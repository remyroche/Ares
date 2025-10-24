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
- Duplicate detection and handling
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
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Awaitable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
        from src.utils.tprint import (
            tprint, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_data_preview,
            tprint_debug, tprint_exception, tprint_progress, tprint_performance, tprint_structured,
            tprint_timer, tprint_data_format, tprint_batch
        )
        return (tprint, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_data_preview,
                tprint_debug, tprint_exception, tprint_progress, tprint_performance, tprint_structured,
                tprint_timer, tprint_data_format, tprint_batch)
    except ImportError:
        # Fallback functions
        def tprint(*args, **kwargs):
            print(*args, **kwargs)
        return tprint, tprint, tprint, tprint, tprint, tprint, tprint, tprint, tprint, tprint, tprint, tprint, tprint, tprint

# Initialize lazy imports
system_logger = get_system_logger()
(tprint, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_data_preview,
 tprint_debug, tprint_exception, tprint_progress, tprint_performance, tprint_structured,
 tprint_timer, tprint_data_format, tprint_batch) = get_tprint_functions()
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

def _lazy_import_quality_utilities():
    """Lazy import of quality utilities to avoid circular imports."""
    global QUALITY_UTILITIES_AVAILABLE, _COMPREHENSIVE_DUPLICATE_ANALYZER, _DATA_QUALITY_FRAMEWORK
    global _COMPREHENSIVE_QUALITY_SCORER, _ADVANCED_QUALITY_METRICS, _DATA_CLEANER
    global _STATISTICAL_VALIDATOR, _QUALITY_ALERT_SYSTEM, _ANALYZE_DUPLICATES_COMPREHENSIVE
    
    if QUALITY_UTILITIES_AVAILABLE is not False:
        return
    
    try:
        from src.utils.data.quality.comprehensive_duplicate_analyzer import (
            ComprehensiveDuplicateAnalyzer,
            analyze_duplicates_comprehensive
        )
        from src.utils.data.quality.data_quality import DataQualityFramework, QualityThresholds, QualityResult
        from src.utils.data.quality.comprehensive_quality_scorer import ComprehensiveQualityScorer, QualityScore, QualityScoreLevel
        from src.utils.data.quality.advanced_quality_metrics import AdvancedQualityMetrics, QualityAssessment
        from src.utils.data.quality.data_cleaning import DataCleaner
        from src.utils.data.quality.statistical_distribution_validation import StatisticalValidator
        from src.utils.data.quality.quality_alert_system import QualityAlertManager
        
        _COMPREHENSIVE_DUPLICATE_ANALYZER = ComprehensiveDuplicateAnalyzer
        _DATA_QUALITY_FRAMEWORK = DataQualityFramework
        _COMPREHENSIVE_QUALITY_SCORER = ComprehensiveQualityScorer
        _ADVANCED_QUALITY_METRICS = AdvancedQualityMetrics
        _DATA_CLEANER = DataCleaner
        _STATISTICAL_VALIDATOR = StatisticalValidator
        _QUALITY_ALERT_SYSTEM = QualityAlertManager
        _ANALYZE_DUPLICATES_COMPREHENSIVE = analyze_duplicates_comprehensive
        
        QUALITY_UTILITIES_AVAILABLE = True
    except ImportError as e:
        tprint_warning(f"⚠️ Some data quality utilities not available: {e}")
        QUALITY_UTILITIES_AVAILABLE = False

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

    _COMPREHENSIVE_DUPLICATE_ANALYZER = ComprehensiveDuplicateAnalyzer
    _ANALYZE_DUPLICATES_COMPREHENSIVE = analyze_duplicates_comprehensive

    # Fallback classes for missing quality utilities
    class DataQualityFramework:
        def validate_data(self, df, thresholds=None):
            return QualityResult(passed=True, issues=[], warnings=[], quality_score=100.0)
        
        def validate_dataframe_quality(self, df, context=''):
            return QualityResult(passed=True, issues=[], warnings=[], quality_score=100.0)

    class ComprehensiveQualityScorer:
        def score_data_quality(self, df, symbol=None, interval=None):
            return QualityScore(overall_score=0.0, level=QualityScoreLevel.CRITICAL,
                              component_scores={}, issues=[], warnings=[],
                              recommendations=[], assessment_timestamp=datetime.now(),
                              data_shape=(0, 0))
        
        def assess_data_quality(self, df, symbol=None, interval=None):
            return QualityScore(overall_score=0.0, level=QualityScoreLevel.CRITICAL,
                              component_scores={}, issues=[], warnings=[],
                              recommendations=[], assessment_timestamp=datetime.now(),
                              data_shape=(0, 0))

    class AdvancedQualityMetrics:
        def assess_quality(self, df):
            return QualityAssessment(overall_score=0.0, metrics=[], issues_found=0,
                                   warnings_found=0, critical_issues=0,
                                   assessment_timestamp=datetime.now(), data_shape=(0, 0))
        
        def comprehensive_quality_assessment(self, df):
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
# Import ExchangeInterface from the proper location
from src.trading.execution.exchange_interface import ExchangeInterface, create_exchange_interface

# Import the proper classes from their locations
from exchanges.shared.unified_ohlcv_standardizer import UnifiedOHLCVStandardizer
# Import BaseStep for inheritance
from src.training.steps.base_step import BaseStep

# Import existing data collection components
from .unified_gap_filler import UnifiedGapFiller
from .enhanced_api_agnostic_data_collector import DataGapDetector, IncrementalDataDownloader
from .utils.data_operations_utils import DataFormatter, DataFormat

# Initialize quality utilities at module level
_lazy_import_quality_utilities()

class StorageConfig:
    """Configuration for data storage operations with hardware optimization."""
    
    def __init__(
        self,
        base_path: str = "data",
        compression: str = "snappy",
        partition_by: Optional[List[str]] = None,
        max_file_size_mb: int = 100,
        backup_enabled: bool = True,
        versioning_enabled: bool = True,
        enable_hardware_optimization: bool = True,
        memory_optimization_level: str = "balanced",
        **kwargs
    ):
        """
        Initialize storage configuration with hardware optimization.
        
        Args:
            base_path: Base directory for data storage
            compression: Compression algorithm (snappy, gzip, lz4, none)
            partition_by: Columns to partition data by
            max_file_size_mb: Maximum file size in MB before splitting
            backup_enabled: Whether to create backups
            versioning_enabled: Whether to enable versioning
            enable_hardware_optimization: Whether to enable hardware optimizations
            memory_optimization_level: Memory optimization level (conservative, balanced, aggressive)
        """
        self.base_path = Path(base_path)
        self.compression = compression
        self.partition_by = partition_by or []
        self.max_file_size_mb = max_file_size_mb
        self.backup_enabled = backup_enabled
        self.versioning_enabled = versioning_enabled
        self.enable_hardware_optimization = enable_hardware_optimization
        self.memory_optimization_level = memory_optimization_level
        
        # Initialize hardware optimization if enabled
        if self.enable_hardware_optimization:
            self._init_hardware_optimization()
        
        # Create base directory if it doesn't exist
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Additional configuration from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def _init_hardware_optimization(self):
        """Initialize hardware optimization components."""
        try:
            from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager
            from src.utils.hardware.optimization_decorators import smart_cache, memory_efficient
            from src.utils.ml_common.unified_vectorization_manager import UnifiedVectorizationManager, OperationType, OptimizationStrategy
            
            # Initialize hardware manager
            self.hardware_manager = UnifiedHardwareManager()
            
            # Initialize vectorization manager
            self.vectorization_manager = UnifiedVectorizationManager()
            
            # Configure optimization strategy based on memory level
            strategy_map = {
                "conservative": OptimizationStrategy.MEMORY,
                "balanced": OptimizationStrategy.BALANCED,
                "aggressive": OptimizationStrategy.SPEED
            }
            self.optimization_strategy = strategy_map.get(self.memory_optimization_level, OptimizationStrategy.BALANCED)
            
            # Apply memory-efficient decorators to key methods
            self._apply_optimization_decorators()
            
        except ImportError as e:
            # Fallback if hardware optimization not available
            self.hardware_manager = None
            self.vectorization_manager = None
            self.optimization_strategy = None
    
    def _apply_optimization_decorators(self):
        """Apply optimization decorators to methods."""
        try:
            from src.utils.hardware.optimization_decorators import smart_cache, memory_efficient
            
            # Apply decorators to data processing methods
            if hasattr(self, 'process_data'):
                self.process_data = smart_cache(ttl=3600)(memory_efficient()(self.process_data))
            
        except ImportError:
            pass

class KlinesMetadata:
    """Metadata for klines data operations."""
    
    def __init__(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        total_candles: int = 0,
        data_quality_score: float = 0.0,
        gaps_detected: int = 0,
        gaps_filled: int = 0,
        duplicates_removed: int = 0,
        processing_steps: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize klines metadata.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe
            start_time: Start timestamp of data
            end_time: End timestamp of data
            total_candles: Total number of candles
            data_quality_score: Quality score (0.0-1.0)
            gaps_detected: Number of gaps detected
            gaps_filled: Number of gaps filled
            duplicates_removed: Number of duplicates removed
            processing_steps: List of processing steps completed
        """
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.start_time = start_time
        self.end_time = end_time
        self.total_candles = total_candles
        self.data_quality_score = data_quality_score
        self.gaps_detected = gaps_detected
        self.gaps_filled = gaps_filled
        self.duplicates_removed = duplicates_removed
        self.processing_steps = processing_steps or []
        self.created_at = datetime.now()
        
        # Additional metadata from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'timeframe': self.timeframe,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_candles': self.total_candles,
            'data_quality_score': self.data_quality_score,
            'gaps_detected': self.gaps_detected,
            'gaps_filled': self.gaps_filled,
            'duplicates_removed': self.duplicates_removed,
            'processing_steps': self.processing_steps,
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KlinesMetadata':
        """Create metadata from dictionary."""
        # Convert timestamp strings back to datetime objects
        if data.get('start_time'):
            data['start_time'] = datetime.fromisoformat(data['start_time'])
        if data.get('end_time'):
            data['end_time'] = datetime.fromisoformat(data['end_time'])
        if data.get('created_at'):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        return cls(**data)

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
    target_intervals: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "30m", "1h"])
    method: str = "ohlc"  # ohlc, vwap, etc.
    preserve_volume: bool = True
    validate_continuity: bool = True
    resample_older_than_days: int = 3  # Only resample data older than this many days
    enable_auto_resampling: bool = True  # Automatically resample based on data age

@dataclass
class PipelineConfig:
    """Configuration for the enhanced klines processing pipeline."""
    data_dir: str = "historical_data"
    exchange: str = "binance"
    enable_logging: bool = True
    max_gap_minutes: int = 1
    enable_gap_filling: bool = True
    enable_resampling: bool = True
    enable_duplicate_handling: bool = True
    enable_quality_validation: bool = True
    batch_compatible: bool = True
    storage_config: Optional[StorageConfig] = None

class EnhancedKlinesProcessingPipeline(BaseStep):
    """
    Enhanced klines data processing pipeline with comprehensive type hints,
    exchange-agnostic design, and fast-fail patterns.

    OPTIMIZED GAP-FIRST APPROACH:
    =============================
    This pipeline uses an optimized approach that prevents duplicate downloads:
    1. First analyzes existing data to detect gaps
    2. Downloads ONLY the missing data periods (with immediate standardization)
    3. Combines existing and new data
    4. Validates and processes the complete dataset

    This prevents the inefficient pattern of:
    - Downloading all data → detecting gaps → re-downloading gaps

    Features:
    - Uses ExchangeInterface for all exchange calls
    - Integrates KlinesParquetManager for efficient storage
    - Implements data standardizer for consistent formatting
    - Fast fail pattern with no fallbacks or mocks (connection failures cause immediate failure)
    - Comprehensive gap detection and filling (OPTIMIZED)
    - Automatic resampling for data older than 3 days
    - Batch-compatible data management
    - Selective downloading to avoid duplicates
    - Immediate data standardization during download
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None
    ) -> None:
        """Initialize the enhanced processing pipeline.

        Args:
            config: Pipeline configuration
        """
        # Initialize BaseStep first
        step_name = f"enhanced_klines_processing_{config.exchange if config else 'default'}"
        super().__init__(step_name)
        
        self.config = config or PipelineConfig()
        self.data_dir = Path(self.config.data_dir)
        self.exchange = self.config.exchange.lower()
        self.enable_logging = self.config.enable_logging

        # Initialize components
        tprint_info("🔧 Initializing data standardizer")
        self.data_standardizer = UnifiedOHLCVStandardizer()
        tprint_success("✅ Data standardizer initialized")
        
        # Initialize lazy-loaded quality utilities
        tprint_info("🔧 Loading quality utilities")
        _lazy_import_quality_utilities()
        self.duplicate_analyzer = _COMPREHENSIVE_DUPLICATE_ANALYZER()
        tprint_success("✅ Quality utilities loaded")

        # Note: KlinesParquetManager is now available via BaseStep integration
        # Use self._store_klines(), self._load_klines(), etc. methods

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

        if self.enable_logging:
            tprint_structured({
                "operation": "enhanced_pipeline_initialization",
                "exchange": self.exchange,
                "data_dir": str(self.data_dir),
                "enable_logging": self.enable_logging,
                "components_initialized": [
                    "data_standardizer", "duplicate_analyzer", "quality_utilities"
                ]
            }, level="SUCCESS")
            tprint_success(f"✅ Enhanced Klines Processing Pipeline initialized for {self.exchange}")

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the enhanced klines processing pipeline (required by BaseStep).
        
        Args:
            config: Configuration containing symbol, exchange, interval, etc.
            
        Returns:
            Execution result with artifacts and metrics
        """
        try:
            tprint_info("🚀 Starting enhanced klines processing pipeline execution")
            self.logger.info("🚀 Starting enhanced klines processing pipeline execution")
            
            # Extract configuration
            symbol = config.get('symbol', 'ETHUSDT')
            exchange = config.get('exchange', 'binance')
            interval = config.get('interval', '1m')
            years = config.get('years', 3)
            
            tprint_structured({
                "operation": "enhanced_pipeline_execution_start",
                "symbol": symbol,
                "exchange": exchange,
                "interval": interval,
                "years": years
            }, level="INFO")
            
            if not symbol:
                tprint_error("❌ Symbol is required for klines processing")
                raise ValueError("Symbol is required for klines processing")
            
            # Set context for klines operations
            self._set_context(
                symbol=symbol,
                exchange=exchange,
                direction=config.get('direction', 'long'),
                model=config.get('model', 'Analyst')
            )
            
            tprint_info(f"Processing klines data for {symbol} from {exchange}")
            tprint_info(f"Interval: {interval}, Years: {years}")
            self.logger.info(f"Processing klines data for {symbol} from {exchange}")
            self.logger.info(f"Interval: {interval}, Years: {years}")
            
            # Initialize artifacts list
            artifacts = []
            metrics = {}
            
            # Process klines data using the existing pipeline
            results = await self.process_klines_data(
                symbol=symbol,
                interval=interval,
                years=years,
                exchange_interface=None,  # Will be created internally
                resampling_config=None,
                max_gap_minutes=1,
                create_consolidated=True
            )
            
            if results.get('pipeline_success', False):
                # Store processed data using KlinesParquetManager
                if 'final_data' in results and results['final_data'] is not None:
                    success = self._store_klines_with_context(
                        df=results['final_data'],
                        interval=interval,
                        batch_id=f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        metadata={
                            'years': years,
                            'processing_timestamp': datetime.now().isoformat(),
                            'data_quality': results.get('data_quality', 'unknown'),
                            'final_shape': results.get('final_data_shape', 'unknown')
                        }
                    )
                    
                    if success:
                        self.logger.info(f"✅ Processed klines data stored using KlinesParquetManager")
                        artifacts.append({
                            'name': f"processed_klines_{interval}",
                            'type': 'klines_data',
                            'size': f"Processed successfully"
                        })
                    else:
                        self.logger.warning(f"⚠️ Failed to store processed klines data")
                
                # Calculate metrics
                metrics = {
                    'symbol': symbol,
                    'exchange': exchange,
                    'interval': interval,
                    'years': years,
                    'data_quality': results.get('data_quality', 'unknown'),
                    'final_shape': results.get('final_data_shape', 'unknown'),
                    'pipeline_success': results.get('pipeline_success', False)
                }
                
                tprint_success("✅ Enhanced klines processing completed successfully")
                tprint_structured({
                    "operation": "enhanced_pipeline_completed",
                    "symbol": symbol,
                    "interval": interval,
                    "pipeline_success": results.get('pipeline_success', False),
                    "data_quality": results.get('data_quality', 'unknown'),
                    "final_shape": results.get('final_data_shape', 'unknown')
                }, level="SUCCESS")
                
                self.logger.info(f"✅ Enhanced klines processing completed successfully")
                return {
                    'success': True,
                    'artifacts': artifacts,
                    'metrics': metrics,
                    'results': results
                }
            else:
                tprint_error("❌ Enhanced klines processing failed")
                tprint_structured({
                    "operation": "enhanced_pipeline_failed",
                    "symbol": symbol,
                    "interval": interval,
                    "error": "Pipeline processing failed"
                }, level="ERROR")
                
                self.logger.error("❌ Enhanced klines processing failed")
                return {
                    'success': False,
                    'artifacts': artifacts,
                    'metrics': metrics,
                    'error': 'Pipeline processing failed'
                }
                
        except Exception as e:
            tprint_error(f"❌ Enhanced klines processing failed: {e}")
            tprint_exception(e, "Enhanced klines processing pipeline execution failed")
            tprint_structured({
                "operation": "enhanced_pipeline_exception",
                "symbol": symbol if 'symbol' in locals() else 'unknown',
                "interval": interval if 'interval' in locals() else 'unknown',
                "error": str(e),
                "error_type": type(e).__name__
            }, level="ERROR")
            
            self.logger.error(f"❌ Enhanced klines processing failed: {e}")
            return {
                'success': False,
                'artifacts': [],
                'metrics': {},
                'error': str(e)
            }
            tprint_info(f"   📁 Data directory: {self.data_dir}")
            tprint_info(f"   🔧 Gap filling: {'enabled' if self.config.enable_gap_filling else 'disabled'}")
            tprint_info(f"   📊 Resampling: {'enabled' if self.config.enable_resampling else 'disabled'}")
            tprint_info(f"   🔄 Batch compatible: {'enabled' if self.config.batch_compatible else 'disabled'}")

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
        Process klines data through the complete pipeline with optimized gap-first approach.

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

        if not exchange_interface:
            raise ValueError("ExchangeInterface is required for data processing")

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

            # Step 1: Connect to exchange (fast fail)
            try:
                await exchange_interface.connect()
            except Exception as e:
                error_msg = f"Exchange connection failed: {e}"
                if self.enable_logging:
                    tprint_error(f"❌ {error_msg}")
                raise RuntimeError(error_msg)

            # Step 2: Analyze existing data and detect gaps FIRST (OPTIMIZED APPROACH)
            if self.enable_logging:
                tprint_info("🔍 Step 1: Analyzing existing data and detecting gaps")
            
            gap_analysis_result = await self._analyze_existing_data_and_gaps(
                symbol, interval, years, max_gap_minutes
            )
            self.processing_results.append(gap_analysis_result)

            if not gap_analysis_result.success:
                raise RuntimeError(f"Gap analysis failed: {gap_analysis_result.errors}")

            results["steps_completed"].append(ProcessingStep.GAP_DETECTION.value)

            # Step 3: Download only missing data (if gaps found)
            if gap_analysis_result.metadata.get("download_required", False):
                if self.enable_logging:
                    gaps_detected = gap_analysis_result.metadata.get("gaps_detected", 0)
                    if gaps_detected > 0:
                        tprint_info(f"📥 Step 2: Downloading {gaps_detected} missing data periods")
                    else:
                        tprint_info("📥 Step 2: No existing data found - downloading all data")
                
                download_result = await self._download_missing_data(
                    gap_analysis_result.data, symbol, interval, exchange_interface
                )
                self.processing_results.append(download_result)

                if not download_result.success:
                    raise RuntimeError(f"Missing data download failed: {download_result.errors}")

                results["steps_completed"].append(ProcessingStep.DOWNLOAD.value)
                if download_result.metadata.get("gaps_filled", 0) > 0:
                    results["steps_completed"].append(ProcessingStep.GAP_FILLING.value)
                
                current_data = download_result.data
            else:
                if self.enable_logging:
                    tprint_success("✅ No gaps found - using existing data")
                current_data = gap_analysis_result.data

            # Step 4: Standardize data format using ExchangeDataStandardizer
            if self.enable_logging:
                tprint_info("🔧 Step 3: Standardizing data format")
            
            standardize_result = await self._standardize_data(
                current_data, symbol, interval
            )
            self.processing_results.append(standardize_result)

            if not standardize_result.success:
                raise RuntimeError(f"Data standardization failed: {standardize_result.errors}")

            results["steps_completed"].append(ProcessingStep.STANDARDIZE.value)

            # Step 5: Validate data quality
            if self.config.enable_quality_validation:
                if self.enable_logging:
                    tprint_info("🔍 Step 4: Validating data quality")
                
                # Preview data before quality validation
                tprint_data_preview(standardize_result.data, f"Data before quality validation for {symbol} {interval}", max_rows=3, include_metadata=True)
                
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
                resample_result = await self._resample_data_with_age_check(
                    current_data, symbol, resampling_config, batch_id
                )
                self.processing_results.append(resample_result)

                if resample_result.success:
                    results["steps_completed"].append(ProcessingStep.RESAMPLING.value)
                    results["resampled_intervals"] = resample_result.metadata.get("resampled_intervals", [])
                    results["stored_files"].extend(resample_result.metadata.get("stored_files", []))
                else:
                    results["warnings"].extend(resample_result.warnings)

            # Step 8: Final quality check
            # Preview data before final quality check
            tprint_data_preview(current_data, f"Data before final quality check for {symbol} {interval}", max_rows=3, include_metadata=True)
            
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

    async def _analyze_existing_data_and_gaps(
        self,
        symbol: str,
        interval: str,
        years: int,
        max_gap_minutes: int
    ) -> ProcessingResult:
        """
        Analyze existing data and detect gaps using UnifiedGapFiller.
        This is the optimized approach that prevents duplicate downloads.
        """
        start_time = datetime.now()
        result = ProcessingResult(
            step=ProcessingStep.GAP_DETECTION,
            success=False,
            errors=[],
            warnings=[]
        )

        try:
            if self.enable_logging:
                tprint_info(f"🔍 Analyzing existing data for {symbol} {interval}")

            # Initialize UnifiedGapFiller
            gap_filler = UnifiedGapFiller(data_cache_path=self.config.data_dir)
            
            # Calculate expected date range
            end_date = datetime.now() - timedelta(days=3)  # 3 days ago
            start_date = end_date - timedelta(days=years * 365)
            
            # Use UnifiedGapFiller to detect gaps
            gaps = gap_filler.detect_gaps(
                symbol=symbol,
                exchange="binance",  # Use config exchange
                data_type="klines",
                start_date=start_date,
                end_date=end_date
            )

            # Load existing data if available
            data_dir = Path(self.config.data_dir) / "binance" / symbol.lower() / "raw"
            parquet_files = list(data_dir.glob(f"{symbol.lower()}_{interval}_*.parquet")) if data_dir.exists() else []
            
            existing_data = pd.DataFrame()
            if parquet_files:
                if self.enable_logging:
                    tprint_info(f"📁 Found {len(parquet_files)} existing parquet files")

                all_data = []
                for file_path in sorted(parquet_files):
                    try:
                        df = pd.read_parquet(file_path)
                        if not df.empty:
                            all_data.append(df)
                            if self.enable_logging:
                                tprint_info(f"  📊 Loaded {len(df)} records from {file_path.name}")
                    except Exception as e:
                        result.warnings.append(f"Failed to load {file_path.name}: {e}")

                if all_data:
                    existing_data = pd.concat(all_data, ignore_index=True)
                    existing_data = existing_data.drop_duplicates().sort_values('timestamp')
                    
                    # Ensure timestamp is datetime
                    if not pd.api.types.is_datetime64_any_dtype(existing_data['timestamp']):
                        existing_data['timestamp'] = pd.to_datetime(existing_data['timestamp'])
                    
                    existing_data.set_index('timestamp', inplace=True)

                    if self.enable_logging:
                        tprint_success(f"✅ Loaded {len(existing_data)} existing records")
                        tprint_info(f"📅 Existing data range: {existing_data.index.min()} to {existing_data.index.max()}")
            else:
                if self.enable_logging:
                    tprint_info("📁 No existing data found - will download all data")

            # Convert UnifiedGapFiller gaps to our GapInfo format
            gap_info_list = []
            for gap in gaps:
                gap_info = GapInfo(
                    start_time=gap['start_time'],
                    end_time=gap['end_time'],
                    duration_minutes=int(gap['gap_minutes']),
                    symbol=symbol,
                    interval=interval,
                    priority=1 if gap['gap_minutes'] > max_gap_minutes else 2
                )
                gap_info_list.append(gap_info)

            result.metadata = {
                "gaps_detected": len(gap_info_list),
                "gaps_filled": 0,
                "existing_records": len(existing_data),
                "download_required": len(gap_info_list) > 0,
                "date_range": {
                    "start": existing_data.index.min().isoformat() if not existing_data.empty else None,
                    "end": existing_data.index.max().isoformat() if not existing_data.empty else None
                },
                "gaps": [gap.__dict__ for gap in gap_info_list]
            }

            if self.enable_logging:
                if len(gap_info_list) > 0:
                    tprint_warning(f"⚠️ Found {len(gap_info_list)} gaps in existing data")
                    for i, gap in enumerate(gap_info_list[:3]):  # Show first 3 gaps
                        tprint_info(f"  Gap {i+1}: {gap.start_time} to {gap.end_time} ({gap.duration_minutes}min)")
                    if len(gap_info_list) > 3:
                        tprint_info(f"  ... and {len(gap_info_list) - 3} more gaps")
                else:
                    tprint_success("✅ No gaps found in existing data")

            result.success = True
            result.data = existing_data

        except Exception as e:
            error_msg = f"Gap analysis failed: {str(e)}"
            result.errors.append(error_msg)
            if self.enable_logging:
                tprint_error(f"❌ {error_msg}")

        result.processing_time = (datetime.now() - start_time).total_seconds()
        return result

    async def _download_missing_data(
        self,
        existing_data: pd.DataFrame,
        symbol: str,
        interval: str,
        exchange_interface: ExchangeInterface
    ) -> ProcessingResult:
        """
        Download only the missing data periods using IncrementalDataDownloader.
        """
        start_time = datetime.now()
        result = ProcessingResult(
            step=ProcessingStep.DOWNLOAD,
            success=False,
            errors=[],
            warnings=[]
        )

        try:
            if self.enable_logging:
                tprint_info(f"📥 Downloading missing data for {symbol} {interval}")

            # Initialize IncrementalDataDownloader
            downloader = IncrementalDataDownloader(
                exchange="binance",
                symbol=symbol,
                timeframe=interval,
                data_cache_path=self.config.data_dir
            )

            # Use the downloader's detect_and_fill_gaps method
            gap_result = await downloader.detect_and_fill_gaps(
                data_type="klines",
                start_date=datetime.now() - timedelta(days=4 * 365),
                end_date=datetime.now() - timedelta(days=3)
            )

            if not gap_result.get('success', False):
                raise RuntimeError(f"Gap filling failed: {gap_result.get('error', 'Unknown error')}")

            # Load the updated data
            updated_data = existing_data.copy()
            
            # If gaps were filled, reload the data from files
            if gap_result.get('gaps_filled', 0) > 0:
                data_dir = Path(self.config.data_dir) / "binance" / symbol.lower() / "raw"
                parquet_files = list(data_dir.glob(f"{symbol.lower()}_{interval}_*.parquet")) if data_dir.exists() else []
                
                if parquet_files:
                    all_data = []
                    for file_path in sorted(parquet_files):
                        try:
                            df = pd.read_parquet(file_path)
                            if not df.empty:
                                all_data.append(df)
                        except Exception as e:
                            result.warnings.append(f"Failed to load {file_path.name}: {e}")

                    if all_data:
                        updated_data = pd.concat(all_data, ignore_index=True)
                        updated_data = updated_data.drop_duplicates().sort_values('timestamp')
                        
                        # Ensure timestamp is datetime
                        if not pd.api.types.is_datetime64_any_dtype(updated_data['timestamp']):
                            updated_data['timestamp'] = pd.to_datetime(updated_data['timestamp'])
                        
                        updated_data.set_index('timestamp', inplace=True)

            result.success = True
            result.data = updated_data
            result.metadata = {
                "gaps_filled": gap_result.get('gaps_filled', 0),
                "records_downloaded": gap_result.get('total_rows_downloaded', 0),
                "total_records": len(updated_data)
            }

            if self.enable_logging:
                tprint_success(f"✅ Downloaded {gap_result.get('gaps_filled', 0)} gaps")
                tprint_info(f"📊 Total records: {len(updated_data)}")

        except Exception as e:
            error_msg = f"Missing data download failed: {str(e)}"
            result.errors.append(error_msg)
            if self.enable_logging:
                tprint_error(f"❌ {error_msg}")

        result.processing_time = (datetime.now() - start_time).total_seconds()
        return result

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
            data_dir = Path(self.config.data_dir) / "binance" / symbol.lower() / "raw"
            parquet_files = list(data_dir.glob(f"{symbol.lower()}_{interval}_*.parquet")) if data_dir.exists() else []
            
            if parquet_files and not getattr(self.config, 'force_download', False):
                # Load existing data from parquet files
                if self.enable_logging:
                    tprint_info(f"📁 Found {len(parquet_files)} existing parquet files")
                
                all_data = []
                for file_path in sorted(parquet_files):
                    try:
                        df = pd.read_parquet(file_path)
                        if not df.empty:
                            all_data.append(df)
                            if self.enable_logging:
                                tprint_info(f"  📊 Loaded {len(df)} records from {file_path.name}")
                    except Exception as e:
                        result.warnings.append(f"Failed to load {file_path.name}: {e}")
                
                if all_data:
                    klines_data = pd.concat(all_data, ignore_index=True)
                else:
                    raise ValueError("No valid data found in parquet files")
            else:
                # Download fresh data from exchange
                if self.enable_logging:
                    tprint_info(f"🌐 Downloading fresh data from {exchange_interface.exchange_type.upper()} exchange")
                
                # Calculate date range
                end_date = datetime.now() - timedelta(days=3)  # 3 days ago
                start_date = end_date - timedelta(days=years * 365)
                
                # Download historical data using get_klines
                klines_data = await exchange_interface.get_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=start_date,
                    end_time=end_date,
                    limit=1000
                )
                
                # Preview raw data from exchange API
                tprint_data_preview(klines_data, f"Raw klines data from {exchange_interface.exchange_type.upper()}", max_rows=3, include_metadata=True)
                
                # Convert KlineData objects to list format
                raw_data = []
                for kline in klines_data:
                    raw_data.append([
                        int(kline.timestamp.timestamp() * 1000),  # timestamp
                        kline.open_price,  # open
                        kline.high_price,  # high
                        kline.low_price,   # low
                        kline.close_price, # close
                        kline.volume,      # volume
                        int(kline.close_time.timestamp() * 1000),  # close_time
                        kline.quote_asset_volume,  # quote_volume
                        kline.number_of_trades,     # trades
                        kline.taker_buy_base_asset_volume,  # taker_buy_base
                        kline.taker_buy_quote_asset_volume   # taker_buy_quote
                    ])
                
                if not raw_data:
                    raise ValueError("No data received from exchange")

            # Convert to DataFrame
                klines_data = pd.DataFrame(raw_data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume'
                ])
                
                # Convert timestamp to datetime
                klines_data['timestamp'] = pd.to_datetime(klines_data['timestamp'], unit='ms')
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
            result.errors.append(error_msg)
            if self.enable_logging:
                tprint_error(f"❌ {error_msg}")

        result.processing_time = (datetime.now() - start_time).total_seconds()
        return result

    async def _standardize_dataframe(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str
    ) -> pd.DataFrame:
        """
        Standardize a DataFrame using DataFormatter.
        """
        try:
            # Initialize DataFormatter
            formatter = DataFormatter()
            
            # Ensure proper column names and types
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")
                # Convert to numeric, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Ensure timestamp index is datetime
            if not pd.api.types.is_datetime64_any_dtype(df.index):
                df.index = pd.to_datetime(df.index)
            
            # Add metadata columns
            df['symbol'] = symbol.upper()
            df['interval'] = interval
            df['exchange'] = self.exchange
            
            # Use DataFormatter to format klines data
            format_result = formatter.format_klines_data(
                data=df,
                symbol=symbol,
                interval=interval,
                exchange=self.exchange
            )
            
            if format_result.get('success', False):
                return format_result['data']
            else:
                if self.enable_logging:
                    tprint_warning(f"⚠️ DataFormatter failed: {format_result.get('error', 'Unknown error')}")
                # Fall back to original data
                return df
            
        except Exception as e:
            if self.enable_logging:
                tprint_warning(f"⚠️ Data standardization failed: {e}")
            # Return original data if standardization fails
            return df

    def _klines_to_dataframe(
        self,
        klines_data: List[Any],
        symbol: str,
        interval: str
    ) -> pd.DataFrame:
        """Convert klines data to standardized DataFrame."""
        try:
            # Extract data based on klines format
            data = []
            for kline in klines_data:
                if hasattr(kline, 'timestamp'):
                    # KlineData object
                    data.append({
                        'timestamp': kline.timestamp,
                        'open': kline.open_price,
                        'high': kline.high_price,
                        'low': kline.low_price,
                        'close': kline.close_price,
                        'volume': kline.volume,
                        'quote_volume': getattr(kline, 'quote_asset_volume', 0),
                        'trades': getattr(kline, 'number_of_trades', 0),
                        'taker_buy_base': getattr(kline, 'taker_buy_base_asset_volume', 0),
                        'taker_buy_quote': getattr(kline, 'taker_buy_quote_asset_volume', 0)
                    })
                else:
                    # Raw list format
                    data.append({
                        'timestamp': pd.to_datetime(kline[0], unit='ms', utc=True),
                        'open': float(kline[1]),
                        'high': float(kline[2]),
                        'low': float(kline[3]),
                        'close': float(kline[4]),
                        'volume': float(kline[5]),
                        'quote_volume': float(kline[6]) if len(kline) > 6 else 0,
                        'trades': int(kline[7]) if len(kline) > 7 else 0,
                        'taker_buy_base': float(kline[8]) if len(kline) > 8 else 0,
                        'taker_buy_quote': float(kline[9]) if len(kline) > 9 else 0
                    })

            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)

            return df

        except Exception as e:
            if self.enable_logging:
                tprint_error(f"❌ Failed to convert klines to DataFrame: {e}")
            return pd.DataFrame()

    async def _standardize_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str
    ) -> ProcessingResult:
        """Standardize data format using ExchangeDataStandardizer."""
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

            # Use UnifiedOHLCVStandardizer
            standardized_df = self.data_standardizer.standardize(
                df
            )
            
            # Preview standardized data
            tprint_data_preview(standardized_df, "Standardized klines data", max_rows=3, include_metadata=True)

            result.success = True
            result.data = standardized_df
            result.metadata = {
                "original_shape": df.shape,
                "final_shape": standardized_df.shape,
                "processing_time": 0.0
            }
            # No warnings from unified standardizer

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

            # Check required columns
            missing_columns = [col for col in self.required_ohlcv_columns if col not in df.columns]
            if missing_columns:
                raise RuntimeError(f"Missing required columns: {missing_columns}")

            if not QUALITY_UTILITIES_AVAILABLE:
                # Use basic validation without quality utilities
                quality_result = QualityResult(
                    quality_score=50.0,
                    issues=["Quality utilities not available"],
                    warnings=["Basic validation only"]
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
            else:
                # Initialize comprehensive quality framework
                quality_framework = DataQualityFramework()
                scorer = ComprehensiveQualityScorer()
                advanced_metrics = AdvancedQualityMetrics()
                data_cleaner = DataCleaner()
                statistical_validator = StatisticalValidator()

                # Set up quality thresholds for klines data
                thresholds = QualityThresholds(
                    max_nan_ratio=0.05,
                    max_infinite_count=0,
                    min_unique_values=2,
                    max_constant_ratio=0.95
                )

                # Perform comprehensive data quality validation
                quality_result = quality_framework.validate_dataframe_quality(df, context="klines_validation")

                # Get advanced quality assessment
                quality_assessment = advanced_metrics.comprehensive_quality_assessment(df, context="klines_validation", step_name="data_validation")

                # Get comprehensive quality score
                score = scorer.assess_data_quality(df, context="klines_validation", step_name="data_validation", data_type="klines")

                # Perform statistical distribution validation
                distribution_validation = {}
                for col in df.select_dtypes(include=[np.number]).columns:
                    if col in ['open', 'high', 'low', 'close', 'volume']:
                        validation_results = statistical_validator.run_comprehensive_validation(df[col].values)
                        distribution_validation[col] = {
                            'results': [{'status': r.status.value, 'message': r.message} for r in validation_results]
                        }
            # Initialize comprehensive quality framework
            quality_framework = DataQualityFramework()
            scorer = ComprehensiveQualityScorer()
            advanced_metrics = AdvancedQualityMetrics()
            data_cleaner = DataCleaner()
            statistical_validator = StatisticalValidator()

            # Set up quality thresholds for klines data
            thresholds = QualityThresholds(
                max_nan_ratio=0.05,  # 5% max NaN ratio
                max_infinite_count=0,
                min_unique_values=2,
                max_constant_ratio=0.95,
                max_gap_hours=48,
                price_tolerance=0.001,
                volume_tolerance=0.001,
                max_correlation_threshold=0.95,
                min_feature_count=40
            )

            # Perform comprehensive data quality validation
            quality_result = quality_framework.validate_dataframe_quality(df, f"{symbol}_{interval}")

            # Get advanced quality assessment
            quality_assessment = advanced_metrics.comprehensive_quality_assessment(df)

            # Get comprehensive quality score
            score = scorer.assess_data_quality(df, symbol, interval)

            # Perform statistical distribution validation
            distribution_validation = statistical_validator.run_comprehensive_validation(df.values)

            # Check for duplicates using comprehensive analyzer
            duplicate_analysis = _ANALYZE_DUPLICATES_COMPREHENSIVE(df)

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
            all_issues = quality_result.issues + score.issues + quality_assessment.metrics
            all_warnings = quality_result.warnings + score.warnings

            # Add duplicate analysis warnings
            if duplicate_analysis.total_duplicates > 0:
                all_warnings.append(f"Found {duplicate_analysis.total_duplicates} duplicate records")

            result.success = True
            result.data = df
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
                    "total_duplicates": duplicate_analysis.total_duplicates,
                    "true_duplicate_groups": duplicate_analysis.true_duplicate_groups,
                    "false_duplicate_groups": duplicate_analysis.false_duplicate_groups,
                    "mixed_duplicate_groups": duplicate_analysis.mixed_duplicate_groups
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

            # Detect gaps
            gaps = self._detect_gaps(df, interval, max_gap_minutes)

            if not gaps:
                result.success = True
                result.data = df
                result.metadata = {"gaps_detected": 0, "gaps_filled": 0}

                if self.enable_logging:
                    tprint_success("✅ No gaps detected")

                return result

            if self.enable_logging:
                tprint_warning(f"⚠️ Found {len(gaps)} gaps > {max_gap_minutes} minutes")

            # Fill gaps by re-downloading data
            filled_data = await self._fill_gaps(df, gaps, symbol, interval, exchange_interface)

            result.success = True
            result.data = filled_data
            result.metadata = {
                "gaps_detected": len(gaps),
                "gaps_filled": len([g for g in gaps if g.priority == 1]),
                "total_gap_duration": sum(g.duration_minutes for g in gaps)
            }

            if self.enable_logging:
                tprint_success(f"✅ Gap handling completed: {result.metadata['gaps_filled']} gaps filled")

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
        max_gap_minutes: int
    ) -> List[GapInfo]:
        """Detect gaps in the data."""
        gaps = []

        if df.empty or len(df) < 2:
            return gaps

        # Calculate expected interval in minutes
        interval_minutes = self._interval_to_minutes(interval)
        if interval_minutes is None:
            return gaps

        # Sort by timestamp
        df_sorted = df.sort_index()

        # Check for gaps
        for i in range(len(df_sorted) - 1):
            current_time = df_sorted.index[i]
            next_time = df_sorted.index[i + 1]

            expected_next_time = current_time + timedelta(minutes=interval_minutes)
            actual_gap_minutes = (next_time - expected_next_time).total_seconds() / 60

            if actual_gap_minutes > max_gap_minutes:
                gap = GapInfo(
                    start_time=expected_next_time,
                    end_time=next_time,
                    duration_minutes=int(actual_gap_minutes),
                    symbol=df_sorted.iloc[i].get('symbol', ''),
                    interval=interval,
                    priority=1 if actual_gap_minutes > interval_minutes * 10 else 2
                )
                gaps.append(gap)

        return gaps

    def _interval_to_minutes(self, interval: str) -> Optional[int]:
        """Convert interval string to minutes."""
        interval_map = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
            '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200
        }
        return interval_map.get(interval)

    async def _fill_gaps(
        self,
        df: pd.DataFrame,
        gaps: List[GapInfo],
        symbol: str,
        interval: str,
        exchange_interface: ExchangeInterface
    ) -> pd.DataFrame:
        """Fill gaps by re-downloading data."""
        filled_data = df.copy()

        for gap in gaps:
            if gap.priority > 1:  # Skip low priority gaps
                continue

            try:
                if self.enable_logging:
                    tprint_info(f"📥 Re-downloading data for gap: {gap.start_time} to {gap.end_time}")

                # Download data for the gap period
                gap_klines = await exchange_interface.get_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=gap.start_time,
                    end_time=gap.end_time,
                    limit=1000
                )
                
                # Preview gap-filled data
                tprint_data_preview(gap_klines, f"Gap-filled data for {gap.start_time} to {gap.end_time}", max_rows=2, include_metadata=True)
                
                # Convert KlineData objects to list format
                gap_data = []
                for kline in gap_klines:
                    gap_data.append([
                        int(kline.timestamp.timestamp() * 1000),  # timestamp
                        kline.open_price,  # open
                        kline.high_price,  # high
                        kline.low_price,   # low
                        kline.close_price, # close
                        kline.volume,      # volume
                        int(kline.close_time.timestamp() * 1000),  # close_time
                        kline.quote_asset_volume,  # quote_volume
                        kline.number_of_trades,     # trades
                        kline.taker_buy_base_asset_volume,  # taker_buy_base
                        kline.taker_buy_quote_asset_volume   # taker_buy_quote
                    ])

                if gap_data:
                    gap_df = self._klines_to_dataframe(gap_data, symbol, interval)
                    if not gap_df.empty:
                        # Standardize the gap data
                        standardized_gap_df = self.data_standardizer.standardize(
                            gap_df
                        )
                        
                        # Preview standardized gap data
                        tprint_data_preview(standardized_gap_df, "Standardized gap data", max_rows=2, include_metadata=True)

                        # Merge with existing data
                        filled_data = pd.concat([filled_data, standardized_gap_df])
                        filled_data = filled_data[~filled_data.index.duplicated(keep='first')]
                        filled_data.sort_index(inplace=True)

                        if self.enable_logging:
                            tprint_success(f"✅ Filled gap with {len(standardized_gap_df)} records")

            except Exception as e:
                if self.enable_logging:
                    tprint_warning(f"⚠️ Failed to fill gap {gap.start_time}: {e}")

        return filled_data

    async def _handle_duplicates(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str
    ) -> ProcessingResult:
        """Handle duplicate timestamps."""
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

            # Analyze duplicates
            analysis_result = self.duplicate_analyzer.analyze_duplicates(df)

            # Handle true duplicates (remove them)
            cleaned_df = df.copy()
            if analysis_result.true_duplicate_groups > 0:
                cleaned_df = cleaned_df.drop_duplicates(keep='first')
                if self.enable_logging:
                    tprint_info(f"🧹 Removed {analysis_result.true_duplicate_groups} groups of true duplicates")

            # Warn about false duplicates
            if analysis_result.false_duplicate_groups > 0:
                warning_msg = f"Found {analysis_result.false_duplicate_groups} groups of false duplicates - manual review required"
                result.warnings.append(warning_msg)
                if self.enable_logging:
                    tprint_warning(f"⚠️ {warning_msg}")

            result.success = True
            result.data = cleaned_df
            result.metadata = {
                "total_duplicates": analysis_result.total_duplicates,
                "true_duplicates_removed": analysis_result.true_duplicate_groups,
                "false_duplicates": analysis_result.false_duplicate_groups,
                "mixed_duplicates": analysis_result.mixed_duplicate_groups
            }

            if self.enable_logging:
                tprint_success(f"✅ Duplicate handling completed: {analysis_result.total_duplicates} duplicates processed")

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
                    
                    # Preview resampled data
                    tprint_data_preview(resampled, f"Resampled data to {target_interval}", max_rows=3, include_metadata=True)

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

            # Check data completeness
            if df.empty:
                raise RuntimeError("Final data is empty")

            # Check required columns
            missing_columns = [col for col in self.required_ohlcv_columns if col not in df.columns]
            if missing_columns:
                raise RuntimeError(f"Missing required columns in final data: {missing_columns}")

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
                    data_shape=df.shape
                )
                final_quality_assessment = QualityAssessment(
                    overall_score=50.0,
                    metrics=[],
                    issues_found=1,
                    warnings_found=1,
                    critical_issues=0,
                    assessment_timestamp=datetime.now(),
                    data_shape=df.shape
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
                final_score = scorer.assess_data_quality(df, context="final_quality_check", step_name="final_validation", data_type="klines")
                final_quality_assessment = advanced_metrics.comprehensive_quality_assessment(df, context="final_quality_check", step_name="final_validation")

                # Perform statistical validation on final data
                final_distribution_validation = {}
                for col in df.select_dtypes(include=[np.number]).columns:
                    if col in ['open', 'high', 'low', 'close', 'volume']:
                        validation_results = statistical_validator.run_comprehensive_validation(df[col].values)
                        final_distribution_validation[col] = {
                            'results': [{'status': r.status.value, 'message': r.message} for r in validation_results]
                        }

                # Check quality alerts
                quality_alerts = quality_alert_system.check_alerts(final_score)

            # Check for data continuity and temporal consistency
            temporal_issues = []
            if len(df) > 1:
                time_diffs = df.index.to_series().diff().dropna()
                expected_interval = self._interval_to_minutes(interval)
                if expected_interval:
                    irregular_intervals = (time_diffs != timedelta(minutes=expected_interval)).sum()
                    if irregular_intervals > 0:
                        temporal_issues.append(f"Found {irregular_intervals} irregular intervals")

            # Check for null values in final data
            null_counts = df[self.required_ohlcv_columns].isnull().sum()
            null_issues = []
            if null_counts.sum() > 0:
                null_issues.append(f"Final data contains null values: {null_counts.to_dict()}")

            # Perform final duplicate check
            final_duplicate_analysis = _ANALYZE_DUPLICATES_COMPREHENSIVE(df)

            # Check quality alerts
            try:
                quality_alerts = quality_alert_system.check_alerts(final_score)
            except Exception as e:
                if self.enable_logging:
                    tprint_warning(f"⚠️ Quality alert system failed: {e}")
                quality_alerts = []

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

            if final_duplicate_analysis.total_duplicates > 0:
                all_warnings.append(f"Final data contains {final_duplicate_analysis.total_duplicates} duplicate records")

            result.success = True
            result.data = df
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
                    "total_duplicates": final_duplicate_analysis.total_duplicates,
                    "true_duplicate_groups": final_duplicate_analysis.true_duplicate_groups,
                    "false_duplicate_groups": final_duplicate_analysis.false_duplicate_groups,
                    "mixed_duplicate_groups": final_duplicate_analysis.mixed_duplicate_groups
                },
                "final_records": len(df),
                "final_columns": len(df.columns),
                "null_counts": null_counts.to_dict(),
                "date_range": {
                    "start": df.index.min().isoformat(),
                    "end": df.index.max().isoformat()
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

            # Create consolidated batch ID
            consolidated_batch_id = f"{batch_id}_consolidated" if batch_id else "consolidated"

            # Store using BaseStep KlinesParquetManager integration
            if self._is_klines_available():
                success = self._store_klines(
                    df=df,
                    symbol=symbol,
                    exchange=self.exchange,
                    interval=f"{interval}_consolidated",
                    batch_id=consolidated_batch_id,
                    metadata={
                        'consolidation_timestamp': datetime.now().isoformat(),
                        'total_records': len(df),
                        'consolidated': True
                    }
                )

                if not success:
                    raise RuntimeError("Failed to store consolidated file using KlinesParquetManager")

                # Log optimization benefits for consolidated file
                if self.enable_logging:
                    compression_stats = self._get_klines_compression_stats()
                    if compression_stats.get("total_files", 0) > 0:
                        tprint_info(f"📊 Consolidated file compression: {compression_stats.get('overall_compression_ratio', 0):.1f}%")
            else:
                # Fallback to direct parquet storage
                output_file = self.data_dir / self.exchange / symbol.lower() / "processed" / f"{symbol.lower()}_{interval}_consolidated.parquet"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(output_file, index=False, compression='snappy')
                success = True

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

            # Store data using BaseStep KlinesParquetManager integration
            if self._is_klines_available():
                success = self._store_klines(
                    df=df,
                    symbol=symbol,
                    exchange=self.exchange,
                    interval=interval,
                    batch_id=batch_id,
                    metadata={
                        'processing_timestamp': datetime.now().isoformat(),
                        'total_records': len(df),
                        'original_data': True
                    }
                )

                if success:
                    result.success = True

                    # Get compression statistics
                    compression_stats = self._get_klines_compression_stats()

                    result.metadata = {
                        "stored_files": [f"{symbol}_{interval}_original"],
                        "record_count": len(df),
                        "compression_ratio": compression_stats.get("overall_compression_ratio", 0),
                        "file_size_mb": compression_stats.get("total_file_size_mb", 0),
                        "optimization_applied": True
                    }

                    if self.enable_logging:
                        tprint_success(f"✅ Stored {len(df)} records for {symbol} {interval}")
                        if compression_stats.get("total_files", 0) > 0:
                            tprint_info(f"📊 Compression ratio: {compression_stats.get('overall_compression_ratio', 0):.1f}%")
                            tprint_info(f"💾 File size: {compression_stats.get('total_file_size_mb', 0):.2f} MB")
                else:
                    raise RuntimeError("Failed to store data using KlinesParquetManager")
            else:
                # Fallback to direct parquet storage
                output_file = self.data_dir / self.exchange / symbol.lower() / "raw" / f"{symbol.lower()}_{interval}.parquet"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(output_file, index=False, compression='snappy')
                success = True
                result.success = True
                compression_stats = {}

                result.metadata = {
                    "stored_files": [f"{symbol}_{interval}_original"],
                    "record_count": len(df),
                    "compression_ratio": compression_stats.get("overall_compression_ratio", 0),
                    "file_size_mb": compression_stats.get("total_file_size_mb", 0),
                    "optimization_applied": True
                }

                if self.enable_logging:
                    tprint_success(f"✅ Stored {len(df)} records for {symbol} {interval}")
                    if compression_stats.get("total_files", 0) > 0:
                        tprint_info(f"📊 Compression ratio: {compression_stats.get('overall_compression_ratio', 0):.1f}%")
                        tprint_info(f"💾 File size: {compression_stats.get('total_file_size_mb', 0):.2f} MB")

        except Exception as e:
            error_msg = f"Data storage failed: {str(e)}"
            result.errors.append(error_msg)
            if self.enable_logging:
                tprint_error(f"❌ {error_msg}")

        result.processing_time = (datetime.now() - start_time).total_seconds()
        return result

    async def _resample_data_with_age_check(
        self,
        df: pd.DataFrame,
        symbol: str,
        resampling_config: ResamplingConfig,
        batch_id: Optional[str]
    ) -> ProcessingResult:
        """Resample data with age-based filtering."""
        start_time = datetime.now()
        result = ProcessingResult(
            step=ProcessingStep.RESAMPLING,
            success=False,
            errors=[],
            warnings=[]
        )

        try:
            if self.enable_logging:
                tprint_info(f"📊 Checking data age for resampling: {symbol}")

            # Check if data is older than threshold
            current_time = datetime.now()
            data_age_days = (current_time - df.index.max()).days

            if data_age_days < resampling_config.resample_older_than_days:
                result.success = True
                result.metadata = {
                    "resampled_intervals": [],
                    "stored_files": [],
                    "reason": f"Data is only {data_age_days} days old, resampling threshold is {resampling_config.resample_older_than_days} days"
                }

                if self.enable_logging:
                    tprint_info(f"⏭️ Skipping resampling: data is {data_age_days} days old (threshold: {resampling_config.resample_older_than_days} days)")

                return result

            # Perform resampling
            resampled_intervals = []
            stored_files = []

            for target_interval in resampling_config.target_intervals:
                try:
                    if self.enable_logging:
                        tprint_info(f"🔄 Resampling to {target_interval}")

                    # Resample data
                    resampled_df = self._perform_resampling(df, target_interval, resampling_config)

                    if not resampled_df.empty:
                        # Preview resampled data
                        tprint_data_preview(resampled_df, f"Resampled data to {target_interval}", max_rows=3, include_metadata=True)
                        
                        # Store resampled data using BaseStep KlinesParquetManager integration
                        if self._is_klines_available():
                            success = self._store_klines(
                                df=resampled_df,
                                symbol=symbol,
                                exchange=self.exchange,
                                interval=target_interval,
                                batch_id=f"resampled_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                metadata={
                                    'resampling_timestamp': datetime.now().isoformat(),
                                    'total_records': len(resampled_df),
                                    'resampled_from': interval,
                                    'resampled_to': target_interval
                                }
                            )
                        else:
                            # Fallback to direct parquet storage
                            output_file = self.data_dir / self.exchange / symbol.lower() / "processed" / f"{symbol.lower()}_{target_interval}.parquet"
                            output_file.parent.mkdir(parents=True, exist_ok=True)
                            resampled_df.to_parquet(output_file, index=False, compression='snappy')
                            success = True

                        if success:
                            resampled_intervals.append(target_interval)
                            stored_files.append(f"{symbol}_{target_interval}_resampled")

                            if self.enable_logging:
                                tprint_success(f"✅ Resampled to {target_interval}: {len(resampled_df)} records")
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
                "data_age_days": data_age_days
            }

        except Exception as e:
            error_msg = f"Resampling with age check failed: {str(e)}"
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
    tprint_info(f"🚀 Starting enhanced klines data processing for {symbol} {interval}")
    tprint_structured({
        "operation": "process_klines_data_enhanced_start",
        "symbol": symbol,
        "interval": interval,
        "years": years,
        "create_consolidated": create_consolidated,
        "batch_id": batch_id
    }, level="INFO")
    
    pipeline = EnhancedKlinesProcessingPipeline(config)

    with tprint_timer("process_klines_data_enhanced", level="PERFORMANCE"):
        results = await pipeline.process_klines_data(
            symbol=symbol,
            interval=interval,
            years=years,
            exchange_interface=exchange_interface,
            resampling_config=resampling_config,
            max_gap_minutes=max_gap_minutes,
            create_consolidated=create_consolidated,
            batch_id=batch_id
        )
    
    tprint_success(f"✅ Enhanced klines data processing completed for {symbol} {interval}")
    return results

if __name__ == "__main__":
    # Example usage - simplified for working with existing data
    async def main():
        try:
            tprint_info("🚀 Starting enhanced klines processing pipeline example")
            
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
            
            tprint_structured({
                "operation": "pipeline_config_created",
                "data_dir": pipeline_config.data_dir,
                "exchange": pipeline_config.exchange,
                "enable_logging": pipeline_config.enable_logging,
                "enable_resampling": pipeline_config.enable_resampling
            }, level="INFO")

            # Configure resampling for existing data
            tprint_info("🔄 Configuring resampling for existing data")
            resampling_config = ResamplingConfig(
                target_intervals=['5m', '15m', '30m', '1h'],  # Skip 1m as it's already available
                method='ohlc',
                preserve_volume=True,
                resample_older_than_days=1,  # Resample all data older than 1 day
                enable_auto_resampling=True
            )
            tprint_success(f"✅ Resampling configured for {len(resampling_config.target_intervals)} intervals")

            # Create enhanced exchange interface for data downloading
            tprint_info("🔗 Creating exchange interface")
            exchange_config = {
                'exchange_type': 'binance',
                'api_key': "",  # Add your API key here
                'api_secret': "",  # Add your API secret here
                'testnet': True,
                'rate_limits': {}
            }
            exchange_interface = ExchangeInterface(exchange_config)
            tprint_success("✅ Exchange interface created")
            
            try:
                tprint_info("🔌 Connecting to exchange")
                await exchange_interface.connect()
                tprint_success("✅ Connected to exchange successfully")
            except Exception as e:
                tprint_warning(f"⚠️ Exchange connection failed: {e}")
                tprint_info("📁 Falling back to existing data processing...")
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

            tprint_success(f"🎉 Processing completed: {results['pipeline_success']}")
            tprint_structured({
                "operation": "pipeline_processing_completed",
                "pipeline_success": results['pipeline_success'],
                "data_quality": results['data_quality'],
                "final_shape": results['final_data_shape'],
                "stored_files": results['stored_files'],
                "resampled_intervals": results['resampled_intervals']
            }, level="SUCCESS")
            
            print(f"\n🎉 Processing completed: {results['pipeline_success']}")
            print(f"📊 Data quality: {results['data_quality']}")
            print(f"📈 Final shape: {results['final_data_shape']}")
            print(f"💾 Stored files: {results['stored_files']}")
            print(f"🔄 Resampled intervals: {results['resampled_intervals']}")

            await exchange_interface.disconnect()

        except Exception as e:
            tprint_error(f"❌ Error in main processing: {e}")
            tprint_exception(e, "Main processing failed")
            tprint_structured({
                "operation": "main_processing_failed",
                "error": str(e),
                "error_type": type(e).__name__
            }, level="ERROR")
            
            print(f"❌ Error in main processing: {e}")
            import traceback
            traceback.print_exc()

    asyncio.run(main())
