"""
Klines Data Collection Module

This module contains scripts and utilities specifically for collecting,
processing, and managing klines (candlestick) data from cryptocurrency exchanges.

Modules:
    - klines_parquet: Handles reading/writing klines data to/from parquet files
    - gap_detector: Detects and fills gaps in klines data
    - historical_data_downloader: Downloads historical klines data from exchanges
    - historical_data_pipeline: Orchestrates the complete klines data pipeline
    - basic_returns_engineer: Processes basic returns and technical features from klines data
    - unified_data_downloader: Centralized download functionality for all data types
    - enhanced_append_data_downloader: Append-based data downloading
    - unified_gap_filler: Unified gap filling functionality
    - unified_resampler: Unified resampling functionality for different timeframes
"""

from .klines_parquet import (
    get_klines_manager,
    get_parquet_utils,
    safe_read_parquet,
    validate_parquet_file,
    safe_read_parquet_with_dtype_normalization,
    repair_parquet_file,
    harmonize_schema_after_read
)
from .gap_detector import GapDetector, detect_and_fill_gaps
from .historical_data_downloader import HistoricalDataDownloader
from .historical_data_pipeline import HistoricalDataPipeline
from .basic_returns_engineer import BasicReturnsEngineer

# Import training step components (optional imports to avoid circular dependencies)
try:
    from .unified_data_downloader import UnifiedDataDownloader
    _training_components_available = True
except ImportError:
    _training_components_available = False

try:
    from .enhanced_append_data_downloader import EnhancedAppendDataDownloader
    _enhanced_downloader_available = True
except ImportError:
    _enhanced_downloader_available = False

try:
    from .unified_gap_filler import UnifiedGapFiller
    _gap_filler_available = True
except ImportError:
    _gap_filler_available = False

try:
    from .unified_resampler import UnifiedResampler
    _resampler_available = True
except ImportError:
    _resampler_available = False

__all__ = [
    # Core klines functionality
    'get_klines_manager',
    'GapDetector',
    'detect_and_fill_gaps',
    'HistoricalDataDownloader',
    'HistoricalDataPipeline',
    'BasicReturnsEngineer',
    # Backward compatibility with parquet_utils
    'get_parquet_utils',
    'safe_read_parquet',
    'validate_parquet_file',
    'safe_read_parquet_with_dtype_normalization',
    'repair_parquet_file',
    'harmonize_schema_after_read'
]

# Add optional components to __all__ if available
if _training_components_available:
    __all__.append('UnifiedDataDownloader')
if _enhanced_downloader_available:
    __all__.append('EnhancedAppendDataDownloader')
if _gap_filler_available:
    __all__.append('UnifiedGapFiller')
if _resampler_available:
    __all__.append('UnifiedResampler')
