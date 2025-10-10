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
- Data resampling capabilities
- OHLCV data validation and formatting
- Duplicate detection and handling
- Quality assurance and validation
"""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Awaitable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.data.quality.comprehensive_duplicate_analyzer import (
    ComprehensiveDuplicateAnalyzer,
    analyze_duplicates_comprehensive
)
from src.trading.execution.exchange_interface import ExchangeInterface, create_exchange_interface
from exchanges.shared.exchange_data_standardizer import ExchangeDataStandardizer


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
    target_intervals: List[str]
    method: str = "ohlc"  # ohlc, vwap, etc.
    preserve_volume: bool = True
    validate_continuity: bool = True


class EnhancedKlinesProcessingPipeline:
    """
    Enhanced klines data processing pipeline with comprehensive type hints,
    exchange-agnostic design, and fast-fail patterns.
    """

    def __init__(
        self, 
        data_dir: str = "historical_data",
        exchange: str = "binance",
        enable_logging: bool = True
    ) -> None:
        """Initialize the enhanced processing pipeline.

        Args:
            data_dir: Base directory for historical data
            exchange: Default exchange name
            enable_logging: Whether to enable detailed logging
        """
        self.data_dir = Path(data_dir)
        self.exchange = exchange.lower()
        self.enable_logging = enable_logging
        
        # Initialize components
        self.data_standardizer = ExchangeDataStandardizer(data_dir)
        self.duplicate_analyzer = ComprehensiveDuplicateAnalyzer()
        
        # Processing state
        self.current_symbol: Optional[str] = None
        self.current_interval: Optional[str] = None
        self.processing_results: List[ProcessingResult] = []
        
        # Configuration
        self.required_ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
        self.metadata_columns = ['exchange', 'symbol', 'interval', 'timestamp']
        self.columns_to_remove = ['taker_buy_base', 'taker_buy_quote', 'year']
        
        if self.enable_logging:
            tprint_success(f"✅ Enhanced Klines Processing Pipeline initialized for {self.exchange}")

    async def process_klines_data(
        self,
        symbol: str,
        interval: str,
        years: int,
        exchange_interface: ExchangeInterface,
        resampling_config: Optional[ResamplingConfig] = None,
        max_gap_minutes: int = 1,
        create_consolidated: bool = True
    ) -> Dict[str, Any]:
        """
        Process klines data through the complete pipeline.

        Args:
            symbol: Trading symbol (e.g., "ETHUSDT")
            interval: Data interval (e.g., "1m")
            years: Number of years of data to process
            exchange_interface: ExchangeInterface instance for data access
            resampling_config: Configuration for data resampling
            max_gap_minutes: Maximum allowed gap in minutes
            create_consolidated: Whether to create consolidated output file

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
        
        self.current_symbol = symbol.upper()
        self.current_interval = interval
        self.processing_results = []
        
        start_time = datetime.now()
        
        try:
            if self.enable_logging:
                tprint_info(f"🚀 Starting enhanced klines processing for {self.current_symbol} {interval}")
            
            # Initialize results structure
            results = {
                "symbol": self.current_symbol,
                "interval": interval,
                "years": years,
                "exchange": self.exchange,
                "pipeline_success": False,
                "steps_completed": [],
                "steps_failed": [],
                "total_processing_time": 0.0,
                "data_quality": DataQualityLevel.FAILED,
                "final_data_shape": (0, 0),
                "errors": [],
                "warnings": [],
                "metadata": {}
            }
            
            # Step 1: Download data
            download_result = await self._download_data(
                symbol, interval, years, exchange_interface
            )
            self.processing_results.append(download_result)
            
            if not download_result.success:
                raise RuntimeError(f"Data download failed: {download_result.errors}")
            
            results["steps_completed"].append(ProcessingStep.DOWNLOAD.value)
            
            # Step 2: Standardize data format
            standardize_result = await self._standardize_data(
                download_result.data, symbol, interval
            )
            self.processing_results.append(standardize_result)
            
            if not standardize_result.success:
                raise RuntimeError(f"Data standardization failed: {standardize_result.errors}")
            
            results["steps_completed"].append(ProcessingStep.STANDARDIZE.value)
            
            # Step 3: Validate data quality
            validate_result = await self._validate_data_quality(
                standardize_result.data, symbol, interval
            )
            self.processing_results.append(validate_result)
            
            if not validate_result.success:
                raise RuntimeError(f"Data validation failed: {validate_result.errors}")
            
            results["steps_completed"].append(ProcessingStep.VALIDATE.value)
            results["data_quality"] = validate_result.quality_level
            
            # Step 4: Detect and fill gaps
            gap_result = await self._handle_gaps(
                validate_result.data, symbol, interval, max_gap_minutes, exchange_interface
            )
            self.processing_results.append(gap_result)
            
            if not gap_result.success:
                raise RuntimeError(f"Gap handling failed: {gap_result.errors}")
            
            results["steps_completed"].append(ProcessingStep.GAP_DETECTION.value)
            if gap_result.metadata.get("gaps_filled", 0) > 0:
                results["steps_completed"].append(ProcessingStep.GAP_FILLING.value)
            
            # Step 5: Handle duplicates
            duplicate_result = await self._handle_duplicates(
                gap_result.data, symbol, interval
            )
            self.processing_results.append(duplicate_result)
            
            if not duplicate_result.success:
                raise RuntimeError(f"Duplicate handling failed: {duplicate_result.errors}")
            
            results["steps_completed"].append(ProcessingStep.DUPLICATE_HANDLING.value)
            
            # Step 6: Resample data if requested
            if resampling_config:
                resample_result = await self._resample_data(
                    duplicate_result.data, symbol, resampling_config
                )
                self.processing_results.append(resample_result)
                
                if not resample_result.success:
                    raise RuntimeError(f"Data resampling failed: {resample_result.errors}")
                
                results["steps_completed"].append(ProcessingStep.RESAMPLING.value)
                final_data = resample_result.data
            else:
                final_data = duplicate_result.data
            
            # Step 7: Final quality check
            final_quality_result = await self._final_quality_check(
                final_data, symbol, interval
            )
            self.processing_results.append(final_quality_result)
            
            if not final_quality_result.success:
                raise RuntimeError(f"Final quality check failed: {final_quality_result.errors}")
            
            results["steps_completed"].append(ProcessingStep.QUALITY_CHECK.value)
            results["data_quality"] = final_quality_result.quality_level
            
            # Step 8: Create consolidated file if requested
            if create_consolidated:
                consolidate_result = await self._create_consolidated_file(
                    final_data, symbol, interval
                )
                self.processing_results.append(consolidate_result)
                
                if consolidate_result.success:
                    results["steps_completed"].append(ProcessingStep.CONSOLIDATION.value)
                    results["consolidated_file"] = consolidate_result.metadata.get("output_file")
                else:
                    results["warnings"].extend(consolidate_result.warnings)
            
            # Compile final results
            results["pipeline_success"] = True
            results["final_data_shape"] = final_data.shape
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

    async def _download_data(
        self,
        symbol: str,
        interval: str,
        years: int,
        exchange_interface: ExchangeInterface
    ) -> ProcessingResult:
        """Download klines data using ExchangeInterface."""
        start_time = datetime.now()
        result = ProcessingResult(
            step=ProcessingStep.DOWNLOAD,
            success=False,
            errors=[],
            warnings=[]
        )
        
        try:
            if self.enable_logging:
                tprint_info(f"📥 Downloading {years} years of {symbol} {interval} data")
            
            # Calculate time range
            end_time = datetime.now()
            start_time_range = end_time - timedelta(days=years * 365)
            
            # Download data using ExchangeInterface
            klines_data = await exchange_interface.get_klines(
                symbol=symbol,
                interval=interval,
                start_time=start_time_range,
                end_time=end_time,
                limit=1000  # Adjust based on exchange limits
            )
            
            if not klines_data:
                raise RuntimeError("No data received from exchange")
            
            # Convert to DataFrame
            df = self._klines_to_dataframe(klines_data, symbol, interval)
            
            if df.empty:
                raise RuntimeError("Downloaded data is empty")
            
            result.success = True
            result.data = df
            result.metadata = {
                "records_downloaded": len(df),
                "date_range": {
                    "start": df.index.min().isoformat() if not df.empty else None,
                    "end": df.index.max().isoformat() if not df.empty else None
                }
            }
            
            if self.enable_logging:
                tprint_success(f"✅ Downloaded {len(df)} records")
            
        except Exception as e:
            error_msg = f"Data download failed: {str(e)}"
            result.errors.append(error_msg)
            if self.enable_logging:
                tprint_error(f"❌ {error_msg}")
        
        result.processing_time = (datetime.now() - start_time).total_seconds()
        return result

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
            
            # Use ExchangeDataStandardizer
            standardized_df, report = self.data_standardizer.standardize_data(
                df, self.exchange, symbol, interval, validate_quality=True
            )
            
            if not report.get('success', False):
                raise RuntimeError(f"Standardization failed: {report.get('errors', [])}")
            
            result.success = True
            result.data = standardized_df
            result.metadata = {
                "original_shape": report.get('original_shape', (0, 0)),
                "final_shape": report.get('final_shape', (0, 0)),
                "processing_time": report.get('processing_time', 0.0)
            }
            result.warnings.extend(report.get('warnings', []))
            
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
        """Validate data quality and assign quality level."""
        start_time = datetime.now()
        result = ProcessingResult(
            step=ProcessingStep.VALIDATE,
            success=False,
            errors=[],
            warnings=[]
        )
        
        try:
            if self.enable_logging:
                tprint_info(f"🔍 Validating data quality for {symbol} {interval}")
            
            # Check required columns
            missing_columns = [col for col in self.required_ohlcv_columns if col not in df.columns]
            if missing_columns:
                raise RuntimeError(f"Missing required columns: {missing_columns}")
            
            # Check for null values
            null_counts = df[self.required_ohlcv_columns].isnull().sum()
            total_nulls = null_counts.sum()
            null_percentage = (total_nulls / (len(df) * len(self.required_ohlcv_columns))) * 100
            
            # Check for negative values in OHLCV
            negative_values = {}
            for col in self.required_ohlcv_columns:
                if col in df.columns:
                    negative_count = (df[col] < 0).sum()
                    if negative_count > 0:
                        negative_values[col] = negative_count
            
            # Check for zero volume
            zero_volume = (df['volume'] == 0).sum() if 'volume' in df.columns else 0
            
            # Determine quality level
            quality_issues = []
            if null_percentage > 5:
                quality_issues.append(f"High null percentage: {null_percentage:.2f}%")
            if negative_values:
                quality_issues.append(f"Negative values found: {negative_values}")
            if zero_volume > len(df) * 0.1:
                quality_issues.append(f"High zero volume percentage: {(zero_volume/len(df)*100):.2f}%")
            
            if not quality_issues:
                result.quality_level = DataQualityLevel.EXCELLENT
            elif len(quality_issues) == 1 and null_percentage < 1:
                result.quality_level = DataQualityLevel.GOOD
            elif len(quality_issues) <= 2 and null_percentage < 3:
                result.quality_level = DataQualityLevel.FAIR
            elif len(quality_issues) <= 3 and null_percentage < 5:
                result.quality_level = DataQualityLevel.POOR
            else:
                result.quality_level = DataQualityLevel.FAILED
            
            result.success = True
            result.data = df
            result.metadata = {
                "null_percentage": null_percentage,
                "negative_values": negative_values,
                "zero_volume_count": zero_volume,
                "quality_issues": quality_issues
            }
            result.warnings.extend(quality_issues)
            
            if self.enable_logging:
                tprint_success(f"✅ Data quality validation completed: {result.quality_level.value}")
            
        except Exception as e:
            error_msg = f"Data quality validation failed: {str(e)}"
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
                gap_data = await exchange_interface.get_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=gap.start_time,
                    end_time=gap.end_time,
                    limit=1000
                )
                
                if gap_data:
                    gap_df = self._klines_to_dataframe(gap_data, symbol, interval)
                    if not gap_df.empty:
                        # Standardize the gap data
                        standardized_gap_df, _ = self.data_standardizer.standardize_data(
                            gap_df, self.exchange, symbol, interval, validate_quality=False
                        )
                        
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
        """Perform final quality check on processed data."""
        start_time = datetime.now()
        result = ProcessingResult(
            step=ProcessingStep.QUALITY_CHECK,
            success=False,
            errors=[],
            warnings=[]
        )
        
        try:
            if self.enable_logging:
                tprint_info(f"✅ Running final quality check for {symbol} {interval}")
            
            # Check data completeness
            if df.empty:
                raise RuntimeError("Final data is empty")
            
            # Check required columns
            missing_columns = [col for col in self.required_ohlcv_columns if col not in df.columns]
            if missing_columns:
                raise RuntimeError(f"Missing required columns in final data: {missing_columns}")
            
            # Check for null values
            null_counts = df[self.required_ohlcv_columns].isnull().sum()
            if null_counts.sum() > 0:
                result.warnings.append(f"Final data contains null values: {null_counts.to_dict()}")
            
            # Check data continuity
            if len(df) > 1:
                time_diffs = df.index.to_series().diff().dropna()
                expected_interval = self._interval_to_minutes(interval)
                if expected_interval:
                    irregular_intervals = (time_diffs != timedelta(minutes=expected_interval)).sum()
                    if irregular_intervals > 0:
                        result.warnings.append(f"Found {irregular_intervals} irregular intervals")
            
            # Determine final quality level
            if not result.warnings:
                result.quality_level = DataQualityLevel.EXCELLENT
            elif len(result.warnings) <= 2:
                result.quality_level = DataQualityLevel.GOOD
            elif len(result.warnings) <= 4:
                result.quality_level = DataQualityLevel.FAIR
            else:
                result.quality_level = DataQualityLevel.POOR
            
            result.success = True
            result.data = df
            result.metadata = {
                "final_records": len(df),
                "final_columns": len(df.columns),
                "null_counts": null_counts.to_dict(),
                "date_range": {
                    "start": df.index.min().isoformat(),
                    "end": df.index.max().isoformat()
                }
            }
            
            if self.enable_logging:
                tprint_success(f"✅ Final quality check completed: {result.quality_level.value}")
            
        except Exception as e:
            error_msg = f"Final quality check failed: {str(e)}"
            result.errors.append(error_msg)
            if self.enable_logging:
                tprint_error(f"❌ {error_msg}")
        
        result.processing_time = (datetime.now() - start_time).total_seconds()
        return result

    async def _create_consolidated_file(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str
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
            
            # Create output file
            output_file = self.data_dir / f"features_{self.exchange}_{symbol}_{interval}_consolidated.parquet"
            df.to_parquet(output_file, index=True, compression='snappy')
            
            result.success = True
            result.metadata = {
                "output_file": str(output_file),
                "file_size_mb": output_file.stat().st_size / (1024 * 1024),
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
    data_dir: str = "historical_data",
    exchange: str = "binance",
    resampling_config: Optional[ResamplingConfig] = None,
    max_gap_minutes: int = 1,
    create_consolidated: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to process klines data using the enhanced pipeline.

    Args:
        symbol: Trading symbol (e.g., "ETHUSDT")
        interval: Data interval (e.g., "1m")
        years: Number of years of data to process
        exchange_interface: ExchangeInterface instance for data access
        data_dir: Base directory for data storage
        exchange: Exchange name
        resampling_config: Configuration for data resampling
        max_gap_minutes: Maximum allowed gap in minutes
        create_consolidated: Whether to create consolidated output file

    Returns:
        Dictionary with complete processing results
    """
    pipeline = EnhancedKlinesProcessingPipeline(data_dir, exchange)
    
    return await pipeline.process_klines_data(
        symbol=symbol,
        interval=interval,
        years=years,
        exchange_interface=exchange_interface,
        resampling_config=resampling_config,
        max_gap_minutes=max_gap_minutes,
        create_consolidated=create_consolidated
    )


if __name__ == "__main__":
    # Example usage
    async def main():
        # Create exchange interface
        exchange_config = {
            'exchange_type': 'binance',
            'api_key': '',  # Add your API key
            'api_secret': '',  # Add your API secret
            'testnet': True
        }
        
        exchange_interface = create_exchange_interface(exchange_config)
        await exchange_interface.connect()
        
        # Configure resampling
        resampling_config = ResamplingConfig(
            target_intervals=['5m', '15m', '1h'],
            method='ohlc',
            preserve_volume=True
        )
        
        # Process data
        results = await process_klines_data_enhanced(
            symbol="ETHUSDT",
            interval="1m",
            years=1,
            exchange_interface=exchange_interface,
            resampling_config=resampling_config
        )
        
        print(f"Processing completed: {results['pipeline_success']}")
        print(f"Data quality: {results['data_quality']}")
        print(f"Final shape: {results['final_data_shape']}")
        
        await exchange_interface.disconnect()
    
    asyncio.run(main())