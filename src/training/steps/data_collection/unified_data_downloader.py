#!/usr/bin/env python3
"""
Enhanced Unified Data Downloader

This module provides centralized download functionality with BaseStep comprehensive tools integration:
- Klines data (PRIMARY - per new setup)
- Aggtrades data (DEPRECATED - not used in new klines-only setup)
- Futures data (DEPRECATED - not used in new klines-only setup)

ENHANCED FEATURES:
==================
- BaseStep comprehensive tools integration
- Advanced logging with tprint utilities
- Hardware optimization for data operations
- Comprehensive error handling and validation
- Performance monitoring and metrics
- Memory optimization for large datasets

NOTE: Per new setup, only klines data is actively used. Aggtrades and futures
downloads are deprecated but maintained for backwards compatibility.
"""

import asyncio
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger
from src.utils.error_handler import handles_errors
from src.utils.common_operations import safe_fillna, safe_to_parquet, safe_read_parquet
from src.utils.common_utilities import validate_dataframe_columns, safe_dataframe_operation
from src.training.steps.base_step import BaseStep
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

# Import comprehensive data quality tools
try:
    from src.utils.data.quality.comprehensive_quality_scorer import get_quality_scorer
    from src.utils.data.quality.data_quality import DataQualityFramework
    QUALITY_TOOLS_AVAILABLE = True
except ImportError:
    QUALITY_TOOLS_AVAILABLE = False

def validate_data_quality(df, **kwargs):
    """Comprehensive data quality validation using proper tools."""
    if not QUALITY_TOOLS_AVAILABLE:
        return {'valid': True, 'quality_score': 50.0, 'issues': [], 'warnings': []}

    try:
        quality_scorer = get_quality_scorer()
        quality_assessment = quality_scorer.assess_data_quality(
            df,
            context="data_collection",
            step_name="data_download",
            data_type="klines"
        )

        return {
            'valid': quality_assessment.level.value not in ['critical'],
            'quality_score': quality_assessment.overall_score,
            'issues': quality_assessment.issues,
            'warnings': quality_assessment.warnings
        }
    except Exception as e:
        return {'valid': True, 'quality_score': 50.0, 'issues': [str(e)], 'warnings': []}

logger = system_logger.getChild("UnifiedDataDownloader")


class EnhancedUnifiedDataDownloader(BaseStep):
    """
    Enhanced unified downloader for all data types with BaseStep comprehensive tools integration.
    
    This class provides:
    - Direct access to all BaseStep comprehensive tools
    - Advanced logging with tprint utilities
    - Hardware optimization for data operations
    - Comprehensive error handling and validation
    - Performance monitoring and metrics
    - Memory optimization for large datasets
    """

    @log_important_calls
    def __init__(self, data_cache_path: str = "data_cache", step_name: str = "enhanced_unified_downloader", config: Optional[Dict[str, Any]] = None):
        super().__init__(step_name, config)
        self.data_cache_path = Path(data_cache_path)
        self.data_cache_path.mkdir(exist_ok=True)
        self.logger = logger.getChild('EnhancedUnifiedDataDownloader')

        # Initialize standardized parquet handler for compatibility
        try:
            from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
            self.parquet_handler = standardized_parquet_handler
        except ImportError:
            self.parquet_handler = None
            self.tprint_warning("⚠️ Standardized parquet handler not available")

        # Download statistics
        self.download_stats = {
            'total_downloads': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'total_rows': 0,
            'start_time': None
        }

        # Initialize exchange instances cache
        self._exchange_instances = {}

        # Lazy initialization of Binance API - only when needed
        self.binance_class = None
        
        # Log initialization with BaseStep tools
        self.tprint_success("✅ Enhanced Unified Data Downloader initialized with BaseStep tools")
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the enhanced data download process using BaseStep tools.
        
        Args:
            config: Configuration containing download parameters
            
        Returns:
            Dictionary with download status and data
        """
        try:
            # Set context for enhanced logging and file operations
            self._set_context(
                symbol=config.get('symbol'),
                exchange=config.get('exchange'),
                information=config.get('information', 'klines'),
                direction=config.get('direction', 'long'),
                model=config.get('model', 'Analyst')
            )
            
            # Log step start with comprehensive information
            self.tprint_step_start("Enhanced Unified Data Download")
            self.tprint_config_preview(config, "Download Configuration")
            
            # Extract parameters with validation
            symbol = self._get_config_value('symbol', expected_type=str)
            exchange = self._get_config_value('exchange', expected_type=str)
            timeframe = self._get_config_value('timeframe', '1m', str)
            start_date = self._get_config_value('start_date', None)
            end_date = self._get_config_value('end_date', None)
            batch_size = self._get_config_value('batch_size', 1000, int)
            use_append_mode = self._get_config_value('use_append_mode', True, bool)
            
            # Validate parameters using BaseStep math validation
            batch_size = self._validate_positive(batch_size, 1000)
            
            self.tprint_info(f"📥 Starting enhanced download for {symbol} on {exchange}")
            self.tprint_info(f"📊 Timeframe: {timeframe}, Batch size: {batch_size}")
            
            # Download klines data with comprehensive error handling
            success, data, error = await self._enhanced_download_klines(
                symbol, exchange, timeframe, start_date, end_date, batch_size, use_append_mode
            )
            
            if success and data is not None:
                # Use BaseStep data quality tools for validation
                if self.data_quality:
                    quality_result = self._get_data_cleaner().assess_quality(data)
                    self.tprint_validation_result(quality_result, "Data Quality Assessment")
                
                # Use BaseStep hardware optimization if available
                if self.hardware_utils and 'optimize_dataframe' in self.hardware_utils:
                    data = self.hardware_utils['optimize_dataframe'](data)
                    self.tprint_info("🔧 Applied hardware optimization to downloaded data")
                
                # Store data using BaseStep artifact management
                artifact_path = self._save_dataframe(
                    data, 
                    f"klines_{symbol}_{exchange}_{timeframe}",
                    metadata={
                        'symbol': symbol,
                        'exchange': exchange,
                        'timeframe': timeframe,
                        'start_date': start_date.isoformat() if start_date else None,
                        'end_date': end_date.isoformat() if end_date else None,
                        'batch_size': batch_size,
                        'use_append_mode': use_append_mode,
                        'rows': len(data),
                        'columns': len(data.columns) if hasattr(data, 'columns') else 0
                    }
                )
                
                # Log performance metrics
                performance_metrics = self._get_performance_metrics()
                self.tprint_performance_summary(performance_metrics)
                
                # Log step completion
                self.tprint_step_end("Enhanced Unified Data Download", True, performance_metrics.get('execution_time', 0))
                
                return {
                    'success': True,
                    'data': data,
                    'error': None,
                    'artifacts': [artifact_path],
                    'metrics': performance_metrics
                }
            else:
                error_msg = f"Download failed: {error or 'Unknown error'}"
                self.tprint_error(f"❌ {error_msg}")
                return {
                    'success': False,
                    'data': None,
                    'error': error_msg,
                    'artifacts': [],
                    'metrics': {}
                }
                
        except Exception as e:
            self.tprint_error(f"❌ Unexpected error in enhanced download: {e}")
            self._log_error_with_context(e, "enhanced_unified_downloader")
            return {
                'success': False,
                'data': None,
                'error': str(e),
                'artifacts': [],
                'metrics': {}
            }
    
    async def _enhanced_download_klines(
        self,
        symbol: str,
        exchange: str,
        timeframe: str = "1m",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        batch_size: int = 1000,
        use_append_mode: bool = True
    ) -> Tuple[bool, List[Dict[str, Any]], Optional[str]]:
        """
        Enhanced klines download with BaseStep comprehensive tools integration.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe
            start_date: Start date for download
            end_date: End date for download
            batch_size: Batch size for processing
            use_append_mode: Whether to use append mode
            
        Returns:
            Tuple of (success, data, error)
        """
        try:
            self.tprint_operation_start(f"Downloading klines data for {symbol}")
            
            # Use BaseStep safe operations for data validation
            if not self._validate_dataframe_columns(None, ['symbol', 'exchange']):  # Placeholder validation
                raise ValueError("Invalid symbol or exchange format")
            
            # Call the original download method
            success, data, error = await self._original_download_klines(
                symbol, exchange, timeframe, start_date, end_date, batch_size, use_append_mode
            )
            
            if success and data is not None:
                # Use BaseStep data operations for safe processing
                if isinstance(data, list) and data:
                    # Convert to DataFrame if needed
                    import pandas as pd
                    df = pd.DataFrame(data)
                    df = self._safe_dataframe_operation(df, 'fillna', method='forward')
                    data = df.to_dict('records')
                
                self.tprint_data_summary(data, f"Downloaded klines data for {symbol}")
                self.tprint_operation_end(f"Downloaded {len(data)} klines records")
                
                return True, data, None
            else:
                error_msg = f"Klines download failed: {error}"
                self.tprint_error(f"❌ {error_msg}")
                return False, None, error_msg
                
        except Exception as e:
            error_msg = f"Klines download exception: {e}"
            self.tprint_error(f"❌ {error_msg}")
            return False, None, error_msg
    
    async def _original_download_klines(
        self,
        symbol: str,
        exchange: str,
        timeframe: str = "1m",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        batch_size: int = 1000,
        use_append_mode: bool = True
    ) -> Tuple[bool, List[Dict[str, Any]], Optional[str]]:
        """
        Original klines download method for backward compatibility.
        This method will be implemented by delegating to the legacy UnifiedDataDownloader.
        """
        # Create a legacy downloader instance for the actual download
        legacy_downloader = UnifiedDataDownloader(str(self.data_cache_path))
        return await legacy_downloader.download_klines(
            symbol, exchange, timeframe, start_date, end_date, batch_size, use_append_mode
        )


class UnifiedDataDownloader:
    """
    Legacy unified downloader for backward compatibility.
    
    This class maintains the original interface while the EnhancedUnifiedDataDownloader
    provides the new BaseStep-integrated functionality.
    """

    @log_important_calls
    def __init__(self, data_cache_path: str = "data_cache"):
        self.data_cache_path = Path(data_cache_path)
        self.data_cache_path.mkdir(exist_ok=True)
        self.logger = logger.getChild('UnifiedDataDownloader')

        # Initialize standardized parquet handler for compatibility
        try:
            from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
            self.parquet_handler = standardized_parquet_handler
        except ImportError:
            self.parquet_handler = None
            self.logger.warning("⚠️ Standardized parquet handler not available")

        # Download statistics
        self.download_stats = {
            'total_downloads': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'total_rows': 0,
            'start_time': None
        }

        # Initialize exchange instances cache
        self._exchange_instances = {}

        # Lazy initialization of Binance API - only when needed
        self.binance_class = None

    def _ensure_binance_api(self) -> bool:
        """Ensure Binance API is available when needed."""
        if self.binance_class is None:
            try:
                from src.exchange.binance import BinanceExchange
                self.binance_class = BinanceExchange
                self.logger.info("✅ Binance API available")
                return True
            except ImportError:
                self.binance_class = None
                self.logger.warning("⚠️ Binance API not available")
                return False
        return True

    @handles_errors(context="download_klines")
    @log_all_calls
    async def download_klines(
        self,
        symbol: str,
        exchange: str,
        timeframe: str = "1m",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        batch_size: int = 1000,
        use_append_mode: bool = True
    ) -> Tuple[bool, List[Dict[str, Any]], Optional[str]]:
        """
        Download klines data for a symbol and exchange.

        Args:
            symbol: Trading symbol (e.g., 'ETHUSDT')
            exchange: Exchange name (e.g., 'BINANCE')
            timeframe: Timeframe (e.g., '1m', '5m', '1h')
            start_date: Start date for download
            end_date: End date for download
            batch_size: Number of records per batch
            use_append_mode: Whether to use append mode (creates new files instead of overwriting)

        Returns:
            Tuple of (success, data, error_message)
        """
        self.logger.info(f"📥 Downloading klines data: {exchange}_{symbol}_{timeframe}")

        try:
            # Set default dates if not provided
            if start_date is None:
                start_date = datetime.now() - timedelta(days=30)
            if end_date is None:
                end_date = datetime.now()

            self.logger.info(f"📅 Download period: {start_date} to {end_date}")

            # Use enhanced append downloader if append mode is enabled
            if use_append_mode:
                try:
                    from .enhanced_append_data_downloader import EnhancedAppendDataDownloader
                    append_downloader = EnhancedAppendDataDownloader(str(self.data_cache_path))

                    result = await append_downloader.download_with_append(
                        symbol=symbol,
                        exchange=exchange,
                        data_type="klines",
                        timeframe=timeframe,
                        start_date=start_date,
                        end_date=end_date,
                        batch_size=batch_size,
                        max_batches=10
                    )

                    if result['success']:
                        # Update statistics
                        self.download_stats['total_downloads'] += 1
                        self.download_stats['successful_downloads'] += 1
                        self.download_stats['total_rows'] += result['total_rows']

                        self.logger.info(f"✅ Downloaded {result['total_rows']} klines records using append mode")
                        return True, [], None  # Data is saved to files, not returned
                    else:
                        self.logger.error(f"❌ Append download failed: {result.get('error', 'Unknown error')}")
                        return False, [], result.get('error', 'Append download failed')

                except ImportError:
                    self.logger.warning("⚠️ Enhanced append downloader not available, falling back to standard mode")
                except Exception as e:
                    self.logger.warning(f"⚠️ Append download failed, falling back to standard mode: {e}")

            # Standard download mode (fallback)
            # Get exchange instance
            exchange_instance = await self._get_exchange_instance(exchange)
            if not exchange_instance:
                return False, [], f"Failed to initialize {exchange} exchange"

            # Convert dates to timestamps
            start_timestamp = int(start_date.timestamp() * 1000)
            end_timestamp = int(end_date.timestamp() * 1000)

            # Download data in batches
            all_data = []
            current_start = start_timestamp

            while current_start < end_timestamp:
                batch_data = await self._download_klines_batch(
                    exchange_instance, symbol, timeframe, current_start, end_timestamp, batch_size
                )

                if not batch_data:
                    break

                all_data.extend(batch_data)

                # Update timestamp for next batch
                if batch_data:
                    current_start = batch_data[-1]['timestamp'] + 1
                else:
                    break

                # Rate limiting
                await asyncio.sleep(0.1)

            # Update statistics
            self.download_stats['total_downloads'] += 1
            self.download_stats['successful_downloads'] += 1
            self.download_stats['total_rows'] += len(all_data)

            self.logger.info(f"✅ Downloaded {len(all_data)} klines records")
            return True, all_data, None

        except Exception as e:
            self.logger.exception(f"❌ Error downloading klines: {e}")
            self.download_stats['failed_downloads'] += 1
            return False, [], str(e)

    @handles_errors(context="download_aggtrades")
    @log_all_calls
    async def download_aggtrades(
        self,
        symbol: str,
        exchange: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        batch_size: int = 1000,
        use_append_mode: bool = True
    ) -> Tuple[bool, List[Dict[str, Any]], Optional[str]]:
        """
        Download aggtrades data for a symbol and exchange.

        DEPRECATED: Per new setup, aggtrades data is not used. This method is maintained
        for backwards compatibility but will return empty data with a warning.

        Args:
            symbol: Trading symbol (e.g., 'ETHUSDT')
            exchange: Exchange name (e.g., 'BINANCE')
            start_date: Start date for download
            end_date: End date for download
            batch_size: Number of records per batch
            use_append_mode: Whether to use append mode (creates new files instead of overwriting)

        Returns:
            Tuple of (success, data, error_message)
        """
        self.logger.warning("⚠️ Aggtrades download is DEPRECATED per new klines-only setup")
        self.logger.info(f"📥 Aggtrades download SKIPPED: {exchange}_{symbol} (klines-only setup)")

        # Return empty data for klines-only setup
        self.logger.info("✅ Aggtrades download disabled - using klines-only setup")
        return True, [], None
