"""
Enhanced Data Downloader for Training Steps

This module provides a unified interface for downloading data using the centralized 
unified downloader with comprehensive type safety, error handling, and BaseStep 
comprehensive tools integration.

ENHANCED FEATURES:
==================
- BaseStep comprehensive tools integration
- Advanced logging with tprint utilities
- Hardware optimization for data operations
- Comprehensive error handling and validation
- Performance monitoring and metrics
- Memory optimization for large datasets
"""

from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

from src.config import CONFIG
from src.training.steps.base_step import BaseStep
from src.utils.tprint import tprint_info, tprint_success, tprint_error, tprint_warning
from src.utils.error_handler import handles_errors as utils_handles_errors

# Import our custom types
from ..types import (
    StepConfig, ExecutionResult, DataLoadResult, ValidationResult,
    DataLoadError, ConfigurationError, validate_config, create_error_result, create_success_result
)

# Import the unified downloader
from .unified_data_downloader import UnifiedDataDownloader


class EnhancedDataDownloader(BaseStep):
    """
    Enhanced data downloader that inherits from BaseStep for comprehensive tool access.
    
    This class provides:
    - Direct access to all BaseStep comprehensive tools
    - Advanced logging with tprint utilities
    - Hardware optimization for data operations
    - Comprehensive error handling and validation
    - Performance monitoring and metrics
    - Memory optimization for large datasets
    """
    
    def __init__(self, step_name: str = "enhanced_data_downloader", config: Optional[StepConfig] = None):
        super().__init__(step_name, config)
        self.downloader = None
        self._initialize_downloader()
    
    def _initialize_downloader(self) -> None:
        """Initialize the unified data downloader with error handling."""
        try:
            data_dir = self._get_config_value('data_dir', 'data_cache')
            self.downloader = UnifiedDataDownloader(data_dir)
            self.tprint_success("✅ UnifiedDataDownloader initialized with BaseStep tools")
        except Exception as e:
            self.tprint_error(f"❌ Failed to initialize UnifiedDataDownloader: {e}")
            raise DataLoadError(f"Failed to initialize UnifiedDataDownloader: {e}") from e
    
    def _validate_symbol_and_exchange(self, symbol: str, exchange: str) -> bool:
        """
        Validate symbol and exchange parameters.
        
        Args:
            symbol: Trading symbol to validate
            exchange: Exchange name to validate
            
        Returns:
            True if both parameters are valid, False otherwise
        """
        try:
            # Validate symbol
            if not symbol or not isinstance(symbol, str):
                self.tprint_error(f"❌ Invalid symbol: {symbol}")
                return False
            
            # Check symbol format (basic validation)
            symbol = symbol.strip().upper()
            if len(symbol) < 2 or len(symbol) > 20:
                self.tprint_error(f"❌ Symbol length invalid: {symbol} (length: {len(symbol)})")
                return False
            
            # Check for valid characters (letters, numbers, and common separators)
            import re
            if not re.match(r'^[A-Z0-9/_-]+$', symbol):
                self.tprint_error(f"❌ Symbol contains invalid characters: {symbol}")
                return False
            
            # Validate exchange
            if not exchange or not isinstance(exchange, str):
                self.tprint_error(f"❌ Invalid exchange: {exchange}")
                return False
            
            # Check exchange format
            exchange = exchange.strip().lower()
            if len(exchange) < 2 or len(exchange) > 20:
                self.tprint_error(f"❌ Exchange length invalid: {exchange} (length: {len(exchange)})")
                return False
            
            # Check for valid characters (letters, numbers, and common separators)
            if not re.match(r'^[a-z0-9_-]+$', exchange):
                self.tprint_error(f"❌ Exchange contains invalid characters: {exchange}")
                return False
            
            # Validate against known exchanges (basic check)
            valid_exchanges = {
                'binance', 'coinbase', 'kraken', 'bitfinex', 'huobi', 'okx', 'bybit',
                'kucoin', 'gate', 'mexc', 'bitget', 'crypto.com', 'binance.us',
                'coinbase_pro', 'gemini', 'bitstamp', 'bittrex', 'poloniex'
            }
            
            if exchange not in valid_exchanges:
                self.tprint_warning(f"⚠️ Unknown exchange: {exchange} (proceeding anyway)")
            
            self.tprint_debug(f"✅ Validated symbol '{symbol}' and exchange '{exchange}'")
            return True
            
        except Exception as e:
            self.tprint_error(f"❌ Validation error: {e}")
            return False
    
    async def execute(self, config: StepConfig) -> ExecutionResult:
        """
        Execute the enhanced data download process.
        
        Args:
            config: Step configuration containing download parameters
            
        Returns:
            ExecutionResult with download status and data
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
            self.tprint_step_start("Enhanced Data Download")
            self.tprint_config_preview(config, "Download Configuration")
            
            # Validate configuration using BaseStep tools
            self._validate_config_common()
            
            # Extract parameters with validation
            symbol = self._get_config_value('symbol', expected_type=str)
            exchange_name = self._get_config_value('exchange', expected_type=str)
            interval = self._get_config_value('interval', '1m', str)
            lookback_years = self._get_config_value('lookback_years', 2, int)
            
            # Validate parameters using BaseStep math validation
            lookback_years = self._validate_positive(lookback_years, 2)
            
            self.tprint_info(f"📥 Starting enhanced data download for {symbol} on {exchange_name}")
            self.tprint_info(f"📊 Interval: {interval}, Lookback: {lookback_years} years")
            
            # Calculate date range with validation
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_years * 365)
            
            self.tprint_info(f"📅 Date range: {start_date} to {end_date}")
            
            # Download klines data with comprehensive error handling
            klines_result = await self._download_klines_data(
                symbol, exchange_name, interval, start_date, end_date
            )
            
            if klines_result['success']:
                klines_data = klines_result['data']
                
                # Use BaseStep data quality tools for validation
                if self.data_quality:
                    quality_result = self._get_data_cleaner().assess_quality(klines_data)
                    self.tprint_validation_result(quality_result, "Data Quality Assessment")
                
                # Use BaseStep hardware optimization if available
                if self.hardware_utils and 'optimize_dataframe' in self.hardware_utils:
                    klines_data = self.hardware_utils['optimize_dataframe'](klines_data)
                    self.tprint_info("🔧 Applied hardware optimization to klines data")
                
                # Store data using BaseStep artifact management
                artifact_path = self._save_dataframe(
                    klines_data, 
                    f"klines_{symbol}_{exchange_name}_{interval}",
                    metadata={
                        'symbol': symbol,
                        'exchange': exchange_name,
                        'interval': interval,
                        'start_date': start_date.isoformat(),
                        'end_date': end_date.isoformat(),
                        'lookback_years': lookback_years,
                        'rows': len(klines_data),
                        'columns': len(klines_data.columns) if hasattr(klines_data, 'columns') else 0
                    }
                )
                
                # Log performance metrics
                performance_metrics = self._get_performance_metrics()
                self.tprint_performance_summary(performance_metrics)
                
                # Log step completion
                self.tprint_step_end("Enhanced Data Download", True, performance_metrics.get('execution_time', 0))
                
                return create_success_result(
                    data=klines_data,
                    source="enhanced_data_downloader",
                    rows=len(klines_data),
                    columns=len(klines_data.columns) if hasattr(klines_data, 'columns') else 0,
                    artifacts=[artifact_path]
                )
            else:
                error_msg = f"Klines download failed: {klines_result.get('error', 'Unknown error')}"
                self.tprint_error(f"❌ {error_msg}")
                return create_error_result(DataLoadError(error_msg), "enhanced_data_downloader")
                
        except Exception as e:
            self.tprint_error(f"❌ Unexpected error in enhanced data download: {e}")
            self._log_error_with_context(e, "enhanced_data_downloader")
            return create_error_result(DataLoadError(f"Enhanced data download failed: {e}"), "enhanced_data_downloader")
    
    async def _download_klines_data(
        self, 
        symbol: str, 
        exchange_name: str, 
        interval: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Download klines data with comprehensive error handling and logging.
        
        Args:
            symbol: Trading symbol
            exchange_name: Exchange name
            interval: Data interval
            start_date: Start date for download
            end_date: End date for download
            
        Returns:
            Dictionary with success status, data, and error information
        """
        try:
            self.tprint_operation_start(f"Downloading klines data for {symbol}")
            
            # Validate input parameters
            if not self._validate_symbol_and_exchange(symbol, exchange_name):
                raise DataLoadError(f"Invalid symbol '{symbol}' or exchange '{exchange_name}' format")
            
            # Download klines data
            klines_success, klines_data, klines_error = await self.downloader.download_klines(
                symbol, exchange_name, interval, start_date, end_date
            )
            
            if klines_success and klines_data is not None:
                # Use BaseStep data operations for safe processing
                klines_data = self._safe_dataframe_operation(klines_data, 'fillna', method='forward')
                
                self.tprint_data_summary(klines_data, f"Downloaded klines data for {symbol}")
                self.tprint_operation_end(f"Downloaded {len(klines_data)} klines records")
                
                return {
                    'success': True,
                    'data': klines_data,
                    'error': None
                }
            else:
                error_msg = f"Klines download failed: {klines_error}"
                self.tprint_error(f"❌ {error_msg}")
                return {
                    'success': False,
                    'data': None,
                    'error': error_msg
                }
                
        except Exception as e:
            error_msg = f"Klines download exception: {e}"
            self.tprint_error(f"❌ {error_msg}")
            return {
                'success': False,
                'data': None,
                'error': error_msg
            }


# Legacy function for backward compatibility
@utils_handles_errors(fallback=False)
async def download_all_data_with_consolidation(
    symbol: str,
    exchange_name: str,
    interval: str = "1m",
    data_dir: Optional[str] = None,
) -> DataLoadResult:
    """
    Unified entrypoint used by training steps to download raw data.

    Uses the centralized unified downloader for all data types with comprehensive
    error handling and type safety.

    Args:
        symbol: Trading symbol (e.g., 'ETHUSDT')
        exchange_name: Exchange name (e.g., 'binance')
        interval: Data interval (e.g., '1m', '15m')
        data_dir: Optional data directory path

    Returns:
        DataLoadResult containing success status, data, and error information

    Raises:
        DataLoadError: If data loading fails
        ConfigurationError: If configuration is invalid
    """
    try:
        # Validate input parameters
        if not symbol or not isinstance(symbol, str):
            raise DataLoadError("Symbol must be a non-empty string")
        if not exchange_name or not isinstance(exchange_name, str):
            raise DataLoadError("Exchange name must be a non-empty string")
        if not interval or not isinstance(interval, str):
            raise DataLoadError("Interval must be a non-empty string")

        tprint_info(f"📥 Starting data download for {symbol} on {exchange_name}")

        # Get lookback years from config
        lookback_years: int = 2
        try:
            if isinstance(CONFIG, dict):
                model_training_cfg: Optional[Dict[str, Any]] = CONFIG.get("MODEL_TRAINING")
                if model_training_cfg and isinstance(model_training_cfg.get("lookback_years"), int):
                    lookback_years = int(model_training_cfg["lookback_years"])
        except Exception as e:
            tprint_warning(f"⚠️ Failed to get lookback_years from config: {e}, using default: {lookback_years}")

        # Initialize downloader
        try:
            downloader = UnifiedDataDownloader(data_dir or "data_cache")
            tprint_success("✅ UnifiedDataDownloader initialized")
        except Exception as e:
            raise DataLoadError(f"Failed to initialize UnifiedDataDownloader: {e}") from e

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_years * 365)

        tprint_info(f"📅 Date range: {start_date} to {end_date}")
        tprint_info(f"📊 Lookback period: {lookback_years} years")

        # Download klines data
        klines_success = False
        klines_data = None
        klines_error = None

        try:
            tprint_info(f"📈 Downloading klines data for {symbol}...")
            klines_success, klines_data, klines_error = await downloader.download_klines(
                symbol, exchange_name, interval, start_date, end_date
            )
            
            if klines_success and klines_data is not None:
                tprint_success(f"✅ Downloaded {len(klines_data)} klines records")
            else:
                tprint_error(f"❌ Klines download failed: {klines_error}")
                
        except Exception as e:
            klines_error = f"Klines download exception: {e}"
            tprint_error(f"❌ Klines download exception: {e}")

        # Skip aggtrades download as per new setup
        tprint_info(f"⚠️ Skipping aggtrades download for {symbol} - aggtrades downloads disabled")

        # Determine overall success
        if klines_success and klines_data is not None:
            tprint_success(f"📊 Data download completed successfully: {len(klines_data)} records")
            return create_success_result(
                data=klines_data,
                source="unified_downloader",
                rows=len(klines_data),
                columns=len(klines_data.columns) if hasattr(klines_data, 'columns') else 0
            )
        else:
            error_msg = f"Data download failed: {klines_error or 'Unknown error'}"
            tprint_error(f"❌ {error_msg}")
            return create_error_result(DataLoadError(error_msg), "download_all_data_with_consolidation")

    except DataLoadError:
        raise
    except Exception as e:
        tprint_error(f"❌ Unexpected error in data download: {e}")
        raise DataLoadError(f"Data download failed: {e}") from e
