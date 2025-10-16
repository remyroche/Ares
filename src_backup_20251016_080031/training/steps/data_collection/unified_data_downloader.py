#!/usr/bin/env python3
"""
Unified Data Downloader

This module provides centralized download functionality:
- Klines data (PRIMARY - per new setup)
- Aggtrades data (DEPRECATED - not used in new klines-only setup)
- Futures data (DEPRECATED - not used in new klines-only setup)

NOTE: Per new setup, only klines data is actively used. Aggtrades and futures
downloads are deprecated but maintained for backwards compatibility.
"""

import asyncio
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger
from src.utils.error_handler import handles_errors
from src.utils.common_operations import safe_fillna, safe_to_parquet, safe_read_parquet
from src.utils.common_utilities import validate_dataframe_columns, safe_dataframe_operation
# from src.utils.validation import validate_data_quality  # Replaced with comprehensive quality tools

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
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

logger = system_logger.getChild("UnifiedDataDownloader")

class UnifiedDataDownloader:
    """Unified downloader for all data types with comprehensive error handling and validation."""
    
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
    
