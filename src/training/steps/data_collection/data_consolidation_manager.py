#!/usr/bin/env python3
"""
Data Consolidation Manager

This module provides comprehensive data consolidation functionality for managing
and merging multiple batch files from data downloads. It ensures data integrity,
handles duplicates, and provides efficient consolidation strategies.

Key Features:
- Intelligent batch file discovery and grouping
- Duplicate detection and removal
- Data quality validation during consolidation
- Memory-efficient processing for large datasets
- Multiple consolidation strategies (by time, by session, by size)
- Comprehensive logging and progress tracking
"""

import asyncio
import sys
import time
import os
import glob
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Set
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger
from src.utils.error_handler import handles_errors
from src.utils.common_operations import safe_fillna, safe_to_parquet, safe_read_parquet
from src.utils.common_utilities import validate_dataframe_columns, safe_dataframe_operation
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

logger = system_logger.getChild("DataConsolidationManager")

class DataConsolidationManager:
    """Manager for consolidating multiple batch files into unified datasets."""
    
    @log_important_calls
    def __init__(self, data_cache_path: str = "data_cache"):
        self.data_cache_path = Path(data_cache_path)
        self.data_cache_path.mkdir(exist_ok=True)
        self.logger = logger.getChild('DataConsolidationManager')
        
        # Initialize standardized parquet handler
        self.parquet_handler = standardized_parquet_handler
        
        # Consolidation statistics
        self.consolidation_stats = {
            'total_consolidations': 0,
            'successful_consolidations': 0,
            'failed_consolidations': 0,
            'total_files_processed': 0,
            'total_rows_consolidated': 0,
            'duplicates_removed': 0,
            'start_time': None
        }
    
    @handles_errors(context="consolidate_by_session")
    @log_all_calls
    async def consolidate_by_session(
        self,
        symbol: str,
        exchange: str,
        session_id: str,
        data_type: str = "klines",
        timeframe: str = "1m",
        remove_originals: bool = False,
        max_memory_mb: int = 1000
    ) -> Dict[str, Any]:
        """
        Consolidate all batch files from a specific session.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_type: Type of data
            timeframe: Timeframe
            session_id: Session ID to consolidate
            remove_originals: Whether to remove original batch files
            max_memory_mb: Maximum memory usage in MB
            
        Returns:
            Dictionary with consolidation results
        """
        self.logger.info(f"🔄 Consolidating session {session_id}: {exchange}_{symbol}_{data_type}_{timeframe}")
        
        try:
            # Find batch files for this session
            batch_files = await self._find_session_batch_files(
                symbol, exchange, data_type, timeframe, session_id
            )
            
            if not batch_files:
                return {
                    'success': False,
                    'error': f'No batch files found for session {session_id}',
                    'consolidated_file': None
                }
            
            self.logger.info(f"📁 Found {len(batch_files)} batch files for session {session_id}")
            
            # Consolidate files
            result = await self._consolidate_files(
                batch_files, symbol, exchange, data_type, timeframe, 
                session_id, remove_originals, max_memory_mb
            )
            
            # Update statistics
            self.consolidation_stats['total_consolidations'] += 1
            if result['success']:
                self.consolidation_stats['successful_consolidations'] += 1
                self.consolidation_stats['total_files_processed'] += len(batch_files)
                self.consolidation_stats['total_rows_consolidated'] += result.get('total_rows', 0)
                self.consolidation_stats['duplicates_removed'] += result.get('duplicates_removed', 0)
            else:
                self.consolidation_stats['failed_consolidations'] += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error consolidating session {session_id}: {e}")
            self.consolidation_stats['failed_consolidations'] += 1
            return {
                'success': False,
                'error': str(e),
                'consolidated_file': None
            }
    
    @handles_errors(context="consolidate_by_time_range")
    @log_all_calls
    async def consolidate_by_time_range(
        self, 
        symbol: str, 
        exchange: str, 
        data_type: str = "klines",
        timeframe: str = "1m",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        remove_originals: bool = False,
        max_memory_mb: int = 1000
    ) -> Dict[str, Any]:
        """
        Consolidate batch files within a specific time range.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_type: Type of data
            timeframe: Timeframe
            start_time: Start time for consolidation
            end_time: End time for consolidation
            remove_originals: Whether to remove original batch files
            max_memory_mb: Maximum memory usage in MB
            
        Returns:
            Dictionary with consolidation results
        """
        self.logger.info(f"🔄 Consolidating time range: {exchange}_{symbol}_{data_type}_{timeframe}")
        
        try:
            # Find batch files within time range
            batch_files = await self._find_time_range_batch_files(
                symbol, exchange, data_type, timeframe, start_time, end_time
            )
            
            if not batch_files:
                return {
                    'success': False,
                    'error': 'No batch files found in time range',
                    'consolidated_file': None
                }
            
            self.logger.info(f"📁 Found {len(batch_files)} batch files in time range")
            
            # Generate session ID for time range
            time_session_id = f"time_range_{start_time.strftime('%Y%m%d_%H%M%S') if start_time else 'start'}_{end_time.strftime('%Y%m%d_%H%M%S') if end_time else 'end'}"
            
            # Consolidate files
            result = await self._consolidate_files(
                batch_files, symbol, exchange, data_type, timeframe, 
                time_session_id, remove_originals, max_memory_mb
            )
            
            # Update statistics
            self.consolidation_stats['total_consolidations'] += 1
            if result['success']:
                self.consolidation_stats['successful_consolidations'] += 1
                self.consolidation_stats['total_files_processed'] += len(batch_files)
                self.consolidation_stats['total_rows_consolidated'] += result.get('total_rows', 0)
                self.consolidation_stats['duplicates_removed'] += result.get('duplicates_removed', 0)
            else:
                self.consolidation_stats['failed_consolidations'] += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error consolidating time range: {e}")
            self.consolidation_stats['failed_consolidations'] += 1
            return {
                'success': False,
                'error': str(e),
                'consolidated_file': None
            }
    
    @handles_errors(context="consolidate_all_available")
    @log_all_calls
    async def consolidate_all_available(
        self, 
        symbol: str, 
        exchange: str, 
        data_type: str = "klines",
        timeframe: str = "1m",
        remove_originals: bool = False,
        max_memory_mb: int = 1000,
        chunk_size: int = 50
    ) -> Dict[str, Any]:
        """
        Consolidate all available batch files for a symbol/exchange combination.
        Uses chunking to handle large numbers of files efficiently.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_type: Type of data
            timeframe: Timeframe
            remove_originals: Whether to remove original batch files
            max_memory_mb: Maximum memory usage in MB
            chunk_size: Number of files to process in each chunk
            
        Returns:
            Dictionary with consolidation results
        """
        self.logger.info(f"🔄 Consolidating all available: {exchange}_{symbol}_{data_type}_{timeframe}")
        
        try:
            # Find all batch files
            batch_files = await self._find_all_batch_files(
                symbol, exchange, data_type, timeframe
            )
            
            if not batch_files:
                return {
                    'success': False,
                    'error': 'No batch files found',
                    'consolidated_file': None
                }
            
            self.logger.info(f"📁 Found {len(batch_files)} batch files to consolidate")
            
            # Process in chunks if there are many files
            if len(batch_files) > chunk_size:
                return await self._consolidate_in_chunks(
                    batch_files, symbol, exchange, data_type, timeframe, 
                    remove_originals, max_memory_mb, chunk_size
                )
            else:
                # Process all files at once
                all_session_id = f"all_available_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                return await self._consolidate_files(
                    batch_files, symbol, exchange, data_type, timeframe, 
                    all_session_id, remove_originals, max_memory_mb
                )
            
        except Exception as e:
            self.logger.error(f"❌ Error consolidating all available: {e}")
            return {
                'success': False,
                'error': str(e),
                'consolidated_file': None
            }
    
    @handles_errors(context="consolidate_files")
    async def _consolidate_files(
        self, 
        batch_files: List[Path], 
        symbol: str, 
        exchange: str, 
        data_type: str, 
        timeframe: str,
        session_id: str, 
        remove_originals: bool, 
        max_memory_mb: int
    ) -> Dict[str, Any]:
        """Consolidate a list of batch files."""
        try:
            self.logger.info(f"🔄 Processing {len(batch_files)} files for consolidation")
            
            # Load and combine all files
            all_dataframes = []
            total_rows_before = 0
            processed_files = 0
            
            for batch_file in batch_files:
                try:
                    df = self.parquet_handler.read_parquet_standardized(batch_file)
                    if df is not None and not df.empty:
                        all_dataframes.append(df)
                        total_rows_before += len(df)
                        processed_files += 1
                        
                        # Memory check
                        current_memory = self._estimate_memory_usage(all_dataframes)
                        if current_memory > max_memory_mb:
                            self.logger.warning(f"⚠️ Memory limit reached ({current_memory:.1f}MB), processing {processed_files} files")
                            break
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to read {batch_file}: {e}")
                    continue
            
            if not all_dataframes:
                return {
                    'success': False,
                    'error': 'No valid data found in batch files',
                    'consolidated_file': None
                }
            
            self.logger.info(f"📊 Loaded {processed_files} files with {total_rows_before} total rows")
            
            # Combine all dataframes
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            
            # Sort by timestamp if available
            if 'timestamp' in combined_df.columns:
                combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
            
            # Remove duplicates with detailed logging
            initial_rows = len(combined_df)
            if 'timestamp' in combined_df.columns:
                # Check for duplicates before removal
                duplicate_count = combined_df['timestamp'].duplicated().sum()
                if duplicate_count > 0:
                    self.logger.warning(f"⚠️ Found {duplicate_count} duplicate timestamps before consolidation")
                combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='first')
            else:
                combined_df = combined_df.drop_duplicates()

            duplicates_removed = initial_rows - len(combined_df)
            if duplicates_removed > 0:
                self.logger.info(f"🧹 Removed {duplicates_removed} duplicate entries during consolidation")
            
            # Generate consolidated filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            consolidated_filename = f"{data_type}_{exchange}_{symbol}_{timeframe}_{session_id}_consolidated_{timestamp}.parquet"
            consolidated_path = self.data_cache_path / exchange.lower() / symbol.lower() / data_type / consolidated_filename
            
            # Save consolidated file
            success = self.parquet_handler.write_parquet_standardized(
                combined_df, consolidated_path, schema_name='unified', validate_quality=True
            )
            
            if not success:
                return {
                    'success': False,
                    'error': 'Failed to save consolidated file',
                    'consolidated_file': None
                }
            
            # Remove original batch files if requested
            removed_files = []
            if remove_originals:
                for batch_file in batch_files[:processed_files]:  # Only remove processed files
                    try:
                        batch_file.unlink()
                        removed_files.append(str(batch_file))
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to remove {batch_file}: {e}")
            
            result = {
                'success': True,
                'consolidated_file': str(consolidated_path),
                'total_rows': len(combined_df),
                'rows_before_dedup': total_rows_before,
                'duplicates_removed': duplicates_removed,
                'files_processed': processed_files,
                'removed_files': removed_files,
                'memory_used_mb': self._estimate_memory_usage([combined_df]),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Consolidation completed: {len(combined_df)} rows in {consolidated_path.name}")
            self.logger.info(f"📊 Removed {duplicates_removed} duplicates from {total_rows_before} rows")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error in file consolidation: {e}")
            return {
                'success': False,
                'error': str(e),
                'consolidated_file': None
            }
    
    @handles_errors(context="consolidate_in_chunks")
    async def _consolidate_in_chunks(
        self, 
        batch_files: List[Path], 
        symbol: str, 
        exchange: str, 
        data_type: str, 
        timeframe: str,
        remove_originals: bool, 
        max_memory_mb: int, 
        chunk_size: int
    ) -> Dict[str, Any]:
        """Consolidate files in chunks to handle large datasets."""
        try:
            self.logger.info(f"🔄 Processing {len(batch_files)} files in chunks of {chunk_size}")
            
            # Split files into chunks
            file_chunks = [batch_files[i:i + chunk_size] for i in range(0, len(batch_files), chunk_size)]
            
            consolidated_files = []
            total_rows = 0
            total_duplicates_removed = 0
            total_files_processed = 0
            
            for chunk_idx, file_chunk in enumerate(file_chunks):
                self.logger.info(f"📦 Processing chunk {chunk_idx + 1}/{len(file_chunks)} ({len(file_chunk)} files)")
                
                # Process chunk
                chunk_session_id = f"chunk_{chunk_idx + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                chunk_result = await self._consolidate_files(
                    file_chunk, symbol, exchange, data_type, timeframe, 
                    chunk_session_id, remove_originals, max_memory_mb
                )
                
                if chunk_result['success']:
                    consolidated_files.append(chunk_result['consolidated_file'])
                    total_rows += chunk_result['total_rows']
                    total_duplicates_removed += chunk_result['duplicates_removed']
                    total_files_processed += chunk_result['files_processed']
                else:
                    self.logger.warning(f"⚠️ Chunk {chunk_idx + 1} failed: {chunk_result['error']}")
            
            if not consolidated_files:
                return {
                    'success': False,
                    'error': 'All chunks failed to consolidate',
                    'consolidated_file': None
                }
            
            # If we have multiple consolidated files, we might want to merge them too
            if len(consolidated_files) > 1:
                self.logger.info(f"🔄 Merging {len(consolidated_files)} consolidated files")
                
                # Create final consolidated file
                final_session_id = f"final_consolidated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                final_result = await self._consolidate_files(
                    [Path(f) for f in consolidated_files], symbol, exchange, data_type, timeframe, 
                    final_session_id, remove_originals=True, max_memory_mb=max_memory_mb
                )
                
                if final_result['success']:
                    # Clean up intermediate consolidated files
                    for consolidated_file in consolidated_files:
                        try:
                            Path(consolidated_file).unlink()
                        except Exception as e:
                            self.logger.warning(f"⚠️ Failed to remove intermediate file {consolidated_file}: {e}")
                    
                    return final_result
                else:
                    return {
                        'success': False,
                        'error': f'Failed to merge consolidated files: {final_result["error"]}',
                        'consolidated_file': None
                    }
            else:
                # Only one consolidated file, return it
                return {
                    'success': True,
                    'consolidated_file': consolidated_files[0],
                    'total_rows': total_rows,
                    'duplicates_removed': total_duplicates_removed,
                    'files_processed': total_files_processed,
                    'chunks_processed': len(file_chunks),
                    'timestamp': datetime.now().isoformat()
                }
            
        except Exception as e:
            self.logger.error(f"❌ Error in chunked consolidation: {e}")
            return {
                'success': False,
                'error': str(e),
                'consolidated_file': None
            }
    
    @handles_errors(context="find_session_batch_files")
    async def _find_session_batch_files(
        self, 
        symbol: str, 
        exchange: str, 
        data_type: str, 
        timeframe: str, 
        session_id: str
    ) -> List[Path]:
        """Find batch files for a specific session."""
        try:
            data_dir = Path("historical_data") / exchange.lower() / symbol.lower() / data_type
            
            if not data_dir.exists():
                return []
            
            pattern = f"{data_type}_{exchange}_{symbol}_{timeframe}_{session_id}_batch_*.parquet"
            batch_files = list(data_dir.glob(pattern))
            batch_files.sort(key=lambda x: x.stat().st_mtime)
            
            return batch_files
            
        except Exception as e:
            self.logger.error(f"❌ Error finding session batch files: {e}")
            return []
    
    @handles_errors(context="find_time_range_batch_files")
    async def _find_time_range_batch_files(
        self, 
        symbol: str, 
        exchange: str, 
        data_type: str, 
        timeframe: str, 
        start_time: Optional[datetime], 
        end_time: Optional[datetime]
    ) -> List[Path]:
        """Find batch files within a time range."""
        try:
            data_dir = Path("historical_data") / exchange.lower() / symbol.lower() / data_type
            
            if not data_dir.exists():
                return []
            
            # Get all batch files
            pattern = f"{data_type}_{exchange}_{symbol}_{timeframe}_*_batch_*.parquet"
            all_files = list(data_dir.glob(pattern))
            
            # Filter by time range
            filtered_files = []
            for file_path in all_files:
                try:
                    # Extract timestamp from filename or use file modification time
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    
                    # Check if file is within time range
                    if start_time and file_mtime < start_time:
                        continue
                    if end_time and file_mtime > end_time:
                        continue
                    
                    filtered_files.append(file_path)
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Error processing file {file_path}: {e}")
                    continue
            
            filtered_files.sort(key=lambda x: x.stat().st_mtime)
            return filtered_files
            
        except Exception as e:
            self.logger.error(f"❌ Error finding time range batch files: {e}")
            return []
    
    @handles_errors(context="find_all_batch_files")
    async def _find_all_batch_files(
        self, 
        symbol: str, 
        exchange: str, 
        data_type: str, 
        timeframe: str
    ) -> List[Path]:
        """Find all batch files for a symbol/exchange combination."""
        try:
            data_dir = Path("historical_data") / exchange.lower() / symbol.lower() / data_type
            
            if not data_dir.exists():
                return []
            
            pattern = f"{data_type}_{exchange}_{symbol}_{timeframe}_*_batch_*.parquet"
            batch_files = list(data_dir.glob(pattern))
            batch_files.sort(key=lambda x: x.stat().st_mtime)
            
            return batch_files
            
        except Exception as e:
            self.logger.error(f"❌ Error finding all batch files: {e}")
            return []
    
    @handles_errors(context="estimate_memory_usage")
    def _estimate_memory_usage(self, dataframes: List[pd.DataFrame]) -> float:
        """Estimate memory usage of dataframes in MB."""
        try:
            total_memory = 0
            for df in dataframes:
                if df is not None:
                    total_memory += df.memory_usage(deep=True).sum()
            return total_memory / (1024 * 1024)  # Convert to MB
        except Exception:
            return 0.0
    
    @handles_errors(context="get_consolidation_stats")
    def get_consolidation_stats(self) -> Dict[str, Any]:
        """Get consolidation statistics."""
        return {
            **self.consolidation_stats,
            'success_rate': (
                self.consolidation_stats['successful_consolidations'] / 
                max(self.consolidation_stats['total_consolidations'], 1) * 100
            )
        }
    
    def reset_stats(self):
        """Reset consolidation statistics."""
        self.consolidation_stats = {
            'total_consolidations': 0,
            'successful_consolidations': 0,
            'failed_consolidations': 0,
            'total_files_processed': 0,
            'total_rows_consolidated': 0,
            'duplicates_removed': 0,
            'start_time': None
        }

# Convenience functions
@handles_errors()
async def consolidate_session_data(
    symbol: str,
    exchange: str,
    session_id: str,
    data_type: str = "klines",
    timeframe: str = "1m",
    **kwargs
) -> Dict[str, Any]:
    """Convenience function for consolidating session data."""
    manager = DataConsolidationManager()
    return await manager.consolidate_by_session(symbol, exchange, data_type, timeframe, session_id, **kwargs)

@handles_errors()
async def consolidate_time_range_data(
    symbol: str, 
    exchange: str, 
    data_type: str = "klines",
    timeframe: str = "1m",
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    **kwargs
) -> Dict[str, Any]:
    """Convenience function for consolidating time range data."""
    manager = DataConsolidationManager()
    return await manager.consolidate_by_time_range(symbol, exchange, data_type, timeframe, start_time, end_time, **kwargs)

@handles_errors()
async def consolidate_all_data(
    symbol: str, 
    exchange: str, 
    data_type: str = "klines",
    timeframe: str = "1m",
    **kwargs
) -> Dict[str, Any]:
    """Convenience function for consolidating all available data."""
    manager = DataConsolidationManager()
    return await manager.consolidate_all_available(symbol, exchange, data_type, timeframe, **kwargs)

if __name__ == "__main__":
    # Example usage
    async def test_consolidation_manager():
        logger.info("🎯 Testing Data Consolidation Manager")
        logger.info("=" * 80)
        
        # Test session consolidation
        logger.info("📊 Testing session consolidation...")
        result = await consolidate_session_data(
            symbol="ETHUSDT",
            exchange="BINANCE",
            data_type="klines",
            timeframe="1m",
            session_id="test_session_20240101_120000"
        )
        
        logger.info(f"✅ Session consolidation result: {result['success']}")
        
        # Test time range consolidation
        logger.info("📅 Testing time range consolidation...")
        time_result = await consolidate_time_range_data(
            symbol="ETHUSDT",
            exchange="BINANCE",
            data_type="klines",
            timeframe="1m",
            start_time=datetime.now() - timedelta(days=1),
            end_time=datetime.now()
        )
        
        logger.info(f"✅ Time range consolidation result: {time_result['success']}")
        
        # Test all data consolidation
        logger.info("🔄 Testing all data consolidation...")
        all_result = await consolidate_all_data(
            symbol="ETHUSDT",
            exchange="BINANCE",
            data_type="klines",
            timeframe="1m"
        )
        
        logger.info(f"✅ All data consolidation result: {all_result['success']}")
        
        logger.info("=" * 80)
        logger.info("🎉 Data Consolidation Manager tests completed!")
        logger.info("=" * 80)
    
    asyncio.run(test_consolidation_manager())