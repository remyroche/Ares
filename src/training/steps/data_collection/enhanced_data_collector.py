#!/usr/bin/env python3
"""
Enhanced Data Collector with Real-time Validation

This module provides enhanced data collection with:
- Real-time schema enforcement during API collection
- Comprehensive data quality validation
- Time gap detection between batches
- Field mapping for different exchanges
- Integration with existing data collection pipeline
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger
from src.utils.pipeline_standards import pipeline_standards
from .enhanced_data_validation_framework import (
    DataType, EnhancedDataValidator, get_validator, 
    ValidationSeverity, ValidationError
)

logger = system_logger.getChild("EnhancedDataCollector")


class EnhancedDataCollector:
    """Enhanced data collector with real-time validation."""
    
    def __init__(self, data_type: DataType, exchange: str, symbol: str, timeframe: str):
        self.data_type = data_type
        self.exchange = exchange.upper()
        self.symbol = symbol
        self.timeframe = timeframe
        self.logger = logger.getChild(f"{data_type.value}.{exchange}.{symbol}")
        
        # Initialize validator
        self.validator = get_validator(data_type)
        
        # Collection state
        self.last_timestamp: Optional[int] = None
        self.collection_stats = {
            'total_batches': 0,
            'total_rows': 0,
            'valid_rows': 0,
            'invalid_rows': 0,
            'time_gaps_detected': 0,
            'collection_start_time': None,
            'last_batch_time': None
        }
        
        # Data storage
        self.validated_data: List[Dict[str, Any]] = []
        self.validation_errors: List[ValidationError] = []
    
    async def collect_data_batch(self, raw_batch_data: List[Dict[str, Any]]) -> bool:
        """
        Collect and validate a batch of raw data from API.
        
        Args:
            raw_batch_data: Raw data from API
            
        Returns:
            True if batch was successfully processed, False otherwise
        """
        batch_start_time = time.time()
        self.collection_stats['total_batches'] += 1
        self.collection_stats['last_batch_time'] = batch_start_time
        
        if not self.collection_stats['collection_start_time']:
            self.collection_stats['collection_start_time'] = batch_start_time
        
        self.logger.info(f"📥 Processing batch {self.collection_stats['total_batches']} with {len(raw_batch_data)} rows")
        
        try:
            # Add metadata to raw data
            enriched_batch = self._enrich_raw_data(raw_batch_data)
            
            # Validate batch
            validated_batch = self.validator.validate_batch(
                enriched_batch, 
                self.last_timestamp
            )
            
            # Update statistics
            self.collection_stats['total_rows'] += len(raw_batch_data)
            self.collection_stats['valid_rows'] += len(validated_batch)
            self.collection_stats['invalid_rows'] += len(raw_batch_data) - len(validated_batch)
            
            # Store validated data
            self.validated_data.extend(validated_batch)
            
            # Update last timestamp
            if validated_batch:
                self.last_timestamp = validated_batch[-1]['timestamp']
            
            # Log batch results
            batch_duration = time.time() - batch_start_time
            success_rate = len(validated_batch) / len(raw_batch_data) * 100 if raw_batch_data else 0
            
            self.logger.info(f"✅ Batch {self.collection_stats['total_batches']} completed:")
            self.logger.info(f"   📊 Rows: {len(validated_batch)}/{len(raw_batch_data)} valid ({success_rate:.1f}%)")
            self.logger.info(f"   ⏱️ Duration: {batch_duration:.2f}s")
            
            if validated_batch:
                self.logger.info(f"   🕐 Last timestamp: {self.last_timestamp}")
            
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Batch {self.collection_stats['total_batches']} failed: {e}")
            return False
    
    def _enrich_raw_data(self, raw_batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich raw data with metadata fields."""
        enriched_batch = []
        
        for row in raw_batch_data:
            enriched_row = row.copy()
            enriched_row['exchange'] = self.exchange
            enriched_row['symbol'] = self.symbol
            enriched_row['timeframe'] = self.timeframe
            enriched_batch.append(enriched_row)
        
        return enriched_batch
    
    async def finalize_collection(self) -> Dict[str, Any]:
        """
        Finalize data collection and return summary.
        
        Returns:
            Collection summary with statistics and validated data
        """
        collection_duration = time.time() - self.collection_stats['collection_start_time']
        
        # Get validation summary
        validation_summary = self.validator.get_validation_summary()
        
        # Create collection summary
        summary = {
            'data_type': self.data_type.value,
            'exchange': self.exchange,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'collection_stats': self.collection_stats.copy(),
            'validation_summary': validation_summary,
            'collection_duration': collection_duration,
            'total_validated_rows': len(self.validated_data),
            'success_rate': self.collection_stats['valid_rows'] / self.collection_stats['total_rows'] * 100 if self.collection_stats['total_rows'] > 0 else 0
        }
        
        # Log final summary
        self.logger.info("=" * 80)
        self.logger.info("📊 DATA COLLECTION SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"🎯 Data Type: {self.data_type.value}")
        self.logger.info(f"🏢 Exchange: {self.exchange}")
        self.logger.info(f"📈 Symbol: {self.symbol}")
        self.logger.info(f"⏰ Timeframe: {self.timeframe}")
        self.logger.info(f"📦 Total Batches: {self.collection_stats['total_batches']}")
        self.logger.info(f"📊 Total Rows: {self.collection_stats['total_rows']}")
        self.logger.info(f"✅ Valid Rows: {self.collection_stats['valid_rows']}")
        self.logger.info(f"❌ Invalid Rows: {self.collection_stats['invalid_rows']}")
        self.logger.info(f"📈 Success Rate: {summary['success_rate']:.1f}%")
        self.logger.info(f"⏱️ Collection Duration: {collection_duration:.2f}s")
        self.logger.info(f"🕐 Time Gaps Detected: {validation_summary['time_gaps_detected']}")
        self.logger.info("=" * 80)
        
        return summary
    
    def get_validated_dataframe(self) -> pd.DataFrame:
        """Convert validated data to DataFrame."""
        if not self.validated_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.validated_data)
        
        # Sort by timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def get_validation_errors(self) -> List[ValidationError]:
        """Get all validation errors."""
        return self.validator.validation_stats['validation_errors']


class EnhancedDataCollectionManager:
    """Manager for enhanced data collection across multiple data types."""
    
    def __init__(self, exchange: str, symbol: str, timeframe: str):
        self.exchange = exchange.upper()
        self.symbol = symbol
        self.timeframe = timeframe
        self.logger = logger.getChild(f"Manager.{exchange}.{symbol}")
        
        # Initialize collectors for each data type
        self.collectors = {
            DataType.KLINES: EnhancedDataCollector(DataType.KLINES, exchange, symbol, timeframe),
            DataType.AGGTRADES: EnhancedDataCollector(DataType.AGGTRADES, exchange, symbol, timeframe),
            DataType.FUTURES: EnhancedDataCollector(DataType.FUTURES, exchange, symbol, timeframe)
        }
        
        # Collection state
        self.collection_start_time = time.time()
        self.collection_results = {}
    
    async def collect_klines_batch(self, raw_klines_data: List[Dict[str, Any]]) -> bool:
        """Collect and validate klines data batch."""
        return await self.collectors[DataType.KLINES].collect_data_batch(raw_klines_data)
    
    async def collect_aggtrades_batch(self, raw_aggtrades_data: List[Dict[str, Any]]) -> bool:
        """Collect and validate aggtrades data batch."""
        return await self.collectors[DataType.AGGTRADES].collect_data_batch(raw_aggtrades_data)
    
    async def collect_futures_batch(self, raw_futures_data: List[Dict[str, Any]]) -> bool:
        """Collect and validate futures data batch."""
        return await self.collectors[DataType.FUTURES].collect_data_batch(raw_futures_data)
    
    async def finalize_all_collections(self) -> Dict[str, Any]:
        """Finalize all data collections and return comprehensive summary."""
        self.logger.info("🏁 Finalizing all data collections...")
        
        # Finalize each collector
        for data_type, collector in self.collectors.items():
            self.collection_results[data_type.value] = await collector.finalize_collection()
        
        # Create overall summary
        total_duration = time.time() - self.collection_start_time
        total_rows = sum(result['collection_stats']['total_rows'] for result in self.collection_results.values())
        total_valid_rows = sum(result['collection_stats']['valid_rows'] for result in self.collection_results.values())
        
        overall_summary = {
            'exchange': self.exchange,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'total_duration': total_duration,
            'total_rows_collected': total_rows,
            'total_valid_rows': total_valid_rows,
            'overall_success_rate': total_valid_rows / total_rows * 100 if total_rows > 0 else 0,
            'collection_results': self.collection_results,
            'data_availability': {
                data_type: len(result['total_validated_rows']) > 0 
                for data_type, result in self.collection_results.items()
            }
        }
        
        # Log overall summary
        self.logger.info("=" * 80)
        self.logger.info("🎉 OVERALL DATA COLLECTION SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"🏢 Exchange: {self.exchange}")
        self.logger.info(f"📈 Symbol: {self.symbol}")
        self.logger.info(f"⏰ Timeframe: {self.timeframe}")
        self.logger.info(f"📊 Total Rows: {total_rows}")
        self.logger.info(f"✅ Valid Rows: {total_valid_rows}")
        self.logger.info(f"📈 Overall Success Rate: {overall_summary['overall_success_rate']:.1f}%")
        self.logger.info(f"⏱️ Total Duration: {total_duration:.2f}s")
        
        # Data availability summary
        self.logger.info("📋 Data Availability:")
        for data_type, available in overall_summary['data_availability'].items():
            status = "✅ Available" if available else "❌ Not Available"
            self.logger.info(f"   {data_type}: {status}")
        
        self.logger.info("=" * 80)
        
        return overall_summary
    
    def get_validated_dataframes(self) -> Dict[str, pd.DataFrame]:
        """Get validated DataFrames for all data types."""
        return {
            data_type.value: collector.get_validated_dataframe()
            for data_type, collector in self.collectors.items()
        }
    
    def get_all_validation_errors(self) -> Dict[str, List[ValidationError]]:
        """Get validation errors for all data types."""
        return {
            data_type.value: collector.get_validation_errors()
            for data_type, collector in self.collectors.items()
        }


# Integration functions for existing pipeline
async def collect_data_with_validation(
    data_type: DataType,
    exchange: str,
    symbol: str,
    timeframe: str,
    raw_data_batches: List[List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """
    Collect data with validation for a single data type.
    
    Args:
        data_type: Type of data to collect
        exchange: Exchange name
        symbol: Trading symbol
        timeframe: Timeframe
        raw_data_batches: List of raw data batches from API
        
    Returns:
        Collection summary with validated data
    """
    collector = EnhancedDataCollector(data_type, exchange, symbol, timeframe)
    
    # Process each batch
    for batch in raw_data_batches:
        await collector.collect_data_batch(batch)
    
    # Finalize collection
    summary = await collector.finalize_collection()
    summary['validated_dataframe'] = collector.get_validated_dataframe()
    
    return summary


async def collect_all_data_with_validation(
    exchange: str,
    symbol: str,
    timeframe: str,
    raw_data: Dict[str, List[List[Dict[str, Any]]]]
) -> Dict[str, Any]:
    """
    Collect all data types with validation.
    
    Args:
        exchange: Exchange name
        symbol: Trading symbol
        timeframe: Timeframe
        raw_data: Dictionary with data type as key and list of batches as value
        
    Returns:
        Comprehensive collection summary
    """
    manager = EnhancedDataCollectionManager(exchange, symbol, timeframe)
    
    # Process each data type
    for data_type_str, batches in raw_data.items():
        try:
            data_type = DataType(data_type_str)
            collector = manager.collectors[data_type]
            
            for batch in batches:
                await collector.collect_data_batch(batch)
                
        except ValueError:
            logger.warning(f"⚠️ Unknown data type: {data_type_str}")
            continue
    
    # Finalize all collections
    summary = await manager.finalize_all_collections()
    summary['validated_dataframes'] = manager.get_validated_dataframes()
    summary['validation_errors'] = manager.get_all_validation_errors()
    
    return summary


if __name__ == "__main__":
    # Example usage
    async def test_enhanced_collection():
        # Simulate raw data from API
        raw_klines_batches = [
            [
                {
                    "open_time": 1640995200000,
                    "open": "3000.0",
                    "high": "3100.0",
                    "low": "2900.0",
                    "close": "3050.0",
                    "volume": "1000.0"
                },
                {
                    "open_time": 1640995260000,
                    "open": "3050.0",
                    "high": "3150.0",
                    "low": "2950.0",
                    "close": "3100.0",
                    "volume": "1200.0"
                }
            ]
        ]
        
        raw_aggtrades_batches = [
            [
                {
                    "T": 1640995200000,
                    "p": "3050.0",
                    "q": "1.5",
                    "m": True
                },
                {
                    "T": 1640995201000,
                    "p": "3051.0",
                    "q": "2.0",
                    "m": False
                }
            ]
        ]
        
        # Collect with validation
        raw_data = {
            "klines": raw_klines_batches,
            "aggtrades": raw_aggtrades_batches
        }
        
        summary = await collect_all_data_with_validation(
            exchange="BINANCE",
            symbol="ETHUSDT",
            timeframe="1m",
            raw_data=raw_data
        )
        
        print("Collection Summary:", summary)
    
    asyncio.run(test_enhanced_collection())