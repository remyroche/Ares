#!/usr/bin/env python3
"""
Enhanced Data Collection Demo

This module demonstrates the complete enhanced data collection framework with:
- Extensive logging and printing
- Integration with utils/ decorators
- Field mapping for different exchanges
- Data qualification with duplicate removal
- API-agnostic data collection
- Comprehensive gap detection and incremental downloading
"""

from __future__ import annotations

import asyncio
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger
from src.utils.common_operations import handles_errors, traced, log_execution_time
from src.utils.enhanced_mlflow_integration import with_enhanced_mlflow_logging
from .enhanced_validation_framework_with_decorators import (
    DataType, get_validator, validate_data_batch, ValidationSeverity
)
from .exchange_field_mappings import get_exchange_mapper, list_supported_exchanges
from .enhanced_api_agnostic_data_collector import (
    EnhancedAPIAgnosticDataCollector,
    collect_data_for_period,
    collect_incremental_data,
    detect_and_fill_gaps
)

logger = system_logger.getChild("EnhancedDataCollectionDemo")


class EnhancedDataCollectionDemo:
    """Comprehensive demo of the enhanced data collection framework."""
    
    def __init__(self):
        self.logger = logger.getChild("Demo")
        self.demo_start_time = time.time()
        
        self.logger.info("🚀 Initializing Enhanced Data Collection Demo")
        self.logger.info("=" * 80)
        self.logger.info("📋 Demo Features:")
        self.logger.info("   ✅ Extensive logging and printing")
        self.logger.info("   ✅ Integration with utils/ decorators")
        self.logger.info("   ✅ Field mapping for different exchanges")
        self.logger.info("   ✅ Data qualification with duplicate removal")
        self.logger.info("   ✅ API-agnostic data collection")
        self.logger.info("   ✅ Comprehensive gap detection")
        self.logger.info("   ✅ Incremental downloading")
        self.logger.info("=" * 80)
    
    @handles_errors(fallback=False, context="demo_field_mappings")
    @traced(span_name="demo_field_mappings", log_args=False, log_result_len_only=True)
    async def demo_field_mappings(self):
        """Demonstrate field mappings for different exchanges."""
        self.logger.info("🎯 DEMO 1: Exchange Field Mappings")
        self.logger.info("-" * 50)
        
        # List supported exchanges
        supported_exchanges = list_supported_exchanges()
        self.logger.info(f"📋 Supported exchanges: {supported_exchanges}")
        
        # Test field mappings for different exchanges
        test_data = {
            "binance": {
                "open_time": 1640995200000,
                "open": "3000.0",
                "high": "3100.0",
                "low": "2900.0",
                "close": "3050.0",
                "volume": "1000.0"
            },
            "coinbase": {
                "timestamp": 1640995200,
                "price_open": "3000.0",
                "price_high": "3100.0",
                "price_low": "2900.0",
                "price_close": "3050.0",
                "volume": "1000.0"
            },
            "kraken": {
                "time": 1640995200,
                "open": "3000.0",
                "high": "3100.0",
                "low": "2900.0",
                "close": "3050.0",
                "vol": "1000.0"
            }
        }
        
        for exchange_name, raw_data in test_data.items():
            self.logger.info(f"🔄 Testing {exchange_name.upper()} field mapping...")
            
            try:
                mapper = get_exchange_mapper(exchange_name)
                mapped_data = mapper.map_fields("klines", raw_data)
                
                self.logger.info(f"✅ {exchange_name.upper()} mapping successful:")
                self.logger.info(f"   📊 Mapped fields: {list(mapped_data.keys())}")
                self.logger.info(f"   🕐 Timestamp: {mapped_data.get('timestamp', 'N/A')}")
                self.logger.info(f"   💰 OHLC: {mapped_data.get('open', 'N/A')}/{mapped_data.get('high', 'N/A')}/{mapped_data.get('low', 'N/A')}/{mapped_data.get('close', 'N/A')}")
                
            except Exception as e:
                self.logger.error(f"❌ {exchange_name.upper()} mapping failed: {e}")
        
        self.logger.info("✅ Field mapping demo completed")
        self.logger.info("-" * 50)
    
    @handles_errors(fallback=False, context="demo_data_validation")
    @traced(span_name="demo_data_validation", log_args=False, log_result_len_only=True)
    async def demo_data_validation(self):
        """Demonstrate enhanced data validation with decorators."""
        self.logger.info("🎯 DEMO 2: Enhanced Data Validation with Decorators")
        self.logger.info("-" * 50)
        
        # Test klines validation
        self.logger.info("📊 Testing klines validation...")
        klines_data = [
            {
                "open_time": 1640995200000,  # Binance format
                "open": "3000.0",
                "high": "3100.0",
                "low": "2900.0",
                "close": "3050.0",
                "volume": "1000.0"
            },
            {
                "open_time": 1640995260000,  # Next minute
                "open": "3050.0",
                "high": "3150.0",
                "low": "2950.0",
                "close": "3100.0",
                "volume": "1200.0"
            }
        ]
        
        validated_klines = validate_data_batch(DataType.KLINES, klines_data, "BINANCE")
        self.logger.info(f"✅ Klines validation: {len(validated_klines)} rows validated")
        
        # Test aggtrades validation
        self.logger.info("📊 Testing aggtrades validation...")
        aggtrades_data = [
            {
                "T": 1640995200000,  # Binance format
                "p": "3050.0",
                "q": "1.5",
                "m": True
            },
            {
                "T": 1640995201000,  # 1 second later
                "p": "3051.0",
                "q": "2.0",
                "m": False
            }
        ]
        
        validated_aggtrades = validate_data_batch(DataType.AGGTRADES, aggtrades_data, "BINANCE")
        self.logger.info(f"✅ Aggtrades validation: {len(validated_aggtrades)} rows validated")
        
        # Test futures validation
        self.logger.info("📊 Testing futures validation...")
        futures_data = [
            {
                "fundingTime": 1640995200000,  # Binance format
                "fundingRate": "0.0001"
            }
        ]
        
        validated_futures = validate_data_batch(DataType.FUTURES, futures_data, "BINANCE")
        self.logger.info(f"✅ Futures validation: {len(validated_futures)} rows validated")
        
        # Test validation with errors
        self.logger.info("📊 Testing validation with errors...")
        invalid_data = [
            {
                "open_time": 1640995200000,
                "open": "0.0",  # Invalid: zero price
                "high": "3100.0",
                "low": "2900.0",
                "close": "3050.0",
                "volume": "1000.0"
            }
        ]
        
        try:
            validated_invalid = validate_data_batch(DataType.KLINES, invalid_data, "BINANCE")
            self.logger.info(f"⚠️ Invalid data validation: {len(validated_invalid)} rows validated")
        except Exception as e:
            self.logger.warning(f"⚠️ Invalid data validation failed as expected: {e}")
        
        self.logger.info("✅ Data validation demo completed")
        self.logger.info("-" * 50)
    
    @handles_errors(fallback=False, context="demo_data_qualification")
    @traced(span_name="demo_data_qualification", log_args=False, log_result_len_only=True)
    async def demo_data_qualification(self):
        """Demonstrate data qualification with duplicate removal."""
        self.logger.info("🎯 DEMO 3: Data Qualification with Duplicate Removal")
        self.logger.info("-" * 50)
        
        # Create test data with duplicates
        test_data = [
            {
                "timestamp": 1640995200000,
                "open": "3000.0",
                "high": "3100.0",
                "low": "2900.0",
                "close": "3050.0",
                "volume": "1000.0",
                "exchange": "BINANCE",
                "symbol": "ETHUSDT",
                "timeframe": "1m"
            },
            {
                "timestamp": 1640995200000,  # Duplicate timestamp
                "open": "3000.0",
                "high": "3100.0",
                "low": "2900.0",
                "close": "3050.0",
                "volume": "1000.0",
                "exchange": "BINANCE",
                "symbol": "ETHUSDT",
                "timeframe": "1m"
            },
            {
                "timestamp": 1640995260000,
                "open": "3050.0",
                "high": "3150.0",
                "low": "2950.0",
                "close": "3100.0",
                "volume": "1200.0",
                "exchange": "BINANCE",
                "symbol": "ETHUSDT",
                "timeframe": "1m"
            }
        ]
        
        self.logger.info(f"📊 Original data: {len(test_data)} rows")
        
        # Validate data (this will remove duplicates)
        validator = get_validator(DataType.KLINES, "BINANCE")
        validated_data = validator.validate_batch(test_data)
        
        self.logger.info(f"✅ Qualified data: {len(validated_data)} rows")
        self.logger.info(f"🔄 Duplicates removed: {len(test_data) - len(validated_data)}")
        
        # Show validation summary
        summary = validator.get_validation_summary()
        self.logger.info(f"📊 Validation Summary:")
        self.logger.info(f"   📈 Success Rate: {summary['success_rate']:.1f}%")
        self.logger.info(f"   🔄 Duplicates Removed: {summary['duplicates_removed']}")
        self.logger.info(f"   ⚠️ Total Errors: {summary['total_errors']}")
        
        self.logger.info("✅ Data qualification demo completed")
        self.logger.info("-" * 50)
    
    @handles_errors(fallback=False, context="demo_api_agnostic_collection")
    @traced(span_name="demo_api_agnostic_collection", log_args=False, log_result_len_only=True)
    async def demo_api_agnostic_collection(self):
        """Demonstrate API-agnostic data collection."""
        self.logger.info("🎯 DEMO 4: API-Agnostic Data Collection")
        self.logger.info("-" * 50)
        
        # Test incremental data collection
        self.logger.info("📊 Testing incremental data collection...")
        
        try:
            result = await collect_incremental_data(
                exchange="BINANCE",
                symbol="ETHUSDT",
                timeframe="1m",
                data_types=["klines"],
                max_batches=2
            )
            
            self.logger.info(f"✅ Incremental collection result:")
            self.logger.info(f"   📊 Success: {result['success']}")
            self.logger.info(f"   📈 Total Rows: {result['total_rows_collected']}")
            self.logger.info(f"   ⏱️ Duration: {result['total_duration']:.2f}s")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Incremental collection failed (expected in demo): {e}")
        
        # Test period-based collection
        self.logger.info("📊 Testing period-based collection...")
        
        try:
            start_time = datetime.now() - timedelta(hours=1)
            end_time = datetime.now()
            
            result = await collect_data_for_period(
                exchange="BINANCE",
                symbol="ETHUSDT",
                timeframe="1m",
                start_time=start_time,
                end_time=end_time,
                data_types=["klines"]
            )
            
            self.logger.info(f"✅ Period collection result:")
            self.logger.info(f"   📊 Success: {result['success']}")
            self.logger.info(f"   📈 Total Rows: {result['total_rows_collected']}")
            self.logger.info(f"   ⏱️ Duration: {result['total_duration']:.2f}s")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Period collection failed (expected in demo): {e}")
        
        # Test gap detection
        self.logger.info("📊 Testing gap detection...")
        
        try:
            result = await detect_and_fill_gaps(
                exchange="BINANCE",
                symbol="ETHUSDT",
                timeframe="1m",
                data_types=["klines"]
            )
            
            self.logger.info(f"✅ Gap detection result:")
            self.logger.info(f"   📊 Success: {result['success']}")
            self.logger.info(f"   🕐 Total Gaps: {result['total_gaps_found']}")
            self.logger.info(f"   ✅ Gaps Filled: {result['total_gaps_filled']}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Gap detection failed (expected in demo): {e}")
        
        self.logger.info("✅ API-agnostic collection demo completed")
        self.logger.info("-" * 50)
    
    @handles_errors(fallback=False, context="demo_comprehensive_features")
    @traced(span_name="demo_comprehensive_features", log_args=False, log_result_len_only=True)
    async def demo_comprehensive_features(self):
        """Demonstrate comprehensive features integration."""
        self.logger.info("🎯 DEMO 5: Comprehensive Features Integration")
        self.logger.info("-" * 50)
        
        # Create a comprehensive data collector
        collector = EnhancedAPIAgnosticDataCollector("BINANCE", "ETHUSDT", "1m")
        
        self.logger.info("📊 Testing comprehensive data collection workflow...")
        
        # Step 1: Collect some data
        self.logger.info("🔄 Step 1: Collecting initial data...")
        try:
            start_time = datetime.now() - timedelta(minutes=10)
            end_time = datetime.now()
            
            result = await collector.collect_data_for_period(
                start_time=start_time,
                end_time=end_time,
                data_types=["klines"]
            )
            
            self.logger.info(f"✅ Initial collection: {result['total_rows_collected']} rows")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Initial collection failed (expected in demo): {e}")
        
        # Step 2: Detect gaps
        self.logger.info("🔄 Step 2: Detecting gaps...")
        try:
            gap_result = await collector.detect_and_fill_gaps(data_types=["klines"])
            
            self.logger.info(f"✅ Gap detection: {gap_result['total_gaps_found']} gaps found")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Gap detection failed (expected in demo): {e}")
        
        # Step 3: Get comprehensive summary
        self.logger.info("🔄 Step 3: Generating comprehensive summary...")
        summary = collector.get_collection_summary()
        
        self.logger.info(f"📊 Comprehensive Summary:")
        self.logger.info(f"   📈 Total Data: {summary['total_data_collected']} rows")
        self.logger.info(f"   📊 Klines: {summary['klines_rows']} rows")
        self.logger.info(f"   🕐 Gaps Detected: {summary['gaps_detected']}")
        self.logger.info(f"   ⏱️ Duration: {summary['total_duration']:.2f}s")
        
        self.logger.info("✅ Comprehensive features demo completed")
        self.logger.info("-" * 50)
    
    @handles_errors(fallback=False, context="run_complete_demo")
    @traced(span_name="run_complete_demo", log_args=False, log_result_len_only=True)
    @with_enhanced_mlflow_logging
    async def run_complete_demo(self):
        """Run the complete enhanced data collection demo."""
        self.logger.info("🚀 Starting Complete Enhanced Data Collection Demo")
        self.logger.info("=" * 80)
        
        try:
            # Run all demos
            await self.demo_field_mappings()
            await self.demo_data_validation()
            await self.demo_data_qualification()
            await self.demo_api_agnostic_collection()
            await self.demo_comprehensive_features()
            
            # Calculate total demo time
            total_duration = time.time() - self.demo_start_time
            
            self.logger.info("=" * 80)
            self.logger.info("🎉 ENHANCED DATA COLLECTION DEMO COMPLETED SUCCESSFULLY!")
            self.logger.info("=" * 80)
            self.logger.info("📊 Demo Summary:")
            self.logger.info("   ✅ Field mappings for multiple exchanges")
            self.logger.info("   ✅ Enhanced data validation with decorators")
            self.logger.info("   ✅ Data qualification with duplicate removal")
            self.logger.info("   ✅ API-agnostic data collection")
            self.logger.info("   ✅ Comprehensive gap detection")
            self.logger.info("   ✅ Incremental downloading")
            self.logger.info(f"   ⏱️ Total Demo Duration: {total_duration:.2f} seconds")
            self.logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Demo failed with exception: {e}")
            return False


# Main execution function
@handles_errors(fallback=False, context="main")
@traced(span_name="main", log_args=False, log_result_len_only=True)
@log_execution_time
async def main():
    """Main function to run the enhanced data collection demo."""
    logger.info("🚀 Enhanced Data Collection Framework Demo")
    logger.info("=" * 80)
    logger.info("📋 This demo showcases all enhanced features:")
    logger.info("   ✅ Extensive logging and printing")
    logger.info("   ✅ Integration with utils/ decorators")
    logger.info("   ✅ Field mapping for different exchanges")
    logger.info("   ✅ Data qualification with duplicate removal")
    logger.info("   ✅ API-agnostic data collection")
    logger.info("   ✅ Comprehensive gap detection")
    logger.info("   ✅ Incremental downloading")
    logger.info("=" * 80)
    
    try:
        # Create and run demo
        demo = EnhancedDataCollectionDemo()
        success = await demo.run_complete_demo()
        
        if success:
            logger.info("🎉 Demo completed successfully!")
            return 0
        else:
            logger.error("❌ Demo failed!")
            return 1
            
    except Exception as e:
        logger.exception(f"❌ Demo failed with exception: {e}")
        return 1


if __name__ == "__main__":
    # Run the demo
    exit_code = asyncio.run(main())
    sys.exit(exit_code)