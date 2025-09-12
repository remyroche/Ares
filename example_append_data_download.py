#!/usr/bin/env python3
"""
Example: Append Data Download System

This script demonstrates how to use the enhanced data download system
that ensures data is appended to existing files rather than overwritten.

Features demonstrated:
- Batch-based data downloading with unique file naming
- Data consolidation and merging
- Comprehensive monitoring and logging
- Error handling and recovery
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.training.steps.data_collection.enhanced_append_data_downloader import (
    EnhancedAppendDataDownloader,
    download_data_with_append,
    consolidate_data_batches,
    list_data_files
)
from src.training.steps.data_collection.data_consolidation_manager import (
    DataConsolidationManager,
    consolidate_session_data,
    consolidate_time_range_data,
    consolidate_all_data
)
from src.training.steps.data_collection.data_download_monitor import (
    DataDownloadMonitor,
    start_download_session,
    update_download_progress,
    end_download_session,
    get_download_status,
    get_monitoring_dashboard
)
from src.utils.logger import system_logger

logger = system_logger.getChild("AppendDataDownloadExample")

async def example_basic_append_download():
    """Example 1: Basic append data download."""
    logger.info("🎯 Example 1: Basic Append Data Download")
    logger.info("=" * 60)
    
    try:
        # Download data with append functionality
        result = await download_data_with_append(
            symbol="ETHUSDT",
            exchange="BINANCE",
            data_type="klines",
            timeframe="1m",
            max_batches=3
        )
        
        if result['success']:
            logger.info(f"✅ Download successful!")
            logger.info(f"📊 Downloaded {result['total_rows']} rows in {result['successful_batches']} batches")
            logger.info(f"📁 Files created: {len(result['batch_results'])}")
            
            # Show file details
            for batch_result in result['batch_results']:
                if batch_result['success']:
                    logger.info(f"   📄 Batch {batch_result['batch_number']}: {batch_result['rows']} rows -> {batch_result['file_path']}")
        else:
            logger.error(f"❌ Download failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"❌ Example 1 failed: {e}")

async def example_monitored_download():
    """Example 2: Monitored data download with progress tracking."""
    logger.info("\n🎯 Example 2: Monitored Data Download")
    logger.info("=" * 60)
    
    try:
        # Initialize monitor
        monitor = DataDownloadMonitor()
        
        # Start a monitored session
        session_id = f"monitored_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session = start_download_session(
            session_id=session_id,
            symbol="BTCUSDT",
            exchange="BINANCE",
            data_type="klines",
            timeframe="5m"
        )
        
        logger.info(f"🚀 Started monitored session: {session['session_id']}")
        
        # Download data with monitoring
        downloader = EnhancedAppendDataDownloader()
        result = await downloader.download_with_append(
            symbol="BTCUSDT",
            exchange="BINANCE",
            data_type="klines",
            timeframe="5m",
            max_batches=2
        )
        
        # Update monitor with batch results
        for batch_result in result['batch_results']:
            update_download_progress(
                session_id=session_id,
                batch_number=batch_result['batch_number'],
                batch_success=batch_result['success'],
                rows_downloaded=batch_result['rows'],
                batch_duration=1.5,  # Simulated duration
                file_path=batch_result.get('file_path')
            )
        
        # End the session
        final_summary = end_download_session(session_id, 'completed')
        
        logger.info(f"✅ Monitored download completed!")
        logger.info(f"📊 Final summary: {final_summary['total_rows']} rows, {final_summary['success_rate']:.1f}% success rate")
        
        # Show monitoring dashboard
        dashboard = get_monitoring_dashboard()
        logger.info(f"📈 Dashboard: {dashboard['overview']['total_sessions']} total sessions")
        
    except Exception as e:
        logger.error(f"❌ Example 2 failed: {e}")

async def example_data_consolidation():
    """Example 3: Data consolidation and merging."""
    logger.info("\n🎯 Example 3: Data Consolidation")
    logger.info("=" * 60)
    
    try:
        # First, download some data to consolidate
        logger.info("📥 Downloading data for consolidation...")
        
        downloader = EnhancedAppendDataDownloader()
        
        # Download multiple sessions
        sessions = []
        for i in range(2):
            session_id = f"consolidation_session_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            result = await downloader.download_with_append(
                symbol="ADAUSDT",
                exchange="BINANCE",
                data_type="klines",
                timeframe="1m",
                max_batches=2
            )
            sessions.append((session_id, result))
        
        logger.info(f"✅ Downloaded data in {len(sessions)} sessions")
        
        # Consolidate by session
        consolidation_manager = DataConsolidationManager()
        
        for session_id, result in sessions:
            if result['success']:
                logger.info(f"🔄 Consolidating session {session_id}...")
                consolidate_result = await consolidation_manager.consolidate_by_session(
                    symbol="ADAUSDT",
                    exchange="BINANCE",
                    data_type="klines",
                    timeframe="1m",
                    session_id=session_id,
                    remove_originals=False
                )
                
                if consolidate_result['success']:
                    logger.info(f"✅ Consolidated: {consolidate_result['total_rows']} rows -> {consolidate_result['consolidated_file']}")
                else:
                    logger.warning(f"⚠️ Consolidation failed: {consolidate_result.get('error')}")
        
        # Consolidate all available data
        logger.info("🔄 Consolidating all available data...")
        all_consolidate_result = await consolidation_manager.consolidate_all_available(
            symbol="ADAUSDT",
            exchange="BINANCE",
            data_type="klines",
            timeframe="1m",
            remove_originals=False
        )
        
        if all_consolidate_result['success']:
            logger.info(f"✅ All data consolidated: {all_consolidate_result['total_rows']} rows")
        else:
            logger.warning(f"⚠️ All data consolidation failed: {all_consolidate_result.get('error')}")
            
    except Exception as e:
        logger.error(f"❌ Example 3 failed: {e}")

async def example_list_and_manage_data():
    """Example 4: List and manage downloaded data."""
    logger.info("\n🎯 Example 4: Data Management")
    logger.info("=" * 60)
    
    try:
        # List all available data
        logger.info("📁 Listing all available data...")
        data_files = await list_data_files()
        
        logger.info(f"📊 Found {data_files['total_files']} files ({data_files['total_size_mb']:.2f} MB)")
        
        # Show batch files
        if data_files['batch_files']:
            logger.info(f"📦 Batch files ({len(data_files['batch_files'])}):")
            for file_info in data_files['batch_files'][:5]:  # Show first 5
                logger.info(f"   📄 {file_info['file_name']} - {file_info['size_mb']:.2f} MB - {file_info['row_count']} rows")
        
        # Show consolidated files
        if data_files['consolidated_files']:
            logger.info(f"📚 Consolidated files ({len(data_files['consolidated_files'])}):")
            for file_info in data_files['consolidated_files'][:5]:  # Show first 5
                logger.info(f"   📄 {file_info['file_name']} - {file_info['size_mb']:.2f} MB - {file_info['row_count']} rows")
        
        # List data for specific symbol
        logger.info("\n🔍 Listing data for ETHUSDT...")
        eth_data = await list_data_files(symbol="ETHUSDT", exchange="BINANCE", data_type="klines")
        
        logger.info(f"📊 ETHUSDT data: {eth_data['total_files']} files ({eth_data['total_size_mb']:.2f} MB)")
        
    except Exception as e:
        logger.error(f"❌ Example 4 failed: {e}")

async def example_error_handling():
    """Example 5: Error handling and recovery."""
    logger.info("\n🎯 Example 5: Error Handling")
    logger.info("=" * 60)
    
    try:
        # Try to download with invalid parameters to demonstrate error handling
        logger.info("🧪 Testing error handling with invalid parameters...")
        
        downloader = EnhancedAppendDataDownloader()
        
        # This should fail gracefully
        result = await downloader.download_with_append(
            symbol="INVALID_SYMBOL",
            exchange="INVALID_EXCHANGE",
            data_type="klines",
            timeframe="1m",
            max_batches=1
        )
        
        if not result['success']:
            logger.info(f"✅ Error handling working: {result.get('error', 'Unknown error')}")
        else:
            logger.warning("⚠️ Expected error did not occur")
        
        # Test with valid parameters but network issues
        logger.info("🧪 Testing with valid parameters...")
        
        result = await downloader.download_with_append(
            symbol="ETHUSDT",
            exchange="BINANCE",
            data_type="klines",
            timeframe="1m",
            max_batches=1
        )
        
        if result['success']:
            logger.info(f"✅ Valid download successful: {result['total_rows']} rows")
        else:
            logger.info(f"ℹ️ Download failed (expected in test environment): {result.get('error')}")
        
    except Exception as e:
        logger.error(f"❌ Example 5 failed: {e}")

async def main():
    """Run all examples."""
    logger.info("🚀 Starting Append Data Download Examples")
    logger.info("=" * 80)
    
    try:
        # Run all examples
        await example_basic_append_download()
        await example_monitored_download()
        await example_data_consolidation()
        await example_list_and_manage_data()
        await example_error_handling()
        
        logger.info("\n" + "=" * 80)
        logger.info("🎉 All examples completed successfully!")
        logger.info("=" * 80)
        
        # Final monitoring summary
        dashboard = get_monitoring_dashboard()
        logger.info(f"\n📊 Final Monitoring Summary:")
        logger.info(f"   📈 Total Sessions: {dashboard['overview']['total_sessions']}")
        logger.info(f"   ✅ Success Rate: {dashboard['overview']['success_rate']:.1f}%")
        logger.info(f"   📊 Total Rows: {dashboard['overview']['total_rows_downloaded']}")
        logger.info(f"   📁 Total Files: {dashboard['overview']['total_files_created']}")
        
    except Exception as e:
        logger.error(f"❌ Main execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())