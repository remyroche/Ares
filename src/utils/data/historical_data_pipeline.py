"""
Complete Historical Data Pipeline

This module provides a complete pipeline for downloading, processing, and managing
historical Binance klines data with gap detection, feature engineering, and resampling.
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logger import system_logger
from src.utils.data.historical_data_downloader import HistoricalDataDownloader
from src.utils.data.gap_detector import GapDetector
from src.utils.data.basic_returns_engineer import BasicReturnsEngineer
from src.utils.data.optimized_parquet_storage import OptimizedParquetStorage
from src.utils.data.klines_parquet import KlinesParquetManager


class HistoricalDataPipeline:
    """Complete pipeline for historical data management."""

    def __init__(self, data_dir: str = "historical_data"):
        """Initialize the historical data pipeline.
        
        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = data_dir
        self.logger = system_logger.getChild("HistoricalDataPipeline")
        
        # Initialize components
        self.downloader = HistoricalDataDownloader(data_dir)
        self.gap_detector = GapDetector(data_dir)
        self.basic_returns_engineer = BasicReturnsEngineer(data_dir)
        self.optimized_storage = OptimizedParquetStorage(data_dir)
        self.klines_manager = KlinesParquetManager(data_dir)
    
    async def run_complete_pipeline(
        self,
        symbol: str = "ETHUSDT",
        years: int = 3,
        api_key: str = "",
        api_secret: str = "",
        target_intervals: List[str] = None,
        max_gap_minutes: int = 1
    ) -> Dict[str, Any]:
        """Run the complete historical data pipeline.
        
        Args:
            symbol: Trading symbol
            years: Number of years to download
            api_key: Binance API key
            api_secret: Binance API secret
            target_intervals: List of target intervals for resampling
            max_gap_minutes: Maximum allowed gap in minutes
            
        Returns:
            Dictionary with pipeline results
        """
        if target_intervals is None:
            target_intervals = ["5m", "15m", "30m", "1h"]
        
        try:
            self.logger.info(f"🚀 Starting complete pipeline for {symbol}")
            
            results = {
                "symbol": symbol,
                "years": years,
                "target_intervals": target_intervals,
                "steps_completed": [],
                "errors": [],
                "summary": {}
            }
            
            # Step 1: Download historical data
            self.logger.info("📥 Step 1: Downloading historical data")
            download_success = await self.downloader.download_historical_klines(
                symbol=symbol,
                interval="1m",
                years=years,
                api_key=api_key,
                api_secret=api_secret
            )
            
            if download_success:
                results["steps_completed"].append("download")
                download_summary = self.downloader.get_data_summary(symbol)
                results["summary"]["download"] = download_summary
                self.logger.info(f"✅ Download completed: {download_summary}")
            else:
                results["errors"].append("Download failed")
                self.logger.error("❌ Download failed")
                return results
            
            # Step 2: Detect and fill gaps
            self.logger.info("🔍 Step 2: Detecting and filling gaps")
            gaps = self.gap_detector.detect_gaps(symbol, "1m", max_gap_minutes)
            self.gap_detector.log_gaps(gaps)
            
            if gaps:
                gap_results = await self.gap_detector.fill_gaps(gaps, api_key, api_secret)
                results["steps_completed"].append("gap_filling")
                results["summary"]["gap_filling"] = gap_results
                self.logger.info(f"✅ Gap filling completed: {gap_results}")
            else:
                results["steps_completed"].append("gap_detection")
                results["summary"]["gap_detection"] = {"gaps_detected": 0}
                self.logger.info("✅ No gaps detected")
            
            # Step 3: Basic returns feature engineering and resampling
            self.logger.info("🔧 Step 3: Basic returns feature engineering and resampling")
            processing_results = self.basic_returns_engineer.process_symbol_data(
                symbol, "1m", target_intervals
            )
            
            if processing_results["success"]:
                results["steps_completed"].append("feature_engineering")
                results["summary"]["feature_engineering"] = processing_results
                self.logger.info(f"✅ Feature engineering completed: {processing_results}")
            else:
                results["errors"].append(f"Basic returns feature engineering failed: {processing_results.get('error', 'Unknown error')}")
                self.logger.error(f"❌ Basic returns feature engineering failed: {processing_results.get('error', 'Unknown error')}")
            
            # Step 4: Verify data integrity
            self.logger.info("🔍 Step 4: Verifying data integrity")
            verification_results = self._verify_data_integrity(symbol, target_intervals)
            results["steps_completed"].append("verification")
            results["summary"]["verification"] = verification_results
            self.logger.info(f"✅ Verification completed: {verification_results}")
            
            # Final summary
            results["pipeline_success"] = len(results["errors"]) == 0
            results["completion_time"] = datetime.now().isoformat()
            
            self.logger.info(f"🎉 Pipeline completed: {len(results['steps_completed'])} steps, {len(results['errors'])} errors")
            return results
            
        except Exception as e:
            self.logger.exception(f"❌ Pipeline failed: {e}")
            results["errors"].append(str(e))
            results["pipeline_success"] = False
            return results
    
    def _verify_data_integrity(
        self,
        symbol: str,
        target_intervals: List[str]
    ) -> Dict[str, Any]:
        """Verify data integrity across all intervals.
        
        Args:
            symbol: Trading symbol
            target_intervals: List of target intervals
            
        Returns:
            Dictionary with verification results
        """
        try:
            verification_results = {
                "raw_data": {},
                "processed_data": {},
                "overall_success": True
            }
            
            # Check raw data
            raw_info = self.klines_manager.get_data_info(symbol, "1m", "raw")
            verification_results["raw_data"] = raw_info
            
            if not raw_info["available"]:
                verification_results["overall_success"] = False
                self.logger.error("❌ No raw data found")
                return verification_results
            
            # Check processed data for each interval
            all_intervals = ["1m"] + target_intervals
            for interval in all_intervals:
                processed_info = self.klines_manager.get_data_info(symbol, interval, "processed")
                verification_results["processed_data"][interval] = processed_info
                
                if not processed_info["available"]:
                    verification_results["overall_success"] = False
                    self.logger.error(f"❌ No processed data found for {interval}")
            
            # Check for data consistency
            if verification_results["overall_success"]:
                self.logger.info("✅ All data integrity checks passed")
            else:
                self.logger.warning("⚠️ Some data integrity checks failed")
            
            return verification_results
            
        except Exception as e:
            self.logger.exception(f"❌ Data integrity verification failed: {e}")
            return {
                "raw_data": {},
                "processed_data": {},
                "overall_success": False,
                "error": str(e)
            }
    
    def get_pipeline_status(self, symbol: str) -> Dict[str, Any]:
        """Get current pipeline status for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with pipeline status
        """
        try:
            status = {
                "symbol": symbol,
                "raw_data_available": False,
                "processed_data_available": {},
                "data_summary": {},
                "recommendations": []
            }
            
            # Check raw data
            raw_info = self.klines_manager.get_data_info(symbol, "1m", "raw")
            status["raw_data_available"] = raw_info["available"]
            status["data_summary"]["raw"] = raw_info
            
            if raw_info["available"]:
                # Check for gaps
                gaps = self.gap_detector.detect_gaps(symbol, "1m", max_gap_minutes=1)
                if gaps:
                    status["recommendations"].append(f"Found {len(gaps)} gaps in raw data - consider running gap filling")
                
                # Check processed data
                intervals = ["1m", "5m", "15m", "30m", "1h"]
                for interval in intervals:
                    processed_info = self.klines_manager.get_data_info(symbol, interval, "processed")
                    status["processed_data_available"][interval] = processed_info["available"]
                    status["data_summary"][f"processed_{interval}"] = processed_info
                
                # Generate recommendations
                if not any(status["processed_data_available"].values()):
                    status["recommendations"].append("No processed data found - consider running feature engineering")
                
                missing_intervals = [interval for interval, available in status["processed_data_available"].items() if not available]
                if missing_intervals:
                    status["recommendations"].append(f"Missing processed data for intervals: {missing_intervals}")
            else:
                status["recommendations"].append("No raw data found - consider running data download")
            
            return status
            
        except Exception as e:
            self.logger.exception(f"❌ Failed to get pipeline status: {e}")
            return {
                "symbol": symbol,
                "raw_data_available": False,
                "processed_data_available": {},
                "data_summary": {},
                "recommendations": [f"Error getting status: {e}"],
                "error": str(e)
            }
    
    def cleanup_old_data(
        self,
        symbol: str,
        keep_days: int = 30,
        data_type: str = "raw"
    ) -> Dict[str, Any]:
        """Clean up old data to save space.
        
        Args:
            symbol: Trading symbol
            keep_days: Number of days to keep
            data_type: 'raw' or 'processed'
            
        Returns:
            Dictionary with cleanup results
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=keep_days)
            
            self.logger.info(f"🧹 Cleaning up {data_type} data older than {cutoff_date}")
            
            # Get data info
            info = self.klines_manager.get_data_info(symbol, "1m", data_type)
            
            if not info["available"]:
                return {"cleaned_files": 0, "freed_space_mb": 0, "message": "No data to clean"}
            
            # Delete old data
            success = self.klines_manager.delete_data(
                symbol, "1m", data_type, end_date=cutoff_date
            )
            
            if success:
                # Get new data info
                new_info = self.klines_manager.get_data_info(symbol, "1m", data_type)
                freed_space = info["file_size_mb"] - new_info["file_size_mb"]
                
                return {
                    "cleaned_files": info["files_count"] - new_info["files_count"],
                    "freed_space_mb": freed_space,
                    "message": f"Cleaned up {freed_space:.2f} MB of old data"
                }
            else:
                return {"cleaned_files": 0, "freed_space_mb": 0, "message": "Cleanup failed"}
                
        except Exception as e:
            self.logger.exception(f"❌ Cleanup failed: {e}")
            return {"cleaned_files": 0, "freed_space_mb": 0, "message": f"Cleanup failed: {e}"}


# Convenience functions
async def run_ethusdt_pipeline(
    years: int = 3,
    data_dir: str = "historical_data",
    api_key: str = "",
    api_secret: str = "",
    target_intervals: List[str] = None
) -> Dict[str, Any]:
    """Run the complete pipeline for ETHUSDT.
    
    Args:
        years: Number of years to download
        data_dir: Base directory for data storage
        api_key: Binance API key
        api_secret: Binance API secret
        target_intervals: List of target intervals for resampling
        
    Returns:
        Dictionary with pipeline results
    """
    if target_intervals is None:
        target_intervals = ["5m", "15m", "30m", "1h"]
    
    pipeline = HistoricalDataPipeline(data_dir)
    return await pipeline.run_complete_pipeline(
        symbol="ETHUSDT",
        years=years,
        api_key=api_key,
        api_secret=api_secret,
        target_intervals=target_intervals
    )


if __name__ == "__main__":
    # Example usage
    async def main():
        pipeline = HistoricalDataPipeline()
        
        # Run complete pipeline
        results = await pipeline.run_complete_pipeline(
            symbol="ETHUSDT",
            years=3,
            target_intervals=["5m", "15m", "30m", "1h"]
        )
        
        print(f"Pipeline results: {results}")
        
        # Check status
        status = pipeline.get_pipeline_status("ETHUSDT")
        print(f"Pipeline status: {status}")
    
    asyncio.run(main())