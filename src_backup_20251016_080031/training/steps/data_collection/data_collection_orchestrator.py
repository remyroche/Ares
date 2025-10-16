#!/usr/bin/env python3
"""
Data Collection Orchestrator

This module orchestrates the complete data collection pipeline:
1. Download data using Binance API
2. Detect and fill gaps
3. Perform data quality checks
4. Resample data to multiple timeframes
5. Store processed data

Ensures all components work together seamlessly.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger
from src.utils.error_handler import handles_errors

# Import data collection components
from .unified_data_downloader import UnifiedDataDownloader
from .unified_gap_filler import UnifiedGapFiller
from .unified_resampler import UnifiedResampler

logger = system_logger.getChild("DataCollectionOrchestrator")

class DataCollectionOrchestrator:
    """Orchestrates the complete data collection pipeline."""

    def __init__(self, data_cache_path: str = "data_cache"):
        """Initialize the data collection orchestrator."""
        self.data_cache_path = Path(data_cache_path)
        self.data_cache_path.mkdir(exist_ok=True)
        self.logger = logger.getChild('Orchestrator')

        # Initialize components
        self.downloader = UnifiedDataDownloader(str(self.data_cache_path))
        self.gap_filler = UnifiedGapFiller(str(self.data_cache_path))
        self.resampler = UnifiedResampler(str(self.data_cache_path))

        # Pipeline statistics
        self.pipeline_stats = {
            'downloads_attempted': 0,
            'downloads_successful': 0,
            'gaps_detected': 0,
            'gaps_filled': 0,
            'quality_checks_passed': 0,
            'resampling_completed': 0,
            'start_time': None,
            'end_time': None
        }

    @handles_errors(context="data_collection_pipeline")
    async def run_complete_pipeline(
        self,
        symbol: str,
        exchange: str = "BINANCE",
        data_types: List[str] = None,
        timeframes: List[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        lookback_days: int = 30
    ) -> Dict[str, Any]:
        """
        Run the complete data collection pipeline.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            exchange: Exchange name (e.g., 'BINANCE')
            data_types: List of data types to collect ('klines', 'futures')
            timeframes: List of timeframes to generate ('1m', '5m', '15m', '1h', etc.)
            start_date: Start date for data collection
            end_date: End date for data collection
            lookback_days: Default lookback period in days

        Returns:
            Pipeline execution results
        """
        self.pipeline_stats['start_time'] = datetime.now()
        self.logger.info("🚀 Starting complete data collection pipeline")
        self.logger.info(f"   Symbol: {symbol}, Exchange: {exchange}")
        self.logger.info(f"   Data types: {data_types or ['klines', 'futures']}")
        self.logger.info(f"   Timeframes: {timeframes or ['1m', '5m', '15m', '1h']}")

        # Set defaults
        if data_types is None:
            data_types = ['klines', 'futures']
        if timeframes is None:
            timeframes = ['1m', '5m', '15m', '30m', '1h']
        if start_date is None:
            start_date = datetime.now() - timedelta(days=lookback_days)
        if end_date is None:
            end_date = datetime.now()

        results = {
            'success': True,
            'stages': {},
            'errors': [],
            'warnings': []
        }

        try:
            # Stage 1: Data Download
            self.logger.info("📥 Stage 1: Data Download")
            download_results = await self._run_download_stage(
                symbol, exchange, data_types, start_date, end_date
            )
            results['stages']['download'] = download_results

            if not download_results['success']:
                results['success'] = False
                results['errors'].append("Download stage failed")
                return results

            # Stage 2: Gap Detection and Filling
            self.logger.info("🔧 Stage 2: Gap Detection and Filling")
            gap_results = await self._run_gap_filling_stage(
                symbol, exchange, data_types, start_date, end_date
            )
            results['stages']['gap_filling'] = gap_results

            # Stage 3: Data Quality Checks
            self.logger.info("✅ Stage 3: Data Quality Checks")
            quality_results = await self._run_quality_checks_stage(
                symbol, exchange, data_types, start_date, end_date
            )
            results['stages']['quality_checks'] = quality_results

            # Stage 4: Data Resampling
            self.logger.info("📊 Stage 4: Data Resampling")
            resampling_results = await self._run_resampling_stage(
                symbol, exchange, timeframes, start_date, end_date
            )
            results['stages']['resampling'] = resampling_results

            # Stage 5: Final Storage and Summary
            self.logger.info("💾 Stage 5: Final Storage and Summary")
            storage_results = await self._run_storage_stage(
                symbol, exchange, data_types, timeframes
            )
            results['stages']['storage'] = storage_results

        except Exception as e:
            self.logger.exception(f"❌ Pipeline execution failed: {e}")
            results['success'] = False
            results['errors'].append(str(e))

        finally:
            self.pipeline_stats['end_time'] = datetime.now()
            results['pipeline_stats'] = self._get_pipeline_stats()
            results['execution_time'] = str(
                self.pipeline_stats['end_time'] - self.pipeline_stats['start_time']
            )

            self.logger.info("🏁 Pipeline execution completed")
            self.logger.info(f"   Success: {results['success']}")
            self.logger.info(f"   Execution time: {results['execution_time']}")

        return results

    async def _run_download_stage(
        self,
        symbol: str,
        exchange: str,
        data_types: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Run the data download stage."""
        results = {
            'success': True,
            'downloads_attempted': 0,
            'downloads_successful': 0,
            'total_rows': 0,
            'errors': []
        }

        for data_type in data_types:
            self.logger.info(f"   Downloading {data_type} data...")
            results['downloads_attempted'] += 1

            try:
                if data_type == 'klines':
                    success, data, error = await self.downloader.download_klines(
                        symbol, exchange, "1m", start_date, end_date
                    )
                else:
                    success, data, error = False, [], f"Unsupported data type: {data_type}"

                if success:
                    results['downloads_successful'] += 1
                    results['total_rows'] += len(data) if data else 0
                    self.pipeline_stats['downloads_successful'] += 1
                    self.logger.info(f"     ✅ Downloaded {len(data) if data else 0} {data_type} records")
                else:
                    results['errors'].append(f"{data_type}: {error}")
                    self.logger.error(f"     ❌ {data_type} download failed: {error}")

            except Exception as e:
                results['errors'].append(f"{data_type}: {str(e)}")
                self.logger.error(f"     ❌ {data_type} download exception: {e}")

        results['success'] = results['downloads_successful'] > 0
        self.pipeline_stats['downloads_attempted'] += results['downloads_attempted']

        return results

    async def _run_gap_filling_stage(
        self,
        symbol: str,
        exchange: str,
        data_types: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Run the gap detection and filling stage."""
        results = {
            'success': True,
            'gaps_detected': 0,
            'gaps_filled': 0,
            'rows_downloaded': 0,
            'errors': []
        }

        for data_type in data_types:
            self.logger.info(f"   Processing {data_type} gaps...")

            try:
                # Detect and fill gaps
                gap_results = await self.gap_filler.detect_and_fill_gaps(
                    symbol, exchange, data_type, start_date, end_date, auto_fill=True
                )

                if gap_results['success']:
                    results['gaps_detected'] += gap_results['gaps_detected']
                    results['gaps_filled'] += gap_results['gaps_filled']
                    results['rows_downloaded'] += gap_results['rows_downloaded']

                    self.pipeline_stats['gaps_detected'] += gap_results['gaps_detected']
                    self.pipeline_stats['gaps_filled'] += gap_results['gaps_filled']

                    self.logger.info(f"     📊 {data_type}: {gap_results['gaps_detected']} gaps detected, {gap_results['gaps_filled']} filled")
                else:
                    results['errors'].extend(gap_results.get('errors', []))
                    self.logger.warning(f"     ⚠️ {data_type} gap processing had issues")

            except Exception as e:
                results['errors'].append(f"{data_type}: {str(e)}")
                self.logger.error(f"     ❌ {data_type} gap processing failed: {e}")

        return results

    async def _run_quality_checks_stage(
        self,
        symbol: str,
        exchange: str,
        data_types: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Run data quality checks stage."""
        results = {
            'success': True,
            'checks_passed': 0,
            'total_checks': 0,
            'quality_scores': {},
            'errors': []
        }

        # Import quality components
        try:
            from .data_quality_components.data_integrity_checker import DataIntegrityChecker
            from .data_quality_components.quality_metrics_calculator import QualityMetricsCalculator

            integrity_checker = DataIntegrityChecker()
            metrics_calculator = QualityMetricsCalculator()

            for data_type in data_types:
                self.logger.info(f"   Checking {data_type} quality...")

                try:
                    # Load recent data file for quality check
                    data_file = self._get_recent_data_file(symbol, exchange, data_type)
                    if data_file and data_file.exists():
                        df = pd.read_parquet(data_file)

                        # Run integrity checks
                        is_valid, integrity_results = integrity_checker.validate_data_integrity(df)

                        # Calculate quality score
                        quality_score = metrics_calculator.calculate_quality_score(integrity_results, df)

                        results['total_checks'] += 1
                        results['quality_scores'][data_type] = quality_score

                        if is_valid and quality_score > 0.7:  # 70% quality threshold
                            results['checks_passed'] += 1
                            self.pipeline_stats['quality_checks_passed'] += 1
                            self.logger.info(f"     ✅ {data_type} quality: {quality_score:.2f}")
                        else:
                            results['errors'].append(f"{data_type} quality below threshold: {quality_score:.2f}")
                            self.logger.warning(f"     ⚠️ {data_type} quality issues: {quality_score:.2f}")
                    else:
                        results['errors'].append(f"No data file found for {data_type}")
                        self.logger.warning(f"     ⚠️ No data file for {data_type} quality check")

                except Exception as e:
                    results['errors'].append(f"{data_type}: {str(e)}")
                    self.logger.error(f"     ❌ {data_type} quality check failed: {e}")

        except ImportError as e:
            results['errors'].append(f"Quality components not available: {e}")
            self.logger.warning(f"⚠️ Quality components not available: {e}")

        results['success'] = results['checks_passed'] == results['total_checks']
        return results

    async def _run_resampling_stage(
        self,
        symbol: str,
        exchange: str,
        timeframes: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Run data resampling stage."""
        results = {
            'success': True,
            'resamples_attempted': 0,
            'resamples_successful': 0,
            'errors': []
        }

        for timeframe in timeframes:
            self.logger.info(f"   Resampling to {timeframe} timeframe...")
            results['resamples_attempted'] += 1

            try:
                # Resample klines data to target timeframe
                resample_result = await self.resampler.resample_to_timeframe(
                    symbol, exchange, "1m", timeframe, start_date, end_date
                )

                if resample_result['success']:
                    results['resamples_successful'] += 1
                    self.pipeline_stats['resampling_completed'] += 1
                    self.logger.info(f"     ✅ Resampled to {timeframe}: {resample_result.get('rows_processed', 0)} rows")
                else:
                    results['errors'].append(f"{timeframe}: {resample_result.get('error', 'Unknown error')}")
                    self.logger.error(f"     ❌ {timeframe} resampling failed: {resample_result.get('error', 'Unknown error')}")

            except Exception as e:
                results['errors'].append(f"{timeframe}: {str(e)}")
                self.logger.error(f"     ❌ {timeframe} resampling exception: {e}")

        results['success'] = results['resamples_successful'] > 0
        return results

    async def _run_storage_stage(
        self,
        symbol: str,
        exchange: str,
        data_types: List[str],
        timeframes: List[str]
    ) -> Dict[str, Any]:
        """Run final storage and organization stage."""
        results = {
            'success': True,
            'files_organized': 0,
            'total_files': 0,
            'storage_path': str(self.data_cache_path),
            'errors': []
        }

        # This is a placeholder for any final storage organization
        # In a real implementation, you might want to:
        # - Move files to organized directories
        # - Create metadata files
        # - Generate summary reports
        # - Compress old files

        self.logger.info(f"   Organizing files in {self.data_cache_path}")

        # Count files by type
        for data_type in data_types:
            pattern = f"*{data_type}*.parquet"
            files = list(self.data_cache_path.glob(f"**/{pattern}"))
            results['total_files'] += len(files)
            self.logger.info(f"     📁 {data_type}: {len(files)} files")

        for timeframe in timeframes:
            pattern = f"*_{timeframe}_*.parquet"
            files = list(self.data_cache_path.glob(f"**/{pattern}"))
            self.logger.info(f"     📊 {timeframe}: {len(files)} files")

        results['files_organized'] = results['total_files']
        self.logger.info(f"   ✅ Organized {results['files_organized']} data files")

        return results

    def _get_recent_data_file(self, symbol: str, exchange: str, data_type: str) -> Optional[Path]:
        """Get the most recent data file for a symbol/exchange/data_type combination."""
        try:
            if data_type == 'klines':
                pattern = f"{data_type}_{exchange}_{symbol}_1m_*.parquet"
            elif data_type == 'futures':
                pattern = f"{data_type}_{exchange}_{symbol}_*.parquet"
            else:
                return None

            files = list(self.data_cache_path.glob(pattern))
            if files:
                # Return the most recent file
                return max(files, key=lambda f: f.stat().st_mtime)
            return None

        except Exception:
            return None

    def _get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        stats = self.pipeline_stats.copy()

        # Calculate derived statistics
        total_time = None
        if stats['start_time'] and stats['end_time']:
            total_time = stats['end_time'] - stats['start_time']
            stats['total_execution_time_seconds'] = total_time.total_seconds()

        # Calculate success rates
        stats['download_success_rate'] = (
            stats['downloads_successful'] / max(stats['downloads_attempted'], 1) * 100
        )
        stats['gap_fill_success_rate'] = (
            stats['gaps_filled'] / max(stats['gaps_detected'], 1) * 100
        )

        return stats

# Convenience functions for backward compatibility
@handles_errors()
async def run_data_collection_pipeline(
    symbol: str,
    exchange: str = "BINANCE",
    **kwargs
) -> Dict[str, Any]:
    """Convenience function to run the complete data collection pipeline."""
    orchestrator = DataCollectionOrchestrator()
    return await orchestrator.run_complete_pipeline(symbol, exchange, **kwargs)

@handles_errors()
async def download_and_process_data(
    symbol: str,
    exchange: str = "BINANCE",
    data_types: List[str] = None,
    timeframes: List[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Convenience function for downloading and processing data."""
    if data_types is None:
        data_types = ['klines', 'futures']
    if timeframes is None:
        timeframes = ['1m', '5m', '15m', '1h']

    orchestrator = DataCollectionOrchestrator()
    return await orchestrator.run_complete_pipeline(
        symbol, exchange, data_types, timeframes, **kwargs
    )


