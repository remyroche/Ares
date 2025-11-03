"""
Collect SR training data from multiple timeframes.

Uses existing data in historical_data/binance/ethusdt/processed/
and resamples as needed for 4h and 1d timeframes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional
import asyncio
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tactician.sr_levels.ml_quality.sr_quality_data_collector import SRQualityDataCollector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiTimeframeDataLoader:
    """Load and resample data from historical_data directory."""
    
    def __init__(self, base_path: str = 'historical_data/binance/ethusdt/processed'):
        self.base_path = Path(base_path)
        logger.info(f"📂 Multi-timeframe data loader initialized")
        logger.info(f"   Base path: {self.base_path}")
    
    def load_timeframe(self, timeframe: str) -> pd.DataFrame:
        """
        Load data for specific timeframe.
        
        Args:
            timeframe: '1m', '5m', '15m', '1h', etc.
            
        Returns:
            DataFrame with OHLCV data
        """
        # Map timeframe to directory
        tf_map = {
            '1m': 'ethusdt_1m',
            '5m': 'ethusdt_5m',
            '15m': 'ethusdt_15m',
            '30m': 'ethusdt_30m',
            '1h': 'ethusdt_1h'
        }
        
        if timeframe not in tf_map:
            logger.error(f"❌ Unsupported timeframe: {timeframe}")
            return pd.DataFrame()
        
        data_dir = self.base_path / tf_map[timeframe]
        
        if not data_dir.exists():
            logger.warning(f"⚠️ Data directory not found: {data_dir}")
            return pd.DataFrame()
        
        # Load all parquet files (partitioned by year)
        all_files = sorted(list(data_dir.rglob('*.parquet')))
        
        if not all_files:
            logger.warning(f"⚠️ No parquet files found in {data_dir}")
            return pd.DataFrame()
        
        logger.info(f"📂 Loading {len(all_files)} files for {timeframe}...")
        
        # Load and concatenate
        dfs = []
        for file_path in all_files:
            try:
                df = pd.read_parquet(file_path)
                
                # Ensure timestamp index
                if 'timestamp' in df.columns and 'timestamp' not in df.index.names:
                    df = df.set_index('timestamp')
                
                dfs.append(df)
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {file_path.name}: {e}")
        
        if not dfs:
            return pd.DataFrame()
        
        combined = pd.concat(dfs, ignore_index=False)
        combined = combined.sort_index()
        
        # Remove duplicates
        combined = combined[~combined.index.duplicated(keep='first')]
        
        # Ensure we have required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in combined.columns for col in required):
            logger.error(f"❌ Missing required columns in {timeframe} data")
            logger.error(f"   Available columns: {combined.columns.tolist()}")
            return pd.DataFrame()
        
        logger.info(f"✅ Loaded {len(combined):,} bars for {timeframe}")
        logger.info(f"   Date range: {combined.index.min()} to {combined.index.max()}")
        
        return combined
    
    def resample_to_timeframe(self, data: pd.DataFrame, target_tf: str) -> pd.DataFrame:
        """
        Resample data to target timeframe.
        
        Args:
            data: Source data (typically 1h or 1m)
            target_tf: Target timeframe ('4h', '1d', '1w')
            
        Returns:
            Resampled DataFrame
        """
        # Resample rules
        resample_map = {
            '4h': '4H',
            '1d': '1D',
            '1w': '1W'
        }
        
        if target_tf not in resample_map:
            logger.error(f"❌ Unsupported resample timeframe: {target_tf}")
            return pd.DataFrame()
        
        rule = resample_map[target_tf]
        
        logger.info(f"🔄 Resampling to {target_tf}...")
        
        resampled = data.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        logger.info(f"✅ Resampled to {len(resampled):,} bars ({target_tf})")
        
        return resampled


async def collect_all_timeframes(
    symbol: str = 'ETHUSDT',
    exchange: str = 'binance',
    start_date: str = '2023-01-01',
    end_date: str = '2024-11-01',
    output_dir: str = 'data_cache/sr_ml_training',
    filter_top_pct: float = 80.0
) -> Dict[str, str]:
    """
    Collect SR training data from all timeframes.
    
    Returns dict mapping timeframe -> output file path.
    """
    loader = MultiTimeframeDataLoader()
    collector = SRQualityDataCollector()
    
    timeframes = {
        '15m': None,  # Direct load
        '1h': None,   # Direct load
        '4h': '1h',   # Resample from 1h
        '1d': '1h'    # Resample from 1h
    }
    
    output_paths = {}
    all_training_data = []
    
    for tf, source_tf in timeframes.items():
        logger.info(f"\n{'='*70}")
        logger.info(f"  Processing {tf} timeframe")
        logger.info(f"{'='*70}")
        
        # Load or resample data
        try:
            if source_tf is None:
                # Direct load
                data = loader.load_timeframe(tf)
            else:
                # Resample
                source_data = loader.load_timeframe(source_tf)
                if source_data.empty:
                    logger.warning(f"⚠️ No source data for {tf}, skipping")
                    continue
                data = loader.resample_to_timeframe(source_data, tf)
            
            if data.empty:
                logger.warning(f"⚠️ No data for {tf}, skipping")
                continue
            
            # Filter date range
            data = data.loc[start_date:end_date]
            
            if data.empty:
                logger.warning(f"⚠️ No data in date range for {tf}")
                continue
            
            logger.info(f"📊 Data ready: {len(data):,} bars")
            
        except Exception as e:
            logger.error(f"❌ Failed to load {tf} data: {e}", exc_info=True)
            continue
        
        # Collect training data
        try:
            # Save the resampled data temporarily if needed
            temp_data_path = Path(output_dir) / f'temp_{symbol}_{tf}.parquet'
            temp_data_path.parent.mkdir(parents=True, exist_ok=True)
            data.to_parquet(temp_data_path)
            
            # Use data loader's collect method
            # Note: This will detect SR levels on historical data
            logger.info(f"🔍 Detecting SR levels and measuring performance...")
            
            training_data = await collector.collect_training_data(
                symbol=symbol,
                exchange=exchange,
                start_date=start_date,
                end_date=end_date,
                timeframe=tf,
                forward_days=10,
                sample_freq_days=7
            )
            
            if training_data is None or len(training_data) == 0:
                logger.warning(f"⚠️ No training data generated for {tf}")
                continue
            
            # Add timeframe column
            training_data['timeframe'] = tf
            
            # Filter to top quality
            if filter_top_pct < 100.0:
                filtered = collector.filter_top_quality_levels(training_data, filter_top_pct)
            else:
                filtered = training_data
            
            # Save individual timeframe
            output_path = Path(output_dir) / f'sr_quality_training_data_{tf}.parquet'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            filtered.to_parquet(output_path)
            
            logger.info(f"✅ Saved {len(filtered):,} samples to {output_path}")
            output_paths[tf] = str(output_path)
            all_training_data.append(filtered)
            
        except Exception as e:
            logger.error(f"❌ Failed to collect {tf} training data: {e}", exc_info=True)
    
    # Combine all timeframes
    if all_training_data:
        logger.info(f"\n{'='*70}")
        logger.info(f"  COMBINING ALL TIMEFRAMES")
        logger.info(f"{'='*70}")
        
        combined = pd.concat(all_training_data, ignore_index=True)
        combined_path = Path(output_dir) / 'sr_quality_training_data_all_timeframes.parquet'
        combined.to_parquet(combined_path)
        
        logger.info(f"\n✅ Combined dataset saved: {combined_path}")
        logger.info(f"   Total samples: {len(combined):,}")
        logger.info(f"\n   Samples by timeframe:")
        
        tf_counts = combined['timeframe'].value_counts().sort_index()
        for tf, count in tf_counts.items():
            pct = count / len(combined) * 100
            logger.info(f"     {tf}: {count:,} ({pct:.1f}%)")
        
        # Quality distribution
        logger.info(f"\n   Overall quality distribution:")
        logger.info(f"     Min:    {combined['quality_score'].min():.3f}")
        logger.info(f"     25th:   {combined['quality_score'].quantile(0.25):.3f}")
        logger.info(f"     Median: {combined['quality_score'].median():.3f}")
        logger.info(f"     75th:   {combined['quality_score'].quantile(0.75):.3f}")
        logger.info(f"     Max:    {combined['quality_score'].max():.3f}")
        
        output_paths['combined'] = str(combined_path)
    else:
        logger.error(f"❌ No data collected from any timeframe!")
    
    return output_paths


async def main():
    """Collect all timeframe data."""
    logger.info("\n" + "="*70)
    logger.info("  MULTI-TIMEFRAME SR TRAINING DATA COLLECTION")
    logger.info("="*70)
    logger.info("\nConfiguration:")
    logger.info("  Symbol: ETHUSDT")
    logger.info("  Exchange: binance")
    logger.info("  Date range: 2023-01-01 to 2024-11-01")
    logger.info("  Filter: Top 20% quality levels only")
    logger.info("  Timeframes: 15m, 1h, 4h, 1d")
    
    try:
        output_paths = await collect_all_timeframes(
            symbol='ETHUSDT',
            exchange='binance',
            start_date='2023-01-01',
            end_date='2024-11-01',
            filter_top_pct=80.0  # Top 20%
        )
        
        if output_paths:
            logger.info(f"\n{'='*70}")
            logger.info(f"  COLLECTION COMPLETE!")
            logger.info(f"{'='*70}")
            logger.info(f"\n✅ Collected data for {len(output_paths)} timeframes:")
            for tf, path in output_paths.items():
                logger.info(f"   {tf}: {path}")
            
            logger.info(f"\n🎯 Next steps:")
            logger.info(f"   1. Run validation: python3 scripts/validate_sr_ml_hypotheses.py")
            logger.info(f"   2. Train model: python ares_launcher.py step2.5 --force-rerun")
            logger.info(f"   3. Check Precision@10 improvement")
        else:
            logger.error(f"❌ No data collected!")
            
    except Exception as e:
        logger.error(f"❌ Collection failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())

