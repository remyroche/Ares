"""
SR Training Data Builder

Wrapper around SRQualityDataCollector for convenient data collection and preparation.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import asyncio

from src.tactician.sr_levels.ml_quality.sr_quality_data_collector import SRQualityDataCollector

logger = logging.getLogger(__name__)


class SRTrainingDataBuilder:
    """Builds training datasets for SR performance prediction.
    
    Wraps SRQualityDataCollector with convenience methods for:
    - Multi-symbol data collection
    - Train/validation splitting
    - Feature subset selection
    - Data quality checks
    """
    
    def __init__(self):
        """Initialize training data builder."""
        self.collector = SRQualityDataCollector()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def collect_data(self,
                          symbol: str,
                          exchange: str,
                          start_date: str,
                          end_date: str,
                          timeframe: str = '1h',
                          forward_days: int = 10,
                          sample_freq_days: int = 7) -> pd.DataFrame:
        """Collect training data for a single symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            exchange: Exchange name (e.g., 'binance')
            start_date: Start date (e.g., '2023-01-01')
            end_date: End date (e.g., '2024-01-01')
            timeframe: Timeframe (e.g., '1h', '4h', '1d')
            forward_days: Days to look forward for labeling
            sample_freq_days: Sample frequency in days
            
        Returns:
            DataFrame with features and labels
        """
        self.logger.info(f"📊 Collecting data for {symbol} {exchange}")
        
        data = await self.collector.collect_training_data(
            symbol=symbol,
            exchange=exchange,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            forward_days=forward_days,
            sample_freq_days=sample_freq_days
        )
        
        self.logger.info(f"✅ Collected {len(data)} samples for {symbol}")
        
        return data
    
    async def collect_multi_symbol(self,
                                   symbols: List[str],
                                   exchange: str,
                                   start_date: str,
                                   end_date: str,
                                   timeframe: str = '1h',
                                   forward_days: int = 10,
                                   sample_freq_days: int = 7) -> pd.DataFrame:
        """Collect training data for multiple symbols.
        
        Args:
            symbols: List of trading symbols
            exchange: Exchange name
            start_date: Start date
            end_date: End date
            timeframe: Timeframe
            forward_days: Days to look forward for labeling
            sample_freq_days: Sample frequency in days
            
        Returns:
            Combined DataFrame with all symbols
        """
        self.logger.info(f"📊 Collecting data for {len(symbols)} symbols")
        
        all_data = []
        
        for symbol in symbols:
            try:
                data = await self.collect_data(
                    symbol=symbol,
                    exchange=exchange,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe=timeframe,
                    forward_days=forward_days,
                    sample_freq_days=sample_freq_days
                )
                
                if not data.empty:
                    all_data.append(data)
                    
            except Exception as e:
                self.logger.error(f"Failed to collect data for {symbol}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No data collected for any symbol")
        
        # Combine all data
        combined = pd.concat(all_data, ignore_index=True)
        
        self.logger.info(f"✅ Collected {len(combined)} total samples from {len(all_data)} symbols")
        
        return combined
    
    def prepare_train_val_split(self,
                                data: pd.DataFrame,
                                val_ratio: float = 0.2,
                                time_based: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and validation sets.
        
        Args:
            data: Full dataset
            val_ratio: Fraction for validation (0-1)
            time_based: If True, use time-based split (recommended for time series)
                       If False, use random split
            
        Returns:
            Tuple of (train_data, val_data)
        """
        if time_based:
            # Sort by date and split chronologically
            data_sorted = data.sort_values('date').reset_index(drop=True)
            split_idx = int(len(data_sorted) * (1 - val_ratio))
            
            train_data = data_sorted.iloc[:split_idx].copy()
            val_data = data_sorted.iloc[split_idx:].copy()
            
            self.logger.info(f"📊 Time-based split:")
            self.logger.info(f"   Train: {len(train_data)} samples ({train_data['date'].min()} to {train_data['date'].max()})")
            self.logger.info(f"   Val:   {len(val_data)} samples ({val_data['date'].min()} to {val_data['date'].max()})")
            
        else:
            # Random split
            val_size = int(len(data) * val_ratio)
            val_data = data.sample(n=val_size, random_state=42)
            train_data = data.drop(val_data.index)
            
            self.logger.info(f"📊 Random split:")
            self.logger.info(f"   Train: {len(train_data)} samples")
            self.logger.info(f"   Val:   {len(val_data)} samples")
        
        return train_data, val_data
    
    def filter_untested_levels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter out SR levels that were not tested in the forward window.
        
        Args:
            data: Full dataset
            
        Returns:
            Filtered dataset with only tested levels
        """
        if 'hit_rate' not in data.columns:
            self.logger.warning("No 'hit_rate' column found, cannot filter untested levels")
            return data
        
        original_len = len(data)
        filtered = data[data['hit_rate'] > 0].copy()
        removed = original_len - len(filtered)
        
        self.logger.info(f"📊 Filtered untested levels:")
        self.logger.info(f"   Original: {original_len} samples")
        self.logger.info(f"   Tested:   {len(filtered)} samples ({len(filtered)/original_len*100:.1f}%)")
        self.logger.info(f"   Removed:  {removed} untested levels")
        
        return filtered
    
    def get_feature_subset(self, 
                          data: pd.DataFrame,
                          feature_groups: Optional[List[str]] = None) -> List[str]:
        """Get subset of features based on groups.
        
        Args:
            data: Dataset with features
            feature_groups: List of feature group prefixes to include
                          Examples: ['basic', 'bounce', 'time', 'market', 'regime']
                          If None, returns all features
            
        Returns:
            List of feature column names
        """
        all_features = [c for c in data.columns if c.startswith('feature_')]
        
        if feature_groups is None:
            return all_features
        
        # Feature group mapping
        group_patterns = {
            'basic': ['strength', 'prominence', 'width', 'volume_confirmation', 
                     'consistency', 'touch_count', 'age_bars', 'failure_count'],
            'bounce': ['bounce_ratio', 'bounce_consistency', 'volume_weighted_bounce',
                      'strong_bounce'],
            'time': ['time_decay', 'recency', 'age_category', 'time_adjusted'],
            'market': ['market_volatility', 'market_volume', 'market_trend', 'market_momentum'],
            'regime': ['vol_adjusted', 'trend_alignment', 'regime'],
            'interaction': ['_x_'],  # Features with '_x_' in name
            'position': ['price_position', 'distance_to_current', 'is_support'],
            'statistical': ['zscore', 'percentile'],
            'confluence': ['method_count', 'method_confluence', 'method_diversity', 'agreement'],
        }
        
        selected = []
        
        for feature in all_features:
            feature_lower = feature.lower()
            
            for group in feature_groups:
                if group in group_patterns:
                    patterns = group_patterns[group]
                    if any(pattern in feature_lower for pattern in patterns):
                        selected.append(feature)
                        break
        
        self.logger.info(f"📊 Feature selection:")
        self.logger.info(f"   Groups: {feature_groups}")
        self.logger.info(f"   Total features: {len(all_features)}")
        self.logger.info(f"   Selected: {len(selected)}")
        
        return selected
    
    def check_data_quality(self, data: pd.DataFrame) -> Dict[str, any]:
        """Check data quality and return statistics.
        
        Args:
            data: Dataset to check
            
        Returns:
            Dictionary with quality metrics
        """
        self.logger.info(f"🔍 Checking data quality...")
        
        stats = {
            'total_samples': len(data),
            'date_range': (data['date'].min(), data['date'].max()) if 'date' in data.columns else None,
            'symbols': data['symbol'].nunique() if 'symbol' in data.columns else 0,
            'timeframes': data['timeframe'].unique().tolist() if 'timeframe' in data.columns else [],
        }
        
        # Target distribution
        targets = ['bounce_strength', 'hold_strength', 'trade_profit']
        for target in targets:
            if target in data.columns:
                stats[f'{target}_mean'] = data[target].mean()
                stats[f'{target}_std'] = data[target].std()
                stats[f'{target}_min'] = data[target].min()
                stats[f'{target}_max'] = data[target].max()
        
        # Missing values
        feature_cols = [c for c in data.columns if c.startswith('feature_')]
        missing_pct = (data[feature_cols].isna().sum() / len(data) * 100).to_dict()
        high_missing = {k: v for k, v in missing_pct.items() if v > 10}
        
        if high_missing:
            stats['high_missing_features'] = high_missing
            self.logger.warning(f"⚠️ Features with >10% missing values: {len(high_missing)}")
        
        # Hit rate distribution
        if 'hit_rate' in data.columns:
            tested_pct = (data['hit_rate'] > 0).sum() / len(data) * 100
            stats['tested_levels_pct'] = tested_pct
            self.logger.info(f"   Tested levels: {tested_pct:.1f}%")
        
        # Log summary
        self.logger.info(f"   Total samples: {stats['total_samples']}")
        self.logger.info(f"   Symbols: {stats['symbols']}")
        self.logger.info(f"   Date range: {stats['date_range']}")
        
        for target in targets:
            if target in data.columns:
                self.logger.info(f"   {target}: μ={stats[f'{target}_mean']:.3f}, σ={stats[f'{target}_std']:.3f}")
        
        return stats
    
    def apply_confidence_weighting(self,
                                   data: pd.DataFrame,
                                   method: str = 'quality_based') -> pd.DataFrame:
        """Apply confidence-based sample weighting.
        
        Args:
            data: Training data
            method: Weighting method ('quality_based', 'tiered', 'exponential')
            
        Returns:
            Data with 'sample_weight' column added
        """
        return self.collector.add_confidence_weights(data, method)
    
    def save_data(self, data: pd.DataFrame, save_path: Path):
        """Save training data to file.
        
        Args:
            data: Training data
            save_path: Path to save (supports .csv, .parquet, .pkl)
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if save_path.suffix == '.csv':
            data.to_csv(save_path, index=False)
        elif save_path.suffix == '.parquet':
            data.to_parquet(save_path, index=False)
        elif save_path.suffix == '.pkl':
            data.to_pickle(save_path)
        else:
            raise ValueError(f"Unsupported file format: {save_path.suffix}")
        
        self.logger.info(f"✅ Saved {len(data)} samples to {save_path}")
    
    def load_data(self, load_path: Path) -> pd.DataFrame:
        """Load training data from file.
        
        Args:
            load_path: Path to load from
            
        Returns:
            Training data
        """
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"File not found: {load_path}")
        
        if load_path.suffix == '.csv':
            data = pd.read_csv(load_path)
        elif load_path.suffix == '.parquet':
            data = pd.read_parquet(load_path)
        elif load_path.suffix == '.pkl':
            data = pd.read_pickle(load_path)
        else:
            raise ValueError(f"Unsupported file format: {load_path.suffix}")
        
        self.logger.info(f"✅ Loaded {len(data)} samples from {load_path}")
        
        return data


def collect_data_sync(symbol: str,
                     exchange: str,
                     start_date: str,
                     end_date: str,
                     **kwargs) -> pd.DataFrame:
    """Synchronous wrapper for data collection.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        start_date: Start date
        end_date: End date
        **kwargs: Additional arguments for collect_data
        
    Returns:
        Training data
    """
    builder = SRTrainingDataBuilder()
    return asyncio.run(builder.collect_data(symbol, exchange, start_date, end_date, **kwargs))

