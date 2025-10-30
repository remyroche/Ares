"""
SR Quality Data Collector

Collects historical SR levels and labels them with forward performance metrics.
Uses artifact_manager to load existing downloaded data (no re-downloading).
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
from tqdm import tqdm

# Import artifact manager
from src.training.steps.pre_training.utils.artifact_manager import (
    get_pretraining_artifact_manager, artifact_context
)

logger = logging.getLogger(__name__)


class SRQualityDataCollector:
    """Collects historical SR levels and labels them with performance metrics.
    
    Uses artifact_manager pattern - loads existing data, does NOT re-download.
    """
    
    def __init__(self):
        self.artifact_manager = get_pretraining_artifact_manager()
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def collect_training_data(self, symbol: str, exchange: str, 
                              start_date: str, end_date: str,
                              timeframe: str = '1h',
                              forward_days: int = 10,
                              sample_freq_days: int = 7) -> pd.DataFrame:
        """Collect SR levels from historical data and label with performance.
        
        Process:
        1. Load full historical OHLCV using artifact_manager (already downloaded)
        2. Walk forward through time
        3. For each date: detect SR, look forward, measure performance
        4. Create training samples
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            exchange: Exchange name (e.g., 'binance')
            start_date: Start date for training data (e.g., '2023-01-01')
            end_date: End date (e.g., '2024-01-01')
            timeframe: Timeframe to analyze (e.g., '1h')
            forward_days: Days to look forward for performance measurement
            sample_freq_days: Sampling frequency (7 = weekly samples)
            
        Returns:
            DataFrame with [features..., quality_score, performance_metrics...]
        """
        
        self.logger.info(f"📊 Collecting SR training data for {symbol} {exchange} {timeframe}")
        self.logger.info(f"   Period: {start_date} to {end_date}")
        self.logger.info(f"   Forward window: {forward_days} days")
        self.logger.info(f"   Sample frequency: every {sample_freq_days} days")
        
        # Load full historical data using artifact_manager
        full_data = self._load_historical_data(symbol, exchange, timeframe)
        
        if full_data is None or full_data.empty:
            raise ValueError(f"No data found for {symbol} {exchange} {timeframe}")
        
        self.logger.info(f"✅ Loaded {len(full_data)} historical bars")
        self.logger.info(f"   Date range: {full_data.index.min()} to {full_data.index.max()}")
        
        # Walk forward through time
        training_samples = []
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        sample_dates = pd.date_range(start_dt, end_dt, freq=f'{sample_freq_days}D')
        
        # Make sample_dates timezone-aware to match data
        if isinstance(full_data.index, pd.DatetimeIndex) and full_data.index.tz is not None:
            sample_dates = sample_dates.tz_localize('UTC')
            self.logger.info(f"   Sample dates made timezone-aware (UTC)")
        
        # Also convert start_dt and end_dt for filtering
        if isinstance(full_data.index, pd.DatetimeIndex) and full_data.index.tz is not None:
            start_dt = start_dt.tz_localize('UTC')
            end_dt = end_dt.tz_localize('UTC')
        
        self.logger.info(f"🔄 Processing {len(sample_dates)} sample dates...")
        
        for current_date in tqdm(sample_dates, desc="Collecting samples"):
            try:
                # Split into historical (for detection) and future (for labeling)
                historical_data = full_data[full_data.index < current_date]
                future_end = current_date + timedelta(days=forward_days)
                future_data = full_data[
                    (full_data.index >= current_date) & 
                    (full_data.index < future_end)
                ]
                
                # Need enough data
                if len(historical_data) < 200 or len(future_data) < 5:
                    continue
                
                # Detect SR levels on historical data
                levels = self._detect_sr_levels(historical_data, symbol, exchange, timeframe)
                
                if not levels:
                    continue
                
                # Label each level with future performance
                for level in levels:
                    try:
                        # Measure performance
                        performance = self._measure_level_performance(
                            level, future_data, historical_data
                        )
                        
                        # Extract ALL features
                        features = self._extract_all_features(level, historical_data)
                        
                        # Create training sample
                        sample = {
                            'date': current_date,
                            'symbol': symbol,
                            'exchange': exchange,
                            'timeframe': timeframe,
                            **features,  # All 30+ features
                            **performance  # Labels
                        }
                        
                        training_samples.append(sample)
                        
                    except Exception as e:
                        self.logger.debug(f"Failed to process level: {e}")
                        continue
            
            except Exception as e:
                self.logger.warning(f"Failed to process date {current_date}: {e}")
                continue
        
        # Convert to DataFrame
        training_df = pd.DataFrame(training_samples)
        
        if len(training_df) == 0:
            raise ValueError("No training samples collected!")
        
        self.logger.info(f"\n✅ Training data collection complete!")
        self.logger.info(f"   Total samples: {len(training_df)}")
        self.logger.info(f"   Date range: {training_df['date'].min()} to {training_df['date'].max()}")
        self.logger.info(f"   Features: {len([c for c in training_df.columns if c.startswith('feature_')])} columns")
        self.logger.info(f"   Quality score range: [{training_df['quality_score'].min():.3f}, {training_df['quality_score'].max():.3f}]")
        self.logger.info(f"   Quality score mean: {training_df['quality_score'].mean():.3f}")
        
        return training_df
    
    def _load_historical_data(self, symbol: str, exchange: str, timeframe: str) -> pd.DataFrame:
        """Load historical data from historical_data directory.
        
        Data is in: historical_data/EXCHANGE/ASSET/processed/
        Loads data that was already downloaded - does NOT trigger new downloads.
        """
        try:
            symbol_lower = symbol.lower()
            exchange_lower = exchange.lower()
            
            # PRIMARY: historical_data directory (partitioned parquet)
            historical_data_path = Path('historical_data') / exchange_lower / symbol_lower / 'processed' / f"{symbol_lower}_{timeframe}"
            
            if historical_data_path.exists():
                self.logger.info(f"✅ Found partitioned data at: {historical_data_path}")
                # Read entire partitioned dataset (pandas handles year/month partitions automatically)
                data = pd.read_parquet(historical_data_path)
                self.logger.info(f"   Loaded {len(data)} bars from partitioned dataset")
                
                # Fix timestamp conversion - check multiple possible timestamp columns
                timestamp_col = None
                for col_name in ['timestamp', 'open_time', 'close_time', 'date', 'datetime']:
                    if col_name in data.columns:
                        timestamp_col = col_name
                        break
                
                if timestamp_col:
                    # Convert to datetime - handle both Unix timestamps (ms) and datetime strings
                    if pd.api.types.is_numeric_dtype(data[timestamp_col]):
                        # Unix timestamp in milliseconds
                        data[timestamp_col] = pd.to_datetime(data[timestamp_col], unit='ms', utc=True)
                    else:
                        # String datetime
                        data[timestamp_col] = pd.to_datetime(data[timestamp_col], utc=True)
                    
                    data = data.set_index(timestamp_col).sort_index()
                    self.logger.info(f"   Set index from '{timestamp_col}' column")
                elif not isinstance(data.index, pd.DatetimeIndex):
                    # Try to convert index if it's numeric (Unix timestamp)
                    if pd.api.types.is_numeric_dtype(data.index):
                        data.index = pd.to_datetime(data.index, unit='ms', utc=True)
                        data = data.sort_index()
                        self.logger.info(f"   Converted numeric index to datetime")
                
                # Verify we have a valid datetime index
                if not isinstance(data.index, pd.DatetimeIndex):
                    self.logger.error(f"❌ Could not create datetime index. Available columns: {list(data.columns)}")
                    return pd.DataFrame()
                
                return data
            
            # Fallback: Try other possible paths
            fallback_paths = [
                # Data cache paths
                Path('data_cache') / exchange_lower / symbol_lower / f"klines_{exchange}_{symbol}_{timeframe}.parquet",
                Path('data_cache') / exchange_lower / symbol_lower / f"klines_{exchange_lower}_{symbol_lower}_{timeframe}.parquet",
                Path('data_cache') / exchange_lower / symbol_lower / f"{timeframe}.parquet",
                
                # Step01 path
                Path('data_cache') / 'step01_data_collection' / symbol_lower / exchange_lower / f"{timeframe}.parquet",
            ]
            
            for cache_path in fallback_paths:
                if cache_path.exists():
                    self.logger.info(f"✅ Found data at fallback path: {cache_path}")
                    data = pd.read_parquet(cache_path)
                    self.logger.info(f"   Loaded {len(data)} bars")
                    return data
            
            # No data found
            self.logger.error(f"❌ No data found for {symbol} {exchange} {timeframe}")
            self.logger.error(f"   Primary path checked: {historical_data_path}")
            self.logger.error(f"   Data should be in: historical_data/{exchange_lower}/{symbol_lower}/processed/{symbol_lower}_{timeframe}/")
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Failed to load historical data: {e}")
            return pd.DataFrame()
    
    def _detect_sr_levels(self, data: pd.DataFrame, symbol: str, 
                         exchange: str, timeframe: str) -> List:
        """Detect SR levels on historical data window."""
        try:
            from ..enhanced_sr_detection import EnhancedSRDetector
            
            # Create detector with multi-TF enabled
            detector = EnhancedSRDetector(config={
                'enable_real_multi_tf': True,
                'multi_tf_config': {
                    'alignment_tolerance': 0.005
                }
            })
            
            # Detect levels
            result = detector.detect_sr_levels(data[-500:])  # Last 500 bars
            
            if isinstance(result, dict) and 'levels' in result:
                return result['levels']
            elif isinstance(result, list):
                return result
            else:
                return []
                
        except Exception as e:
            self.logger.warning(f"SR detection failed: {e}")
            return []
    
    def _measure_level_performance(self, level, future_data: pd.DataFrame,
                                   historical_data: pd.DataFrame) -> Dict[str, float]:
        """Measure level performance in future data.
        
        This defines what "quality" means for SR levels.
        
        Returns:
            Dictionary with performance metrics including quality_score (target label)
        """
        tolerance = level.price * 0.005  # 0.5% tolerance
        level_type = level.type if hasattr(level, 'type') else 'unknown'
        level_price = level.price if hasattr(level, 'price') else 0
        
        # Check if price hit the level
        if level_type == 'support':
            hits = future_data[future_data['low'] <= level_price + tolerance]
        elif level_type == 'resistance':
            hits = future_data[future_data['high'] >= level_price - tolerance]
        else:
            # Unknown type
            return self._get_default_performance()
        
        if len(hits) == 0:
            # Level NOT tested - assign low quality
            return {
                'hit_rate': 0.0,
                'bounce_strength': 0.0,
                'hold_strength': 0.5,
                'trade_profit': 0.0,
                'quality_score': 0.2  # Low quality (untested)
            }
        
        # Level WAS hit - measure bounce
        first_hit_idx = hits.index[0]
        hit_bar = hits.loc[first_hit_idx]
        
        # 1. Bounce Strength
        if level_type == 'support':
            future_highs = future_data.loc[first_hit_idx:, 'high']
            max_bounce = future_highs.max() - hit_bar['low']
            bounce_pct = max_bounce / level_price
        else:  # resistance
            future_lows = future_data.loc[first_hit_idx:, 'low']
            max_bounce = hit_bar['high'] - future_lows.min()
            bounce_pct = max_bounce / level_price
        
        bounce_strength = min(bounce_pct / 0.02, 1.0)  # 2% bounce = 1.0
        
        # 2. Hold Strength (did level hold?)
        if level_type == 'support':
            breaks = future_data.loc[first_hit_idx:][
                future_data['close'] < level_price - tolerance
            ]
        else:
            breaks = future_data.loc[first_hit_idx:][
                future_data['close'] > level_price + tolerance
            ]
        
        if len(breaks) == 0:
            hold_strength = 1.0  # Held perfectly
        else:
            bars_until_break = len(future_data.loc[first_hit_idx:breaks.index[0]])
            hold_strength = min(bars_until_break / 20, 1.0)  # 20+ bars = 1.0
        
        # 3. Simulated Trade Profit
        trade_profit = self._simulate_trade(level_type, level_price, future_data, first_hit_idx)
        
        # 4. QUALITY SCORE (target label)
        quality_score = (
            bounce_strength * 0.35 +    # Strong bounces = good
            hold_strength * 0.35 +      # Levels that hold = good
            max(trade_profit, 0) * 0.30 # Profitable = good
        )
        
        return {
            'hit_rate': 1.0,
            'bounce_strength': float(bounce_strength),
            'hold_strength': float(hold_strength),
            'trade_profit': float(trade_profit),
            'quality_score': float(np.clip(quality_score, 0, 1))
        }
    
    def _simulate_trade(self, level_type: str, entry_price: float,
                       future_data: pd.DataFrame, hit_idx) -> float:
        """Simulate trade at level with 1% SL and 2% TP (2:1 R/R).
        
        Returns normalized profit (-1 to +1).
        """
        if level_type == 'support':
            stop_loss = entry_price * 0.99
            take_profit = entry_price * 1.02
            direction = 1
        else:  # resistance
            stop_loss = entry_price * 1.01
            take_profit = entry_price * 0.98
            direction = -1
        
        # Check next 10 bars
        future_bars = future_data.loc[hit_idx:].iloc[:10]
        
        for _, bar in future_bars.iterrows():
            if direction == 1:  # Long
                if bar['low'] <= stop_loss:
                    return -0.5  # Loss hit
                elif bar['high'] >= take_profit:
                    return 1.0  # TP hit (2:1 R/R)
            else:  # Short
                if bar['high'] >= stop_loss:
                    return -0.5
                elif bar['low'] <= take_profit:
                    return 1.0
        
        # No SL/TP hit - exit at close
        exit_price = future_bars.iloc[-1]['close']
        pnl_pct = (exit_price - entry_price) / entry_price * direction
        
        # Normalize: 2% = 1.0, -2% = -1.0
        return np.clip(pnl_pct * 50, -1, 1)
    
    def _extract_all_features(self, level, data: pd.DataFrame) -> Dict[str, float]:
        """Extract ALL 30+ features for ML training.
        
        Returns dict with 'feature_' prefix for each feature.
        """
        current_price = data['close'].iloc[-1]
        
        # Get level attributes safely
        def get_attr(name, default=0.0):
            if isinstance(level, dict):
                return level.get(name, default)
            return getattr(level, name, default)
        
        features = {
            # Basic SR features (always available)
            'feature_strength': get_attr('strength', 0.5),
            'feature_prominence': get_attr('prominence_score', 0.5),
            'feature_width': get_attr('width_score', 1.0),
            'feature_volume_confirmation': get_attr('volume_confirmation_score', 0.5),
            'feature_consistency': get_attr('consistency_score', 0.5),
            'feature_touch_count': get_attr('touch_count', 1),
            'feature_age_bars': get_attr('age_bars', 0),
            'feature_failure_count': get_attr('failure_count', 0),
            'feature_avg_bounce_ratio': get_attr('avg_bounce_ratio', 0),
            'feature_max_bounce_ratio': get_attr('max_bounce_ratio', 0),
            
            # Phase 1 features (dynamics & clustering)
            'feature_approach_velocity': get_attr('approach_velocity', 0),
            'feature_rejection_velocity': get_attr('rejection_velocity', 0),
            'feature_cluster_density': get_attr('cluster_density', 0),
            'feature_recency_weighted_strength': get_attr('recency_weighted_strength', 0),
            'feature_dwell_time': get_attr('dwell_time', 0),
            
            # Phase 3 features (multi-TF)
            'feature_multi_tf_score': get_attr('multi_tf_score', 0),
            'feature_multi_tf_confirmations': get_attr('confirmation_count', 0),
            
            # Interaction features
            'feature_strength_x_volume': get_attr('strength', 0.5) * get_attr('volume_confirmation_score', 0.5),
            'feature_prominence_x_width': get_attr('prominence_score', 0.5) * get_attr('width_score', 1.0) / 50.0,
            'feature_touch_x_consistency': get_attr('touch_count', 1) * get_attr('consistency_score', 0.5) / 10.0,
            'feature_cluster_x_multi_tf': get_attr('cluster_density', 0) * get_attr('multi_tf_score', 0),
            
            # Position features
            'feature_price_position': (get_attr('price', current_price) - data['close'].min()) / (data['close'].max() - data['close'].min() + 1e-8),
            'feature_distance_to_current_pct': abs(get_attr('price', current_price) - current_price) / current_price,
            'feature_is_support': 1.0 if get_attr('type', 'support') == 'support' else 0.0,
            
            # Market context features
            'feature_market_volatility': data['close'].pct_change().std(),
            'feature_market_volume_avg': data['volume'].mean() / 1e6,  # Normalize
            'feature_market_trend': (data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20],
            'feature_market_momentum': data['close'].pct_change(5).iloc[-1],
            
            # Statistical features
            'feature_price_zscore': (get_attr('price', current_price) - data['close'].mean()) / (data['close'].std() + 1e-8),
            'feature_price_percentile': (get_attr('price', current_price) < data['close']).sum() / len(data),
            
            # Time features
            'feature_hour_of_day': data.index[-1].hour if hasattr(data.index[-1], 'hour') else 0,
            'feature_day_of_week': data.index[-1].dayofweek if hasattr(data.index[-1], 'dayofweek') else 0,
        }
        
        # Add more features from metadata if available
        if hasattr(level, 'metadata') and level.metadata:
            regime = level.metadata.get('regime', {})
            features['feature_volatility_regime_score'] = regime.get('vol_score', 0.5)
            features['feature_trend_strength'] = regime.get('trend_strength', 0.0)
        
        return features
    
    def _get_default_performance(self) -> Dict[str, float]:
        """Default performance when measurement fails."""
        return {
            'hit_rate': 0.0,
            'bounce_strength': 0.0,
            'hold_strength': 0.5,
            'trade_profit': 0.0,
            'quality_score': 0.3
        }
    
    def save_training_data(self, training_df: pd.DataFrame, 
                          output_path: str = None) -> str:
        """Save collected training data.
        
        Args:
            training_df: Training data DataFrame
            output_path: Optional custom path
            
        Returns:
            Path to saved file
        """
        if output_path is None:
            output_path = 'data_cache/sr_ml_training/sr_quality_training_data.parquet'
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        training_df.to_parquet(output_file, index=False)
        
        # Save metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'samples': len(training_df),
            'date_range': {
                'start': str(training_df['date'].min()),
                'end': str(training_df['date'].max())
            },
            'symbols': training_df['symbol'].unique().tolist(),
            'timeframes': training_df['timeframe'].unique().tolist(),
            'feature_count': len([c for c in training_df.columns if c.startswith('feature_')]),
            'quality_score_stats': {
                'mean': float(training_df['quality_score'].mean()),
                'std': float(training_df['quality_score'].std()),
                'min': float(training_df['quality_score'].min()),
                'max': float(training_df['quality_score'].max())
            }
        }
        
        metadata_path = str(output_file).replace('.parquet', '_metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"✅ Training data saved to {output_file}")
        self.logger.info(f"✅ Metadata saved to {metadata_path}")
        
        return str(output_file)


# Convenience function
def collect_sr_training_data(symbol: str, exchange: str, timeframe: str,
                            start_date: str, end_date: str) -> pd.DataFrame:
    """Convenience function to collect training data."""
    collector = SRQualityDataCollector()
    return collector.collect_training_data(symbol, exchange, start_date, end_date, timeframe)

