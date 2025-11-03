"""
Simplified SR Quality Data Collector

PURE DATA-DRIVEN: No heuristic components!
Only calculates realized_pnl_pct (actual trading profit) aligned with 0.5-1% price goals.

Removed:
- bounce_strength (heuristic normalization)
- hold_strength (heuristic normalization)
- rejection_speed (heuristic calculation)
- volume_quality (heuristic normalization)
- quality_score (heuristic weighted sum)

Kept:
- realized_pnl_pct (ACTUAL trading profit/loss)
- feature_* columns (historical SR characteristics)
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List
from pathlib import Path
from tqdm import tqdm

from src.utils.data.real_data_loader import RealDataLoader

logger = logging.getLogger(__name__)


class SimplifiedSRDataCollector:
    """
    Simplified data collector for pure data-driven approach.
    
    Philosophy:
    - INPUTS: Historical SR features (feature_strength, touch_count, etc.)
    - OUTPUT: realized_pnl_pct (actual profit from trading with 0.5-1% goals)
    - NO intermediate heuristics!
    """
    
    def __init__(self, 
                 stop_loss_pct: float = 0.005,     # 0.5% stop loss
                 take_profit_pct: float = 0.01,    # 1.0% take profit (2:1 R/R)
                 max_hold_bars: int = 20):
        """
        Initialize simplified data collector.
        
        Args:
            stop_loss_pct: Stop loss percentage (default: 0.5%)
            take_profit_pct: Take profit percentage (default: 1.0%)
            max_hold_bars: Maximum bars to hold trade (default: 20)
        """
        self.data_loader = RealDataLoader()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Trading parameters aligned with 0.5-1% price goals
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_hold_bars = max_hold_bars
        
        self.logger.info(f"✅ Simplified SR Data Collector initialized")
        self.logger.info(f"   Trading setup: SL={stop_loss_pct*100:.1f}%, TP={take_profit_pct*100:.1f}% (R/R={take_profit_pct/stop_loss_pct:.1f}:1)")
        self.logger.info(f"   Max hold: {max_hold_bars} bars")
        
        # Initialize SR detector (reused for speed)
        from ..enhanced_sr_detection import EnhancedSRDetector
        
        self.sr_detector = EnhancedSRDetector(config={
            'disable_dbscan_clustering': True,
            'disable_backtesting_validation': True,
            'max_levels_per_method': 10,
            'fractal_periods': [5],
            'pivot_periods': [5],
            'use_optimized_fractals': True,
            'enable_fractal_caching': True,
            'enable_pivot_caching': True,
        })
        
    async def collect_training_data(self, 
                                    symbol: str, 
                                    exchange: str,
                                    start_date: str, 
                                    end_date: str,
                                    timeframe: str = '1h',
                                    forward_days: int = 10,
                                    sample_freq_days: int = 7) -> pd.DataFrame:
        """
        Collect SR training data with ONLY realized_pnl_pct as target.
        
        Process:
        1. Walk through historical dates
        2. Detect SR levels on historical data
        3. Measure forward performance: realized_pnl_pct ONLY
        4. Extract historical features
        5. Create training samples
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            start_date: Start date for data collection
            end_date: End date
            timeframe: Timeframe to analyze
            forward_days: Days to look forward for performance
            sample_freq_days: Sampling frequency
            
        Returns:
            DataFrame with [feature_*, realized_pnl_pct]
        """
        
        self.logger.info(f"📊 Collecting SIMPLIFIED training data for {symbol} {exchange} {timeframe}")
        self.logger.info(f"   Period: {start_date} to {end_date}")
        self.logger.info(f"   Forward window: {forward_days} days")
        self.logger.info(f"   Sample frequency: every {sample_freq_days} days")
        self.logger.info(f"   Target: realized_pnl_pct ONLY (no heuristics)")
        
        # Load full historical data
        full_data = await self._load_historical_data(symbol, exchange, timeframe, start_date, end_date)
        
        if full_data is None or full_data.empty:
            raise ValueError(f"No data found for {symbol} {exchange} {timeframe}")
        
        self.logger.info(f"✅ Loaded {len(full_data)} historical bars")
        
        # Walk forward through time
        training_samples = []
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        sample_dates = pd.date_range(start_dt, end_dt, freq=f'{sample_freq_days}D')
        
        # Make timezone-aware if needed
        if isinstance(full_data.index, pd.DatetimeIndex) and full_data.index.tz is not None:
            sample_dates = sample_dates.tz_localize('UTC')
            start_dt = start_dt.tz_localize('UTC')
            end_dt = end_dt.tz_localize('UTC')
        
        self.logger.info(f"🔄 Processing {len(sample_dates)} sample dates...")
        
        for current_date in tqdm(sample_dates, desc="Collecting samples"):
            try:
                # Split data
                historical_data = full_data[full_data.index < current_date]
                future_end = current_date + timedelta(days=forward_days)
                future_data = full_data[
                    (full_data.index >= current_date) & 
                    (full_data.index < future_end)
                ]
                
                # Need enough data
                if len(historical_data) < 200 or len(future_data) < 5:
                    continue
                
                # Detect SR levels
                levels = self._detect_sr_levels(historical_data, symbol, exchange, timeframe)
                
                if not levels:
                    continue
                
                # Process each level
                for level in levels:
                    try:
                        # Extract ONLY historical features (no future peeking!)
                        features = self._extract_historical_features(level, historical_data)
                        
                        # Calculate ONLY the target: realized_pnl_pct
                        target = self._calculate_realized_pnl(level, future_data)
                        
                        # Create training sample
                        sample = {
                            'date': current_date,
                            'symbol': symbol,
                            'exchange': exchange,
                            'timeframe': timeframe,
                            **features,                      # Historical features
                            'realized_pnl_pct': target,      # ✅ ONLY target!
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
        
        # Filter out untested levels (zero P&L means never hit)
        initial_count = len(training_df)
        training_df = training_df[training_df['realized_pnl_pct'] != 0.0].copy()
        
        self.logger.info(f"\n✅ Training data collection complete!")
        self.logger.info(f"   Total samples: {len(training_df)} (filtered {initial_count - len(training_df)} untested)")
        self.logger.info(f"   Features: {len([c for c in training_df.columns if c.startswith('feature_')])} columns")
        self.logger.info(f"   Target: realized_pnl_pct")
        self.logger.info(f"   P&L range: [{training_df['realized_pnl_pct'].min()*100:.2f}%, {training_df['realized_pnl_pct'].max()*100:.2f}%]")
        self.logger.info(f"   Mean P&L: {training_df['realized_pnl_pct'].mean()*100:.2f}%")
        self.logger.info(f"   Win rate: {(training_df['realized_pnl_pct'] > 0).sum() / len(training_df) * 100:.1f}%")
        
        return training_df
    
    def _calculate_realized_pnl(self, level, future_data: pd.DataFrame) -> float:
        """
        Calculate ONLY realized_pnl_pct - no heuristic components!
        
        This is the ENTIRE performance measurement - just actual trading profit.
        Aligned with 0.5-1% price goals.
        
        Args:
            level: SR level object
            future_data: Future price data
            
        Returns:
            realized_pnl_pct: Actual P&L percentage (-stop_loss_pct to +take_profit_pct)
        """
        
        # Get level info
        level_price = getattr(level, 'price', None) if not isinstance(level, dict) else level.get('price')
        level_type = getattr(level, 'type', None) if not isinstance(level, dict) else level.get('type')
        
        if level_price is None or level_type not in ['support', 'resistance']:
            return 0.0
        
        tolerance = level_price * 0.005  # 0.5% tolerance for hit detection
        
        # Check if level was hit in forward window
        if level_type == 'support':
            hits = future_data[future_data['low'] <= level_price + tolerance]
        else:  # resistance
            hits = future_data[future_data['high'] >= level_price - tolerance]
        
        if len(hits) == 0:
            return 0.0  # Level never tested
        
        # Simulate trade from first hit
        first_hit_idx = hits.index[0]
        
        # Define trade parameters (aligned with 0.5-1% goals)
        if level_type == 'support':
            entry_price = level_price
            stop_loss = entry_price * (1 - self.stop_loss_pct)   # 0.5% below
            take_profit = entry_price * (1 + self.take_profit_pct)  # 1.0% above
            direction = 1  # Long
        else:  # resistance
            entry_price = level_price
            stop_loss = entry_price * (1 + self.stop_loss_pct)   # 0.5% above
            take_profit = entry_price * (1 - self.take_profit_pct)  # 1.0% below
            direction = -1  # Short
        
        # Check what happens in next bars
        future_bars = future_data.loc[first_hit_idx:].iloc[:self.max_hold_bars]
        
        for bar_idx, (_, bar) in enumerate(future_bars.iterrows()):
            if direction == 1:  # Long from support
                if bar['low'] <= stop_loss:
                    return -self.stop_loss_pct  # Lost 0.5%
                if bar['high'] >= take_profit:
                    return self.take_profit_pct  # Made 1.0%
            else:  # Short from resistance
                if bar['high'] >= stop_loss:
                    return -self.stop_loss_pct  # Lost 0.5%
                if bar['low'] <= take_profit:
                    return self.take_profit_pct  # Made 1.0%
        
        # Neither SL nor TP hit - exit at market close
        exit_price = future_bars.iloc[-1]['close']
        pnl_pct = (exit_price - entry_price) / entry_price * direction
        
        return float(pnl_pct)
    
    def _extract_historical_features(self, level, data: pd.DataFrame) -> Dict[str, float]:
        """
        Extract ONLY historical features (available at prediction time).
        
        No future-peeking! No heuristic calculations!
        Just the SR level's historical characteristics.
        
        Args:
            level: SR level object
            data: Historical price data
            
        Returns:
            Dictionary of feature values
        """
        
        current_price = data['close'].iloc[-1]
        
        # Get level attributes safely
        def get_attr(name, default=0.0):
            if isinstance(level, dict):
                return level.get(name, default)
            return getattr(level, name, default)
        
        # Core SR characteristics (from historical data only)
        strength = get_attr('strength', 0.5)
        touch_count = get_attr('touch_count', 1)
        age_bars = get_attr('age_bars', 0)
        consistency = get_attr('consistency_score', 0.5)
        avg_bounce = get_attr('avg_bounce_ratio', 0)
        max_bounce = get_attr('max_bounce_ratio', 0)
        
        # Position relative to current price
        level_price = get_attr('price', current_price)
        distance_pct = abs(level_price - current_price) / current_price
        price_zscore = (level_price - data['close'].mean()) / (data['close'].std() + 1e-8)
        
        # Market context (from historical data only)
        volatility = data['close'].pct_change().std()
        trend = (data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20]
        
        # Time features
        hour_normalized = 0.0
        if len(data) > 0 and hasattr(data.index[-1], 'hour'):
            hour_normalized = float(data.index[-1].hour) / 24.0
        
        # SIMPLIFIED FEATURE SET (19 features)
        features = {
            # Core SR metrics (6 features)
            'feature_strength': float(strength),
            'feature_touch_count': int(touch_count),
            'feature_age_bars': int(age_bars),
            'feature_consistency': float(consistency),
            'feature_avg_bounce_ratio': float(avg_bounce),
            'feature_max_bounce_ratio': float(max_bounce),
            
            # Position features (3 features)
            'feature_distance_to_current_pct': float(distance_pct),
            'feature_price_zscore': float(price_zscore),
            'feature_is_support': 1.0 if get_attr('type', 'support') == 'support' else 0.0,
            
            # Market context (4 features)
            'feature_market_volatility': float(volatility),
            'feature_market_trend': float(trend),
            'feature_is_high_volatility': 1.0 if volatility > 0.03 else 0.0,
            'feature_is_uptrend': 1.0 if trend > 0.02 else 0.0,
            
            # Quality indicators (3 features)
            'feature_volume_confirmation': float(get_attr('volume_confirmation_score', 0.5)),
            'feature_bounce_consistency': float(get_attr('bounce_consistency', 0.0)),
            'feature_recency_weighted_strength': float(strength * np.exp(-age_bars / 50)),
            
            # Time features (3 features)
            'feature_hour_of_day': float(hour_normalized),
            'feature_quality_tier': float(min(strength * 2.0, 1.0)),
            'feature_touch_quality_score': float(
                (avg_bounce * 0.5) + (get_attr('avg_touch_volume_ratio', 0) * 0.3) + (consistency * 0.2)
            ),
        }
        
        return features
    
    async def _load_historical_data(self, symbol: str, exchange: str, timeframe: str,
                                    start_date: str, end_date: str) -> pd.DataFrame:
        """Load historical market data."""
        try:
            data = await self.data_loader.load_market_data(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                force_download=False
            )
            
            if data is not None and len(data) > 0:
                self.logger.info(f"✅ Loaded {len(data)} bars from data loader")
                return data
            else:
                self.logger.error(f"❌ No data found")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def _detect_sr_levels(self, data: pd.DataFrame, symbol: str,
                         exchange: str, timeframe: str) -> List:
        """Detect SR levels on historical data."""
        try:
            result = self.sr_detector.detect_sr_levels(data[-500:])
            
            if isinstance(result, dict) and 'levels' in result:
                return result['levels']
            elif isinstance(result, list):
                return result
            else:
                return []
                
        except Exception as e:
            self.logger.warning(f"SR detection failed: {e}")
            return []
    
    def save_training_data(self, training_df: pd.DataFrame,
                          output_path: str = None) -> str:
        """
        Save training data with metadata.
        
        Args:
            training_df: Training DataFrame
            output_path: Optional custom path
            
        Returns:
            Path to saved file
        """
        
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f'data_cache/sr_ml_training/sr_quality_SIMPLIFIED_{timestamp}.parquet'
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        training_df.to_parquet(output_file, index=False)
        
        # Save metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'approach': 'simplified_data_driven',
            'samples': len(training_df),
            'date_range': {
                'start': str(training_df['date'].min()),
                'end': str(training_df['date'].max())
            },
            'symbols': training_df['symbol'].unique().tolist(),
            'timeframes': training_df['timeframe'].unique().tolist(),
            'feature_count': len([c for c in training_df.columns if c.startswith('feature_')]),
            'target': 'realized_pnl_pct',
            'trading_params': {
                'stop_loss_pct': self.stop_loss_pct,
                'take_profit_pct': self.take_profit_pct,
                'max_hold_bars': self.max_hold_bars,
                'risk_reward_ratio': self.take_profit_pct / self.stop_loss_pct
            },
            'pnl_stats': {
                'mean': float(training_df['realized_pnl_pct'].mean()),
                'std': float(training_df['realized_pnl_pct'].std()),
                'min': float(training_df['realized_pnl_pct'].min()),
                'max': float(training_df['realized_pnl_pct'].max()),
                'win_rate': float((training_df['realized_pnl_pct'] > 0).sum() / len(training_df))
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
async def collect_simplified_training_data(symbol: str, exchange: str, timeframe: str,
                                          start_date: str, end_date: str,
                                          stop_loss_pct: float = 0.005,
                                          take_profit_pct: float = 0.01) -> pd.DataFrame:
    """
    Convenience function to collect simplified training data.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        start_date: Start date
        end_date: End date
        stop_loss_pct: Stop loss % (default: 0.5%)
        take_profit_pct: Take profit % (default: 1.0%)
        
    Returns:
        Training DataFrame with [feature_*, realized_pnl_pct]
    """
    
    collector = SimplifiedSRDataCollector(
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct
    )
    
    return await collector.collect_training_data(
        symbol, exchange, start_date, end_date, timeframe
    )

