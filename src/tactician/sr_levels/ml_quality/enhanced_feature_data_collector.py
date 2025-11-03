"""
Enhanced SR Quality Data Collector with FeatureBank Integration

Uses powerful features from feature_generation/:
1. SR-specific features (support/resistance generators)
2. Market regime features (volatility, trend states)
3. Price action features (momentum, candlestick patterns)
4. Multi-timeframe features (1D SR tested on 1h/4h)
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List
from pathlib import Path
from tqdm import tqdm

from src.utils.data.real_data_loader import RealDataLoader
from src.feature_generation import FeatureBank, get_feature_bank
from src.feature_generation.core.feature_generator import FeatureCategory

logger = logging.getLogger(__name__)


class EnhancedFeatureSRDataCollector:
    """
    Enhanced SR data collector using FeatureBank for powerful features.
    
    Key improvements:
    - Uses FeatureBank for 100+ technical features
    - Adds SR-specific features
    - Adds market regime features
    - Adds multi-timeframe SR features
    - NO heuristic components!
    """
    
    def __init__(self,
                 stop_loss_pct: float = 0.01,
                 take_profit_pct: float = 0.01,
                 max_hold_bars: int = 20):
        """
        Initialize enhanced data collector with FeatureBank.
        
        Args:
            stop_loss_pct: Stop loss % (default: 1.0%)
            take_profit_pct: Take profit % (default: 1.0%)
            max_hold_bars: Max bars to hold trade
        """
        self.data_loader = RealDataLoader()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_hold_bars = max_hold_bars
        
        # Initialize FeatureBank for powerful features
        self.logger.info("🚀 Initializing FeatureBank for enhanced features...")
        self.feature_bank = get_feature_bank()
        
        # Initialize SR detector
        from ..enhanced_sr_detection import EnhancedSRDetector
        self.sr_detector = EnhancedSRDetector(config={
            'disable_dbscan_clustering': True,
            'disable_backtesting_validation': True,
            'max_levels_per_method': 15,
            'fractal_periods': [3, 5],
            'pivot_periods': [5],
            'use_optimized_fractals': True,
        })
        
        self.logger.info(f"✅ Enhanced collector initialized")
        self.logger.info(f"   SL={stop_loss_pct*100:.1f}%, TP={take_profit_pct*100:.1f}%, R/R={take_profit_pct/stop_loss_pct:.1f}:1")
        
    async def collect_training_data(self,
                                    symbol: str,
                                    exchange: str,
                                    start_date: str,
                                    end_date: str,
                                    timeframe: str = '1h',
                                    forward_days: int = 10,
                                    sample_freq_days: int = 1) -> pd.DataFrame:
        """
        Collect training data with ENHANCED FEATURES from FeatureBank.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            start_date: Start date
            end_date: End date
            timeframe: Timeframe to analyze
            forward_days: Days to look forward
            sample_freq_days: Sampling frequency
            
        Returns:
            DataFrame with [enhanced_feature_*, realized_pnl_pct]
        """
        
        self.logger.info(f"📊 Collecting ENHANCED training data for {symbol} {exchange} {timeframe}")
        self.logger.info(f"   Period: {start_date} to {end_date}")
        self.logger.info(f"   Features: Using FeatureBank + SR-specific + Multi-timeframe")
        
        # Load historical data
        full_data = await self._load_historical_data(symbol, exchange, timeframe, start_date, end_date)
        
        if full_data is None or full_data.empty:
            raise ValueError(f"No data found for {symbol} {exchange} {timeframe}")
        
        self.logger.info(f"✅ Loaded {len(full_data)} bars")
        
        # Load multi-timeframe data (1D for SR detection)
        full_data_1d = await self._load_historical_data(symbol, exchange, '1d', start_date, end_date)
        
        if full_data_1d is not None and len(full_data_1d) > 0:
            self.logger.info(f"✅ Loaded {len(full_data_1d)} daily bars for multi-timeframe features")
            has_multi_tf = True
        else:
            self.logger.warning("⚠️  Could not load 1D data, skipping multi-timeframe features")
            has_multi_tf = False
        
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
                
                # Detect SR levels (on current timeframe)
                levels = self._detect_sr_levels(historical_data[-500:], symbol, exchange, timeframe)
                
                if not levels:
                    continue
                
                # Process each level
                for level in levels:
                    try:
                        # Extract ENHANCED features using FeatureBank
                        features = self._extract_enhanced_features(
                            level, 
                            historical_data,
                            full_data_1d if has_multi_tf else None,
                            current_date
                        )
                        
                        # Calculate target
                        target = self._calculate_realized_pnl(level, future_data)
                        
                        # Create sample
                        sample = {
                            'date': current_date,
                            'symbol': symbol,
                            'exchange': exchange,
                            'timeframe': timeframe,
                            **features,
                            'realized_pnl_pct': target,
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
        
        # Filter untested levels
        initial_count = len(training_df)
        training_df = training_df[training_df['realized_pnl_pct'] != 0.0].copy()
        
        self.logger.info(f"\n✅ Enhanced training data collection complete!")
        self.logger.info(f"   Total samples: {len(training_df)} (filtered {initial_count - len(training_df)} untested)")
        self.logger.info(f"   Features: {len([c for c in training_df.columns if c.startswith('feature_')])} columns")
        self.logger.info(f"   Categories: SR-specific, Market regime, Price action, Multi-timeframe")
        
        return training_df
    
    def _extract_enhanced_features(self, level, historical_data: pd.DataFrame,
                                   daily_data: pd.DataFrame, current_date) -> Dict[str, float]:
        """
        Extract ENHANCED features using FeatureBank + SR-specific + Multi-timeframe.
        
        Feature categories:
        1. Basic SR features (19 from before)
        2. SR-specific features from FeatureBank (support_resistance category)
        3. Market regime features (volatility, trend regimes)
        4. Price action features (momentum, candlestick patterns)
        5. Multi-timeframe features (daily SR levels)
        
        Args:
            level: SR level object
            historical_data: Historical price data (current timeframe)
            daily_data: Daily price data (for multi-TF features)
            current_date: Current date
            
        Returns:
            Dictionary of enhanced features
        """
        
        features = {}
        
        # ====================================================================
        # 1. BASIC SR FEATURES (from original implementation)
        # ====================================================================
        
        def get_attr(name, default=0.0):
            if isinstance(level, dict):
                return level.get(name, default)
            return getattr(level, name, default)
        
        current_price = historical_data['close'].iloc[-1]
        level_price = get_attr('price', current_price)
        level_type = get_attr('type', 'support')
        
        basic_features = {
            'feature_strength': float(get_attr('strength', 0.5)),
            'feature_touch_count': int(get_attr('touch_count', 1)),
            'feature_age_bars': int(get_attr('age_bars', 0)),
            'feature_consistency': float(get_attr('consistency_score', 0.5)),
            'feature_distance_to_current_pct': float(abs(level_price - current_price) / current_price),
            'feature_is_support': 1.0 if level_type == 'support' else 0.0,
        }
        
        features.update(basic_features)
        
        # ====================================================================
        # 2. FEATUREBANK FEATURES - Market Regime
        # ====================================================================
        
        try:
            # Generate regime features using FeatureBank
            regime_features_df = self.feature_bank.generate_features(
                data=historical_data[-100:].copy(),  # Last 100 bars
                categories=[
                    FeatureCategory.VOLATILITY,  # Volatility regime
                    FeatureCategory.TREND,       # Trend regime
                ],
                lookback_optimization=True
            )
            
            # Extract last row (current state)
            if regime_features_df is not None and len(regime_features_df) > 0:
                regime_features = regime_features_df.iloc[-1].to_dict()
                
                # Add top regime features
                for feat_name, feat_value in regime_features.items():
                    if pd.notna(feat_value):
                        features[f'feature_regime_{feat_name}'] = float(feat_value)
                
                self.logger.debug(f"Added {len(regime_features)} regime features")
        
        except Exception as e:
            self.logger.debug(f"Regime features failed: {e}")
        
        # ====================================================================
        # 3. FEATUREBANK FEATURES - Price Action
        # ====================================================================
        
        try:
            # Generate price action features
            price_action_df = self.feature_bank.generate_features(
                data=historical_data[-100:].copy(),
                categories=[
                    FeatureCategory.MOMENTUM,           # RSI, MACD, etc.
                    FeatureCategory.CANDLESTICK_PATTERN, # Candle patterns
                ],
                lookback_optimization=True
            )
            
            if price_action_df is not None and len(price_action_df) > 0:
                price_features = price_action_df.iloc[-1].to_dict()
                
                for feat_name, feat_value in price_features.items():
                    if pd.notna(feat_value):
                        features[f'feature_price_action_{feat_name}'] = float(feat_value)
                
                self.logger.debug(f"Added {len(price_features)} price action features")
        
        except Exception as e:
            self.logger.debug(f"Price action features failed: {e}")
        
        # ====================================================================
        # 4. SR-SPECIFIC FEATURES from FeatureBank
        # ====================================================================
        
        try:
            # Generate SR-specific features
            sr_features_df = self.feature_bank.generate_features(
                data=historical_data[-100:].copy(),
                categories=[
                    FeatureCategory.SUPPORT_RESISTANCE,
                ],
                lookback_optimization=True
            )
            
            if sr_features_df is not None and len(sr_features_df) > 0:
                sr_features = sr_features_df.iloc[-1].to_dict()
                
                for feat_name, feat_value in sr_features.items():
                    if pd.notna(feat_value):
                        features[f'feature_sr_{feat_name}'] = float(feat_value)
                
                self.logger.debug(f"Added {len(sr_features)} SR-specific features")
        
        except Exception as e:
            self.logger.debug(f"SR-specific features failed: {e}")
        
        # ====================================================================
        # 5. MULTI-TIMEFRAME FEATURES (1D SR on 1h/4h timeframe)
        # ====================================================================
        
        if daily_data is not None:
            try:
                # Detect SR levels on DAILY timeframe
                daily_hist = daily_data[daily_data.index < current_date]
                
                if len(daily_hist) >= 50:
                    daily_levels = self._detect_sr_levels(daily_hist[-100:], '', '', '1d')
                    
                    if daily_levels:
                        # Find nearest daily SR level
                        daily_prices = [getattr(lvl, 'price', 0) if not isinstance(lvl, dict) else lvl.get('price', 0) 
                                      for lvl in daily_levels]
                        daily_prices = [p for p in daily_prices if p > 0]
                        
                        if daily_prices:
                            distances = [abs(p - current_price) / current_price for p in daily_prices]
                            nearest_idx = np.argmin(distances)
                            nearest_daily_level = daily_levels[nearest_idx]
                            
                            # Multi-timeframe features
                            features['feature_mtf_aligned_with_1d'] = 1.0 if distances[nearest_idx] < 0.02 else 0.0
                            features['feature_mtf_1d_distance_pct'] = float(distances[nearest_idx])
                            features['feature_mtf_1d_strength'] = float(get_attr('strength', 0.5)) if hasattr(nearest_daily_level, '__dict__') else 0.5
                            
                            self.logger.debug(f"Added 3 multi-timeframe features")
            
            except Exception as e:
                self.logger.debug(f"Multi-timeframe features failed: {e}")
        
        # ====================================================================
        # 6. RECENT SR PERFORMANCE (actual predictive features!)
        # ====================================================================
        
        try:
            # How did THIS specific level perform recently?
            # This is highly predictive!
            
            # Get recent bars near this level
            tolerance = level_price * 0.01  # 1% tolerance
            recent_data = historical_data[-50:]  # Last 50 bars
            
            if level_type == 'support':
                tests = recent_data[recent_data['low'] <= level_price + tolerance]
            else:
                tests = recent_data[recent_data['high'] >= level_price - tolerance]
            
            # Recent performance features
            features['feature_recent_tests_count'] = len(tests)
            features['feature_days_since_last_test'] = (current_date - tests.index[-1]).days if len(tests) > 0 else 999
            
            # Did it bounce last time?
            if len(tests) > 0:
                last_test_idx = tests.index[-1]
                bars_after_test = historical_data.loc[last_test_idx:].iloc[:5]
                
                if len(bars_after_test) >= 2:
                    if level_type == 'support':
                        bounced = bars_after_test['close'].iloc[-1] > level_price
                    else:
                        bounced = bars_after_test['close'].iloc[-1] < level_price
                    
                    features['feature_bounced_last_test'] = 1.0 if bounced else 0.0
                else:
                    features['feature_bounced_last_test'] = 0.5
            else:
                features['feature_bounced_last_test'] = 0.5
            
            self.logger.debug(f"Added 3 recent performance features")
        
        except Exception as e:
            self.logger.debug(f"Recent performance features failed: {e}")
            features['feature_recent_tests_count'] = 0
            features['feature_days_since_last_test'] = 999
            features['feature_bounced_last_test'] = 0.5
        
        return features
    
    def _calculate_realized_pnl(self, level, future_data: pd.DataFrame) -> float:
        """Calculate realized P&L (same as simplified version)."""
        
        level_price = getattr(level, 'price', None) if not isinstance(level, dict) else level.get('price')
        level_type = getattr(level, 'type', None) if not isinstance(level, dict) else level.get('type')
        
        if level_price is None or level_type not in ['support', 'resistance']:
            return 0.0
        
        tolerance = level_price * 0.005
        
        # Check if hit
        if level_type == 'support':
            hits = future_data[future_data['low'] <= level_price + tolerance]
        else:
            hits = future_data[future_data['high'] >= level_price - tolerance]
        
        if len(hits) == 0:
            return 0.0
        
        # Simulate trade
        first_hit_idx = hits.index[0]
        
        if level_type == 'support':
            entry_price = level_price
            stop_loss = entry_price * (1 - self.stop_loss_pct)
            take_profit = entry_price * (1 + self.take_profit_pct)
            direction = 1
        else:
            entry_price = level_price
            stop_loss = entry_price * (1 + self.stop_loss_pct)
            take_profit = entry_price * (1 - self.take_profit_pct)
            direction = -1
        
        future_bars = future_data.loc[first_hit_idx:].iloc[:self.max_hold_bars]
        
        for _, bar in future_bars.iterrows():
            if direction == 1:
                if bar['low'] <= stop_loss:
                    return -self.stop_loss_pct
                if bar['high'] >= take_profit:
                    return self.take_profit_pct
            else:
                if bar['high'] >= stop_loss:
                    return -self.stop_loss_pct
                if bar['low'] <= take_profit:
                    return self.take_profit_pct
        
        # Exit at market
        exit_price = future_bars.iloc[-1]['close']
        pnl_pct = (exit_price - entry_price) / entry_price * direction
        
        return float(pnl_pct)
    
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
            
            return data if data is not None else pd.DataFrame()
                
        except Exception as e:
            self.logger.debug(f"Failed to load data: {e}")
            return pd.DataFrame()
    
    def _detect_sr_levels(self, data: pd.DataFrame, symbol: str,
                         exchange: str, timeframe: str) -> List:
        """Detect SR levels."""
        try:
            result = self.sr_detector.detect_sr_levels(data)
            
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
        """Save training data with metadata."""
        
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f'data_cache/sr_ml_training/sr_quality_ENHANCED_{timestamp}.parquet'
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        training_df.to_parquet(output_file, index=False)
        
        # Save metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'approach': 'enhanced_featurebank_integration',
            'samples': len(training_df),
            'feature_count': len([c for c in training_df.columns if c.startswith('feature_')]),
            'feature_categories': [
                'basic_sr',
                'featurebank_regime',
                'featurebank_price_action',
                'featurebank_sr_specific',
                'multi_timeframe',
                'recent_performance'
            ],
            'trading_params': {
                'stop_loss_pct': self.stop_loss_pct,
                'take_profit_pct': self.take_profit_pct,
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

