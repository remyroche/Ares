"""
Feature Enhancement Module for Regime Clustering.

This module implements advanced features to improve regime separation and economic relevance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import warnings
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error
)


class AdvancedFeatureGenerator:
    """Generates advanced features for regime clustering."""
    
    def __init__(self, lookback_periods: int = 20):
        """
        Initialize feature generator.
        
        Args:
            lookback_periods: Number of periods for rolling calculations
        """
        self.lookback_periods = lookback_periods
        self.scaler = StandardScaler()
        
    def generate_enhanced_features(
        self, 
        market_data: pd.DataFrame,
        existing_features: Optional[np.ndarray] = None,
        existing_feature_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate enhanced features for regime clustering.
        
        Args:
            market_data: Market data with OHLCV columns
            existing_features: Optional existing feature matrix
            existing_feature_names: Optional existing feature names
            
        Returns:
            Tuple of (enhanced_features, feature_names)
        """
        try:
            tprint_info("Generating enhanced features for regime clustering...")
            
            enhanced_features = []
            feature_names = []
            
            # Add existing features if provided
            if existing_features is not None and existing_feature_names is not None:
                enhanced_features.append(existing_features)
                feature_names.extend(existing_feature_names)
            
            # Generate volatility regime features
            vol_features, vol_names = self._generate_volatility_regime_features(market_data)
            if vol_features is not None:
                enhanced_features.append(vol_features)
                feature_names.extend(vol_names)
            
            # Generate trend regime features
            trend_features, trend_names = self._generate_trend_regime_features(market_data)
            if trend_features is not None:
                enhanced_features.append(trend_features)
                feature_names.extend(trend_names)
            
            # Generate momentum regime features
            momentum_features, momentum_names = self._generate_momentum_regime_features(market_data)
            if momentum_features is not None:
                enhanced_features.append(momentum_features)
                feature_names.extend(momentum_names)
            
            # Generate volume regime features
            volume_features, volume_names = self._generate_volume_regime_features(market_data)
            if volume_features is not None:
                enhanced_features.append(volume_features)
                feature_names.extend(volume_names)
            
            # Generate regime persistence features
            persistence_features, persistence_names = self._generate_regime_persistence_features(market_data)
            if persistence_features is not None:
                enhanced_features.append(persistence_features)
                feature_names.extend(persistence_names)
            
            # Generate economic regime features
            economic_features, economic_names = self._generate_economic_regime_features(market_data)
            if economic_features is not None:
                enhanced_features.append(economic_features)
                feature_names.extend(economic_names)
            
            # Generate market microstructure features
            microstructure_features, microstructure_names = self._generate_microstructure_features(market_data)
            if microstructure_features is not None:
                enhanced_features.append(microstructure_features)
                feature_names.extend(microstructure_names)
            
            # Combine all features
            if enhanced_features:
                combined_features = np.hstack(enhanced_features)
            else:
                combined_features = np.array([]).reshape(len(market_data), 0)
            
            tprint_success(f"Generated {len(feature_names)} enhanced features")
            return combined_features, feature_names
            
        except Exception as e:
            tprint_error(f"Enhanced feature generation failed: {e}")
            return np.array([]).reshape(len(market_data), 0), []
    
    def _generate_volatility_regime_features(self, market_data: pd.DataFrame) -> Tuple[Optional[np.ndarray], List[str]]:
        """Generate volatility regime features."""
        try:
            if 'close' not in market_data.columns:
                return None, []
            
            features = []
            feature_names = []
            
            returns = market_data['close'].pct_change().dropna()
            if len(returns) < self.lookback_periods:
                return None, []
            
            # Multi-timeframe volatility
            for period in [5, 10, 20, 40, 60, 120]:
                vol = returns.rolling(period).std() * np.sqrt(252)
                features.append(vol.fillna(vol.mean()).values)
                feature_names.append(f'volatility_{period}')
            
            # ATR-based volatility
            if all(col in market_data.columns for col in ['high', 'low', 'close']):
                high = market_data['high']
                low = market_data['low']
                close = market_data['close']
                
                for period in [5, 10, 20, 40]:
                    tr1 = high - low
                    tr2 = np.abs(high - close.shift(1))
                    tr3 = np.abs(low - close.shift(1))
                    tr = np.maximum(tr1, np.maximum(tr2, tr3))
                    atr = tr.rolling(period).mean()
                    features.append(atr.fillna(atr.mean()).values)
                    feature_names.append(f'atr_{period}')
            
            # Volatility regime classification
            vol_20 = returns.rolling(20).std() * np.sqrt(252)
            vol_mean_60 = vol_20.rolling(60).mean()
            vol_std_60 = vol_20.rolling(60).std()
            
            # Z-score
            vol_zscore = (vol_20 - vol_mean_60) / (vol_std_60 + 1e-8)
            features.append(vol_zscore.fillna(0).values)
            feature_names.append('vol_regime_zscore')
            
            # Percentile rank
            vol_percentile = vol_20.rolling(252).rank(pct=True)
            features.append(vol_percentile.fillna(0.5).values)
            feature_names.append('vol_regime_percentile')
            
            # Volatility clustering
            vol_cluster_high = (vol_20 > vol_mean_60 + vol_std_60).astype(int)
            vol_cluster_low = (vol_20 < vol_mean_60 - vol_std_60).astype(int)
            vol_cluster_normal = ((vol_20 >= vol_mean_60 - vol_std_60) & 
                                (vol_20 <= vol_mean_60 + vol_std_60)).astype(int)
            
            features.append(vol_cluster_high.fillna(0).values)
            feature_names.append('vol_cluster_high')
            features.append(vol_cluster_low.fillna(0).values)
            feature_names.append('vol_cluster_low')
            features.append(vol_cluster_normal.fillna(1).values)
            feature_names.append('vol_cluster_normal')
            
            # Volatility momentum
            vol_momentum = vol_20.pct_change(5)
            features.append(vol_momentum.fillna(0).values)
            feature_names.append('vol_momentum_5')
            
            # Volatility acceleration
            vol_acceleration = vol_momentum.diff()
            features.append(vol_acceleration.fillna(0).values)
            feature_names.append('vol_acceleration')
            
            if features:
                return np.column_stack(features), feature_names
            else:
                return None, []
                
        except Exception as e:
            tprint_warning(f"Volatility regime feature generation failed: {e}")
            return None, []
    
    def _generate_trend_regime_features(self, market_data: pd.DataFrame) -> Tuple[Optional[np.ndarray], List[str]]:
        """Generate trend regime features."""
        try:
            if 'close' not in market_data.columns:
                return None, []
            
            features = []
            feature_names = []
            close = market_data['close']
            returns = close.pct_change().dropna()
            
            if len(close) < self.lookback_periods:
                return None, []
            
            # Multiple timeframe moving averages
            for period in [5, 10, 20, 50, 100, 200]:
                sma = close.rolling(period).mean()
                features.append(sma.fillna(close).values)
                feature_names.append(f'sma_{period}')
                
                ema = close.ewm(span=period).mean()
                features.append(ema.fillna(close).values)
                feature_names.append(f'ema_{period}')
            
            # Trend strength indicators
            sma_20 = close.rolling(20).mean()
            sma_5 = close.rolling(5).mean()
            
            # Normalized trend strength
            trend_strength = np.abs(sma_20 - sma_5) / (close + 1e-8)
            features.append(trend_strength.fillna(0).values)
            feature_names.append('trend_strength')
            
            # Trend consistency
            trend_consistency = (returns > 0).rolling(20).mean() - 0.5
            features.append(trend_consistency.fillna(0).values)
            feature_names.append('trend_consistency')
            
            # Trend acceleration
            trend_acceleration = trend_strength.diff()
            features.append(trend_acceleration.fillna(0).values)
            feature_names.append('trend_acceleration')
            
            # ADX (Average Directional Index)
            if all(col in market_data.columns for col in ['high', 'low', 'close']):
                adx_features, adx_names = self._calculate_adx(market_data)
                if adx_features is not None:
                    features.append(adx_features)
                    feature_names.extend(adx_names)
            
            # Trend regime classification
            sma_short = close.rolling(10).mean()
            sma_long = close.rolling(30).mean()
            
            # Trend direction
            trend_up = (sma_short > sma_long).astype(int)
            trend_down = (sma_short < sma_long).astype(int)
            trend_sideways = ((sma_short - sma_long).abs() < close * 0.02).astype(int)
            
            features.append(trend_up.fillna(0).values)
            feature_names.append('trend_up')
            features.append(trend_down.fillna(0).values)
            feature_names.append('trend_down')
            features.append(trend_sideways.fillna(1).values)
            feature_names.append('trend_sideways')
            
            # Trend persistence
            trend_persistence = self._calculate_trend_persistence(trend_up, trend_down)
            features.append(trend_persistence.fillna(0).values)
            feature_names.append('trend_persistence')
            
            if features:
                return np.column_stack(features), feature_names
            else:
                return None, []
                
        except Exception as e:
            tprint_warning(f"Trend regime feature generation failed: {e}")
            return None, []
    
    def _calculate_adx(self, market_data: pd.DataFrame, period: int = 14) -> Tuple[Optional[np.ndarray], List[str]]:
        """Calculate ADX and related indicators."""
        try:
            high = market_data['high']
            low = market_data['low']
            close = market_data['close']
            
            # True Range
            tr1 = high - low
            tr2 = np.abs(high - close.shift(1))
            tr3 = np.abs(low - close.shift(1))
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # Directional Movement
            dm_plus = high.diff()
            dm_minus = -low.diff()
            
            dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
            dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
            
            # Smoothed values
            atr = tr.rolling(period).mean()
            di_plus = 100 * (dm_plus.rolling(period).mean() / atr)
            di_minus = 100 * (dm_minus.rolling(period).mean() / atr)
            
            # ADX
            dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-8)
            adx = dx.rolling(period).mean()
            
            features = np.column_stack([
                adx.fillna(0).values,
                di_plus.fillna(0).values,
                di_minus.fillna(0).values
            ])
            
            return features, ['adx', 'dmi_plus', 'dmi_minus']
            
        except Exception as e:
            tprint_warning(f"ADX calculation failed: {e}")
            return None, []
    
    def _calculate_trend_persistence(self, trend_up: pd.Series, trend_down: pd.Series) -> pd.Series:
        """Calculate trend persistence."""
        try:
            trend_signal = trend_up - trend_down  # 1 for up, -1 for down, 0 for sideways
            persistence = pd.Series(0, index=trend_signal.index)
            count = 0
            prev_trend = 0
            
            for i, trend in enumerate(trend_signal):
                if trend == prev_trend and trend != 0:
                    count += 1
                else:
                    count = 1
                persistence.iloc[i] = count
                prev_trend = trend
            
            return persistence
            
        except Exception:
            return pd.Series(0, index=trend_signal.index)
    
    def _generate_momentum_regime_features(self, market_data: pd.DataFrame) -> Tuple[Optional[np.ndarray], List[str]]:
        """Generate momentum regime features."""
        try:
            if 'close' not in market_data.columns:
                return None, []
            
            features = []
            feature_names = []
            close = market_data['close']
            returns = close.pct_change().dropna()
            
            if len(close) < self.lookback_periods:
                return None, []
            
            # RSI
            for period in [14, 21, 50]:
                rsi = self._calculate_rsi(close, period)
                features.append(rsi.fillna(50).values)
                feature_names.append(f'rsi_{period}')
            
            # MACD
            macd_features, macd_names = self._calculate_macd(close)
            if macd_features is not None:
                features.append(macd_features)
                feature_names.extend(macd_names)
            
            # Stochastic Oscillator
            if all(col in market_data.columns for col in ['high', 'low', 'close']):
                stoch_features, stoch_names = self._calculate_stochastic(market_data)
                if stoch_features is not None:
                    features.append(stoch_features)
                    feature_names.extend(stoch_names)
            
            # Williams %R
            if all(col in market_data.columns for col in ['high', 'low', 'close']):
                williams_r = self._calculate_williams_r(market_data)
                features.append(williams_r.fillna(-50).values)
                feature_names.append('williams_r')
            
            # Rate of Change
            for period in [5, 10, 20]:
                roc = close.pct_change(period) * 100
                features.append(roc.fillna(0).values)
                feature_names.append(f'roc_{period}')
            
            # Momentum
            for period in [5, 10, 20]:
                momentum = close / close.shift(period) - 1
                features.append(momentum.fillna(0).values)
                feature_names.append(f'momentum_{period}')
            
            # Momentum regime classification
            rsi_14 = self._calculate_rsi(close, 14)
            momentum_regime_oversold = (rsi_14 < 30).astype(int)
            momentum_regime_overbought = (rsi_14 > 70).astype(int)
            momentum_regime_neutral = ((rsi_14 >= 30) & (rsi_14 <= 70)).astype(int)
            
            features.append(momentum_regime_oversold.fillna(0).values)
            feature_names.append('momentum_regime_oversold')
            features.append(momentum_regime_overbought.fillna(0).values)
            feature_names.append('momentum_regime_overbought')
            features.append(momentum_regime_neutral.fillna(1).values)
            feature_names.append('momentum_regime_neutral')
            
            if features:
                return np.column_stack(features), feature_names
            else:
                return None, []
                
        except Exception as e:
            tprint_warning(f"Momentum regime feature generation failed: {e}")
            return None, []
    
    def _calculate_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        try:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception:
            return pd.Series(50, index=close.index)
    
    def _calculate_macd(self, close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[Optional[np.ndarray], List[str]]:
        """Calculate MACD."""
        try:
            ema_fast = close.ewm(span=fast).mean()
            ema_slow = close.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal).mean()
            macd_histogram = macd - macd_signal
            
            features = np.column_stack([
                macd.fillna(0).values,
                macd_signal.fillna(0).values,
                macd_histogram.fillna(0).values
            ])
            
            return features, ['macd', 'macd_signal', 'macd_histogram']
            
        except Exception as e:
            tprint_warning(f"MACD calculation failed: {e}")
            return None, []
    
    def _calculate_stochastic(self, market_data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[Optional[np.ndarray], List[str]]:
        """Calculate Stochastic Oscillator."""
        try:
            high = market_data['high']
            low = market_data['low']
            close = market_data['close']
            
            lowest_low = low.rolling(k_period).min()
            highest_high = high.rolling(k_period).max()
            
            k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-8)
            d_percent = k_percent.rolling(d_period).mean()
            
            features = np.column_stack([
                k_percent.fillna(50).values,
                d_percent.fillna(50).values
            ])
            
            return features, ['stoch_k', 'stoch_d']
            
        except Exception as e:
            tprint_warning(f"Stochastic calculation failed: {e}")
            return None, []
    
    def _calculate_williams_r(self, market_data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        try:
            high = market_data['high']
            low = market_data['low']
            close = market_data['close']
            
            highest_high = high.rolling(period).max()
            lowest_low = low.rolling(period).min()
            
            williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-8)
            return williams_r
            
        except Exception:
            return pd.Series(-50, index=market_data.index)
    
    def _generate_volume_regime_features(self, market_data: pd.DataFrame) -> Tuple[Optional[np.ndarray], List[str]]:
        """Generate volume regime features."""
        try:
            if 'volume' not in market_data.columns or 'close' not in market_data.columns:
                return None, []
            
            features = []
            feature_names = []
            volume = market_data['volume']
            close = market_data['close']
            returns = close.pct_change().dropna()
            
            if len(volume) < self.lookback_periods:
                return None, []
            
            # Volume moving averages
            for period in [5, 10, 20, 50]:
                vol_sma = volume.rolling(period).mean()
                features.append(vol_sma.fillna(volume.mean()).values)
                feature_names.append(f'volume_sma_{period}')
            
            # Volume ratio
            vol_ratio = volume / (volume.rolling(20).mean() + 1e-8)
            features.append(vol_ratio.fillna(1).values)
            feature_names.append('volume_ratio')
            
            # Volume regime classification
            vol_percentile = volume.rolling(252).rank(pct=True)
            vol_regime_high = (vol_percentile > 0.8).astype(int)
            vol_regime_low = (vol_percentile < 0.2).astype(int)
            vol_regime_normal = ((vol_percentile >= 0.2) & (vol_percentile <= 0.8)).astype(int)
            
            features.append(vol_regime_high.fillna(0).values)
            feature_names.append('vol_regime_high')
            features.append(vol_regime_low.fillna(0).values)
            feature_names.append('vol_regime_low')
            features.append(vol_regime_normal.fillna(1).values)
            feature_names.append('vol_regime_normal')
            
            # On-Balance Volume
            obv = self._calculate_obv(close, volume)
            features.append(obv.fillna(0).values)
            feature_names.append('obv')
            
            # Accumulation/Distribution Line
            ad_line = self._calculate_ad_line(market_data)
            if ad_line is not None:
                features.append(ad_line.fillna(0).values)
                feature_names.append('ad_line')
            
            # Money Flow Index
            mfi = self._calculate_mfi(market_data)
            if mfi is not None:
                features.append(mfi.fillna(50).values)
                feature_names.append('mfi')
            
            # Volume-Price Trend
            vpt = self._calculate_vpt(close, volume)
            features.append(vpt.fillna(0).values)
            feature_names.append('vpt')
            
            if features:
                return np.column_stack(features), feature_names
            else:
                return None, []
                
        except Exception as e:
            tprint_warning(f"Volume regime feature generation failed: {e}")
            return None, []
    
    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume."""
        try:
            obv = pd.Series(0, index=close.index)
            obv.iloc[0] = volume.iloc[0]
            
            for i in range(1, len(close)):
                if close.iloc[i] > close.iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
                elif close.iloc[i] < close.iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            
            return obv
        except Exception:
            return pd.Series(0, index=close.index)
    
    def _calculate_ad_line(self, market_data: pd.DataFrame) -> Optional[pd.Series]:
        """Calculate Accumulation/Distribution Line."""
        try:
            if not all(col in market_data.columns for col in ['high', 'low', 'close', 'volume']):
                return None
            
            high = market_data['high']
            low = market_data['low']
            close = market_data['close']
            volume = market_data['volume']
            
            clv = ((close - low) - (high - close)) / (high - low + 1e-8)
            ad_line = (clv * volume).cumsum()
            
            return ad_line
        except Exception:
            return None
    
    def _calculate_mfi(self, market_data: pd.DataFrame, period: int = 14) -> Optional[pd.Series]:
        """Calculate Money Flow Index."""
        try:
            if not all(col in market_data.columns for col in ['high', 'low', 'close', 'volume']):
                return None
            
            high = market_data['high']
            low = market_data['low']
            close = market_data['close']
            volume = market_data['volume']
            
            typical_price = (high + low + close) / 3
            money_flow = typical_price * volume
            
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(period).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(period).sum()
            
            mfi = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-8)))
            
            return mfi
        except Exception:
            return None
    
    def _calculate_vpt(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Volume-Price Trend."""
        try:
            returns = close.pct_change()
            vpt = (returns * volume).cumsum()
            return vpt
        except Exception:
            return pd.Series(0, index=close.index)
    
    def _generate_regime_persistence_features(self, market_data: pd.DataFrame) -> Tuple[Optional[np.ndarray], List[str]]:
        """Generate regime persistence features."""
        try:
            if 'close' not in market_data.columns:
                return None, []
            
            features = []
            feature_names = []
            close = market_data['close']
            returns = close.pct_change().dropna()
            
            if len(close) < self.lookback_periods:
                return None, []
            
            # Volatility regime persistence
            vol_20 = returns.rolling(20).std() * np.sqrt(252)
            vol_mean_60 = vol_20.rolling(60).mean()
            vol_std_60 = vol_20.rolling(60).std()
            vol_regime = (vol_20 > vol_mean_60 + vol_std_60).astype(int)
            
            vol_persistence = self._calculate_persistence(vol_regime)
            features.append(vol_persistence.fillna(0).values)
            feature_names.append('vol_regime_persistence')
            
            # Trend regime persistence
            sma_short = close.rolling(10).mean()
            sma_long = close.rolling(30).mean()
            trend_regime = (sma_short > sma_long).astype(int)
            
            trend_persistence = self._calculate_persistence(trend_regime)
            features.append(trend_persistence.fillna(0).values)
            feature_names.append('trend_regime_persistence')
            
            # Return regime persistence
            return_regime = (returns > 0).astype(int)
            return_persistence = self._calculate_persistence(return_regime)
            features.append(return_persistence.fillna(0).values)
            feature_names.append('return_regime_persistence')
            
            # Regime transition probability
            transition_prob = self._calculate_transition_probability(vol_regime)
            features.append(transition_prob.fillna(0).values)
            feature_names.append('regime_transition_probability')
            
            # Regime stability
            stability = self._calculate_regime_stability(vol_regime, trend_regime)
            features.append(stability.fillna(0).values)
            feature_names.append('regime_stability')
            
            if features:
                return np.column_stack(features), feature_names
            else:
                return None, []
                
        except Exception as e:
            tprint_warning(f"Regime persistence feature generation failed: {e}")
            return None, []
    
    def _calculate_persistence(self, regime_signal: pd.Series) -> pd.Series:
        """Calculate regime persistence."""
        try:
            persistence = pd.Series(0, index=regime_signal.index)
            count = 0
            prev_regime = -1
            
            for i, regime in enumerate(regime_signal):
                if regime == prev_regime:
                    count += 1
                else:
                    count = 1
                persistence.iloc[i] = count
                prev_regime = regime
            
            return persistence
        except Exception:
            return pd.Series(0, index=regime_signal.index)
    
    def _calculate_transition_probability(self, regime_signal: pd.Series, window: int = 20) -> pd.Series:
        """Calculate regime transition probability."""
        try:
            transitions = regime_signal.diff().abs()
            transition_prob = transitions.rolling(window).mean()
            return transition_prob
        except Exception:
            return pd.Series(0, index=regime_signal.index)
    
    def _calculate_regime_stability(self, vol_regime: pd.Series, trend_regime: pd.Series) -> pd.Series:
        """Calculate regime stability."""
        try:
            # Combine regimes
            combined_regime = vol_regime * 2 + trend_regime
            
            # Calculate stability as inverse of transition frequency
            transitions = combined_regime.diff().abs()
            stability = 1 - transitions.rolling(20).mean()
            
            return stability
        except Exception:
            return pd.Series(0, index=vol_regime.index)
    
    def _generate_economic_regime_features(self, market_data: pd.DataFrame) -> Tuple[Optional[np.ndarray], List[str]]:
        """Generate economic regime features."""
        try:
            if 'close' not in market_data.columns:
                return None, []
            
            features = []
            feature_names = []
            close = market_data['close']
            returns = close.pct_change().dropna()
            
            if len(close) < self.lookback_periods:
                return None, []
            
            # Market phase indicators
            sma_50 = close.rolling(50).mean()
            sma_200 = close.rolling(200).mean()
            
            # Bull market indicator
            bull_market = (sma_50 > sma_200).astype(int)
            features.append(bull_market.fillna(0).values)
            feature_names.append('bull_market')
            
            # Bear market indicator
            bear_market = (sma_50 < sma_200).astype(int)
            features.append(bear_market.fillna(0).values)
            feature_names.append('bear_market')
            
            # Market breadth proxy (using price vs moving average)
            breadth = (close > sma_50).rolling(20).mean()
            features.append(breadth.fillna(0.5).values)
            feature_names.append('market_breadth')
            
            # Fear and Greed indicators
            # VIX proxy using volatility
            vol_20 = returns.rolling(20).std() * np.sqrt(252)
            vix_proxy = vol_20 * 100
            features.append(vix_proxy.fillna(20).values)
            feature_names.append('vix_proxy')
            
            # Fear index (high volatility = high fear)
            fear_index = (vol_20 > vol_20.rolling(252).quantile(0.8)).astype(int)
            features.append(fear_index.fillna(0).values)
            feature_names.append('fear_index')
            
            # Greed index (low volatility = high greed)
            greed_index = (vol_20 < vol_20.rolling(252).quantile(0.2)).astype(int)
            features.append(greed_index.fillna(0).values)
            feature_names.append('greed_index')
            
            # Market sentiment
            sentiment = (returns > 0).rolling(20).mean() - 0.5
            features.append(sentiment.fillna(0).values)
            feature_names.append('market_sentiment')
            
            # Economic cycle indicators
            # Recession indicator (negative returns over long period)
            recession_indicator = (returns.rolling(60).sum() < -0.1).astype(int)
            features.append(recession_indicator.fillna(0).values)
            feature_names.append('recession_indicator')
            
            # Recovery indicator (positive returns after decline)
            recovery_indicator = ((returns.rolling(60).sum() > 0.05) & 
                                (returns.rolling(20).sum() > 0.02)).astype(int)
            features.append(recovery_indicator.fillna(0).values)
            feature_names.append('recovery_indicator')
            
            if features:
                return np.column_stack(features), feature_names
            else:
                return None, []
                
        except Exception as e:
            tprint_warning(f"Economic regime feature generation failed: {e}")
            return None, []
    
    def _generate_microstructure_features(self, market_data: pd.DataFrame) -> Tuple[Optional[np.ndarray], List[str]]:
        """Generate market microstructure features."""
        try:
            if not all(col in market_data.columns for col in ['high', 'low', 'close', 'volume']):
                return None, []
            
            features = []
            feature_names = []
            high = market_data['high']
            low = market_data['low']
            close = market_data['close']
            volume = market_data['volume']
            
            # Price impact features
            returns = close.pct_change()
            
            # Volume-weighted average price proxy
            vwap = (close * volume).rolling(20).sum() / volume.rolling(20).sum()
            price_impact = (close - vwap) / vwap
            features.append(price_impact.fillna(0).values)
            feature_names.append('price_impact')
            
            # Bid-ask spread proxy
            spread_proxy = (high - low) / close
            features.append(spread_proxy.fillna(0).values)
            feature_names.append('spread_proxy')
            
            # Order flow imbalance
            ofi = (close - low) / (high - low + 1e-8) - 0.5
            features.append(ofi.fillna(0).values)
            feature_names.append('order_flow_imbalance')
            
            # Volume-price relationship
            volume_price_corr = returns.rolling(20).corr(volume.pct_change())
            features.append(volume_price_corr.fillna(0).values)
            feature_names.append('volume_price_correlation')
            
            # Intraday volatility
            intraday_vol = (high - low) / close
            features.append(intraday_vol.fillna(0).values)
            feature_names.append('intraday_volatility')
            
            # Price efficiency
            price_efficiency = np.abs(returns) / (high - low + 1e-8)
            features.append(price_efficiency.fillna(0).values)
            feature_names.append('price_efficiency')
            
            if features:
                return np.column_stack(features), feature_names
            else:
                return None, []
                
        except Exception as e:
            tprint_warning(f"Microstructure feature generation failed: {e}")
            return None, []


def create_feature_generator(lookback_periods: int = 20) -> AdvancedFeatureGenerator:
    """Create feature generator instance."""
    return AdvancedFeatureGenerator(lookback_periods=lookback_periods)