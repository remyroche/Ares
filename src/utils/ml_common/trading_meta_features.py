"""
Trading-Specific Meta-Features for High Leverage Trading HPO

This module provides comprehensive meta-feature extraction specifically designed
for financial trading datasets and high leverage trading scenarios.

Key Features:
- Market regime characterization
- Volatility and risk metrics
- Trading frequency and liquidity analysis
- Regime transition detection
- Financial performance indicators
- High leverage specific features
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from scipy import stats
from scipy.signal import find_peaks
import warnings

from ..math_validation import safe_divide, safe_log
from ..common_operations import create_fallback_logger

logger = logging.getLogger(__name__)

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib not available - limited technical analysis features")


class TradingMetaFeaturesExtractor:
    """Extract comprehensive meta-features for trading datasets."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize trading meta-features extractor."""
        self.config = config or {}
        self.logger = logger.getChild('TradingMetaFeatures')
        
        # Configuration defaults
        self.lookback_periods = self.config.get('lookback_periods', [5, 10, 20, 50, 100])
        self.volatility_windows = self.config.get('volatility_windows', [10, 20, 50])
        self.regime_detection_window = self.config.get('regime_detection_window', 50)
        
    def extract_trading_meta_features(self, 
                                    price_data: pd.DataFrame,
                                    volume_data: Optional[pd.Series] = None,
                                    returns_data: Optional[pd.Series] = None,
                                    regime_labels: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Extract comprehensive trading meta-features.
        
        Args:
            price_data: DataFrame with OHLCV data or price series
            volume_data: Optional volume data
            returns_data: Optional pre-calculated returns
            regime_labels: Optional market regime labels
            
        Returns:
            Dictionary of trading meta-features
        """
        try:
            self.logger.info("📊 Extracting trading meta-features")
            
            meta_features = {}
            
            # Basic dataset characteristics
            meta_features.update(self._extract_basic_characteristics(price_data, volume_data))
            
            # Price and returns analysis
            if returns_data is not None:
                returns = returns_data
            else:
                returns = self._calculate_returns(price_data)
            
            meta_features.update(self._extract_returns_characteristics(returns))
            
            # Volatility analysis
            meta_features.update(self._extract_volatility_features(returns, price_data))
            
            # Market regime analysis
            meta_features.update(self._extract_regime_features(returns, regime_labels))
            
            # Technical analysis features
            meta_features.update(self._extract_technical_features(price_data, volume_data))
            
            # High leverage specific features
            meta_features.update(self._extract_leverage_features(returns, price_data))
            
            # Risk metrics
            meta_features.update(self._extract_risk_metrics(returns))
            
            # Trading frequency and liquidity
            meta_features.update(self._extract_liquidity_features(price_data, volume_data))
            
            # Regime transition analysis
            meta_features.update(self._extract_regime_transition_features(returns, regime_labels))
            
            self.logger.info(f"✅ Extracted {len(meta_features)} trading meta-features")
            return meta_features
            
        except Exception as e:
            self.logger.error(f"❌ Trading meta-features extraction failed: {e}")
            return {}
    
    def _extract_basic_characteristics(self, 
                                     price_data: pd.DataFrame, 
                                     volume_data: Optional[pd.Series]) -> Dict[str, float]:
        """Extract basic dataset characteristics."""
        try:
            characteristics = {}
            
            # Dataset size
            characteristics['n_observations'] = len(price_data)
            characteristics['n_features'] = price_data.shape[1] if hasattr(price_data, 'shape') else 1
            
            # Time span
            if hasattr(price_data, 'index') and isinstance(price_data.index, pd.DatetimeIndex):
                time_span = (price_data.index[-1] - price_data.index[0]).days
                characteristics['time_span_days'] = time_span
                characteristics['observations_per_day'] = len(price_data) / max(1, time_span)
            
            # Price range
            if 'close' in price_data.columns:
                close_prices = price_data['close']
            elif 'Close' in price_data.columns:
                close_prices = price_data['Close']
            else:
                close_prices = price_data.iloc[:, -1] if hasattr(price_data, 'iloc') else price_data
            
            characteristics['price_range_ratio'] = safe_divide(
                close_prices.max() - close_prices.min(), close_prices.mean()
            )
            characteristics['price_volatility'] = float(close_prices.std() / close_prices.mean())
            
            # Volume characteristics
            if volume_data is not None:
                characteristics['volume_mean'] = float(volume_data.mean())
                characteristics['volume_std'] = float(volume_data.std())
                characteristics['volume_skewness'] = float(stats.skew(volume_data.dropna()))
                characteristics['volume_kurtosis'] = float(stats.kurtosis(volume_data.dropna()))
            
            return characteristics
            
        except Exception as e:
            self.logger.warning(f"Basic characteristics extraction failed: {e}")
            return {}
    
    def _calculate_returns(self, price_data: pd.DataFrame) -> pd.Series:
        """Calculate returns from price data."""
        try:
            if 'close' in price_data.columns:
                prices = price_data['close']
            elif 'Close' in price_data.columns:
                prices = price_data['Close']
            else:
                prices = price_data.iloc[:, -1] if hasattr(price_data, 'iloc') else price_data
            
            returns = prices.pct_change().dropna()
            return returns
            
        except Exception as e:
            self.logger.warning(f"Returns calculation failed: {e}")
            return pd.Series()
    
    def _extract_returns_characteristics(self, returns: pd.Series) -> Dict[str, float]:
        """Extract returns-based characteristics."""
        try:
            if len(returns) == 0:
                return {}
            
            characteristics = {}
            
            # Basic statistics
            characteristics['returns_mean'] = float(returns.mean())
            characteristics['returns_std'] = float(returns.std())
            characteristics['returns_skewness'] = float(stats.skew(returns.dropna()))
            characteristics['returns_kurtosis'] = float(stats.kurtosis(returns.dropna()))
            
            # Risk metrics
            characteristics['sharpe_ratio'] = safe_divide(returns.mean(), returns.std())
            characteristics['max_drawdown'] = self._calculate_max_drawdown(returns)
            characteristics['var_95'] = float(np.percentile(returns.dropna(), 5))
            characteristics['var_99'] = float(np.percentile(returns.dropna(), 1))
            
            # Autocorrelation
            if len(returns) > 1:
                characteristics['returns_autocorr_1'] = float(returns.autocorr(lag=1))
                characteristics['returns_autocorr_5'] = float(returns.autocorr(lag=5))
            
            # Volatility clustering
            characteristics['volatility_clustering'] = self._calculate_volatility_clustering(returns)
            
            # Fat tails
            characteristics['fat_tails'] = self._calculate_fat_tails(returns)
            
            return characteristics
            
        except Exception as e:
            self.logger.warning(f"Returns characteristics extraction failed: {e}")
            return {}
    
    def _extract_volatility_features(self, 
                                   returns: pd.Series, 
                                   price_data: pd.DataFrame) -> Dict[str, float]:
        """Extract volatility-related features."""
        try:
            if len(returns) == 0:
                return {}
            
            features = {}
            
            # Rolling volatility
            for window in self.volatility_windows:
                if len(returns) >= window:
                    rolling_vol = returns.rolling(window=window).std()
                    features[f'volatility_mean_{window}'] = float(rolling_vol.mean())
                    features[f'volatility_std_{window}'] = float(rolling_vol.std())
                    features[f'volatility_skewness_{window}'] = float(stats.skew(rolling_vol.dropna()))
            
            # Volatility of volatility
            if len(returns) >= 20:
                vol_20 = returns.rolling(window=20).std()
                features['vol_of_vol'] = float(vol_20.std())
            
            # GARCH-like features
            features['volatility_persistence'] = self._calculate_volatility_persistence(returns)
            
            # Regime-dependent volatility
            features['volatility_regime_switching'] = self._detect_volatility_regime_switching(returns)
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Volatility features extraction failed: {e}")
            return {}
    
    def _extract_regime_features(self, 
                               returns: pd.Series, 
                               regime_labels: Optional[pd.Series]) -> Dict[str, float]:
        """Extract market regime features."""
        try:
            features = {}
            
            if regime_labels is not None:
                # Regime distribution
                regime_counts = regime_labels.value_counts()
                features['n_regimes'] = len(regime_counts)
                features['regime_entropy'] = -sum((count/len(regime_labels)) * safe_log(count/len(regime_labels)) 
                                                 for count in regime_counts if count > 0)
                features['dominant_regime_ratio'] = float(regime_counts.max() / len(regime_labels))
                
                # Regime-specific returns
                for regime in regime_counts.index:
                    regime_returns = returns[regime_labels == regime]
                    if len(regime_returns) > 0:
                        features[f'regime_{regime}_mean_return'] = float(regime_returns.mean())
                        features[f'regime_{regime}_volatility'] = float(regime_returns.std())
            else:
                # Detect regimes automatically
                detected_regimes = self._detect_market_regimes(returns)
                if detected_regimes is not None:
                    features.update(self._extract_regime_features(returns, detected_regimes))
            
            # Trend strength
            features['trend_strength'] = self._calculate_trend_strength(returns)
            
            # Mean reversion
            features['mean_reversion'] = self._calculate_mean_reversion(returns)
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Regime features extraction failed: {e}")
            return {}
    
    def _extract_technical_features(self, 
                                  price_data: pd.DataFrame, 
                                  volume_data: Optional[pd.Series]) -> Dict[str, float]:
        """Extract technical analysis features."""
        try:
            features = {}
            
            if 'close' in price_data.columns:
                close = price_data['close']
                high = price_data.get('high', close)
                low = price_data.get('low', close)
                open_price = price_data.get('open', close)
            elif 'Close' in price_data.columns:
                close = price_data['Close']
                high = price_data.get('High', close)
                low = price_data.get('Low', close)
                open_price = price_data.get('Open', close)
            else:
                close = price_data.iloc[:, -1] if hasattr(price_data, 'iloc') else price_data
                high = low = open_price = close
            
            # Moving averages
            for period in [5, 10, 20, 50]:
                if len(close) >= period:
                    ma = close.rolling(window=period).mean()
                    features[f'price_vs_ma_{period}'] = float((close.iloc[-1] - ma.iloc[-1]) / ma.iloc[-1])
            
            # RSI
            if TALIB_AVAILABLE and len(close) >= 14:
                rsi = talib.RSI(close.values, timeperiod=14)
                if not np.isnan(rsi[-1]):
                    features['rsi'] = float(rsi[-1])
                    features['rsi_extreme'] = 1.0 if rsi[-1] > 80 or rsi[-1] < 20 else 0.0
            
            # Bollinger Bands
            if len(close) >= 20:
                bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close, 20, 2)
                features['bb_position'] = float((close.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]))
                features['bb_squeeze'] = float((bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_middle.iloc[-1])
            
            # Volume features
            if volume_data is not None:
                features['volume_price_trend'] = self._calculate_volume_price_trend(close, volume_data)
                features['volume_volatility'] = float(volume_data.rolling(window=20).std().iloc[-1])
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Technical features extraction failed: {e}")
            return {}
    
    def _extract_leverage_features(self, 
                                 returns: pd.Series, 
                                 price_data: pd.DataFrame) -> Dict[str, float]:
        """Extract high leverage specific features."""
        try:
            features = {}
            
            # Leverage risk metrics
            features['leverage_risk'] = self._calculate_leverage_risk(returns)
            features['margin_call_risk'] = self._calculate_margin_call_risk(returns)
            
            # High frequency features
            features['high_freq_volatility'] = self._calculate_high_freq_volatility(returns)
            features['microstructure_noise'] = self._calculate_microstructure_noise(returns)
            
            # Execution risk
            features['execution_risk'] = self._calculate_execution_risk(returns, price_data)
            
            # Correlation with market
            features['market_correlation'] = self._calculate_market_correlation(returns)
            
            # Liquidity risk
            features['liquidity_risk'] = self._calculate_liquidity_risk(returns)
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Leverage features extraction failed: {e}")
            return {}
    
    def _extract_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Extract comprehensive risk metrics."""
        try:
            if len(returns) == 0:
                return {}
            
            metrics = {}
            
            # Downside risk
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                metrics['downside_deviation'] = float(downside_returns.std())
                metrics['downside_ratio'] = safe_divide(returns.mean(), downside_returns.std())
            
            # Tail risk
            metrics['tail_ratio'] = safe_divide(
                np.percentile(returns.dropna(), 95) - np.percentile(returns.dropna(), 50),
                np.percentile(returns.dropna(), 50) - np.percentile(returns.dropna(), 5)
            )
            
            # Expected shortfall
            var_95 = np.percentile(returns.dropna(), 5)
            metrics['expected_shortfall_95'] = float(returns[returns <= var_95].mean())
            
            # Maximum adverse excursion
            metrics['max_adverse_excursion'] = self._calculate_max_adverse_excursion(returns)
            
            # Calmar ratio
            max_dd = self._calculate_max_drawdown(returns)
            metrics['calmar_ratio'] = safe_divide(returns.mean() * 252, abs(max_dd))
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Risk metrics extraction failed: {e}")
            return {}
    
    def _extract_liquidity_features(self, 
                                  price_data: pd.DataFrame, 
                                  volume_data: Optional[pd.Series]) -> Dict[str, float]:
        """Extract liquidity-related features."""
        try:
            features = {}
            
            if volume_data is not None:
                # Volume-based liquidity
                features['avg_volume'] = float(volume_data.mean())
                features['volume_consistency'] = 1.0 - float(volume_data.std() / volume_data.mean())
                
                # Volume spikes
                volume_threshold = volume_data.quantile(0.95)
                features['volume_spike_frequency'] = float((volume_data > volume_threshold).mean())
            
            # Price impact
            if 'close' in price_data.columns or 'Close' in price_data.columns:
                close = price_data['close'] if 'close' in price_data.columns else price_data['Close']
                features['price_impact'] = self._calculate_price_impact(close, volume_data)
            
            # Bid-ask spread proxy
            if 'high' in price_data.columns and 'low' in price_data.columns:
                high = price_data['high']
                low = price_data['low']
                features['spread_proxy'] = float(((high - low) / close).mean())
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Liquidity features extraction failed: {e}")
            return {}
    
    def _extract_regime_transition_features(self, 
                                          returns: pd.Series, 
                                          regime_labels: Optional[pd.Series]) -> Dict[str, float]:
        """Extract regime transition features."""
        try:
            features = {}
            
            if regime_labels is not None:
                # Transition frequency
                transitions = (regime_labels != regime_labels.shift(1)).sum()
                features['regime_transition_frequency'] = float(transitions / len(regime_labels))
                
                # Transition patterns
                features['regime_persistence'] = 1.0 - features['regime_transition_frequency']
                
                # Volatility around transitions
                transition_indices = regime_labels[regime_labels != regime_labels.shift(1)].index
                if len(transition_indices) > 0:
                    transition_volatility = []
                    for idx in transition_indices:
                        window_start = max(0, idx - 5)
                        window_end = min(len(returns), idx + 5)
                        window_vol = returns.iloc[window_start:window_end].std()
                        transition_volatility.append(window_vol)
                    
                    features['transition_volatility'] = float(np.mean(transition_volatility))
            
            # Volatility regime changes
            vol_regimes = self._detect_volatility_regimes(returns)
            if vol_regimes is not None:
                vol_transitions = (vol_regimes != vol_regimes.shift(1)).sum()
                features['volatility_regime_changes'] = float(vol_transitions / len(returns))
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Regime transition features extraction failed: {e}")
            return {}
    
    # Helper methods for calculations
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return float(drawdown.min())
        except:
            return 0.0
    
    def _calculate_volatility_clustering(self, returns: pd.Series) -> float:
        """Calculate volatility clustering measure."""
        try:
            if len(returns) < 20:
                return 0.0
            
            vol_20 = returns.rolling(window=20).std()
            return float(vol_20.autocorr(lag=1))
        except:
            return 0.0
    
    def _calculate_fat_tails(self, returns: pd.Series) -> float:
        """Calculate fat tails measure."""
        try:
            kurtosis = stats.kurtosis(returns.dropna())
            return float(max(0, kurtosis - 3))  # Excess kurtosis
        except:
            return 0.0
    
    def _calculate_volatility_persistence(self, returns: pd.Series) -> float:
        """Calculate volatility persistence (GARCH-like)."""
        try:
            if len(returns) < 50:
                return 0.0
            
            vol_10 = returns.rolling(window=10).std()
            vol_20 = returns.rolling(window=20).std()
            
            # Correlation between short-term and long-term volatility
            correlation = vol_10.corr(vol_20)
            return float(correlation) if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _detect_volatility_regime_switching(self, returns: pd.Series) -> float:
        """Detect volatility regime switching."""
        try:
            if len(returns) < 100:
                return 0.0
            
            vol_20 = returns.rolling(window=20).std()
            vol_50 = returns.rolling(window=50).std()
            
            # Count regime switches
            regime_switches = (vol_20 > vol_50).astype(int).diff().abs().sum()
            return float(regime_switches / len(returns))
        except:
            return 0.0
    
    def _detect_market_regimes(self, returns: pd.Series) -> Optional[pd.Series]:
        """Detect market regimes automatically."""
        try:
            if len(returns) < 100:
                return None
            
            # Simple regime detection based on volatility and trend
            vol_20 = returns.rolling(window=20).std()
            trend_20 = returns.rolling(window=20).mean()
            
            # High volatility regime
            high_vol = vol_20 > vol_20.quantile(0.7)
            
            # Bull/bear regimes
            bull_market = trend_20 > trend_20.quantile(0.6)
            bear_market = trend_20 < trend_20.quantile(0.4)
            
            # Combine into regimes
            regimes = pd.Series('normal', index=returns.index)
            regimes[high_vol & bull_market] = 'high_vol_bull'
            regimes[high_vol & bear_market] = 'high_vol_bear'
            regimes[~high_vol & bull_market] = 'low_vol_bull'
            regimes[~high_vol & bear_market] = 'low_vol_bear'
            
            return regimes
        except:
            return None
    
    def _calculate_trend_strength(self, returns: pd.Series) -> float:
        """Calculate trend strength."""
        try:
            if len(returns) < 20:
                return 0.0
            
            # Linear regression slope
            x = np.arange(len(returns))
            slope, _, _, _, _ = stats.linregress(x, returns.values)
            return float(slope)
        except:
            return 0.0
    
    def _calculate_mean_reversion(self, returns: pd.Series) -> float:
        """Calculate mean reversion strength."""
        try:
            if len(returns) < 20:
                return 0.0
            
            # Hurst exponent approximation
            lags = range(2, min(20, len(returns)//2))
            tau = [np.sqrt(np.std(np.subtract(returns[lag:], returns[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return float(poly[0] * 2.0)  # Hurst exponent
        except:
            return 0.0
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        try:
            rolling_mean = prices.rolling(window=window).mean()
            rolling_std = prices.rolling(window=window).std()
            
            upper_band = rolling_mean + (rolling_std * std_dev)
            lower_band = rolling_mean - (rolling_std * std_dev)
            
            return upper_band, rolling_mean, lower_band
        except:
            return pd.Series(), pd.Series(), pd.Series()
    
    def _calculate_volume_price_trend(self, prices: pd.Series, volume: pd.Series) -> float:
        """Calculate volume-price trend correlation."""
        try:
            if len(prices) != len(volume):
                return 0.0
            
            price_change = prices.pct_change()
            volume_change = volume.pct_change()
            
            correlation = price_change.corr(volume_change)
            return float(correlation) if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _calculate_leverage_risk(self, returns: pd.Series) -> float:
        """Calculate leverage risk metric."""
        try:
            # Risk increases exponentially with leverage
            # This is a simplified measure
            max_loss = abs(returns.min())
            leverage_risk = min(1.0, max_loss * 10)  # Assume 10x leverage
            return float(leverage_risk)
        except:
            return 0.0
    
    def _calculate_margin_call_risk(self, returns: pd.Series) -> float:
        """Calculate margin call risk."""
        try:
            # Probability of hitting margin call threshold
            margin_threshold = -0.1  # 10% loss triggers margin call
            margin_call_prob = (returns < margin_threshold).mean()
            return float(margin_call_prob)
        except:
            return 0.0
    
    def _calculate_high_freq_volatility(self, returns: pd.Series) -> float:
        """Calculate high frequency volatility."""
        try:
            if len(returns) < 5:
                return 0.0
            
            # Intraday volatility proxy
            high_freq_vol = returns.rolling(window=5).std().mean()
            return float(high_freq_vol)
        except:
            return 0.0
    
    def _calculate_microstructure_noise(self, returns: pd.Series) -> float:
        """Calculate microstructure noise."""
        try:
            if len(returns) < 10:
                return 0.0
            
            # Autocorrelation at lag 1 as noise proxy
            autocorr = returns.autocorr(lag=1)
            return float(abs(autocorr)) if not np.isnan(autocorr) else 0.0
        except:
            return 0.0
    
    def _calculate_execution_risk(self, returns: pd.Series, price_data: pd.DataFrame) -> float:
        """Calculate execution risk."""
        try:
            # Price impact and slippage risk
            if 'high' in price_data.columns and 'low' in price_data.columns:
                high = price_data['high']
                low = price_data['low']
                close = price_data.get('close', price_data.iloc[:, -1])
                
                execution_risk = ((high - low) / close).mean()
                return float(execution_risk)
            else:
                return float(returns.std() * 0.1)  # Simplified measure
        except:
            return 0.0
    
    def _calculate_market_correlation(self, returns: pd.Series) -> float:
        """Calculate correlation with market (simplified)."""
        try:
            # This would typically use a market index
            # For now, use autocorrelation as proxy
            market_corr = returns.autocorr(lag=1)
            return float(market_corr) if not np.isnan(market_corr) else 0.0
        except:
            return 0.0
    
    def _calculate_liquidity_risk(self, returns: pd.Series) -> float:
        """Calculate liquidity risk."""
        try:
            # Volatility of volatility as liquidity risk proxy
            vol_20 = returns.rolling(window=20).std()
            liquidity_risk = vol_20.std()
            return float(liquidity_risk)
        except:
            return 0.0
    
    def _calculate_max_adverse_excursion(self, returns: pd.Series) -> float:
        """Calculate maximum adverse excursion."""
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            adverse_excursion = (cumulative - running_max) / running_max
            return float(adverse_excursion.min())
        except:
            return 0.0
    
    def _calculate_price_impact(self, prices: pd.Series, volume: pd.Series) -> float:
        """Calculate price impact measure."""
        try:
            if volume is None or len(prices) != len(volume):
                return 0.0
            
            # Price change per unit volume
            price_change = prices.pct_change()
            volume_change = volume.pct_change()
            
            # Avoid division by zero
            volume_change = volume_change.replace(0, np.nan)
            price_impact = (price_change / volume_change).mean()
            
            return float(price_impact) if not np.isnan(price_impact) else 0.0
        except:
            return 0.0
    
    def _detect_volatility_regimes(self, returns: pd.Series) -> Optional[pd.Series]:
        """Detect volatility regimes."""
        try:
            if len(returns) < 50:
                return None
            
            vol_20 = returns.rolling(window=20).std()
            vol_threshold = vol_20.quantile(0.5)
            
            regimes = pd.Series('low_vol', index=returns.index)
            regimes[vol_20 > vol_threshold] = 'high_vol'
            
            return regimes
        except:
            return None