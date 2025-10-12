from src.utils.tprint import tprint

from src.core.decorators import handles_errors
"""Utility functions and classes for advanced feature engineering."""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

from ..utils.logger import system_logger

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


class TechnicalIndicatorCalculator:
    """Collection of technical indicator calculation methods."""

    def __init__(self, config: List[Dict[str, Any]]):
        """Initialize with configuration."""
        self.config = config
        self.logger = system_logger.getChild('TechnicalIndicatorCalculator')

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all configured technical indicators."""
        results = pd.DataFrame(index=data.index)

        for indicator_config in self.config:
            name = indicator_config.get('name', '')
            params = indicator_config.get('params', {})

            try:
                if name == 'RSI':
                    period = params.get('period', 14)
                    results[f'rsi_{period}'] = self.calculate_rsi(data['close'], period)
                elif name == 'SMA':
                    period = params.get('period', 20)
                    results[f'sma_{period}'] = data['close'].rolling(period).mean()
                elif name == 'EMA':
                    period = params.get('period', 12)
                    results[f'ema_{period}'] = data['close'].ewm(span=period).mean()
                elif name == 'MACD':
                    fast = params.get('fast', 12)
                    slow = params.get('slow', 26)
                    signal = params.get('signal', 9)
                    results[f'macd_{fast}_{slow}_{signal}'] = self.calculate_macd(data['close'], fast, slow)
                elif name == 'ATR':
                    period = params.get('period', 14)
                    results[f'atr_{period}'] = self.calculate_atr(data, period)
                elif name == 'BB':
                    window = params.get('window', 20)
                    num_std = params.get('num_std', 2)
                    bb_features = self.calculate_bollinger_bands(data['close'], window, num_std)
                    for col in bb_features.columns:
                        results[f'{col}_{window}'] = bb_features[col]
                elif name == 'ADX':
                    period = params.get('period', 14)
                    results[f'adx_{period}'] = self.calculate_adx(data, period)
                else:
                    self.logger.warning(f"Unknown indicator: {name}")
            except Exception as e:
                self.logger.error(f"Error calculating {name}: {e}")

        return results
    
    @staticmethod
    @handles_errors(fallback = pd.Series())
    def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index using VectorBT for optimization."""
        if VECTORBT_AVAILABLE and len(prices) >= 1000:  # Use VectorBT for large datasets
            try:
                # Use VectorBT native RSI calculation
                rsi_result = vbt.RSI.run(prices, window=window)
                return rsi_result.rsi
            except Exception as e:
                logger.warning(f"VectorBT RSI failed: {e}, using pandas fallback")
        
        # Fallback to pandas implementation
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window = window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window = window).mean()
        rs = gain / loss
        rsi = 100 - 100 / (1 + rs)
        return rsi

    @staticmethod
    @handles_errors(fallback = pd.Series())
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD using VectorBT for optimization."""
        if VECTORBT_AVAILABLE and len(prices) >= 1000:  # Use VectorBT for large datasets
            try:
                # Use VectorBT native MACD calculation
                macd_result = vbt.MACD.run(prices, fast=fast, slow=slow, signal=signal)
                return macd_result.macd
            except Exception as e:
                logger.warning(f"VectorBT MACD failed: {e}, using pandas fallback")
        
        # Fallback to pandas implementation
        ema_fast = prices.ewm(span = fast).mean()
        ema_slow = prices.ewm(span = slow).mean()
        macd = ema_fast - ema_slow
        return macd

    @staticmethod
    @handles_errors(fallback = pd.Series())
    def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range using VectorBT for optimization."""
        if VECTORBT_AVAILABLE and len(df) >= 1000:  # Use VectorBT for large datasets
            try:
                # Use VectorBT native ATR calculation
                atr_result = vbt.ATR.run(df['high'], df['low'], df['close'], window=window)
                return atr_result.atr
            except Exception as e:
                logger.warning(f"VectorBT ATR failed: {e}, using pandas fallback")
        
        # Fallback to pandas implementation
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis = 1).max(axis = 1)
        atr = tr.rolling(window = window).mean()
        return atr

    @staticmethod
    @handles_errors(fallback = pd.DataFrame())
    def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands using VectorBT for optimization."""
        if VECTORBT_AVAILABLE and len(prices) >= 1000:  # Use VectorBT for large datasets
            try:
                # Use VectorBT native Bollinger Bands calculation
                bb_result = vbt.BBANDS.run(prices, window=window, alpha=num_std)
                bb_features = pd.DataFrame({
                    'bb_upper': bb_result.upper, 
                    'bb_middle': bb_result.middle, 
                    'bb_lower': bb_result.lower, 
                    'bb_width': bb_result.width, 
                    'bb_position': bb_result.percent
                })
                return bb_features
            except Exception as e:
                logger.warning(f"VectorBT Bollinger Bands failed: {e}, using pandas fallback")
        
        # Fallback to pandas implementation
        sma = prices.rolling(window = window).mean()
        std = prices.rolling(window = window).std()
        bb_upper = sma + std * num_std
        bb_lower = sma - std * num_std
        bb_width = (bb_upper - bb_lower) / sma
        bb_position = (prices - bb_lower) / (bb_upper - bb_lower)
        bb_features = pd.DataFrame({
            'bb_upper': bb_upper, 
            'bb_middle': sma, 
            'bb_lower': bb_lower, 
            'bb_width': bb_width, 
            'bb_position': bb_position
        })
        return bb_features

    @staticmethod
    @handles_errors(fallback = pd.Series())
    def calculate_adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average Directional Index (ADX)."""
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis = 1).max(axis = 1)
        dm_plus = high - high.shift(1)
        dm_minus = low.shift(1) - low
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
        tr_smooth = tr.rolling(window = window).mean()
        dm_plus_smooth = dm_plus.rolling(window = window).mean()
        dm_minus_smooth = dm_minus.rolling(window = window).mean()
        di_plus = 100 * (dm_plus_smooth / tr_smooth)
        di_minus = 100 * (dm_minus_smooth / tr_smooth)
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window = window).mean()
        return adx


class VolatilityCalculator:
    """Handles volatility calculations and regime detection."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    @handles_errors(fallback = pd.Series())
    def calculate_parkinson_volatility(self, price_data: pd.DataFrame) -> pd.Series:
        """Calculate Parkinson volatility estimator using VectorBT optimization."""
        try:
            high_low_ratio = np.log(price_data["high"] / price_data["low"]) ** 2
            parkinson_vol = np.sqrt(high_low_ratio / (4 * np.log(2)))
            
            # Use VectorBT rolling mean for optimization
            if VECTORBT_AVAILABLE and len(parkinson_vol) >= 1000:
                try:
                    return rolling_mean(parkinson_vol, window=20)
                except Exception as e:
                    self.logger.debug(f"VectorBT rolling mean failed: {e}, using pandas fallback")
            
            return parkinson_vol.rolling(20).mean()
        except (ValueError, TypeError, IndexError) as e:
            self.logger.debug(f"Error calculating Parkinson volatility: {e}")
            return pd.Series()

    @handles_errors(fallback = pd.Series())
    def calculate_garman_klass_volatility(self, price_data: pd.DataFrame) -> pd.Series:
        """Calculate Garman-Klass volatility estimator using VectorBT optimization."""
        try:
            c = np.log(price_data["close"] / price_data["close"].shift(1))
            h = np.log(price_data["high"] / price_data["close"].shift(1))
            l = np.log(price_data["low"] / price_data["close"].shift(1))

            gk_vol = np.sqrt(0.5 * (h - l) ** 2 - (2 * np.log(2) - 1) * c**2)
            
            # Use VectorBT rolling mean for optimization
            if VECTORBT_AVAILABLE and len(gk_vol) >= 1000:
                try:
                    return rolling_mean(gk_vol, window=20)
                except Exception as e:
                    self.logger.debug(f"VectorBT rolling mean failed: {e}, using pandas fallback")
            
            return gk_vol.rolling(20).mean()
        except (ValueError, TypeError, IndexError) as e:
            self.logger.debug(f"Error calculating Garman-Klass volatility: {e}")
            return pd.Series()

    def calculate_volatility_regime(self, realized_vol: pd.Series) -> str:
        """Calculate volatility regime classification."""
        try:
            vol_percentile = realized_vol.rank(pct = True).iloc[-1]
            
            if vol_percentile > 0.8:
                return "high"
            elif vol_percentile < 0.2:
                return "low"
            else:
                return "medium"
        except Exception as e:
            self.logger.warning(f"Error calculating volatility regime: {e}")
            return "medium"


class MomentumCalculator:
    """Handles momentum calculations and analysis."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def calculate_momentum_features(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive momentum features using VectorBT optimization."""
        try:
            returns = price_data["close"].pct_change().dropna()
            
            # Use VectorBT rolling operations for optimization
            if VECTORBT_AVAILABLE and len(returns) >= 1000:
                try:
                    # Momentum indicators using VectorBT
                    momentum_5 = rolling_mean(returns, window=5)
                    momentum_20 = rolling_mean(returns, window=20)
                    momentum_50 = rolling_mean(returns, window=50)
                    
                    # Momentum acceleration
                    momentum_accel = momentum_5 - momentum_20
                    
                    # Momentum strength using VectorBT rolling std
                    momentum_20_std = rolling_std(returns, window=20)
                    momentum_strength = momentum_5 / momentum_20_std
                    
                except Exception as e:
                    self.logger.debug(f"VectorBT momentum calculations failed: {e}, using pandas fallback")
                    # Fallback to pandas
                    momentum_5 = returns.rolling(5).mean()
                    momentum_20 = returns.rolling(20).mean()
                    momentum_50 = returns.rolling(50).mean()
                    momentum_accel = momentum_5 - momentum_20
                    momentum_strength = momentum_5 / momentum_20.std()
            else:
                # Use pandas for smaller datasets
                momentum_5 = returns.rolling(5).mean()
                momentum_20 = returns.rolling(20).mean()
                momentum_50 = returns.rolling(50).mean()
                momentum_accel = momentum_5 - momentum_20
                momentum_strength = momentum_5 / momentum_20.std()
            
            # Momentum divergence
            price_momentum = price_data["close"].pct_change(5)
            volume_momentum = (
                price_data["volume"].pct_change(5)
                if "volume" in price_data.columns
                else pd.Series(0)
            )
            momentum_divergence = price_momentum - volume_momentum
            
            return {
                "momentum_5": momentum_5.iloc[-1] if not momentum_5.empty else 0.0,
                "momentum_20": momentum_20.iloc[-1] if not momentum_20.empty else 0.0,
                "momentum_50": momentum_50.iloc[-1] if not momentum_50.empty else 0.0,
                "momentum_acceleration": momentum_accel.iloc[-1] if not momentum_accel.empty else 0.0,
                "momentum_strength": momentum_strength.iloc[-1] if not momentum_strength.empty else 0.0,
                "momentum_divergence": momentum_divergence.iloc[-1] if not momentum_divergence.empty else 0.0,
            }
        except (KeyError, IndexError, ValueError) as e:
            self.logger.debug(f"Error calculating momentum features: {e}")
            return {}


class LiquidityCalculator:
    """Handles liquidity calculations and analysis."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def calculate_liquidity_features(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        order_flow_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """Calculate comprehensive liquidity features."""
        try:
            # Volume-based liquidity measures
            avg_volume = volume_data["volume"].rolling(20).mean()
            volume_liquidity = volume_data["volume"] / avg_volume
            
            # Price-based liquidity measures
            price_changes = price_data["close"].pct_change()
            price_impact = np.abs(price_changes) / volume_data["volume"]
            price_impact = price_impact.rolling(20).mean()
            
            # Spread-based liquidity (if order flow data available)
            spread_liquidity = 0.0
            if order_flow_data is not None and "spread" in order_flow_data.columns:
                spread_liquidity = order_flow_data["spread"].rolling(20).mean().iloc[-1]
            
            # Liquidity regime classification
            liquidity_percentile = volume_liquidity.rank(pct = True).iloc[-1]
            
            if liquidity_percentile > 0.8:
                liquidity_regime = "high"
            elif liquidity_percentile < 0.2:
                liquidity_regime = "low"
            else:
                liquidity_regime = "medium"
            
            return {
                "volume_liquidity": volume_liquidity.iloc[-1] if not volume_liquidity.empty else 1.0,
                "price_impact": price_impact.iloc[-1] if not price_impact.empty else 0.0,
                "spread_liquidity": spread_liquidity,
                "liquidity_regime": liquidity_regime,
                "liquidity_percentile": liquidity_percentile,
            }
        except (KeyError, IndexError, ValueError) as e:
            self.logger.debug(f"Error calculating liquidity features: {e}")
            return {}


class CorrelationCalculator:
    """Handles correlation calculations and analysis."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def calculate_correlation_features(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate correlation features using VectorBT optimization."""
        try:
            returns = price_data["close"].pct_change().dropna()
            
            # Use VectorBT rolling correlation for optimization
            if VECTORBT_AVAILABLE and len(returns) >= 1000:
                try:
                    # Rolling correlations using VectorBT
                    corr_5 = rolling_corr(returns, returns.shift(1), window=5)
                    corr_20 = rolling_corr(returns, returns.shift(1), window=20)
                except Exception as e:
                    self.logger.debug(f"VectorBT correlation calculations failed: {e}, using pandas fallback")
                    # Fallback to pandas
                    corr_5 = returns.rolling(5).corr(returns.shift(1))
                    corr_20 = returns.rolling(20).corr(returns.shift(1))
            else:
                # Use pandas for smaller datasets
                corr_5 = returns.rolling(5).corr(returns.shift(1))
                corr_20 = returns.rolling(20).corr(returns.shift(1))
            
            # Cross-timeframe correlations
            returns_5m = returns.resample("5T").last()
            returns_1h = returns.resample("1H").last()
            
            cross_corr = (
                returns_5m.corr(returns_1h)
                if len(returns_5m) > 1 and len(returns_1h) > 1
                else 0.0
            )
            
            return {
                "autocorrelation_5": corr_5.iloc[-1] if not corr_5.empty else 0.0,
                "autocorrelation_20": corr_20.iloc[-1] if not corr_20.empty else 0.0,
                "cross_timeframe_correlation": cross_corr,
            }
        except (KeyError, IndexError, ValueError) as e:
            self.logger.debug(f"Error calculating correlation features: {e}")
            return {}


class MicrostructureCalculator:
    """Handles market microstructure calculations."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def calculate_price_impact(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
    ) -> Dict[str, float]:
        """Calculate price impact metrics using VectorBT optimization."""
        try:
            # Calculate price changes
            price_changes = price_data["close"].pct_change()
            
            # Use VectorBT rolling operations for optimization
            if VECTORBT_AVAILABLE and len(price_changes) >= 1000:
                try:
                    # Calculate volume-weighted price impact using VectorBT
                    volume_weighted_impact = rolling_mean(
                        price_changes * volume_data["volume"], window=20
                    )
                    
                    # Calculate Kyle's lambda using VectorBT
                    kyle_lambda = (
                        rolling_mean(np.abs(price_changes), window=50) /
                        rolling_mean(volume_data["volume"], window=50)
                    )
                    
                    # Calculate Amihud illiquidity measure using VectorBT
                    amihud_illiquidity = rolling_mean(
                        np.abs(price_changes) / volume_data["volume"], window=20
                    )
                    
                except Exception as e:
                    self.logger.debug(f"VectorBT microstructure calculations failed: {e}, using pandas fallback")
                    # Fallback to pandas
                    volume_weighted_impact = (
                        (price_changes * volume_data["volume"]).rolling(20).mean()
                    )
                    kyle_lambda = (
                        np.abs(price_changes).rolling(50).mean()
                        / volume_data["volume"].rolling(50).mean()
                    )
                    amihud_illiquidity = np.abs(price_changes) / volume_data["volume"]
                    amihud_illiquidity = amihud_illiquidity.rolling(20).mean()
            else:
                # Use pandas for smaller datasets
                volume_weighted_impact = (
                    (price_changes * volume_data["volume"]).rolling(20).mean()
                )
                kyle_lambda = (
                    np.abs(price_changes).rolling(50).mean()
                    / volume_data["volume"].rolling(50).mean()
                )
                amihud_illiquidity = np.abs(price_changes) / volume_data["volume"]
                amihud_illiquidity = amihud_illiquidity.rolling(20).mean()
            
            return {
                "price_impact": volume_weighted_impact.iloc[-1]
                if not volume_weighted_impact.empty
                else 0.0,
                "kyle_lambda": kyle_lambda.iloc[-1] if not kyle_lambda.empty else 0.0,
                "amihud_illiquidity": amihud_illiquidity.iloc[-1]
                if not amihud_illiquidity.empty
                else 0.0,
            }
        except (KeyError, IndexError, ValueError) as e:
            self.logger.debug(f"Error calculating price impact: {e}")
            return {}

    def calculate_order_flow_imbalance(
        self,
        order_flow_data: pd.DataFrame,
    ) -> Dict[str, float]:
        """Calculate order flow imbalance metrics."""
        try:
            # Calculate buy/sell pressure
            buy_volume = order_flow_data.get("buy_volume", pd.Series(0))
            sell_volume = order_flow_data.get("sell_volume", pd.Series(0))
            
            # Order flow imbalance
            total_volume = buy_volume + sell_volume
            imbalance = (buy_volume - sell_volume) / total_volume
            imbalance = imbalance.rolling(20).mean()
            
            # Large order detection
            avg_volume = total_volume.rolling(50).mean()
            large_order_ratio = (total_volume > 2 * avg_volume).rolling(20).mean()
            
            return {
                "order_flow_imbalance": imbalance.iloc[-1]
                if not imbalance.empty
                else 0.0,
                "large_order_ratio": large_order_ratio.iloc[-1]
                if not large_order_ratio.empty
                else 0.0,
            }
        except (KeyError, IndexError, ValueError) as e:
            self.logger.debug(f"Error calculating order flow imbalance: {e}")
            return {}

    def calculate_volume_profile(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
    ) -> Dict[str, float]:
        """Calculate volume profile metrics."""
        try:
            # Volume-weighted average price (VWAP)
            vwap = (price_data["close"] * volume_data["volume"]).rolling(
                20,
            ).sum() / volume_data["volume"].rolling(20).sum()
            
            # Volume price trend (VPT)
            vpt = (volume_data["volume"] * price_data["close"].pct_change()).cumsum()
            
            # Volume rate of change
            volume_roc = volume_data["volume"].pct_change(5)
            
            # Volume moving average ratio
            volume_ma_ratio = (
                volume_data["volume"] / volume_data["volume"].rolling(20).mean()
            )
            
            return {
                "vwap": vwap.iloc[-1]
                if not vwap.empty
                else price_data["close"].iloc[-1],
                "vpt": vpt.iloc[-1] if not vpt.empty else 0.0,
                "volume_roc": volume_roc.iloc[-1] if not volume_roc.empty else 0.0,
                "volume_ma_ratio": volume_ma_ratio.iloc[-1]
                if not volume_ma_ratio.empty
                else 1.0,
            }
        except (KeyError, IndexError, ValueError) as e:
            self.logger.debug(f"Error calculating volume profile: {e}")
            return {}


class AdaptiveIndicatorCalculator:
    """Handles adaptive technical indicator calculations."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def calculate_adaptive_moving_averages(
        self,
        price_data: pd.DataFrame,
    ) -> Dict[str, float]:
        """Calculate adaptive moving averages based on volatility."""
        try:
            # Calculate volatility
            returns = price_data["close"].pct_change()
            volatility = returns.rolling(20).std()
            
            # Adaptive periods based on volatility
            base_period = 20
            volatility_factor = volatility / volatility.rolling(100).mean()
            adaptive_period = (base_period * volatility_factor).clip(5, 50)
            
            # Adaptive SMA
            adaptive_sma = (
                price_data["close"].rolling(window = adaptive_period.astype(int)).mean()
            )
            
            # Adaptive EMA
            adaptive_alpha = 2 / (adaptive_period + 1)
            adaptive_ema = price_data["close"].ewm(alpha = adaptive_alpha).mean()
            
            return {
                "adaptive_sma": adaptive_sma.iloc[-1]
                if not adaptive_sma.empty
                else price_data["close"].iloc[-1],
                "adaptive_ema": adaptive_ema.iloc[-1]
                if not adaptive_ema.empty
                else price_data["close"].iloc[-1],
                "adaptive_period": adaptive_period.iloc[-1]
                if not adaptive_period.empty
                else base_period,
            }
        except (KeyError, IndexError, ValueError) as e:
            self.logger.debug(f"Error calculating adaptive moving averages: {e}")
            return {}


def initialization_error(message: str) -> str:
    """Format initialization error message."""
    return f"❌ {message}"


def print_message(message: str) -> None:
    """Print message with proper formatting."""
    tprint(message)


class VectorBTOptimizedFeatureCalculator:
    """
    VectorBT-optimized feature calculator that provides high-performance
    feature generation using VectorBT's native implementations.
    """
    
    def __init__(self, enable_gpu: bool = False, enable_parallel: bool = True):
        """Initialize VectorBT optimized calculator."""
        self.enable_gpu = enable_gpu
        self.enable_parallel = enable_parallel
        self.logger = system_logger.getChild('VectorBTOptimizedFeatureCalculator')
        
        # Performance tracking
        self.stats = {
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'gpu_accelerations': 0,
            'total_operations': 0
        }
    
    def calculate_batch_technical_indicators(self, data: pd.DataFrame, 
                                           indicators: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Calculate multiple technical indicators in batch using VectorBT.
        
        Args:
            data: OHLCV data
            indicators: List of indicator configurations
            
        Returns:
            DataFrame with calculated indicators
        """
        if not VECTORBT_AVAILABLE:
            self.logger.warning("VectorBT not available, using pandas fallback")
            return self._pandas_batch_indicators(data, indicators)
        
        results = {}
        
        try:
            for indicator_config in indicators:
                name = indicator_config['name']
                indicator_type = indicator_config['type']
                params = indicator_config.get('params', {})
                
                if indicator_type == 'rsi':
                    result = vbt.RSI.run(data['close'], **params).rsi
                elif indicator_type == 'macd':
                    macd_result = vbt.MACD.run(data['close'], **params)
                    result = macd_result.macd
                elif indicator_type == 'atr':
                    result = vbt.ATR.run(data['high'], data['low'], data['close'], **params).atr
                elif indicator_type == 'bbands_upper':
                    bb_result = vbt.BBANDS.run(data['close'], **params)
                    result = bb_result.upper
                elif indicator_type == 'bbands_lower':
                    bb_result = vbt.BBANDS.run(data['close'], **params)
                    result = bb_result.lower
                elif indicator_type == 'stoch_k':
                    stoch_result = vbt.STOCH.run(data['high'], data['low'], data['close'], **params)
                    result = stoch_result.stoch_k
                elif indicator_type == 'willr':
                    result = vbt.WILLR.run(data['high'], data['low'], data['close'], **params).willr
                elif indicator_type == 'cci':
                    result = vbt.CCI.run(data['high'], data['low'], data['close'], **params).cci
                elif indicator_type == 'mfi':
                    result = vbt.MFI.run(data['high'], data['low'], data['close'], data['volume'], **params).mfi
                elif indicator_type == 'adx':
                    result = vbt.ADX.run(data['high'], data['low'], data['close'], **params).adx
                elif indicator_type == 'roc':
                    result = vbt.ROC.run(data['close'], **params).roc
                elif indicator_type == 'mom':
                    result = vbt.MOM.run(data['close'], **params).mom
                elif indicator_type == 'obv':
                    result = vbt.OBV.run(data['close'], data['volume'], **params).obv
                else:
                    self.logger.warning(f"Unknown indicator type: {indicator_type}")
                    continue
                
                results[name] = result
                self.stats['vectorbt_operations'] += 1
                
        except Exception as e:
            self.logger.warning(f"VectorBT batch indicators failed: {e}, using pandas fallback")
            return self._pandas_batch_indicators(data, indicators)
        
        self.stats['total_operations'] += len(indicators)
        return pd.DataFrame(results, index=data.index)
    
    def calculate_batch_rolling_features(self, data: pd.DataFrame, 
                                       features: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Calculate multiple rolling features in batch using VectorBT.
        
        Args:
            data: Input data
            features: List of rolling feature configurations
            
        Returns:
            DataFrame with calculated features
        """
        if not VECTORBT_AVAILABLE:
            return self._pandas_batch_rolling(data, features)
        
        results = {}
        
        try:
            for feature_config in features:
                name = feature_config['name']
                column = feature_config.get('column', 'close')
                operation = feature_config.get('operation', 'mean')
                window = feature_config.get('window', 20)
                
                if column not in data.columns:
                    self.logger.warning(f"Column {column} not found for feature {name}")
                    continue
                
                if operation == 'mean':
                    result = rolling_mean(data[column], window=window)
                elif operation == 'std':
                    result = rolling_std(data[column], window=window)
                elif operation == 'var':
                    result = rolling_var(data[column], window=window)
                elif operation == 'min':
                    result = rolling_min(data[column], window=window)
                elif operation == 'max':
                    result = rolling_max(data[column], window=window)
                elif operation == 'sum':
                    result = rolling_sum(data[column], window=window)
                else:
                    self.logger.warning(f"Unknown rolling operation: {operation}")
                    continue
                
                results[name] = result
                self.stats['vectorbt_operations'] += 1
                
        except Exception as e:
            self.logger.warning(f"VectorBT batch rolling failed: {e}, using pandas fallback")
            return self._pandas_batch_rolling(data, features)
        
        self.stats['total_operations'] += len(features)
        return pd.DataFrame(results, index=data.index)
    
    def calculate_batch_scaling_features(self, data: pd.DataFrame, 
                                       features: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Calculate multiple scaling features in batch using VectorBT.
        
        Args:
            data: Input data
            features: List of scaling feature configurations
            
        Returns:
            DataFrame with calculated features
        """
        if not VECTORBT_AVAILABLE:
            return self._pandas_batch_scaling(data, features)
        
        results = {}
        
        try:
            for feature_config in features:
                name = feature_config['name']
                column = feature_config.get('column', 'close')
                method = feature_config.get('method', 'zscore')
                
                if column not in data.columns:
                    self.logger.warning(f"Column {column} not found for feature {name}")
                    continue
                
                if method == 'zscore':
                    result = zscore(data[column])
                elif method == 'minmax':
                    result = scale(data[column], method='minmax')
                elif method == 'robust':
                    result = scale(data[column], method='robust')
                elif method == 'quantile':
                    result = quantile(data[column])
                elif method == 'winsorize':
                    result = winsorize(data[column])
                elif method == 'rank':
                    result = rank(data[column])
                elif method == 'clip':
                    result = clip(data[column])
                else:
                    self.logger.warning(f"Unknown scaling method: {method}")
                    continue
                
                results[name] = result
                self.stats['vectorbt_operations'] += 1
                
        except Exception as e:
            self.logger.warning(f"VectorBT batch scaling failed: {e}, using pandas fallback")
            return self._pandas_batch_scaling(data, features)
        
        self.stats['total_operations'] += len(features)
        return pd.DataFrame(results, index=data.index)
    
    def _pandas_batch_indicators(self, data: pd.DataFrame, indicators: List[Dict[str, Any]]) -> pd.DataFrame:
        """Fallback pandas implementation for batch indicators."""
        results = {}
        for indicator_config in indicators:
            name = indicator_config['name']
            indicator_type = indicator_config['type']
            params = indicator_config.get('params', {})
            
            try:
                if indicator_type == 'rsi':
                    # Simple RSI calculation
                    delta = data['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=params.get('window', 14)).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=params.get('window', 14)).mean()
                    rs = gain / loss
                    result = 100 - (100 / (1 + rs))
                elif indicator_type == 'macd':
                    # Simple MACD calculation
                    ema_fast = data['close'].ewm(span=params.get('fast_window', 12)).mean()
                    ema_slow = data['close'].ewm(span=params.get('slow_window', 26)).mean()
                    result = ema_fast - ema_slow
                else:
                    # For other indicators, return NaN series
                    result = pd.Series(np.nan, index=data.index)
                
                results[name] = result
                self.stats['pandas_fallbacks'] += 1
                
            except Exception as e:
                self.logger.warning(f"Pandas indicator {indicator_type} failed: {e}")
                results[name] = pd.Series(np.nan, index=data.index)
        
        return pd.DataFrame(results, index=data.index)
    
    def _pandas_batch_rolling(self, data: pd.DataFrame, features: List[Dict[str, Any]]) -> pd.DataFrame:
        """Fallback pandas implementation for batch rolling."""
        results = {}
        for feature_config in features:
            name = feature_config['name']
            column = feature_config.get('column', 'close')
            operation = feature_config.get('operation', 'mean')
            window = feature_config.get('window', 20)
            
            if column not in data.columns:
                continue
            
            try:
                if operation == 'mean':
                    result = data[column].rolling(window=window).mean()
                elif operation == 'std':
                    result = data[column].rolling(window=window).std()
                elif operation == 'var':
                    result = data[column].rolling(window=window).var()
                elif operation == 'min':
                    result = data[column].rolling(window=window).min()
                elif operation == 'max':
                    result = data[column].rolling(window=window).max()
                elif operation == 'sum':
                    result = data[column].rolling(window=window).sum()
                else:
                    continue
                
                results[name] = result
                self.stats['pandas_fallbacks'] += 1
                
            except Exception as e:
                self.logger.warning(f"Pandas rolling {operation} failed: {e}")
                results[name] = pd.Series(np.nan, index=data.index)
        
        return pd.DataFrame(results, index=data.index)
    
    def _pandas_batch_scaling(self, data: pd.DataFrame, features: List[Dict[str, Any]]) -> pd.DataFrame:
        """Fallback pandas implementation for batch scaling."""
        results = {}
        for feature_config in features:
            name = feature_config['name']
            column = feature_config.get('column', 'close')
            method = feature_config.get('method', 'zscore')
            
            if column not in data.columns:
                continue
            
            try:
                if method == 'zscore':
                    result = (data[column] - data[column].mean()) / data[column].std()
                elif method == 'minmax':
                    result = (data[column] - data[column].min()) / (data[column].max() - data[column].min())
                elif method == 'robust':
                    median = data[column].median()
                    mad = (data[column] - median).abs().median()
                    result = (data[column] - median) / mad
                else:
                    continue
                
                results[name] = result
                self.stats['pandas_fallbacks'] += 1
                
            except Exception as e:
                self.logger.warning(f"Pandas scaling {method} failed: {e}")
                results[name] = pd.Series(np.nan, index=data.index)
        
        return pd.DataFrame(results, index=data.index)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.stats.copy()
        if stats['total_operations'] > 0:
            stats['vectorbt_usage_percentage'] = (
                stats['vectorbt_operations'] / stats['total_operations'] * 100
            )
            stats['pandas_fallback_percentage'] = (
                stats['pandas_fallbacks'] / stats['total_operations'] * 100
            )
        else:
            stats['vectorbt_usage_percentage'] = 0
            stats['pandas_fallback_percentage'] = 0
        return stats

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
