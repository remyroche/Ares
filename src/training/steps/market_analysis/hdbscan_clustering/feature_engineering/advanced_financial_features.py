"""
Advanced Financial Feature Engineering for Data-Driven Clustering

This module provides comprehensive feature engineering for financial data,
including risk dimensions, distributional features, volume analysis, and
volatility-aware features for regime discovery.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from scipy import stats
from scipy.signal import find_peaks
import warnings

# Import tprint utilities for extensive logging
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, 
    tprint_debug, tprint_performance, tprint_progress, tprint_timer,
    tprint_logged, LogLevel
)

logger = logging.getLogger(__name__)

@dataclass
class AdvancedFeatureConfig:
    """Configuration for advanced financial feature engineering."""
    # Risk and distributional features
    enable_skewness_features: bool = True
    enable_kurtosis_features: bool = True
    enable_var_features: bool = True
    enable_cvar_features: bool = True
    enable_drawdown_features: bool = True
    
    # Volatility features
    enable_volatility_regimes: bool = True
    enable_volatility_scaling: bool = True
    enable_garch_features: bool = False  # Requires arch library
    
    # Volume features
    enable_volume_features: bool = True
    enable_volume_momentum: bool = True
    enable_volume_volatility: bool = True
    enable_volume_price_correlation: bool = True
    
    # Technical indicators
    enable_technical_indicators: bool = True
    enable_momentum_indicators: bool = True
    enable_volatility_indicators: bool = True
    
    # Feature windows
    short_window: int = 5
    medium_window: int = 20
    long_window: int = 50
    
    # Risk parameters
    var_levels: List[float] = None
    cvar_levels: List[float] = None
    
    # Volatility parameters
    volatility_window: int = 20
    volatility_threshold: float = 0.02
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.var_levels is None:
            self.var_levels = [0.01, 0.05, 0.10]
        if self.cvar_levels is None:
            self.cvar_levels = [0.01, 0.05, 0.10]

class AdvancedFinancialFeatureEngineer:
    """
    Advanced financial feature engineer for clustering and regime discovery.
    
    Creates comprehensive feature sets including risk dimensions, distributional
    features, volume analysis, and volatility-aware features.
    """
    
    @tprint_logged(LogLevel.INFO, include_args=True)
    def __init__(self, config: Optional[AdvancedFeatureConfig] = None):
        """Initialize advanced feature engineer."""
        tprint_info("🔧 Initializing AdvancedFinancialFeatureEngineer")
        start_time = time.perf_counter()
        
        self.config = config or AdvancedFeatureConfig()
        self.feature_names = []
        self.feature_categories = {}
        
        init_time = time.perf_counter() - start_time
        tprint_success(f"✅ AdvancedFinancialFeatureEngineer initialized in {init_time:.3f}s")
        
    @tprint_logged(LogLevel.INFO, include_args=True, include_result=True)
    def engineer_features(self, market_data: pd.DataFrame) -> Tuple[np.ndarray, List[str], Dict[str, List[str]]]:
        """
        Engineer comprehensive financial features.
        
        Args:
            market_data: Market data with OHLCV columns
            
        Returns:
            Tuple of (features_array, feature_names, feature_categories)
        """
        try:
            tprint_info("🔧 Starting advanced financial feature engineering...")
            tprint_debug(f"Market data shape: {market_data.shape}")
            
            # Validate input data
            with tprint_timer("Market data validation"):
                market_data = self._validate_market_data(market_data)
                tprint_debug(f"Validated market data shape: {market_data.shape}")
            
            # Initialize feature storage
            features = []
            feature_names = []
            feature_categories = {
                'returns': [],
                'volatility': [],
                'volume': [],
                'risk': [],
                'distributional': [],
                'technical': [],
                'momentum': []
            }
            
            # Calculate base returns
            with tprint_timer("Base returns calculation"):
                returns = self._calculate_returns(market_data)
                tprint_debug(f"Calculated returns: {len(returns)} samples")
            
            # 1. Risk and Distributional Features
            if (self.config.enable_skewness_features or 
                self.config.enable_kurtosis_features or 
                self.config.enable_var_features or 
                self.config.enable_cvar_features or 
                self.config.enable_drawdown_features):
                
                tprint_info("📊 Engineering risk and distributional features...")
                with tprint_timer("Risk features engineering"):
                    risk_features, risk_names, risk_categories = self._engineer_risk_features(
                        market_data, returns
                    )
                    features.extend(risk_features)
                    feature_names.extend(risk_names)
                    feature_categories.update(risk_categories)
                    tprint_debug(f"Risk features: {len(risk_features)} features")
            
            # 2. Volatility Features
            if (self.config.enable_volatility_regimes or 
                self.config.enable_volatility_scaling or 
                self.config.enable_garch_features):
                
                tprint_info("📈 Engineering volatility features...")
                with tprint_timer("Volatility features engineering"):
                    vol_features, vol_names, vol_categories = self._engineer_volatility_features(
                        market_data, returns
                    )
                    features.extend(vol_features)
                    feature_names.extend(vol_names)
                    feature_categories.update(vol_categories)
                    tprint_debug(f"Volatility features: {len(vol_features)} features")
            
            # 3. Volume Features
            if (self.config.enable_volume_features or 
                self.config.enable_volume_momentum or 
                self.config.enable_volume_volatility or 
                self.config.enable_volume_price_correlation):
                
                tprint_info("📊 Engineering volume features...")
                with tprint_timer("Volume features engineering"):
                    volume_features, volume_names, volume_categories = self._engineer_volume_features(
                        market_data, returns
                    )
                    features.extend(volume_features)
                    feature_names.extend(volume_names)
                    feature_categories.update(volume_categories)
                    tprint_debug(f"Volume features: {len(volume_features)} features")
            
            # 4. Technical Indicators
            if (self.config.enable_technical_indicators or 
                self.config.enable_momentum_indicators or 
                self.config.enable_volatility_indicators):
                
                tprint_info("🔧 Engineering technical indicator features...")
                with tprint_timer("Technical features engineering"):
                    tech_features, tech_names, tech_categories = self._engineer_technical_features(
                        market_data, returns
                    )
                    features.extend(tech_features)
                    feature_names.extend(tech_names)
                    feature_categories.update(tech_categories)
                    tprint_debug(f"Technical features: {len(tech_features)} features")
            
            # Convert to numpy array
            with tprint_timer("Feature array conversion"):
                features_array = np.column_stack(features) if features else np.array([]).reshape(len(market_data), 0)
                tprint_debug(f"Features array shape: {features_array.shape}")
            
            # Store feature information
            self.feature_names = feature_names
            self.feature_categories = feature_categories
            
            tprint_success(f"✅ Advanced feature engineering completed: {features_array.shape[1]} features")
            tprint_info(f"📊 Feature categories: {list(feature_categories.keys())}")
            
            return features_array, feature_names, feature_categories
            
        except Exception as e:
            tprint_error(f"❌ Advanced feature engineering failed: {e}")
            raise
    
    @tprint_logged(LogLevel.DEBUG, include_args=True, include_result=True)
    def _validate_market_data(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean market data."""
        try:
            tprint_debug(f"🔍 Validating market data: {market_data.shape}")
            
            # Check required columns
            required_columns = ['close']
            missing_columns = [col for col in required_columns if col not in market_data.columns]
            if missing_columns:
                tprint_error(f"Missing required columns: {missing_columns}")
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            tprint_debug("✅ Required columns validation passed")
            
            # Ensure numeric data
            for col in ['close', 'open', 'high', 'low', 'volume']:
                if col in market_data.columns:
                    market_data[col] = pd.to_numeric(market_data[col], errors='coerce')
                    tprint_debug(f"Converted {col} to numeric")
            
            # Remove rows with NaN values in critical columns
            initial_rows = len(market_data)
            market_data = market_data.dropna(subset=['close'])
            dropped_rows = initial_rows - len(market_data)
            if dropped_rows > 0:
                tprint_debug(f"Dropped {dropped_rows} rows with NaN close prices")
            
            # Ensure positive prices
            initial_rows = len(market_data)
            market_data = market_data[market_data['close'] > 0]
            dropped_rows = initial_rows - len(market_data)
            if dropped_rows > 0:
                tprint_debug(f"Dropped {dropped_rows} rows with non-positive close prices")
            
            if 'volume' in market_data.columns:
                initial_rows = len(market_data)
                market_data = market_data[market_data['volume'] >= 0]
                dropped_rows = initial_rows - len(market_data)
                if dropped_rows > 0:
                    tprint_debug(f"Dropped {dropped_rows} rows with negative volume")
            
            tprint_success(f"✅ Market data validation completed: {market_data.shape}")
            return market_data
            
        except Exception as e:
            tprint_error(f"Market data validation failed: {e}")
            raise
    
    @tprint_logged(LogLevel.DEBUG, include_args=True, include_result=True)
    def _calculate_returns(self, market_data: pd.DataFrame) -> pd.Series:
        """Calculate returns from market data."""
        try:
            tprint_debug("📊 Calculating returns from market data")
            
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change().dropna()
                tprint_debug(f"Calculated returns: {len(returns)} samples, range: [{returns.min():.4f}, {returns.max():.4f}]")
                return returns
            else:
                tprint_error("No close price data available")
                raise ValueError("No close price data available")
        except Exception as e:
            tprint_error(f"Returns calculation failed: {e}")
            raise
    
    def _engineer_risk_features(self, market_data: pd.DataFrame, returns: pd.Series) -> Tuple[List[np.ndarray], List[str], Dict[str, List[str]]]:
        """Engineer risk and distributional features."""
        try:
            features = []
            feature_names = []
            feature_categories = {
                'risk': [],
                'distributional': []
            }
            
            # Skewness features
            if self.config.enable_skewness_features:
                for window in [self.config.short_window, self.config.medium_window, self.config.long_window]:
                    if window <= len(returns):
                        skewness = returns.rolling(window).skew()
                        features.append(skewness.fillna(0).values)
                        feature_names.append(f'skewness_{window}')
                        feature_categories['distributional'].append(f'skewness_{window}')
            
            # Kurtosis features
            if self.config.enable_kurtosis_features:
                for window in [self.config.short_window, self.config.medium_window, self.config.long_window]:
                    if window <= len(returns):
                        kurtosis = returns.rolling(window).kurt()
                        features.append(kurtosis.fillna(0).values)
                        feature_names.append(f'kurtosis_{window}')
                        feature_categories['distributional'].append(f'kurtosis_{window}')
            
            # VaR features
            if self.config.enable_var_features:
                for var_level in self.config.var_levels:
                    for window in [self.config.short_window, self.config.medium_window]:
                        if window <= len(returns):
                            var_values = returns.rolling(window).quantile(var_level)
                            features.append(var_values.fillna(0).values)
                            feature_names.append(f'var_{var_level}_{window}')
                            feature_categories['risk'].append(f'var_{var_level}_{window}')
            
            # CVaR features
            if self.config.enable_cvar_features:
                for cvar_level in self.config.cvar_levels:
                    for window in [self.config.short_window, self.config.medium_window]:
                        if window <= len(returns):
                            cvar_values = returns.rolling(window).apply(
                                lambda x: self._calculate_cvar(x, cvar_level), raw=False
                            )
                            features.append(cvar_values.fillna(0).values)
                            feature_names.append(f'cvar_{cvar_level}_{window}')
                            feature_categories['risk'].append(f'cvar_{cvar_level}_{window}')
            
            # Drawdown features
            if self.config.enable_drawdown_features:
                for window in [self.config.medium_window, self.config.long_window]:
                    if window <= len(market_data):
                        drawdowns = self._calculate_drawdowns(market_data['close'], window)
                        features.append(drawdowns.fillna(0).values)
                        feature_names.append(f'drawdown_{window}')
                        feature_categories['risk'].append(f'drawdown_{window}')
                        
                        # Maximum drawdown
                        max_drawdowns = drawdowns.rolling(window).min()
                        features.append(max_drawdowns.fillna(0).values)
                        feature_names.append(f'max_drawdown_{window}')
                        feature_categories['risk'].append(f'max_drawdown_{window}')
            
            return features, feature_names, feature_categories
            
        except Exception as e:
            logger.warning(f"Risk features engineering failed: {e}")
            return [], [], {'risk': [], 'distributional': []}
    
    def _calculate_cvar(self, returns: pd.Series, level: float) -> float:
        """Calculate Conditional Value at Risk (CVaR)."""
        try:
            if len(returns) < 5:
                return 0.0
            
            var_value = returns.quantile(level)
            cvar_value = returns[returns <= var_value].mean()
            return cvar_value if not pd.isna(cvar_value) else 0.0
        except Exception:
            return 0.0
    
    def _calculate_drawdowns(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate rolling drawdowns."""
        try:
            rolling_max = prices.rolling(window).max()
            drawdowns = (prices - rolling_max) / rolling_max
            return drawdowns
        except Exception:
            return pd.Series(0, index=prices.index)
    
    def _engineer_volatility_features(self, market_data: pd.DataFrame, returns: pd.Series) -> Tuple[List[np.ndarray], List[str], Dict[str, List[str]]]:
        """Engineer volatility-aware features."""
        try:
            features = []
            feature_names = []
            feature_categories = {
                'volatility': []
            }
            
            # Basic volatility features
            for window in [self.config.short_window, self.config.medium_window, self.config.long_window]:
                if window <= len(returns):
                    # Rolling standard deviation
                    volatility = returns.rolling(window).std()
                    features.append(volatility.fillna(0).values)
                    feature_names.append(f'volatility_{window}')
                    feature_categories['volatility'].append(f'volatility_{window}')
                    
                    # Rolling variance
                    variance = returns.rolling(window).var()
                    features.append(variance.fillna(0).values)
                    feature_names.append(f'variance_{window}')
                    feature_categories['volatility'].append(f'variance_{window}')
            
            # Volatility regimes
            if self.config.enable_volatility_regimes:
                vol_regime = self._calculate_volatility_regimes(returns)
                features.append(vol_regime.values)
                feature_names.append('volatility_regime')
                feature_categories['volatility'].append('volatility_regime')
            
            # Volatility scaling
            if self.config.enable_volatility_scaling:
                vol_scaled_returns = self._calculate_volatility_scaled_returns(returns)
                features.append(vol_scaled_returns.values)
                feature_names.append('volatility_scaled_returns')
                feature_categories['volatility'].append('volatility_scaled_returns')
            
            # GARCH features (if enabled)
            if self.config.enable_garch_features:
                try:
                    garch_features = self._calculate_garch_features(returns)
                    features.extend(garch_features)
                    feature_names.extend(['garch_volatility', 'garch_residuals'])
                    feature_categories['volatility'].extend(['garch_volatility', 'garch_residuals'])
                except ImportError:
                    logger.warning("GARCH features require arch library")
                except Exception as e:
                    logger.warning(f"GARCH features failed: {e}")
            
            return features, feature_names, feature_categories
            
        except Exception as e:
            logger.warning(f"Volatility features engineering failed: {e}")
            return [], [], {'volatility': []}
    
    def _calculate_volatility_regimes(self, returns: pd.Series) -> pd.Series:
        """Calculate volatility regimes (high/medium/low)."""
        try:
            # Calculate rolling volatility
            vol_window = self.config.volatility_window
            volatility = returns.rolling(vol_window).std()
            
            # Calculate percentiles
            vol_33 = volatility.quantile(0.33)
            vol_67 = volatility.quantile(0.67)
            
            # Assign regimes
            regimes = pd.Series(1, index=volatility.index)  # Medium by default
            regimes[volatility <= vol_33] = 0  # Low volatility
            regimes[volatility >= vol_67] = 2  # High volatility
            
            return regimes.fillna(1)
        except Exception:
            return pd.Series(1, index=returns.index)
    
    def _calculate_volatility_scaled_returns(self, returns: pd.Series) -> pd.Series:
        """Calculate volatility-scaled returns."""
        try:
            vol_window = self.config.volatility_window
            volatility = returns.rolling(vol_window).std()
            scaled_returns = returns / (volatility + 1e-10)
            return scaled_returns.fillna(0)
        except Exception:
            return pd.Series(0, index=returns.index)
    
    def _calculate_garch_features(self, returns: pd.Series) -> List[np.ndarray]:
        """Calculate GARCH features (requires arch library)."""
        try:
            from arch import arch_model
            
            # Fit GARCH(1,1) model
            model = arch_model(returns * 100, vol='Garch', p=1, q=1)
            fitted_model = model.fit(disp='off')
            
            # Get volatility forecasts
            volatility_forecast = fitted_model.conditional_volatility / 100
            residuals = fitted_model.resid / 100
            
            return [volatility_forecast.values, residuals.values]
        except ImportError:
            raise ImportError("GARCH features require arch library")
        except Exception as e:
            logger.warning(f"GARCH model fitting failed: {e}")
            return [np.zeros(len(returns)), np.zeros(len(returns))]
    
    def _engineer_volume_features(self, market_data: pd.DataFrame, returns: pd.Series) -> Tuple[List[np.ndarray], List[str], Dict[str, List[str]]]:
        """Engineer volume-based features."""
        try:
            features = []
            feature_names = []
            feature_categories = {
                'volume': []
            }
            
            if 'volume' not in market_data.columns:
                logger.warning("No volume data available for volume features")
                return [], [], {'volume': []}
            
            volume = market_data['volume']
            
            # Basic volume features
            for window in [self.config.short_window, self.config.medium_window, self.config.long_window]:
                if window <= len(volume):
                    # Volume moving averages
                    vol_ma = volume.rolling(window).mean()
                    features.append(vol_ma.fillna(0).values)
                    feature_names.append(f'volume_ma_{window}')
                    feature_categories['volume'].append(f'volume_ma_{window}')
                    
                    # Volume standard deviation
                    vol_std = volume.rolling(window).std()
                    features.append(vol_std.fillna(0).values)
                    feature_names.append(f'volume_std_{window}')
                    feature_categories['volume'].append(f'volume_std_{window}')
            
            # Relative volume (RVOL)
            for window in [self.config.medium_window, self.config.long_window]:
                if window <= len(volume):
                    vol_ma = volume.rolling(window).mean()
                    rvol = volume / (vol_ma + 1e-10)
                    features.append(rvol.fillna(1).values)
                    feature_names.append(f'rvol_{window}')
                    feature_categories['volume'].append(f'rvol_{window}')
            
            # Volume Z-score
            if self.config.enable_volume_features:
                for window in [self.config.medium_window, self.config.long_window]:
                    if window <= len(volume):
                        vol_ma = volume.rolling(window).mean()
                        vol_std = volume.rolling(window).std()
                        vol_zscore = (volume - vol_ma) / (vol_std + 1e-10)
                        features.append(vol_zscore.fillna(0).values)
                        feature_names.append(f'volume_zscore_{window}')
                        feature_categories['volume'].append(f'volume_zscore_{window}')
            
            # Volume momentum
            if self.config.enable_volume_momentum:
                for short_win, long_win in [(5, 20), (10, 50)]:
                    if long_win <= len(volume):
                        vol_ma_short = volume.rolling(short_win).mean()
                        vol_ma_long = volume.rolling(long_win).mean()
                        vol_momentum = vol_ma_short / (vol_ma_long + 1e-10)
                        features.append(vol_momentum.fillna(1).values)
                        feature_names.append(f'volume_momentum_{short_win}_{long_win}')
                        feature_categories['volume'].append(f'volume_momentum_{short_win}_{long_win}')
            
            # Volume volatility
            if self.config.enable_volume_volatility:
                for window in [self.config.medium_window, self.config.long_window]:
                    if window <= len(volume):
                        vol_returns = volume.pct_change()
                        vol_volatility = vol_returns.rolling(window).std()
                        features.append(vol_volatility.fillna(0).values)
                        feature_names.append(f'volume_volatility_{window}')
                        feature_categories['volume'].append(f'volume_volatility_{window}')
            
            # Volume-price correlation
            if self.config.enable_volume_price_correlation:
                for window in [self.config.medium_window, self.config.long_window]:
                    if window <= len(volume) and window <= len(returns):
                        vol_price_corr = returns.rolling(window).corr(volume)
                        features.append(vol_price_corr.fillna(0).values)
                        feature_names.append(f'vol_price_corr_{window}')
                        feature_categories['volume'].append(f'vol_price_corr_{window}')
            
            return features, feature_names, feature_categories
            
        except Exception as e:
            logger.warning(f"Volume features engineering failed: {e}")
            return [], [], {'volume': []}
    
    def _engineer_technical_features(self, market_data: pd.DataFrame, returns: pd.Series) -> Tuple[List[np.ndarray], List[str], Dict[str, List[str]]]:
        """Engineer technical indicator features."""
        try:
            features = []
            feature_names = []
            feature_categories = {
                'technical': [],
                'momentum': []
            }
            
            # RSI (Relative Strength Index)
            if self.config.enable_technical_indicators:
                for window in [14, 21, 50]:
                    if window <= len(market_data):
                        rsi = self._calculate_rsi(market_data['close'], window)
                        features.append(rsi.fillna(50).values)
                        feature_names.append(f'rsi_{window}')
                        feature_categories['technical'].append(f'rsi_{window}')
            
            # MACD (Moving Average Convergence Divergence)
            if self.config.enable_technical_indicators:
                macd_line, macd_signal, macd_histogram = self._calculate_macd(market_data['close'])
                features.extend([macd_line.fillna(0).values, macd_signal.fillna(0).values, macd_histogram.fillna(0).values])
                feature_names.extend(['macd_line', 'macd_signal', 'macd_histogram'])
                feature_categories['technical'].extend(['macd_line', 'macd_signal', 'macd_histogram'])
            
            # Bollinger Bands
            if self.config.enable_volatility_indicators:
                bb_upper, bb_lower, bb_width = self._calculate_bollinger_bands(market_data['close'])
                features.extend([bb_upper.fillna(0).values, bb_lower.fillna(0).values, bb_width.fillna(0).values])
                feature_names.extend(['bb_upper', 'bb_lower', 'bb_width'])
                feature_categories['technical'].extend(['bb_upper', 'bb_lower', 'bb_width'])
            
            # ATR (Average True Range)
            if self.config.enable_volatility_indicators and all(col in market_data.columns for col in ['high', 'low', 'close']):
                for window in [14, 21]:
                    if window <= len(market_data):
                        atr = self._calculate_atr(market_data, window)
                        features.append(atr.fillna(0).values)
                        feature_names.append(f'atr_{window}')
                        feature_categories['technical'].append(f'atr_{window}')
            
            # Momentum indicators
            if self.config.enable_momentum_indicators:
                for window in [5, 10, 20]:
                    if window <= len(returns):
                        momentum = returns.rolling(window).apply(lambda x: (1 + x).prod() - 1, raw=False)
                        features.append(momentum.fillna(0).values)
                        feature_names.append(f'momentum_{window}')
                        feature_categories['momentum'].append(f'momentum_{window}')
            
            return features, feature_names, feature_categories
            
        except Exception as e:
            logger.warning(f"Technical features engineering failed: {e}")
            return [], [], {'technical': [], 'momentum': []}
    
    def _calculate_rsi(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate Relative Strength Index."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception:
            return pd.Series(50, index=prices.index)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicators."""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            macd_signal = macd_line.ewm(span=signal).mean()
            macd_histogram = macd_line - macd_signal
            return macd_line, macd_signal, macd_histogram
        except Exception:
            return pd.Series(0, index=prices.index), pd.Series(0, index=prices.index), pd.Series(0, index=prices.index)
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        try:
            ma = prices.rolling(window).mean()
            std = prices.rolling(window).std()
            bb_upper = ma + (std * num_std)
            bb_lower = ma - (std * num_std)
            bb_width = (bb_upper - bb_lower) / ma
            return bb_upper, bb_lower, bb_width
        except Exception:
            return pd.Series(0, index=prices.index), pd.Series(0, index=prices.index), pd.Series(0, index=prices.index)
    
    def _calculate_atr(self, market_data: pd.DataFrame, window: int) -> pd.Series:
        """Calculate Average True Range."""
        try:
            high = market_data['high']
            low = market_data['low']
            close = market_data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window).mean()
            
            return atr
        except Exception:
            return pd.Series(0, index=market_data.index)
    
    def get_feature_importance_scores(self, features: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance scores using mutual information."""
        try:
            from sklearn.feature_selection import mutual_info_regression
            
            # Calculate mutual information
            mi_scores = mutual_info_regression(features, target, random_state=42)
            
            # Create feature importance dictionary
            importance_scores = {}
            for i, feature_name in enumerate(self.feature_names):
                if i < len(mi_scores):
                    importance_scores[feature_name] = float(mi_scores[i])
            
            return importance_scores
            
        except Exception as e:
            logger.warning(f"Feature importance calculation failed: {e}")
            return {name: 0.0 for name in self.feature_names}
    
    def get_feature_categories(self) -> Dict[str, List[str]]:
        """Get feature categories."""
        return self.feature_categories.copy()
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        return self.feature_names.copy()