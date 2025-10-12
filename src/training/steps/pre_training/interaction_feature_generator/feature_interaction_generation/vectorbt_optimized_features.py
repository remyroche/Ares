"""
VectorBT-Optimized Feature Generation

This module provides VectorBT-optimized implementations of technical indicators,
rolling statistics, and feature generation operations for maximum performance.

Key Features:
- VectorBT-optimized technical indicators with GPU acceleration
- Vectorized rolling operations for cross-timeframe features
- Memory-efficient feature generation using VectorBT data structures
- Enhanced interaction feature generation with matrix operations
- Comprehensive validation using VectorBT utilities
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
import warnings

# VectorBT imports
try:
    import vectorbt as vbt
    from vectorbt.indicators import RSI, MACD, BollingerBands, SMA, EMA
    from vectorbt.portfolio.base import Portfolio
    from vectorbt.utils import checks
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    RSI = None
    MACD = None
    BollingerBands = None
    SMA = None
    EMA = None

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance
)

logger = logging.getLogger(__name__)


@dataclass
class VectorBTFeatureConfig:
    """Configuration for VectorBT-optimized feature generation."""
    # Technical indicators
    enable_rsi: bool = True
    enable_macd: bool = True
    enable_bollinger: bool = True
    enable_sma: bool = True
    enable_ema: bool = True
    
    # RSI parameters
    rsi_periods: List[int] = None
    
    # MACD parameters
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Bollinger Bands parameters
    bb_periods: List[int] = None
    bb_std_devs: List[float] = None
    
    # SMA/EMA parameters
    sma_periods: List[int] = None
    ema_periods: List[int] = None
    
    # Rolling operations
    rolling_windows: List[int] = None
    
    # Cross-timeframe
    cross_timeframe_periods: List[int] = None
    
    # Performance settings
    use_gpu: bool = True
    chunk_size: int = 50000
    memory_limit_gb: float = 8.0
    enable_parallel: bool = True
    
    # Validation
    min_valid_ratio: float = 0.8
    max_constant_ratio: float = 0.1
    
    def __post_init__(self):
        if self.rsi_periods is None:
            self.rsi_periods = [14, 21, 28]
        if self.bb_periods is None:
            self.bb_periods = [20, 30, 50]
        if self.bb_std_devs is None:
            self.bb_std_devs = [1.5, 2.0, 2.5]
        if self.sma_periods is None:
            self.sma_periods = [5, 10, 20, 50, 100]
        if self.ema_periods is None:
            self.ema_periods = [5, 10, 20, 50, 100]
        if self.rolling_windows is None:
            self.rolling_windows = [5, 10, 20, 50, 100]
        if self.cross_timeframe_periods is None:
            self.cross_timeframe_periods = [5, 15, 30, 60]


class VectorBTFeatureGenerator:
    """VectorBT-optimized feature generator with GPU acceleration."""
    
    def __init__(self, config: Optional[VectorBTFeatureConfig] = None):
        """Initialize the VectorBT feature generator."""
        self.config = config or VectorBTFeatureConfig()
        
        if not VECTORBT_AVAILABLE:
            raise ImportError("VectorBT is required for VectorBTFeatureGenerator")
        
        # Initialize VectorBT settings
        self._setup_vectorbt()
        
        tprint_success("🚀 VectorBT Feature Generator initialized")
        tprint_info(f"📊 GPU acceleration: {'✅' if self.config.use_gpu and CUPY_AVAILABLE else '❌'}")
        tprint_info(f"📊 Parallel processing: {'✅' if self.config.enable_parallel else '❌'}")
        tprint_info(f"📊 Memory limit: {self.config.memory_limit_gb} GB")
    
    def _setup_vectorbt(self):
        """Setup VectorBT configuration for optimal performance."""
        try:
            # Configure VectorBT for performance
            vbt.settings.set_theme("dark")
            vbt.settings['array_wrapper']['freq_precision'] = 0
            vbt.settings['array_wrapper']['freq_rep'] = 'auto'
            
            # Enable parallel processing if requested
            if self.config.enable_parallel:
                vbt.settings['array_wrapper']['freq_rep'] = 'auto'
            
            # Configure memory settings
            if hasattr(vbt.settings, 'memory'):
                vbt.settings['memory']['limit'] = self.config.memory_limit_gb * 1024**3
            
            tprint_debug("✅ VectorBT configuration applied")
            
        except Exception as e:
            tprint_warning(f"⚠️ Could not configure VectorBT settings: {e}")
    
    def generate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate technical indicators using the feature bank with VectorBT optimizations.
        
        Args:
            data: OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            DataFrame with technical indicators from the feature bank
        """
        tprint_info("🔧 Generating technical indicators using feature bank with VectorBT optimizations...")
        start_time = time.time()
        
        # Validate input data
        self._validate_ohlcv_data(data)
        
        try:
            # Import feature bank components
            from src.feature_generation.core.feature_bank import FeatureBank, FeatureBankConfig
            from src.feature_generation.categories import (
                create_default_momentum_generators,
                create_default_volatility_generators,
                create_default_oscillator_generators,
                create_default_trend_generators
            )
            
            # Create feature bank configuration with VectorBT optimizations
            bank_config = FeatureBankConfig(
                enable_matrix_operations=True,
                enable_gpu_acceleration=self.config.use_gpu,
                enable_parallel_processing=self.config.enable_parallel,
                memory_efficient=True,
                chunk_size=self.config.chunk_size
            )
            
            # Initialize feature bank
            feature_bank = FeatureBank(bank_config)
            
            # Generate features from different categories
            all_features = []
            
            # Generate momentum features (RSI, MACD, etc.)
            if self.config.enable_rsi or self.config.enable_macd:
                momentum_generators = create_default_momentum_generators()
                for generator in momentum_generators:
                    try:
                        momentum_features = generator.generate(data)
                        if not momentum_features.empty:
                            all_features.append(momentum_features)
                            tprint_debug(f"✅ Generated momentum features: {momentum_features.shape[1]} features")
                    except Exception as e:
                        tprint_warning(f"⚠️ Failed to generate momentum features with {generator.__class__.__name__}: {e}")
            
            # Generate volatility features (Bollinger Bands, ATR, etc.)
            if self.config.enable_bollinger:
                volatility_generators = create_default_volatility_generators()
                for generator in volatility_generators:
                    try:
                        volatility_features = generator.generate(data)
                        if not volatility_features.empty:
                            all_features.append(volatility_features)
                            tprint_debug(f"✅ Generated volatility features: {volatility_features.shape[1]} features")
                    except Exception as e:
                        tprint_warning(f"⚠️ Failed to generate volatility features with {generator.__class__.__name__}: {e}")
            
            # Generate oscillator features (Stochastic, Williams %R, etc.)
            oscillator_generators = create_default_oscillator_generators()
            for generator in oscillator_generators:
                try:
                    oscillator_features = generator.generate(data)
                    if not oscillator_features.empty:
                        all_features.append(oscillator_features)
                        tprint_debug(f"✅ Generated oscillator features: {oscillator_features.shape[1]} features")
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to generate oscillator features with {generator.__class__.__name__}: {e}")
            
            # Generate trend features (SMA, EMA, etc.)
            if self.config.enable_sma or self.config.enable_ema:
                trend_generators = create_default_trend_generators()
                for generator in trend_generators:
                    try:
                        trend_features = generator.generate(data)
                        if not trend_features.empty:
                            all_features.append(trend_features)
                            tprint_debug(f"✅ Generated trend features: {trend_features.shape[1]} features")
                    except Exception as e:
                        tprint_warning(f"⚠️ Failed to generate trend features with {generator.__class__.__name__}: {e}")
            
            # Combine all features
            if all_features:
                result_df = pd.concat(all_features, axis=1)
                # Remove duplicate columns
                result_df = result_df.loc[:, ~result_df.columns.duplicated(keep='first')]
                result_df = self._optimize_dataframe_dtypes(result_df)
            else:
                result_df = pd.DataFrame(index=data.index)
                tprint_warning("⚠️ No technical indicators generated from feature bank")
            
            execution_time = time.time() - start_time
            tprint_success(f"✅ Generated {len(result_df.columns)} technical indicators from feature bank in {execution_time:.3f}s")
            
            return result_df
            
        except ImportError as e:
            tprint_warning(f"⚠️ Feature bank not available, falling back to direct VectorBT generation: {e}")
            return self._generate_technical_indicators_fallback(data)
        except Exception as e:
            tprint_warning(f"⚠️ Feature bank generation failed, falling back to direct VectorBT generation: {e}")
            return self._generate_technical_indicators_fallback(data)
    
    def _generate_technical_indicators_fallback(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fallback to direct VectorBT technical indicator generation."""
        tprint_info("🔧 Using fallback VectorBT technical indicator generation...")
        start_time = time.time()
        
        # Convert to VectorBT format
        ohlcv_data = self._prepare_ohlcv_data(data)
        
        # Use VectorBT for batch technical indicator processing
        if self.config.use_gpu and CUPY_AVAILABLE:
            features = self._generate_technical_indicators_gpu_batch(ohlcv_data)
        else:
            features = self._generate_technical_indicators_cpu_batch(ohlcv_data)
        
        # Create result DataFrame
        if features:
            result_df = pd.DataFrame(features, index=data.index)
            result_df = self._optimize_dataframe_dtypes(result_df)
        else:
            result_df = pd.DataFrame(index=data.index)
        
        execution_time = time.time() - start_time
        tprint_success(f"✅ Generated {len(result_df.columns)} technical indicators (fallback) in {execution_time:.3f}s")
        
        return result_df
    
    def _generate_technical_indicators_cpu_batch(self, ohlcv_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate technical indicators using VectorBT CPU batch processing."""
        features = {}
        
        try:
            # Use VectorBT for batch technical indicator processing
            close = ohlcv_data['close']
            high = ohlcv_data['high']
            low = ohlcv_data['low']
            volume = ohlcv_data['volume']
            
            # Generate RSI indicators in batch
            if self.config.enable_rsi:
                for period in self.config.rsi_periods:
                    try:
                        rsi = RSI.run(close, window=period)
                        features[f'rsi_{period}'] = rsi.rsi.values
                        features[f'rsi_{period}_signal'] = (rsi.rsi > 50).astype(int)
                        features[f'rsi_{period}_oversold'] = (rsi.rsi < 30).astype(int)
                        features[f'rsi_{period}_overbought'] = (rsi.rsi > 70).astype(int)
                        
                        # Additional RSI features
                        features[f'rsi_{period}_momentum'] = rsi.rsi.diff()
                        features[f'rsi_{period}_divergence'] = self._calculate_rsi_divergence(close, rsi.rsi, period)
                    except Exception as e:
                        tprint_warning(f"⚠️ Failed to generate RSI {period}: {e}")
            
            # Generate MACD indicators
            if self.config.enable_macd:
                try:
                    macd = MACD.run(
                        close,
                        fast_window=self.config.macd_fast,
                        slow_window=self.config.macd_slow,
                        signal_window=self.config.macd_signal
                    )
                    
                    features['macd'] = macd.macd.values
                    features['macd_signal'] = macd.signal.values
                    features['macd_histogram'] = macd.histogram.values
                    features['macd_crossover'] = (macd.macd > macd.signal).astype(int)
                    features['macd_crossunder'] = (macd.macd < macd.signal).astype(int)
                    
                    # Additional MACD features
                    features['macd_momentum'] = macd.histogram.diff()
                    features['macd_signal_strength'] = np.abs(macd.histogram)
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to generate MACD: {e}")
            
            # Generate Bollinger Bands in batch
            if self.config.enable_bollinger:
                for period in self.config.bb_periods:
                    for std_dev in self.config.bb_std_devs:
                        try:
                            bb = BollingerBands.run(close, window=period, alpha=std_dev)
                            
                            features[f'bb_{period}_{std_dev}_upper'] = bb.upper.values
                            features[f'bb_{period}_{std_dev}_middle'] = bb.middle.values
                            features[f'bb_{period}_{std_dev}_lower'] = bb.lower.values
                            features[f'bb_{period}_{std_dev}_width'] = bb.width.values
                            features[f'bb_{period}_{std_dev}_percent'] = bb.percent.values
                            features[f'bb_{period}_{std_dev}_zscore'] = bb.zscore.values
                            
                            # Additional Bollinger Bands features
                            features[f'bb_{period}_{std_dev}_squeeze'] = (bb.width < bb.width.rolling(20).mean()).astype(int)
                            features[f'bb_{period}_{std_dev}_breakout'] = ((close > bb.upper) | (close < bb.lower)).astype(int)
                        except Exception as e:
                            tprint_warning(f"⚠️ Failed to generate Bollinger Bands {period}_{std_dev}: {e}")
            
            # Generate SMA indicators in batch
            if self.config.enable_sma:
                for period in self.config.sma_periods:
                    try:
                        sma = SMA.run(close, window=period)
                        features[f'sma_{period}'] = sma.sma.values
                        features[f'sma_{period}_signal'] = (close > sma.sma).astype(int)
                        features[f'sma_{period}_distance'] = (close / sma.sma - 1) * 100
                        features[f'sma_{period}_slope'] = sma.sma.diff()
                    except Exception as e:
                        tprint_warning(f"⚠️ Failed to generate SMA {period}: {e}")
            
            # Generate EMA indicators in batch
            if self.config.enable_ema:
                for period in self.config.ema_periods:
                    try:
                        ema = EMA.run(close, window=period)
                        features[f'ema_{period}'] = ema.ema.values
                        features[f'ema_{period}_signal'] = (close > ema.ema).astype(int)
                        features[f'ema_{period}_distance'] = (close / ema.ema - 1) * 100
                        features[f'ema_{period}_slope'] = ema.ema.diff()
                    except Exception as e:
                        tprint_warning(f"⚠️ Failed to generate EMA {period}: {e}")
            
            # Generate additional advanced indicators
            features.update(self._generate_advanced_indicators(ohlcv_data))
            
            tprint_success(f"✅ Generated {len(features)} technical indicator features using VectorBT batch processing")
            
        except Exception as e:
            tprint_warning(f"⚠️ VectorBT batch technical indicator processing failed: {e}")
            # Fallback to individual processing
            return self._generate_technical_indicators_individual(ohlcv_data)
        
        return features
    
    def _generate_technical_indicators_gpu_batch(self, ohlcv_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate technical indicators using GPU batch processing."""
        features = {}
        
        try:
            # Convert OHLCV data to GPU arrays
            close_gpu = cp.asarray(ohlcv_data['close'].values)
            high_gpu = cp.asarray(ohlcv_data['high'].values)
            low_gpu = cp.asarray(ohlcv_data['low'].values)
            volume_gpu = cp.asarray(ohlcv_data['volume'].values)
            
            # Generate basic technical indicators on GPU
            if self.config.enable_sma:
                for period in self.config.sma_periods:
                    sma_gpu = self._gpu_rolling_mean(close_gpu.reshape(-1, 1), period).flatten()
                    features[f'sma_{period}'] = cp.asnumpy(sma_gpu)
                    features[f'sma_{period}_signal'] = cp.asnumpy(close_gpu > sma_gpu).astype(int)
            
            if self.config.enable_ema:
                for period in self.config.ema_periods:
                    ema_gpu = self._gpu_ema(close_gpu, period)
                    features[f'ema_{period}'] = cp.asnumpy(ema_gpu)
                    features[f'ema_{period}_signal'] = cp.asnumpy(close_gpu > ema_gpu).astype(int)
            
            # Generate volatility indicators on GPU
            returns_gpu = cp.diff(cp.log(close_gpu + 1e-8))
            for period in [5, 10, 20]:
                vol_gpu = self._gpu_rolling_std(returns_gpu.reshape(-1, 1), period).flatten()
                features[f'volatility_{period}'] = cp.asnumpy(vol_gpu)
            
            tprint_success(f"✅ Generated {len(features)} technical indicator features using GPU batch processing")
            
        except Exception as e:
            tprint_warning(f"⚠️ GPU technical indicator processing failed, falling back to CPU: {e}")
            return self._generate_technical_indicators_cpu_batch(ohlcv_data)
        
        return features
    
    def _generate_advanced_indicators(self, ohlcv_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate advanced technical indicators."""
        features = {}
        
        try:
            close = ohlcv_data['close']
            high = ohlcv_data['high']
            low = ohlcv_data['low']
            volume = ohlcv_data['volume']
            
            # Stochastic Oscillator
            for period in [14, 21]:
                try:
                    stoch = vbt.indicators.Stochastic.run(high, low, close, window=period)
                    features[f'stoch_k_{period}'] = stoch.stoch_k.values
                    features[f'stoch_d_{period}'] = stoch.stoch_d.values
                    features[f'stoch_signal_{period}'] = (stoch.stoch_k > stoch.stoch_d).astype(int)
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to generate Stochastic {period}: {e}")
            
            # Williams %R
            for period in [14, 21]:
                try:
                    williams_r = vbt.indicators.WilliamsR.run(high, low, close, window=period)
                    features[f'williams_r_{period}'] = williams_r.williams_r.values
                    features[f'williams_r_oversold_{period}'] = (williams_r.williams_r < -80).astype(int)
                    features[f'williams_r_overbought_{period}'] = (williams_r.williams_r > -20).astype(int)
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to generate Williams %R {period}: {e}")
            
            # Commodity Channel Index (CCI)
            for period in [14, 20]:
                try:
                    cci = vbt.indicators.CCI.run(high, low, close, window=period)
                    features[f'cci_{period}'] = cci.cci.values
                    features[f'cci_signal_{period}'] = (cci.cci > 100).astype(int) - (cci.cci < -100).astype(int)
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to generate CCI {period}: {e}")
            
            # Average True Range (ATR)
            for period in [14, 21]:
                try:
                    atr = vbt.indicators.ATR.run(high, low, close, window=period)
                    features[f'atr_{period}'] = atr.atr.values
                    features[f'atr_percent_{period}'] = (atr.atr / close) * 100
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to generate ATR {period}: {e}")
            
        except Exception as e:
            tprint_warning(f"⚠️ Advanced indicator generation failed: {e}")
        
        return features
    
    def _gpu_ema(self, data_gpu: cp.ndarray, period: int) -> cp.ndarray:
        """GPU-accelerated exponential moving average."""
        alpha = 2.0 / (period + 1)
        ema = cp.zeros_like(data_gpu)
        ema[0] = data_gpu[0]
        
        for i in range(1, len(data_gpu)):
            ema[i] = alpha * data_gpu[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def _calculate_rsi_divergence(self, price: pd.Series, rsi: pd.Series, period: int) -> pd.Series:
        """Calculate RSI divergence."""
        try:
            # Simple divergence calculation
            price_peaks = price.rolling(window=period).max() == price
            rsi_peaks = rsi.rolling(window=period).max() == rsi
            
            divergence = np.zeros(len(price))
            for i in range(period, len(price)):
                if price_peaks.iloc[i] and rsi_peaks.iloc[i]:
                    # Check if price is higher but RSI is lower (bearish divergence)
                    if price.iloc[i] > price.iloc[i-period] and rsi.iloc[i] < rsi.iloc[i-period]:
                        divergence[i] = -1
                    # Check if price is lower but RSI is higher (bullish divergence)
                    elif price.iloc[i] < price.iloc[i-period] and rsi.iloc[i] > rsi.iloc[i-period]:
                        divergence[i] = 1
            
            return pd.Series(divergence, index=price.index)
        except Exception:
            return pd.Series(0, index=price.index)
    
    def _generate_technical_indicators_individual(self, ohlcv_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate technical indicators using individual VectorBT operations (fallback)."""
        features = {}
        
        # Generate RSI indicators
        if self.config.enable_rsi:
            features.update(self._generate_rsi_indicators(ohlcv_data))
        
        # Generate MACD indicators
        if self.config.enable_macd:
            features.update(self._generate_macd_indicators(ohlcv_data))
        
        # Generate Bollinger Bands
        if self.config.enable_bollinger:
            features.update(self._generate_bollinger_indicators(ohlcv_data))
        
        # Generate SMA indicators
        if self.config.enable_sma:
            features.update(self._generate_sma_indicators(ohlcv_data))
        
        # Generate EMA indicators
        if self.config.enable_ema:
            features.update(self._generate_ema_indicators(ohlcv_data))
        
        return features
    
    def generate_rolling_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate rolling statistics using VectorBT optimizations.
        
        Args:
            data: Input data for rolling calculations
            
        Returns:
            DataFrame with rolling features
        """
        tprint_info("📊 Generating VectorBT-optimized rolling features...")
        start_time = time.time()
        
        # Validate input data
        if data.empty:
            raise ValueError("Input data is empty")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found for rolling calculations")
        
        features = {}
        
        # Use VectorBT for batch processing of all windows and columns
        if self.config.use_gpu and CUPY_AVAILABLE:
            features = self._generate_rolling_features_gpu_batch(data, numeric_cols)
        else:
            features = self._generate_rolling_features_cpu_batch(data, numeric_cols)
        
        # Create result DataFrame
        if features:
            result_df = pd.DataFrame(features, index=data.index)
            result_df = self._optimize_dataframe_dtypes(result_df)
        else:
            result_df = pd.DataFrame(index=data.index)
        
        execution_time = time.time() - start_time
        tprint_success(f"✅ Generated {len(result_df.columns)} rolling features in {execution_time:.3f}s")
        
        return result_df
    
    def _generate_rolling_features_cpu_batch(self, data: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, np.ndarray]:
        """Generate rolling features using VectorBT CPU batch processing."""
        features = {}
        
        # Process all windows and columns in batches for better performance
        for window in self.config.rolling_windows:
            tprint_debug(f"🔧 Processing rolling window: {window}")
            
            # Create a matrix of all numeric columns for batch processing
            data_matrix = data[numeric_cols].values
            
            try:
                # Use VectorBT for batch rolling operations
                rolling_obj = vbt.Rolling.from_1d(data_matrix, window=window)
                
                # Batch compute all rolling statistics
                rolling_mean = rolling_obj.mean()
                rolling_std = rolling_obj.std()
                rolling_min = rolling_obj.min()
                rolling_max = rolling_obj.max()
                rolling_median = rolling_obj.median()
                rolling_skew = rolling_obj.skew()
                rolling_kurt = rolling_obj.kurt()
                
                # Add features for each column
                for i, col in enumerate(numeric_cols):
                    features[f'rolling_{window}_{col}_mean'] = rolling_mean[:, i]
                    features[f'rolling_{window}_{col}_std'] = rolling_std[:, i]
                    features[f'rolling_{window}_{col}_min'] = rolling_min[:, i]
                    features[f'rolling_{window}_{col}_max'] = rolling_max[:, i]
                    features[f'rolling_{window}_{col}_median'] = rolling_median[:, i]
                    features[f'rolling_{window}_{col}_skew'] = rolling_skew[:, i]
                    features[f'rolling_{window}_{col}_kurt'] = rolling_kurt[:, i]
                    
                    # Additional quantile features
                    for q in [0.25, 0.75, 0.9, 0.95]:
                        rolling_q = rolling_obj.quantile(q)
                        features[f'rolling_{window}_{col}_q{int(q*100)}'] = rolling_q[:, i]
                
                tprint_debug(f"✅ Processed window {window} with {len(numeric_cols)} columns")
                
            except Exception as e:
                tprint_warning(f"⚠️ VectorBT batch processing failed for window {window}: {e}")
                # Fallback to individual column processing
                for col in numeric_cols:
                    series = data[col]
                    try:
                        rolling_obj = vbt.Rolling.from_1d(series, window=window)
                        features[f'rolling_{window}_{col}_mean'] = rolling_obj.mean()
                        features[f'rolling_{window}_{col}_std'] = rolling_obj.std()
                        features[f'rolling_{window}_{col}_min'] = rolling_obj.min()
                        features[f'rolling_{window}_{col}_max'] = rolling_obj.max()
                        features[f'rolling_{window}_{col}_median'] = rolling_obj.median()
                        features[f'rolling_{window}_{col}_skew'] = rolling_obj.skew()
                        features[f'rolling_{window}_{col}_kurt'] = rolling_obj.kurt()
                    except Exception as col_e:
                        tprint_warning(f"⚠️ Failed to process {col} for window {window}: {col_e}")
        
        return features
    
    def _generate_rolling_features_gpu_batch(self, data: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, np.ndarray]:
        """Generate rolling features using VectorBT GPU batch processing."""
        features = {}
        
        try:
            # Convert data to GPU array
            data_gpu = cp.asarray(data[numeric_cols].values)
            
            for window in self.config.rolling_windows:
                tprint_debug(f"🔧 Processing rolling window {window} on GPU...")
                
                # Use CuPy for GPU-accelerated rolling operations
                rolling_mean = self._gpu_rolling_mean(data_gpu, window)
                rolling_std = self._gpu_rolling_std(data_gpu, window)
                rolling_min = self._gpu_rolling_min(data_gpu, window)
                rolling_max = self._gpu_rolling_max(data_gpu, window)
                
                # Convert back to CPU and add features
                for i, col in enumerate(numeric_cols):
                    features[f'rolling_{window}_{col}_mean'] = cp.asnumpy(rolling_mean[:, i])
                    features[f'rolling_{window}_{col}_std'] = cp.asnumpy(rolling_std[:, i])
                    features[f'rolling_{window}_{col}_min'] = cp.asnumpy(rolling_min[:, i])
                    features[f'rolling_{window}_{col}_max'] = cp.asnumpy(rolling_max[:, i])
                
                tprint_debug(f"✅ Processed window {window} on GPU with {len(numeric_cols)} columns")
                
        except Exception as e:
            tprint_warning(f"⚠️ GPU processing failed, falling back to CPU: {e}")
            return self._generate_rolling_features_cpu_batch(data, numeric_cols)
        
        return features
    
    def _gpu_rolling_mean(self, data_gpu: cp.ndarray, window: int) -> cp.ndarray:
        """GPU-accelerated rolling mean."""
        return cp.convolve(data_gpu, cp.ones(window) / window, mode='valid')
    
    def _gpu_rolling_std(self, data_gpu: cp.ndarray, window: int) -> cp.ndarray:
        """GPU-accelerated rolling standard deviation."""
        rolling_mean = self._gpu_rolling_mean(data_gpu, window)
        rolling_var = cp.convolve(data_gpu**2, cp.ones(window) / window, mode='valid') - rolling_mean**2
        return cp.sqrt(cp.maximum(rolling_var, 0))
    
    def _gpu_rolling_min(self, data_gpu: cp.ndarray, window: int) -> cp.ndarray:
        """GPU-accelerated rolling minimum."""
        return cp.minimum.accumulate(data_gpu, axis=0)[window-1:]
    
    def _gpu_rolling_max(self, data_gpu: cp.ndarray, window: int) -> cp.ndarray:
        """GPU-accelerated rolling maximum."""
        return cp.maximum.accumulate(data_gpu, axis=0)[window-1:]
    
    def generate_cross_timeframe_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate cross-timeframe features using VectorBT optimizations.
        
        Args:
            data: Input data for cross-timeframe calculations
            
        Returns:
            DataFrame with cross-timeframe features
        """
        tprint_info("⏰ Generating VectorBT-optimized cross-timeframe features...")
        start_time = time.time()
        
        # Validate input data
        if data.empty:
            raise ValueError("Input data is empty")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found for cross-timeframe calculations")
        
        # Use VectorBT for batch cross-timeframe processing
        if self.config.use_gpu and CUPY_AVAILABLE:
            features = self._generate_cross_timeframe_features_gpu_batch(data, numeric_cols)
        else:
            features = self._generate_cross_timeframe_features_cpu_batch(data, numeric_cols)
        
        # Create result DataFrame
        if features:
            result_df = pd.DataFrame(features, index=data.index)
            result_df = self._optimize_dataframe_dtypes(result_df)
        else:
            result_df = pd.DataFrame(index=data.index)
        
        execution_time = time.time() - start_time
        tprint_success(f"✅ Generated {len(result_df.columns)} cross-timeframe features in {execution_time:.3f}s")
        
        return result_df
    
    def _generate_cross_timeframe_features_cpu_batch(self, data: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, np.ndarray]:
        """Generate cross-timeframe features using VectorBT CPU batch processing."""
        features = {}
        
        # Create a matrix of all numeric columns for batch processing
        data_matrix = data[numeric_cols].values
        
        for period in self.config.cross_timeframe_periods:
            tprint_debug(f"🔧 Processing cross-timeframe period: {period}")
            
            try:
                # Use VectorBT for batch cross-timeframe operations
                rolling_obj = vbt.Rolling.from_1d(data_matrix, window=period)
                
                # Batch compute all cross-timeframe statistics
                ctf_mean = rolling_obj.mean()
                ctf_std = rolling_obj.std()
                ctf_min = rolling_obj.min()
                ctf_max = rolling_obj.max()
                ctf_median = rolling_obj.median()
                ctf_skew = rolling_obj.skew()
                ctf_kurt = rolling_obj.kurt()
                
                # Add features for each column
                for i, col in enumerate(numeric_cols):
                    features[f'ctf_{period}m_{col}_mean'] = ctf_mean[:, i]
                    features[f'ctf_{period}m_{col}_std'] = ctf_std[:, i]
                    features[f'ctf_{period}m_{col}_min'] = ctf_min[:, i]
                    features[f'ctf_{period}m_{col}_max'] = ctf_max[:, i]
                    features[f'ctf_{period}m_{col}_median'] = ctf_median[:, i]
                    features[f'ctf_{period}m_{col}_skew'] = ctf_skew[:, i]
                    features[f'ctf_{period}m_{col}_kurt'] = ctf_kurt[:, i]
                    
                    # Additional quantile features
                    for q in [0.25, 0.75, 0.9, 0.95]:
                        ctf_q = rolling_obj.quantile(q)
                        features[f'ctf_{period}m_{col}_q{int(q*100)}'] = ctf_q[:, i]
                    
                    # Cross-timeframe momentum and volatility features
                    if col in ['close', 'open', 'high', 'low']:
                        # Price momentum across timeframes
                        price_series = data[col]
                        ctf_momentum = (price_series / rolling_obj.mean()[:, i] - 1) * 100
                        features[f'ctf_{period}m_{col}_momentum'] = ctf_momentum
                        
                        # Volatility ratio (current vs cross-timeframe)
                        current_vol = price_series.pct_change().rolling(window=5).std()
                        ctf_vol = rolling_obj.std()[:, i]
                        vol_ratio = current_vol / (ctf_vol + 1e-8)
                        features[f'ctf_{period}m_{col}_vol_ratio'] = vol_ratio
                
                tprint_debug(f"✅ Processed cross-timeframe period {period} with {len(numeric_cols)} columns")
                
            except Exception as e:
                tprint_warning(f"⚠️ VectorBT cross-timeframe batch processing failed for period {period}: {e}")
                # Fallback to individual column processing
                for col in numeric_cols:
                    series = data[col]
                    try:
                        rolling_obj = vbt.Rolling.from_1d(series, window=period)
                        features[f'ctf_{period}m_{col}_mean'] = rolling_obj.mean()
                        features[f'ctf_{period}m_{col}_std'] = rolling_obj.std()
                        features[f'ctf_{period}m_{col}_min'] = rolling_obj.min()
                        features[f'ctf_{period}m_{col}_max'] = rolling_obj.max()
                        features[f'ctf_{period}m_{col}_median'] = rolling_obj.median()
                        features[f'ctf_{period}m_{col}_skew'] = rolling_obj.skew()
                        features[f'ctf_{period}m_{col}_kurt'] = rolling_obj.kurt()
                    except Exception as col_e:
                        tprint_warning(f"⚠️ Failed to process {col} for period {period}: {col_e}")
        
        return features
    
    def _generate_cross_timeframe_features_gpu_batch(self, data: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, np.ndarray]:
        """Generate cross-timeframe features using GPU batch processing."""
        features = {}
        
        try:
            # Convert data to GPU array
            data_gpu = cp.asarray(data[numeric_cols].values)
            
            for period in self.config.cross_timeframe_periods:
                tprint_debug(f"🔧 Processing cross-timeframe period {period} on GPU...")
                
                # Use CuPy for GPU-accelerated cross-timeframe operations
                ctf_mean = self._gpu_rolling_mean(data_gpu, period)
                ctf_std = self._gpu_rolling_std(data_gpu, period)
                ctf_min = self._gpu_rolling_min(data_gpu, period)
                ctf_max = self._gpu_rolling_max(data_gpu, period)
                
                # Convert back to CPU and add features
                for i, col in enumerate(numeric_cols):
                    features[f'ctf_{period}m_{col}_mean'] = cp.asnumpy(ctf_mean[:, i])
                    features[f'ctf_{period}m_{col}_std'] = cp.asnumpy(ctf_std[:, i])
                    features[f'ctf_{period}m_{col}_min'] = cp.asnumpy(ctf_min[:, i])
                    features[f'ctf_{period}m_{col}_max'] = cp.asnumpy(ctf_max[:, i])
                    
                    # Additional cross-timeframe features
                    if col in ['close', 'open', 'high', 'low']:
                        price_series = data[col]
                        ctf_momentum = (price_series / cp.asnumpy(ctf_mean[:, i]) - 1) * 100
                        features[f'ctf_{period}m_{col}_momentum'] = ctf_momentum
                
                tprint_debug(f"✅ Processed cross-timeframe period {period} on GPU with {len(numeric_cols)} columns")
                
        except Exception as e:
            tprint_warning(f"⚠️ GPU cross-timeframe processing failed, falling back to CPU: {e}")
            return self._generate_cross_timeframe_features_cpu_batch(data, numeric_cols)
        
        return features
    
    def generate_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate interaction features using VectorBT matrix operations.
        
        Args:
            data: Input data for interaction calculations
            
        Returns:
            DataFrame with interaction features
        """
        tprint_info("🔗 Generating VectorBT-optimized interaction features...")
        start_time = time.time()
        
        # Validate input data
        if data.empty:
            raise ValueError("Input data is empty")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            raise ValueError("Need at least 2 numeric columns for interactions")
        
        # Use VectorBT for batch matrix operations
        if self.config.use_gpu and CUPY_AVAILABLE:
            features = self._generate_interaction_features_gpu_batch(data, numeric_cols)
        else:
            features = self._generate_interaction_features_cpu_batch(data, numeric_cols)
        
        # Create result DataFrame
        if features:
            result_df = pd.DataFrame(features, index=data.index)
            result_df = self._optimize_dataframe_dtypes(result_df)
        else:
            result_df = pd.DataFrame(index=data.index)
        
        execution_time = time.time() - start_time
        tprint_success(f"✅ Generated {len(result_df.columns)} interaction features in {execution_time:.3f}s")
        
        return result_df
    
    def _generate_interaction_features_cpu_batch(self, data: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, np.ndarray]:
        """Generate interaction features using VectorBT CPU batch processing."""
        features = {}
        
        # Convert to numpy arrays for VectorBT operations
        numeric_data = data[numeric_cols].values
        
        try:
            # Use VectorBT for batch matrix operations
            data_vbt = vbt.ArrayWrapper.from_1d(numeric_data)
            
            # Generate all pairwise interactions in batch
            n_cols = len(numeric_cols)
            interaction_count = 0
            max_interactions = min(100, n_cols * (n_cols - 1) // 2)
            
            for i in range(n_cols):
                if interaction_count >= max_interactions:
                    break
                    
                for j in range(i + 1, n_cols):
                    if interaction_count >= max_interactions:
                        break
                    
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    
                    # Extract individual series
                    series1 = data_vbt[:, i]
                    series2 = data_vbt[:, j]
                    
                    # Basic interactions
                    features[f'{col1}_div_{col2}'] = (series1 / (series2 + 1e-8)).values
                    features[f'{col1}_mul_{col2}'] = (series1 * series2).values
                    features[f'{col1}_sub_{col2}'] = (series1 - series2).values
                    features[f'{col1}_add_{col2}'] = (series1 + series2).values
                    
                    # Advanced interactions
                    features[f'{col1}_pow_{col2}'] = (series1 ** (series2 + 1e-8)).values
                    features[f'{col1}_log_ratio_{col2}'] = (np.log(series1 + 1e-8) - np.log(series2 + 1e-8)).values
                    features[f'{col1}_abs_diff_{col2}'] = np.abs(series1 - series2).values
                    features[f'{col1}_max_{col2}'] = np.maximum(series1, series2).values
                    features[f'{col1}_min_{col2}'] = np.minimum(series1, series2).values
                    
                    # Correlation-based interactions
                    if len(series1) > 10:  # Need sufficient data for correlation
                        rolling_corr = vbt.Rolling.from_1d(series1, window=min(20, len(series1)//2)).corr(series2)
                        features[f'{col1}_corr_{col2}'] = rolling_corr.values
                    
                    interaction_count += 9
                
                tprint_debug(f"✅ Processed interactions for column {i+1}/{n_cols}")
            
            tprint_success(f"✅ Generated {interaction_count} interaction features using VectorBT batch processing")
            
        except Exception as e:
            tprint_warning(f"⚠️ VectorBT batch interaction processing failed: {e}")
            # Fallback to individual processing
            return self._generate_interaction_features_individual(data, numeric_cols)
        
        return features
    
    def _generate_interaction_features_gpu_batch(self, data: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, np.ndarray]:
        """Generate interaction features using GPU batch processing."""
        features = {}
        
        try:
            # Convert to GPU arrays
            data_gpu = cp.asarray(data[numeric_cols].values)
            
            n_cols = len(numeric_cols)
            interaction_count = 0
            max_interactions = min(100, n_cols * (n_cols - 1) // 2)
            
            for i in range(n_cols):
                if interaction_count >= max_interactions:
                    break
                    
                for j in range(i + 1, n_cols):
                    if interaction_count >= max_interactions:
                        break
                    
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    
                    # Extract individual series
                    series1_gpu = data_gpu[:, i]
                    series2_gpu = data_gpu[:, j]
                    
                    # Basic interactions on GPU
                    features[f'{col1}_div_{col2}'] = cp.asnumpy(series1_gpu / (series2_gpu + 1e-8))
                    features[f'{col1}_mul_{col2}'] = cp.asnumpy(series1_gpu * series2_gpu)
                    features[f'{col1}_sub_{col2}'] = cp.asnumpy(series1_gpu - series2_gpu)
                    features[f'{col1}_add_{col2}'] = cp.asnumpy(series1_gpu + series2_gpu)
                    
                    # Advanced interactions on GPU
                    features[f'{col1}_pow_{col2}'] = cp.asnumpy(cp.power(series1_gpu, series2_gpu + 1e-8))
                    features[f'{col1}_log_ratio_{col2}'] = cp.asnumpy(cp.log(series1_gpu + 1e-8) - cp.log(series2_gpu + 1e-8))
                    features[f'{col1}_abs_diff_{col2}'] = cp.asnumpy(cp.abs(series1_gpu - series2_gpu))
                    features[f'{col1}_max_{col2}'] = cp.asnumpy(cp.maximum(series1_gpu, series2_gpu))
                    features[f'{col1}_min_{col2}'] = cp.asnumpy(cp.minimum(series1_gpu, series2_gpu))
                    
                    interaction_count += 9
                
                tprint_debug(f"✅ Processed GPU interactions for column {i+1}/{n_cols}")
            
            tprint_success(f"✅ Generated {interaction_count} interaction features using GPU batch processing")
            
        except Exception as e:
            tprint_warning(f"⚠️ GPU interaction processing failed, falling back to CPU: {e}")
            return self._generate_interaction_features_cpu_batch(data, numeric_cols)
        
        return features
    
    def _generate_interaction_features_individual(self, data: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, np.ndarray]:
        """Generate interaction features using individual VectorBT operations (fallback)."""
        features = {}
        
        interaction_count = 0
        max_interactions = min(100, len(numeric_cols) * (len(numeric_cols) - 1) // 2)
        
        for i, col1 in enumerate(numeric_cols):
            if interaction_count >= max_interactions:
                break
                
            for j, col2 in enumerate(numeric_cols[i+1:], i+1):
                if interaction_count >= max_interactions:
                    break
                
                try:
                    # Use VectorBT for optimized matrix operations
                    series1 = data[col1]
                    series2 = data[col2]
                    
                    # Basic interactions
                    features[f'{col1}_div_{col2}'] = (series1 / (series2 + 1e-8)).values
                    features[f'{col1}_mul_{col2}'] = (series1 * series2).values
                    features[f'{col1}_sub_{col2}'] = (series1 - series2).values
                    features[f'{col1}_add_{col2}'] = (series1 + series2).values
                    
                    interaction_count += 4
                    
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to generate interactions for {col1} and {col2}: {e}")
                    continue
        
        return features
    
    def _validate_ohlcv_data(self, data: pd.DataFrame):
        """Validate OHLCV data for technical indicators."""
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = set(required_cols) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required OHLCV columns: {missing_cols}")
        
        # Check for non-positive values in price columns
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if (data[col] <= 0).any():
                raise ValueError(f"Non-positive values found in {col} column")
        
        # Check OHLC relationships
        invalid_high = data['high'] < data[['open', 'close']].max(axis=1)
        if invalid_high.any():
            raise ValueError(f"Found {invalid_high.sum()} rows where high < max(open, close)")
        
        invalid_low = data['low'] > data[['open', 'close']].min(axis=1)
        if invalid_low.any():
            raise ValueError(f"Found {invalid_low.sum()} rows where low > min(open, close)")
    
    def _prepare_ohlcv_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare OHLCV data for VectorBT operations."""
        ohlcv_data = data[['open', 'high', 'low', 'close', 'volume']].copy()
        
        # Ensure proper data types
        for col in ohlcv_data.columns:
            ohlcv_data[col] = pd.to_numeric(ohlcv_data[col], errors='coerce')
        
        # Remove any remaining NaN values
        ohlcv_data = ohlcv_data.dropna()
        
        return ohlcv_data
    
    def _generate_rsi_indicators(self, ohlcv_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate RSI indicators using VectorBT."""
        features = {}
        close = ohlcv_data['close']
        
        for period in self.config.rsi_periods:
            try:
                rsi = RSI.run(close, window=period)
                features[f'rsi_{period}'] = rsi.rsi.values
                features[f'rsi_{period}_signal'] = (rsi.rsi > 50).astype(int)
                features[f'rsi_{period}_oversold'] = (rsi.rsi < 30).astype(int)
                features[f'rsi_{period}_overbought'] = (rsi.rsi > 70).astype(int)
            except Exception as e:
                tprint_warning(f"⚠️ Failed to generate RSI {period}: {e}")
        
        return features
    
    def _generate_macd_indicators(self, ohlcv_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate MACD indicators using VectorBT."""
        features = {}
        close = ohlcv_data['close']
        
        try:
            macd = MACD.run(
                close,
                fast_window=self.config.macd_fast,
                slow_window=self.config.macd_slow,
                signal_window=self.config.macd_signal
            )
            
            features['macd'] = macd.macd.values
            features['macd_signal'] = macd.signal.values
            features['macd_histogram'] = macd.histogram.values
            features['macd_crossover'] = (macd.macd > macd.signal).astype(int)
            features['macd_crossunder'] = (macd.macd < macd.signal).astype(int)
        except Exception as e:
            tprint_warning(f"⚠️ Failed to generate MACD: {e}")
        
        return features
    
    def _generate_bollinger_indicators(self, ohlcv_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate Bollinger Bands indicators using VectorBT."""
        features = {}
        close = ohlcv_data['close']
        
        for period in self.config.bb_periods:
            for std_dev in self.config.bb_std_devs:
                try:
                    bb = BollingerBands.run(close, window=period, alpha=std_dev)
                    
                    features[f'bb_{period}_{std_dev}_upper'] = bb.upper.values
                    features[f'bb_{period}_{std_dev}_middle'] = bb.middle.values
                    features[f'bb_{period}_{std_dev}_lower'] = bb.lower.values
                    features[f'bb_{period}_{std_dev}_width'] = bb.width.values
                    features[f'bb_{period}_{std_dev}_percent'] = bb.percent.values
                    features[f'bb_{period}_{std_dev}_zscore'] = bb.zscore.values
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to generate Bollinger Bands {period}_{std_dev}: {e}")
        
        return features
    
    def _generate_sma_indicators(self, ohlcv_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate SMA indicators using VectorBT."""
        features = {}
        close = ohlcv_data['close']
        
        for period in self.config.sma_periods:
            try:
                sma = SMA.run(close, window=period)
                features[f'sma_{period}'] = sma.sma.values
                features[f'sma_{period}_signal'] = (close > sma.sma).astype(int)
            except Exception as e:
                tprint_warning(f"⚠️ Failed to generate SMA {period}: {e}")
        
        return features
    
    def _generate_ema_indicators(self, ohlcv_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate EMA indicators using VectorBT."""
        features = {}
        close = ohlcv_data['close']
        
        for period in self.config.ema_periods:
            try:
                ema = EMA.run(close, window=period)
                features[f'ema_{period}'] = ema.ema.values
                features[f'ema_{period}_signal'] = (close > ema.ema).astype(int)
            except Exception as e:
                tprint_warning(f"⚠️ Failed to generate EMA {period}: {e}")
        
        return features
    
    def _optimize_dataframe_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame dtypes for memory efficiency using VectorBT utilities."""
        if df.empty:
            return df
        
        try:
            # Use VectorBT for memory optimization if available
            if VECTORBT_AVAILABLE:
                # Convert to VectorBT ArrayWrapper for optimization
                vbt_df = vbt.ArrayWrapper.from_1d(df.values)
                
                # Optimize float columns
                for col in df.select_dtypes(include=['float64']).columns:
                    series = df[col]
                    
                    # Check if values fit in float32
                    if series.min() >= np.finfo(np.float32).min and series.max() <= np.finfo(np.float32).max:
                        # Check if precision loss is acceptable
                        if np.allclose(series, series.astype(np.float32), rtol=1e-6):
                            df[col] = series.astype(np.float32)
                    
                    # Check if values fit in float16
                    elif series.min() >= np.finfo(np.float16).min and series.max() <= np.finfo(np.float16).max:
                        if np.allclose(series, series.astype(np.float16), rtol=1e-3):
                            df[col] = series.astype(np.float16)
                
                # Optimize integer columns
                for col in df.select_dtypes(include=['int64']).columns:
                    series = df[col]
                    
                    # Check if values fit in int32
                    if series.min() >= np.iinfo(np.int32).min and series.max() <= np.iinfo(np.int32).max:
                        df[col] = series.astype(np.int32)
                    
                    # Check if values fit in int16
                    elif series.min() >= np.iinfo(np.int16).min and series.max() <= np.iinfo(np.int16).max:
                        df[col] = series.astype(np.int16)
                    
                    # Check if values fit in int8
                    elif series.min() >= np.iinfo(np.int8).min and series.max() <= np.iinfo(np.int8).max:
                        df[col] = series.astype(np.int8)
                
                # Optimize boolean columns
                for col in df.select_dtypes(include=['bool']).columns:
                    df[col] = df[col].astype(np.uint8)
                
                tprint_debug(f"✅ Optimized DataFrame dtypes using VectorBT utilities")
                
            else:
                # Fallback to basic optimization
                for col in df.select_dtypes(include=['float64']).columns:
                    df[col] = pd.to_numeric(df[col], downcast='float')
                
                for col in df.select_dtypes(include=['int64']).columns:
                    df[col] = pd.to_numeric(df[col], downcast='integer')
            
            # Calculate memory savings
            original_memory = df.memory_usage(deep=True).sum()
            optimized_memory = df.memory_usage(deep=True).sum()
            memory_saved = original_memory - optimized_memory
            
            if memory_saved > 0:
                tprint_debug(f"💾 Memory optimization saved {memory_saved / 1024 / 1024:.2f} MB")
            
        except Exception as e:
            tprint_warning(f"⚠️ DataFrame dtype optimization failed: {e}")
            # Fallback to basic optimization
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = pd.to_numeric(df[col], downcast='float')
            
            for col in df.select_dtypes(include=['int64']).columns:
                df[col] = pd.to_numeric(df[col], downcast='integer')
        
        return df
    
    def validate_features(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate generated features using VectorBT utilities.
        
        Args:
            features: DataFrame with features to validate
            
        Returns:
            Validation results dictionary
        """
        if features.empty:
            return {
                'passed': False,
                'quality_score': 0.0,
                'issues': ['No features to validate']
            }
        
        issues = []
        quality_metrics = {}
        
        try:
            # Use VectorBT for enhanced validation if available
            if VECTORBT_AVAILABLE:
                # Convert to VectorBT ArrayWrapper for validation
                vbt_features = vbt.ArrayWrapper.from_1d(features.values)
                
                # Check for infinite values using VectorBT
                inf_mask = vbt_features.isinf()
                inf_count = inf_mask.sum()
                if inf_count > 0:
                    issues.append(f"Found {inf_count} infinite values")
                quality_metrics['infinite_ratio'] = inf_count / (features.size or 1)
                
                # Check for NaN values using VectorBT
                nan_mask = vbt_features.isnull()
                nan_count = nan_mask.sum()
                nan_ratio = nan_count / (features.size or 1)
                if nan_ratio > (1 - self.config.min_valid_ratio):
                    issues.append(f"Too many NaN values: {nan_ratio:.1%}")
                quality_metrics['nan_ratio'] = nan_ratio
                
                # Check for constant features using VectorBT
                constant_cols = []
                for i in range(features.shape[1]):
                    col_values = vbt_features[:, i]
                    if col_values.nunique() <= 1:
                        constant_cols.append(i)
                
                constant_count = len(constant_cols)
                constant_ratio = constant_count / len(features.columns)
                if constant_ratio > self.config.max_constant_ratio:
                    issues.append(f"Too many constant features: {constant_ratio:.1%}")
                quality_metrics['constant_ratio'] = constant_ratio
                
                # Check for correlation issues using VectorBT
                numeric_features = features.select_dtypes(include=[np.number])
                if len(numeric_features.columns) > 1:
                    try:
                        # Calculate correlation matrix using VectorBT
                        corr_matrix = vbt.ArrayWrapper.from_1d(numeric_features.values).corr()
                        
                        # Check for high correlations (potential multicollinearity)
                        high_corr_pairs = []
                        for i in range(len(numeric_features.columns)):
                            for j in range(i+1, len(numeric_features.columns)):
                                corr_val = abs(corr_matrix[i, j])
                                if corr_val > 0.95:  # Very high correlation
                                    high_corr_pairs.append((numeric_features.columns[i], numeric_features.columns[j], corr_val))
                        
                        if len(high_corr_pairs) > len(numeric_features.columns) * 0.1:  # More than 10% of pairs
                            issues.append(f"High correlation detected in {len(high_corr_pairs)} feature pairs")
                        quality_metrics['high_correlation_pairs'] = len(high_corr_pairs)
                        
                    except Exception as e:
                        tprint_warning(f"⚠️ Correlation validation failed: {e}")
                
                # Check for data quality using VectorBT statistics
                try:
                    # Calculate basic statistics using VectorBT
                    vbt_stats = vbt_features.stats()
                    
                    # Check for extreme values
                    extreme_values = 0
                    for i in range(features.shape[1]):
                        col_values = vbt_features[:, i]
                        q1, q3 = col_values.quantile([0.25, 0.75])
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        extreme_count = ((col_values < lower_bound) | (col_values > upper_bound)).sum()
                        extreme_values += extreme_count
                    
                    extreme_ratio = extreme_values / (features.size or 1)
                    if extreme_ratio > 0.1:  # More than 10% extreme values
                        issues.append(f"High number of extreme values: {extreme_ratio:.1%}")
                    quality_metrics['extreme_values_ratio'] = extreme_ratio
                    
                except Exception as e:
                    tprint_warning(f"⚠️ Statistical validation failed: {e}")
                
                tprint_debug(f"✅ Enhanced validation completed using VectorBT utilities")
                
            else:
                # Fallback to basic validation
                self._basic_validation(features, issues, quality_metrics)
            
        except Exception as e:
            tprint_warning(f"⚠️ VectorBT validation failed, using fallback: {e}")
            self._basic_validation(features, issues, quality_metrics)
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(quality_metrics)
        
        return {
            'passed': len(issues) == 0,
            'quality_score': quality_score,
            'issues': issues,
            'metrics': quality_metrics
        }
    
    def _basic_validation(self, features: pd.DataFrame, issues: List[str], quality_metrics: Dict[str, Any]):
        """Basic validation fallback."""
        # Check for infinite values
        inf_count = np.isinf(features.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            issues.append(f"Found {inf_count} infinite values")
        quality_metrics['infinite_ratio'] = inf_count / (features.size or 1)
        
        # Check for NaN values
        nan_count = features.isnull().sum().sum()
        nan_ratio = nan_count / (features.size or 1)
        if nan_ratio > (1 - self.config.min_valid_ratio):
            issues.append(f"Too many NaN values: {nan_ratio:.1%}")
        quality_metrics['nan_ratio'] = nan_ratio
        
        # Check for constant features
        constant_cols = features.nunique() <= 1
        constant_count = constant_cols.sum()
        constant_ratio = constant_count / len(features.columns)
        if constant_ratio > self.config.max_constant_ratio:
            issues.append(f"Too many constant features: {constant_ratio:.1%}")
        quality_metrics['constant_ratio'] = constant_ratio
    
    def _calculate_quality_score(self, quality_metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score."""
        # Base score components
        infinite_penalty = quality_metrics.get('infinite_ratio', 0)
        nan_penalty = quality_metrics.get('nan_ratio', 0)
        constant_penalty = quality_metrics.get('constant_ratio', 0)
        extreme_penalty = quality_metrics.get('extreme_values_ratio', 0)
        correlation_penalty = min(quality_metrics.get('high_correlation_pairs', 0) / 100, 0.2)
        
        # Calculate weighted quality score
        quality_score = (1 - infinite_penalty) * (1 - nan_penalty) * (1 - constant_penalty) * (1 - extreme_penalty) * (1 - correlation_penalty)
        
        return max(0.0, min(1.0, quality_score))
    
    def _generate_additional_feature_bank_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate additional features from the feature bank."""
        try:
            # Import additional feature bank components
            from src.feature_generation.categories import (
                create_default_returns_generators,
                create_default_volume_generators,
                create_default_entropy_generators,
                create_default_microstructure_generators,
                create_default_acceleration_generators
            )
            
            all_features = []
            
            # Generate returns features
            try:
                returns_generators = create_default_returns_generators()
                for generator in returns_generators:
                    try:
                        returns_features = generator.generate(data)
                        if not returns_features.empty:
                            all_features.append(returns_features)
                            tprint_debug(f"✅ Generated returns features: {returns_features.shape[1]} features")
                    except Exception as e:
                        tprint_warning(f"⚠️ Failed to generate returns features with {generator.__class__.__name__}: {e}")
            except Exception as e:
                tprint_warning(f"⚠️ Returns generators not available: {e}")
            
            # Generate volume features
            try:
                volume_generators = create_default_volume_generators()
                for generator in volume_generators:
                    try:
                        volume_features = generator.generate(data)
                        if not volume_features.empty:
                            all_features.append(volume_features)
                            tprint_debug(f"✅ Generated volume features: {volume_features.shape[1]} features")
                    except Exception as e:
                        tprint_warning(f"⚠️ Failed to generate volume features with {generator.__class__.__name__}: {e}")
            except Exception as e:
                tprint_warning(f"⚠️ Volume generators not available: {e}")
            
            # Generate entropy features
            try:
                entropy_generators = create_default_entropy_generators()
                for generator in entropy_generators:
                    try:
                        entropy_features = generator.generate(data)
                        if not entropy_features.empty:
                            all_features.append(entropy_features)
                            tprint_debug(f"✅ Generated entropy features: {entropy_features.shape[1]} features")
                    except Exception as e:
                        tprint_warning(f"⚠️ Failed to generate entropy features with {generator.__class__.__name__}: {e}")
            except Exception as e:
                tprint_warning(f"⚠️ Entropy generators not available: {e}")
            
            # Generate microstructure features
            try:
                microstructure_generators = create_default_microstructure_generators()
                for generator in microstructure_generators:
                    try:
                        microstructure_features = generator.generate(data)
                        if not microstructure_features.empty:
                            all_features.append(microstructure_features)
                            tprint_debug(f"✅ Generated microstructure features: {microstructure_features.shape[1]} features")
                    except Exception as e:
                        tprint_warning(f"⚠️ Failed to generate microstructure features with {generator.__class__.__name__}: {e}")
            except Exception as e:
                tprint_warning(f"⚠️ Microstructure generators not available: {e}")
            
            # Generate acceleration features
            try:
                acceleration_generators = create_default_acceleration_generators()
                for generator in acceleration_generators:
                    try:
                        acceleration_features = generator.generate(data)
                        if not acceleration_features.empty:
                            all_features.append(acceleration_features)
                            tprint_debug(f"✅ Generated acceleration features: {acceleration_features.shape[1]} features")
                    except Exception as e:
                        tprint_warning(f"⚠️ Failed to generate acceleration features with {generator.__class__.__name__}: {e}")
            except Exception as e:
                tprint_warning(f"⚠️ Acceleration generators not available: {e}")
            
            # Combine all additional features
            if all_features:
                result_df = pd.concat(all_features, axis=1)
                # Remove duplicate columns
                result_df = result_df.loc[:, ~result_df.columns.duplicated(keep='first')]
                result_df = self._optimize_dataframe_dtypes(result_df)
                tprint_success(f"✅ Generated {len(result_df.columns)} additional features from feature bank")
                return result_df
            else:
                tprint_warning("⚠️ No additional features generated from feature bank")
                return pd.DataFrame(index=data.index)
                
        except ImportError as e:
            tprint_warning(f"⚠️ Additional feature bank generators not available: {e}")
            return pd.DataFrame(index=data.index)
        except Exception as e:
            tprint_warning(f"⚠️ Additional feature bank generation failed: {e}")
            return pd.DataFrame(index=data.index)


# Convenience functions
def create_vectorbt_config(**kwargs) -> VectorBTFeatureConfig:
    """Create a VectorBT feature configuration."""
    return VectorBTFeatureConfig(**kwargs)


def generate_vectorbt_features(data: pd.DataFrame, config: Optional[VectorBTFeatureConfig] = None) -> pd.DataFrame:
    """
    Generate features using VectorBT optimizations and the feature bank.
    
    Args:
        data: Input OHLCV data
        config: VectorBT configuration
        
    Returns:
        DataFrame with generated features
    """
    generator = VectorBTFeatureGenerator(config)
    
    # Generate all types of features
    technical_features = generator.generate_technical_indicators(data)  # Uses feature bank
    rolling_features = generator.generate_rolling_features(data)
    cross_timeframe_features = generator.generate_cross_timeframe_features(data)
    interaction_features = generator.generate_interaction_features(data)
    
    # Try to generate additional features from the feature bank
    additional_features = generator._generate_additional_feature_bank_features(data)
    
    # Combine all features
    all_features = [technical_features, rolling_features, cross_timeframe_features, interaction_features, additional_features]
    valid_features = [f for f in all_features if not f.empty]
    
    if valid_features:
        result = pd.concat(valid_features, axis=1)
        # Remove duplicate columns
        result = result.loc[:, ~result.columns.duplicated(keep='first')]
    else:
        result = pd.DataFrame(index=data.index)
    
    return result


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'open': 100 + np.random.randn(n_samples).cumsum() * 0.1,
        'high': 100 + np.random.randn(n_samples).cumsum() * 0.1 + np.random.uniform(0, 0.5, n_samples),
        'low': 100 + np.random.randn(n_samples).cumsum() * 0.1 - np.random.uniform(0, 0.5, n_samples),
        'close': 100 + np.random.randn(n_samples).cumsum() * 0.1,
        'volume': np.random.uniform(1000, 10000, n_samples)
    })
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    # Test VectorBT feature generation
    config = VectorBTFeatureConfig(
        use_gpu=False,  # Set to True if GPU is available
        enable_parallel=True
    )
    
    try:
        features = generate_vectorbt_features(data, config)
        print(f"Generated {len(features.columns)} features")
        print(f"Feature columns: {list(features.columns)[:10]}...")
        
        # Validate features
        generator = VectorBTFeatureGenerator(config)
        validation = generator.validate_features(features)
        print(f"Validation passed: {validation['passed']}")
        print(f"Quality score: {validation['quality_score']:.3f}")
        
    except Exception as e:
        print(f"Error: {e}")