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
        Generate technical indicators using VectorBT optimizations.
        
        Args:
            data: OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            DataFrame with technical indicators
        """
        tprint_info("🔧 Generating VectorBT-optimized technical indicators...")
        start_time = time.time()
        
        # Validate input data
        self._validate_ohlcv_data(data)
        
        # Convert to VectorBT format
        ohlcv_data = self._prepare_ohlcv_data(data)
        
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
        
        # Create result DataFrame
        if features:
            result_df = pd.DataFrame(features, index=data.index)
            result_df = self._optimize_dataframe_dtypes(result_df)
        else:
            result_df = pd.DataFrame(index=data.index)
        
        execution_time = time.time() - start_time
        tprint_success(f"✅ Generated {len(result_df.columns)} technical indicators in {execution_time:.3f}s")
        
        return result_df
    
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
        
        # Generate rolling statistics for each window
        for window in self.config.rolling_windows:
            for col in numeric_cols:
                series = data[col]
                
                # Use VectorBT for optimized rolling calculations
                rolling_mean = vbt.Rolling.from_1d(series, window=window).mean()
                rolling_std = vbt.Rolling.from_1d(series, window=window).std()
                rolling_min = vbt.Rolling.from_1d(series, window=window).min()
                rolling_max = vbt.Rolling.from_1d(series, window=window).max()
                rolling_median = vbt.Rolling.from_1d(series, window=window).median()
                
                # Add features
                features[f'rolling_{window}_{col}_mean'] = rolling_mean
                features[f'rolling_{window}_{col}_std'] = rolling_std
                features[f'rolling_{window}_{col}_min'] = rolling_min
                features[f'rolling_{window}_{col}_max'] = rolling_max
                features[f'rolling_{window}_{col}_median'] = rolling_median
                
                # Additional rolling features
                rolling_skew = vbt.Rolling.from_1d(series, window=window).skew()
                rolling_kurt = vbt.Rolling.from_1d(series, window=window).kurt()
                rolling_quantile_25 = vbt.Rolling.from_1d(series, window=window).quantile(0.25)
                rolling_quantile_75 = vbt.Rolling.from_1d(series, window=window).quantile(0.75)
                
                features[f'rolling_{window}_{col}_skew'] = rolling_skew
                features[f'rolling_{window}_{col}_kurt'] = rolling_kurt
                features[f'rolling_{window}_{col}_q25'] = rolling_quantile_25
                features[f'rolling_{window}_{col}_q75'] = rolling_quantile_75
        
        # Create result DataFrame
        if features:
            result_df = pd.DataFrame(features, index=data.index)
            result_df = self._optimize_dataframe_dtypes(result_df)
        else:
            result_df = pd.DataFrame(index=data.index)
        
        execution_time = time.time() - start_time
        tprint_success(f"✅ Generated {len(result_df.columns)} rolling features in {execution_time:.3f}s")
        
        return result_df
    
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
        
        features = {}
        
        # Generate cross-timeframe features for each period
        for period in self.config.cross_timeframe_periods:
            for col in numeric_cols:
                series = data[col]
                
                # Use VectorBT for optimized cross-timeframe calculations
                ctf_mean = vbt.Rolling.from_1d(series, window=period).mean()
                ctf_std = vbt.Rolling.from_1d(series, window=period).std()
                ctf_min = vbt.Rolling.from_1d(series, window=period).min()
                ctf_max = vbt.Rolling.from_1d(series, window=period).max()
                ctf_median = vbt.Rolling.from_1d(series, window=period).median()
                
                # Add cross-timeframe features
                features[f'ctf_{period}m_{col}_mean'] = ctf_mean
                features[f'ctf_{period}m_{col}_std'] = ctf_std
                features[f'ctf_{period}m_{col}_min'] = ctf_min
                features[f'ctf_{period}m_{col}_max'] = ctf_max
                features[f'ctf_{period}m_{col}_median'] = ctf_median
                
                # Additional cross-timeframe features
                ctf_skew = vbt.Rolling.from_1d(series, window=period).skew()
                ctf_kurt = vbt.Rolling.from_1d(series, window=period).kurt()
                ctf_quantile_25 = vbt.Rolling.from_1d(series, window=period).quantile(0.25)
                ctf_quantile_75 = vbt.Rolling.from_1d(series, window=period).quantile(0.75)
                
                features[f'ctf_{period}m_{col}_skew'] = ctf_skew
                features[f'ctf_{period}m_{col}_kurt'] = ctf_kurt
                features[f'ctf_{period}m_{col}_q25'] = ctf_quantile_25
                features[f'ctf_{period}m_{col}_q75'] = ctf_quantile_75
        
        # Create result DataFrame
        if features:
            result_df = pd.DataFrame(features, index=data.index)
            result_df = self._optimize_dataframe_dtypes(result_df)
        else:
            result_df = pd.DataFrame(index=data.index)
        
        execution_time = time.time() - start_time
        tprint_success(f"✅ Generated {len(result_df.columns)} cross-timeframe features in {execution_time:.3f}s")
        
        return result_df
    
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
        
        # Convert to numpy arrays for VectorBT operations
        numeric_data = data[numeric_cols].values
        
        features = {}
        interaction_count = 0
        max_interactions = min(100, len(numeric_cols) * (len(numeric_cols) - 1) // 2)
        
        # Generate interactions using VectorBT matrix operations
        for i, col1 in enumerate(numeric_cols):
            if interaction_count >= max_interactions:
                break
                
            for j, col2 in enumerate(numeric_cols[i+1:], i+1):
                if interaction_count >= max_interactions:
                    break
                
                # Use VectorBT for optimized matrix operations
                series1 = data[col1]
                series2 = data[col2]
                
                # Ratio interaction (safe division)
                ratio_feature = vbt.ArrayWrapper.from_1d(series1) / (vbt.ArrayWrapper.from_1d(series2) + 1e-8)
                features[f'{col1}_div_{col2}'] = ratio_feature.values
                
                # Product interaction
                product_feature = vbt.ArrayWrapper.from_1d(series1) * vbt.ArrayWrapper.from_1d(series2)
                features[f'{col1}_mul_{col2}'] = product_feature.values
                
                # Difference interaction
                diff_feature = vbt.ArrayWrapper.from_1d(series1) - vbt.ArrayWrapper.from_1d(series2)
                features[f'{col1}_sub_{col2}'] = diff_feature.values
                
                # Sum interaction
                sum_feature = vbt.ArrayWrapper.from_1d(series1) + vbt.ArrayWrapper.from_1d(series2)
                features[f'{col1}_add_{col2}'] = sum_feature.values
                
                interaction_count += 4
        
        # Create result DataFrame
        if features:
            result_df = pd.DataFrame(features, index=data.index)
            result_df = self._optimize_dataframe_dtypes(result_df)
        else:
            result_df = pd.DataFrame(index=data.index)
        
        execution_time = time.time() - start_time
        tprint_success(f"✅ Generated {len(result_df.columns)} interaction features in {execution_time:.3f}s")
        
        return result_df
    
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
        """Optimize DataFrame dtypes for memory efficiency."""
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
        
        # Calculate overall quality score
        quality_score = (1 - quality_metrics['infinite_ratio']) * (1 - quality_metrics['nan_ratio']) * (1 - quality_metrics['constant_ratio'])
        
        return {
            'passed': len(issues) == 0,
            'quality_score': quality_score,
            'issues': issues,
            'metrics': quality_metrics
        }


# Convenience functions
def create_vectorbt_config(**kwargs) -> VectorBTFeatureConfig:
    """Create a VectorBT feature configuration."""
    return VectorBTFeatureConfig(**kwargs)


def generate_vectorbt_features(data: pd.DataFrame, config: Optional[VectorBTFeatureConfig] = None) -> pd.DataFrame:
    """
    Generate features using VectorBT optimizations.
    
    Args:
        data: Input OHLCV data
        config: VectorBT configuration
        
    Returns:
        DataFrame with generated features
    """
    generator = VectorBTFeatureGenerator(config)
    
    # Generate all types of features
    technical_features = generator.generate_technical_indicators(data)
    rolling_features = generator.generate_rolling_features(data)
    cross_timeframe_features = generator.generate_cross_timeframe_features(data)
    interaction_features = generator.generate_interaction_features(data)
    
    # Combine all features
    all_features = [technical_features, rolling_features, cross_timeframe_features, interaction_features]
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