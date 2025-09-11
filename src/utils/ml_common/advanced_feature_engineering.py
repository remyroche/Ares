"""
Advanced Feature Engineering Utilities

This module provides advanced feature engineering capabilities for trading data,
including technical indicators, statistical features, and domain-specific transformations.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import gc
from scipy import stats
from scipy.signal import find_peaks
import talib

# Core utilities
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, safe_file_exists, ensure_directory,
    safe_mean, safe_std, safe_float, safe_int, get_current_datetime,
    safe_append, safe_extend, safe_dict_get, safe_lower, safe_upper,
    format_datetime, validate_file_path, get_file_size, check_disk_space
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, MathValidationError
)
from src.core.decorators import (
    handles_errors, validates, traced, log_execution_time, 
    timeout, error_boundary, compose, validate_data_quality, 
    monitor_step_execution, ensure_data_integrity, validate_pipeline_step
)
from src.utils.enhanced_error_handler import handle_errors_with_tracking
from src.utils.logger import system_logger

class FeatureType(Enum):
    """Feature engineering types."""
    TECHNICAL_INDICATORS = "technical_indicators"
    STATISTICAL_FEATURES = "statistical_features"
    TIME_SERIES_FEATURES = "time_series_features"
    VOLATILITY_FEATURES = "volatility_features"
    MOMENTUM_FEATURES = "momentum_features"
    VOLUME_FEATURES = "volume_features"
    CROSS_ASSET_FEATURES = "cross_asset_features"
    REGIME_FEATURES = "regime_features"

class FeatureComplexity(Enum):
    """Feature complexity levels."""
    SIMPLE = "simple"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    feature_type: FeatureType
    complexity: FeatureComplexity
    lookback_periods: List[int]
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    priority: int = 1

@dataclass
class FeatureResult:
    """Feature engineering result."""
    feature_name: str
    feature_values: np.ndarray
    feature_type: FeatureType
    complexity: FeatureComplexity
    lookback_period: int
    parameters: Dict[str, Any]
    computation_time: float
    quality_score: float

class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for trading data.
    
    This class provides comprehensive feature engineering capabilities including:
    - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
    - Statistical features (rolling statistics, percentiles, etc.)
    - Time series features (trends, seasonality, etc.)
    - Volatility features (GARCH, realized volatility, etc.)
    - Momentum features (rate of change, acceleration, etc.)
    - Volume features (volume profiles, accumulation, etc.)
    - Cross-asset features (correlations, spreads, etc.)
    - Regime features (market state indicators, etc.)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize advanced feature engineer."""
        self.config = config
        self.logger = system_logger.getChild('AdvancedFeatureEngineer')
        
        # Feature engineering configuration
        self.feature_config = config.get('feature_engineering', {})
        self.enabled_features = self.feature_config.get('enabled_features', {})
        self.complexity_level = self.feature_config.get('complexity_level', FeatureComplexity.INTERMEDIATE)
        self.max_features = self.feature_config.get('max_features', 1000)
        self.feature_selection_enabled = self.feature_config.get('feature_selection_enabled', True)
        
        # Feature registry
        self.feature_registry: Dict[str, Callable] = {}
        self.feature_results: List[FeatureResult] = []
        
        # Initialize feature registry
        self._initialize_feature_registry()

    def _initialize_feature_registry(self) -> None:
        """Initialize the feature engineering registry."""
        # Technical indicators
        self.feature_registry.update({
            'rsi': self._calculate_rsi,
            'macd': self._calculate_macd,
            'bollinger_bands': self._calculate_bollinger_bands,
            'stochastic': self._calculate_stochastic,
            'williams_r': self._calculate_williams_r,
            'cci': self._calculate_cci,
            'atr': self._calculate_atr,
            'adx': self._calculate_adx,
            'obv': self._calculate_obv,
            'mfi': self._calculate_mfi,
        })
        
        # Statistical features
        self.feature_registry.update({
            'rolling_mean': self._calculate_rolling_mean,
            'rolling_std': self._calculate_rolling_std,
            'rolling_skew': self._calculate_rolling_skew,
            'rolling_kurtosis': self._calculate_rolling_kurtosis,
            'rolling_percentile': self._calculate_rolling_percentile,
            'rolling_zscore': self._calculate_rolling_zscore,
            'rolling_correlation': self._calculate_rolling_correlation,
        })
        
        # Time series features
        self.feature_registry.update({
            'trend_strength': self._calculate_trend_strength,
            'seasonality': self._calculate_seasonality,
            'autocorrelation': self._calculate_autocorrelation,
            'partial_autocorrelation': self._calculate_partial_autocorrelation,
            'spectral_features': self._calculate_spectral_features,
        })
        
        # Volatility features
        self.feature_registry.update({
            'realized_volatility': self._calculate_realized_volatility,
            'parkinson_volatility': self._calculate_parkinson_volatility,
            'garman_klass_volatility': self._calculate_garman_klass_volatility,
            'rogers_satchell_volatility': self._calculate_rogers_satchell_volatility,
            'yang_zhang_volatility': self._calculate_yang_zhang_volatility,
        })
        
        # Momentum features
        self.feature_registry.update({
            'rate_of_change': self._calculate_rate_of_change,
            'momentum': self._calculate_momentum,
            'acceleration': self._calculate_acceleration,
            'jerk': self._calculate_jerk,
            'velocity': self._calculate_velocity,
        })
        
        # Volume features
        self.feature_registry.update({
            'volume_profile': self._calculate_volume_profile,
            'volume_weighted_price': self._calculate_volume_weighted_price,
            'accumulation_distribution': self._calculate_accumulation_distribution,
            'chaikin_money_flow': self._calculate_chaikin_money_flow,
            'ease_of_movement': self._calculate_ease_of_movement,
        })

    @handles_errors(Exception, fallback=False, log_level='ERROR')
    @validates(strict=True)
    @traced("engineer_features")
    @log_execution_time
    async def engineer_features(
        self, 
        data: pd.DataFrame,
        feature_configs: Optional[List[FeatureConfig]] = None
    ) -> pd.DataFrame:
        """
        Engineer features from trading data.
        
        Args:
            data: Trading data DataFrame
            feature_configs: List of feature configurations
            
        Returns:
            pd.DataFrame: DataFrame with engineered features
        """
        try:
            self.logger.info(f"🔧 Starting feature engineering for {len(data)} records")
            
            if feature_configs is None:
                feature_configs = self._get_default_feature_configs()
            
            # Initialize result DataFrame
            result_df = data.copy()
            self.feature_results = []
            
            # Engineer features
            for config in feature_configs:
                if not config.enabled:
                    continue
                
                try:
                    await self._engineer_feature_type(result_df, config)
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to engineer {config.feature_type.value}: {e}")
                    continue
            
            # Feature selection if enabled
            if self.feature_selection_enabled:
                result_df = await self._select_features(result_df)
            
            self.logger.info(f"✅ Feature engineering completed: {len(result_df.columns)} features")
            return result_df
            
        except Exception as e:
            self.logger.error(f"❌ Feature engineering failed: {e}")
            raise

    def _get_default_feature_configs(self) -> List[FeatureConfig]:
        """Get default feature configurations based on complexity level."""
        configs = []
        
        if self.complexity_level in [FeatureComplexity.SIMPLE, FeatureComplexity.INTERMEDIATE, FeatureComplexity.ADVANCED, FeatureComplexity.EXPERT]:
            # Technical indicators
            configs.append(FeatureConfig(
                feature_type=FeatureType.TECHNICAL_INDICATORS,
                complexity=FeatureComplexity.SIMPLE,
                lookback_periods=[14, 21, 30],
                parameters={'indicators': ['rsi', 'macd', 'bollinger_bands']}
            ))
        
        if self.complexity_level in [FeatureComplexity.INTERMEDIATE, FeatureComplexity.ADVANCED, FeatureComplexity.EXPERT]:
            # Statistical features
            configs.append(FeatureConfig(
                feature_type=FeatureType.STATISTICAL_FEATURES,
                complexity=FeatureComplexity.INTERMEDIATE,
                lookback_periods=[5, 10, 20, 50],
                parameters={'features': ['rolling_mean', 'rolling_std', 'rolling_skew']}
            ))
        
        if self.complexity_level in [FeatureComplexity.ADVANCED, FeatureComplexity.EXPERT]:
            # Volatility features
            configs.append(FeatureConfig(
                feature_type=FeatureType.VOLATILITY_FEATURES,
                complexity=FeatureComplexity.ADVANCED,
                lookback_periods=[10, 20, 30],
                parameters={'features': ['realized_volatility', 'parkinson_volatility']}
            ))
        
        if self.complexity_level == FeatureComplexity.EXPERT:
            # Advanced features
            configs.append(FeatureConfig(
                feature_type=FeatureType.TIME_SERIES_FEATURES,
                complexity=FeatureComplexity.EXPERT,
                lookback_periods=[20, 50, 100],
                parameters={'features': ['trend_strength', 'seasonality', 'spectral_features']}
            ))
        
        return configs

    @handles_errors(Exception, fallback=False, log_level='WARNING')
    @traced("engineer_feature_type")
    async def _engineer_feature_type(self, df: pd.DataFrame, config: FeatureConfig) -> None:
        """Engineer features of a specific type."""
        feature_type = config.feature_type
        
        if feature_type == FeatureType.TECHNICAL_INDICATORS:
            await self._engineer_technical_indicators(df, config)
        elif feature_type == FeatureType.STATISTICAL_FEATURES:
            await self._engineer_statistical_features(df, config)
        elif feature_type == FeatureType.VOLATILITY_FEATURES:
            await self._engineer_volatility_features(df, config)
        elif feature_type == FeatureType.TIME_SERIES_FEATURES:
            await self._engineer_time_series_features(df, config)
        elif feature_type == FeatureType.MOMENTUM_FEATURES:
            await self._engineer_momentum_features(df, config)
        elif feature_type == FeatureType.VOLUME_FEATURES:
            await self._engineer_volume_features(df, config)

    @handles_errors(Exception, fallback=False, log_level='WARNING')
    @traced("engineer_technical_indicators")
    async def _engineer_technical_indicators(self, df: pd.DataFrame, config: FeatureConfig) -> None:
        """Engineer technical indicators."""
        indicators = config.parameters.get('indicators', ['rsi', 'macd', 'bollinger_bands'])
        
        for indicator in indicators:
            if indicator in self.feature_registry:
                for period in config.lookback_periods:
                    try:
                        start_time = time.time()
                        feature_values = self.feature_registry[indicator](df, period)
                        computation_time = time.time() - start_time
                        
                        if feature_values is not None and len(feature_values) > 0:
                            feature_name = f"{indicator}_{period}"
                            df[feature_name] = feature_values
                            
                            # Store result
                            result = FeatureResult(
                                feature_name=feature_name,
                                feature_values=feature_values,
                                feature_type=FeatureType.TECHNICAL_INDICATORS,
                                complexity=config.complexity,
                                lookback_period=period,
                                parameters={'indicator': indicator},
                                computation_time=computation_time,
                                quality_score=self._calculate_feature_quality(feature_values)
                            )
                            self.feature_results.append(result)
                    
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to calculate {indicator} with period {period}: {e}")

    @handles_errors(Exception, fallback=False, log_level='WARNING')
    @traced("engineer_statistical_features")
    async def _engineer_statistical_features(self, df: pd.DataFrame, config: FeatureConfig) -> None:
        """Engineer statistical features."""
        features = config.parameters.get('features', ['rolling_mean', 'rolling_std'])
        
        for feature in features:
            if feature in self.feature_registry:
                for period in config.lookback_periods:
                    try:
                        start_time = time.time()
                        feature_values = self.feature_registry[feature](df, period)
                        computation_time = time.time() - start_time
                        
                        if feature_values is not None and len(feature_values) > 0:
                            feature_name = f"{feature}_{period}"
                            df[feature_name] = feature_values
                            
                            # Store result
                            result = FeatureResult(
                                feature_name=feature_name,
                                feature_values=feature_values,
                                feature_type=FeatureType.STATISTICAL_FEATURES,
                                complexity=config.complexity,
                                lookback_period=period,
                                parameters={'feature': feature},
                                computation_time=computation_time,
                                quality_score=self._calculate_feature_quality(feature_values)
                            )
                            self.feature_results.append(result)
                    
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to calculate {feature} with period {period}: {e}")

    @handles_errors(Exception, fallback=False, log_level='WARNING')
    @traced("engineer_volatility_features")
    async def _engineer_volatility_features(self, df: pd.DataFrame, config: FeatureConfig) -> None:
        """Engineer volatility features."""
        features = config.parameters.get('features', ['realized_volatility'])
        
        for feature in features:
            if feature in self.feature_registry:
                for period in config.lookback_periods:
                    try:
                        start_time = time.time()
                        feature_values = self.feature_registry[feature](df, period)
                        computation_time = time.time() - start_time
                        
                        if feature_values is not None and len(feature_values) > 0:
                            feature_name = f"{feature}_{period}"
                            df[feature_name] = feature_values
                            
                            # Store result
                            result = FeatureResult(
                                feature_name=feature_name,
                                feature_values=feature_values,
                                feature_type=FeatureType.VOLATILITY_FEATURES,
                                complexity=config.complexity,
                                lookback_period=period,
                                parameters={'feature': feature},
                                computation_time=computation_time,
                                quality_score=self._calculate_feature_quality(feature_values)
                            )
                            self.feature_results.append(result)
                    
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to calculate {feature} with period {period}: {e}")

    @handles_errors(Exception, fallback=False, log_level='WARNING')
    @traced("engineer_time_series_features")
    async def _engineer_time_series_features(self, df: pd.DataFrame, config: FeatureConfig) -> None:
        """Engineer time series features."""
        features = config.parameters.get('features', ['trend_strength'])
        
        for feature in features:
            if feature in self.feature_registry:
                for period in config.lookback_periods:
                    try:
                        start_time = time.time()
                        feature_values = self.feature_registry[feature](df, period)
                        computation_time = time.time() - start_time
                        
                        if feature_values is not None and len(feature_values) > 0:
                            feature_name = f"{feature}_{period}"
                            df[feature_name] = feature_values
                            
                            # Store result
                            result = FeatureResult(
                                feature_name=feature_name,
                                feature_values=feature_values,
                                feature_type=FeatureType.TIME_SERIES_FEATURES,
                                complexity=config.complexity,
                                lookback_period=period,
                                parameters={'feature': feature},
                                computation_time=computation_time,
                                quality_score=self._calculate_feature_quality(feature_values)
                            )
                            self.feature_results.append(result)
                    
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to calculate {feature} with period {period}: {e}")

    @handles_errors(Exception, fallback=False, log_level='WARNING')
    @traced("engineer_momentum_features")
    async def _engineer_momentum_features(self, df: pd.DataFrame, config: FeatureConfig) -> None:
        """Engineer momentum features."""
        features = config.parameters.get('features', ['rate_of_change'])
        
        for feature in features:
            if feature in self.feature_registry:
                for period in config.lookback_periods:
                    try:
                        start_time = time.time()
                        feature_values = self.feature_registry[feature](df, period)
                        computation_time = time.time() - start_time
                        
                        if feature_values is not None and len(feature_values) > 0:
                            feature_name = f"{feature}_{period}"
                            df[feature_name] = feature_values
                            
                            # Store result
                            result = FeatureResult(
                                feature_name=feature_name,
                                feature_values=feature_values,
                                feature_type=FeatureType.MOMENTUM_FEATURES,
                                complexity=config.complexity,
                                lookback_period=period,
                                parameters={'feature': feature},
                                computation_time=computation_time,
                                quality_score=self._calculate_feature_quality(feature_values)
                            )
                            self.feature_results.append(result)
                    
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to calculate {feature} with period {period}: {e}")

    @handles_errors(Exception, fallback=False, log_level='WARNING')
    @traced("engineer_volume_features")
    async def _engineer_volume_features(self, df: pd.DataFrame, config: FeatureConfig) -> None:
        """Engineer volume features."""
        features = config.parameters.get('features', ['volume_profile'])
        
        for feature in features:
            if feature in self.feature_registry:
                for period in config.lookback_periods:
                    try:
                        start_time = time.time()
                        feature_values = self.feature_registry[feature](df, period)
                        computation_time = time.time() - start_time
                        
                        if feature_values is not None and len(feature_values) > 0:
                            feature_name = f"{feature}_{period}"
                            df[feature_name] = feature_values
                            
                            # Store result
                            result = FeatureResult(
                                feature_name=feature_name,
                                feature_values=feature_values,
                                feature_type=FeatureType.VOLUME_FEATURES,
                                complexity=config.complexity,
                                lookback_period=period,
                                parameters={'feature': feature},
                                computation_time=computation_time,
                                quality_score=self._calculate_feature_quality(feature_values)
                            )
                            self.feature_results.append(result)
                    
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to calculate {feature} with period {period}: {e}")

    # Technical Indicator Calculations
    def _calculate_rsi(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate RSI indicator."""
        if 'close' not in df.columns:
            return None
        return talib.RSI(df['close'].values, timeperiod=period)

    def _calculate_macd(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate MACD indicator."""
        if 'close' not in df.columns:
            return None
        macd, _, _ = talib.MACD(df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
        return macd

    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate Bollinger Bands."""
        if 'close' not in df.columns:
            return None
        upper, middle, lower = talib.BBANDS(df['close'].values, timeperiod=period, nbdevup=2, nbdevdn=2)
        return (upper - lower) / middle  # Band width

    def _calculate_stochastic(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate Stochastic oscillator."""
        if not all(col in df.columns for col in ['high', 'low', 'close']):
            return None
        slowk, slowd = talib.STOCH(df['high'].values, df['low'].values, df['close'].values, 
                                  fastk_period=period, slowk_period=3, slowd_period=3)
        return slowk

    def _calculate_williams_r(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate Williams %R."""
        if not all(col in df.columns for col in ['high', 'low', 'close']):
            return None
        return talib.WILLR(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)

    def _calculate_cci(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate Commodity Channel Index."""
        if not all(col in df.columns for col in ['high', 'low', 'close']):
            return None
        return talib.CCI(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate Average True Range."""
        if not all(col in df.columns for col in ['high', 'low', 'close']):
            return None
        return talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)

    def _calculate_adx(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate Average Directional Index."""
        if not all(col in df.columns for col in ['high', 'low', 'close']):
            return None
        return talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)

    def _calculate_obv(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate On Balance Volume."""
        if not all(col in df.columns for col in ['close', 'volume']):
            return None
        obv = talib.OBV(df['close'].values, df['volume'].values)
        return pd.Series(obv).rolling(window=period).mean().values

    def _calculate_mfi(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate Money Flow Index."""
        if not all(col in df.columns for col in ['high', 'low', 'close', 'volume']):
            return None
        return talib.MFI(df['high'].values, df['low'].values, df['close'].values, df['volume'].values, timeperiod=period)

    # Statistical Feature Calculations
    def _calculate_rolling_mean(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate rolling mean."""
        if 'close' not in df.columns:
            return None
        return df['close'].rolling(window=period).mean().values

    def _calculate_rolling_std(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate rolling standard deviation."""
        if 'close' not in df.columns:
            return None
        return df['close'].rolling(window=period).std().values

    def _calculate_rolling_skew(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate rolling skewness."""
        if 'close' not in df.columns:
            return None
        return df['close'].rolling(window=period).skew().values

    def _calculate_rolling_kurtosis(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate rolling kurtosis."""
        if 'close' not in df.columns:
            return None
        return df['close'].rolling(window=period).kurt().values

    def _calculate_rolling_percentile(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate rolling percentile."""
        if 'close' not in df.columns:
            return None
        return df['close'].rolling(window=period).quantile(0.5).values

    def _calculate_rolling_zscore(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate rolling z-score."""
        if 'close' not in df.columns:
            return None
        rolling_mean = df['close'].rolling(window=period).mean()
        rolling_std = df['close'].rolling(window=period).std()
        return ((df['close'] - rolling_mean) / rolling_std).values

    def _calculate_rolling_correlation(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate rolling correlation between high and low."""
        if not all(col in df.columns for col in ['high', 'low']):
            return None
        return df['high'].rolling(window=period).corr(df['low']).values

    # Volatility Feature Calculations
    def _calculate_realized_volatility(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate realized volatility."""
        if 'close' not in df.columns:
            return None
        returns = df['close'].pct_change().dropna()
        return returns.rolling(window=period).std().values * np.sqrt(252)  # Annualized

    def _calculate_parkinson_volatility(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate Parkinson volatility."""
        if not all(col in df.columns for col in ['high', 'low']):
            return None
        log_hl = np.log(df['high'] / df['low'])
        return np.sqrt(log_hl.rolling(window=period).mean() / (4 * np.log(2))) * np.sqrt(252)

    def _calculate_garman_klass_volatility(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate Garman-Klass volatility."""
        if not all(col in df.columns for col in ['high', 'low', 'open', 'close']):
            return None
        log_hl = np.log(df['high'] / df['low'])
        log_co = np.log(df['close'] / df['open'])
        gk = 0.5 * log_hl**2 - (2*np.log(2) - 1) * log_co**2
        return np.sqrt(gk.rolling(window=period).mean()) * np.sqrt(252)

    def _calculate_rogers_satchell_volatility(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate Rogers-Satchell volatility."""
        if not all(col in df.columns for col in ['high', 'low', 'open', 'close']):
            return None
        log_ho = np.log(df['high'] / df['open'])
        log_hc = np.log(df['high'] / df['close'])
        log_lo = np.log(df['low'] / df['open'])
        log_lc = np.log(df['low'] / df['close'])
        rs = log_ho * log_hc + log_lo * log_lc
        return np.sqrt(rs.rolling(window=period).mean()) * np.sqrt(252)

    def _calculate_yang_zhang_volatility(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate Yang-Zhang volatility."""
        if not all(col in df.columns for col in ['high', 'low', 'open', 'close']):
            return None
        # Simplified Yang-Zhang implementation
        log_co = np.log(df['close'] / df['open'])
        log_oc = np.log(df['open'] / df['close'].shift(1))
        yz = log_co**2 + log_oc**2
        return np.sqrt(yz.rolling(window=period).mean()) * np.sqrt(252)

    # Time Series Feature Calculations
    def _calculate_trend_strength(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate trend strength."""
        if 'close' not in df.columns:
            return None
        # Linear regression slope as trend strength
        def slope(y):
            x = np.arange(len(y))
            return np.polyfit(x, y, 1)[0] if len(y) > 1 else 0
        
        return df['close'].rolling(window=period).apply(slope, raw=False).values

    def _calculate_seasonality(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate seasonality strength."""
        if 'close' not in df.columns:
            return None
        # Simplified seasonality calculation
        returns = df['close'].pct_change().dropna()
        seasonal = returns.rolling(window=period).apply(lambda x: np.std(x) if len(x) > 1 else 0, raw=False)
        return seasonal.values

    def _calculate_autocorrelation(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate autocorrelation."""
        if 'close' not in df.columns:
            return None
        returns = df['close'].pct_change().dropna()
        autocorr = returns.rolling(window=period).apply(lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False)
        return autocorr.values

    def _calculate_partial_autocorrelation(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate partial autocorrelation."""
        if 'close' not in df.columns:
            return None
        returns = df['close'].pct_change().dropna()
        # Simplified partial autocorrelation
        pacf = returns.rolling(window=period).apply(lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False)
        return pacf.values

    def _calculate_spectral_features(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate spectral features."""
        if 'close' not in df.columns:
            return None
        # Simplified spectral features - dominant frequency
        returns = df['close'].pct_change().dropna()
        def dominant_freq(x):
            if len(x) < 4:
                return 0
            fft = np.fft.fft(x)
            freqs = np.fft.fftfreq(len(x))
            return freqs[np.argmax(np.abs(fft[1:len(fft)//2])) + 1]
        
        spectral = returns.rolling(window=period).apply(dominant_freq, raw=False)
        return spectral.values

    # Momentum Feature Calculations
    def _calculate_rate_of_change(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate rate of change."""
        if 'close' not in df.columns:
            return None
        return df['close'].pct_change(periods=period).values

    def _calculate_momentum(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate momentum."""
        if 'close' not in df.columns:
            return None
        return (df['close'] - df['close'].shift(period)).values

    def _calculate_acceleration(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate acceleration (second derivative)."""
        if 'close' not in df.columns:
            return None
        momentum = df['close'] - df['close'].shift(period)
        acceleration = momentum - momentum.shift(period)
        return acceleration.values

    def _calculate_jerk(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate jerk (third derivative)."""
        if 'close' not in df.columns:
            return None
        momentum = df['close'] - df['close'].shift(period)
        acceleration = momentum - momentum.shift(period)
        jerk = acceleration - acceleration.shift(period)
        return jerk.values

    def _calculate_velocity(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate velocity (first derivative)."""
        if 'close' not in df.columns:
            return None
        return df['close'].diff(periods=period).values

    # Volume Feature Calculations
    def _calculate_volume_profile(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate volume profile."""
        if not all(col in df.columns for col in ['volume', 'close']):
            return None
        # Volume-weighted average price
        vwap = (df['volume'] * df['close']).rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
        return vwap.values

    def _calculate_volume_weighted_price(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate volume-weighted price."""
        if not all(col in df.columns for col in ['volume', 'close']):
            return None
        return self._calculate_volume_profile(df, period)

    def _calculate_accumulation_distribution(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate Accumulation/Distribution line."""
        if not all(col in df.columns for col in ['high', 'low', 'close', 'volume']):
            return None
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        clv = clv.fillna(0)
        ad = (clv * df['volume']).rolling(window=period).sum()
        return ad.values

    def _calculate_chaikin_money_flow(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate Chaikin Money Flow."""
        if not all(col in df.columns for col in ['high', 'low', 'close', 'volume']):
            return None
        mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
        mfv = mfv.fillna(0)
        cmf = mfv.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
        return cmf.values

    def _calculate_ease_of_movement(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate Ease of Movement."""
        if not all(col in df.columns for col in ['high', 'low', 'close', 'volume']):
            return None
        distance = (df['high'] + df['low']) / 2 - (df['high'].shift(1) + df['low'].shift(1)) / 2
        box_height = df['volume'] / (df['high'] - df['low'])
        box_height = box_height.replace([np.inf, -np.inf], np.nan).fillna(0)
        eom = distance / box_height
        return eom.rolling(window=period).mean().values

    def _calculate_feature_quality(self, feature_values: np.ndarray) -> float:
        """Calculate feature quality score."""
        if len(feature_values) == 0:
            return 0.0
        
        # Remove NaN values
        clean_values = feature_values[~np.isnan(feature_values)]
        if len(clean_values) == 0:
            return 0.0
        
        # Quality metrics
        completeness = len(clean_values) / len(feature_values)
        variance = np.var(clean_values)
        stability = 1.0 - (np.std(clean_values) / (np.mean(clean_values) + 1e-8))
        
        # Combined quality score
        quality_score = (completeness * 0.4 + min(variance, 1.0) * 0.3 + max(0, stability) * 0.3)
        return min(1.0, max(0.0, quality_score))

    @handles_errors(Exception, fallback=False, log_level='WARNING')
    @traced("select_features")
    async def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select best features based on quality scores."""
        if not self.feature_results:
            return df
        
        # Sort features by quality score
        sorted_features = sorted(self.feature_results, key=lambda x: x.quality_score, reverse=True)
        
        # Select top features
        selected_features = sorted_features[:self.max_features]
        selected_feature_names = [f.feature_name for f in selected_features]
        
        # Filter DataFrame to include only selected features
        original_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        available_original = [col for col in original_columns if col in df.columns]
        final_columns = available_original + selected_feature_names
        
        result_df = df[final_columns].copy()
        
        self.logger.info(f"📊 Selected {len(selected_features)} features from {len(self.feature_results)} candidates")
        
        return result_df

    @handles_errors(Exception, fallback=False, log_level='ERROR')
    @traced("save_feature_report")
    async def save_feature_report(self, output_dir: str) -> str:
        """Save feature engineering report."""
        ensure_directory(output_dir)
        
        report_data = {
            'feature_results': [
                {
                    'feature_name': result.feature_name,
                    'feature_type': result.feature_type.value,
                    'complexity': result.complexity.value,
                    'lookback_period': result.lookback_period,
                    'parameters': result.parameters,
                    'computation_time': result.computation_time,
                    'quality_score': result.quality_score
                }
                for result in self.feature_results
            ],
            'summary': {
                'total_features': len(self.feature_results),
                'feature_types': {
                    ft.value: len([r for r in self.feature_results if r.feature_type == ft])
                    for ft in FeatureType
                },
                'complexity_distribution': {
                    fc.value: len([r for r in self.feature_results if r.complexity == fc])
                    for fc in FeatureComplexity
                },
                'average_quality_score': safe_mean([r.quality_score for r in self.feature_results]) if self.feature_results else 0.0,
                'total_computation_time': sum(r.computation_time for r in self.feature_results)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        report_file = f"{output_dir}/feature_engineering_report.json"
        safe_json_dump(report_data, report_file, indent=2)
        
        self.logger.info(f"💾 Feature engineering report saved to: {report_file}")
        return report_file