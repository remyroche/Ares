"""
NAS Feature Extractor for short-term trading regime detection.

This module provides enhanced feature extraction optimized for NAS-driven
clustering with micro-regime detection capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, RobustScaler
import talib
from datetime import datetime, timedelta

# Import matrix operations for optimized computations
from src.utils.matrix_operations import UnifiedMatrixOperations

# Import hardware optimization
from src.utils.hardware.unified_hardware_manager import (
    UnifiedHardwareManager, HardwareConfig, WorkloadType, OptimizationLevel
)
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager

logger = logging.getLogger(__name__)


@dataclass
class NASFeatureResult:
    """Result of NAS feature extraction."""
    features: np.ndarray
    feature_names: List[str]
    timestamps: np.ndarray
    regime_features: Dict[str, np.ndarray]
    micro_regime_features: Dict[str, np.ndarray]
    feature_metadata: Dict[str, Any]
    execution_time: float


class NASFeatureExtractor:
    """Enhanced feature extractor for NAS-driven clustering."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize NAS feature extractor.
        
        Args:
            config: Feature extraction configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize scalers
        self.scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        
        # Feature extraction settings
        self.exclude_complex_features = config.get('exclude_complex_features', True)
        self.include_technical_indicators = config.get('include_technical_indicators', True)
        self.include_volume_features = config.get('include_volume_features', True)
        self.include_volatility_features = config.get('include_volatility_features', True)
        self.include_momentum_features = config.get('include_momentum_features', True)
        self.include_trend_features = config.get('include_trend_features', True)
        
        # Timeframe settings
        self.timeframe = config.get('timeframe', '15m')
        self.micro_timeframe = config.get('micro_timeframe', '5m')
        
        # Initialize matrix operations for optimized computations
        self.matrix_ops = UnifiedMatrixOperations()
        self.logger.info("✅ Matrix operations initialized")
        
        # Initialize hardware optimization
        self.hardware_manager = None
        self.memory_optimizer = None
        self.cpu_optimizer = None
        self.gpu_manager = None
        
        if config.get('enable_hardware_acceleration', True):
            self._initialize_hardware_optimization()
        
        self.logger.info(f"✅ NAS Feature Extractor initialized for {self.timeframe} timeframe")
        self.logger.info(f"🖥️ Hardware optimization: {self.hardware_manager is not None}")
        self.logger.info(f"🔢 Matrix operations: {self.matrix_ops is not None}")
    
    def _initialize_hardware_optimization(self):
        """Initialize hardware optimization components."""
        try:
            # Initialize unified hardware manager
            hardware_config = HardwareConfig(
                cpu_optimization_level=OptimizationLevel.BALANCED,
                gpu_optimization_level=OptimizationLevel.BALANCED,
                memory_optimization_level=OptimizationLevel.BALANCED,
                memory_limit_gb=8.0,
                enable_adaptive_optimization=True,
                learning_enabled=True,
                auto_tuning_enabled=True
            )
            self.hardware_manager = UnifiedHardwareManager(hardware_config)
            
            # Initialize M1-specific optimizers
            self.memory_optimizer = get_m1_memory_optimizer()
            self.cpu_optimizer = get_m1_cpu_optimizer()
            self.gpu_manager = get_m1_gpu_manager()
            
            self.logger.info("✅ Hardware optimization components initialized")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Hardware optimization initialization failed: {e}")
            self.hardware_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
            self.gpu_manager = None
    
    def _optimize_data_array_with_matrix_ops(self, data_array: np.ndarray) -> np.ndarray:
        """Optimize data array using matrix operations."""
        try:
            # Use matrix operations for data preprocessing
            if self.matrix_ops:
                # Normalize data using matrix operations
                normalized_data = self.matrix_ops.matrix_normalize(data_array)
                self.logger.info("✅ Data array optimized with matrix operations")
                return normalized_data
            else:
                # Fallback to standard normalization
                return self.scaler.fit_transform(data_array)
        except Exception as e:
            self.logger.warning(f"⚠️ Matrix operations optimization failed: {e}")
            return data_array
    
    def _extract_features_with_matrix_ops(self, data_array: np.ndarray) -> np.ndarray:
        """Extract features using matrix operations for optimization."""
        try:
            if self.matrix_ops:
                # Use matrix operations for feature extraction
                features = self.matrix_ops.extract_technical_features(data_array)
                self.logger.info("✅ Features extracted with matrix operations")
                return features
            else:
                # Fallback to standard feature extraction
                return self._extract_base_features(data_array)
        except Exception as e:
            self.logger.warning(f"⚠️ Matrix operations feature extraction failed: {e}")
            return self._extract_base_features(data_array)
    
    def extract_features(self, data: Union[pd.DataFrame, np.ndarray], 
                        timestamps: Optional[np.ndarray] = None) -> NASFeatureResult:
        """Extract features optimized for NAS-driven clustering with matrix operations and hardware optimization.
        
        Args:
            data: Market data (DataFrame or numpy array)
            timestamps: Optional timestamps array
            
        Returns:
            NASFeatureResult with extracted features
        """
        import time
        start_time = time.time()
        
        # Start hardware optimization if available
        if self.hardware_manager:
            self.hardware_manager.start_optimization(
                workload_type=WorkloadType.FEATURE_ENGINEERING,
                optimization_level=OptimizationLevel.BALANCED
            )
        
        try:
            # Prepare data with matrix operations optimization
            if isinstance(data, pd.DataFrame):
                data_array = data.values
                if timestamps is None and 'timestamp' in data.columns:
                    timestamps = data['timestamp'].values
            else:
                data_array = data
                if timestamps is None:
                    timestamps = np.arange(len(data))
            
            # Optimize data array using matrix operations
            if self.matrix_ops:
                self.logger.info("🔢 Optimizing data array with matrix operations...")
                data_array = self._optimize_data_array_with_matrix_ops(data_array)
            
            # Extract base features
            base_features = self._extract_base_features(data_array)
            
            # Extract technical indicators
            technical_features = self._extract_technical_indicators(data_array)
            
            # Extract volume features
            volume_features = self._extract_volume_features(data_array)
            
            # Extract volatility features
            volatility_features = self._extract_volatility_features(data_array)
            
            # Extract momentum features
            momentum_features = self._extract_momentum_features(data_array)
            
            # Extract trend features
            trend_features = self._extract_trend_features(data_array)
            
            # Extract micro-regime features
            micro_regime_features = self._extract_micro_regime_features(data_array, timestamps)
            
            # Combine features
            all_features = []
            feature_names = []
            
            if base_features is not None:
                all_features.append(base_features)
                feature_names.extend([f'base_{i}' for i in range(base_features.shape[1])])
            
            if technical_features is not None:
                all_features.append(technical_features)
                feature_names.extend([f'tech_{i}' for i in range(technical_features.shape[1])])
            
            if volume_features is not None:
                all_features.append(volume_features)
                feature_names.extend([f'volume_{i}' for i in range(volume_features.shape[1])])
            
            if volatility_features is not None:
                all_features.append(volatility_features)
                feature_names.extend([f'volatility_{i}' for i in range(volatility_features.shape[1])])
            
            if momentum_features is not None:
                all_features.append(momentum_features)
                feature_names.extend([f'momentum_{i}' for i in range(momentum_features.shape[1])])
            
            if trend_features is not None:
                all_features.append(trend_features)
                feature_names.extend([f'trend_{i}' for i in range(trend_features.shape[1])])
            
            if micro_regime_features is not None:
                all_features.append(micro_regime_features)
                feature_names.extend([f'micro_{i}' for i in range(micro_regime_features.shape[1])])
            
            # Combine all features
            if all_features:
                combined_features = np.column_stack(all_features)
            else:
                combined_features = data_array
            
            # Normalize features
            if self.config.get('normalize_features', True):
                combined_features = self._normalize_features(combined_features)
            
            # Create regime features dictionary
            regime_features = {
                'base': base_features,
                'technical': technical_features,
                'volume': volume_features,
                'volatility': volatility_features,
                'momentum': momentum_features,
                'trend': trend_features
            }
            
            # Create micro-regime features dictionary
            micro_regime_features_dict = {
                'micro_regime': micro_regime_features
            }
            
            execution_time = time.time() - start_time
            
            # Create result
            result = NASFeatureResult(
                features=combined_features,
                feature_names=feature_names,
                timestamps=timestamps,
                regime_features=regime_features,
                micro_regime_features=micro_regime_features_dict,
                feature_metadata={
                    'timeframe': self.timeframe,
                    'micro_timeframe': self.micro_timeframe,
                    'feature_count': len(feature_names),
                    'exclude_complex_features': self.exclude_complex_features,
                    'normalization_applied': self.config.get('normalize_features', True)
                },
                execution_time=execution_time
            )
            
            self.logger.info(f"✅ NAS feature extraction completed: {len(feature_names)} features in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ NAS feature extraction failed: {e}")
            return NASFeatureResult(
                features=np.array([]),
                feature_names=[],
                timestamps=timestamps,
                regime_features={},
                micro_regime_features={},
                feature_metadata={'error': str(e)},
                execution_time=execution_time
            )
    
    def _extract_base_features(self, data: np.ndarray) -> Optional[np.ndarray]:
        """Extract base features from market data."""
        try:
            if data.shape[1] < 4:  # Need OHLC data
                return None
            
            # Extract OHLC features
            open_price = data[:, 0]
            high_price = data[:, 1]
            low_price = data[:, 2]
            close_price = data[:, 3]
            
            # Basic price features
            price_range = high_price - low_price
            price_change = close_price - open_price
            price_change_pct = price_change / open_price
            
            # Body and wick features
            body_size = np.abs(price_change)
            upper_wick = high_price - np.maximum(open_price, close_price)
            lower_wick = np.minimum(open_price, close_price) - low_price
            
            # Combine base features
            base_features = np.column_stack([
                price_range, price_change, price_change_pct,
                body_size, upper_wick, lower_wick
            ])
            
            return base_features
            
        except Exception as e:
            self.logger.warning(f"⚠️ Base feature extraction failed: {e}")
            return None
    
    def _extract_technical_indicators(self, data: np.ndarray) -> Optional[np.ndarray]:
        """Extract technical indicators."""
        try:
            if not self.include_technical_indicators or data.shape[1] < 4:
                return None
            
            close_price = data[:, 3]
            high_price = data[:, 1]
            low_price = data[:, 2]
            volume = data[:, 4] if data.shape[1] > 4 else np.ones(len(close_price))
            
            technical_features = []
            
            # Moving averages
            if len(close_price) >= 20:
                sma_5 = talib.SMA(close_price, timeperiod=5)
                sma_10 = talib.SMA(close_price, timeperiod=10)
                sma_20 = talib.SMA(close_price, timeperiod=20)
                technical_features.extend([sma_5, sma_10, sma_20])
            
            # RSI
            if len(close_price) >= 14:
                rsi = talib.RSI(close_price, timeperiod=14)
                technical_features.append(rsi)
            
            # MACD
            if len(close_price) >= 26:
                macd, macd_signal, macd_hist = talib.MACD(close_price)
                technical_features.extend([macd, macd_signal, macd_hist])
            
            # Bollinger Bands
            if len(close_price) >= 20:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(close_price)
                technical_features.extend([bb_upper, bb_middle, bb_lower])
            
            # Stochastic
            if len(close_price) >= 14:
                stoch_k, stoch_d = talib.STOCH(high_price, low_price, close_price)
                technical_features.extend([stoch_k, stoch_d])
            
            if technical_features:
                return np.column_stack(technical_features)
            else:
                return None
                
        except Exception as e:
            self.logger.warning(f"⚠️ Technical indicator extraction failed: {e}")
            return None
    
    def _extract_volume_features(self, data: np.ndarray) -> Optional[np.ndarray]:
        """Extract volume-related features."""
        try:
            if not self.include_volume_features or data.shape[1] < 5:
                return None
            
            volume = data[:, 4]
            close_price = data[:, 3]
            
            volume_features = []
            
            # Volume moving averages
            if len(volume) >= 10:
                volume_sma_5 = talib.SMA(volume, timeperiod=5)
                volume_sma_10 = talib.SMA(volume, timeperiod=10)
                volume_features.extend([volume_sma_5, volume_sma_10])
            
            # Volume ratio
            if len(volume) >= 5:
                volume_ratio = volume / talib.SMA(volume, timeperiod=5)
                volume_features.append(volume_ratio)
            
            # Volume price trend
            if len(volume) >= 5:
                vpt = talib.VPT(close_price, volume)
                volume_features.append(vpt)
            
            # On Balance Volume
            if len(volume) >= 5:
                obv = talib.OBV(close_price, volume)
                volume_features.append(obv)
            
            if volume_features:
                return np.column_stack(volume_features)
            else:
                return None
                
        except Exception as e:
            self.logger.warning(f"⚠️ Volume feature extraction failed: {e}")
            return None
    
    def _extract_volatility_features(self, data: np.ndarray) -> Optional[np.ndarray]:
        """Extract volatility-related features."""
        try:
            if not self.include_volatility_features or data.shape[1] < 4:
                return None
            
            close_price = data[:, 3]
            high_price = data[:, 1]
            low_price = data[:, 2]
            
            volatility_features = []
            
            # ATR (Average True Range)
            if len(close_price) >= 14:
                atr = talib.ATR(high_price, low_price, close_price, timeperiod=14)
                volatility_features.append(atr)
            
            # NATR (Normalized ATR)
            if len(close_price) >= 14:
                natr = talib.NATR(high_price, low_price, close_price, timeperiod=14)
                volatility_features.append(natr)
            
            # Rolling volatility
            if len(close_price) >= 20:
                returns = np.diff(close_price) / close_price[:-1]
                rolling_vol = talib.SMA(returns, timeperiod=20)
                volatility_features.append(rolling_vol)
            
            # Volatility ratio
            if len(close_price) >= 20:
                short_vol = talib.SMA(np.abs(np.diff(close_price)), timeperiod=5)
                long_vol = talib.SMA(np.abs(np.diff(close_price)), timeperiod=20)
                vol_ratio = short_vol / long_vol
                volatility_features.append(vol_ratio)
            
            if volatility_features:
                return np.column_stack(volatility_features)
            else:
                return None
                
        except Exception as e:
            self.logger.warning(f"⚠️ Volatility feature extraction failed: {e}")
            return None
    
    def _extract_momentum_features(self, data: np.ndarray) -> Optional[np.ndarray]:
        """Extract momentum-related features."""
        try:
            if not self.include_momentum_features or data.shape[1] < 4:
                return None
            
            close_price = data[:, 3]
            high_price = data[:, 1]
            low_price = data[:, 2]
            
            momentum_features = []
            
            # Rate of Change
            if len(close_price) >= 10:
                roc = talib.ROC(close_price, timeperiod=10)
                momentum_features.append(roc)
            
            # Momentum
            if len(close_price) >= 10:
                momentum = talib.MOM(close_price, timeperiod=10)
                momentum_features.append(momentum)
            
            # Williams %R
            if len(close_price) >= 14:
                willr = talib.WILLR(high_price, low_price, close_price, timeperiod=14)
                momentum_features.append(willr)
            
            # Commodity Channel Index
            if len(close_price) >= 14:
                cci = talib.CCI(high_price, low_price, close_price, timeperiod=14)
                momentum_features.append(cci)
            
            if momentum_features:
                return np.column_stack(momentum_features)
            else:
                return None
                
        except Exception as e:
            self.logger.warning(f"⚠️ Momentum feature extraction failed: {e}")
            return None
    
    def _extract_trend_features(self, data: np.ndarray) -> Optional[np.ndarray]:
        """Extract trend-related features."""
        try:
            if not self.include_trend_features or data.shape[1] < 4:
                return None
            
            close_price = data[:, 3]
            high_price = data[:, 1]
            low_price = data[:, 2]
            
            trend_features = []
            
            # ADX (Average Directional Index)
            if len(close_price) >= 14:
                adx = talib.ADX(high_price, low_price, close_price, timeperiod=14)
                trend_features.append(adx)
            
            # Plus DI and Minus DI
            if len(close_price) >= 14:
                plus_di = talib.PLUS_DI(high_price, low_price, close_price, timeperiod=14)
                minus_di = talib.MINUS_DI(high_price, low_price, close_price, timeperiod=14)
                trend_features.extend([plus_di, minus_di])
            
            # Aroon
            if len(close_price) >= 14:
                aroon_up, aroon_down = talib.AROON(high_price, low_price, timeperiod=14)
                trend_features.extend([aroon_up, aroon_down])
            
            # Parabolic SAR
            if len(close_price) >= 14:
                sar = talib.SAR(high_price, low_price)
                trend_features.append(sar)
            
            if trend_features:
                return np.column_stack(trend_features)
            else:
                return None
                
        except Exception as e:
            self.logger.warning(f"⚠️ Trend feature extraction failed: {e}")
            return None
    
    def _extract_micro_regime_features(self, data: np.ndarray, timestamps: np.ndarray) -> Optional[np.ndarray]:
        """Extract micro-regime features for subtle market changes."""
        try:
            if data.shape[1] < 4:
                return None
            
            close_price = data[:, 3]
            high_price = data[:, 1]
            low_price = data[:, 2]
            volume = data[:, 4] if data.shape[1] > 4 else np.ones(len(close_price))
            
            micro_features = []
            
            # Price acceleration (second derivative)
            if len(close_price) >= 3:
                price_acceleration = np.diff(np.diff(close_price))
                micro_features.append(price_acceleration)
            
            # Volume acceleration
            if len(volume) >= 3:
                volume_acceleration = np.diff(np.diff(volume))
                micro_features.append(volume_acceleration)
            
            # Micro-trend changes
            if len(close_price) >= 5:
                micro_trend = np.diff(close_price, n=2)  # 2-period difference
                micro_features.append(micro_trend)
            
            # Micro-volatility
            if len(close_price) >= 5:
                micro_volatility = np.abs(np.diff(close_price, n=2))
                micro_features.append(micro_volatility)
            
            # Micro-volume spikes
            if len(volume) >= 5:
                volume_ma = talib.SMA(volume, timeperiod=5)
                micro_volume_spike = (volume - volume_ma) / volume_ma
                micro_features.append(micro_volume_spike)
            
            if micro_features:
                return np.column_stack(micro_features)
            else:
                return None
                
        except Exception as e:
            self.logger.warning(f"⚠️ Micro-regime feature extraction failed: {e}")
            return None
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using robust scaling."""
        try:
            # Use robust scaling to handle outliers
            normalized_features = self.robust_scaler.fit_transform(features)
            return normalized_features
        except Exception as e:
            self.logger.warning(f"⚠️ Feature normalization failed: {e}")
            return features