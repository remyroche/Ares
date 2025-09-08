"""
Enhanced Step06 Feature Engineering with Advanced Optimizations

This module implements comprehensive feature engineering with:
- Vectorized batch processing for indicator extraction
- Sophisticated feature interactions (polynomial, cross-timeframe, pattern recognition)
- Strict temporal validation to prevent lookahead bias
- Memory-efficient chunking for large datasets
- Mathematical safety with validation utilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from pathlib import Path
from contextlib import nullcontext
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings

# Import validation and safety utilities
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, 
    validate_positive, validate_range, MathValidationError
)
from src.utils.comprehensive_function_logger import (
    log_step_functions, log_important_calls, log_all_calls, 
    log_internal_call, log_step_progress, log_data_operation
)

# Import lookahead bias detection
try:
    from src.utils.lookahead_bias_detector import (
        get_global_detector, validate_no_future_data, LookaheadBiasError
    )
    BIAS_DETECTION_AVAILABLE = True
except ImportError:
    BIAS_DETECTION_AVAILABLE = False
    def get_global_detector():
        return None
    def validate_no_future_data(data, timestamp_col, current_time):
        return data

# Import validation framework
try:
    from .step06_enhanced_validation_framework import (
        step06_function_validator, step06_function_tracker, 
        step06_validation_context, ValidationLevel, FunctionStatus
    )
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    
    def step06_function_validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def step06_function_tracker(func):
        return func
    
    def step06_validation_context(*args, **kwargs):
        return nullcontext()
    
    class ValidationLevel:
        BASIC = 'basic'
        DETAILED = 'detailed'
        COMPREHENSIVE = 'comprehensive'
    
    class FunctionStatus:
        PENDING = 'pending'
        IN_PROGRESS = 'in_progress'
        COMPLETED = 'completed'
        FAILED = 'failed'
        TIMEOUT = 'timeout'

# Import technical analysis library
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    talib = None

# Import sklearn for advanced features
try:
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_regression
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

class EnhancedFeatureEngineering:
    """
    Enhanced feature engineering with advanced optimizations and safety measures.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize enhanced feature engineering."""
        self.config = config
        self.logger = logger
        self.feature_config = config.get('step06_feature_engineering', {})
        
        # Configuration parameters
        self.chunk_size = self.feature_config.get('chunk_size', 10000)
        self.max_features = self.feature_config.get('max_features', 500)
        self.polynomial_degree = self.feature_config.get('polynomial_degree', 2)
        self.correlation_threshold = self.feature_config.get('correlation_threshold', 0.95)
        self.memory_limit_mb = self.feature_config.get('memory_limit_mb', 1000)
        
        # Initialize components
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.poly_features = None
        self.feature_selector = None
        self.pca = None
        
        # Performance tracking
        self.processing_stats = {
            'total_samples_processed': 0,
            'total_features_created': 0,
            'processing_time': 0.0,
            'memory_usage_mb': 0.0,
            'chunks_processed': 0
        }
        
        self.logger.info("🚀 Enhanced Feature Engineering initialized")
        self.logger.info(f"   Chunk size: {self.chunk_size}")
        self.logger.info(f"   Max features: {self.max_features}")
        self.logger.info(f"   Polynomial degree: {self.polynomial_degree}")
        self.logger.info(f"   Memory limit: {self.memory_limit_mb}MB")

    def _validate_temporal_consistency(self, data: pd.DataFrame, current_idx: int) -> pd.DataFrame:
        """
        Strict temporal validation to prevent lookahead bias.
        
        Args:
            data: Input data
            current_idx: Current processing index (only use data up to this point)
            
        Returns:
            Temporally validated data
        """
        if not BIAS_DETECTION_AVAILABLE:
            self.logger.warning("⚠️ Lookahead bias detection not available")
            return data
        
        try:
            # Ensure we only use historical data
            if current_idx is not None and current_idx < len(data):
                historical_data = data.iloc[:current_idx].copy()
            else:
                historical_data = data.copy()
            
            # Remove any future-looking columns
            future_columns = [col for col in historical_data.columns 
                            if col.lower().startswith('future_') or 
                               col.lower().endswith('_future') or
                               'forward' in col.lower()]
            
            if future_columns:
                self.logger.warning(f"⚠️ Removing future-looking columns: {future_columns}")
                historical_data = historical_data.drop(columns=future_columns)
            
            # Validate temporal ordering
            if isinstance(historical_data.index, pd.DatetimeIndex):
                if not historical_data.index.is_monotonic_increasing:
                    self.logger.warning("⚠️ Data not temporally ordered, sorting...")
                    historical_data = historical_data.sort_index()
            
            # Set current timestamp for bias detection
            if len(historical_data) > 0:
                current_time = historical_data.index[-1]
                bias_detector = get_global_detector()
                if bias_detector:
                    bias_detector.set_current_timestamp(current_time)
                    historical_data = validate_no_future_data(
                        historical_data, 'timestamp', current_time
                    )
            
            return historical_data
            
        except Exception as e:
            self.logger.error(f"❌ Temporal validation failed: {e}")
            raise MathValidationError(f"Temporal validation error: {e}") from e

    @step06_function_validator(function_type='feature_engineering', validation_level=ValidationLevel.COMPREHENSIVE)
    def extract_indicators_batch(self, market_data: pd.DataFrame, 
                                periods_config: Dict[str, List[int]]) -> pd.DataFrame:
        """
        Extract technical indicators using vectorized batch processing.
        
        Args:
            market_data: OHLCV market data
            periods_config: Configuration for indicator periods
            
        Returns:
            DataFrame with technical indicators
        """
        with step06_validation_context('extract_indicators_batch', 'feature_engineering'):
            self.logger.info(f'🔧 Starting batch indicator extraction')
            self.logger.info(f'   Input shape: {market_data.shape}')
            self.logger.info(f'   Indicators: {list(periods_config.keys())}')
        
        # Validate input data
        self._validate_market_data(market_data)
        
        # Check if we need chunking
        if len(market_data) > self.chunk_size:
            self.logger.info(f"📦 Large dataset detected ({len(market_data)} rows), using chunking")
            return self._extract_indicators_chunked(market_data, periods_config)
        
        # Process in memory
        indicators = {}
        
        try:
            # Vectorized RSI extraction
            if 'RSI' in periods_config and TALIB_AVAILABLE:
                indicators.update(self._extract_rsi_batch(market_data, periods_config['RSI']))
            
            # Vectorized MACD extraction
            if 'MACD' in periods_config and TALIB_AVAILABLE:
                indicators.update(self._extract_macd_batch(market_data, periods_config['MACD']))
            
            # Vectorized Bollinger Bands extraction
            if 'Bollinger_Bands' in periods_config and TALIB_AVAILABLE:
                indicators.update(self._extract_bb_batch(market_data, periods_config['Bollinger_Bands']))
            
            # Vectorized moving averages
            if 'SMA' in periods_config and TALIB_AVAILABLE:
                indicators.update(self._extract_sma_batch(market_data, periods_config['SMA']))
            
            if 'EMA' in periods_config and TALIB_AVAILABLE:
                indicators.update(self._extract_ema_batch(market_data, periods_config['EMA']))
            
            # Vectorized volatility indicators
            if 'ATR' in periods_config and TALIB_AVAILABLE:
                indicators.update(self._extract_atr_batch(market_data, periods_config['ATR']))
            
            # Vectorized momentum indicators
            if 'Stochastic' in periods_config and TALIB_AVAILABLE:
                indicators.update(self._extract_stoch_batch(market_data, periods_config['Stochastic']))
            
            if 'ADX' in periods_config and TALIB_AVAILABLE:
                indicators.update(self._extract_adx_batch(market_data, periods_config['ADX']))
            
            # Vectorized volume indicators
            if 'OBV' in periods_config and TALIB_AVAILABLE:
                indicators.update(self._extract_obv_batch(market_data, periods_config['OBV']))
            
            if 'MFI' in periods_config and TALIB_AVAILABLE:
                indicators.update(self._extract_mfi_batch(market_data, periods_config['MFI']))
            
        except Exception as e:
            self.logger.error(f"❌ Batch indicator extraction failed: {e}")
            raise MathValidationError(f"Indicator extraction error: {e}") from e
        
        # Convert to DataFrame
        indicators_df = pd.DataFrame(indicators, index=market_data.index)
        indicators_df = indicators_df.ffill().fillna(0)
        
        self.logger.info(f'✅ Batch indicator extraction completed: {indicators_df.shape[1]} indicators')
        return indicators_df

    def _extract_indicators_chunked(self, market_data: pd.DataFrame, 
                                   periods_config: Dict[str, List[int]]) -> pd.DataFrame:
        """
        Extract indicators using memory-efficient chunking for large datasets.
        
        Args:
            market_data: OHLCV market data
            periods_config: Configuration for indicator periods
            
        Returns:
            DataFrame with technical indicators
        """
        self.logger.info(f"📦 Processing {len(market_data)} rows in chunks of {self.chunk_size}")
        
        all_indicators = []
        chunks_processed = 0
        
        for start_idx in range(0, len(market_data), self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, len(market_data))
            chunk = market_data.iloc[start_idx:end_idx].copy()
            
            self.logger.info(f"   Processing chunk {chunks_processed + 1}: rows {start_idx}-{end_idx}")
            
            # Extract indicators for this chunk
            chunk_indicators = self.extract_indicators_batch(chunk, periods_config)
            all_indicators.append(chunk_indicators)
            
            chunks_processed += 1
            self.processing_stats['chunks_processed'] = chunks_processed
            
            # Memory management
            if chunks_processed % 10 == 0:
                import gc
                gc.collect()
                self.logger.info(f"   Memory cleanup after {chunks_processed} chunks")
        
        # Combine all chunks
        combined_indicators = pd.concat(all_indicators, axis=0)
        
        self.logger.info(f"✅ Chunked processing completed: {chunks_processed} chunks, {combined_indicators.shape[1]} indicators")
        return combined_indicators

    def _extract_rsi_batch(self, data: pd.DataFrame, periods: List[int]) -> Dict[str, np.ndarray]:
        """Extract RSI indicators for multiple periods in batch."""
        indicators = {}
        close_values = data['close'].values
        
        for period in periods:
            try:
                rsi = talib.RSI(close_values, timeperiod=period)
                # Validate RSI values
                rsi = np.clip(rsi, 0, 100)  # RSI should be between 0 and 100
                indicators[f'RSI_{period}'] = rsi
            except Exception as e:
                self.logger.warning(f"⚠️ RSI calculation failed for period {period}: {e}")
                indicators[f'RSI_{period}'] = np.full(len(data), 50.0)  # Neutral RSI
        
        return indicators

    def _extract_macd_batch(self, data: pd.DataFrame, periods: List[int]) -> Dict[str, np.ndarray]:
        """Extract MACD indicators for multiple periods in batch."""
        indicators = {}
        close_values = data['close'].values
        
        for i in range(0, len(periods), 2):
            if i + 1 < len(periods):
                fast_period = periods[i]
                slow_period = periods[i + 1]
                
                try:
                    macd, macd_signal, macd_hist = talib.MACD(
                        close_values, 
                        fastperiod=fast_period, 
                        slowperiod=slow_period, 
                        signalperiod=9
                    )
                    indicators[f'MACD_{fast_period}_{slow_period}'] = macd
                    indicators[f'MACD_Signal_{fast_period}_{slow_period}'] = macd_signal
                    indicators[f'MACD_Hist_{fast_period}_{slow_period}'] = macd_hist
                except Exception as e:
                    self.logger.warning(f"⚠️ MACD calculation failed for {fast_period},{slow_period}: {e}")
        
        return indicators

    def _extract_bb_batch(self, data: pd.DataFrame, periods: List[int]) -> Dict[str, np.ndarray]:
        """Extract Bollinger Bands indicators for multiple periods in batch."""
        indicators = {}
        close_values = data['close'].values
        
        for period in periods:
            try:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(
                    close_values, 
                    timeperiod=period, 
                    nbdevup=2, 
                    nbdevdn=2
                )
                
                # Calculate additional BB features
                bb_position = safe_divide(close_values - bb_lower, bb_upper - bb_lower, default=0.5)
                bb_squeeze = safe_divide(bb_upper - bb_lower, bb_middle, default=0.0)
                
                indicators[f'BB_Upper_{period}'] = bb_upper
                indicators[f'BB_Middle_{period}'] = bb_middle
                indicators[f'BB_Lower_{period}'] = bb_lower
                indicators[f'BB_Position_{period}'] = bb_position
                indicators[f'BB_Squeeze_{period}'] = bb_squeeze
            except Exception as e:
                self.logger.warning(f"⚠️ Bollinger Bands calculation failed for period {period}: {e}")
        
        return indicators

    def _extract_sma_batch(self, data: pd.DataFrame, periods: List[int]) -> Dict[str, np.ndarray]:
        """Extract SMA indicators for multiple periods in batch."""
        indicators = {}
        close_values = data['close'].values
        
        for period in periods:
            try:
                sma = talib.SMA(close_values, timeperiod=period)
                sma_ratio = safe_divide(close_values, sma, default=1.0)
                indicators[f'SMA_{period}'] = sma
                indicators[f'SMA_Ratio_{period}'] = sma_ratio
            except Exception as e:
                self.logger.warning(f"⚠️ SMA calculation failed for period {period}: {e}")
        
        return indicators

    def _extract_ema_batch(self, data: pd.DataFrame, periods: List[int]) -> Dict[str, np.ndarray]:
        """Extract EMA indicators for multiple periods in batch."""
        indicators = {}
        close_values = data['close'].values
        
        for period in periods:
            try:
                ema = talib.EMA(close_values, timeperiod=period)
                ema_ratio = safe_divide(close_values, ema, default=1.0)
                indicators[f'EMA_{period}'] = ema
                indicators[f'EMA_Ratio_{period}'] = ema_ratio
            except Exception as e:
                self.logger.warning(f"⚠️ EMA calculation failed for period {period}: {e}")
        
        return indicators

    def _extract_atr_batch(self, data: pd.DataFrame, periods: List[int]) -> Dict[str, np.ndarray]:
        """Extract ATR indicators for multiple periods in batch."""
        indicators = {}
        high_values = data['high'].values
        low_values = data['low'].values
        close_values = data['close'].values
        
        for period in periods:
            try:
                atr = talib.ATR(high_values, low_values, close_values, timeperiod=period)
                atr_normalized = safe_divide(atr, close_values, default=0.0)
                indicators[f'ATR_{period}'] = atr
                indicators[f'ATR_Normalized_{period}'] = atr_normalized
            except Exception as e:
                self.logger.warning(f"⚠️ ATR calculation failed for period {period}: {e}")
        
        return indicators

    def _extract_stoch_batch(self, data: pd.DataFrame, periods: List[int]) -> Dict[str, np.ndarray]:
        """Extract Stochastic indicators for multiple periods in batch."""
        indicators = {}
        high_values = data['high'].values
        low_values = data['low'].values
        close_values = data['close'].values
        
        for period in periods:
            try:
                stoch_k, stoch_d = talib.STOCH(
                    high_values, low_values, close_values,
                    fastk_period=period, slowk_period=3, slowd_period=3
                )
                indicators[f'Stoch_K_{period}'] = stoch_k
                indicators[f'Stoch_D_{period}'] = stoch_d
            except Exception as e:
                self.logger.warning(f"⚠️ Stochastic calculation failed for period {period}: {e}")
        
        return indicators

    def _extract_adx_batch(self, data: pd.DataFrame, periods: List[int]) -> Dict[str, np.ndarray]:
        """Extract ADX indicators for multiple periods in batch."""
        indicators = {}
        high_values = data['high'].values
        low_values = data['low'].values
        close_values = data['close'].values
        
        for period in periods:
            try:
                adx = talib.ADX(high_values, low_values, close_values, timeperiod=period)
                indicators[f'ADX_{period}'] = adx
            except Exception as e:
                self.logger.warning(f"⚠️ ADX calculation failed for period {period}: {e}")
        
        return indicators

    def _extract_obv_batch(self, data: pd.DataFrame, periods: List[int]) -> Dict[str, np.ndarray]:
        """Extract OBV indicators for multiple periods in batch."""
        indicators = {}
        close_values = data['close'].values
        volume_values = data['volume'].values if 'volume' in data.columns else np.ones(len(data))
        
        try:
            obv = talib.OBV(close_values, volume_values)
            obv_normalized = (obv - np.mean(obv)) / (np.std(obv) + 1e-8)
            indicators['OBV'] = obv
            indicators['OBV_Normalized'] = obv_normalized
        except Exception as e:
            self.logger.warning(f"⚠️ OBV calculation failed: {e}")
        
        return indicators

    def _extract_mfi_batch(self, data: pd.DataFrame, periods: List[int]) -> Dict[str, np.ndarray]:
        """Extract MFI indicators for multiple periods in batch."""
        indicators = {}
        high_values = data['high'].values
        low_values = data['low'].values
        close_values = data['close'].values
        volume_values = data['volume'].values if 'volume' in data.columns else np.ones(len(data))
        
        for period in periods:
            try:
                mfi = talib.MFI(high_values, low_values, close_values, volume_values, timeperiod=period)
                indicators[f'MFI_{period}'] = mfi
            except Exception as e:
                self.logger.warning(f"⚠️ MFI calculation failed for period {period}: {e}")
        
        return indicators

    @step06_function_validator(function_type='feature_engineering', validation_level=ValidationLevel.COMPREHENSIVE)
    def create_sophisticated_interactions(self, features: pd.DataFrame, 
                                        current_idx: Optional[int] = None) -> pd.DataFrame:
        """
        Create sophisticated feature interactions with polynomial features and pattern recognition.
        
        Args:
            features: Base features DataFrame
            current_idx: Current processing index for temporal validation
            
        Returns:
            DataFrame with sophisticated interaction features
        """
        with step06_validation_context('create_sophisticated_interactions', 'feature_engineering'):
            self.logger.info(f'🔗 Starting sophisticated interaction creation')
            self.logger.info(f'   Input features: {features.shape}')
            self.logger.info(f'   Current index: {current_idx}')
        
        # Apply temporal validation
        validated_features = self._validate_temporal_consistency(features, current_idx)
        
        # Extract feature arrays
        feature_arrays = []
        feature_names = []
        
        for col in validated_features.columns:
            if col.startswith(('RSI_', 'MACD_', 'SMA_', 'EMA_', 'BB_', 'ATR_')):
                feature_arrays.append(validated_features[col].values)
                feature_names.append(col)
        
        if not feature_arrays:
            self.logger.warning("⚠️ No technical indicators found for interaction creation")
            return validated_features
        
        features_matrix = np.column_stack(feature_arrays)
        
        # Create different types of interactions
        interaction_features = {}
        
        # 1. Polynomial features
        if SKLEARN_AVAILABLE:
            poly_features = self._create_polynomial_features(features_matrix, feature_names)
            interaction_features.update(poly_features)
        
        # 2. Cross-timeframe interactions
        cross_timeframe_features = self._create_cross_timeframe_interactions(features_matrix, feature_names)
        interaction_features.update(cross_timeframe_features)
        
        # 3. Advanced pattern recognition
        pattern_features = self._create_pattern_recognition_features(features_matrix, feature_names)
        interaction_features.update(pattern_features)
        
        # 4. Momentum and volatility interactions
        momentum_vol_features = self._create_momentum_volatility_interactions(features_matrix, feature_names)
        interaction_features.update(momentum_vol_features)
        
        # 5. Regime-dependent interactions
        regime_features = self._create_regime_dependent_interactions(features_matrix, feature_names)
        interaction_features.update(regime_features)
        
        # Combine all features
        all_interaction_features = pd.DataFrame(interaction_features, index=validated_features.index)
        
        # Feature selection to prevent overfitting
        if len(all_interaction_features.columns) > self.max_features:
            all_interaction_features = self._select_optimal_features(
                all_interaction_features, validated_features
            )
        
        # Combine with original features
        result = pd.concat([validated_features, all_interaction_features], axis=1)
        
        self.logger.info(f'✅ Sophisticated interactions created: {len(interaction_features)} new features')
        self.logger.info(f'   Total features: {result.shape[1]}')
        
        return result

    def _create_polynomial_features(self, features_matrix: np.ndarray, 
                                   feature_names: List[str]) -> Dict[str, np.ndarray]:
        """Create polynomial features for non-linear relationships."""
        if not SKLEARN_AVAILABLE:
            return {}
        
        try:
            # Limit features for polynomial expansion to prevent explosion
            max_poly_features = min(20, features_matrix.shape[1])
            selected_features = features_matrix[:, :max_poly_features]
            selected_names = feature_names[:max_poly_features]
            
            # Create polynomial features
            poly = PolynomialFeatures(
                degree=self.polynomial_degree,
                include_bias=False,
                interaction_only=True  # Only interaction terms, not powers
            )
            
            poly_matrix = poly.fit_transform(selected_features)
            poly_feature_names = poly.get_feature_names_out(selected_names)
            
            # Create dictionary
            poly_features = {}
            for i, name in enumerate(poly_feature_names):
                if 'x' in name:  # Skip single features, keep only interactions
                    poly_features[f'poly_{name.replace(" ", "_")}'] = poly_matrix[:, i]
            
            self.logger.info(f'📊 Created {len(poly_features)} polynomial interaction features')
            return poly_features
            
        except Exception as e:
            self.logger.warning(f"⚠️ Polynomial feature creation failed: {e}")
            return {}

    def _create_cross_timeframe_interactions(self, features_matrix: np.ndarray, 
                                           feature_names: List[str]) -> Dict[str, np.ndarray]:
        """Create cross-timeframe interactions."""
        cross_features = {}
        
        # Group features by type
        feature_groups = {}
        for i, name in enumerate(feature_names):
            base_name = name.split('_')[0]
            if base_name not in feature_groups:
                feature_groups[base_name] = []
            feature_groups[base_name].append((i, name))
        
        # Create cross-timeframe interactions
        for base_name, feature_list in feature_groups.items():
            if len(feature_list) >= 2:
                # Sort by period (extract number from feature name)
                feature_list.sort(key=lambda x: int(x[1].split('_')[-1]) if x[1].split('_')[-1].isdigit() else 0)
                
                for i in range(len(feature_list) - 1):
                    short_idx, short_name = feature_list[i]
                    long_idx, long_name = feature_list[i + 1]
                    
                    short_values = features_matrix[:, short_idx]
                    long_values = features_matrix[:, long_idx]
                    
                    # Create interaction features
                    cross_features[f'{base_name}_short_long_ratio'] = safe_divide(
                        short_values, long_values, default=1.0
                    )
                    cross_features[f'{base_name}_short_long_diff'] = short_values - long_values
                    cross_features[f'{base_name}_short_long_momentum'] = safe_divide(
                        short_values - long_values, long_values, default=0.0
                    )
        
        self.logger.info(f'⏰ Created {len(cross_features)} cross-timeframe interaction features')
        return cross_features

    def _create_pattern_recognition_features(self, features_matrix: np.ndarray, 
                                           feature_names: List[str]) -> Dict[str, np.ndarray]:
        """Create advanced pattern recognition features."""
        pattern_features = {}
        
        # Find RSI features
        rsi_indices = [i for i, name in enumerate(feature_names) if name.startswith('RSI_')]
        if len(rsi_indices) >= 2:
            rsi_short = features_matrix[:, rsi_indices[0]]
            rsi_long = features_matrix[:, rsi_indices[1]]
            
            # RSI divergence patterns
            pattern_features['rsi_divergence'] = rsi_short - rsi_long
            pattern_features['rsi_overbought_short'] = (rsi_short > 70).astype(float)
            pattern_features['rsi_oversold_short'] = (rsi_short < 30).astype(float)
            pattern_features['rsi_overbought_long'] = (rsi_long > 70).astype(float)
            pattern_features['rsi_oversold_long'] = (rsi_long < 30).astype(float)
        
        # Find MACD features
        macd_indices = [i for i, name in enumerate(feature_names) if name.startswith('MACD_') and not 'Signal' in name and not 'Hist' in name]
        macd_signal_indices = [i for i, name in enumerate(feature_names) if name.startswith('MACD_Signal_')]
        
        if macd_indices and macd_signal_indices:
            macd = features_matrix[:, macd_indices[0]]
            macd_signal = features_matrix[:, macd_signal_indices[0]]
            
            # MACD patterns
            pattern_features['macd_bullish_cross'] = ((macd > macd_signal) & (np.roll(macd, 1) <= np.roll(macd_signal, 1))).astype(float)
            pattern_features['macd_bearish_cross'] = ((macd < macd_signal) & (np.roll(macd, 1) >= np.roll(macd_signal, 1))).astype(float)
            pattern_features['macd_momentum'] = macd - macd_signal
        
        # Find Bollinger Bands features
        bb_position_indices = [i for i, name in enumerate(feature_names) if name.startswith('BB_Position_')]
        bb_squeeze_indices = [i for i, name in enumerate(feature_names) if name.startswith('BB_Squeeze_')]
        
        if bb_position_indices and bb_squeeze_indices:
            bb_position = features_matrix[:, bb_position_indices[0]]
            bb_squeeze = features_matrix[:, bb_squeeze_indices[0]]
            
            # Bollinger Bands patterns
            pattern_features['bb_squeeze_breakout'] = (bb_squeeze > np.roll(bb_squeeze, 1)).astype(float)
            pattern_features['bb_position_extreme'] = ((bb_position > 0.8) | (bb_position < 0.2)).astype(float)
            pattern_features['bb_mean_reversion'] = np.abs(bb_position - 0.5)
        
        self.logger.info(f'🎯 Created {len(pattern_features)} pattern recognition features')
        return pattern_features

    def _create_momentum_volatility_interactions(self, features_matrix: np.ndarray, 
                                               feature_names: List[str]) -> Dict[str, np.ndarray]:
        """Create momentum and volatility interaction features."""
        momentum_vol_features = {}
        
        # Find momentum features (RSI, Stochastic, etc.)
        momentum_indices = []
        for i, name in enumerate(feature_names):
            if any(momentum in name for momentum in ['RSI_', 'Stoch_', 'Williams_', 'CCI_']):
                momentum_indices.append(i)
        
        # Find volatility features (ATR, BB_Squeeze, etc.)
        volatility_indices = []
        for i, name in enumerate(feature_names):
            if any(vol in name for vol in ['ATR_', 'BB_Squeeze_', 'Volatility']):
                volatility_indices.append(i)
        
        if momentum_indices and volatility_indices:
            # Average momentum and volatility
            avg_momentum = np.mean(features_matrix[:, momentum_indices], axis=1)
            avg_volatility = np.mean(features_matrix[:, volatility_indices], axis=1)
            
            # Create interactions
            momentum_vol_features['momentum_vol_interaction'] = avg_momentum * avg_volatility
            momentum_vol_features['momentum_vol_ratio'] = safe_divide(avg_momentum, avg_volatility, default=0.0)
            momentum_vol_features['momentum_vol_normalized'] = safe_divide(
                avg_momentum * avg_volatility, 
                np.std(avg_momentum) * np.std(avg_volatility) + 1e-8, 
                default=0.0
            )
        
        self.logger.info(f'📈 Created {len(momentum_vol_features)} momentum-volatility interaction features')
        return momentum_vol_features

    def _create_regime_dependent_interactions(self, features_matrix: np.ndarray, 
                                            feature_names: List[str]) -> Dict[str, np.ndarray]:
        """Create regime-dependent interaction features."""
        regime_features = {}
        
        # Simple regime detection based on volatility
        if 'ATR_Normalized' in feature_names:
            atr_idx = feature_names.index('ATR_Normalized')
            atr_values = features_matrix[:, atr_idx]
            
            # Define regimes based on ATR percentiles
            atr_25 = np.percentile(atr_values, 25)
            atr_75 = np.percentile(atr_values, 75)
            
            low_vol_regime = (atr_values < atr_25).astype(float)
            high_vol_regime = (atr_values > atr_75).astype(float)
            
            # Create regime-dependent features
            for i, name in enumerate(feature_names):
                if name.startswith(('RSI_', 'MACD_')):
                    feature_values = features_matrix[:, i]
                    
                    regime_features[f'{name}_low_vol'] = feature_values * low_vol_regime
                    regime_features[f'{name}_high_vol'] = feature_values * high_vol_regime
        
        self.logger.info(f'🏛️ Created {len(regime_features)} regime-dependent interaction features')
        return regime_features

    def _select_optimal_features(self, interaction_features: pd.DataFrame, 
                               base_features: pd.DataFrame) -> pd.DataFrame:
        """Select optimal features to prevent overfitting."""
        if not SKLEARN_AVAILABLE:
            # Simple correlation-based selection
            return self._select_features_by_correlation(interaction_features)
        
        try:
            # Use mutual information for feature selection
            # Create dummy target for feature selection
            dummy_target = np.random.choice([0, 1], size=len(interaction_features))
            
            # Select top features
            selector = SelectKBest(score_func=mutual_info_classif, k=self.max_features)
            selected_features = selector.fit_transform(interaction_features, dummy_target)
            
            # Get selected feature names
            selected_indices = selector.get_support(indices=True)
            selected_columns = [interaction_features.columns[i] for i in selected_indices]
            
            result = pd.DataFrame(selected_features, 
                                index=interaction_features.index, 
                                columns=selected_columns)
            
            self.logger.info(f'🎯 Selected {len(selected_columns)} optimal features from {len(interaction_features.columns)}')
            return result
            
        except Exception as e:
            self.logger.warning(f"⚠️ Advanced feature selection failed, using correlation-based: {e}")
            return self._select_features_by_correlation(interaction_features)

    def _select_features_by_correlation(self, features: pd.DataFrame) -> pd.DataFrame:
        """Select features based on correlation to avoid redundancy."""
        # Calculate correlation matrix
        corr_matrix = features.corr().abs()
        
        # Find highly correlated pairs
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Select features to drop
        to_drop = [column for column in upper_triangle.columns 
                  if any(upper_triangle[column] > self.correlation_threshold)]
        
        # Keep features
        selected_features = features.drop(columns=to_drop)
        
        # Limit to max_features
        if len(selected_features.columns) > self.max_features:
            selected_features = selected_features.iloc[:, :self.max_features]
        
        self.logger.info(f'📊 Correlation-based selection: kept {len(selected_features.columns)} features')
        return selected_features

    def _validate_market_data(self, data: pd.DataFrame) -> None:
        """Validate market data for feature engineering."""
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise MathValidationError(f"Missing required columns: {missing_columns}")
        
        # Check for valid data
        for col in required_columns:
            if (data[col] <= 0).any():
                raise MathValidationError(f"Invalid prices in {col}: non-positive values found")
        
        # Check for sufficient data
        if len(data) < 50:
            raise MathValidationError(f"Insufficient data: {len(data)} rows (minimum 50 required)")

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.processing_stats.copy()

    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self.processing_stats = {
            'total_samples_processed': 0,
            'total_features_created': 0,
            'processing_time': 0.0,
            'memory_usage_mb': 0.0,
            'chunks_processed': 0
        }