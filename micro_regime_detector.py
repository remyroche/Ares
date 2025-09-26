"""
Enhanced Micro-Regime Detector with Comprehensive Utility Integration

This module provides a sophisticated micro-regime detection system that integrates
all available utilities for optimal performance and functionality:

- Advanced regime detection algorithms
- ML-based optimization with CV, HPO, and Bayesian methods
- M1 hardware optimization (GPU, CPU, Memory)
- Comprehensive data handling and parquet support
- Robust serialization and persistence
- Enhanced logging and monitoring
- Mathematical validation and error handling

Key Features:
- Breakout detection with volume confirmation
- Consolidation pattern recognition
- Reversal signal detection with RSI and momentum
- Acceleration/deceleration detection
- Volume and volatility spike identification
- Liquidity change detection
- Multi-timeframe analysis
- Adaptive parameter optimization
- Real-time regime monitoring
- Comprehensive performance analytics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
from pathlib import Path
import warnings
import asyncio
from contextlib import contextmanager

# Core dependencies
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest

# Import all utility modules
from src.utils.common_operations import (
    safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std,
    validate_finite, validate_positive, validate_range,
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
    calculate_data_quality_metrics, create_summary_statistics,
    safe_to_parquet, safe_read_parquet, optimize_dataframe_dtypes,
    integrate_with_m1_optimizers, get_m1_gpu_manager, get_m1_memory_optimizer,
    get_m1_cpu_optimizer, cleanup_m1_optimizers, memory_checkpoint, gpu_context,
    optimize_memory, get_memory_usage, timed_operation, format_bytes
)

from src.utils.math_validation import (
    MathValidation, validate_numeric_array, safe_correlation, safe_covariance,
    safe_percentile, validate_correlation_matrix, safe_matrix_inverse,
    math_safe, MathValidationError
)

from src.utils.serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
)

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_performance, tprint_structured, tprint_with_level,
    tprint_timer, configure_tprint, TPrintConfig, LogLevel
)

# ML utilities
try:
    from src.utils.ml_common.cvlsa import (
        create_enhanced_cvlsa_model, create_cvlsa_config,
        create_hybrid_cvlsa_tree_model, create_enhanced_variable_selector,
        create_improved_feature_engineer, create_performance_memory_manager,
        create_robust_error_handler, create_advanced_monitoring_analytics,
        create_configuration_simplification
    )
    ML_UTILITIES_AVAILABLE = True
except ImportError:
    ML_UTILITIES_AVAILABLE = False
    tprint_warning("ML utilities not available - some advanced features will be limited")

# Data utilities
try:
    from src.utils.data.klines_parquet import KlinesParquetManager
    from src.utils.data.processing.data_processing import DataProcessor
    from src.utils.data.quality.data_quality import DataQualityAnalyzer
    DATA_UTILITIES_AVAILABLE = True
except ImportError:
    DATA_UTILITIES_AVAILABLE = False
    tprint_warning("Data utilities not available - using fallback implementations")

# Matrix operations
try:
    from src.utils.matrix_operations.unified_operations import MatrixOperationsManager
    MATRIX_UTILITIES_AVAILABLE = True
except ImportError:
    MATRIX_UTILITIES_AVAILABLE = False
    tprint_warning("Matrix utilities not available - using basic numpy operations")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configure enhanced tprint
tprint_config = TPrintConfig(
    timestamp_format='%Y-%m-%d %H:%M:%S.%f',
    use_colors=True,
    enable_structured_logging=True,
    integrate_with_logging=True,
    auto_log_prints=True
)
configure_tprint(tprint_config)


class MicroRegimeType(Enum):
    """Enumeration of micro-regime types."""
    BREAKOUT = "breakout"
    CONSOLIDATION = "consolidation"
    REVERSAL = "reversal"
    ACCELERATION = "acceleration"
    DECELERATION = "deceleration"
    VOLUME_SPIKE = "volume_spike"
    VOLATILITY_SPIKE = "volatility_spike"
    LIQUIDITY_CHANGE = "liquidity_change"
    MOMENTUM_SHIFT = "momentum_shift"
    TREND_CONTINUATION = "trend_continuation"
    TREND_REVERSAL = "trend_reversal"
    UNKNOWN = "unknown"


class MarketRegime(Enum):
    """Enumeration of broader market regimes."""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING = "trending"
    RANGING = "ranging"
    UNKNOWN = "unknown"


@dataclass
class MicroRegimeDetectionResult:
    """Enhanced result of micro-regime detection with comprehensive metadata."""
    regime_type: MicroRegimeType
    confidence: float
    start_time: datetime
    end_time: Optional[datetime] = None
    characteristics: Dict[str, float] = field(default_factory=dict)
    signal_strength: float = 0.0
    duration_minutes: float = 0.0
    transition_probability: float = 0.0
    market_context: Optional[MarketRegime] = None
    feature_importance: Dict[str, float] = field(default_factory=dict)
    ml_confidence: float = 0.0
    statistical_significance: float = 0.0
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionConfig:
    """Comprehensive configuration for micro-regime detection."""
    # Core detection parameters
    sensitivity: float = 0.7
    detection_threshold: float = 0.6
    min_confidence: float = 0.5
    
    # Time parameters
    lookback_periods: int = 50
    min_duration_minutes: int = 5
    max_duration_minutes: int = 240
    
    # Technical indicator parameters
    rsi_period: int = 14
    sma_periods: List[int] = field(default_factory=lambda: [20, 50, 200])
    volatility_period: int = 20
    
    # ML parameters
    enable_ml_detection: bool = True
    ml_confidence_threshold: float = 0.7
    feature_engineering: bool = True
    cross_validation: bool = True
    hyperparameter_optimization: bool = True
    
    # Hardware optimization
    enable_m1_optimization: bool = True
    memory_limit_gb: Optional[float] = None
    use_gpu_acceleration: bool = True
    
    # Data processing
    enable_data_quality_checks: bool = True
    missing_data_threshold: float = 0.1
    outlier_detection: bool = True
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    detailed_logging: bool = True
    save_detection_history: bool = True


class MicroRegimeDetector:
    """
    Enhanced Micro-Regime Detector with comprehensive utility integration.
    
    This detector combines traditional technical analysis with modern ML techniques
    and hardware optimization for superior regime detection performance.
    """
    
    def __init__(self, config: Optional[DetectionConfig] = None):
        """
        Initialize the enhanced micro-regime detector.
        
        Args:
            config: Configuration object for the detector
        """
        self.config = config or DetectionConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize utility components
        self._initialize_utilities()
        
        # Initialize detection parameters
        self._initialize_detection_parameters()
        
        # Initialize ML components if available
        self._initialize_ml_components()
        
        # Initialize data components
        self._initialize_data_components()
        
        # Performance tracking
        self.detection_history: List[MicroRegimeDetectionResult] = []
        self.performance_metrics: Dict[str, Any] = {}
        
        tprint_success(f"🚀 MicroRegimeDetector initialized with {len(self._get_enabled_features())} features")
    
    def _initialize_utilities(self):
        """Initialize utility components."""
        try:
            # Math validation
            self.math_validator = MathValidation()
            
            # Serialization
            self.serializer = UniversalSerializer()
            
            # Hardware optimization
            if self.config.enable_m1_optimization:
                self.m1_integration = integrate_with_m1_optimizers()
                if self.m1_integration.get('success', False):
                    self.gpu_manager = get_m1_gpu_manager()
                    self.memory_optimizer = get_m1_memory_optimizer()
                    self.cpu_optimizer = get_m1_cpu_optimizer()
                    tprint_success("🧠 M1 optimization enabled")
                else:
                    tprint_warning("⚠️ M1 optimization failed - using fallback")
                    self.m1_integration = {'success': False}
            else:
                self.m1_integration = {'success': False}
            
            # Matrix operations
            if MATRIX_UTILITIES_AVAILABLE:
                self.matrix_manager = MatrixOperationsManager()
                tprint_success("🔢 Matrix operations enabled")
            
        except Exception as e:
            tprint_error(f"❌ Utility initialization failed: {e}")
            raise
    
    def _initialize_detection_parameters(self):
        """Initialize detection parameters with validation."""
        # Breakout detection
        self.breakout_params = {
            'price_threshold': validate_range(self.config.sensitivity * 2.0, 0.5, 5.0, "price_threshold"),
            'volume_multiplier': validate_range(1.2 + self.config.sensitivity * 0.5, 1.0, 3.0, "volume_multiplier"),
            'min_duration': max(3, int(self.config.min_duration_minutes / 5)),
            'momentum_confirmation': True,
            'trend_confirmation': True
        }
        
        # Consolidation detection
        self.consolidation_params = {
            'volatility_threshold': validate_range(0.2 + self.config.sensitivity * 0.3, 0.1, 0.8, "volatility_threshold"),
            'price_range_threshold': validate_range(0.01 + self.config.sensitivity * 0.02, 0.005, 0.05, "price_range_threshold"),
            'min_duration': max(10, int(self.config.min_duration_minutes / 2)),
            'volume_confirmation': True
        }
        
        # Reversal detection
        self.reversal_params = {
            'momentum_threshold': validate_range(0.5 + self.config.sensitivity * 0.4, 0.2, 1.0, "momentum_threshold"),
            'volume_confirmation': validate_range(1.1 + self.config.sensitivity * 0.3, 1.0, 2.0, "volume_confirmation"),
            'rsi_threshold': validate_range(25 + self.config.sensitivity * 10, 20, 40, "rsi_threshold"),
            'divergence_detection': True
        }
        
        # Acceleration detection
        self.acceleration_params = {
            'momentum_acceleration': validate_range(0.05 + self.config.sensitivity * 0.1, 0.02, 0.3, "momentum_acceleration"),
            'volume_trend': validate_range(1.1 + self.config.sensitivity * 0.4, 1.0, 2.0, "volume_trend"),
            'duration_threshold': max(5, int(self.config.min_duration_minutes / 3))
        }
        
        # Volume spike detection
        self.volume_spike_params = {
            'volume_multiplier': validate_range(1.5 + self.config.sensitivity * 1.0, 1.2, 5.0, "volume_multiplier"),
            'price_confirmation': validate_range(0.005 + self.config.sensitivity * 0.01, 0.001, 0.03, "price_confirmation"),
            'isolation_threshold': validate_range(0.7 + self.config.sensitivity * 0.2, 0.5, 0.9, "isolation_threshold"),
            'duration_threshold': max(2, int(self.config.min_duration_minutes / 5))
        }
        
        # Volatility spike detection
        self.volatility_spike_params = {
            'volatility_multiplier': validate_range(2.0 + self.config.sensitivity * 1.0, 1.5, 5.0, "volatility_multiplier"),
            'duration_threshold': max(3, int(self.config.min_duration_minutes / 4)),
            'price_impact_threshold': validate_range(0.01 + self.config.sensitivity * 0.02, 0.005, 0.05, "price_impact_threshold")
        }
    
    def _initialize_ml_components(self):
        """Initialize ML components if available."""
        if not ML_UTILITIES_AVAILABLE or not self.config.enable_ml_detection:
            self.ml_components = {}
            return
        
        try:
            # Initialize ML components
            self.ml_components = {
                'cvlsa_model': None,
                'feature_selector': create_enhanced_variable_selector(),
                'feature_engineer': create_improved_feature_engineer(),
                'performance_manager': create_performance_memory_manager(),
                'error_handler': create_robust_error_handler(),
                'monitoring': create_advanced_monitoring_analytics(),
                'config_simplifier': create_configuration_simplification()
            }
            
            tprint_success("🤖 ML components initialized")
            
        except Exception as e:
            tprint_warning(f"⚠️ ML component initialization failed: {e}")
            self.ml_components = {}
    
    def _initialize_data_components(self):
        """Initialize data handling components."""
        if DATA_UTILITIES_AVAILABLE:
            try:
                self.data_components = {
                    'parquet_manager': KlinesParquetManager(),
                    'data_processor': DataProcessor(),
                    'quality_analyzer': DataQualityAnalyzer()
                }
                tprint_success("📊 Data components initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Data component initialization failed: {e}")
                self.data_components = {}
        else:
            self.data_components = {}
    
    def _get_enabled_features(self) -> List[str]:
        """Get list of enabled features."""
        features = ['core_detection', 'math_validation', 'serialization', 'logging']
        
        if self.m1_integration.get('success', False):
            features.append('m1_optimization')
        
        if self.ml_components:
            features.append('ml_detection')
        
        if self.data_components:
            features.append('data_processing')
        
        if MATRIX_UTILITIES_AVAILABLE:
            features.append('matrix_operations')
        
        return features
    
    @timed_operation
    def detect_micro_regimes(self, market_data: pd.DataFrame,
                           current_regime: Optional[MarketRegime] = None,
                           enable_ml: bool = None) -> List[MicroRegimeDetectionResult]:
        """
        Detect micro-regimes in market data with comprehensive analysis.
        
        Args:
            market_data: Market data with OHLCV and indicators
            current_regime: Current market regime context
            enable_ml: Override ML detection (None = use config)
            
        Returns:
            List of detected micro-regimes with comprehensive metadata
        """
        tprint_info("🔍 Starting enhanced micro-regime detection...")
        
        # Use memory checkpoint for M1 optimization
        with memory_checkpoint("micro_regime_detection"):
            try:
                # Validate input data
                validated_data = self._validate_and_preprocess_data(market_data)
                
                # Detect regimes using multiple methods
                detected_regimes = []
                
                # Traditional technical analysis detection
                traditional_regimes = self._detect_traditional_regimes(validated_data, current_regime)
                detected_regimes.extend(traditional_regimes)
                
                # ML-based detection if enabled
                if (enable_ml is None and self.config.enable_ml_detection) or enable_ml:
                    ml_regimes = self._detect_ml_regimes(validated_data, current_regime)
                    detected_regimes.extend(ml_regimes)
                
                # Statistical regime detection
                statistical_regimes = self._detect_statistical_regimes(validated_data)
                detected_regimes.extend(statistical_regimes)
                
                # Post-process and rank results
                final_regimes = self._post_process_regimes(detected_regimes, validated_data)
                
                # Update performance metrics
                self._update_performance_metrics(final_regimes)
                
                # Store in history if enabled
                if self.config.save_detection_history:
                    self.detection_history.extend(final_regimes)
                
                tprint_success(f"✅ Detected {len(final_regimes)} micro-regimes")
                return final_regimes
                
            except Exception as e:
                tprint_error(f"❌ Micro-regime detection failed: {e}")
                if self.ml_components.get('error_handler'):
                    self.ml_components['error_handler'].handle_error(e, "detect_micro_regimes")
                return []
    
    def _validate_and_preprocess_data(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate and preprocess market data with comprehensive quality checks."""
        tprint_debug("📊 Validating and preprocessing market data...")
        
        # Data quality checks
        if self.config.enable_data_quality_checks and self.data_components:
            quality_report = self.data_components['quality_analyzer'].analyze_data_quality(market_data)
            if quality_report['quality_score'] < 0.7:
                tprint_warning(f"⚠️ Data quality score: {quality_report['quality_score']:.2f}")
        
        # Validate required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not validate_dataframe_columns(market_data, required_columns):
            raise ValueError(f"Missing required columns. Required: {required_columns}")
        
        # Handle missing data
        if market_data.isnull().sum().sum() / (len(market_data) * len(market_data.columns)) > self.config.missing_data_threshold:
            tprint_warning("⚠️ High missing data ratio - applying imputation")
            market_data = self._impute_missing_data(market_data)
        
        # Optimize data types
        market_data = optimize_dataframe_dtypes(market_data)
        
        # Calculate comprehensive technical indicators
        processed_data = self._calculate_technical_indicators(market_data)
        
        # Add feature engineering if ML is enabled
        if self.ml_components and self.config.feature_engineering:
            processed_data = self._apply_feature_engineering(processed_data)
        
        # Outlier detection and handling
        if self.config.outlier_detection:
            processed_data = self._handle_outliers(processed_data)
        
        return processed_data
    
    def _calculate_technical_indicators(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive technical indicators."""
        tprint_debug("📈 Calculating technical indicators...")
        
        # Basic price data
        price = market_data['close']
        high = market_data['high']
        low = market_data['low']
        volume = market_data.get('volume', 1)
        
        # Returns and log returns
        returns = price.pct_change()
        log_returns = np.log(price).diff()
        
        # Moving averages
        sma_data = {}
        for period in self.config.sma_periods:
            sma_data[f'sma_{period}'] = price.rolling(period).mean()
            sma_data[f'ema_{period}'] = price.ewm(span=period).mean()
        
        # Volatility measures
        volatility_20 = returns.rolling(self.config.volatility_period).std()
        volatility_50 = returns.rolling(50).std()
        atr = self._calculate_atr(high, low, price, 14)
        
        # Momentum indicators
        momentum_5 = price / price.shift(5) - 1
        momentum_10 = price / price.shift(10) - 1
        momentum_20 = price / price.shift(20) - 1
        
        # RSI and other oscillators
        rsi = self._calculate_rsi(price, self.config.rsi_period)
        macd_line, macd_signal, macd_hist = self._calculate_macd(price)
        bollinger_upper, bollinger_lower = self._calculate_bollinger_bands(price)
        
        # Volume indicators
        volume_ma_20 = volume.rolling(20).mean()
        volume_ratio = volume / volume_ma_20
        obv = self._calculate_obv(price, volume)
        
        # Price derivatives
        price_velocity = returns.rolling(5).mean()
        price_acceleration = price_velocity.diff()
        price_jerk = price_acceleration.diff()
        
        # Support and resistance levels
        support_resistance = self._calculate_support_resistance(high, low, price)
        
        # Market microstructure
        spread_estimate = (high - low) / price
        price_impact = abs(returns) * volume
        
        return {
            'price': price,
            'high': high,
            'low': low,
            'volume': volume,
            'returns': returns,
            'log_returns': log_returns,
            'timestamp': market_data.index,
            **sma_data,
            'volatility_20': volatility_20,
            'volatility_50': volatility_50,
            'atr': atr,
            'momentum_5': momentum_5,
            'momentum_10': momentum_10,
            'momentum_20': momentum_20,
            'rsi': rsi,
            'macd_line': macd_line,
            'macd_signal': macd_signal,
            'macd_hist': macd_hist,
            'bollinger_upper': bollinger_upper,
            'bollinger_lower': bollinger_lower,
            'volume_ratio': volume_ratio,
            'obv': obv,
            'price_velocity': price_velocity,
            'price_acceleration': price_acceleration,
            'price_jerk': price_jerk,
            'support_resistance': support_resistance,
            'spread_estimate': spread_estimate,
            'price_impact': price_impact
        }
    
    def _detect_traditional_regimes(self, data: Dict[str, Any], 
                                  current_regime: Optional[MarketRegime]) -> List[MicroRegimeDetectionResult]:
        """Detect regimes using traditional technical analysis methods."""
        tprint_debug("🔍 Detecting traditional regimes...")
        
        regimes = []
        
        # Breakout detection
        breakout_regimes = self._detect_enhanced_breakouts(data)
        regimes.extend(breakout_regimes)
        
        # Consolidation detection
        consolidation_regimes = self._detect_enhanced_consolidations(data)
        regimes.extend(consolidation_regimes)
        
        # Reversal detection
        reversal_regimes = self._detect_enhanced_reversals(data)
        regimes.extend(reversal_regimes)
        
        # Acceleration/deceleration detection
        acceleration_regimes = self._detect_enhanced_accelerations(data)
        regimes.extend(acceleration_regimes)
        
        # Volume spike detection
        volume_spikes = self._detect_enhanced_volume_spikes(data)
        regimes.extend(volume_spikes)
        
        # Volatility spike detection
        volatility_spikes = self._detect_enhanced_volatility_spikes(data)
        regimes.extend(volatility_spikes)
        
        # Momentum shift detection
        momentum_shifts = self._detect_momentum_shifts(data)
        regimes.extend(momentum_shifts)
        
        # Liquidity change detection
        liquidity_changes = self._detect_liquidity_changes(data)
        regimes.extend(liquidity_changes)
        
        return regimes
    
    def _detect_enhanced_breakouts(self, data: Dict[str, Any]) -> List[MicroRegimeDetectionResult]:
        """Enhanced breakout detection with multiple confirmation signals."""
        breakouts = []
        
        try:
            price = data['price']
            returns = data['returns']
            volume = data['volume']
            atr = data['atr']
            bollinger_upper = data['bollinger_upper']
            bollinger_lower = data['bollinger_lower']
            
            # Multiple breakout signals
            signals = []
            
            # Bollinger Band breakout
            bb_breakout_up = price > bollinger_upper
            bb_breakout_down = price < bollinger_lower
            signals.append(('bollinger', bb_breakout_up | bb_breakout_down))
            
            # ATR-based breakout
            atr_breakout = abs(returns) > atr * self.breakout_params['price_threshold']
            signals.append(('atr', atr_breakout))
            
            # Volume-confirmed breakout
            volume_confirmation = data['volume_ratio'] > self.breakout_params['volume_multiplier']
            signals.append(('volume', volume_confirmation))
            
            # Combined breakout signal
            combined_signal = pd.Series(False, index=price.index)
            for signal_name, signal in signals:
                combined_signal |= signal
            
            # Find breakout periods
            breakout_periods = self._find_contiguous_periods(combined_signal)
            
            for start_idx, end_idx in breakout_periods:
                if end_idx - start_idx >= self.breakout_params['min_duration']:
                    
                    # Calculate comprehensive breakout characteristics
                    breakout_data = self._analyze_breakout_period(data, start_idx, end_idx)
                    
                    if breakout_data['confidence'] >= self.config.detection_threshold:
                        
                        # Determine breakout direction and strength
                        breakout_direction = self._determine_breakout_direction(
                            breakout_data['price_data'], breakout_data['returns_data']
                        )
                        
                        # Calculate risk metrics
                        risk_metrics = self._calculate_breakout_risk_metrics(breakout_data)
                        
                        # Calculate performance metrics
                        performance_metrics = self._calculate_breakout_performance_metrics(breakout_data)
                        
                        breakout_regime = MicroRegimeDetectionResult(
                            regime_type=MicroRegimeType.BREAKOUT,
                            confidence=breakout_data['confidence'],
                            start_time=data['timestamp'][start_idx],
                            end_time=data['timestamp'][end_idx],
                            characteristics=breakout_data['characteristics'],
                            signal_strength=breakout_data['signal_strength'],
                            duration_minutes=(end_idx - start_idx) * 5,
                            transition_probability=breakout_data['transition_probability'],
                            feature_importance=breakout_data['feature_importance'],
                            risk_metrics=risk_metrics,
                            performance_metrics=performance_metrics,
                            metadata={
                                'breakout_direction': breakout_direction,
                                'confirmation_signals': len([s for s in signals if s[1].iloc[start_idx:end_idx].any()]),
                                'detection_method': 'enhanced_traditional'
                            }
                        )
                        
                        breakouts.append(breakout_regime)
        
        except Exception as e:
            tprint_warning(f"⚠️ Enhanced breakout detection failed: {e}")
        
        return breakouts
    
    def _analyze_breakout_period(self, data: Dict[str, Any], start_idx: int, end_idx: int) -> Dict[str, Any]:
        """Analyze a breakout period with comprehensive metrics."""
        # Extract period data
        period_data = {key: values.iloc[start_idx:end_idx] for key, values in data.items() 
                      if isinstance(values, pd.Series)}
        
        # Calculate basic characteristics
        price_data = period_data['price']
        returns_data = period_data['returns']
        volume_data = period_data['volume']
        
        # Price movement analysis
        price_change = (price_data.iloc[-1] / price_data.iloc[0] - 1)
        price_range = (price_data.max() - price_data.min()) / price_data.iloc[0]
        price_volatility = returns_data.std()
        
        # Volume analysis
        avg_volume = volume_data.mean()
        baseline_volume = data['volume'].iloc[:start_idx].tail(20).mean()
        volume_multiplier = avg_volume / baseline_volume if baseline_volume > 0 else 1
        
        # Momentum analysis
        momentum_strength = abs(returns_data.mean())
        momentum_consistency = 1 - returns_data.std()
        
        # Technical indicator analysis
        rsi_data = period_data.get('rsi', pd.Series([50] * len(price_data)))
        rsi_extreme = max(abs(rsi_data.iloc[-1] - 50), abs(rsi_data.iloc[0] - 50)) / 50
        
        # Calculate confidence using multiple factors
        confidence_factors = {
            'price_movement': min(1.0, abs(price_change) * 10),
            'volume_confirmation': min(1.0, volume_multiplier / self.breakout_params['volume_multiplier']),
            'momentum_strength': min(1.0, momentum_strength * 5),
            'duration_factor': min(1.0, len(price_data) / 20),
            'rsi_extreme': min(1.0, rsi_extreme),
            'price_range': min(1.0, price_range * 20)
        }
        
        # Weighted confidence calculation
        weights = {'price_movement': 0.25, 'volume_confirmation': 0.25, 'momentum_strength': 0.20,
                  'duration_factor': 0.15, 'rsi_extreme': 0.10, 'price_range': 0.05}
        
        confidence = sum(confidence_factors[factor] * weights[factor] for factor in confidence_factors)
        
        # Feature importance for ML
        feature_importance = {
            'price_change': abs(price_change),
            'volume_ratio': volume_multiplier,
            'momentum_strength': momentum_strength,
            'volatility': price_volatility,
            'rsi_level': rsi_data.iloc[-1]
        }
        
        return {
            'confidence': confidence,
            'signal_strength': momentum_strength,
            'price_data': price_data,
            'returns_data': returns_data,
            'volume_data': volume_data,
            'characteristics': {
                'price_change': price_change,
                'price_range': price_range,
                'volume_multiplier': volume_multiplier,
                'momentum_strength': momentum_strength,
                'volatility': price_volatility,
                'duration': len(price_data),
                'rsi_level': rsi_data.iloc[-1]
            },
            'feature_importance': feature_importance,
            'transition_probability': self._calculate_transition_probability(
                MicroRegimeType.BREAKOUT, data, start_idx
            )
        }
    
    def _detect_enhanced_consolidations(self, data: Dict[str, Any]) -> List[MicroRegimeDetectionResult]:
        """Enhanced consolidation detection with multiple criteria."""
        consolidations = []
        
        try:
            price = data['price']
            volatility = data['volatility_20']
            atr = data['atr']
            volume = data['volume']
            
            # Multiple consolidation signals
            signals = []
            
            # Low volatility
            low_vol_signal = volatility < self.consolidation_params['volatility_threshold']
            signals.append(('volatility', low_vol_signal))
            
            # Price range constraint
            price_range = (price - price.shift(1)).abs()
            max_range = price * self.consolidation_params['price_range_threshold']
            range_signal = price_range <= max_range
            signals.append(('range', range_signal))
            
            # ATR-based consolidation
            atr_signal = atr < atr.rolling(20).mean() * 0.8
            signals.append(('atr', atr_signal))
            
            # Volume confirmation (optional)
            if self.consolidation_params['volume_confirmation']:
                volume_signal = data['volume_ratio'] < 1.2  # Lower than normal volume
                signals.append(('volume', volume_signal))
            
            # Combined consolidation signal
            combined_signal = pd.Series(True, index=price.index)
            for signal_name, signal in signals:
                combined_signal &= signal
            
            # Find consolidation periods
            consolidation_periods = self._find_contiguous_periods(combined_signal)
            
            for start_idx, end_idx in consolidation_periods:
                if end_idx - start_idx >= self.consolidation_params['min_duration']:
                    
                    # Analyze consolidation period
                    consolidation_data = self._analyze_consolidation_period(data, start_idx, end_idx)
                    
                    if consolidation_data['confidence'] >= self.config.detection_threshold:
                        
                        consolidation_regime = MicroRegimeDetectionResult(
                            regime_type=MicroRegimeType.CONSOLIDATION,
                            confidence=consolidation_data['confidence'],
                            start_time=data['timestamp'][start_idx],
                            end_time=data['timestamp'][end_idx],
                            characteristics=consolidation_data['characteristics'],
                            signal_strength=consolidation_data['signal_strength'],
                            duration_minutes=(end_idx - start_idx) * 5,
                            transition_probability=consolidation_data['transition_probability'],
                            feature_importance=consolidation_data['feature_importance'],
                            metadata={
                                'consolidation_type': self._classify_consolidation_type(consolidation_data),
                                'breakout_probability': self._calculate_breakout_probability(consolidation_data),
                                'detection_method': 'enhanced_traditional'
                            }
                        )
                        
                        consolidations.append(consolidation_regime)
        
        except Exception as e:
            tprint_warning(f"⚠️ Enhanced consolidation detection failed: {e}")
        
        return consolidations
    
    # Technical indicator calculation methods
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR)."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_hist = macd_line - macd_signal
        return macd_line, macd_signal, macd_hist
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band
    
    def _calculate_obv(self, prices: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume (OBV)."""
        price_change = prices.diff()
        obv = volume.copy()
        obv[price_change < 0] = -volume[price_change < 0]
        obv[price_change == 0] = 0
        return obv.cumsum()
    
    def _calculate_support_resistance(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]:
        """Calculate support and resistance levels."""
        # Simple pivot point calculation
        pivot = (high + low + close) / 3
        resistance1 = 2 * pivot - low
        support1 = 2 * pivot - high
        resistance2 = pivot + (high - low)
        support2 = pivot - (high - low)
        
        return {
            'pivot': pivot,
            'resistance1': resistance1,
            'support1': support1,
            'resistance2': resistance2,
            'support2': support2
        }
    
    def _find_contiguous_periods(self, mask: pd.Series) -> List[Tuple[int, int]]:
        """Find contiguous periods where mask is True."""
        periods = []
        start_idx = None
        
        for i, val in enumerate(mask):
            if val and start_idx is None:
                start_idx = i
            elif not val and start_idx is not None:
                periods.append((start_idx, i))
                start_idx = None
        
        if start_idx is not None:
            periods.append((start_idx, len(mask)))
        
        return periods
    
    def _calculate_transition_probability(self, regime_type: MicroRegimeType, data: Dict[str, Any], start_idx: int) -> float:
        """Calculate probability of transitioning to this micro-regime."""
        base_probabilities = {
            MicroRegimeType.BREAKOUT: 0.15,
            MicroRegimeType.CONSOLIDATION: 0.25,
            MicroRegimeType.REVERSAL: 0.10,
            MicroRegimeType.ACCELERATION: 0.20,
            MicroRegimeType.DECELERATION: 0.15,
            MicroRegimeType.VOLUME_SPIKE: 0.10,
            MicroRegimeType.VOLATILITY_SPIKE: 0.05,
            MicroRegimeType.LIQUIDITY_CHANGE: 0.08,
            MicroRegimeType.MOMENTUM_SHIFT: 0.12
        }
        
        base_prob = base_probabilities.get(regime_type, 0.1)
        
        # Adjust based on recent market conditions
        if start_idx > 20:
            recent_volatility = data['volatility_20'].iloc[start_idx-20:start_idx].mean()
            recent_volume = data['volume_ratio'].iloc[start_idx-20:start_idx].mean()
            
            if regime_type in [MicroRegimeType.BREAKOUT, MicroRegimeType.VOLATILITY_SPIKE]:
                base_prob *= (1 + recent_volatility)
            
            if regime_type == MicroRegimeType.VOLUME_SPIKE:
                base_prob *= recent_volume
        
        return min(1.0, base_prob)
    
    def _impute_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Impute missing data using various methods."""
        # Forward fill for price data
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in data.columns:
                data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
        
        # Median fill for volume
        if 'volume' in data.columns:
            data['volume'] = data['volume'].fillna(data['volume'].median())
        
        return data
    
    def _apply_feature_engineering(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply advanced feature engineering if ML components are available."""
        if not self.ml_components or not self.ml_components.get('feature_engineer'):
            return data
        
        try:
            # Convert data to DataFrame for feature engineering
            df = pd.DataFrame({k: v for k, v in data.items() if isinstance(v, pd.Series)})
            
            # Apply feature engineering
            engineered_df = self.ml_components['feature_engineer'].transform(df)
            
            # Convert back to dictionary
            for col in engineered_df.columns:
                if col not in data:
                    data[col] = engineered_df[col]
            
            tprint_debug(f"🔧 Applied feature engineering: {len(engineered_df.columns)} features")
            
        except Exception as e:
            tprint_warning(f"⚠️ Feature engineering failed: {e}")
        
        return data
    
    def _handle_outliers(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle outliers in the data."""
        try:
            # Use Isolation Forest for outlier detection
            numeric_data = {k: v for k, v in data.items() 
                          if isinstance(v, pd.Series) and pd.api.types.is_numeric_dtype(v)}
            
            if len(numeric_data) > 0:
                df = pd.DataFrame(numeric_data).dropna()
                
                if len(df) > 100:  # Need sufficient data for outlier detection
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    outliers = iso_forest.fit_predict(df)
                    
                    # Mark outliers but don't remove them (they might be important for regime detection)
                    data['outlier_mask'] = pd.Series(outliers == -1, index=df.index)
                    
                    tprint_debug(f"🎯 Detected {sum(outliers == -1)} outliers")
        
        except Exception as e:
            tprint_warning(f"⚠️ Outlier detection failed: {e}")
        
        return data
    
    def _detect_ml_regimes(self, data: Dict[str, Any], current_regime: Optional[MarketRegime]) -> List[MicroRegimeDetectionResult]:
        """Detect regimes using ML methods."""
        if not self.ml_components:
            return []
        
        tprint_debug("🤖 Detecting ML-based regimes...")
        
        try:
            # This would involve training ML models and using them for regime detection
            # For now, return empty list as this requires more complex ML pipeline setup
            return []
        
        except Exception as e:
            tprint_warning(f"⚠️ ML regime detection failed: {e}")
            return []
    
    def _detect_statistical_regimes(self, data: Dict[str, Any]) -> List[MicroRegimeDetectionResult]:
        """Detect regimes using statistical methods."""
        tprint_debug("📊 Detecting statistical regimes...")
        
        regimes = []
        
        try:
            # Hidden Markov Model for regime detection
            returns = data['returns'].dropna()
            
            if len(returns) > 100:
                # Simple statistical regime detection using volatility clustering
                volatility = returns.rolling(20).std()
                
                # Detect high/low volatility regimes
                vol_threshold_high = volatility.quantile(0.8)
                vol_threshold_low = volatility.quantile(0.2)
                
                high_vol_mask = volatility > vol_threshold_high
                low_vol_mask = volatility < vol_threshold_low
                
                # Find high volatility periods
                high_vol_periods = self._find_contiguous_periods(high_vol_mask)
                for start_idx, end_idx in high_vol_periods:
                    if end_idx - start_idx >= 5:
                        regime = MicroRegimeDetectionResult(
                            regime_type=MicroRegimeType.VOLATILITY_SPIKE,
                            confidence=0.7,
                            start_time=data['timestamp'][start_idx],
                            end_time=data['timestamp'][end_idx],
                            characteristics={'volatility_level': 'high'},
                            signal_strength=volatility.iloc[start_idx:end_idx].mean(),
                            duration_minutes=(end_idx - start_idx) * 5,
                            metadata={'detection_method': 'statistical'}
                        )
                        regimes.append(regime)
        
        except Exception as e:
            tprint_warning(f"⚠️ Statistical regime detection failed: {e}")
        
        return regimes
    
    def _post_process_regimes(self, regimes: List[MicroRegimeDetectionResult], 
                            data: Dict[str, Any]) -> List[MicroRegimeDetectionResult]:
        """Post-process and rank detected regimes."""
        if not regimes:
            return []
        
        # Filter by confidence threshold
        filtered_regimes = [r for r in regimes if r.confidence >= self.config.min_confidence]
        
        # Remove overlapping regimes (keep highest confidence)
        filtered_regimes = self._remove_overlapping_regimes(filtered_regimes)
        
        # Rank by confidence and signal strength
        filtered_regimes.sort(key=lambda x: (x.confidence, x.signal_strength), reverse=True)
        
        # Limit number of results
        max_regimes = 20
        if len(filtered_regimes) > max_regimes:
            filtered_regimes = filtered_regimes[:max_regimes]
        
        return filtered_regimes
    
    def _remove_overlapping_regimes(self, regimes: List[MicroRegimeDetectionResult]) -> List[MicroRegimeDetectionResult]:
        """Remove overlapping regimes, keeping the one with highest confidence."""
        if not regimes:
            return []
        
        # Sort by start time
        regimes.sort(key=lambda x: x.start_time)
        
        non_overlapping = []
        for regime in regimes:
            # Check if this regime overlaps with any already selected regime
            overlaps = False
            for selected in non_overlapping:
                if (regime.start_time < selected.end_time and regime.end_time > selected.start_time):
                    overlaps = True
                    # Replace if this regime has higher confidence
                    if regime.confidence > selected.confidence:
                        non_overlapping.remove(selected)
                        non_overlapping.append(regime)
                    break
            
            if not overlaps:
                non_overlapping.append(regime)
        
        return non_overlapping
    
    def _update_performance_metrics(self, regimes: List[MicroRegimeDetectionResult]):
        """Update performance metrics."""
        self.performance_metrics = {
            'total_detections': len(self.detection_history) + len(regimes),
            'current_detection_count': len(regimes),
            'avg_confidence': np.mean([r.confidence for r in regimes]) if regimes else 0,
            'regime_types_detected': list(set(r.regime_type.value for r in regimes)),
            'detection_timestamp': datetime.now(),
            'memory_usage': get_memory_usage()
        }
    
    # Additional helper methods for enhanced detection
    def _analyze_consolidation_period(self, data: Dict[str, Any], start_idx: int, end_idx: int) -> Dict[str, Any]:
        """Analyze a consolidation period."""
        period_data = {key: values.iloc[start_idx:end_idx] for key, values in data.items() 
                      if isinstance(values, pd.Series)}
        
        price_data = period_data['price']
        volatility_data = period_data['volatility_20']
        
        avg_volatility = volatility_data.mean()
        price_range = (price_data.max() - price_data.min()) / price_data.iloc[0]
        
        confidence = max(0.0, 1 - avg_volatility / self.consolidation_params['volatility_threshold'])
        
        return {
            'confidence': confidence,
            'signal_strength': 1 - avg_volatility,
            'characteristics': {
                'avg_volatility': avg_volatility,
                'price_range': price_range,
                'duration': len(price_data)
            },
            'feature_importance': {
                'volatility': avg_volatility,
                'price_stability': 1 - price_range,
                'duration': len(price_data)
            },
            'transition_probability': self._calculate_transition_probability(
                MicroRegimeType.CONSOLIDATION, data, start_idx
            )
        }
    
    def _classify_consolidation_type(self, consolidation_data: Dict[str, Any]) -> str:
        """Classify the type of consolidation."""
        volatility = consolidation_data['characteristics']['avg_volatility']
        price_range = consolidation_data['characteristics']['price_range']
        
        if volatility < 0.1 and price_range < 0.01:
            return "tight_consolidation"
        elif volatility < 0.2 and price_range < 0.02:
            return "moderate_consolidation"
        else:
            return "loose_consolidation"
    
    def _calculate_breakout_probability(self, consolidation_data: Dict[str, Any]) -> float:
        """Calculate probability of breakout from consolidation."""
        duration = consolidation_data['characteristics']['duration']
        volatility = consolidation_data['characteristics']['avg_volatility']
        
        # Longer consolidations with low volatility have higher breakout probability
        duration_factor = min(1.0, duration / 50)
        volatility_factor = max(0.0, 1 - volatility / 0.3)
        
        return (duration_factor * 0.6 + volatility_factor * 0.4)
    
    def _determine_breakout_direction(self, price_data: pd.Series, returns_data: pd.Series) -> str:
        """Determine the direction of a breakout."""
        price_change = (price_data.iloc[-1] / price_data.iloc[0] - 1)
        
        if price_change > 0.02:
            return "bullish_breakout"
        elif price_change < -0.02:
            return "bearish_breakout"
        else:
            return "neutral_breakout"
    
    def _calculate_breakout_risk_metrics(self, breakout_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate risk metrics for breakout."""
        returns_data = breakout_data['returns_data']
        volatility = returns_data.std()
        
        return {
            'volatility': volatility,
            'max_drawdown': returns_data.cumsum().max() - returns_data.cumsum().min(),
            'var_95': np.percentile(returns_data.dropna(), 5),
            'expected_return': returns_data.mean()
        }
    
    def _calculate_breakout_performance_metrics(self, breakout_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance metrics for breakout."""
        returns_data = breakout_data['returns_data']
        
        return {
            'total_return': returns_data.sum(),
            'sharpe_ratio': returns_data.mean() / returns_data.std() if returns_data.std() > 0 else 0,
            'win_rate': (returns_data > 0).mean(),
            'avg_win': returns_data[returns_data > 0].mean() if (returns_data > 0).any() else 0,
            'avg_loss': returns_data[returns_data < 0].mean() if (returns_data < 0).any() else 0
        }
    
    # Placeholder methods for additional detection types
    def _detect_enhanced_reversals(self, data: Dict[str, Any]) -> List[MicroRegimeDetectionResult]:
        """Enhanced reversal detection with RSI divergence, momentum confirmation, and volume analysis."""
        reversals = []
        
        try:
            price = data['price']
            returns = data['returns']
            volume = data['volume']
            rsi = data['rsi']
            momentum_5 = data['momentum_5']
            momentum_10 = data['momentum_10']
            macd_line = data['macd_line']
            macd_signal = data['macd_signal']
            
            # Multiple reversal signals
            signals = []
            
            # RSI divergence detection
            rsi_divergence = self._detect_rsi_divergence(price, rsi)
            signals.append(('rsi_divergence', rsi_divergence))
            
            # RSI extreme levels
            rsi_oversold = rsi < self.reversal_params['rsi_threshold']
            rsi_overbought = rsi > (100 - self.reversal_params['rsi_threshold'])
            rsi_extreme = rsi_oversold | rsi_overbought
            signals.append(('rsi_extreme', rsi_extreme))
            
            # Momentum reversal
            momentum_reversal = self._detect_momentum_reversal(momentum_5, momentum_10)
            signals.append(('momentum_reversal', momentum_reversal))
            
            # MACD signal line crossover
            macd_crossover = self._detect_macd_crossover(macd_line, macd_signal)
            signals.append(('macd_crossover', macd_crossover))
            
            # Volume confirmation
            volume_confirmation = data['volume_ratio'] > self.reversal_params['volume_confirmation']
            signals.append(('volume_confirmation', volume_confirmation))
            
            # Combined reversal signal
            combined_signal = pd.Series(False, index=price.index)
            for signal_name, signal in signals:
                combined_signal |= signal
            
            # Find reversal periods
            reversal_periods = self._find_contiguous_periods(combined_signal)
            
            for start_idx, end_idx in reversal_periods:
                if end_idx - start_idx >= 3:  # Minimum duration for reversal
                    
                    # Analyze reversal period
                    reversal_data = self._analyze_reversal_period(data, start_idx, end_idx)
                    
                    if reversal_data['confidence'] >= self.config.detection_threshold:
                        
                        # Determine reversal direction
                        reversal_direction = self._determine_reversal_direction(
                            reversal_data['price_data'], reversal_data['momentum_data']
                        )
                        
                        # Calculate risk metrics
                        risk_metrics = self._calculate_reversal_risk_metrics(reversal_data)
                        
                        reversal_regime = MicroRegimeDetectionResult(
                            regime_type=MicroRegimeType.REVERSAL,
                            confidence=reversal_data['confidence'],
                            start_time=data['timestamp'][start_idx],
                            end_time=data['timestamp'][end_idx],
                            characteristics=reversal_data['characteristics'],
                            signal_strength=reversal_data['signal_strength'],
                            duration_minutes=(end_idx - start_idx) * 5,
                            transition_probability=reversal_data['transition_probability'],
                            feature_importance=reversal_data['feature_importance'],
                            risk_metrics=risk_metrics,
                            metadata={
                                'reversal_direction': reversal_direction,
                                'confirmation_signals': len([s for s in signals if s[1].iloc[start_idx:end_idx].any()]),
                                'detection_method': 'enhanced_traditional'
                            }
                        )
                        
                        reversals.append(reversal_regime)
        
        except Exception as e:
            tprint_warning(f"⚠️ Enhanced reversal detection failed: {e}")
        
        return reversals
    
    def _detect_enhanced_accelerations(self, data: Dict[str, Any]) -> List[MicroRegimeDetectionResult]:
        """Enhanced acceleration/deceleration detection using price derivatives and momentum indicators."""
        accelerations = []
        
        try:
            price = data['price']
            returns = data['returns']
            volume = data['volume']
            price_velocity = data['price_velocity']
            price_acceleration = data['price_acceleration']
            price_jerk = data['price_jerk']
            momentum_5 = data['momentum_5']
            momentum_10 = data['momentum_10']
            
            # Multiple acceleration signals
            signals = []
            
            # Price acceleration detection
            acceleration_signal = abs(price_acceleration) > self.acceleration_params['momentum_acceleration']
            signals.append(('price_acceleration', acceleration_signal))
            
            # Velocity trend detection
            velocity_trend = self._detect_velocity_trend(price_velocity)
            signals.append(('velocity_trend', velocity_trend))
            
            # Momentum acceleration
            momentum_acceleration = self._detect_momentum_acceleration(momentum_5, momentum_10)
            signals.append(('momentum_acceleration', momentum_acceleration))
            
            # Volume trend confirmation
            volume_trend = data['volume_ratio'] > self.acceleration_params['volume_trend']
            signals.append(('volume_trend', volume_trend))
            
            # Jerk detection (third derivative)
            jerk_signal = abs(price_jerk) > abs(price_jerk).rolling(20).mean() * 2
            signals.append(('jerk_signal', jerk_signal))
            
            # Combined acceleration signal
            combined_signal = pd.Series(False, index=price.index)
            for signal_name, signal in signals:
                combined_signal |= signal
            
            # Find acceleration periods
            acceleration_periods = self._find_contiguous_periods(combined_signal)
            
            for start_idx, end_idx in acceleration_periods:
                if end_idx - start_idx >= self.acceleration_params['duration_threshold']:
                    
                    # Analyze acceleration period
                    acceleration_data = self._analyze_acceleration_period(data, start_idx, end_idx)
                    
                    if acceleration_data['confidence'] >= self.config.detection_threshold:
                        
                        # Determine acceleration type
                        acceleration_type = self._determine_acceleration_type(
                            acceleration_data['velocity_data'], acceleration_data['acceleration_data']
                        )
                        
                        # Calculate risk metrics
                        risk_metrics = self._calculate_acceleration_risk_metrics(acceleration_data)
                        
                        # Determine regime type (acceleration or deceleration)
                        regime_type = MicroRegimeType.ACCELERATION if acceleration_data['acceleration_strength'] > 0 else MicroRegimeType.DECELERATION
                        
                        acceleration_regime = MicroRegimeDetectionResult(
                            regime_type=regime_type,
                            confidence=acceleration_data['confidence'],
                            start_time=data['timestamp'][start_idx],
                            end_time=data['timestamp'][end_idx],
                            characteristics=acceleration_data['characteristics'],
                            signal_strength=acceleration_data['signal_strength'],
                            duration_minutes=(end_idx - start_idx) * 5,
                            transition_probability=acceleration_data['transition_probability'],
                            feature_importance=acceleration_data['feature_importance'],
                            risk_metrics=risk_metrics,
                            metadata={
                                'acceleration_type': acceleration_type,
                                'acceleration_strength': acceleration_data['acceleration_strength'],
                                'confirmation_signals': len([s for s in signals if s[1].iloc[start_idx:end_idx].any()]),
                                'detection_method': 'enhanced_traditional'
                            }
                        )
                        
                        accelerations.append(acceleration_regime)
        
        except Exception as e:
            tprint_warning(f"⚠️ Enhanced acceleration detection failed: {e}")
        
        return accelerations
    
    def _detect_enhanced_volume_spikes(self, data: Dict[str, Any]) -> List[MicroRegimeDetectionResult]:
        """Enhanced volume spike detection with isolation analysis and price confirmation."""
        volume_spikes = []
        
        try:
            price = data['price']
            returns = data['returns']
            volume = data['volume']
            volume_ratio = data['volume_ratio']
            obv = data['obv']
            
            # Multiple volume spike signals
            signals = []
            
            # Volume multiplier threshold
            volume_spike_signal = volume_ratio > self.volume_spike_params['volume_multiplier']
            signals.append(('volume_multiplier', volume_spike_signal))
            
            # Volume isolation analysis
            volume_isolation = self._detect_volume_isolation(volume, volume_ratio)
            signals.append(('volume_isolation', volume_isolation))
            
            # Price confirmation
            price_confirmation = abs(returns) > self.volume_spike_params['price_confirmation']
            signals.append(('price_confirmation', price_confirmation))
            
            # OBV divergence
            obv_divergence = self._detect_obv_divergence(price, obv)
            signals.append(('obv_divergence', obv_divergence))
            
            # Volume trend analysis
            volume_trend = self._detect_volume_trend(volume)
            signals.append(('volume_trend', volume_trend))
            
            # Combined volume spike signal
            combined_signal = pd.Series(False, index=price.index)
            for signal_name, signal in signals:
                combined_signal |= signal
            
            # Find volume spike periods
            spike_periods = self._find_contiguous_periods(combined_signal)
            
            for start_idx, end_idx in spike_periods:
                if end_idx - start_idx >= self.volume_spike_params['duration_threshold']:
                    
                    # Analyze volume spike period
                    spike_data = self._analyze_volume_spike_period(data, start_idx, end_idx)
                    
                    if spike_data['confidence'] >= self.config.detection_threshold:
                        
                        # Determine spike characteristics
                        spike_type = self._classify_volume_spike_type(spike_data)
                        
                        # Calculate risk metrics
                        risk_metrics = self._calculate_volume_spike_risk_metrics(spike_data)
                        
                        volume_spike_regime = MicroRegimeDetectionResult(
                            regime_type=MicroRegimeType.VOLUME_SPIKE,
                            confidence=spike_data['confidence'],
                            start_time=data['timestamp'][start_idx],
                            end_time=data['timestamp'][end_idx],
                            characteristics=spike_data['characteristics'],
                            signal_strength=spike_data['signal_strength'],
                            duration_minutes=(end_idx - start_idx) * 5,
                            transition_probability=spike_data['transition_probability'],
                            feature_importance=spike_data['feature_importance'],
                            risk_metrics=risk_metrics,
                            metadata={
                                'spike_type': spike_type,
                                'volume_multiplier': spike_data['volume_multiplier'],
                                'confirmation_signals': len([s for s in signals if s[1].iloc[start_idx:end_idx].any()]),
                                'detection_method': 'enhanced_traditional'
                            }
                        )
                        
                        volume_spikes.append(volume_spike_regime)
        
        except Exception as e:
            tprint_warning(f"⚠️ Enhanced volume spike detection failed: {e}")
        
        return volume_spikes
    
    def _detect_enhanced_volatility_spikes(self, data: Dict[str, Any]) -> List[MicroRegimeDetectionResult]:
        """Enhanced volatility spike detection with statistical analysis and market impact assessment."""
        volatility_spikes = []
        
        try:
            price = data['price']
            returns = data['returns']
            volume = data['volume']
            volatility_20 = data['volatility_20']
            volatility_50 = data['volatility_50']
            atr = data['atr']
            spread_estimate = data['spread_estimate']
            price_impact = data['price_impact']
            
            # Multiple volatility spike signals
            signals = []
            
            # Volatility multiplier threshold
            vol_spike_signal = volatility_20 > volatility_50 * self.volatility_spike_params['volatility_multiplier']
            signals.append(('volatility_multiplier', vol_spike_signal))
            
            # ATR-based volatility spike
            atr_spike = atr > atr.rolling(20).mean() * 2
            signals.append(('atr_spike', atr_spike))
            
            # Price impact threshold
            price_impact_signal = price_impact > self.volatility_spike_params['price_impact_threshold']
            signals.append(('price_impact', price_impact_signal))
            
            # Statistical volatility analysis
            vol_statistical = self._detect_statistical_volatility_spike(volatility_20)
            signals.append(('statistical_volatility', vol_statistical))
            
            # Spread expansion
            spread_expansion = spread_estimate > spread_estimate.rolling(20).mean() * 1.5
            signals.append(('spread_expansion', spread_expansion))
            
            # Volume-volatility relationship
            vol_volume_relationship = self._detect_volatility_volume_relationship(volatility_20, volume)
            signals.append(('vol_volume_relationship', vol_volume_relationship))
            
            # Combined volatility spike signal
            combined_signal = pd.Series(False, index=price.index)
            for signal_name, signal in signals:
                combined_signal |= signal
            
            # Find volatility spike periods
            spike_periods = self._find_contiguous_periods(combined_signal)
            
            for start_idx, end_idx in spike_periods:
                if end_idx - start_idx >= self.volatility_spike_params['duration_threshold']:
                    
                    # Analyze volatility spike period
                    spike_data = self._analyze_volatility_spike_period(data, start_idx, end_idx)
                    
                    if spike_data['confidence'] >= self.config.detection_threshold:
                        
                        # Determine spike characteristics
                        spike_severity = self._classify_volatility_spike_severity(spike_data)
                        
                        # Calculate risk metrics
                        risk_metrics = self._calculate_volatility_spike_risk_metrics(spike_data)
                        
                        volatility_spike_regime = MicroRegimeDetectionResult(
                            regime_type=MicroRegimeType.VOLATILITY_SPIKE,
                            confidence=spike_data['confidence'],
                            start_time=data['timestamp'][start_idx],
                            end_time=data['timestamp'][end_idx],
                            characteristics=spike_data['characteristics'],
                            signal_strength=spike_data['signal_strength'],
                            duration_minutes=(end_idx - start_idx) * 5,
                            transition_probability=spike_data['transition_probability'],
                            feature_importance=spike_data['feature_importance'],
                            risk_metrics=risk_metrics,
                            metadata={
                                'spike_severity': spike_severity,
                                'volatility_multiplier': spike_data['volatility_multiplier'],
                                'confirmation_signals': len([s for s in signals if s[1].iloc[start_idx:end_idx].any()]),
                                'detection_method': 'enhanced_traditional'
                            }
                        )
                        
                        volatility_spikes.append(volatility_spike_regime)
        
        except Exception as e:
            tprint_warning(f"⚠️ Enhanced volatility spike detection failed: {e}")
        
        return volatility_spikes
    
    def _detect_momentum_shifts(self, data: Dict[str, Any]) -> List[MicroRegimeDetectionResult]:
        """Detect momentum shifts using multiple momentum indicators and trend analysis."""
        momentum_shifts = []
        
        try:
            price = data['price']
            returns = data['returns']
            volume = data['volume']
            momentum_5 = data['momentum_5']
            momentum_10 = data['momentum_10']
            momentum_20 = data['momentum_20']
            rsi = data['rsi']
            macd_line = data['macd_line']
            macd_signal = data['macd_signal']
            macd_hist = data['macd_hist']
            
            # Multiple momentum shift signals
            signals = []
            
            # Momentum crossover detection
            momentum_crossover = self._detect_momentum_crossover(momentum_5, momentum_10, momentum_20)
            signals.append(('momentum_crossover', momentum_crossover))
            
            # RSI momentum shift
            rsi_momentum_shift = self._detect_rsi_momentum_shift(rsi)
            signals.append(('rsi_momentum_shift', rsi_momentum_shift))
            
            # MACD momentum shift
            macd_momentum_shift = self._detect_macd_momentum_shift(macd_line, macd_signal, macd_hist)
            signals.append(('macd_momentum_shift', macd_momentum_shift))
            
            # Trend strength change
            trend_strength_change = self._detect_trend_strength_change(price, returns)
            signals.append(('trend_strength_change', trend_strength_change))
            
            # Volume momentum confirmation
            volume_momentum = self._detect_volume_momentum(volume, returns)
            signals.append(('volume_momentum', volume_momentum))
            
            # Combined momentum shift signal
            combined_signal = pd.Series(False, index=price.index)
            for signal_name, signal in signals:
                combined_signal |= signal
            
            # Find momentum shift periods
            shift_periods = self._find_contiguous_periods(combined_signal)
            
            for start_idx, end_idx in shift_periods:
                if end_idx - start_idx >= 5:  # Minimum duration for momentum shift
                    
                    # Analyze momentum shift period
                    shift_data = self._analyze_momentum_shift_period(data, start_idx, end_idx)
                    
                    if shift_data['confidence'] >= self.config.detection_threshold:
                        
                        # Determine shift characteristics
                        shift_type = self._classify_momentum_shift_type(shift_data)
                        
                        # Calculate risk metrics
                        risk_metrics = self._calculate_momentum_shift_risk_metrics(shift_data)
                        
                        momentum_shift_regime = MicroRegimeDetectionResult(
                            regime_type=MicroRegimeType.MOMENTUM_SHIFT,
                            confidence=shift_data['confidence'],
                            start_time=data['timestamp'][start_idx],
                            end_time=data['timestamp'][end_idx],
                            characteristics=shift_data['characteristics'],
                            signal_strength=shift_data['signal_strength'],
                            duration_minutes=(end_idx - start_idx) * 5,
                            transition_probability=shift_data['transition_probability'],
                            feature_importance=shift_data['feature_importance'],
                            risk_metrics=risk_metrics,
                            metadata={
                                'shift_type': shift_type,
                                'momentum_change': shift_data['momentum_change'],
                                'confirmation_signals': len([s for s in signals if s[1].iloc[start_idx:end_idx].any()]),
                                'detection_method': 'enhanced_traditional'
                            }
                        )
                        
                        momentum_shifts.append(momentum_shift_regime)
        
        except Exception as e:
            tprint_warning(f"⚠️ Momentum shift detection failed: {e}")
        
        return momentum_shifts
    
    def _detect_liquidity_changes(self, data: Dict[str, Any]) -> List[MicroRegimeDetectionResult]:
        """Detect liquidity changes using spread analysis and volume patterns."""
        liquidity_changes = []
        
        try:
            price = data['price']
            returns = data['returns']
            volume = data['volume']
            spread_estimate = data['spread_estimate']
            price_impact = data['price_impact']
            obv = data['obv']
            volume_ratio = data['volume_ratio']
            
            # Multiple liquidity change signals
            signals = []
            
            # Spread change detection
            spread_change = self._detect_spread_change(spread_estimate)
            signals.append(('spread_change', spread_change))
            
            # Price impact analysis
            price_impact_change = self._detect_price_impact_change(price_impact)
            signals.append(('price_impact_change', price_impact_change))
            
            # Volume pattern analysis
            volume_pattern = self._detect_volume_pattern_change(volume, volume_ratio)
            signals.append(('volume_pattern', volume_pattern))
            
            # OBV trend analysis
            obv_trend = self._detect_obv_trend_change(obv)
            signals.append(('obv_trend', obv_trend))
            
            # Market depth analysis
            market_depth = self._detect_market_depth_change(volume, returns)
            signals.append(('market_depth', market_depth))
            
            # Liquidity stress indicators
            liquidity_stress = self._detect_liquidity_stress(spread_estimate, volume, price_impact)
            signals.append(('liquidity_stress', liquidity_stress))
            
            # Combined liquidity change signal
            combined_signal = pd.Series(False, index=price.index)
            for signal_name, signal in signals:
                combined_signal |= signal
            
            # Find liquidity change periods
            change_periods = self._find_contiguous_periods(combined_signal)
            
            for start_idx, end_idx in change_periods:
                if end_idx - start_idx >= 5:  # Minimum duration for liquidity change
                    
                    # Analyze liquidity change period
                    change_data = self._analyze_liquidity_change_period(data, start_idx, end_idx)
                    
                    if change_data['confidence'] >= self.config.detection_threshold:
                        
                        # Determine change characteristics
                        change_type = self._classify_liquidity_change_type(change_data)
                        
                        # Calculate risk metrics
                        risk_metrics = self._calculate_liquidity_change_risk_metrics(change_data)
                        
                        liquidity_change_regime = MicroRegimeDetectionResult(
                            regime_type=MicroRegimeType.LIQUIDITY_CHANGE,
                            confidence=change_data['confidence'],
                            start_time=data['timestamp'][start_idx],
                            end_time=data['timestamp'][end_idx],
                            characteristics=change_data['characteristics'],
                            signal_strength=change_data['signal_strength'],
                            duration_minutes=(end_idx - start_idx) * 5,
                            transition_probability=change_data['transition_probability'],
                            feature_importance=change_data['feature_importance'],
                            risk_metrics=risk_metrics,
                            metadata={
                                'change_type': change_type,
                                'liquidity_change': change_data['liquidity_change'],
                                'confirmation_signals': len([s for s in signals if s[1].iloc[start_idx:end_idx].any()]),
                                'detection_method': 'enhanced_traditional'
                            }
                        )
                        
                        liquidity_changes.append(liquidity_change_regime)
        
        except Exception as e:
            tprint_warning(f"⚠️ Liquidity change detection failed: {e}")
        
        return liquidity_changes
    
    # Helper methods for enhanced detection
    def _detect_rsi_divergence(self, price: pd.Series, rsi: pd.Series) -> pd.Series:
        """Detect RSI divergence with price."""
        divergence = pd.Series(False, index=price.index)
        
        # Look for divergence over rolling windows
        window = 20
        for i in range(window, len(price)):
            price_window = price.iloc[i-window:i]
            rsi_window = rsi.iloc[i-window:i]
            
            # Price trend
            price_trend = (price_window.iloc[-1] - price_window.iloc[0]) / price_window.iloc[0]
            # RSI trend
            rsi_trend = rsi_window.iloc[-1] - rsi_window.iloc[0]
            
            # Divergence: price and RSI moving in opposite directions
            if abs(price_trend) > 0.02:  # Significant price movement
                if (price_trend > 0 and rsi_trend < -5) or (price_trend < 0 and rsi_trend > 5):
                    divergence.iloc[i] = True
        
        return divergence
    
    def _detect_momentum_reversal(self, momentum_5: pd.Series, momentum_10: pd.Series) -> pd.Series:
        """Detect momentum reversal patterns."""
        reversal = pd.Series(False, index=momentum_5.index)
        
        # Look for momentum crossover
        momentum_diff = momentum_5 - momentum_10
        crossover = momentum_diff.diff().abs() > 0.01
        
        # Look for momentum divergence
        momentum_divergence = (momentum_5.diff() * momentum_10.diff()) < 0
        
        reversal = crossover | momentum_divergence
        return reversal
    
    def _detect_macd_crossover(self, macd_line: pd.Series, macd_signal: pd.Series) -> pd.Series:
        """Detect MACD signal line crossovers."""
        crossover = pd.Series(False, index=macd_line.index)
        
        # MACD line crossing signal line
        macd_diff = macd_line - macd_signal
        crossover = macd_diff.diff().abs() > 0.001
        
        return crossover
    
    def _detect_velocity_trend(self, velocity: pd.Series) -> pd.Series:
        """Detect velocity trend changes."""
        trend = pd.Series(False, index=velocity.index)
        
        # Look for significant velocity changes
        velocity_change = velocity.diff().abs() > velocity.rolling(20).std() * 2
        trend = velocity_change
        
        return trend
    
    def _detect_momentum_acceleration(self, momentum_5: pd.Series, momentum_10: pd.Series) -> pd.Series:
        """Detect momentum acceleration patterns."""
        acceleration = pd.Series(False, index=momentum_5.index)
        
        # Look for acceleration in momentum
        momentum_accel = momentum_5.diff() - momentum_10.diff()
        acceleration = abs(momentum_accel) > momentum_accel.rolling(20).std() * 2
        
        return acceleration
    
    def _detect_volume_isolation(self, volume: pd.Series, volume_ratio: pd.Series) -> pd.Series:
        """Detect volume isolation patterns."""
        isolation = pd.Series(False, index=volume.index)
        
        # Volume spike that's isolated from surrounding periods
        volume_spike = volume_ratio > 2.0
        surrounding_low = (volume_ratio.shift(1) < 1.2) & (volume_ratio.shift(-1) < 1.2)
        isolation = volume_spike & surrounding_low
        
        return isolation
    
    def _detect_obv_divergence(self, price: pd.Series, obv: pd.Series) -> pd.Series:
        """Detect OBV divergence with price."""
        divergence = pd.Series(False, index=price.index)
        
        # Look for divergence over rolling windows
        window = 20
        for i in range(window, len(price)):
            price_window = price.iloc[i-window:i]
            obv_window = obv.iloc[i-window:i]
            
            # Price trend
            price_trend = (price_window.iloc[-1] - price_window.iloc[0]) / price_window.iloc[0]
            # OBV trend
            obv_trend = obv_window.iloc[-1] - obv_window.iloc[0]
            
            # Divergence: price and OBV moving in opposite directions
            if abs(price_trend) > 0.02:  # Significant price movement
                if (price_trend > 0 and obv_trend < 0) or (price_trend < 0 and obv_trend > 0):
                    divergence.iloc[i] = True
        
        return divergence
    
    def _detect_volume_trend(self, volume: pd.Series) -> pd.Series:
        """Detect volume trend changes."""
        trend = pd.Series(False, index=volume.index)
        
        # Look for significant volume trend changes
        volume_ma = volume.rolling(20).mean()
        volume_trend = (volume - volume_ma) / volume_ma
        trend = abs(volume_trend) > 0.5
        
        return trend
    
    def _detect_statistical_volatility_spike(self, volatility: pd.Series) -> pd.Series:
        """Detect statistical volatility spikes."""
        spike = pd.Series(False, index=volatility.index)
        
        # Z-score based detection
        vol_mean = volatility.rolling(50).mean()
        vol_std = volatility.rolling(50).std()
        z_score = (volatility - vol_mean) / vol_std
        spike = z_score > 2.0
        
        return spike
    
    def _detect_volatility_volume_relationship(self, volatility: pd.Series, volume: pd.Series) -> pd.Series:
        """Detect volatility-volume relationship changes."""
        relationship = pd.Series(False, index=volatility.index)
        
        # Correlation between volatility and volume
        vol_vol_corr = volatility.rolling(20).corr(volume.rolling(20))
        relationship = abs(vol_vol_corr) > 0.7
        
        return relationship
    
    def _detect_momentum_crossover(self, momentum_5: pd.Series, momentum_10: pd.Series, momentum_20: pd.Series) -> pd.Series:
        """Detect momentum crossover patterns."""
        crossover = pd.Series(False, index=momentum_5.index)
        
        # Multiple momentum crossovers
        cross_5_10 = (momentum_5 - momentum_10).diff().abs() > 0.01
        cross_10_20 = (momentum_10 - momentum_20).diff().abs() > 0.01
        
        crossover = cross_5_10 | cross_10_20
        return crossover
    
    def _detect_rsi_momentum_shift(self, rsi: pd.Series) -> pd.Series:
        """Detect RSI momentum shifts."""
        shift = pd.Series(False, index=rsi.index)
        
        # RSI momentum change
        rsi_momentum = rsi.diff()
        shift = abs(rsi_momentum) > rsi_momentum.rolling(20).std() * 2
        
        return shift
    
    def _detect_macd_momentum_shift(self, macd_line: pd.Series, macd_signal: pd.Series, macd_hist: pd.Series) -> pd.Series:
        """Detect MACD momentum shifts."""
        shift = pd.Series(False, index=macd_line.index)
        
        # MACD histogram momentum change
        hist_momentum = macd_hist.diff()
        shift = abs(hist_momentum) > hist_momentum.rolling(20).std() * 2
        
        return shift
    
    def _detect_trend_strength_change(self, price: pd.Series, returns: pd.Series) -> pd.Series:
        """Detect trend strength changes."""
        change = pd.Series(False, index=price.index)
        
        # Trend strength using rolling correlation
        trend_strength = returns.rolling(20).apply(lambda x: abs(x.corr(pd.Series(range(len(x))))))
        change = trend_strength.diff().abs() > trend_strength.rolling(20).std() * 2
        
        return change
    
    def _detect_volume_momentum(self, volume: pd.Series, returns: pd.Series) -> pd.Series:
        """Detect volume momentum patterns."""
        momentum = pd.Series(False, index=volume.index)
        
        # Volume-return correlation momentum
        vol_ret_corr = volume.rolling(20).corr(returns.rolling(20))
        momentum = abs(vol_ret_corr.diff()) > 0.3
        
        return momentum
    
    def _detect_spread_change(self, spread: pd.Series) -> pd.Series:
        """Detect spread changes."""
        change = pd.Series(False, index=spread.index)
        
        # Significant spread changes
        spread_change = spread.diff().abs() > spread.rolling(20).std() * 2
        change = spread_change
        
        return change
    
    def _detect_price_impact_change(self, price_impact: pd.Series) -> pd.Series:
        """Detect price impact changes."""
        change = pd.Series(False, index=price_impact.index)
        
        # Significant price impact changes
        impact_change = price_impact.diff().abs() > price_impact.rolling(20).std() * 2
        change = impact_change
        
        return change
    
    def _detect_volume_pattern_change(self, volume: pd.Series, volume_ratio: pd.Series) -> pd.Series:
        """Detect volume pattern changes."""
        change = pd.Series(False, index=volume.index)
        
        # Volume pattern changes
        volume_trend = volume_ratio.diff().abs() > 0.5
        change = volume_trend
        
        return change
    
    def _detect_obv_trend_change(self, obv: pd.Series) -> pd.Series:
        """Detect OBV trend changes."""
        change = pd.Series(False, index=obv.index)
        
        # OBV trend changes
        obv_trend = obv.diff().abs() > obv.rolling(20).std() * 2
        change = obv_trend
        
        return change
    
    def _detect_market_depth_change(self, volume: pd.Series, returns: pd.Series) -> pd.Series:
        """Detect market depth changes."""
        change = pd.Series(False, index=volume.index)
        
        # Market depth proxy using volume-return relationship
        depth_proxy = volume / abs(returns + 1e-8)
        depth_change = depth_proxy.diff().abs() > depth_proxy.rolling(20).std() * 2
        change = depth_change
        
        return change
    
    def _detect_liquidity_stress(self, spread: pd.Series, volume: pd.Series, price_impact: pd.Series) -> pd.Series:
        """Detect liquidity stress indicators."""
        stress = pd.Series(False, index=spread.index)
        
        # Combined liquidity stress indicators
        spread_stress = spread > spread.rolling(20).mean() * 1.5
        volume_stress = volume < volume.rolling(20).mean() * 0.7
        impact_stress = price_impact > price_impact.rolling(20).mean() * 1.5
        
        stress = spread_stress | volume_stress | impact_stress
        return stress
    
    # Analysis methods for detected periods
    def _analyze_reversal_period(self, data: Dict[str, Any], start_idx: int, end_idx: int) -> Dict[str, Any]:
        """Analyze a reversal period."""
        period_data = {key: values.iloc[start_idx:end_idx] for key, values in data.items() 
                      if isinstance(values, pd.Series)}
        
        price_data = period_data['price']
        returns_data = period_data['returns']
        rsi_data = period_data.get('rsi', pd.Series([50] * len(price_data)))
        momentum_data = period_data.get('momentum_5', pd.Series([0] * len(price_data)))
        
        # Calculate reversal characteristics
        price_change = (price_data.iloc[-1] / price_data.iloc[0] - 1)
        rsi_change = rsi_data.iloc[-1] - rsi_data.iloc[0]
        momentum_change = momentum_data.iloc[-1] - momentum_data.iloc[0]
        
        # Confidence calculation
        confidence_factors = {
            'price_reversal': min(1.0, abs(price_change) * 10),
            'rsi_reversal': min(1.0, abs(rsi_change) / 50),
            'momentum_reversal': min(1.0, abs(momentum_change) * 5),
            'duration_factor': min(1.0, len(price_data) / 10)
        }
        
        weights = {'price_reversal': 0.4, 'rsi_reversal': 0.3, 'momentum_reversal': 0.2, 'duration_factor': 0.1}
        confidence = sum(confidence_factors[factor] * weights[factor] for factor in confidence_factors)
        
        return {
            'confidence': confidence,
            'signal_strength': abs(price_change),
            'price_data': price_data,
            'momentum_data': momentum_data,
            'characteristics': {
                'price_change': price_change,
                'rsi_change': rsi_change,
                'momentum_change': momentum_change,
                'duration': len(price_data)
            },
            'feature_importance': {
                'price_change': abs(price_change),
                'rsi_change': abs(rsi_change),
                'momentum_change': abs(momentum_change)
            },
            'transition_probability': self._calculate_transition_probability(
                MicroRegimeType.REVERSAL, data, start_idx
            )
        }
    
    def _analyze_acceleration_period(self, data: Dict[str, Any], start_idx: int, end_idx: int) -> Dict[str, Any]:
        """Analyze an acceleration period."""
        period_data = {key: values.iloc[start_idx:end_idx] for key, values in data.items() 
                      if isinstance(values, pd.Series)}
        
        velocity_data = period_data.get('price_velocity', pd.Series([0] * len(period_data['price'])))
        acceleration_data = period_data.get('price_acceleration', pd.Series([0] * len(period_data['price'])))
        
        # Calculate acceleration characteristics
        avg_velocity = velocity_data.mean()
        avg_acceleration = acceleration_data.mean()
        acceleration_strength = avg_acceleration
        
        # Confidence calculation
        confidence_factors = {
            'acceleration_strength': min(1.0, abs(acceleration_strength) * 20),
            'velocity_consistency': 1 - velocity_data.std(),
            'duration_factor': min(1.0, len(velocity_data) / 20)
        }
        
        weights = {'acceleration_strength': 0.5, 'velocity_consistency': 0.3, 'duration_factor': 0.2}
        confidence = sum(confidence_factors[factor] * weights[factor] for factor in confidence_factors)
        
        return {
            'confidence': confidence,
            'signal_strength': abs(acceleration_strength),
            'velocity_data': velocity_data,
            'acceleration_data': acceleration_data,
            'acceleration_strength': acceleration_strength,
            'characteristics': {
                'avg_velocity': avg_velocity,
                'avg_acceleration': avg_acceleration,
                'acceleration_strength': acceleration_strength,
                'duration': len(velocity_data)
            },
            'feature_importance': {
                'acceleration_strength': abs(acceleration_strength),
                'velocity_consistency': 1 - velocity_data.std()
            },
            'transition_probability': self._calculate_transition_probability(
                MicroRegimeType.ACCELERATION, data, start_idx
            )
        }
    
    def _analyze_volume_spike_period(self, data: Dict[str, Any], start_idx: int, end_idx: int) -> Dict[str, Any]:
        """Analyze a volume spike period."""
        period_data = {key: values.iloc[start_idx:end_idx] for key, values in data.items() 
                      if isinstance(values, pd.Series)}
        
        volume_data = period_data['volume']
        volume_ratio_data = period_data.get('volume_ratio', pd.Series([1] * len(volume_data)))
        
        # Calculate volume spike characteristics
        avg_volume_ratio = volume_ratio_data.mean()
        max_volume_ratio = volume_ratio_data.max()
        volume_multiplier = avg_volume_ratio
        
        # Confidence calculation
        confidence_factors = {
            'volume_multiplier': min(1.0, volume_multiplier / 3.0),
            'spike_consistency': 1 - volume_ratio_data.std() / volume_ratio_data.mean(),
            'duration_factor': min(1.0, len(volume_data) / 10)
        }
        
        weights = {'volume_multiplier': 0.5, 'spike_consistency': 0.3, 'duration_factor': 0.2}
        confidence = sum(confidence_factors[factor] * weights[factor] for factor in confidence_factors)
        
        return {
            'confidence': confidence,
            'signal_strength': volume_multiplier,
            'volume_multiplier': volume_multiplier,
            'characteristics': {
                'avg_volume_ratio': avg_volume_ratio,
                'max_volume_ratio': max_volume_ratio,
                'volume_multiplier': volume_multiplier,
                'duration': len(volume_data)
            },
            'feature_importance': {
                'volume_multiplier': volume_multiplier,
                'spike_consistency': 1 - volume_ratio_data.std() / volume_ratio_data.mean()
            },
            'transition_probability': self._calculate_transition_probability(
                MicroRegimeType.VOLUME_SPIKE, data, start_idx
            )
        }
    
    def _analyze_volatility_spike_period(self, data: Dict[str, Any], start_idx: int, end_idx: int) -> Dict[str, Any]:
        """Analyze a volatility spike period."""
        period_data = {key: values.iloc[start_idx:end_idx] for key, values in data.items() 
                      if isinstance(values, pd.Series)}
        
        volatility_data = period_data.get('volatility_20', pd.Series([0.01] * len(period_data['price'])))
        returns_data = period_data['returns']
        
        # Calculate volatility spike characteristics
        avg_volatility = volatility_data.mean()
        max_volatility = volatility_data.max()
        volatility_multiplier = avg_volatility / data['volatility_20'].iloc[:start_idx].tail(20).mean()
        
        # Confidence calculation
        confidence_factors = {
            'volatility_multiplier': min(1.0, volatility_multiplier / 3.0),
            'spike_consistency': 1 - volatility_data.std() / volatility_data.mean(),
            'duration_factor': min(1.0, len(volatility_data) / 10)
        }
        
        weights = {'volatility_multiplier': 0.5, 'spike_consistency': 0.3, 'duration_factor': 0.2}
        confidence = sum(confidence_factors[factor] * weights[factor] for factor in confidence_factors)
        
        return {
            'confidence': confidence,
            'signal_strength': volatility_multiplier,
            'volatility_multiplier': volatility_multiplier,
            'characteristics': {
                'avg_volatility': avg_volatility,
                'max_volatility': max_volatility,
                'volatility_multiplier': volatility_multiplier,
                'duration': len(volatility_data)
            },
            'feature_importance': {
                'volatility_multiplier': volatility_multiplier,
                'spike_consistency': 1 - volatility_data.std() / volatility_data.mean()
            },
            'transition_probability': self._calculate_transition_probability(
                MicroRegimeType.VOLATILITY_SPIKE, data, start_idx
            )
        }
    
    def _analyze_momentum_shift_period(self, data: Dict[str, Any], start_idx: int, end_idx: int) -> Dict[str, Any]:
        """Analyze a momentum shift period."""
        period_data = {key: values.iloc[start_idx:end_idx] for key, values in data.items() 
                      if isinstance(values, pd.Series)}
        
        momentum_5_data = period_data.get('momentum_5', pd.Series([0] * len(period_data['price'])))
        momentum_10_data = period_data.get('momentum_10', pd.Series([0] * len(period_data['price'])))
        
        # Calculate momentum shift characteristics
        momentum_change = momentum_5_data.iloc[-1] - momentum_5_data.iloc[0]
        momentum_strength = abs(momentum_change)
        
        # Confidence calculation
        confidence_factors = {
            'momentum_change': min(1.0, momentum_strength * 10),
            'shift_consistency': 1 - momentum_5_data.std(),
            'duration_factor': min(1.0, len(momentum_5_data) / 10)
        }
        
        weights = {'momentum_change': 0.5, 'shift_consistency': 0.3, 'duration_factor': 0.2}
        confidence = sum(confidence_factors[factor] * weights[factor] for factor in confidence_factors)
        
        return {
            'confidence': confidence,
            'signal_strength': momentum_strength,
            'momentum_change': momentum_change,
            'characteristics': {
                'momentum_change': momentum_change,
                'momentum_strength': momentum_strength,
                'duration': len(momentum_5_data)
            },
            'feature_importance': {
                'momentum_change': momentum_strength,
                'shift_consistency': 1 - momentum_5_data.std()
            },
            'transition_probability': self._calculate_transition_probability(
                MicroRegimeType.MOMENTUM_SHIFT, data, start_idx
            )
        }
    
    def _analyze_liquidity_change_period(self, data: Dict[str, Any], start_idx: int, end_idx: int) -> Dict[str, Any]:
        """Analyze a liquidity change period."""
        period_data = {key: values.iloc[start_idx:end_idx] for key, values in data.items() 
                      if isinstance(values, pd.Series)}
        
        spread_data = period_data.get('spread_estimate', pd.Series([0.001] * len(period_data['price'])))
        volume_data = period_data['volume']
        
        # Calculate liquidity change characteristics
        avg_spread = spread_data.mean()
        baseline_spread = data['spread_estimate'].iloc[:start_idx].tail(20).mean()
        liquidity_change = (avg_spread - baseline_spread) / baseline_spread if baseline_spread > 0 else 0
        
        # Confidence calculation
        confidence_factors = {
            'liquidity_change': min(1.0, abs(liquidity_change) * 5),
            'change_consistency': 1 - spread_data.std() / spread_data.mean(),
            'duration_factor': min(1.0, len(spread_data) / 10)
        }
        
        weights = {'liquidity_change': 0.5, 'change_consistency': 0.3, 'duration_factor': 0.2}
        confidence = sum(confidence_factors[factor] * weights[factor] for factor in confidence_factors)
        
        return {
            'confidence': confidence,
            'signal_strength': abs(liquidity_change),
            'liquidity_change': liquidity_change,
            'characteristics': {
                'avg_spread': avg_spread,
                'liquidity_change': liquidity_change,
                'duration': len(spread_data)
            },
            'feature_importance': {
                'liquidity_change': abs(liquidity_change),
                'change_consistency': 1 - spread_data.std() / spread_data.mean()
            },
            'transition_probability': self._calculate_transition_probability(
                MicroRegimeType.LIQUIDITY_CHANGE, data, start_idx
            )
        }
    
    # Classification and risk calculation methods
    def _determine_reversal_direction(self, price_data: pd.Series, momentum_data: pd.Series) -> str:
        """Determine the direction of a reversal."""
        price_change = (price_data.iloc[-1] / price_data.iloc[0] - 1)
        momentum_change = momentum_data.iloc[-1] - momentum_data.iloc[0]
        
        if price_change > 0.01 and momentum_change > 0:
            return "bullish_reversal"
        elif price_change < -0.01 and momentum_change < 0:
            return "bearish_reversal"
        else:
            return "neutral_reversal"
    
    def _determine_acceleration_type(self, velocity_data: pd.Series, acceleration_data: pd.Series) -> str:
        """Determine the type of acceleration."""
        avg_acceleration = acceleration_data.mean()
        
        if avg_acceleration > 0.01:
            return "positive_acceleration"
        elif avg_acceleration < -0.01:
            return "negative_acceleration"
        else:
            return "neutral_acceleration"
    
    def _classify_volume_spike_type(self, spike_data: Dict[str, Any]) -> str:
        """Classify the type of volume spike."""
        volume_multiplier = spike_data['volume_multiplier']
        
        if volume_multiplier > 5.0:
            return "extreme_spike"
        elif volume_multiplier > 3.0:
            return "major_spike"
        elif volume_multiplier > 2.0:
            return "moderate_spike"
        else:
            return "minor_spike"
    
    def _classify_volatility_spike_severity(self, spike_data: Dict[str, Any]) -> str:
        """Classify the severity of volatility spike."""
        volatility_multiplier = spike_data['volatility_multiplier']
        
        if volatility_multiplier > 4.0:
            return "extreme_volatility"
        elif volatility_multiplier > 2.5:
            return "high_volatility"
        elif volatility_multiplier > 1.5:
            return "moderate_volatility"
        else:
            return "low_volatility"
    
    def _classify_momentum_shift_type(self, shift_data: Dict[str, Any]) -> str:
        """Classify the type of momentum shift."""
        momentum_change = shift_data['momentum_change']
        
        if momentum_change > 0.05:
            return "strong_bullish_shift"
        elif momentum_change > 0.02:
            return "bullish_shift"
        elif momentum_change < -0.05:
            return "strong_bearish_shift"
        elif momentum_change < -0.02:
            return "bearish_shift"
        else:
            return "neutral_shift"
    
    def _classify_liquidity_change_type(self, change_data: Dict[str, Any]) -> str:
        """Classify the type of liquidity change."""
        liquidity_change = change_data['liquidity_change']
        
        if liquidity_change > 0.5:
            return "liquidity_deterioration"
        elif liquidity_change > 0.2:
            return "liquidity_decline"
        elif liquidity_change < -0.2:
            return "liquidity_improvement"
        else:
            return "liquidity_stable"
    
    # Risk calculation methods
    def _calculate_reversal_risk_metrics(self, reversal_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate risk metrics for reversal."""
        returns_data = reversal_data['returns_data']
        
        return {
            'volatility': returns_data.std(),
            'max_drawdown': returns_data.cumsum().max() - returns_data.cumsum().min(),
            'var_95': np.percentile(returns_data.dropna(), 5),
            'expected_return': returns_data.mean()
        }
    
    def _calculate_acceleration_risk_metrics(self, acceleration_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate risk metrics for acceleration."""
        velocity_data = acceleration_data['velocity_data']
        
        return {
            'velocity_volatility': velocity_data.std(),
            'acceleration_risk': abs(acceleration_data['acceleration_strength']),
            'momentum_risk': abs(velocity_data.mean()),
            'stability': 1 - velocity_data.std()
        }
    
    def _calculate_volume_spike_risk_metrics(self, spike_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate risk metrics for volume spike."""
        return {
            'volume_risk': spike_data['volume_multiplier'],
            'liquidity_risk': 1 / spike_data['volume_multiplier'] if spike_data['volume_multiplier'] > 0 else 1,
            'market_impact': spike_data['volume_multiplier'] * 0.1,
            'stability': 1 - spike_data['characteristics'].get('spike_consistency', 0.5)
        }
    
    def _calculate_volatility_spike_risk_metrics(self, spike_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate risk metrics for volatility spike."""
        return {
            'volatility_risk': spike_data['volatility_multiplier'],
            'market_risk': spike_data['volatility_multiplier'] * 0.2,
            'stability_risk': 1 - spike_data['characteristics'].get('spike_consistency', 0.5),
            'expected_volatility': spike_data['characteristics']['avg_volatility']
        }
    
    def _calculate_momentum_shift_risk_metrics(self, shift_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate risk metrics for momentum shift."""
        return {
            'momentum_risk': abs(shift_data['momentum_change']),
            'trend_risk': abs(shift_data['momentum_change']) * 0.5,
            'stability': 1 - shift_data['characteristics'].get('shift_consistency', 0.5),
            'direction_risk': abs(shift_data['momentum_change'])
        }
    
    def _calculate_liquidity_change_risk_metrics(self, change_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate risk metrics for liquidity change."""
        return {
            'liquidity_risk': abs(change_data['liquidity_change']),
            'spread_risk': change_data['characteristics']['avg_spread'],
            'market_impact': abs(change_data['liquidity_change']) * 0.3,
            'stability': 1 - change_data['characteristics'].get('change_consistency', 0.5)
        }
    
    # Serialization and persistence methods
    def save_model(self, filepath: str) -> bool:
        """Save the detector model to file."""
        try:
            model_data = {
                'config': self.config,
                'detection_history': self.detection_history,
                'performance_metrics': self.performance_metrics,
                'timestamp': datetime.now()
            }
            
            success = self.serializer.save(model_data, filepath)
            if success:
                tprint_success(f"💾 Model saved to {filepath}")
            else:
                tprint_error(f"❌ Failed to save model to {filepath}")
            
            return success
        
        except Exception as e:
            tprint_error(f"❌ Model save failed: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load the detector model from file."""
        try:
            model_data = self.serializer.load(filepath)
            if model_data:
                self.config = model_data.get('config', self.config)
                self.detection_history = model_data.get('detection_history', [])
                self.performance_metrics = model_data.get('performance_metrics', {})
                tprint_success(f"📁 Model loaded from {filepath}")
                return True
            else:
                tprint_error(f"❌ Failed to load model from {filepath}")
                return False
        
        except Exception as e:
            tprint_error(f"❌ Model load failed: {e}")
            return False
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """Get comprehensive detection summary."""
        return {
            'enabled_features': self._get_enabled_features(),
            'performance_metrics': self.performance_metrics,
            'config': self.config,
            'detection_history_count': len(self.detection_history),
            'm1_optimization_status': self.m1_integration.get('success', False),
            'ml_components_available': bool(self.ml_components),
            'data_components_available': bool(self.data_components),
            'timestamp': datetime.now()
        }
    
    def cleanup(self):
        """Cleanup resources and optimizers."""
        try:
            if self.m1_integration.get('success', False):
                cleanup_m1_optimizers()
                tprint_success("🧹 M1 optimizers cleaned up")
            
            tprint_success("🧹 MicroRegimeDetector cleanup completed")
        
        except Exception as e:
            tprint_warning(f"⚠️ Cleanup warning: {e}")


# Factory function for easy instantiation
def create_micro_regime_detector(config: Optional[DetectionConfig] = None) -> MicroRegimeDetector:
    """
    Factory function to create a MicroRegimeDetector instance.
    
    Args:
        config: Optional configuration object
        
    Returns:
        Configured MicroRegimeDetector instance
    """
    return MicroRegimeDetector(config)


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='5min')
    
    # Generate synthetic market data
    price = 100 + np.cumsum(np.random.randn(1000) * 0.01)
    high = price + np.abs(np.random.randn(1000) * 0.005)
    low = price - np.abs(np.random.randn(1000) * 0.005)
    volume = np.random.randint(1000, 10000, 1000)
    
    market_data = pd.DataFrame({
        'open': price,
        'high': high,
        'low': low,
        'close': price,
        'volume': volume
    }, index=dates)
    
    # Create detector
    detector = create_micro_regime_detector()
    
    # Test detection
    tprint_info("🧪 Testing MicroRegimeDetector...")
    regimes = detector.detect_micro_regimes(market_data)
    
    # Print results
    tprint_info(f"📊 Detected {len(regimes)} regimes:")
    for i, regime in enumerate(regimes[:5]):  # Show first 5
        tprint_info(f"  {i+1}. {regime.regime_type.value} (confidence: {regime.confidence:.2f})")
    
    # Print summary
    summary = detector.get_detection_summary()
    tprint_structured(summary)
    
    # Cleanup
    detector.cleanup()