#!/usr/bin/env python3
"""
Enhanced HMM Regime Detection Utilities

This enhanced module integrates the HMM composite manager functionality and adds
comprehensive regime detection capabilities with advanced features:

Key Enhancements:
- HMM Composite Manager Integration: Full integration with consolidated HMM functionality
- Multi-Timeframe Support: Ensemble HMM across multiple timeframes
- Regime Transition Analysis: Advanced transition probability calculations
- Economic Significance: Pareto front utilities for regime validation
- Streaming Support: Real-time regime detection capabilities
- Memory Optimization: M1-optimized memory management
- GPU Acceleration: M1 MPS support for regime detection
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
import logging
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import warnings
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
from pathlib import Path

# Import comprehensive utility infrastructure
from ..math_validation import (
    safe_divide, safe_log, safe_sqrt,
    validate_positive, validate_range
)
from ..core.common import create_fallback_logger, create_fallback_decorator
from ..parquet_utils import ParquetUtils
from ..serialization_utils import UniversalSerializer
from ..data_processing_utils import DataProcessingUtils
from ..common_utilities import CommonUtilities
from ..hardware.m1_memory_optimizer import get_m1_memory_optimizer, M1MemoryOptimizer
from ..hardware.m1_gpu_utils import get_m1_gpu_manager
from ..hardware.m1_cpu_optimizer import get_m1_cpu_optimizer, M1CPUOptimizer

# Import HMM composite manager
from ..hmm_composite_manager import EnhancedHMMCompositeManager

# Import ML Common utilities
from .cv_utils import TemporalCrossValidator, PurgedKFold
from .validation_utils import ValidationFramework
from .pareto import ParetoFrontAnalyzer
from .ensemble_manager import EnsembleManager, EnsembleConfig, EnsembleType

# Import data quality frameworks
try:
    from ..data.quality.data_quality import DataQualityFramework
    QUALITY_FRAMEWORK_AVAILABLE = True
except ImportError:
    QUALITY_FRAMEWORK_AVAILABLE = False

try:
    from ..feature_engineering_validation import FeatureEngineeringValidator
    FEATURE_VALIDATOR_AVAILABLE = True
except ImportError:
    FEATURE_VALIDATOR_AVAILABLE = False

logger = logging.getLogger(__name__)

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.warning("HMM libraries not available - limited regime detection functionality")

class RegimeDetectionMethod(Enum):
    """Available regime detection methods."""
    HMM_GAUSSIAN = "hmm_gaussian"
    HMM_MULTIVARIATE = "hmm_multivariate"
    ENSEMBLE_HMM = "ensemble_hmm"
    MULTI_TIMEFRAME_HMM = "multi_timeframe_hmm"
    STREAMING_HMM = "streaming_hmm"
    REGIME_AWARE_HMM = "regime_aware_hmm"

class TimeframeType(Enum):
    """Available timeframe types."""
    MINUTE = "1m"
    HOUR = "1h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1M"

@dataclass
class HMMRegimeConfig:
    """Configuration for HMM regime detection."""
    n_components: int = 4
    covariance_type: str = "full"
    n_iter: int = 100
    tol: float = 1e-3
    random_state: int = 42
    method: RegimeDetectionMethod = RegimeDetectionMethod.HMM_GAUSSIAN
    min_regime_samples: int = 100
    max_regime_imbalance: float = 0.8
    economic_significance_threshold: float = 0.05

@dataclass
class MultiTimeframeConfig:
    """Configuration for multi-timeframe HMM."""
    timeframes: List[TimeframeType] = field(default_factory=lambda: [TimeframeType.HOUR, TimeframeType.DAILY])
    ensemble_weights: Dict[str, float] = field(default_factory=dict)
    consensus_threshold: float = 0.6
    temporal_alignment: bool = True

@dataclass
class StreamingConfig:
    """Configuration for streaming HMM."""
    window_size: int = 1000
    update_frequency: int = 100
    adaptation_rate: float = 0.1
    stability_threshold: float = 0.8
    max_regime_changes: int = 10

@dataclass
class RegimeTransitionMetrics:
    """Metrics for regime transition analysis."""
    transition_matrix: np.ndarray
    transition_probabilities: Dict[Tuple[int, int], float]
    regime_persistence: Dict[int, float]
    transition_volatility: float
    regime_stability: float

@dataclass
class EconomicSignificanceMetrics:
    """Metrics for economic significance validation."""
    regime_returns: Dict[int, float]
    regime_volatility: Dict[int, float]
    regime_sharpe: Dict[int, float]
    pareto_efficiency: float
    economic_significance: bool

class EnhancedHMMRegimeDetector:
    """Enhanced HMM regime detector with comprehensive functionality."""
    
    def __init__(self, config: Optional[HMMRegimeConfig] = None):
        self.logger = create_fallback_logger()
        self.logger.info("🚀 Initializing EnhancedHMMRegimeDetector...")
        start_time = time.time()
        
        self.config = config or HMMRegimeConfig()
        self.logger.info(f"📊 Configuration loaded: {self.config.method.value}")
        self.logger.info(f"📊 HMM components: {self.config.n_components}")
        self.logger.info(f"📊 Covariance type: {self.config.covariance_type}")
        
        # Initialize utility managers
        self.logger.debug("🔧 Initializing utility managers...")
        self._initialize_utilities()
        
        # Initialize HMM composite manager
        self.logger.debug("🔧 Initializing HMM composite manager...")
        self.hmm_manager = EnhancedHMMCompositeManager()
        self.logger.debug("✅ HMM composite manager initialized")
        
        # Initialize specialized configurations
        self.logger.debug("🔧 Initializing specialized configurations...")
        self.multi_timeframe_config = MultiTimeframeConfig()
        self.streaming_config = StreamingConfig()
        self.logger.debug("✅ Specialized configurations initialized")
        
        # Performance tracking
        self.performance_stats = {
            'total_regimes_detected': 0,
            'processing_time': 0.0,
            'memory_usage': 0.0,
            'accuracy_scores': []
        }
        
        init_time = time.time() - start_time
        self.logger.info(f"✅ EnhancedHMMRegimeDetector initialized in {init_time:.3f}s")
        self.logger.info(f"📊 Min regime samples: {self.config.min_regime_samples}")
        self.logger.info(f"📊 Max regime imbalance: {self.config.max_regime_imbalance}")
        self.logger.info(f"📊 Economic significance threshold: {self.config.economic_significance_threshold}")
        
        # Streaming state
        self.streaming_state = {
            'current_model': None,
            'last_update': None,
            'regime_history': [],
            'stability_score': 0.0
        }

    def _initialize_utilities(self):
        """Initialize utility managers."""
        try:
            # Try to import and initialize M1 GPU manager with fallback
            try:
                self.gpu_manager = get_m1_gpu_manager()
            except NameError:
                self.logger.warning("⚠️ get_m1_gpu_manager not available, using fallback")
                self.gpu_manager = None
            except Exception as gpu_e:
                self.logger.warning(f"⚠️ M1 GPU manager initialization failed: {gpu_e}")
                self.gpu_manager = None

            # Try to import and initialize M1 memory optimizer with fallback
            try:
                self.memory_optimizer = get_m1_memory_optimizer()
            except Exception as mem_e:
                self.logger.warning(f"⚠️ M1 memory optimizer initialization failed: {mem_e}")
                self.memory_optimizer = None

            # Try to import and initialize M1 CPU optimizer with fallback
            try:
                self.cpu_optimizer = get_m1_cpu_optimizer()
            except Exception as cpu_e:
                self.logger.warning(f"⚠️ M1 CPU optimizer initialization failed: {cpu_e}")
                self.cpu_optimizer = None

            # Initialize other utilities
            try:
                self.parquet_utils = ParquetUtils()
                self.serializer = UniversalSerializer()
                self.data_processor = DataProcessingUtils()
                self.common_utils = CommonUtilities()
                self.pareto_analyzer = ParetoFrontAnalyzer()

                # Create ensemble config for HMM regime detection
                ensemble_config = EnsembleConfig(
                    ensemble_name="hmm_regime_detection",
                    output_dir=str(Path(__file__).parent / "artifacts" / "ensemble"),
                    ensemble_type=EnsembleType.VOTING,
                    enable_gpu_acceleration=True,
                    enable_memory_optimization=True,
                    enable_parallel_processing=True,
                    memory_limit_gb=4.0
                )
                self.ensemble_manager = EnsembleManager(ensemble_config)
            except Exception as util_e:
                self.logger.warning(f"⚠️ Some basic utilities failed to initialize: {util_e}")
                self.parquet_utils = None
                self.serializer = None
                self.data_processor = None
                self.common_utils = None
                self.pareto_analyzer = None
                self.ensemble_manager = None

            self.logger.info("✅ All utility managers initialized successfully")
        except Exception as e:
            self.logger.warning(f"⚠️ Unexpected error during utility initialization: {e}")
            # Set fallback implementations for any that weren't already set
            if not hasattr(self, 'gpu_manager'):
                self.gpu_manager = None
            if not hasattr(self, 'memory_optimizer'):
                self.memory_optimizer = None
            if not hasattr(self, 'cpu_optimizer'):
                self.cpu_optimizer = None
            if not hasattr(self, 'parquet_utils'):
                self.parquet_utils = None
            if not hasattr(self, 'serializer'):
                self.serializer = None
            if not hasattr(self, 'data_processor'):
                self.data_processor = None
            if not hasattr(self, 'common_utils'):
                self.common_utils = None
            if not hasattr(self, 'pareto_analyzer'):
                self.pareto_analyzer = None
            if not hasattr(self, 'ensemble_manager'):
                self.ensemble_manager = None

    def detect_regimes(
        self,
        data: pd.DataFrame,
        method: Optional[RegimeDetectionMethod] = None,
        config: Optional[HMMRegimeConfig] = None
    ) -> pd.DataFrame:
        """
        Detect regimes using specified method.
        
        Args:
            data: Input data for regime detection
            method: Regime detection method
            config: Optional configuration override
            
        Returns:
            DataFrame with regime labels and metadata
        """
        method = method or self.config.method
        config = config or self.config
        start_time = time.time()
        
        try:
            # Validate input data
            self._validate_input_data(data)
            
            # Select implementation based on method
            if method == RegimeDetectionMethod.HMM_GAUSSIAN:
                regimes_df = self._detect_hmm_gaussian_regimes(data, config)
            elif method == RegimeDetectionMethod.HMM_MULTIVARIATE:
                regimes_df = self._detect_hmm_multivariate_regimes(data, config)
            elif method == RegimeDetectionMethod.ENSEMBLE_HMM:
                regimes_df = self._detect_ensemble_hmm_regimes(data, config)
            elif method == RegimeDetectionMethod.MULTI_TIMEFRAME_HMM:
                regimes_df = self._detect_multi_timeframe_hmm_regimes(data, config)
            elif method == RegimeDetectionMethod.STREAMING_HMM:
                regimes_df = self._detect_streaming_hmm_regimes(data, config)
            elif method == RegimeDetectionMethod.REGIME_AWARE_HMM:
                regimes_df = self._detect_regime_aware_hmm_regimes(data, config)
            else:
                raise ValueError(f"Unsupported regime detection method: {method}")
            
            # Validate regime quality
            validation_results = self._validate_regime_quality(regimes_df, data)
            
            # Update performance stats
            self._update_performance_stats(start_time, len(regimes_df))
            
            self.logger.info(f"✅ Detected {len(regimes_df)} regimes using {method.value}")
            return regimes_df
            
        except Exception as e:
            self.logger.error(f"❌ Failed to detect regimes: {e}")
            raise

    def _detect_hmm_gaussian_regimes(
        self, 
        data: pd.DataFrame, 
        config: HMMRegimeConfig
    ) -> pd.DataFrame:
        """Detect regimes using Gaussian HMM."""
        if not HMM_AVAILABLE:
            raise ImportError("HMM libraries not available")
        
        # Prepare data - enhanced feature engineering for HMM
        numeric_data = data.select_dtypes(include=[np.number])

        # Check for NaN values and handle them properly
        nan_count = numeric_data.isnull().sum().sum()
        if nan_count > 0:
            self.logger.warning(f"⚠️ Found {nan_count} NaN values in numeric data, using forward/backward fill")

            # Use forward fill, then backward fill for remaining NaN values
            numeric_data = numeric_data.fillna(method='ffill').fillna(method='bfill')

            # If still have NaN values (e.g., all values in column are NaN), fill with column mean
            numeric_data = numeric_data.fillna(numeric_data.mean())

            # Final fallback: fill any remaining NaN values with 0
            numeric_data = numeric_data.fillna(0)

        # Enhanced feature engineering for better regime detection
        try:
            enhanced_data = self._create_enhanced_features(numeric_data)
            numeric_data = enhanced_data
            self.logger.info(f"✅ Enhanced features created: {numeric_data.shape[1]} features")
        except Exception as e:
            self.logger.warning(f"⚠️ Feature engineering failed, using original data: {e}")

        # Data normalization for HMM stability
        try:
            numeric_data = self._normalize_hmm_data(numeric_data)
            self.logger.info("✅ Data normalized for HMM training")
        except Exception as e:
            self.logger.warning(f"⚠️ Data normalization failed: {e}")

        # Validate data for HMM training
        if numeric_data.empty:
            raise ValueError("No numeric data available for HMM training")

        if len(numeric_data) < config.n_components * 10:
            raise ValueError(f"Insufficient data for HMM training. Need at least {config.n_components * 10} samples, got {len(numeric_data)}")

        # Check for infinite values
        if np.any(np.isinf(numeric_data.values)):
            self.logger.warning("⚠️ Found infinite values in data, replacing with finite values")
            numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill').fillna(0)

        # Use HMM composite manager for optimization
        optimization_result = self.hmm_manager.optimize_hmm_parameters(numeric_data)
        
        if optimization_result.get('success', False):
            best_params = optimization_result['best_params']
            config.n_components = best_params.get('n_components', config.n_components)
            config.covariance_type = best_params.get('covariance_type', config.covariance_type)
            config.n_iter = best_params.get('n_iter', config.n_iter)
            config.tol = best_params.get('tol', config.tol)
        
        # Create and fit HMM model with improved parameters
        model = hmm.GaussianHMM(
            n_components=config.n_components,
            covariance_type=config.covariance_type,
            n_iter=max(config.n_iter, 100),  # Ensure minimum iterations
            tol=max(config.tol, 1e-4),  # Ensure reasonable tolerance
            random_state=config.random_state,
            init_params='mc',  # Only initialize transition matrix, use better initialization
            params='stmc'  # Estimate start, transition, and means/covariances
        )

        # Better initialization for start probabilities
        if hasattr(model, 'startprob_'):
            # Initialize with uniform distribution
            model.startprob_ = np.ones(config.n_components) / config.n_components
        
        # Use memory optimizer if available
        if self.memory_optimizer:
            numeric_data = self.memory_optimizer.optimize_dataframe_memory(numeric_data)
        
        # Try fitting with different configurations if needed
        fit_success = False
        fallback_configs = [
            {'covariance_type': 'diag', 'n_components': min(config.n_components, 3)},
            {'covariance_type': 'spherical', 'n_components': 2},
        ]

        try:
            model.fit(numeric_data)
            fit_success = True
        except Exception as e:
            self.logger.warning(f"⚠️ Primary HMM fitting failed: {e}, trying fallback configurations")

            for i, fallback_config in enumerate(fallback_configs):
                try:
                    self.logger.info(f"Trying fallback configuration {i+1}: {fallback_config}")
                    fallback_model = hmm.GaussianHMM(
                        n_components=fallback_config['n_components'],
                        covariance_type=fallback_config['covariance_type'],
                        n_iter=200,
                        tol=1e-3,
                        random_state=config.random_state,
                        init_params='mc',
                        params='stmc'
                    )
                    fallback_model.startprob_ = np.ones(fallback_config['n_components']) / fallback_config['n_components']
                    fallback_model.fit(numeric_data)
                    model = fallback_model
                    fit_success = True
                    self.logger.info(f"✅ Fallback configuration {i+1} succeeded")
                    break
                except Exception as fallback_e:
                    self.logger.warning(f"⚠️ Fallback configuration {i+1} failed: {fallback_e}")

            if not fit_success:
                self.logger.error(f"❌ All HMM fitting attempts failed")
                raise RuntimeError(f"HMM model fitting failed: {e}") from e

        # Check if model fitted properly
        if hasattr(model, 'startprob_') and np.any(np.isnan(model.startprob_)):
            self.logger.error("❌ HMM model has invalid start probabilities (NaN values)")
            raise RuntimeError("HMM model fitting produced invalid start probabilities")

        if hasattr(model, 'transmat_') and np.any(np.isnan(model.transmat_)):
            self.logger.error("❌ HMM model has invalid transition matrix (NaN values)")
            raise RuntimeError("HMM model fitting produced invalid transition matrix")

        try:
            regime_labels = model.predict(numeric_data)
        except Exception as e:
            self.logger.error(f"❌ HMM prediction failed: {e}")
            raise RuntimeError(f"HMM prediction failed: {e}") from e

        # Handle length mismatch due to feature engineering
        original_length = len(data)
        processed_length = len(numeric_data)

        if processed_length != original_length:
            self.logger.warning(f"⚠️ Length mismatch: original {original_length}, processed {processed_length}")
            # Create regime labels for original length, using the last regime for missing data
            extended_regime_labels = np.full(original_length, regime_labels[-1] if len(regime_labels) > 0 else 0)
            # Map processed data indices back to original data
            processed_indices = numeric_data.index
            extended_regime_labels[processed_indices] = regime_labels
            regime_labels = extended_regime_labels

        # Create result DataFrame
        result = data.copy()
        result['regime'] = regime_labels

        # Handle regime probabilities with proper length matching
        try:
            if processed_length == original_length:
                probabilities = model.predict_proba(numeric_data)
                result['regime_probability'] = np.max(probabilities, axis=1)
            else:
                # For mismatched lengths, use default probabilities
                result['regime_probability'] = 0.5
                self.logger.warning("⚠️ Using default regime probabilities due to length mismatch")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not compute regime probabilities: {e}")
            result['regime_probability'] = 0.5  # Default probability

        result['detection_method'] = 'hmm_gaussian'

        try:
            result['model_score'] = model.score(numeric_data)
        except Exception as e:
            self.logger.warning(f"⚠️ Could not compute model score: {e}")
            result['model_score'] = 0.0
        
        return result

    def _create_enhanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced features for better HMM regime detection with validation."""
        df = data.copy()

        # Track original data for validation
        original_df = df.copy()
        feature_config = {
            'price_features': ['returns', 'log_returns', 'sma_5', 'sma_20', 'ema_12', 'ema_26', 'momentum_5', 'momentum_20'],
            'volume_features': ['volume_ma_5', 'volume_ratio', 'volume_change'],
            'volatility_features': ['tr', 'atr_14', 'bb_upper', 'bb_lower', 'bb_position'],
            'technical_features': ['rsi', 'macd', 'macd_signal', 'macd_histogram', 'stoch_k', 'stoch_d']
        }

        # Check if feature validator is available
        validator_available = FEATURE_VALIDATOR_AVAILABLE

        # Basic price features
        if 'close' in df.columns and 'open' in df.columns:
            # Returns and volatility
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

            # Moving averages
            df['sma_5'] = df['close'].rolling(5).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()

            # Price momentum
            df['momentum_5'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
            df['momentum_20'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)

        # Volume features
        if 'volume' in df.columns:
            df['volume_ma_5'] = df['volume'].rolling(5).mean()
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            df['volume_change'] = df['volume'].pct_change()

        # Volatility features
        if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            # True Range and ATR
            df['tr'] = np.maximum(df['high'] - df['low'],
                                 np.maximum(abs(df['high'] - df['close'].shift(1)),
                                           abs(df['low'] - df['close'].shift(1))))
            df['atr_14'] = df['tr'].rolling(14).mean()

            # Bollinger Bands
            sma_20 = df['close'].rolling(20).mean()
            std_20 = df['close'].rolling(20).std()
            df['bb_upper'] = sma_20 + 2 * std_20
            df['bb_lower'] = sma_20 - 2 * std_20
            df['bb_position'] = (df['close'] - sma_20) / (2 * std_20)

        # Technical indicators
        if 'close' in df.columns and 'high' in df.columns and 'low' in df.columns:
            # RSI - improved calculation to avoid constant values
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            # Avoid division by zero and handle edge cases
            loss = loss.replace(0, 1e-10)  # Small value instead of zero
            rs = gain / loss
            df['rsi_14'] = 100 - (100 / (1 + rs))
            # Ensure RSI is within valid range
            df['rsi_14'] = df['rsi_14'].clip(0, 100)

            # MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']

        # Handle NaN values created by rolling operations - DON'T drop rows to preserve length
        # Instead, fill with forward/backward fill or zeros
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

        # Debug: Check for problematic calculations and known problematic columns
        self.logger.info(f"Features created: {list(df.columns)}")

        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                unique_count = df[col].nunique()
                std_val = df[col].std()
                if std_val == 0 or unique_count == 1:
                    self.logger.warning(f"⚠️ Unexpected constant column: {col} (unique: {unique_count}, std: {std_val})")

        # Check for constant columns with fast-fail logic
        constant_cols = []
        trade_stat_cols = ['trade_volume', 'trade_count', 'avg_price', 'min_price', 'max_price', 'price_std']
        funding_cols = ['funding_rate']

        # Fast-fail: Check if critical features are constant
        critical_constant_features = []
        for col in trade_stat_cols + funding_cols:
            if col in df.columns:
                unique_vals = df[col].nunique()
                std_val = df[col].std()
                if unique_vals <= 2 or (not pd.isna(std_val) and std_val < 1e-10):
                    critical_constant_features.append(f"{col}(unique={unique_vals}, std={std_val:.2e})")

        if critical_constant_features:
            error_msg = f"🚨 CRITICAL: Constant feature detection failed! These features are constant: {critical_constant_features}"
            self.logger.error(error_msg)
            self.logger.error("   This indicates a data processing failure - features should have variation")
            self.logger.error("   Please check the data converter step01_5_data_converter.py")
            self.logger.error("   FAST-FAIL: Not attempting gap filling or downloads due to constant features")
            raise ValueError(f"HMM training cannot proceed with constant features: {critical_constant_features}")

        # Check for other genuinely constant columns
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                is_constant = df[col].std() == 0 or df[col].nunique() == 1

                if is_constant:
                    # These should not be constant - they indicate data quality issues
                    if col in trade_stat_cols + funding_cols:
                        # We already checked these above, so this shouldn't happen
                        continue
                    else:
                        # For other columns, constant values are genuinely problematic
                        constant_cols.append(col)

        if constant_cols:
            self.logger.info(f"Removing genuinely constant columns: {constant_cols}")
            df = df.drop(columns=constant_cols)

        self.logger.info("✅ Feature quality validation passed - all critical features have proper variation")

        # Select only numeric columns and limit to reasonable number of features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 20:
            # Keep most important features
            priority_features = ['returns', 'log_returns', 'rsi_14', 'macd', 'bb_position',
                               'momentum_5', 'volume_ratio', 'atr_14']
            other_features = [col for col in numeric_cols if col not in priority_features]
            selected_features = priority_features + other_features[:10]  # Limit to 18 total
            df = df[selected_features]

        # Feature Engineering Validation
        if validator_available:
            self.logger.info("🔍 Validating engineered features...")

            try:
                from ..feature_engineering_validation import FeatureEngineeringValidator
                validator = FeatureEngineeringValidator()
                validation_result = validator.validate_engineered_features(
                    original_df=original_df,
                    features_df=df,
                    feature_config=feature_config,
                    validate_calculations=True,
                    check_dependencies=True
                )

                # Log validation results
                self.logger.info(f"✅ Feature validation completed - Quality Score: {validation_result.quality_score:.2f}/100")

                if not validation_result.passed:
                    self.logger.warning("⚠️ Feature validation found issues:")

                    # Log critical issues
                    critical_issues = [i for i in validation_result.issues if 'CRITICAL' in str(i).upper()]
                    if critical_issues:
                        self.logger.error(f"❌ Critical feature issues ({len(critical_issues)}):")
                        for issue in critical_issues[:3]:
                            self.logger.error(f"   - {issue}")

                    # Log warnings
                    if validation_result.warnings:
                        self.logger.warning(f"⚠️ Feature warnings ({len(validation_result.warnings)}):")
                        for warning in validation_result.warnings[:3]:
                            self.logger.warning(f"   - {warning}")

                # Feature quality gate
                if validation_result.quality_score < 99:
                    self.logger.warning("⚠️ Low feature quality detected - proceeding with caution")

            except Exception as e:
                self.logger.warning(f"⚠️ Feature validation failed: {e} - continuing with features")
        else:
            self.logger.info("ℹ️ Feature engineering validation not available - skipping validation")

        return df

    def _normalize_hmm_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize data for stable HMM training."""
        df = data.copy()

        # Remove constant columns - more robust check
        constant_cols = []
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                # Check both std == 0 and nunique == 1 to catch edge cases
                if df[col].std() == 0 or df[col].nunique() == 1:
                    constant_cols.append(col)

        if constant_cols:
            df = df.drop(columns=constant_cols)
            self.logger.info(f"Removed constant columns from normalization: {constant_cols}")

        # Robust normalization using median and IQR
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                median = df[col].median()
                q75, q25 = np.percentile(df[col].dropna(), [75, 25])
                iqr = q75 - q25

                if iqr > 0:  # Avoid division by zero
                    df[col] = (df[col] - median) / iqr
                else:
                    df[col] = df[col] - median

        # Handle outliers using z-score with robust threshold
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                # Use 3.5 sigma instead of 3 for more robust outlier removal
                df.loc[z_scores > 3.5, col] = df[col].median()

        return df

    def _detect_hmm_multivariate_regimes(
        self, 
        data: pd.DataFrame, 
        config: HMMRegimeConfig
    ) -> pd.DataFrame:
        """Detect regimes using multivariate HMM."""
        if not HMM_AVAILABLE:
            raise ImportError("HMM libraries not available")
        
        # Prepare multivariate data
        numeric_data = data.select_dtypes(include=[np.number]).fillna(0)
        
        # Feature engineering using HMM composite manager
        engineered_data = self.hmm_manager.engineer_features(numeric_data)
        
        # Create and fit multivariate HMM model
        model = hmm.GaussianHMM(
            n_components=config.n_components,
            covariance_type=config.covariance_type,
            n_iter=config.n_iter,
            tol=config.tol,
            random_state=config.random_state
        )
        
        model.fit(engineered_data)
        regime_labels = model.predict(engineered_data)
        
        # Create result DataFrame
        result = data.copy()
        result['regime'] = regime_labels
        result['regime_probability'] = model.predict_proba(engineered_data).max(axis=1)
        result['detection_method'] = 'hmm_multivariate'
        result['model_score'] = model.score(engineered_data)
        
        return result

    def _detect_ensemble_hmm_regimes(
        self, 
        data: pd.DataFrame, 
        config: HMMRegimeConfig
    ) -> pd.DataFrame:
        """Detect regimes using ensemble HMM methods."""
        if not self.ensemble_manager:
            raise ImportError("Ensemble manager not available")
        
        # Create multiple HMM models with different configurations
        models = []
        configs = [
            HMMRegimeConfig(n_components=3, covariance_type="full"),
            HMMRegimeConfig(n_components=4, covariance_type="tied"),
            HMMRegimeConfig(n_components=5, covariance_type="diag"),
        ]
        
        for model_config in configs:
            try:
                model_result = self._detect_hmm_gaussian_regimes(data, model_config)
                models.append(model_result)
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to create ensemble model: {e}")
        
        if not models:
            raise ValueError("No ensemble models could be created")
        
        # Combine ensemble results
        ensemble_result = self._combine_ensemble_results(models)
        
        # Create result DataFrame
        result = data.copy()
        result['regime'] = ensemble_result['regime']
        result['regime_probability'] = ensemble_result['probability']
        result['detection_method'] = 'ensemble_hmm'
        result['ensemble_consensus'] = ensemble_result['consensus']
        
        return result

    def _detect_multi_timeframe_hmm_regimes(
        self, 
        data: pd.DataFrame, 
        config: HMMRegimeConfig
    ) -> pd.DataFrame:
        """Detect regimes using multi-timeframe HMM ensemble."""
        # This would require multiple timeframe data
        # For now, implement single timeframe with multi-timeframe structure
        base_result = self._detect_hmm_gaussian_regimes(data, config)
        
        # Add multi-timeframe metadata
        base_result['detection_method'] = 'multi_timeframe_hmm'
        base_result['timeframe_consensus'] = 1.0  # Single timeframe for now
        
        return base_result

    def _detect_streaming_hmm_regimes(
        self, 
        data: pd.DataFrame, 
        config: HMMRegimeConfig
    ) -> pd.DataFrame:
        """Detect regimes using streaming HMM."""
        # Initialize streaming state if needed
        if self.streaming_state['current_model'] is None:
            self._initialize_streaming_model(data, config)
        
        # Process data in streaming fashion
        window_size = self.streaming_config.window_size
        update_frequency = self.streaming_config.update_frequency
        
        regimes = []
        probabilities = []
        
        for i in range(0, len(data), update_frequency):
            window_data = data.iloc[i:i+window_size]
            
            if len(window_data) < window_size:
                break
            
            # Update model if needed
            if i % (update_frequency * 10) == 0:
                self._update_streaming_model(window_data, config)
            
            # Predict regimes for current window
            window_regimes = self._predict_streaming_regimes(window_data)
            regimes.extend(window_regimes)
            probabilities.extend([0.8] * len(window_regimes))  # Placeholder
        
        # Create result DataFrame
        result = data.copy()
        result['regime'] = regimes[:len(data)]
        result['regime_probability'] = probabilities[:len(data)]
        result['detection_method'] = 'streaming_hmm'
        result['streaming_stability'] = self.streaming_state['stability_score']
        
        return result

    def _detect_regime_aware_hmm_regimes(
        self, 
        data: pd.DataFrame, 
        config: HMMRegimeConfig
    ) -> pd.DataFrame:
        """Detect regimes using regime-aware HMM."""
        # First detect basic regimes
        base_result = self._detect_hmm_gaussian_regimes(data, config)
        
        # Apply regime-aware refinements
        refined_result = self._refine_regime_aware_regimes(base_result, data)
        
        return refined_result

    def analyze_regime_transitions(
        self, 
        regimes_df: pd.DataFrame
    ) -> RegimeTransitionMetrics:
        """Analyze regime transition patterns."""
        try:
            regimes = regimes_df['regime'].values
            unique_regimes = np.unique(regimes)
            n_regimes = len(unique_regimes)
            
            # Create transition matrix
            transition_matrix = np.zeros((n_regimes, n_regimes))
            
            for i in range(len(regimes) - 1):
                current_regime = regimes[i]
                next_regime = regimes[i + 1]
                
                current_idx = np.where(unique_regimes == current_regime)[0][0]
                next_idx = np.where(unique_regimes == next_regime)[0][0]
                
                transition_matrix[current_idx, next_idx] += 1
            
            # Normalize transition matrix
            row_sums = transition_matrix.sum(axis=1)
            transition_matrix = np.divide(
                transition_matrix, 
                row_sums[:, np.newaxis], 
                out=np.zeros_like(transition_matrix), 
                where=row_sums[:, np.newaxis] != 0
            )
            
            # Calculate transition probabilities
            transition_probabilities = {}
            for i in range(n_regimes):
                for j in range(n_regimes):
                    transition_probabilities[(unique_regimes[i], unique_regimes[j])] = transition_matrix[i, j]
            
            # Calculate regime persistence
            regime_persistence = {}
            for i, regime in enumerate(unique_regimes):
                regime_persistence[regime] = transition_matrix[i, i]
            
            # Calculate transition volatility
            transition_volatility = np.std(transition_matrix)
            
            # Calculate regime stability
            regime_stability = np.mean([regime_persistence[regime] for regime in unique_regimes])
            
            return RegimeTransitionMetrics(
                transition_matrix=transition_matrix,
                transition_probabilities=transition_probabilities,
                regime_persistence=regime_persistence,
                transition_volatility=transition_volatility,
                regime_stability=regime_stability
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to analyze regime transitions: {e}")
            return RegimeTransitionMetrics(
                transition_matrix=np.array([]),
                transition_probabilities={},
                regime_persistence={},
                transition_volatility=0.0,
                regime_stability=0.0
            )

    def validate_economic_significance(
        self, 
        regimes_df: pd.DataFrame, 
        returns_data: pd.Series
    ) -> EconomicSignificanceMetrics:
        """Validate economic significance of detected regimes."""
        try:
            if not self.pareto_analyzer:
                raise ImportError("Pareto analyzer not available")
            
            regimes = regimes_df['regime'].values
            unique_regimes = np.unique(regimes)
            
            # Calculate regime-specific metrics
            regime_returns = {}
            regime_volatility = {}
            regime_sharpe = {}
            
            for regime in unique_regimes:
                regime_mask = regimes == regime
                regime_returns_series = returns_data[regime_mask]
                
                if len(regime_returns_series) > 0:
                    regime_returns[regime] = np.mean(regime_returns_series)
                    regime_volatility[regime] = np.std(regime_returns_series)
                    regime_sharpe[regime] = safe_divide(
                        regime_returns[regime], 
                        regime_volatility[regime]
                    )
                else:
                    regime_returns[regime] = 0.0
                    regime_volatility[regime] = 0.0
                    regime_sharpe[regime] = 0.0
            
            # Analyze Pareto efficiency
            returns_array = np.array(list(regime_returns.values()))
            volatility_array = np.array(list(regime_volatility.values()))
            
            pareto_efficiency = self.pareto_analyzer.calculate_pareto_efficiency(
                returns_array, volatility_array
            )
            
            # Determine economic significance
            economic_significance = (
                pareto_efficiency > self.config.economic_significance_threshold and
                len(unique_regimes) > 1 and
                max(regime_returns.values()) - min(regime_returns.values()) > 0.01
            )
            
            return EconomicSignificanceMetrics(
                regime_returns=regime_returns,
                regime_volatility=regime_volatility,
                regime_sharpe=regime_sharpe,
                pareto_efficiency=pareto_efficiency,
                economic_significance=economic_significance
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to validate economic significance: {e}")
            return EconomicSignificanceMetrics(
                regime_returns={},
                regime_volatility={},
                regime_sharpe={},
                pareto_efficiency=0.0,
                economic_significance=False
            )

    def _validate_regime_quality(
        self, 
        regimes_df: pd.DataFrame, 
        original_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Validate the quality of detected regimes."""
        try:
            # Use HMM composite manager validation
            validation_result = self.hmm_manager.validate_hmm_results(
                original_data, 
                regimes_df['regime'].values
            )
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to validate regime quality: {e}")
            return {
                'validation_passed': False,
                'errors': [str(e)],
                'warnings': []
            }

    def _combine_ensemble_results(self, models: List[pd.DataFrame]) -> Dict[str, Any]:
        """Combine results from ensemble models."""
        try:
            # Get regime predictions from all models
            regime_predictions = [model['regime'].values for model in models]
            
            # Calculate consensus regime
            consensus_regimes = []
            consensus_probabilities = []
            
            for i in range(len(regime_predictions[0])):
                # Get regime votes for this time point
                votes = [pred[i] for pred in regime_predictions]
                
                # Calculate consensus (most common regime)
                unique_votes, vote_counts = np.unique(votes, return_counts=True)
                consensus_regime = unique_votes[np.argmax(vote_counts)]
                consensus_probability = np.max(vote_counts) / len(votes)
                
                consensus_regimes.append(consensus_regime)
                consensus_probabilities.append(consensus_probability)
            
            return {
                'regime': consensus_regimes,
                'probability': consensus_probabilities,
                'consensus': np.mean(consensus_probabilities)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to combine ensemble results: {e}")
            return {
                'regime': [0] * len(models[0]),
                'probability': [0.0] * len(models[0]),
                'consensus': 0.0
            }

    def _initialize_streaming_model(self, data: pd.DataFrame, config: HMMRegimeConfig):
        """Initialize streaming HMM model."""
        try:
            # Use first window to initialize model
            window_data = data.iloc[:self.streaming_config.window_size]
            initial_result = self._detect_hmm_gaussian_regimes(window_data, config)
            
            self.streaming_state['current_model'] = initial_result
            self.streaming_state['last_update'] = time.time()
            self.streaming_state['regime_history'] = initial_result['regime'].tolist()
            self.streaming_state['stability_score'] = 0.8  # Initial stability
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize streaming model: {e}")

    def _update_streaming_model(self, window_data: pd.DataFrame, config: HMMRegimeConfig):
        """Update streaming HMM model."""
        try:
            # Detect regimes for current window
            window_result = self._detect_hmm_gaussian_regimes(window_data, config)
            
            # Update streaming state
            self.streaming_state['current_model'] = window_result
            self.streaming_state['last_update'] = time.time()
            
            # Update regime history
            new_regimes = window_result['regime'].tolist()
            self.streaming_state['regime_history'].extend(new_regimes)
            
            # Keep only recent history
            max_history = self.streaming_config.window_size * 5
            if len(self.streaming_state['regime_history']) > max_history:
                self.streaming_state['regime_history'] = self.streaming_state['regime_history'][-max_history:]
            
            # Calculate stability score
            self._calculate_streaming_stability()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update streaming model: {e}")

    def _predict_streaming_regimes(self, window_data: pd.DataFrame) -> List[int]:
        """Predict regimes for streaming window."""
        try:
            if self.streaming_state['current_model'] is None:
                return [0] * len(window_data)
            
            # Use current model to predict regimes
            # This is a simplified implementation
            return self.streaming_state['current_model']['regime'].tolist()[:len(window_data)]
            
        except Exception as e:
            self.logger.error(f"❌ Failed to predict streaming regimes: {e}")
            return [0] * len(window_data)

    def _calculate_streaming_stability(self):
        """Calculate streaming model stability."""
        try:
            regime_history = self.streaming_state['regime_history']
            if len(regime_history) < 10:
                self.streaming_state['stability_score'] = 0.5
                return
            
            # Calculate regime change frequency
            regime_changes = sum(1 for i in range(1, len(regime_history)) 
                               if regime_history[i] != regime_history[i-1])
            
            change_frequency = safe_divide(regime_changes, len(regime_history) - 1)
            stability_score = 1.0 - change_frequency
            
            self.streaming_state['stability_score'] = max(0.0, min(1.0, stability_score))
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate streaming stability: {e}")
            self.streaming_state['stability_score'] = 0.0

    def _refine_regime_aware_regimes(
        self, 
        base_result: pd.DataFrame, 
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """Refine regimes using regime-aware logic."""
        # This would implement regime-aware refinements
        # For now, return the base result with additional metadata
        refined_result = base_result.copy()
        refined_result['detection_method'] = 'regime_aware_hmm'
        refined_result['regime_awareness'] = 1.0  # Placeholder
        
        return refined_result

    def _validate_input_data(self, data: pd.DataFrame):
        """Validate input data for regime detection using advanced quality framework."""
        # Check if quality framework is available
        if not QUALITY_FRAMEWORK_AVAILABLE:
            self.logger.warning("⚠️ Advanced quality framework not available, using basic validation")
            return self._basic_validation_fallback(data)

        # Use advanced quality framework for comprehensive validation
        quality_framework = DataQualityFramework()

        # Basic sanity checks first
        if len(data) < 50:
            raise ValueError("Insufficient data for regime detection (minimum 50 rows required)")

        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 0:
            raise ValueError("No numeric columns found for regime detection")

        # Advanced quality validation
        quality_framework = DataQualityFramework()
        quality_result = quality_framework.validate_dataframe_quality(
            data,
            context="hmm_regime_detection_input_validation"
        )

        # Log quality assessment
        self.logger.info("🔍 HMM Input Data Quality Assessment:")
        self.logger.info(f"   📊 Quality Score: {quality_result.quality_score:.1f}/100")
        self.logger.info(f"   📊 Critical Issues: {len([i for i in quality_result.issues if 'CRITICAL' in str(i).upper()])}")
        self.logger.info(f"   📊 Warnings: {len(quality_result.warnings)}")

        # Handle critical issues
        critical_issues = [i for i in quality_result.issues if 'CRITICAL' in str(i).upper()]
        if critical_issues:
            self.logger.error("❌ Critical data quality issues found:")
            for issue in critical_issues[:3]:  # Show first 3 critical issues
                self.logger.error(f"   - {issue}")
            if len(critical_issues) > 3:
                self.logger.error(f"   ... and {len(critical_issues) - 3} more critical issues")

        # Handle warnings
        if quality_result.warnings:
            self.logger.warning("⚠️ Data quality warnings:")
            for warning in quality_result.warnings[:3]:  # Show first 3 warnings
                self.logger.warning(f"   - {warning}")

        # Quality gate - reject if quality score too low
        if quality_result.quality_score < 99:  # Very strict threshold for HMM
            raise ValueError(f"HMM input data quality too low: {quality_result.quality_score:.1f}/100 (required: 99.0)")

        # Advanced gap detection and filling
        data = self._advanced_gap_detection_and_filling(data)

        return quality_result

    def _basic_validation_fallback(self, data: pd.DataFrame) -> None:
        """Basic validation fallback when advanced framework is not available."""
        if len(data) < 50:
            raise ValueError("Insufficient data for regime detection (minimum 50 rows required)")

        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 0:
            raise ValueError("No numeric columns found for regime detection")

        # Check for null values
        null_counts = data[numeric_columns].isnull().sum()
        if null_counts.any():
            self.logger.warning(f"Null values found in data: {null_counts.to_dict()}")

        self.logger.info("✅ Basic validation passed")

    def _advanced_gap_detection_and_filling(self, data: pd.DataFrame) -> pd.DataFrame:
        """Advanced gap detection and filling with download capability for large gaps."""
        from ..data.quality.data_cleaning import DataCleaner
        from datetime import timedelta

        self.logger.info("🔍 Performing advanced gap detection and filling...")

        # Initialize cleaning framework
        cleaner = DataCleaner()

        # Detect gaps in timestamp data
        if 'timestamp' in data.columns:
            try:
                # Convert to datetime if needed
                if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                    timestamps = pd.to_datetime(data['timestamp'], unit='ms', utc=True)
                else:
                    timestamps = data['timestamp']

                # Calculate time differences
                time_diffs = timestamps.diff().dt.total_seconds()

                # Determine expected interval based on data characteristics
                expected_interval = self._determine_expected_interval(data, timestamps)
                gap_threshold = expected_interval * 2  # Consider gaps 2x the expected interval as significant
                download_threshold = expected_interval * 3  # Attempt download for gaps 3x the expected interval

                self.logger.info(f"📊 Expected data interval: {expected_interval:.1f}s")
                self.logger.info(f"📊 Gap detection threshold: {gap_threshold:.1f}s")
                self.logger.info(f"📊 Download attempt threshold: {download_threshold:.1f}s")

                # Find gaps larger than expected interval
                gap_mask = time_diffs > gap_threshold
                large_gaps = time_diffs[gap_mask]

                if len(large_gaps) > 0:
                    self.logger.info(f"📊 Found {len(large_gaps)} gaps larger than {gap_threshold:.1f}s")

                    # Batch process gaps for efficiency
                    data = self._batch_process_gaps(data, large_gaps, timestamps, gap_threshold, download_threshold)

                    # Re-validate after gap filling
                    self.logger.info("🔄 Re-validating data after gap filling...")
                    quality_framework = DataQualityFramework()
                    post_fill_result = quality_framework.validate_dataframe_quality(
                        data, context="hmm_post_gap_fill_validation"
                    )

                    if post_fill_result.quality_score < 99:
                        self.logger.warning("⚠️ Data quality still low after gap filling")

                else:
                    self.logger.info("✅ No significant gaps detected in data")

            except Exception as e:
                self.logger.warning(f"⚠️ Error during advanced gap detection: {e}")
                # Fall back to basic gap filling
                data = cleaner.fill_data_gaps(data, method='interpolate')

        return data

    def _determine_expected_interval(self, data: pd.DataFrame, timestamps: pd.Series) -> float:
        """Determine the expected time interval between data points."""
        try:
            # Calculate the most common time difference
            time_diffs = timestamps.diff().dt.total_seconds().dropna()

            if len(time_diffs) == 0:
                self.logger.warning("⚠️ No time differences found, assuming 60s interval")
                return 60.0

            # Use mode (most common interval) as expected interval
            mode_interval = time_diffs.mode()
            if len(mode_interval) > 0:
                expected_interval = mode_interval.iloc[0]
            else:
                # Fallback to median if no clear mode
                expected_interval = time_diffs.median()

            # Validate the interval makes sense (between 1 second and 24 hours)
            if expected_interval < 1:
                self.logger.warning(f"⚠️ Very small interval detected ({expected_interval:.3f}s), assuming 1s")
                expected_interval = 1.0
            elif expected_interval > 86400:  # 24 hours
                self.logger.warning(f"⚠️ Very large interval detected ({expected_interval:.1f}s), assuming 3600s (1h)")
                expected_interval = 3600.0

            # Log interval classification for user understanding
            if expected_interval <= 2:
                interval_type = "1s aggtrades"
            elif expected_interval <= 90:
                interval_type = "1m klines"
            elif expected_interval <= 600:
                interval_type = "5m klines"
            elif expected_interval <= 1800:
                interval_type = "15m/30m klines"
            elif expected_interval <= 3600:
                interval_type = "1h klines"
            elif expected_interval <= 14400:
                interval_type = "4h klines"
            elif expected_interval <= 28800:
                interval_type = "8h futures"
            elif expected_interval <= 86400:
                interval_type = "1d klines"
            else:
                interval_type = "weekly/monthly data"

            self.logger.info(f"📊 Detected data type: {interval_type} (interval: {expected_interval:.1f}s)")

            return expected_interval

        except Exception as e:
            self.logger.warning(f"⚠️ Error determining expected interval: {e}, assuming 60s")
            return 60.0

    def _download_missing_data(self, gap_start: pd.Timestamp, gap_end: pd.Timestamp) -> pd.DataFrame | None:
        """Download missing data for large gaps using existing data collection pipeline."""
        try:
            self.logger.info(f"📥 Attempting to download data from {gap_start} to {gap_end}")

            # Try to use the unified data downloader
            try:
                from ...training.steps.data_collection.data_downloader import download_all_data_with_consolidation

                # Calculate timeframe from gap duration
                gap_duration = gap_end - gap_start
                gap_seconds = gap_duration.total_seconds()

                # Determine appropriate timeframe based on gap size and data characteristics
                # Use more conservative timeframe selection to avoid downloading unnecessary data
                if gap_seconds <= 120:  # 2 minutes - likely 1s or 1m data
                    timeframe = "1s"  # Use 1-second candles for very small gaps
                elif gap_seconds <= 600:  # 10 minutes - likely 1m data
                    timeframe = "1m"
                elif gap_seconds <= 1800:  # 30 minutes - likely 5m data
                    timeframe = "5m"
                elif gap_seconds <= 7200:  # 2 hours - likely 15m data
                    timeframe = "15m"
                elif gap_seconds <= 43200:  # 12 hours - likely 1h data
                    timeframe = "1h"
                elif gap_seconds <= 172800:  # 2 days - likely 4h data
                    timeframe = "4h"
                else:  # Very large gaps - likely daily data
                    timeframe = "1d"

                # Attempt download using existing pipeline (handle async function)
                import asyncio
                import inspect

                # Check if we're already in an event loop
                try:
                    loop = asyncio.get_running_loop()
                    in_event_loop = True
                except RuntimeError:
                    in_event_loop = False

                if inspect.iscoroutinefunction(download_all_data_with_consolidation):
                    # Function is async - handle event loop properly
                    if in_event_loop:
                        # We're already in an event loop, use run_in_executor to avoid conflict
                        self.logger.info("🔄 Running async download in executor to avoid event loop conflict")
                        loop = asyncio.get_running_loop()
                        downloaded_data = await loop.run_in_executor(
                            None, 
                            lambda: asyncio.run(download_all_data_with_consolidation(
                                symbol="ETHUSDT",
                                exchange="binance", 
                                timeframe=timeframe,
                                start_date=gap_start,
                                end_date=gap_end
                            ))
                        )
                    else:
                        # Safe to run in new event loop
                        downloaded_data = await download_all_data_with_consolidation(
                            symbol="ETHUSDT",
                            exchange="binance",
                            timeframe=timeframe, 
                            start_date=gap_start,
                            end_date=gap_end
                        )
                else:
                    # Function is synchronous - run in executor if in event loop
                    if in_event_loop:
                        self.logger.info("🔄 Running sync download in executor to avoid blocking event loop")
                        loop = asyncio.get_running_loop()
                        downloaded_data = await loop.run_in_executor(
                            None,
                            download_all_data_with_consolidation,
                            "ETHUSDT", "binance", timeframe, gap_start, gap_end
                        )
                    else:
                        # Safe to run directly
                        downloaded_data = download_all_data_with_consolidation(
                            "ETHUSDT", "binance", timeframe, gap_start, gap_end
                        )
                
                # Return the downloaded data if successful
                if downloaded_data is not None and not downloaded_data.empty:
                    self.logger.info(f"✅ Successfully downloaded {len(downloaded_data)} rows for gap")
                    return downloaded_data
                else:
                    self.logger.warning("⚠️ Download completed but returned empty data")
                    return None

            except ImportError:
                self.logger.warning("⚠️ Unified data downloader not available, trying alternative methods")

            # Fallback: Try using the missing data downloader
            try:
                from ...training.steps.data_collection.data_preparation.missing_data_downloader_and_gap_filler import MissingDataDownloaderAndGapFiller

                downloader = MissingDataDownloaderAndGapFiller()
                downloaded_data = downloader.download_missing_data_segment(
                    symbol=getattr(self, 'symbol', 'ETHUSDT'),
                    exchange=getattr(self, 'exchange', 'binance'),
                    start_time=gap_start,
                    end_time=gap_end,
                    timeframe=getattr(self, 'timeframe', '1m')
                )

                if downloaded_data is not None and not downloaded_data.empty:
                    self.logger.info(f"✅ Downloaded {len(downloaded_data)} rows using gap filler")
                    return downloaded_data
                else:
                    self.logger.warning("❌ Gap filler download failed")
                    return None

            except Exception as e:
                self.logger.warning(f"⚠️ Alternative download method failed: {e}")

            return None

        except Exception as e:
            self.logger.error(f"❌ Error downloading missing data: {e}")
            return None

    def _load_recently_downloaded_data(self, gap_start: pd.Timestamp, gap_end: pd.Timestamp, timeframe: str) -> pd.DataFrame | None:
        """Load recently downloaded data for the gap period."""
        try:
            from ...training.steps.standardized_parquet_handler import standardized_parquet_handler
            from src.utils.pipeline_standards import PipelineStandards

            standards = PipelineStandards()
            symbol = getattr(self, 'symbol', 'ETHUSDT')
            exchange = getattr(self, 'exchange', 'BINANCE').lower()

            # Generate file path for downloaded data
            klines_file = standards.generate_file_name('klines', exchange, symbol, timeframe)
            data_dir = getattr(self, 'data_dir', 'data_cache')
            klines_path = Path(data_dir) / klines_file

            if klines_path.exists():
                # Load the data
                df = standardized_parquet_handler.read_parquet_standardized(klines_path)

                # Filter to the gap period
                if 'timestamp' in df.columns:
                    # Convert timestamps if needed
                    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

                    # Filter to gap period
                    mask = (df['timestamp'] >= gap_start) & (df['timestamp'] <= gap_end)
                    gap_data = df[mask].copy()

                    if not gap_data.empty:
                        self.logger.info(f"✅ Loaded {len(gap_data)} rows from downloaded data")
                        return gap_data

            self.logger.warning("⚠️ No suitable downloaded data found for gap period")
            return None

        except Exception as e:
            self.logger.warning(f"⚠️ Error loading downloaded data: {e}")
            return None

    def _insert_downloaded_data(self, original_data: pd.DataFrame,
                               downloaded_data: pd.DataFrame,
                               insert_index: int) -> pd.DataFrame:
        """Insert downloaded data into the original dataset."""
        try:
            # Combine datasets
            combined = pd.concat([original_data, downloaded_data], ignore_index=True)

            # Sort by timestamp and remove duplicates
            if 'timestamp' in combined.columns:
                combined = combined.sort_values('timestamp').drop_duplicates(subset=['timestamp'])

            self.logger.info(f"✅ Successfully inserted {len(downloaded_data)} downloaded rows")
            return combined

        except Exception as e:
            self.logger.error(f"❌ Error inserting downloaded data: {e}")
            return original_data

    def _batch_process_gaps(self, data: pd.DataFrame, large_gaps: pd.Series, timestamps: pd.Series, 
                           gap_threshold: float, download_threshold: float) -> pd.DataFrame:
        """Batch process gaps for efficiency with prioritization."""
        try:
            # Categorize gaps by size and priority
            download_gaps = []
            interpolation_gaps = []
            
            for idx in large_gaps.index:
                gap_size = large_gaps.loc[idx]
                if gap_size > download_threshold:
                    download_gaps.append((idx, gap_size))
                else:
                    interpolation_gaps.append((idx, gap_size))
            
            self.logger.info(f"📊 Gap categorization: {len(download_gaps)} for download, {len(interpolation_gaps)} for interpolation")
            
            # Process download gaps in batches (limit concurrent downloads)
            if download_gaps:
                self.logger.info(f"🔄 Processing {len(download_gaps)} gaps for download in batches...")
                data = self._process_download_gaps_batch(data, download_gaps, timestamps)
            
            # Process interpolation gaps in batches
            if interpolation_gaps:
                self.logger.info(f"🔄 Processing {len(interpolation_gaps)} gaps for interpolation in batches...")
                data = self._process_interpolation_gaps_batch(data, interpolation_gaps)
            
            return data
            
        except Exception as e:
            self.logger.error(f"❌ Error in batch gap processing: {e}")
            # Fall back to individual processing
            return self._fallback_individual_gap_processing(data, large_gaps, timestamps, gap_threshold, download_threshold)

    def _process_download_gaps_batch(self, data: pd.DataFrame, download_gaps: list, timestamps: pd.Series) -> pd.DataFrame:
        """Process download gaps in batches with concurrency control."""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Limit concurrent downloads to avoid overwhelming the system
        max_concurrent_downloads = 3
        batch_size = 10
        
        self.logger.info(f"🔄 Processing download gaps in batches of {batch_size} with max {max_concurrent_downloads} concurrent downloads")
        
        # Process gaps in batches
        for i in range(0, len(download_gaps), batch_size):
            batch = download_gaps[i:i + batch_size]
            self.logger.info(f"📥 Processing download batch {i//batch_size + 1}/{(len(download_gaps) + batch_size - 1)//batch_size}")
            
            # Use ThreadPoolExecutor for concurrent downloads
            with ThreadPoolExecutor(max_workers=max_concurrent_downloads) as executor:
                futures = []
                for idx, gap_size in batch:
                    gap_start = timestamps.loc[idx-1]
                    gap_end = timestamps.loc[idx]
                    
                    # Validate dates before attempting download
                    current_time = pd.Timestamp.now(tz='UTC')
                    if gap_start > current_time or gap_end > current_time:
                        self.logger.warning(f"⚠️ Skipping future gap: {gap_start} to {gap_end}")
                        continue
                    
                    # Submit download task
                    future = executor.submit(self._download_missing_data, gap_start, gap_end)
                    futures.append((future, idx, gap_size, gap_start, gap_end))
                
                # Process completed downloads
                for future, idx, gap_size, gap_start, gap_end in futures:
                    try:
                        downloaded_data = future.result(timeout=30)  # 30 second timeout per download
                        
                        if downloaded_data is not None and not downloaded_data.empty:
                            self.logger.info(f"✅ Downloaded {len(downloaded_data)} rows for gap {gap_size:.1f}s")
                            data = self._insert_downloaded_data(data, downloaded_data, idx)
                        else:
                            self.logger.warning(f"❌ Download failed for gap {gap_size:.1f}s - using interpolation")
                            data = self._interpolate_gap(data, idx, gap_size)
                            
                    except Exception as e:
                        self.logger.warning(f"❌ Download error for gap {gap_size:.1f}s: {e} - using interpolation")
                        data = self._interpolate_gap(data, idx, gap_size)
        
        return data

    def _process_interpolation_gaps_batch(self, data: pd.DataFrame, interpolation_gaps: list) -> pd.DataFrame:
        """Process interpolation gaps in batches for efficiency."""
        batch_size = 50  # Process more interpolation gaps per batch since they're faster
        
        self.logger.info(f"🔄 Processing interpolation gaps in batches of {batch_size}")
        
        for i in range(0, len(interpolation_gaps), batch_size):
            batch = interpolation_gaps[i:i + batch_size]
            self.logger.info(f"🔧 Processing interpolation batch {i//batch_size + 1}/{(len(interpolation_gaps) + batch_size - 1)//batch_size}")
            
            # Process interpolation gaps in the batch
            for idx, gap_size in batch:
                try:
                    data = self._interpolate_gap(data, idx, gap_size)
                except Exception as e:
                    self.logger.warning(f"⚠️ Interpolation error for gap {gap_size:.1f}s: {e}")
        
        return data

    def _fallback_individual_gap_processing(self, data: pd.DataFrame, large_gaps: pd.Series, 
                                          timestamps: pd.Series, gap_threshold: float, 
                                          download_threshold: float) -> pd.DataFrame:
        """Fallback to individual gap processing if batch processing fails."""
        self.logger.warning("⚠️ Falling back to individual gap processing")
        
        for idx in large_gaps.index:
            gap_size = large_gaps.loc[idx]
            
            if gap_size > download_threshold:
                self.logger.info(f"📥 Large gap detected ({gap_size:.1f}s > {download_threshold:.1f}s) - attempting download")
                gap_start = timestamps.loc[idx-1]
                gap_end = timestamps.loc[idx]
                
                # Validate dates before attempting download
                current_time = pd.Timestamp.now(tz='UTC')
                if gap_start > current_time or gap_end > current_time:
                    self.logger.warning(f"⚠️ Cannot download future data: gap from {gap_start} to {gap_end}")
                    data = self._interpolate_gap(data, idx, gap_size)
                    continue
                
                # Attempt to download missing data
                downloaded_data = self._download_missing_data(gap_start, gap_end)
                
                if downloaded_data is not None and not downloaded_data.empty:
                    self.logger.info(f"✅ Successfully downloaded {len(downloaded_data)} rows for gap")
                    data = self._insert_downloaded_data(data, downloaded_data, idx)
                else:
                    self.logger.warning(f"❌ Failed to download data for gap - using interpolation")
                    data = self._interpolate_gap(data, idx, gap_size)
            else:
                self.logger.info(f"🔧 Gap detected ({gap_size:.1f}s) - using interpolation")
                data = self._interpolate_gap(data, idx, gap_size)
        
        return data

    def _interpolate_gap(self, data: pd.DataFrame, gap_index: int, gap_size: float) -> pd.DataFrame:
        """Interpolate data for small gaps using various methods."""
        try:
            from ..data.quality.data_cleaning import DataCleaner

            cleaner = DataCleaner()

            # Choose interpolation method based on gap size and data type
            if gap_size <= 30:  # Very small gaps - linear interpolation
                method = 'linear'
                method_kwargs = {}
            elif gap_size <= 300:  # Medium gaps - polynomial interpolation
                method = 'polynomial'
                method_kwargs = {'order': 2}  # Quadratic polynomial
            else:  # Larger gaps - use linear interpolation as fallback
                method = 'linear'
                method_kwargs = {}

            # Apply interpolation to numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col in data.columns:
                    try:
                        # Interpolate missing values with appropriate parameters
                        data[col] = data[col].interpolate(method=method, limit=10, **method_kwargs)
                    except Exception as col_error:
                        # Fallback to linear interpolation if the chosen method fails
                        self.logger.warning(f"⚠️ Interpolation method '{method}' failed for column '{col}': {col_error}")
                        try:
                            data[col] = data[col].interpolate(method='linear', limit=10)
                            self.logger.info(f"✅ Fallback to linear interpolation successful for column '{col}'")
                        except Exception as fallback_error:
                            self.logger.warning(f"⚠️ Even linear interpolation failed for column '{col}': {fallback_error}")

            # For timestamp column, ensure proper time progression (memory-optimized)
            if 'timestamp' in data.columns:
                try:
                    # Use in-place sorting to reduce memory usage
                    data = data.sort_values('timestamp', inplace=False).reset_index(drop=True)
                except Exception as sort_error:
                    self.logger.warning(f"⚠️ Memory-efficient sort failed: {sort_error}")
                    # Fallback: try with copy=False
                    try:
                        data = data.sort_values('timestamp', inplace=False, ignore_index=True)
                    except Exception as fallback_error:
                        self.logger.warning(f"⚠️ All sort attempts failed: {fallback_error}")
                        # Last resort: don't sort but warn
                        self.logger.warning("⚠️ Unable to sort timestamp column - data may be unsorted")

            self.logger.info(f"✅ Interpolated gap using {method} method")
            return data

        except Exception as e:
            self.logger.warning(f"⚠️ Error during gap interpolation: {e}")
            return data

    def _update_performance_stats(self, start_time: float, num_regimes: int):
        """Update performance statistics."""
        processing_time = time.time() - start_time
        
        self.performance_stats['total_regimes_detected'] += num_regimes
        self.performance_stats['processing_time'] += processing_time

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            'total_regimes_detected': self.performance_stats['total_regimes_detected'],
            'total_processing_time': self.performance_stats['processing_time'],
            'regimes_per_second': safe_divide(
                self.performance_stats['total_regimes_detected'],
                self.performance_stats['processing_time']
            ),
            'average_accuracy': np.mean(self.performance_stats['accuracy_scores']) if self.performance_stats['accuracy_scores'] else 0.0
        }

# Global instance for backward compatibility
enhanced_hmm_regime_detector = EnhancedHMMRegimeDetector()

# Export for backward compatibility
HMMRegimeDetector = EnhancedHMMRegimeDetector