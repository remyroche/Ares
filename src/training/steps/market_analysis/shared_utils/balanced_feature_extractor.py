"""
Balanced Feature Extractor - Shared Utility

This module provides a comprehensive, balanced feature extraction system that can be used
by both NAS and TAS regime detection systems. It leverages existing feature generation
tools and implements TAS-style balanced feature extraction to prevent imbalanced clusters.

Key Features:
- 7D Feature Categories: Price, Volume, Volatility, Momentum, Trend, Technical, Statistical
- TAS-style balanced feature extraction
- Integration with PID-based feature generation
- Integration with existing feature_generation tools
- Hardware-optimized computations
- Prevents clustering imbalance through balanced feature design
"""

import numpy as np
import pandas as pd
import gc
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, TypeVar, Generic
import logging
from dataclasses import dataclass, field
from enum import Enum
import time
from datetime import datetime

# Import tprint for consistent logging
from src.utils.tprint import tprint

# Core dependencies with fallback support
try:
    from sklearn.preprocessing import RobustScaler, StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Import feature generation tools from unified system
try:
    from src.feature_generation import (
        FeatureBank,
        ReturnsFeatureGenerator,
        MomentumFeatureGenerator,
        VolumeFeatureGenerator,
        VolatilityFeatureGenerator,
        TrendFeatureGenerator,
        InteractionFeatureGenerator,
        CrossTimeframeFeatureGenerator,
        generate_features_by_category,
        FeatureGenerationOptimizer,
        get_feature_optimizer
    )
    FEATURE_GENERATION_AVAILABLE = True
except ImportError:
    FEATURE_GENERATION_AVAILABLE = False

# Hybrid NAS/TAS features removed - no longer needed for market_analysis
HYBRID_FEATURES_AVAILABLE = False

# Import matrix operations for hardware optimization
try:
    from src.utils.matrix_operations import get_unified_matrix_operations
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError:
    MATRIX_OPERATIONS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Custom exception classes for better error handling
class FeatureExtractionError(Exception):
    """Base exception for feature extraction errors."""
    pass

class MemoryLimitExceededError(FeatureExtractionError):
    """Raised when memory limit is exceeded during feature extraction."""
    pass

class FeatureCategoryError(FeatureExtractionError):
    """Raised when feature category extraction fails."""
    pass

class DataValidationError(FeatureExtractionError):
    """Raised when input data validation fails."""
    pass

class HardwareOptimizationError(FeatureExtractionError):
    """Raised when hardware optimization fails."""
    pass

class FeatureCategory(Enum):
    """Feature categories for balanced extraction."""
    PRICE: str = "price"
    VOLUME: str = "volume"
    VOLATILITY: str = "volatility"
    MOMENTUM: str = "momentum"
    TREND: str = "trend"
    TECHNICAL: str = "technical"
    REGIME: str = "regime"
    INTERACTION: str = "interaction"

@dataclass
class BalancedFeatureConfig:
    """Configuration for balanced feature extraction."""
    # Feature categories to include (INTERACTION removed for unified feature set)
    enabled_categories: List[FeatureCategory] = field(default_factory=lambda: [
        FeatureCategory.PRICE, FeatureCategory.VOLUME, FeatureCategory.VOLATILITY,
        FeatureCategory.MOMENTUM, FeatureCategory.TREND, FeatureCategory.TECHNICAL,
        FeatureCategory.REGIME
    ])

    # TAS-style balanced extraction settings
    use_tas_style_extraction: bool = True
    use_balanced_scaling: bool = True
    use_quantile_features: bool = True
    use_ratio_based_features: bool = True

    # Feature engineering settings
    enable_pid_features: bool = True
    enable_hybrid_features: bool = True
    enable_hardware_optimization: bool = True

    # Clustering balance settings
    max_feature_range: float = 3.0  # 3-sigma bounds for features
    min_cluster_balance_ratio: float = 0.1  # Minimum 10% of data per cluster
    feature_normalization_method: str = "robust"  # "robust", "standard", "minmax"

    # Feature selection settings
    max_features_per_category: int = 20
    total_max_features: int = 100
    enable_feature_selection: bool = True

    # Performance settings
    batch_size: int = 1000
    enable_caching: bool = True

    # Enhanced regime detection settings
    use_numpy_optimization: bool = True
    enable_temporal_features: bool = True
    enable_micro_regime_features: bool = True
    target_regime_count: int = 6  # Increased from 4 to 6-8
    micro_regime_threshold: float = 0.3  # Sensitivity for micro-regime detection
    regime_stability_window: int = 10  # Window for regime stability analysis

@dataclass
class BalancedFeatureResult:
    """Result from balanced feature extraction."""
    features: np.ndarray
    feature_names: List[str]
    feature_categories: Dict[str, List[str]]
    extraction_metadata: Dict[str, Any]
    processing_time: float
    balance_metrics: Dict[str, float]
    success: bool
    error_message: Optional[str] = None

class BalancedFeatureExtractor:
    """
    Comprehensive balanced feature extractor that prevents clustering imbalance.

    This extractor uses TAS-style balanced feature extraction combined with
    existing feature generation tools to create well-distributed features.
    """

    def __init__(self, config: Optional[BalancedFeatureConfig] = None) -> None:
        """Initialize the balanced feature extractor.

        Args:
            config: Configuration for feature extraction
        """
        tprint("🚀 Initializing BalancedFeatureExtractor", color="cyan", bold=True)

        self.config = config or BalancedFeatureConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize components
        self.scaler = None
        self.feature_selector = None
        self.pca = None

        # Initialize matrix operations for hardware optimization
        self.matrix_ops = None
        if MATRIX_OPERATIONS_AVAILABLE and self.config.enable_hardware_optimization:
            try:
                self.matrix_ops = get_unified_matrix_operations()
                tprint("✅ Matrix operations initialized for hardware optimization", color="green")
            except Exception as e:
                tprint(f"⚠️ Matrix operations initialization failed: {e}", color="yellow")

        # Initialize feature generation system
        self.feature_bank = None
        self.feature_optimizer = None
        if FEATURE_GENERATION_AVAILABLE and self.config.enable_pid_features:
            try:
                from src.feature_generation.core.feature_bank import get_global_feature_bank
                self.feature_bank = get_global_feature_bank()
                self.feature_optimizer = get_feature_optimizer()
                tprint("✅ Feature generation system initialized", color="green")
            except Exception as e:
                tprint(f"⚠️ Feature generation system initialization failed: {e}", color="yellow")

        # Initialize hybrid feature calculator
        self.hybrid_calculator = None
        if HYBRID_FEATURES_AVAILABLE and self.config.enable_hybrid_features:
            try:
                collection_config = FeatureCollectionConfig(
                    use_standardized_features=True,
                    feature_categories=['momentum', 'volatility', 'volume', 'trend'],
                    lookback_periods=[5, 10, 20],
                    use_hardware_acceleration=True,
                    use_matrix_operations=True
                )
                self.hybrid_calculator = StandardizedFeatureCalculator(collection_config)
                tprint("✅ Hybrid feature calculator initialized", color="green")
            except Exception as e:
                tprint(f"⚠️ Hybrid calculator initialization failed: {e}", color="yellow")

        tprint("✅ BalancedFeatureExtractor initialized successfully", color="green")

    def extract_balanced_features(self, data: Union[np.ndarray, pd.DataFrame],
                                labels: Optional[np.ndarray] = None) -> BalancedFeatureResult:
        """
        Extract balanced features to prevent clustering imbalance.

        Args:
            data: Input data (numpy array or DataFrame)
            labels: Optional labels for supervised feature selection

        Returns:
            BalancedFeatureResult with extracted features and metadata
        """
        start_time = time.time()
        tprint("🔍 Starting balanced feature extraction", color="blue")

        try:
            # Convert to DataFrame if needed
            if isinstance(data, np.ndarray):
                data_df = self._array_to_dataframe(data)
            else:
                data_df = data.copy()

            # Initialize result containers
            all_features = []
            feature_names = []
            feature_categories = {category.value: [] for category in FeatureCategory}
            # Add temporal and micro_regime categories
            feature_categories['temporal'] = []
            feature_categories['micro_regime'] = []
            extraction_metadata = {}

            # Extract features by category
            for category in self.config.enabled_categories:
                tprint(f"📊 Extracting {category.value} features", color="cyan")

                try:
                    if category == FeatureCategory.PRICE:
                        features, names = self._extract_price_features_balanced(data_df)
                    elif category == FeatureCategory.VOLUME:
                        features, names = self._extract_volume_features_balanced(data_df)
                    elif category == FeatureCategory.VOLATILITY:
                        features, names = self._extract_volatility_features_balanced(data_df)
                    elif category == FeatureCategory.MOMENTUM:
                        features, names = self._extract_momentum_features_balanced(data_df)
                    elif category == FeatureCategory.TREND:
                        features, names = self._extract_trend_features_balanced(data_df)
                    elif category == FeatureCategory.TECHNICAL:
                        features, names = self._extract_technical_features_balanced(data_df)
                    elif category == FeatureCategory.REGIME:
                        features, names = self._extract_statistical_features_balanced(data_df)
                    elif category == FeatureCategory.INTERACTION:
                        features, names = self._extract_interaction_features_balanced(data_df)
                    else:
                        continue

                    if features is not None and len(features) > 0:
                        all_features.append(features)
                        feature_names.extend(names)
                        feature_categories[category.value].extend(names)
                        extraction_metadata[category.value] = {
                            'feature_count': len(names),
                            'feature_shape': features.shape
                        }
                        tprint(f"✅ {category.value}: {len(names)} features extracted", color="green")
                    else:
                        tprint(f"⚠️ {category.value}: No features extracted", color="yellow")

                except Exception as e:
                    tprint(f"❌ {category.value} extraction failed: {e}", color="red")
                    self.logger.warning(f"Feature extraction failed for {category.value}: {e}")
                    continue

            # Extract temporal features if enabled
            if self.config.enable_temporal_features:
                tprint("📊 Extracting temporal features", color="cyan")
                try:
                    temporal_features, temporal_names = self._extract_temporal_features_balanced(data_df)
                    if temporal_features is not None and len(temporal_features) > 0:
                        all_features.append(temporal_features)
                        feature_names.extend(temporal_names)
                        feature_categories['temporal'].extend(temporal_names)
                        extraction_metadata['temporal'] = {
                            'feature_count': len(temporal_names),
                            'feature_shape': temporal_features.shape
                        }
                        tprint(f"✅ temporal: {len(temporal_names)} features extracted", color="green")
                except Exception as e:
                    tprint(f"❌ temporal extraction failed: {e}", color="red")
                    self.logger.warning(f"Temporal feature extraction failed: {e}")

            # Extract micro-regime features if enabled
            if self.config.enable_micro_regime_features:
                tprint("📊 Extracting micro-regime features", color="cyan")
                try:
                    micro_features, micro_names = self._extract_micro_regime_features_balanced(data_df)
                    if micro_features is not None and len(micro_features) > 0:
                        all_features.append(micro_features)
                        feature_names.extend(micro_names)
                        feature_categories['micro_regime'].extend(micro_names)
                        extraction_metadata['micro_regime'] = {
                            'feature_count': len(micro_names),
                            'feature_shape': micro_features.shape
                        }
                        tprint(f"✅ micro_regime: {len(micro_names)} features extracted", color="green")
                except Exception as e:
                    tprint(f"❌ micro-regime extraction failed: {e}", color="red")
                    self.logger.warning(f"Micro-regime feature extraction failed: {e}")

            # Combine all features
            if not all_features:
                raise ValueError("No features were successfully extracted")

            combined_features = np.concatenate(all_features, axis=1)
            tprint(f"📊 Combined features shape: {combined_features.shape}", color="blue")

            # Apply balanced scaling
            if self.config.use_balanced_scaling:
                combined_features = self._apply_balanced_scaling(combined_features)
                tprint("✅ Balanced scaling applied", color="green")

            # Feature selection if enabled
            if self.config.enable_feature_selection and labels is not None:
                combined_features, selected_names = self._select_features_balanced(
                    combined_features, feature_names, labels
                )
                feature_names = selected_names
                tprint(f"✅ Feature selection completed: {len(selected_names)} features selected", color="green")

            # Calculate balance metrics
            balance_metrics = self._calculate_balance_metrics(combined_features)

            processing_time = time.time() - start_time

            result = BalancedFeatureResult(
                features=combined_features,
                feature_names=feature_names,
                feature_categories=feature_categories,
                extraction_metadata=extraction_metadata,
                processing_time=processing_time,
                balance_metrics=balance_metrics,
                success=True
            )

            tprint(f"✅ Balanced feature extraction completed in {processing_time:.2f}s", color="green")
            tprint(f"📊 Final features: {combined_features.shape[1]} features, {combined_features.shape[0]} samples", color="blue")

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Balanced feature extraction failed: {e}"
            tprint(f"❌ {error_msg}", color="red")
            self.logger.error(error_msg)

            return BalancedFeatureResult(
                features=data if isinstance(data, np.ndarray) else data.values,
                feature_names=[],
                feature_categories={},
                extraction_metadata={},
                processing_time=processing_time,
                balance_metrics={},
                success=False,
                error_message=error_msg
            )

    def _validate_input_data(self, data: Union[np.ndarray, pd.DataFrame]) -> None:
        """Validate input data for feature extraction."""
        try:
            if data is None:
                raise DataValidationError("Input data cannot be None")

            if isinstance(data, np.ndarray):
                if data.size == 0:
                    raise DataValidationError("Input array is empty")
                if len(data.shape) < 2:
                    raise DataValidationError("Input array must be at least 2D")
            elif isinstance(data, pd.DataFrame):
                if len(data) == 0:
                    raise DataValidationError("Input DataFrame is empty")
                if len(data.columns) < 2:
                    raise DataValidationError("Input DataFrame must have at least 2 columns")
            else:
                raise DataValidationError(f"Unsupported data type: {type(data)}")

        except Exception as e:
            if isinstance(e, DataValidationError):
                raise
            else:
                raise DataValidationError(f"Data validation failed: {e}")

    def _check_memory_usage(self) -> None:
        """Check memory usage and trigger cleanup if needed."""
        if not self.memory_optimizer:
            return

        try:
            memory_stats = self.memory_optimizer.get_memory_stats()
            memory_percent = memory_stats.get('memory_percent', 0)

            if memory_percent > self.config.memory_cleanup_threshold * 100:
                tprint(f"🧠 Memory usage high ({memory_percent:.1f}%), triggering cleanup", color="yellow")
                self.memory_optimizer._moderate_memory_cleanup()

                # Check if still high after cleanup
                memory_stats = self.memory_optimizer.get_memory_stats()
                memory_percent = memory_stats.get('memory_percent', 0)

                if memory_percent > 95:  # Critical threshold
                    raise MemoryLimitExceededError(f"Memory usage too high: {memory_percent:.1f}%")

        except Exception as e:
            if isinstance(e, MemoryLimitExceededError):
                raise
            else:
                self.logger.warning(f"Memory check failed: {e}")

    def _memory_efficient_concatenate(self, feature_arrays: List[np.ndarray]) -> np.ndarray:
        """Memory-efficient concatenation for large feature arrays."""
        try:
            if not feature_arrays:
                return np.array([])

            if len(feature_arrays) == 1:
                return feature_arrays[0]

            # Calculate total shape
            n_samples = feature_arrays[0].shape[0]
            total_features = sum(arr.shape[1] for arr in feature_arrays)

            # Pre-allocate result array
            result = np.empty((n_samples, total_features), dtype=feature_arrays[0].dtype)

            # Fill result array in chunks to avoid memory spikes
            start_idx = 0
            for arr in feature_arrays:
                end_idx = start_idx + arr.shape[1]
                result[:, start_idx:end_idx] = arr
                start_idx = end_idx

                # Force garbage collection after each chunk
                if self.memory_optimizer:
                    gc.collect()

            return result

        except Exception as e:
            self.logger.warning(f"Memory-efficient concatenation failed: {e}")
            # Fallback to standard concatenation
            return np.concatenate(feature_arrays, axis=1)

    def _numpy_rolling_std(self, data: np.ndarray, window: int) -> np.ndarray:
        """Numpy-based rolling standard deviation for better performance."""
        try:
            if len(data) < window:
                return np.full(len(data), np.nan)

            # Use numpy for rolling calculations instead of pandas
            result = np.full(len(data), np.nan)

            for i in range(window - 1, len(data)):
                window_data = data[i - window + 1:i + 1]
                if not np.any(np.isnan(window_data)):
                    result[i] = np.std(window_data)

            return result

        except Exception as e:
            self.logger.warning(f"Numpy rolling std failed: {e}")
            # Fallback to pandas
            return pd.Series(data).rolling(window=window).std().values

    def _numpy_rolling_mean(self, data: np.ndarray, window: int) -> np.ndarray:
        """Numpy-based rolling mean for better performance."""
        try:
            if len(data) < window:
                return np.full(len(data), np.nan)

            result = np.full(len(data), np.nan)

            for i in range(window - 1, len(data)):
                window_data = data[i - window + 1:i + 1]
                if not np.any(np.isnan(window_data)):
                    result[i] = np.mean(window_data)

            return result

        except Exception as e:
            self.logger.warning(f"Numpy rolling mean failed: {e}")
            # Fallback to pandas
            return pd.Series(data).rolling(window=window).mean().values

    def _numpy_rolling_skew(self, data: np.ndarray, window: int) -> np.ndarray:
        """Numpy-based rolling skewness for better performance."""
        try:
            if len(data) < window:
                return np.full(len(data), np.nan)

            result = np.full(len(data), np.nan)

            for i in range(window - 1, len(data)):
                window_data = data[i - window + 1:i + 1]
                if not np.any(np.isnan(window_data)) and len(window_data) == window:
                    # Calculate skewness manually
                    mean_val = np.mean(window_data)
                    std_val = np.std(window_data)
                    if std_val > 0:
                        skew = np.mean(((window_data - mean_val) / std_val) ** 3)
                        result[i] = skew
                    else:
                        result[i] = 0.0

            return result

        except Exception as e:
            self.logger.warning(f"Numpy rolling skew failed: {e}")
            # Fallback to pandas
            return pd.Series(data).rolling(window=window).apply(
                lambda x: x.skew() if len(x) == window else np.nan
            ).values

    def _numpy_rolling_kurtosis(self, data: np.ndarray, window: int) -> np.ndarray:
        """Numpy-based rolling kurtosis for better performance."""
        try:
            if len(data) < window:
                return np.full(len(data), np.nan)

            result = np.full(len(data), np.nan)

            for i in range(window - 1, len(data)):
                window_data = data[i - window + 1:i + 1]
                if not np.any(np.isnan(window_data)) and len(window_data) == window:
                    # Calculate kurtosis manually
                    mean_val = np.mean(window_data)
                    std_val = np.std(window_data)
                    if std_val > 0:
                        kurt = np.mean(((window_data - mean_val) / std_val) ** 4) - 3
                        result[i] = kurt
                    else:
                        result[i] = 0.0

            return result

        except Exception as e:
            self.logger.warning(f"Numpy rolling kurtosis failed: {e}")
            # Fallback to pandas
            return pd.Series(data).rolling(window=window).apply(
                lambda x: x.kurtosis() if len(x) == window else np.nan
            ).values

    def _array_to_dataframe(self, data: np.ndarray) -> pd.DataFrame:
        """Convert numpy array to DataFrame for processing."""
        try:
            # Assume OHLCV format if 5+ columns
            if data.shape[1] >= 5:
                columns = ['open', 'high', 'low', 'close', 'volume']
                if data.shape[1] > 5:
                    columns.extend([f'feature_{i}' for i in range(5, data.shape[1])])
            elif data.shape[1] >= 4:
                columns = ['open', 'high', 'low', 'close']
                if data.shape[1] > 4:
                    columns.extend([f'feature_{i}' for i in range(4, data.shape[1])])
            else:
                columns = [f'feature_{i}' for i in range(data.shape[1])]

            return pd.DataFrame(data, columns=columns)

        except Exception as e:
            self.logger.warning(f"Array to DataFrame conversion failed: {e}")
            return pd.DataFrame(data, columns=[f'feature_{i}' for i in range(data.shape[1])])

    def _extract_price_features_balanced(self, data_df: pd.DataFrame) -> Tuple[Optional[np.ndarray], List[str]]:
        """Extract balanced price features using TAS-style approach."""
        try:
            features = []
            names = []

            if 'close' in data_df.columns:
                close_price = data_df['close'].values

                # Normalized price (TAS-style)
                if self.config.use_tas_style_extraction:
                    normalized_price = (close_price - np.mean(close_price)) / (np.std(close_price) + 1e-8)
                    features.append(normalized_price.reshape(-1, 1))
                    names.append('normalized_close')

                # Price ratios (bounded)
                if len(close_price) > 1:
                    price_ratios = close_price[1:] / close_price[:-1]
                    price_ratios_padded = np.concatenate([[1], price_ratios])
                    # Log transform to reduce extreme values
                    log_ratios = np.log(price_ratios_padded + 1e-8)
                    log_ratios = np.clip(log_ratios, -2, 2)  # Bounded
                    features.append(log_ratios.reshape(-1, 1))
                    names.append('log_price_ratio')

                # Price position within range (if high/low available)
                if 'high' in data_df.columns and 'low' in data_df.columns:
                    high_low_range = data_df['high'].values - data_df['low'].values
                    price_position = (data_df['close'].values - data_df['low'].values) / (high_low_range + 1e-8)
                    price_position = np.clip(price_position, 0, 1)  # Bounded
                    features.append(price_position.reshape(-1, 1))
                    names.append('price_position')

                # Quantile-based features (balanced)
                if self.config.use_quantile_features:
                    quantiles = [0.25, 0.5, 0.75]
                    for q in quantiles:
                        q_value = np.percentile(close_price, q * 100)
                        distance_to_q = (close_price - q_value) / (np.std(close_price) + 1e-8)
                        distance_to_q = np.clip(distance_to_q, -3, 3)  # Bounded
                        features.append(distance_to_q.reshape(-1, 1))
                        names.append(f'distance_to_q{q}')

            if features:
                return np.concatenate(features, axis=1), names
            else:
                return None, []

        except Exception as e:
            self.logger.warning(f"Price features extraction failed: {e}")
            return None, []

    def _extract_volume_features_balanced(self, data_df: pd.DataFrame) -> Tuple[Optional[np.ndarray], List[str]]:
        """Extract balanced volume features."""
        try:
            features = []
            names = []

            if 'volume' in data_df.columns:
                volume = data_df['volume'].values

                # Normalized volume
                if self.config.use_tas_style_extraction:
                    normalized_volume = (volume - np.mean(volume)) / (np.std(volume) + 1e-8)
                    normalized_volume = np.clip(normalized_volume, -3, 3)
                    features.append(normalized_volume.reshape(-1, 1))
                    names.append('normalized_volume')

                # Volume ratios (bounded)
                if len(volume) > 1:
                    volume_ratios = volume[1:] / (volume[:-1] + 1e-8)
                    volume_ratios_padded = np.concatenate([[1], volume_ratios])
                    log_volume_ratios = np.log(volume_ratios_padded + 1e-8)
                    log_volume_ratios = np.clip(log_volume_ratios, -2, 2)
                    features.append(log_volume_ratios.reshape(-1, 1))
                    names.append('log_volume_ratio')

                # Volume position (if price available)
                if 'close' in data_df.columns:
                    price_volume_ratio = volume / (data_df['close'].values + 1e-8)
                    price_volume_ratio = (price_volume_ratio - np.mean(price_volume_ratio)) / (np.std(price_volume_ratio) + 1e-8)
                    price_volume_ratio = np.clip(price_volume_ratio, -3, 3)
                    features.append(price_volume_ratio.reshape(-1, 1))
                    names.append('price_volume_ratio')

            if features:
                return np.concatenate(features, axis=1), names
            else:
                return None, []

        except Exception as e:
            self.logger.warning(f"Volume features extraction failed: {e}")
            return None, []

    def _extract_volatility_features_balanced(self, data_df: pd.DataFrame) -> Tuple[Optional[np.ndarray], List[str]]:
        """Extract balanced volatility features using TAS-style ratio-based approach."""
        try:
            features = []
            names = []

            if 'close' not in data_df.columns:
                raise FeatureCategoryError("Close price column not found for volatility features")

            close_price = data_df['close'].values

            # Volatility periods (balanced set)
            periods = [5, 10, 20]

            for period in periods:
                if len(close_price) <= period:
                    continue

                try:
                    # Use numpy-based rolling calculations for better performance
                    if self.config.use_numpy_optimization:
                        rolling_std = self._numpy_rolling_std(close_price, period)
                        rolling_mean_vol = self._numpy_rolling_mean(rolling_std, period)
                    else:
                        # Fallback to pandas
                        rolling_std = pd.Series(close_price).rolling(window=period).std().values
                        rolling_mean_vol = pd.Series(rolling_std).rolling(window=period).mean().values

                    # TAS-style volatility ratio (more balanced)
                    vol_ratio = rolling_std / (rolling_mean_vol + 1e-8)

                    # Bounded volatility features
                    valid_std = rolling_std[~np.isnan(rolling_std)]
                    if len(valid_std) > 0:
                        vol_normalized = np.clip(rolling_std, 0, np.percentile(valid_std, 95))
                    else:
                        vol_normalized = rolling_std

                    vol_ratio_normalized = np.clip(vol_ratio, 0, 5)  # Cap extreme ratios

                    # Fill NaN values
                    vol_normalized = np.nan_to_num(vol_normalized, nan=np.nanmean(vol_normalized))
                    vol_ratio_normalized = np.nan_to_num(vol_ratio_normalized, nan=1.0)

                    features.append(vol_normalized.reshape(-1, 1))
                    features.append(vol_ratio_normalized.reshape(-1, 1))
                    names.extend([f'volatility_{period}', f'vol_ratio_{period}'])

                except Exception as e:
                    self.logger.warning(f"Volatility period {period} failed: {e}")
                    continue

            # Returns-based volatility (bounded)
            if len(close_price) > 1:
                try:
                    returns = np.diff(close_price) / (close_price[:-1] + 1e-8)
                    abs_returns = np.abs(returns)
                    squared_returns = returns ** 2

                    # Pad to match original length
                    abs_returns_padded = np.concatenate([[0], abs_returns])
                    squared_returns_padded = np.concatenate([[0], squared_returns])

                    # Bounded returns
                    abs_returns_padded = np.clip(abs_returns_padded, 0, 0.1)  # Cap at 10%
                    squared_returns_padded = np.clip(squared_returns_padded, 0, 0.01)  # Cap at 1%

                    features.append(abs_returns_padded.reshape(-1, 1))
                    features.append(squared_returns_padded.reshape(-1, 1))
                    names.extend(['abs_returns', 'squared_returns'])

                except Exception as e:
                    self.logger.warning(f"Returns-based volatility failed: {e}")

            if features:
                return np.concatenate(features, axis=1), names
            else:
                return None, []

        except FeatureCategoryError:
            raise
        except Exception as e:
            self.logger.warning(f"Volatility features extraction failed: {e}")
            return None, []

    def _extract_momentum_features_balanced(self, data_df: pd.DataFrame) -> Tuple[Optional[np.ndarray], List[str]]:
        """Extract balanced momentum features using TAS-style bounded approach."""
        try:
            features = []
            names = []

            if 'close' in data_df.columns:
                close_price = data_df['close'].values

                # Momentum periods (balanced set)
                periods = [3, 7, 14]

                for period in periods:
                    if len(close_price) > period:
                        # Price momentum (TAS approach - bounded)
                        momentum = close_price / np.roll(close_price, period) - 1
                        momentum = np.clip(momentum, -1, 2)  # Bounded momentum

                        # Rate of change (percentage-based)
                        roc = pd.Series(close_price).pct_change(period).values
                        roc = np.clip(roc, -1, 2)  # Bounded ROC

                        # Fill NaN values
                        momentum = np.nan_to_num(momentum, nan=0.0)
                        roc = np.nan_to_num(roc, nan=0.0)

                        features.append(momentum.reshape(-1, 1))
                        features.append(roc.reshape(-1, 1))
                        names.extend([f'momentum_{period}', f'roc_{period}'])

                # Stochastic-like momentum (if high/low available)
                if 'high' in data_df.columns and 'low' in data_df.columns:
                    high_low = data_df['high'].values - data_df['low'].values
                    price_position = (close_price - data_df['low'].values) / (high_low + 1e-8)
                    price_position = np.clip(price_position, 0, 1)  # Bounded between 0 and 1

                    features.append(price_position.reshape(-1, 1))
                    names.append('stochastic_position')

            if features:
                return np.concatenate(features, axis=1), names
            else:
                return None, []

        except Exception as e:
            self.logger.warning(f"Momentum features extraction failed: {e}")
            return None, []

    def _extract_trend_features_balanced(self, data_df: pd.DataFrame) -> Tuple[Optional[np.ndarray], List[str]]:
        """Extract balanced trend features using TAS-style binary indicators."""
        try:
            features = []
            names = []

            if 'close' in data_df.columns:
                close_price = data_df['close'].values

                # Trend periods (balanced set)
                periods = [5, 10, 20]

                for period in periods:
                    if len(close_price) > period:
                        # Trend direction (TAS approach - binary)
                        trend_dir = np.where(close_price > np.roll(close_price, period), 1, -1)

                        # Trend strength (normalized)
                        trend_strength = np.abs(close_price - np.roll(close_price, period))
                        trend_strength = trend_strength / (close_price + 1e-8)  # Normalized
                        trend_strength = np.clip(trend_strength, 0, 1)  # Bounded

                        features.append(trend_dir.reshape(-1, 1))
                        features.append(trend_strength.reshape(-1, 1))
                        names.extend([f'trend_dir_{period}', f'trend_strength_{period}'])

                # Volume trend (if volume available)
                if 'volume' in data_df.columns:
                    volume = data_df['volume'].values
                    if len(volume) > 10:
                        # Volume trend direction
                        vol_trend = np.where(volume > np.roll(volume, 10), 1, -1)
                        features.append(vol_trend.reshape(-1, 1))
                        names.append('volume_trend')

            if features:
                return np.concatenate(features, axis=1), names
            else:
                return None, []

        except Exception as e:
            self.logger.warning(f"Trend features extraction failed: {e}")
            return None, []

    def _extract_technical_features_balanced(self, data_df: pd.DataFrame) -> Tuple[Optional[np.ndarray], List[str]]:
        """Extract balanced technical features."""
        try:
            features = []
            names = []

            if 'close' not in data_df.columns:
                raise FeatureCategoryError("Close price column not found for technical features")

            close_price = data_df['close'].values

            # Simple moving average ratios (bounded)
            periods = [5, 10, 20]
            for period in periods:
                if len(close_price) <= period:
                    continue

                try:
                    # Use numpy-based rolling mean for better performance
                    if self.config.use_numpy_optimization:
                        sma = self._numpy_rolling_mean(close_price, period)
                    else:
                        sma = pd.Series(close_price).rolling(window=period).mean().values

                    sma_ratio = close_price / (sma + 1e-8)
                    sma_ratio = np.clip(sma_ratio, 0.5, 2.0)  # Bounded
                    sma_ratio = np.nan_to_num(sma_ratio, nan=1.0)

                    features.append(sma_ratio.reshape(-1, 1))
                    names.append(f'sma_ratio_{period}')

                except Exception as e:
                    self.logger.warning(f"SMA period {period} failed: {e}")
                    continue

            # RSI-like momentum (bounded)
            if len(close_price) > 14:
                try:
                    delta = np.diff(close_price)
                    gains = np.where(delta > 0, delta, 0)
                    losses = np.where(delta < 0, -delta, 0)

                    # Use numpy-based rolling mean for RSI calculation
                    if self.config.use_numpy_optimization:
                        avg_gain = self._numpy_rolling_mean(gains, 14)
                        avg_loss = self._numpy_rolling_mean(losses, 14)
                    else:
                        avg_gain = pd.Series(gains).rolling(window=14).mean().values
                        avg_loss = pd.Series(losses).rolling(window=14).mean().values

                    rs = avg_gain / (avg_loss + 1e-8)
                    rsi = 100 - (100 / (1 + rs))
                    rsi = np.clip(rsi, 0, 100)  # Bounded
                    rsi = np.nan_to_num(rsi, nan=50)  # Neutral RSI

                    # Pad to match original length
                    rsi_padded = np.concatenate([[50], rsi])
                    features.append(rsi_padded.reshape(-1, 1))
                    names.append('rsi')

                except Exception as e:
                    self.logger.warning(f"RSI calculation failed: {e}")

            if features:
                return np.concatenate(features, axis=1), names
            else:
                return None, []

        except FeatureCategoryError:
            raise
        except Exception as e:
            self.logger.warning(f"Technical features extraction failed: {e}")
            return None, []

    def _extract_statistical_features_balanced(self, data_df: pd.DataFrame) -> Tuple[Optional[np.ndarray], List[str]]:
        """Extract balanced statistical features."""
        try:
            features = []
            names = []

            if 'close' not in data_df.columns:
                raise FeatureCategoryError("Close price column not found for statistical features")

            close_price = data_df['close'].values

            # Rolling statistics (bounded)
            periods = [10, 20]
            for period in periods:
                if len(close_price) <= period:
                    continue

                try:
                    # Use numpy-based rolling calculations for better performance
                    if self.config.use_numpy_optimization:
                        rolling_mean = self._numpy_rolling_mean(close_price, period)
                        rolling_std = self._numpy_rolling_std(close_price, period)
                    else:
                        rolling_mean = pd.Series(close_price).rolling(window=period).mean().values
                        rolling_std = pd.Series(close_price).rolling(window=period).std().values

                    # Rolling mean ratio
                    mean_ratio = close_price / (rolling_mean + 1e-8)
                    mean_ratio = np.clip(mean_ratio, 0.5, 2.0)
                    mean_ratio = np.nan_to_num(mean_ratio, nan=1.0)

                    # Rolling std (normalized)
                    std_normalized = rolling_std / (np.std(close_price) + 1e-8)
                    std_normalized = np.clip(std_normalized, 0, 3)
                    std_normalized = np.nan_to_num(std_normalized, nan=1.0)

                    features.extend([mean_ratio.reshape(-1, 1), std_normalized.reshape(-1, 1)])
                    names.extend([f'mean_ratio_{period}', f'std_normalized_{period}'])

                except Exception as e:
                    self.logger.warning(f"Statistical period {period} failed: {e}")
                    continue

            # Skewness and kurtosis (if enough data)
            if len(close_price) > 50:
                try:
                    from scipy import stats

                    # Use numpy-based rolling calculations for skewness and kurtosis
                    if self.config.use_numpy_optimization:
                        rolling_skew = self._numpy_rolling_skew(close_price, 20)
                        rolling_kurt = self._numpy_rolling_kurtosis(close_price, 20)
                    else:
                        rolling_skew = pd.Series(close_price).rolling(window=20).apply(
                            lambda x: stats.skew(x) if len(x) == 20 else np.nan
                        ).values
                        rolling_kurt = pd.Series(close_price).rolling(window=20).apply(
                            lambda x: stats.kurtosis(x) if len(x) == 20 else np.nan
                        ).values

                    # Bounded skewness and kurtosis
                    rolling_skew = np.clip(rolling_skew, -3, 3)
                    rolling_kurt = np.clip(rolling_kurt, -3, 10)
                    rolling_skew = np.nan_to_num(rolling_skew, nan=0.0)
                    rolling_kurt = np.nan_to_num(rolling_kurt, nan=0.0)

                    features.extend([rolling_skew.reshape(-1, 1), rolling_kurt.reshape(-1, 1)])
                    names.extend(['rolling_skewness', 'rolling_kurtosis'])

                except ImportError:
                    self.logger.warning("SciPy not available for skewness and kurtosis")
                except Exception as e:
                    self.logger.warning(f"Skewness/kurtosis calculation failed: {e}")

            if features:
                return np.concatenate(features, axis=1), names
            else:
                return None, []

        except FeatureCategoryError:
            raise
        except Exception as e:
            self.logger.warning(f"Statistical features extraction failed: {e}")
            return None, []

    def _extract_interaction_features_balanced(self, data_df: pd.DataFrame) -> Tuple[Optional[np.ndarray], List[str]]:
        """Extract balanced interaction features using unified feature generation."""
        try:
            features = []
            names = []

            # Use unified feature generation system if available
            if self.feature_bank is not None:
                try:
                    # Generate interaction features using FeatureBank
                    interaction_categories = ['interaction', 'cross_timeframe']
                    interaction_result = self.feature_bank.generate_features(
                        data=data_df,
                        categories=interaction_categories,
                        lookback_optimization=True
                    )

                    if interaction_result is not None and hasattr(interaction_result, 'features'):
                        # Limit interaction features to prevent imbalance
                        interaction_features = interaction_result.features.values
                        if interaction_features.shape[1] > self.config.max_features_per_category:
                            # Select most important features based on variance
                            feature_importance = np.var(interaction_features, axis=0)
                            top_indices = np.argsort(feature_importance)[-self.config.max_features_per_category:]
                            interaction_features = interaction_features[:, top_indices]
                            feature_names = [interaction_result.features.columns[i] for i in top_indices]
                        else:
                            feature_names = list(interaction_result.features.columns)

                        # Apply balanced scaling to interaction features
                        from sklearn.preprocessing import RobustScaler
                        scaler = RobustScaler()
                        interaction_features = scaler.fit_transform(interaction_features)
                        interaction_features = np.clip(interaction_features, -self.config.max_feature_range, self.config.max_feature_range)

                        features.append(interaction_features)
                        names.extend(feature_names)

                except Exception as e:
                    self.logger.warning(f"Unified interaction features failed: {e}")

            # Fallback: Simple interaction features
            if not features and 'close' in data_df.columns and 'volume' in data_df.columns:
                close_price = data_df['close'].values
                volume = data_df['volume'].values

                # Price-volume interaction (normalized)
                price_volume_interaction = close_price * volume
                price_volume_interaction = (price_volume_interaction - np.mean(price_volume_interaction)) / (np.std(price_volume_interaction) + 1e-8)
                price_volume_interaction = np.clip(price_volume_interaction, -3, 3)

                features.append(price_volume_interaction.reshape(-1, 1))
                names.append('price_volume_interaction')

            if features:
                return np.concatenate(features, axis=1), names
            else:
                return None, []

        except Exception as e:
            self.logger.warning(f"Interaction features extraction failed: {e}")
            return None, []

    def _apply_balanced_scaling(self, features: np.ndarray) -> np.ndarray:
        """Apply balanced scaling to prevent extreme values."""
        try:
            if self.config.feature_normalization_method == "robust":
                scaler = RobustScaler()
            elif self.config.feature_normalization_method == "standard":
                scaler = StandardScaler()
            else:
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()

            scaled_features = scaler.fit_transform(features)

            # Apply TAS-style bounds to prevent extreme values
            scaled_features = np.clip(scaled_features, -self.config.max_feature_range, self.config.max_feature_range)

            # Handle any remaining NaN or inf values
            scaled_features = np.nan_to_num(scaled_features, nan=0.0, posinf=1.0, neginf=-1.0)

            self.scaler = scaler
            return scaled_features

        except Exception as e:
            self.logger.warning(f"Balanced scaling failed: {e}")
            return features

    def _select_features_balanced(self, features: np.ndarray, feature_names: List[str],
                                labels: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Select features using balanced approach."""
        try:
            if features.shape[1] <= self.config.total_max_features:
                return features, feature_names

            # Use variance-based selection for balance
            feature_variance = np.var(features, axis=0)

            # Select features with good variance (not too low, not too high)
            # This helps prevent both sparse and overly concentrated features
            variance_threshold_low = np.percentile(feature_variance, 10)
            variance_threshold_high = np.percentile(feature_variance, 90)

            balanced_mask = (feature_variance >= variance_threshold_low) & (feature_variance <= variance_threshold_high)

            if np.sum(balanced_mask) < self.config.total_max_features:
                # Fallback to top variance features
                top_indices = np.argsort(feature_variance)[-self.config.total_max_features:]
                balanced_mask = np.zeros(len(feature_names), dtype=bool)
                balanced_mask[top_indices] = True

            selected_features = features[:, balanced_mask]
            selected_names = [name for i, name in enumerate(feature_names) if balanced_mask[i]]

            return selected_features, selected_names

        except Exception as e:
            self.logger.warning(f"Feature selection failed: {e}")
            return features, feature_names

    def _extract_temporal_features_balanced(self, data_df: pd.DataFrame) -> Tuple[Optional[np.ndarray], List[str]]:
        """Extract temporal features for regime detection."""
        try:
            features = []
            names = []

            if 'close' not in data_df.columns:
                return None, []

            close_price = data_df['close'].values

            # Time-based features
            if self.config.enable_temporal_features:
                # Hour of day effect (if we have timestamps)
                if hasattr(data_df, 'index') and hasattr(data_df.index, 'hour'):
                    hour_values = np.array(data_df.index.hour)
                    hour_features = np.sin(2 * np.pi * hour_values / 24)
                    features.append(hour_features.reshape(-1, 1))
                    names.append('hour_sin')

                    hour_features = np.cos(2 * np.pi * hour_values / 24)
                    features.append(hour_features.reshape(-1, 1))
                    names.append('hour_cos')

                # Day of week effect
                if hasattr(data_df, 'index') and hasattr(data_df.index, 'dayofweek'):
                    day_values = np.array(data_df.index.dayofweek)
                    day_features = np.sin(2 * np.pi * day_values / 7)
                    features.append(day_features.reshape(-1, 1))
                    names.append('day_sin')

                    day_features = np.cos(2 * np.pi * day_values / 7)
                    features.append(day_features.reshape(-1, 1))
                    names.append('day_cos')

                # Time-based volatility patterns
                for period in [5, 10, 20]:
                    if len(close_price) > period:
                        # Rolling volatility with time weighting
                        returns = np.diff(close_price) / (close_price[:-1] + 1e-8)
                        time_weights = np.exp(-np.arange(len(returns)) * 0.01)  # Exponential decay
                        weighted_vol = np.full(len(close_price), 0.0)  # Initialize with correct length

                        for i in range(period, len(returns)):
                            window_returns = returns[i-period:i]
                            window_weights = time_weights[i-period:i]
                            weighted_std = np.sqrt(np.average(window_returns**2, weights=window_weights))
                            weighted_vol[i+1] = weighted_std  # +1 because returns[i] corresponds to close_price[i+1]

                        features.append(weighted_vol.reshape(-1, 1))
                        names.append(f'time_weighted_vol_{period}')

            if features:
                return np.concatenate(features, axis=1), names
            else:
                return None, []

        except Exception as e:
            self.logger.warning(f"Temporal features extraction failed: {e}")
            return None, []

    def _extract_micro_regime_features_balanced(self, data_df: pd.DataFrame) -> Tuple[Optional[np.ndarray], List[str]]:
        """Extract micro-regime features for short-term regime detection."""
        try:
            features = []
            names = []

            if 'close' not in data_df.columns:
                return None, []

            close_price = data_df['close'].values

            if self.config.enable_micro_regime_features:
                # Short-term volatility changes (micro-regime indicators)
                for short_period in [2, 3, 5]:
                    if len(close_price) > short_period:
                        # Rolling volatility
                        returns = np.diff(close_price) / (close_price[:-1] + 1e-8)
                        rolling_vol = np.full(len(close_price), 0.0)  # Initialize with correct length

                        for i in range(short_period, len(returns)):
                            vol = np.std(returns[i-short_period:i])
                            rolling_vol[i+1] = vol  # +1 because returns[i] corresponds to close_price[i+1]

                        # Volatility change rate (micro-regime sensitivity)
                        vol_change = np.full(len(close_price), 0.0)  # Initialize with correct length
                        vol_change[1:] = np.diff(rolling_vol)  # Fill from index 1 onwards

                        # Normalize and bound
                        vol_change = np.clip(vol_change, -self.config.max_feature_range, self.config.max_feature_range)

                        features.append(vol_change.reshape(-1, 1))
                        names.append(f'micro_vol_change_{short_period}')

                        # Volatility acceleration (second derivative)
                        if len(vol_change) > 2:
                            vol_acceleration = np.full(len(close_price), 0.0)  # Initialize with correct length
                            vol_acceleration[2:] = np.diff(vol_change[1:])  # Fill from index 2 onwards
                            vol_acceleration = np.clip(vol_acceleration, -self.config.max_feature_range, self.config.max_feature_range)

                            features.append(vol_acceleration.reshape(-1, 1))
                            names.append(f'micro_vol_acceleration_{short_period}')

                # Price momentum micro-features
                for short_period in [2, 3, 5]:
                    if len(close_price) > short_period:
                        # Short-term momentum
                        momentum = np.full(len(close_price), 0.0)  # Initialize with correct length
                        momentum_values = (close_price[short_period:] - close_price[:-short_period]) / (close_price[:-short_period] + 1e-8)
                        momentum[short_period:] = momentum_values  # Fill from short_period onwards
                        momentum = np.clip(momentum, -self.config.max_feature_range, self.config.max_feature_range)

                        features.append(momentum.reshape(-1, 1))
                        names.append(f'micro_momentum_{short_period}')

                        # Momentum change rate
                        if len(momentum) > 1:
                            momentum_change = np.full(len(close_price), 0.0)  # Initialize with correct length
                            momentum_change[1:] = np.diff(momentum)  # Fill from index 1 onwards
                            momentum_change = np.clip(momentum_change, -self.config.max_feature_range, self.config.max_feature_range)

                            features.append(momentum_change.reshape(-1, 1))
                            names.append(f'micro_momentum_change_{short_period}')

            if features:
                return np.concatenate(features, axis=1), names
            else:
                return None, []

        except Exception as e:
            self.logger.warning(f"Micro-regime features extraction failed: {e}")
            return None, []

    def _analyze_regime_stability(self, regime_assignments: np.ndarray) -> Dict[str, Any]:
        """Analyze regime stability and persistence."""
        try:
            stability_metrics = {}

            # Regime persistence analysis
            regime_changes = np.diff(regime_assignments) != 0
            change_points = np.where(regime_changes)[0]

            if len(change_points) > 0:
                # Calculate regime durations
                regime_durations = np.diff(np.concatenate([[0], change_points, [len(regime_assignments)]]))

                stability_metrics['total_regime_changes'] = len(change_points)
                stability_metrics['avg_regime_duration'] = np.mean(regime_durations)
                stability_metrics['min_regime_duration'] = np.min(regime_durations)
                stability_metrics['max_regime_duration'] = np.max(regime_durations)
                stability_metrics['regime_stability_score'] = 1.0 / (1.0 + len(change_points) / len(regime_assignments))
            else:
                stability_metrics['total_regime_changes'] = 0
                stability_metrics['avg_regime_duration'] = len(regime_assignments)
                stability_metrics['min_regime_duration'] = len(regime_assignments)
                stability_metrics['max_regime_duration'] = len(regime_assignments)
                stability_metrics['regime_stability_score'] = 1.0

            # Regime distribution analysis
            unique_regimes, regime_counts = np.unique(regime_assignments, return_counts=True)
            regime_distribution = dict(zip(unique_regimes, regime_counts))

            stability_metrics['regime_distribution'] = regime_distribution
            stability_metrics['regime_balance'] = 1.0 - (np.std(list(regime_counts.values())) / np.mean(list(regime_counts.values())))

            return stability_metrics

        except Exception as e:
            self.logger.warning(f"Regime stability analysis failed: {e}")
            return {}

    def _calculate_balance_metrics(self, features: np.ndarray) -> Dict[str, float]:
        """Calculate metrics to assess feature balance."""
        try:
            metrics = {}

            # Feature variance distribution
            feature_variance = np.var(features, axis=0)
            metrics['variance_mean'] = np.mean(feature_variance)
            metrics['variance_std'] = np.std(feature_variance)
            metrics['variance_cv'] = metrics['variance_std'] / (metrics['variance_mean'] + 1e-8)

            # Feature range distribution
            feature_ranges = np.max(features, axis=0) - np.min(features, axis=0)
            metrics['range_mean'] = np.mean(feature_ranges)
            metrics['range_std'] = np.std(feature_ranges)

            # Outlier percentage (features with extreme values)
            outlier_threshold = self.config.max_feature_range
            outlier_count = np.sum(np.abs(features) > outlier_threshold)
            metrics['outlier_percentage'] = outlier_count / features.size * 100

            # Feature correlation (to assess redundancy)
            try:
                correlation_matrix = np.corrcoef(features.T)
                # Remove diagonal
                mask = ~np.eye(correlation_matrix.shape[0], dtype=bool)
                correlations = correlation_matrix[mask]
                metrics['avg_correlation'] = np.mean(np.abs(correlations))
                metrics['high_correlation_percentage'] = np.sum(np.abs(correlations) > 0.8) / len(correlations) * 100
            except:
                metrics['avg_correlation'] = 0.0
                metrics['high_correlation_percentage'] = 0.0

            return metrics

        except Exception as e:
            self.logger.warning(f"Balance metrics calculation failed: {e}")
            return {}

    def analyze_feature_importance_for_regimes(self, features: np.ndarray,
                                             feature_names: List[str],
                                             regime_labels: np.ndarray,
                                             method: str = "mutual_information") -> Dict[str, Any]:
        """
        Analyze feature importance for regime discovery and characterization.

        This method is crucial for regime discovery because it helps:
        1. Identify which features are most discriminative between regimes
        2. Understand what market characteristics define each regime
        3. Validate clustering quality based on meaningful features
        4. Provide interpretability for regime assignments
        5. Guide feature selection for improved clustering

        Args:
            features: Feature matrix (n_samples, n_features)
            feature_names: List of feature names
            regime_labels: Regime/cluster labels for each sample
            method: Importance calculation method ('mutual_information', 'f_classif', 'variance')

        Returns:
            Dictionary containing importance scores, rankings, and regime analysis
        """
        tprint("🔍 Analyzing feature importance for regime discovery", color="blue")

        try:
            n_regimes = len(np.unique(regime_labels))
            n_features = features.shape[1]

            importance_scores = np.zeros(n_features)
            regime_feature_profiles = {}

            if method == "mutual_information":
                importance_scores = self._calculate_mutual_information_importance(features, regime_labels)
            elif method == "f_classif" and SKLEARN_AVAILABLE:
                importance_scores = self._calculate_anova_importance(features, regime_labels)
            else:
                # Fallback to variance-based importance
                importance_scores = self._calculate_variance_importance(features, regime_labels)

            # Create feature importance ranking
            feature_importance = list(zip(feature_names, importance_scores))
            feature_importance.sort(key=lambda x: x[1], reverse=True)

            # Calculate regime-specific feature profiles
            for regime_id in np.unique(regime_labels):
                regime_mask = regime_labels == regime_id
                regime_features = features[regime_mask]

                regime_profile = {
                    'mean_features': np.mean(regime_features, axis=0),
                    'std_features': np.std(regime_features, axis=0),
                    'dominant_features': self._get_dominant_features(regime_features, feature_names),
                    'sample_count': np.sum(regime_mask),
                    'feature_variance': np.var(regime_features, axis=0)
                }
                regime_feature_profiles[f"regime_{regime_id}"] = regime_profile

            # Calculate regime separability metrics
            separability_metrics = self._calculate_regime_separability(features, regime_labels)

            result = {
                'feature_importance_ranking': feature_importance,
                'regime_feature_profiles': regime_feature_profiles,
                'regime_separability': separability_metrics,
                'method_used': method,
                'n_regimes': n_regimes,
                'most_important_features': [name for name, _ in feature_importance[:10]],
                'least_important_features': [name for name, _ in feature_importance[-10:]],
                'interpretation': self._generate_regime_interpretation(feature_importance, regime_feature_profiles)
            }

            tprint(f"✅ Feature importance analysis completed for {n_regimes} regimes", color="green")
            return result

        except Exception as e:
            self.logger.error(f"Feature importance analysis failed: {e}")
            tprint(f"❌ Feature importance analysis failed: {e}", color="red")
            return {}

    def _calculate_mutual_information_importance(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Calculate feature importance using mutual information."""
        try:
            from sklearn.feature_selection import mutual_info_classif
            return mutual_info_classif(features, labels)
        except ImportError:
            # Fallback to simplified mutual information calculation
            return self._calculate_variance_importance(features, labels)

    def _calculate_anova_importance(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Calculate feature importance using ANOVA F-statistic."""
        try:
            from sklearn.feature_selection import f_classif
            f_scores, _ = f_classif(features, labels)
            return f_scores
        except ImportError:
            return self._calculate_variance_importance(features, labels)

    def _calculate_variance_importance(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Calculate feature importance based on between-class variance."""
        try:
            unique_labels = np.unique(labels)
            between_class_variance = np.zeros(features.shape[1])

            overall_mean = np.mean(features, axis=0)

            for label in unique_labels:
                mask = labels == label
                class_mean = np.mean(features[mask], axis=0)
                class_size = np.sum(mask)

                # Weighted variance contribution
                between_class_variance += class_size * np.sum((class_mean - overall_mean) ** 2)

            return between_class_variance / len(unique_labels)
        except Exception as e:
            self.logger.warning(f"Variance importance calculation failed: {e}")
            return np.var(features, axis=0)

    def _get_dominant_features(self, regime_features: np.ndarray, feature_names: List[str],
                              top_k: int = 5) -> List[str]:
        """Get the most distinctive features for a specific regime."""
        try:
            # Features with highest absolute deviation from overall mean
            feature_means = np.mean(regime_features, axis=0)
            feature_stds = np.std(regime_features, axis=0)

            # Z-score of feature means (how distinctive they are)
            z_scores = np.abs(feature_means) / (feature_stds + 1e-8)

            # Get top distinctive features
            top_indices = np.argsort(z_scores)[-top_k:]
            return [feature_names[i] for i in top_indices]
        except Exception:
            return feature_names[:top_k]

    def _calculate_regime_separability(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate metrics to assess how well-separated the regimes are."""
        try:
            metrics = {}

            # Silhouette score (if sklearn available)
            if SKLEARN_AVAILABLE:
                try:
                    from sklearn.metrics import silhouette_score
                    silhouette = silhouette_score(features, labels)
                    metrics['silhouette_score'] = silhouette
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not calculate silhouette score: {e}")
                    metrics['silhouette_score'] = 0.0

            # Between vs within class variance ratio
            unique_labels = np.unique(labels)
            total_variance = np.var(features, axis=0)

            between_variance = np.zeros(features.shape[1])
            within_variance = np.zeros(features.shape[1])

            for label in unique_labels:
                mask = labels == label
                class_features = features[mask]
                class_mean = np.mean(class_features, axis=0)

                between_variance += np.sum((class_mean - np.mean(features, axis=0)) ** 2)
                within_variance += np.var(class_features, axis=0)

            # Average separability ratio
            separability_ratio = np.mean(between_variance / (within_variance + 1e-8))
            metrics['separability_ratio'] = separability_ratio

            return metrics
        except Exception as e:
            self.logger.warning(f"Regime separability calculation failed: {e}")
            return {}

    def _generate_regime_interpretation(self, feature_importance: List[Tuple[str, float]],
                                      regime_profiles: Dict[str, Dict[str, Any]]) -> str:
        """Generate human-readable interpretation of regime characteristics."""
        try:
            interpretation_parts = []

            # Most important features overall with scores
            top_features_with_scores = []
            for name, score in feature_importance[:5]:
                top_features_with_scores.append(f"{name}({score:.3f})")
            interpretation_parts.append(f"Key regime discriminators: {', '.join(top_features_with_scores)}")

            # Regime-specific characteristics with detailed explanations
            for regime_name, profile in regime_profiles.items():
                dominant = profile.get('dominant_features', [])[:3]
                if dominant:
                    # Get importance scores for these features to provide context
                    feature_scores = []
                    for feature_name in dominant:
                        # Find the score for this feature in the global ranking
                        for name, score in feature_importance:
                            if name == feature_name:
                                feature_scores.append(f"{feature_name}({score:.3f})")
                                break

                    if feature_scores:
                        interpretation_parts.append(f"{regime_name}: most characteristic features are {', '.join(feature_scores)}")
                    else:
                        interpretation_parts.append(f"{regime_name}: dominated by {', '.join(dominant)}")

            return " | ".join(interpretation_parts)
        except Exception:
            return "Feature importance analysis completed"

# Convenience functions for easy integration
def extract_balanced_features(data: Union[np.ndarray, pd.DataFrame],
                            config: Optional[BalancedFeatureConfig] = None,
                            labels: Optional[np.ndarray] = None) -> BalancedFeatureResult:
    """
    Convenience function to extract balanced features.

    Args:
        data: Input data
        config: Optional configuration
        labels: Optional labels for supervised feature selection

    Returns:
        BalancedFeatureResult with extracted features
    """
    extractor = BalancedFeatureExtractor(config)
    return extractor.extract_balanced_features(data, labels)

def create_unified_config() -> BalancedFeatureConfig:
    """Create unified configuration for both NAS and TAS to ensure identical features."""
    return BalancedFeatureConfig(
        enabled_categories=[
            FeatureCategory.PRICE, FeatureCategory.VOLUME, FeatureCategory.VOLATILITY,
            FeatureCategory.MOMENTUM, FeatureCategory.TREND, FeatureCategory.TECHNICAL,
            FeatureCategory.REGIME
        ],
        use_tas_style_extraction=True,
        use_balanced_scaling=True,
        max_feature_range=3.0,  # Unified bounds
        total_max_features=100,  # Sufficient for both systems
        enable_feature_selection=True,
        enable_pid_features=True,  # Use unified feature generation
        enable_hybrid_features=True,
        use_numpy_optimization=True,  # Enhanced performance
        enable_temporal_features=True,  # Add temporal awareness
        enable_micro_regime_features=True  # Add micro-regime detection
    )

def create_nas_config() -> BalancedFeatureConfig:
    """Create configuration for NAS clustering - now uses unified config."""
    return create_unified_config()

def create_tas_config() -> BalancedFeatureConfig:
    """Create configuration for TAS regime detection - now uses unified config."""
    return create_unified_config()

def analyze_regime_feature_importance(features: np.ndarray,
                                    feature_names: List[str],
                                    regime_labels: np.ndarray,
                                    method: str = "mutual_information",
                                    config: Optional[BalancedFeatureConfig] = None) -> Dict[str, Any]:
    """
    Convenience function to analyze feature importance for regime discovery.

    This function demonstrates why feature importance analysis is crucial for regime discovery:
    1. Identifies the most discriminative features between regimes
    2. Provides insights into market characteristics defining each regime
    3. Validates clustering quality based on meaningful features
    4. Enables interpretability of regime assignments
    5. Guides future feature engineering efforts

    Args:
        features: Feature matrix (n_samples, n_features)
        feature_names: List of feature names corresponding to features
        regime_labels: Regime/cluster labels for each sample
        method: Importance calculation method ('mutual_information', 'f_classif', 'variance')
        config: Optional configuration (mainly for logging)

    Returns:
        Dictionary containing comprehensive regime analysis results
    """
    extractor = BalancedFeatureExtractor(config)
    return extractor.analyze_feature_importance_for_regimes(
        features, feature_names, regime_labels, method
    )

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
