"""
Normalization Feature Generators

This module provides feature generators for data normalization and scaling operations,
leveraging the comprehensive scaling infrastructure from features_common.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

# Import from feature_generation core
from src.feature_generation.core.feature_generator import FeatureGenerator, FeatureCategory, FeatureConfig, FeatureResult

# Import the comprehensive scaling infrastructure from features_common
try:
    from .transforms.scaling_normalization import ScalingNormalizer
    from .transforms.base_scaler import BaseScaler, create_optimized_scaler
    SCALING_AVAILABLE = True
except ImportError as e:
    SCALING_AVAILABLE = False
    ScalingNormalizer = None
    BaseScaler = None
    create_optimized_scaler = None

try:
    from src.utils.tprint import tprint
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)

logger = logging.getLogger(__name__)

@dataclass
class NormalizationConfig(FeatureConfig):
    """Configuration for normalization feature generators."""
    method: str = "zscore"  # "zscore", "minmax", "robust", "quantile"
    rolling_window: int = 20
    min_periods: int = 10
    exclude_outliers: bool = True
    outlier_threshold: float = 3.0
    use_features_common: bool = True  # Use features_common infrastructure

class NormalizationFeatureGenerator(FeatureGenerator):
    """
    Base class for normalization feature generators using features_common infrastructure.

    Leverages the comprehensive ScalingNormalizer from features_common for robust
    normalization operations with VectorBT optimization when available.
    """

    def __init__(self, config: Optional[NormalizationConfig] = None):
        """
        Initialize the normalization feature generator.

        Args:
            config: Normalization configuration
        """
        if config is None:
            config = NormalizationConfig(
                name="normalization",
                category=FeatureCategory.NORMALIZATION,
                description="Data normalization features using features_common infrastructure",
                required_columns=["close"],
                default_lookback=20
            )

        super().__init__(config)
        self.normalization_config = config

        # Initialize features_common scaler if available
        self.scaler = None
        if config.use_features_common and SCALING_AVAILABLE:
            try:
                scaler_config = {
                    'default_strategy': config.method,
                    'auto_select': True,
                    'handle_outliers': config.exclude_outliers,
                    'outlier_threshold': config.outlier_threshold,
                    'use_vectorbt': True  # Enable VectorBT optimization
                }
                self.scaler = ScalingNormalizer(scaler_config)
                if TPRINT_AVAILABLE:
                    tprint("✅ Initialized with features_common ScalingNormalizer")
            except Exception as e:
                logger.warning(f"Failed to initialize ScalingNormalizer: {e}")
                self.scaler = None
        else:
            if TPRINT_AVAILABLE:
                tprint("⚠️ features_common not available, using basic normalization")

    def generate(self, data: pd.DataFrame, **kwargs) -> FeatureResult:
        """
        Generate normalization features using features_common infrastructure.

        Args:
            data: Input data DataFrame
            **kwargs: Additional parameters

        Returns:
            Feature result with normalized data
        """
        start_time = pd.Timestamp.now()

        try:
            # Validate input data
            if data.empty:
                return FeatureResult(
                    name=self.config.name,
                    data=pd.Series(dtype=float, index=data.index),
                    config=self.config,
                    computation_time=0.0,
                    success=False,
                    error_message="Empty data provided"
                )

            # Use features_common scaler if available
            if self.scaler is not None:
                normalized_data = self._apply_features_common_normalization(data, **kwargs)
            else:
                # Fallback to basic normalization
                normalized_data = self._apply_basic_normalization(data, **kwargs)

            computation_time = (pd.Timestamp.now() - start_time).total_seconds()

            return FeatureResult(
                name=self.config.name,
                data=normalized_data,
                config=self.config,
                computation_time=computation_time,
                success=True,
                metadata={
                    'normalization_method': self.normalization_config.method,
                    'rolling_window': self.normalization_config.rolling_window,
                    'using_features_common': self.scaler is not None,
                    'original_shape': data.shape,
                    'normalized_shape': normalized_data.shape if isinstance(normalized_data, pd.DataFrame) else (len(normalized_data), 1)
                }
            )

        except Exception as e:
            computation_time = (pd.Timestamp.now() - start_time).total_seconds()
            logger.error(f"Error in normalization generation: {e}")

            return FeatureResult(
                name=self.config.name,
                data=pd.Series(dtype=float, index=data.index),
                config=self.config,
                computation_time=computation_time,
                success=False,
                error_message=str(e)
            )

    def _apply_features_common_normalization(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Apply normalization using features_common ScalingNormalizer.

        Args:
            data: Input data DataFrame
            **kwargs: Additional parameters

        Returns:
            Normalized data series
        """
        if self.scaler is None:
            raise ValueError("ScalingNormalizer not initialized")

        # Get the primary column for normalization
        primary_column = self.config.required_columns[0] if self.config.required_columns else 'close'

        if primary_column not in data.columns:
            # Fallback to first numeric column
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                primary_column = numeric_cols[0]
            else:
                raise ValueError("No numeric columns found for normalization")

        # Create a single-column DataFrame for the scaler
        feature_data = data[[primary_column]].copy()

        # Apply normalization using features_common
        try:
            normalized_df = self.scaler.fit_transform(
                feature_data,
                strategy=kwargs.get('method', self.normalization_config.method),
                feature_list=[primary_column]
            )

            result = normalized_df[primary_column]

            if TPRINT_AVAILABLE:
                tprint(f"✅ Applied {self.normalization_config.method} normalization using features_common")

            return result

        except Exception as e:
            logger.warning(f"features_common normalization failed: {e}, using fallback")
            return self._apply_basic_normalization(data, **kwargs)

    def _apply_basic_normalization(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Apply basic normalization as fallback.

        Args:
            data: Input data DataFrame
            **kwargs: Additional parameters

        Returns:
            Normalized data series
        """
        method = kwargs.get('method', self.normalization_config.method)
        window = kwargs.get('rolling_window', self.normalization_config.rolling_window)
        min_periods = kwargs.get('min_periods', self.normalization_config.min_periods)

        # Get the primary column for normalization
        primary_column = self.config.required_columns[0] if self.config.required_columns else 'close'

        if primary_column not in data.columns:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                primary_column = numeric_cols[0]
            else:
                raise ValueError("No numeric columns found for normalization")

        series = data[primary_column]

        if method == "zscore":
            return self._zscore_normalization(series, window, min_periods)
        elif method == "minmax":
            return self._minmax_normalization(series, window, min_periods)
        elif method == "robust":
            return self._robust_normalization(series, window, min_periods)
        else:
            # Default to z-score
            return self._zscore_normalization(series, window, min_periods)

    def _zscore_normalization(self, series: pd.Series, window: int, min_periods: int) -> pd.Series:
        """Apply rolling z-score normalization."""
        rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
        rolling_std = series.rolling(window=window, min_periods=min_periods).std()

        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)

        zscore = (series - rolling_mean) / rolling_std

        # Handle outliers if enabled
        if self.normalization_config.exclude_outliers:
            threshold = self.normalization_config.outlier_threshold
            zscore = zscore.clip(-threshold, threshold)

        return zscore

    def _minmax_normalization(self, series: pd.Series, window: int, min_periods: int) -> pd.Series:
        """Apply rolling min-max normalization."""
        rolling_min = series.rolling(window=window, min_periods=min_periods).min()
        rolling_max = series.rolling(window=window, min_periods=min_periods).max()

        # Avoid division by zero
        range_vals = rolling_max - rolling_min
        range_vals = range_vals.replace(0, np.nan)

        return (series - rolling_min) / range_vals

    def _robust_normalization(self, series: pd.Series, window: int, min_periods: int) -> pd.Series:
        """Apply rolling robust normalization using median and MAD."""
        rolling_median = series.rolling(window=window, min_periods=min_periods).median()
        rolling_mad = series.rolling(window=window, min_periods=min_periods).apply(
            lambda x: np.median(np.abs(x - np.median(x)))
        )

        # Avoid division by zero
        rolling_mad = rolling_mad.replace(0, np.nan)

        return (series - rolling_median) / rolling_mad

class RollingZScoreGenerator(NormalizationFeatureGenerator):
    """
    Rolling Z-Score normalization feature generator using features_common infrastructure.

    Computes rolling z-score normalization using a specified window size with
    VectorBT optimization when available.
    """

    def __init__(self, rolling_window: int = 20, **kwargs):
        """
        Initialize the rolling z-score generator.

        Args:
            rolling_window: Window size for rolling calculations
            **kwargs: Additional parameters
        """
        config = NormalizationConfig(
            name="rolling_zscore",
            category=FeatureCategory.NORMALIZATION,
            description=f"Rolling z-score normalization (window={rolling_window}) using features_common",
            required_columns=["close"],
            default_lookback=rolling_window,
            method="zscore",
            rolling_window=rolling_window,
            use_features_common=True
        )

        super().__init__(config)

class VolatilityScalingGenerator(NormalizationFeatureGenerator):
    """
    Volatility scaling normalization feature generator.

    Scales features by their rolling volatility to achieve unit volatility.
    """

    def __init__(self, rolling_window: int = 20, **kwargs):
        """
        Initialize the volatility scaling generator.

        Args:
            rolling_window: Window size for volatility calculation
            **kwargs: Additional parameters
        """
        config = NormalizationConfig(
            name="volatility_scaling",
            category=FeatureCategory.NORMALIZATION,
            description=f"Volatility scaling normalization (window={rolling_window})",
            required_columns=["close"],
            default_lookback=rolling_window,
            method="zscore",  # Use z-score as base, but scale by volatility
            rolling_window=rolling_window,
            use_features_common=True
        )

        super().__init__(config)

    def _apply_features_common_normalization(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Apply volatility scaling normalization using features_common.

        Args:
            data: Input data DataFrame
            **kwargs: Additional parameters

        Returns:
            Volatility-scaled data series
        """
        primary_column = self.config.required_columns[0] if self.config.required_columns else 'close'

        if primary_column not in data.columns:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                primary_column = numeric_cols[0]
            else:
                raise ValueError("No numeric columns found for normalization")

        series = data[primary_column]
        window = kwargs.get('rolling_window', self.normalization_config.rolling_window)
        min_periods = kwargs.get('min_periods', self.normalization_config.min_periods)

        # Calculate returns
        returns = series.pct_change()

        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=window, min_periods=min_periods).std()

        # Scale returns by volatility
        vol_scaled = returns / rolling_vol

        # Handle infinite values
        vol_scaled = vol_scaled.replace([np.inf, -np.inf], np.nan)

        return vol_scaled

class CrossSectionalNormalizer(NormalizationFeatureGenerator):
    """
    Cross-sectional normalization feature generator.

    Normalizes features across assets/symbols at each time point using
    features_common infrastructure when available.
    """

    def __init__(self, **kwargs):
        """
        Initialize the cross-sectional normalizer.

        Args:
            **kwargs: Additional parameters
        """
        config = NormalizationConfig(
            name="cross_sectional_normalizer",
            category=FeatureCategory.NORMALIZATION,
            description="Cross-sectional normalization across assets using features_common",
            required_columns=["close"],
            default_lookback=1,
            method="zscore",
            use_features_common=True
        )

        super().__init__(config)

    def _apply_features_common_normalization(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Apply cross-sectional normalization using features_common.

        Args:
            data: Input data DataFrame
            **kwargs: Additional parameters

        Returns:
            Cross-sectionally normalized data series
        """
        method = kwargs.get('method', self.normalization_config.method)

        # For cross-sectional normalization, we need multiple assets
        # For now, we'll apply it to the primary column
        primary_column = self.config.required_columns[0] if self.config.required_columns else 'close'

        if primary_column not in data.columns:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                primary_column = numeric_cols[0]
            else:
                raise ValueError("No numeric columns found for normalization")

        series = data[primary_column]

        if method == "zscore":
            # Cross-sectional z-score (normalize by current cross-section)
            cross_sectional_mean = series.mean()
            cross_sectional_std = series.std()

            if cross_sectional_std == 0:
                return pd.Series(0, index=series.index)

            return (series - cross_sectional_mean) / cross_sectional_std

        elif method == "minmax":
            # Cross-sectional min-max normalization
            cross_sectional_min = series.min()
            cross_sectional_max = series.max()

            if cross_sectional_max == cross_sectional_min:
                return pd.Series(0, index=series.index)

            return (series - cross_sectional_min) / (cross_sectional_max - cross_sectional_min)

        else:
            # Default to z-score
            cross_sectional_mean = series.mean()
            cross_sectional_std = series.std()

            if cross_sectional_std == 0:
                return pd.Series(0, index=series.index)

            return (series - cross_sectional_mean) / cross_sectional_std

def create_data_normalizer(method: str = "zscore",
                          rolling_window: int = 20,
                          use_features_common: bool = True,
                          **kwargs) -> NormalizationFeatureGenerator:
    """
    Create a data normalizer with specified parameters using features_common infrastructure.

    Args:
        method: Normalization method ("zscore", "minmax", "robust", "quantile")
        rolling_window: Rolling window size for normalization
        use_features_common: Whether to use features_common infrastructure
        **kwargs: Additional parameters

    Returns:
        Configured normalization feature generator
    """
    if method == "zscore":
        return RollingZScoreGenerator(rolling_window=rolling_window, **kwargs)
    elif method == "volatility_scaling":
        return VolatilityScalingGenerator(rolling_window=rolling_window, **kwargs)
    elif method == "cross_sectional":
        return CrossSectionalNormalizer(**kwargs)
    else:
        # Default to base normalization generator
        config = NormalizationConfig(
            name=f"normalization_{method}",
            category=FeatureCategory.NORMALIZATION,
            description=f"Data normalization using {method} method with features_common",
            required_columns=["close"],
            default_lookback=rolling_window,
            method=method,
            rolling_window=rolling_window,
            use_features_common=use_features_common
        )
        return NormalizationFeatureGenerator(config)

def create_default_normalization_generators() -> List[NormalizationFeatureGenerator]:
    """
    Create default normalization feature generators using features_common infrastructure.

    Returns:
        List of default normalization generators
    """
    generators = []

    try:
        # Rolling z-score generators with different windows
        for window in [10, 20, 50]:
            generator = RollingZScoreGenerator(rolling_window=window)
            generators.append(generator)

        # Volatility scaling generator
        vol_scaler = VolatilityScalingGenerator(rolling_window=20)
        generators.append(vol_scaler)

        # Cross-sectional normalizer
        cross_sectional = CrossSectionalNormalizer()
        generators.append(cross_sectional)

        if TPRINT_AVAILABLE:
            tprint(f"✅ Created {len(generators)} normalization generators using features_common infrastructure")

    except Exception as e:
        if TPRINT_AVAILABLE:
            tprint(f"⚠️ Failed to create normalization generators: {e}")
        logger.warning(f"Failed to create normalization generators: {e}")

    return generators

# Export the main classes and functions
__all__ = [
    'NormalizationFeatureGenerator',
    'RollingZScoreGenerator',
    'VolatilityScalingGenerator',
    'CrossSectionalNormalizer',
    'create_data_normalizer',
    'create_default_normalization_generators',
    'NormalizationConfig'
]
