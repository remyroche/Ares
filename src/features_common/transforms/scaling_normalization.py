"""
Scaling and Normalization for Unified Data-Driven Pipeline.

This module provides comprehensive scaling and normalization capabilities
for consistent feature preprocessing across the pipeline with VectorBT integration.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler,
    QuantileTransformer, PowerTransformer, Normalizer
)
from sklearn.compose import ColumnTransformer
import logging

from src.utils.feature_common.atr_normalization import (
    calculate_atr,
    should_use_atr_normalization,
)

try:
    from src.utils.feature_common.volume_transforms import log1p_zscore_normalize
    VOLUME_TRANSFORMS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    log1p_zscore_normalize = None  # type: ignore[assignment]
    VOLUME_TRANSFORMS_AVAILABLE = False

# Import VectorBT components
try:
    from src.features_common.vectorbt_extensions.unified_manager import UnifiedVectorizationManager
    from src.features_common.transforms.vectorbt_scaler import VectorBTScaler
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False

try:
    from src.utils.tprint import tprint, tprint_info, tprint_error, tprint_warning, tprint_success, tprint_debug
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

# Functional helpers are re-exported for legacy consumers that expect
# lightweight normalization utilities without instantiating the full
# ScalingNormalizer class.
__all__ = [
    "ScalingNormalizer",
    "zscore_normalize",
    "winsorized_zscore_normalize",
    "robust_normalize",
    "rank_normalize",
    "rolling_zscore_normalize",
    "rolling_winsorized_zscore_normalize",
    "rolling_minmax_normalize",
    "rolling_adaptive_normalize",
]


def _normalize_input(data: Union[pd.DataFrame, pd.Series]) -> Tuple[pd.DataFrame, bool]:
    """Convert input to DataFrame and track whether it was originally a Series."""
    if isinstance(data, pd.Series):
        return data.to_frame(), True
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas Series or DataFrame")
    return data, False


def _restore_output(result: pd.DataFrame, was_series: bool, original: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
    """Convert back to Series if input was Series, preserving name/index."""
    if was_series:
        series = result.iloc[:, 0]
        series.name = getattr(original, "name", None)
        return series
    return result


def zscore_normalize(
    data: Union[pd.DataFrame, pd.Series],
    window: int = 600,
    ddof: int = 1,
) -> Union[pd.DataFrame, pd.Series]:
    """
    Apply rolling z-score normalization column-wise with growing window.

    Uses rolling windows (growing windows for the first rows) to ensure causality
    and prevent look-ahead bias.

    Args:
        data: Input data (Series or DataFrame)
        window: Rolling window size (default: 600 samples)
        ddof: Degrees of freedom for standard deviation calculation

    Returns:
        Rolling z-score normalized data
    """
    # Delegate to rolling_zscore_normalize which already implements this correctly
    return rolling_zscore_normalize(data, window=window, min_periods=1, ddof=ddof)


def winsorized_zscore_normalize(
    data: Union[pd.DataFrame, pd.Series],
    window: int = 600,
    ddof: int = 0,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> Union[pd.DataFrame, pd.Series]:
    """
    Apply rolling z-score normalization via Polars with a growing window for the first rows.

    This uses rolling windows (growing windows for the first rows) to compute
    z-score normalization with winsorization, with 600 samples per window by default.
    This ensures causality and prevents look-ahead bias.

    Args:
        data: Input data (Series or DataFrame)
        window: Rolling window size (default: 600 samples)
        ddof: Degrees of freedom for standard deviation calculation
        lower_quantile: Lower quantile for winsorization (default: 0.01 = 1st percentile)
        upper_quantile: Upper quantile for winsorization (default: 0.99 = 99th percentile)

    Returns:
        Rolling z-score normalized data with winsorization
    """
    try:
        import polars as pl
    except ImportError:
        # Fallback to pandas implementation if polars not available
        return rolling_winsorized_zscore_normalize(data, window=window, ddof=ddof,
                                                   lower_quantile=lower_quantile,
                                                   upper_quantile=upper_quantile)

    df, was_series = _normalize_input(data)

    # Convert to Polars for efficient rolling operations
    pl_df = pl.from_pandas(df)

    # Process each column
    normalized_cols = []
    for col in pl_df.columns:
        # Calculate rolling quantiles with expanding window for first rows
        rolling_lower = (
            pl_df.select(col)
            .with_columns([
                pl.col(col).rolling_quantile(
                    quantile=lower_quantile,
                    window_size=window,
                    min_periods=1  # Growing window for first rows
                ).alias('lower_bound')
            ])
        )['lower_bound']

        rolling_upper = (
            pl_df.select(col)
            .with_columns([
                pl.col(col).rolling_quantile(
                    quantile=upper_quantile,
                    window_size=window,
                    min_periods=1  # Growing window for first rows
                ).alias('upper_bound')
            ])
        )['upper_bound']

        # Winsorize: clip values to rolling quantile bounds
        winsorized = pl_df.select(col).with_columns([
            pl.col(col).clip(rolling_lower, rolling_upper).alias('winsorized')
        ])['winsorized']

        # Calculate rolling mean and std with growing window
        rolling_mean = (
            pl_df.select(col)
            .with_columns([
                winsorized.rolling_mean(window_size=window, min_periods=1).alias('mean')
            ])
        )['mean']

        rolling_std = (
            pl_df.select(col)
            .with_columns([
                winsorized.rolling_std(window_size=window, min_periods=1, ddof=ddof).alias('std')
            ])
        )['std']

        # Z-score normalization
        normalized = (winsorized - rolling_mean) / rolling_std
        normalized = normalized.fill_null(0.0).fill_nan(0.0)
        normalized_cols.append(normalized.alias(col))

    # Combine normalized columns
    result_pl = pl.DataFrame(normalized_cols)

    # Convert back to pandas
    result = result_pl.to_pandas()
    result.index = df.index

    return _restore_output(result.astype(float), was_series, data)


def robust_normalize(
    data: Union[pd.DataFrame, pd.Series],
    window: int = 600,
    center: str = "median",
    scale: str = "mad",
    epsilon: float = 1e-9,
) -> Union[pd.DataFrame, pd.Series]:
    """
    Apply rolling robust normalization using median/IQR style scaling.

    Uses rolling windows (growing windows for the first rows) to ensure causality
    and prevent look-ahead bias.

    Args:
        data: Input data (Series or DataFrame)
        window: Rolling window size (default: 600 samples)
        center: Center metric ('median' or 'mean')
        scale: Scale metric ('mad' or 'iqr')
        epsilon: Small value to prevent division by zero

    Returns:
        Rolling robust normalized data
    """
    df, was_series = _normalize_input(data)

    # Calculate rolling centers with growing window
    if center == "median":
        centers = df.rolling(window=window, min_periods=1).median()
    elif center == "mean":
        centers = df.rolling(window=window, min_periods=1).mean()
    else:
        raise ValueError("center must be 'median' or 'mean'")

    # Calculate rolling scales with growing window
    if scale == "mad":
        # Rolling MAD (Median Absolute Deviation)
        scales = (df - centers).abs().rolling(window=window, min_periods=1).median()
    elif scale == "iqr":
        # Rolling IQR (Interquartile Range)
        q75 = df.rolling(window=window, min_periods=1).quantile(0.75)
        q25 = df.rolling(window=window, min_periods=1).quantile(0.25)
        scales = q75 - q25
    else:
        raise ValueError("scale must be 'mad' or 'iqr'")

    scales = scales.replace(0, np.nan)
    normalized = (df - centers) / (scales + epsilon)
    normalized = normalized.fillna(0.0)

    return _restore_output(normalized.astype(float), was_series, data)


def _is_volume_feature(
    feature_name: str,
    volume_columns: Optional[List[str]] = None,
) -> bool:
    """Heuristic check for pure volume features.

    This intentionally focuses on explicit volume naming patterns to avoid
    catching generic volatility metrics (e.g. "volatility", "vol_of_vol").
    """
    if volume_columns and feature_name in volume_columns:
        return True

    name = feature_name.lower()

    # Avoid double-transforming already log-transformed series such as
    # 'log_volume' or 'log1p_volume'. These should typically be treated as
    # generic continuous features and left to the default branch.
    if name.startswith("log_") or name.startswith("log1p_"):
        return False

    if name == "volume":
        return True
    if name.endswith("_volume") or name.startswith("volume_"):
        return True
    return False


def rolling_adaptive_normalize(
    data: Union[pd.DataFrame, pd.Series],
    window: int,
    min_periods: Optional[int] = None,
    ddof: int = 1,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    close: Optional[pd.Series] = None,
    volume_columns: Optional[List[str]] = None,
    enable_log1p_volume: bool = True,
    atr_window: int = 14,
) -> Union[pd.DataFrame, pd.Series]:
    """Adaptive rolling normalization for feature matrices.

    Routing logic per feature column:

    - If ``enable_log1p_volume`` and the feature is identified as a pure
      volume feature (via ``_is_volume_feature`` / ``volume_columns``) and
      ``log1p_zscore_normalize`` is available → apply log1p + rolling
      z-score normalization from :mod:`volume_transforms`.

    - Else if OHLC data is provided and ``should_use_atr_normalization``
      returns ``True`` for the feature name → normalize by ATR using a
      fixed ``atr_window`` (price-distance/level semantics).

    - Else → fall back to ``rolling_winsorized_zscore_normalize`` with the
      provided rolling window and quantile parameters.

    This keeps winsorized z-score as the default for momentum/speed and
    general statistical features while giving ATR-normalized behaviour for
    spatial distance/level features and robust log1p-normalization for
    pure volume series.
    """
    df, was_series = _normalize_input(data)

    if min_periods is None:
        min_periods = window

    result = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

    # Pre-compute ATR once if possible and only if any column might use it
    atr_series: Optional[pd.Series] = None
    if high is not None and low is not None and close is not None:
        try:
            atr_raw = calculate_atr(high, low, close, window=atr_window)
            if isinstance(atr_raw, pd.Series):
                atr_series = atr_raw.clip(lower=1e-8)
            else:
                atr_series = pd.Series(atr_raw, index=high.index).clip(lower=1e-8)
        except Exception:
            atr_series = None

    for col in df.columns:
        series = df[col]
        col_name = str(col)

        # 1) Volume-style features → optional log1p + z-score normalization
        if (
            enable_log1p_volume
            and VOLUME_TRANSFORMS_AVAILABLE
            and log1p_zscore_normalize is not None
            and _is_volume_feature(col_name, volume_columns)
        ):
            try:
                result[col] = log1p_zscore_normalize(
                    series,
                    window=window,
                    min_periods=min_periods,
                    ddof=ddof,
                )
                continue
            except Exception:
                # Fall through to winsorized z-score if robust volume transform fails
                pass

        # 2) Spatial distance/level features → ATR normalization when OHLC is available
        if atr_series is not None and should_use_atr_normalization(col_name):
            try:
                # Align indices and divide by ATR
                aligned_series = series.astype(float)
                normalized = aligned_series.div(atr_series, axis=0)
                result[col] = normalized.fillna(0.0)
                continue
            except Exception:
                # If ATR route fails, fall back to winsorized z-score
                pass

        # 3) Default: rolling winsorized z-score normalization
        result[col] = rolling_winsorized_zscore_normalize(
            series,
            window=window,
            min_periods=min_periods,
            ddof=ddof,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
        )

    return _restore_output(result.astype(float), was_series, data)


def rank_normalize(
    data: Union[pd.DataFrame, pd.Series],
    method: str = "average",
    ascending: bool = True,
) -> Union[pd.DataFrame, pd.Series]:
    """Apply percentile rank normalization column-wise."""
    df, was_series = _normalize_input(data)

    normalized = df.rank(method=method, ascending=ascending, pct=True)
    normalized = normalized.fillna(0.5)

    return _restore_output(normalized.astype(float), was_series, data)


def rolling_zscore_normalize(
    data: Union[pd.DataFrame, pd.Series],
    window: int,
    min_periods: Optional[int] = None,
    ddof: int = 1,
) -> Union[pd.DataFrame, pd.Series]:
    """
    Apply rolling z-score normalization using only data available at time t.

    This ensures causality: the normalization at time t uses only data from t-window to t-1.
    This is critical for preventing look-ahead bias in ML models.

    Args:
        data: Input data (Series or DataFrame)
        window: Rolling window size (e.g., 500 for ~500 bars)
        min_periods: Minimum number of observations required (default: window)
        ddof: Degrees of freedom for standard deviation calculation (default: 1)

    Returns:
        Rolling z-score normalized data
    """
    df, was_series = _normalize_input(data)

    if min_periods is None:
        min_periods = window

    # Calculate rolling statistics using only past data
    rolling_mean = df.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = df.rolling(window=window, min_periods=min_periods).std(ddof=ddof)

    # Normalize using rolling statistics
    rolling_std_safe = rolling_std.replace(0, np.nan)
    normalized = (df - rolling_mean) / rolling_std_safe
    normalized = normalized.fillna(0.0)

    return _restore_output(normalized.astype(float), was_series, data)


def rolling_winsorized_zscore_normalize(
    data: Union[pd.DataFrame, pd.Series],
    window: int,
    min_periods: Optional[int] = None,
    ddof: int = 1,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> Union[pd.DataFrame, pd.Series]:
    """
    Apply rolling winsorized z-score normalization using only data available at time t.

    This combines rolling window approach with winsorization to handle outliers while
    ensuring causality. At each time t, it uses data from t-window to t-1 to compute
    the normalization parameters.

    Args:
        data: Input data (Series or DataFrame)
        window: Rolling window size (e.g., 500 for ~500 bars)
        min_periods: Minimum number of observations required (default: window)
        ddof: Degrees of freedom for standard deviation calculation (default: 1)
        lower_quantile: Lower quantile for winsorization (default: 0.01)
        upper_quantile: Upper quantile for winsorization (default: 0.99)

    Returns:
        Rolling winsorized z-score normalized data
    """
    df, was_series = _normalize_input(data)

    if min_periods is None:
        min_periods = window

    # Initialize result with NaN
    normalized = pd.DataFrame(np.nan, index=df.index, columns=df.columns)

    # Process each column separately
    for col in df.columns:
        series = df[col]
        result_col = pd.Series(np.nan, index=series.index)

        # Calculate rolling quantiles for winsorization
        rolling_lower = series.rolling(window=window, min_periods=min_periods).quantile(lower_quantile)
        rolling_upper = series.rolling(window=window, min_periods=min_periods).quantile(upper_quantile)

        # Winsorize: clip values to rolling quantile bounds
        winsorized = series.clip(lower=rolling_lower, upper=rolling_upper, axis=0)

        # Calculate rolling statistics on winsorized data
        rolling_mean = winsorized.rolling(window=window, min_periods=min_periods).mean()
        rolling_std = winsorized.rolling(window=window, min_periods=min_periods).std(ddof=ddof)

        # Normalize
        rolling_std_safe = rolling_std.replace(0, np.nan)
        result_col = (winsorized - rolling_mean) / rolling_std_safe
        result_col = result_col.fillna(0.0)

        normalized[col] = result_col

    return _restore_output(normalized.astype(float), was_series, data)


def rolling_minmax_normalize(
    data: Union[pd.DataFrame, pd.Series],
    window: int,
    min_periods: Optional[int] = None,
    feature_range: Tuple[float, float] = (0.0, 1.0),
) -> Union[pd.DataFrame, pd.Series]:
    """
    Apply rolling min-max normalization using only data available at time t.

    This normalizes each value to the range [feature_range[0], feature_range[1]]
    based on the rolling min/max from the past window.

    Args:
        data: Input data (Series or DataFrame)
        window: Rolling window size (e.g., 500 for ~500 bars)
        min_periods: Minimum number of observations required (default: window)
        feature_range: Desired output range (default: (0, 1))

    Returns:
        Rolling min-max normalized data
    """
    df, was_series = _normalize_input(data)

    if min_periods is None:
        min_periods = window

    # Calculate rolling min and max using only past data
    rolling_min = df.rolling(window=window, min_periods=min_periods).min()
    rolling_max = df.rolling(window=window, min_periods=min_periods).max()

    # Normalize to [0, 1]
    range_val = rolling_max - rolling_min
    range_val_safe = range_val.replace(0, np.nan)
    normalized_01 = (df - rolling_min) / range_val_safe

    # Scale to desired range
    min_val, max_val = feature_range
    normalized = normalized_01 * (max_val - min_val) + min_val
    normalized = normalized.fillna(min_val)

    return _restore_output(normalized.astype(float), was_series, data)


class ScalingNormalizer:
    """
    Comprehensive scaling and normalization for the unified pipeline.

    Provides multiple scaling strategies with automatic selection based on
    data characteristics and feature types. Integrates with VectorBT for
    optimized performance when available.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the scaling normalizer."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Initialize VectorBT components if available
        self.vectorbt_manager = None
        self.vectorbt_scaler = None

        if VECTORBT_AVAILABLE:
            try:
                self.vectorbt_manager = UnifiedVectorizationManager()
                self.vectorbt_scaler = VectorBTScaler()
                tprint_success("✅ VectorBT components initialized for scaling")
            except Exception as e:
                tprint_warning(f"⚠️ VectorBT initialization failed: {e}")

        # Traditional scaling strategies
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler(),
            'maxabs': MaxAbsScaler(),
            'quantile': QuantileTransformer(output_distribution='normal'),
            'power': PowerTransformer(method='yeo-johnson'),
            'normalizer': Normalizer()
        }

        # Configuration
        # Changed default from 'robust' to 'winsorized_zscore' for better outlier handling
        self.default_strategy = self.config.get('default_strategy', 'winsorized_zscore')
        self.auto_select = self.config.get('auto_select', True)
        self.handle_outliers = self.config.get('handle_outliers', True)
        self.outlier_threshold = self.config.get('outlier_threshold', 3.0)
        self.use_vectorbt = self.config.get('use_vectorbt', VECTORBT_AVAILABLE)
        self.winsorize_quantiles = self.config.get('winsorize_quantiles', (0.01, 0.99))

        # Fitted scalers and feature mappings
        self.fitted_scalers = {}
        self.feature_mappings = {}
        self.scaling_stats = {}

        tprint_success("✅ ScalingNormalizer initialized")

    def analyze_data_characteristics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze data characteristics to recommend scaling strategy.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary with data characteristics and recommendations
        """
        characteristics = {
            'numeric_features': [],
            'categorical_features': [],
            'outlier_features': [],
            'skewed_features': [],
            'recommended_strategy': self.default_strategy,
            'feature_stats': {},
            'vectorbt_suitable': False
        }

        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                characteristics['numeric_features'].append(col)

                # Calculate statistics
                col_data = data[col].dropna()
                if len(col_data) > 0:
                    stats = {
                        'mean': col_data.mean(),
                        'std': col_data.std(),
                        'min': col_data.min(),
                        'max': col_data.max(),
                        'skewness': col_data.skew(),
                        'kurtosis': col_data.kurtosis(),
                        'outlier_count': self._count_outliers(col_data),
                        'total_count': len(col_data)
                    }
                    characteristics['feature_stats'][col] = stats

                    # Check for outliers
                    if stats['outlier_count'] > len(col_data) * 0.05:  # 5% outliers
                        characteristics['outlier_features'].append(col)

                    # Check for skewness
                    if abs(stats['skewness']) > 1.0:
                        characteristics['skewed_features'].append(col)
            else:
                characteristics['categorical_features'].append(col)

        # Check if data is suitable for VectorBT optimization
        if (self.use_vectorbt and VECTORBT_AVAILABLE and
            len(characteristics['numeric_features']) > 0 and
            len(data) > 1000):  # VectorBT is more efficient for larger datasets
            characteristics['vectorbt_suitable'] = True

        # Recommend scaling strategy based on characteristics
        if characteristics['outlier_features']:
            characteristics['recommended_strategy'] = 'robust'
        elif characteristics['skewed_features']:
            characteristics['recommended_strategy'] = 'quantile'
        else:
            characteristics['recommended_strategy'] = 'standard'

        # tprint_info(f"📊 Data analysis: {len(characteristics['numeric_features'])} numeric, "
        #            f"{len(characteristics['outlier_features'])} with outliers, "
        #            f"{len(characteristics['skewed_features'])} skewed, "
        #            f"VectorBT suitable: {characteristics['vectorbt_suitable']}")

        return characteristics

    def _count_outliers(self, data: pd.Series, method: str = 'iqr') -> int:
        """Count outliers in a data series."""
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return ((data < lower_bound) | (data > upper_bound)).sum()
        elif method == 'zscore':
            z_scores = np.abs((data - data.mean()) / data.std())
            return (z_scores > self.outlier_threshold).sum()
        else:
            return 0

    def select_scaling_strategy(self, feature_name: str, feature_stats: Dict[str, Any]) -> str:
        """
        Select appropriate scaling strategy for a feature.

        Args:
            feature_name: Name of the feature
            feature_stats: Statistics of the feature

        Returns:
            Selected scaling strategy name
        """
        if not feature_stats:
            return self.default_strategy

        total_count = feature_stats.get('total_count', 0)
        outlier_count = feature_stats.get('outlier_count', 0)
        if not total_count or total_count <= 0:
            outlier_ratio = 0.0
        else:
            outlier_ratio = outlier_count / total_count

        skewness = abs(feature_stats.get('skewness', 0.0))
        feature_min = feature_stats.get('min', 0.0)

        # Strategy selection logic
        if outlier_ratio > 0.1:  # More than 10% outliers
            return 'robust'
        elif skewness > 2.0:  # Highly skewed
            return 'quantile'
        elif skewness > 1.0:  # Moderately skewed
            return 'power'
        elif feature_min < 0:  # Has negative values
            return 'standard'
        else:  # Normal distribution, positive values
            return 'minmax'

    def fit_transform(self, data: pd.DataFrame,
                     strategy: Optional[str] = None,
                     feature_list: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fit and transform data using appropriate scaling strategies.

        Args:
            data: Input DataFrame
            strategy: Specific scaling strategy (if None, auto-select)
            feature_list: Specific features to scale (if None, scale all numeric)

        Returns:
            Scaled DataFrame
        """
        # tprint_info("🔧 Starting scaling and normalization")

        # Analyze data characteristics
        characteristics = self.analyze_data_characteristics(data)

        # Determine features to scale
        if feature_list is None:
            feature_list = characteristics['numeric_features']

        if not feature_list:
            tprint_info("ℹ️ No numeric features to scale")
            return data

        # Try VectorBT optimization if suitable
        if (characteristics['vectorbt_suitable'] and
            self.vectorbt_scaler is not None and
            strategy in ['standard', 'robust', 'minmax']):

            try:
                tprint_info("🚀 Using VectorBT for optimized scaling")
                scaled_data = self._vectorbt_fit_transform(data, feature_list, strategy)
                if scaled_data is not None:
                    tprint_success("✅ VectorBT scaling completed")
                    return scaled_data
                else:
                    tprint_warning("⚠️ VectorBT scaling failed, falling back to traditional methods")
            except Exception as e:
                tprint_warning(f"⚠️ VectorBT scaling error: {e}, using traditional methods")

        # Traditional scaling approach
        scaled_data = data.copy()

        # Process each feature
        for feature in feature_list:
            if feature not in data.columns:
                tprint_warning(f"⚠️ Feature {feature} not found in data")
                continue

            try:
                # Select scaling strategy
                if strategy is None and self.auto_select:
                    feature_stats = characteristics['feature_stats'].get(feature, {})
                    selected_strategy = self.select_scaling_strategy(feature, feature_stats)
                else:
                    selected_strategy = strategy or self.default_strategy

                # Normalize naming differences (e.g. zscore -> standard)
                selected_strategy = self._normalize_strategy_name(selected_strategy)

                # Apply scaling
                scaled_feature = self._apply_scaling(
                    data[feature], feature, selected_strategy, fit=True
                )

                if scaled_feature is not None:
                    scaled_data[feature] = scaled_feature
                    # tprint_success(f"✅ Scaled {feature} using {selected_strategy}")

            except Exception as e:
                tprint_error(f"❌ Error scaling feature {feature}: {e}")
                continue

        # tprint_success(f"✅ Scaling completed: {len(feature_list)} features processed")
        return scaled_data

    @staticmethod
    def _normalize_strategy_name(name: Optional[str]) -> Optional[str]:
        if name is None:
            return None
        aliases = {
            'zscore': 'standard',
            'standardize': 'standard',
            'robust_zscore': 'robust',
        }
        lowered = name.lower()
        return aliases.get(lowered, lowered)

    def _vectorbt_fit_transform(self, data: pd.DataFrame,
                               feature_list: List[str],
                               strategy: str) -> Optional[pd.DataFrame]:
        """Use VectorBT for optimized scaling."""
        try:
            if not self.vectorbt_scaler:
                return None

            # Prepare data for VectorBT
            numeric_data = data[feature_list].copy()

            # Apply VectorBT scaling
            if strategy == 'standard':
                scaled_data = self.vectorbt_scaler.zscore_normalize(numeric_data)
            elif strategy == 'robust':
                scaled_data = self.vectorbt_scaler.robust_normalize(numeric_data)
            elif strategy == 'minmax':
                scaled_data = self.vectorbt_scaler.minmax_normalize(numeric_data)
            else:
                return None

            # Store fitted scalers for inverse transform
            for feature in feature_list:
                self.fitted_scalers[feature] = {
                    'scaler': self.vectorbt_scaler,
                    'strategy': strategy,
                    'vectorbt': True
                }

            # Create result DataFrame with original structure
            result = data.copy()
            result[feature_list] = scaled_data

            return result

        except Exception as e:
            tprint_error(f"❌ VectorBT scaling failed: {e}")
            return None

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using previously fitted scalers.

        Args:
            data: Input DataFrame

        Returns:
            Transformed DataFrame
        """
        if not self.fitted_scalers:
            tprint_warning("⚠️ No fitted scalers found, returning original data")
            return data

        tprint_info("🔄 Applying fitted scaling transformations")

        # Create a copy to avoid modifying original data
        transformed_data = data.copy()

        for feature, scaler_info in self.fitted_scalers.items():
            if feature not in data.columns:
                tprint_warning(f"⚠️ Feature {feature} not found in data")
                continue

            try:
                scaler = scaler_info['scaler']
                strategy = scaler_info['strategy']
                is_vectorbt = scaler_info.get('vectorbt', False)

                # Apply transformation
                if strategy == 'winsorized_zscore':
                    # Apply winsorized z-score normalization
                    lower_q = scaler_info.get('lower_quantile', 0.01)
                    upper_q = scaler_info.get('upper_quantile', 0.99)
                    transformed_feature = winsorized_zscore_normalize(
                        data[feature],
                        lower_quantile=lower_q,
                        upper_quantile=upper_q
                    )
                    transformed_data[feature] = transformed_feature
                elif is_vectorbt and hasattr(scaler, f'{strategy}_normalize'):
                    # Use VectorBT transformation
                    feature_data = data[[feature]].copy()
                    transformed_feature = getattr(scaler, f'{strategy}_normalize')(feature_data)
                    transformed_data[feature] = transformed_feature[feature]
                else:
                    # Use traditional transformation
                    transformed_feature = self._apply_scaling(
                        data[feature], feature, strategy, fit=False
                    )
                    if transformed_feature is not None:
                        transformed_data[feature] = transformed_feature

                tprint_debug(f"✅ Transformed {feature} using {strategy}")

            except Exception as e:
                tprint_error(f"❌ Error transforming feature {feature}: {e}")
                continue

        tprint_success("✅ Transformation completed")
        return transformed_data

    def _apply_scaling(self, data: pd.Series, feature_name: str,
                      strategy: str, fit: bool = True) -> Optional[pd.Series]:
        """Apply specific scaling strategy to a feature."""
        try:
            # Handle winsorized_zscore as a special case
            if strategy == 'winsorized_zscore':
                try:
                    lower_q, upper_q = self.winsorize_quantiles
                    scaled_series = winsorized_zscore_normalize(
                        data,
                        lower_quantile=lower_q,
                        upper_quantile=upper_q
                    )
                    # Store metadata for this strategy
                    if fit:
                        self.fitted_scalers[feature_name] = {
                            'scaler': None,  # Functional approach, no scaler object
                            'strategy': 'winsorized_zscore',
                            'vectorbt': False,
                            'lower_quantile': lower_q,
                            'upper_quantile': upper_q
                        }
                    return scaled_series
                except Exception as e:
                    tprint_error(f"❌ Winsorized z-score normalization failed for {feature_name}: {e}")
                    return None

            # Get scaler
            if strategy not in self.scalers:
                tprint_warning(f"⚠️ Unknown scaling strategy: {strategy}")
                return None

            scaler = self.scalers[strategy]

            # Handle missing values
            data_clean = data.dropna()
            if len(data_clean) == 0:
                tprint_warning(f"⚠️ No valid data for feature {feature_name}")
                return None

            # Handle infinities
            inf_mask = np.isinf(data_clean.values)
            if inf_mask.any():
                inf_count = int(np.count_nonzero(inf_mask))
                tprint_warning(f"⚠️ Feature {feature_name} contains {inf_count} infinity values; replacing with finite min/max")
                
                # Get finite values
                finite_mask = np.isfinite(data_clean.values)
                if finite_mask.any():
                    finite_values = data_clean.values[finite_mask]
                    finite_max = np.max(finite_values)
                    finite_min = np.min(finite_values)
                    
                    # Replace infinities
                    clean_values = data_clean.values.copy()
                    clean_values[np.isposinf(clean_values)] = finite_max
                    clean_values[np.isneginf(clean_values)] = finite_min
                    data_clean = pd.Series(clean_values, index=data_clean.index)
                else:
                    # All values are inf, replace with 0
                    tprint_warning(f"⚠️ Feature {feature_name} has only infinity values; replacing with 0")
                    data_clean = pd.Series(0.0, index=data_clean.index)

            # Fit and transform
            if fit:
                scaled_values = scaler.fit_transform(data_clean.values.reshape(-1, 1)).flatten()
                # Store fitted scaler
                self.fitted_scalers[feature_name] = {
                    'scaler': scaler,
                    'strategy': strategy,
                    'vectorbt': False
                }
            else:
                # Use previously fitted scaler
                if feature_name in self.fitted_scalers:
                    scaler = self.fitted_scalers[feature_name]['scaler']
                    scaled_values = scaler.transform(data_clean.values.reshape(-1, 1)).flatten()
                else:
                    tprint_warning(f"⚠️ No fitted scaler found for feature {feature_name}")
                    return None

            # Create Series with original index
            scaled_series = pd.Series(index=data.index, dtype=float)
            scaled_series.loc[data_clean.index] = scaled_values

            return scaled_series

        except Exception as e:
            tprint_error(f"❌ Error applying {strategy} scaling to {feature_name}: {e}")
            return None

    def inverse_transform(self, data: pd.DataFrame, feature_name: str) -> pd.Series:
        """Inverse transform a scaled feature back to original scale."""
        if feature_name not in self.fitted_scalers:
            tprint_warning(f"⚠️ No fitted scaler found for feature {feature_name}")
            return data[feature_name]

        try:
            scaler_info = self.fitted_scalers[feature_name]
            scaler = scaler_info['scaler']
            is_vectorbt = scaler_info.get('vectorbt', False)

            # Handle missing values
            data_clean = data[feature_name].dropna()
            if len(data_clean) == 0:
                return data[feature_name]

            # Inverse transform
            if is_vectorbt and hasattr(scaler, 'inverse_transform'):
                # Use VectorBT inverse transform
                feature_data = data_clean.values.reshape(-1, 1)
                original_values = scaler.inverse_transform(feature_data).flatten()
            else:
                # Use traditional inverse transform
                original_values = scaler.inverse_transform(data_clean.values.reshape(-1, 1)).flatten()

            # Create Series with original index
            original_series = pd.Series(index=data[feature_name].index, dtype=float)
            original_series.loc[data_clean.index] = original_values

            return original_series

        except Exception as e:
            tprint_error(f"❌ Error in inverse transform for {feature_name}: {e}")
            return data[feature_name]

    def get_scaling_summary(self) -> Dict[str, Any]:
        """Get summary of scaling operations."""
        summary = {
            'total_features_scaled': len(self.fitted_scalers),
            'scaling_strategies_used': {},
            'feature_details': {},
            'vectorbt_usage': 0,
            'traditional_usage': 0
        }

        for feature, scaler_info in self.fitted_scalers.items():
            strategy = scaler_info['strategy']
            is_vectorbt = scaler_info.get('vectorbt', False)

            summary['scaling_strategies_used'][strategy] = summary['scaling_strategies_used'].get(strategy, 0) + 1

            if is_vectorbt:
                summary['vectorbt_usage'] += 1
            else:
                summary['traditional_usage'] += 1

            summary['feature_details'][feature] = {
                'strategy': strategy,
                'scaler_type': type(scaler_info['scaler']).__name__,
                'vectorbt': is_vectorbt
            }

        return summary

    def reset(self):
        """Reset all fitted scalers and mappings."""
        self.fitted_scalers = {}
        self.feature_mappings = {}
        self.scaling_stats = {}
        tprint_success("✅ Scaling normalizer reset")
