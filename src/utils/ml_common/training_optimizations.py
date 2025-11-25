"""
Training Optimizations for Analyst Base Models.

Provides optimization techniques to speed up training:
1. Histogram binning for continuous features
2. GOSS (Gradient-based One-Side Sampling) for LightGBM
3. Precision reduction (float32 vs float64)
4. Pre-calculation and vectorization
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import KBinsDiscretizer

logger = logging.getLogger(__name__)


class HistogramBinner:
    """
    Bins continuous features into discrete buckets for faster training.

    Used by XGBoost's histogram-based algorithm to reduce memory and
    computation while maintaining model quality.
    """

    def __init__(
        self,
        n_bins: int = 256,
        strategy: str = 'quantile',
        subsample: Optional[int] = None
    ):
        """
        Initialize histogram binner.

        Args:
            n_bins: Number of bins (default: 256, XGBoost default)
            strategy: Binning strategy ('quantile', 'uniform', 'kmeans')
            subsample: Optional number of samples to use for bin calculation
        """
        self.n_bins = n_bins
        self.strategy = strategy
        self.subsample = subsample
        self.binner = None
        self.feature_names = None

    def fit(self, X: pd.DataFrame) -> 'HistogramBinner':
        """
        Fit binner on training data.

        Args:
            X: Training data

        Returns:
            Self
        """
        self.feature_names = X.columns.tolist()

        # Subsample if requested (for large datasets)
        if self.subsample and len(X) > self.subsample:
            logger.info(f"Subsampling {self.subsample} samples for binning")
            X_sample = X.sample(n=self.subsample, random_state=42)
        else:
            X_sample = X

        # Fit KBinsDiscretizer
        self.binner = KBinsDiscretizer(
            n_bins=self.n_bins,
            encode='ordinal',
            strategy=self.strategy
        )

        self.binner.fit(X_sample)

        logger.info(
            f"Histogram binner fitted: {len(self.feature_names)} features, "
            f"{self.n_bins} bins per feature"
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features into binned representation.

        Args:
            X: Input data

        Returns:
            Binned data
        """
        if self.binner is None:
            raise ValueError("Binner not fitted. Call fit() first.")

        X_binned = self.binner.transform(X)

        return pd.DataFrame(
            X_binned,
            index=X.index,
            columns=self.feature_names
        )

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class PrecisionReducer:
    """
    Reduces feature precision from float64 to float32 to save memory.

    This can reduce memory usage by 50% with minimal impact on model quality.
    """

    def __init__(self, target_dtype: str = 'float32'):
        """
        Initialize precision reducer.

        Args:
            target_dtype: Target data type ('float32' or 'float16')
        """
        self.target_dtype = np.dtype(target_dtype)

    def reduce_precision(
        self,
        df: pd.DataFrame,
        exclude_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Reduce precision of numeric columns.

        Args:
            df: Input DataFrame
            exclude_columns: Columns to exclude from reduction

        Returns:
            DataFrame with reduced precision
        """
        df = df.copy()
        exclude_columns = exclude_columns or []

        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.float64, np.float32]).columns

        # Exclude specified columns
        cols_to_reduce = [col for col in numeric_cols if col not in exclude_columns]

        if cols_to_reduce:
            df[cols_to_reduce] = df[cols_to_reduce].astype(self.target_dtype)

            memory_before = df.memory_usage(deep=True).sum() / 1024**2
            logger.info(
                f"Reduced precision for {len(cols_to_reduce)} columns to {self.target_dtype}. "
            )

        return df


class FeatureVectorizer:
    """
    Pre-calculates vectorized feature operations for faster training.

    Identifies and pre-computes features that can benefit from vectorization.
    """

    def __init__(self):
        """Initialize feature vectorizer."""
        self.vectorized_features = {}

    def identify_vectorizable_features(
        self,
        df: pd.DataFrame
    ) -> List[str]:
        """
        Identify features that can benefit from vectorization.

        Args:
            df: Input DataFrame

        Returns:
            List of feature names
        """
        vectorizable = []

        for col in df.columns:
            # Check if feature has repeated calculations (e.g., rolling stats)
            if any(keyword in col.lower() for keyword in ['rolling', 'ema', 'sma', 'std', 'mean']):
                vectorizable.append(col)

        return vectorizable

    def pre_calculate_features(
        self,
        df: pd.DataFrame,
        feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Pre-calculate features using vectorized operations.

        Args:
            df: Input DataFrame
            feature_names: Features to pre-calculate

        Returns:
            DataFrame with pre-calculated features
        """
        # For now, just ensure features are contiguous in memory
        df = df.copy()

        for feature in feature_names:
            if feature in df.columns:
                df[feature] = np.ascontiguousarray(df[feature].values)

        return df


def configure_xgboost_optimizations(
    enable_histogram: bool = True,
    max_bin: int = 256,
    tree_method: str = 'hist',
    enable_categorical: bool = False
) -> Dict[str, Any]:
    """
    Configure XGBoost optimization parameters.

    Args:
        enable_histogram: Enable histogram-based algorithm
        max_bin: Maximum number of bins (histogram method)
        tree_method: Tree construction method ('hist', 'approx', 'exact')
        enable_categorical: Enable categorical feature support

    Returns:
        Dictionary of XGBoost parameters
    """
    params = {}

    if enable_histogram:
        params['tree_method'] = tree_method
        params['max_bin'] = max_bin
        logger.info(f"XGBoost histogram method enabled: tree_method={tree_method}, max_bin={max_bin}")

    if enable_categorical:
        params['enable_categorical'] = True

    return params


def configure_lightgbm_optimizations(
    enable_goss: bool = True,
    top_rate: float = 0.2,
    other_rate: float = 0.1,
    max_bin: int = 255,
    enable_bundle: bool = True
) -> Dict[str, Any]:
    """
    Configure LightGBM optimization parameters including GOSS.

    GOSS (Gradient-based One-Side Sampling) keeps instances with large gradients
    and randomly samples instances with small gradients, reducing training time
    while maintaining accuracy.

    Args:
        enable_goss: Enable GOSS sampling
        top_rate: Percentage of large gradient instances to keep (0.0-1.0)
        other_rate: Percentage of small gradient instances to sample (0.0-1.0)
        max_bin: Maximum number of bins for feature values
        enable_bundle: Enable feature bundling (EFB)

    Returns:
        Dictionary of LightGBM parameters
    """
    params = {
        'max_bin': max_bin,
        'feature_pre_filter': False  # Disable pre-filtering for speed
    }

    if enable_goss:
        params['boosting_type'] = 'goss'
        params['top_rate'] = top_rate
        params['other_rate'] = other_rate
        logger.info(
            f"LightGBM GOSS enabled: "
            f"top_rate={top_rate}, other_rate={other_rate}, "
            f"effective sample rate={top_rate + other_rate}"
        )
    else:
        params['boosting_type'] = 'gbdt'

    if enable_bundle:
        params['enable_bundle'] = True
        params['max_conflict_rate'] = 0.0  # For exclusive feature bundling

    return params


def estimate_memory_usage(
    n_samples: int,
    n_features: int,
    dtype: np.dtype = np.float32
) -> float:
    """
    Estimate memory usage for dataset.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        dtype: Data type

    Returns:
        Estimated memory usage in MB
    """
    bytes_per_element = dtype.itemsize
    total_bytes = n_samples * n_features * bytes_per_element

    # Add overhead for pandas (roughly 2x)
    total_bytes *= 2

    return total_bytes / 1024**2


def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage.

    Args:
        df: Input DataFrame

    Returns:
        Optimized DataFrame
    """
    df = df.copy()
    memory_before = df.memory_usage(deep=True).sum() / 1024**2

    # Reduce float precision
    float_cols = df.select_dtypes(include=[np.float64]).columns
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].astype(np.float32)

    # Reduce integer precision
    int_cols = df.select_dtypes(include=[np.int64]).columns
    for col in int_cols:
        col_min = df[col].min()
        col_max = df[col].max()

        if col_min >= 0:
            if col_max < 255:
                df[col] = df[col].astype(np.uint8)
            elif col_max < 65535:
                df[col] = df[col].astype(np.uint16)
            elif col_max < 4294967295:
                df[col] = df[col].astype(np.uint32)
        else:
            if col_min > -128 and col_max < 127:
                df[col] = df[col].astype(np.int8)
            elif col_min > -32768 and col_max < 32767:
                df[col] = df[col].astype(np.int16)
            elif col_min > -2147483648 and col_max < 2147483647:
                df[col] = df[col].astype(np.int32)

    memory_after = df.memory_usage(deep=True).sum() / 1024**2
    reduction = (memory_before - memory_after) / memory_before * 100

    logger.info(
        f"Memory optimization: {memory_before:.2f} MB → {memory_after:.2f} MB "
        f"({reduction:.1f}% reduction)"
    )

    return df


def create_efficient_train_params(
    model_type: str = 'xgboost',
    n_samples: int = 100000,
    n_features: int = 100,
    enable_gpu: bool = False
) -> Dict[str, Any]:
    """
    Create optimized training parameters based on dataset size.

    Args:
        model_type: 'xgboost' or 'lightgbm'
        n_samples: Number of training samples
        n_features: Number of features
        enable_gpu: Enable GPU training

    Returns:
        Dictionary of optimized parameters
    """
    params = {}

    if model_type == 'xgboost':
        # Use histogram method for large datasets
        if n_samples > 100000:
            params['tree_method'] = 'hist'
            params['max_bin'] = 256
        else:
            params['tree_method'] = 'approx'

        # GPU acceleration if available
        if enable_gpu:
            params['tree_method'] = 'gpu_hist'
            params['predictor'] = 'gpu_predictor'

        # Subsample for very large datasets
        if n_samples > 500000:
            params['subsample'] = 0.8
            params['colsample_bytree'] = 0.8

    elif model_type == 'lightgbm':
        # Use GOSS for large datasets
        if n_samples > 100000:
            params['boosting_type'] = 'goss'
            params['top_rate'] = 0.2
            params['other_rate'] = 0.1

        # Histogram binning
        params['max_bin'] = 255

        # GPU acceleration if available
        if enable_gpu:
            params['device'] = 'gpu'
            params['gpu_platform_id'] = 0
            params['gpu_device_id'] = 0

        # Feature bundling for many features
        if n_features > 100:
            params['enable_bundle'] = True

    return params
