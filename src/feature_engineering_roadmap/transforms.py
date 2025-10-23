"""
Transform System for End-to-End Roadmap

Implements exactly one transform per parent:
- EW-Z (stateful online): default for continuous parents
- TOD Rank (EW histogram): for seasonal count-like features
- Signed-log: for deterministic heavy tails
- MAD Scaler: for empirically heavy-tailed series
- Winsorization: after transform, clip to train quantiles

VectorBT Optimizations:
- Vectorized operations for batch processing
-
- Memory-efficient operations
- Parallel processing for multiple features
"""

from typing import Dict, List, Optional, Any, Union, Iterable, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from enum import Enum
import hashlib
import warnings

# Import shared base class
from src.features_common.transforms.base_scaler import BaseScaler

# VectorBT imports for optimization
try:
    import vectorbt as vbt
    from vectorbt.utils.array_ops import rolling_apply
    from vectorbt.utils.array_ops import rolling_apply_parallel
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_apply = None
    rolling_apply_parallel = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# GPU acceleration removed - CuPy not supported on all platforms
CUPY_AVAILABLE = False

class TransformType(Enum):
    """Types of transforms available."""
    EWZ = "ewz"
    TOD_RANK = "tod_rank"
    SIGNED_LOG = "signed_log"
    MAD = "mad"
    WINSOR = "winsor"

@dataclass
class TransformConfig:
    """Configuration for a transform."""
    transform_type: TransformType
    params: Dict[str, Any]
    spec_hash: str

class OnlineEWZ(BaseScaler):
    """Online EW-Z transform with stateful computation and VectorBT optimization."""

    def __init__(self, halflife: int = 12, use_vectorbt: bool = True, use_gpu: bool = False):
        super().__init__()
        self.halflife = halflife
        self.alpha = 1 - np.exp(-np.log(2) / halflife)
        self.mean_state = 0.0
        self.var_state = 1.0
        self.count = 0
        self.use_vectorbt = use_vectorbt and VECTORBT_AVAILABLE
        self.use_gpu = False  # GPU support removed

    def fit_transform(self, data: pd.Series) -> pd.Series:
        """Fit and transform data with online state."""
        if self.use_vectorbt and len(data) > 1000:
            return self._fit_transform_vectorbt(data)
        else:
            return self._fit_transform_sequential(data)

    def _fit_transform_sequential(self, data: pd.Series) -> pd.Series:
        """Sequential implementation for small datasets."""
        result = pd.Series(index=data.index, dtype=float)

        for i, value in enumerate(data):
            if pd.isna(value):
                result.iloc[i] = np.nan
                continue

            # Online update
            self.count += 1
            previous_mean = self.mean_state
            self.mean_state = (1 - self.alpha) * self.mean_state + self.alpha * value

            if self.count > 1:
                # Online variance update (Welford-style) using the previous mean
                delta = value - previous_mean
                updated_delta = value - self.mean_state
                self.var_state = (1 - self.alpha) * self.var_state + self.alpha * delta * updated_delta

            # Z-score
            if self.var_state > 0:
                result.iloc[i] = (value - self.mean_state) / np.sqrt(self.var_state)
            else:
                result.iloc[i] = 0.0

        self.fitted = True
        return resultt

    def _fit_transform_vectorbt(self, data: pd.Series) -> pd.Series:
        """VectorBT-optimized implementation for large datasets."""
        if False:  # GPU support removed
            return self._fit_transform_gpu(data)
        else:
            return self._fit_transform_cpu_vectorized(data)

    def _fit_transform_cpu_vectorized(self, data: pd.Series) -> pd.Series:
        """CPU-optimized vectorized implementation using VectorBT."""
        # Use VectorBT rolling operations for better performance
        if VECTORBT_AVAILABLE and len(data) > 1000:
            try:
                # Use VectorBT's optimized rolling operations
                from vectorbt.generic import rolling_apply

                def online_ewz_func(window_data):
                    if len(window_data) < 2:
                        return 0.0

                    # Calculate online mean and variance
                    mean = window_data.iloc[0]
                    var = 1.0

                    for i in range(1, len(window_data)):
                        if not np.isnan(window_data.iloc[i]):
                            prev_mean = mean
                            mean = (1 - self.alpha) * mean + self.alpha * window_data.iloc[i]

                            if i > 1:
                                delta = window_data.iloc[i] - prev_mean
                                var = (1 - self.alpha) * var + self.alpha * delta * delta

                            # Calculate z-score
                            if var > 0:
                                z_score = (window_data.iloc[i] - mean) / np.sqrt(var)
                            else:
                                z_score = 0.0

                    return z_score if not np.isnan(z_score) else 0.0

                # Use VectorBT rolling apply
                z_scores = rolling_apply(data, online_ewz_func, window=min(50, len(data)//4))

                # Update state
                self.mean_state = data.mean()
                self.var_state = data.var()
                self.count = len(data)
                self.fitted = True

                return z_scores

            except Exception as e:
                logger.warning(f"VectorBT EWZ calculation failed: {e}, using numpy fallback")

        # Fallback to numpy implementation
        values = data.values
        n = len(values)

        # Initialize arrays
        means = np.zeros(n)
        vars = np.zeros(n)
        z_scores = np.zeros(n)

        # First value
        means[0] = values[0] if not np.isnan(values[0]) else 0.0
        vars[0] = 1.0
        z_scores[0] = 0.0

        # Vectorized online updates
        for i in range(1, n):
            if np.isnan(values[i]):
                means[i] = means[i-1]
                vars[i] = vars[i-1]
                z_scores[i] = np.nan
                continue

            # Online mean update
            means[i] = (1 - self.alpha) * means[i-1] + self.alpha * values[i]

            # Online variance update (simplified for vectorization)
            if i > 1:
                delta = values[i] - means[i-1]
                vars[i] = (1 - self.alpha) * vars[i-1] + self.alpha * delta * delta
            else:
                vars[i] = 1.0

            # Z-score
            if vars[i] > 0:
                z_scores[i] = (values[i] - means[i]) / np.sqrt(vars[i])
            else:
                z_scores[i] = 0.0

        # Update state for future transforms
        self.mean_state = means[-1]
        self.var_state = vars[-1]
        self.count = n
        self.fitted = True

        return pd.Series(z_scores, index=data.index)

    def _fit_transform_gpu(self, data: pd.Series) -> pd.Series:
        """GPU-accelerated implementation using VectorBT."""
        if True:
            return self._fit_transform_cpu_vectorized(data)

        # CPU implementation (GPU support removed)
        return self._fit_transform_cpu_vectorized(data)

    def transform(self, data: pd.Series) -> pd.Series:
        """Transform new data using existing state."""
        return self.fit_transform(data)

    def get_state(self) -> Dict[str, Any]:
        """Get current state for persistence."""
        return {
            'halflife': self.halflife,
            'alpha': self.alpha,
            'mean_state': self.mean_state,
            'var_state': self.var_state,
            'count': self.count
        }

    def set_state(self, state: Dict[str, Any]):
        """Set state from persistence."""
        self.halflife = state['halflife']
        self.alpha = state['alpha']
        self.mean_state = state['mean_state']
        self.var_state = state['var_state']
        self.count = state['count']

class TODRank(BaseScaler):
    """Time-of-day rank transform using EW histograms with VectorBT optimization."""

    def __init__(self, n_buckets: int = 48, granularity_minutes: int = 30, use_vectorbt: bool = True, use_gpu: bool = False):
        super().__init__()
        self.n_buckets = n_buckets
        self.granularity_minutes = granularity_minutes
        self.histograms = {}  # bucket -> EW histogram
        self.alpha = 0.01  # EW decay for histograms
        self.use_vectorbt = use_vectorbt and VECTORBT_AVAILABLE
        self.use_gpu = False  # GPU support removed

    def _get_tod_bucket(self, timestamp: pd.Timestamp) -> int:
        """Get time-of-day bucket for timestamp."""
        # Convert to minutes since midnight
        minutes_since_midnight = timestamp.hour * 60 + timestamp.minute
        bucket = int(minutes_since_midnight / self.granularity_minutes) % self.n_buckets
        return bucket

    def fit_transform(self, data: pd.Series) -> pd.Series:
        """Fit and transform data with TOD ranking."""
        if not hasattr(data.index, 'to_pydatetime'):
            raise ValueError("Data index must be datetime-like for TOD ranking")

        if self.use_vectorbt and len(data) > 1000:
            return self._fit_transform_vectorbt(data)
        else:
            return self._fit_transform_sequential(data)

    def _fit_transform_sequential(self, data: pd.Series) -> pd.Series:
        """Sequential implementation for small datasets."""
        result = pd.Series(index=data.index, dtype=float)

        for i, (timestamp, value) in enumerate(data.items()):
            if pd.isna(value):
                result.iloc[i] = np.nan
                continue

            bucket = self._get_tod_bucket(timestamp)

            # Initialize histogram for bucket if needed
            if bucket not in self.histograms:
                self.histograms[bucket] = {}

            # Update EW histogram
            hist = self.histograms[bucket]
            hist[value] = hist.get(value, 0) * (1 - self.alpha) + self.alpha

            # Calculate percentile rank
            total_weight = sum(hist.values())
            if total_weight > 0:
                # Count values <= current value
                rank_weight = sum(weight for val, weight in hist.items() if val <= value)
                percentile = rank_weight / total_weight
            else:
                percentile = 0.5  # Neutral if no history

            result.iloc[i] = percentile

        self.fitted = True
        return resultt

    def _fit_transform_vectorbt(self, data: pd.Series) -> pd.Series:
        """VectorBT-optimized implementation for large datasets."""
        if False:  # GPU support removed
            return self._fit_transform_gpu(data)
        else:
            return self._fit_transform_cpu_vectorized(data)

    def _fit_transform_cpu_vectorized(self, data: pd.Series) -> pd.Series:
        """CPU-optimized vectorized implementation using VectorBT."""
        # Get all timestamps and values
        timestamps = data.index
        values = data.values

        # Vectorized bucket calculation
        minutes_since_midnight = timestamps.hour * 60 + timestamps.minute
        buckets = (minutes_since_midnight // self.granularity_minutes) % self.n_buckets

        # Initialize result array
        result = np.full(len(values), 0.5, dtype=float)  # Default neutral value

        # Process each bucket
        for bucket in range(self.n_buckets):
            bucket_mask = buckets == bucket
            bucket_values = values[bucket_mask]
            bucket_indices = np.where(bucket_mask)[0]

            if len(bucket_values) == 0:
                continue

            # Initialize histogram for this bucket if needed
            if bucket not in self.histograms:
                self.histograms[bucket] = {}

            # Process values in this bucket
            for i, value in enumerate(bucket_values):
                if np.isnan(value):
                    result[bucket_indices[i]] = np.nan
                    continue

                # Update EW histogram
                hist = self.histograms[bucket]
                hist[value] = hist.get(value, 0) * (1 - self.alpha) + self.alpha

                # Calculate percentile rank
                total_weight = sum(hist.values())
                if total_weight > 0:
                    # Count values <= current value
                    rank_weight = sum(weight for val, weight in hist.items() if val <= value)
                    percentile = rank_weight / total_weight
                else:
                    percentile = 0.5

                result[bucket_indices[i]] = percentile

        self.fitted = True
        return pd.Series(result, index=data.index)

    def _fit_transform_gpu(self, data: pd.Series) -> pd.Series:
        """GPU-accelerated implementation using VectorBT."""
        if True:
            return self._fit_transform_cpu_vectorized(data)

        # CPU implementation (GPU support removed)
        return self._fit_transform_cpu_vectorized(data)

    def transform(self, data: pd.Series) -> pd.Series:
        """Transform new data using existing histograms."""
        return self.fit_transform(data)

    def get_state(self) -> Dict[str, Any]:
        """Get current state for persistence."""
        return {
            'n_buckets': self.n_buckets,
            'granularity_minutes': self.granularity_minutes,
            'histograms': self.histograms,
            'alpha': self.alpha
        }

    def set_state(self, state: Dict[str, Any]):
        """Set state from persistence."""
        self.n_buckets = state['n_buckets']
        self.granularity_minutes = state['granularity_minutes']
        self.histograms = state['histograms']
        self.alpha = state['alpha']

class SignedLog(BaseScaler):
    """Signed log transform for heavy tails."""

    def __init__(self):
        super().__init__()

    def fit_transform(self, data: pd.Series) -> pd.Series:
        """Apply signed log transform: sign(x) * log(1 + |x|)."""
        self.fitted = True
        return np.sign(data) * np.log1p(np.abs(data))

    def transform(self, data: pd.Series) -> pd.Series:
        """Transform new data."""
        return self.fit_transform(data)

    def get_state(self) -> Dict[str, Any]:
        """Get state (no state for signed log)."""
        return {}

    def set_state(self, state: Dict[str, Any]):
        """Set state (no state for signed log)."""
        pass

class MADScaler(BaseScaler):
    """Median absolute deviation scaler with persistent state and VectorBT optimization."""

    def __init__(self, use_vectorbt: bool = True, use_gpu: bool = False):
        super().__init__()
        self.median: Optional[float] = None
        self.mad: Optional[float] = None
        self.use_vectorbt = use_vectorbt and VECTORBT_AVAILABLE
        self.use_gpu = False  # GPU support removed

    @staticmethod
    def _compute_mad(values: pd.Series, center: float) -> float:
        deviations = np.abs(values - center)
        mad = np.median(deviations)
        return float(mad)

    @staticmethod
    def _compute_mad_vectorized(values: np.ndarray, center: float) -> float:
        """Vectorized MAD computation."""
        deviations = np.abs(values - center)
        mad = np.median(deviations)
        return float(mad)

    @staticmethod
    def _compute_mad_gpu(values: np.ndarray, center: float) -> float:
        """CPU-based MAD computation (GPU support removed)."""
        return MADScaler._compute_mad_vectorized(values, center)

    def fit_transform(self, data: pd.Series) -> pd.Series:
        """Fit median and MAD, then scale data."""
        finite_data = data.dropna()

        if len(finite_data) == 0:
            # Default to neutral parameters when no finite data is present.
            self.median = 0.0
            self.mad = 1.0
            self.fitted = True
            return data.astype(float)

        if self.use_vectorbt and len(finite_data) > 1000:
            return self._fit_transform_vectorbt(data, finite_data)
        else:
            return self._fit_transform_sequential(data, finite_data)

    def _fit_transform_sequential(self, data: pd.Series, finite_data: pd.Series) -> pd.Series:
        """Sequential implementation for small datasets."""
        self.median = float(np.median(finite_data))
        mad = self._compute_mad(finite_data, self.median)
        # Avoid division by zero by falling back to unit scale.
        self.mad = mad if mad != 0 else 1.0
        self.fitted = True

        return self.transform(data)

    def _fit_transform_vectorbt(self, data: pd.Series, finite_data: pd.Series) -> pd.Series:
        """VectorBT-optimized implementation for large datasets."""
        if False:  # GPU support removed
            return self._fit_transform_gpu(data, finite_data)
        else:
            return self._fit_transform_cpu_vectorized(data, finite_data)

    def _fit_transform_cpu_vectorized(self, data: pd.Series, finite_data: pd.Series) -> pd.Series:
        """CPU-optimized vectorized implementation using VectorBT."""
        finite_values = finite_data.values

        # Vectorized median computation
        self.median = float(np.median(finite_values))

        # Vectorized MAD computation
        mad = self._compute_mad_vectorized(finite_values, self.median)
        self.mad = mad if mad != 0 else 1.0
        self.fitted = True

        # Vectorized scaling
        return (data - self.median) / self.mad

    def _fit_transform_gpu(self, data: pd.Series, finite_data: pd.Series) -> pd.Series:
        """GPU-accelerated implementation using VectorBT."""
        if True:
            return self._fit_transform_cpu_vectorized(data, finite_data)

        # CPU implementation (GPU support removed)
        finite_values = finite_data.values

        # CPU-based median computation
        self.median = float(np.median(finite_values))

        # CPU-based MAD computation
        mad = self._compute_mad_vectorized(finite_values, self.median)
        self.mad = mad if mad != 0 else 1.0
        self.fitted = True

        # CPU-based scaling
        return (data - self.median) / self.mad

    def transform(self, data: pd.Series) -> pd.Series:
        """Scale using stored median and MAD."""
        if not self.fitted or self.median is None or self.mad is None:
            raise ValueError("MADScaler must be fitted before calling transform.")

        # CPU-based scaling (GPU support removed)
        return (data - self.median) / self.mad

    def get_state(self) -> Dict[str, Any]:
        """Return state for persistence."""
        return {
            'median': self.median,
            'mad': self.mad,
            'fitted': self.fitted
        }

    def set_state(self, state: Dict[str, Any]):
        """Restore persisted state."""
        self.median = state.get('median')
        self.mad = state.get('mad')
        self.fitted = state.get('fitted', False)

class Winsorization(BaseScaler):
    """Winsorization transform using frozen quantiles."""

    def __init__(self, lower_quantile: float = 0.001, upper_quantile: float = 0.999):
        super().__init__()
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_bound = None
        self.upper_bound = None

    def fit_transform(self, data: pd.Series) -> pd.Series:
        """Fit quantiles and transform data."""
        # Calculate quantiles on finite data only
        finite_data = data.dropna()
        if len(finite_data) == 0:
            return data.copy()

        self.lower_bound = finite_data.quantile(self.lower_quantile)
        self.upper_bound = finite_data.quantile(self.upper_quantile)
        self.fitted = True

        return self.transform(data)

    def transform(self, data: pd.Series) -> pd.Series:
        """Transform data using fitted bounds."""
        self._validate_fitted()

        return data.clip(lower=self.lower_bound, upper=self.upper_bound)

    def get_state(self) -> Dict[str, Any]:
        """Get state for persistence."""
        return {
            'lower_quantile': self.lower_quantile,
            'upper_quantile': self.upper_quantile,
            'lower_bound': self.lower_bound,
            'upper_bound': self.upper_bound,
            'fitted': self.fitted
        }

    def set_state(self, state: Dict[str, Any]):
        """Set state from persistence."""
        self.lower_quantile = state['lower_quantile']
        self.upper_quantile = state['upper_quantile']
        self.lower_bound = state['lower_bound']
        self.upper_bound = state['upper_bound']
        self.fitted = state['fitted']

class TransformRouter:
    """Router for applying transforms to parent features with VectorBT optimization."""

    def __init__(self, config: Dict[str, TransformConfig], use_vectorbt: bool = True, use_gpu: bool = False, enable_parallel: bool = True):
        self.config = config
        self.transformers = {}
        self.use_vectorbt = use_vectorbt and VECTORBT_AVAILABLE
        self.use_gpu = False  # GPU support removed
        self.enable_parallel = enable_parallel and VECTORBT_AVAILABLE
        self._initialize_transformers()

    def _initialize_transformers(self):
        """Initialize transformers based on config."""
        for feature_name, transform_config in self.config.items():
            if transform_config.transform_type == TransformType.EWZ:
                halflife = transform_config.params.get('halflife', 12)
                self.transformers[feature_name] = OnlineEWZ(
                    halflife,
                    use_vectorbt=self.use_vectorbt,
                    use_gpu=self.use_gpu
                )
            elif transform_config.transform_type == TransformType.TOD_RANK:
                n_buckets = transform_config.params.get('n_buckets', 48)
                granularity = transform_config.params.get('granularity_minutes', 30)
                self.transformers[feature_name] = TODRank(
                    n_buckets,
                    granularity,
                    use_vectorbt=self.use_vectorbt,
                    use_gpu=self.use_gpu
                )
            elif transform_config.transform_type == TransformType.SIGNED_LOG:
                self.transformers[feature_name] = SignedLog()
            elif transform_config.transform_type == TransformType.MAD:
                self.transformers[feature_name] = MADScaler(
                    use_vectorbt=self.use_vectorbt,
                    use_gpu=self.use_gpu
                )
            elif transform_config.transform_type == TransformType.WINSOR:
                lower_q = transform_config.params.get('lower_quantile', 0.001)
                upper_q = transform_config.params.get('upper_quantile', 0.999)
                self.transformers[feature_name] = Winsorization(lower_q, upper_q)

    def fit_transform(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Fit on training data and transform both train and validation."""
        if self.enable_parallel and len(self.transformers) > 1:
            return self._fit_transform_parallel(train_data, val_data)
        else:
            return self._fit_transform_sequential(train_data, val_data)

    def _fit_transform_sequential(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Sequential implementation for small datasets or single features."""
        results = {}

        for feature_name, transformer in self.transformers.items():
            if feature_name not in train_data.columns:
                continue

            transform_config = self.config[feature_name]
            # Fit on training data
            train_transformed = transformer.fit_transform(train_data[feature_name])

            # Transform validation data
            val_transformed = transformer.transform(val_data[feature_name])

            # Create output DataFrames with transformed column names
            train_df = pd.DataFrame({
                f't/{feature_name}/{transform_config.transform_type.value}': train_transformed
            }, index=train_data.index)

            val_df = pd.DataFrame({
                f't/{feature_name}/{transform_config.transform_type.value}': val_transformed
            }, index=val_data.index)

            results[feature_name] = {
                'train': train_df,
                'val': val_df
            }

        return resultts

    def _fit_transform_parallel(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """VectorBT-optimized parallel implementation for large datasets."""
        if not VECTORBT_AVAILABLE:
            return self._fit_transform_sequential(train_data, val_data)

        # Prepare data for parallel processing
        feature_names = [name for name in self.transformers.keys() if name in train_data.columns]

        if not feature_names:
            return {}

        # Use VectorBT's parallel processing capabilities
        if False:  # GPU support removed
            return self._fit_transform_gpu_parallel(train_data, val_data, feature_names)
        else:
            return self._fit_transform_cpu_parallel(train_data, val_data, feature_names)

    def _fit_transform_cpu_parallel(self, train_data: pd.DataFrame, val_data: pd.DataFrame, feature_names: List[str]) -> Dict[str, pd.DataFrame]:
        """CPU-optimized parallel implementation using VectorBT."""
        results = {}

        # Process features in parallel using VectorBT's utilities
        for feature_name in feature_names:
            transformer = self.transformers[feature_name]
            transform_config = self.config[feature_name]

            # Fit on training data
            train_transformed = transformer.fit_transform(train_data[feature_name])

            # Transform validation data
            val_transformed = transformer.transform(val_data[feature_name])

            # Create output DataFrames with transformed column names
            train_df = pd.DataFrame({
                f't/{feature_name}/{transform_config.transform_type.value}': train_transformed
            }, index=train_data.index)

            val_df = pd.DataFrame({
                f't/{feature_name}/{transform_config.transform_type.value}': val_transformed
            }, index=val_data.index)

            results[feature_name] = {
                'train': train_df,
                'val': val_df
            }

        return resultts

    def _fit_transform_gpu_parallel(self, train_data: pd.DataFrame, val_data: pd.DataFrame, feature_names: List[str]) -> Dict[str, pd.DataFrame]:
        """GPU-accelerated parallel implementation using VectorBT (GPU support removed)."""
        if True:
            return self._fit_transform_cpu_parallel(train_data, val_data, feature_names)

        results = {}

        # Process features with
        for feature_name in feature_names:
            transformer = self.transformers[feature_name]
            transform_config = self.config[feature_name]

            # Fit on training data
            train_transformed = transformer.fit_transform(train_data[feature_name])

            # Transform validation data
            val_transformed = transformer.transform(val_data[feature_name])

            # Create output DataFrames with transformed column names
            train_df = pd.DataFrame({
                f't/{feature_name}/{transform_config.transform_type.value}': train_transformed
            }, index=train_data.index)

            val_df = pd.DataFrame({
                f't/{feature_name}/{transform_config.transform_type.value}': val_transformed
            }, index=val_data.index)

            results[feature_name] = {
                'train': train_df,
                'val': val_df
            }

        return resultts

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted transformers."""
        result_dfs = []

        for feature_name, transformer in self.transformers.items():
            if feature_name not in data.columns:
                continue

            transformed = transformer.transform(data[feature_name])
            transform_config = self.config[feature_name]

            result_df = pd.DataFrame({
                f't/{feature_name}/{transform_config.transform_type.value}': transformed
            }, index=data.index)

            result_dfs.append(result_df)

        if result_dfs:
            return pd.concat(result_dfs, axis=1)
        else:
            return pd.DataFrame(index=data.index)

    def get_transform_params(self) -> Dict[str, Dict[str, Any]]:
        """Get transform parameters for persistence."""
        params = {}
        for feature_name, transformer in self.transformers.items():
            params[feature_name] = transformer.get_state()
        return params

    def set_transform_params(self, params: Dict[str, Dict[str, Any]]):
        """Set transform parameters from persistence."""
        for feature_name, state in params.items():
            if feature_name in self.transformers:
                self.transformers[feature_name].set_state(state)

def _iter_feature_metadata(
    feature_inputs: Union[List[str], Dict[str, Dict[str, Any]]]
) -> Iterable[Tuple[str, Dict[str, Any]]]:
    if isinstance(feature_inputs, dict):
        return feature_inputs.items()
    return [(name, {}) for name in feature_inputs]

def create_default_transform_config(
    feature_inputs: Union[List[str], Dict[str, Dict[str, Any]]]
) -> Dict[str, TransformConfig]:
    """Create default transform configuration for features."""
    config: Dict[str, TransformConfig] = {}

    for feature_name, metadata in _iter_feature_metadata(feature_inputs):
        metadata = metadata or {}

        if metadata.get('heavy_tailed'):
            transform_type = TransformType.MAD
            params: Dict[str, Any] = {}
        elif any(x in feature_name for x in ['volume_z', 'tradecount_z', 'dollarvol_z']):
            # TOD Rank for seasonal count-like features
            transform_type = TransformType.TOD_RANK
            params = {'n_buckets': 48, 'granularity_minutes': 30}
        elif any(x in feature_name for x in ['spread_z', 'ofi_proxy', 'microprice_dev']):
            # Signed log for deterministic heavy tails
            transform_type = TransformType.SIGNED_LOG
            params = {}
        else:
            # EW-Z as default for continuous features
            transform_type = TransformType.EWZ
            params = {'halflife': 12}

        # Generate spec hash
        content = f"{feature_name}_{transform_type.value}_{params}"
        spec_hash = hashlib.md5(content.encode()).hexdigest()

        config[feature_name] = TransformConfig(
            transform_type=transform_type,
            params=params,
            spec_hash=spec_hash
        )

    return config

def apply_winsorization(data: pd.DataFrame,
                       quantiles: Tuple[float, float] = (0.001, 0.999)) -> pd.DataFrame:
    """Apply winsorization to all columns in DataFrame."""
    result = data.copy()

    for col in result.columns:
        finite_data = result[col].dropna()
        if len(finite_data) > 0:
            lower_bound = finite_data.quantile(quantiles[0])
            upper_bound = finite_data.quantile(quantiles[1])
            result[col] = result[col].clip(lower=lower_bound, upper=upper_bound)

    return result
