"""
import warnings
Autoencoder Feature Generator

This module provides feature generators for autoencoder-based indicators,
including encoded features, reconstruction error, and deep learning features.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory
# Optimization utilities
try:
    from ..utils.vectorization_optimizer import get_vectorization_optimizer
    from ..utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

except ImportError:

    cp = None

from ..base_calculations import (
    BaseCalculationType,
    create_base_calculator
)

class AutoencoderFeatureGenerator(VectorizedFeatureGenerator):
    """
    Feature generator for autoencoder-based features.

    This generator creates various autoencoder indicators including:
    - Encoded features (latent representations)
    - Reconstruction error
    - Deep learning features
    - Dimensionality reduction features
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize the autoencoder feature generator.

        Args:
            config: Feature configuration (uses default if None)
        """
        if config is None:
            config = self._create_default_config()

        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        """Create default configuration for autoencoder features."""
        return FeatureConfig(
            name="autoencoder_features",
            category=FeatureCategory.AUTOENCODER,
            description="Comprehensive autoencoder features including encoded features and reconstruction error",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=20,
            min_lookback=1,
            max_lookback=100,
            parameters={
                "encoding_dimensions": [10, 20, 30],
                "reconstruction_windows": [5, 10, 20],
                "autoencoder_windows": [5, 10, 20]
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )

    @classmethod
    def create_default(cls) -> 'AutoencoderFeatureGenerator':
        return cls()

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        # Placeholder implementation
        close_prices = data['close'].values
        autoencoder = np.zeros_like(close_prices)
        return pd.Series(autoencoder, index=data.index, name='autoencoder_placeholder')

# Autoencoder Encoded Feature Generator

class AutoencoderEncodedGenerator(VectorizedFeatureGenerator):
    """Generator for autoencoder encoded features."""

    def __init__(self,
                 encoding_dimension: int = 10,
                 window: int = 20,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize autoencoder encoded generator.

        Args:
            encoding_dimension: Dimension of the encoded representation
            window: Rolling window for autoencoder calculations
            base_calculation: Base calculation type
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)

        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"autoencoder_encoded_{encoding_dimension}_{window}_{base_calculation.value}",
            category=FeatureCategory.AUTOENCODER,
            description=f"Autoencoder encoded feature {encoding_dimension} over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={
                'encoding_dimension': encoding_dimension,
                'window': window,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.encoding_dimension = encoding_dimension
        self.window = window
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate autoencoder encoded feature."""
        base_values = self.base_calculator.calculate(data)

        # Simulate autoencoder encoding using PCA-like transformation
        # In practice, this would use a trained autoencoder model
        encoded_feature = base_values.rolling(window=self.window).apply(
            lambda x: np.mean(x) + np.std(x) * np.sin(self.encoding_dimension * np.pi * x.index / len(x))
        )

        return encoded_feature

# Autoencoder Reconstruction Error Generator

class AutoencoderReconstructionErrorGenerator(VectorizedFeatureGenerator):
    """Generator for autoencoder reconstruction error features."""

    def __init__(self,
                 window: int = 20,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize autoencoder reconstruction error generator.

        Args:
            window: Rolling window for reconstruction error calculations
            base_calculation: Base calculation type
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)

        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"autoencoder_reconstruction_error_{window}_{base_calculation.value}",
            category=FeatureCategory.AUTOENCODER,
            description=f"Autoencoder reconstruction error over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={
                'window': window,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate autoencoder reconstruction error."""
        base_values = self.base_calculator.calculate(data)

        # Simulate reconstruction error using rolling mean as reconstruction
        # In practice, this would use a trained autoencoder model
        reconstruction = base_values.rolling(window=self.window).mean()
        reconstruction_error = np.abs(base_values - reconstruction)

        return reconstruction_error

# Autoencoder Reconstruction Error MA Generator

class AutoencoderReconstructionErrorMAGenerator(VectorizedFeatureGenerator):
    """Generator for autoencoder reconstruction error moving average features."""

    def __init__(self,
                 window: int = 20,
                 ma_window: int = 10,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize autoencoder reconstruction error MA generator.

        Args:
            window: Rolling window for reconstruction error calculations
            ma_window: Moving average window
            base_calculation: Base calculation type
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)

        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"autoencoder_reconstruction_error_ma_{window}_{ma_window}_{base_calculation.value}",
            category=FeatureCategory.AUTOENCODER,
            description=f"Autoencoder reconstruction error MA over {window} periods with MA {ma_window} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={
                'window': window,
                'ma_window': ma_window,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.ma_window = ma_window
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate autoencoder reconstruction error MA."""
        base_values = self.base_calculator.calculate(data)

        # Calculate reconstruction error
        reconstruction = base_values.rolling(window=self.window).mean()
        reconstruction_error = np.abs(base_values - reconstruction)

        # Calculate moving average of reconstruction error
        reconstruction_error_ma = reconstruction_error.rolling(window=self.ma_window).mean()

        return reconstruction_error_ma

# Autoencoder Reconstruction Error Std Generator

class AutoencoderReconstructionErrorStdGenerator(VectorizedFeatureGenerator):
    """Generator for autoencoder reconstruction error standard deviation features."""

    def __init__(self,
                 window: int = 20,
                 std_window: int = 10,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize autoencoder reconstruction error std generator.

        Args:
            window: Rolling window for reconstruction error calculations
            std_window: Standard deviation window
            base_calculation: Base calculation type
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)

        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"autoencoder_reconstruction_error_std_{window}_{std_window}_{base_calculation.value}",
            category=FeatureCategory.AUTOENCODER,
            description=f"Autoencoder reconstruction error std over {window} periods with std {std_window} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={
                'window': window,
                'std_window': std_window,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.std_window = std_window
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate autoencoder reconstruction error std."""
        base_values = self.base_calculator.calculate(data)

        # Calculate reconstruction error
        reconstruction = base_values.rolling(window=self.window).mean()
        reconstruction_error = np.abs(base_values - reconstruction)

        # Calculate standard deviation of reconstruction error
        reconstruction_error_std = reconstruction_error.rolling(window=self.std_window).std()

        return reconstruction_error_std

# Autoencoder Reconstruction Error Skew Generator

class AutoencoderReconstructionErrorSkewGenerator(VectorizedFeatureGenerator):
    """Generator for autoencoder reconstruction error skewness features."""

    def __init__(self,
                 window: int = 20,
                 skew_window: int = 10,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize autoencoder reconstruction error skew generator.

        Args:
            window: Rolling window for reconstruction error calculations
            skew_window: Skewness window
            base_calculation: Base calculation type
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)

        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"autoencoder_reconstruction_error_skew_{window}_{skew_window}_{base_calculation.value}",
            category=FeatureCategory.AUTOENCODER,
            description=f"Autoencoder reconstruction error skew over {window} periods with skew {skew_window} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={
                'window': window,
                'skew_window': skew_window,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.skew_window = skew_window
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate autoencoder reconstruction error skew."""
        base_values = self.base_calculator.calculate(data)

        # Calculate reconstruction error
        reconstruction = base_values.rolling(window=self.window).mean()
        reconstruction_error = np.abs(base_values - reconstruction)

        # Calculate skewness of reconstruction error
        reconstruction_error_skew = reconstruction_error.rolling(window=self.skew_window).skew()

        return reconstruction_error_skew

# Autoencoder Reconstruction Error Kurtosis Generator

class AutoencoderReconstructionErrorKurtosisGenerator(VectorizedFeatureGenerator):
    """Generator for autoencoder reconstruction error kurtosis features."""

    def __init__(self,
                 window: int = 20,
                 kurtosis_window: int = 10,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize autoencoder reconstruction error kurtosis generator.

        Args:
            window: Rolling window for reconstruction error calculations
            kurtosis_window: Kurtosis window
            base_calculation: Base calculation type
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)

        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"autoencoder_reconstruction_error_kurtosis_{window}_{kurtosis_window}_{base_calculation.value}",
            category=FeatureCategory.AUTOENCODER,
            description=f"Autoencoder reconstruction error kurtosis over {window} periods with kurtosis {kurtosis_window} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={
                'window': window,
                'kurtosis_window': kurtosis_window,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.kurtosis_window = kurtosis_window
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate autoencoder reconstruction error kurtosis."""
        base_values = self.base_calculator.calculate(data)

        # Calculate reconstruction error
        reconstruction = base_values.rolling(window=self.window).mean()
        reconstruction_error = np.abs(base_values - reconstruction)

        # Calculate kurtosis of reconstruction error
        reconstruction_error_kurtosis = reconstruction_error.rolling(window=self.kurtosis_window).kurt()

        return reconstruction_error_kurtosis

# Autoencoder Reconstruction Error Ratio Generator

class AutoencoderReconstructionErrorRatioGenerator(VectorizedFeatureGenerator):
    """Generator for autoencoder reconstruction error ratio features."""

    def __init__(self,
                 window: int = 20,
                 ratio_window: int = 10,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize autoencoder reconstruction error ratio generator.

        Args:
            window: Rolling window for reconstruction error calculations
            ratio_window: Ratio window
            base_calculation: Base calculation type
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)

        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"autoencoder_reconstruction_error_ratio_{window}_{ratio_window}_{base_calculation.value}",
            category=FeatureCategory.AUTOENCODER,
            description=f"Autoencoder reconstruction error ratio over {window} periods with ratio {ratio_window} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={
                'window': window,
                'ratio_window': ratio_window,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.ratio_window = ratio_window
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate autoencoder reconstruction error ratio."""
        base_values = self.base_calculator.calculate(data)

        # Calculate reconstruction error
        reconstruction = base_values.rolling(window=self.window).mean()
        reconstruction_error = np.abs(base_values - reconstruction)

        # Calculate ratio of reconstruction error to its moving average
        reconstruction_error_ma = reconstruction_error.rolling(window=self.ratio_window).mean()
        reconstruction_error_ratio = reconstruction_error / reconstruction_error_ma.replace(0, 1)

        return reconstruction_error_ratio

# Autoencoder Reconstruction Error Diff Generator

class AutoencoderReconstructionErrorDiffGenerator(VectorizedFeatureGenerator):
    """Generator for autoencoder reconstruction error difference features."""

    def __init__(self,
                 window: int = 20,
                 diff_window: int = 10,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize autoencoder reconstruction error diff generator.

        Args:
            window: Rolling window for reconstruction error calculations
            diff_window: Difference window
            base_calculation: Base calculation type
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)

        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"autoencoder_reconstruction_error_diff_{window}_{diff_window}_{base_calculation.value}",
            category=FeatureCategory.AUTOENCODER,
            description=f"Autoencoder reconstruction error diff over {window} periods with diff {diff_window} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={
                'window': window,
                'diff_window': diff_window,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.diff_window = diff_window
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate autoencoder reconstruction error diff."""
        base_values = self.base_calculator.calculate(data)

        # Calculate reconstruction error
        reconstruction = base_values.rolling(window=self.window).mean()
        reconstruction_error = np.abs(base_values - reconstruction)

        # Calculate difference of reconstruction error
        reconstruction_error_diff = reconstruction_error.diff(periods=self.diff_window)

        return reconstruction_error_diff

# Autoencoder Reconstruction Error Product Generator

class AutoencoderReconstructionErrorProductGenerator(VectorizedFeatureGenerator):
    """Generator for autoencoder reconstruction error product features."""

    def __init__(self,
                 window: int = 20,
                 product_window: int = 10,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize autoencoder reconstruction error product generator.

        Args:
            window: Rolling window for reconstruction error calculations
            product_window: Product window
            base_calculation: Base calculation type
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)

        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"autoencoder_reconstruction_error_product_{window}_{product_window}_{base_calculation.value}",
            category=FeatureCategory.AUTOENCODER,
            description=f"Autoencoder reconstruction error product over {window} periods with product {product_window} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={
                'window': window,
                'product_window': product_window,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.product_window = product_window
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate autoencoder reconstruction error product."""
        base_values = self.base_calculator.calculate(data)

        # Calculate reconstruction error
        reconstruction = base_values.rolling(window=self.window).mean()
        reconstruction_error = np.abs(base_values - reconstruction)

        # Calculate product of reconstruction error with its lagged version
        reconstruction_error_lagged = reconstruction_error.shift(self.product_window)
        reconstruction_error_product = reconstruction_error * reconstruction_error_lagged

        return reconstruction_error_product

# Autoencoder Reconstruction Error Squared Generator

class AutoencoderReconstructionErrorSquaredGenerator(VectorizedFeatureGenerator):
    """Generator for autoencoder reconstruction error squared features."""

    def __init__(self,
                 window: int = 20,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize autoencoder reconstruction error squared generator.

        Args:
            window: Rolling window for reconstruction error calculations
            base_calculation: Base calculation type
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)

        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"autoencoder_reconstruction_error_squared_{window}_{base_calculation.value}",
            category=FeatureCategory.AUTOENCODER,
            description=f"Autoencoder reconstruction error squared over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={
                'window': window,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate autoencoder reconstruction error squared."""
        base_values = self.base_calculator.calculate(data)

        # Calculate reconstruction error
        reconstruction = base_values.rolling(window=self.window).mean()
        reconstruction_error = np.abs(base_values - reconstruction)

        # Calculate squared reconstruction error
        reconstruction_error_squared = reconstruction_error ** 2

        return reconstruction_error_squared

# Autoencoder Reconstruction Error Cubed Generator

class AutoencoderReconstructionErrorCubedGenerator(VectorizedFeatureGenerator):
    """Generator for autoencoder reconstruction error cubed features."""

    def __init__(self,
                 window: int = 20,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize autoencoder reconstruction error cubed generator.

        Args:
            window: Rolling window for reconstruction error calculations
            base_calculation: Base calculation type
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)

        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"autoencoder_reconstruction_error_cubed_{window}_{base_calculation.value}",
            category=FeatureCategory.AUTOENCODER,
            description=f"Autoencoder reconstruction error cubed over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={
                'window': window,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate autoencoder reconstruction error cubed."""
        base_values = self.base_calculator.calculate(data)

        # Calculate reconstruction error
        reconstruction = base_values.rolling(window=self.window).mean()
        reconstruction_error = np.abs(base_values - reconstruction)

        # Calculate cubed reconstruction error
        reconstruction_error_cubed = reconstruction_error ** 3

        return reconstruction_error_cubed

# Autoencoder Reconstruction Error Cross Timeframe Generator

class AutoencoderReconstructionErrorCrossTimeframeGenerator(VectorizedFeatureGenerator):
    """Generator for autoencoder reconstruction error cross timeframe features."""

    def __init__(self,
                 window: int = 20,
                 cross_timeframe: int = 5,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize autoencoder reconstruction error cross timeframe generator.

        Args:
            window: Rolling window for reconstruction error calculations
            cross_timeframe: Cross timeframe period
            base_calculation: Base calculation type
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)

        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"autoencoder_reconstruction_error_cross_timeframe_{window}_{cross_timeframe}_{base_calculation.value}",
            category=FeatureCategory.AUTOENCODER,
            description=f"Autoencoder reconstruction error cross timeframe over {window} periods with cross timeframe {cross_timeframe} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={
                'window': window,
                'cross_timeframe': cross_timeframe,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.cross_timeframe = cross_timeframe
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate autoencoder reconstruction error cross timeframe."""
        base_values = self.base_calculator.calculate(data)

        # Calculate reconstruction error
        reconstruction = base_values.rolling(window=self.window).mean()
        reconstruction_error = np.abs(base_values - reconstruction)

        # Calculate cross timeframe reconstruction error
        reconstruction_error_cross_timeframe = reconstruction_error.rolling(window=self.cross_timeframe).mean()

        return reconstruction_error_cross_timeframe

# Autoencoder Reconstruction Error Regime Generator

class AutoencoderReconstructionErrorRegimeGenerator(VectorizedFeatureGenerator):
    """Generator for autoencoder reconstruction error regime features."""

    def __init__(self,
                 window: int = 20,
                 regime_window: int = 10,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize autoencoder reconstruction error regime generator.

        Args:
            window: Rolling window for reconstruction error calculations
            regime_window: Regime window
            base_calculation: Base calculation type
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)

        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"autoencoder_reconstruction_error_regime_{window}_{regime_window}_{base_calculation.value}",
            category=FeatureCategory.AUTOENCODER,
            description=f"Autoencoder reconstruction error regime over {window} periods with regime {regime_window} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={
                'window': window,
                'regime_window': regime_window,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.regime_window = regime_window
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate autoencoder reconstruction error regime."""
        base_values = self.base_calculator.calculate(data)

        # Calculate reconstruction error
        reconstruction = base_values.rolling(window=self.window).mean()
        reconstruction_error = np.abs(base_values - reconstruction)

        # Calculate regime-based reconstruction error
        reconstruction_error_regime = reconstruction_error.rolling(window=self.regime_window).apply(
            lambda x: np.mean(x) + np.std(x) * np.sin(2 * np.pi * x.index / len(x))
        )

        return reconstruction_error_regime

# Autoencoder Reconstruction Error Interaction Generator

class AutoencoderReconstructionErrorInteractionGenerator(VectorizedFeatureGenerator):
    """Generator for autoencoder reconstruction error interaction features."""

    def __init__(self,
                 window: int = 20,
                 interaction_window: int = 10,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize autoencoder reconstruction error interaction generator.

        Args:
            window: Rolling window for reconstruction error calculations
            interaction_window: Interaction window
            base_calculation: Base calculation type
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)

        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"autoencoder_reconstruction_error_interaction_{window}_{interaction_window}_{base_calculation.value}",
            category=FeatureCategory.AUTOENCODER,
            description=f"Autoencoder reconstruction error interaction over {window} periods with interaction {interaction_window} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={
                'window': window,
                'interaction_window': interaction_window,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.interaction_window = interaction_window
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate autoencoder reconstruction error interaction."""
        base_values = self.base_calculator.calculate(data)

        # Calculate reconstruction error
        reconstruction = base_values.rolling(window=self.window).mean()
        reconstruction_error = np.abs(base_values - reconstruction)

        # Calculate interaction-based reconstruction error
        reconstruction_error_interaction = reconstruction_error.rolling(window=self.interaction_window).apply(
            lambda x: np.mean(x) * np.std(x) * np.cos(2 * np.pi * x.index / len(x))
        )

        return reconstruction_error_interaction

def create_autoencoder_generators(encoding_dimensions: List[int] = None, windows: List[int] = None) -> List[FeatureGenerator]:
    """Create a set of autoencoder feature generators."""
    if encoding_dimensions is None:
        encoding_dimensions = [10, 20, 30]
    if windows is None:
        windows = [5, 10, 20]

    generators = []

    # Create encoded feature generators
    for encoding_dim in encoding_dimensions:
        for window in windows:
            generators.append(AutoencoderEncodedGenerator(encoding_dim, window))

    # Create reconstruction error generators
    for window in windows:
        generators.extend([
            AutoencoderReconstructionErrorGenerator(window),
            AutoencoderReconstructionErrorMAGenerator(window, 10),
            AutoencoderReconstructionErrorStdGenerator(window, 10),
            AutoencoderReconstructionErrorSkewGenerator(window, 10),
            AutoencoderReconstructionErrorKurtosisGenerator(window, 10),
            AutoencoderReconstructionErrorRatioGenerator(window, 10),
            AutoencoderReconstructionErrorDiffGenerator(window, 10),
            AutoencoderReconstructionErrorProductGenerator(window, 10),
            AutoencoderReconstructionErrorSquaredGenerator(window),
            AutoencoderReconstructionErrorCubedGenerator(window),
            AutoencoderReconstructionErrorCrossTimeframeGenerator(window, 5),
            AutoencoderReconstructionErrorRegimeGenerator(window, 10),
            AutoencoderReconstructionErrorInteractionGenerator(window, 10)
        ])

    return generators

def create_default_autoencoder_generators() -> List[FeatureGenerator]:
    """Create default autoencoder feature generators."""
    return create_autoencoder_generators()
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and
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
