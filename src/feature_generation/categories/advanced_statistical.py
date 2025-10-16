"""
Advanced Statistical Features Module

This module provides comprehensive advanced statistical feature generators
for quantitative finance, including sophisticated statistical indicators
and risk measures.

Key Features:
- Advanced statistical indicators (Hurst exponent, jump indicators, CVaR, drawdown measures)
- Risk measures and tail risk analysis
- Distribution analysis and statistical tests
- Full VectorBT integration for optimal performance
"""

# Standard library imports
import warnings
from typing import Any, Dict, List, Optional, Union, Tuple

# Third-party imports
import numpy as np
import pandas as pd

# Optional third-party imports
try:
    from scipy import stats
    from scipy.signal import find_peaks
    from scipy.stats import skew, kurtosis, jarque_bera
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None
    find_peaks = None
    skew = None
    kurtosis = None
    jarque_bera = None
    warnings.warn("SciPy not available. Some advanced statistical features may not work properly")

try:
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    StandardScaler = None
    warnings.warn("Scikit-learn not available. Some ML features may not work properly")

try:
    import vectorbt as vbt
    from vectorbt.generic import (
        rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max,
        rolling_sum, rolling_apply, rolling_corr, rolling_cov,
        rolling_skew, rolling_kurt, rolling_quantile
    )
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
    rolling_skew = None
    rolling_kurt = None
    rolling_quantile = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Local imports
from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

# Optimization utilities
try:
    from src.feature_generation.utils.vectorization_optimizer import get_vectorization_optimizer
    from src.feature_generation.utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    get_vectorization_optimizer = None
    get_optimized_feature_pipeline = None

try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
    VECTORBT_ROLLING_OPTIMIZER_AVAILABLE = True
except ImportError:
    VECTORBT_ROLLING_OPTIMIZER_AVAILABLE = False
    VectorBTRollingOptimizer = None

# Import tprint for consistent logging
try:
    from src.utils.tprint import tprint
except ImportError:
    def tprint(*args, **kwargs):
        print(*args, **kwargs)

class HurstExponentGenerator(VectorizedFeatureGenerator):
    """
    Generator for Hurst exponent features using VectorBT optimization.

    The Hurst exponent is a statistical measure used to analyze long-range dependence
    in time series data. It helps identify whether a time series is trending, mean-reverting,
    or random walk.

    Hurst Exponent Interpretation:
    - H > 0.5: Persistent/trending behavior (long memory)
    - H = 0.5: Random walk (no memory)
    - H < 0.5: Mean-reverting behavior (anti-persistent)

    Parameters:
    - window: Lookback window for calculation (default: 50)
    - min_periods: Minimum periods required for valid calculation (default: 20)

    Returns:
    - pd.Series: Hurst exponent values (0.0 to 1.0)

    Example:
        >>> generator = HurstExponentGenerator(window=30)
        >>> hurst_values = generator._generate_feature(data)
        >>> print(f"Average Hurst exponent: {hurst_values.mean():.3f}")
    """

    def __init__(self, window: int = 50, min_periods: int = 20):
        config = FeatureConfig(
            name="hurst_exponent",
            category=FeatureCategory.STATISTICAL,
            description="Hurst exponent for long-range dependence analysis",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=min_periods,
            max_lookback=200,
            parameters={"window": window, "min_periods": min_periods}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.min_periods = min_periods

        # Initialize VectorBT optimizer
        self.vectorbt_optimizer = None
        if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            try:
                self.vectorbt_optimizer = get_vectorbt_rolling_optimizer()
            except Exception as e:
                tprint(f"⚠️ VectorBT optimizer initialization failed: {e}")

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Hurst exponent feature."""
        close = data['close']

        if VECTORBT_AVAILABLE and self.vectorbt_optimizer:
            try:
                # Use VectorBT rolling apply for Hurst exponent calculation
                hurst_series = self.vectorbt_optimizer.rolling_apply(
                    close,
                    self._calculate_hurst_exponent,
                    window=self.window,
                    min_periods=self.min_periods
                )
                return hurst_series
            except Exception as e:
                warnings.warn(f"VectorBT Hurst exponent calculation failed: {e}, using fallback")
                return self._calculate_hurst_fallback(close, self.window, self.min_periods, data.index)
        else:
            return self._calculate_hurst_fallback(close, self.window, self.min_periods, data.index)

    def _calculate_hurst_exponent(self, segment: np.ndarray) -> float:
        """Calculate Hurst exponent for a segment."""
        return self._hurst_exponent(segment)

    def _calculate_hurst_fallback(self, close: pd.Series, window: int, min_periods: int, index: pd.Index) -> pd.Series:
        """Fallback Hurst exponent calculation using pandas rolling."""
        hurst_values = []
        for i in range(len(close)):
            if i < min_periods - 1:
                hurst_values.append(np.nan)
            else:
                segment = close.iloc[i-window+1:i+1].values
                hurst = self._hurst_exponent(segment)
                hurst_values.append(hurst)

        return pd.Series(hurst_values, index=index)

    def _hurst_exponent(self, data: np.ndarray) -> float:
        """Calculate Hurst exponent using R/S analysis."""
        try:
            if len(data) < 10:
                return 0.5

            # Remove NaN values
            data = data[~np.isnan(data)]
            if len(data) < 10:
                return 0.5

            # Calculate returns
            returns = np.diff(np.log(data))
            if len(returns) < 5:
                return 0.5

            # R/S analysis
            n = len(returns)
            mean_return = np.mean(returns)
            deviations = returns - mean_return
            cumulative_deviations = np.cumsum(deviations)

            # Calculate range
            R = np.max(cumulative_deviations) - np.min(cumulative_deviations)

            # Calculate standard deviation
            S = np.std(returns)

            if S == 0 or R == 0:
                return 0.5

            # R/S ratio
            rs_ratio = R / S

            # Hurst exponent
            hurst = np.log(rs_ratio) / np.log(n)

            # Clamp to reasonable range
            return max(0.0, min(1.0, hurst))

        except Exception:
            return 0.5

class JumpIndicatorsGenerator(VectorizedFeatureGenerator):
    """
    Generator for jump detection indicators using VectorBT optimization.

    Jump indicators help identify sudden price movements that may indicate
    market stress, news events, or structural changes in market dynamics.

    Parameters:
    - window: Lookback window for calculation (default: 20)
    - threshold: Threshold for jump detection (default: 3.0)

    Returns:
    - pd.Series: Binary jump indicators (0 or 1)

    Example:
        >>> generator = JumpIndicatorsGenerator(window=15, threshold=2.5)
        >>> jumps = generator._generate_feature(data)
        >>> print(f"Jump frequency: {jumps.mean():.3f}")
    """

    def __init__(self, window: int = 20, threshold: float = 3.0):
        config = FeatureConfig(
            name="jump_indicators",
            category=FeatureCategory.STATISTICAL,
            description="Jump detection indicators for volatility analysis",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=10,
            max_lookback=100,
            parameters={"window": window, "threshold": threshold}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.threshold = threshold

        # Initialize VectorBT optimizer
        self.vectorbt_optimizer = None
        if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            try:
                self.vectorbt_optimizer = get_vectorbt_rolling_optimizer()
            except Exception as e:
                tprint(f"⚠️ VectorBT optimizer initialization failed: {e}")

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate jump indicators feature."""
        close = data['close']

        if VECTORBT_AVAILABLE and self.vectorbt_optimizer:
            try:
                # Use VectorBT rolling apply for jump detection
                jump_series = self.vectorbt_optimizer.rolling_apply(
                    close,
                    self._detect_jumps,
                    window=self.window
                )
                return jump_series
            except Exception as e:
                warnings.warn(f"VectorBT jump detection failed: {e}, using fallback")
                return self._detect_jumps_fallback(close, self.window, self.threshold, data.index)
        else:
            return self._detect_jumps_fallback(close, self.window, self.threshold, data.index)

    def _detect_jumps(self, segment: np.ndarray) -> float:
        """Detect jumps in a segment."""
        return self._jump_indicator(segment, self.threshold)

    def _detect_jumps_fallback(self, close: pd.Series, window: int, threshold: float, index: pd.Index) -> pd.Series:
        """Fallback jump detection using pandas rolling."""
        jump_values = []
        for i in range(len(close)):
            if i < window - 1:
                jump_values.append(np.nan)
            else:
                segment = close.iloc[i-window+1:i+1].values
                jump_indicator = self._jump_indicator(segment, threshold)
                jump_values.append(jump_indicator)

        return pd.Series(jump_values, index=index)

    def _jump_indicator(self, data: np.ndarray, threshold: float) -> float:
        """Calculate jump indicator using Bipower Variation."""
        try:
            if len(data) < 5:
                return 0.0

            # Calculate returns
            returns = np.diff(np.log(data))
            if len(returns) < 3:
                return 0.0

            # Bipower variation
            abs_returns = np.abs(returns)
            bipower_variation = np.mean(abs_returns[:-1] * abs_returns[1:])

            # Realized variance
            realized_variance = np.mean(returns ** 2)

            # Jump test statistic
            if bipower_variation == 0:
                return 0.0

            jump_stat = (realized_variance - bipower_variation) / bipower_variation

            # Binary jump indicator
            return 1.0 if jump_stat > threshold else 0.0

        except Exception:
            return 0.0

class CVaRGenerator(VectorizedFeatureGenerator):
    """
    Generator for Conditional Value at Risk (CVaR) features.

    CVaR, also known as Expected Shortfall, measures the expected loss
    beyond the Value at Risk threshold, providing a more comprehensive
    measure of tail risk.

    Parameters:
    - window: Lookback window for calculation (default: 20)
    - confidence_level: Confidence level for VaR calculation (default: 0.05)

    Returns:
    - pd.Series: CVaR values

    Example:
        >>> generator = CVaRGenerator(window=30, confidence_level=0.01)
        >>> cvar = generator._generate_feature(data)
        >>> print(f"Average CVaR: {cvar.mean():.4f}")
    """

    def __init__(self, window: int = 20, confidence_level: float = 0.05):
        config = FeatureConfig(
            name="cvar",
            category=FeatureCategory.STATISTICAL,
            description="Conditional Value at Risk (CVaR) for tail risk analysis",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=10,
            max_lookback=100,
            parameters={"window": window, "confidence_level": confidence_level}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.confidence_level = confidence_level

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate CVaR feature."""
        close = data['close']
        returns = close.pct_change().dropna()

        if VECTORBT_AVAILABLE:
            try:
                # Use VectorBT rolling quantile for VaR calculation
                var_series = rolling_quantile(
                    returns,
                    window=self.window,
                    q=self.confidence_level
                )

                # Calculate CVaR as mean of returns below VaR
                cvar_series = returns.rolling(window=self.window).apply(
                    lambda x: self._calculate_cvar(x, self.confidence_level),
                    raw=False
                )

                return cvar_series
            except Exception as e:
                warnings.warn(f"VectorBT CVaR calculation failed: {e}, using fallback")
                return self._calculate_cvar_fallback(returns, self.window, self.confidence_level, data.index)
        else:
            return self._calculate_cvar_fallback(returns, self.window, self.confidence_level, data.index)

    def _calculate_cvar(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate CVaR for a series of returns."""
        try:
            if len(returns) < 5:
                return np.nan

            # Calculate VaR
            var = np.percentile(returns, confidence_level * 100)

            # Calculate CVaR as mean of returns below VaR
            tail_returns = returns[returns <= var]
            if len(tail_returns) == 0:
                return var

            return np.mean(tail_returns)

        except Exception:
            return np.nan

    def _calculate_cvar_fallback(self, returns: pd.Series, window: int, confidence_level: float, index: pd.Index) -> pd.Series:
        """Fallback CVaR calculation using pandas rolling."""
        cvar_values = []
        for i in range(len(returns)):
            if i < window - 1:
                cvar_values.append(np.nan)
            else:
                segment = returns.iloc[i-window+1:i+1]
                cvar = self._calculate_cvar(segment, confidence_level)
                cvar_values.append(cvar)

        return pd.Series(cvar_values, index=index)

class MaxDrawdownGenerator(VectorizedFeatureGenerator):
    """
    Generator for maximum drawdown features.

    Maximum drawdown measures the largest peak-to-trough decline
    in a time series, providing a key risk metric for portfolio management.

    Parameters:
    - window: Lookback window for calculation (default: 50)

    Returns:
    - pd.Series: Maximum drawdown values (negative values)

    Example:
        >>> generator = MaxDrawdownGenerator(window=30)
        >>> drawdown = generator._generate_feature(data)
        >>> print(f"Average max drawdown: {drawdown.mean():.4f}")
    """

    def __init__(self, window: int = 50):
        config = FeatureConfig(
            name="max_drawdown",
            category=FeatureCategory.STATISTICAL,
            description="Maximum drawdown for risk analysis",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=20,
            max_lookback=200,
            parameters={"window": window}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate maximum drawdown feature."""
        close = data['close']

        if VECTORBT_AVAILABLE:
            try:
                # Use VectorBT rolling apply for drawdown calculation
                drawdown_series = close.rolling(window=self.window).apply(
                    lambda x: self._calculate_max_drawdown(x),
                    raw=False
                )
                return drawdown_series
            except Exception as e:
                warnings.warn(f"VectorBT drawdown calculation failed: {e}, using fallback")
                return self._calculate_drawdown_fallback(close, self.window, data.index)
        else:
            return self._calculate_drawdown_fallback(close, self.window, data.index)

    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown for a price series."""
        try:
            if len(prices) < 2:
                return 0.0

            # Calculate running maximum
            running_max = prices.expanding().max()

            # Calculate drawdown
            drawdown = (prices - running_max) / running_max

            # Return maximum drawdown (most negative value)
            return drawdown.min()

        except Exception:
            return 0.0

    def _calculate_drawdown_fallback(self, close: pd.Series, window: int, index: pd.Index) -> pd.Series:
        """Fallback drawdown calculation using pandas rolling."""
        drawdown_values = []
        for i in range(len(close)):
            if i < window - 1:
                drawdown_values.append(np.nan)
            else:
                segment = close.iloc[i-window+1:i+1]
                max_dd = self._calculate_max_drawdown(segment)
                drawdown_values.append(max_dd)

        return pd.Series(drawdown_values, index=index)

class RollingSkewnessKurtosisGenerator(VectorizedFeatureGenerator):
    """
    Generator for rolling skewness and kurtosis features.

    Skewness and kurtosis measure the asymmetry and tail heaviness
    of return distributions, providing insights into market behavior
    and risk characteristics.

    Parameters:
    - window: Lookback window for calculation (default: 20)

    Returns:
    - pd.Series: Combined skewness and kurtosis values

    Example:
        >>> generator = RollingSkewnessKurtosisGenerator(window=15)
        >>> skew_kurt = generator._generate_feature(data)
        >>> print(f"Average skewness+kurtosis: {skew_kurt.mean():.3f}")
    """

    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name="rolling_skewness_kurtosis",
            category=FeatureCategory.STATISTICAL,
            description="Rolling skewness and kurtosis for distribution analysis",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=10,
            max_lookback=100,
            parameters={"window": window}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate rolling skewness and kurtosis feature."""
        close = data['close']
        returns = close.pct_change().dropna()

        if VECTORBT_AVAILABLE:
            try:
                # Use VectorBT rolling skew and kurt
                skew_series = rolling_skew(returns, window=self.window)
                kurt_series = rolling_kurt(returns, window=self.window)

                # Combine skewness and kurtosis
                combined = (skew_series + kurt_series) / 2
                return combined
            except Exception as e:
                warnings.warn(f"VectorBT skewness/kurtosis calculation failed: {e}, using fallback")
                return self._calculate_skew_kurt_fallback(returns, self.window, data.index)
        else:
            return self._calculate_skew_kurt_fallback(returns, self.window, data.index)

    def _calculate_skew_kurt_fallback(self, returns: pd.Series, window: int, index: pd.Index) -> pd.Series:
        """Fallback skewness/kurtosis calculation using pandas rolling."""
        skew_kurt_values = []
        for i in range(len(returns)):
            if i < window - 1:
                skew_kurt_values.append(np.nan)
            else:
                segment = returns.iloc[i-window+1:i+1]
                if len(segment) >= 3:
                    skew_val = segment.skew()
                    kurt_val = segment.kurtosis()
                    combined = (skew_val + kurt_val) / 2
                else:
                    combined = np.nan
                skew_kurt_values.append(combined)

        return pd.Series(skew_kurt_values, index=index)

class TrendPersistenceGenerator(VectorizedFeatureGenerator):
    """
    Generator for trend persistence features.

    Trend persistence measures the degree to which price movements
    tend to continue in the same direction, indicating market momentum
    and trend strength.

    Parameters:
    - window: Lookback window for calculation (default: 20)

    Returns:
    - pd.Series: Trend persistence values (-1 to 1)

    Example:
        >>> generator = TrendPersistenceGenerator(window=15)
        >>> persistence = generator._generate_feature(data)
        >>> print(f"Average trend persistence: {persistence.mean():.3f}")
    """

    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name="trend_persistence",
            category=FeatureCategory.STATISTICAL,
            description="Trend persistence analysis using autocorrelation",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=10,
            max_lookback=100,
            parameters={"window": window}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate trend persistence feature."""
        close = data['close']
        returns = close.pct_change().dropna()

        if VECTORBT_AVAILABLE:
            try:
                # Use VectorBT rolling correlation for autocorrelation
                autocorr_series = returns.rolling(window=self.window).apply(
                    lambda x: x.autocorr(lag=1) if len(x) > 1 else 0,
                    raw=False
                )
                return autocorr_series
            except Exception as e:
                warnings.warn(f"VectorBT trend persistence calculation failed: {e}, using fallback")
                return self._calculate_trend_persistence_fallback(returns, self.window, data.index)
        else:
            return self._calculate_trend_persistence_fallback(returns, self.window, data.index)

    def _calculate_trend_persistence_fallback(self, returns: pd.Series, window: int, index: pd.Index) -> pd.Series:
        """Fallback trend persistence calculation using pandas rolling."""
        persistence_values = []
        for i in range(len(returns)):
            if i < window - 1:
                persistence_values.append(np.nan)
            else:
                segment = returns.iloc[i-window+1:i+1]
                if len(segment) > 1:
                    autocorr = segment.autocorr(lag=1)
                    persistence_values.append(autocorr if not np.isnan(autocorr) else 0)
                else:
                    persistence_values.append(0)

        return pd.Series(persistence_values, index=index)

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_advanced_statistical_generators() -> List[FeatureGenerator]:
    """Create all advanced statistical feature generators."""
    generators = []

    for window in [20, 30, 50]:
        generators.append(HurstExponentGenerator(window))
        generators.append(JumpIndicatorsGenerator(window))
        generators.append(CVaRGenerator(window))
        generators.append(MaxDrawdownGenerator(window))
        generators.append(RollingSkewnessKurtosisGenerator(window))
        generators.append(TrendPersistenceGenerator(window))

    return generators

def process_advanced_statistical_features_batch(data: pd.DataFrame,
                                             generators: Optional[List[FeatureGenerator]] = None,
                                             use_vectorbt: bool = True,
                                             **kwargs) -> pd.DataFrame:
    """
    Process advanced statistical features in batch using VectorBT optimizations.

    Args:
        data: Input OHLCV data
        generators: List of feature generators (uses default if None)
        use_vectorbt: Whether to use VectorBT batch processing
        **kwargs: Additional parameters

    Returns:
        DataFrame with generated advanced statistical features
    """
    if generators is None:
        generators = create_advanced_statistical_generators()

    if use_vectorbt and OPTIMIZATION_AVAILABLE:
        try:
            # Use unified optimization system for batch processing
            from src.feature_generation.utils.unified_optimization_system import get_unified_optimization_system
            unified_optimizer = get_unified_optimization_system()

            # Process features in batch
            result = unified_optimizer.process_features_batch(data, generators, **kwargs)
            return result

        except Exception as e:
            warnings.warn(f"VectorBT batch processing failed: {e}, using sequential processing")
            return _process_advanced_statistical_features_sequential(data, generators, **kwargs)
    else:
        return _process_advanced_statistical_features_sequential(data, generators, **kwargs)

def _process_advanced_statistical_features_sequential(data: pd.DataFrame,
                                                    generators: List[FeatureGenerator],
                                                    **kwargs) -> pd.DataFrame:
    """Process advanced statistical features sequentially (fallback)."""
    results = []

    for generator in generators:
        try:
            feature_result = generator._generate_feature(data, **kwargs)
            if not feature_result.empty:
                results.append(feature_result)
        except Exception as e:
            warnings.warn(f"Generator {generator.__class__.__name__} failed: {e}")
            continue

    if results:
        return pd.concat(results, axis=1)
    else:
        return pd.DataFrame(index=data.index)

__all__ = [
    'HurstExponentGenerator',
    'JumpIndicatorsGenerator',
    'CVaRGenerator',
    'MaxDrawdownGenerator',
    'RollingSkewnessKurtosisGenerator',
    'TrendPersistenceGenerator',
    'create_advanced_statistical_generators',
    'process_advanced_statistical_features_batch'
]
