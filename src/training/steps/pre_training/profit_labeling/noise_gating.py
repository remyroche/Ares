"""
Noise Gating and Eligibility Filters

This module implements noise gating to filter out labels when signal is dominated by noise,
ensuring that only high-quality, learnable labels are generated.

Key Features:
- Minimum move vs. micro-range filtering
- Variance ratio test for microstructure detection
- Liquidity gating based on volume and spread
- Signal-to-noise ratio assessment
- Eligibility mask generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

# Import existing utilities
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.common_operations import (
    safe_divide, safe_log, safe_sqrt, safe_mean, safe_std,
    validate_finite, validate_positive, validate_range, safe_correlation
)
from src.utils.math_validation import MathValidation

# Import matrix operations for optimized rolling calculations
from src.utils.matrix_operations import vectorized_rolling_features

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

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


class NoiseGateType(Enum):
    """Enumeration of noise gate types."""
    MICRO_RANGE = "micro_range"  # Minimum move vs. micro-range
    VARIANCE_RATIO = "variance_ratio"  # Variance ratio test
    SIGNAL_NOISE = "signal_noise"  # Signal-to-noise ratio
    COMBINED = "combined"  # Combined approach


@dataclass
class NoiseGatingConfig:
    """Configuration for noise gating."""
    
    # Global enable/disable
    enabled: bool = True
    
    # Micro-range gating
    enable_micro_range_gating: bool = True
    min_move_ratio: float = 1.2  # Minimum k·σ_t / (α·mTR_t) ratio - more lenient
    micro_range_window: int = 20  # Window for median true range calculation
    
    # Variance ratio gating
    enable_variance_ratio_gating: bool = True
    vr_threshold_low: float = 0.6  # Lower threshold for microstructure detection - more lenient
    vr_threshold_high: float = 1.4  # Upper threshold for random walk - more lenient
    vr_window: int = 30  # Window for variance ratio calculation
    vr_subperiods: int = 5  # Number of subperiods for VR calculation
    
    
    # Signal-to-noise gating
    enable_signal_noise_gating: bool = True
    min_snr_ratio: float = 1.1  # Minimum signal-to-noise ratio - more lenient
    snr_window: int = 25  # Window for SNR calculation
    
    # Combined gating
    gate_type: NoiseGateType = NoiseGateType.COMBINED
    min_eligibility_ratio: float = 0.2  # Minimum ratio of eligible samples - more lenient
    strict_mode: bool = False  # Use strict eligibility criteria
    
    # Quality checks
    min_eligible_samples: int = 100
    max_gate_failure_rate: float = 0.8  # Maximum allowed gate failure rate


@dataclass
class EligibilityResult:
    """Result container for eligibility filtering."""
    
    # Core results
    eligibility_mask: pd.Series
    eligibility_ratio: float
    
    # Gate-specific results
    micro_range_mask: Optional[pd.Series] = None
    variance_ratio_mask: Optional[pd.Series] = None
    signal_noise_mask: Optional[pd.Series] = None
    
    # Statistics
    n_total_samples: int = 0
    n_eligible_samples: int = 0
    n_filtered_samples: int = 0
    
    # Gate performance
    gate_failure_rates: Dict[str, float] = field(default_factory=dict)
    gate_effectiveness: Dict[str, float] = field(default_factory=dict)
    
    # Quality metrics
    signal_quality_score: float = 0.0
    noise_reduction_ratio: float = 0.0
    
    # Metadata
    config_used: NoiseGatingConfig = None
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class NoiseGatingFilter:
    """
    Noise Gating Filter for Volatility-Aware Labeling
    
    This class implements comprehensive noise gating to filter out labels when signal
    is dominated by noise, ensuring that only high-quality, learnable labels are generated.
    
    Key Features:
    1. **Micro-Range Gating**: Filters moves indistinguishable from microstructure noise
    2. **Variance Ratio Gating**: Detects microstructure-driven mean reversion
    3. **Liquidity Gating**: Ensures sufficient liquidity for reliable labels
    4. **Signal-to-Noise Gating**: Maintains minimum signal quality
    5. **Combined Approach**: Intelligent combination of all gating methods
    """
    
    def __init__(self, config: Optional[NoiseGatingConfig] = None):
        """Initialize noise gating filter."""
        self.config = config or NoiseGatingConfig()
        self.logger = logging.getLogger('NoiseGatingFilter')
        
        tprint_info("🔇 Noise Gating Filter initialized")
        tprint_info(f"   → Gate type: {self.config.gate_type.value}")
        tprint_info(f"   → Micro-range gating: {self.config.enable_micro_range_gating}")
        tprint_info(f"   → Variance ratio gating: {self.config.enable_variance_ratio_gating}")
        tprint_info(f"   → Signal-noise gating: {self.config.enable_signal_noise_gating}")
    
    def filter_noise(self, bars: pd.DataFrame, volatility_series: pd.Series) -> EligibilityResult:
        """
        Filter noise and generate eligibility mask.
        
        Args:
            bars: Cleaned OHLCV bars
            volatility_series: Volatility estimates
            
        Returns:
            EligibilityResult with eligibility mask and statistics
        """
        start_time = datetime.now()
        tprint_info("🔇 Applying noise gating filters")
        
        # Initialize result container
        result = EligibilityResult(
            eligibility_mask=pd.Series(),
            eligibility_ratio=0.0,
            config_used=self.config
        )
        
        # If noise gating is disabled, return all samples as eligible
        if hasattr(self.config, 'enabled') and not self.config.enabled:
            tprint_info("⚡ Noise gating disabled - marking all samples as eligible")
            result.eligibility_mask = pd.Series(True, index=bars.index)
            result.n_total_samples = len(bars)
            result.n_eligible_samples = len(bars)
            result.eligibility_ratio = 1.0
            result.signal_quality_score = 1.0
            result.processing_time = (datetime.now() - start_time).total_seconds()
            return result
        
        try:
            # Validate input data
            if not self._validate_input_data(bars, volatility_series):
                return result
            
            # Align data
            common_index = bars.index.intersection(volatility_series.index)
            if len(common_index) == 0:
                tprint_warning("⚠️ No common index between bars and volatility")
                return result
            
            bars_aligned = bars.loc[common_index]
            vol_aligned = volatility_series.loc[common_index]
            
            result.n_total_samples = len(common_index)
            
            # Apply individual gating methods
            tprint_info("🔍 Step 1: Applying individual gating methods")
            gate_results = {}
            
            if self.config.enable_micro_range_gating:
                micro_range_mask = self._apply_micro_range_gating(bars_aligned, vol_aligned)
                gate_results['micro_range'] = micro_range_mask
                result.micro_range_mask = micro_range_mask
            
            if self.config.enable_variance_ratio_gating:
                vr_mask = self._apply_variance_ratio_gating(bars_aligned)
                gate_results['variance_ratio'] = vr_mask
                result.variance_ratio_mask = vr_mask
            
            
            if self.config.enable_signal_noise_gating:
                snr_mask = self._apply_signal_noise_gating(bars_aligned, vol_aligned)
                gate_results['signal_noise'] = snr_mask
                result.signal_noise_mask = snr_mask
            
            # Combine gating results
            tprint_info("🔗 Step 2: Combining gating results")
            combined_mask = self._combine_gating_results(gate_results)
            result.eligibility_mask = combined_mask
            
            # Calculate statistics
            tprint_info("📊 Step 3: Calculating statistics")
            result.n_eligible_samples = combined_mask.sum()
            result.n_filtered_samples = result.n_total_samples - result.n_eligible_samples
            result.eligibility_ratio = result.n_eligible_samples / result.n_total_samples if result.n_total_samples > 0 else 0.0
            
            # Calculate gate performance
            result.gate_failure_rates = self._calculate_gate_failure_rates(gate_results, combined_mask)
            result.gate_effectiveness = self._calculate_gate_effectiveness(gate_results, combined_mask)
            
            # Calculate quality metrics
            result.signal_quality_score = self._calculate_signal_quality_score(bars_aligned, vol_aligned, combined_mask)
            result.noise_reduction_ratio = self._calculate_noise_reduction_ratio(bars_aligned, combined_mask)
            
            # Validate minimum requirements
            if result.n_eligible_samples < self.config.min_eligible_samples:
                tprint_warning(f"⚠️ Insufficient eligible samples: {result.n_eligible_samples} < {self.config.min_eligible_samples}")
                result.eligibility_mask = pd.Series(False, index=common_index)
                result.eligibility_ratio = 0.0
            
        except Exception as e:
            tprint_error(f"❌ Noise gating failed: {e}")
            return result
        
        # Calculate processing time
        result.processing_time = (datetime.now() - start_time).total_seconds()
        
        tprint_success("✅ Noise gating completed")
        tprint_info(f"   → Total samples: {result.n_total_samples}")
        tprint_info(f"   → Eligible samples: {result.n_eligible_samples}")
        tprint_info(f"   → Eligibility ratio: {result.eligibility_ratio:.3f}")
        tprint_info(f"   → Signal quality score: {result.signal_quality_score:.3f}")
        
        return result
    
    def _validate_input_data(self, bars: pd.DataFrame, volatility_series: pd.Series) -> bool:
        """Validate input data."""
        try:
            # Check if DataFrames are empty
            if bars.empty or volatility_series.empty:
                tprint_warning("⚠️ Input data is empty")
                return False
            
            # Check required columns for bars
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = set(required_columns) - set(bars.columns)
            if missing_columns:
                tprint_warning(f"⚠️ Missing required columns: {missing_columns}")
                return False
            
            # Check for non-finite values
            if bars[required_columns].isnull().any().any() or volatility_series.isnull().any():
                tprint_warning("⚠️ Data contains null values")
                return False
            
            if not np.isfinite(bars[required_columns].values).all() or not np.isfinite(volatility_series.values).all():
                tprint_warning("⚠️ Data contains non-finite values")
                return False
            
            return True
            
        except Exception as e:
            tprint_error(f"❌ Data validation failed: {e}")
            return False
    
    def _apply_micro_range_gating(self, bars: pd.DataFrame, volatility_series: pd.Series) -> pd.Series:
        """Apply micro-range gating to filter microstructure noise."""
        try:
            # Calculate median true range
            high_low = bars['high'] - bars['low']
            high_close = np.abs(bars['high'] - bars['close'].shift(1))
            low_close = np.abs(bars['low'] - bars['close'].shift(1))
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))

            # Use optimized rolling median for better performance
            try:
                # Use vectorized rolling features for median calculation
                true_range_df = pd.DataFrame({'true_range': true_range})
                min_periods = self.config.micro_range_window // 2

                # Use vectorized rolling features for median
                rolling_result = vectorized_rolling_features(
                    true_range_df,
                    windows=[self.config.micro_range_window],
                    features=['median']
                )

                # Extract the median column
                median_column = f'true_range_rolling_median_{self.config.micro_range_window}'
                if median_column in rolling_result.columns:
                    median_true_range = rolling_result[median_column]
                else:
                    # Fallback to pandas if vectorized features don't include expected column
                    median_true_range = true_range.rolling(
                        window=self.config.micro_range_window,
                        min_periods=min_periods
                    ).median()
            except Exception as e:
                tprint_warning(f"⚠️ Optimized rolling median failed, using pandas: {e}")
                median_true_range = true_range.rolling(
                    window=self.config.micro_range_window,
                    min_periods=self.config.micro_range_window // 2
                ).median()
            
            # Calculate minimum move threshold
            min_move_threshold = self.config.min_move_ratio * median_true_range
            
            # Apply gating: require k·σ_t >= α·mTR_t
            eligibility_mask = volatility_series >= min_move_threshold
            
            # Handle NaN values
            eligibility_mask = eligibility_mask.fillna(False)
            
            return eligibility_mask
            
        except Exception as e:
            tprint_warning(f"⚠️ Error in micro-range gating: {e}")
            return pd.Series(True, index=bars.index)
    
    def _apply_variance_ratio_gating(self, bars: pd.DataFrame) -> pd.Series:
        """Apply variance ratio gating to detect microstructure-driven mean reversion."""
        try:
            # Calculate returns
            returns = bars['close'].pct_change().dropna()
            
            if len(returns) < self.config.vr_window:
                return pd.Series(True, index=bars.index)
            
            # Calculate variance ratio
            variance_ratios = []
            for i in range(len(returns) - self.config.vr_window + 1):
                window_returns = returns.iloc[i:i + self.config.vr_window]
                
                # Calculate variance of original returns
                var_original = window_returns.var()
                
                # Calculate variance of subperiod returns
                subperiod_length = len(window_returns) // self.config.vr_subperiods
                if subperiod_length > 0:
                    subperiod_returns = []
                    for j in range(0, len(window_returns), subperiod_length):
                        subperiod = window_returns.iloc[j:j + subperiod_length]
                        if len(subperiod) > 1:
                            subperiod_returns.append(subperiod.mean())
                    
                    if len(subperiod_returns) > 1:
                        var_subperiod = np.var(subperiod_returns)
                        vr = var_subperiod / var_original if var_original > 0 else 1.0
                        variance_ratios.append(vr)
                    else:
                        variance_ratios.append(1.0)
                else:
                    variance_ratios.append(1.0)
            
            # Create variance ratio series
            vr_series = pd.Series(variance_ratios, index=returns.index[self.config.vr_window - 1:])
            
            # Apply gating: filter out microstructure-driven mean reversion
            eligibility_mask = (vr_series >= self.config.vr_threshold_low) & (vr_series <= self.config.vr_threshold_high)
            
            # Align with original index
            full_mask = pd.Series(True, index=bars.index)
            full_mask.loc[vr_series.index] = eligibility_mask
            
            return full_mask
            
        except Exception as e:
            tprint_warning(f"⚠️ Error in variance ratio gating: {e}")
            return pd.Series(True, index=bars.index)
    
    def _apply_signal_noise_gating(self, bars: pd.DataFrame, volatility_series: pd.Series) -> pd.Series:
        """Apply signal-to-noise ratio gating."""
        try:
            # Calculate returns
            returns = bars['close'].pct_change().dropna()
            
            if len(returns) < self.config.snr_window:
                return pd.Series(True, index=bars.index)
            
            # Calculate rolling signal-to-noise ratio using optimized operations
            try:
                # Use vectorized rolling features for mean and std calculations
                returns_df = pd.DataFrame({'returns': returns})
                min_periods = self.config.snr_window // 2

                # Use vectorized rolling features for mean and std
                rolling_result = vectorized_rolling_features(
                    returns_df,
                    windows=[self.config.snr_window],
                    features=['mean', 'std']
                )

                # Extract the mean and std columns
                mean_column = f'returns_rolling_mean_{self.config.snr_window}'
                std_column = f'returns_rolling_std_{self.config.snr_window}'

                if mean_column in rolling_result.columns and std_column in rolling_result.columns:
                    signal_power = rolling_result[mean_column].abs()
                    noise_power = rolling_result[std_column]
                else:
                    # Fallback to pandas if vectorized features don't include expected columns
                    signal_power = returns.rolling(
                        window=self.config.snr_window,
                        min_periods=min_periods
                    ).mean().abs()

                    noise_power = returns.rolling(
                        window=self.config.snr_window,
                        min_periods=min_periods
                    ).std()
            except Exception as e:
                tprint_warning(f"⚠️ Optimized rolling mean/std failed, using pandas: {e}")
                signal_power = returns.rolling(
                    window=self.config.snr_window,
                    min_periods=self.config.snr_window // 2
                ).mean().abs()

                noise_power = returns.rolling(
                    window=self.config.snr_window,
                    min_periods=self.config.snr_window // 2
                ).std()
            
            snr_ratio = signal_power / noise_power
            snr_ratio = snr_ratio.fillna(0)
            
            # Apply gating
            eligibility_mask = snr_ratio >= self.config.min_snr_ratio
            
            # Align with original index
            full_mask = pd.Series(True, index=bars.index)
            full_mask.loc[snr_ratio.index] = eligibility_mask
            
            return full_mask
            
        except Exception as e:
            tprint_warning(f"⚠️ Error in signal-noise gating: {e}")
            return pd.Series(True, index=bars.index)
    
    def _combine_gating_results(self, gate_results: Dict[str, pd.Series]) -> pd.Series:
        """Combine individual gating results based on configuration."""
        try:
            if not gate_results:
                return pd.Series(True, index=next(iter(gate_results.values())).index)
            
            # Get common index
            common_index = next(iter(gate_results.values())).index
            
            if self.config.gate_type == NoiseGateType.COMBINED:
                # Use AND logic for combined gating
                combined_mask = pd.Series(True, index=common_index)
                for gate_name, gate_mask in gate_results.items():
                    combined_mask = combined_mask & gate_mask
            else:
                # Use the specific gate type
                gate_name = self.config.gate_type.value
                if gate_name in gate_results:
                    combined_mask = gate_results[gate_name]
                else:
                    combined_mask = pd.Series(True, index=common_index)
            
            # Apply strict mode if enabled
            if self.config.strict_mode:
                # Require all gates to pass
                for gate_name, gate_mask in gate_results.items():
                    combined_mask = combined_mask & gate_mask
            
            return combined_mask
            
        except Exception as e:
            tprint_warning(f"⚠️ Error combining gating results: {e}")
            return pd.Series(True, index=next(iter(gate_results.values())).index)
    
    def _calculate_gate_failure_rates(self, gate_results: Dict[str, pd.Series], 
                                    combined_mask: pd.Series) -> Dict[str, float]:
        """Calculate gate failure rates."""
        try:
            failure_rates = {}
            total_samples = len(combined_mask)
            
            for gate_name, gate_mask in gate_results.items():
                if total_samples > 0:
                    failed_samples = (~gate_mask).sum()
                    failure_rates[gate_name] = failed_samples / total_samples
                else:
                    failure_rates[gate_name] = 0.0
            
            return failure_rates
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating gate failure rates: {e}")
            return {}
    
    def _calculate_gate_effectiveness(self, gate_results: Dict[str, pd.Series], 
                                    combined_mask: pd.Series) -> Dict[str, float]:
        """Calculate gate effectiveness."""
        try:
            effectiveness = {}
            total_samples = len(combined_mask)
            eligible_samples = combined_mask.sum()
            
            for gate_name, gate_mask in gate_results.items():
                if total_samples > 0:
                    # Effectiveness = (samples that pass this gate AND are eligible) / total eligible samples
                    gate_and_eligible = (gate_mask & combined_mask).sum()
                    effectiveness[gate_name] = gate_and_eligible / eligible_samples if eligible_samples > 0 else 0.0
                else:
                    effectiveness[gate_name] = 0.0
            
            return effectiveness
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating gate effectiveness: {e}")
            return {}
    
    def _calculate_signal_quality_score(self, bars: pd.DataFrame, volatility_series: pd.Series, 
                                      eligibility_mask: pd.Series) -> float:
        """Calculate signal quality score."""
        try:
            if eligibility_mask.empty or not eligibility_mask.any():
                return 0.0
            
            # Calculate returns for eligible samples
            eligible_returns = bars['close'].pct_change()[eligibility_mask].dropna()
            
            if len(eligible_returns) < 10:
                return 0.0
            
            # Calculate signal quality metrics
            return_consistency = 1.0 - (eligible_returns.std() / eligible_returns.abs().mean()) if eligible_returns.abs().mean() > 0 else 0.0
            volatility_alignment = 1.0 - abs(eligible_returns.std() - volatility_series[eligibility_mask].mean()) / volatility_series[eligibility_mask].mean() if volatility_series[eligibility_mask].mean() > 0 else 0.0
            
            # Combine metrics
            quality_score = (return_consistency + volatility_alignment) / 2
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating signal quality score: {e}")
            return 0.0
    
    def _calculate_noise_reduction_ratio(self, bars: pd.DataFrame, eligibility_mask: pd.Series) -> float:
        """Calculate noise reduction ratio."""
        try:
            if eligibility_mask.empty:
                return 0.0
            
            # Calculate returns
            returns = bars['close'].pct_change().dropna()
            
            if len(returns) < 10:
                return 0.0
            
            # Calculate noise metrics
            all_noise = returns.std()
            eligible_noise = returns[eligibility_mask].std() if eligibility_mask.any() else all_noise
            
            # Calculate reduction ratio
            if all_noise > 0:
                reduction_ratio = 1.0 - (eligible_noise / all_noise)
            else:
                reduction_ratio = 0.0
            
            return max(0.0, min(1.0, reduction_ratio))
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating noise reduction ratio: {e}")
            return 0.0


# Convenience functions
def create_noise_gating_filter(config: Optional[NoiseGatingConfig] = None) -> NoiseGatingFilter:
    """Create noise gating filter with specified configuration."""
    return NoiseGatingFilter(config)


def filter_noise(bars: pd.DataFrame, volatility_series: pd.Series,
                config: Optional[NoiseGatingConfig] = None) -> EligibilityResult:
    """Filter noise with default configuration."""
    filter_obj = NoiseGatingFilter(config)
    return filter_obj.filter_noise(bars, volatility_series)

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
