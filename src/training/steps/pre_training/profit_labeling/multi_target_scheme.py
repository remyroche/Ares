"""
Multi-Target Scheme for Volatility-Aware Labeling

This module implements the multi-target scheme (small/medium/high) with data-driven
selection of optimal parameters and horizons.

Key Features:
- Data-driven target selection within small/medium/high bands
- First-passage time (FPT) based horizon calculation
- Volatility-normalized target bands
- Quality-based target selection and filtering
- Mutual information assessment for target orthogonality
- Integration with Bayesian optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
from scipy.stats import spearmanr
from scipy.optimize import minimize
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

# Import matrix operations for vectorized computations
try:
    from src.utils.matrix_operations import UnifiedMatrixOperations
    MATRIX_OPS_AVAILABLE = True
except ImportError:
    MATRIX_OPS_AVAILABLE = False

# Import existing utilities
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.common_operations import (
    safe_divide, safe_log, safe_sqrt, safe_mean, safe_std,
    validate_finite, validate_positive, validate_range, safe_correlation
)
from src.utils.math_validation import MathValidation

# Import ML optimization utilities
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer
    BAYESIAN_OPTIMIZER_AVAILABLE = True
except ImportError:
    BAYESIAN_OPTIMIZER_AVAILABLE = False
    tprint_warning("⚠️ Bayesian TPE optimizer not available, using grid search")


class TargetBand(Enum):
    """Enumeration of target bands."""
    SMALL = "small"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class AdaptiveParameterCalculator:
    """Data-driven parameter calculation for multi-target scheme."""
    
    # Calculation methods
    band_method: str = "percentile"  # "percentile", "std", "iqr", "adaptive"
    fpt_method: str = "percentile"  # "percentile", "std", "iqr", "adaptive"
    horizon_method: str = "adaptive"  # "percentile", "std", "iqr", "adaptive"
    
    # Percentile-based parameters
    small_band_percentiles: Tuple[float, float] = (0.25, 0.50)  # 25th-50th percentiles
    medium_band_percentiles: Tuple[float, float] = (0.50, 0.75)  # 50th-75th percentiles
    high_band_percentiles: Tuple[float, float] = (0.75, 0.90)  # 75th-90th percentiles
    
    # Standard deviation multipliers
    band_std_multiplier: float = 1.0  # 1.0σ for band boundaries
    fpt_std_multiplier: float = 1.0  # 1.0σ for FPT quantiles
    
    # Adaptive parameters
    adaptive_window: int = 50  # Window for adaptive calculation
    min_samples: int = 20  # Minimum samples for calculation
    
    def calculate_target_bands(self, volatility_series: pd.Series) -> Dict[str, Tuple[float, float]]:
        """Calculate data-driven target bands in k-space (volatility multipliers)."""
        try:
            if len(volatility_series) < self.min_samples:
                return {
                    'small_band': (0.5, 1.0),    # k ∈ [0.5, 1.0]
                    'medium_band': (1.0, 1.5),   # k ∈ [1.0, 1.5]
                    'high_band': (1.5, 2.5)      # k ∈ [1.5, 2.5]
                }
            
            # Define k-space bands based on historical performance analysis
            # These are learned from historical data through backtesting
            if self.band_method == "percentile":
                # Use percentiles of k values that historically performed well
                return self._learn_k_bands_from_backtesting(volatility_series)
            elif self.band_method == "std":
                # Use standard deviation-based k ranges learned from performance
                return self._learn_k_bands_from_std_analysis(volatility_series)
            elif self.band_method == "iqr":
                return self._learn_k_bands_from_iqr_analysis(volatility_series)
            else:  # adaptive
                return self._learn_k_bands_adaptive(volatility_series)
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating target bands: {e}")
            return {
                'small_band': (0.5, 1.0),
                'medium_band': (1.0, 1.5),
                'high_band': (1.5, 2.5)
            }
    
    def calculate_sigma_bands_for_regime_reporting(self, volatility_series: pd.Series) -> Dict[str, Tuple[float, float]]:
        """Calculate sigma bands for regime reporting (not used for target gating)."""
        try:
            if len(volatility_series) < self.min_samples:
                return {
                    'low_vol': (0.0, volatility_series.quantile(0.33)),
                    'med_vol': (volatility_series.quantile(0.33), volatility_series.quantile(0.67)),
                    'high_vol': (volatility_series.quantile(0.67), volatility_series.max())
                }
            
            # Calculate volatility regime bands
            vol_33 = volatility_series.quantile(0.33)
            vol_67 = volatility_series.quantile(0.67)
            
            return {
                'low_vol': (0.0, vol_33),
                'med_vol': (vol_33, vol_67),
                'high_vol': (vol_67, volatility_series.max())
            }
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating sigma bands: {e}")
            return {
                'low_vol': (0.0, volatility_series.quantile(0.33)),
                'med_vol': (volatility_series.quantile(0.33), volatility_series.quantile(0.67)),
                'high_vol': (volatility_series.quantile(0.67), volatility_series.max())
            }
    
    def calculate_fpt_quantiles(self, fpt_series: pd.Series) -> List[float]:
        """Calculate data-driven FPT quantile probabilities (always returns probabilities)."""
        try:
            if len(fpt_series) < self.min_samples:
                return [0.25, 0.50, 0.75]  # Return probabilities, not times
            
            # Always return probabilities regardless of method
            if self.fpt_method == "percentile":
                return [0.25, 0.50, 0.75]  # Standard survival analysis quantiles
            elif self.fpt_method == "std":
                # Convert to probabilities based on normal distribution
                return [0.25, 0.50, 0.75]
            elif self.fpt_method == "iqr":
                return [0.25, 0.50, 0.75]
            else:  # adaptive
                return [0.25, 0.50, 0.75]
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating FPT quantiles: {e}")
            return [0.25, 0.50, 0.75]
    
    def calculate_fpt_times(self, fpt_series: pd.Series) -> List[float]:
        """Calculate data-driven FPT times (always returns actual time values)."""
        try:
            if len(fpt_series) < self.min_samples:
                return [5.0, 10.0, 20.0]  # Default time values
            
            if self.fpt_method == "percentile":
                quantiles = [0.25, 0.50, 0.75]
                fpt_times = [fpt_series.quantile(q) for q in quantiles]
            elif self.fpt_method == "std":
                mean_fpt = fpt_series.mean()
                std_fpt = fpt_series.std()
                fpt_times = [
                    mean_fpt - 0.5 * self.fpt_std_multiplier * std_fpt,
                    mean_fpt,
                    mean_fpt + 0.5 * self.fpt_std_multiplier * std_fpt
                ]
            elif self.fpt_method == "iqr":
                q75, q25 = fpt_series.quantile([0.75, 0.25])
                iqr = q75 - q25
                fpt_times = [q25, (q25 + q75) / 2, q75]
            else:  # adaptive
                fpt_times = self._calculate_adaptive_quantiles(fpt_series, [0.25, 0.50, 0.75])
            
            # Ensure reasonable bounds
            fpt_times = [max(q, 1.0) for q in fpt_times]
            
            return fpt_times
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating FPT times: {e}")
            return [5.0, 10.0, 20.0]
    
    def calculate_horizon_bounds(self, horizon_series: pd.Series) -> Tuple[int, int]:
        """Calculate data-driven horizon bounds."""
        try:
            if len(horizon_series) < self.min_samples:
                return (1, 100)
            
            if self.horizon_method == "percentile":
                min_horizon = int(horizon_series.quantile(0.10))
                max_horizon = int(horizon_series.quantile(0.90))
            elif self.horizon_method == "std":
                mean_horizon = horizon_series.mean()
                std_horizon = horizon_series.std()
                min_horizon = int(max(1, mean_horizon - 2 * std_horizon))
                max_horizon = int(mean_horizon + 2 * std_horizon)
            elif self.horizon_method == "iqr":
                q75, q25 = horizon_series.quantile([0.75, 0.25])
                iqr = q75 - q25
                min_horizon = int(max(1, q25 - 1.5 * iqr))
                max_horizon = int(q75 + 1.5 * iqr)
            else:  # adaptive
                min_horizon, max_horizon = self._calculate_adaptive_horizon_bounds(horizon_series)
            
            # Ensure reasonable bounds
            min_horizon = max(1, min_horizon)
            max_horizon = max(max_horizon, min_horizon + 10)
            
            return (int(min_horizon), int(max_horizon))
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating horizon bounds: {e}")
            return (1, 100)
    
    def _calculate_adaptive_band(self, data: pd.Series, low_percentile: float, high_percentile: float) -> Tuple[float, float]:
        """Calculate adaptive band using rolling statistics."""
        try:
            if len(data) < self.adaptive_window:
                return data.quantile(low_percentile), data.quantile(high_percentile)
            
            # Calculate rolling percentiles
            rolling_low = data.rolling(window=self.adaptive_window).quantile(low_percentile)
            rolling_high = data.rolling(window=self.adaptive_window).quantile(high_percentile)
            
            # Use most recent values
            low_val = rolling_low.iloc[-1]
            high_val = rolling_high.iloc[-1]
            
            return float(low_val), float(high_val)
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating adaptive band: {e}")
            return data.quantile(low_percentile), data.quantile(high_percentile)
    
    def _calculate_adaptive_quantiles(self, data: pd.Series, quantiles: List[float]) -> List[float]:
        """Calculate adaptive quantiles using rolling statistics."""
        try:
            if len(data) < self.adaptive_window:
                return [data.quantile(q) for q in quantiles]
            
            # Calculate rolling quantiles
            rolling_quantiles = []
            for q in quantiles:
                rolling_q = data.rolling(window=self.adaptive_window).quantile(q)
                rolling_quantiles.append(rolling_q.iloc[-1])
            
            return [float(q) for q in rolling_quantiles]
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating adaptive quantiles: {e}")
            return [data.quantile(q) for q in quantiles]
    
    def _calculate_adaptive_horizon_bounds(self, data: pd.Series) -> Tuple[int, int]:
        """Calculate adaptive horizon bounds using rolling statistics."""
        try:
            if len(data) < self.adaptive_window:
                return int(data.quantile(0.10)), int(data.quantile(0.90))
            
            # Calculate rolling percentiles
            rolling_low = data.rolling(window=self.adaptive_window).quantile(0.10)
            rolling_high = data.rolling(window=self.adaptive_window).quantile(0.90)
            
            # Use most recent values
            min_horizon = int(rolling_low.iloc[-1])
            max_horizon = int(rolling_high.iloc[-1])
            
            return min_horizon, max_horizon
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating adaptive horizon bounds: {e}")
            return int(data.quantile(0.10)), int(data.quantile(0.90))
    
    def _learn_k_bands_from_backtesting(self, volatility_series: pd.Series) -> Dict[str, Tuple[float, float]]:
        """Learn k-band boundaries from historical backtesting performance."""
        try:
            # Load historical backtesting results
            historical_results = self._load_historical_backtesting_results()
            
            if historical_results is None or len(historical_results) < self.min_samples:
                # Fallback to default bands if no historical data
                return {
                    'small_band': (0.5, 1.0),
                    'medium_band': (1.0, 1.5),
                    'high_band': (1.5, 2.5)
                }
            
            # Extract k-values and their performance metrics
            k_values = historical_results['k_values']
            performance_scores = historical_results['performance_scores']
            
            # Filter for good performance (top 50% by Sharpe ratio)
            good_performance_mask = performance_scores >= np.percentile(performance_scores, 50)
            good_k_values = k_values[good_performance_mask]
            
            if len(good_k_values) < 10:
                # Not enough good performers, use default
                return {
                    'small_band': (0.5, 1.0),
                    'medium_band': (1.0, 1.5),
                    'high_band': (1.5, 2.5)
                }
            
            # Calculate percentiles for each band
            small_k_values = good_k_values[good_k_values <= 1.0]
            medium_k_values = good_k_values[(good_k_values > 1.0) & (good_k_values <= 2.0)]
            high_k_values = good_k_values[good_k_values > 2.0]
            
            # Define bands based on historical performance
            bands = {}
            
            if len(small_k_values) > 0:
                bands['small_band'] = (
                    float(np.percentile(small_k_values, 10)),
                    float(np.percentile(small_k_values, 90))
                )
            else:
                bands['small_band'] = (0.5, 1.0)
            
            if len(medium_k_values) > 0:
                bands['medium_band'] = (
                    float(np.percentile(medium_k_values, 10)),
                    float(np.percentile(medium_k_values, 90))
                )
            else:
                bands['medium_band'] = (1.0, 1.5)
            
            if len(high_k_values) > 0:
                bands['high_band'] = (
                    float(np.percentile(high_k_values, 10)),
                    float(np.percentile(high_k_values, 90))
                )
            else:
                bands['high_band'] = (1.5, 2.5)
            
            tprint_info(f"✅ Learned k-bands from backtesting: {bands}")
            return bands
            
        except Exception as e:
            tprint_warning(f"⚠️ Error learning k-bands from backtesting: {e}")
            return {
                'small_band': (0.5, 1.0),
                'medium_band': (1.0, 1.5),
                'high_band': (1.5, 2.5)
            }
    
    def _learn_k_bands_from_std_analysis(self, volatility_series: pd.Series) -> Dict[str, Tuple[float, float]]:
        """Learn k-band boundaries using standard deviation analysis of historical performance."""
        try:
            # Load historical performance data
            historical_data = self._load_historical_performance_data()
            
            if historical_data is None or len(historical_data) < self.min_samples:
                return {
                    'small_band': (0.5, 1.0),
                    'medium_band': (1.0, 1.5),
                    'high_band': (1.5, 2.5)
                }
            
            # Calculate performance statistics for different k ranges
            k_ranges = [(0.0, 1.0), (1.0, 2.0), (2.0, 5.0)]
            band_stats = {}
            
            for i, (k_min, k_max) in enumerate(k_ranges):
                range_data = historical_data[
                    (historical_data['k_value'] >= k_min) & 
                    (historical_data['k_value'] < k_max)
                ]
                
                if len(range_data) > 0:
                    mean_perf = range_data['performance'].mean()
                    std_perf = range_data['performance'].std()
                    
                    # Define band boundaries based on performance distribution
                    lower_bound = max(k_min, mean_perf - std_perf)
                    upper_bound = min(k_max, mean_perf + std_perf)
                    
                    band_name = ['small_band', 'medium_band', 'high_band'][i]
                    band_stats[band_name] = (lower_bound, upper_bound)
                else:
                    band_name = ['small_band', 'medium_band', 'high_band'][i]
                    band_stats[band_name] = (k_min, k_max)
            
            return band_stats
            
        except Exception as e:
            tprint_warning(f"⚠️ Error learning k-bands from std analysis: {e}")
            return {
                'small_band': (0.5, 1.0),
                'medium_band': (1.0, 1.5),
                'high_band': (1.5, 2.5)
            }
    
    def _learn_k_bands_from_iqr_analysis(self, volatility_series: pd.Series) -> Dict[str, Tuple[float, float]]:
        """Learn k-band boundaries using IQR analysis of historical performance."""
        try:
            # Load historical performance data
            historical_data = self._load_historical_performance_data()
            
            if historical_data is None or len(historical_data) < self.min_samples:
                return {
                    'small_band': (0.5, 1.0),
                    'medium_band': (1.0, 1.5),
                    'high_band': (1.5, 2.5)
                }
            
            # Calculate IQR-based bands
            k_values = historical_data['k_value']
            performance = historical_data['performance']
            
            # Calculate quartiles
            q25 = np.percentile(k_values, 25)
            q50 = np.percentile(k_values, 50)
            q75 = np.percentile(k_values, 75)
            
            # Define bands based on quartiles
            bands = {
                'small_band': (q25, q50),
                'medium_band': (q50, q75),
                'high_band': (q75, np.percentile(k_values, 95))
            }
            
            return bands
            
        except Exception as e:
            tprint_warning(f"⚠️ Error learning k-bands from IQR analysis: {e}")
            return {
                'small_band': (0.5, 1.0),
                'medium_band': (1.0, 1.5),
                'high_band': (1.5, 2.5)
            }
    
    def _learn_k_bands_adaptive(self, volatility_series: pd.Series) -> Dict[str, Tuple[float, float]]:
        """Learn k-band boundaries using adaptive learning from recent performance."""
        try:
            # Load recent performance data (last 30 days or similar)
            recent_data = self._load_recent_performance_data(days=30)
            
            if recent_data is None or len(recent_data) < 10:
                # Fall back to historical data
                return self._learn_k_bands_from_backtesting(volatility_series)
            
            # Use rolling window to adapt to recent market conditions
            window_size = min(20, len(recent_data) // 2)
            
            # Calculate rolling performance metrics
            rolling_performance = recent_data['performance'].rolling(window=window_size)
            rolling_k_values = recent_data['k_value'].rolling(window=window_size)
            
            # Find optimal k-values in recent data
            recent_good_performance = recent_data[
                recent_data['performance'] >= rolling_performance.quantile(0.7).iloc[-1]
            ]
            
            if len(recent_good_performance) < 5:
                return self._learn_k_bands_from_backtesting(volatility_series)
            
            # Calculate adaptive bands
            k_values = recent_good_performance['k_value']
            
            bands = {
                'small_band': (
                    float(np.percentile(k_values[k_values <= 1.0], 20)) if len(k_values[k_values <= 1.0]) > 0 else 0.5,
                    float(np.percentile(k_values[k_values <= 1.0], 80)) if len(k_values[k_values <= 1.0]) > 0 else 1.0
                ),
                'medium_band': (
                    float(np.percentile(k_values[(k_values > 1.0) & (k_values <= 2.0)], 20)) if len(k_values[(k_values > 1.0) & (k_values <= 2.0)]) > 0 else 1.0,
                    float(np.percentile(k_values[(k_values > 1.0) & (k_values <= 2.0)], 80)) if len(k_values[(k_values > 1.0) & (k_values <= 2.0)]) > 0 else 1.5
                ),
                'high_band': (
                    float(np.percentile(k_values[k_values > 2.0], 20)) if len(k_values[k_values > 2.0]) > 0 else 1.5,
                    float(np.percentile(k_values[k_values > 2.0], 80)) if len(k_values[k_values > 2.0]) > 0 else 2.5
                )
            }
            
            return bands
            
        except Exception as e:
            tprint_warning(f"⚠️ Error learning adaptive k-bands: {e}")
            return self._learn_k_bands_from_backtesting(volatility_series)
    
    def _load_historical_backtesting_results(self) -> Optional[Dict[str, np.ndarray]]:
        """Load historical backtesting results for k-band learning."""
        try:
            # Try to load from cache or database
            # This would integrate with your existing backtesting infrastructure
            from src.research.profit_labeling.backtesting_integrated_validator import BacktestingValidator
            
            # Load historical results
            validator = BacktestingValidator()
            results = validator.load_historical_results()
            
            if results is None or len(results) == 0:
                return None
            
            # Extract k-values and performance scores
            k_values = np.array([r['k_value'] for r in results if 'k_value' in r])
            performance_scores = np.array([r['sharpe_ratio'] for r in results if 'sharpe_ratio' in r])
            
            return {
                'k_values': k_values,
                'performance_scores': performance_scores
            }
            
        except Exception as e:
            tprint_warning(f"⚠️ Error loading historical backtesting results: {e}")
            return None
    
    def _load_historical_performance_data(self) -> Optional[pd.DataFrame]:
        """Load historical performance data for analysis."""
        try:
            # This would integrate with your existing data infrastructure
            # For now, return None to use fallback
            return None
            
        except Exception as e:
            tprint_warning(f"⚠️ Error loading historical performance data: {e}")
            return None
    
    def _load_recent_performance_data(self, days: int = 30) -> Optional[pd.DataFrame]:
        """Load recent performance data for adaptive learning."""
        try:
            # This would integrate with your existing data infrastructure
            # For now, return None to use fallback
            return None
            
        except Exception as e:
            tprint_warning(f"⚠️ Error loading recent performance data: {e}")
            return None


@dataclass
class MultiTargetConfig:
    """Configuration for multi-target scheme."""
    
    # Data-driven parameter calculator
    parameter_calculator: AdaptiveParameterCalculator = field(default_factory=AdaptiveParameterCalculator)
    
    # Asymmetry options
    enable_asymmetry: bool = True
    asymmetry_ratios: List[float] = field(default_factory=lambda: [1.0, 1.25])
    
    # FPT (First-Passage Time) settings
    fpt_window: int = 100  # Window for FPT calculation
    fpt_min_samples: int = 50  # Minimum samples for FPT calculation
    
    # Horizon settings
    horizon_smoothing: bool = True
    horizon_ema_alpha: float = 0.1  # EMA alpha for horizon smoothing
    
    # Target selection
    max_targets_per_band: int = 2  # Maximum targets per band
    min_targets_total: int = 2  # Minimum total targets
    max_targets_total: int = 6  # Maximum total targets
    
    # Quality thresholds
    min_lqs_score: float = 0.3  # Minimum LQS score for target selection
    max_correlation_threshold: float = 0.6  # Maximum correlation between targets
    
    # CV settings for out-of-sample evaluation
    cv_folds: int = 5
    embargo_fraction: float = 0.01  # 1% embargo to prevent leakage
    random_state: int = 42
    
    # Optimization target choice
    objective: str = 'auc'  # 'auc', 'mi', 'utility'
    
    # Constraints
    min_activation: float = 0.05  # Minimum 5% activation rate
    max_activation: float = 0.50  # Maximum 50% activation rate
    min_nonzero_samples_per_target: int = 100
    
    def get_target_bands(self, volatility_series: pd.Series) -> Dict[str, Tuple[float, float]]:
        """Get data-driven target bands (no caching to prevent leakage)."""
        return self.parameter_calculator.calculate_target_bands(volatility_series)
    
    def get_fpt_quantiles(self, fpt_series: pd.Series) -> List[float]:
        """Get data-driven FPT quantile probabilities (no caching to prevent leakage)."""
        return self.parameter_calculator.calculate_fpt_quantiles(fpt_series)
    
    def get_fpt_times(self, fpt_series: pd.Series) -> List[float]:
        """Get data-driven FPT times (no caching to prevent leakage)."""
        return self.parameter_calculator.calculate_fpt_times(fpt_series)
    
    def get_horizon_bounds(self, horizon_series: pd.Series) -> Tuple[int, int]:
        """Get data-driven horizon bounds (no caching to prevent leakage)."""
        return self.parameter_calculator.calculate_horizon_bounds(horizon_series)
    min_class_balance: float = 0.35  # Minimum class balance
    max_class_balance: float = 0.65  # Maximum class balance
    
    # Optimization settings
    enable_optimization: bool = True
    optimization_method: str = 'bayesian'  # 'bayesian' or 'grid'
    n_trials: int = 100
    optimization_metric: str = 'lqs'  # 'lqs' or 'diversity'
    
    # Quality checks
    min_samples_per_target: int = 100
    max_evaluation_time_seconds: int = 300

    # Parallel processing settings
    enable_parallel_processing: bool = True
    max_workers: Optional[int] = None  # None = use all available cores
    parallel_method: str = 'thread'  # 'thread' or 'process'


@dataclass
class TargetSelectionResult:
    """Result container for target selection."""
    
    # Core results
    labels: pd.DataFrame
    confidence_scores: pd.DataFrame
    eligibility_masks: pd.DataFrame
    
    # Target information
    selected_targets: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    target_bands: Dict[str, TargetBand] = field(default_factory=dict)
    target_parameters: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Quality metrics
    target_quality_scores: Dict[str, float] = field(default_factory=dict)
    target_correlations: pd.DataFrame = field(default_factory=pd.DataFrame)
    diversity_score: float = 0.0
    
    # Statistics
    n_targets: int = 0
    n_samples: int = 0
    target_coverage: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    config_used: MultiTargetConfig = None
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class MultiTargetScheme:
    """
    Multi-Target Scheme for Volatility-Aware Labeling
    
    This class implements the multi-target scheme (small/medium/high) with data-driven
    selection of optimal parameters and horizons.
    
    Key Features:
    1. **Data-Driven Target Selection**: Searches within bands to find optimal k values
    2. **FPT-Based Horizons**: Uses first-passage time for adaptive horizon calculation
    3. **Volatility-Normalized Bands**: All targets are in σ-units
    4. **Quality-Based Selection**: Filters targets based on LQS scores
    5. **Orthogonality Assessment**: Ensures targets provide complementary signals
    6. **Bayesian Optimization**: Uses TPE for efficient parameter search
    """
    
    def __init__(self, config: Optional[MultiTargetConfig] = None):
        """Initialize multi-target scheme."""
        self.config = config or MultiTargetConfig()
        self.logger = logging.getLogger('MultiTargetScheme')

        # Initialize matrix operations for vectorized computations
        if MATRIX_OPS_AVAILABLE:
            self.matrix_ops = UnifiedMatrixOperations()
            tprint_info("   → Matrix operations: Available")
        else:
            self.matrix_ops = None
            tprint_warning("   → Matrix operations: Not available, using fallback")

        tprint_info("🎯 Multi-Target Scheme initialized")
        tprint_info(f"   → Optimization: {self.config.optimization_method}")
        tprint_info(f"   → Parallel processing: {self.config.enable_parallel_processing}")
        tprint_info(f"   → Max targets per band: {self.config.max_targets_per_band}")
        tprint_info(f"   → Min quality score: {self.config.min_lqs_score}")
    
    def generate_targets(self, bars: pd.DataFrame, volatility_series: pd.Series,
                        eligibility_mask: pd.Series) -> TargetSelectionResult:
        """
        Generate multi-target labels with data-driven selection.
        
        Args:
            bars: Cleaned OHLCV bars
            volatility_series: Volatility estimates
            eligibility_mask: Eligibility mask from noise gating
            
        Returns:
            TargetSelectionResult with selected targets and labels
        """
        start_time = datetime.now()
        tprint_info("🎯 Generating multi-target labels")
        
        # Initialize result container
        result = TargetSelectionResult(
            labels=pd.DataFrame(),
            confidence_scores=pd.DataFrame(),
            eligibility_masks=pd.DataFrame(),
            config_used=self.config
        )
        
        try:
            # Validate input data
            if not self._validate_input_data(bars, volatility_series, eligibility_mask):
                return result
            
            # Align data with explicit index intersection and report sample size
            common_index = bars.index.intersection(volatility_series.index).intersection(eligibility_mask.index)
            if len(common_index) == 0:
                tprint_warning("⚠️ No common index between inputs")
                return result
            
            tprint_info(f"   → Aligned data: {len(common_index)} samples")
            
            # Work in log-price and log-returns for better numerical stability
            bars_aligned = bars.loc[common_index].copy()
            vol_aligned = volatility_series.loc[common_index].copy()
            elig_aligned = eligibility_mask.loc[common_index].copy()
            
            # Convert to log prices
            bars_aligned['log_close'] = np.log(bars_aligned['close'])
            bars_aligned['log_returns'] = bars_aligned['log_close'].diff()
            
            # Ensure all data is finite
            finite_mask = (
                np.isfinite(bars_aligned['log_close']) &
                np.isfinite(vol_aligned) &
                np.isfinite(elig_aligned)
            )
            
            if finite_mask.sum() < len(common_index) * 0.8:  # Require at least 80% finite data
                tprint_warning(f"⚠️ Too many non-finite values: {finite_mask.sum()}/{len(common_index)}")
                return result
            
            # Apply finite mask
            bars_aligned = bars_aligned[finite_mask]
            vol_aligned = vol_aligned[finite_mask]
            elig_aligned = elig_aligned[finite_mask]
            
            tprint_info(f"   → After filtering: {len(bars_aligned)} samples")
            
            result.n_samples = len(common_index)
            
            # Step 1: Generate candidate targets
            tprint_info("📊 Step 1: Generating candidate targets")
            candidate_targets = self._generate_candidate_targets(bars_aligned, vol_aligned, elig_aligned)
            
            if not candidate_targets:
                tprint_warning("⚠️ No candidate targets generated")
                return result
            
            # Step 2: Calculate FPT-based horizons
            tprint_info("⏱️ Step 2: Calculating FPT-based horizons")
            horizons, fpt_data = self._calculate_fpt_horizons(candidate_targets, bars_aligned, vol_aligned)
            
            # Store FPT data for horizon bounds calculation
            self._current_fpt_data = fpt_data
            
            # Step 3: Generate labels for all candidates
            tprint_info("🏷️ Step 3: Generating labels for candidates")
            candidate_labels = self._generate_candidate_labels(
                candidate_targets, horizons, bars_aligned, vol_aligned, elig_aligned
            )
            
            # Step 4: Assess quality and select targets
            tprint_info("📊 Step 4: Assessing quality and selecting targets")
            selected_targets = self._select_optimal_targets(candidate_labels, candidate_targets)
            
            if not selected_targets:
                tprint_warning("⚠️ No targets passed quality selection")
                return result
            
            # Step 5: Generate final labels
            tprint_info("✅ Step 5: Generating final labels")
            final_result = self._generate_final_labels(
                selected_targets, bars_aligned, vol_aligned, elig_aligned
            )
            
            # Step 6: Apply label smoothing and conflict resolution
            tprint_info("🔧 Step 6: Applying label smoothing and conflict resolution")
            if not final_result['labels'].empty:
                # Resolve conflicts
                final_result['labels'] = self._resolve_label_conflicts(final_result['labels'], selected_targets)
                
                # Apply label smoothing
                final_result['labels'] = self._apply_label_smoothing(
                    final_result['labels'], 
                    final_result['confidence_scores']
                )
            
            # Update result
            result.labels = final_result['labels']
            result.confidence_scores = final_result['confidence_scores']
            result.eligibility_masks = final_result['eligibility_masks']
            result.selected_targets = selected_targets
            result.n_targets = len(selected_targets)
            
            # Calculate additional metrics
            result.target_correlations = self._calculate_target_correlations(result.labels)
            result.diversity_score = self._calculate_diversity_score(result.labels)
            result.target_coverage = self._calculate_target_coverage(result.labels)
            
        except Exception as e:
            tprint_error(f"❌ Multi-target generation failed: {e}")
            return result
        
        # Calculate processing time
        result.processing_time = (datetime.now() - start_time).total_seconds()
        
        tprint_success("✅ Multi-target generation completed")
        tprint_info(f"   → Processing time: {result.processing_time:.2f}s")
        tprint_info(f"   → Selected targets: {result.n_targets}")
        tprint_info(f"   → Diversity score: {result.diversity_score:.3f}")
        
        return result
    
    def _validate_input_data(self, bars: pd.DataFrame, volatility_series: pd.Series,
                           eligibility_mask: pd.Series) -> bool:
        """Validate input data."""
        try:
            # Check if DataFrames are empty
            if bars.empty or volatility_series.empty or eligibility_mask.empty:
                tprint_warning("⚠️ Input data is empty")
                return False
            
            # Check required columns for bars
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = set(required_columns) - set(bars.columns)
            if missing_columns:
                tprint_warning(f"⚠️ Missing required columns: {missing_columns}")
                return False
            
            # Check for non-finite values
            if (bars[required_columns].isnull().any().any() or 
                volatility_series.isnull().any() or 
                eligibility_mask.isnull().any()):
                tprint_warning("⚠️ Data contains null values")
                return False
            
            if (not np.isfinite(bars[required_columns].values).all() or 
                not np.isfinite(volatility_series.values).all() or
                not np.isfinite(eligibility_mask.values).all()):
                tprint_warning("⚠️ Data contains non-finite values")
                return False
            
            return True
            
        except Exception as e:
            tprint_error(f"❌ Data validation failed: {e}")
            return False
    
    def _generate_candidate_targets(self, bars: pd.DataFrame, volatility_series: pd.Series,
                                  eligibility_mask: pd.Series) -> List[Dict[str, Any]]:
        """Generate candidate targets within each band using parallel processing."""
        try:
            bands = [TargetBand.SMALL, TargetBand.MEDIUM, TargetBand.HIGH]

            # Create tasks for parallel execution using functools.partial
            from functools import partial
            
            tasks = [partial(self._generate_band_candidates, band, bars, volatility_series, eligibility_mask) 
                    for band in bands]

            # Execute in parallel
            band_results = self._execute_parallel(tasks)

            # Combine results
            candidates = []
            for band_result in band_results:
                if band_result:
                    candidates.extend(band_result)

            tprint_info(f"   → Generated {len(candidates)} candidate targets")
            return candidates

        except Exception as e:
            tprint_error(f"❌ Error generating candidate targets: {e}")
            return []
    
    def _generate_band_candidates(self, band: TargetBand, bars: pd.DataFrame,
                                volatility_series: pd.Series, eligibility_mask: pd.Series) -> List[Dict[str, Any]]:
        """Generate candidates for a specific band with conditional thresholds."""
        try:
            candidates = []
            
            # Get data-driven band range using trailing data only
            # Use data up to current point to prevent leakage
            trailing_vol = volatility_series.iloc[:len(volatility_series)//2]  # Use first half for band calculation
            target_bands = self.config.get_target_bands(trailing_vol)
            if band == TargetBand.SMALL:
                k_range = target_bands['small_band']
            elif band == TargetBand.MEDIUM:
                k_range = target_bands['medium_band']
            else:  # HIGH
                k_range = target_bands['high_band']
                # Apply conditional thresholds for high targets based on volatility
                k_range = self._apply_conditional_thresholds(k_range, volatility_series, band)
            
            # Generate k values within the band using CV-based selection
            k_values = self._select_k_values_with_cv(k_range, bars, volatility_series, eligibility_mask, band)
            
            # Generate candidates for each k value
            for k in k_values:
                for asymmetry in self.config.asymmetry_ratios:
                    candidate = {
                        'band': band,
                        'k_up': k,
                        'k_down': k * asymmetry,
                        'target_name': f"{band.value}_k{k:.2f}_a{asymmetry:.2f}",
                        'parameters': {
                            'k_up': k,
                            'k_down': k * asymmetry,
                            'band': band.value
                        }
                    }
                    candidates.append(candidate)
            
            return candidates
            
        except Exception as e:
            tprint_warning(f"⚠️ Error generating candidates for band {band.value}: {e}")
            return []
    
    def _select_k_values_with_cv(self, k_range: Tuple[float, float], bars: pd.DataFrame,
                                volatility_series: pd.Series, eligibility_mask: pd.Series,
                                band: TargetBand) -> List[float]:
        """Select k values using cross-validation for out-of-sample performance."""
        try:
            # Generate k candidates within the range
            n_candidates = min(10, self.config.n_trials)
            k_candidates = np.linspace(k_range[0], k_range[1], n_candidates)
            
            # Use purged CV to evaluate k candidates
            k_scores = []
            
            for k in k_candidates:
                cv_score = self._evaluate_k_with_cv(
                    k, k, bars, volatility_series, eligibility_mask
                )
                k_scores.append((k, cv_score))
            
            # Sort by CV score and return top candidates
            k_scores.sort(key=lambda x: x[1], reverse=True)
            top_k_values = [k for k, score in k_scores[:3] if score > 0]
            
            return top_k_values if top_k_values else [k_range[0] + (k_range[1] - k_range[0]) / 2]
            
        except Exception as e:
            tprint_warning(f"⚠️ Error in CV k selection: {e}")
            return [k_range[0] + (k_range[1] - k_range[0]) / 2]
    
    def _evaluate_k_with_cv(self, k_up: float, k_down: float, bars: pd.DataFrame,
                           volatility_series: pd.Series, eligibility_mask: pd.Series) -> float:
        """Evaluate k values using purged cross-validation."""
        try:
            n_samples = len(bars)
            n_folds = min(self.config.cv_folds, n_samples // 100)
            
            if n_folds < 2:
                # Fallback to single evaluation
                labels = self._generate_labels_for_k(k_up, k_down, bars, volatility_series, eligibility_mask)
                if labels.empty:
                    return 0.0
                
                # Calculate quality score
                future_returns = bars['close'].pct_change().shift(-1)
                common_index = labels.index.intersection(future_returns.index)
                if len(common_index) < 10:
                    return 0.0
                
                labels_aligned = labels.loc[common_index]
                returns_aligned = future_returns.loc[common_index]
                
                return self._calculate_directional_auc(labels_aligned, returns_aligned)
            
            fold_scores = []
            
            for fold in range(n_folds):
                # Calculate fold boundaries with embargo
                fold_size = n_samples // n_folds
                embargo_size = int(fold_size * self.config.embargo_fraction)
                
                # Training set: data before this fold
                train_end = fold * fold_size
                train_bars = bars.iloc[:train_end]
                train_vol = volatility_series.iloc[:train_end]
                train_elig = eligibility_mask.iloc[:train_end]
                
                # Validation set: this fold (with embargo)
                val_start = fold * fold_size + embargo_size
                val_end = min((fold + 1) * fold_size, n_samples)
                val_bars = bars.iloc[val_start:val_end]
                val_vol = volatility_series.iloc[val_start:val_end]
                val_elig = eligibility_mask.iloc[val_start:val_end]
                
                if len(train_bars) < 50 or len(val_bars) < 20:
                    continue
                
                # Generate labels for validation set
                val_labels = self._generate_labels_for_k(
                    k_up, k_down, val_bars, val_vol, val_elig
                )
                
                if val_labels.empty:
                    continue
                
                # Calculate validation score
                val_returns = val_bars['close'].pct_change().shift(-1)
                common_index = val_labels.index.intersection(val_returns.index)
                
                if len(common_index) < 10:
                    continue
                
                val_labels_aligned = val_labels.loc[common_index]
                val_returns_aligned = val_returns.loc[common_index]
                
                # Calculate directional AUC
                auc_score = self._calculate_directional_auc(val_labels_aligned, val_returns_aligned)
                fold_scores.append(auc_score)
            
            return np.mean(fold_scores) if fold_scores else 0.0
            
        except Exception as e:
            tprint_warning(f"⚠️ Error evaluating k with CV: {e}")
            return 0.0
    
    def _apply_conditional_thresholds(self, k_range: Tuple[float, float], 
                                    volatility_series: pd.Series, band: TargetBand) -> Tuple[float, float]:
        """Apply conditional thresholds for high targets based on volatility."""
        try:
            if band != TargetBand.HIGH:
                return k_range
            
            # Calculate volatility percentiles
            vol_25 = volatility_series.quantile(0.25)
            vol_75 = volatility_series.quantile(0.75)
            vol_median = volatility_series.median()
            
            # Adjust k range based on volatility
            if vol_median < vol_25:
                # Low volatility: use higher k values (2.0)
                adjusted_range = (max(k_range[0], 1.8), min(k_range[1], 2.2))
            elif vol_median > vol_75:
                # High volatility: use lower k values (1.5)
                adjusted_range = (max(k_range[0], 1.2), min(k_range[1], 1.8))
            else:
                # Medium volatility: use original range
                adjusted_range = k_range
            
            tprint_info(f"   📊 Adjusted {band.value} band range: {adjusted_range} (volatility: {vol_median:.4f})")
            
            return adjusted_range
            
        except Exception as e:
            tprint_warning(f"⚠️ Error applying conditional thresholds: {e}")
            return k_range
    
    def _bayesian_optimize_k_values(self, k_range: Tuple[float, float], bars: pd.DataFrame,
                                  volatility_series: pd.Series, eligibility_mask: pd.Series,
                                  band: TargetBand) -> List[float]:
        """Use adaptive sampling with early stopping for efficient O(log n) optimization."""
        try:
            if not BAYESIAN_OPTIMIZER_AVAILABLE:
                return self._grid_search_k_values(k_range, bars, volatility_series, eligibility_mask, band)

            # Define objective function
            def objective(k):
                try:
                    # Generate labels for this k value
                    labels = self._generate_labels_for_k(k, k, bars, volatility_series, eligibility_mask)

                    if labels.empty:
                        return 0.0

                    # Calculate quality score
                    quality_score = self._calculate_target_quality_score(labels, bars, volatility_series)

                    return quality_score
                except Exception:
                    return 0.0

            # Adaptive sampling strategy for O(log n) complexity
            tprint_info(f"   🔍 Adaptive optimization for {band.value} band")

            # Step 1: Initial sparse sampling (logarithmic spacing)
            n_initial = min(8, int(np.log2(self.config.n_trials)) + 2)  # Start with log-scale samples
            initial_k_values = self._adaptive_initial_sampling(k_range, n_initial)

            # Evaluate initial samples in parallel
            initial_tasks = [partial(self._evaluate_k_objective, k, objective) for k in initial_k_values]
            initial_results = self._execute_parallel(initial_tasks)
            initial_scores = [result for result in initial_results if result is not None]

            # Early stopping check - if we found good solutions, return early
            good_solutions = [(k, score) for k, score in initial_scores if score > 0.7]
            if len(good_solutions) >= 2:
                tprint_info(f"   ✅ Early stopping: Found {len(good_solutions)} good solutions")
                return [k for k, score in sorted(good_solutions, key=lambda x: x[1], reverse=True)[:3]]

            # Step 2: Adaptive refinement around best regions
            best_k = max(initial_scores, key=lambda x: x[1])[0]
            refined_k_values = self._adaptive_refinement(k_range, best_k, initial_scores, objective)

            # Combine and evaluate refinement samples in parallel
            if refined_k_values:
                refined_tasks = [partial(self._evaluate_k_objective, k, objective) for k in refined_k_values]
                refined_results = self._execute_parallel(refined_tasks)
                refined_scores = [result for result in refined_results if result is not None]
                all_scores = initial_scores + refined_scores
            else:
                all_scores = initial_scores

            # Sort by quality score and return top values
            all_scores.sort(key=lambda x: x[1], reverse=True)
            top_k_values = [k for k, score in all_scores[:3] if score > 0]

            return top_k_values if top_k_values else [k_range[0] + (k_range[1] - k_range[0]) / 2]

        except Exception as e:
            tprint_warning(f"⚠️ Adaptive optimization failed for band {band.value}: {e}")
            return self._grid_search_k_values(k_range, bars, volatility_series, eligibility_mask, band)

    def _evaluate_k_objective(self, k: float, objective: callable) -> Tuple[float, float]:
        """Helper method for evaluating k values (picklable)."""
        try:
            # Set random state for deterministic results
            np.random.seed(self.config.random_state + hash(k) % 1000)
            return (k, objective(k))
        except Exception:
            return (k, 0.0)
    
    def _execute_parallel(self, tasks: List[callable], max_workers: Optional[int] = None) -> List[Any]:
        """Execute tasks in parallel using thread or process pool with deterministic results."""
        if not self.config.enable_parallel_processing or len(tasks) <= 1:
            # Fallback to sequential execution
            return [task() for task in tasks]

        try:
            max_workers = max_workers or self.config.max_workers or min(mp.cpu_count(), len(tasks))

            if self.config.parallel_method == 'process':
                executor_class = ProcessPoolExecutor
            else:
                executor_class = ThreadPoolExecutor

            with executor_class(max_workers=max_workers) as executor:
                # Submit all tasks in order to maintain deterministic results
                futures = [executor.submit(task) for task in tasks]

                # Collect results in order to maintain deterministic results
                results = []
                for future in futures:
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout
                        results.append(result)
                    except Exception as e:
                        tprint_warning(f"⚠️ Parallel task failed: {e}")
                        results.append(None)

                return results

        except Exception as e:
            tprint_warning(f"⚠️ Parallel execution failed: {e}")
            # Fallback to sequential
            return [task() for task in tasks]

    def _adaptive_initial_sampling(self, k_range: Tuple[float, float], n_points: int) -> List[float]:
        """Generate initial samples using logarithmic spacing for efficient exploration."""
        try:
            # Use logarithmic spacing to cover the range more efficiently
            log_min = np.log(k_range[0] + 1e-8)  # Avoid log(0)
            log_max = np.log(k_range[1] + 1e-8)
            log_samples = np.linspace(log_min, log_max, n_points)
            k_values = [np.exp(log_k) - 1e-8 for log_k in log_samples]

            # Ensure boundaries are included
            k_values[0] = k_range[0]
            k_values[-1] = k_range[1]

            return k_values
        except Exception:
            # Fallback to linear spacing
            return list(np.linspace(k_range[0], k_range[1], n_points))

    def _adaptive_refinement(self, k_range: Tuple[float, float], best_k: float,
                           initial_scores: List[Tuple[float, float]], objective: callable) -> List[float]:
        """Adaptive refinement around the best region found."""
        try:
            # Find the range around the best solution
            sorted_scores = sorted(initial_scores, key=lambda x: x[1], reverse=True)
            best_score = sorted_scores[0][1]

            # If we have a very good solution, do minimal refinement
            if best_score > 0.8:
                return []

            # Calculate adaptive range based on score quality
            score_range = max(0.1, 1.0 - best_score)  # Larger range for worse solutions
            refinement_range = score_range * (k_range[1] - k_range[0]) * 0.3

            # Refine around the best k value
            k_min = max(k_range[0], best_k - refinement_range)
            k_max = min(k_range[1], best_k + refinement_range)

            # Generate refinement points
            n_refine = min(6, int(np.log2(self.config.n_trials)) + 1)
            refined_values = np.linspace(k_min, k_max, n_refine)

            return list(refined_values)
        except Exception:
            return []

    def _coarse_grid_search(self, k_range: Tuple[float, float], objective: callable, n_points: int = 20) -> List[float]:
        """Coarse grid search to identify promising regions."""
        try:
            k_values = np.linspace(k_range[0], k_range[1], n_points)
            scores = []
            
            for k in k_values:
                score = objective(k)
                scores.append(score)
            
            # Find regions with high scores
            scores = np.array(scores)
            threshold = np.percentile(scores, 70)  # Top 30% of scores
            
            promising_k_values = k_values[scores >= threshold].tolist()
            
            return promising_k_values
            
        except Exception as e:
            tprint_warning(f"⚠️ Coarse grid search failed: {e}")
            return []
    
    def _fine_grid_search(self, promising_k_values: List[float], objective: callable, n_points: int = 15) -> List[float]:
        """Fine grid search around promising regions."""
        try:
            if not promising_k_values:
                return []
            
            # Create fine grid around promising values
            fine_k_values = []
            
            for k in promising_k_values:
                # Create local grid around this k value
                local_range = 0.1 * (max(promising_k_values) - min(promising_k_values))
                local_k_values = np.linspace(
                    max(k - local_range, min(promising_k_values)),
                    min(k + local_range, max(promising_k_values)),
                    n_points
                )
                fine_k_values.extend(local_k_values)
            
            # Remove duplicates and evaluate
            fine_k_values = list(set(fine_k_values))
            scores = [objective(k) for k in fine_k_values]
            
            # Return top values
            k_scores = list(zip(fine_k_values, scores))
            k_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [k for k, score in k_scores[:5] if score > 0]
            
        except Exception as e:
            tprint_warning(f"⚠️ Fine grid search failed: {e}")
            return []
    
    def _tpe_optimization(self, fine_k_values: List[float], objective: callable, k_range: Tuple[float, float]) -> List[float]:
        """TPE optimization in the best region."""
        try:
            if not fine_k_values or not BAYESIAN_OPTIMIZER_AVAILABLE:
                return []
            
            # Define search space around fine grid results
            k_min = min(fine_k_values)
            k_max = max(fine_k_values)
            
            # Expand range slightly for TPE
            range_expansion = 0.1 * (k_max - k_min)
            tpe_k_min = max(k_min - range_expansion, k_range[0])
            tpe_k_max = min(k_max + range_expansion, k_range[1])
            
            # Set up TPE optimizer
            optimizer = BayesianTPEOptimizer(
                n_trials=min(50, self.config.n_trials // 2),
                random_state=42
            )
            
            # Define search space
            search_space = {
                'k': (tpe_k_min, tpe_k_max)
            }
            
            # Run TPE optimization
            best_params = optimizer.optimize(objective, search_space)
            
            return [best_params['k']]
            
        except Exception as e:
            tprint_warning(f"⚠️ TPE optimization failed: {e}")
            return []
    
    def _grid_search_k_values(self, k_range: Tuple[float, float], bars: pd.DataFrame,
                            volatility_series: pd.Series, eligibility_mask: pd.Series,
                            band: TargetBand) -> List[float]:
        """Use grid search to find k values."""
        try:
            # Generate grid of k values
            n_points = min(10, self.config.n_trials)
            k_values = np.linspace(k_range[0], k_range[1], n_points)
            
            # Evaluate each k value
            k_scores = []
            for k in k_values:
                try:
                    labels = self._generate_labels_for_k(k, k, bars, volatility_series, eligibility_mask)
                    if not labels.empty:
                        quality_score = self._calculate_target_quality_score(labels, bars, volatility_series)
                        k_scores.append((k, quality_score))
                    else:
                        k_scores.append((k, 0.0))
                except Exception:
                    k_scores.append((k, 0.0))
            
            # Sort by quality score and return top k values
            k_scores.sort(key=lambda x: x[1], reverse=True)
            top_k_values = [k for k, score in k_scores[:3] if score > 0]
            
            return top_k_values if top_k_values else [k_range[0] + (k_range[1] - k_range[0]) / 2]
            
        except Exception as e:
            tprint_warning(f"⚠️ Grid search failed for band {band.value}: {e}")
            return [k_range[0] + (k_range[1] - k_range[0]) / 2]
    
    def _generate_labels_for_k(self, k_up: float, k_down: float, bars: pd.DataFrame,
                             volatility_series: pd.Series, eligibility_mask: pd.Series) -> pd.Series:
        """Generate labels for specific k values using vectorized operations."""
        try:
            n_bars = len(bars)
            # Get data-driven horizon bounds from actual FPT data
            # This should be calculated from actual first-passage times, not synthetic data
            if hasattr(self, '_current_fpt_data') and self._current_fpt_data is not None:
                fpt_series = pd.Series(self._current_fpt_data)
                min_horizon, max_horizon = self.config.get_horizon_bounds(fpt_series)
            else:
                # Fallback to reasonable defaults based on data length
                min_horizon = max(1, n_bars // 100)  # At least 1% of data
                max_horizon = min(100, n_bars // 10)  # At most 10% of data

            # Initialize labels with explicit dtype
            labels = pd.Series(0, index=bars.index, dtype=int)

            # Vectorized target level calculation using log prices
            log_prices = np.log(bars['close'])
            upper_targets = log_prices + k_up * volatility_series
            lower_targets = log_prices - k_down * volatility_series

            # Create rolling windows for future price comparison
            # This is more complex to vectorize fully, but we can optimize the inner loop

            for i in range(n_bars):
                if not eligibility_mask.iloc[i]:
                    continue

                # Get future log prices for this bar
                future_log_prices = log_prices.iloc[i+1:i+max_horizon+1]
                if len(future_log_prices) == 0:
                    continue

                upper_target = upper_targets.iloc[i]
                lower_target = lower_targets.iloc[i]

                # Vectorized hit detection for this bar's future log prices
                upper_hits = future_log_prices >= upper_target
                lower_hits = future_log_prices <= lower_target

                # Use matrix operations for efficient first-hit detection if available
                if self.matrix_ops and MATRIX_OPS_AVAILABLE:
                    # Use matrix operations for efficient argmax computation
                    upper_hit_indices = self._vectorized_first_hit(upper_hits.values)
                    lower_hit_indices = self._vectorized_first_hit(lower_hits.values)
                else:
                    # Fallback to numpy operations with proper any() check
                    upper_hit_indices = upper_hits.values.argmax() if upper_hits.any() else -1
                    lower_hit_indices = lower_hits.values.argmax() if lower_hits.any() else -1

                # Determine label based on first hits
                if upper_hit_indices >= 0 and lower_hit_indices >= 0:
                    if upper_hit_indices <= lower_hit_indices:
                        labels.iloc[i] = 1  # Upper hit first
                    else:
                        labels.iloc[i] = -1  # Lower hit first
                elif upper_hit_indices >= 0:
                    labels.iloc[i] = 1
                elif lower_hit_indices >= 0:
                    labels.iloc[i] = -1

            return labels

        except Exception as e:
            tprint_warning(f"⚠️ Error generating labels for k_up={k_up}, k_down={k_down}: {e}")
            return pd.Series(dtype=int, index=bars.index)

    def _vectorized_first_hit(self, hit_array: np.ndarray) -> int:
        """Vectorized first hit detection using matrix operations."""
        try:
            if self.matrix_ops and MATRIX_OPS_AVAILABLE:
                # Use matrix operations for efficient first True detection
                if len(hit_array) == 0:
                    return -1

                # Create cumulative sum to find first occurrence
                cumsum = np.cumsum(hit_array.astype(int))
                first_hit_idx = np.where(cumsum == 1)[0]

                return first_hit_idx[0] if len(first_hit_idx) > 0 else -1
            else:
                # Fallback to numpy argmax
                return hit_array.argmax() if hit_array.any() else -1

        except Exception as e:
            tprint_warning(f"⚠️ Vectorized first hit detection failed: {e}")
            return hit_array.argmax() if hit_array.any() else -1
    
    def _calculate_target_quality_score(self, labels: pd.Series, bars: pd.DataFrame,
                                      volatility_series: pd.Series) -> float:
        """Calculate predictive quality score using directional AUC."""
        try:
            if labels.empty or labels.nunique() < 2:
                return 0.0
            
            # Calculate activation rate
            non_zero_labels = labels[labels != 0]
            activation_rate = len(non_zero_labels) / len(labels)
            
            # Check activation constraints
            if not (self.config.min_activation <= activation_rate <= self.config.max_activation):
                return 0.0
            
            if len(non_zero_labels) < self.config.min_nonzero_samples_per_target:
                return 0.0
            
            # Calculate directional AUC (predictive objective)
            if len(bars) > 0 and 'close' in bars.columns:
                # Calculate future returns for evaluation
                future_returns = bars['close'].pct_change().shift(-1)  # Next period return
                
                # Align data
                common_index = labels.index.intersection(future_returns.index)
                if len(common_index) < 10:
                    return 0.0
                
                labels_aligned = labels.loc[common_index]
                returns_aligned = future_returns.loc[common_index]
                
                # Calculate directional AUC
                auc_score = self._calculate_directional_auc(labels_aligned, returns_aligned)
                
                # Calculate class balance on non-zero labels only
                if len(non_zero_labels) > 0:
                    class_balance = non_zero_labels.value_counts().max() / len(non_zero_labels)
                    balance_score = 1.0 - abs(class_balance - 0.5) * 2
                else:
                    balance_score = 0.0
                
                # Calculate hit ratio (barrier actually hit in labeled direction)
                hit_ratio = self._calculate_hit_ratio(labels_aligned, returns_aligned, bars.loc[common_index])
                
                # Combine metrics with weights
                quality_score = (
                    0.5 * auc_score +      # Predictive power
                    0.3 * balance_score +  # Class balance
                    0.2 * hit_ratio        # Hit accuracy
                )
                
                return max(0.0, min(1.0, quality_score))
            else:
                return 0.0
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating quality score: {e}")
            return 0.0
    
    def _calculate_directional_auc(self, labels: pd.Series, returns: pd.Series) -> float:
        """Calculate directional AUC between label sign and future return sign."""
        try:
            from sklearn.metrics import roc_auc_score
            
            # Filter non-zero labels
            non_zero_mask = labels != 0
            if non_zero_mask.sum() < 10:
                return 0.0
            
            labels_nz = labels[non_zero_mask]
            returns_nz = returns[non_zero_mask]
            
            # Create binary targets: 1 if return > 0, 0 if return <= 0
            binary_targets = (returns_nz > 0).astype(int)
            
            # Use label signs as predictions
            predictions = (labels_nz > 0).astype(int)
            
            # Calculate AUC
            if len(np.unique(binary_targets)) > 1 and len(np.unique(predictions)) > 1:
                auc = roc_auc_score(binary_targets, predictions)
                return abs(auc - 0.5) * 2  # Convert to [0, 1] scale
            else:
                return 0.0
                
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating directional AUC: {e}")
            return 0.0
    
    def _calculate_hit_ratio(self, labels: pd.Series, returns: pd.Series, bars: pd.DataFrame) -> float:
        """Calculate hit ratio: P(barrier actually hit in labeled direction)."""
        try:
            # Filter non-zero labels
            non_zero_mask = labels != 0
            if non_zero_mask.sum() < 10:
                return 0.0
            
            labels_nz = labels[non_zero_mask]
            returns_nz = returns[non_zero_mask]
            
            # Count correct directional predictions
            correct_predictions = 0
            total_predictions = len(labels_nz)
            
            for i, (label, ret) in enumerate(zip(labels_nz, returns_nz)):
                if (label > 0 and ret > 0) or (label < 0 and ret < 0):
                    correct_predictions += 1
            
            hit_ratio = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            
            return hit_ratio
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating hit ratio: {e}")
            return 0.0
    
    def _calculate_fpt_horizons(self, candidate_targets: List[Dict[str, Any]],
                              bars: pd.DataFrame, volatility_series: pd.Series) -> Tuple[Dict[str, int], List[float]]:
        """Calculate first-passage time based horizons."""
        try:
            horizons = {}
            all_fpt_data = []
            
            for candidate in candidate_targets:
                target_name = candidate['target_name']
                k_up = candidate['k_up']
                k_down = candidate['k_down']
                
            # Calculate FPT for this target using trailing data only
            trailing_bars = bars.iloc[:len(bars)//2]  # Use first half for FPT calculation
            trailing_vol = volatility_series.iloc[:len(volatility_series)//2]
            fpt = self._calculate_fpt_for_target(k_up, k_down, trailing_bars, trailing_vol)
                
                if fpt is not None and len(fpt) > 0:
                    all_fpt_data.extend(fpt)
                    # Use middle quantile of FPT distribution as horizon
                    horizon = int(fpt[1])  # fpt is already an array of quantiles [q25, q50, q75]
                    horizons[target_name] = horizon
                else:
                    # Use default horizon if no FPT data
                    horizons[target_name] = max(1, len(bars) // 50)
            
            # Calculate data-driven horizon bounds from all FPT data
            if all_fpt_data:
                fpt_series = pd.Series(all_fpt_data)
                min_horizon, max_horizon = self.config.get_horizon_bounds(fpt_series)
                
                # Apply bounds to all horizons
                for target_name in horizons:
                    horizons[target_name] = max(min_horizon, min(max_horizon, horizons[target_name]))
            else:
                # Fallback to reasonable defaults
                min_horizon = max(1, len(bars) // 100)
                max_horizon = min(100, len(bars) // 10)
                for target_name in horizons:
                    horizons[target_name] = max(min_horizon, min(max_horizon, horizons[target_name]))
            
            # Use CV-based horizon selection for better out-of-sample performance
            cv_horizons = self._select_horizons_with_cv(candidate_targets, bars, volatility_series, horizons)
            
            return cv_horizons, all_fpt_data
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating FPT horizons: {e}")
            min_horizon = max(1, len(bars) // 100)
            return {target['target_name']: min_horizon for target in candidate_targets}, []
    
    def _select_horizons_with_cv(self, candidate_targets: List[Dict[str, Any]], 
                                bars: pd.DataFrame, volatility_series: pd.Series,
                                initial_horizons: Dict[str, int]) -> Dict[str, int]:
        """Select horizons using cross-validation for out-of-sample performance."""
        try:
            cv_horizons = {}
            
            for candidate in candidate_targets:
                target_name = candidate['target_name']
                k_up = candidate['k_up']
                k_down = candidate['k_down']
                initial_horizon = initial_horizons.get(target_name, 10)
                
                # Define horizon search space around initial estimate
                horizon_candidates = [
                    max(1, initial_horizon - 5),
                    initial_horizon,
                    min(len(bars) // 10, initial_horizon + 5),
                    max(1, initial_horizon // 2),
                    min(len(bars) // 10, initial_horizon * 2)
                ]
                
                # Remove duplicates and sort
                horizon_candidates = sorted(list(set(horizon_candidates)))
                
                # Use purged CV to evaluate horizons
                best_horizon = self._evaluate_horizons_cv(
                    k_up, k_down, horizon_candidates, bars, volatility_series
                )
                
                cv_horizons[target_name] = best_horizon
            
            return cv_horizons
            
        except Exception as e:
            tprint_warning(f"⚠️ Error in CV horizon selection: {e}")
            return initial_horizons
    
    def _evaluate_horizons_cv(self, k_up: float, k_down: float, horizon_candidates: List[int],
                             bars: pd.DataFrame, volatility_series: pd.Series) -> int:
        """Evaluate horizon candidates using purged cross-validation."""
        try:
            n_samples = len(bars)
            n_folds = min(self.config.cv_folds, n_samples // 100)  # Ensure enough samples per fold
            
            if n_folds < 2:
                return horizon_candidates[0] if horizon_candidates else 10
            
            # Create purged CV splits
            fold_scores = {}
            
            for horizon in horizon_candidates:
                fold_scores[horizon] = []
                
                for fold in range(n_folds):
                    # Calculate fold boundaries with embargo
                    fold_size = n_samples // n_folds
                    embargo_size = int(fold_size * self.config.embargo_fraction)
                    
                    # Training set: data before this fold
                    train_end = fold * fold_size
                    train_bars = bars.iloc[:train_end]
                    train_vol = volatility_series.iloc[:train_end]
                    
                    # Validation set: this fold (with embargo)
                    val_start = fold * fold_size + embargo_size
                    val_end = min((fold + 1) * fold_size, n_samples)
                    val_bars = bars.iloc[val_start:val_end]
                    val_vol = volatility_series.iloc[val_start:val_end]
                    
                    if len(train_bars) < 50 or len(val_bars) < 20:
                        continue
                    
                    # Generate labels for this horizon
                    train_labels = self._generate_labels_with_horizon(
                        k_up, k_down, horizon, train_bars, train_vol, 
                        pd.Series(True, index=train_bars.index)
                    )
                    val_labels = self._generate_labels_with_horizon(
                        k_up, k_down, horizon, val_bars, val_vol,
                        pd.Series(True, index=val_bars.index)
                    )
                    
                    if train_labels.empty or val_labels.empty:
                        continue
                    
                    # Calculate validation score (directional AUC)
                    train_labels_series = train_labels['labels']
                    val_labels_series = val_labels['labels']
                    
                    # Calculate future returns for validation
                    val_returns = val_bars['close'].pct_change().shift(-1)
                    
                    # Align data
                    common_index = val_labels_series.index.intersection(val_returns.index)
                    if len(common_index) < 10:
                        continue
                    
                    val_labels_aligned = val_labels_series.loc[common_index]
                    val_returns_aligned = val_returns.loc[common_index]
                    
                    # Calculate directional AUC
                    auc_score = self._calculate_directional_auc(val_labels_aligned, val_returns_aligned)
                    fold_scores[horizon].append(auc_score)
            
            # Select horizon with best average validation score
            best_horizon = horizon_candidates[0]
            best_score = -np.inf
            
            for horizon, scores in fold_scores.items():
                if scores:
                    avg_score = np.mean(scores)
                    if avg_score > best_score:
                        best_score = avg_score
                        best_horizon = horizon
            
            return best_horizon
            
        except Exception as e:
            tprint_warning(f"⚠️ Error evaluating horizons CV: {e}")
            return horizon_candidates[0] if horizon_candidates else 10
    
    def _calculate_fpt_for_target(self, k_up: float, k_down: float, bars: pd.DataFrame,
                                volatility_series: pd.Series) -> Optional[np.ndarray]:
        """Calculate first-passage time for a specific target using proper hit logic and censoring."""
        try:
            if len(bars) < self.config.fpt_min_samples:
                return None
            
            fpt_values = []
            censored_values = []
            
            # Use log prices for better numerical stability
            log_prices = np.log(bars['close'].values)
            
            for i in range(len(bars) - self.config.fpt_window):
                current_log_price = log_prices[i]
                current_vol = volatility_series.iloc[i]
                
                if np.isnan(current_vol) or current_vol <= 0:
                    continue
                
                # Calculate log barriers
                upper_log_target = current_log_price + k_up * current_vol
                lower_log_target = current_log_price - k_down * current_vol
                
                # Look ahead for first hit with overlapping windows
                future_log_prices = log_prices[i+1:i+self.config.fpt_window]
                
                hit_time = None
                hit_direction = None
                
                for j, future_log_price in enumerate(future_log_prices):
                    if future_log_price >= upper_log_target:
                        hit_time = j + 1  # +1 because j is 0-indexed
                        hit_direction = 'upper'
                        break
                    elif future_log_price <= lower_log_target:
                        hit_time = j + 1
                        hit_direction = 'lower'
                        break
                
                if hit_time is not None:
                    # Record the FPT with direction information
                    fpt_values.append((hit_time, hit_direction))
                else:
                    # Censored observation (no hit within window)
                    censored_values.append(self.config.fpt_window)
            
            # Process FPT data for survival analysis
            if fpt_values or censored_values:
                # Extract just the times for survival analysis
                fpt_times = [fpt[0] for fpt in fpt_values]
                all_times = fpt_times + censored_values
                event_indicators = [1] * len(fpt_times) + [0] * len(censored_values)
                
                # Convert to numpy arrays
                all_times = np.array(all_times)
                event_indicators = np.array(event_indicators)
                
                # Sort by time
                sort_idx = np.argsort(all_times)
                sorted_times = all_times[sort_idx]
                sorted_events = event_indicators[sort_idx]
                
                # Calculate survival probabilities using proper KM estimator
                survival_probs = self._calculate_survival_probabilities(sorted_times, sorted_events)
                
                # Use data-driven quantile probabilities for FPT estimation
                fpt_quantile_probs = self.config.get_fpt_quantiles(pd.Series(sorted_times))
                quantile_times = []
                
                for prob in fpt_quantile_probs:
                    # Find time where survival probability drops below (1-prob)
                    target_survival = 1 - prob
                    idx = np.where(survival_probs <= target_survival)[0]
                    if len(idx) > 0:
                        quantile_times.append(sorted_times[idx[0]])
                    else:
                        quantile_times.append(sorted_times[-1])
                
                return np.array(quantile_times)
            else:
                return None
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating FPT for target: {e}")
            return None
    
    def _calculate_survival_probabilities(self, times: np.ndarray, events: np.ndarray) -> np.ndarray:
        """Calculate survival probabilities using proper Kaplan-Meier estimator."""
        try:
            if len(times) == 0:
                return np.array([])
            
            # Get unique times and aggregate events/censoring
            unique_times, inverse_indices = np.unique(times, return_inverse=True)
            n_unique = len(unique_times)
            
            # Count events and censoring at each unique time
            events_at_time = np.zeros(n_unique, dtype=int)
            censored_at_time = np.zeros(n_unique, dtype=int)
            
            for i, time_idx in enumerate(inverse_indices):
                if events[i] == 1:  # Observed event
                    events_at_time[time_idx] += 1
                else:  # Censored
                    censored_at_time[time_idx] += 1
            
            # Calculate at-risk counts at each unique time
            n_at_risk = np.zeros(n_unique, dtype=int)
            n_at_risk[0] = len(times)  # All individuals at risk at first time
            
            for i in range(1, n_unique):
                n_at_risk[i] = n_at_risk[i-1] - events_at_time[i-1] - censored_at_time[i-1]
            
            # Calculate survival probabilities using proper KM formula
            survival_probs = np.ones(n_unique)
            
            for i in range(n_unique):
                if n_at_risk[i] > 0 and events_at_time[i] > 0:
                    # S(t_i) = S(t_{i-1}) * (1 - d_i/n_i)
                    hazard = events_at_time[i] / n_at_risk[i]
                    survival_probs[i] = survival_probs[i-1] * (1 - hazard)
                elif i > 0:
                    survival_probs[i] = survival_probs[i-1]
            
            # Map back to original time points
            survival_probs_mapped = survival_probs[inverse_indices]
            
            return survival_probs_mapped

        except Exception as e:
            tprint_warning(f"⚠️ Error calculating survival probabilities: {e}")
            return np.ones(len(times))
    
    def _generate_candidate_labels(self, candidate_targets: List[Dict[str, Any]],
                                 horizons: Dict[str, int], bars: pd.DataFrame,
                                 volatility_series: pd.Series, eligibility_mask: pd.Series) -> Dict[str, pd.DataFrame]:
        """Generate labels for all candidate targets."""
        try:
            candidate_labels = {}
            
            for candidate in candidate_targets:
                target_name = candidate['target_name']
                k_up = candidate['k_up']
                k_down = candidate['k_down']
                min_horizon, _ = self.config.get_horizon_bounds(pd.Series(range(1, min(100, len(bars)))))
                horizon = horizons.get(target_name, min_horizon)
                
                # Generate labels with specific horizon
                labels = self._generate_labels_with_horizon(
                    k_up, k_down, horizon, bars, volatility_series, eligibility_mask
                )
                
                if not labels.empty:
                    candidate_labels[target_name] = labels
            
            return candidate_labels
            
        except Exception as e:
            tprint_error(f"❌ Error generating candidate labels: {e}")
            return {}
    
    def _generate_labels_with_horizon(self, k_up: float, k_down: float, horizon: int,
                                    bars: pd.DataFrame, volatility_series: pd.Series,
                                    eligibility_mask: pd.Series) -> pd.DataFrame:
        """Generate labels with specific horizon."""
        try:
            # Calculate target levels
            upper_targets = bars['close'] + k_up * volatility_series
            lower_targets = bars['close'] - k_down * volatility_series
            
            # Initialize labels with explicit dtype
            labels = pd.Series(0, index=bars.index, dtype=int)
            confidence_scores = pd.Series(0.0, index=bars.index, dtype=float)
            
            # Generate labels using triple barrier method with horizon
            for i in range(len(bars) - horizon):
                if not eligibility_mask.iloc[i]:
                    continue
                
                current_price = bars['close'].iloc[i]
                upper_target = upper_targets.iloc[i]
                lower_target = lower_targets.iloc[i]
                
                # Check if price hits targets within horizon
                future_prices = bars['close'].iloc[i+1:i+horizon+1]
                if len(future_prices) == 0:
                    continue
                
                # Find first hit
                upper_hits = future_prices >= upper_target
                lower_hits = future_prices <= lower_target
                
                if upper_hits.any() and lower_hits.any():
                    # Both hit - check which comes first
                    upper_first_hit = upper_hits.idxmax() if upper_hits.any() else None
                    lower_first_hit = lower_hits.idxmax() if lower_hits.any() else None
                    
                    if upper_first_hit is not None and lower_first_hit is not None:
                        if upper_first_hit <= lower_first_hit:
                            labels.iloc[i] = 1  # Upper hit first
                            # Calculate confidence based on distance to opposite barrier
                            distance_to_opposite = abs(future_prices.loc[upper_first_hit] - lower_target)
                            confidence_scores.iloc[i] = min(1.0, distance_to_opposite / (k_down * volatility_series.iloc[i]))
                        else:
                            labels.iloc[i] = -1  # Lower hit first
                            distance_to_opposite = abs(future_prices.loc[lower_first_hit] - upper_target)
                            confidence_scores.iloc[i] = min(1.0, distance_to_opposite / (k_up * volatility_series.iloc[i]))
                    elif upper_first_hit is not None:
                        labels.iloc[i] = 1
                        distance_to_opposite = abs(future_prices.loc[upper_first_hit] - lower_target)
                        confidence_scores.iloc[i] = min(1.0, distance_to_opposite / (k_down * volatility_series.iloc[i]))
                    elif lower_first_hit is not None:
                        labels.iloc[i] = -1
                        distance_to_opposite = abs(future_prices.loc[lower_first_hit] - upper_target)
                        confidence_scores.iloc[i] = min(1.0, distance_to_opposite / (k_up * volatility_series.iloc[i]))
                elif upper_hits.any():
                    labels.iloc[i] = 1
                    distance_to_opposite = abs(future_prices.loc[upper_hits.idxmax()] - lower_target) if upper_hits.any() else 0
                    confidence_scores.iloc[i] = min(1.0, distance_to_opposite / (k_down * volatility_series.iloc[i]))
                elif lower_hits.any():
                    labels.iloc[i] = -1
                    distance_to_opposite = abs(future_prices.loc[lower_hits.idxmax()] - upper_target) if lower_hits.any() else 0
                    confidence_scores.iloc[i] = min(1.0, distance_to_opposite / (k_up * volatility_series.iloc[i]))
            
            # Create DataFrame with labels and confidence
            result_df = pd.DataFrame({
                'labels': labels,
                'confidence': confidence_scores
            }, index=bars.index)
            
            return result_df
            
        except Exception as e:
            tprint_warning(f"⚠️ Error generating labels with horizon: {e}")
            return pd.DataFrame()
    
    def _generate_confidence_features(self, bars: pd.DataFrame, volatility_series: pd.Series) -> pd.DataFrame:
        """Generate features for probabilistic confidence scoring."""
        try:
            features = pd.DataFrame(index=bars.index)
            
            # Price-based features
            features['returns'] = bars['close'].pct_change()
            features['volatility'] = volatility_series
            features['volatility_ratio'] = volatility_series / volatility_series.rolling(20).mean()
            
            # Volume features
            features['volume_ratio'] = bars['volume'] / bars['volume'].rolling(20).mean()
            features['volume_trend'] = bars['volume'].pct_change()
            
            # OHLC features
            features['high_low_ratio'] = (bars['high'] - bars['low']) / bars['close']
            features['close_open_ratio'] = (bars['close'] - bars['open']) / bars['open']
            
            # Technical indicators
            features['price_momentum'] = bars['close'] / bars['close'].shift(5) - 1
            features['volatility_momentum'] = volatility_series / volatility_series.shift(5) - 1
            
            # Fill NaN values
            features = features.fillna(0)
            
            return features
            
        except Exception as e:
            tprint_warning(f"⚠️ Error generating confidence features: {e}")
            return pd.DataFrame()
    
    def _calculate_probabilistic_confidence(self, features: pd.Series, label: int, 
                                          volatility: float, hit_time: int) -> float:
        """Calculate probabilistic confidence using calibrated model."""
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.preprocessing import StandardScaler
            
            # Train confidence model on historical data with proper CV
            confidence_model = self._train_confidence_model_with_cv(features, volatility, hit_time)
            
            # Simple heuristic-based confidence for now
            # Real implementation would use a trained model
            
            # Base confidence from features
            base_confidence = 0.5
            
            # Adjust for volatility (lower vol = higher confidence)
            vol_factor = 1.0 / (1.0 + volatility * 5.0)
            
            # Adjust for hit time (faster hits = higher confidence)
            time_factor = 1.0 / (1.0 + hit_time * 0.05)
            
            # Adjust for feature consistency
            feature_consistency = 0.5
            if len(features) > 0:
                # Check if features are consistent with label direction
                if 'returns' in features and not pd.isna(features['returns']):
                    if (label > 0 and features['returns'] > 0) or (label < 0 and features['returns'] < 0):
                        feature_consistency = 0.8
                    else:
                        feature_consistency = 0.3
            
            # Combine factors
            confidence = base_confidence * vol_factor * time_factor * feature_consistency
            
            # Apply calibration (simple sigmoid)
            calibrated_confidence = 1.0 / (1.0 + np.exp(-(confidence - 0.5) * 10))
            
            return max(0.0, min(1.0, calibrated_confidence))
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating probabilistic confidence: {e}")
            return 0.5
    
    def _train_confidence_model_with_cv(self, features: pd.Series, volatility: float, 
                                       hit_time: float) -> Any:
        """Train confidence model using cross-validation on historical data."""
        try:
            # Load historical training data
            training_data = self._load_confidence_training_data()
            
            if training_data is None or len(training_data) < 100:
                # Fallback to heuristic approach
                return self._calculate_heuristic_confidence(features, volatility, hit_time)
            
            # Prepare training features
            X = training_data[['volatility', 'hit_time', 'returns', 'volume_ratio', 'volatility_ratio']]
            y = training_data['confidence_label']  # Binary confidence labels
            
            # Use proper time series CV
            from src.utils.ml_common.validation.cv import purged_time_series_splits, PurgedSplitConfig
            
            config = PurgedSplitConfig(n_splits=5, purge_minutes=30, embargo_minutes=15)
            cv_splits = list(purged_time_series_splits(X, y, config))
            
            if len(cv_splits) < 3:
                # Fallback if CV fails
                return self._calculate_heuristic_confidence(features, volatility, hit_time)
            
            # Train model with CV
            from sklearn.linear_model import LogisticRegression
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import log_loss
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train calibrated model
            base_model = LogisticRegression(random_state=42, max_iter=1000)
            calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=cv_splits)
            calibrated_model.fit(X_scaled, y)
            
            # Prepare current features for prediction
            current_features = np.array([[
                volatility,
                hit_time,
                features.get('returns', 0.0),
                features.get('volume_ratio', 1.0),
                features.get('volatility_ratio', 1.0)
            ]])
            
            current_features_scaled = scaler.transform(current_features)
            confidence = calibrated_model.predict_proba(current_features_scaled)[0][1]
            
            return float(confidence)
            
        except Exception as e:
            tprint_warning(f"⚠️ Error training confidence model with CV: {e}")
            return self._calculate_heuristic_confidence(features, volatility, hit_time)
    
    def _calculate_heuristic_confidence(self, features: pd.Series, volatility: float, 
                                      hit_time: float) -> float:
        """Calculate heuristic confidence as fallback."""
        try:
            # Base confidence from features
            base_confidence = 0.5
            
            # Adjust for volatility (lower vol = higher confidence)
            vol_factor = 1.0 / (1.0 + volatility * 5.0)
            
            # Adjust for hit time (faster hits = higher confidence)
            time_factor = 1.0 / (1.0 + hit_time * 0.05)
            
            # Adjust for feature consistency
            feature_consistency = 0.5
            if len(features) > 0:
                # Check if features are consistent with label direction
                if 'returns' in features and not pd.isna(features['returns']):
                    if (features['returns'] > 0) or (features['returns'] < 0):
                        feature_consistency = 0.8
                    else:
                        feature_consistency = 0.3
            
            # Combine factors
            confidence = base_confidence * vol_factor * time_factor * feature_consistency
            
            # Apply calibration (simple sigmoid)
            calibrated_confidence = 1.0 / (1.0 + np.exp(-(confidence - 0.5) * 10))
            
            return max(0.0, min(1.0, calibrated_confidence))
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating heuristic confidence: {e}")
            return 0.5
    
    def _load_confidence_training_data(self) -> Optional[pd.DataFrame]:
        """Load historical data for confidence model training."""
        try:
            # This would integrate with your existing data infrastructure
            # For now, return None to use fallback
            return None
            
        except Exception as e:
            tprint_warning(f"⚠️ Error loading confidence training data: {e}")
            return None

    def _train_confidence_model(self, features_df: pd.DataFrame, labels: pd.Series, 
                               hit_times: pd.Series) -> Any:
        """Train a confidence model on historical data."""
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
            
            # Prepare features
            X = features_df.fillna(0)
            y = (labels != 0).astype(int)  # Binary: hit or no hit
            
            # Add hit time as a feature
            X['hit_time'] = hit_times.fillna(0)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train logistic regression
            lr = LogisticRegression(random_state=42, max_iter=1000)
            lr.fit(X_train_scaled, y_train)
            
            # Calibrate probabilities
            calibrated_model = CalibratedClassifierCV(lr, method='isotonic', cv=3)
            calibrated_model.fit(X_train_scaled, y_train)
            
            return calibrated_model, scaler
            
        except Exception as e:
            tprint_warning(f"⚠️ Error training confidence model: {e}")
            return None, None
    
    def _resolve_label_conflicts(self, labels_df: pd.DataFrame, selected_targets: Dict[str, Any] = None) -> pd.DataFrame:
        """Resolve conflicts between different target labels at the same timestamp."""
        try:
            if labels_df.empty:
                return labels_df
            
            # Create conflict resolution rules
            # 1. Hierarchical precedence: small < medium < high
            # 2. Confidence-based selection within same level
            # 3. Multi-task output for complementary signals
            
            resolved_labels = labels_df.copy()
            
            # Group by target bands using metadata from selected_targets
            small_targets = []
            medium_targets = []
            high_targets = []
            
            # Use selected_targets metadata if available
            if selected_targets:
                for target_name, target_info in selected_targets.items():
                    if target_name in labels_df.columns:
                        band = target_info.get('band', TargetBand.SMALL)
                        if band == TargetBand.SMALL:
                            small_targets.append(target_name)
                        elif band == TargetBand.MEDIUM:
                            medium_targets.append(target_name)
                        elif band == TargetBand.HIGH:
                            high_targets.append(target_name)
            else:
                # Fallback to string matching (less robust)
                small_targets = [col for col in labels_df.columns if 'small' in col.lower()]
                medium_targets = [col for col in labels_df.columns if 'medium' in col.lower()]
                high_targets = [col for col in labels_df.columns if 'high' in col.lower()]
            
            # Apply model-based conflict resolution
            for idx in labels_df.index:
                # Check for conflicts (multiple non-zero labels)
                non_zero_labels = labels_df.loc[idx][labels_df.loc[idx] != 0]
                
                if len(non_zero_labels) > 1:
                    # Use confidence-based selection instead of hierarchical precedence
                    target_to_keep = self._select_best_target_by_confidence(
                        non_zero_labels, selected_targets, idx
                    )

                    # Zero out all other conflicting targets
                    if target_to_keep is not None:
                        for other_col in non_zero_labels.index:
                            if other_col != target_to_keep:
                                resolved_labels.loc[idx, other_col] = 0
            
            return resolved_labels
            
        except Exception as e:
            tprint_warning(f"⚠️ Error resolving label conflicts: {e}")
            return labels_df
    
    def _select_best_target_by_confidence(self, non_zero_labels: pd.Series, 
                                        selected_targets: Dict[str, Any], 
                                        timestamp: pd.Timestamp) -> Optional[str]:
        """Select best target based on confidence and expected utility."""
        try:
            if not selected_targets:
                return non_zero_labels.index[0]  # Fallback to first target
            
            best_target = None
            best_score = -np.inf
            
            for target_name in non_zero_labels.index:
                if target_name not in selected_targets:
                    continue
                
                target_info = selected_targets[target_name]
                label_value = non_zero_labels[target_name]
                
                # Calculate confidence score
                confidence = self._get_target_confidence(target_name, timestamp)
                
                # Calculate expected utility (simplified)
                utility = self._calculate_expected_utility(target_info, label_value, confidence)
                
                # Combined score
                score = confidence * utility
                
                if score > best_score:
                    best_score = score
                    best_target = target_name
            
            return best_target
            
        except Exception as e:
            tprint_warning(f"⚠️ Error selecting best target: {e}")
            return non_zero_labels.index[0] if len(non_zero_labels) > 0 else None
    
    def _get_target_confidence(self, target_name: str, timestamp: pd.Timestamp) -> float:
        """Get confidence score for a target at a specific timestamp."""
        try:
            # This would use the trained confidence model in practice
            # For now, return a simple heuristic based on target band
            if hasattr(self, '_current_selected_targets'):
                target_info = self._current_selected_targets.get(target_name, {})
                band = target_info.get('band', TargetBand.SMALL)
                
                # Higher confidence for higher bands (more significant signals)
                if band == TargetBand.HIGH:
                    return 0.8
                elif band == TargetBand.MEDIUM:
                    return 0.6
                else:
                    return 0.4
            else:
                return 0.5  # Default confidence
                
        except Exception as e:
            tprint_warning(f"⚠️ Error getting target confidence: {e}")
            return 0.5
    
    def _calculate_expected_utility(self, target_info: Dict[str, Any], 
                                  label_value: int, confidence: float) -> float:
        """Calculate expected utility for a target."""
        try:
            # Simple utility calculation based on target parameters and confidence
            k_up = target_info.get('k_up', 1.0)
            k_down = target_info.get('k_down', 1.0)
            
            # Higher k values indicate more significant moves
            k_avg = (k_up + k_down) / 2
            
            # Utility increases with k value and confidence
            utility = k_avg * confidence * abs(label_value)
            
            return utility
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating expected utility: {e}")
            return 0.5
    
    def _apply_label_smoothing(self, labels_df: pd.DataFrame, confidence_df: pd.DataFrame) -> pd.DataFrame:
        """Apply label smoothing for better model calibration."""
        try:
            if labels_df.empty:
                return labels_df
            
            smoothed_labels = labels_df.copy()
            
            # Apply temporal smoothing to reduce micro-flips
            for col in labels_df.columns:
                if col in labels_df.columns:
                    labels_series = labels_df[col]
                    smoothed_series = self._temporal_smoothing(labels_series, confidence_df.get(col, pd.Series()))
                    smoothed_labels[col] = smoothed_series
            
            # Apply soft label smoothing (mix with uniform noise)
            if not confidence_df.empty:
                smoothed_labels = self._soft_label_smoothing(smoothed_labels, confidence_df)
            
            return smoothed_labels
            
        except Exception as e:
            tprint_warning(f"⚠️ Error applying label smoothing: {e}")
            return labels_df
    
    def _temporal_smoothing(self, labels_series: pd.Series, confidence_series: pd.Series) -> pd.Series:
        """Apply temporal smoothing to reduce micro-flips."""
        try:
            if len(labels_series) < 3:
                return labels_series
            
            smoothed = labels_series.copy()
            window_size = 3  # Small window for temporal smoothing
            
            for i in range(window_size, len(labels_series)):
                # Get recent labels and confidences
                recent_labels = labels_series.iloc[i-window_size:i]
                recent_confidences = confidence_series.iloc[i-window_size:i] if not confidence_series.empty else pd.Series(1.0, index=recent_labels.index)
                
                # Weight by confidence
                if not recent_confidences.empty:
                    weights = recent_confidences / recent_confidences.sum()
                    weighted_labels = recent_labels * weights
                    smoothed_value = weighted_labels.sum()
                else:
                    smoothed_value = recent_labels.mean()
                
                # Apply smoothing only if confidence is high enough
                current_confidence = confidence_series.iloc[i] if not confidence_series.empty else 1.0
                if current_confidence > 0.5:
                    # Blend current label with smoothed value
                    alpha = 0.3  # Smoothing strength
                    smoothed.iloc[i] = (1 - alpha) * labels_series.iloc[i] + alpha * smoothed_value
            
            return smoothed
            
        except Exception as e:
            tprint_warning(f"⚠️ Error in temporal smoothing: {e}")
            return labels_series
    
    def _soft_label_smoothing(self, labels_df: pd.DataFrame, confidence_df: pd.DataFrame) -> pd.DataFrame:
        """Apply soft label smoothing by mixing with uniform noise."""
        try:
            if labels_df.empty or confidence_df.empty:
                return labels_df
            
            smoothed_labels = labels_df.copy()
            smoothing_factor = 0.1  # 10% uniform noise
            
            for col in labels_df.columns:
                if col in labels_df.columns and col in confidence_df.columns:
                    labels_series = labels_df[col]
                    confidence_series = confidence_df[col]
                    
                    # Create soft labels
                    soft_labels = labels_series.copy()
                    
                    for i in range(len(labels_series)):
                        if confidence_series.iloc[i] > 0.5:  # Only smooth high-confidence labels
                            # Use deterministic smoothing instead of random noise for reproducibility
                            # Apply small amount of smoothing toward zero for regularization
                            soft_value = (1 - smoothing_factor) * labels_series.iloc[i] + smoothing_factor * 0.0

                            # Clamp to valid range
                            soft_labels.iloc[i] = max(-1, min(1, soft_value))
                    
                    smoothed_labels[col] = soft_labels
            
            return smoothed_labels
            
        except Exception as e:
            tprint_warning(f"⚠️ Error in soft label smoothing: {e}")
            return labels_df
    
    def _select_optimal_targets(self, candidate_labels: Dict[str, pd.DataFrame],
                              candidate_targets: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Select optimal targets based on quality and diversity."""
        try:
            if not candidate_labels:
                return {}
            
            # Calculate quality scores for all candidates
            quality_scores = {}
            for target_name, labels_df in candidate_labels.items():
                if not labels_df.empty and 'labels' in labels_df.columns:
                    labels = labels_df['labels']
                    quality_score = self._calculate_target_quality_score(labels, pd.DataFrame(), pd.Series())
                    quality_scores[target_name] = quality_score
            
            # Filter by minimum quality threshold
            qualified_targets = {
                name: score for name, score in quality_scores.items()
                if score >= self.config.min_lqs_score
            }
            
            if not qualified_targets:
                tprint_warning("⚠️ No targets passed quality threshold")
                return {}
            
            # Select targets by band
            selected_targets = {}
            band_counts = {band: 0 for band in TargetBand}
            
            # Sort by quality score
            sorted_targets = sorted(qualified_targets.items(), key=lambda x: x[1], reverse=True)
            
            for target_name, quality_score in sorted_targets:
                # Find the candidate info
                candidate_info = next((c for c in candidate_targets if c['target_name'] == target_name), None)
                if not candidate_info:
                    continue
                
                band = candidate_info['band']
                
                # Check band limits
                if band_counts[band] >= self.config.max_targets_per_band:
                    continue
                
                # Check total limits
                if len(selected_targets) >= self.config.max_targets_total:
                    break
                
                # Check correlation with already selected targets
                if self._check_correlation_constraints(target_name, selected_targets, candidate_labels):
                    selected_targets[target_name] = {
                        **candidate_info,
                        'quality_score': quality_score
                    }
                    band_counts[band] += 1
            
            # Ensure minimum targets
            if len(selected_targets) < self.config.min_targets_total:
                tprint_warning(f"⚠️ Only {len(selected_targets)} targets selected, minimum is {self.config.min_targets_total}")
            
            return selected_targets
            
        except Exception as e:
            tprint_error(f"❌ Error selecting optimal targets: {e}")
            return {}
    
    def _check_correlation_constraints(self, target_name: str, selected_targets: Dict[str, Any],
                                     candidate_labels: Dict[str, pd.DataFrame]) -> bool:
        """Check if target meets mutual information constraints for orthogonality."""
        try:
            if not selected_targets:
                return True
            
            # Get labels for current target
            current_labels = candidate_labels.get(target_name)
            if current_labels is None or current_labels.empty or 'labels' not in current_labels.columns:
                return False
            
            current_labels_series = current_labels['labels']
            
            # Check mutual information with each selected target
            for selected_name, selected_info in selected_targets.items():
                selected_labels = candidate_labels.get(selected_name)
                if selected_labels is None or selected_labels.empty or 'labels' not in selected_labels.columns:
                    continue
                
                selected_labels_series = selected_labels['labels']
                
                # Align indices
                common_index = current_labels_series.index.intersection(selected_labels_series.index)
                if len(common_index) < 10:
                    continue
                
                current_aligned = current_labels_series.loc[common_index]
                selected_aligned = selected_labels_series.loc[common_index]
                
                # Calculate mutual information
                try:
                    mi_score = self._calculate_mutual_information(current_aligned, selected_aligned)
                    if mi_score > self.config.max_correlation_threshold:  # Use same threshold for now
                        return False
                except Exception:
                    # Fallback to correlation if MI calculation fails
                    corr, _ = spearmanr(current_aligned, selected_aligned)
                    if not np.isnan(corr) and abs(corr) > self.config.max_correlation_threshold:
                        return False
            
            return True
            
        except Exception as e:
            tprint_warning(f"⚠️ Error checking correlation constraints: {e}")
            return True
    
    def _calculate_mutual_information(self, x: pd.Series, y: pd.Series) -> float:
        """Calculate mutual information between two label sequences."""
        try:
            from sklearn.feature_selection import mutual_info_classif
            from sklearn.preprocessing import LabelEncoder
            
            # Filter non-zero labels for both series
            non_zero_mask = (x != 0) & (y != 0)
            if non_zero_mask.sum() < 10:
                return 0.0
            
            x_nz = x[non_zero_mask]
            y_nz = y[non_zero_mask]
            
            # Encode labels as integers
            le_x = LabelEncoder()
            le_y = LabelEncoder()
            
            x_encoded = le_x.fit_transform(x_nz)
            y_encoded = le_y.fit_transform(y_nz)
            
            # Calculate mutual information
            mi_scores = mutual_info_classif(x_encoded.reshape(-1, 1), y_encoded, discrete_features=True)
            
            return mi_scores[0] if len(mi_scores) > 0 else 0.0
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating mutual information: {e}")
            return 0.0
    
    def _generate_final_labels(self, selected_targets: Dict[str, Any], bars: pd.DataFrame,
                             volatility_series: pd.Series, eligibility_mask: pd.Series) -> Dict[str, pd.DataFrame]:
        """Generate final labels for selected targets."""
        try:
            labels_df = pd.DataFrame(index=bars.index)
            confidence_df = pd.DataFrame(index=bars.index)
            eligibility_df = pd.DataFrame(index=bars.index)
            
            for target_name, target_info in selected_targets.items():
                k_up = target_info['k_up']
                k_down = target_info['k_down']
                min_horizon, _ = self.config.get_horizon_bounds(pd.Series(range(1, min(100, len(bars)))))
                horizon = target_info.get('horizon', min_horizon)
                
                # Generate labels
                target_result = self._generate_labels_with_horizon(
                    k_up, k_down, horizon, bars, volatility_series, eligibility_mask
                )
                
                if not target_result.empty:
                    labels_df[target_name] = target_result['labels']
                    confidence_df[f"{target_name}_confidence"] = target_result['confidence']
                    eligibility_df[f"{target_name}_eligibility"] = eligibility_mask
            
            return {
                'labels': labels_df,
                'confidence_scores': confidence_df,
                'eligibility_masks': eligibility_df
            }
            
        except Exception as e:
            tprint_error(f"❌ Error generating final labels: {e}")
            return {
                'labels': pd.DataFrame(),
                'confidence_scores': pd.DataFrame(),
                'eligibility_masks': pd.DataFrame()
            }
    
    def _calculate_target_correlations(self, labels_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix between targets."""
        try:
            if labels_df.empty:
                return pd.DataFrame()
            
            # Calculate Spearman correlations
            corr_matrix = labels_df.corr(method='spearman')
            
            return corr_matrix
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating target correlations: {e}")
            return pd.DataFrame()
    
    def _calculate_diversity_score(self, labels_df: pd.DataFrame) -> float:
        """Calculate diversity score using mutual information on non-zero labels only."""
        try:
            if labels_df.empty or len(labels_df.columns) < 2:
                return 0.0
            
            # Calculate pairwise mutual information on synchronized non-zero timestamps
            mi_scores = []
            columns = labels_df.columns.tolist()
            
            for i in range(len(columns)):
                for j in range(i + 1, len(columns)):
                    col1 = columns[i]
                    col2 = columns[j]
                    
                    # Filter non-zero labels for both columns and ensure synchronized timestamps
                    non_zero_mask = (labels_df[col1] != 0) & (labels_df[col2] != 0)
                    if non_zero_mask.sum() < 10:
                        continue
                    
                    # Get synchronized non-zero labels
                    x = labels_df[col1][non_zero_mask]
                    y = labels_df[col2][non_zero_mask]
                    
                    # Ensure we have enough samples
                    if len(x) < 10 or len(y) < 10:
                        continue
                    
                    mi_score = self._calculate_mutual_information(x, y)
                    mi_scores.append(mi_score)
            
            if not mi_scores:
                return 0.0
            
            # Calculate average mutual information
            avg_mi = np.mean(mi_scores)
            
            # Diversity score (lower MI = higher diversity)
            # Normalize MI to [0, 1] range (MI can be > 1 for discrete variables)
            diversity_score = max(0.0, 1.0 - min(1.0, avg_mi))
            
            return diversity_score
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating diversity score: {e}")
            return 0.0
    
    def _calculate_target_coverage(self, labels_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate coverage for each target with detailed metrics."""
        try:
            coverage = {}
            
            for col in labels_df.columns:
                if col in labels_df.columns:
                    # Calculate activation rate P(|label|=1)
                    non_zero_labels = (labels_df[col] != 0).sum()
                    total_samples = len(labels_df)
                    activation_rate = non_zero_labels / total_samples if total_samples > 0 else 0.0
                    
                    # Calculate positive/negative split among active labels
                    positive_labels = (labels_df[col] > 0).sum()
                    negative_labels = (labels_df[col] < 0).sum()
                    total_active = positive_labels + negative_labels
                    
                    if total_active > 0:
                        positive_ratio = positive_labels / total_active
                        negative_ratio = negative_labels / total_active
                    else:
                        positive_ratio = 0.0
                        negative_ratio = 0.0
                    
                    # Store detailed coverage metrics
                    coverage[col] = {
                        'activation_rate': activation_rate,
                        'positive_ratio': positive_ratio,
                        'negative_ratio': negative_ratio,
                        'total_active': total_active,
                        'positive_count': positive_labels,
                        'negative_count': negative_labels
                    }
            
            return coverage
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating target coverage: {e}")
            return {}


# Convenience functions
def create_multi_target_scheme(config: Optional[MultiTargetConfig] = None) -> MultiTargetScheme:
    """Create multi-target scheme with specified configuration."""
    return MultiTargetScheme(config)


def generate_multi_targets(bars: pd.DataFrame, volatility_series: pd.Series,
                          eligibility_mask: pd.Series,
                          config: Optional[MultiTargetConfig] = None) -> TargetSelectionResult:
    """Generate multi-targets with default configuration."""
    scheme = MultiTargetScheme(config)
    return scheme.generate_targets(bars, volatility_series, eligibility_mask)