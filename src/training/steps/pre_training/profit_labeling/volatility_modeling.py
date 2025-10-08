"""
Volatility Modeling for Volatility-Aware Labeling

This module implements volatility modeling to normalize all thresholds and horizons
using volatility units instead of fixed percentages.

Key Features:
- Realized volatility estimation using high-frequency returns
- ATR (Average True Range) calculation
- EWMA volatility for responsiveness without whipsaw
- Volatility unit definition with floor to avoid division blowups
- Integration with existing ML optimization utilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

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
    validate_finite, validate_positive, validate_range
)
from src.utils.math_validation import MathValidation

# Import ML optimization utilities
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer
    BAYESIAN_OPTIMIZER_AVAILABLE = True
except ImportError:
    BAYESIAN_OPTIMIZER_AVAILABLE = False
    tprint_warning("⚠️ Bayesian TPE optimizer not available, using grid search")


class VolatilityMethod(Enum):
    """Enumeration of volatility estimation methods."""
    REALIZED = "realized"  # High-frequency realized volatility
    ATR = "atr"  # Average True Range
    EWMA = "ewma"  # Exponentially Weighted Moving Average
    GARCH = "garch"  # GARCH model (if available)
    COMBINED = "combined"  # Combined approach


@dataclass
class VolatilityConfig:
    """Configuration for volatility modeling."""
    
    # Volatility method
    method: VolatilityMethod = VolatilityMethod.COMBINED
    
    # Realized volatility settings
    rv_window: int = 20  # Window for realized volatility calculation
    rv_min_periods: int = 10  # Minimum periods for RV calculation
    
    # ATR settings
    atr_window: int = 14  # Window for ATR calculation
    atr_min_periods: int = 7  # Minimum periods for ATR calculation
    
    # EWMA settings
    ewma_alpha: float = 0.06  # EWMA decay factor (λ ≈ 0.94-0.97)
    ewma_min_periods: int = 10  # Minimum periods for EWMA calculation
    
    # Volatility unit settings
    volatility_floor: float = 1e-6  # Floor to avoid division blowups
    volatility_cap: float = 1.0  # Cap to avoid extreme volatility values
    
    # Smoothing settings
    enable_smoothing: bool = True
    smoothing_window: int = 5  # Window for additional smoothing
    
    # Quality checks
    min_volatility_samples: int = 50
    max_volatility_ratio: float = 10.0  # Max ratio between consecutive volatility values

    def _validate_config(self) -> None:
        """Basic validation for volatility configuration parameters."""
        if self.rv_window < 1:
            raise ValueError("rv_window must be at least 1")
        if self.atr_window < 1:
            raise ValueError("atr_window must be at least 1")
        if not (0 < self.ewma_alpha <= 1):
            raise ValueError("ewma_alpha must be between 0 and 1")
        if self.volatility_floor <= 0:
            raise ValueError("volatility_floor must be positive")
        if self.volatility_cap <= 0:
            raise ValueError("volatility_cap must be positive")
        if self.min_volatility_samples < 1:
            raise ValueError("min_volatility_samples must be at least 1")
        if self.max_volatility_ratio <= 0:
            raise ValueError("max_volatility_ratio must be positive")


@dataclass
class VolatilityResult:
    """Result container for volatility modeling."""
    
    # Core results
    volatility_series: pd.Series
    volatility_method: VolatilityMethod
    
    # Component results
    realized_volatility: Optional[pd.Series] = None
    atr_volatility: Optional[pd.Series] = None
    ewma_volatility: Optional[pd.Series] = None
    
    # Statistics
    mean_volatility: float = 0.0
    volatility_std: float = 0.0
    volatility_percentiles: Dict[str, float] = field(default_factory=dict)
    
    # Quality metrics
    volatility_consistency: float = 0.0
    volatility_stability: float = 0.0
    
    # Metadata
    config_used: VolatilityConfig = None
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class VolatilityModeler:
    """
    Volatility Modeler for Volatility-Aware Labeling
    
    This class implements comprehensive volatility modeling to normalize all thresholds
    and horizons using volatility units instead of fixed percentages.
    
    Key Features:
    1. **Realized Volatility**: High-frequency return-based volatility estimation
    2. **ATR Volatility**: Average True Range for price movement volatility
    3. **EWMA Volatility**: Exponentially weighted moving average for responsiveness
    4. **Combined Approach**: Intelligent combination of multiple methods
    5. **Quality Validation**: Comprehensive volatility quality assessment
    """
    
    def __init__(self, config: Optional[VolatilityConfig] = None):
        """Initialize volatility modeler."""
        self.config = config or VolatilityConfig()
        self.logger = logging.getLogger('VolatilityModeler')

        # Initialize matrix operations for vectorized computations
        if MATRIX_OPS_AVAILABLE:
            self.matrix_ops = UnifiedMatrixOperations()
            tprint_info("   → Matrix operations: Available")
        else:
            self.matrix_ops = None
            tprint_warning("   → Matrix operations: Not available, using fallback")

        tprint_info("📈 Volatility Modeler initialized")
        tprint_info(f"   → Method: {self.config.method.value}")
        tprint_info(f"   → RV window: {self.config.rv_window}")
        tprint_info(f"   → ATR window: {self.config.atr_window}")
        tprint_info(f"   → EWMA alpha: {self.config.ewma_alpha}")
    
    def model_volatility(self, bars: pd.DataFrame) -> VolatilityResult:
        """
        Model volatility from cleaned bars.
        
        Args:
            bars: Cleaned OHLCV bars with datetime index
            
        Returns:
            VolatilityResult with volatility estimates and statistics
        """
        start_time = datetime.now()
        tprint_info("📊 Modeling volatility")
        
        # Initialize result container
        result = VolatilityResult(
            volatility_series=pd.Series(),
            volatility_method=self.config.method,
            config_used=self.config
        )
        
        try:
            # Validate input data
            if not self._validate_input_data(bars):
                return result
            
            # Calculate individual volatility components
            tprint_info("📈 Step 1: Calculating volatility components")
            rv_series = self._calculate_realized_volatility(bars)
            atr_series = self._calculate_atr_volatility(bars)
            ewma_series = self._calculate_ewma_volatility(bars)
            
            result.realized_volatility = rv_series
            result.atr_volatility = atr_series
            result.ewma_volatility = ewma_series
            
            # Combine volatility estimates based on method
            tprint_info("🔗 Step 2: Combining volatility estimates")
            if self.config.method == VolatilityMethod.REALIZED:
                combined_volatility = rv_series
            elif self.config.method == VolatilityMethod.ATR:
                combined_volatility = atr_series
            elif self.config.method == VolatilityMethod.EWMA:
                combined_volatility = ewma_series
            else:  # COMBINED
                combined_volatility = self._combine_volatility_estimates(rv_series, atr_series, ewma_series)
            
            # Apply volatility unit normalization
            tprint_info("⚖️ Step 3: Applying volatility unit normalization")
            normalized_volatility = self._normalize_volatility_units(combined_volatility)
            
            # Apply smoothing if enabled
            if self.config.enable_smoothing:
                tprint_info("🔧 Step 4: Applying smoothing")
                normalized_volatility = self._apply_smoothing(normalized_volatility)
            
            result.volatility_series = normalized_volatility
            
            # Calculate statistics and quality metrics
            tprint_info("📊 Step 5: Calculating statistics and quality metrics")
            stats = self._calculate_volatility_statistics(normalized_volatility)
            result.mean_volatility = stats['mean_volatility']
            result.volatility_std = stats['volatility_std']
            result.volatility_percentiles = stats['volatility_percentiles']
            
            quality_metrics = self._calculate_volatility_quality(normalized_volatility)
            result.volatility_consistency = quality_metrics['consistency']
            result.volatility_stability = quality_metrics['stability']
            
        except Exception as e:
            tprint_error(f"❌ Volatility modeling failed: {e}")
            return result
        
        # Calculate processing time
        result.processing_time = (datetime.now() - start_time).total_seconds()
        
        tprint_success("✅ Volatility modeling completed")
        tprint_info(f"   → Volatility samples: {len(result.volatility_series)}")
        tprint_info(f"   → Mean volatility: {result.mean_volatility:.6f}")
        tprint_info(f"   → Volatility std: {result.volatility_std:.6f}")
        tprint_info(f"   → Consistency: {result.volatility_consistency:.3f}")
        
        return result
    
    def _validate_input_data(self, bars: pd.DataFrame) -> bool:
        """Validate input bar data."""
        try:
            # Check if DataFrame is empty
            if bars.empty:
                tprint_warning("⚠️ Input bars are empty")
                return False
            
            # Check required columns
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = set(required_columns) - set(bars.columns)
            if missing_columns:
                tprint_warning(f"⚠️ Missing required columns: {missing_columns}")
                return False
            
            # Check minimum samples
            if len(bars) < self.config.min_volatility_samples:
                tprint_warning(f"⚠️ Insufficient samples: {len(bars)} < {self.config.min_volatility_samples}")
                return False
            
            # Check for non-finite values
            if bars[required_columns].isnull().any().any():
                tprint_warning("⚠️ Data contains null values")
                return False
            
            if not np.isfinite(bars[required_columns].values).all():
                tprint_warning("⚠️ Data contains non-finite values")
                return False
            
            return True
            
        except Exception as e:
            tprint_error(f"❌ Data validation failed: {e}")
            return False
    
    def _calculate_realized_volatility(self, bars: pd.DataFrame) -> pd.Series:
        """Calculate realized volatility from high-frequency returns."""
        try:
            # Calculate returns (retain index alignment)
            returns = bars['close'].pct_change()
            
            if len(returns) < self.config.rv_min_periods:
                return pd.Series(dtype=float, index=bars.index)
            
            # Calculate rolling realized volatility
            rv = returns.rolling(
                window=self.config.rv_window,
                min_periods=self.config.rv_min_periods
            ).std()

            # Annualize if needed (assuming daily data)
            rv = rv * np.sqrt(252)

            # Use past-only window (exclude current bar)
            rv = rv.shift(1)

            return rv
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating realized volatility: {e}")
            return pd.Series(dtype=float, index=bars.index)
    
    def _calculate_atr_volatility(self, bars: pd.DataFrame) -> pd.Series:
        """Calculate ATR-based volatility using vectorized operations."""
        try:
            # Vectorized True Range calculation
            high = bars['high'].values
            low = bars['low'].values
            close = bars['close'].values

            # True Range components
            high_low = high - low
            high_close = np.abs(high - np.roll(close, 1))
            low_close = np.abs(low - np.roll(close, 1))

            # Fix first element for high_close and low_close
            high_close[0] = high_low[0]  # Use high-low for first element
            low_close[0] = high_low[0]

            # Vectorized true range calculation
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))

            if len(true_range) < self.config.atr_min_periods:
                return pd.Series(dtype=float, index=bars.index)

            # Use matrix operations for rolling mean if available
            if self.matrix_ops and MATRIX_OPS_AVAILABLE:
                # Convert to pandas Series for rolling operation (still efficient)
                tr_series = pd.Series(true_range, index=bars.index)

                # Vectorized rolling mean calculation
                atr_values = np.zeros_like(true_range, dtype=float)
                window = self.config.atr_window

                for i in range(len(true_range)):
                    if i < window - 1:
                        # Use available data for initial values
                        atr_values[i] = np.mean(true_range[max(0, i - window + 1):i+1])
                    else:
                        atr_values[i] = np.mean(true_range[i - window + 1:i+1])

                atr = pd.Series(atr_values, index=bars.index)
            else:
                # Fallback to pandas rolling
                tr_series = pd.Series(true_range, index=bars.index)
                atr = tr_series.rolling(
                    window=self.config.atr_window,
                    min_periods=self.config.atr_min_periods
                ).mean()

            # Normalize by price level (vectorized)
            atr = atr.shift(1)

            atr_volatility = atr / bars['close']

            return atr_volatility

        except Exception as e:
            tprint_warning(f"⚠️ Error calculating ATR volatility: {e}")
            return pd.Series(dtype=float, index=bars.index)
    
    def _calculate_ewma_volatility(self, bars: pd.DataFrame) -> pd.Series:
        """Calculate EWMA volatility using vectorized operations."""
        try:
            # Vectorized returns calculation
            close_prices = bars['close'].values
            returns = np.diff(close_prices) / close_prices[:-1]
            returns = np.concatenate([[0], returns])  # Pad first value

            if len(returns) < self.config.ewma_min_periods:
                return pd.Series(dtype=float, index=bars.index)

            # Vectorized EWMA calculation for variance
            alpha = self.config.ewma_alpha
            min_periods = self.config.ewma_min_periods

            # Use matrix operations for EWMA if available
            if self.matrix_ops and MATRIX_OPS_AVAILABLE:
                # Convert to pandas for ewm operation (still efficient)
                returns_series = pd.Series(returns, index=bars.index)

                # Calculate EWMA variance
                ewma_var = returns_series.ewm(
                    alpha=alpha,
                    min_periods=min_periods
                ).var().shift(1)

                # Convert to volatility
                ewma_volatility = np.sqrt(ewma_var)
            else:
                # Fallback implementation using vectorized operations
                returns_series = pd.Series(returns, index=bars.index)
                ewma_var = returns_series.ewm(alpha=alpha, min_periods=min_periods).var().shift(1)
                ewma_volatility = np.sqrt(ewma_var)

            # Annualize if needed (vectorized)
            ewma_volatility = ewma_volatility * np.sqrt(252)

            return ewma_volatility

        except Exception as e:
            tprint_warning(f"⚠️ Error calculating EWMA volatility: {e}")
            return pd.Series(dtype=float, index=bars.index)
    
    def _combine_volatility_estimates(self, rv_series: pd.Series, atr_series: pd.Series, 
                                    ewma_series: pd.Series) -> pd.Series:
        """Intelligently combine multiple volatility estimates using hybrid approach."""
        try:
            # Align all series to the same index
            common_index = rv_series.index.intersection(atr_series.index).intersection(ewma_series.index)
            
            if len(common_index) == 0:
                return pd.Series(dtype=float, index=rv_series.index)
            
            # Get aligned series
            rv_aligned = rv_series.loc[common_index]
            atr_aligned = atr_series.loc[common_index]
            ewma_aligned = ewma_series.loc[common_index]
            
            # Use hybrid RV+ATR approach as suggested
            # σ_t = 0.5·RV_t + 0.5·ATR_t for better balance of responsiveness and smoothness
            hybrid_volatility = 0.5 * rv_aligned + 0.5 * atr_aligned
            
            # Add EWMA as a smoothing component (smaller weight)
            ewma_weight = 0.2
            hybrid_weight = 0.8
            
            # Final combination: 80% hybrid (RV+ATR) + 20% EWMA
            combined = hybrid_weight * hybrid_volatility + ewma_weight * ewma_aligned
            
            return combined
            
        except Exception as e:
            tprint_warning(f"⚠️ Error combining volatility estimates: {e}")
            return rv_series  # Fallback to RV
    
    def _calculate_volatility_weight(self, vol_series: pd.Series, method: str) -> float:
        """Calculate weight for volatility method based on reliability."""
        try:
            if vol_series.empty or vol_series.isnull().all():
                return 0.0
            
            # Calculate reliability metrics
            non_null_ratio = vol_series.notna().sum() / len(vol_series)
            stability = 1.0 - (vol_series.std() / vol_series.mean()) if vol_series.mean() > 0 else 0.0
            consistency = 1.0 - (vol_series.diff().abs().mean() / vol_series.mean()) if vol_series.mean() > 0 else 0.0
            
            # Method-specific adjustments
            if method == 'rv':
                # RV is generally more reliable for high-frequency data
                method_bonus = 1.2
            elif method == 'atr':
                # ATR is good for price-level volatility
                method_bonus = 1.0
            elif method == 'ewma':
                # EWMA is good for responsiveness
                method_bonus = 0.8
            else:
                method_bonus = 1.0
            
            # Calculate final weight
            weight = non_null_ratio * stability * consistency * method_bonus
            
            return max(0.0, min(1.0, weight))
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating volatility weight for {method}: {e}")
            return 0.0
    
    def _normalize_volatility_units(self, volatility_series: pd.Series) -> pd.Series:
        """Normalize volatility to units with floor and cap."""
        try:
            if volatility_series.empty:
                return volatility_series
            
            # Apply floor to avoid division blowups
            normalized = np.maximum(volatility_series, self.config.volatility_floor)
            
            # Apply cap to avoid extreme values
            normalized = np.minimum(normalized, self.config.volatility_cap)
            
            # Ensure non-finite values are handled
            normalized = np.where(np.isfinite(normalized), normalized, self.config.volatility_floor)
            
            return pd.Series(normalized, index=volatility_series.index)
            
        except Exception as e:
            tprint_warning(f"⚠️ Error normalizing volatility units: {e}")
            return volatility_series
    
    def _apply_smoothing(self, volatility_series: pd.Series) -> pd.Series:
        """Apply additional smoothing to volatility series."""
        try:
            if volatility_series.empty or len(volatility_series) < self.config.smoothing_window:
                return volatility_series
            
            # Apply rolling mean smoothing
            smoothed = volatility_series.rolling(
                window=self.config.smoothing_window,
                min_periods=1
            ).mean()
            
            return smoothed
            
        except Exception as e:
            tprint_warning(f"⚠️ Error applying smoothing: {e}")
            return volatility_series
    
    def _calculate_volatility_statistics(self, volatility_series: pd.Series) -> Dict[str, Any]:
        """Calculate volatility statistics."""
        try:
            if volatility_series.empty:
                return {
                    'mean_volatility': 0.0,
                    'volatility_std': 0.0,
                    'volatility_percentiles': {}
                }
            
            # Basic statistics
            mean_vol = volatility_series.mean()
            std_vol = volatility_series.std()
            
            # Percentiles
            percentiles = [5, 10, 25, 50, 75, 90, 95]
            vol_percentiles = {}
            for p in percentiles:
                vol_percentiles[f'p{p}'] = volatility_series.quantile(p / 100)
            
            return {
                'mean_volatility': mean_vol,
                'volatility_std': std_vol,
                'volatility_percentiles': vol_percentiles
            }
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating volatility statistics: {e}")
            return {
                'mean_volatility': 0.0,
                'volatility_std': 0.0,
                'volatility_percentiles': {}
            }
    
    def _calculate_volatility_quality(self, volatility_series: pd.Series) -> Dict[str, float]:
        """Calculate volatility quality metrics."""
        try:
            if volatility_series.empty:
                return {'consistency': 0.0, 'stability': 0.0}
            
            # Consistency: how stable the volatility is over time
            vol_changes = volatility_series.diff().abs()
            consistency = 1.0 - (vol_changes.mean() / volatility_series.mean()) if volatility_series.mean() > 0 else 0.0
            
            # Stability: how consistent the volatility distribution is
            vol_std = volatility_series.std()
            vol_mean = volatility_series.mean()
            stability = 1.0 - (vol_std / vol_mean) if vol_mean > 0 else 0.0
            
            return {
                'consistency': max(0.0, min(1.0, consistency)),
                'stability': max(0.0, min(1.0, stability))
            }
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating volatility quality: {e}")
            return {'consistency': 0.0, 'stability': 0.0}


# Convenience functions
def create_volatility_modeler(config: Optional[VolatilityConfig] = None) -> VolatilityModeler:
    """Create volatility modeler with specified configuration."""
    return VolatilityModeler(config)


def model_volatility(bars: pd.DataFrame,
                    config: Optional[VolatilityConfig] = None) -> VolatilityResult:
    """Model volatility with default configuration."""
    modeler = VolatilityModeler(config)
    return modeler.model_volatility(bars)