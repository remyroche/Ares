"""
Unified Price Generation for Layer2 Context Generation
Integrates Layer0-optimized Kalman+VWAP parameters across all Layer2 models
"""

import logging
import json
import glob
import numpy as np
import pandas as pd
from typing import Dict, Optional

logger = logging.getLogger(__name__)

def load_layer0_params() -> dict:
    """Load optimized Kalman+VWAP parameters from Layer0 summary."""
    try:
        # Find latest Layer0 summary
        layer0_files = glob.glob("outcomes/layer0_summary_*.csv")
        layer0_json_files = glob.glob("outcomes/layer0_summary_*.json")
        layer0_files.extend(layer0_json_files)
        if not layer0_files:
            logger.warning(f"No Layer0 summary found in outcomes/, checked {len(glob.glob('outcomes/layer0_summary_*'))} files")
            logger.warning("No Layer0 summary found, using defaults")
            return {
                'kalman_Q': 1e-4, 
                'kalman_R': 0.01, 
                'vwap_weight': 0.4, 
                'vwap_lookback': 50,
                'median_filter_enabled': False,
                'median_window': 5,
                'hampel_filter_enabled': False,
                'hampel_window': 5,
                'hampel_threshold': 3.0,
                'adaptive_kalman_enabled': False,
                'adaptive_noise_window': 50,
                'adaptive_adaptation_rate': 0.1,
                'robust_vwap_enabled': False,
                'robust_min_lookback': 20,
                'robust_max_lookback': 200,
                'robust_volatility_window': 20
            }
        
        latest_file = max(layer0_files)
        
        # Load based on file type
        if latest_file.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(latest_file)
            if len(df) > 0:
                summary = df.iloc[0].to_dict()
            else:
                summary = {}
        else:
            with open(latest_file, 'r') as f:
                summary = json.load(f)
        
        # Extract best parameters (handle different summary formats)
        if 'best_params' in summary:
            params = summary['best_params']
        else:
            # Fallback to direct parameter extraction
            params = {
                'kalman_Q': summary.get('kalman_Q', 1e-4),
                'kalman_R': summary.get('kalman_R', 0.01),
                'vwap_weight': summary.get('vwap_weight', 0.4),
                'vwap_lookback': summary.get('vwap_lookback', 50)
            }
        
        # Add enhanced parameters with defaults
        enhanced_params = {
            'kalman_Q': params.get('kalman_Q', 1e-4),
            'kalman_R': params.get('kalman_R', 0.01),
            'vwap_weight': params.get('vwap_weight', 0.4),
            'vwap_lookback': params.get('vwap_lookback', 50),
            'median_filter_enabled': summary.get('median_filter_enabled', False),
            'median_window': summary.get('median_window', 5),
            'hampel_filter_enabled': summary.get('hampel_filter_enabled', False),
            'hampel_window': summary.get('hampel_window', 5),
            'hampel_threshold': summary.get('hampel_threshold', 3.0),
            'adaptive_kalman_enabled': summary.get('adaptive_kalman_enabled', False),
            'adaptive_noise_window': summary.get('adaptive_noise_window', 50),
            'adaptive_adaptation_rate': summary.get('adaptive_adaptation_rate', 0.1),
            'robust_vwap_enabled': summary.get('robust_vwap_enabled', False),
            'robust_min_lookback': summary.get('robust_min_lookback', 20),
            'robust_max_lookback': summary.get('robust_max_lookback', 200),
            'robust_volatility_window': summary.get('robust_volatility_window', 20)
        }
        
        logger.info(f"Loaded Layer0 params: Q={enhanced_params['kalman_Q']}, R={enhanced_params['kalman_R']}, vwap_weight={enhanced_params['vwap_weight']}")
        logger.info(f"Enhanced features: hampel_filter={enhanced_params['hampel_filter_enabled']}, adaptive_kalman={enhanced_params['adaptive_kalman_enabled']}, robust_vwap={enhanced_params['robust_vwap_enabled']}")
        return enhanced_params
        
    except Exception as e:
        logger.warning(f"Failed to load Layer0 params: {e}, using defaults")
        return {
            'kalman_Q': 1e-4, 
            'kalman_R': 0.01, 
            'vwap_weight': 0.4, 
            'vwap_lookback': 50,
            'median_filter_enabled': False,
            'median_window': 5,
            'hampel_filter_enabled': False,
            'hampel_window': 5,
            'hampel_threshold': 3.0,
            'adaptive_kalman_enabled': False,
            'adaptive_noise_window': 50,
            'adaptive_adaptation_rate': 0.1,
            'robust_vwap_enabled': False,
            'robust_min_lookback': 20,
            'robust_max_lookback': 200,
            'robust_volatility_window': 20
        }

def generate_unified_layer2_price(df: pd.DataFrame, layer0_params: dict = None) -> pd.Series:
    """
    Generate unified Kalman+VWAP composite price for all Layer2 models.
    
    Enhanced with Median Filter, Adaptive Kalman, Robust VWAP, and Savitzky-Golay options.
    
    Args:
        df: DataFrame with close, volume columns
        layer0_params: Optimized parameters from Layer0 (auto-loaded if None)
        
    Returns:
        Composite price series for Layer2 context generation
    """
    if layer0_params is None:
        layer0_params = load_layer0_params()
    
    # Extract Layer0-optimized parameters
    Q = layer0_params['kalman_Q']
    R = layer0_params['kalman_R']
    vwap_weight = layer0_params['vwap_weight']
    vwap_lookback = layer0_params['vwap_lookback']
    hampel_filter_enabled = layer0_params.get('hampel_filter_enabled', False)
    hampel_window = layer0_params.get('hampel_window', 5)
    hampel_threshold = layer0_params.get('hampel_threshold', 3.0)
    adaptive_kalman_enabled = layer0_params.get('adaptive_kalman_enabled', False)
    adaptive_noise_window = layer0_params.get('adaptive_noise_window', 50)
    adaptive_adaptation_rate = layer0_params.get('adaptive_adaptation_rate', 0.1)
    robust_vwap_enabled = layer0_params.get('robust_vwap_enabled', False)
    robust_min_lookback = layer0_params.get('robust_min_lookback', 20)
    robust_max_lookback = layer0_params.get('robust_max_lookback', 200)
    robust_volatility_window = layer0_params.get('robust_volatility_window', 20)
    savgol_filter_enabled = layer0_params.get('savgol_filter_enabled', False)
    savgol_window = layer0_params.get('savgol_window', 21)
    savgol_order = layer0_params.get('savgol_order', 3)
    
    # Validate required columns
    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain 'close' column")
    if 'volume' not in df.columns:
        logger.warning("No volume column found, using pure Kalman price")
        vwap_weight = 0.0
        robust_vwap_enabled = False
    
    try:
        # Import KalmanFilter1D from orthogonal_label_generation
        from .orthogonal_label_generation import KalmanFilter1D, calc_vwap
        
        # Choose price generation method based on enabled features
        if adaptive_kalman_enabled:
            # Use Adaptive Kalman with dynamic noise estimation
            logger.debug("Using Adaptive Kalman with dynamic noise estimation")
            kalman_price = generate_adaptive_kalman_price(
                df, Q, R, adaptive_noise_window, adaptive_adaptation_rate
            )
        else:
            # Use standard Kalman filter
            kalman_price, kalman_uncertainty = KalmanFilter1D(Q=Q, R=R).filter_series(df['close'])
        
        # Generate VWAP price with enhanced robust method if enabled
        if robust_vwap_enabled and 'volume' in df.columns:
            logger.debug("Using Robust VWAP with adaptive window sizing")
            vwap_price = generate_robust_vwap_price(
                df, vwap_lookback, robust_min_lookback, robust_max_lookback, robust_volatility_window
            )
        elif vwap_weight > 0 and 'volume' in df.columns:
            vwap_price = calc_vwap(df['close'], df['volume'], vwap_lookback)
        else:
            vwap_price = kalman_price
        
        # Composite price with Layer0-optimized weights
        composite_price = (1 - vwap_weight) * kalman_price + vwap_weight * vwap_price
        
        # Apply Hampel Filter if enabled (outlier removal)
        if hampel_filter_enabled and len(composite_price) > hampel_window:
            composite_price = apply_hampel_filter(composite_price, hampel_window, hampel_threshold)
        
        # Apply Savitzky-Golay Filter if enabled (feature preservation)
        if savgol_filter_enabled and len(composite_price) > savgol_window:
            composite_price = apply_savgol_filter(composite_price, savgol_window, savgol_order)
        
        # Ensure no NaN values
        composite_price = composite_price.fillna(method='ffill').fillna(df['close'])
        
        logger.debug(f"Generated unified Layer2 price: Q={Q}, R={R}, vwap_weight={vwap_weight}")
        logger.debug(f"Enhanced features: hampel_filter={hampel_filter_enabled}, adaptive_kalman={adaptive_kalman_enabled}, robust_vwap={robust_vwap_enabled}, savgol_filter={savgol_filter_enabled}")
        return composite_price
        
    except Exception as e:
        logger.error(f"Failed to generate unified price: {e}, falling back to raw close")
        return df['close']

def apply_savgol_filter(price_series: pd.Series, window_length: int = 21, poly_order: int = 3) -> pd.Series:
    """
    Apply Savitzky-Golay filter for feature preservation and timing accuracy.
    
    The Savitzky-Golay filter excels at:
    - Maintaining peaks, valleys, and momentum patterns
    - Preserving signal phase relationships (no lag)
    - O(N) complexity, suitable for real-time
    - Simple parameters: window size and polynomial order
    - Better edge preservation than moving averages for turning points
    
    Args:
        price_series: Input price series
        window_length: Window length (must be odd)
        poly_order: Polynomial order (must be < window_length)
        
    Returns:
        Savitzky-Golay filtered price series
    """
    try:
        from scipy.signal import savgol_filter
        
        # Ensure window length is odd and appropriate
        if window_length % 2 == 0:
            window_length += 1
        
        # Ensure polynomial order is valid
        poly_order = min(poly_order, window_length - 1)
        
        # Apply Savitzky-Golay filter
        filtered_price = savgol_filter(price_series.values, window_length, poly_order)
        
        return pd.Series(filtered_price, index=price_series.index)
        
    except ImportError:
        logger.warning("SciPy not available, falling back to moving average")
        return price_series.rolling(window_length, center=True, min_periods=1).mean()
    except Exception as e:
        logger.error(f"Savitzky-Golay filtering failed: {e}")
        return price_series

def apply_hampel_filter(price_series: pd.Series, window: int = 5, threshold: float = 3.0) -> pd.Series:
    """
    Apply Hampel filter to remove outliers while preserving signal characteristics.
    
    The Hampel filter identifies and replaces outliers using a sliding window
    approach with median and MAD (Median Absolute Deviation) statistics.
    
    Args:
        price_series: Input price series
        window: Window size for outlier detection (must be odd)
        threshold: Threshold in MAD units for outlier detection (default: 3.0)
        
    Returns:
        Hampel-filtered price series
    """
    if window % 2 == 0:
        window += 1  # Ensure odd window size
    
    filtered_price = price_series.copy()
    half_window = window // 2
    
    for i in range(half_window, len(price_series) - half_window):
        # Extract window
        window_data = price_series.iloc[i - half_window:i + half_window + 1]
        
        # Calculate median and MAD
        median = window_data.median()
        mad = np.median(np.abs(window_data - median))
        
        # Calculate threshold
        if mad > 0:
            threshold_value = threshold * mad
        else:
            threshold_value = 0
        
        # Check if current point is outlier
        if abs(price_series.iloc[i] - median) > threshold_value:
            # Replace with median
            filtered_price.iloc[i] = median
    
    return filtered_price

def generate_adaptive_kalman_price(df: pd.DataFrame, 
                                 base_Q: float = 1e-4,
                                 base_R: float = 0.01,
                                 noise_window: int = 50,
                                 adaptation_rate: float = 0.1) -> pd.Series:
    """
    Generate Adaptive Kalman price with dynamic noise estimation.
    
    Extends current Kalman filter by dynamically estimating measurement noise (R)
    from recent price volatility, making it responsive to changing market conditions.
    
    Args:
        df: DataFrame with close column
        base_Q: Base process noise parameter
        base_R: Base measurement noise parameter  
        noise_window: Window for noise estimation
        adaptation_rate: Rate of parameter adaptation (0-1)
        
    Returns:
        Adaptive Kalman-filtered price series
    """
    try:
        from .orthogonal_label_generation import KalmanFilter1D
        
        close = df['close']
        adaptive_price = close.copy()
        current_Q = base_Q
        current_R = base_R
        
        # Initialize Kalman filter
        kalman_filter = KalmanFilter1D(Q=current_Q, R=current_R)
        
        # Process price series with adaptive noise estimation
        for i in range(1, len(close)):
            # Estimate recent volatility (measurement noise proxy)
            if i >= noise_window:
                recent_returns = close.iloc[i-noise_window:i].pct_change().dropna()
                if len(recent_returns) > 1:
                    # Estimate measurement noise from recent volatility
                    recent_volatility = recent_returns.std()
                    # Adaptive R estimation (higher volatility = higher measurement noise)
                    estimated_R = max(base_R * 0.1, recent_volatility * 0.5)
                    
                    # Smooth adaptation
                    current_R = (1 - adaptation_rate) * current_R + adaptation_rate * estimated_R
            
            # Update Kalman filter with current parameters
            kalman_filter.Q = current_Q
            kalman_filter.R = current_R
            
            # Filter current price point
            filtered_price, _ = kalman_filter.filter_point(close.iloc[i])
            adaptive_price.iloc[i] = filtered_price
        
        return adaptive_price
        
    except Exception as e:
        logger.error(f"Adaptive Kalman filtering failed: {e}")
        return df['close']

def generate_robust_vwap_price(df: pd.DataFrame,
                              base_lookback: int = 50,
                              min_lookback: int = 20,
                              max_lookback: int = 200,
                              volatility_window: int = 20) -> pd.Series:
    """
    Generate Robust Volume-Weighted Average Price with adaptive window sizing.
    
    Extends VWAP with adaptive window sizing based on volume volatility profile.
    More responsive to liquidity changes and volume pattern shifts.
    
    Args:
        df: DataFrame with close, volume columns
        base_lookback: Base VWAP lookback period
        min_lookback: Minimum lookback (for high volatility)
        max_lookback: Maximum lookback (for low volatility)
        volatility_window: Window for volatility calculation
        
    Returns:
        Robust VWAP price series
    """
    try:
        close = df['close']
        volume = df['volume']
        
        # Calculate volume volatility profile
        volume_volatility = volume.pct_change().rolling(volatility_window).std()
        
        # Adaptive lookback based on volume volatility
        # High volatility = shorter lookback (more responsive)
        # Low volatility = longer lookback (smoother)
        volatility_normalized = volume_volatility / volume_volatility.rolling(100).mean()
        adaptive_lookback = (
            min_lookback + 
            (max_lookback - min_lookback) * (1 - volatility_normalized)
        ).astype(int)
        
        # Ensure lookback doesn't exceed available data
        adaptive_lookback = min(adaptive_lookback, len(df))
        
        # Calculate robust VWAP with adaptive lookback
        pv = close * volume
        cum_pv = pv.rolling(adaptive_lookback).sum()
        cum_vol = volume.rolling(adaptive_lookback).sum()
        robust_vwap = cum_pv / (cum_vol + 1e-9)
        
        # Fill initial NaN values with simple VWAP
        simple_vwap = calc_vwap(close, volume, base_lookback)
        robust_vwap = robust_vwap.fillna(simple_vwap)
        
        return robust_vwap
        
    except Exception as e:
        logger.error(f"Robust VWAP generation failed: {e}")
        return calc_vwap(df['close'], df.get('volume', pd.Series(0, index=df.index)), base_lookback)

class UnifiedPriceMixin:
    """Mixin class for Layer2 generators to use unified price."""
    
    def __init__(self, use_unified_price: bool = True, layer0_params: dict = None):
        self.use_unified_price = use_unified_price
        self._layer0_params = layer0_params or load_layer0_params()
        self._cached_unified_price = None
        self._cached_timestamp = None
    
    def _get_unified_price(self, df: pd.DataFrame) -> pd.Series:
        """Get cached unified price or generate new one."""
        if not self.use_unified_price:
            return df['close']
        
        # Check cache validity (avoid re-computation)
        current_time = df.index[-1] if len(df) > 0 else None
        if (self._cached_unified_price is not None and 
            self._cached_timestamp == current_time and 
            len(self._cached_unified_price) == len(df)):
            return self._cached_unified_price
        
        # Generate unified price
        unified_price = generate_unified_layer2_price(df, self._layer0_params)
        
        # Cache for reuse
        self._cached_unified_price = unified_price
        self._cached_timestamp = current_time
        
        return unified_price
    
    def _invalidate_price_cache(self):
        """Invalidate cached price (for parameter updates)."""
        self._cached_unified_price = None
        self._cached_timestamp = None

# Live Trading Edge Preservation
class LiveTradingPriceManager:
    """
    Manages unified price generation in live trading to preserve edge.
    
    Key strategies:
    1. Parameter stability monitoring
    2. Real-time price quality validation  
    3. Adaptive fallback mechanisms
    4. Performance tracking
    """
    
    def __init__(self, layer0_params: dict = None):
        self.layer0_params = layer0_params or load_layer0_params()
        self.price_quality_history = []
        self.parameter_drift_threshold = 0.1
        self.last_quality_check = None
        
    def generate_live_price(self, df: pd.DataFrame) -> pd.Series:
        """Generate unified price with live trading safeguards."""
        try:
            # Generate unified price
            unified_price = generate_unified_layer2_price(df, self.layer0_params)
            
            # Quality validation
            quality_score = self._validate_price_quality(df, unified_price)
            self.price_quality_history.append(quality_score)
            
            # Check for parameter drift
            if self._detect_parameter_drift():
                logger.warning("Parameter drift detected, consider re-optimization")
                self._trigger_parameter_update()
            
            return unified_price
            
        except Exception as e:
            logger.error(f"Live price generation failed: {e}, using fallback")
            return self._generate_fallback_price(df)
    
    def _validate_price_quality(self, df: pd.DataFrame, unified_price: pd.Series) -> float:
        """Validate unified price quality vs raw price."""
        try:
            # Calculate quality metrics
            raw_price = df['close']
            
            # 1. Smoothness (less noise = better)
            raw_volatility = raw_price.pct_change().std()
            unified_volatility = unified_price.pct_change().std()
            smoothness_score = 1 - (unified_volatility / raw_volatility)
            
            # 2. Tracking accuracy (how well it follows raw price)
            tracking_error = np.mean((unified_price - raw_price) ** 2)
            max_acceptable_error = (raw_price.std() * 0.01) ** 2  # 1% of std
            tracking_score = 1 - min(tracking_error / max_acceptable_error, 1.0)
            
            # 3. Volume consistency (if volume available)
            volume_score = 1.0
            if 'volume' in df.columns:
                # Check if price moves align with volume
                price_volume_corr = abs(unified_price.pct_change().corr(df['volume'].pct_change()))
                volume_score = price_volume_corr  # Higher correlation = better
            
            # Composite quality score
            quality_score = (smoothness_score * 0.4 + 
                            tracking_score * 0.4 + 
                            volume_score * 0.2)
            
            return np.clip(quality_score, 0, 1)
            
        except Exception as e:
            logger.warning(f"Quality validation failed: {e}")
            return 0.5  # Neutral score
    
    def _detect_parameter_drift(self) -> bool:
        """Detect if current parameters are drifting from optimal."""
        if len(self.price_quality_history) < 100:
            return False
        
        # Check recent quality vs historical average
        recent_quality = np.mean(self.price_quality_history[-20:])
        historical_quality = np.mean(self.price_quality_history[:-20])
        
        quality_drop = historical_quality - recent_quality
        return quality_drop > self.parameter_drift_threshold
    
    def _trigger_parameter_update(self):
        """Trigger parameter re-optimization in live trading."""
        logger.info("Triggering parameter update due to quality degradation")
        # In live trading, this would trigger a background re-optimization
        # For now, just log the event
        pass
    
    def _generate_fallback_price(self, df: pd.DataFrame) -> pd.Series:
        """Generate fallback price if unified price fails."""
        try:
            # Fallback to simple exponential smoothing
            raw_price = df['close']
            fallback_price = raw_price.ewm(span=20, adjust=False).mean()
            return fallback_price.fillna(method='ffill').fillna(raw_price)
        except Exception as e:
            logger.error(f"Fallback price generation failed: {e}")
            return df['close']
    
    def get_performance_metrics(self) -> dict:
        """Get live trading performance metrics."""
        if not self.price_quality_history:
            return {}
        
        return {
            'avg_quality': np.mean(self.price_quality_history),
            'quality_trend': np.mean(self.price_quality_history[-10:]) - np.mean(self.price_quality_history[-50:-10]) if len(self.price_quality_history) > 50 else 0,
            'quality_stability': np.std(self.price_quality_history[-20:]) if len(self.price_quality_history) >= 20 else 0,
            'total_samples': len(self.price_quality_history)
        }
