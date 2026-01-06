import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from src.utils.ml_common.afml_utils import (
    get_daily_vol, get_t_events, get_vertical_barrier, 
    apply_triple_barrier, get_bins, get_weights_by_uniqueness,
    frac_diff_fixed
)

logger = logging.getLogger(__name__)

class AFMLSpecialistMixin:
    """
    Mixin providing AFML-specific logic for enhanced specialists:
    - CUSUM Filtering for event-based sampling
    - Triple Barrier Method for path-aware labeling
    - Sample Uniqueness weighting to handle overlap (Concurrence)
    - Fractional Differentiation for memory preservation
    """
    
    def apply_afml_sampling(self, df: pd.DataFrame, config: Dict[str, Any], filter_type: str = 'price') -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
        """
        Apply CUSUM filtering to sample meaningful events, targeting ~10% of total bars.
        Uses an enhanced binary search to find the optimal threshold factor.
        Supports: 'price', 'volatility', 'volume', 'spread'
        """
        if filter_type == 'price':
            series = df['close']
            threshold_base = get_daily_vol(series)
        elif filter_type == 'volatility':
            # Log-volatility returns
            vol = df['close'].pct_change().rolling(20).std().fillna(0)
            # Use small epsilon to avoid log(0)
            series = np.log((vol + 1e-9) / (vol.shift(1) + 1e-9)).fillna(0)
            threshold_base = series.rolling(100).std()
        elif filter_type == 'volume':
            volume = df.get('volume', pd.Series(1, index=df.index))
            log_vol = np.log1p(volume)
            series = log_vol.diff().fillna(0)
            threshold_base = series.rolling(100).std()
        elif filter_type == 'spread':
            # Use high-low range as spread proxy
            spread = (df['high'] - df['low']) / (df['close'] + 1e-8)
            series = spread.diff().fillna(0)
            threshold_base = series.rolling(100).std()
        else:
            series = df['close']
            threshold_base = get_daily_vol(series)
            
        threshold_base = threshold_base.fillna(method='bfill').fillna(method='ffill')
        
        # Binary search for threshold_factor to target ~10% sampling rate
        target_rate = config.get('afml_target_sampling_rate', 0.10)
        target_count = int(len(df) * target_rate)
        
        # Wide search range [1e-6, 1e6] for extreme series like spread or volatility
        low, high = 1e-6, 1000000.0
        best_factor = 1.0
        best_events = df.index
        min_diff = float('inf')
        
        # 25 steps of binary search for high precision across wide range
        for _ in range(25):
            mid = (low + high) / 2
            t_events = get_t_events(series, threshold_base * mid)
            count = len(t_events)
            
            if count == 0:
                # Too high, lower it
                high = mid
                continue
                
            if abs(count - target_count) < min_diff:
                min_diff = abs(count - target_count)
                best_factor = mid
                best_events = t_events
                
            if count > target_count:
                # Too many events, increase threshold
                low = mid
            else:
                # Too few events, decrease threshold
                high = mid
                
        t_events = best_events
        if len(t_events) == 0:
            # Emergency fallback if search failed
            t_events = df.index[::int(1/target_rate)]
            logger.warning(f"AFML {filter_type} Sampling FAILED to find threshold. Using periodic fallback.")
            
        logger.info(f"AFML {filter_type.capitalize()} Sampling: {len(t_events)} events (Rate: {len(t_events)/len(df):.1%}, Target: {target_rate:.1%}, Factor: {best_factor:.6f})")
        
        return df.loc[t_events], t_events

    def apply_sequential_bootstrap(self, t1: pd.Series, close_index: pd.Index, num_samples: Optional[int] = None) -> List[pd.Timestamp]:
        """Apply sequential bootstrapping to get robust non-overlapping samples."""
        from src.utils.ml_common.afml_utils import seq_bootstrap
        return seq_bootstrap(t1, close_index, num_samples)

    def generate_tbm_labels(self, df: pd.DataFrame, t_events: pd.DatetimeIndex, 
                           config: Dict[str, Any], pt_sl: List[float]) -> pd.DataFrame:
        """Generate Triple Barrier Method labels."""
        close = df['close']
        vol = get_daily_vol(close)
        vol = vol.fillna(method='bfill').fillna(method='ffill')
        
        lookforward = config.get('lookforward_bars', 35)
        vertical_barrier = get_vertical_barrier(close, t_events, lookforward)
        
        # Apply Triple Barrier
        tbm_events = apply_triple_barrier(
            close=close,
            t_events=t_events,
            pt_sl=pt_sl,
            target=vol,
            min_ret=config.get('min_ret', 0.001),
            vertical_barrier=vertical_barrier
        )
        
        labels_df = get_bins(tbm_events, close)
        return labels_df

    def get_concurrent_weights(self, t1: pd.Series, close_index: pd.Index) -> pd.Series:
        """Calculate sample weights based on average uniqueness (Concurrence fix)."""
        weights = get_weights_by_uniqueness(t1, close_index)
        return weights

    def apply_fractional_diff(self, series: pd.Series, d: float = 0.5) -> pd.Series:
        """Apply fractional differentiation to preserve memory in non-stationary series."""
        return frac_diff_fixed(series, d)

    def compute_binned_mi(self, x: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
        """Compute a fast binned MI score for reporting."""
        try:
            from sklearn.metrics import mutual_info_score
            if len(x) < 2 or len(np.unique(y)) < 2:
                return 0.0
            # Clean data
            mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
            x_c, y_c = x[mask], y[mask]
            if len(x_c) < 2:
                return 0.0
            
            # Bin continuous x
            x_edges = np.histogram_bin_edges(x_c, bins=bins)
            x_binned = np.digitize(x_c, x_edges)
            
            return float(mutual_info_score(x_binned, y_c))
        except Exception:
            return 0.0
