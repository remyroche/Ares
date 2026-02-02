"""
Causal Residual Computation for Layer 2.5 Chaser

Computes the target for the Chaser: y~ = y_actual - y_causal_anchor

This ensures the Chaser only learns "unexplained alpha" and doesn't
waste capacity on simple linear relationships already captured by
the Causal Anchor.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Tuple
import warnings

# Try to import numba
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    # Fallback print functions
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")

@jit(nopython=True, fastmath=True, cache=True)
def _compute_residuals_numba(
    y_actual: np.ndarray,
    y_anchor: np.ndarray,
    min_residual_threshold: float,
    max_residual_threshold: float,
    clip_residuals: bool
) -> np.ndarray:
    """
    Numba-optimized residual computation.
    Single-pass (mostly) logic for validation, subtraction and clipping.
    Operates on float32/64 arrays.
    """
    n = len(y_actual)
    # Ensure we use float32 for output to save memory, assuming inputs are compatible
    out = np.empty(n, dtype=np.float32)

    # Constants
    FLOAT_LIMIT = 1e30 # Safe limit below max float32

    # Stats accumulators for pass 1
    sum_val = 0.0
    sum_sq = 0.0

    # Pass 1: Compute raw residuals, handling Inf/NaN/Clipping
    for i in range(n):
        act = y_actual[i]
        anc = y_anchor[i]

        # Check validity (NaN or Inf)
        # Note: In Numba, np.isnan/isinf work on scalars
        act_invalid = np.isnan(act) or np.isinf(act)
        anc_invalid = np.isnan(anc) or np.isinf(anc)

        if act_invalid or anc_invalid:
            # If either is missing/invalid, we cannot compute a valid residual.
            # Standard logic is to treat as 0 (neutral).
            res = 0.0
        else:
            # Clip extreme inputs to prevent overflow
            if act > FLOAT_LIMIT: act = FLOAT_LIMIT
            elif act < -FLOAT_LIMIT: act = -FLOAT_LIMIT

            if anc > FLOAT_LIMIT: anc = FLOAT_LIMIT
            elif anc < -FLOAT_LIMIT: anc = -FLOAT_LIMIT

            res = act - anc

            # Tiny residual check (set to 0 if too small)
            # if abs(res) < min_residual_threshold: res = 0.0
            # Kept separate to match original logic flow or do here?
            # Original does it at the end. We'll do it here for efficiency.
            if res > 0 and res < min_residual_threshold:
                res = 0.0
            elif res < 0 and res > -min_residual_threshold:
                res = 0.0

        out[i] = res
        sum_val += res
        sum_sq += res * res

    if not clip_residuals:
        return out

    # Calculate stats for clipping
    mean = sum_val / n
    var = (sum_sq / n) - (mean * mean)
    std = np.sqrt(var) if var > 0 else 0.0

    if std == 0:
        return out

    # Determine clip value
    # Original logic: min(max_threshold * std, 3 * std)
    # If max_threshold is typically 10, then 3*std is the bound.
    limit_mult = 3.0
    if max_residual_threshold < 3.0:
        limit_mult = max_residual_threshold

    clip_val = limit_mult * std

    # Pass 2: Clip residuals
    for i in range(n):
        val = out[i]
        if val > clip_val:
            out[i] = clip_val
        elif val < -clip_val:
            out[i] = -clip_val

    return out

def compute_causal_residuals(
    y_actual: Union[pd.Series, np.ndarray],
    y_causal_anchor: Union[pd.Series, np.ndarray],
    min_residual_threshold: float = 1e-8,
    max_residual_threshold: float = 10.0,
    clip_residuals: bool = True,
    verbose: bool = True,
    use_robust_stats: bool = False
) -> pd.Series:
    """
    Compute causal residuals for Chaser targeting.
    
    Formula: y~ = y_actual - y_causal_anchor
    
    Args:
        y_actual: Actual returns/targets
        y_causal_anchor: Causal Anchor predictions
        min_residual_threshold: Minimum residual threshold
        max_residual_threshold: Maximum residual threshold multiplier (std devs)
        clip_residuals: Whether to clip extreme residuals
        verbose: Whether to print statistics
        use_robust_stats: (Unused in fast path) Reserved for future robust scaling
        
    Returns:
        Causal residuals (unexplained alpha)
    """
    try:
        # --- Alignment Handling ---
        # Determine if we need to align indices
        has_index_actual = isinstance(y_actual, (pd.Series, pd.DataFrame))
        has_index_anchor = isinstance(y_causal_anchor, (pd.Series, pd.DataFrame))
        
        result_index = None
        
        if has_index_actual and has_index_anchor:
            # Both have index -> Align safely
            if len(y_actual) != len(y_causal_anchor) or not y_actual.index.equals(y_causal_anchor.index):
                if verbose: tprint_info("🔄 Aligning input Series by index...")
                y_actual, y_causal_anchor = y_actual.align(y_causal_anchor, join='inner')
            result_index = y_actual.index

        elif has_index_actual:
            # Only actual has index -> Assume positional, verify length
            if len(y_actual) != len(y_causal_anchor):
                raise ValueError(f"Length mismatch: y_actual ({len(y_actual)}) vs y_anchor ({len(y_causal_anchor)})")
            result_index = y_actual.index

        elif has_index_anchor:
            # Only anchor has index -> Assume positional
            if len(y_actual) != len(y_causal_anchor):
                raise ValueError(f"Length mismatch: y_actual ({len(y_actual)}) vs y_anchor ({len(y_causal_anchor)})")
            # Usually we want the target's index, but if missing, maybe use anchor's?
            result_index = y_causal_anchor.index
        else:
            # Neither has index
            if len(y_actual) != len(y_causal_anchor):
                raise ValueError(f"Length mismatch: {len(y_actual)} vs {len(y_causal_anchor)}")
        
        # --- Data Extraction & Conversion ---
        # Convert to numpy arrays (float32 for speed/memory)
        # Using values attribute if available, else standard casting
        vals_actual = y_actual.values if has_index_actual else y_actual
        vals_anchor = y_causal_anchor.values if has_index_anchor else y_causal_anchor
        
        # Ensure contiguous float32 array for Numba
        vals_actual = np.ascontiguousarray(vals_actual, dtype=np.float32)
        vals_anchor = np.ascontiguousarray(vals_anchor, dtype=np.float32)
        
        # --- Computation ---
        if NUMBA_AVAILABLE:
            residuals_arr = _compute_residuals_numba(
                vals_actual,
                vals_anchor,
                min_residual_threshold,
                max_residual_threshold,
                clip_residuals
            )
        else:
            # Fallback logic (vectorized numpy)
            if verbose: tprint_warning("Numba not available, using NumPy fallback")

            # Replace Inf with NaN
            vals_actual[np.isinf(vals_actual)] = np.nan
            vals_anchor[np.isinf(vals_anchor)] = np.nan

            # Clip Inputs
            limit = 1e30
            vals_actual = np.clip(vals_actual, -limit, limit)
            vals_anchor = np.clip(vals_anchor, -limit, limit)

            # Compute
            residuals_arr = vals_actual - vals_anchor

            # Handle NaN -> 0
            residuals_arr[np.isnan(residuals_arr)] = 0.0

            # Clip Residuals
            if clip_residuals:
                std = np.std(residuals_arr)
                if std > 0:
                    clip_val = min(max_residual_threshold * std, 3 * std)
                    residuals_arr = np.clip(residuals_arr, -clip_val, clip_val)

            # Tiny threshold
            residuals_arr[np.abs(residuals_arr) < min_residual_threshold] = 0.0

        # --- Output Wrapping ---
        if result_index is not None:
            residuals = pd.Series(residuals_arr, index=result_index, name='residuals')
        else:
            residuals = pd.Series(residuals_arr, name='residuals')

        # --- Logging ---
        if verbose:
            tprint_success("✅ Causal residuals computed:")
            tprint_info(f"   - Samples: {len(residuals)}")
            tprint_info(f"   - Mean: {residuals.mean():.6f}")
            tprint_info(f"   - Std: {residuals.std():.6f}")
            
            # Only compute costly correlation if needed
            if len(residuals) > 100:
                # Use numpy for correlation to stay fast
                # Need to handle potential NaNs in input for correlation if we want accuracy,
                # but our residuals are clean. y_actual might have NaNs (replaced by 0 or ignored).
                # _compute_residuals_numba doesn't fix y_actual in place, so vals_actual might have Infs/NaNs?
                # Actually, vals_actual was converted to float32. Inf/NaNs are still there.
                
                # Safe correlation
                mask = np.isfinite(vals_actual) & np.isfinite(vals_anchor)
                if mask.sum() > 10:
                    corr = np.corrcoef(vals_actual[mask], vals_anchor[mask])[0, 1]
                    tprint_info(f"   - Actual vs Anchor correlation: {corr:.4f}")

                    res_corr = np.corrcoef(residuals_arr[mask], vals_actual[mask])[0, 1]
                    tprint_info(f"   - Residual vs Actual correlation: {res_corr:.4f}")
        
        return residuals
        
    except Exception as e:
        if verbose:
            tprint_error(f"❌ Causal residual computation failed: {e}")
        raise

def analyze_residual_quality(
    y_actual: Union[pd.Series, np.ndarray],
    y_causal_anchor: Union[pd.Series, np.ndarray],
    residuals: Optional[pd.Series] = None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Analyze the quality of causal residuals.
    """
    try:
        if residuals is None:
            residuals = compute_causal_residuals(y_actual, y_causal_anchor, verbose=False)
        
        # Ensure we work with numpy arrays for stats
        res_vals = residuals.values.astype(np.float32)
        act_vals = y_actual.values if hasattr(y_actual, 'values') else y_actual
        anc_vals = y_causal_anchor.values if hasattr(y_causal_anchor, 'values') else y_causal_anchor

        # Mask for valid data
        mask = np.isfinite(act_vals) & np.isfinite(anc_vals) & np.isfinite(res_vals)
        
        if mask.sum() < 2:
            return {}

        res_clean = res_vals[mask]
        act_clean = act_vals[mask]
        anc_clean = anc_vals[mask]

        # Stats
        residual_mean = float(np.mean(res_clean))
        residual_std = float(np.std(res_clean))

        # Use pandas for skew/kurt as it's convenient and not performance critical here
        res_series = pd.Series(res_clean)
        residual_skew = res_series.skew()
        residual_kurt = res_series.kurt() # Use .kurt() per pandas 3.0 standards
        
        # Correlations
        actual_anchor_corr = np.corrcoef(act_clean, anc_clean)[0, 1]
        residual_actual_corr = np.corrcoef(res_clean, act_clean)[0, 1]
        residual_anchor_corr = np.corrcoef(res_clean, anc_clean)[0, 1]

        # Signal Quality
        # Signal-to-Noise: Mean is usually bias (bad). We want predictable volatility?
        # Standard definition: Mean / Std.
        signal_to_noise = abs(residual_mean) / residual_std if residual_std > 0 else 0
        predictability = abs(residual_actual_corr)
        
        # Explained Variance
        anchor_explained_variance = actual_anchor_corr ** 2
        
        # Orthogonality (Target: 1.0)
        orthogonality = 1.0 - abs(residual_anchor_corr)

        quality_score = orthogonality * predictability
        
        metrics = {
            'residual_mean': residual_mean,
            'residual_std': residual_std,
            'residual_skew': residual_skew,
            'residual_kurtosis': residual_kurt,
            'actual_anchor_correlation': actual_anchor_corr,
            'residual_actual_correlation': residual_actual_corr,
            'residual_anchor_correlation': residual_anchor_corr,
            'signal_to_noise': signal_to_noise,
            'predictability': predictability,
            'anchor_explained_variance': anchor_explained_variance,
            'orthogonality': orthogonality,
            'quality_score': quality_score
        }
        
        if verbose:
            tprint_info("📊 Residual Quality Analysis:")
            tprint_info(f"   - Residual std: {residual_std:.6f}")
            tprint_info(f"   - Orthogonality: {orthogonality:.4f}")
            tprint_info(f"   - Quality score: {quality_score:.4f}")
        
        return metrics
        
    except Exception as e:
        if verbose:
            tprint_error(f"❌ Residual quality analysis failed: {e}")
        raise

def validate_residual_targets(
    residuals: pd.Series,
    min_samples: int = 100,
    min_variance: float = 1e-6,
    max_skewness: float = 10.0,
    max_kurtosis: float = 50.0,
    verbose: bool = True
) -> Dict[str, bool]:
    """
    Validate that residuals are suitable for Chaser training.
    """
    try:
        validation_results = {}
        
        # Sample size check
        validation_results['sufficient_samples'] = len(residuals) >= min_samples
        
        # Variance check
        residual_variance = residuals.var()
        validation_results['sufficient_variance'] = residual_variance >= min_variance
        
        # Distribution checks
        residual_skew = residuals.skew()
        residual_kurt = residuals.kurt() # Updated to .kurt()

        validation_results['reasonable_skewness'] = abs(residual_skew) <= max_skewness
        validation_results['reasonable_kurtosis'] = abs(residual_kurt) <= max_kurtosis
        
        # Overall validity
        validation_results['valid_for_training'] = all(validation_results.values())
        
        if verbose:
            tprint_info("🔍 Residual Validation:")
            for check, passed in validation_results.items():
                status = "✅" if passed else "❌"
                tprint_info(f"   {status} {check}: {passed}")
        
        return validation_results
        
    except Exception as e:
        if verbose:
            tprint_error(f"❌ Residual validation failed: {e}")
        raise

def residual_pipeline(
    y_actual: Union[pd.Series, np.ndarray],
    y_causal_anchor: Union[pd.Series, np.ndarray],
    validate: bool = True,
    analyze: bool = True,
    **kwargs
) -> Tuple[pd.Series, Optional[Dict[str, float]], Optional[Dict[str, bool]]]:
    """
    Complete residual pipeline: compute, analyze, and validate.
    """
    # Compute residuals
    residuals = compute_causal_residuals(y_actual, y_causal_anchor, **kwargs)
    
    # Analyze quality
    quality_metrics = None
    if analyze:
        quality_metrics = analyze_residual_quality(y_actual, y_causal_anchor, residuals)
    
    # Validate
    validation_results = None
    if validate:
        validation_results = validate_residual_targets(residuals)
    
    return residuals, quality_metrics, validation_results
