"""
Causal Residual Computation for Layer 2.5 Chaser

Computes the target for the Chaser: y~ = y_actual - y_causal_anchor

This ensures the Chaser only learns "unexplained alpha" and doesn't
waste capacity on simple linear relationships already captured by
the Causal Anchor.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Tuple, Literal
import warnings

# Try to import numba
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def prange(n):
        return range(n)

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
def _compute_residuals_numba_1d(
    y_actual: np.ndarray,
    y_anchor: np.ndarray,
    min_residual_threshold: float,
    max_residual_threshold: float,
    clipping_method: int, # 0=none, 1=std, 2=mad, 3=iqr
    adjust_beta: bool
) -> np.ndarray:
    """
    Numba-optimized residual computation for 1D arrays (float32).
    """
    n = len(y_actual)
    out = np.empty(n, dtype=np.float32)
    FLOAT_LIMIT = 1e30

    # Stats accumulators for pass 1 (if needed)
    sum_act = 0.0
    sum_anc = 0.0
    sum_act_sq = 0.0
    sum_anc_sq = 0.0
    sum_prod = 0.0
    count_valid = 0

    # Beta adjustment
    beta = 1.0

    # Pass 1: Compute beta (if requested) or just count valid
    if adjust_beta:
        for i in range(n):
            act = y_actual[i]
            anc = y_anchor[i]
            if not (np.isnan(act) or np.isinf(act) or np.isnan(anc) or np.isinf(anc)):
                sum_act += act
                sum_anc += anc
                sum_act_sq += act * act
                sum_anc_sq += anc * anc
                sum_prod += act * anc
                count_valid += 1

        if count_valid > 1:
            mean_act = sum_act / count_valid
            mean_anc = sum_anc / count_valid
            var_anc = (sum_anc_sq / count_valid) - (mean_anc * mean_anc)
            cov = (sum_prod / count_valid) - (mean_act * mean_anc)
            if var_anc > 1e-12:
                beta = cov / var_anc
            else:
                beta = 0.0 if var_anc == 0 else 1.0 # Fallback
        else:
            beta = 1.0 # Not enough data

    # Pass 2: Compute raw residuals
    # We can compute stats for clipping on the fly if method=std
    sum_res = 0.0
    sum_res_sq = 0.0
    count_res_valid = 0

    # Temp storage for median/iqr not feasible without allocation,
    # but for MAD we need median. For now, let's store valid residuals back in `out`
    # and mark invalids with NaN.

    for i in range(n):
        act = y_actual[i]
        anc = y_anchor[i]

        act_invalid = np.isnan(act) or np.isinf(act)
        anc_invalid = np.isnan(anc) or np.isinf(anc)

        if act_invalid or anc_invalid:
            res = 0.0 # Default to 0
            # Mark as NaN temporarily if we need to compute robust stats?
            # Actually, filling with 0 is safer for pipelines.
            # But for stats calc, 0 might bias.
            # Let's use 0.0 as neutral.
        else:
            # Clip extreme inputs
            if act > FLOAT_LIMIT: act = FLOAT_LIMIT
            elif act < -FLOAT_LIMIT: act = -FLOAT_LIMIT

            if anc > FLOAT_LIMIT: anc = FLOAT_LIMIT
            elif anc < -FLOAT_LIMIT: anc = -FLOAT_LIMIT

            res = act - (beta * anc)

            # Tiny residual check
            if res > 0 and res < min_residual_threshold:
                res = 0.0
            elif res < 0 and res > -min_residual_threshold:
                res = 0.0

        out[i] = res
        sum_res += res
        sum_res_sq += res * res
        count_res_valid += 1

    if clipping_method == 0:
        return out

    # Determine clip value
    clip_val = FLOAT_LIMIT

    if clipping_method == 1: # STD
        mean = sum_res / count_res_valid if count_res_valid > 0 else 0.0
        var = (sum_res_sq / count_res_valid) - (mean * mean) if count_res_valid > 0 else 0.0
        std = np.sqrt(var) if var > 0 else 0.0

        limit_mult = 3.0
        if max_residual_threshold < 10.0: # Heuristic
            limit_mult = max_residual_threshold

        if std > 0:
            clip_val = limit_mult * std

    elif clipping_method == 2 or clipping_method == 3: # MAD or IQR
        # Need to sort valid residuals
        # Copy to new array to sort
        # Using a fixed size buffer or dynamic? Numba supports dynamic alloc
        # But `out` is already filled. We can sort `out`? No, order matters.
        # We need a copy.
        temp = out.copy()
        # Filter out 0s? No, 0 is a valid residual.
        # But if we filled NaNs with 0, it affects distribution.
        # Assuming 0 is correct neutral value.

        # Sort
        temp.sort()

        if count_res_valid > 0:
            if clipping_method == 2: # MAD
                # median = np.median(temp)
                mid = count_res_valid // 2
                median = temp[mid] # Approx

                # Compute MAD
                # MAD = median(|x - median|)
                # In place update of temp to absolute deviation
                for i in range(count_res_valid):
                    val = temp[i] - median
                    temp[i] = -val if val < 0 else val

                temp.sort()
                mad = temp[mid]

                sigma = 1.4826 * mad
                clip_val = max_residual_threshold * sigma

            elif clipping_method == 3: # IQR
                q1 = temp[int(count_res_valid * 0.25)]
                q3 = temp[int(count_res_valid * 0.75)]
                iqr = q3 - q1
                # Tukey fences: Q3 + 1.5 IQR
                # But here we want symmetric clipping around 0 or median?
                # Usually we clip to [Q1 - k*IQR, Q3 + k*IQR]
                upper = q3 + max_residual_threshold * iqr
                lower = q1 - max_residual_threshold * iqr

                # Apply asymmetric clipping
                for i in range(n):
                    val = out[i]
                    if val > upper:
                        out[i] = upper
                    elif val < lower:
                        out[i] = lower
                return out

    # Symmetric Clipping (STD and MAD)
    if clip_val < FLOAT_LIMIT:
        for i in range(n):
            val = out[i]
            if val > clip_val:
                out[i] = clip_val
            elif val < -clip_val:
                out[i] = -clip_val

    return out

@jit(nopython=True, fastmath=True, parallel=True)
def _compute_residuals_numba_2d(
    y_actual: np.ndarray,
    y_anchor: np.ndarray,
    min_residual_threshold: float,
    max_residual_threshold: float,
    clipping_method: int,
    adjust_beta: bool
) -> np.ndarray:
    """
    Numba-optimized residual computation for 2D arrays (Time x Assets).
    Processing is done column-wise (per asset) in parallel.
    """
    rows, cols = y_actual.shape
    out = np.empty((rows, cols), dtype=np.float32)

    for j in prange(cols):
        # Extract columns
        # Numba creates copies or views? Views usually.
        # We can call the 1D function

        # Need to create contiguous copies for the 1D function if passing slices?
        # Numba handles slices.
        col_act = y_actual[:, j]
        col_anc = y_anchor[:, j]

        # We cannot call another @jit function easily inside parallel loop unless inlined?
        # Actually we can call other jit functions.
        # However, to be safe and avoid overhead, we can just call it.
        # NOTE: Parallel loop in numba requires careful management of allocations.
        # But _compute_residuals_numba_1d allocates 'out'. We want to write to 'out[:, j]'.
        # So we should modify 1D function to accept output buffer or return it.
        # Returning array in parallel loop is tricky (allocation).
        # Better to inline the logic or make a helper that writes to buffer.

        # Let's assume we return array and assign it.
        # out[:, j] = result

        # Note: In nopython mode, assigning array to slice works.
        res_col = _compute_residuals_numba_1d(
            col_act, col_anc,
            min_residual_threshold, max_residual_threshold,
            clipping_method, adjust_beta
        )
        for i in range(rows):
            out[i, j] = res_col[i]

    return out

def compute_causal_residuals(
    y_actual: Union[pd.Series, pd.DataFrame, np.ndarray],
    y_causal_anchor: Union[pd.Series, pd.DataFrame, np.ndarray],
    min_residual_threshold: float = 1e-8,
    max_residual_threshold: float = 10.0,
    clip_residuals: bool = True, # Deprecated/Wrapper for method='std'
    clipping_method: Literal['none', 'std', 'mad', 'iqr'] = 'std',
    adjust_beta: bool = False,
    standardize_output: bool = False,
    verbose: bool = True
) -> Union[pd.Series, pd.DataFrame]:
    """
    Compute causal residuals for Chaser targeting.
    
    Formula: y~ = y_actual - (beta * y_causal_anchor)
    
    Args:
        y_actual: Actual returns/targets (1D or 2D)
        y_causal_anchor: Causal Anchor predictions (1D or 2D)
        min_residual_threshold: Minimum residual threshold (below this = 0)
        max_residual_threshold: Maximum residual threshold multiplier (std/mad devs)
        clip_residuals: (Legacy) If False, sets clipping_method='none'
        clipping_method: Method for clipping ('std', 'mad', 'iqr', 'none')
        adjust_beta: If True, computes dynamic beta to remove linear correlation
        standardize_output: If True, z-scores the residuals
        verbose: Whether to print statistics
        
    Returns:
        Causal residuals (unexplained alpha)
    """
    try:
        # Handle legacy clip_residuals
        if not clip_residuals:
            clipping_method = 'none'

        # Map method to int for Numba
        method_map = {'none': 0, 'std': 1, 'mad': 2, 'iqr': 3}
        method_int = method_map.get(clipping_method, 1)

        # --- Alignment Handling ---
        # Determine if we need to align indices
        has_index_actual = isinstance(y_actual, (pd.Series, pd.DataFrame))
        has_index_anchor = isinstance(y_causal_anchor, (pd.Series, pd.DataFrame))
        
        result_obj = None # Template for result (index/columns)
        
        if has_index_actual and has_index_anchor:
            # Check for DataFrame vs Series mix
            # We assume structure matches (both Series or both DataFrame with same cols)
            # Alignment handles index.
            if len(y_actual) != len(y_causal_anchor) or not y_actual.index.equals(y_causal_anchor.index):
                if verbose: tprint_info("🔄 Aligning input data by index...")
                y_actual, y_causal_anchor = y_actual.align(y_causal_anchor, join='inner', axis=0)
            result_obj = y_actual

        elif has_index_actual:
            if len(y_actual) != len(y_causal_anchor):
                raise ValueError(f"Length mismatch: y_actual ({len(y_actual)}) vs y_anchor ({len(y_causal_anchor)})")
            result_obj = y_actual
        elif has_index_anchor:
            if len(y_actual) != len(y_causal_anchor):
                raise ValueError(f"Length mismatch: {len(y_actual)} vs {len(y_causal_anchor)}")
            result_obj = y_causal_anchor
        else:
             if len(y_actual) != len(y_causal_anchor):
                raise ValueError(f"Length mismatch: {len(y_actual)} vs {len(y_causal_anchor)}")

        # --- Data Extraction & Conversion ---
        # Convert to numpy arrays (float32 for speed/memory)
        vals_actual = y_actual.values if hasattr(y_actual, 'values') else y_actual
        vals_anchor = y_causal_anchor.values if hasattr(y_causal_anchor, 'values') else y_causal_anchor
        
        # Ensure contiguous float32 array for Numba
        vals_actual = np.ascontiguousarray(vals_actual, dtype=np.float32)
        vals_anchor = np.ascontiguousarray(vals_anchor, dtype=np.float32)
        
        # Check dimensions
        is_2d = vals_actual.ndim == 2

        # --- Computation ---
        if NUMBA_AVAILABLE:
            if is_2d:
                residuals_arr = _compute_residuals_numba_2d(
                    vals_actual,
                    vals_anchor,
                    min_residual_threshold,
                    max_residual_threshold,
                    method_int,
                    adjust_beta
                )
            else:
                residuals_arr = _compute_residuals_numba_1d(
                    vals_actual,
                    vals_anchor,
                    min_residual_threshold,
                    max_residual_threshold,
                    method_int,
                    adjust_beta
                )
        else:
            # Fallback logic (vectorized numpy)
            if verbose: tprint_warning("Numba not available, using NumPy fallback")

            # Simple 1D fallback logic applied to flattened or broadcast
            # For 2D, we should iterate or broadcast. Numpy handles broadcasting.

            # Replace Inf with NaN
            vals_actual_clean = vals_actual.copy()
            vals_anchor_clean = vals_anchor.copy()
            vals_actual_clean[np.isinf(vals_actual_clean)] = np.nan
            vals_anchor_clean[np.isinf(vals_anchor_clean)] = np.nan

            # Clip Inputs
            limit = 1e30
            vals_actual_clean = np.clip(vals_actual_clean, -limit, limit)
            vals_anchor_clean = np.clip(vals_anchor_clean, -limit, limit)

            beta = 1.0
            if adjust_beta:
                 # Calculate simple global beta (if 2D, this computes global beta across all assets?
                 # Or per column? Fallback is usually simple. Let's do global for simplicity or iterate)
                 # Doing per-column beta in numpy loop is cleaner.
                 if is_2d:
                     # Iterate columns
                     residuals_arr = np.zeros_like(vals_actual_clean)
                     for j in range(vals_actual_clean.shape[1]):
                         act = vals_actual_clean[:, j]
                         anc = vals_anchor_clean[:, j]
                         mask = np.isfinite(act) & np.isfinite(anc)
                         if mask.sum() > 1:
                             cov = np.cov(act[mask], anc[mask])[0, 1]
                             var = np.var(anc[mask])
                             b = cov / var if var > 1e-12 else 0.0
                         else:
                             b = 1.0
                         residuals_arr[:, j] = act - b * anc
                 else:
                     mask = np.isfinite(vals_actual_clean) & np.isfinite(vals_anchor_clean)
                     if mask.sum() > 1:
                         cov = np.cov(vals_actual_clean[mask], vals_anchor_clean[mask])[0, 1]
                         var = np.var(vals_anchor_clean[mask])
                         beta = cov / var if var > 1e-12 else 0.0
                     residuals_arr = vals_actual_clean - beta * vals_anchor_clean
            else:
                residuals_arr = vals_actual_clean - vals_anchor_clean

            # Handle NaN -> 0
            residuals_arr[np.isnan(residuals_arr)] = 0.0

            # Clip Residuals
            if clipping_method != 'none':
                # Per column clipping if 2D? Yes.
                if is_2d:
                     for j in range(residuals_arr.shape[1]):
                         col = residuals_arr[:, j]
                         std = np.std(col)
                         if std > 0:
                             clip_val = max_residual_threshold * std
                             residuals_arr[:, j] = np.clip(col, -clip_val, clip_val)
                else:
                    std = np.std(residuals_arr)
                    if std > 0:
                        clip_val = max_residual_threshold * std
                        residuals_arr = np.clip(residuals_arr, -clip_val, clip_val)

            # Tiny threshold
            residuals_arr[np.abs(residuals_arr) < min_residual_threshold] = 0.0

        # Standardize Output
        if standardize_output:
            if is_2d:
                # Per column Z-score
                 mean = np.mean(residuals_arr, axis=0)
                 std = np.std(residuals_arr, axis=0)
                 # Avoid div by zero
                 std[std < 1e-12] = 1.0
                 residuals_arr = (residuals_arr - mean) / std
            else:
                 std = np.std(residuals_arr)
                 if std > 1e-12:
                     residuals_arr = (residuals_arr - np.mean(residuals_arr)) / std

        # --- Output Wrapping ---
        if result_obj is not None:
            if isinstance(result_obj, pd.DataFrame):
                 residuals = pd.DataFrame(residuals_arr, index=result_obj.index, columns=result_obj.columns)
            else:
                 residuals = pd.Series(residuals_arr, index=result_obj.index, name='residuals')
        else:
            # If input was array
            residuals = residuals_arr

        # --- Logging ---
        if verbose:
            tprint_success("✅ Causal residuals computed:")
            n_samples = len(residuals)
            tprint_info(f"   - Samples: {n_samples}")
            if hasattr(residuals, 'mean'):
                # For DataFrame, take mean of mean?
                mean_val = residuals.values.mean()
                std_val = residuals.values.std()
                tprint_info(f"   - Global Mean: {mean_val:.6f}")
                tprint_info(f"   - Global Std: {std_val:.6f}")
            
        return residuals
        
    except Exception as e:
        if verbose:
            tprint_error(f"❌ Causal residual computation failed: {e}")
        raise

def analyze_residual_quality(
    y_actual: Union[pd.Series, pd.DataFrame, np.ndarray],
    y_causal_anchor: Union[pd.Series, pd.DataFrame, np.ndarray],
    residuals: Optional[Union[pd.Series, pd.DataFrame]] = None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Analyze the quality of causal residuals.
    Supports 1D and 2D inputs (computes average metrics for 2D).
    """
    try:
        if residuals is None:
            residuals = compute_causal_residuals(y_actual, y_causal_anchor, verbose=False)
        
        # Ensure we work with flattened numpy arrays for global stats
        # Or should we do per-asset? Global is easier for summary.

        def to_flat(x):
            v = x.values if hasattr(x, 'values') else x
            return v.flatten().astype(np.float32)

        res_vals = to_flat(residuals)
        act_vals = to_flat(y_actual)
        anc_vals = to_flat(y_causal_anchor)

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

        # Use pandas for skew/kurt
        res_series = pd.Series(res_clean)
        residual_skew = res_series.skew()
        residual_kurt = res_series.kurt()
        
        # Correlations
        actual_anchor_corr = np.corrcoef(act_clean, anc_clean)[0, 1]
        residual_actual_corr = np.corrcoef(res_clean, act_clean)[0, 1]
        residual_anchor_corr = np.corrcoef(res_clean, anc_clean)[0, 1]

        # Signal Quality
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
    residuals: Union[pd.Series, pd.DataFrame],
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
        
        # Flatten for check
        if hasattr(residuals, 'values'):
             vals = residuals.values.flatten()
             vals = vals[np.isfinite(vals)]
             res_flat = pd.Series(vals)
        else:
             res_flat = pd.Series(residuals).dropna()

        # Sample size check
        validation_results['sufficient_samples'] = len(res_flat) >= min_samples
        
        # Variance check
        residual_variance = res_flat.var()
        validation_results['sufficient_variance'] = residual_variance >= min_variance
        
        # Distribution checks
        residual_skew = res_flat.skew()
        residual_kurt = res_flat.kurt()

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
    y_actual: Union[pd.Series, pd.DataFrame, np.ndarray],
    y_causal_anchor: Union[pd.Series, pd.DataFrame, np.ndarray],
    validate: bool = True,
    analyze: bool = True,
    **kwargs
) -> Tuple[Union[pd.Series, pd.DataFrame], Optional[Dict[str, float]], Optional[Dict[str, bool]]]:
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
