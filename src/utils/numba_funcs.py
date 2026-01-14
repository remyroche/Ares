
import numpy as np
from src.utils.tprint import tprint_warning

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Dummy decorator if numba is not installed
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

@jit(nopython=True)
def _numba_generate_dollar_bars(times, opens, highs, lows, closes, vols, thresholds):
    """
    Generate dollar bars using Numba JIT.
    """
    n_rows = len(times)
    
    # Pre-allocate output arrays (max expected size = n_rows)
    # Note: timestamps are handled as int64 (nanoseconds) in Numba if passed as datetime64[ns]
    out_times = np.empty_like(times)
    out_opens = np.zeros(n_rows, dtype=np.float64)
    out_highs = np.zeros(n_rows, dtype=np.float64)
    out_lows = np.zeros(n_rows, dtype=np.float64)
    out_closes = np.zeros(n_rows, dtype=np.float64)
    out_vols = np.zeros(n_rows, dtype=np.float64)
    
    count = 0
    current_vol = 0.0
    current_open = opens[0]
    current_high = highs[0]
    current_low = lows[0]
    
    for i in range(n_rows):
        current_vol += vols[i]
        val_high = highs[i]
        val_low = lows[i]
        
        if val_high > current_high:
            current_high = val_high
        if val_low < current_low:
            current_low = val_low
            
        target = thresholds[i]
        if np.isnan(target) or target <= 0:
             target = 1000000.0
             
        if current_vol >= target:
            out_times[count] = times[i]
            out_opens[count] = current_open
            out_highs[count] = current_high
            out_lows[count] = current_low
            out_closes[count] = closes[i]
            out_vols[count] = current_vol
            count += 1
            
            # Reset
            current_vol = 0.0
            if i + 1 < n_rows:
                current_open = opens[i+1]
                current_high = highs[i+1]
                current_low = lows[i+1]
                
    return out_times[:count], out_opens[:count], out_highs[:count], out_lows[:count], out_closes[:count], out_vols[:count]

@jit(nopython=True)
def _numba_generate_range_bars(times, opens, highs, lows, closes, vols, thresholds):
    """
    Generate range bars using Numba JIT.
    thresholds: Array of dynamic thresholds (delta_p) for each timestamp.
    """
    n_rows = len(times)
    
    # Pre-allocate output arrays
    out_times = np.empty_like(times)
    out_opens = np.zeros(n_rows, dtype=np.float64)
    out_highs = np.zeros(n_rows, dtype=np.float64)
    out_lows = np.zeros(n_rows, dtype=np.float64)
    out_closes = np.zeros(n_rows, dtype=np.float64)
    out_vols = np.zeros(n_rows, dtype=np.float64)
    out_durations = np.zeros(n_rows, dtype=np.float64) # Duration in seconds + 60.0
    
    count = 0
    
    current_open = opens[0]
    current_high = highs[0]
    current_low = lows[0]
    current_vol = 0.0
    start_ts_idx = 0 # Index of start time
    
    for i in range(n_rows):
        p = closes[i]
        val_high = highs[i]
        val_low = lows[i]
        
        if val_high > current_high: current_high = val_high
        if val_low < current_low: current_low = val_low
        current_vol += vols[i]
        
        delta_p = thresholds[i]
        if np.isnan(delta_p) or delta_p <= 0:
            delta_p = p * 0.01

        # Check range condition (absolute change from open >= threshold)
        if abs(p - current_open) >= delta_p:
            out_times[count] = times[i]
            out_opens[count] = current_open
            out_highs[count] = current_high
            out_lows[count] = current_low
            out_closes[count] = p
            out_vols[count] = current_vol
            
            # Duration: (times[i] - times[start_ts_idx]) in seconds + 60
            # Numba handles datetime64 subtraction getting timedelta64.
            # Convert to seconds is tricky in pure Numba nopython depending on version.
            # Standard: (t2 - t1).astype(np.float64) / 1e9 for ns -> seconds?
            # Or just store raw diff and process later?
            # Let's try simple subtraction div 1e9. 
            # Note: times is int64 array of nanoseconds usually.
            diff_ns = (times[i] - times[start_ts_idx])
            # Convert timedelta64 to nanoseconds (int64) in Numba-compatible way
            diff_ns_int = np.int64(diff_ns)
            out_durations[count] = float(diff_ns_int) / 1e9 + 60.0
            
            count += 1
            
            # Reset
            if i + 1 < n_rows:
                current_open = p
                current_high = p
                current_low = p
                current_vol = 0.0
                start_ts_idx = i + 1
                
    return out_times[:count], out_opens[:count], out_highs[:count], out_lows[:count], out_closes[:count], out_vols[:count], out_durations[:count]

@jit(nopython=True)
def _numba_rolling_entropy(x, window, bins=5):
    """
    Calculate rolling Shannon entropy using Numba.
    x: Input array (returns)
    window: Rolling window size
    bins: Number of histogram bins
    """
    n = len(x)
    output = np.zeros(n, dtype=np.float64)
    
    # We need at least 'window' elements
    for i in range(window, n):
        # Slice window
        chunk = x[i-window:i]
        
        # Calculate histogram manually
        c_min = np.min(chunk)
        c_max = np.max(chunk)
        
        if c_max == c_min:
            output[i] = 0.0
            continue
            
        bin_width = (c_max - c_min) / bins
        hist = np.zeros(bins, dtype=np.float64)
        
        for val in chunk:
            bin_idx = int((val - c_min) / bin_width)
            if bin_idx >= bins:
                bin_idx = bins - 1
            hist[bin_idx] += 1
            
        # Normalize to probability
        # density=True in numpy means sum(hist * bin_width) = 1
        # But Shannon entropy usually uses p_i = count_i / total_count
        # The python code used: hist, _ = np.histogram(..., density=True).
        # hist values are count / (total * width).
        # Entropy = -sum(p * log10(p)). p is the probability density?
        # Usually entropy uses probabilities p_i summing to 1.
        # If density=True, values can be > 1.
        # However, code used: `hist = hist / np.sum(hist) * (1/bin_width)` effectively?
        # Wait, if `density=True`, then `sum(hist * bin_width) = 1`.
        # Code: `return -np.sum(hist * np.log10(hist + 1e-9))`
        # If hist contains PDF values, this is Differential Entropy proxy.
        # I'll replicate the exact logic of `np.histogram(density=True)`.
        
        # density=True: element = count / (n_samples * bin_width)
        norm_factor = window * bin_width
        probs = hist / norm_factor
        
        # Filter > 0
        probs_valid = probs[probs > 0]
        ent = -np.sum(probs_valid * np.log10(probs_valid + 1e-9))
        output[i] = ent
        
    return output

@jit(nopython=True)
def _numba_run_regime_filter(log_probs, weights_raw, entropies, n_regimes, transition_matrix, chaos_idx, log_lik_ema_start, log_lik_std, base_smoothing):
    """
    Run the sequential forward filter for regime detection (HMM-like).
    Handles adaptive smoothing, OOD tracking, and inertia updates.
    """
    n_rows = len(log_probs)
    
    # Output arrays
    final_weights = np.zeros_like(weights_raw)
    z_familiars = np.zeros(n_rows, dtype=np.float64)
    confidences = np.zeros(n_rows, dtype=np.float64)
    
    # State validation
    log_lik_ema = log_lik_ema_start
    # Initialize with first observation if no prior
    # Or uniform? Code uses last_weights=None -> weights_blended logic
    # We will track last_weights
    last_weights = np.zeros(n_regimes, dtype=np.float64)
    has_prior = False # Logic matches: if last_weights is None...
    
    max_entropy = np.log(n_regimes)
    if max_entropy == 0: max_entropy = 1e-9

    chaos_onehot = np.zeros(n_regimes, dtype=np.float64)
    if chaos_idx >= 0 and chaos_idx < n_regimes:
        chaos_onehot[chaos_idx] = 1.0

    for i in range(n_rows):
        # 1. Inputs
        log_prob = log_probs[i]
        w_raw = weights_raw[i]
        raw_ent = entropies[i]
        
        # 2. Adaptive Params
        # dynamic_alpha = min_alpha + ...
        # min_alpha = 0.2
        ratio = raw_ent / (max_entropy + 1e-9)
        dynamic_alpha = 0.2 + (base_smoothing - 0.2) * (1.0 - ratio)
        
        # 3. OOD / Z-Familiar
        z_score = (log_prob - log_lik_ema) / (log_lik_std + 1e-9)
        z_familiars[i] = z_score
        
        # 4. Chaos Boost
        # chaos_boost = 0.4 * sigmoid(-(z + 2.0))
        # Numba doesn't have scipy.special.expit? 
        # sigmoid(x) = 1 / (1 + exp(-x))
        # -(z+2)
        val = -(z_score + 2.0)
        sig_val = 1.0 / (1.0 + np.exp(-val))
        chaos_boost = 0.4 * sig_val
        
        # 5. Blend Raw
        weights_blended = (1.0 - chaos_boost) * w_raw + (chaos_boost * chaos_onehot)
        
        # 6. Forward Filter (Consistency)
        weights_update = weights_blended # Default if no prior
        
        if has_prior:
            # predicted_weights = dot(last, trans)
            # last [1xD], trans [DxD] -> [1xD]
            # Numba dot support
            pred_weights = np.dot(last_weights, transition_matrix)
            
            # Update: pred * evidence (Bayesian update)
            raw_updated = pred_weights * weights_blended
            sum_updated = np.sum(raw_updated)
            norm_updated = raw_updated / (sum_updated + 1e-9)
            
            # Inertia blend
            weights_update = (dynamic_alpha * last_weights) + ((1.0 - dynamic_alpha) * norm_updated)
            
        final_weights[i] = weights_update
        last_weights = weights_update
        has_prior = True
        
        # 7. Update OOD Stats
        if z_score > -3.0:
            log_lik_ema = 0.999 * log_lik_ema + 0.001 * log_prob
            
        # 8. Confidence
        # (1 - ratio) * sigmoid(z)
        sig_z = 1.0 / (1.0 + np.exp(-z_score))
        confidences[i] = (1.0 - ratio) * sig_z
        
    return final_weights, z_familiars, confidences

@jit(nopython=True)
def _numba_rolling_mad(x, window):
    """
    Calculate rolling MAD (Median Absolute Deviation) using Numba.
    MAD = median(|x - median(x)|)
    Results are aligned to the right edge of the window.
    First (window-1) elements will be 0.
    """
    n = len(x)
    output = np.zeros(n, dtype=np.float64)
    
    # Needs at least 'window' elements
    for i in range(window, n + 1):
        # Slice: ending at i (exclusive) -> window size
        chunk = x[i-window:i]
        
        # 1. Median
        med = np.median(chunk)
        
        # 2. Abs Deviations
        devs = np.abs(chunk - med)
        
        # 3. MAD = median of deviations
        mad = np.median(devs)
        
        output[i-1] = mad
        
    return output

@jit(nopython=True)
def _numba_rolling_slope(y, window):
    """
    Calculate rolling linear regression slope using Numba.
    Slope = (N*sum(xy) - sum(x)*sum(y)) / (N*sum(x^2) - (sum(x))^2)
    where x = [0, 1, ..., window-1]
    """
    n = len(y)
    output = np.zeros(n, dtype=np.float64)
    
    # Pre-calculate common terms for x = range(window)
    n_w = float(window)
    sum_x = (n_w * (n_w - 1.0)) / 2.0
    sum_x2 = (n_w * (n_w - 1.0) * (2.0 * n_w - 1.0)) / 6.0
    denom = (n_w * sum_x2 - sum_x**2)
    
    if denom == 0:
        return output
        
    for i in range(window, n + 1):
        chunk = y[i-window:i]
        
        sum_y = 0.0
        sum_xy = 0.0
        for j in range(window):
            sum_y += chunk[j]
            sum_xy += j * chunk[j]
            
        slope = (n_w * sum_xy - sum_x * sum_y) / denom
        output[i-1] = slope
        
    return output

@jit(nopython=True)
def _numba_volatility_clustering(returns, window=20):
    """
    Calculate volatility clustering coefficient (GARCH-like behavior).
    Measures persistence of volatility - high values indicate clustering (chaos).
    """
    n = len(returns)
    output = np.zeros(n, dtype=np.float64)
    
    for i in range(window, n):
        # Calculate squared returns in window
        sq_returns = returns[i-window:i] ** 2
        
        # Calculate autocorrelation of squared returns at lag 1
        if len(sq_returns) > 1:
            x = sq_returns[:-1]
            y = sq_returns[1:]
            
            # Pearson correlation
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            
            x_dev = x - x_mean
            y_dev = y - y_mean
            
            numerator = np.sum(x_dev * y_dev)
            denominator = np.sqrt(np.sum(x_dev**2) * np.sum(y_dev**2))
            
            if denominator > 1e-10:
                output[i] = numerator / denominator
            else:
                output[i] = 0.0
                
    return output

@jit(nopython=True)
def _numba_return_autocorrelation(returns, window=20, lag=1):
    """
    Calculate return autocorrelation at specified lag.
    Negative autocorrelation often indicates chaotic behavior.
    """
    n = len(returns)
    output = np.zeros(n, dtype=np.float64)
    
    for i in range(window + lag, n):
        # Get returns in window
        x = returns[i-window:i-lag]
        y = returns[i-window+lag:i]
        
        # Pearson correlation
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        x_dev = x - x_mean
        y_dev = y - y_mean
        
        numerator = np.sum(x_dev * y_dev)
        denominator = np.sqrt(np.sum(x_dev**2) * np.sum(y_dev**2))
        
        if denominator > 1e-10:
            output[i] = numerator / denominator
        else:
            output[i] = 0.0
            
    return output

@jit(nopython=True)
def _numba_price_jump_frequency(returns, window=20, threshold=2.0):
    """
    Count large price moves (>threshold std deviations) in rolling window.
    High jump frequency indicates market turbulence/chaos.
    """
    n = len(returns)
    output = np.zeros(n, dtype=np.float64)
    
    for i in range(window, n):
        # Get returns in window
        window_returns = returns[i-window:i]
        
        # Calculate rolling mean and std
        mean_ret = np.mean(window_returns)
        std_ret = np.std(window_returns)
        
        if std_ret < 1e-10:
            output[i] = 0.0
            continue
            
        # Count jumps > threshold std deviations
        jump_count = 0
        for ret in window_returns:
            z_score = abs(ret - mean_ret) / std_ret
            if z_score > threshold:
                jump_count += 1
                
        # Normalize by window length
        output[i] = float(jump_count) / window
        
    return output

@jit(nopython=True)
def _numba_detect_gaps_vectorized(timestamps, expected_interval_minutes):
    """
    Vectorized gap detection using Numba.
    Returns array of gap durations in minutes.
    """
    n = len(timestamps)
    gaps = np.zeros(n, dtype=np.int64)
    expected_interval_ns = expected_interval_minutes * 60 * 1_000_000_000  # Convert to nanoseconds
    
    for i in range(1, n):
        time_diff = timestamps[i] - timestamps[i-1]
        gap_minutes = time_diff / (60 * 1_000_000_000)
        
        if gap_minutes > expected_interval_minutes:
            gaps[i] = int(gap_minutes - expected_interval_minutes)
    
    return gaps

@jit(nopython=True)
def _numba_fill_gaps_vectorized(timestamps, opens, highs, lows, closes, volumes, gap_indices):
    """
    Vectorized gap filling using forward fill logic.
    """
    n = len(timestamps)
    filled_timestamps = timestamps.copy()
    filled_opens = opens.copy()
    filled_highs = highs.copy()
    filled_lows = lows.copy()
    filled_closes = closes.copy()
    filled_volumes = volumes.copy()
    
    for gap_idx in gap_indices:
        if gap_idx > 0 and gap_idx < n:
            # Forward fill with last known values
            filled_opens[gap_idx] = filled_closes[gap_idx-1]
            filled_highs[gap_idx] = filled_closes[gap_idx-1]
            filled_lows[gap_idx] = filled_closes[gap_idx-1]
            filled_closes[gap_idx] = filled_closes[gap_idx-1]
            filled_volumes[gap_idx] = 0.0  # Zero volume for filled bars
    
    return filled_timestamps, filled_opens, filled_highs, filled_lows, filled_closes, filled_volumes

@jit(nopython=True)
def _numba_ohlc_resample_vectorized(timestamps, opens, highs, lows, closes, volumes, 
                                   resample_interval_ns, start_timestamp, end_timestamp):
    """
    Vectorized OHLC resampling using Numba.
    """
    # Calculate number of output bars
    total_duration = end_timestamp - start_timestamp
    n_bars = int(total_duration / resample_interval_ns) + 1
    
    # Initialize output arrays
    out_timestamps = np.zeros(n_bars, dtype=np.int64)
    out_opens = np.zeros(n_bars, dtype=np.float64)
    out_highs = np.zeros(n_bars, dtype=np.float64)
    out_lows = np.zeros(n_bars, dtype=np.float64)
    out_closes = np.zeros(n_bars, dtype=np.float64)
    out_volumes = np.zeros(n_bars, dtype=np.float64)
    
    # Initialize with NaN
    out_opens[:] = np.nan
    out_highs[:] = np.nan
    out_lows[:] = np.nan
    out_closes[:] = np.nan
    
    # Process each input bar
    for i in range(len(timestamps)):
        ts = timestamps[i]
        if ts < start_timestamp or ts > end_timestamp:
            continue
            
        # Find output bar index
        bar_idx = int((ts - start_timestamp) / resample_interval_ns)
        if bar_idx < 0 or bar_idx >= n_bars:
            continue
            
        # Set timestamp for this bar
        out_timestamps[bar_idx] = start_timestamp + bar_idx * resample_interval_ns
        
        # Update OHLCV
        if np.isnan(out_opens[bar_idx]):
            out_opens[bar_idx] = opens[i]
        
        if np.isnan(out_highs[bar_idx]) or highs[i] > out_highs[bar_idx]:
            out_highs[bar_idx] = highs[i]
            
        if np.isnan(out_lows[bar_idx]) or lows[i] < out_lows[bar_idx]:
            out_lows[bar_idx] = lows[i]
            
        out_closes[bar_idx] = closes[i]
        out_volumes[bar_idx] += volumes[i]
    
    return out_timestamps, out_opens, out_highs, out_lows, out_closes, out_volumes

@jit(nopython=True)
def _numba_verify_data_quality(opens, highs, lows, closes, volumes):
    """
    Vectorized data quality verification using Numba.
    Returns counts of various quality issues.
    """
    n = len(opens)
    ohlc_issues = 0
    volume_issues = 0
    price_issues = 0
    
    for i in range(n):
        # Check OHLC consistency
        if highs[i] < lows[i]:
            ohlc_issues += 1
        if opens[i] > highs[i] or opens[i] < lows[i]:
            ohlc_issues += 1
        if closes[i] > highs[i] or closes[i] < lows[i]:
            ohlc_issues += 1
            
        # Check volume
        if volumes[i] < 0:
            volume_issues += 1
            
        # Check for price issues (zero, negative, extreme values)
        if opens[i] <= 0 or highs[i] <= 0 or lows[i] <= 0 or closes[i] <= 0:
            price_issues += 1
            
        # Check for extreme price movements ( > 50% in one bar)
        if i > 0:
            price_change = abs(closes[i] - closes[i-1]) / closes[i-1]
            if price_change > 0.5:
                price_issues += 1
    
    return ohlc_issues, volume_issues, price_issues

@jit(nopython=True)
def _numba_streak_persistence(close, window=20):
    """
    Calculate Momentum Persistence (Z-score of price streaks) using Numba.
    Streak = sequence of returns with same sign.
    Persistence = mean(streak_lengths) / std(streak_lengths)
    
    This replaces the slow rolling().apply(streak_z) loop.
    """
    n = len(close)
    output = np.zeros(n, dtype=np.float64)
    
    # Pre-calculate signs of differences
    # diff[i] = close[i] - close[i-1]
    # We first calculate diffs manually to avoid dependency
    diffs = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        diffs[i] = close[i] - close[i-1]
    
    signs = np.sign(diffs)
    
    # Iterate through windows
    for i in range(window, n):
        # Window from i-window+1 to i (inclusive)
        # Note: We look at the diffs occurring WITHIN the window.
        # indices of interest: i-window+1 to i
        
        # Extract window signs
        # slice: [i-window+1 : i+1]
        start_idx = i - window + 1
        end_idx = i + 1
        
        if start_idx < 0: start_idx = 0
        
        # Numba slice
        window_signs = signs[start_idx:end_idx]
        
        # Calculate streaks in this window
        if len(window_signs) < 2:
            continue
            
        current_streak = 0.0
        # We can't use list append in nopython mode effectively if type inference is tricky, 
        # using a fixed size array or just computing stats online is better.
        # But for streaks, we need the variance of lengths.
        # Let's use a pre-allocated array buffer for this window
        streak_buffer = np.zeros(window, dtype=np.float64)
        streak_count = 0
        
        last_sign = 0.0
        
        for j in range(len(window_signs)):
            s = window_signs[j]
            if s == 0: continue
            
            if s == last_sign:
                current_streak += 1.0
            else:
                if current_streak > 0:
                    streak_buffer[streak_count] = current_streak
                    streak_count += 1
                current_streak = 1.0
                last_sign = s
        
        # Append last streak
        if current_streak > 0:
            streak_buffer[streak_count] = current_streak
            streak_count += 1
            
        # Compute Z-score stats
        if streak_count < 2:
            output[i] = 0.0
        else:
            # Calculate mean and std manually
            sum_val = 0.0
            sum_sq = 0.0
            for k in range(streak_count):
                val = streak_buffer[k]
                sum_val += val
                sum_sq += val * val
            
            mean_val = sum_val / streak_count
            var_val = (sum_sq / streak_count) - (mean_val * mean_val)
            
            if var_val < 1e-9:
                output[i] = 0.0
            else:
                std_val = np.sqrt(var_val)
                output[i] = mean_val / std_val
                
    return output

@jit(nopython=True)
def _numba_rolling_hurst(arr, window):
    """
    Compute rolling Hurst exponent using Rescaled Range (R/S) method.
    """
    n = len(arr)
    output = np.full(n, 0.5, dtype=np.float64)

    # Needs at least 'window' elements
    for i in range(window, n + 1):
        # Slice window
        chunk = arr[i-window:i]

        # Check segment length
        m = len(chunk)
        if m < 20:
            continue

        # Mean-adjusted series
        mean_val = np.mean(chunk)
        mean_adj = chunk - mean_val

        # Cumulative deviate series
        cumdev = np.cumsum(mean_adj)

        # Range
        R = np.max(cumdev) - np.min(cumdev)

        # Standard deviation
        S = np.std(chunk)

        if S < 1e-9 or R < 1e-9:
            output[i-1] = 0.5
            continue

        # R/S calculation
        rs = R / S

        # Hurst approximation: H = log(R/S) / log(n)
        H = np.log(rs) / np.log(m)

        if H < 0.0: H = 0.0
        if H > 1.0: H = 1.0

        output[i-1] = H

    return output

@jit(nopython=True)
def _numba_fracdiff(arr, d, threshold=1e-5):
    """
    Apply fractional differentiation using fixed-width window.
    """
    n = len(arr)
    output = np.full(n, np.nan, dtype=np.float64)

    # 1. Calculate weights
    # We don't know the size of weights beforehand, but it won't exceed n
    w = np.empty(n, dtype=np.float64)
    w[0] = 1.0
    width = 1

    for k in range(1, n):
        w_k = -w[k-1] * (d - k + 1) / k
        if np.abs(w_k) < threshold:
            break
        w[k] = w_k
        width += 1

    # Slice valid weights
    weights = w[:width]

    # 2. Apply convolution
    # For each point i >= width - 1
    for i in range(width - 1, n):
        # Segment: arr[i-width+1 : i+1] reversed
        # Dot product with weights
        val = 0.0
        for j in range(width):
            val += weights[j] * arr[i - j]

        output[i] = val

    return output

@jit(nopython=True)
def _numba_anchored_zscore(values, anchor_mask):
    """
    Compute z-scores anchored to the last event in the mask.
    values: float array
    anchor_mask: boolean array (True where anchor event occurs)
    """
    n = len(values)
    output = np.zeros(n, dtype=np.float64)
    last_anchor_idx = -1

    # Pre-calculate accumulators for running variance if needed?
    # Or just recompute on segment? Recomputing on segment O(N^2) worst case if no anchors.
    # But usually anchors appear periodically.
    # Also, we can optimize running mean/std from anchor.
    # Let's use running Welford's algorithm from the anchor point.

    count = 0
    mean_val = 0.0
    m2_val = 0.0
    anchor_val = 0.0

    for i in range(n):
        if anchor_mask[i]:
            last_anchor_idx = i
            count = 1
            anchor_val = values[i]
            mean_val = anchor_val
            m2_val = 0.0
            output[i] = 0.0
            continue

        if last_anchor_idx != -1:
            val = values[i]

            # Update running stats
            count += 1
            delta = val - mean_val
            mean_val += delta / count
            delta2 = val - mean_val
            m2_val += delta * delta2

            # Compute Z-score
            # z = (val - anchor_val) / std_dev
            # std_dev = sqrt(m2 / (count - 1)) if count > 1 else 0

            if count > 4: # Min observations
                variance = m2_val / (count - 1)
                if variance > 1e-9:
                    std_dev = np.sqrt(variance)
                    z = (val - anchor_val) / std_dev

                    # Clip
                    if z > 5.0: z = 5.0
                    if z < -5.0: z = -5.0

                    output[i] = z
                else:
                    output[i] = 0.0
            else:
                output[i] = 0.0
        else:
            output[i] = 0.0

    return output

@jit(nopython=True)
def _numba_time_since_shock(n, shock_mask, decay_lambda=0.02):
    """
    Compute time-since-shock with exponential decay.
    n: length of array
    shock_mask: boolean array
    """
    output = np.zeros(n, dtype=np.float64)
    last_shock_idx = -1

    for i in range(n):
        if shock_mask[i]:
            last_shock_idx = i

        if last_shock_idx != -1:
            delta_t = i - last_shock_idx
            output[i] = np.exp(-decay_lambda * delta_t)
        else:
            output[i] = 0.0

    return output
