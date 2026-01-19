import numpy as np
import math

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


# Small epsilon to prevent division by zero in Numba JIT functions
_EPS = 1e-12


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
                current_open = opens[i + 1]
                current_high = highs[i + 1]
                current_low = lows[i + 1]

    return (
        out_times[:count],
        out_opens[:count],
        out_highs[:count],
        out_lows[:count],
        out_closes[:count],
        out_vols[:count],
    )


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
    out_durations = np.zeros(n_rows, dtype=np.float64)  # Duration in seconds + 60.0

    count = 0

    current_open = opens[0]
    current_high = highs[0]
    current_low = lows[0]
    current_vol = 0.0
    start_ts_idx = 0  # Index of start time

    for i in range(n_rows):
        p = closes[i]
        val_high = highs[i]
        val_low = lows[i]

        if val_high > current_high:
            current_high = val_high
        if val_low < current_low:
            current_low = val_low
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
            diff_ns = times[i] - times[start_ts_idx]
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

    return (
        out_times[:count],
        out_opens[:count],
        out_highs[:count],
        out_lows[:count],
        out_closes[:count],
        out_vols[:count],
        out_durations[:count],
    )


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
        chunk = x[i - window : i]

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
def _numba_run_regime_filter(
    log_probs,
    weights_raw,
    entropies,
    n_regimes,
    transition_matrix,
    chaos_idx,
    log_lik_ema_start,
    log_lik_std,
    base_smoothing,
):
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
    has_prior = False  # Logic matches: if last_weights is None...

    max_entropy = np.log(n_regimes)
    if max_entropy == 0:
        max_entropy = 1e-9

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
        weights_update = weights_blended  # Default if no prior

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
            weights_update = (dynamic_alpha * last_weights) + (
                (1.0 - dynamic_alpha) * norm_updated
            )

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
        chunk = x[i - window : i]

        # 1. Median
        med = np.median(chunk)

        # 2. Abs Deviations
        devs = np.abs(chunk - med)

        # 3. MAD = median of deviations
        mad = np.median(devs)

        output[i - 1] = mad

    return output


@jit(nopython=True)
def _numba_rolling_median(x, window):
    """
    Calculate rolling Median using Numba.
    Results are aligned to the right edge of the window.
    First (window-1) elements will be 0.
    """
    n = len(x)
    output = np.zeros(n, dtype=np.float64)

    # Needs at least 'window' elements
    for i in range(window, n + 1):
        # Slice: ending at i (exclusive) -> window size
        chunk = x[i - window : i]

        # Median
        med = np.median(chunk)

        output[i - 1] = med

    return output


@jit(nopython=True)
def _numba_rolling_slope(y, window):
    """
    Calculate rolling linear regression slope using Numba.
    Slope = (N*sum(xy) - sum(x)*sum(y)) / (N*sum(x^2) - (sum(x))^2)
    where x = [0, 1, ..., window-1]

    Optimized to O(N) using incremental updates:
    S_xy_new = S_xy_old - S_y_old + y_leaving + (W-1)*y_entering
    S_y_new = S_y_old - y_leaving + y_entering
    """
    n = len(y)
    output = np.zeros(n, dtype=np.float64)

    if n < window:
        return output

    # Pre-calculate common terms for x = range(window)
    n_w = float(window)
    sum_x = (n_w * (n_w - 1.0)) / 2.0
    sum_x2 = (n_w * (n_w - 1.0) * (2.0 * n_w - 1.0)) / 6.0
    denom = n_w * sum_x2 - sum_x**2

    if denom == 0:
        return output

    # Initialize first window
    sum_y = 0.0
    sum_xy = 0.0
    for j in range(window):
        val = y[j]
        sum_y += val
        sum_xy += j * val

    output[window - 1] = (n_w * sum_xy - sum_x * sum_y) / denom

    # Loop for remaining windows
    for i in range(window, n):
        y_leaving = y[i - window]
        y_entering = y[i]

        # Update sums incrementally
        sum_xy = sum_xy - sum_y + y_leaving + (n_w - 1) * y_entering
        sum_y = sum_y - y_leaving + y_entering

        output[i] = (n_w * sum_xy - sum_x * sum_y) / denom

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
        sq_returns = returns[i - window : i] ** 2

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
        x = returns[i - window : i - lag]
        y = returns[i - window + lag : i]

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
        window_returns = returns[i - window : i]

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
    expected_interval_ns = (
        expected_interval_minutes * 60 * 1_000_000_000
    )  # Convert to nanoseconds

    for i in range(1, n):
        time_diff = timestamps[i] - timestamps[i - 1]
        gap_minutes = time_diff / (60 * 1_000_000_000)

        if gap_minutes > expected_interval_minutes:
            gaps[i] = int(gap_minutes - expected_interval_minutes)

    return gaps


@jit(nopython=True)
def _numba_fill_gaps_vectorized(
    timestamps, opens, highs, lows, closes, volumes, gap_indices
):
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
            filled_opens[gap_idx] = filled_closes[gap_idx - 1]
            filled_highs[gap_idx] = filled_closes[gap_idx - 1]
            filled_lows[gap_idx] = filled_closes[gap_idx - 1]
            filled_closes[gap_idx] = filled_closes[gap_idx - 1]
            filled_volumes[gap_idx] = 0.0  # Zero volume for filled bars

    return (
        filled_timestamps,
        filled_opens,
        filled_highs,
        filled_lows,
        filled_closes,
        filled_volumes,
    )


@jit(nopython=True)
def _numba_ohlc_resample_vectorized(
    timestamps,
    opens,
    highs,
    lows,
    closes,
    volumes,
    resample_interval_ns,
    start_timestamp,
    end_timestamp,
):
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
            price_change = abs(closes[i] - closes[i - 1]) / closes[i - 1]
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
        diffs[i] = close[i] - close[i - 1]

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

        if start_idx < 0:
            start_idx = 0

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
            if s == 0:
                continue

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
def _numba_rolling_sum(x: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling sum aligned to the right edge of the window.
    First (window-1) elements are 0.0.
    """
    n = len(x)
    out = np.zeros(n, dtype=np.float64)

    if window <= 0:
        return out
    if window == 1:
        # Right-aligned sum with window=1 is the value itself
        for i in range(n):
            out[i] = x[i]
        return out

    s = 0.0
    for i in range(n):
        s += x[i]
        if i >= window:
            s -= x[i - window]
        if i >= window - 1:
            out[i] = s
    return out


@jit(nopython=True)
def _numba_rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling mean aligned to the right edge of the window.
    First (window-1) elements are 0.0.
    """
    n = len(x)
    out = np.zeros(n, dtype=np.float64)

    if window <= 0:
        return out
    if window == 1:
        for i in range(n):
            out[i] = x[i]
        return out

    s = 0.0
    inv_w = 1.0 / window
    for i in range(n):
        s += x[i]
        if i >= window:
            s -= x[i - window]
        if i >= window - 1:
            out[i] = s * inv_w
    return out


@jit(nopython=True)
def _numba_rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling standard deviation aligned to the right edge of the window.
    Uses E[x^2] - (E[x])^2 (fast; fine for typical small windows).
    """
    n = len(x)
    out = np.zeros(n, dtype=np.float64)

    if window <= 0:
        return out
    if window == 1:
        # Std over a single point is 0
        return out

    s = 0.0
    ss = 0.0
    inv_w = 1.0 / window

    for i in range(n):
        v = x[i]
        s += v
        ss += v * v

        if i >= window:
            r = x[i - window]
            s -= r
            ss -= r * r

        if i >= window - 1:
            mean = s * inv_w
            var = (ss * inv_w) - (mean * mean)
            if var <= _EPS:
                out[i] = 0.0
            else:
                out[i] = np.sqrt(var)

    return out


@jit(nopython=True)
def _numba_rolling_correlation(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling correlation Corr(x,y) aligned to the right edge of the window.

    Corr = Cov(x,y) / (Std(x)*Std(y))
    Cov(x,y) = E[xy] - E[x]E[y]
    """
    n = len(x)
    out = np.zeros(n, dtype=np.float64)

    if window <= 0:
        return out
    if len(y) != n:
        # In nopython mode, avoid raising; caller should ensure matching lengths.
        return out
    if window == 1:
        # Correlation over a single point is undefined; return 0s (consistent with other kernels)
        return out

    sx = 0.0
    sy = 0.0
    sxx = 0.0
    syy = 0.0
    sxy = 0.0
    inv_w = 1.0 / window

    for i in range(n):
        vx = x[i]
        vy = y[i]

        sx += vx
        sy += vy
        sxx += vx * vx
        syy += vy * vy
        sxy += vx * vy

        if i >= window:
            rx = x[i - window]
            ry = y[i - window]
            sx -= rx
            sy -= ry
            sxx -= rx * rx
            syy -= ry * ry
            sxy -= rx * ry

        if i >= window - 1:
            mx = sx * inv_w
            my = sy * inv_w

            varx = (sxx * inv_w) - (mx * mx)
            vary = (syy * inv_w) - (my * my)
            cov = (sxy * inv_w) - (mx * my)

            if varx <= _EPS or vary <= _EPS:
                out[i] = 0.0
            else:
                out[i] = cov / np.sqrt(varx * vary)

    return out


@jit(nopython=True)
def _numba_calculate_continuous_weight(vals, gamma, beta, quantile_threshold):
    """
    Calculate continuous sample weights using Numba.
    Optimized replacement for calculate_continuous_weight.
    """
    n = len(vals)
    weights = np.zeros(n, dtype=np.float64)

    # Work on absolute values
    abs_vals = np.abs(vals)

    # Sort indices
    # argsort handles NaNs by putting them at the end
    sorted_idxs = np.argsort(abs_vals)

    # Count valid entries (non-NaN)
    valid_count = 0
    for i in range(n):
        if not np.isnan(abs_vals[i]):
            valid_count += 1

    # Process valid entries
    i = 0
    while i < valid_count:
        start = i
        val = abs_vals[sorted_idxs[i]]

        # Find end of tie group
        end = i + 1
        while end < valid_count:
            if abs_vals[sorted_idxs[end]] != val:
                break
            end += 1

        # Average rank for the group
        avg_rank = (start + 1 + end) / 2.0
        pct_rank = avg_rank / valid_count

        # Apply to all elements in the tie group
        for k in range(start, end):
            orig_idx = sorted_idxs[k]
            # Re-fetch value to compute sigmoid (it is 'val')

            # Sigmoid: 1 / (1 + exp(-beta * (val - 2.0)))
            # Use math.exp for scalar speed
            z = val
            sig = 1.0 / (1.0 + math.exp(-beta * (z - 2.0)))

            if pct_rank > quantile_threshold:
                weights[orig_idx] = (pct_rank ** gamma) * sig
            else:
                weights[orig_idx] = 0.0

        i = end

    # NaNs (indices >= valid_count) remain 0.0 as initialized
    # OR should they be NaN?
    # Original behavior: NaNs resulted in NaN weight?
    # (0.5 ** gamma) * NaN -> NaN.
    # So if we want to match exactly, we should set them to NaN.
    for k in range(valid_count, n):
        orig_idx = sorted_idxs[k]
        weights[orig_idx] = np.nan

    return weights

@jit(nopython=True)
def _numba_ewma(x: np.ndarray, alpha: float, adjust: bool = False) -> np.ndarray:
    """
    Calculate Exponential Weighted Moving Average (EWMA) using Numba.
    Matches pandas ewm(alpha=alpha, adjust=adjust).mean().

    Args:
        x: Input array (1D)
        alpha: Smoothing factor (0 < alpha <= 1)
        adjust: Whether to adjust for starting bias (Pandas-style)

    Returns:
        EWMA array
    """
    n = len(x)
    out = np.empty(n, dtype=np.float64)

    if n == 0:
        return out

    # Handle first value (and NaNs at start)
    # Find first valid index
    first_valid_idx = -1
    for i in range(n):
        if not np.isnan(x[i]):
            first_valid_idx = i
            break

    if first_valid_idx == -1:
        # All NaNs
        out[:] = np.nan
        return out

    # Fill NaNs before first valid with NaN
    out[:first_valid_idx] = np.nan

    # Initialize
    if adjust:
        # With adjust=True: y[t] = sum(w_i * x[t-i]) / sum(w_i)
        # Recursive:
        # weighted_sum[t] = x[t] + (1-alpha)*weighted_sum[t-1]
        # sum_weights[t] = 1 + (1-alpha)*sum_weights[t-1]
        # out[t] = weighted_sum[t] / sum_weights[t]

        weighted_sum = x[first_valid_idx]
        sum_weights = 1.0
        out[first_valid_idx] = weighted_sum / sum_weights

        for i in range(first_valid_idx + 1, n):
            val = x[i]
            if np.isnan(val):
                out[i] = np.nan
                # Simplification: if NaN encountered, result becomes NaN for that step.
                weighted_sum = np.nan
            else:
                weighted_sum = val + (1.0 - alpha) * weighted_sum
                sum_weights = 1.0 + (1.0 - alpha) * sum_weights
                out[i] = weighted_sum / sum_weights
    else:
        # With adjust=False: y[t] = (1-alpha)*y[t-1] + alpha*x[t]
        # Initialization: y[0] = x[0]

        last_val = x[first_valid_idx]
        out[first_valid_idx] = last_val

        for i in range(first_valid_idx + 1, n):
            val = x[i]
            if np.isnan(val):
                out[i] = np.nan
                last_val = np.nan # Propagate NaN
            else:
                if np.isnan(last_val):
                    last_val = val
                else:
                    last_val = (1.0 - alpha) * last_val + alpha * val
                out[i] = last_val

    return out

@jit(nopython=True)
def _numba_ewm_std(x: np.ndarray, alpha: float, adjust: bool = False) -> np.ndarray:
    """
    Calculate Exponential Weighted Moving Standard Deviation using Numba.

    Args:
        x: Input array (1D)
        alpha: Smoothing factor
        adjust: Bias adjustment

    Returns:
        EWM Std array
    """
    n = len(x)
    out = np.empty(n, dtype=np.float64)

    if n == 0:
        return out

    # Find first valid
    first_valid_idx = -1
    for i in range(n):
        if not np.isnan(x[i]):
            first_valid_idx = i
            break

    if first_valid_idx == -1:
        out[:] = np.nan
        return out

    out[:first_valid_idx] = np.nan
    out[first_valid_idx] = np.nan # Variance is undefined for the first point

    # Re-implementation using Two-Pass approach (E[x] and E[x^2]) is safer for consistency with Pandas
    # Calculating means
    ewm_x = _numba_ewma(x, alpha, adjust)
    x2 = x * x
    ewm_x2 = _numba_ewma(x2, alpha, adjust)

    for i in range(n):
        if np.isnan(ewm_x[i]) or np.isnan(ewm_x2[i]):
            out[i] = np.nan
        else:
            # Var = E[x^2] - (E[x])^2
            # Numerical noise might cause negative small values
            v = ewm_x2[i] - ewm_x[i]**2
            if v < 0:
                v = 0.0
            out[i] = np.sqrt(v)

    return out

@jit(nopython=True)
def _numba_rolling_skew(x, window):
    """
    Calculate rolling skewness using Numba with online update algorithm (Sum of Powers).
    Optimized for speed O(N) instead of O(N*W).
    Assumes clean input (no NaNs), or NaNs will propagate.
    """
    n = len(x)
    out = np.empty(n, dtype=np.float64)
    out[:] = np.nan

    if window < 3:
        return out

    s1 = 0.0
    s2 = 0.0
    s3 = 0.0

    # Initialize first window
    for i in range(window):
        val = x[i]
        s1 += val
        s2 += val*val
        s3 += val*val*val

    w = float(window)
    # Compute first output
    mean = s1 / w
    # Variance = E[x^2] - (E[x])^2
    var = (s2 / w) - (mean * mean)

    # m3 = E[x^3] - 3*E[x]*E[x^2] + 2*(E[x])^3
    m3 = (s3 / w) - 3.0 * mean * (s2 / w) + 2.0 * (mean * mean * mean)

    if var > 1e-12:
        stdev = np.sqrt(var)
        pop_skew = m3 / (stdev * stdev * stdev)
        # Bias correction
        adj = np.sqrt(w * (w - 1.0)) / (w - 2.0)
        out[window-1] = adj * pop_skew
    else:
        out[window-1] = 0.0

    # Rolling update
    for i in range(window, n):
        leaving = x[i-window]
        entering = x[i]

        s1 = s1 - leaving + entering
        s2 = s2 - leaving*leaving + entering*entering
        s3 = s3 - leaving*leaving*leaving + entering*entering*entering

        mean = s1 / w
        var = (s2 / w) - (mean * mean)

        if var > 1e-12:
            m3 = (s3 / w) - 3.0 * mean * (s2 / w) + 2.0 * (mean * mean * mean)
            stdev = np.sqrt(var)
            pop_skew = m3 / (stdev * stdev * stdev)

            adj = np.sqrt(w * (w - 1.0)) / (w - 2.0)
            out[i] = adj * pop_skew
        else:
            out[i] = 0.0

    return out

@jit(nopython=True)
def _numba_rolling_kurt(x, window):
    """
    Calculate rolling kurtosis using Numba with online update algorithm (Sum of Powers).
    Optimized for speed O(N) instead of O(N*W).
    Assumes clean input (no NaNs), or NaNs will propagate.
    """
    n = len(x)
    out = np.empty(n, dtype=np.float64)
    out[:] = np.nan

    if window < 4:
        return out

    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    s4 = 0.0

    for i in range(window):
        val = x[i]
        s1 += val
        s2 += val*val
        s3 += val*val*val
        s4 += val*val*val*val

    w = float(window)

    # Constants for bias correction
    term1 = (w * (w + 1.0)) / ((w - 1.0) * (w - 2.0) * (w - 3.0))
    term2 = (3.0 * (w - 1.0)**2) / ((w - 2.0) * (w - 3.0))

    # Helper logic inlined (Numba inlining is automatic usually)
    mean = s1 / w
    m2 = (s2 / w) - (mean * mean)

    if m2 > 1e-12:
        e4 = s4 / w
        e3 = s3 / w
        e2 = s2 / w
        e1 = mean
        m4 = e4 - 4.0*e1*e3 + 6.0*e1*e1*e2 - 3.0*e1*e1*e1*e1
        kurt_pop = m4 / (m2 * m2)
        val = ((w - 1.0)**2 / w) * kurt_pop
        out[window-1] = term1 * val - term2
    else:
        out[window-1] = 0.0

    for i in range(window, n):
        leaving = x[i-window]
        entering = x[i]

        s1 = s1 - leaving + entering
        s2 = s2 - leaving*leaving + entering*entering
        s3 = s3 - leaving*leaving*leaving + entering*entering*entering
        s4 = s4 - leaving*leaving*leaving*leaving + entering*entering*entering*entering

        mean = s1 / w
        m2 = (s2 / w) - (mean * mean)

        if m2 > 1e-12:
            e4 = s4 / w
            e3 = s3 / w
            e2 = s2 / w
            e1 = mean
            m4 = e4 - 4.0*e1*e3 + 6.0*e1*e1*e2 - 3.0*e1*e1*e1*e1
            kurt_pop = m4 / (m2 * m2)
            val = ((w - 1.0)**2 / w) * kurt_pop
            out[i] = term1 * val - term2
        else:
            out[i] = 0.0

    return out
