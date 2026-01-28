import numpy as np
from numba import jit

@jit(nopython=True)
def _numba_kalman_filter_1d(values, Q, R, initial_x, initial_P):
    n = len(values)
    x_hat = np.zeros(n)
    P_hat = np.zeros(n)
    x, P = initial_x, initial_P

    for i in range(n):
        x_pred = x
        P_pred = P + Q
        z = values[i]
        K = P_pred / (P_pred + R)
        x = x_pred + K * (z - x_pred)
        P = (1 - K) * P_pred
        x_hat[i] = x
        P_hat[i] = P

    return x_hat, P_hat

@jit(nopython=True)
def _numba_apply_fracdiff(series_values, w):
    n = len(series_values)
    width = len(w)
    result = np.full(n, np.nan)

    # Pre-reverse weights for convolution if needed, or iterate appropriately
    # The original code: np.dot(w, series.iloc[i - width + 1:i + 1].values[::-1])
    # series slice [i-width+1 : i+1] length is width.
    # Reversed means latest value first?
    # series[i] is latest. series[i-width+1] is oldest.
    # values[::-1] means [series[i], series[i-1], ... series[i-width+1]]
    # w is [1, -d, ...]
    # So result[i] = w[0]*series[i] + w[1]*series[i-1] + ...

    for i in range(width - 1, n):
        val = 0.0
        for j in range(width):
            val += w[j] * series_values[i - j]
        result[i] = val

    return result

@jit(nopython=True)
def _numba_rolling_hurst(series, window):
    n = len(series)
    output = np.full(n, 0.5)

    if n < window:
        return output

    # Initialize sums for the first window
    current_sum = 0.0
    current_sq = 0.0
    for j in range(window):
        val = series[j]
        current_sum += val
        current_sq += val * val

    for i in range(window, n + 1):
        # Update sums incrementally if shifted
        if i > window:
            entering = series[i - 1]
            leaving = series[i - 1 - window]
            current_sum += entering - leaving
            current_sq += entering * entering - leaving * leaving

        mean_val = current_sum / window

        # Calculate Standard Deviation (S)
        # Var = E[x^2] - (E[x])^2
        variance_term = (current_sq / window) - (mean_val * mean_val)

        if variance_term < 1e-12:
            S = 0.0
        else:
            S = np.sqrt(variance_term)

        # Range (R) calculation - requires loop over the window
        # We optimize by avoiding the extra loop for mean/sum calculation
        min_cumdev = 0.0
        max_cumdev = 0.0
        current_cumdev = 0.0

        start_idx = i - window
        end_idx = i

        for j in range(start_idx, end_idx):
            val = series[j]
            # Calculate deviation on the fly
            current_cumdev += (val - mean_val)

            if current_cumdev < min_cumdev:
                min_cumdev = current_cumdev
            if current_cumdev > max_cumdev:
                max_cumdev = current_cumdev

        R = max_cumdev - min_cumdev

        if S < 1e-9 or R < 1e-9:
            continue

        rs = R / S
        H = np.log(rs) / np.log(window)

        # Clip
        if H < 0.0:
            H = 0.0
        if H > 1.0:
            H = 1.0

        output[i - 1] = H

    return output

@jit(nopython=True)
def _numba_anchored_zscore(values, current_indices, anchor_indices):
    """
    values: array of feature values
    current_indices: array of indices [0, 1, ..., n-1] usually, or specific query indices
    anchor_indices: sorted array of indices where anchors occur
    """
    n = len(current_indices)
    result = np.zeros(n)
    num_anchors = len(anchor_indices)

    if num_anchors == 0:
        return result

    for i in range(n):
        curr_idx = current_indices[i]

        # Find last anchor < curr_idx
        # Linear search backwards or binary search
        # Since we iterate i sequentially, we can keep a pointer
        # But here we assume random access logic for safety

        # Binary search for curr_idx in anchor_indices
        # np.searchsorted(a, v, side='left') returns index where v should be inserted
        # idx = searchsorted -> anchor_indices[idx-1] < curr_idx <= anchor_indices[idx]

        # Numba supports searchsorted
        idx = np.searchsorted(anchor_indices, curr_idx)

        if idx == 0:
            continue

        last_anchor_idx = anchor_indices[idx - 1]

        if curr_idx <= last_anchor_idx:
            continue

        # Segment: values[last_anchor_idx : curr_idx + 1]
        length = curr_idx - last_anchor_idx + 1
        if length < 5:
            continue

        # Compute stats
        # We need std of segment and first element

        start_val = values[last_anchor_idx]
        end_val = values[curr_idx]

        # Compute Std
        # Single pass Welford's algorithm or two pass

        mean_val = 0.0
        m2 = 0.0
        count = 0

        for k in range(last_anchor_idx, curr_idx + 1):
            val = values[k]
            count += 1
            delta = val - mean_val
            mean_val += delta / count
            delta2 = val - mean_val
            m2 += delta * delta2

        if count < 2:
            std_val = 0.0
        else:
            std_val = np.sqrt(m2 / (count - 1)) # Sample std

        if std_val > 1e-9:
            result[i] = (end_val - start_val) / std_val

    return result

@jit(nopython=True)
def _numba_time_since_shock(n_samples, current_indices, shock_indices, decay_lambda):
    result = np.zeros(n_samples)
    num_shocks = len(shock_indices)

    if num_shocks == 0:
        return result

    for i in range(n_samples):
        curr_idx = current_indices[i]

        idx = np.searchsorted(shock_indices, curr_idx)
        if idx == 0:
            continue

        last_shock_idx = shock_indices[idx - 1]

        delta_t = curr_idx - last_shock_idx
        result[i] = np.exp(-decay_lambda * delta_t)

    return result

@jit(nopython=True)
def _numba_build_indicator_matrix(event_indices, n_bars, horizon, binary=True):
    arr = np.zeros(n_bars, dtype=np.int64)

    for loc in event_indices:
        if loc == -1: continue
        end_loc = loc + horizon
        if end_loc > n_bars:
            end_loc = n_bars

        if binary:
            arr[loc:end_loc] = 1
        else:
            # Numba loop for range assignment
            for k in range(loc, end_loc):
                arr[k] += 1

    return arr

@jit(nopython=True)
def _numba_get_uniqueness(event_indices, indicator_arr, horizon):
    n_events = len(event_indices)
    uniqueness = np.zeros(n_events)
    n_bars = len(indicator_arr)

    for i in range(n_events):
        loc = event_indices[i]
        if loc == -1: continue

        end_loc = loc + horizon
        if end_loc > n_bars: end_loc = n_bars

        sum_inv = 0.0
        count = 0

        for k in range(loc, end_loc):
            c = indicator_arr[k]
            if c > 0:
                sum_inv += 1.0 / c
            count += 1

        if count > 0:
            uniqueness[i] = sum_inv / count

    return uniqueness

@jit(nopython=True)
def _numba_create_ridge_targets(close_arr, vol_arr, event_indices, horizon, weights):
    # weights: array of length 4 (for 4 horizons)
    # horizons: [h//4, h//2, h, h*2]

    n_events = len(event_indices)
    n_bars = len(close_arr)
    result = np.full(n_events, np.nan)

    h1 = max(1, horizon // 4)
    h2 = max(1, horizon // 2)
    h3 = horizon
    h4 = horizon * 2

    horizons = np.array([h1, h2, h3, h4])

    for i in range(n_events):
        evt_idx = event_indices[i]
        if evt_idx < 0 or evt_idx >= n_bars:
            continue

        base_price = close_arr[evt_idx]
        evt_vol = vol_arr[evt_idx]
        if evt_vol <= 0: evt_vol = 1.0

        weighted_ret_sum = 0.0
        weight_sum = 0.0

        for j in range(4):
            h = horizons[j]
            end_idx = evt_idx + h
            if end_idx >= n_bars: end_idx = n_bars - 1

            price_end = close_arr[end_idx]
            ret = (price_end / base_price) - 1.0

            weighted_ret_sum += ret * weights[j]
            weight_sum += weights[j]

        avg_ret = weighted_ret_sum / weight_sum

        norm_ret = avg_ret / (evt_vol * np.sqrt(horizon) + 1e-9)
        result[i] = norm_ret

    return result

@jit(nopython=True)
def _numba_create_tree_targets(close_arr, vol_arr, event_indices, horizon):
    n_events = len(event_indices)
    n_bars = len(close_arr)
    result = np.full(n_events, np.nan)

    for i in range(n_events):
        evt_idx = event_indices[i]
        if evt_idx < 0 or evt_idx >= n_bars:
            continue

        end_idx = evt_idx + horizon
        if end_idx >= n_bars: end_idx = n_bars - 1

        base_price = close_arr[evt_idx]
        end_price = close_arr[end_idx]

        ret = (end_price / base_price) - 1.0

        # Path volatility
        # segment: close_arr[evt_idx : end_idx + 1]

        # Calculate std of returns in this segment
        # returns = diff / prev

        if end_idx > evt_idx + 1:
            # Need return std
            mean_r = 0.0
            m2_r = 0.0
            count_r = 0

            prev_p = close_arr[evt_idx]
            path_abs_diff_sum = 0.0

            for k in range(evt_idx + 1, end_idx + 1):
                curr_p = close_arr[k]
                if prev_p != 0:
                    r = (curr_p - prev_p) / prev_p
                else:
                    r = 0.0

                count_r += 1
                delta = r - mean_r
                mean_r += delta / count_r
                delta2 = r - mean_r
                m2_r += delta * delta2

                path_abs_diff_sum += abs(curr_p - prev_p)
                prev_p = curr_p

            if count_r > 1:
                path_vol = np.sqrt(m2_r / (count_r - 1))
            else:
                path_vol = vol_arr[evt_idx]
        else:
            path_vol = vol_arr[evt_idx]
            path_abs_diff_sum = abs(end_price - base_price)

        if path_vol <= 0: path_vol = 1e-9

        # Sharpe target (Multi-Horizon Blended)
        # Blend returns over [h/4, h/2, h] to improve SNR
        h1 = max(1, horizon // 4)
        h2 = max(1, horizon // 2)
        h3 = horizon

        idx1 = min(n_bars - 1, evt_idx + h1)
        idx2 = min(n_bars - 1, evt_idx + h2)
        idx3 = min(n_bars - 1, evt_idx + h3)

        # Calculate returns for each horizon
        r1 = (close_arr[idx1] / base_price) - 1.0
        r2 = (close_arr[idx2] / base_price) - 1.0
        r3 = (close_arr[idx3] / base_price) - 1.0

        # Weighted blend (50% full horizon, 30% half, 20% quarter)
        blended_ret = 0.5 * r3 + 0.3 * r2 + 0.2 * r1

        sharpe = blended_ret / (path_vol * np.sqrt(horizon) + 1e-9)

        # Efficiency (based on full horizon)
        net_move = abs(end_price - base_price)
        if path_abs_diff_sum > 0:
            efficiency = net_move / path_abs_diff_sum
        else:
            efficiency = 1.0

        adjusted = sharpe * (0.5 + 0.5 * efficiency)
        result[i] = adjusted

    return result
