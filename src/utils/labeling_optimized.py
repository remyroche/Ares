import numpy as np
from numba import jit

@jit(nopython=True)
def triple_barrier_labels_numba(prices, events_idx, vols, pt_mult, sl_mult, horizon):
    """
    Numba implementation of Triple Barrier Method.
    Avoids creating N x Horizon window matrix.
    Returns array of labels: 1 (profit), -1 (loss), 0 (timeout/none).
    """
    n_events = len(events_idx)
    labels = np.zeros(n_events, dtype=np.int8)
    n_prices = len(prices)

    for i in range(n_events):
        idx = events_idx[i]

        # Check if enough data for full horizon
        if idx + horizon >= n_prices:
            continue

        entry_price = prices[idx]
        vol = vols[idx]

        # Guard against zero volatility
        if vol <= 0:
            vol = 1e-6

        up_barrier = pt_mult * vol
        down_barrier = sl_mult * vol

        hit_up = False
        hit_down = False

        # Scan forward
        for j in range(1, horizon + 1):
            curr_price = prices[idx + j]
            ret = (curr_price / entry_price) - 1.0

            # Check upper barrier
            if ret >= up_barrier:
                hit_up = True
                break

            # Check lower barrier (note: down_barrier is positive magnitude)
            if ret <= -down_barrier:
                hit_down = True
                break

        # Assign label based on first hit
        if hit_up:
            labels[i] = 1
        elif hit_down:
            labels[i] = -1
        # else 0

    return labels

@jit(nopython=True)
def persistence_label_numba(series_values, events_idx, horizon, threshold):
    """
    Numba implementation of Persistence Label.
    Calculates if average value over horizon > threshold.
    """
    n_events = len(events_idx)
    labels = np.zeros(n_events, dtype=np.int8)
    n_series = len(series_values)

    for i in range(n_events):
        idx = events_idx[i]

        if idx + horizon >= n_series:
            continue

        sum_val = 0.0
        count = 0

        for j in range(1, horizon + 1):
            sum_val += series_values[idx + j]
            count += 1

        if count > 0:
            avg_val = sum_val / count
            if avg_val > threshold:
                labels[i] = 1

    return labels

@jit(nopython=True)
def window_stats_close_numba(prices, events_idx, horizon):
    """
    Calculate MFE/MAE/Returns using Close prices only.
    Returns: mfe, mae, final_ret
    """
    n_events = len(events_idx)
    n_prices = len(prices)

    mfe = np.zeros(n_events, dtype=np.float64)
    mae = np.zeros(n_events, dtype=np.float64)
    final_ret = np.zeros(n_events, dtype=np.float64)

    for i in range(n_events):
        idx = events_idx[i]
        if idx + horizon >= n_prices:
            # Handle edge case or leave as 0
            # If partial horizon, calculate on what's available
            end_idx = n_prices - 1
        else:
            end_idx = idx + horizon

        entry_price = prices[idx]

        local_max = -999.0
        local_min = 999.0

        for k in range(idx + 1, end_idx + 1):
            p = prices[k]
            ret = (p / entry_price) - 1.0

            if ret > local_max:
                local_max = ret
            if ret < local_min:
                local_min = ret

        final_ret[i] = (prices[end_idx] / entry_price) - 1.0

        mfe[i] = max(0.0, local_max) if local_max != -999.0 else 0.0
        mae[i] = -local_min if local_min != 999.0 else 0.0

    return mfe, mae, final_ret

@jit(nopython=True)
def window_stats_high_low_numba(prices_high, prices_low, prices_close, events_idx, horizon):
    """
    Calculate MFE/MAE using High/Low prices.
    MFE uses Highs, MAE uses Lows.
    """
    n_events = len(events_idx)
    n_prices = len(prices_close)

    mfe = np.zeros(n_events, dtype=np.float64)
    mae = np.zeros(n_events, dtype=np.float64)
    final_ret = np.zeros(n_events, dtype=np.float64)

    for i in range(n_events):
        idx = events_idx[i]
        if idx + horizon >= n_prices:
            end_idx = n_prices - 1
        else:
            end_idx = idx + horizon

        entry_price = prices_close[idx]

        local_max = -999.0
        local_min = 999.0

        for k in range(idx + 1, end_idx + 1):
            # High return
            ret_h = (prices_high[k] / entry_price) - 1.0
            if ret_h > local_max:
                local_max = ret_h

            # Low return
            ret_l = (prices_low[k] / entry_price) - 1.0
            if ret_l < local_min:
                local_min = ret_l

        final_ret[i] = (prices_close[end_idx] / entry_price) - 1.0

        mfe[i] = max(0.0, local_max) if local_max != -999.0 else 0.0
        mae[i] = -local_min if local_min != 999.0 else 0.0

    return mfe, mae, final_ret

@jit(nopython=True)
def first_hit_numba(prices, events_idx, pt_thresholds, sl_thresholds, horizon):
    """
    Calculates the 'first hit' indices for PT and SL.
    Returns:
       first_pt_idx: index relative to event (1 to H), or H+1 if not hit
       first_sl_idx: index relative to event (1 to H), or H+1 if not hit
       any_pt: boolean
       any_sl: boolean
    """
    n_events = len(events_idx)
    n_prices = len(prices)

    first_pt = np.full(n_events, horizon + 1, dtype=np.int32)
    first_sl = np.full(n_events, horizon + 1, dtype=np.int32)
    any_pt_arr = np.zeros(n_events, dtype=np.int8)
    any_sl_arr = np.zeros(n_events, dtype=np.int8)

    for i in range(n_events):
        idx = events_idx[i]
        pt_thresh = pt_thresholds[i]
        sl_thresh = sl_thresholds[i]

        if idx + horizon >= n_prices:
            end_idx = n_prices - 1
        else:
            end_idx = idx + horizon

        entry_price = prices[idx]

        for k in range(1, horizon + 1):
            curr_idx = idx + k
            if curr_idx >= n_prices:
                break

            p = prices[curr_idx]
            ret = (p / entry_price) - 1.0

            # Check PT
            if first_pt[i] == horizon + 1:
                if ret > pt_thresh:
                    first_pt[i] = k
                    any_pt_arr[i] = 1

            # Check SL
            if first_sl[i] == horizon + 1:
                if ret < sl_thresh:
                    first_sl[i] = k
                    any_sl_arr[i] = 1

            if first_pt[i] <= k and first_sl[i] <= k:
                break

    return first_pt, first_sl, any_pt_arr, any_sl_arr

@jit(nopython=True)
def first_hit_high_low_numba(prices_high, prices_low, prices_close, events_idx, pt_thresholds, sl_thresholds, horizon):
    """
    First hit logic using High/Low.
    """
    n_events = len(events_idx)
    n_prices = len(prices_close)

    first_pt = np.full(n_events, horizon + 1, dtype=np.int32)
    first_sl = np.full(n_events, horizon + 1, dtype=np.int32)
    any_pt_arr = np.zeros(n_events, dtype=np.int8)
    any_sl_arr = np.zeros(n_events, dtype=np.int8)

    for i in range(n_events):
        idx = events_idx[i]
        pt_thresh = pt_thresholds[i]
        sl_thresh = sl_thresholds[i]

        if idx + horizon >= n_prices:
            end_idx = n_prices - 1
        else:
            end_idx = idx + horizon

        entry_price = prices_close[idx]

        for k in range(1, horizon + 1):
            curr_idx = idx + k
            if curr_idx >= n_prices:
                break

            # Check PT with High
            if first_pt[i] == horizon + 1:
                ret_h = (prices_high[curr_idx] / entry_price) - 1.0
                if ret_h > pt_thresh:
                    first_pt[i] = k
                    any_pt_arr[i] = 1

            # Check SL with Low
            if first_sl[i] == horizon + 1:
                ret_l = (prices_low[curr_idx] / entry_price) - 1.0
                if ret_l < sl_thresh:
                    first_sl[i] = k
                    any_sl_arr[i] = 1

            if first_pt[i] <= k and first_sl[i] <= k:
                break

    return first_pt, first_sl, any_pt_arr, any_sl_arr

@jit(nopython=True)
def batch_mi_score_numba(interaction_matrix, target_binned, n_bins=5):
    """
    Compute Mutual Information for each column in interaction_matrix against target_binned.
    interaction_matrix: (N, M) float array
    target_binned: (N,) int array (values 0..n_bins-1)
    """
    N, M = interaction_matrix.shape
    scores = np.zeros(M, dtype=np.float64)

    # Pre-compute target probs
    target_counts = np.zeros(n_bins, dtype=np.float64)
    valid_samples_global = 0

    for i in range(N):
        t = target_binned[i]
        if t >= 0 and t < n_bins:
            target_counts[t] += 1
            valid_samples_global += 1

    # If no valid targets, return 0
    if valid_samples_global == 0:
        return scores

    p_target = target_counts / valid_samples_global

    for j in range(M):
        col = interaction_matrix[:, j]

        # Quantile Binning
        # Create a copy to sort for quantiles
        col_sorted = np.sort(col)

        # Find bin edges
        edges = np.zeros(n_bins + 1, dtype=np.float64)
        edges[0] = col_sorted[0]
        edges[n_bins] = col_sorted[N-1]

        # Percentiles
        for b in range(1, n_bins):
            idx = int(b * N / n_bins)
            if idx >= N: idx = N - 1
            edges[b] = col_sorted[idx]

        # Add small epsilon to max
        edges[n_bins] += 1e-9

        # Binning and Contingency Table
        joint_counts = np.zeros((n_bins, n_bins), dtype=np.float64)
        x_counts = np.zeros(n_bins, dtype=np.float64)

        valid_samples = 0

        for i in range(N):
            val = col[i]
            y = target_binned[i]
            if y < 0 or y >= n_bins: continue

            # Find x bin
            x = 0
            found = False
            for b in range(n_bins):
                if val <= edges[b+1]:
                    x = b
                    found = True
                    break
            if not found: x = n_bins - 1

            joint_counts[x, y] += 1
            x_counts[x] += 1
            valid_samples += 1

        if valid_samples == 0:
            scores[j] = 0.0
            continue

        # Compute MI
        mi = 0.0
        for x in range(n_bins):
            for y in range(n_bins):
                count = joint_counts[x, y]
                if count > 0:
                    px = x_counts[x] / valid_samples
                    # Recalculate py locally for this filtered set (though mostly same as global if only -1s filtered)
                    # But wait, target_binned -1s are filtered in global count.
                    # Is it possible valid_samples != valid_samples_global?
                    # Only if interaction values are NaN?
                    # We didn't check for NaN in interaction col. Assuming pre-filled.
                    # Use global p_target? No, standard MI uses marginals of the joint distribution.

                    # Marginal for y in this joint set
                    py_local = 0.0
                    for xx in range(n_bins):
                        py_local += joint_counts[xx, y]
                    py = py_local / valid_samples

                    pxy = count / valid_samples

                    if px > 0 and py > 0:
                        term = pxy * np.log(pxy / (px * py))
                        mi += term

        scores[j] = mi

    return scores
