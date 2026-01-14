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
    Returns: mfe, mae, final_ret, hit_high_matrix (simulated), hit_low_matrix (simulated)
    To match full logic, we return raw arrays to Python for threshold checking.
    Actually, to save memory, we should compute mfe/mae directly.
    """
    n_events = len(events_idx)
    n_prices = len(prices)

    mfe = np.zeros(n_events, dtype=np.float64)
    mae = np.zeros(n_events, dtype=np.float64)
    final_ret = np.zeros(n_events, dtype=np.float64)

    # We also need return paths for "first hit" logic if we want to do it in python.
    # But optimal is to do it here. However, `compute_dominance_labels` logic is complex.
    # It constructs full boolean matrices `hit_pt` and `hit_sl`.
    # To avoid O(N*H) memory, we return mfe/mae/final_ret and handle complex logic differently?
    # No, `compute_dominance_labels` uses `hit_pt` (matrix) to find `first_pt_idx`.
    # We can compute `first_pt_idx` and `first_sl_idx` here given thresholds.
    # BUT thresholds depend on `vol`.
    # So we should pass thresholds or vol arrays.

    # Let's keep this function simple: just MFE, MAE, Returns.
    # The boolean matrix logic in original code is:
    # hit_pt = returns_matrix > pt_thresh
    # first_pt_idx = np.argmax(hit_pt, axis=1)

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

        # Scan window
        for k in range(idx + 1, end_idx + 1):
            p = prices[k]
            ret = (p / entry_price) - 1.0

            if ret > local_max:
                local_max = ret
            if ret < local_min:
                local_min = ret

        # Final return
        final_ret[i] = (prices[end_idx] / entry_price) - 1.0

        # MFE is max positive excursion
        mfe[i] = max(0.0, local_max) if local_max != -999.0 else 0.0

        # MAE is max negative excursion (magnitude)
        # Original: mae = np.max(-returns_matrix) -> max of negative returns inverted
        # i.e., min return is -0.05, mae is 0.05.
        # But if all returns are positive, mae is 0?
        # Original: -returns_matrix. So if ret=0.01, -ret=-0.01. max is negative?
        # No, usually MAE is positive.
        # If returns are [0.01, 0.02], -returns are [-0.01, -0.02]. max is -0.01.
        # Wait, the original code: `mae = np.max(-returns_matrix, axis=1)`
        # If all returns are positive, `mae` would be negative? That seems wrong for "Magnitude".
        # Let's check original code logic again.
        # `risk_used = mae / np.maximum(stop_dist, 1e-9)`
        # If mae is negative, risk_used is negative.
        # Usually MAE is positive number representing draw-down.
        # If `returns_matrix` has negative values (drawdowns), `-returns` has positive values.
        # So `max(-returns)` captures the largest drawdown as a positive number.
        # If all returns are positive, `max(-returns)` is the least positive return (closest to 0) but negative.
        # e.g. [-0.01, -0.02]. max is -0.01.
        # So if price never drops below entry, MAE should be 0?
        # Standard MAE definition: Maximum Adverse Excursion.
        # Logic: -min(ret). If min(ret) is positive (never drops), -min is negative.
        # We should clip at 0.

        val_mae = -local_min if local_min != 999.0 else 0.0
        # If val_mae is negative (meaning local_min was positive), clamp to 0?
        # Original code doesn't clamp explicitly but risk logic might handle it.
        # `risk_used` would be negative, `risk_mask = risk_used <= risk_budget` (negative <= positive) is True.
        # So it works.
        mae[i] = val_mae

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

        entry_price = prices_close[idx] # Entry is always on Close? Or Open of next? Usually Close of signal bar.

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
        sl_thresh = sl_thresholds[i] # Note: this should be negative value or magnitude?
        # Based on original: sl_thresh = -vol * sl_mult (so it's negative)
        # hit_sl = ret < sl_thresh

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

            # Optimization: if both hit, we can stop?
            # We need the *first* index. If we found both, we are done?
            # Yes, because we scan sequentially. The first time we see a hit is the min index.
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
