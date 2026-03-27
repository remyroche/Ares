import re

with open("extreme_price_movements/labeling.py", "r") as f:
    c = f.read()

# Modification 1: _numba_triple_barrier_outcomes_fast
target_alloc1 = """    n = len(closes)
    outcomes = np.zeros(n, dtype=np.int8)
    quality = np.zeros(n, dtype=np.float32)
    returns = np.zeros(n, dtype=np.float32)
    exit_idxs = np.zeros(n, dtype=np.int64)
    conflict_j = np.full(n, -1, dtype=np.int64)"""

replace_alloc1 = """    n = len(closes)
    outcomes = np.zeros(n, dtype=np.int8)
    quality = np.zeros(n, dtype=np.float32)
    returns = np.zeros(n, dtype=np.float32)
    exit_idxs = np.zeros(n, dtype=np.int64)
    conflict_j = np.full(n, -1, dtype=np.int64)
    mfe_arr = np.zeros(n, dtype=np.float32)
    mae_arr = np.zeros(n, dtype=np.float32)
    time_to_mfe = np.zeros(n, dtype=np.float32)
    time_to_mae = np.zeros(n, dtype=np.float32)"""

c = c.replace(target_alloc1, replace_alloc1)

target_loop1 = """        mfe_val = 0.0
        mae_val = 0.0

        for j in range(j_start, min(j_end, n)):
            tt = times[j]
            hh = highs[j]
            ll = lows[j]
            cc = closes[j]

            # Update MFE/MAE
            if side == 1:
                cur_mfe = max(0.0, hh - entry_p)
                cur_mae = max(0.0, entry_p - ll)
            else:
                cur_mfe = max(0.0, entry_p - ll)
                cur_mae = max(0.0, hh - entry_p)

            if cur_mfe > mfe_val: mfe_val = cur_mfe
            if cur_mae > mae_val: mae_val = cur_mae"""

replace_loop1 = """        mfe_val = 0.0
        mae_val = 0.0
        t_mfe = 0.0
        t_mae = 0.0

        for j in range(j_start, min(j_end, n)):
            tt = times[j]
            hh = highs[j]
            ll = lows[j]
            cc = closes[j]

            # Update MFE/MAE
            if side == 1:
                cur_mfe = max(0.0, hh - entry_p)
                cur_mae = max(0.0, entry_p - ll)
            else:
                cur_mfe = max(0.0, entry_p - ll)
                cur_mae = max(0.0, hh - entry_p)

            if cur_mfe > mfe_val:
                mfe_val = cur_mfe
                t_mfe = (tt - entry_t) / 1e9 / 3600.0
            if cur_mae > mae_val:
                mae_val = cur_mae
                t_mae = (tt - entry_t) / 1e9 / 3600.0"""

c = c.replace(target_loop1, replace_loop1)

target_ret1 = """        if not exit_found:
            outcomes[i] = OUT_TO
            if side == 1: returns[i] = (closes[n-1] / entry_p) - 1.0
            else: returns[i] = (entry_p / closes[n-1]) - 1.0
            exit_idxs[i] = n - 1
            den_tp = max(abs(activation), _QUALITY_EPS)
            rel_prog = returns[i] / den_tp
            quality[i] = 0.5 + _clip_scalar(rel_prog * 0.4, -0.4, 0.4)

    # Numba compatibility: avoid nan_to_num keyword args unsupported in some versions.
    quality = np.nan_to_num(quality)
    quality = np.clip(quality, 0.0, 1.0).astype(np.float32)
    return outcomes, returns, quality, exit_idxs, conflict_j"""

replace_ret1 = """        if not exit_found:
            outcomes[i] = OUT_TO
            if side == 1: returns[i] = (closes[n-1] / entry_p) - 1.0
            else: returns[i] = (entry_p / closes[n-1]) - 1.0
            exit_idxs[i] = n - 1
            den_tp = max(abs(activation), _QUALITY_EPS)
            rel_prog = returns[i] / den_tp
            quality[i] = 0.5 + _clip_scalar(rel_prog * 0.4, -0.4, 0.4)

        mfe_arr[i] = mfe_val / entry_p
        mae_arr[i] = mae_val / entry_p
        time_to_mfe[i] = t_mfe
        time_to_mae[i] = t_mae

    # Numba compatibility: avoid nan_to_num keyword args unsupported in some versions.
    quality = np.nan_to_num(quality)
    quality = np.clip(quality, 0.0, 1.0).astype(np.float32)
    return outcomes, returns, quality, exit_idxs, conflict_j, mfe_arr, mae_arr, time_to_mfe, time_to_mae"""

c = c.replace(target_ret1, replace_ret1)


# Also need to fix _numba_triple_barrier_outcomes (without fast)
target_ret_o1 = """                rel_prog = returns[i] / den_tp
                quality[i] = 0.5 + _clip_scalar(rel_prog * 0.4, -0.4, 0.4)
                break

    return outcomes, quality, returns, exit_idxs, conflict_j"""
replace_ret_o1 = """                rel_prog = returns[i] / den_tp
                quality[i] = 0.5 + _clip_scalar(rel_prog * 0.4, -0.4, 0.4)
                break

        mfe_arr[i] = mfe_val / entry_p
        mae_arr[i] = mae_val / entry_p
        time_to_mfe[i] = t_mfe
        time_to_mae[i] = t_mae

    return outcomes, quality, returns, exit_idxs, conflict_j, mfe_arr, mae_arr, time_to_mfe, time_to_mae"""
c = c.replace(target_ret_o1, replace_ret_o1)


# Next: _numba_triple_barrier_fast
target_alloc2 = """    n = len(closes)
    labels = np.zeros(n, dtype=np.int8)
    returns = np.zeros(n, dtype=np.float32)
    exit_idxs = np.zeros(n, dtype=np.int64)
    conflict_j = np.full(n, -1, dtype=np.int64)"""

replace_alloc2 = """    n = len(closes)
    labels = np.zeros(n, dtype=np.int8)
    returns = np.zeros(n, dtype=np.float32)
    exit_idxs = np.zeros(n, dtype=np.int64)
    conflict_j = np.full(n, -1, dtype=np.int64)
    mfe_arr = np.zeros(n, dtype=np.float32)
    mae_arr = np.zeros(n, dtype=np.float32)
    time_to_mfe = np.zeros(n, dtype=np.float32)
    time_to_mae = np.zeros(n, dtype=np.float32)"""

c = c.replace(target_alloc2, replace_alloc2)

target_loop2 = """        trailing_active = False
        exit_found = False

        for j in range(j_start, min(j_end, n)):
            tt = times[j]
            hh = highs[j]
            ll = lows[j]
            cc = closes[j]

            # Handle NaN high/low"""

replace_loop2 = """        trailing_active = False
        exit_found = False

        mfe_val = 0.0
        mae_val = 0.0
        t_mfe = 0.0
        t_mae = 0.0

        for j in range(j_start, min(j_end, n)):
            tt = times[j]
            hh = highs[j]
            ll = lows[j]
            cc = closes[j]

            if side == 1:
                cur_mfe = max(0.0, hh - entry_p)
                cur_mae = max(0.0, entry_p - ll)
            else:
                cur_mfe = max(0.0, entry_p - ll)
                cur_mae = max(0.0, hh - entry_p)

            if cur_mfe > mfe_val:
                mfe_val = cur_mfe
                t_mfe = (tt - entry_t) / 1e9 / 3600.0
            if cur_mae > mae_val:
                mae_val = cur_mae
                t_mae = (tt - entry_t) / 1e9 / 3600.0

            # Handle NaN high/low"""
c = c.replace(target_loop2, replace_loop2)

target_ret2 = """        if not exit_found:
            # Timeout at end of window or data
            final_idx = min(j_end, n - 1)
            labels[i] = OUT_TO
            returns[i] = (closes[final_idx] / entry_p - 1.0) if side == 1 else (entry_p / closes[final_idx] - 1.0)
            exit_idxs[i] = final_idx

    return labels, returns, exit_idxs, conflict_j"""

replace_ret2 = """        if not exit_found:
            # Timeout at end of window or data
            final_idx = min(j_end, n - 1)
            labels[i] = OUT_TO
            returns[i] = (closes[final_idx] / entry_p - 1.0) if side == 1 else (entry_p / closes[final_idx] - 1.0)
            exit_idxs[i] = final_idx

        mfe_arr[i] = mfe_val / entry_p
        mae_arr[i] = mae_val / entry_p
        time_to_mfe[i] = t_mfe
        time_to_mae[i] = t_mae

    return labels, returns, exit_idxs, conflict_j, mfe_arr, mae_arr, time_to_mfe, time_to_mae"""
c = c.replace(target_ret2, replace_ret2)


target_compute_ret = """        if return_outcomes:
            out, rets, qual, _, conflict_j = _numba_triple_barrier_outcomes_fast(
                times, o_arr, h_arr, l_arr, c_arr, tp_arr, sl_arr, horizon, side_int, horizons_arr=h_arr_custom
            )
            return asset, out, rets, qual, conflict_j, tp_arr, sl_arr
        else:
            lbs, rets, _, conflict_j = _numba_triple_barrier_fast(times, o_arr, h_arr, l_arr, c_arr, tp_arr, sl_arr, horizon, side_int, horizons_arr=h_arr_custom)
            return asset, lbs, rets, None, conflict_j, tp_arr, sl_arr"""

replace_compute_ret = """        if return_outcomes:
            out, rets, qual, _, conflict_j, mfe_arr, mae_arr, t_mfe, t_mae = _numba_triple_barrier_outcomes_fast(
                times, o_arr, h_arr, l_arr, c_arr, tp_arr, sl_arr, horizon, side_int, horizons_arr=h_arr_custom
            )
            return asset, out, rets, qual, conflict_j, tp_arr, sl_arr, mfe_arr, mae_arr, t_mfe, t_mae
        else:
            lbs, rets, _, conflict_j, mfe_arr, mae_arr, t_mfe, t_mae = _numba_triple_barrier_fast(times, o_arr, h_arr, l_arr, c_arr, tp_arr, sl_arr, horizon, side_int, horizons_arr=h_arr_custom)
            return asset, lbs, rets, None, conflict_j, tp_arr, sl_arr, mfe_arr, mae_arr, t_mfe, t_mae"""

c = c.replace(target_compute_ret, replace_compute_ret)

with open("extreme_price_movements/labeling.py", "w") as f:
    f.write(c)
