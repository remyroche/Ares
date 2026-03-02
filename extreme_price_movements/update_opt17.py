import re

with open("extreme_price_movements/labeling.py", "r") as f:
    content = f.read()

# Let's add ambiguous_idxs to _numba_triple_barrier_outcomes output
search1 = """    n = len(closes)
    outcomes = np.zeros(n, dtype=np.int8)
    quality = np.zeros(n, dtype=np.float32)
    returns = np.zeros(n, dtype=np.float32)
    exit_idxs = np.zeros(n, dtype=np.int64)"""

replace1 = """    n = len(closes)
    outcomes = np.zeros(n, dtype=np.int8)
    quality = np.zeros(n, dtype=np.float32)
    returns = np.zeros(n, dtype=np.float32)
    exit_idxs = np.zeros(n, dtype=np.int64)
    ambiguous_flags = np.zeros(n, dtype=np.bool_)"""

content = content.replace(search1, replace1)

search2 = """            if hit_sl and hit_tp:
                # Ambiguous bar - Fallback logic: close proximity to extrema
                # Consider it a win if close price of the ambiguous bar is closer to the high (for longs) / low (for shorts)
                dist_to_high = abs(hh - cc)"""

replace2 = """            if hit_sl and hit_tp:
                ambiguous_flags[i] = True
                # Ambiguous bar - Fallback logic: close proximity to extrema
                # Consider it a win if close price of the ambiguous bar is closer to the high (for longs) / low (for shorts)
                dist_to_high = abs(hh - cc)"""

content = content.replace(search2, replace2)

search3 = """    # Numba compatibility: avoid nan_to_num keyword args unsupported in some versions.
    quality = np.nan_to_num(quality)
    quality = np.clip(quality, 0.0, 1.0).astype(np.float32)
    return outcomes, returns, quality, exit_idxs"""

replace3 = """    # Numba compatibility: avoid nan_to_num keyword args unsupported in some versions.
    quality = np.nan_to_num(quality)
    quality = np.clip(quality, 0.0, 1.0).astype(np.float32)
    return outcomes, returns, quality, exit_idxs, ambiguous_flags"""

content = content.replace(search3, replace3)


search4 = """        if return_outcomes:
            rets, ret_arr, qual, exit_indices = _numba_triple_barrier_outcomes(
                times, o_arr, h_arr, l_arr, c_arr,
                tp_arr, sl_arr, float(horizon), side_int, h_arr_custom
            )
            out_labels[asset] = rets
            out_returns[asset] = ret_arr
            out_quality[asset] = qual
        else:
            rets, ret_arr, exit_indices = _numba_triple_barrier(
                times, o_arr, h_arr, l_arr, c_arr,
                tp_arr, sl_arr, float(horizon), side_int, h_arr_custom
            )
            out_labels[asset] = rets
            out_returns[asset] = ret_arr"""

replace4 = """        if return_outcomes:
            rets, ret_arr, qual, exit_indices, ambig_flags = _numba_triple_barrier_outcomes(
                times, o_arr, h_arr, l_arr, c_arr,
                tp_arr, sl_arr, float(horizon), side_int, h_arr_custom
            )
            out_labels[asset] = rets
            out_returns[asset] = ret_arr
            out_quality[asset] = qual

            # If 15m resolution logic is needed, we could fetch here.
            # But downloading in Numba loop / parallel loop is bad.
        else:
            rets, ret_arr, exit_indices = _numba_triple_barrier(
                times, o_arr, h_arr, l_arr, c_arr,
                tp_arr, sl_arr, float(horizon), side_int, h_arr_custom
            )
            out_labels[asset] = rets
            out_returns[asset] = ret_arr"""

content = content.replace(search4, replace4)

with open("extreme_price_movements/labeling.py", "w") as f:
    f.write(content)
