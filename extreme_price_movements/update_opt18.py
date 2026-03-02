import re

with open("extreme_price_movements/labeling.py", "r") as f:
    content = f.read()

# Let's add full 15m fetching to compute_triple_barrier_labels!

search_import = "from numba import jit"
replace_import = """from numba import jit
import warnings
import ccxt
from extreme_price_movements.hf_data_loader import get_15m_ohlcv"""

content = content.replace(search_import, replace_import)

search4 = """        if return_outcomes:
            rets, ret_arr, qual, exit_indices, ambig_flags = _numba_triple_barrier_outcomes(
                times, o_arr, h_arr, l_arr, c_arr,
                tp_arr, sl_arr, float(horizon), side_int, h_arr_custom
            )
            out_labels[asset] = rets
            out_returns[asset] = ret_arr
            out_quality[asset] = qual

            # If 15m resolution logic is needed, we could fetch here.
            # But downloading in Numba loop / parallel loop is bad.
        else:"""

replace4 = """        if return_outcomes:
            rets, ret_arr, qual, exit_indices, ambig_flags = _numba_triple_barrier_outcomes(
                times, o_arr, h_arr, l_arr, c_arr,
                tp_arr, sl_arr, float(horizon), side_int, h_arr_custom
            )

            # Use 15m OHLCV data to resolve ambiguous paths
            if np.any(ambig_flags):
                try:
                    exchange = ccxt.binance({'enableRateLimit': True})
                    ccxt_sym = asset if "/" in asset else asset.replace("USDT", "/USDT")

                    ambig_indices = np.where(ambig_flags)[0]
                    for i in ambig_indices:
                        entry_idx = i
                        exit_idx = exit_indices[i]

                        entry_t = times[entry_idx]
                        exit_t = times[exit_idx]

                        # Convert ns to Timestamp
                        ts_start = pd.Timestamp(entry_t, unit='ns', tz='UTC')
                        ts_end = pd.Timestamp(exit_t, unit='ns', tz='UTC')

                        # Max hold hours is the horizon (for fetching)
                        h_eff = float(horizon) if h_arr_custom is None else h_arr_custom[i]
                        max_hold = int(np.ceil(h_eff))

                        # Fetch 15m data
                        df_15m = get_15m_ohlcv(exchange, ccxt_sym, ts_start, max_hold)

                        if not df_15m.empty:
                            # Filter to the specific 1h ambiguous bar
                            df_15m_bar = df_15m[(df_15m.index >= ts_end) & (df_15m.index < ts_end + pd.Timedelta(hours=1))]

                            if not df_15m_bar.empty:
                                # Re-simulate the ambiguous bar with 15m resolution
                                # We want to see which hit first: SL or TP
                                tp_price = c_arr[entry_idx] * (1.0 + (tp_arr[entry_idx] if side == "long" else -tp_arr[entry_idx]))
                                sl_price = c_arr[entry_idx] * (1.0 - (sl_arr[entry_idx] if side == "long" else -sl_arr[entry_idx]))

                                hit_tp_15m = False
                                hit_sl_15m = False

                                for _, row15 in df_15m_bar.iterrows():
                                    if side == "long":
                                        if row15['low'] <= sl_price: hit_sl_15m = True
                                        if row15['high'] >= tp_price: hit_tp_15m = True
                                    else:
                                        if row15['high'] >= sl_price: hit_sl_15m = True
                                        if row15['low'] <= tp_price: hit_tp_15m = True

                                    if hit_sl_15m or hit_tp_15m:
                                        break

                                if hit_tp_15m and not hit_sl_15m:
                                    rets[i] = OUT_TP
                                    ret_arr[i] = tp_arr[entry_idx]
                                    qual[i] = 1.0 # simplified quality update
                                elif hit_sl_15m and not hit_tp_15m:
                                    rets[i] = OUT_SL
                                    ret_arr[i] = -sl_arr[entry_idx]
                                    qual[i] = 0.0 # simplified quality update
                                # If STILL ambiguous on 15m, the fallback logic (close to extreme) that already ran in numba is kept!

                except Exception as e:
                    warnings.warn(f"Failed to fetch 15m data for ambiguity resolution on {asset}: {e}")

            out_labels[asset] = rets
            out_returns[asset] = ret_arr
            out_quality[asset] = qual

        else:"""

content = content.replace(search4, replace4)

with open("extreme_price_movements/labeling.py", "w") as f:
    f.write(content)
