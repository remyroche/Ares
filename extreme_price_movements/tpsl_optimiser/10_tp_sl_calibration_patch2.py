import pandas as pd
import numpy as np

def precompute_15m_bars_for_trades(trades: pd.DataFrame, data_15m: dict) -> list:
    """
    Precomputes the relevant 15m bars for each trade so we don't index/mask
    inside the nested optimization loop.
    Returns a list of length len(trades). Each element is either None
    or a 2D numpy array: [[high, low, close], ...]
    """
    bars_list = []

    if "timestamp" not in trades.columns or "asset" not in trades.columns:
        return [None] * len(trades)

    for i in range(len(trades)):
        asset = trades["asset"].values[i]
        if asset not in data_15m:
            bars_list.append(None)
            continue

        df_15 = data_15m[asset]
        ts = pd.to_datetime(trades["timestamp"].values[i], utc=True)
        end_ts = ts + pd.Timedelta(hours=1)

        # mask is just O(log N) if index is sorted, but doing it outside loop is faster.
        mask = (df_15.index >= ts) & (df_15.index < end_ts)
        bars = df_15.loc[mask]

        if bars.empty:
            bars_list.append(None)
        else:
            # We need high, low, close
            bars_arr = bars[['high', 'low', 'close']].to_numpy(dtype=float)
            bars_list.append(bars_arr)

    return bars_list

def resolve_double_hits_fast(
    trades: pd.DataFrame,
    base_ret: np.ndarray,
    tp_pct: np.ndarray,
    sl_pct: np.ndarray,
    bars_list: list
) -> np.ndarray:
    clipped = np.clip(base_ret, -sl_pct, tp_pct)

    if not bars_list:
        return clipped

    resolved = clipped.copy()

    is_long_arr = trades["is_long"].values
    entry_p_arr = trades["entry_price"].values

    for i in range(len(trades)):
        bars = bars_list[i]
        if bars is None:
            continue

        entry_p = float(entry_p_arr[i])
        if entry_p <= 0:
            continue

        tp = float(tp_pct[i])
        sl = float(sl_pct[i])
        is_long = int(is_long_arr[i])

        if is_long == 1:
            sl_price = entry_p * (1.0 - sl)
            tp_price = entry_p * (1.0 + tp)
        else:
            sl_price = entry_p * (1.0 + sl)
            tp_price = entry_p * (1.0 - tp)

        for j in range(bars.shape[0]):
            hh = bars[j, 0]
            ll = bars[j, 1]
            cc = bars[j, 2]

            bar_hit_tp = False
            bar_hit_sl = False

            if is_long == 1:
                if ll <= sl_price: bar_hit_sl = True
                if hh >= tp_price: bar_hit_tp = True
            else:
                if hh >= sl_price: bar_hit_sl = True
                if ll <= tp_price: bar_hit_tp = True

            if bar_hit_tp and not bar_hit_sl:
                resolved[i] = tp
                break
            elif bar_hit_sl and not bar_hit_tp:
                resolved[i] = -sl
                break
            elif bar_hit_sl and bar_hit_tp:
                d_tp = abs(cc - tp_price)
                d_sl = abs(cc - sl_price)
                if d_tp < d_sl:
                    resolved[i] = tp
                else:
                    resolved[i] = -sl
                break

    return resolved
