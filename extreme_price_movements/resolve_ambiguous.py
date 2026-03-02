import pandas as pd
import numpy as np
import ccxt
import time
from extreme_price_movements.hf_data_loader import get_15m_ohlcv

# Singleton exchange
_EXCHANGE = None

def get_exchange():
    global _EXCHANGE
    if _EXCHANGE is None:
        _EXCHANGE = ccxt.binance({
            'enableRateLimit': True,
        })
    return _EXCHANGE

def resolve_ambiguous_bar(symbol, ts, side, entry_p, tp_dist, sl_dist, tp_price, sl_price, open_p, high_p, low_p, close_p):
    """
    Resolves an ambiguous bar (where both TP and SL are hit in the same bar)
    by fetching 15m data and simulating.
    If 15m is also ambiguous or missing, fallback to Open/Close logic.
    """
    exc = get_exchange()
    try:
        # The ambiguous bar is at timestamp `ts`. Since it might be a 1h bar or 4h,
        # we fetch 12 hours from `ts` to be safe and cover the whole bar.
        df_15m = get_15m_ohlcv(exc, symbol, pd.Timestamp(ts, unit='ns', tz='UTC'), max_hold_hours=24)

        if not df_15m.empty:
            # We only want bars starting from `ts` until the end of the ambiguous bar.
            # But wait, we don't know the timeframe of the ambiguous bar exactly here.
            # Usually it's 1h or 4h or 15m.
            # We can just iterate through df_15m starting from `ts` and see which hits first.

            # Start simulation from `ts`
            sub_df = df_15m[df_15m.index >= pd.Timestamp(ts, unit='ns', tz='UTC')]

            for _, row in sub_df.iterrows():
                h = row['high']
                l = row['low']
                c = row['close']
                o = row['open']

                hit_tp = False
                hit_sl = False

                if side == 1:
                    if l <= sl_price: hit_sl = True
                    if h >= tp_price: hit_tp = True
                else:
                    if h >= sl_price: hit_sl = True
                    if l <= tp_price: hit_tp = True

                if hit_tp and hit_sl:
                    # Still ambiguous on 15m! Break and use fallback
                    break
                elif hit_tp:
                    return 2, tp_dist
                elif hit_sl:
                    return 0, -sl_dist

                # If we passed the ambiguous bar (price goes outside the bar's range), break
                # But typically one of them will hit.
                if (side == 1 and (h > high_p or l < low_p)) or (side == -1 and (h > high_p or l < low_p)):
                    # We might have left the ambiguous bar's true period
                    pass

    except Exception as e:
        print(f"Warning: Failed to fetch 15m data for {symbol} at {ts}: {e}")

    # Fallback logic: "if price is higher than high (longs) or lower than low (shorts) and vice versa"
    # Interpreted as: if it's a green bar (Close > Open) for Longs -> Win.
    # If it's a red bar (Close < Open) for Shorts -> Win.
    if side == 1:
        if close_p > open_p:
            return 2, tp_dist
        else:
            return 0, -sl_dist
    else:
        if close_p < open_p:
            return 2, tp_dist
        else:
            return 0, -sl_dist
