import re

file_path = "extreme_price_movements/lgbm_based_mask_generation.py"

with open(file_path, "r") as f:
    content = f.read()

# Replace tbm_outcomes_atr_nb
old_tbm = """@njit(cache=True, fastmath=True)
def tbm_outcomes_atr_nb(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    horizon: int,
    tp_atr: float,
    sl_atr: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = close.shape[0]
    tp_first = np.zeros(n, dtype=np.int8)
    sl_first = np.zeros(n, dtype=np.int8)
    timeout = np.zeros(n, dtype=np.int8)

    if horizon <= 0:
        return tp_first, sl_first, timeout

    for i in range(n - horizon):
        entry = close[i]
        atr_i = max(atr[i], 1e-9)

        tp_price = entry + tp_atr * atr_i
        sl_price = entry - sl_atr * atr_i

        for j in range(i + 1, i + horizon + 1):
            hi = high[j]
            lo = low[j]

            hit_tp = hi >= tp_price
            hit_sl = lo <= sl_price

            if hit_tp and not hit_sl:
                tp_first[i] = 1
                break
            if hit_sl and not hit_tp:
                sl_first[i] = 1
                break
            if hit_tp and hit_sl:
                median = 0.5 * (hi + lo)
                d_tp = abs(median - tp_price)
                d_sl = abs(median - sl_price)
                if d_tp < d_sl:
                    tp_first[i] = 1
                else:
                    sl_first[i] = 1
                break

        if tp_first[i] == 0 and sl_first[i] == 0:
            timeout[i] = 1

    return tp_first, sl_first, timeout"""

new_tbm = """@njit(cache=True, fastmath=True)
def tbm_outcomes_atr_nb(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    horizon: int,
    tp_atr: float,
    sl_atr: float,
    side_mult: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = close.shape[0]
    tp_first = np.zeros(n, dtype=np.int8)
    sl_first = np.zeros(n, dtype=np.int8)
    timeout = np.zeros(n, dtype=np.int8)

    if horizon <= 0:
        return tp_first, sl_first, timeout

    for i in range(n - horizon):
        entry = close[i]

        # Guard against zero or negative entry prices
        if entry <= 1e-9:
            timeout[i] = 1
            continue

        atr_i = max(atr[i], 1e-9)
        atr_pct = atr_i / entry

        # Thresholds in terms of percentage returns
        tp_thresh = tp_atr * atr_pct
        sl_thresh = sl_atr * atr_pct

        for j in range(i + 1, i + horizon + 1):
            hi = high[j]
            lo = low[j]

            if side_mult > 0:
                ret_fav = (hi - entry) / entry
                ret_adv = (entry - lo) / entry
            else:
                ret_fav = (entry - lo) / entry
                ret_adv = (hi - entry) / entry

            hit_tp = ret_fav >= tp_thresh
            hit_sl = ret_adv >= sl_thresh

            if hit_tp and not hit_sl:
                tp_first[i] = 1
                break
            if hit_sl and not hit_tp:
                sl_first[i] = 1
                break
            if hit_tp and hit_sl:
                # If both hit in the same bar, fallback to checking which price median is closer to
                median = 0.5 * (hi + lo)

                if side_mult > 0:
                    tp_price = entry * (1.0 + tp_thresh)
                    sl_price = entry * (1.0 - sl_thresh)
                else:
                    tp_price = entry * (1.0 - tp_thresh)
                    sl_price = entry * (1.0 + sl_thresh)

                d_tp = abs(median - tp_price)
                d_sl = abs(median - sl_price)
                if d_tp < d_sl:
                    tp_first[i] = 1
                else:
                    sl_first[i] = 1
                break

        if tp_first[i] == 0 and sl_first[i] == 0:
            timeout[i] = 1

    return tp_first, sl_first, timeout"""

content = content.replace(old_tbm, new_tbm)

# Replace compute_tbm_outcomes_per_symbol
old_compute = """def compute_tbm_outcomes_per_symbol(
    data: pd.DataFrame,
    horizon: int,
    tp_atr: float,
    sl_atr: float,
    side: str = "long",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    \"\"\"
    Compute TBM outcomes independently within each symbol's time series.

    Assumes `data` has columns:
      - symbol
      - timestamp
      - close
      - high
      - low
      - atr

    Returns arrays aligned to `data.index`.
    \"\"\"
    if data.empty:
        z = np.zeros(0, dtype=np.int8)
        return z, z, z

    # Preserve original row order for final alignment
    out_tp = np.zeros(len(data), dtype=np.int8)
    out_sl = np.zeros(len(data), dtype=np.int8)
    out_to = np.zeros(len(data), dtype=np.int8)

    # Sort once for temporal correctness inside each symbol
    work = data.reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    work = work.sort_values(["symbol", "timestamp"], kind="mergesort")

    for sym, g in work.groupby("symbol", sort=False):
        idx = g["_orig_idx"].to_numpy()

        close = g["close"].to_numpy(dtype=np.float64, copy=False)
        high = g["high"].to_numpy(dtype=np.float64, copy=False)
        low = g["low"].to_numpy(dtype=np.float64, copy=False)
        atr = g["atr"].to_numpy(dtype=np.float64, copy=False)

        if side == "short":
            c, h, l = -close, -low, -high
        else:
            c, h, l = close, high, low

        tp_f, sl_f, to_f = tbm_outcomes_atr_nb(
            close=c,
            high=h,
            low=l,
            atr=atr,
            horizon=horizon,
            tp_atr=tp_atr,
            sl_atr=sl_atr,
        )

        out_tp[idx] = tp_f
        out_sl[idx] = sl_f
        out_to[idx] = to_f

    return out_tp, out_sl, out_to"""

new_compute = """def compute_tbm_outcomes_per_symbol(
    data: pd.DataFrame,
    horizon: int,
    tp_atr: float,
    sl_atr: float,
    side: str = "long",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    \"\"\"
    Compute TBM outcomes independently within each symbol's time series.
    Evaluated using true forward percentage returns as required.

    Assumes `data` has columns:
      - symbol
      - timestamp
      - close
      - high
      - low
      - atr

    Returns arrays aligned to `data.index`.
    \"\"\"
    if data.empty:
        z = np.zeros(0, dtype=np.int8)
        return z, z, z

    # Preserve original row order for final alignment
    out_tp = np.zeros(len(data), dtype=np.int8)
    out_sl = np.zeros(len(data), dtype=np.int8)
    out_to = np.zeros(len(data), dtype=np.int8)

    # Sort once for temporal correctness inside each symbol
    work = data.reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    work = work.sort_values(["symbol", "timestamp"], kind="mergesort")

    side_mult = -1.0 if side == "short" else 1.0

    for sym, g in work.groupby("symbol", sort=False):
        idx = g["_orig_idx"].to_numpy()

        close = g["close"].to_numpy(dtype=np.float64, copy=False)
        high = g["high"].to_numpy(dtype=np.float64, copy=False)
        low = g["low"].to_numpy(dtype=np.float64, copy=False)
        atr = g["atr"].to_numpy(dtype=np.float64, copy=False)

        tp_f, sl_f, to_f = tbm_outcomes_atr_nb(
            close=close,
            high=high,
            low=low,
            atr=atr,
            horizon=horizon,
            tp_atr=tp_atr,
            sl_atr=sl_atr,
            side_mult=side_mult,
        )

        out_tp[idx] = tp_f
        out_sl[idx] = sl_f
        out_to[idx] = to_f

    return out_tp, out_sl, out_to"""

content = content.replace(old_compute, new_compute)

with open(file_path, "w") as f:
    f.write(content)

print("Patched.")
