import os

import numpy as np
import pandas as pd

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from src.utils.tprint import tprint_info, tprint_warning

from src.features_common.transforms.scaling_normalization import winsorized_zscore_normalize
from src.utils.feature_common.atr_normalization import atr_normalize
from src.utils.feature_common.volume_transforms import log1p_zscore_normalize

try:
    import lightgbm as lgb

    _LGBM_AVAILABLE = True
except Exception:
    _LGBM_AVAILABLE = False


try:
    from sklearn.model_selection import TimeSeriesSplit

    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False
    TimeSeriesSplit = None


def _is_sorted_unique_index(index: pd.Index) -> bool:
    try:
        if not bool(index.is_unique):
            return False
        if isinstance(index, pd.DatetimeIndex):
            return bool(index.is_monotonic_increasing)
        return True
    except Exception:
        return False


def _ensure_sorted_unique(
    df: pd.DataFrame,
    name: str,
    *,
    hardening_cfg: Dict[str, Any],
    verbose: bool,
) -> pd.DataFrame:
    ensure = bool(hardening_cfg.get("ensure_sorted_unique", True))
    if not ensure:
        return df

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df

    if _is_sorted_unique_index(df.index):
        return df

    auto_sort = bool(hardening_cfg.get("auto_sort", True))
    raise_on_fail = bool(hardening_cfg.get("raise_on_bad_index", True))
    try:
        if not bool(df.index.is_unique):
            msg = f"[{name}] index is not unique"
            if raise_on_fail:
                raise ValueError(msg)
            if verbose:
                tprint_warning(msg)
        if isinstance(df.index, pd.DatetimeIndex) and not bool(df.index.is_monotonic_increasing):
            msg = f"[{name}] DatetimeIndex not sorted"
            if auto_sort:
                if verbose:
                    tprint_warning(msg + " -> sorting")
                df = df.sort_index()
            else:
                if raise_on_fail:
                    raise ValueError(msg)
                if verbose:
                    tprint_warning(msg)
    except Exception as exc:
        if raise_on_fail:
            raise
        if verbose:
            tprint_warning(f"[{name}] index hardening failed: {exc}")
    return df


def _detect_forbidden_feature_columns(
    cols: Iterable[str],
    hardening_cfg: Dict[str, Any],
) -> List[str]:
    explicit = hardening_cfg.get("forbidden_columns")
    if not isinstance(explicit, (list, tuple, set)):
        explicit = []
    explicit_set = {str(c).lower() for c in explicit}

    forbidden_substrings = hardening_cfg.get("forbidden_substrings")
    if not isinstance(forbidden_substrings, (list, tuple, set)):
        forbidden_substrings = [
            "label",
            "target",
            "realized_return",
            "future_",
            "meta_probability",
            "probability",
            "tpsl",
            "barrier",
            "triple_barrier",
            "sample_weight",
            "event_",
        ]
    forbidden_substrings = [str(s).lower() for s in forbidden_substrings]

    out = []
    for c in cols:
        lc = str(c).lower()
        if lc in explicit_set:
            out.append(str(c))
            continue
        if any(tok in lc for tok in forbidden_substrings):
            out.append(str(c))
    return sorted(set(out))


def _time_features_from_index(index: pd.Index, cfg: Dict[str, Any]) -> pd.DataFrame:
    if not bool(cfg.get("enabled", True)):
        return pd.DataFrame(index=index)
    if not isinstance(index, pd.DatetimeIndex):
        return pd.DataFrame(index=index)

    minutes = index.hour.astype(float) * 60.0 + index.minute.astype(float)
    phase = 2.0 * np.pi * (minutes / 1440.0)
    tod_sin = np.sin(phase)
    tod_cos = np.cos(phase)

    dow = index.dayofweek.astype(float)
    dow_phase = 2.0 * np.pi * (dow / 7.0)
    dow_sin = np.sin(dow_phase)
    dow_cos = np.cos(dow_phase)

    out = pd.DataFrame(index=index)
    out["regime_time_sin__tod"] = tod_sin
    out["regime_time_cos__tod"] = tod_cos
    out["regime_time_sin__dow"] = dow_sin
    out["regime_time_cos__dow"] = dow_cos

    if bool(cfg.get("include_month", False)):
        m = (index.month.astype(float) - 1.0) / 12.0
        m_phase = 2.0 * np.pi * m
        out["regime_time_sin__month"] = np.sin(m_phase)
        out["regime_time_cos__month"] = np.cos(m_phase)
    return out


def _infer_max_lookahead_bars_from_targets_cfg(targets_cfg: dict) -> int:
    try:
        explicit = targets_cfg.get("max_lookahead_bars")
        if explicit is not None:
            return int(explicit)
    except Exception:
        pass

    cands = []
    for k in [
        "volatility_window",
        "volatility_window_short",
        "macro_trend_horizon",
        "trend_efficiency_window",
        "liquidity_window",
        "liquidity_norm_window",
        "memory_window",
        "range_window",
        "downside_horizon",
        "vol_of_vol_window",
    ]:
        try:
            if k in targets_cfg and targets_cfg.get(k) is not None:
                cands.append(int(targets_cfg.get(k)))
        except Exception:
            pass

    try:
        mh = targets_cfg.get("macro_trend_horizons")
        if isinstance(mh, (list, tuple)) and mh:
            cands.append(int(max([int(x) for x in mh if x is not None] + [1])))
    except Exception:
        pass
    return max([1] + cands)


def _prune_train_for_lookahead(X_train: pd.DataFrame, y_train: pd.Series, max_lookahead_bars: int):
    if max_lookahead_bars is None:
        return X_train, y_train
    try:
        max_lookahead_bars = int(max_lookahead_bars)
    except Exception:
        return X_train, y_train
    if max_lookahead_bars <= 0:
        return X_train, y_train

    if len(X_train) <= max_lookahead_bars + 5:
        return X_train, y_train
    # Drop last bars so that y(t) cannot depend on prices inside the test window
    X_train_pruned = X_train.iloc[: -max_lookahead_bars]
    y_train_pruned = y_train.reindex(X_train_pruned.index)
    return X_train_pruned, y_train_pruned


def _find_elbow_k(values_desc: np.ndarray, min_k: int = 5) -> int:
    try:
        vals = np.asarray(values_desc, dtype=float)
        if vals.size == 0:
            return 0
        if vals.size <= min_k:
            return int(vals.size)

        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        denom = (vmax - vmin) if (vmax - vmin) > 1e-12 else 1.0
        norm = (vals - vmin) / denom

        x = np.arange(norm.size, dtype=float)
        line = norm[0] + (norm[-1] - norm[0]) * (x / max(1.0, float(norm.size - 1)))
        dist = np.abs(norm - line)
        elbow_idx = int(np.argmax(dist))
        elbow_idx = max(elbow_idx, int(min_k - 1))
        return int(elbow_idx + 1)
    except Exception:
        return int(max(0, min_k))


def _predict_leaf_matrix(model: Any, X_test: pd.DataFrame) -> Optional[np.ndarray]:
    try:
        leaf_mat = model.predict(X_test, pred_leaf=True)
        arr = np.asarray(leaf_mat)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim == 2 and int(arr.shape[0]) != int(len(X_test)) and int(arr.shape[1]) == int(len(X_test)):
            arr = arr.T
        return arr
    except Exception:
        pass
    try:
        booster = getattr(model, "booster_", None)
        if booster is None:
            return None
        leaf_mat = booster.predict(X_test, pred_leaf=True)
        arr = np.asarray(leaf_mat)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim == 2 and int(arr.shape[0]) != int(len(X_test)) and int(arr.shape[1]) == int(len(X_test)):
            arr = arr.T
        return arr
    except Exception:
        return None


def _compute_future_volatility(close: pd.Series, window: int) -> pd.Series:
    log_ret = np.log(close).diff()
    fut = log_ret.shift(-1)
    return fut.rolling(window=window, min_periods=max(5, window // 4)).std().shift(-(window - 1))


def _compute_future_return(close: pd.Series, horizon: int) -> pd.Series:
    return (close.shift(-horizon) / (close + 1e-12) - 1.0).astype(float)


def _compute_trend_efficiency(close: pd.Series, window: int) -> pd.Series:
    fut_signal = (close.shift(-window) - close).abs()
    noise = close.diff().abs().shift(-1)
    fut_noise = noise.rolling(window=window, min_periods=max(5, window // 4)).sum().shift(-(window - 1))
    return (fut_signal / (fut_noise + 1e-12)).clip(lower=0.0, upper=1.0)


def _compute_future_liquidity_z(volume: pd.Series, window: int, min_periods: int = 200) -> pd.Series:
    fut_mean = volume.shift(-1).rolling(window=window, min_periods=max(5, window // 4)).mean().shift(-(window - 1))
    liq_raw = np.log(fut_mean.replace(0.0, np.nan))

    mean_past = liq_raw.expanding(min_periods=min_periods).mean().shift(1)
    std_past = liq_raw.expanding(min_periods=min_periods).std().shift(1)
    return ((liq_raw - mean_past) / (std_past + 1e-12)).clip(lower=-10.0, upper=10.0)


def _compute_future_liquidity_z_rolling(
    volume: pd.Series,
    *,
    window: int,
    norm_window: int,
    min_periods: int = 200,
) -> pd.Series:
    fut_mean = volume.shift(-1).rolling(window=window, min_periods=max(5, window // 4)).mean().shift(-(window - 1))
    liq_raw = np.log(fut_mean.replace(0.0, np.nan))
    norm_window = int(max(10, norm_window))
    mean_past = liq_raw.rolling(window=norm_window, min_periods=min_periods).mean().shift(1)
    std_past = liq_raw.rolling(window=norm_window, min_periods=min_periods).std().shift(1)
    return ((liq_raw - mean_past) / (std_past + 1e-12)).clip(lower=-10.0, upper=10.0)


def _compute_future_range_pct(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int,
) -> pd.Series:
    h = pd.to_numeric(high, errors="coerce")
    l = pd.to_numeric(low, errors="coerce")
    c = pd.to_numeric(close, errors="coerce")
    fut_high = h.shift(-1).rolling(window=window, min_periods=max(5, window // 4)).max().shift(-(window - 1))
    fut_low = l.shift(-1).rolling(window=window, min_periods=max(5, window // 4)).min().shift(-(window - 1))
    return ((fut_high - fut_low) / (c.abs() + 1e-12)).astype(float)


def _compute_future_min_return(close: pd.Series, horizon: int) -> pd.Series:
    c = pd.to_numeric(close, errors="coerce")
    fut_min = c.shift(-1).rolling(window=horizon, min_periods=max(5, horizon // 4)).min().shift(-(horizon - 1))
    return (fut_min / (c + 1e-12) - 1.0).astype(float)


def _compute_future_max_return(close: pd.Series, horizon: int) -> pd.Series:
    c = pd.to_numeric(close, errors="coerce")
    fut_max = c.shift(-1).rolling(window=horizon, min_periods=max(5, horizon // 4)).max().shift(-(horizon - 1))
    return (fut_max / (c + 1e-12) - 1.0).astype(float)


def _compute_future_min_bar_return(close: pd.Series, horizon: int) -> pd.Series:
    c = pd.to_numeric(close, errors="coerce")
    bar_ret = c.pct_change().shift(-1)
    return bar_ret.rolling(window=horizon, min_periods=max(5, horizon // 4)).min().shift(-(horizon - 1)).astype(float)


def _compute_future_max_abs_bar_return(close: pd.Series, horizon: int) -> pd.Series:
    c = pd.to_numeric(close, errors="coerce")
    bar_ret = c.pct_change().shift(-1)
    abs_ret = bar_ret.abs()
    return abs_ret.rolling(window=horizon, min_periods=max(5, horizon // 4)).max().shift(-(horizon - 1)).astype(float)


def _compute_future_vol_of_vol(close: pd.Series, vol_window_short: int, vol_of_vol_window: int) -> pd.Series:
    vol = _compute_future_volatility(close, window=int(vol_window_short))
    fut_vol = vol.shift(-1)
    return fut_vol.rolling(window=int(vol_of_vol_window), min_periods=max(5, int(vol_of_vol_window) // 4)).std().shift(
        -(int(vol_of_vol_window) - 1)
    )


def _compute_memory_autocorr(returns: pd.Series, window: int) -> pd.Series:
    shifted = returns.shift(1)
    return returns.rolling(window=window, min_periods=max(10, window // 4)).corr(shifted)


def _compute_true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _safe_log_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    ratio = (a / (b.replace(0.0, np.nan))).replace([np.inf, -np.inf], np.nan)
    return np.log(ratio)


def _binary_entropy_from_prob(p: pd.Series) -> pd.Series:
    p = pd.to_numeric(p, errors="coerce").astype(float)
    p = p.clip(lower=1e-12, upper=1.0 - 1e-12)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)).astype(float)


def build_regime_embedding_features(market_data: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    close_col = str(cfg.get("close_col", "close"))
    open_col = str(cfg.get("open_col", "open"))
    high_col = str(cfg.get("high_col", "high"))
    low_col = str(cfg.get("low_col", "low"))
    volume_col = str(cfg.get("volume_col", "volume"))

    if close_col not in market_data.columns:
        raise KeyError(f"close_col '{close_col}' missing from market_data")

    close = pd.to_numeric(market_data[close_col], errors="coerce")
    high = pd.to_numeric(market_data[high_col], errors="coerce") if high_col in market_data.columns else close
    low = pd.to_numeric(market_data[low_col], errors="coerce") if low_col in market_data.columns else close
    open_px = (
        pd.to_numeric(market_data[open_col], errors="coerce")
        if open_col in market_data.columns
        else close
    )

    out = pd.DataFrame(index=market_data.index)

    log_ret = np.log(close).diff()
    ret = close.pct_change()

    out["reg_ohlcv__log_ret"] = log_ret
    out["reg_ohlcv__ret"] = ret

    # Legacy-compatible volatility windows (used by create_meta_features)
    try:
        out["reg_ohlcv__volatility_1h"] = log_ret.rolling(window=4, min_periods=2).std()
        out["reg_ohlcv__volatility_4h"] = log_ret.rolling(window=16, min_periods=4).std()
        out["reg_ohlcv__volatility_1d"] = log_ret.rolling(window=96, min_periods=16).std()
        out["reg_ohlcv__vol_of_vol_1h20"] = out["reg_ohlcv__volatility_1h"].rolling(window=20, min_periods=5).std()

        vol_baseline = out["reg_ohlcv__volatility_1d"].rolling(96, min_periods=24).mean()
        out["reg_ohlcv__vol_ratio"] = out["reg_ohlcv__volatility_1d"] / (vol_baseline + 1e-8)
    except Exception:
        out["reg_ohlcv__volatility_1h"] = np.nan
        out["reg_ohlcv__volatility_4h"] = np.nan
        out["reg_ohlcv__volatility_1d"] = np.nan
        out["reg_ohlcv__vol_of_vol_1h20"] = np.nan
        out["reg_ohlcv__vol_ratio"] = np.nan

    # Legacy-compatible volatility regime labels (categorical + dummies)
    try:
        vol_for_regime = out["reg_ohlcv__volatility_1d"].copy()
        vol_non_null = pd.to_numeric(vol_for_regime, errors="coerce").dropna()
        if len(vol_non_null) >= 3:
            split_idx = max(1, int(len(vol_non_null) * 0.7))
            vol_train = vol_non_null.iloc[:split_idx]
            q1, q2 = vol_train.quantile([1 / 3, 2 / 3])
            bins = [-np.inf, q1, q2, np.inf]
            labels = ["low", "medium", "high"]
            regime_full = pd.cut(vol_for_regime, bins=bins, labels=labels)
            out["reg_ohlcv__volatility_regime"] = regime_full
            d = pd.get_dummies(regime_full, prefix="reg_ohlcv__vol_regime", drop_first=True)
            out = out.join(d)
            out = out.drop(columns=["reg_ohlcv__volatility_regime"], errors="ignore")
        else:
            out["reg_ohlcv__volatility_regime"] = pd.Series(index=out.index, dtype="category")
            out["reg_ohlcv__vol_regime_medium"] = 0.0
            out["reg_ohlcv__vol_regime_high"] = 0.0
    except Exception:
        out["reg_ohlcv__volatility_regime"] = pd.Series(index=out.index, dtype="category")
        out["reg_ohlcv__vol_regime_medium"] = 0.0
        out["reg_ohlcv__vol_regime_high"] = 0.0

    out = out.drop(columns=["reg_ohlcv__volatility_regime"], errors="ignore")

    windows = cfg.get("vol_windows")
    if not isinstance(windows, (list, tuple)) or not windows:
        windows = [8, 24, 96]
    windows = [int(w) for w in windows if int(w) > 1]
    for w in windows:
        out[f"reg_ohlcv__rv_std_logret_w{w}"] = log_ret.rolling(window=w, min_periods=max(5, w // 4)).std()
        out[f"reg_ohlcv__rv_std_ret_w{w}"] = ret.rolling(window=w, min_periods=max(5, w // 4)).std()

    trend_windows = cfg.get("trend_windows")
    if not isinstance(trend_windows, (list, tuple)) or not trend_windows:
        trend_windows = [16, 32, 64]
    trend_windows = [int(w) for w in trend_windows if int(w) > 1]
    for w in trend_windows:
        out[f"reg_ohlcv__log_ret_sum_w{w}"] = log_ret.rolling(window=w, min_periods=max(5, w // 4)).sum()
        out[f"reg_ohlcv__ret_sum_w{w}"] = ret.rolling(window=w, min_periods=max(5, w // 4)).sum()
        out[f"reg_ohlcv__ret_mean_w{w}"] = ret.rolling(window=w, min_periods=max(5, w // 4)).mean()

    ema_fast_w = int(cfg.get("ema_fast_window", 16))
    ema_slow_w = int(cfg.get("ema_slow_window", 64))
    ema_fast = close.ewm(span=ema_fast_w, adjust=False, min_periods=max(5, ema_fast_w // 4)).mean()
    ema_slow = close.ewm(span=ema_slow_w, adjust=False, min_periods=max(5, ema_slow_w // 4)).mean()
    out["reg_ohlcv__ema_fast_slow_ratio"] = (ema_fast / (ema_slow + 1e-12) - 1.0).astype(float)
    out["reg_ohlcv__ema_fast_slope"] = (ema_fast.diff() / (ema_fast.abs() + 1e-12)).astype(float)
    out["reg_ohlcv__ema_slow_slope"] = (ema_slow.diff() / (ema_slow.abs() + 1e-12)).astype(float)

    eff_w = int(cfg.get("efficiency_window", 32))
    path = close.diff().abs()
    denom = path.rolling(window=eff_w, min_periods=max(5, eff_w // 4)).sum()
    num = (close - close.shift(eff_w)).abs()
    out["reg_ohlcv__efficiency_ratio"] = (num / (denom + 1e-12)).clip(lower=0.0, upper=1.0).astype(float)

    tr = _compute_true_range(high=high, low=low, close=close)
    atr_w = int(cfg.get("atr_window", 14))
    atr = tr.rolling(window=atr_w, min_periods=max(5, atr_w // 4)).mean()
    out["reg_ohlcv__atr_pct"] = atr / (close.abs() + 1e-12)
    try:
        out["reg_ohlcv__tr_atr"] = atr_normalize(tr, high=high, low=low, close=close, window=atr_w)
    except Exception:
        out["reg_ohlcv__tr_atr"] = (tr / (atr + 1e-12)).replace([np.inf, -np.inf], np.nan)

    log_hl = _safe_log_ratio(high, low)
    pk_w = int(cfg.get("parkinson_window", 48))
    pk_mean = (log_hl**2).rolling(window=pk_w, min_periods=max(10, pk_w // 4)).mean()
    out["reg_ohlcv__parkinson_vol"] = np.sqrt(pk_mean / (4.0 * np.log(2.0) + 1e-12))

    log_co = _safe_log_ratio(close, open_px)
    gk_w = int(cfg.get("gk_window", 48))
    gk_term = 0.5 * (log_hl**2) - (2.0 * np.log(2.0) - 1.0) * (log_co**2)
    gk_mean = gk_term.rolling(window=gk_w, min_periods=max(10, gk_w // 4)).mean()
    out["reg_ohlcv__gk_vol"] = np.sqrt(gk_mean.clip(lower=0.0))

    rng = (high - low).replace(0.0, np.nan)
    body = (close - open_px).abs()
    upper = high - pd.concat([open_px, close], axis=1).max(axis=1)
    lower = pd.concat([open_px, close], axis=1).min(axis=1) - low
    out["reg_ohlcv__body_frac"] = (body / rng).replace([np.inf, -np.inf], np.nan)
    out["reg_ohlcv__upper_wick_frac"] = (upper / rng).replace([np.inf, -np.inf], np.nan)
    out["reg_ohlcv__lower_wick_frac"] = (lower / rng).replace([np.inf, -np.inf], np.nan)
    out["reg_ohlcv__range_pct"] = ((high - low) / (close.abs() + 1e-12)).astype(float)

    gap = (open_px - close.shift(1)).astype(float)
    out["reg_ohlcv__gap_abs_atr"] = (gap.abs() / (atr + 1e-12)).replace([np.inf, -np.inf], np.nan)
    out["reg_ohlcv__gap_signed_atr"] = (gap / (atr + 1e-12)).replace([np.inf, -np.inf], np.nan)

    dd_w = int(cfg.get("drawdown_window", 192))
    roll_max = close.rolling(window=dd_w, min_periods=max(10, dd_w // 6)).max()
    out["reg_ohlcv__drawdown"] = (close / (roll_max + 1e-12) - 1.0).astype(float)

    dvol_w = int(cfg.get("downside_vol_window", 96))
    neg_ret = ret.where(ret < 0.0, 0.0)
    out["reg_ohlcv__downside_vol"] = neg_ret.rolling(window=dvol_w, min_periods=max(10, dvol_w // 4)).std()
    out["reg_ohlcv__downside_semivar"] = (neg_ret**2).rolling(window=dvol_w, min_periods=max(10, dvol_w // 4)).mean()

    vov_base_w = int(cfg.get("vol_of_vol_base_window", 24))
    vov_w = int(cfg.get("vol_of_vol_window", 192))
    base_vol = log_ret.rolling(window=vov_base_w, min_periods=max(5, vov_base_w // 4)).std()
    out["reg_ohlcv__vol_of_vol"] = base_vol.rolling(window=vov_w, min_periods=max(10, vov_w // 6)).std()

    try:
        vol_trend_cfg = cfg.get("vol_trend")
        if not isinstance(vol_trend_cfg, dict):
            vol_trend_cfg = {}
        vol_trend_windows = vol_trend_cfg.get("windows")
        if not isinstance(vol_trend_windows, (list, tuple)) or not vol_trend_windows:
            vol_trend_windows = [48, 96]
        vol_trend_windows = [int(w) for w in vol_trend_windows if int(w) >= 10]
    except Exception:
        vol_trend_windows = [48, 96]

    try:
        vol_state = pd.to_numeric(out.get("reg_ohlcv__volatility_1d"), errors="coerce")
        for w in vol_trend_windows:
            vol_ewm = vol_state.ewm(span=w, adjust=False, min_periods=max(5, w // 4)).mean()
            out[f"reg_ohlcv__vol_ewm_slope_w{w}"] = (vol_ewm.diff() / (vol_ewm.abs() + 1e-12)).astype(float)
            out[f"reg_ohlcv__vol_pct_chg_w{w}"] = vol_state.pct_change(periods=max(1, w // 4)).replace([np.inf, -np.inf], np.nan)
    except Exception:
        for w in vol_trend_windows:
            out[f"reg_ohlcv__vol_ewm_slope_w{w}"] = np.nan
            out[f"reg_ohlcv__vol_pct_chg_w{w}"] = np.nan

    try:
        chop_cfg = cfg.get("choppiness")
        if not isinstance(chop_cfg, dict):
            chop_cfg = {}
        chop_windows = chop_cfg.get("windows")
        if not isinstance(chop_windows, (list, tuple)) or not chop_windows:
            chop_windows = [48, 96]
        chop_windows = [int(w) for w in chop_windows if int(w) >= 10]
        for w in chop_windows:
            sum_tr = tr.rolling(window=w, min_periods=max(10, w // 4)).sum()
            rng_w = (close.rolling(window=w, min_periods=max(10, w // 4)).max() - close.rolling(window=w, min_periods=max(10, w // 4)).min())
            chop = 100.0 * (np.log10((sum_tr + 1e-12) / (rng_w.abs() + 1e-12))) / (np.log10(float(w)) + 1e-12)
            out[f"reg_ohlcv__choppiness_w{w}"] = chop.replace([np.inf, -np.inf], np.nan)
    except Exception:
        try:
            for w in list(chop_windows):
                out[f"reg_ohlcv__choppiness_w{w}"] = np.nan
        except Exception:
            pass

    try:
        tail_cfg = cfg.get("tail_risk")
        if not isinstance(tail_cfg, dict):
            tail_cfg = {}
        tail_windows = tail_cfg.get("windows")
        if not isinstance(tail_windows, (list, tuple)) or not tail_windows:
            tail_windows = [96, 192]
        tail_windows = [int(w) for w in tail_windows if int(w) >= 20]
        q_lo = float(tail_cfg.get("q_lo", 0.05))
        q_hi = float(tail_cfg.get("q_hi", 0.95))
        for w in tail_windows:
            r = pd.to_numeric(ret, errors="coerce")
            ql = r.rolling(window=w, min_periods=max(10, w // 4)).quantile(q_lo)
            qh = r.rolling(window=w, min_periods=max(10, w // 4)).quantile(q_hi)
            out[f"reg_ohlcv__ret_q{int(q_lo*100):02d}_w{w}"] = ql
            out[f"reg_ohlcv__ret_q{int(q_hi*100):02d}_w{w}"] = qh
            out[f"reg_ohlcv__tail_asym_qratio_w{w}"] = (ql.abs() / (qh.abs() + 1e-12)).replace([np.inf, -np.inf], np.nan)
            dd = (close / (close.rolling(window=w, min_periods=max(10, w // 4)).max() + 1e-12) - 1.0).astype(float)
            out[f"reg_ohlcv__max_drawdown_w{w}"] = dd.rolling(window=w, min_periods=max(10, w // 4)).min()
    except Exception:
        try:
            for w in list(tail_windows):
                out[f"reg_ohlcv__tail_asym_qratio_w{w}"] = np.nan
                out[f"reg_ohlcv__max_drawdown_w{w}"] = np.nan
        except Exception:
            pass

    jump_w = int(cfg.get("jump_window", 96))
    jump_mult = float(cfg.get("jump_sigma_mult", 3.0))
    rv = ret.rolling(window=jump_w, min_periods=max(10, jump_w // 4)).std()
    jump = (ret.abs() > (jump_mult * (rv + 1e-12))).astype(float)
    out["reg_ohlcv__jump_rate"] = jump.rolling(window=jump_w, min_periods=max(10, jump_w // 4)).mean()
    out["reg_ohlcv__jump_abs_ret_mean"] = (ret.abs() * jump).rolling(window=jump_w, min_periods=max(10, jump_w // 4)).mean()

    zero_thr = float(cfg.get("zero_return_threshold", 1e-8))
    zero_flag = (ret.abs() <= zero_thr).astype(float)
    zr_w = int(cfg.get("zero_return_window", 96))
    out["reg_ohlcv__zero_ret_frac"] = zero_flag.rolling(window=zr_w, min_periods=max(10, zr_w // 4)).mean()

    skew_w = int(cfg.get("moment_window", 96))
    out["reg_ohlcv__ret_skew"] = ret.rolling(window=skew_w, min_periods=max(10, skew_w // 4)).skew()
    out["reg_ohlcv__ret_kurt"] = ret.rolling(window=skew_w, min_periods=max(10, skew_w // 4)).kurt()

    ac_cfg = cfg.get("autocorr")
    if not isinstance(ac_cfg, dict):
        ac_cfg = {}
    ac_windows = ac_cfg.get("windows")
    if not isinstance(ac_windows, (list, tuple)) or not ac_windows:
        ac_windows = [48, 96]
    ac_lags = ac_cfg.get("lags")
    if not isinstance(ac_lags, (list, tuple)) or not ac_lags:
        ac_lags = [1, 2, 5]
    for w in [int(x) for x in ac_windows if int(x) > 5]:
        for lag in [int(x) for x in ac_lags if int(x) > 0]:
            out[f"reg_ohlcv__ret_autocorr_l{lag}_w{w}"] = ret.rolling(
                window=w, min_periods=max(10, w // 4)
            ).corr(ret.shift(lag))

    enable_persistence = bool(cfg.get("enable_persistence_features", True))
    if enable_persistence:
        vol_cluster_cfg = cfg.get("vol_clustering")
        if not isinstance(vol_cluster_cfg, dict):
            vol_cluster_cfg = {}
        vc_windows = vol_cluster_cfg.get("windows")
        if not isinstance(vc_windows, (list, tuple)) or not vc_windows:
            vc_windows = [48, 96]
        vc_windows = [int(w) for w in vc_windows if int(w) >= 10]
        abs_ret = ret.abs()
        sq_ret = (ret**2).astype(float)
        for w in vc_windows:
            out[f"reg_ohlcv__absret_autocorr_l1_w{w}"] = abs_ret.rolling(window=w, min_periods=max(10, w // 4)).corr(
                abs_ret.shift(1)
            )
            out[f"reg_ohlcv__sqret_autocorr_l1_w{w}"] = sq_ret.rolling(window=w, min_periods=max(10, w // 4)).corr(
                sq_ret.shift(1)
            )

        dir_cfg = cfg.get("directional_persistence")
        if not isinstance(dir_cfg, dict):
            dir_cfg = {}
        dir_windows = dir_cfg.get("windows")
        if not isinstance(dir_windows, (list, tuple)) or not dir_windows:
            dir_windows = [48, 96]
        dir_windows = [int(w) for w in dir_windows if int(w) >= 10]
        up = (ret > 0.0).astype(float)
        for w in dir_windows:
            p_up = up.rolling(window=w, min_periods=max(10, w // 4)).mean()
            out[f"reg_ohlcv__p_up_w{w}"] = p_up
            out[f"reg_ohlcv__dir_entropy_w{w}"] = _binary_entropy_from_prob(p_up)

    if volume_col in market_data.columns:
        vol = pd.to_numeric(market_data[volume_col], errors="coerce")
        turnover = (close * vol).astype(float)
        norm_w = int(cfg.get("volume_norm_window", 500))
        try:
            out["reg_ohlcv__volume_log1p_z"] = log1p_zscore_normalize(vol, window=norm_w, min_periods=1, ddof=1)
        except Exception:
            out["reg_ohlcv__volume_log1p_z"] = np.nan
        try:
            out["reg_ohlcv__turnover_log1p_z"] = log1p_zscore_normalize(turnover, window=norm_w, min_periods=1, ddof=1)
        except Exception:
            out["reg_ohlcv__turnover_log1p_z"] = np.nan

        amihud_w = int(cfg.get("amihud_window", 96))
        illiq = (ret.abs() / (turnover.abs() + 1e-12)).replace([np.inf, -np.inf], np.nan)
        out["reg_ohlcv__amihud_illiq"] = illiq.rolling(window=amihud_w, min_periods=max(10, amihud_w // 4)).mean()

        signed_vol = (np.sign(ret.fillna(0.0)) * vol).astype(float)
        kyle_w = int(cfg.get("kyle_window", 96))
        cov = ret.rolling(window=kyle_w, min_periods=max(10, kyle_w // 4)).cov(signed_vol)
        var = signed_vol.rolling(window=kyle_w, min_periods=max(10, kyle_w // 4)).var()
        out["reg_ohlcv__kyle_lambda_proxy"] = (cov / (var + 1e-12)).replace([np.inf, -np.inf], np.nan)

        try:
            coupling_cfg = cfg.get("coupling")
            if not isinstance(coupling_cfg, dict):
                coupling_cfg = {}
            coupling_windows = coupling_cfg.get("windows")
            if not isinstance(coupling_windows, (list, tuple)) or not coupling_windows:
                coupling_windows = [48, 96]
            coupling_windows = [int(w) for w in coupling_windows if int(w) >= 10]

            vol_log = np.log1p(vol).replace([np.inf, -np.inf], np.nan)
            vol_chg = vol.pct_change().replace([np.inf, -np.inf], np.nan)
            abs_r = pd.to_numeric(ret, errors="coerce").abs()
            for w in coupling_windows:
                out[f"reg_ohlcv__corr_absret_logvol_w{w}"] = abs_r.rolling(window=w, min_periods=max(10, w // 4)).corr(
                    vol_log
                )
                out[f"reg_ohlcv__corr_ret_logvol_w{w}"] = pd.to_numeric(ret, errors="coerce").rolling(
                    window=w, min_periods=max(10, w // 4)
                ).corr(vol_log)
                out[f"reg_ohlcv__corr_ret_volchg_w{w}"] = pd.to_numeric(ret, errors="coerce").rolling(
                    window=w, min_periods=max(10, w // 4)
                ).corr(vol_chg)

                out[f"reg_ohlcv__corr_volatility_logvol_w{w}"] = pd.to_numeric(base_vol, errors="coerce").rolling(
                    window=w, min_periods=max(10, w // 4)
                ).corr(vol_log)
        except Exception:
            try:
                for w in list(coupling_windows):
                    out[f"reg_ohlcv__corr_absret_logvol_w{w}"] = np.nan
                    out[f"reg_ohlcv__corr_ret_logvol_w{w}"] = np.nan
                    out[f"reg_ohlcv__corr_ret_volchg_w{w}"] = np.nan
                    out[f"reg_ohlcv__corr_volatility_logvol_w{w}"] = np.nan
            except Exception:
                pass

        if enable_persistence:
            liq_cfg = cfg.get("liquidity_persistence")
            if not isinstance(liq_cfg, dict):
                liq_cfg = {}
            liq_windows = liq_cfg.get("windows")
            if not isinstance(liq_windows, (list, tuple)) or not liq_windows:
                liq_windows = [48, 96]
            liq_windows = [int(w) for w in liq_windows if int(w) >= 10]
            vol_change = vol.pct_change().replace([np.inf, -np.inf], np.nan)
            to_change = turnover.pct_change().replace([np.inf, -np.inf], np.nan)
            for w in liq_windows:
                out[f"reg_ohlcv__volume_autocorr_l1_w{w}"] = vol.rolling(window=w, min_periods=max(10, w // 4)).corr(
                    vol.shift(1)
                )
                out[f"reg_ohlcv__turnover_autocorr_l1_w{w}"] = turnover.rolling(
                    window=w, min_periods=max(10, w // 4)
                ).corr(turnover.shift(1))
                out[f"reg_ohlcv__volume_chg_std_w{w}"] = vol_change.rolling(window=w, min_periods=max(10, w // 4)).std()
                out[f"reg_ohlcv__turnover_chg_std_w{w}"] = to_change.rolling(window=w, min_periods=max(10, w // 4)).std()

        # Legacy-compatible OFI proxy block
        try:
            close_in_range = (close - low) / (high - low + 1e-8)
            price_direction = np.sign(close - open_px)
            signed_volume = vol * price_direction
            cvd_proxy = signed_volume.cumsum()
            cvd_normalized = (cvd_proxy - cvd_proxy.rolling(96, min_periods=24).mean()) / (cvd_proxy.rolling(96, min_periods=24).std() + 1e-8)
            out["reg_ohlcv__cvd_proxy"] = cvd_normalized

            volume_pressure = (close_in_range - 0.5) * vol
            out["reg_ohlcv__volume_pressure"] = volume_pressure.ewm(span=20, adjust=False, min_periods=5).mean()

            upper_wick = high - pd.concat([open_px, close], axis=1).max(axis=1)
            lower_wick = pd.concat([open_px, close], axis=1).min(axis=1) - low
            total_range = high - low + 1e-8
            supply_rejection = (upper_wick / total_range) * vol
            demand_rejection = (lower_wick / total_range) * vol
            ofi_proxy = (demand_rejection - supply_rejection).rolling(20, min_periods=5).sum()
            out["reg_ohlcv__ofi_proxy"] = ofi_proxy / (ofi_proxy.rolling(96, min_periods=24).std() + 1e-8)

            buy_volume = vol * close_in_range
            sell_volume = vol * (1.0 - close_in_range)
            volume_imbalance = (buy_volume - sell_volume) / (vol + 1e-8)
            out["reg_ohlcv__volume_imbalance"] = volume_imbalance.ewm(span=20, adjust=False, min_periods=5).mean()

            is_at_extreme = (close_in_range < 0.2) | (close_in_range > 0.8)
            extreme_volume = vol.where(is_at_extreme, 0.0).rolling(20, min_periods=5).sum()
            total_volume = vol.rolling(20, min_periods=5).sum()
            out["reg_ohlcv__absorption_ratio"] = extreme_volume / (total_volume + 1e-8)

            out["reg_ohlcv__trade_aggressor_ratio"] = close_in_range.ewm(span=20, adjust=False, min_periods=5).mean()

            prev_close = close.shift(1)
            gap_raw = open_px - prev_close
            out["reg_ohlcv__liquidity_gap_up"] = np.maximum(gap_raw, 0.0) / (prev_close + 1e-8)
            out["reg_ohlcv__liquidity_gap_down"] = np.maximum(-gap_raw, 0.0) / (prev_close + 1e-8)
            out["reg_ohlcv__liquidity_gap_abs"] = gap_raw.abs() / (atr + 1e-8)
        except Exception:
            out["reg_ohlcv__cvd_proxy"] = 0.0
            out["reg_ohlcv__volume_pressure"] = 0.0
            out["reg_ohlcv__ofi_proxy"] = 0.0
            out["reg_ohlcv__volume_imbalance"] = 0.0
            out["reg_ohlcv__absorption_ratio"] = 0.0
            out["reg_ohlcv__trade_aggressor_ratio"] = 0.5
            out["reg_ohlcv__liquidity_gap_up"] = 0.0
            out["reg_ohlcv__liquidity_gap_down"] = 0.0
            out["reg_ohlcv__liquidity_gap_abs"] = 0.0
    else:
        out["reg_ohlcv__volume_log1p_z"] = np.nan
        out["reg_ohlcv__turnover_log1p_z"] = np.nan
        out["reg_ohlcv__amihud_illiq"] = np.nan
        out["reg_ohlcv__kyle_lambda_proxy"] = np.nan

    enable_transition = bool(cfg.get("enable_transition_features", True))
    if enable_transition:
        trans_cfg = cfg.get("transition")
        if not isinstance(trans_cfg, dict):
            trans_cfg = {}
        trans_windows = trans_cfg.get("windows")
        if not isinstance(trans_windows, (list, tuple)) or not trans_windows:
            trans_windows = [48, 96]
        trans_windows = [int(w) for w in trans_windows if int(w) >= 10]

        # Key state deltas (causal)
        try:
            out["reg_ohlcv__d_volatility_1d"] = pd.to_numeric(out.get("reg_ohlcv__volatility_1d"), errors="coerce").diff()
        except Exception:
            out["reg_ohlcv__d_volatility_1d"] = np.nan
        try:
            out["reg_ohlcv__d_vol_ratio"] = pd.to_numeric(out.get("reg_ohlcv__vol_ratio"), errors="coerce").diff()
        except Exception:
            out["reg_ohlcv__d_vol_ratio"] = np.nan

        # Shock / change intensity from state deltas
        dv = pd.to_numeric(out["reg_ohlcv__d_volatility_1d"], errors="coerce")
        dvr = pd.to_numeric(out["reg_ohlcv__d_vol_ratio"], errors="coerce")
        for w in trans_windows:
            dv_sigma = dv.rolling(window=w, min_periods=max(10, w // 4)).std()
            dvr_sigma = dvr.rolling(window=w, min_periods=max(10, w // 4)).std()
            out[f"reg_ohlcv__vol_change_rate_w{w}"] = (dv.abs() > (2.0 * (dv_sigma + 1e-12))).astype(float).rolling(
                window=w, min_periods=max(10, w // 4)
            ).mean()
            out[f"reg_ohlcv__vol_ratio_change_rate_w{w}"] = (dvr.abs() > (2.0 * (dvr_sigma + 1e-12))).astype(float).rolling(
                window=w, min_periods=max(10, w // 4)
            ).mean()

        # Short/long volatility ratio features (if windows exist)
        try:
            w_short = int(trans_cfg.get("ratio_short_window", 8))
            w_long = int(trans_cfg.get("ratio_long_window", 96))
            c_short = f"reg_ohlcv__rv_std_logret_w{w_short}"
            c_long = f"reg_ohlcv__rv_std_logret_w{w_long}"
            if c_short in out.columns and c_long in out.columns:
                out[f"reg_ohlcv__rv_ratio_logret_w{w_short}_w{w_long}"] = (
                    pd.to_numeric(out[c_short], errors="coerce") / (pd.to_numeric(out[c_long], errors="coerce") + 1e-12)
                ).astype(float)
        except Exception:
            pass

    enable_complexity = bool(cfg.get("enable_complexity_features", False))
    if enable_complexity:
        comp_cfg = cfg.get("complexity")
        if not isinstance(comp_cfg, dict):
            comp_cfg = {}
        ent_windows = comp_cfg.get("entropy_windows")
        if not isinstance(ent_windows, (list, tuple)) or not ent_windows:
            ent_windows = [96, 192]
        ent_windows = [int(w) for w in ent_windows if int(w) >= 20]

        up = (ret > 0.0).astype(float)
        for w in ent_windows:
            p_up = up.rolling(window=w, min_periods=max(10, w // 4)).mean()
            out[f"reg_ohlcv__entropy_sign_w{w}"] = _binary_entropy_from_prob(p_up)

        if bool(comp_cfg.get("enable_hurst", False)):
            hurst_windows = comp_cfg.get("hurst_windows")
            if not isinstance(hurst_windows, (list, tuple)) or not hurst_windows:
                hurst_windows = [96, 192]
            hurst_windows = [int(w) for w in hurst_windows if int(w) >= 30]

            def _hurst_rs(x: np.ndarray) -> float:
                x = np.asarray(x, dtype=float)
                x = x[np.isfinite(x)]
                n = int(x.size)
                if n < 20:
                    return float("nan")
                mu = float(np.mean(x))
                y = np.cumsum(x - mu)
                r = float(np.max(y) - np.min(y))
                s = float(np.std(x, ddof=0))
                if not np.isfinite(r) or not np.isfinite(s) or s <= 1e-12 or r <= 1e-12:
                    return float("nan")
                return float(np.log(r / s) / np.log(float(n)))

            for w in hurst_windows:
                out[f"reg_ohlcv__hurst_rs_w{w}"] = ret.rolling(window=w, min_periods=max(20, w // 2)).apply(
                    lambda a: _hurst_rs(np.asarray(a, dtype=float)), raw=False
                )

    # Legacy-compatible time-of-day / seasonality flags
    try:
        if isinstance(out.index, pd.DatetimeIndex):
            hour_arr = out.index.hour.to_numpy()
            dow_arr = out.index.dayofweek.to_numpy()
            out["reg_ohlcv__hour"] = hour_arr.astype(float)
            out["reg_ohlcv__day_of_week"] = dow_arr.astype(float)
            out["reg_ohlcv__hour_sin"] = np.sin(2 * np.pi * hour_arr / 24.0)
            out["reg_ohlcv__hour_cos"] = np.cos(2 * np.pi * hour_arr / 24.0)
            out["reg_ohlcv__is_good_hour"] = np.isin(hour_arr, [3, 5, 10]).astype(float)
            out["reg_ohlcv__is_bad_hour"] = np.isin(hour_arr, [0, 13, 19]).astype(float)
            out["reg_ohlcv__is_sunday"] = (dow_arr == 6).astype(float)
        else:
            out["reg_ohlcv__hour"] = 0.0
            out["reg_ohlcv__day_of_week"] = 0.0
            out["reg_ohlcv__hour_sin"] = 0.0
            out["reg_ohlcv__hour_cos"] = 1.0
            out["reg_ohlcv__is_good_hour"] = 0.0
            out["reg_ohlcv__is_bad_hour"] = 0.0
            out["reg_ohlcv__is_sunday"] = 0.0
    except Exception:
        out["reg_ohlcv__hour"] = 0.0
        out["reg_ohlcv__day_of_week"] = 0.0
        out["reg_ohlcv__hour_sin"] = 0.0
        out["reg_ohlcv__hour_cos"] = 1.0
        out["reg_ohlcv__is_good_hour"] = 0.0
        out["reg_ohlcv__is_bad_hour"] = 0.0
        out["reg_ohlcv__is_sunday"] = 0.0

    out = out.replace([np.inf, -np.inf], np.nan)

    try:
        drop_non_numeric = bool(cfg.get("drop_non_numeric", True))
    except Exception:
        drop_non_numeric = True

    if drop_non_numeric:
        non_numeric = [c for c in out.columns if not pd.api.types.is_numeric_dtype(out[c])]
        if non_numeric:
            out = out.drop(columns=non_numeric, errors="ignore")
    else:
        for c in list(out.columns):
            if not pd.api.types.is_numeric_dtype(out[c]):
                out[c] = pd.to_numeric(out[c], errors="coerce")

    try:
        win_w = int(cfg.get("winsor_zscore_window", 600))
        lo_q = float(cfg.get("winsor_lower_q", 0.01))
        hi_q = float(cfg.get("winsor_upper_q", 0.99))

        cols_other = [
            c
            for c in out.columns
            if c not in {"reg_ohlcv__volume_log1p_z", "reg_ohlcv__turnover_log1p_z"}
        ]
        if cols_other:
            normed = winsorized_zscore_normalize(
                out[cols_other].astype(float),
                window=win_w,
                ddof=0,
                lower_quantile=lo_q,
                upper_quantile=hi_q,
            )
            out[cols_other] = normed
    except Exception:
        try:
            cols_other = [
                c
                for c in out.columns
                if c not in {"reg_ohlcv__volume_log1p_z", "reg_ohlcv__turnover_log1p_z"}
            ]
            if cols_other:
                out[cols_other] = out[cols_other].apply(pd.to_numeric, errors="coerce")
        except Exception:
            pass

    return out


def _robust_scale_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if X_train is None or getattr(X_train, "empty", True):
        return X_train, X_test

    skip_low_cardinality = bool(cfg.get("skip_low_cardinality", True))
    low_cardinality_max = int(cfg.get("low_cardinality_max", 5))
    clip_abs = cfg.get("clip_abs")
    try:
        clip_abs = float(clip_abs) if clip_abs is not None else None
    except Exception:
        clip_abs = None

    Xtr = X_train.copy()
    Xte = X_test.copy() if X_test is not None else X_test

    for col in list(Xtr.columns):
        s = pd.to_numeric(Xtr[col], errors="coerce")
        if skip_low_cardinality:
            uniq = pd.unique(s.dropna())
            if uniq.size <= 2 and set(uniq.tolist()).issubset({0, 1}):
                continue
            if int(pd.Series(uniq).nunique()) <= low_cardinality_max:
                continue

        med = float(s.median()) if s.notna().any() else 0.0
        q75 = float(s.quantile(0.75)) if s.notna().any() else 0.0
        q25 = float(s.quantile(0.25)) if s.notna().any() else 0.0
        iqr = q75 - q25
        if not np.isfinite(iqr) or iqr <= 1e-12:
            iqr = float(s.std()) if s.notna().any() else 1.0
        if not np.isfinite(iqr) or iqr <= 1e-12:
            iqr = 1.0

        Xtr[col] = (pd.to_numeric(Xtr[col], errors="coerce") - med) / (iqr + 1e-12)
        if Xte is not None:
            Xte[col] = (pd.to_numeric(Xte[col], errors="coerce") - med) / (iqr + 1e-12)

        if clip_abs is not None and np.isfinite(clip_abs) and clip_abs > 0:
            Xtr[col] = Xtr[col].clip(lower=-clip_abs, upper=clip_abs)
            if Xte is not None:
                Xte[col] = Xte[col].clip(lower=-clip_abs, upper=clip_abs)

    return Xtr, Xte


def _add_missing_indicators(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not bool(cfg.get("enabled", True)):
        return X_train, X_test

    if X_train is None or getattr(X_train, "empty", True):
        return X_train, X_test

    cols = list(X_train.columns)
    ind_cols = []
    for col in cols:
        tr_na = bool(pd.to_numeric(X_train[col], errors="coerce").isna().any())
        te_na = False
        if X_test is not None and not getattr(X_test, "empty", True):
            te_na = bool(pd.to_numeric(X_test[col], errors="coerce").isna().any())
        if tr_na or te_na:
            ind_cols.append(col)

    if not ind_cols:
        return X_train, X_test

    Xtr = X_train.copy()
    Xte = X_test.copy() if X_test is not None else X_test

    for col in ind_cols:
        ind_name = f"isna__{col}"
        Xtr[ind_name] = pd.to_numeric(Xtr[col], errors="coerce").isna().astype(float)
        if Xte is not None:
            Xte[ind_name] = pd.to_numeric(Xte[col], errors="coerce").isna().astype(float)
    return Xtr, Xte


def compute_regime_targets_from_ohlcv(
    market_data: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    close_col = str(config.get("close_col", "close"))
    volume_col = str(config.get("volume_col", "volume"))
    high_col = str(config.get("high_col", "high"))
    low_col = str(config.get("low_col", "low"))

    if close_col not in market_data.columns:
        raise KeyError(f"close_col '{close_col}' missing from market_data")

    close = pd.to_numeric(market_data[close_col], errors="coerce")

    vol_window = int(config.get("volatility_window", 24))
    macro_trend_horizon = config.get("macro_trend_horizon")
    macro_trend_horizons = config.get("macro_trend_horizons")
    if isinstance(macro_trend_horizons, (list, tuple)) and macro_trend_horizons:
        macro_trend_horizons = [int(x) for x in macro_trend_horizons if x is not None]
    else:
        if macro_trend_horizon is None:
            macro_trend_horizons = [16, 32, 64]
        else:
            macro_trend_horizons = [int(macro_trend_horizon)]
    macro_trend_horizons = [int(h) for h in macro_trend_horizons if int(h) > 0]
    if not macro_trend_horizons:
        macro_trend_horizons = [32]

    efficiency_window = int(config.get("trend_efficiency_window", 16))
    liquidity_window = int(config.get("liquidity_window", 50))
    liquidity_norm = str(config.get("liquidity_norm", "rolling")).lower()
    liquidity_norm_window = int(config.get("liquidity_norm_window", 1000))
    liquidity_min_periods = int(config.get("liquidity_min_periods", 200))

    memory_window = int(config.get("memory_window", 40))
    range_window = int(config.get("range_window", 16))
    downside_horizon = int(config.get("downside_horizon", 32))
    vol_window_short = int(config.get("volatility_window_short", 8))
    vol_of_vol_window = int(config.get("vol_of_vol_window", 64))

    targets = pd.DataFrame(index=market_data.index)
    targets["regime_volatility"] = _compute_future_volatility(close, window=vol_window)

    for h in macro_trend_horizons:
        targets[f"regime_macro_trend_h{int(h)}"] = _compute_future_return(close, horizon=int(h))
    try:
        targets["regime_macro_trend"] = targets[f"regime_macro_trend_h{int(max(macro_trend_horizons))}"].astype(float)
    except Exception:
        targets["regime_macro_trend"] = _compute_future_return(close, horizon=int(max(macro_trend_horizons)))

    targets["regime_trend_efficiency"] = _compute_trend_efficiency(close, window=efficiency_window)

    if volume_col in market_data.columns:
        vol_series = pd.to_numeric(market_data[volume_col], errors="coerce")
        if liquidity_norm == "expanding":
            targets["regime_liquidity"] = _compute_future_liquidity_z(
                vol_series,
                window=liquidity_window,
                min_periods=liquidity_min_periods,
            )
        else:
            targets["regime_liquidity"] = _compute_future_liquidity_z_rolling(
                vol_series,
                window=liquidity_window,
                norm_window=liquidity_norm_window,
                min_periods=liquidity_min_periods,
            )
    else:
        targets["regime_liquidity"] = np.nan

    returns = close.pct_change().astype(float)
    targets["regime_memory"] = _compute_memory_autocorr(returns, window=memory_window)

    if high_col in market_data.columns and low_col in market_data.columns:
        targets["regime_future_range"] = _compute_future_range_pct(
            market_data[high_col],
            market_data[low_col],
            close,
            window=range_window,
        )
    else:
        targets["regime_future_range"] = np.nan

    targets["regime_downside_ae"] = _compute_future_min_return(close, horizon=downside_horizon)
    targets["regime_upside_ae"] = _compute_future_max_return(close, horizon=downside_horizon)
    targets["regime_tail_min_bar"] = _compute_future_min_bar_return(close, horizon=downside_horizon)
    targets["regime_jump_max_abs_bar"] = _compute_future_max_abs_bar_return(close, horizon=downside_horizon)
    targets["regime_vol_of_vol"] = _compute_future_vol_of_vol(
        close,
        vol_window_short=vol_window_short,
        vol_of_vol_window=vol_of_vol_window,
    )

    return targets


def _default_lgbm_params(config: dict, random_state: int, *, n_train_samples: Optional[int] = None) -> dict:
    num_leaves = int(config.get("num_leaves", 6))
    max_depth = int(config.get("max_depth", 3))
    n_estimators = int(config.get("n_estimators", 50))
    learning_rate = float(config.get("learning_rate", 0.1))

    min_data_in_leaf = int(config.get("min_data_in_leaf", 50))
    try:
        n_train = int(n_train_samples) if n_train_samples is not None else None
    except Exception:
        n_train = None
    if n_train is not None and n_train > 0:
        try:
            cap = int(max(10, round(0.05 * float(n_train))))
            min_data_in_leaf = int(max(10, min(int(min_data_in_leaf), int(cap), max(10, n_train - 1))))
        except Exception:
            min_data_in_leaf = int(max(10, min_data_in_leaf))

    min_gain_to_split = float(config.get("min_gain_to_split", 0.05))
    lambda_l1 = float(config.get("lambda_l1", 3.0))
    lambda_l2 = float(config.get("lambda_l2", 3.0))

    subsample = float(config.get("subsample", 0.9))
    colsample_bytree = float(config.get("colsample_bytree", 1.0))

    return {
        "n_estimators": n_estimators,
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "min_data_in_leaf": min_data_in_leaf,
        "min_gain_to_split": min_gain_to_split,
        "lambda_l1": lambda_l1,
        "lambda_l2": lambda_l2,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "random_state": random_state,
        "verbosity": -1,
        "n_jobs": -1,
    }


def _extract_leaf_paths(struct: Any, feature_names: Optional[List[str]] = None) -> Dict[int, List[Dict[str, Any]]]:
    if struct is None or not isinstance(struct, dict):
        return {}

    def _fname(split_feature: Any) -> str:
        try:
            if isinstance(split_feature, (int, np.integer)) and feature_names and 0 <= int(split_feature) < len(feature_names):
                return str(feature_names[int(split_feature)])
        except Exception:
            pass
        return str(split_feature)

    out: Dict[int, List[Dict[str, Any]]] = {}
    stack: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = [(struct, [])]
    while stack:
        node, path = stack.pop()
        if not isinstance(node, dict):
            continue
        if "leaf_index" in node:
            try:
                out[int(node["leaf_index"])] = list(path)
            except Exception:
                pass
            continue

        sf = node.get("split_feature")
        thr = node.get("threshold")
        dt = node.get("decision_type")
        default_left = node.get("default_left")
        feat = _fname(sf)

        left_child = node.get("left_child")
        right_child = node.get("right_child")
        left_rule = {"feature": feat, "threshold": thr, "decision": "<=", "decision_type": dt, "default_left": default_left}
        right_rule = {"feature": feat, "threshold": thr, "decision": ">", "decision_type": dt, "default_left": default_left}
        if isinstance(left_child, dict):
            stack.append((left_child, path + [left_rule]))
        if isinstance(right_child, dict):
            stack.append((right_child, path + [right_rule]))
    return out


def _leaf_summary_stats(
    *,
    raw_leaf_series: pd.Series,
    y_all: pd.Series,
    X_num: pd.DataFrame,
    kept_leaf_ids: Sequence[int],
    random_state: int,
    top_features_per_leaf: int,
    max_samples_per_leaf: int,
) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    try:
        global_mean = X_num.mean(numeric_only=True)
        global_std = X_num.std(numeric_only=True).replace(0.0, np.nan)
    except Exception:
        global_mean = None
        global_std = None

    rs = np.random.RandomState(int(random_state))

    for li in kept_leaf_ids:
        try:
            li_int = int(li)
        except Exception:
            continue
        try:
            mask = pd.to_numeric(raw_leaf_series, errors="coerce").astype(float).eq(float(li_int))
        except Exception:
            continue

        n_rows = int(mask.sum())
        y_leaf = pd.to_numeric(y_all.where(mask), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

        y_stats: Dict[str, Any] = {"n": int(len(y_leaf))}
        try:
            if len(y_leaf) > 0:
                y_stats.update(
                    {
                        "mean": float(y_leaf.mean()),
                        "std": float(y_leaf.std()),
                        "min": float(y_leaf.min()),
                        "p25": float(y_leaf.quantile(0.25)),
                        "p50": float(y_leaf.quantile(0.50)),
                        "p75": float(y_leaf.quantile(0.75)),
                        "max": float(y_leaf.max()),
                    }
                )
        except Exception:
            pass

        top_features = []
        try:
            if global_mean is not None and global_std is not None and n_rows > 0:
                idx = X_num.index[mask.values]
                if len(idx) > int(max_samples_per_leaf):
                    sel = rs.choice(np.arange(len(idx)), size=int(max_samples_per_leaf), replace=False)
                    idx = idx[np.asarray(sel, dtype=int)]
                X_leaf = X_num.reindex(idx)
                leaf_mean = X_leaf.mean(numeric_only=True)
                z = (leaf_mean - global_mean) / (global_std + 1e-12)
                z = z.replace([np.inf, -np.inf], np.nan).dropna()
                if not z.empty:
                    topk = z.abs().sort_values(ascending=False).head(int(max(0, top_features_per_leaf)))
                    for feat, z_val in topk.items():
                        try:
                            top_features.append(
                                {
                                    "feature": str(feat),
                                    "z_diff": float(z_val),
                                    "leaf_mean": float(leaf_mean.get(feat)),
                                    "global_mean": float(global_mean.get(feat)),
                                }
                            )
                        except Exception:
                            continue
        except Exception:
            top_features = []

        out[li_int] = {
            "n_rows": int(n_rows),
            "y": y_stats,
            "top_features": top_features,
        }

    return out


def _iter_anchored_windows(index: pd.Index, cfg: dict):
    if not isinstance(index, pd.DatetimeIndex):
        n = int(len(index))
        initial_train_frac = float(cfg.get("initial_train_frac", 0.4))
        step_frac = float(cfg.get("step_frac", 0.1))

        train_end = max(10, int(n * initial_train_frac))
        step = max(10, int(n * step_frac))

        while train_end < n - 1:
            test_end = min(n, train_end + step)
            yield slice(0, train_end), slice(train_end, test_end)
            train_end = test_end
        return

    initial_train_days = float(cfg.get("initial_train_days", 365.0 * 2.0))
    step_days = float(cfg.get("step_days", 90.0))

    try:
        span_days = float((index.max() - index.min()).total_seconds() / 86400.0)
    except Exception:
        span_days = None
    if span_days is None or span_days <= 0 or initial_train_days >= span_days:
        n = int(len(index))
        initial_train_frac = float(cfg.get("initial_train_frac", 0.4))
        step_frac = float(cfg.get("step_frac", 0.1))
        train_end = max(10, int(n * initial_train_frac))
        step = max(10, int(n * step_frac))
        while train_end < n - 1:
            test_end = min(n, train_end + step)
            yield slice(0, train_end), slice(train_end, test_end)
            train_end = test_end
        return

    if step_days <= 0:
        step_days = 1.0

    start = index.min()
    train_end_ts = start + pd.Timedelta(days=initial_train_days)

    while train_end_ts < index.max():
        test_end_ts = train_end_ts + pd.Timedelta(days=step_days)
        train_mask = index <= train_end_ts
        test_mask = (index > train_end_ts) & (index <= test_end_ts)
        yield train_mask, test_mask
        train_end_ts = test_end_ts


def _iter_cross_fit_splits(index: pd.Index, cfg: Dict[str, Any]):
    if not _SKLEARN_AVAILABLE or TimeSeriesSplit is None:
        raise ImportError("scikit-learn is required for cross_fit mode")
    n = int(len(index))
    n_splits = int(cfg.get("n_splits", 5))
    if n_splits < 2:
        n_splits = 2

    custom = cfg.get("cv_splits")
    if isinstance(custom, (list, tuple)) and custom:
        for pair in custom:
            try:
                tr, te = pair
                tr_idx = np.asarray(tr, dtype=int)
                te_idx = np.asarray(te, dtype=int)
                if tr_idx.size == 0 or te_idx.size == 0:
                    continue
                yield tr_idx, te_idx
            except Exception:
                continue
        return

    tss = TimeSeriesSplit(n_splits=n_splits)
    for tr_idx, te_idx in tss.split(np.arange(n)):
        yield np.asarray(tr_idx, dtype=int), np.asarray(te_idx, dtype=int)


def _safe_pearson_corr(a: Any, b: Any) -> Optional[float]:
    try:
        aa = np.asarray(a, dtype=float)
        bb = np.asarray(b, dtype=float)
        mask = np.isfinite(aa) & np.isfinite(bb)
        if int(mask.sum()) < 3:
            return None
        aa = aa[mask]
        bb = bb[mask]
        if float(np.std(aa)) < 1e-12 or float(np.std(bb)) < 1e-12:
            return 0.0
        c = float(np.corrcoef(aa, bb)[0, 1])
        return c if np.isfinite(c) else None
    except Exception:
        return None


def _safe_spearman_corr(a: Any, b: Any) -> Optional[float]:
    try:
        aa = np.asarray(a, dtype=float)
        bb = np.asarray(b, dtype=float)
        mask = np.isfinite(aa) & np.isfinite(bb)
        if int(mask.sum()) < 3:
            return None
        aa = aa[mask]
        bb = bb[mask]
        if float(np.std(aa)) < 1e-12 or float(np.std(bb)) < 1e-12:
            return 0.0
        ra = pd.Series(aa).rank(method="average").to_numpy(dtype=float)
        rb = pd.Series(bb).rank(method="average").to_numpy(dtype=float)
        c = float(np.corrcoef(ra, rb)[0, 1])
        return c if np.isfinite(c) else None
    except Exception:
        return None


def _sel_to_positions(index: pd.Index, sel: Any, *, is_cross_fit: bool) -> np.ndarray:
    n = int(len(index))
    if is_cross_fit:
        arr = np.asarray(sel, dtype=int)
        if arr.size == 0:
            return np.asarray([], dtype=int)
        return np.asarray(np.sort(arr), dtype=int)
    if isinstance(sel, slice):
        return np.arange(*sel.indices(n), dtype=int)
    try:
        b = np.asarray(sel, dtype=bool)
        if b.shape[0] == n:
            return np.asarray(np.flatnonzero(b), dtype=int)
    except Exception:
        pass
    try:
        return np.asarray(np.flatnonzero(index.isin(sel)), dtype=int)
    except Exception:
        return np.asarray([], dtype=int)


def _split_plan_summary(index: pd.Index, split_plan: Sequence[Tuple[Any, Any]], *, is_cross_fit: bool) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    n = int(len(index))
    for k, (tr, te) in enumerate(list(split_plan)):
        try:
            tr_pos = _sel_to_positions(index, tr, is_cross_fit=is_cross_fit)
            te_pos = _sel_to_positions(index, te, is_cross_fit=is_cross_fit)
            if tr_pos.size == 0 or te_pos.size == 0:
                continue
            out.append(
                {
                    "fold": int(k),
                    "train_start": str(index[int(tr_pos[0])]) if n else None,
                    "train_end": str(index[int(tr_pos[-1])]) if n else None,
                    "test_start": str(index[int(te_pos[0])]) if n else None,
                    "test_end": str(index[int(te_pos[-1])]) if n else None,
                    "n_train": int(tr_pos.size),
                    "n_test": int(te_pos.size),
                }
            )
        except Exception:
            continue
    return out


def _split_time_bins(index: pd.Index, n_bins: int) -> List[np.ndarray]:
    try:
        n = int(len(index))
        n_bins = int(n_bins)
        if n <= 0 or n_bins <= 1:
            return [np.arange(n, dtype=int)] if n > 0 else []
        n_bins = int(min(n_bins, max(1, n)))
        return [np.asarray(b, dtype=int) for b in np.array_split(np.arange(n, dtype=int), n_bins) if b.size > 0]
    except Exception:
        return []


def extract_regime_leaf_onehot_features(
    X: pd.DataFrame,
    market_data: pd.DataFrame,
    config: dict,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    if not _LGBM_AVAILABLE:
        raise ImportError("lightgbm is required for regime leaf feature extraction")

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    if X.empty:
        return pd.DataFrame(index=X.index)

    if market_data is None or not isinstance(market_data, pd.DataFrame) or market_data.empty:
        raise ValueError("market_data must be a non-empty DataFrame")

    targets_cfg = dict(config.get("targets", {}))
    wfv_cfg = dict(config.get("walk_forward", {}))
    input_cfg = dict(config.get("inputs", {}))
    preprocess_cfg = dict(config.get("preprocessing", {}))
    preprocess_cfg.setdefault("add_missing_indicators", True)
    preprocess_cfg.setdefault("enable_robust_scaling", False)

    hardening_cfg = dict(config.get("hardening", {}))
    time_feat_cfg = dict(config.get("time_features", {}))
    try:
        time_feat_cfg.setdefault("enabled", False)
        time_feat_cfg.setdefault("include_in_output", False)
    except Exception:
        pass

    reporting_cfg = dict(config.get("reporting", {}))
    reporting_enabled = bool(reporting_cfg.get("enabled", False))
    report_path = None

    enabled_targets = config.get("enabled_targets")
    if not isinstance(enabled_targets, (list, tuple, set)):
        enabled_targets = None
    else:
        try:
            if len(enabled_targets) == 0:
                enabled_targets = None
        except Exception:
            enabled_targets = None

    raw_score_cfg = config.get("raw_score")
    if not isinstance(raw_score_cfg, dict):
        raw_score_cfg = {}
    raw_score_enabled = bool(raw_score_cfg.get("enabled", False))

    onehot_cfg = config.get("onehot")
    if not isinstance(onehot_cfg, dict):
        onehot_cfg = {}
    onehot_enabled = bool(onehot_cfg.get("enabled", True))

    interaction_cfg = config.get("interaction_feature")
    if not isinstance(interaction_cfg, dict):
        interaction_cfg = {}
    interaction_enabled = bool(interaction_cfg.get("enabled", True))
    interaction_include_base = bool(interaction_cfg.get("include_base", True))

    max_lookahead_bars = int(
        wfv_cfg.get("max_lookahead_bars", _infer_max_lookahead_bars_from_targets_cfg(targets_cfg))
    )

    if verbose:
        try:
            tprint_info(
                "[regime_leaf] start "
                f"X_rows={int(len(X))} X_cols={int(X.shape[1])} "
                f"market_rows={int(len(market_data))} "
                f"max_lookahead_bars={int(max_lookahead_bars)}"
            )
        except Exception:
            pass

    try:
        market_data = _ensure_sorted_unique(market_data, "market_data", hardening_cfg=hardening_cfg, verbose=verbose)
        X = _ensure_sorted_unique(X, "X", hardening_cfg=hardening_cfg, verbose=verbose)
    except Exception as exc:
        raise

    try:
        targets_full = compute_regime_targets_from_ohlcv(market_data=market_data, config=targets_cfg)
    except Exception as exc:
        raise RuntimeError(f"failed_to_compute_regime_targets: {exc}")

    try:
        targets = targets_full.reindex(X.index)
    except Exception:
        targets = targets_full.loc[targets_full.index.intersection(X.index)].reindex(X.index)

    if enabled_targets is not None:
        try:
            cols = [c for c in targets.columns if str(c) in set([str(x) for x in enabled_targets])]
            targets = targets[cols]
        except Exception:
            pass

    input_source = str(input_cfg.get("input_source", "ohlcv_only")).lower()
    if input_source == "provided_x":
        X_num = X.copy()
        for col in X_num.columns:
            X_num[col] = pd.to_numeric(X_num[col], errors="coerce")
        X_num = X_num.replace([np.inf, -np.inf], np.nan)
    else:
        feat_cfg = dict(input_cfg.get("ohlcv_feature_config", {}))
        try:
            X_ohlcv = build_regime_embedding_features(market_data=market_data, cfg=feat_cfg)
        except Exception as exc:
            raise RuntimeError(f"failed_to_build_regime_embedding_features: {exc}")
        X_num = X_ohlcv.reindex(X.index)

    try:
        forbidden = _detect_forbidden_feature_columns(X_num.columns, hardening_cfg)
        if forbidden:
            action = str(hardening_cfg.get("forbidden_action", "drop")).lower()
            msg = f"[regime_leaf] forbidden_feature_columns={forbidden} action={action}"
            if action == "raise":
                raise ValueError(msg)
            if verbose:
                tprint_warning(msg)
            if action == "drop":
                X_num = X_num.drop(columns=forbidden, errors="ignore")
    except Exception:
        raise

    time_features = _time_features_from_index(X_num.index, time_feat_cfg)
    if time_features is not None and not time_features.empty:
        try:
            X_num = pd.concat([X_num, time_features], axis=1)
        except Exception:
            pass

    leaf_frames = []
    score_frames = []
    interaction_frames = []
    report: dict = {
        "enabled": bool(True),
        "random_state": int(random_state),
        "max_lookahead_bars": int(max_lookahead_bars),
        "X_shape": [int(X_num.shape[0]), int(X_num.shape[1])],
        "market_shape": [int(market_data.shape[0]), int(market_data.shape[1])],
        "input_source": str(input_source),
        "targets_cfg": dict(targets_cfg),
        "walk_forward_cfg": dict(wfv_cfg),
        "lgbm_cfg": dict(config.get("lgbm", {})),
        "topk_min": int(config.get("topk_min", 5)),
        "topk_max": (int(config.get("topk_max")) if config.get("topk_max") is not None else None),
        "max_features": (int(config.get("max_features")) if config.get("max_features") is not None else None),
        "targets": {},
        "report_path": None,
    }

    split_mode = str(wfv_cfg.get("mode", "walk_forward")).lower()
    is_cross_fit = bool(split_mode == "cross_fit")
    try:
        if is_cross_fit:
            split_plan = list(_iter_cross_fit_splits(X_num.index, dict(wfv_cfg.get("cross_fit", {}))))
        else:
            split_plan = list(_iter_anchored_windows(X_num.index, wfv_cfg))
    except Exception:
        split_plan = []

    leakage_cfg = dict(wfv_cfg.get("leakage", {})) if isinstance(wfv_cfg.get("leakage", {}), dict) else {}
    leakage_enabled = bool(leakage_cfg.get("enabled", True))
    try:
        report["split_mode"] = str(split_mode)
        report["split_plan"] = _split_plan_summary(X_num.index, split_plan, is_cross_fit=is_cross_fit)
        report["leakage"] = dict(leakage_cfg)
        report["leakage_policy"] = "purge_and_embargo_train_gap"
    except Exception:
        pass

    for target_name in list(targets.columns):
        y_all = pd.to_numeric(targets[target_name], errors="coerce")

        n_total_target = int(len(X_num))
        try:
            cfg_min_train = int(wfv_cfg.get("min_train_samples", 500))
        except Exception:
            cfg_min_train = 500
        try:
            cfg_min_test = int(wfv_cfg.get("min_test_samples", 50))
        except Exception:
            cfg_min_test = 50
        if n_total_target > 0:
            cfg_min_train = int(min(cfg_min_train, max(20, int(round(0.6 * n_total_target)))))
            cfg_min_test = int(min(cfg_min_test, max(10, int(round(0.2 * n_total_target)))))

        if verbose:
            try:
                nn = int(np.sum(pd.notna(y_all).values))
                tprint_info(f"[regime_leaf] target_begin target={target_name} y_non_null={nn}")
            except Exception:
                pass

        target_lookahead = None
        try:
            if target_name == "regime_volatility":
                target_lookahead = int(targets_cfg.get("volatility_window", 24))
            elif target_name == "regime_macro_trend":
                target_lookahead = int(targets_cfg.get("macro_trend_horizon", 96))
            elif target_name == "regime_trend_efficiency":
                target_lookahead = int(targets_cfg.get("trend_efficiency_window", 16))
            elif target_name == "regime_liquidity":
                target_lookahead = int(targets_cfg.get("liquidity_window", 50))
            elif target_name == "regime_memory":
                target_lookahead = int(targets_cfg.get("memory_window", 40))
        except Exception:
            target_lookahead = None
        if target_lookahead is None or target_lookahead <= 0:
            target_lookahead = int(max_lookahead_bars)

        leaves_oos = pd.DataFrame(index=X_num.index, dtype=float)
        contrib_oos = pd.DataFrame(index=X_num.index, dtype=float)
        base_oos = pd.Series(index=X_num.index, dtype=float)
        interaction_oos_raw = pd.Series(index=X_num.index, dtype=float)
        interaction_oos = pd.Series(index=X_num.index, dtype=float)
        interaction_center_oof = pd.Series(index=X_num.index, dtype=float)
        interaction_scale_oof = pd.Series(index=X_num.index, dtype=float)
        any_pred = False
        windows_trained = 0
        oos_rows_total = 0
        oos_rows_total_effective = 0
        oos_rows_pred = 0
        last_model_dump = None
        last_fit_error = None
        last_pred_error = None
        skipped_small_train = 0
        skipped_small_test = 0

        fold_test_indexes = []
        fold_test_indexes_raw = []

        fold_ic_spearman = []
        fold_ic_pearson = []
        fold_n = []

        standardize_cfg = interaction_cfg.get("standardize") if isinstance(interaction_cfg.get("standardize"), dict) else {}
        standardize_enabled = bool(standardize_cfg.get("enabled", True))
        standardize_method = str(standardize_cfg.get("method", "robust_zscore")).lower()
        try:
            standardize_clip_abs = standardize_cfg.get("clip_abs")
            standardize_clip_abs = float(standardize_clip_abs) if standardize_clip_abs is not None else None
        except Exception:
            standardize_clip_abs = None

        try:
            purge_bars = int(leakage_cfg.get("purge_bars", target_lookahead))
        except Exception:
            purge_bars = int(target_lookahead)
        purge_bars = int(max(int(target_lookahead), int(max(0, purge_bars))))

        try:
            embargo_bars = int(leakage_cfg.get("embargo_bars", purge_bars))
        except Exception:
            embargo_bars = int(purge_bars)
        embargo_bars = int(max(0, embargo_bars))

        for train_sel, test_sel in list(split_plan):
            try:
                tr_pos = _sel_to_positions(X_num.index, train_sel, is_cross_fit=is_cross_fit)
                te_pos = _sel_to_positions(X_num.index, test_sel, is_cross_fit=is_cross_fit)
                if tr_pos.size == 0 or te_pos.size == 0:
                    continue

                te_pos_raw = np.asarray(te_pos, dtype=int)
                te_pos_eff = te_pos_raw

                if leakage_enabled and te_pos_raw.size > 0 and tr_pos.size > 0:
                    cutoff = int(np.min(te_pos_raw)) - int(purge_bars) - int(embargo_bars)
                    if cutoff >= 0:
                        tr_pos = tr_pos[tr_pos < cutoff]

                if tr_pos.size == 0:
                    skipped_small_train += 1
                    continue
                if te_pos_eff.size == 0:
                    skipped_small_test += 1
                    continue

                X_train = X_num.iloc[tr_pos]
                y_train = y_all.iloc[tr_pos]
                X_test = X_num.iloc[te_pos_eff]
                test_index = X_test.index
                fold_test_indexes.append(test_index)
                fold_test_indexes_raw.append(X_num.index[te_pos_raw])
            except Exception:
                continue

            try:
                X_train, y_train = _prune_train_for_lookahead(X_train, y_train, target_lookahead)
            except Exception:
                pass

            y_train = pd.to_numeric(y_train, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(y_train) < int(cfg_min_train):
                skipped_small_train += 1
                continue
            if len(test_index) < int(cfg_min_test):
                skipped_small_test += 1
                continue

            X_train = X_train.reindex(y_train.index)
            X_test = X_test.reindex(test_index)

            if bool(preprocess_cfg.get("add_missing_indicators", True)):
                X_train, X_test = _add_missing_indicators(
                    X_train,
                    X_test,
                    dict(preprocess_cfg.get("missing_indicators", {})),
                )

            if bool(preprocess_cfg.get("enable_robust_scaling", True)):
                X_train, X_test = _robust_scale_train_test(
                    X_train,
                    X_test,
                    dict(preprocess_cfg.get("robust_scaling", {})),
                )

            params = _default_lgbm_params(
                dict(config.get("lgbm", {})),
                random_state=random_state,
                n_train_samples=int(len(y_train)),
            )
            model = lgb.LGBMRegressor(**params)

            try:
                model.fit(X_train, y_train)
            except Exception as fit_exc:
                last_fit_error = str(fit_exc)
                continue

            windows_trained += 1

            pred_test = None
            try:
                pred_test = np.asarray(model.predict(X_test), dtype=float).reshape(-1)
                if pred_test.shape[0] == len(test_index):
                    interaction_oos_raw.loc[test_index] = pred_test

                    center = 0.0
                    scale = 1.0
                    pred_out = pred_test

                    if standardize_enabled and interaction_enabled:
                        pred_train = None
                        try:
                            pred_train = np.asarray(model.predict(X_train), dtype=float).reshape(-1)
                        except Exception:
                            pred_train = None
                        if pred_train is not None:
                            tr = np.asarray(pred_train, dtype=float)
                            tr = tr[np.isfinite(tr)]
                        else:
                            tr = np.asarray([], dtype=float)

                        if tr.size >= 10:
                            if standardize_method == "zscore":
                                center = float(np.mean(tr))
                                scale = float(np.std(tr))
                            elif standardize_method == "rank":
                                tr_sorted = np.sort(tr)
                                denom = float(max(1, tr_sorted.size - 1))
                                ranks = np.searchsorted(tr_sorted, pred_test, side="right").astype(float)
                                pred_out = ranks / denom
                                center = 0.0
                                scale = 1.0
                            else:
                                center = float(np.median(tr))
                                q75 = float(np.quantile(tr, 0.75))
                                q25 = float(np.quantile(tr, 0.25))
                                scale = float(q75 - q25)
                            if not np.isfinite(scale) or scale <= 1e-12:
                                scale = float(np.std(tr))
                            if not np.isfinite(scale) or scale <= 1e-12:
                                scale = 1.0

                        if standardize_method in {"robust_zscore", "zscore"}:
                            pred_out = (pred_test - float(center)) / (float(scale) + 1e-12)

                    if standardize_clip_abs is not None and np.isfinite(standardize_clip_abs) and standardize_clip_abs > 0:
                        pred_out = np.clip(np.asarray(pred_out, dtype=float), -float(standardize_clip_abs), float(standardize_clip_abs))

                    interaction_oos.loc[test_index] = np.asarray(pred_out, dtype=float)
                    interaction_center_oof.loc[test_index] = float(center)
                    interaction_scale_oof.loc[test_index] = float(scale)

                    try:
                        y_test = pd.to_numeric(y_all.reindex(test_index), errors="coerce").to_numpy(dtype=float)
                        fold_ic_spearman.append(_safe_spearman_corr(pred_out, y_test))
                        fold_ic_pearson.append(_safe_pearson_corr(pred_out, y_test))
                        fold_n.append(int(np.isfinite(y_test).sum()))
                    except Exception:
                        pass
            except Exception:
                pred_test = None

            leaf_mat = _predict_leaf_matrix(model, X_test)
            if leaf_mat is None:
                last_pred_error = "pred_leaf_failed"
                continue

            if leaf_mat.ndim != 2 or leaf_mat.shape[0] != len(test_index):
                continue

            cols = [f"regime_leaf_raw__{target_name}__t{j}" for j in range(int(leaf_mat.shape[1]))]
            leaf_chunk = pd.DataFrame(leaf_mat, index=test_index, columns=cols)
            for c in cols:
                leaves_oos.loc[test_index, c] = leaf_chunk[c]
            any_pred = True

            try:
                booster = getattr(model, "booster_", None)
                if booster is not None:
                    dump = booster.dump_model()
                    last_model_dump = dump if isinstance(dump, dict) else last_model_dump
                    tree_info = dump.get("tree_info", []) if isinstance(dump, dict) else []
                    lr = float(params.get("learning_rate", 0.1))

                    contrib_mat = np.zeros_like(leaf_mat, dtype=float)
                    for j in range(int(leaf_mat.shape[1])):
                        tree = tree_info[j] if j < len(tree_info) else None
                        struct = tree.get("tree_structure") if isinstance(tree, dict) else None
                        if struct is None:
                            continue

                        mapping = {}

                        stack = [struct]
                        while stack:
                            node = stack.pop()
                            if not isinstance(node, dict):
                                continue
                            if "leaf_index" in node and "leaf_value" in node:
                                try:
                                    mapping[int(node["leaf_index"])] = float(node["leaf_value"])
                                except Exception:
                                    continue
                            else:
                                if "left_child" in node:
                                    stack.append(node.get("left_child"))
                                if "right_child" in node:
                                    stack.append(node.get("right_child"))

                        if not mapping:
                            continue

                        max_idx = int(max(mapping.keys()))
                        arr = np.zeros(max_idx + 1, dtype=float)
                        for k, v in mapping.items():
                            if 0 <= int(k) <= max_idx:
                                arr[int(k)] = float(v)

                        idxs = leaf_mat[:, j].astype(int)
                        idxs = np.clip(idxs, 0, max_idx)
                        contrib_mat[:, j] = lr * arr[idxs]

                    contrib_cols = [f"regime_leaf_contrib__{target_name}__t{j}" for j in range(int(contrib_mat.shape[1]))]
                    contrib_chunk = pd.DataFrame(contrib_mat, index=test_index, columns=contrib_cols)
                    for c in contrib_cols:
                        contrib_oos.loc[test_index, c] = contrib_chunk[c]

                    try:
                        if pred_test is None:
                            raw_pred = np.asarray(model.predict(X_test), dtype=float).reshape(-1)
                        else:
                            raw_pred = pred_test
                        base_vals = raw_pred - np.sum(contrib_mat, axis=1)
                        base_oos.loc[test_index] = base_vals
                    except Exception:
                        pass
            except Exception:
                pass

            try:
                oos_rows_total += int(len(fold_test_indexes_raw[-1]))
                oos_rows_total_effective += int(len(test_index))
                oos_rows_pred += int(np.sum(pd.notna(leaf_chunk.iloc[:, 0]).values))
            except Exception:
                pass

        if not any_pred:
            try:
                min_train_samples = int(cfg_min_train)
                min_test_samples = int(cfg_min_test)
                n_total = int(len(X_num))
                if n_total >= (min_train_samples + max(1, min_test_samples)):
                    test_start = int(max(0, n_total - min_test_samples))
                    train_end = int(n_total - min_test_samples - int(target_lookahead))
                    if train_end < min_train_samples:
                        train_end = int(max(min_train_samples, n_total - min_test_samples))

                    X_train = X_num.iloc[:train_end]
                    y_train = y_all.iloc[:train_end]
                    X_test = X_num.iloc[test_start:]
                    test_index = X_test.index

                    X_train, y_train = _prune_train_for_lookahead(X_train, y_train, target_lookahead)
                    y_train = y_train.dropna()

                    if len(y_train) >= min_train_samples and len(test_index) >= max(1, min_test_samples):
                        X_train = X_train.reindex(y_train.index)

                        if bool(preprocess_cfg.get("add_missing_indicators", True)):
                            X_train, X_test = _add_missing_indicators(
                                X_train,
                                X_test,
                                dict(preprocess_cfg.get("missing_indicators", {})),
                            )

                        if bool(preprocess_cfg.get("enable_robust_scaling", True)):
                            X_train, X_test = _robust_scale_train_test(
                                X_train,
                                X_test,
                                dict(preprocess_cfg.get("robust_scaling", {})),
                            )

                        params = _default_lgbm_params(
                            dict(config.get("lgbm", {})),
                            random_state=random_state,
                            n_train_samples=int(len(y_train)),
                        )
                        model = lgb.LGBMRegressor(**params)
                        try:
                            model.fit(X_train, y_train)
                        except Exception as fit_exc:
                            last_fit_error = str(fit_exc)
                            raise
                        windows_trained += 1

                        try:
                            pred_test = np.asarray(model.predict(X_test), dtype=float).reshape(-1)
                            if pred_test.shape[0] == len(test_index):
                                interaction_oos_raw.loc[test_index] = pred_test
                                pred_out = pred_test

                                if standardize_enabled and interaction_enabled and standardize_method in {"robust_zscore", "zscore"}:
                                    try:
                                        pred_train = np.asarray(model.predict(X_train), dtype=float).reshape(-1)
                                        tr = np.asarray(pred_train, dtype=float)
                                        tr = tr[np.isfinite(tr)]
                                        if tr.size >= 10:
                                            if standardize_method == "zscore":
                                                center = float(np.mean(tr))
                                                scale = float(np.std(tr))
                                            else:
                                                center = float(np.median(tr))
                                                q75 = float(np.quantile(tr, 0.75))
                                                q25 = float(np.quantile(tr, 0.25))
                                                scale = float(q75 - q25)
                                            if not np.isfinite(scale) or scale <= 1e-12:
                                                scale = float(np.std(tr))
                                            if not np.isfinite(scale) or scale <= 1e-12:
                                                scale = 1.0
                                            pred_out = (pred_test - float(center)) / (float(scale) + 1e-12)
                                    except Exception:
                                        pass

                                if standardize_clip_abs is not None and np.isfinite(standardize_clip_abs) and standardize_clip_abs > 0:
                                    pred_out = np.clip(np.asarray(pred_out, dtype=float), -float(standardize_clip_abs), float(standardize_clip_abs))

                                interaction_oos.loc[test_index] = np.asarray(pred_out, dtype=float)
                                any_pred = True
                                oos_rows_total += int(len(test_index))
                                oos_rows_total_effective += int(len(test_index))
                                try:
                                    y_test = pd.to_numeric(y_all.reindex(test_index), errors="coerce").to_numpy(dtype=float)
                                    fold_ic_spearman.append(_safe_spearman_corr(pred_out, y_test))
                                    fold_ic_pearson.append(_safe_pearson_corr(pred_out, y_test))
                                    fold_n.append(int(np.isfinite(y_test).sum()))
                                except Exception:
                                    pass
                        except Exception:
                            pass

                        leaf_mat = _predict_leaf_matrix(model, X_test)
                        if leaf_mat is None:
                            last_pred_error = "pred_leaf_failed"
                            continue

                        if leaf_mat.ndim == 2 and leaf_mat.shape[0] == len(test_index):
                            cols = [f"regime_leaf_raw__{target_name}__t{j}" for j in range(int(leaf_mat.shape[1]))]
                            leaf_chunk = pd.DataFrame(leaf_mat, index=test_index, columns=cols)
                            for c in cols:
                                leaves_oos.loc[test_index, c] = leaf_chunk[c]
                            any_pred = True
                            try:
                                oos_rows_total += int(len(test_index))
                                oos_rows_pred += int(np.sum(pd.notna(leaf_chunk.iloc[:, 0]).values))
                            except Exception:
                                pass
            except Exception:
                pass

        if not any_pred:
            last_fit_error = last_fit_error or "no_oos_predictions"

        if not any_pred:
            if verbose:
                err_bits = []
                if last_fit_error:
                    err_bits.append(f"fit={last_fit_error}")
                if last_pred_error:
                    err_bits.append(f"pred={last_pred_error}")
                suffix = (" " + " ".join(err_bits)) if err_bits else ""
                tprint_warning(f"[regime_leaf] no_oos_predictions target={target_name}{suffix}")
            try:
                report["targets"][str(target_name)] = {
                    "windows_trained": int(windows_trained),
                    "oos_rows": int(oos_rows_total),
                    "oos_rows_effective": int(oos_rows_total_effective),
                    "oos_coverage": 0.0,
                    "onehot_features": 0,
                    "raw_score_included": False,
                    "error": "no_oos_predictions",
                    "last_fit_error": last_fit_error,
                    "last_pred_error": last_pred_error,
                    "skipped_small_train": int(skipped_small_train),
                    "skipped_small_test": int(skipped_small_test),
                    "target_lookahead_bars": int(target_lookahead),
                    "purge_bars": int(purge_bars) if leakage_enabled else 0,
                    "embargo_bars": int(embargo_bars) if leakage_enabled else 0,
                    "standardize_method": str(standardize_method) if standardize_enabled else None,
                    "fold_ic_spearman": [],
                    "fold_ic_pearson": [],
                    "fold_n": [],
                }
            except Exception:
                pass
            continue

        if verbose:
            try:
                denom = int(oos_rows_total_effective) if int(oos_rows_total_effective) > 0 else int(oos_rows_total)
                coverage = float(oos_rows_pred) / float(max(1, denom))
                tprint_info(
                    f"[regime_leaf] target={target_name} windows={int(windows_trained)} "
                    f"oos_rows={int(oos_rows_total)} coverage={coverage:.2%}"
                )
            except Exception:
                pass

        raw_cols = [c for c in leaves_oos.columns if str(c).startswith(f"regime_leaf_raw__{target_name}__t")]

        pair_freqs = []
        keep_pairs_by_tree = {}
        for raw_col in raw_cols:
            try:
                s = leaves_oos[raw_col]
                vc = s.dropna().value_counts()
                denom = float(max(1, int(vc.sum())))
                for leaf_val, cnt in vc.items():
                    try:
                        pair_freqs.append((raw_col, float(leaf_val), float(cnt) / denom))
                    except Exception:
                        continue
            except Exception:
                continue

        if pair_freqs:
            pair_freqs_sorted = sorted(pair_freqs, key=lambda x: x[2], reverse=True)
            freqs = np.asarray([p[2] for p in pair_freqs_sorted], dtype=float)
            min_k = int(config.get("topk_min", 5))
            max_k = (config.get("topk_max") if "topk_max" in config else 7)
            try:
                if max_k is not None:
                    max_k = int(max_k)
            except Exception:
                max_k = None

            k_elbow = _find_elbow_k(freqs, min_k=min_k)
            if max_k is not None and max_k > 0:
                k_elbow = int(min(k_elbow, max_k))

            if verbose:
                try:
                    tprint_info(
                        f"[regime_leaf] target={target_name} topk_elbow_k={int(k_elbow)} "
                        f"pairs_total={int(len(pair_freqs_sorted))}"
                    )
                except Exception:
                    pass

            kept = pair_freqs_sorted[: int(max(0, k_elbow))]
            for raw_col, leaf_val, _ in kept:
                keep_pairs_by_tree.setdefault(raw_col, set()).add(float(leaf_val))

        pair_freqs_present = bool(pair_freqs)

        for raw_col in raw_cols:
            raw_series = leaves_oos[raw_col]
            dummies = pd.get_dummies(raw_series, prefix=raw_col, dummy_na=False)
            dummies = dummies.reindex(index=X_num.index).fillna(0.0)

            keep_vals = keep_pairs_by_tree.get(raw_col)
            if keep_vals is None:
                if pair_freqs_present:
                    dummies = dummies.iloc[:, 0:0]
            elif len(keep_vals) > 0:
                keep_cols = []
                for v in keep_vals:
                    key = f"{raw_col}_{v}"
                    if key in dummies.columns:
                        keep_cols.append(key)
                    else:
                        key_alt = f"{raw_col}_{int(v)}"
                        if key_alt in dummies.columns:
                            keep_cols.append(key_alt)
                if keep_cols:
                    dummies = dummies[keep_cols]
                else:
                    dummies = dummies.iloc[:, 0:0]

            if onehot_enabled and dummies is not None and not dummies.empty:
                leaf_frames.append(dummies)

        try:
            contrib_cols = [c for c in contrib_oos.columns if str(c).startswith(f"regime_leaf_contrib__{target_name}__t")]
            if raw_score_enabled and contrib_cols and raw_cols:
                masked_sum = pd.Series(0.0, index=X_num.index)
                for j, raw_col in enumerate(raw_cols):
                    if j >= len(contrib_cols):
                        continue
                    keep_vals = keep_pairs_by_tree.get(raw_col)
                    if keep_vals is None or len(keep_vals) == 0:
                        continue
                    mask = pd.to_numeric(leaves_oos[raw_col], errors="coerce").isin(list(keep_vals))
                    contrib_series = pd.to_numeric(contrib_oos[contrib_cols[j]], errors="coerce").fillna(0.0)
                    masked_sum = masked_sum + contrib_series.where(mask, 0.0)

                base_fill = pd.to_numeric(base_oos, errors="coerce")
                base_fill = base_fill.fillna(base_fill.dropna().median() if base_fill.notna().any() else 0.0)
                raw_score = (base_fill + masked_sum).astype(float)
                score_frames.append(
                    pd.DataFrame({f"regime_leaf_raw_score__{target_name}": raw_score}, index=X_num.index)
                )

                if verbose:
                    try:
                        rs_vals = pd.to_numeric(raw_score, errors="coerce").replace([np.inf, -np.inf], np.nan)
                        rs_mu = float(rs_vals.mean()) if rs_vals.notna().any() else float("nan")
                        rs_sd = float(rs_vals.std()) if rs_vals.notna().any() else float("nan")
                        tprint_info(
                            f"[regime_leaf] raw_score target={target_name} mean={rs_mu:.6f} std={rs_sd:.6f}"
                        )
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            kept_pairs = {k: sorted([int(v) for v in list(vals)]) for k, vals in keep_pairs_by_tree.items()}
        except Exception:
            kept_pairs = {}

        leaf_values_kept = {}
        try:
            if isinstance(last_model_dump, dict):
                tree_info = last_model_dump.get("tree_info", [])
                for j, tree in enumerate(tree_info):
                    raw_col = f"regime_leaf_raw__{target_name}__t{j}"
                    kept = kept_pairs.get(raw_col)
                    if not kept:
                        continue
                    struct = tree.get("tree_structure") if isinstance(tree, dict) else None
                    if struct is None:
                        continue
                    mapping = {}
                    stack = [struct]
                    while stack:
                        node = stack.pop()
                        if not isinstance(node, dict):
                            continue
                        if "leaf_index" in node and "leaf_value" in node:
                            try:
                                mapping[int(node["leaf_index"])] = float(node["leaf_value"])
                            except Exception:
                                continue
                        else:
                            if "left_child" in node:
                                stack.append(node.get("left_child"))
                            if "right_child" in node:
                                stack.append(node.get("right_child"))

                    vals = {}
                    for li in kept:
                        if int(li) in mapping:
                            vals[int(li)] = float(mapping[int(li)])
                    if vals:
                        leaf_values_kept[raw_col] = vals
        except Exception:
            leaf_values_kept = {}

        kept_leaf_paths = {}
        kept_leaf_stats = {}
        if reporting_enabled:
            try:
                include_leaf_paths = bool(reporting_cfg.get("include_leaf_paths", True))
                include_leaf_stats = bool(reporting_cfg.get("include_leaf_stats", True))
                top_features_per_leaf = int(reporting_cfg.get("top_features_per_leaf", 10))
                max_samples_per_leaf = int(reporting_cfg.get("max_samples_per_leaf", 2000))

                feature_names = None
                tree_info = None
                try:
                    if isinstance(last_model_dump, dict):
                        feature_names = last_model_dump.get("feature_names")
                        tree_info = last_model_dump.get("tree_info")
                except Exception:
                    feature_names = None
                    tree_info = None

                for raw_col, leaf_ids in kept_pairs.items():
                    try:
                        if not isinstance(raw_col, str) or not isinstance(leaf_ids, (list, tuple)) or not leaf_ids:
                            continue

                        if include_leaf_paths and isinstance(tree_info, list):
                            try:
                                tree_idx = int(str(raw_col).rsplit("__t", 1)[-1])
                            except Exception:
                                tree_idx = None
                            if tree_idx is not None and 0 <= int(tree_idx) < len(tree_info):
                                tree = tree_info[int(tree_idx)]
                                struct = tree.get("tree_structure") if isinstance(tree, dict) else None
                                all_paths = _extract_leaf_paths(
                                    struct,
                                    feature_names=feature_names if isinstance(feature_names, list) else None,
                                )
                                kept_path = {}
                                for li in list(leaf_ids):
                                    try:
                                        li_int = int(li)
                                    except Exception:
                                        continue
                                    if li_int in all_paths:
                                        kept_path[str(li_int)] = all_paths[li_int]
                                if kept_path:
                                    kept_leaf_paths[raw_col] = kept_path

                        if include_leaf_stats:
                            try:
                                raw_series = leaves_oos.get(raw_col)
                                if raw_series is None:
                                    continue
                                stats = _leaf_summary_stats(
                                    raw_leaf_series=raw_series,
                                    y_all=y_all,
                                    X_num=X_num,
                                    kept_leaf_ids=[int(v) for v in list(leaf_ids)],
                                    random_state=int(random_state),
                                    top_features_per_leaf=int(top_features_per_leaf),
                                    max_samples_per_leaf=int(max_samples_per_leaf),
                                )
                                if stats:
                                    kept_leaf_stats[raw_col] = {str(k): v for k, v in stats.items()}
                            except Exception:
                                pass
                    except Exception:
                        continue
            except Exception:
                kept_leaf_paths = {}
                kept_leaf_stats = {}

        try:
            report["targets"][str(target_name)] = {
                "windows_trained": int(windows_trained),
                "oos_rows": int(oos_rows_total),
                "oos_rows_effective": int(oos_rows_total_effective),
                "oos_coverage": float(oos_rows_pred) / float(max(1, oos_rows_total_effective if int(oos_rows_total_effective) > 0 else oos_rows_total)),
                "target_lookahead_bars": int(target_lookahead),
                "purge_bars": int(purge_bars) if leakage_enabled else 0,
                "embargo_bars": int(embargo_bars) if leakage_enabled else 0,
                "standardize_method": str(standardize_method) if standardize_enabled else None,
                "fold_ic_spearman": [float(x) if x is not None and np.isfinite(x) else None for x in fold_ic_spearman],
                "fold_ic_pearson": [float(x) if x is not None and np.isfinite(x) else None for x in fold_ic_pearson],
                "fold_n": [int(x) for x in fold_n],
                "fold_ic_spearman_summary": {},
                "kept_pairs_by_tree": kept_pairs,
                "kept_leaf_values": leaf_values_kept,
                "kept_leaf_paths": kept_leaf_paths,
                "kept_leaf_stats": kept_leaf_stats,
                "interaction_feature_included": bool(interaction_enabled),
                "model_params": dict(_default_lgbm_params(dict(config.get("lgbm", {})), random_state=random_state)),
            }

            try:
                ic_vals = [float(x) for x in fold_ic_spearman if x is not None and np.isfinite(x)]
                if ic_vals:
                    ic_mean = float(np.mean(ic_vals))
                    ic_std = float(np.std(ic_vals))
                    report["targets"][str(target_name)]["fold_ic_spearman_summary"] = {
                        "mean": ic_mean,
                        "std": ic_std,
                        "icir": (ic_mean / (ic_std + 1e-12)) if np.isfinite(ic_std) else None,
                        "sign_consistency": float(np.mean([1.0 if v >= 0 else 0.0 for v in ic_vals])),
                        "n_folds": int(len(ic_vals)),
                    }
            except Exception:
                pass
        except Exception:
            pass

        if interaction_enabled:
            try:
                if interaction_include_base:
                    interaction_series = pd.to_numeric(interaction_oos, errors="coerce")
                else:
                    base_raw = pd.to_numeric(base_oos, errors="coerce")
                    pred_raw = pd.to_numeric(interaction_oos_raw, errors="coerce")
                    scale = pd.to_numeric(interaction_scale_oof, errors="coerce")
                    resid_raw = (pred_raw - base_raw).astype(float)
                    resid_scaled = resid_raw / (scale.replace(0.0, np.nan) + 1e-12)
                    if standardize_enabled:
                        interaction_series = resid_scaled.where(resid_scaled.notna(), pd.to_numeric(interaction_oos, errors="coerce"))
                    else:
                        interaction_series = resid_raw.where(resid_raw.notna(), pd.to_numeric(interaction_oos, errors="coerce"))

                interaction_series = interaction_series.reindex(X_num.index).astype(float)
                interaction_frames.append(
                    pd.DataFrame({f"regime_leaf_interaction__{target_name}": interaction_series.fillna(0.0)}, index=X_num.index)
                )

                transition_cfg = interaction_cfg.get("transition") if isinstance(interaction_cfg.get("transition"), dict) else {}
                transition_enabled = bool(transition_cfg.get("enabled", True))
                if transition_enabled:
                    s_ff = interaction_series.ffill().fillna(0.0)
                    d1 = s_ff.diff(1).fillna(0.0).astype(float)
                    interaction_frames.append(
                        pd.DataFrame({f"regime_leaf_interaction_transition__{target_name}": d1}, index=X_num.index)
                    )
                    interaction_frames.append(
                        pd.DataFrame({f"regime_leaf_interaction_transition_abs__{target_name}": d1.abs()}, index=X_num.index)
                    )

                try:
                    stats = {
                        "interaction_non_null": int(pd.to_numeric(interaction_series, errors="coerce").notna().sum()),
                        "interaction_mean": float(pd.to_numeric(interaction_series, errors="coerce").mean()),
                        "interaction_std": float(pd.to_numeric(interaction_series, errors="coerce").std()),
                    }
                    if transition_enabled:
                        stats.update(
                            {
                                "transition_mean_abs": float(d1.abs().mean()),
                                "transition_std": float(d1.std()),
                            }
                        )
                    if isinstance(report.get("targets", {}).get(str(target_name)), dict):
                        report["targets"][str(target_name)]["interaction_stats"] = stats
                except Exception:
                    pass

                try:
                    include_bins = bool(reporting_cfg.get("include_interaction_ic_bins", True)) if isinstance(reporting_cfg, dict) else True
                    if include_bins:
                        try:
                            n_bins = int(reporting_cfg.get("interaction_ic_bins", 8)) if isinstance(reporting_cfg, dict) else 8
                        except Exception:
                            n_bins = 8
                        bins = _split_time_bins(X_num.index, n_bins)
                        bin_ics = []
                        for bi, pos in enumerate(bins):
                            try:
                                idx_bin = X_num.index[np.asarray(pos, dtype=int)]
                                ic_b = _safe_spearman_corr(
                                    pd.to_numeric(interaction_series.reindex(idx_bin), errors="coerce").to_numpy(dtype=float),
                                    pd.to_numeric(y_all.reindex(idx_bin), errors="coerce").to_numpy(dtype=float),
                                )
                                bin_ics.append(float(ic_b) if ic_b is not None and np.isfinite(ic_b) else None)
                            except Exception:
                                bin_ics.append(None)
                        bin_ics_f = [float(v) for v in bin_ics if v is not None and np.isfinite(v)]
                        stability = {
                            "n_bins": int(len(bin_ics)),
                            "ic_bins": bin_ics,
                            "ic_mean": float(np.mean(bin_ics_f)) if bin_ics_f else None,
                            "ic_std": float(np.std(bin_ics_f)) if bin_ics_f else None,
                            "sign_consistency": float(np.mean([1.0 if v >= 0 else 0.0 for v in bin_ics_f])) if bin_ics_f else None,
                        }
                        report["targets"][str(target_name)]["interaction_ic_stability"] = stability
                except Exception:
                    pass
            except Exception:
                pass

    if not leaf_frames:
        if score_frames or interaction_frames:
            parts = []
            if score_frames:
                parts.append(pd.concat(score_frames, axis=1))
            if interaction_frames:
                parts.append(pd.concat(interaction_frames, axis=1))
            out_df = pd.concat(parts, axis=1).reindex(X_num.index).fillna(0.0)
            if time_features is not None and not time_features.empty and bool(time_feat_cfg.get("include_in_output", True)):
                try:
                    out_df = pd.concat([out_df, time_features.reindex(out_df.index)], axis=1).fillna(0.0)
                except Exception:
                    pass
            try:
                if reporting_enabled:
                    try:
                        out_dir = str(reporting_cfg.get("output_dir", "outcomes"))
                        prefix = str(reporting_cfg.get("prefix", "regime_leaf_report"))
                        tag = reporting_cfg.get("run_tag")
                        if tag is None:
                            tag = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
                        os.makedirs(out_dir, exist_ok=True)
                        report_path = f"{out_dir}/{prefix}_{tag}.json"
                        report["report_path"] = str(report_path)
                        with open(report_path, "w") as f:
                            import json as _json

                            _json.dump(report, f, indent=2, default=str)
                    except Exception:
                        pass
                    out_df.attrs["regime_leaf_report"] = dict(report)
            except Exception:
                pass
            return out_df
        return pd.DataFrame(index=X_num.index)

    leaf_onehot = pd.concat(leaf_frames, axis=1)
    if score_frames:
        leaf_onehot = pd.concat([leaf_onehot] + score_frames, axis=1)

    if interaction_frames:
        try:
            leaf_onehot = pd.concat([leaf_onehot] + interaction_frames, axis=1)
        except Exception:
            pass

    if time_features is not None and not time_features.empty and bool(time_feat_cfg.get("include_in_output", True)):
        try:
            leaf_onehot = pd.concat([leaf_onehot, time_features.reindex(leaf_onehot.index)], axis=1).fillna(0.0)
        except Exception:
            pass

    max_features = config.get("max_features")
    if max_features is not None:
        try:
            max_features = int(max_features)
            if max_features > 0 and leaf_onehot.shape[1] > max_features:
                leaf_onehot = leaf_onehot.iloc[:, :max_features]
        except Exception:
            pass

    leaf_onehot.columns = [str(c) for c in leaf_onehot.columns]

    if verbose:
        tprint_info(f"[regime_leaf] onehot_features={int(leaf_onehot.shape[1])}")

    if reporting_enabled:
        try:
            out_dir = str(reporting_cfg.get("output_dir", "outcomes"))
            prefix = str(reporting_cfg.get("prefix", "regime_leaf_report"))
            tag = reporting_cfg.get("run_tag")
            if tag is None:
                tag = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
            os.makedirs(out_dir, exist_ok=True)
            report_path = f"{out_dir}/{prefix}_{tag}.json"
            report["report_path"] = str(report_path)
            with open(report_path, "w") as f:
                import json as _json

                _json.dump(report, f, indent=2, default=str)
            if verbose:
                tprint_info(f"[regime_leaf] report_saved path={report_path}")
        except Exception as rep_exc:
            if verbose:
                tprint_warning(f"[regime_leaf] report_failed error={rep_exc}")

    try:
        leaf_onehot.attrs["regime_leaf_report"] = dict(report)
    except Exception:
        pass

    return leaf_onehot
