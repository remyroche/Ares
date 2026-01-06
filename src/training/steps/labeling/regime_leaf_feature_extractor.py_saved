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
from src.feature_generation.categories.smc_regime_features import generate_smc_regime_features
from src.feature_generation.categories.liquidity_regime_features import generate_liquidity_regime_features
from src.feature_generation.categories.volume_force_features import generate_volume_force_features

# Temporal Conv Encoder (optional, for neural temporal features)
try:
    from src.training.steps.labeling.temporal_conv_encoder import (
        generate_temporal_embeddings,
        TemporalConvEncoder,
    )
    _TEMPORAL_ENCODER_AVAILABLE = True
except ImportError:
    _TEMPORAL_ENCODER_AVAILABLE = False

# Stacked NN Sequence Encoder (optional, for Conv+LSTM+Attention)
try:
    from src.training.steps.labeling.short_nn_sequence_template import (
        generate_nn_sequence_embeddings,
        StackedSequenceEncoder,
    )
    _NN_SEQUENCE_AVAILABLE = True
except ImportError:
    _NN_SEQUENCE_AVAILABLE = False

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


def _infer_time_tolerance(index: pd.DatetimeIndex) -> Optional[pd.Timedelta]:
    try:
        if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
            return None
        diffs = pd.Series(index).diff().dropna()
        if diffs.empty:
            return None
        med = diffs.median()
        if med is None:
            return None
        tol = pd.to_timedelta(med) * 2
        return tol if tol > pd.Timedelta(0) else None
    except Exception:
        return None


def _align_frame_to_index(
    df: pd.DataFrame,
    target_index: pd.Index,
    *,
    method: str = "ffill",
    tolerance: Optional[pd.Timedelta] = None,
) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(index=target_index)
    if not isinstance(df.index, pd.DatetimeIndex) or not isinstance(target_index, pd.DatetimeIndex):
        return df.reindex(target_index)

    if tolerance is None:
        tolerance = _infer_time_tolerance(df.index)
    method = str(method or "ffill").lower()
    if method in {"ffill", "pad"}:
        return df.reindex(target_index, method="ffill", tolerance=tolerance)
    if method in {"bfill", "backfill"}:
        return df.reindex(target_index, method="bfill", tolerance=tolerance)
    if method in {"nearest"}:
        return df.reindex(target_index, method="nearest", tolerance=tolerance)
    return df.reindex(target_index)


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
        "volume_force_horizon",
        "volume_force_lookahead",
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

    try:
        eh = targets_cfg.get("trend_efficiency_horizons")
        if isinstance(eh, (list, tuple)) and eh:
            cands.append(int(max([int(x) for x in eh if x is not None] + [1])))
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


def _compute_future_past_vol_log_ratio(close: pd.Series, window: int) -> pd.Series:
    log_ret = np.log(close).diff()
    past = log_ret.rolling(window=window, min_periods=max(5, window // 4)).std()
    future = log_ret.shift(-1).rolling(window=window, min_periods=max(5, window // 4)).std().shift(-(window - 1))
    return (np.log(future + 1e-12) - np.log(past.shift(1) + 1e-12)).astype(float)


def _compute_future_return(close: pd.Series, horizon: int) -> pd.Series:
    return (close.shift(-horizon) / (close + 1e-12) - 1.0).astype(float)


def _compute_trend_efficiency(close: pd.Series, window: int) -> pd.Series:
    c = pd.to_numeric(close, errors="coerce").replace(0.0, np.nan)
    logp = np.log(c)
    fut_signal = (logp.shift(-window) - logp).abs()
    noise = logp.diff().abs().shift(-1)
    fut_noise = noise.rolling(window=window, min_periods=max(5, window // 4)).sum().shift(-(window - 1))
    fut_noise = fut_noise.clip(lower=1e-6)
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


def _score_leaf_pairs_effect_support(
    *,
    leaves_oos: pd.DataFrame,
    y_all: pd.Series,
    raw_cols: Sequence[str],
    min_support: float,
    max_support: float,
    dominant_support_max: float,
    min_effect_z: float,
    max_pairs: Optional[int],
) -> Tuple[List[Tuple[str, float, float, float]], Dict[str, set]]:
    pair_scores: List[Tuple[str, float, float, float]] = []
    keep_pairs_by_tree: Dict[str, set] = {}

    y_vals = pd.to_numeric(y_all, errors="coerce")
    y_mu = float(y_vals.mean()) if y_vals.notna().any() else 0.0
    y_sd = float(y_vals.std()) if y_vals.notna().any() else 0.0
    y_sd = float(y_sd) if np.isfinite(y_sd) and y_sd > 1e-12 else 1.0

    for raw_col in list(raw_cols):
        try:
            s = pd.to_numeric(leaves_oos[raw_col], errors="coerce")
        except Exception:
            continue

        try:
            df_col = pd.DataFrame({'leaf': s, 'y': y_vals}).dropna()
            if df_col.empty:
                continue

            n = len(df_col)
            grouped = df_col.groupby('leaf')['y']

            counts = grouped.count()
            means = grouped.mean()

            supports = counts / float(n)

            for leaf_val, count in counts.items():
                support = supports[leaf_val]

                if support < float(min_support) or support > float(max_support):
                    continue
                if support > float(dominant_support_max):
                    continue
                if count < 5:
                    continue

                leaf_mean = means[leaf_val]
                effect_z = abs(leaf_mean - y_mu) / (y_sd + 1e-12)

                if not np.isfinite(effect_z) or effect_z < float(min_effect_z):
                    continue

                score = effect_z * np.sqrt(max(1e-12, support))
                if not np.isfinite(score):
                    continue

                pair_scores.append((str(raw_col), float(leaf_val), float(score), float(support)))

        except Exception:
            continue

    pair_scores_sorted = sorted(pair_scores, key=lambda x: x[2], reverse=True)
    if max_pairs is not None:
        try:
            max_pairs = int(max_pairs)
        except Exception:
            max_pairs = None
    if max_pairs is not None and max_pairs > 0:
        pair_scores_sorted = pair_scores_sorted[: int(max_pairs)]

    for raw_col, leaf_val, _, _ in pair_scores_sorted:
        keep_pairs_by_tree.setdefault(raw_col, set()).add(float(leaf_val))

    return pair_scores_sorted, keep_pairs_by_tree


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


def _safe_tanh(x: pd.Series, scale: float) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").astype(float)
    try:
        scale = float(scale)
    except Exception:
        scale = 1.0
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return np.tanh(x / scale).astype(float)


def _interaction_gating_features(
    score: pd.Series,
    *,
    prefix: str,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Generate gating features for regime leaf interactions.

    Default configuration reduces feature count to prevent overfitting:
    - include_sign: True  - directional indicator (+1/-1/0)
    - include_soft: True  - tanh-scaled continuous version
    - include_bins: False - disabled by default (adds 2 dummy cols per target = 10 extra features)

    Total features per target with defaults: 2 (sign + soft)
    Previously with bins enabled: 4 (sign + soft + 2 bins)
    """
    out = pd.DataFrame(index=score.index)
    include_sign = bool(cfg.get("include_sign", True))
    include_soft = bool(cfg.get("include_soft", True))
    # Disabled by default to reduce feature count and overfitting risk
    # Bins add 2 dummy columns per target (10 total for 5 targets)
    include_bins = bool(cfg.get("include_bins", False))

    if include_sign:
        s = pd.to_numeric(score, errors="coerce")
        out[f"{prefix}gate_sign"] = np.sign(s.fillna(0.0)).astype(float)

    if include_soft:
        scale = float(cfg.get("soft_scale", 1.0))
        out[f"{prefix}gate_soft"] = _safe_tanh(score, scale=scale)

    if include_bins:
        try:
            n_bins = int(cfg.get("n_bins", 3))
        except Exception:
            n_bins = 3
        n_bins = int(max(2, min(8, n_bins)))

        s = pd.to_numeric(score, errors="coerce")
        s_non_null = s.dropna()
        if len(s_non_null) >= max(50, 10 * n_bins):
            split_idx = max(1, int(len(s_non_null) * float(cfg.get("train_frac", 0.7))))
            train = s_non_null.iloc[:split_idx]
            try:
                qs = [float(q) for q in np.linspace(0.0, 1.0, n_bins + 1)]
                edges = train.quantile(qs).to_numpy(dtype=float)
                edges[0] = -np.inf
                edges[-1] = np.inf
                for i in range(1, len(edges) - 1):
                    if not np.isfinite(edges[i]) or edges[i] <= edges[i - 1]:
                        edges[i] = edges[i - 1] + 1e-12
                labels = [f"b{i}" for i in range(n_bins)]
                binned = pd.cut(s, bins=edges, labels=labels, include_lowest=True)
                d = pd.get_dummies(binned, prefix=f"{prefix}gate_bin", drop_first=True)
                out = out.join(d)
            except Exception:
                pass

    return out


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

    try:
        spread = (ema_fast - ema_slow).astype(float)
        out["reg_ohlcv__ema_spread_slope"] = (spread.diff() / (spread.abs() + 1e-12)).astype(float)
        out["reg_ohlcv__ema_spread_accel"] = (out["reg_ohlcv__ema_spread_slope"].diff() / (out["reg_ohlcv__ema_spread_slope"].abs() + 1e-12)).astype(float)
    except Exception:
        out["reg_ohlcv__ema_spread_slope"] = np.nan
        out["reg_ohlcv__ema_spread_accel"] = np.nan

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

    try:
        re_cfg = cfg.get("range_expansion")
        if not isinstance(re_cfg, dict):
            re_cfg = {}
        re_windows = re_cfg.get("windows")
        if not isinstance(re_windows, (list, tuple)) or not re_windows:
            re_windows = [96]
        re_windows = [int(w) for w in re_windows if int(w) >= 10]
        for w in re_windows:
            hh = high.rolling(window=w, min_periods=max(10, w // 4)).max()
            ll = low.rolling(window=w, min_periods=max(10, w // 4)).min()
            out[f"reg_ohlcv__range_expansion_atr_w{w}"] = ((hh - ll) / (atr + 1e-12)).replace([np.inf, -np.inf], np.nan)
    except Exception:
        pass

    gap = (open_px - close.shift(1)).astype(float)
    out["reg_ohlcv__gap_abs_atr"] = (gap.abs() / (atr + 1e-12)).replace([np.inf, -np.inf], np.nan)
    out["reg_ohlcv__gap_signed_atr"] = (gap / (atr + 1e-12)).replace([np.inf, -np.inf], np.nan)

    try:
        gap_cfg = cfg.get("gap_frequency")
        if not isinstance(gap_cfg, dict):
            gap_cfg = {}
        gap_windows = gap_cfg.get("windows")
        if not isinstance(gap_windows, (list, tuple)) or not gap_windows:
            gap_windows = [96]
        gap_windows = [int(w) for w in gap_windows if int(w) >= 10]
        gap_thr = float(gap_cfg.get("gap_abs_atr_threshold", 1.0))
        for w in gap_windows:
            gflag = (pd.to_numeric(out["reg_ohlcv__gap_abs_atr"], errors="coerce") > gap_thr).astype(float)
            out[f"reg_ohlcv__gap_freq_gt{gap_thr:g}_w{w}"] = gflag.rolling(
                window=w, min_periods=max(10, w // 4)
            ).mean()
    except Exception:
        pass

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

        try:
            shock_cfg = cfg.get("liquidity_shock")
            if not isinstance(shock_cfg, dict):
                shock_cfg = {}
            shock_w = int(shock_cfg.get("window", norm_w))
            clip_abs = float(shock_cfg.get("clip_abs", 6.0))
            vol_chg = vol.pct_change().replace([np.inf, -np.inf], np.nan)
            t_chg = turnover.pct_change().replace([np.inf, -np.inf], np.nan)
            out["reg_ohlcv__volume_chg_z"] = winsorized_zscore_normalize(vol_chg, window=shock_w, min_periods=max(10, shock_w // 10)).clip(
                lower=-clip_abs, upper=clip_abs
            )
            out["reg_ohlcv__turnover_chg_z"] = winsorized_zscore_normalize(t_chg, window=shock_w, min_periods=max(10, shock_w // 10)).clip(
                lower=-clip_abs, upper=clip_abs
            )
        except Exception:
            out["reg_ohlcv__volume_chg_z"] = np.nan
            out["reg_ohlcv__turnover_chg_z"] = np.nan

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

        # Advanced Volatility Features (Vol-of-Vol, ATR Rank, Compression)
        try:
            adv_vol_windows = [24, 96]
            
            # Vol-of-Vol: CoV of volatility (std / mean)
            for w in adv_vol_windows:
                vol_rolling_mean = vol.rolling(window=w, min_periods=max(5, w//4)).mean()
                vol_rolling_std = vol.rolling(window=w, min_periods=max(5, w//4)).std()
                out[f"reg_ohlcv__vol_of_vol_w{w}"] = (vol_rolling_std / (vol_rolling_mean + 1e-12)).replace([np.inf, -np.inf], np.nan)
            
            # ATR Rank: Percentile of current ATR within historical window
            # (ATR - MinATR) / (MaxATR - MinATR)
            for w in adv_vol_windows:
                atr_min = atr.rolling(window=w, min_periods=max(5, w//4)).min()
                atr_max = atr.rolling(window=w, min_periods=max(5, w//4)).max()
                out[f"reg_ohlcv__atr_rank_w{w}"] = ((atr - atr_min) / (atr_max - atr_min + 1e-12)).clip(0.0, 1.0)

            # Compression Ratio: Short-term range / Long-term range (BB Width proxy)
            # Using ATR ratio as proxy: ATR(short) / ATR(long)
            # Low values (< 1) imply compression, High values (> 1) imply expansion
            short_w = 12
            long_w = 48
            atr_short = atr.rolling(window=short_w, min_periods=5).mean()
            atr_long = atr.rolling(window=long_w, min_periods=20).mean()
            out["reg_ohlcv__vol_compression_ratio"] = (atr_short / (atr_long + 1e-12)).replace([np.inf, -np.inf], np.nan)
            
        except Exception:
            pass

        # Multi-Horizon Regime Agreement
        # Proxies for Volatility (Log Ratio) and Trend (Returns) across scales
        try:
            mh_windows = [8, 16, 32]
            
            # 1. Volatility Proxies
            vol_proxies = []
            for w in mh_windows:
                # Log(Vol / MA(Vol, w))
                v_ma = vol.rolling(window=w, min_periods=max(5, w//4)).mean()
                vp = np.log((vol / (v_ma + 1e-12)).replace(0, np.nan)).fillna(0.0)
                vol_proxies.append(vp)
            
            # Vol Dispersion (Std across horizons)
            # Use concat(axis=1).std(axis=1)
            vol_proxy_df = pd.concat(vol_proxies, axis=1)
            out["reg_ohlcv__vol_multi_horizon_dispersion"] = vol_proxy_df.std(axis=1)
            out["reg_ohlcv__vol_multi_horizon_mean"] = vol_proxy_df.mean(axis=1)

            # 2. Trend Proxies
            trend_proxies = []
            for w in mh_windows:
                # Returns over w bars
                tp = close.pct_change(periods=w).fillna(0.0)
                trend_proxies.append(tp)

            trend_proxy_df = pd.concat(trend_proxies, axis=1)

            # Trend Agreement: Mean of Signs (1.0 = All Agree, 0.0 = Mixed)
            # We want "Agreement Strength" -> abs(mean(signs))
            # If 3 windows: (+1, +1, +1) -> mean 1.0 -> abs 1.0
            # (+1, +1, -1) -> mean 0.33 -> abs 0.33
            trend_signs = np.sign(trend_proxy_df)
            out["reg_ohlcv__trend_multi_horizon_agreement"] = trend_signs.mean(axis=1).abs()
            out["reg_ohlcv__trend_multi_horizon_mean"] = trend_proxy_df.mean(axis=1)

        except Exception:
            out["reg_ohlcv__vol_multi_horizon_dispersion"] = 0.0
            out["reg_ohlcv__vol_multi_horizon_mean"] = 0.0
            out["reg_ohlcv__trend_multi_horizon_agreement"] = 0.5
            out["reg_ohlcv__trend_multi_horizon_mean"] = 0.0

        # =====================================================================
        # TREND STRENGTH FEATURES (ADX, Donchian, Trend Persistence)
        # Added to improve performance in trending/volatile markets
        # =====================================================================

        # --- ADX (Average Directional Index) ---
        # Measures trend strength regardless of direction (0-100 scale)
        # Windows: 16=4h, 24=6h, 32=8h at 15m timeframe
        try:
            adx_cfg = cfg.get("adx", {})
            if not isinstance(adx_cfg, dict):
                adx_cfg = {}
            adx_windows = adx_cfg.get("windows", [16, 24, 32])  # 4h, 6h, 8h at 15m
            if not isinstance(adx_windows, (list, tuple)):
                adx_windows = [16, 24, 32]
            adx_windows = [int(w) for w in adx_windows if int(w) >= 5]

            for adx_w in adx_windows:
                # True Range (already computed as `tr`)
                # +DM (positive directional movement)
                high_diff = high.diff()
                low_diff = low.diff()
                plus_dm = high_diff.where((high_diff > 0) & (high_diff > -low_diff), 0.0)
                minus_dm = (-low_diff).where((low_diff < 0) & (-low_diff > high_diff), 0.0)

                # Smoothed averages (Wilder's smoothing = EWM with alpha=1/window)
                alpha = 1.0 / float(adx_w)
                atr_smooth = tr.ewm(alpha=alpha, adjust=False, min_periods=max(5, adx_w // 4)).mean()
                plus_dm_smooth = plus_dm.ewm(alpha=alpha, adjust=False, min_periods=max(5, adx_w // 4)).mean()
                minus_dm_smooth = minus_dm.ewm(alpha=alpha, adjust=False, min_periods=max(5, adx_w // 4)).mean()

                # +DI and -DI
                plus_di = 100.0 * plus_dm_smooth / (atr_smooth + 1e-12)
                minus_di = 100.0 * minus_dm_smooth / (atr_smooth + 1e-12)

                # DX = |+DI - -DI| / |+DI + -DI| * 100
                di_sum = plus_di + minus_di
                di_diff = (plus_di - minus_di).abs()
                dx = 100.0 * di_diff / (di_sum + 1e-12)

                # ADX = smoothed DX
                adx_val = dx.ewm(alpha=alpha, adjust=False, min_periods=max(5, adx_w // 4)).mean()

                out[f"reg_ohlcv__adx_w{adx_w}"] = adx_val.clip(0.0, 100.0)
                out[f"reg_ohlcv__plus_di_w{adx_w}"] = plus_di.clip(0.0, 100.0)
                out[f"reg_ohlcv__minus_di_w{adx_w}"] = minus_di.clip(0.0, 100.0)
                # DI spread: bullish when positive
                out[f"reg_ohlcv__di_spread_w{adx_w}"] = (plus_di - minus_di).clip(-100.0, 100.0)

                # --- ADX × Direction Interaction Features ---
                # Combines trend strength with trend direction for macro_trend prediction
                # Direction: sign of return over the window
                htf_ret = close.pct_change(adx_w)
                htf_dir = np.sign(htf_ret)

                # ADX × Direction: positive for strong uptrend, negative for strong downtrend
                out[f"reg_ohlcv__adx_x_direction_w{adx_w}"] = (adx_val * htf_dir / 100.0).clip(-1.0, 1.0)

                # ADX × |Return|: trend strength weighted by move size
                out[f"reg_ohlcv__adx_x_abs_ret_w{adx_w}"] = (adx_val * htf_ret.abs() * 10.0).clip(0.0, 10.0)

                # Directional trend score: ADX * DI_spread / 100 (captures both strength and direction)
                di_spread_norm = (plus_di - minus_di) / 100.0  # -1 to +1
                out[f"reg_ohlcv__directional_trend_w{adx_w}"] = (adx_val / 100.0 * di_spread_norm).clip(-1.0, 1.0)

        except Exception:
            for w in [16, 24, 32]:
                out[f"reg_ohlcv__adx_w{w}"] = np.nan
                out[f"reg_ohlcv__plus_di_w{w}"] = np.nan
                out[f"reg_ohlcv__minus_di_w{w}"] = np.nan
                out[f"reg_ohlcv__di_spread_w{w}"] = np.nan
                out[f"reg_ohlcv__adx_x_direction_w{w}"] = 0.0
                out[f"reg_ohlcv__adx_x_abs_ret_w{w}"] = 0.0
                out[f"reg_ohlcv__directional_trend_w{w}"] = 0.0

        # --- Donchian Channel Features ---
        # Position within channel (0=at low, 1=at high), channel width
        try:
            donch_cfg = cfg.get("donchian", {})
            if not isinstance(donch_cfg, dict):
                donch_cfg = {}
            donch_windows = donch_cfg.get("windows", [20, 50])
            if not isinstance(donch_windows, (list, tuple)):
                donch_windows = [20, 50]
            donch_windows = [int(w) for w in donch_windows if int(w) >= 5]

            for donch_w in donch_windows:
                donch_high = high.rolling(window=donch_w, min_periods=max(5, donch_w // 4)).max()
                donch_low = low.rolling(window=donch_w, min_periods=max(5, donch_w // 4)).min()
                donch_mid = (donch_high + donch_low) / 2.0
                donch_width = donch_high - donch_low

                # Position in channel (0-1 range)
                out[f"reg_ohlcv__donch_position_w{donch_w}"] = ((close - donch_low) / (donch_width + 1e-12)).clip(0.0, 1.0)
                # Width normalized by ATR
                out[f"reg_ohlcv__donch_width_atr_w{donch_w}"] = (donch_width / (atr + 1e-12)).replace([np.inf, -np.inf], np.nan)
                # Distance from channel midpoint (normalized)
                out[f"reg_ohlcv__donch_mid_dist_w{donch_w}"] = ((close - donch_mid) / (donch_width + 1e-12)).clip(-1.0, 1.0)
                # Breakout signals: close at/above high or at/below low
                out[f"reg_ohlcv__donch_at_high_w{donch_w}"] = ((close >= donch_high * 0.998) & (donch_width > 0)).astype(float)
                out[f"reg_ohlcv__donch_at_low_w{donch_w}"] = ((close <= donch_low * 1.002) & (donch_width > 0)).astype(float)

        except Exception:
            for w in [20, 50]:
                out[f"reg_ohlcv__donch_position_w{w}"] = np.nan
                out[f"reg_ohlcv__donch_width_atr_w{w}"] = np.nan
                out[f"reg_ohlcv__donch_mid_dist_w{w}"] = np.nan
                out[f"reg_ohlcv__donch_at_high_w{w}"] = 0.0
                out[f"reg_ohlcv__donch_at_low_w{w}"] = 0.0

        # --- Trend Persistence Features ---
        # Consecutive up/down bars, streak strength
        try:
            # Up/Down classification
            is_up = (close > close.shift(1)).astype(int)
            is_down = (close < close.shift(1)).astype(int)

            # Count consecutive up bars
            up_reset = (is_up != is_up.shift(1)).cumsum()
            consec_up = is_up.groupby(up_reset).cumsum()
            consec_up = consec_up.where(is_up == 1, 0)

            # Count consecutive down bars
            down_reset = (is_down != is_down.shift(1)).cumsum()
            consec_down = is_down.groupby(down_reset).cumsum()
            consec_down = consec_down.where(is_down == 1, 0)

            out["reg_ohlcv__consec_up_bars"] = consec_up.astype(float)
            out["reg_ohlcv__consec_down_bars"] = consec_down.astype(float)
            out["reg_ohlcv__consec_net_bars"] = (consec_up - consec_down).astype(float)

            # Streak strength: consecutive count weighted by cumulative return
            streak_ret_up = ret.where(is_up == 1, 0.0).groupby(up_reset).cumsum()
            streak_ret_down = ret.where(is_down == 1, 0.0).groupby(down_reset).cumsum()
            out["reg_ohlcv__streak_strength_up"] = (consec_up * streak_ret_up.abs()).fillna(0.0)
            out["reg_ohlcv__streak_strength_down"] = (consec_down * streak_ret_down.abs()).fillna(0.0)

            # Rolling trend persistence: how often are consecutive moves > N bars
            for persist_w in [24, 96]:
                long_streak = ((consec_up >= 3) | (consec_down >= 3)).astype(float)
                out[f"reg_ohlcv__trend_persist_rate_w{persist_w}"] = long_streak.rolling(
                    window=persist_w, min_periods=max(10, persist_w // 4)
                ).mean()

        except Exception:
            out["reg_ohlcv__consec_up_bars"] = 0.0
            out["reg_ohlcv__consec_down_bars"] = 0.0
            out["reg_ohlcv__consec_net_bars"] = 0.0
            out["reg_ohlcv__streak_strength_up"] = 0.0
            out["reg_ohlcv__streak_strength_down"] = 0.0
            out["reg_ohlcv__trend_persist_rate_w24"] = 0.0
            out["reg_ohlcv__trend_persist_rate_w96"] = 0.0

        # --- Higher-Timeframe Trend ---
        # 4h and 1d equivalent trend signals using rolling windows
        try:
            htf_cfg = cfg.get("higher_tf_trend", {})
            if not isinstance(htf_cfg, dict):
                htf_cfg = {}
            htf_windows = htf_cfg.get("windows", [16, 96])  # 4h = 16 bars, 1d = 96 bars at 15m
            if not isinstance(htf_windows, (list, tuple)):
                htf_windows = [16, 96]
            htf_windows = [int(w) for w in htf_windows if int(w) >= 4]

            for htf_w in htf_windows:
                # HTF close approximation (last close in window)
                htf_close_start = close.shift(htf_w)
                htf_return = ((close - htf_close_start) / (htf_close_start.abs() + 1e-12)).replace([np.inf, -np.inf], np.nan)
                htf_direction = np.sign(htf_return)

                out[f"reg_ohlcv__htf_return_w{htf_w}"] = htf_return
                out[f"reg_ohlcv__htf_direction_w{htf_w}"] = htf_direction

                # HTF trend strength: HTF return / ATR (momentum normalized by vol)
                htf_atr = atr.rolling(window=htf_w, min_periods=max(5, htf_w // 4)).mean()
                out[f"reg_ohlcv__htf_trend_strength_w{htf_w}"] = (htf_return.abs() / (htf_atr / close.abs() + 1e-12)).clip(0.0, 10.0)

        except Exception:
            for w in [16, 96]:
                out[f"reg_ohlcv__htf_return_w{w}"] = np.nan
                out[f"reg_ohlcv__htf_direction_w{w}"] = 0.0
                out[f"reg_ohlcv__htf_trend_strength_w{w}"] = 0.0

        # --- Direction Dominance (for predicting trend efficiency) ---
        # Measures how one-sided recent moves have been
        try:
            dom_cfg = cfg.get("direction_dominance", {})
            if not isinstance(dom_cfg, dict):
                dom_cfg = {}
            dom_windows = dom_cfg.get("windows", [8, 16, 24])  # Match macro_trend horizons
            if not isinstance(dom_windows, (list, tuple)):
                dom_windows = [8, 16, 24]
            dom_windows = [int(w) for w in dom_windows if int(w) >= 4]

            for dom_w in dom_windows:
                # Count up vs down moves
                n_up = (ret > 0).astype(float).rolling(window=dom_w, min_periods=max(3, dom_w // 4)).sum()
                n_down = (ret < 0).astype(float).rolling(window=dom_w, min_periods=max(3, dom_w // 4)).sum()

                # Direction dominance: |n_up - n_down| / total moves (0 = choppy, 1 = trending)
                total_moves = n_up + n_down
                out[f"reg_ohlcv__direction_dominance_w{dom_w}"] = ((n_up - n_down).abs() / (total_moves + 1e-12)).clip(0.0, 1.0)

                # Signed direction bias (-1 = bearish, +1 = bullish)
                out[f"reg_ohlcv__direction_bias_w{dom_w}"] = ((n_up - n_down) / (total_moves + 1e-12)).clip(-1.0, 1.0)

                # Up/Down ratio
                out[f"reg_ohlcv__up_down_ratio_w{dom_w}"] = (n_up / (n_down + 1e-12)).clip(0.0, 10.0)

        except Exception:
            for w in [8, 16, 24]:
                out[f"reg_ohlcv__direction_dominance_w{w}"] = 0.5
                out[f"reg_ohlcv__direction_bias_w{w}"] = 0.0
                out[f"reg_ohlcv__up_down_ratio_w{w}"] = 1.0

        # --- Trend Stability Features ---
        # Predict whether current trend will persist
        try:
            stab_cfg = cfg.get("trend_stability", {})
            if not isinstance(stab_cfg, dict):
                stab_cfg = {}
            stab_windows = stab_cfg.get("windows", [16, 24])
            if not isinstance(stab_windows, (list, tuple)):
                stab_windows = [16, 24]
            stab_windows = [int(w) for w in stab_windows if int(w) >= 8]

            for stab_w in stab_windows:
                # Return autocorrelation (high = trending, low = mean-reverting)
                ret_ac = ret.rolling(window=stab_w, min_periods=max(5, stab_w // 4)).corr(ret.shift(1))
                out[f"reg_ohlcv__ret_autocorr_w{stab_w}"] = ret_ac.fillna(0.0).clip(-1.0, 1.0)

                # Volatility-adjusted momentum (trend strength adjusted for noise)
                mom = close.pct_change(stab_w)
                vol = ret.rolling(window=stab_w, min_periods=max(5, stab_w // 4)).std()
                out[f"reg_ohlcv__vol_adj_momentum_w{stab_w}"] = (mom / (vol + 1e-12)).clip(-5.0, 5.0)

                # Price path linearity (R-squared of price vs time)
                # High R² = smooth trend, Low R² = choppy
                def rolling_r2(x):
                    if len(x) < 3:
                        return np.nan
                    t = np.arange(len(x))
                    if np.std(x) < 1e-12:
                        return 0.0
                    corr = np.corrcoef(t, x)[0, 1]
                    return corr ** 2 if np.isfinite(corr) else 0.0

                out[f"reg_ohlcv__price_linearity_w{stab_w}"] = close.rolling(window=stab_w, min_periods=max(5, stab_w // 4)).apply(rolling_r2, raw=True).fillna(0.0)

        except Exception:
            for w in [16, 24]:
                out[f"reg_ohlcv__ret_autocorr_w{w}"] = 0.0
                out[f"reg_ohlcv__vol_adj_momentum_w{w}"] = 0.0
                out[f"reg_ohlcv__price_linearity_w{w}"] = 0.0

        # --- Path Efficiency Change (delta of efficiency ratio) ---
        # Predicts if market is becoming more or less trendy
        try:
            eff_val = pd.to_numeric(out.get("reg_ohlcv__efficiency_ratio"), errors="coerce")
            if eff_val is not None:
                out["reg_ohlcv__efficiency_delta_1"] = eff_val.diff(1)
                out["reg_ohlcv__efficiency_delta_4"] = eff_val.diff(4)
                out["reg_ohlcv__efficiency_acceleration"] = eff_val.diff(1).diff(1)  # 2nd derivative
        except Exception:
            out["reg_ohlcv__efficiency_delta_1"] = 0.0
            out["reg_ohlcv__efficiency_delta_4"] = 0.0
            out["reg_ohlcv__efficiency_acceleration"] = 0.0

        # --- Momentum-Based Features for Trend Regimes ---
        # Added to improve trend regime prediction
        try:
            mom_cfg = cfg.get("momentum_features", {})
            if not isinstance(mom_cfg, dict):
                mom_cfg = {}
            mom_windows = mom_cfg.get("windows", [8, 16, 32])
            if not isinstance(mom_windows, (list, tuple)):
                mom_windows = [8, 16, 32]
            mom_windows = [int(w) for w in mom_windows if int(w) >= 4]

            for mom_w in mom_windows:
                # Momentum (rate of change)
                mom = (close / close.shift(mom_w) - 1.0).replace([np.inf, -np.inf], np.nan)
                out[f"reg_ohlcv__momentum_w{mom_w}"] = mom.clip(-0.5, 0.5)

                # Momentum acceleration (change in momentum)
                mom_prev = (close.shift(mom_w) / close.shift(2 * mom_w) - 1.0).replace([np.inf, -np.inf], np.nan)
                mom_accel = (mom - mom_prev).fillna(0.0)
                out[f"reg_ohlcv__momentum_accel_w{mom_w}"] = mom_accel.clip(-0.2, 0.2)

                # Momentum consistency (ratio of returns in same direction)
                ret_signs = np.sign(ret)
                sign_consistency = ret_signs.rolling(window=mom_w, min_periods=max(3, mom_w // 4)).apply(
                    lambda x: np.abs(np.sum(x)) / (len(x) + 1e-12), raw=True
                )
                out[f"reg_ohlcv__momentum_consistency_w{mom_w}"] = sign_consistency.fillna(0.0).clip(0.0, 1.0)

            # Multi-horizon momentum divergence (short vs long momentum difference)
            if len(mom_windows) >= 2:
                short_w = min(mom_windows)
                long_w = max(mom_windows)
                mom_short = (close / close.shift(short_w) - 1.0).replace([np.inf, -np.inf], np.nan)
                mom_long = (close / close.shift(long_w) - 1.0).replace([np.inf, -np.inf], np.nan)
                out["reg_ohlcv__momentum_divergence"] = (mom_short - mom_long).fillna(0.0).clip(-0.3, 0.3)

                # Momentum alignment (sign agreement between short/long)
                out["reg_ohlcv__momentum_alignment"] = (np.sign(mom_short) == np.sign(mom_long)).astype(float)

        except Exception:
            for w in [8, 16, 32]:
                out[f"reg_ohlcv__momentum_w{w}"] = 0.0
                out[f"reg_ohlcv__momentum_accel_w{w}"] = 0.0
                out[f"reg_ohlcv__momentum_consistency_w{w}"] = 0.0
            out["reg_ohlcv__momentum_divergence"] = 0.0
            out["reg_ohlcv__momentum_alignment"] = 0.5

        # --- Order Flow Proxies (OHLCV-based) ---
        # Microstructure signals without order book data
        try:
            oflow_cfg = cfg.get("order_flow_proxies", {})
            if not isinstance(oflow_cfg, dict):
                oflow_cfg = {}
            oflow_windows = oflow_cfg.get("windows", [8, 24, 48])
            if not isinstance(oflow_windows, (list, tuple)):
                oflow_windows = [8, 24, 48]
            oflow_windows = [int(w) for w in oflow_windows if int(w) >= 4]

            # 1. Trade Imbalance Proxy (Close position in candle range)
            candle_range = high - low
            close_position = (close - low) / (candle_range + 1e-12)  # 0-1, 1 = buying pressure
            out["reg_ohlcv__close_position"] = close_position.clip(0.0, 1.0)

            # 2. Volume-Weighted Price Pressure
            if volume is not None:
                # Buying volume proxy: (close-low)/(high-low) * volume
                buy_vol_proxy = close_position * volume
                sell_vol_proxy = (1.0 - close_position) * volume
                out["reg_ohlcv__buy_sell_ratio"] = (buy_vol_proxy / (sell_vol_proxy + 1e-12)).clip(0.0, 10.0)

            for oflow_w in oflow_windows:
                # 3. Cumulative Volume Delta Proxy (CVD)
                cvd = (close_position - 0.5) * volume if volume is not None else (close_position - 0.5)
                cvd_rolling = cvd.rolling(window=oflow_w, min_periods=max(3, oflow_w // 4)).sum()
                out[f"reg_ohlcv__cvd_proxy_w{oflow_w}"] = cvd_rolling.fillna(0.0)

                # 4. CVD momentum (speed of accumulation/distribution)
                if volume is not None:
                    cvd_mom = cvd_rolling.diff(max(4, oflow_w // 6))
                    out[f"reg_ohlcv__cvd_momentum_w{oflow_w}"] = cvd_mom.fillna(0.0)

                # 5. Price-Volume Divergence (price up, volume down = weak)
                price_trend = close.pct_change(oflow_w)
                vol_trend = volume.pct_change(oflow_w) if volume is not None else pd.Series(0.0, index=close.index)
                pv_diverg = np.sign(price_trend) * np.sign(vol_trend)  # -1 = divergence
                out[f"reg_ohlcv__pv_divergence_w{oflow_w}"] = pv_diverg.fillna(0.0)

                # 6. Absorption Ratio (large candle with little price change = absorption)
                price_change = close.diff(oflow_w).abs()
                vol_sum = volume.rolling(window=oflow_w, min_periods=max(3, oflow_w // 4)).sum() if volume is not None else pd.Series(1.0, index=close.index)
                absorption = vol_sum / (price_change + 1e-12)
                out[f"reg_ohlcv__absorption_w{oflow_w}"] = absorption.clip(0.0, 1e6).replace([np.inf, -np.inf], np.nan).fillna(0.0)

                # 7. Intrabar Volatility (wick-to-body ratio as uncertainty proxy)
                body = (close - open_s).abs() if open_s is not None else close.diff().abs()
                wick = candle_range - body
                wick_ratio = wick / (candle_range + 1e-12)
                out[f"reg_ohlcv__wick_ratio_ema_w{oflow_w}"] = wick_ratio.ewm(span=oflow_w).mean().clip(0.0, 1.0)

        except Exception:
            out["reg_ohlcv__close_position"] = 0.5
            out["reg_ohlcv__buy_sell_ratio"] = 1.0
            for w in [8, 24, 48]:
                out[f"reg_ohlcv__cvd_proxy_w{w}"] = 0.0
                out[f"reg_ohlcv__cvd_momentum_w{w}"] = 0.0
                out[f"reg_ohlcv__pv_divergence_w{w}"] = 0.0
                out[f"reg_ohlcv__absorption_w{w}"] = 0.0
                out[f"reg_ohlcv__wick_ratio_ema_w{w}"] = 0.5

        # --- Recency-Weighted Rolling Statistics (EWM) ---
        # Exponential weighted features emphasizing recent behavior
        try:
            ewm_cfg = cfg.get("ewm_features", {})
            if not isinstance(ewm_cfg, dict):
                ewm_cfg = {}
            ewm_halflifes = ewm_cfg.get("halflifes", [12, 24, 48])  # In bars
            if not isinstance(ewm_halflifes, (list, tuple)):
                ewm_halflifes = [12, 24, 48]
            ewm_halflifes = [int(h) for h in ewm_halflifes if int(h) >= 4]

            for hl in ewm_halflifes:
                # EWM volatility
                ewm_vol = ret.ewm(halflife=hl).std()
                out[f"reg_ohlcv__ewm_vol_h{hl}"] = ewm_vol.fillna(0.0)

                # EWM trend (price vs EWM mean)
                ewm_mean = close.ewm(halflife=hl).mean()
                ewm_trend = (close - ewm_mean) / (ewm_vol * close + 1e-12)
                out[f"reg_ohlcv__ewm_trend_h{hl}"] = ewm_trend.clip(-5.0, 5.0).fillna(0.0)

                # EWM momentum
                ewm_ret = ret.ewm(halflife=hl).mean()
                out[f"reg_ohlcv__ewm_momentum_h{hl}"] = ewm_ret.fillna(0.0)

            # Fast vs Slow EWM Divergence (MACD-like)
            if len(ewm_halflifes) >= 2:
                fast_hl = min(ewm_halflifes)
                slow_hl = max(ewm_halflifes)
                ewm_fast = close.ewm(halflife=fast_hl).mean()
                ewm_slow = close.ewm(halflife=slow_hl).mean()
                ewm_div = (ewm_fast - ewm_slow) / (close * 0.01 + 1e-12)  # Normalize by 1%
                out["reg_ohlcv__ewm_divergence"] = ewm_div.clip(-10.0, 10.0).fillna(0.0)

                # EWM divergence momentum
                out["reg_ohlcv__ewm_divergence_momentum"] = ewm_div.diff(4).fillna(0.0)

        except Exception:
            for h in [12, 24, 48]:
                out[f"reg_ohlcv__ewm_vol_h{h}"] = 0.0
                out[f"reg_ohlcv__ewm_trend_h{h}"] = 0.0
                out[f"reg_ohlcv__ewm_momentum_h{h}"] = 0.0
            out["reg_ohlcv__ewm_divergence"] = 0.0
            out["reg_ohlcv__ewm_divergence_momentum"] = 0.0

        # --- Regime Age / Persistence ---
        # How long current regime has lasted
        try:
            regime_cfg = cfg.get("regime_age", {})
            if not isinstance(regime_cfg, dict):
                regime_cfg = {}

            # Volatility regime age
            vol_q = atr.rolling(window=96, min_periods=24).quantile(0.5)
            vol_regime = (atr > vol_q).astype(int)
            vol_regime_change = (vol_regime != vol_regime.shift(1)).astype(int)
            vol_regime_age = vol_regime_change.groupby(vol_regime_change.cumsum()).cumcount()
            out["reg_ohlcv__vol_regime_age"] = vol_regime_age.astype(float)

            # Trend regime age (based on price vs SMA)
            sma_trend = close.rolling(window=24, min_periods=8).mean()
            trend_regime = (close > sma_trend).astype(int)
            trend_regime_change = (trend_regime != trend_regime.shift(1)).astype(int)
            trend_regime_age = trend_regime_change.groupby(trend_regime_change.cumsum()).cumcount()
            out["reg_ohlcv__trend_regime_age"] = trend_regime_age.astype(float)

            # Normalized regime age (0 = just changed, 1 = typical duration)
            expected_regime_duration = 24.0  # bars
            out["reg_ohlcv__vol_regime_age_norm"] = (vol_regime_age / expected_regime_duration).clip(0.0, 3.0)
            out["reg_ohlcv__trend_regime_age_norm"] = (trend_regime_age / expected_regime_duration).clip(0.0, 3.0)

            # Regime stability (fewer changes = more stable)
            regime_changes_24 = vol_regime_change.rolling(window=24, min_periods=8).sum()
            out["reg_ohlcv__regime_change_rate_24"] = regime_changes_24.fillna(0.0)

        except Exception:
            out["reg_ohlcv__vol_regime_age"] = 0.0
            out["reg_ohlcv__trend_regime_age"] = 0.0
            out["reg_ohlcv__vol_regime_age_norm"] = 0.0
            out["reg_ohlcv__trend_regime_age_norm"] = 0.0
            out["reg_ohlcv__regime_change_rate_24"] = 0.0

        # --- Temporal Momentum / Feature Velocity ---
        # Speed of change in key market indicators
        try:
            vel_cfg = cfg.get("feature_velocity", {})
            if not isinstance(vel_cfg, dict):
                vel_cfg = {}
            vel_lookback = int(vel_cfg.get("lookback", 4))

            # Volatility velocity (is vol increasing or decreasing?)
            vol_velocity = atr.pct_change(vel_lookback).replace([np.inf, -np.inf], np.nan)
            out["reg_ohlcv__vol_velocity"] = vol_velocity.clip(-2.0, 2.0).fillna(0.0)

            # Volatility acceleration (is velocity changing?)
            vol_accel = vol_velocity.diff(vel_lookback)
            out["reg_ohlcv__vol_acceleration"] = vol_accel.clip(-1.0, 1.0).fillna(0.0)

            # Trend velocity (how fast is the trend changing?)
            trend_sma = close.rolling(window=24, min_periods=8).mean()
            trend_dist = close - trend_sma
            trend_velocity = trend_dist.diff(vel_lookback) / (atr + 1e-12)
            out["reg_ohlcv__trend_velocity"] = trend_velocity.clip(-5.0, 5.0).fillna(0.0)

            # Volume velocity
            if volume is not None:
                vol_sma = volume.rolling(window=24, min_periods=8).mean()
                vol_vol_velocity = vol_sma.pct_change(vel_lookback).replace([np.inf, -np.inf], np.nan)
                out["reg_ohlcv__volume_velocity"] = vol_vol_velocity.clip(-2.0, 2.0).fillna(0.0)
            else:
                out["reg_ohlcv__volume_velocity"] = 0.0

        except Exception:
            out["reg_ohlcv__vol_velocity"] = 0.0
            out["reg_ohlcv__vol_acceleration"] = 0.0
            out["reg_ohlcv__trend_velocity"] = 0.0
            out["reg_ohlcv__volume_velocity"] = 0.0

        # --- Lookback Decay Statistics ---
        # Compare short vs long lookback to detect regime shifts
        try:
            decay_cfg = cfg.get("lookback_decay", {})
            if not isinstance(decay_cfg, dict):
                decay_cfg = {}

            short_w = int(decay_cfg.get("short", 8))
            long_w = int(decay_cfg.get("long", 48))

            # Volatility ratio (short/long)
            vol_short = ret.rolling(window=short_w, min_periods=max(3, short_w // 2)).std()
            vol_long = ret.rolling(window=long_w, min_periods=max(10, long_w // 2)).std()
            vol_ratio_sl = vol_short / (vol_long + 1e-12)
            out["reg_ohlcv__vol_ratio_short_long"] = vol_ratio_sl.clip(0.1, 5.0).fillna(1.0)

            # Return mean ratio (recent vs long-term)
            ret_short = ret.rolling(window=short_w, min_periods=max(3, short_w // 2)).mean()
            ret_long = ret.rolling(window=long_w, min_periods=max(10, long_w // 2)).mean()
            ret_divergence = ret_short - ret_long
            out["reg_ohlcv__ret_divergence_sl"] = ret_divergence.fillna(0.0)

            # Efficiency ratio comparison
            eff_short = close.diff(short_w).abs() / (close.diff().abs().rolling(window=short_w, min_periods=3).sum() + 1e-12)
            eff_long = close.diff(long_w).abs() / (close.diff().abs().rolling(window=long_w, min_periods=10).sum() + 1e-12)
            eff_ratio = eff_short / (eff_long + 1e-12)
            out["reg_ohlcv__efficiency_ratio_sl"] = eff_ratio.clip(0.1, 5.0).fillna(1.0)

            # Correlation with recent data (rolling correlation of returns with time index)
            # High = trending, Low = mean-reverting
            recent_w = 24
            time_idx = pd.Series(np.arange(len(close)), index=close.index).astype(float)
            time_corr = close.rolling(window=recent_w, min_periods=max(10, recent_w // 2)).corr(time_idx)
            out["reg_ohlcv__recency_trend_corr"] = time_corr.clip(-1.0, 1.0).fillna(0.0)

        except Exception:
            out["reg_ohlcv__vol_ratio_short_long"] = 1.0
            out["reg_ohlcv__ret_divergence_sl"] = 0.0
            out["reg_ohlcv__efficiency_ratio_sl"] = 1.0
            out["reg_ohlcv__recency_trend_corr"] = 0.0

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
        # Key state deltas (causal)
        try:
            out["reg_ohlcv__d_volatility_1d"] = pd.to_numeric(out.get("reg_ohlcv__volatility_1d"), errors="coerce").diff()
        except Exception:
            out["reg_ohlcv__d_volatility_1d"] = np.nan
        try:
            out["reg_ohlcv__d_vol_ratio"] = pd.to_numeric(out.get("reg_ohlcv__vol_ratio"), errors="coerce").diff()
        except Exception:
            out["reg_ohlcv__d_vol_ratio"] = np.nan

        # Regime Transition Features (Efficiency & Autocorr Derivatives)
        try:
            # 1. Volatility Acceleration (2nd derivative)
            out["reg_ohlcv__vol_acceleration_1d"] = pd.to_numeric(out["reg_ohlcv__d_volatility_1d"], errors="coerce").diff()

            # 2. Efficiency and Autocorr Deltas
            # Compute rolling features first if needed
            eff_window = 24

            # Efficiency (Kaufman-like): Abs(TotalMove) / Sum(Abs(Moves))
            # Directional Move over W
            dir_move = close.diff(eff_window).abs()
            # Path Length over W (sum of 1-bar abs diffs)
            path_len = close.diff().abs().rolling(window=eff_window, min_periods=max(5, eff_window//2)).sum()
            eff_val = (dir_move / (path_len + 1e-12)).clip(0.0, 1.0)

            out[f"reg_ohlcv__efficiency_w{eff_window}"] = eff_val
            out[f"reg_ohlcv__d_efficiency_w{eff_window}"] = eff_val.diff()

            # Autocorrelation (Market Memory)
            # Rolling 1-lag autocorrelation of returns
            mem_window = 24
            auto_corr = ret.rolling(window=mem_window, min_periods=max(10, mem_window//2)).corr(ret.shift(1)).fillna(0.0).clip(-0.99, 0.99)

            out[f"reg_ohlcv__autocorr_w{mem_window}"] = auto_corr
            out[f"reg_ohlcv__d_autocorr_w{mem_window}"] = auto_corr.diff()

        except Exception:
            pass

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

            # New Contextual Features (Dist from Open, Session Pos)
            # Distance from Daily Open (approximate using resampling)
            try:
                # Resample to daily open prices
                daily_open = close.resample('D').first().reindex(close.index).ffill()
                out["reg_ohlcv__dist_from_open_day"] = (close / (daily_open + 1e-8) - 1.0).astype(float)

                # Distance from Weekly Open
                weekly_open = close.resample('W').first().reindex(close.index).ffill()
                out["reg_ohlcv__dist_from_open_week"] = (close / (weekly_open + 1e-8) - 1.0).astype(float)

                # Session Position (0.0 to 1.0)
                # minute of day / 1440
                minutes = out.index.hour * 60 + out.index.minute
                out["reg_ohlcv__session_pos_24h"] = (minutes / 1440.0).astype(float)
            except Exception:
                out["reg_ohlcv__dist_from_open_day"] = 0.0
                out["reg_ohlcv__dist_from_open_week"] = 0.0
                out["reg_ohlcv__session_pos_24h"] = 0.0

        else:
            out["reg_ohlcv__hour"] = 0.0
            out["reg_ohlcv__day_of_week"] = 0.0
            out["reg_ohlcv__hour_sin"] = 0.0
            out["reg_ohlcv__hour_cos"] = 1.0
            out["reg_ohlcv__is_good_hour"] = 0.0
            out["reg_ohlcv__is_bad_hour"] = 0.0
            out["reg_ohlcv__is_sunday"] = 0.0
            out["reg_ohlcv__dist_from_open_day"] = 0.0
            out["reg_ohlcv__dist_from_open_week"] = 0.0
            out["reg_ohlcv__session_pos_24h"] = 0.0
    except Exception:
        out["reg_ohlcv__hour"] = 0.0
        out["reg_ohlcv__day_of_week"] = 0.0
        out["reg_ohlcv__hour_sin"] = 0.0
        out["reg_ohlcv__hour_cos"] = 1.0
        out["reg_ohlcv__is_good_hour"] = 0.0
        out["reg_ohlcv__is_bad_hour"] = 0.0
        out["reg_ohlcv__is_sunday"] = 0.0


    # =========================================================================
    # REGIME QUALITY & DEGRADATION FEATURES
    # These features help identify "weak" regime combinations where trading
    # performance is historically poor (e.g., vol_low + trend_high)
    # =========================================================================
    try:
        enable_regime_quality = bool(cfg.get("enable_regime_quality_features", True))
    except Exception:
        enable_regime_quality = True

    if enable_regime_quality:
        try:
            # 1. Volatility-Trend Mismatch Score
            # Low vol + high trend is suspicious: real trends have confirming volatility
            vol_1d = out.get("reg_ohlcv__volatility_1d")
            trend_efficiency = out.get("reg_ohlcv__efficiency_ratio")

            if vol_1d is not None and trend_efficiency is not None:
                vol_1d = pd.to_numeric(vol_1d, errors="coerce")
                trend_efficiency = pd.to_numeric(trend_efficiency, errors="coerce")

                # Rolling percentile ranks for contextualization
                vol_pctl = vol_1d.rolling(window=192, min_periods=48).apply(
                    lambda x: (x.iloc[-1] <= x).mean() if len(x) > 0 else 0.5, raw=False
                )
                trend_pctl = trend_efficiency.rolling(window=192, min_periods=48).apply(
                    lambda x: (x.iloc[-1] <= x).mean() if len(x) > 0 else 0.5, raw=False
                )

                # Mismatch: low vol (< 0.33 pctl) + high trend (> 0.66 pctl)
                vol_low_flag = (vol_pctl < 0.33).astype(float)
                trend_high_flag = (trend_pctl > 0.66).astype(float)
                mismatch = vol_low_flag * trend_high_flag
                out["reg_ohlcv__vol_trend_mismatch"] = mismatch

                # Mismatch severity (continuous)
                out["reg_ohlcv__vol_trend_mismatch_score"] = (
                    (0.33 - vol_pctl).clip(lower=0.0) * (trend_pctl - 0.66).clip(lower=0.0) * 10.0
                ).fillna(0.0)

            # 2. Bars-in-Weak-Regime Counter
            # Count consecutive bars where volatility is low and trend is high
            if "reg_ohlcv__vol_trend_mismatch" in out.columns:
                mismatch_flag = out["reg_ohlcv__vol_trend_mismatch"]
                # Cumulative counter of consecutive weak regime bars
                # Reset on transitions
                group = (mismatch_flag != mismatch_flag.shift()).cumsum()
                bars_counter = mismatch_flag.groupby(group).cumsum()
                out["reg_ohlcv__bars_in_weak_regime"] = bars_counter

            # 3. Volatility Trend Within Low-Vol Regime
            # Rising volatility in a low-vol regime is a warning sign
            if vol_1d is not None:
                vol_slope_48 = out.get("reg_ohlcv__vol_ewm_slope_w48")
                if vol_slope_48 is not None:
                    vol_slope_48 = pd.to_numeric(vol_slope_48, errors="coerce")
                    # Warning: rising vol in low-vol regime
                    vol_rising_in_low = vol_low_flag * (vol_slope_48 > 0.0).astype(float)
                    out["reg_ohlcv__vol_rising_in_low_regime"] = vol_rising_in_low

            # 4. Regime Transition Recency
            # How many bars since the last regime transition?
            try:
                vol_regime_med = out.get("reg_ohlcv__vol_regime_medium")
                vol_regime_high = out.get("reg_ohlcv__vol_regime_high")
                if vol_regime_med is not None and vol_regime_high is not None:
                    vol_state = (
                        pd.to_numeric(vol_regime_high, errors="coerce").fillna(0.0) * 2 +
                        pd.to_numeric(vol_regime_med, errors="coerce").fillna(0.0)
                    )
                    transitions = (vol_state != vol_state.shift()).astype(float)
                    out["reg_ohlcv__is_regime_transition"] = transitions

                    # Bars since last transition
                    group_trans = transitions.cumsum()
                    bars_since_trans = vol_state.groupby(group_trans).cumcount()
                    out["reg_ohlcv__bars_since_regime_transition"] = bars_since_trans.astype(float)

                    # Normalize to 0-1 scale (capped at 96 bars = ~1 day)
                    out["reg_ohlcv__regime_stability"] = (bars_since_trans / 96.0).clip(upper=1.0)
            except Exception:
                pass

            # 5. Multi-Horizon Regime Agreement
            # Do short/medium/long volatility windows agree on the regime?
            vol_8 = out.get("reg_ohlcv__rv_std_logret_w8")
            vol_24 = out.get("reg_ohlcv__rv_std_logret_w24")
            vol_96 = out.get("reg_ohlcv__rv_std_logret_w96")
            if vol_8 is not None and vol_24 is not None and vol_96 is not None:
                vol_8 = pd.to_numeric(vol_8, errors="coerce")
                vol_24 = pd.to_numeric(vol_24, errors="coerce")
                vol_96 = pd.to_numeric(vol_96, errors="coerce")

                # Agreement: all three trending same direction
                dir_8 = np.sign(vol_8.diff())
                dir_24 = np.sign(vol_24.diff())
                dir_96 = np.sign(vol_96.diff())

                agreement = ((dir_8 == dir_24) & (dir_24 == dir_96)).astype(float)
                out["reg_ohlcv__multi_horizon_vol_agreement"] = agreement

                # Disagreement score
                disagreement = (
                    (dir_8 != dir_24).astype(float) +
                    (dir_24 != dir_96).astype(float) +
                    (dir_8 != dir_96).astype(float)
                ) / 3.0
                out["reg_ohlcv__multi_horizon_vol_disagreement"] = disagreement

        except Exception as regime_qual_exc:
            tprint_warning(f"Failed to compute regime quality features: {regime_qual_exc}")

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

    # =========================================================================
    # EXPERT FEATURES INTEGRATION (SMC, Liquidity, Volume Force)
    # Knowledge Distillation from specialized components
    # =========================================================================
    try:
        # SMC Features (Market Structure, FVG, Order Blocks)
        try:
            smc_feat = generate_smc_regime_features(market_data, cfg)
            # Ensure index alignment and no duplicates
            smc_feat = smc_feat.reindex(out.index)
            # Drop columns that might already exist (though unlikely due to prefixing)
            new_cols = [c for c in smc_feat.columns if c not in out.columns]
            if new_cols:
                out = pd.concat([out, smc_feat[new_cols]], axis=1)
        except Exception:
            pass

        # Liquidity Features (Regimes, Volume Profile)
        try:
            # Note: generate_liquidity_regime_features might produce many cols.
            # We assume it handles its own config extraction.
            liq_feat = generate_liquidity_regime_features(market_data, cfg)
            liq_feat = liq_feat.reindex(out.index)
            new_cols = [c for c in liq_feat.columns if c not in out.columns]
            if new_cols:
                out = pd.concat([out, liq_feat[new_cols]], axis=1)
        except Exception:
            pass

        # Volume Force Features (Impulse, Pressure)
        try:
            vf_feat = generate_volume_force_features(market_data, cfg)
            vf_feat = vf_feat.reindex(out.index)
            new_cols = [c for c in vf_feat.columns if c not in out.columns]
            if new_cols:
                out = pd.concat([out, vf_feat[new_cols]], axis=1)
        except Exception:
            pass

        # Temporal Conv Embeddings (Neural Network features for temporal patterns)
        enable_temporal = bool(cfg.get("enable_temporal_embeddings", False))
        if enable_temporal and _TEMPORAL_ENCODER_AVAILABLE:
            try:
                temporal_cfg = cfg.get("temporal_encoder", {})
                if not isinstance(temporal_cfg, dict):
                    temporal_cfg = {}

                seq_len = int(temporal_cfg.get("seq_len", 24))
                embed_dim = int(temporal_cfg.get("embed_dim", 8))
                device = str(temporal_cfg.get("device", "cpu"))
                pretrained_path = temporal_cfg.get("pretrained_path", None)

                temporal_embed = generate_temporal_embeddings(
                    market_data=market_data,
                    seq_len=seq_len,
                    embed_dim=embed_dim,
                    device=device,
                    pretrained_path=pretrained_path,
                )

                if temporal_embed is not None and len(temporal_embed) > 0:
                    temporal_embed = temporal_embed.reindex(out.index)
                    new_cols = [c for c in temporal_embed.columns if c not in out.columns]
                    if new_cols:
                        out = pd.concat([out, temporal_embed[new_cols]], axis=1)
                        tprint_info(f"Added {len(new_cols)} temporal embedding features")
            except Exception as e:
                tprint_warning(f"Failed to generate temporal embeddings: {e}")

        # Stacked NN Sequence Embeddings (Conv + LSTM + optional Attention)
        enable_nn_sequence = bool(cfg.get("enable_nn_sequence_embeddings", False))
        if enable_nn_sequence and _NN_SEQUENCE_AVAILABLE:
            try:
                nn_cfg = cfg.get("nn_sequence_encoder", {})
                if not isinstance(nn_cfg, dict):
                    nn_cfg = {}

                seq_len = int(nn_cfg.get("seq_len", 24))
                embed_dim = int(nn_cfg.get("embed_dim", 8))
                device = str(nn_cfg.get("device", "cpu"))
                encoder_type = str(nn_cfg.get("encoder_type", "stacked"))
                use_conv = bool(nn_cfg.get("use_conv", True))
                use_lstm = bool(nn_cfg.get("use_lstm", True))
                use_attention = bool(nn_cfg.get("use_attention", False))
                pretrained_path = nn_cfg.get("pretrained_path", None)

                nn_embed = generate_nn_sequence_embeddings(
                    market_data=market_data,
                    encoder_type=encoder_type,
                    seq_len=seq_len,
                    embed_dim=embed_dim,
                    use_conv=use_conv,
                    use_lstm=use_lstm,
                    use_attention=use_attention,
                    device=device,
                    pretrained_path=pretrained_path,
                )

                if nn_embed is not None and len(nn_embed) > 0:
                    nn_embed = nn_embed.reindex(out.index)
                    new_cols = [c for c in nn_embed.columns if c not in out.columns]
                    if new_cols:
                        out = pd.concat([out, nn_embed[new_cols]], axis=1)
                        tprint_info(f"Added {len(new_cols)} stacked NN embedding features")
            except Exception as e:
                tprint_warning(f"Failed to generate NN sequence embeddings: {e}")

        # Drop duplicates just in case
        out = out.loc[:, ~out.columns.duplicated()]

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
            macro_trend_horizons = [8, 16, 24]  # 2h, 4h, 6h at 15m (reduced from [16,32,64])
        else:
            macro_trend_horizons = [int(macro_trend_horizon)]
    macro_trend_horizons = [int(h) for h in macro_trend_horizons if int(h) > 0]
    if not macro_trend_horizons:
        macro_trend_horizons = [16]  # 4h at 15m

    efficiency_window = int(config.get("trend_efficiency_window", 16))
    try:
        vf_horizon = config.get("volume_force_horizon")
        if vf_horizon is None:
            vf_horizon = config.get("volume_force_lookahead")
        vf_horizon = int(vf_horizon) if vf_horizon is not None else 16
    except Exception:
        vf_horizon = 16
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
    vol_target_mode = str(config.get("volatility_target_mode", "future_past_log_ratio")).lower()
    if vol_target_mode in {"future_past_log_ratio", "log_ratio", "ratio"}:
        targets["regime_volatility"] = _compute_future_past_vol_log_ratio(close, window=vol_window)
    else:
        targets["regime_volatility"] = _compute_future_volatility(close, window=vol_window)

    log_ret = np.log(close).diff()
    vol_base = log_ret.rolling(window=vol_window, min_periods=max(5, vol_window // 4)).std()

    for h in macro_trend_horizons:
        targets[f"regime_macro_trend_h{int(h)}"] = _compute_future_return(close, horizon=int(h))

    # Normalize macro trend target by volatility to ensure consistent scale for LGBM (avoiding L1/L2 regularization collapse)
    # CHANGED: Convert from regression to CLASSIFICATION (+1/0/-1) since exact return prediction has IC=-0.01
    # Classification predicts trend direction: +1 (bullish), 0 (neutral), -1 (bearish)
    h_max = int(max(macro_trend_horizons))
    try:
        raw_trend = targets[f"regime_macro_trend_h{h_max}"].astype(float)
    except Exception:
        raw_trend = _compute_future_return(close, horizon=h_max)

    # Scale: vol_base * sqrt(h) approximates vol over horizon h
    trend_zscore = (raw_trend / (vol_base * np.sqrt(float(h_max)) + 1e-12)).replace([np.inf, -np.inf], np.nan)

    # Convert to classification: thresholds at +/- 0.5 sigma (conservative)
    # +1 = bullish (zscore > 0.5), -1 = bearish (zscore < -0.5), 0 = neutral
    trend_class = pd.Series(0.0, index=close.index, dtype=float)
    trend_class = trend_class.where(trend_zscore.abs() <= 0.5, np.sign(trend_zscore))
    targets["regime_macro_trend"] = trend_class

    try:
        vf_ret = (np.log(close.replace(0.0, np.nan)) - np.log(close.shift(int(vf_horizon)).replace(0.0, np.nan))).shift(-int(vf_horizon))
        vf_vol = _compute_future_volatility(close, window=int(vf_horizon))
        vf = (vf_ret / (vf_vol * np.sqrt(float(max(1, int(vf_horizon)))) + 1e-12)).replace([np.inf, -np.inf], np.nan)
        targets["regime_volume_force_direction"] = vf.clip(-6.0, 6.0).astype(float)
    except Exception:
        targets["regime_volume_force_direction"] = np.nan

    # Trend Efficiency is [0,1], scale ~0.2. L1/L2 reg of 0.5 dampens it.
    # Normalize with rolling robust z-score (median/IQR) to boost scale to ~1.0
    eff_horizons = config.get("trend_efficiency_horizons")
    if isinstance(eff_horizons, (list, tuple)) and eff_horizons:
        eff_horizons = [int(h) for h in eff_horizons if h is not None and int(h) > 1]
    else:
        eff_horizons = [int(efficiency_window)]
    if not eff_horizons:
        eff_horizons = [int(efficiency_window)]

    eff_parts = []
    for h in eff_horizons:
        eff_parts.append(_compute_trend_efficiency(close, window=int(h)))
    if len(eff_parts) == 1:
        eff_raw = eff_parts[0]
    else:
        eff_raw = pd.concat(eff_parts, axis=1).median(axis=1)

    eff_transform = str(config.get("trend_efficiency_transform", "robust_zscore")).lower()
    if eff_transform in {"logit", "logit_zscore"}:
        eps = 1e-6
        eff_for_scale = np.log((eff_raw.clip(lower=0.0, upper=1.0) + eps) / (1.0 - eff_raw.clip(lower=0.0, upper=1.0) + eps))
    else:
        eff_for_scale = eff_raw

    eff_med = eff_for_scale.rolling(window=efficiency_window*4, min_periods=efficiency_window).median()
    eff_q75 = eff_for_scale.rolling(window=efficiency_window*4, min_periods=efficiency_window).quantile(0.75)
    eff_q25 = eff_for_scale.rolling(window=efficiency_window*4, min_periods=efficiency_window).quantile(0.25)
    eff_iqr = (eff_q75 - eff_q25).replace(0.0, np.nan)
    # Fallback to std if IQR is 0
    eff_std = eff_for_scale.rolling(window=efficiency_window*4, min_periods=efficiency_window).std()
    eff_scale = eff_iqr.fillna(eff_std).fillna(0.2) # Default scale 0.2

    targets["regime_trend_efficiency"] = ((eff_for_scale - eff_med) / (eff_scale + 1e-12)).clip(-5.0, 5.0).astype(float)

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

    # --- Breakout Target (from Volume Force) ---
    try:
        # Check if High/Low available
        if high_col in market_data.columns and low_col in market_data.columns:
            L_brk = 20
            H_brk = 12
            thresh_mult = 1.0

            # Use raw high/low
            h_s = pd.to_numeric(market_data[high_col], errors="coerce")
            l_s = pd.to_numeric(market_data[low_col], errors="coerce")

            past_h = h_s.rolling(L_brk).max()
            past_l = l_s.rolling(L_brk).min()

            tr_brk = np.maximum(h_s - l_s, (h_s - close.shift(1)).abs())
            atr_brk = tr_brk.rolling(14).mean()
            thresh_val = atr_brk * thresh_mult

            # Future max/min close - Checks if price SUSTAINS the break (close basis)
            f_max = close.shift(-H_brk).rolling(H_brk).max()
            f_min = close.shift(-H_brk).rolling(H_brk).min()

            # Binary Breakout (1.0 = Breakout, 0.0 = Range)
            is_brk = ((f_max > (past_h + thresh_val)) | (f_min < (past_l - thresh_val))).astype(float)
            targets["regime_breakout"] = is_brk
        else:
            targets["regime_breakout"] = np.nan
    except Exception:
        targets["regime_breakout"] = np.nan

    return targets


def _default_lgbm_params(config: dict, random_state: int, *, n_train_samples: Optional[int] = None) -> dict:
    # Increased defaults to produce more diverse leaves for regime detection
    # Previous values (6 leaves, depth 3) collapsed to single leaf OOS
    num_leaves = int(config.get("num_leaves", 15))
    max_depth = int(config.get("max_depth", 5))
    n_estimators = int(config.get("n_estimators", 80))
    learning_rate = float(config.get("learning_rate", 0.08))

    min_data_in_leaf = int(config.get("min_data_in_leaf", 30))
    try:
        n_train = int(n_train_samples) if n_train_samples is not None else None
    except Exception:
        n_train = None
    if n_train is not None and n_train > 0:
        try:
            # Allow smaller leaves (down to 1% of training data) for more granular regimes
            cap = int(max(10, round(0.01 * float(n_train))))
            min_data_in_leaf = int(max(10, min(int(min_data_in_leaf), int(cap), max(10, n_train - 1))))
        except Exception:
            min_data_in_leaf = int(max(10, min_data_in_leaf))

    min_gain_to_split = float(config.get("min_gain_to_split", 0.01))
    # Reduced regularization to allow more splits
    lambda_l1 = float(config.get("lambda_l1", 0.5))
    lambda_l2 = float(config.get("lambda_l2", 0.5))

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

    kept_set = set([int(x) for x in kept_leaf_ids])

    leaf_vals = pd.to_numeric(raw_leaf_series, errors="coerce")
    mask = leaf_vals.isin(kept_set)
    if not mask.any():
        return {}

    df_leaf = pd.DataFrame({'leaf': leaf_vals[mask], 'y': y_all[mask]})

    grouped = df_leaf.groupby('leaf')['y']
    stats = grouped.agg(['count', 'mean', 'std', 'min', 'max'])

    for li_float, row in stats.iterrows():
        li_int = int(li_float)

        y_stats = {
            "n": int(row['count']),
            "mean": float(row['mean']),
            "std": float(row['std']),
            "min": float(row['min']),
            "max": float(row['max']),
        }

        top_features = []
        if global_mean is not None and global_std is not None:
            try:
                leaf_mask = (leaf_vals == li_int)
                idx = X_num.index[leaf_mask]

                n_samples = len(idx)
                if n_samples > 0:
                    if n_samples > max_samples_per_leaf:
                        rs = np.random.RandomState(random_state + li_int)
                        choice = rs.choice(n_samples, size=max_samples_per_leaf, replace=False)
                        idx_sub = idx[choice]
                        X_leaf = X_num.reindex(idx_sub)
                    else:
                        X_leaf = X_num.reindex(idx)

                    leaf_mean = X_leaf.mean(numeric_only=True)
                    z = (leaf_mean - global_mean) / (global_std + 1e-12)
                    z = z.replace([np.inf, -np.inf], np.nan).dropna()

                    if not z.empty:
                        topk = z.abs().sort_values(ascending=False).head(max(0, top_features_per_leaf))
                        for feat, z_val in topk.items():
                            top_features.append({
                                "feature": str(feat),
                                "z_diff": float(z_val),
                                "leaf_mean": float(leaf_mean.get(feat)),
                                "global_mean": float(global_mean.get(feat)),
                            })
            except Exception:
                pass

        out[li_int] = {
            "n_rows": int(y_stats['n']),
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
    try:
        if "trend_efficiency_horizons" not in targets_cfg or targets_cfg.get("trend_efficiency_horizons") in (None, [], ()):
            te_w = targets_cfg.get("trend_efficiency_window", 16)
            te_w = int(te_w) if te_w is not None else 16
            targets_cfg["trend_efficiency_horizons"] = [max(2, te_w // 2), max(2, te_w), max(2, te_w * 2)]
    except Exception:
        pass
    try:
        if "trend_efficiency_transform" not in targets_cfg or targets_cfg.get("trend_efficiency_transform") in (None, ""):
            targets_cfg["trend_efficiency_transform"] = "logit"
    except Exception:
        pass
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

    align_cfg = dict(input_cfg.get("alignment", {})) if isinstance(input_cfg.get("alignment"), dict) else {}
    align_enabled = bool(align_cfg.get("enabled", True))
    align_method = str(align_cfg.get("method", "ffill")).lower()
    tolerance = None
    try:
        tolerance_cfg = align_cfg.get("tolerance")
        if tolerance_cfg is not None:
            tolerance = pd.to_timedelta(tolerance_cfg)
    except Exception:
        tolerance = None

    if align_enabled:
        targets = _align_frame_to_index(targets_full, X.index, method=align_method, tolerance=tolerance)
    else:
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

        if align_enabled:
            X_num = _align_frame_to_index(X_ohlcv, X.index, method=align_method, tolerance=tolerance)
        else:
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

    try:
        X_numeric = X_num.select_dtypes(include=[np.number, "bool"])
        dropped_cols = [c for c in list(X_num.columns) if c not in set(X_numeric.columns)]
        if dropped_cols:
            if verbose:
                try:
                    tprint_warning(f"[regime_leaf] dropping_non_numeric_feature_columns={dropped_cols} action=drop")
                except Exception:
                    pass
            X_num = X_numeric
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
        "alignment": {
            "enabled": bool(align_enabled),
            "method": str(align_method),
            "tolerance": str(tolerance) if tolerance is not None else None,
        },
        "targets": {},
        "report_path": None,
    }

    try:
        nan_frac = float(pd.to_numeric(X_num.stack(), errors="coerce").isna().mean()) if not X_num.empty else 0.0
        report["X_nan_frac"] = float(nan_frac)
    except Exception:
        pass

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

        try:
            y_non_null = int(y_all.notna().sum())
            y_nan_frac = float(1.0 - (float(y_non_null) / float(max(1, len(y_all)))))
        except Exception:
            y_non_null = 0
            y_nan_frac = 1.0

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
                tprint_info(
                    f"[regime_leaf] target_begin target={target_name} y_non_null={int(y_non_null)} y_nan_frac={float(y_nan_frac):.2%}"
                )
            except Exception:
                pass

        target_lookahead = None
        try:
            if target_name == "regime_volatility":
                target_lookahead = int(targets_cfg.get("volatility_window", 24))
            elif target_name == "regime_macro_trend":
                target_lookahead = int(targets_cfg.get("macro_trend_horizon", 96))
            elif target_name == "regime_volume_force_direction":
                vf_h = targets_cfg.get("volume_force_horizon")
                if vf_h is None:
                    vf_h = targets_cfg.get("volume_force_lookahead")
                target_lookahead = int(vf_h) if vf_h is not None else 16
            elif target_name == "regime_trend_efficiency":
                eff_h = targets_cfg.get("trend_efficiency_horizons")
                if isinstance(eff_h, (list, tuple)) and eff_h:
                    eff_h = [int(h) for h in eff_h if h is not None and int(h) > 1]
                    target_lookahead = int(max(eff_h)) if eff_h else int(targets_cfg.get("trend_efficiency_window", 16))
                else:
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
        last_top_shap_features = []
        last_top_mdi_features = []

        fold_test_indexes = []
        fold_test_indexes_raw = []

        fold_ic_spearman = []
        fold_ic_pearson = []
        fold_n = []

        fold_train_n_raw = []
        fold_train_n_after_leakage = []
        fold_train_n_after_lookahead = []
        fold_y_train_n = []
        fold_y_train_std = []

        leaf_window_presence: Dict[str, Dict[int, int]] = {}
        leaf_distinct_values: Dict[str, set] = {}

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

        try:
            min_train_samples_effective = wfv_cfg.get("min_train_samples_effective")
            min_train_samples_effective = int(min_train_samples_effective) if min_train_samples_effective is not None else int(cfg_min_train)
        except Exception:
            min_train_samples_effective = int(cfg_min_train)

        for train_sel, test_sel in list(split_plan):
            try:
                tr_pos = _sel_to_positions(X_num.index, train_sel, is_cross_fit=is_cross_fit)
                te_pos = _sel_to_positions(X_num.index, test_sel, is_cross_fit=is_cross_fit)
                if tr_pos.size == 0 or te_pos.size == 0:
                    continue

                te_pos_raw = np.asarray(te_pos, dtype=int)
                te_pos_eff = te_pos_raw

                try:
                    fold_train_n_raw.append(int(tr_pos.size))
                except Exception:
                    pass

                if leakage_enabled and te_pos_raw.size > 0 and tr_pos.size > 0:
                    cutoff = int(np.min(te_pos_raw)) - int(purge_bars) - int(embargo_bars)
                    if cutoff >= 0:
                        tr_pos = tr_pos[tr_pos < cutoff]

                try:
                    fold_train_n_after_leakage.append(int(tr_pos.size))
                except Exception:
                    pass

                if int(tr_pos.size) < int(min_train_samples_effective):
                    skipped_small_train += 1
                    continue

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

            try:
                fold_train_n_after_lookahead.append(int(len(X_train)))
            except Exception:
                pass

            if int(len(X_train)) < int(min_train_samples_effective):
                skipped_small_train += 1
                continue

            y_train = pd.to_numeric(y_train, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

            try:
                fold_y_train_n.append(int(len(y_train)))
                fold_y_train_std.append(float(y_train.std()))
            except Exception:
                pass

            try:
                min_target_std = float(wfv_cfg.get("min_target_std", 0.0))
            except Exception:
                min_target_std = 0.0
            if min_target_std is not None and np.isfinite(min_target_std) and float(min_target_std) > 0.0:
                try:
                    y_sd = float(y_train.std()) if len(y_train) > 1 else 0.0
                    if not np.isfinite(y_sd) or y_sd < float(min_target_std):
                        skipped_small_train += 1
                        continue
                except Exception:
                    pass

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

            lgbm_cfg = dict(config.get("lgbm", {}))
            try:
                lgbm_by_target = config.get("lgbm_by_target")
                if isinstance(lgbm_by_target, dict):
                    tgt_overrides = lgbm_by_target.get(str(target_name))
                    if isinstance(tgt_overrides, dict) and tgt_overrides:
                        lgbm_cfg.update(dict(tgt_overrides))
            except Exception:
                pass

            params = _default_lgbm_params(
                lgbm_cfg,
                random_state=random_state,
                n_train_samples=int(len(y_train)),
            )
            model = lgb.LGBMRegressor(**params)

            try:
                model.fit(X_train, y_train)
            except Exception as fit_exc:
                last_fit_error = str(fit_exc)
                continue

            # SHAP Feature Attribution
            try:
                import shap
                explainer = shap.TreeExplainer(model)
                # Sample background if train is too large
                x_sample = X_train.sample(n=min(len(X_train), 200), random_state=42) if len(X_train) > 200 else X_train
                shap_values = explainer.shap_values(x_sample)

                # Handle LightGBM/SHAP output formats (list for classifier, array for regressor)
                if isinstance(shap_values, list):
                    # For binary classifier, index 1 is usually positive class
                    if len(shap_values) > 1:
                        vals = shap_values[1]
                    else:
                        vals = shap_values[0]
                else:
                    vals = shap_values

                if hasattr(vals, "values"): # If shap returned an object with values
                     vals = vals.values

                # Mean Abs SHAP per feature
                if vals.shape == x_sample.shape:
                    shap_imp = np.mean(np.abs(vals), axis=0)
                    # Map to columns
                    feat_imp = pd.Series(shap_imp, index=X_train.columns).sort_values(ascending=False)
                    last_top_shap_features = feat_imp.head(5).index.tolist()
            except Exception:
                pass

            # MDI Feature Attribution (Fallback/Complement)
            try:
                mdi_imp = model.feature_importances_
                feat_imp_mdi = pd.Series(mdi_imp, index=X_train.columns).sort_values(ascending=False)
                last_top_mdi_features = feat_imp_mdi.head(5).index.tolist()
            except Exception:
                pass

            prediction_start_timestamp = pd.Timestamp.utcnow()
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

            try:
                for c in cols:
                    s = pd.to_numeric(leaf_chunk[c], errors="coerce").dropna()
                    if s.empty:
                        continue
                    uniq = pd.unique(s.astype(int))
                    dset = leaf_distinct_values.setdefault(c, set())
                    dset.update([int(x) for x in uniq.tolist()])
                    pc = leaf_window_presence.setdefault(c, {})
                    for x in uniq:
                        xi = int(x)
                        pc[xi] = int(pc.get(xi, 0)) + 1
            except Exception:
                pass
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
                    "y_non_null": int(y_non_null),
                    "y_nan_frac": float(y_nan_frac),
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

        leaf_stability_cfg = config.get("leaf_stability")
        if not isinstance(leaf_stability_cfg, dict):
            leaf_stability_cfg = {}
        try:
            min_survival_frac = float(leaf_stability_cfg.get("min_survival_frac", 0.0))
        except Exception:
            min_survival_frac = 0.0
        try:
            min_distinct_leaves = int(leaf_stability_cfg.get("min_distinct_leaves", 2))
        except Exception:
            min_distinct_leaves = 2
        min_survival_frac = float(max(0.0, min(1.0, min_survival_frac)))
        min_distinct_leaves = int(max(1, min_distinct_leaves))

        drop_trees = set()
        try:
            for rc in list(raw_cols):
                n_distinct = int(len(leaf_distinct_values.get(rc, set())))
                if n_distinct < int(min_distinct_leaves):
                    drop_trees.add(str(rc))
        except Exception:
            drop_trees = set()

        leaf_sel_cfg = config.get("leaf_selection")
        if not isinstance(leaf_sel_cfg, dict):
            leaf_sel_cfg = {}
        leaf_sel_mode = str(leaf_sel_cfg.get("mode", "effect_support")).lower()

        # Defaults: avoid selecting the dominant leaf (collapse) and prefer informative leaves.
        min_support = float(leaf_sel_cfg.get("min_support", 0.01))
        max_support = float(leaf_sel_cfg.get("max_support", 0.80))
        dominant_support_max = float(leaf_sel_cfg.get("dominant_support_max", 0.95))
        min_effect_z = float(leaf_sel_cfg.get("min_effect_z", 0.25))
        try:
            max_pairs = leaf_sel_cfg.get("max_pairs")
            if max_pairs is None:
                max_pairs = config.get("topk_max") if "topk_max" in config else None
            max_pairs = int(max_pairs) if max_pairs is not None else None
        except Exception:
            max_pairs = None

        keep_pairs_by_tree = {}
        pair_scores_sorted: List[Tuple[str, float, float, float]] = []
        if leaf_sel_mode in {"effect_support", "effect", "effect_support_v1"}:
            try:
                pair_scores_sorted, keep_pairs_by_tree = _score_leaf_pairs_effect_support(
                    leaves_oos=leaves_oos,
                    y_all=y_all,
                    raw_cols=raw_cols,
                    min_support=min_support,
                    max_support=max_support,
                    dominant_support_max=dominant_support_max,
                    min_effect_z=min_effect_z,
                    max_pairs=max_pairs,
                )
            except Exception:
                keep_pairs_by_tree = {}
                pair_scores_sorted = []

        if not keep_pairs_by_tree:
            # Fallback: frequency-based, but still avoid the dominant leaf
            pair_freqs = []
            for raw_col in raw_cols:
                try:
                    s = leaves_oos[raw_col]
                    vc = s.dropna().value_counts()
                    denom = float(max(1, int(vc.sum())))
                    for leaf_val, cnt in vc.items():
                        try:
                            freq = float(cnt) / denom
                            if not np.isfinite(freq):
                                continue
                            if freq > float(dominant_support_max):
                                continue
                            pair_freqs.append((raw_col, float(leaf_val), float(freq)))
                        except Exception:
                            continue
                except Exception:
                    continue

            if pair_freqs:
                pair_freqs_sorted = sorted(pair_freqs, key=lambda x: x[2], reverse=True)
                min_k = int(config.get("topk_min", 5))
                max_k = (config.get("topk_max") if "topk_max" in config else 7)
                try:
                    if max_k is not None:
                        max_k = int(max_k)
                except Exception:
                    max_k = None
                k_elbow = _find_elbow_k(np.asarray([p[2] for p in pair_freqs_sorted], dtype=float), min_k=min_k)
                if max_k is not None and max_k > 0:
                    k_elbow = int(min(k_elbow, max_k))
                kept = pair_freqs_sorted[: int(max(0, k_elbow))]
                for raw_col, leaf_val, _ in kept:
                    keep_pairs_by_tree.setdefault(raw_col, set()).add(float(leaf_val))

        pair_freqs_present = bool(keep_pairs_by_tree)

        kept_pairs_before_stability = 0
        try:
            kept_pairs_before_stability = int(sum([len(v) for v in keep_pairs_by_tree.values() if isinstance(v, set)]))
        except Exception:
            kept_pairs_before_stability = 0

        if min_survival_frac > 0.0 and int(windows_trained) > 0 and keep_pairs_by_tree:
            try:
                for raw_col, keep_vals in list(keep_pairs_by_tree.items()):
                    if raw_col in drop_trees:
                        keep_pairs_by_tree[raw_col] = set()
                        continue
                    if not isinstance(keep_vals, set) or not keep_vals:
                        continue
                    pc = leaf_window_presence.get(raw_col, {})
                    filtered = set()
                    for v in list(keep_vals):
                        try:
                            vi = int(v)
                        except Exception:
                            continue
                        surv = float(pc.get(vi, 0)) / float(max(1, int(windows_trained)))
                        if surv >= float(min_survival_frac):
                            filtered.add(float(vi))
                    keep_pairs_by_tree[raw_col] = filtered
            except Exception:
                pass

        kept_pairs_after_stability = 0
        try:
            kept_pairs_after_stability = int(sum([len(v) for v in keep_pairs_by_tree.values() if isinstance(v, set)]))
        except Exception:
            kept_pairs_after_stability = 0

        if verbose:
            try:
                n_pairs = int(sum([len(v) for v in keep_pairs_by_tree.values()]))
                tprint_info(f"[regime_leaf] target={target_name} leaf_selection={leaf_sel_mode} kept_pairs={n_pairs}")
            except Exception:
                pass

        try:
            if isinstance(report.get("targets", {}).get(str(target_name)), dict):
                report["targets"][str(target_name)]["y_non_null"] = int(y_non_null)
                report["targets"][str(target_name)]["y_nan_frac"] = float(y_nan_frac)
                report["targets"][str(target_name)]["leaf_stability"] = {
                    "min_survival_frac": float(min_survival_frac),
                    "min_distinct_leaves": int(min_distinct_leaves),
                    "kept_pairs_before": int(kept_pairs_before_stability),
                    "kept_pairs_after": int(kept_pairs_after_stability),
                    "dropped_trees": sorted(list(drop_trees)),
                    "n_dropped_trees": int(len(drop_trees)),
                }
        except Exception:
            pass

        for raw_col in raw_cols:
            if str(raw_col) in drop_trees:
                continue
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
                "min_train_samples_effective": int(min_train_samples_effective),
                "standardize_method": str(standardize_method) if standardize_enabled else None,
                "fold_ic_spearman": [float(x) if x is not None and np.isfinite(x) else None for x in fold_ic_spearman],
                "fold_ic_pearson": [float(x) if x is not None and np.isfinite(x) else None for x in fold_ic_pearson],
                "fold_n": [int(x) for x in fold_n],
                "fold_train_n_raw": [int(x) for x in fold_train_n_raw],
                "fold_train_n_after_leakage": [int(x) for x in fold_train_n_after_leakage],
                "fold_train_n_after_lookahead": [int(x) for x in fold_train_n_after_lookahead],
                "fold_y_train_n": [int(x) for x in fold_y_train_n],
                "fold_y_train_std": [float(x) if x is not None and np.isfinite(x) else None for x in fold_y_train_std],
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
                        "top_shap_features": list(last_top_shap_features) if last_top_shap_features else [],
                        "top_mdi_features": list(last_top_mdi_features) if last_top_mdi_features else [],
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

                interaction_fillna = str(interaction_cfg.get("fillna", "nan")).lower()
                if interaction_fillna in {"zero", "0", "0.0"}:
                    interaction_out = interaction_series.fillna(0.0)
                elif interaction_fillna in {"median"}:
                    med = float(interaction_series.dropna().median()) if interaction_series.notna().any() else 0.0
                    interaction_out = interaction_series.fillna(med)
                else:
                    interaction_out = interaction_series

                interaction_frames.append(
                    pd.DataFrame({f"regime_leaf_interaction__{target_name}": interaction_out}, index=X_num.index)
                )

                gating_cfg = interaction_cfg.get("gating") if isinstance(interaction_cfg.get("gating"), dict) else {}
                try:
                    gdf = _interaction_gating_features(
                        pd.to_numeric(interaction_out, errors="coerce"),
                        prefix=f"regime_leaf_interaction__{target_name}__",
                        cfg=dict(gating_cfg),
                    )
                    if gdf is not None and not gdf.empty:
                        interaction_frames.append(gdf.reindex(X_num.index).fillna(0.0))
                except Exception:
                    pass

                transition_cfg = interaction_cfg.get("transition") if isinstance(interaction_cfg.get("transition"), dict) else {}
                transition_enabled = bool(transition_cfg.get("enabled", True))
                if transition_enabled:
                    transition_fillna = str(transition_cfg.get("fillna", "none")).lower()
                    if transition_fillna in {"ffill_zero", "ffill0"}:
                        s_for_diff = interaction_series.ffill().fillna(0.0)
                    elif transition_fillna in {"ffill"}:
                        s_for_diff = interaction_series.ffill()
                    elif transition_fillna in {"zero", "0", "0.0"}:
                        s_for_diff = interaction_series.fillna(0.0)
                    else:
                        s_for_diff = interaction_series

                    d1 = s_for_diff.diff(1).astype(float)
                    interaction_frames.append(
                        pd.DataFrame({f"regime_leaf_interaction_transition__{target_name}": d1}, index=X_num.index)
                    )
                    interaction_frames.append(
                        pd.DataFrame({f"regime_leaf_interaction_transition_abs__{target_name}": d1.abs()}, index=X_num.index)
                    )

                    # ----------------------------------------------------------------
                    # REGIME MOMENTUM FEATURES
                    # Captures the rate of change of regime indicator over multiple windows.
                    # This helps the model understand regime dynamics: accelerating,
                    # decelerating, or stable regime conditions.
                    # ----------------------------------------------------------------
                    momentum_cfg = transition_cfg.get("momentum") if isinstance(transition_cfg.get("momentum"), dict) else {}
                    momentum_enabled = bool(momentum_cfg.get("enabled", True))
                    if momentum_enabled:
                        try:
                            momentum_windows = momentum_cfg.get("windows", [3, 5, 10])
                            if not isinstance(momentum_windows, (list, tuple)):
                                momentum_windows = [3, 5, 10]
                            momentum_windows = [int(w) for w in momentum_windows if int(w) > 0]

                            for mw in momentum_windows:
                                # Rolling mean of regime change (momentum)
                                regime_momentum = d1.rolling(window=mw, min_periods=1).mean()
                                interaction_frames.append(
                                    pd.DataFrame(
                                        {f"regime_leaf_momentum_{mw}__{target_name}": regime_momentum.astype(float)},
                                        index=X_num.index
                                    )
                                )

                                # Rolling std of regime change (volatility of regime transitions)
                                regime_momentum_std = d1.rolling(window=mw, min_periods=2).std()
                                interaction_frames.append(
                                    pd.DataFrame(
                                        {f"regime_leaf_momentum_std_{mw}__{target_name}": regime_momentum_std.astype(float)},
                                        index=X_num.index
                                    )
                                )

                            # Regime acceleration: second derivative (change of momentum)
                            regime_accel = d1.diff(1).astype(float)
                            interaction_frames.append(
                                pd.DataFrame(
                                    {f"regime_leaf_acceleration__{target_name}": regime_accel},
                                    index=X_num.index
                                )
                            )

                        except Exception:
                            pass

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
                        # Add momentum stats if enabled
                        if momentum_enabled:
                            try:
                                # Momentum over default 5-bar window
                                regime_mom_5 = d1.rolling(window=5, min_periods=1).mean()
                                stats.update({
                                    "momentum_5_mean": float(regime_mom_5.mean()) if regime_mom_5.notna().any() else None,
                                    "momentum_5_std": float(regime_mom_5.std()) if regime_mom_5.notna().any() else None,
                                    "momentum_5_abs_mean": float(regime_mom_5.abs().mean()) if regime_mom_5.notna().any() else None,
                                })
                            except Exception:
                                pass
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
            out_df = pd.concat(parts, axis=1).reindex(X_num.index)

            try:
                interaction_cols = [
                    c
                    for c in list(out_df.columns)
                    if str(c).startswith("regime_leaf_interaction__")
                    or str(c).startswith("regime_leaf_interaction_transition__")
                    or str(c).startswith("regime_leaf_interaction_transition_abs__")
                    or str(c).startswith("regime_leaf_momentum_")
                    or str(c).startswith("regime_leaf_acceleration__")
                ]
            except Exception:
                interaction_cols = []
            if interaction_cols:
                non_inter_cols = [c for c in list(out_df.columns) if c not in set(interaction_cols)]
                try:
                    if non_inter_cols:
                        out_df[non_inter_cols] = out_df[non_inter_cols].fillna(0.0)
                except Exception:
                    pass
            else:
                out_df = out_df.fillna(0.0)

            if time_features is not None and not time_features.empty and bool(time_feat_cfg.get("include_in_output", True)):
                try:
                    tf = time_features.reindex(out_df.index)
                    tf = tf.fillna(0.0)
                    out_df = pd.concat([out_df, tf], axis=1)
                except Exception:
                    pass

            # Explicitly pass through engineered regime features to the Meta Model
            try:
                # Include Multi-Horizon Agreement and Advanced Volatility/Transition features
                passthrough_prefixes = [
                    "reg_ohlcv__vol_multi_horizon",
                    "reg_ohlcv__trend_multi_horizon",
                    "reg_ohlcv__vol_acceleration",
                    "reg_ohlcv__d_efficiency",
                    "reg_ohlcv__d_autocorr",
                    "reg_ohlcv__vol_compression",
                    "reg_ohlcv__atr_rank",
                    "reg_ohlcv__vol_of_vol"
                ]
                cols_to_pass = [c for c in X_num.columns if any(str(c).startswith(p) for p in passthrough_prefixes)]
                if cols_to_pass:
                    pt_df = X_num[cols_to_pass].reindex(out_df.index).fillna(0.0)
                    out_df = pd.concat([out_df, pt_df], axis=1)
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
            tf = time_features.reindex(leaf_onehot.index).fillna(0.0)
            leaf_onehot = pd.concat([leaf_onehot, tf], axis=1)
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

    # Print summary of per-target metrics before saving report
    if verbose:
        try:
            tprint_success("=" * 60)
            tprint_success("REGIME LEAF MODEL: Per-Target Metrics Summary")
            tprint_success("=" * 60)
            targets_data = report.get("targets", {})
            if targets_data:
                for tgt_name, tgt_data in targets_data.items():
                    summary = tgt_data.get("fold_ic_spearman_summary", {})
                    fold_ics = tgt_data.get("fold_ic_spearman", [])
                    if summary:
                        ic_mean = summary.get("mean", "N/A")
                        ic_std = summary.get("std", "N/A")
                        icir = summary.get("icir", "N/A")
                        sign_cons = summary.get("sign_consistency", "N/A")
                        # Format values nicely
                        ic_mean_str = f"{ic_mean:.4f}" if isinstance(ic_mean, (int, float)) else str(ic_mean)
                        ic_std_str = f"{ic_std:.4f}" if isinstance(ic_std, (int, float)) else str(ic_std)
                        icir_str = f"{icir:.2f}" if isinstance(icir, (int, float)) else str(icir)
                        sign_str = f"{sign_cons:.1%}" if isinstance(sign_cons, (int, float)) else str(sign_cons)
                        fold_ic_str = ", ".join([f"{v:.3f}" if isinstance(v, (int, float)) else "N/A" for v in fold_ics[:6]])
                        # Classification for quick assessment
                        verdict = "✅ GOOD" if isinstance(ic_mean, (int, float)) and float(ic_mean) > 0.1 else ("⚠️ WEAK" if isinstance(ic_mean, (int, float)) and float(ic_mean) > 0.0 else "❌ NO SIGNAL")
                        tprint_info(f"  [{tgt_name}] {verdict}")
                        tprint_info(f"    IC Mean: {ic_mean_str}  |  IC Std: {ic_std_str}  |  ICIR: {icir_str}  |  Sign: {sign_str}")
                        tprint_info(f"    Fold ICs: [{fold_ic_str}]")
                    else:
                        tprint_warning(f"  [{tgt_name}] No summary available")
            tprint_success("=" * 60)
        except Exception as summary_exc:
            tprint_warning(f"[regime_leaf] metrics summary print failed: {summary_exc}")

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


def generate_regime_feature_interactions(
    feature_df: pd.DataFrame,
    regime_scores: pd.DataFrame,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Generate interaction features between regime scores and key market indicators.

    Creates products: RegimeScore * MarketFeature
    Targeting specific interactions: Volatility, Momentum (RSI), and Liquidity.
    Supports non-linear transformations (Squared, Threshold) to sharpen regime focus.

    Args:
        feature_df: DataFrame with raw market features (must contain keys like 'volatility', 'rsi', etc.)
        regime_scores: DataFrame with regime leaf scores (e.g. from extract_regime_leaf_onehot_features)
        config: Configuration dictionary for interactions

    Returns:
        DataFrame of interaction features
    """
    out = pd.DataFrame(index=feature_df.index)
    
    # Check for dynamic features (e.g. from SHAP mining)
    dynamic_feats = config.get("dynamic_interaction_features", [])
    if isinstance(dynamic_feats, list) and len(dynamic_feats) > 0:
        # Use dynamic SHAP features
        # Filter to only those present in the dataframe
        valid_dynamic = [f for f in dynamic_feats if f in feature_df.columns]
        if valid_dynamic:
            key_feats = {"shap": valid_dynamic}
        else:
             # Fallback if dynamic features not found in DF
            key_feats = config.get("interaction_keys", {
                "volatility": ["reg_ohlcv__volatility_1d", "volatility_1d"],
                "rsi": ["reg_ohlcv__rsi_14", "rsi_14", "rsi"],
                "volume": ["reg_ohlcv__volume_log1p_z", "volume_z"]
            })
    else:
        # Fallback to heuristics
        key_feats = config.get("interaction_keys", {
            "volatility": ["reg_ohlcv__volatility_1d", "volatility_1d"],
            "rsi": ["reg_ohlcv__rsi_14", "rsi_14", "rsi"],
            "volume": ["reg_ohlcv__volume_log1p_z", "volume_z"]
        })

    interaction_types = config.get("interaction_types", ["linear", "squared", "threshold"])
    threshold_val = float(config.get("interaction_threshold", 0.5))

    # Identify regime score columns
    score_cols = [c for c in regime_scores.columns if "regime_leaf_raw_score" in str(c)]
    
    for score_col in score_cols:
        score_raw = pd.to_numeric(regime_scores[score_col], errors="coerce").fillna(0.0)
        
        # Pre-compute score variants
        score_variants = {}
        if "linear" in interaction_types:
            score_variants["lin"] = score_raw
        if "squared" in interaction_types:
            score_variants["sq"] = score_raw ** 2
        if "threshold" in interaction_types:
            score_variants["thr"] = (score_raw - threshold_val).clip(lower=0.0)
        
        for key, candidate_list in key_feats.items():
            feat_col = next((c for c in candidate_list if c in feature_df.columns), None)
            if feat_col:
                feat_val = pd.to_numeric(feature_df[feat_col], errors="coerce").fillna(0.0)
                
                for var_name, score_vec in score_variants.items():
                    # e.g. regime_leaf_raw_score__regime_volatility_x_volatility_lin
                    col_name = f"{score_col}_x_{key}_{var_name}"
                    out[col_name] = score_vec * feat_val
                
    return out
