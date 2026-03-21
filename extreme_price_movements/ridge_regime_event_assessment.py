import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional

from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr

from extreme_price_movements.config import RIDGE_FEATURE_META, RIDGE_FEATURE_COLS

# =========================================================
# DATACLASSES
# =========================================================
@dataclass(frozen=True)
class Phase3ConditionerSeed:
    feature: str
    family: str
    feature_type: str
    direction: str
    coefficient: float
    stability_ratio: float
    sign_consistency: float
    abs_signed_importance: float
    thresholds: Optional[Dict[str, float]] = None


# =========================================================
# 1) FEATURE ENGINEERING
# =========================================================
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rolling_zscore(series, window):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std

def rolling_slope(series, window):
    from extreme_price_movements import fast_funcs as ff
    arr = series.to_numpy(dtype=np.float32, copy=False)
    res = ff.slope_nb(arr, window)
    return pd.Series(res, index=series.index)

def true_range(df):
    prev_close = df["close"].shift()
    return np.maximum(
        df["high"] - df["low"],
        np.maximum(
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ),
    )

def atr(df, window=14):
    return true_range(df).rolling(window).mean()

def rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    from extreme_price_movements import fast_funcs as ff
    return ff.numba_rolling_rank_pct(series, window)

def build_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Ensure numeric columns are actually numeric
    for col in ["high", "low", "close", "volume", "open"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)

    # -----------------------------
    # Trend
    # -----------------------------
    df["ema20"] = ema(df["close"], 40)
    df["ema50"] = ema(df["close"], 100)
    df["ema200"] = ema(df["close"], 400)

    df["ema20_gt_ema50"] = (df["ema20"] > df["ema50"]).astype(int)
    df["ema50_gt_ema200"] = (df["ema50"] > df["ema200"]).astype(int)
    df["price_gt_ema50"] = (df["close"] > df["ema50"]).astype(int)
    df["price_lt_ema200"] = (df["close"] < df["ema200"]).astype(int)

    df["ema20_slope"] = rolling_slope(df["ema20"], 40)
    df["ema50_slope"] = rolling_slope(df["ema50"], 100)
    df["ema200_slope"] = rolling_slope(df["ema200"], 400)

    # Fixed: rolling percentile
    df["trend_strength_percentile"] = rolling_percentile_rank(df["ema50_slope"], 168)

    # Fixed: bars_since_trend_flip
    trend_sign = np.sign(df["ema20_slope"])
    flip_flag = trend_sign.ne(trend_sign.shift()).fillna(False)
    regime_id = flip_flag.cumsum()
    df["bars_since_trend_flip"] = df.groupby(regime_id).cumcount()

    # -----------------------------
    # Volatility
    # -----------------------------
    df["rolling_std_4h"] = df["close"].rolling(32).std()
    df["realized_volatility_24h"] = df["close"].pct_change(fill_method=None).rolling(192).std()

    df["atr14"] = atr(df, 28)
    df["atr_change_rate"] = df["atr14"].pct_change(fill_method=None)

    df["true_range"] = true_range(df)
    # Fixed: rolling percentile
    df["true_range_percentile"] = rolling_percentile_rank(df["true_range"], 168)

    # -----------------------------
    # Stretch / distance
    # -----------------------------
    df["dist_ema20_atr"] = (df["close"] - df["ema20"]) / df["atr14"].replace(0, np.nan)
    df["dist_ema50_atr"] = (df["close"] - df["ema50"]) / df["atr14"].replace(0, np.nan)
    df["dist_ema200_atr"] = (df["close"] - df["ema200"]) / df["atr14"].replace(0, np.nan)

    df["zscore_price_50"] = rolling_zscore(df["close"], 100)
    df["zscore_price_200"] = rolling_zscore(df["close"], 400)

    # -----------------------------
    # VWAP
    # -----------------------------
    df["date"] = df["timestamp"].dt.date

    if "volume" not in df.columns:
        df["volume"] = 1.0

    df["cum_vol"] = df.groupby("date")["volume"].cumsum()
    df["cum_vol_price"] = (df["close"] * df["volume"]).groupby(df["date"]).cumsum()
    df["vwap_intraday"] = df["cum_vol_price"] / df["cum_vol"].replace(0, np.nan)
    df["dist_vwap_atr"] = (df["close"] - df["vwap_intraday"]) / df["atr14"].replace(0, np.nan)

    df["week"] = df["timestamp"].dt.isocalendar().week
    df["cum_vol_w"] = df.groupby("week")["volume"].cumsum()
    df["cum_vol_price_w"] = ((df["close"] * df["volume"]).groupby(df["week"]).cumsum())
    df["anchored_vwap_week"] = df["cum_vol_price_w"] / df["cum_vol_w"].replace(0, np.nan)
    df["dist_weekly_vwap"] = df["close"] - df["anchored_vwap_week"]

    # -----------------------------
    # Compression / expansion
    # -----------------------------
    df["bollinger_mid"] = df["close"].rolling(40).mean()
    df["bollinger_std"] = df["close"].rolling(40).std()
    df["bollinger_band_width"] = 2.0 * df["bollinger_std"] / df["bollinger_mid"].replace(0, np.nan)

    df["rolling_range_20"] = df["high"].rolling(40).max() - df["low"].rolling(40).min()
    # Fixed: rolling percentile
    df["atr_percentile"] = rolling_percentile_rank(df["atr14"], 168)

    # -----------------------------
    # Structure location
    # -----------------------------
    daily_high = df.groupby("date")["high"].transform("max")
    daily_low = df.groupby("date")["low"].transform("min")

    df["prior_day_high"] = daily_high.shift(1)
    df["prior_day_low"] = daily_low.shift(1)

    df["dist_prior_day_high"] = df["close"] - df["prior_day_high"]
    df["dist_prior_day_low"] = df["close"] - df["prior_day_low"]

    # Fixed: Rename to rolling_7d_high
    df["rolling_7d_high"] = df["high"].rolling(7 * 192).max()
    df["dist_rolling_7d_high"] = df["close"] - df["rolling_7d_high"]

    df["local_swing_high"] = df["high"].rolling(40).max()
    df["dist_local_swing"] = df["close"] - df["local_swing_high"]

    # -----------------------------
    # Candle structure / liquidity proxy
    # -----------------------------
    if "open" not in df.columns:
        df["open"] = df["close"].shift(1).fillna(df["close"])

    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    df["wick_length"] = upper_wick + lower_wick

    df["candle_range"] = df["high"] - df["low"]
    df["wick_to_range"] = df["wick_length"] / df["candle_range"].replace(0, np.nan)

    df["volume_ma20"] = df["volume"].rolling(40).mean()
    df["volume_spike"] = df["volume"] / df["volume_ma20"].replace(0, np.nan)

    df["orderflow_imbalance"] = (
        (df["close"] - df["open"]) /
        (df["high"] - df["low"]).replace(0, np.nan)
    )

    # -----------------------------
    # Momentum
    # -----------------------------
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.rolling(28).mean()
    avg_loss = loss.rolling(28).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100.0 - (100.0 / (1.0 + rs))

    ema12 = ema(df["close"], 24)
    ema26 = ema(df["close"], 52)
    macd = ema12 - ema26
    signal = ema(macd, 18)
    df["MACD_histogram"] = macd - signal

    df["rate_of_change"] = df["close"].pct_change(24, fill_method=None)
    df["momentum_zscore"] = rolling_zscore(df["rate_of_change"], 100)

    # -----------------------------
    # Prior move context
    # -----------------------------
    df["prior_return_1h"] = df["close"].pct_change(8, fill_method=None)
    df["prior_return_4h"] = df["close"].pct_change(32, fill_method=None)
    df["prior_return_12h"] = df["close"].pct_change(96, fill_method=None)

    df["prior_range"] = df["high"].rolling(32).max() - df["low"].rolling(32).min()
    df["prior_volatility"] = df["close"].pct_change(fill_method=None).rolling(32).std()

    df["acceleration_of_move"] = df["close"].pct_change(fill_method=None).diff()

    # -----------------------------
    # Micro-regimes / Short Timeframe
    # -----------------------------
    # Path Structure
    net_move_20 = (df["close"] - df["close"].shift(40)).abs()
    abs_moves_20 = df["close"].diff().abs().rolling(40).sum()
    df["efficiency_ratio_20"] = net_move_20 / abs_moves_20.replace(0, np.nan)

    atr_sum_20 = df["true_range"].rolling(40).sum()
    range_20 = df["high"].rolling(40).max() - df["low"].rolling(40).min()
    # 100 * log10( sum(ATR(1), 40) / (highest(40) - lowest(40)) ) / log10(40)
    # Clip to avoid log(0) or log(<0)
    chop_ratio = (atr_sum_20 / range_20.replace(0, np.nan)).clip(lower=1e-6)
    df["choppiness_index_20"] = 100.0 * np.log10(chop_ratio) / np.log10(40)

    # Direction Entropy 20: entropy of up/down signs over 40 bars (Vectorized)
    # 🎯 Why: Iterating over DataFrame columns in Python using pd.Series.rolling().apply() is phenomenally slow.
    diff = df["close"].diff()
    # sum of up/down over 40 bars
    up_sum = (diff > 0).astype(float).rolling(40).sum()
    dn_sum = (diff < 0).astype(float).rolling(40).sum()

    p_up = up_sum / 40.0
    p_dn = dn_sum / 40.0

    # log2(0) handles by np.where/masking or adding a small epsilon. Since pandas can handle log2 with 0s as -inf
    # and 0 * -inf is nan which we replace with 0.
    e_up = -(p_up * np.log2(p_up.replace(0, np.nan))).fillna(0.0)
    e_dn = -(p_dn * np.log2(p_dn.replace(0, np.nan))).fillna(0.0)

    # restore NaNs for warmup period
    e_total = e_up + e_dn
    e_total.iloc[:39] = np.nan
    df["direction_entropy_20"] = e_total

    # Volatility Term Structure
    std_20 = df["close"].pct_change(fill_method=None).rolling(40).std()
    std_100 = df["close"].pct_change(fill_method=None).rolling(200).std()
    df["compression_ratio"] = std_20 / std_100.replace(0, np.nan)

    df["range_expansion_ratio"] = df["true_range"] / df["atr14"].replace(0, np.nan)

    # Extra pre-existing family additions
    highest_20 = df["high"].rolling(40).max()
    lowest_20 = df["low"].rolling(40).min()
    range_mid_20 = (highest_20 + lowest_20) / 2.0
    df["dist_range_mid_atr"] = (df["close"] - range_mid_20) / df["atr14"].replace(0, np.nan)

    ema100 = ema(df["close"], 200)
    df["dist_ma100_atr"] = (df["close"] - ema100) / df["atr14"].replace(0, np.nan)

    df["volatility_ratio_short_long"] = df["rolling_std_4h"] / df["realized_volatility_24h"].replace(0, np.nan)

    df["volume_percentile"] = rolling_percentile_rank(df["volume"], 168)

    # -----------------------------
    # v17: Technical Regime Additions
    # -----------------------------
    df["range_atr"] = (df["high"] - df["low"]) / df["atr14"].replace(0, np.nan)
    df["body_ratio"] = (df["close"] - df["open"]).abs() / (df["high"] - df["low"]).replace(0, np.nan)
    
    # Clearer wick definitions
    df["upper_wick_ratio"] = (df["high"] - df[["open", "close"]].max(axis=1)) / (df["high"] - df["low"]).replace(0, np.nan)
    df["lower_wick_ratio"] = (df[["open", "close"]].min(axis=1) - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    
    # Specific slopes and depths
    df["ema20_slope_5h"] = (df["ema20"] - df["ema20"].shift(5)) / df["atr14"].replace(0, np.nan)
    df["pullback_depth"] = (df["ema20"] - df["low"]) / df["atr14"].replace(0, np.nan)
    
    # ATR-based compression
    df["atr_long"] = true_range(df).rolling(200).mean()
    df["atr_compression_ratio"] = df["atr14"] / df["atr_long"].replace(0, np.nan)
    
    # Normalized second-order acceleration
    # accel = close - 2*close[-1] + close[-2]
    df["acceleration_norm"] = (df["close"] - 2*df["close"].shift(1) + df["close"].shift(2)) / df["atr14"].replace(0, np.nan)

    return df

# =========================================================
# 2) SELECTION & PHASE 3 SEEDING
# =========================================================
def select_promising_regime_variables(
    ranked_features: pd.DataFrame,
    event_df: pd.DataFrame,
    max_total: int = 8,
    max_per_family: int = 2,
    min_importance: float = 0.05,
    corr_threshold: float = 0.90
) -> pd.DataFrame:
    """
    Selects top features avoiding family concentration and high collinearity.
    """
    selected = []
    family_counts = {f["family"]: 0 for f in RIDGE_FEATURE_META.values()}

    # We only check correlation against already selected continuous features
    selected_continuous_cols = []

    for _, row in ranked_features.iterrows():
        if len(selected) >= max_total:
            break

        fname = row["feature"]
        importance = row["abs_signed_importance"]

        if pd.isna(importance) or importance < min_importance:
            continue

        meta = RIDGE_FEATURE_META.get(fname, {"family": "unknown", "type": "continuous"})
        family = meta["family"]
        ftype = meta["type"]

        if family_counts.get(family, 0) >= max_per_family:
            continue

        # Duplicate suppression (only for continuous)
        is_duplicate = False
        if ftype == "continuous" and fname in event_df.columns:
            for s_col in selected_continuous_cols:
                if s_col in event_df.columns:
                    # calculate correlation on event rows
                    mask = event_df[fname].notna() & event_df[s_col].notna()
                    if mask.sum() > 10:
                        corr = np.corrcoef(event_df.loc[mask, fname], event_df.loc[mask, s_col])[0, 1]
                        if abs(corr) > corr_threshold:
                            is_duplicate = True
                            break
        if is_duplicate:
            continue

        selected.append(row)
        family_counts[family] = family_counts.get(family, 0) + 1
        if ftype == "continuous":
            selected_continuous_cols.append(fname)

    if not selected:
        return pd.DataFrame(columns=ranked_features.columns)

    return pd.DataFrame(selected)

def build_phase3_conditioner_seeds(
    selected_features: pd.DataFrame,
    event_df: pd.DataFrame
) -> List[Phase3ConditionerSeed]:
    """
    Transforms selected Ridge features into robust seeds for Phase 3 conditioned events.
    Continuous features will carry quantiles evaluated strictly over the event_df rows.
    """
    if selected_features.empty:
        return []

    ranked = selected_features.copy()
    if "coef" in ranked.columns:
        abs_strength = ranked["coef"].abs().astype(float)
        if len(abs_strength) > 1 and np.isfinite(abs_strength).any():
            strength_cutoff = float(np.nanpercentile(abs_strength.values, 50.0))
            keep_mask = abs_strength >= strength_cutoff
            if keep_mask.any():
                ranked = ranked.loc[keep_mask].copy()

    seeds = []
    for _, row in ranked.iterrows():
        fname = row["feature"]
        coef = row["coef"]

        meta = RIDGE_FEATURE_META.get(fname, {"family": "unknown", "type": "continuous"})
        direction = "positive" if coef > 0 else "negative"

        thresholds = None
        if meta["type"] == "continuous" and fname in event_df.columns:
            valid_vals = event_df[fname].dropna().values
            if len(valid_vals) > 10:
                qs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
                thresholds = {f"q{int(q*100)}": float(np.percentile(valid_vals, q*100)) for q in qs}

        sign_cons = row.get("sign_consistency", 1.0)

        seed = Phase3ConditionerSeed(
            feature=fname,
            family=meta["family"],
            feature_type=meta["type"],
            direction=direction,
            coefficient=float(coef),
            stability_ratio=float(row.get("stability_ratio", 0.0)),
            sign_consistency=float(sign_cons),
            abs_signed_importance=float(row.get("abs_signed_importance", 0.0)),
            thresholds=thresholds
        )
        seeds.append(seed)

    return seeds


def _impute_and_scale_train_valid(
    X_train: np.ndarray,
    X_valid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    X_tr = np.asarray(X_train, dtype=np.float32).copy()
    X_va = np.asarray(X_valid, dtype=np.float32).copy()
    X_tr[~np.isfinite(X_tr)] = np.nan
    X_va[~np.isfinite(X_va)] = np.nan

    n_features = X_tr.shape[1]
    mean = np.zeros(n_features, dtype=np.float32)
    std = np.ones(n_features, dtype=np.float32)

    for j in range(n_features):
        col = X_tr[:, j]
        valid = ~np.isnan(col)
        if np.any(valid):
            med = np.median(col[valid]).astype(np.float32)
            X_tr[~valid, j] = med
            X_va[np.isnan(X_va[:, j]), j] = med
        else:
            X_tr[:, j] = 0.0
            X_va[:, j] = 0.0
        mean[j] = np.mean(X_tr[:, j]).astype(np.float32)
        s = np.std(X_tr[:, j]).astype(np.float32)
        std[j] = s if s > 1e-6 else 1.0

    X_tr = ((X_tr - mean) / std).astype(np.float32, copy=False)
    X_va = ((X_va - mean) / std).astype(np.float32, copy=False)
    return X_tr, X_va


def _ridge_fit_predict_closed_form(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    X_tr = np.asarray(X_train, dtype=np.float32)
    y_tr = np.asarray(y_train, dtype=np.float32)
    X_va = np.asarray(X_valid, dtype=np.float32)
    n_features = X_tr.shape[1]
    if X_tr.shape[0] == 0 or n_features == 0:
        return np.zeros(X_va.shape[0], dtype=np.float32), np.zeros(n_features, dtype=np.float32)

    y_mean = float(np.mean(y_tr))
    y_centered = (y_tr - y_mean).astype(np.float32, copy=False)
    xtx = (X_tr.T @ X_tr).astype(np.float32, copy=False)
    xty = (X_tr.T @ y_centered).astype(np.float32, copy=False)
    reg = np.eye(n_features, dtype=np.float32) * np.float32(alpha)
    coef = np.linalg.solve(xtx + reg, xty).astype(np.float32, copy=False)
    preds = (X_va @ coef + np.float32(y_mean)).astype(np.float32, copy=False)
    return preds, coef


def _r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_f = np.asarray(y_true, dtype=np.float32)
    y_pred_f = np.asarray(y_pred, dtype=np.float32)
    sst = float(np.sum((y_true_f - np.mean(y_true_f)) ** 2))
    if sst < 1e-9:
        return 0.0
    ssr = float(np.sum((y_true_f - y_pred_f) ** 2))
    return float(1.0 - ssr / sst)


def fit_ridge_regime_scan_arrays(
    feature_matrix: np.ndarray,
    feature_cols: List[str],
    event_mask: np.ndarray,
    target_values: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    n_splits: int = 4,
    alphas: tuple = (1.0, 3.0, 10.0, 30.0),
    min_test_rows_per_fold: int = 10,
) -> Optional[Dict[str, Any]]:
    X_all = np.asarray(feature_matrix, dtype=np.float32)
    y_all = np.asarray(target_values, dtype=np.float32)
    event_mask_arr = np.asarray(event_mask, dtype=bool)
    if X_all.ndim != 2 or X_all.shape[0] != y_all.shape[0] or y_all.shape[0] != event_mask_arr.shape[0]:
        return None

    event_idx = np.flatnonzero(event_mask_arr & np.isfinite(y_all)).astype(np.int32)
    if event_idx.shape[0] == 0:
        return None
    X_event = X_all[event_idx]
    y_event = y_all[event_idx]
    ts_event = np.asarray(timestamps)[event_idx] if timestamps is not None else None

    if event_idx.shape[0] < (n_splits + 1) * min_test_rows_per_fold:
        return None

    usable_positions = [
        i for i, _ in enumerate(feature_cols)
        if i < X_event.shape[1] and np.isfinite(X_event[:, i]).any()
    ]
    if not usable_positions:
        return None

    used_features = [feature_cols[i] for i in usable_positions]
    X_event = X_event[:, usable_positions].astype(np.float32, copy=False)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_coefs: List[np.ndarray] = []
    fold_scores: List[Dict[str, Any]] = []
    cv_scores: List[float] = []
    fold_diagnostics: List[Dict[str, Any]] = []

    best_alpha = float(alphas[0])
    best_alpha_score = -np.inf
    for alpha in alphas:
        alpha_scores: List[float] = []
        valid_alpha = True
        for train_idx, test_idx in tscv.split(X_event):
            if len(test_idx) < min_test_rows_per_fold:
                return None
            X_tr, X_te = _impute_and_scale_train_valid(X_event[train_idx], X_event[test_idx])
            preds, _ = _ridge_fit_predict_closed_form(X_tr, y_event[train_idx], X_te, float(alpha))
            alpha_scores.append(_r2_score_np(y_event[test_idx], preds))
        if not valid_alpha or not alpha_scores:
            continue
        mean_score = float(np.mean(alpha_scores))
        if mean_score > best_alpha_score:
            best_alpha_score = mean_score
            best_alpha = float(alpha)

    for fold_id, (train_idx, test_idx) in enumerate(tscv.split(X_event), start=1):
        if len(test_idx) < min_test_rows_per_fold:
            return None
        X_tr, X_te = _impute_and_scale_train_valid(X_event[train_idx], X_event[test_idx])
        preds, coef = _ridge_fit_predict_closed_form(X_tr, y_event[train_idx], X_te, best_alpha)
        r2 = _r2_score_np(y_event[test_idx], preds)
        cv_scores.append(r2)
        fold_coefs.append(coef.astype(np.float32, copy=False))
        fold_scores.append(
            {
                "fold": fold_id,
                "test_r2": r2,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
            }
        )
        if ts_event is not None:
            fold_diagnostics.append(
                {
                    "fold": fold_id,
                    "train_start_ts": ts_event[train_idx[0]],
                    "train_end_ts": ts_event[train_idx[-1]],
                    "test_start_ts": ts_event[test_idx[0]],
                    "test_end_ts": ts_event[test_idx[-1]],
                }
            )

    X_full, _ = _impute_and_scale_train_valid(X_event, X_event[:1])
    _, full_coef = _ridge_fit_predict_closed_form(X_full, y_event, X_full[:1], best_alpha)

    event_feature_df = pd.DataFrame(X_event, columns=used_features)
    fold_coef_arr = np.vstack(fold_coefs).astype(np.float32, copy=False)
    coef_df = pd.DataFrame({"feature": used_features, "coef": full_coef.astype(np.float32, copy=False)})
    fold_coef_df = pd.DataFrame(fold_coef_arr, columns=used_features)
    signs = np.sign(fold_coef_df)
    sign_cons = signs.sum(axis=0).abs() / max(n_splits, 1)
    coef_stability = pd.DataFrame(
        {
            "feature": used_features,
            "coef_mean": fold_coef_df.mean(axis=0).values,
            "coef_std": fold_coef_df.std(axis=0).values,
            "coef_abs_mean": fold_coef_df.abs().mean(axis=0).values,
            "sign_consistency": sign_cons.values,
        }
    )
    out = coef_df.merge(coef_stability, on="feature", how="left")
    out["stability_ratio"] = out["coef_abs_mean"] / out["coef_std"].replace(0, np.nan)
    out["signed_importance"] = out["coef"] * out["stability_ratio"].fillna(0.0)
    out["abs_signed_importance"] = out["signed_importance"].abs()
    out = out.sort_values("abs_signed_importance", ascending=False).reset_index(drop=True)

    selected_features = select_promising_regime_variables(out, event_feature_df, max_total=8, max_per_family=2)
    phase3_conditioner_seeds = build_phase3_conditioner_seeds(selected_features, event_feature_df)
    summary = {
        "n_event_rows": int(event_idx.shape[0]),
        "best_alpha_full_fit": float(best_alpha),
        "cv_r2_mean": float(np.mean(cv_scores)),
        "cv_r2_std": float(np.std(cv_scores)),
        "cv_r2_scores": list(map(float, cv_scores)),
        "fold_scores": fold_scores,
    }
    return {
        "summary": summary,
        "ranked_features": out,
        "selected_features": selected_features,
        "phase3_conditioner_seeds": phase3_conditioner_seeds,
        "fold_coefs": fold_coef_df,
        "fold_diagnostics": fold_diagnostics,
    }


def fit_lgbm_regime_scan_arrays(
    feature_matrix: np.ndarray,
    feature_cols: List[str],
    event_mask: np.ndarray,
    target_values: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    n_splits: int = 4,
    min_test_rows_per_fold: int = 10,
) -> Optional[Dict[str, Any]]:
    """
    Economical but superior non-linear alternative to Ridge.
    Uses LGBM with gain importance and Spearman correlation for directionality.
    """
    X_all = np.asarray(feature_matrix, dtype=np.float32)
    y_all = np.asarray(target_values, dtype=np.float32)
    event_mask_arr = np.asarray(event_mask, dtype=bool)
    if X_all.ndim != 2 or X_all.shape[0] != y_all.shape[0] or y_all.shape[0] != event_mask_arr.shape[0]:
        return None

    event_idx = np.flatnonzero(event_mask_arr & np.isfinite(y_all)).astype(np.int32)
    if event_idx.shape[0] < (n_splits + 1) * min_test_rows_per_fold:
        return None

    X_event = X_all[event_idx]
    y_event = y_all[event_idx]
    ts_event = np.asarray(timestamps)[event_idx] if timestamps is not None else None

    usable_positions = [
        i for i, _ in enumerate(feature_cols)
        if i < X_event.shape[1] and np.isfinite(X_event[:, i]).any()
    ]
    if not usable_positions:
        return None

    used_features = [feature_cols[i] for i in usable_positions]
    X_event = X_event[:, usable_positions].astype(np.float32, copy=False)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_importances: List[np.ndarray] = []
    cv_scores: List[float] = []

    for train_idx, test_idx in tscv.split(X_event):
        X_tr, X_te = _impute_and_scale_train_valid(X_event[train_idx], X_event[test_idx])
        y_tr, y_te = y_event[train_idx], y_event[test_idx]

        # Cheap yet better config
        model = LGBMRegressor(
            n_estimators=50,
            max_depth=3,
            num_leaves=8,
            learning_rate=0.07,
            importance_type='gain',
            min_child_samples=max(5, len(train_idx) // 20),
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=1,
            verbosity=-1,
            random_state=42
        )
        try:
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            fold_importances.append(model.feature_importances_.astype(np.float32))
            cv_scores.append(_r2_score_np(y_te, preds))
        except:
            continue

    if not fold_importances:
        return None

    # Aggregate importance stability
    imp_arr = np.vstack(fold_importances)
    imp_mean = np.mean(imp_arr, axis=0)
    imp_std = np.std(imp_arr, axis=0)
    stability_ratio = imp_mean / (imp_std + 1e-9)

    # Directionality via Spearman correlation on the whole event set
    # (Since tree importance is unsigned)
    directions = []
    for j in range(X_event.shape[1]):
        col = X_event[:, j]
        mask = np.isfinite(col) & np.isfinite(y_event)
        if np.sum(mask) > 10:
            rho, _ = spearmanr(col[mask], y_event[mask])
            directions.append(float(np.nan_to_num(rho)))
        else:
            directions.append(0.0)
    
    dir_arr = np.asarray(directions, dtype=np.float32)

    out = pd.DataFrame({
        "feature": used_features,
        "coef": dir_arr * imp_mean, # "Synthetic" coefficient for compatibility
        "gain_mean": imp_mean,
        "stability_ratio": stability_ratio,
        "spearman_rho": dir_arr
    })
    out["abs_signed_importance"] = out["gain_mean"] * stability_ratio
    out = out.sort_values("abs_signed_importance", ascending=False).reset_index(drop=True)

    summary = {
        "n_event_rows": int(event_idx.shape[0]),
        "cv_r2_mean": float(np.mean(cv_scores)) if cv_scores else 0.0,
        "cv_r2_std": float(np.std(cv_scores)) if cv_scores else 0.0,
    }

    selected_features = select_promising_regime_variables(out, pd.DataFrame(X_event, columns=used_features), max_total=8, max_per_family=2)
    phase3_conditioner_seeds = build_phase3_conditioner_seeds(selected_features, pd.DataFrame(X_event, columns=used_features))

    return {
        "summary": summary,
        "ranked_features": out,
        "selected_features": selected_features,
        "phase3_conditioner_seeds": phase3_conditioner_seeds,
    }

# =========================================================
# 3) RIDGE REGIME ATTRIBUTION
# =========================================================
def fit_ridge_regime_scan(
    df: pd.DataFrame,
    feature_cols: list[str],
    event_col: str,
    target_col: str,
    n_splits: int = 4,
    alphas: tuple = (1.0, 3.0, 10.0, 30.0),
    min_test_rows_per_fold: int = 10,
):
    """
    Phase 2.5 expects exact survivor event masks to already exist on the dataframe.
    This function fits Ridge purely on those rows.
    """
    if df is None or df.empty:
        return None
    usable_feature_cols = [col for col in feature_cols if col in df.columns]
    if not usable_feature_cols:
        return None
    feature_matrix = (
        df[usable_feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(dtype=np.float32, copy=True)
    )
    event_mask = df[event_col].to_numpy(dtype=np.int8, copy=False).astype(bool, copy=False)
    target_values = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=np.float32, copy=False)
    timestamps = df["timestamp"].to_numpy() if "timestamp" in df.columns else None
    return fit_ridge_regime_scan_arrays(
        feature_matrix=feature_matrix,
        feature_cols=usable_feature_cols,
        event_mask=event_mask,
        target_values=target_values,
        timestamps=timestamps,
        n_splits=n_splits,
        alphas=alphas,
        min_test_rows_per_fold=min_test_rows_per_fold,
    )
