import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =========================================================
# FEATURE METADATA
# =========================================================
RIDGE_FEATURE_META = {
    "ema20_gt_ema50": {"family": "trend", "type": "binary"},
    "ema50_gt_ema200": {"family": "trend", "type": "binary"},
    "price_gt_ema50": {"family": "trend", "type": "binary"},
    "price_lt_ema200": {"family": "trend", "type": "binary"},
    "ema20_slope": {"family": "trend", "type": "continuous"},
    "ema50_slope": {"family": "trend", "type": "continuous"},
    "ema200_slope": {"family": "trend", "type": "continuous"},
    "trend_strength_percentile": {"family": "trend", "type": "continuous"},
    "bars_since_trend_flip": {"family": "trend", "type": "continuous"},

    "rolling_std_4h": {"family": "volatility", "type": "continuous"},
    "realized_volatility_24h": {"family": "volatility", "type": "continuous"},
    "atr_change_rate": {"family": "volatility", "type": "continuous"},
    "true_range_percentile": {"family": "volatility", "type": "continuous"},

    "dist_ema20_atr": {"family": "stretch", "type": "continuous"},
    "dist_ema50_atr": {"family": "stretch", "type": "continuous"},
    "dist_ema200_atr": {"family": "stretch", "type": "continuous"},
    "zscore_price_50": {"family": "stretch", "type": "continuous"},
    "zscore_price_200": {"family": "stretch", "type": "continuous"},
    "dist_vwap_atr": {"family": "stretch", "type": "continuous"},
    "dist_weekly_vwap": {"family": "stretch", "type": "continuous"},

    "bollinger_band_width": {"family": "compression", "type": "continuous"},
    "rolling_range_20": {"family": "compression", "type": "continuous"},
    "atr_percentile": {"family": "compression", "type": "continuous"},

    "dist_prior_day_high": {"family": "structure", "type": "continuous"},
    "dist_prior_day_low": {"family": "structure", "type": "continuous"},
    "dist_rolling_7d_high": {"family": "structure", "type": "continuous"},
    "dist_local_swing": {"family": "structure", "type": "continuous"},

    "wick_to_range": {"family": "liquidity", "type": "continuous"},
    "volume_spike": {"family": "liquidity", "type": "continuous"},
    "orderflow_imbalance": {"family": "liquidity", "type": "continuous"},

    "RSI": {"family": "momentum", "type": "continuous"},
    "MACD_histogram": {"family": "momentum", "type": "continuous"},
    "rate_of_change": {"family": "momentum", "type": "continuous"},
    "momentum_zscore": {"family": "momentum", "type": "continuous"},

    "prior_return_1h": {"family": "context", "type": "continuous"},
    "prior_return_4h": {"family": "context", "type": "continuous"},
    "prior_return_12h": {"family": "context", "type": "continuous"},
    "prior_range": {"family": "context", "type": "continuous"},
    "prior_volatility": {"family": "context", "type": "continuous"},
    "acceleration_of_move": {"family": "context", "type": "continuous"},

    # Micro-regime updates
    "efficiency_ratio_20": {"family": "path_structure", "type": "continuous"},
    "choppiness_index_20": {"family": "path_structure", "type": "continuous"},
    "direction_entropy_20": {"family": "path_structure", "type": "continuous"},

    "compression_ratio": {"family": "volatility_term_structure", "type": "continuous"},
    "range_expansion_ratio": {"family": "volatility_term_structure", "type": "continuous"},

    "dist_range_mid_atr": {"family": "structure", "type": "continuous"},
    "dist_ma100_atr": {"family": "stretch", "type": "continuous"},
    "volatility_ratio_short_long": {"family": "volatility", "type": "continuous"},
    "volume_percentile": {"family": "liquidity", "type": "continuous"},
    
    # User-requested technical regimes (v17)
    "range_atr": {"family": "context", "type": "continuous"},
    "body_ratio": {"family": "path_structure", "type": "continuous"},
    "upper_wick_ratio": {"family": "liquidity", "type": "continuous"},
    "lower_wick_ratio": {"family": "liquidity", "type": "continuous"},
    "ema20_slope_5h": {"family": "trend", "type": "continuous"},
    "pullback_depth": {"family": "trend", "type": "continuous"},
    "atr_compression_ratio": {"family": "volatility_term_structure", "type": "continuous"},
    "acceleration_norm": {"family": "context", "type": "continuous"},
}

RIDGE_FEATURE_COLS = list(RIDGE_FEATURE_META.keys())

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
    idx = np.arange(window)
    def slope(x):
        return np.polyfit(idx, x, 1)[0]
    return series.rolling(window).apply(slope, raw=True)

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
    def pct_rank(x):
        if len(x) == 0:
            return np.nan
        return float(pd.Series(x).rank(pct=True).iloc[-1])
    return series.rolling(window).apply(pct_rank, raw=True)

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
    df["realized_volatility_24h"] = df["close"].pct_change().rolling(192).std()

    df["atr14"] = atr(df, 28)
    df["atr_change_rate"] = df["atr14"].pct_change()

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

    df["rate_of_change"] = df["close"].pct_change(24)
    df["momentum_zscore"] = rolling_zscore(df["rate_of_change"], 100)

    # -----------------------------
    # Prior move context
    # -----------------------------
    df["prior_return_1h"] = df["close"].pct_change(8)
    df["prior_return_4h"] = df["close"].pct_change(32)
    df["prior_return_12h"] = df["close"].pct_change(96)

    df["prior_range"] = df["high"].rolling(32).max() - df["low"].rolling(32).min()
    df["prior_volatility"] = df["close"].pct_change().rolling(32).std()

    df["acceleration_of_move"] = df["close"].pct_change().diff()

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

    # Direction Entropy 20: entropy of up/down signs over 40 bars
    def _entropy(x):
        if len(x) == 0:
            return np.nan
        p_up = (x > 0).sum() / len(x)
        p_dn = (x < 0).sum() / len(x)
        e = 0.0
        if p_up > 0: e -= p_up * np.log2(p_up)
        if p_dn > 0: e -= p_dn * np.log2(p_dn)
        return e

    df["direction_entropy_20"] = df["close"].diff().rolling(40).apply(_entropy, raw=True)

    # Volatility Term Structure
    std_20 = df["close"].pct_change().rolling(40).std()
    std_100 = df["close"].pct_change().rolling(200).std()
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
    seeds = []
    for _, row in selected_features.iterrows():
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
    req_cols = feature_cols + [target_col]
    if "timestamp" in df.columns:
        req_cols.append("timestamp")

    event_df = df.loc[df[event_col] == 1, req_cols].copy()
    event_df = event_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[target_col])

    if "timestamp" in event_df.columns:
        event_df = event_df.sort_values("timestamp").reset_index(drop=True)
    else:
        event_df = event_df.reset_index(drop=True)

    if len(event_df) < (n_splits + 1) * min_test_rows_per_fold:
        return None

    X = event_df[feature_cols]
    y = event_df[target_col]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_cols,
            )
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("ridge", RidgeCV(alphas=alphas)),
        ]
    )

    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_coefs = []
    fold_scores = []
    cv_scores = []
    fold_diagnostics = []

    for fold_id, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        if len(test_idx) < min_test_rows_per_fold:
            return None # Reject if any fold is too small

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        fold_model = Pipeline(
            steps=[
                ("prep", preprocessor),
                ("ridge", Ridge(alpha=1.0)),
            ]
        )

        fold_model.fit(X_train, y_train)
        y_pred = fold_model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        cv_scores.append(r2)

        ridge = fold_model.named_steps["ridge"]
        coefs = ridge.coef_

        fold_coefs.append(coefs)
        fold_scores.append(
            {
                "fold": fold_id,
                "test_r2": r2,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            }
        )

        if "timestamp" in event_df.columns:
            fold_diagnostics.append({
                "fold": fold_id,
                "train_start_ts": event_df["timestamp"].iloc[train_idx[0]],
                "train_end_ts": event_df["timestamp"].iloc[train_idx[-1]],
                "test_start_ts": event_df["timestamp"].iloc[test_idx[0]],
                "test_end_ts": event_df["timestamp"].iloc[test_idx[-1]],
            })

    model.fit(X, y)
    full_ridge = model.named_steps["ridge"]
    best_alpha = full_ridge.alpha_
    
    # Preprocessor might drop columns if they are all NaN
    # We need to get the actual feature names passed to the ridge model
    try:
        # scikit-learn >= 1.2
        used_features = model.named_steps["prep"].get_feature_names_out()
        # strip the prefix 'num__'
        used_features = [f.replace("num__", "") for f in used_features]
    except:
        used_features = feature_cols

    coef_df = pd.DataFrame(
        {
            "feature": used_features,
            "coef": full_ridge.coef_,
        }
    )

    fold_coef_df = pd.DataFrame(fold_coefs, columns=used_features)

    signs = np.sign(fold_coef_df)
    sign_cons = signs.sum(axis=0).abs() / n_splits

    coef_stability = pd.DataFrame(
        {
            "feature": used_features,
            "coef_mean": fold_coef_df.mean(axis=0).values,
            "coef_std": fold_coef_df.std(axis=0).values,
            "coef_abs_mean": fold_coef_df.abs().mean(axis=0).values,
            "sign_consistency": sign_cons.values
        }
    )

    out = coef_df.merge(coef_stability, on="feature", how="left")
    out["stability_ratio"] = out["coef_abs_mean"] / out["coef_std"].replace(0, np.nan)
    out["signed_importance"] = out["coef"] * out["stability_ratio"].fillna(0.0)
    out["abs_signed_importance"] = out["signed_importance"].abs()

    out = out.sort_values("abs_signed_importance", ascending=False).reset_index(drop=True)

    selected_features = select_promising_regime_variables(out, event_df, max_total=8, max_per_family=2)
    phase3_conditioner_seeds = build_phase3_conditioner_seeds(selected_features, event_df)

    summary = {
        "n_event_rows": int(len(event_df)),
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
        "fold_diagnostics": fold_diagnostics
    }
