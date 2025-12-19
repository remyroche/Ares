import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    log_loss,
    brier_score_loss,
    roc_auc_score,
    average_precision_score,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from typing import List, Tuple, Optional, Any, Dict
import shap

from src.feature_generation.categories.layer3_specific_features import generate_layer3_features
from src.training.steps.labeling.generate_weights_per_label import (
    finalize_sample_weights,
)

from src.utils.purged_kfold import PurgedKFoldTime


def _fast_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true_arr = np.asarray(y_true, dtype=float).reshape(-1)
    y_prob_arr = np.asarray(y_prob, dtype=float).reshape(-1)

    mask = np.isfinite(y_true_arr) & np.isfinite(y_prob_arr)
    if not np.any(mask):
        return 0.0

    y_true_arr = y_true_arr[mask]
    y_prob_arr = y_prob_arr[mask]
    n = int(y_prob_arr.size)
    if n <= 0:
        return 0.0

    y_prob_arr = np.clip(y_prob_arr, 0.0, 1.0)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(y_prob_arr, bin_edges, right=True) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    counts = np.bincount(bin_idx, minlength=n_bins).astype(float)
    sum_prob = np.bincount(bin_idx, weights=y_prob_arr, minlength=n_bins).astype(float)
    sum_true = np.bincount(bin_idx, weights=y_true_arr, minlength=n_bins).astype(float)

    nonzero = counts > 0
    if not np.any(nonzero):
        return 0.0

    mean_prob = np.zeros(n_bins, dtype=float)
    mean_true = np.zeros(n_bins, dtype=float)
    mean_prob[nonzero] = sum_prob[nonzero] / counts[nonzero]
    mean_true[nonzero] = sum_true[nonzero] / counts[nonzero]

    return float(np.sum((counts[nonzero] / n) * np.abs(mean_prob[nonzero] - mean_true[nonzero])))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return np.log(p / (1.0 - p))


def _clip_probs(p: np.ndarray, low: float, high: float) -> np.ndarray:
    arr = np.asarray(p, dtype=float)
    lo = float(low)
    hi = float(high)
    if (not np.isfinite(lo)) or lo <= 0.0:
        lo = 1e-6
    if (not np.isfinite(hi)) or hi >= 1.0:
        hi = 1.0 - 1e-6
    if hi <= lo:
        hi = min(1.0 - 1e-6, lo + 1e-3)
    return np.clip(arr, lo, hi)


def _fit_temperature(y_true: np.ndarray, p_cal: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> float:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(p_cal, dtype=float).reshape(-1)

    m = np.isfinite(y) & np.isfinite(p)
    y = y[m]
    p = p[m]

    if y.size < 50:
        return 1.0

    try:
        if int(np.unique(y).size) < 2:
            return 1.0
    except Exception:
        return 1.0

    w = None
    if sample_weight is not None:
        try:
            w0 = np.asarray(sample_weight, dtype=float).reshape(-1)
            w0 = w0[m]
            w0 = np.where(np.isfinite(w0), w0, 1.0)
            w = w0
        except Exception:
            w = None

    z = _logit(p)
    temps = np.concatenate([
        np.linspace(0.6, 1.8, 25),
        np.linspace(2.0, 6.0, 21),
    ])
    best_t = 1.0
    best_loss = float('inf')
    for t in temps:
        try:
            pt = _sigmoid(z / float(t))
            loss = float(log_loss(y.astype(int), pt, labels=[0, 1], sample_weight=w))
            if np.isfinite(loss) and loss < best_loss:
                best_loss = loss
                best_t = float(t)
        except Exception:
            continue
    return float(best_t) if np.isfinite(best_t) and best_t > 0.0 else 1.0


def _apply_temperature(p: np.ndarray, temperature: float) -> np.ndarray:
    t = float(temperature)
    if (not np.isfinite(t)) or t <= 1e-9:
        t = 1.0
    return _sigmoid(_logit(p) / t)


def _dump_layer3_feature_inventory(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    outcomes_dir: Path,
    symbol: str,
    timeframe: str,
    ts: str,
    stage: str,
    cfg: Optional[Dict[str, Any]] = None,
    meta_prob_col: str = 'meta_prob',
) -> None:
    try:
        if df is None or df.empty:
            return
        if not feature_cols:
            return

        try:
            include_spearman = bool(cfg.get('layer3_feature_inventory_include_spearman', True)) if isinstance(cfg, dict) else True
        except Exception:
            include_spearman = True

        try:
            tail_frac = float(cfg.get('layer3_feature_inventory_tail_frac', 0.2)) if isinstance(cfg, dict) else 0.2
        except Exception:
            tail_frac = 0.2
        if (not np.isfinite(tail_frac)) or tail_frac <= 0.05 or tail_frac >= 0.5:
            tail_frac = 0.2

        n = int(df.shape[0])
        split_idx = int(max(0, min(n, int(np.floor(n * (1.0 - tail_frac))))))
        split_idx = int(max(1, min(n - 1, split_idx))) if n >= 2 else 0

        y = pd.to_numeric(df.get(target_col), errors='coerce').astype(float) if target_col in df.columns else None
        p = pd.to_numeric(df.get(meta_prob_col), errors='coerce').astype(float) if meta_prob_col in df.columns else None

        rows = []
        for col in feature_cols:
            if col not in df.columns:
                continue

            s = pd.to_numeric(df[col], errors='coerce').astype(float)
            arr = s.to_numpy(dtype=float, copy=False)
            finite = np.isfinite(arr)
            n_finite = int(np.sum(finite))
            n_nan = int(arr.size - n_finite)
            pct_nan = float(n_nan / max(1, arr.size))

            mean = float(np.nanmean(arr)) if n_finite > 0 else float('nan')
            std = float(np.nanstd(arr)) if n_finite > 1 else float('nan')
            vmin = float(np.nanmin(arr)) if n_finite > 0 else float('nan')
            vmax = float(np.nanmax(arr)) if n_finite > 0 else float('nan')

            p01 = float(np.nanpercentile(arr, 1)) if n_finite > 10 else float('nan')
            p50 = float(np.nanpercentile(arr, 50)) if n_finite > 0 else float('nan')
            p99 = float(np.nanpercentile(arr, 99)) if n_finite > 10 else float('nan')

            zero_frac = float(np.mean((arr[finite] == 0.0))) if n_finite > 0 else float('nan')

            def _pearson(a: np.ndarray, b: pd.Series) -> float:
                try:
                    bv = b.to_numpy(dtype=float, copy=False)
                    m = np.isfinite(a) & np.isfinite(bv)
                    if int(np.sum(m)) < 10:
                        return float('nan')
                    aa = a[m]
                    bb = bv[m]
                    if float(np.nanstd(aa)) <= 1e-12 or float(np.nanstd(bb)) <= 1e-12:
                        return float('nan')
                    return float(np.corrcoef(aa, bb)[0, 1])
                except Exception:
                    return float('nan')

            corr_target = _pearson(arr, y) if y is not None else float('nan')
            corr_meta = _pearson(arr, p) if p is not None else float('nan')

            spearman_target = float('nan')
            spearman_meta = float('nan')
            if include_spearman and (y is not None):
                try:
                    m = np.isfinite(arr) & np.isfinite(y.to_numpy(dtype=float, copy=False))
                    if int(np.sum(m)) >= 10:
                        spearman_target = float(spearmanr(arr[m], y.to_numpy(dtype=float, copy=False)[m]).correlation)
                except Exception:
                    spearman_target = float('nan')
            if include_spearman and (p is not None):
                try:
                    m = np.isfinite(arr) & np.isfinite(p.to_numpy(dtype=float, copy=False))
                    if int(np.sum(m)) >= 10:
                        spearman_meta = float(spearmanr(arr[m], p.to_numpy(dtype=float, copy=False)[m]).correlation)
                except Exception:
                    spearman_meta = float('nan')

            drift_smd = float('nan')
            drift_nan_delta = float('nan')
            try:
                if n >= 2 and 0 < split_idx < n:
                    a0 = arr[:split_idx]
                    a1 = arr[split_idx:]
                    m0 = np.isfinite(a0)
                    m1 = np.isfinite(a1)
                    if int(np.sum(m0)) >= 10 and int(np.sum(m1)) >= 10:
                        mu0 = float(np.nanmean(a0))
                        mu1 = float(np.nanmean(a1))
                        sd0 = float(np.nanstd(a0))
                        sd1 = float(np.nanstd(a1))
                        pool = float(np.sqrt(0.5 * (sd0 * sd0 + sd1 * sd1)))
                        drift_smd = float(abs(mu1 - mu0) / (pool + 1e-12))
                    drift_nan_delta = float((np.mean(~m1) - np.mean(~m0)))
            except Exception:
                drift_smd = float('nan')

            rows.append(
                {
                    'feature': str(col),
                    'n': int(arr.size),
                    'n_finite': n_finite,
                    'n_nan': n_nan,
                    'pct_nan': pct_nan,
                    'mean': mean,
                    'std': std,
                    'min': vmin,
                    'p01': p01,
                    'p50': p50,
                    'p99': p99,
                    'max': vmax,
                    'zero_frac': zero_frac,
                    'pearson_target': corr_target,
                    'pearson_meta_prob': corr_meta,
                    'spearman_target': spearman_target,
                    'spearman_meta_prob': spearman_meta,
                    'drift_smd': drift_smd,
                    'drift_nan_delta': drift_nan_delta,
                }
            )

        if not rows:
            return

        inv_df = pd.DataFrame(rows)
        try:
            inv_df = inv_df.sort_values(['pct_nan', 'drift_smd'], ascending=[True, False])
        except Exception:
            pass

        try:
            out_csv = outcomes_dir / f"layer3_feature_inventory_{stage}_{symbol}_{timeframe}_{ts}.csv"
            inv_df.to_csv(out_csv, index=False)
        except Exception:
            pass

        try:
            meta = {
                'stage': str(stage),
                'timestamp': str(ts),
                'symbol': str(symbol),
                'timeframe': str(timeframe),
                'n_rows': int(n),
                'n_features': int(len(feature_cols)),
                'target_col': str(target_col),
                'meta_prob_col': str(meta_prob_col) if meta_prob_col in df.columns else None,
                'tail_frac': float(tail_frac),
                'split_idx': int(split_idx),
            }
            out_json = outcomes_dir / f"layer3_feature_inventory_{stage}_{symbol}_{timeframe}_{ts}.json"
            out_json.write_text(json.dumps(meta, indent=2))
        except Exception:
            pass
    except Exception:
        return


def layer3_analyst_lgbm(
    oof_df: pd.DataFrame,
    base_model_cols: List[str],
    target_col: str,
    train_split_date: Optional[str] = None,
    sample_weight: Optional[np.ndarray] = None,
    # New arguments for Scheme comparison
    layer1_weight: Optional[np.ndarray] = None,
    layer2_weight: Optional[np.ndarray] = None,
    net_returns: Optional[np.ndarray] = None,
    market_data: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Any]:
    """
    Transforms diverse Base Model scores into a single Calibrated Probability using LGBM.

    Performs a comparison of 7 specified weighting schemes using ScoreL3 logic:
      ScoreL3 = 100*(AUC-0.5) + 50*(0.693-LogLoss) - 200*ECE

    Selects the best scheme and trains the final production model.
    """
    print(f"\n{'='*60}")
    print("LAYER 3: ANALYST META-MODEL (LGBM + CALIBRATION) - COMPARATIVE MODE")
    print(f"{'='*60}")

    df = oof_df.copy()

    cfg = config if isinstance(config, dict) else {}
    enable_timing = False
    try:
        enable_timing = bool(cfg.get('layer3_timing', False))
    except Exception:
        enable_timing = False
    t0_all = time.perf_counter() if enable_timing else None

    # ---------------------------------------------------------
    # 1. Feature Engineering: Curated Feature Set
    # ---------------------------------------------------------
    print("<< Generating Layer 3 Features (centralized in generate_layer3_features)...")

    # Base models are probabilities; keep them centered around 0.5 when missing.
    if base_model_cols:
        safe_base_cols = [c for c in base_model_cols if c in df.columns]
        if safe_base_cols:
            df[safe_base_cols] = df[safe_base_cols].replace([np.inf, -np.inf], np.nan).fillna(0.5)
    else:
        safe_base_cols = []

    # Attach OHLCV inputs for Layer 3 feature computation (volume/candle/regime).
    if market_data is not None and isinstance(market_data, pd.DataFrame) and not market_data.empty:
        for c in ['volume', 'high', 'low', 'close']:
            if c in market_data.columns:
                df[c] = market_data[c].reindex(df.index)

    # Centralized feature engineering (adds ensemble/disagreement/logit/regime/time features)
    try:
        df = generate_layer3_features(df, safe_base_cols)
    except Exception as e:
        print(f"⚠️ generate_layer3_features failed: {e}")

    # Centralized feature list (only keep columns that exist)
    candidate_features = []
    candidate_features.extend(safe_base_cols)
    candidate_features.extend(
        [
            # Ensemble/core
            'ensemble_prob',
            'logit_prob', 'logit_momentum_5', 'logit_momentum_1',
            'vol_at_signal', 'candle_shape', 'candle_shape_4',
            'base_pred_mean', 'base_pred_std', 'base_pred_range',
            # Cross-timeframe momentum agreement
            'momentum_agreement',
            'momentum_agreement_abs',
            # Removed: 'momentum_weighted_agreement',
            'trend_consistency_12',
            # Disagreement
            'ens_prediction_dispersion',
            'ens_confidence_gap',
            'ens_uncertainty',
            'ens_prediction_range',
            'ens_avg_divergence',
            'ens_max_confidence',
            'ens_disagreement_rate',
            'ens_snr_internal',
            'ens_snr_consensus',
            # Regime/time (GateModel-derived)
            'rv_short', 'rv_short_over_med', 'rv_z_short',
            'slope_short', 'adx_proxy', 'momentum_short', 'snr',
            'time_since_last_vol_spike', 'time_since_last_large_candle',
            'choppiness_index', 'variance_ratio', 'permutation_entropy',
            'hour', 'day_of_week', 'hour_sin', 'hour_cos', 'is_weekend',
            'efficiency_ratio',
        ]
    )

    # Optional context columns already present on events
    for c in ['volatility_1d']:
        if c in df.columns:
            candidate_features.append(c)

    meta_features = [c for c in list(dict.fromkeys(candidate_features)) if c in df.columns]

    # Ensure numeric + stable missingness handling
    other_cols = [c for c in meta_features if c not in set(safe_base_cols)]
    if other_cols:
        df[other_cols] = df[other_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    print(f"   Final Feature Set ({len(meta_features)}): {meta_features}")
    
    # [VERIFICATION] Log presence of critical features
    has_mom = 'momentum_agreement' in meta_features
    has_dis = any(c.startswith('ens_') for c in meta_features)
    base_feats = [c for c in meta_features if c in safe_base_cols]
    print(f"   [FEATURE_CHECK] Momentum Agreement: {has_mom}, Disagreement: {has_dis}")
    print(f"   [FEATURE_CHECK] Base Models Included ({len(base_feats)}): {base_feats}")

    # Clean target
    try:
        neutral_value = cfg.get('layer3_neutral_target_value', 0.5) if isinstance(cfg, dict) else 0.5
        neutral_value = float(neutral_value)
    except Exception:
        neutral_value = 0.5
    try:
        y_num = pd.to_numeric(df[target_col], errors='coerce').astype(float)
        df[target_col] = y_num
        if np.isfinite(neutral_value):
            df.loc[np.isclose(df[target_col].astype(float), neutral_value, atol=1e-12), target_col] = np.nan
    except Exception:
        pass
    # Add logging for target dropna analysis
    initial_rows = len(df)
    nan_counts = df.isna().sum()
    target_nan_count = nan_counts.get(target_col, 0)
    
    print(f"\n=== TARGET DROPNA ANALYSIS ===")
    print(f"Initial rows: {initial_rows}")
    print(f"Target column '{target_col}' NaN count: {target_nan_count}")
    
    # Show top 10 columns with most NaNs
    top_nans = nan_counts.sort_values(ascending=False).head(10)
    print(f"Top 10 columns with NaNs:")
    for col, count in top_nans.items():
        if count > 0:
            pct = (count / initial_rows) * 100
            print(f"  {col}: {count} ({pct:.1f}%)")
    
    df = df.dropna(subset=[target_col])
    final_rows = len(df)
    rows_dropped = initial_rows - final_rows
    
    print(f"After target dropna: {final_rows} rows ({rows_dropped} dropped, {(rows_dropped/initial_rows)*100:.1f}% loss)")
    print("=" * 35)

    # Tight alignment: require Series aligned to oof_df.index.
    # Avoid silent truncation/padding because it invalidates OOF + scheme selection.
    def _require_series_aligned(vec, name: str) -> pd.Series:
        if vec is None:
            raise ValueError(f"{name} is required for Layer3 scheme comparison and must be a pd.Series aligned to oof_df.index")
        if not isinstance(vec, pd.Series):
            raise TypeError(f"{name} must be a pd.Series aligned to oof_df.index (got {type(vec)})")
        if not vec.index.equals(oof_df.index):
            raise ValueError(
                f"{name} index mismatch vs oof_df.index. "
                "Pass a pd.Series with exactly the same DatetimeIndex as oof_df."
            )
        s = pd.to_numeric(vec, errors="coerce").astype(float)
        s = s.replace([np.inf, -np.inf], np.nan)
        return s

    w_l1_s = _require_series_aligned(layer1_weight, "layer1_weight")
    w_l2_s = _require_series_aligned(layer2_weight, "layer2_weight")
    ret_s = _require_series_aligned(net_returns, "net_returns")

    # After target dropna, align by index (no reordering)
    w_l1 = w_l1_s.reindex(df.index).values
    w_l2 = w_l2_s.reindex(df.index).values
    ret_vec = ret_s.reindex(df.index).values

    if len(w_l1) != len(df) or len(w_l2) != len(df) or len(ret_vec) != len(df):
        raise ValueError("Layer3 internal alignment error: weight/return lengths do not match df after target filtering")

    # Outcome-derived magnitude factor (optional; see cfg['layer3_allow_outcome_weighting']).
    magnitude_log = np.log1p(np.clip(ret_vec, 0, None))

    def _coerce_weights(w: np.ndarray) -> np.ndarray:
        arr = np.asarray(w, dtype=float).reshape(-1)
        arr = np.where(np.isfinite(arr), arr, 1.0)
        return arr

    # ---------------------------------------------------------
    # 2. Define Weighting Schemes
    # ---------------------------------------------------------
    # Note: All schemes are finalized using robust MAD scaling (finalize_sample_weights)
    # to ensure they are comparable and standardized (mean=1.0, clipped extremes).
    schemes = {}

    # Always ensure finite weights before MAD-scaling.
    w_l1 = _coerce_weights(w_l1)
    w_l2 = _coerce_weights(w_l2)
    magnitude_log = _coerce_weights(magnitude_log)

    # Scheme 1: target_sample_weight (layer1)
    schemes["S1_L1"] = w_l1

    # Scheme 2: target_sample_weight * final composite weight (layer2)
    schemes["S2_L1_L2"] = (w_l1 * w_l2)

    # Scheme 3: final composite weight (layer2)
    schemes["S3_L2"] = w_l2

    try:
        allow_outcome_weighting = bool(cfg.get('layer3_allow_outcome_weighting', False))
    except Exception:
        allow_outcome_weighting = False

    if allow_outcome_weighting:
        # Scheme 4: log(1+NetReturns) for magnitude integration
        schemes["S4_Mag"] = magnitude_log

        # Scheme 5: target_sample_weight * log(1+NetReturns)
        schemes["S5_L1_Mag"] = (w_l1 * magnitude_log)

        # Scheme 6: final composite weight * log(1+NetReturns)
        schemes["S6_L2_Mag"] = (w_l2 * magnitude_log)

        # Scheme 7: target_sample_weight * final composite weight * log(1+NetReturns)
        schemes["S7_All"] = (w_l1 * w_l2 * magnitude_log)

        # Scheme 8: Asymmetric weighting - downweight losing trades (loss aversion)
        loss_mask = ret_vec < 0
        raw_s8 = np.where(
            loss_mask,
            w_l2 * 0.9,  # Downweight losing trades
            w_l2 * 1.1   # Boost winning trades
        )
        schemes["S8_Asymmetric"] = raw_s8

    # Scheme 9: Class Balanced weighting - compensate for low base rate
    # Ensures winners and losers have equal aggregate weight in training
    y_values = y_num.reindex(df.index).to_numpy()
    try:
        y_bin = (y_values > 0.5).astype(int)
        pos_count = np.sum(y_bin == 1)
        neg_count = np.sum(y_bin == 0)
        if pos_count > 0 and neg_count > 0:
            scale_pos = neg_count / pos_count
            raw_s9 = np.where(y_bin == 1, w_l2 * scale_pos, w_l2)
        else:
            raw_s9 = w_l2
        schemes["S9_ClassBalanced"] = raw_s9
    except Exception:
        schemes["S9_ClassBalanced"] = w_l2

    # ---------------------------------------------------------
    # 3. Comparative Evaluation (2-Phase Scheme Pruning)
    # ---------------------------------------------------------
    # Phase 1: Quick screening on fold 1 only for all schemes
    # Phase 2: Full 5-fold evaluation for top 3 schemes
    # This reduces training calls from 105+ to ~66 (37% reduction)
    print(f"\n>> Phase 1: Quick Screening ({len(schemes)} Schemes, Fold 1 Only)...")

    results = []

    best_score = -float('inf')
    best_scheme_name = None
    best_model_artifacts = None # To store OOF preds and Final Model

    lgbm_params = {
        'objective': 'cross_entropy',
        'metric': 'xentropy',
        'n_estimators': 800,
        'learning_rate': 0.02,
        'max_depth': 7,
        'num_leaves': 63,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 1.0,
        'reg_lambda': 2.0,
        'min_child_samples': 20,
        'min_gain_to_split': 0.005,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbose': -1,
        'n_jobs': 1
    }

    X = df[meta_features]
    y = df[target_col]

    X_values = X.to_numpy(copy=False)
    y_values = y.to_numpy(copy=False)

    # De Prado-style: purged + embargoed sequential folds.
    # Default purge/embargo are derived from bar time delta and lookahead horizon.
    try:
        n_splits = int(cfg.get('cv_splits', 5))
    except Exception:
        n_splits = 5
    n_splits = int(max(3, min(8, n_splits)))  # Ensure 3-8 splits for robust CV

    # Infer bar duration
    bar_td = None
    try:
        if isinstance(df.index, pd.DatetimeIndex) and len(df.index) >= 3:
            deltas = df.index.to_series().diff().dropna()
            if len(deltas) > 0:
                bar_td = deltas.median()
    except Exception:
        bar_td = None

    try:
        purge_bars = int(cfg.get('layer3_purge_bars', 0))
    except Exception:
        purge_bars = 0
    try:
        embargo_bars = int(cfg.get('layer3_embargo_bars', 0))
    except Exception:
        embargo_bars = 0

    if purge_bars <= 0:
        try:
            # Reduced purge to preserve more data
            purge_bars = int(cfg.get('layer3_max_lookahead_bars', 50))
        except Exception:
            purge_bars = 50
    if embargo_bars <= 0:
        # Reduced embargo to preserve more data
        embargo_bars = int(max(1, int(purge_bars // 3)))

    if bar_td is not None and isinstance(bar_td, pd.Timedelta) and pd.notna(bar_td):
        purge = bar_td * int(max(0, purge_bars))
        embargo = bar_td * int(max(0, embargo_bars))
    else:
        purge = int(max(0, purge_bars))
        embargo = int(max(0, embargo_bars))

    cv = PurgedKFoldTime(n_splits=n_splits, purge=purge, embargo=embargo)
    fold_indices = list(cv.split(df))

    # Helper function to evaluate a scheme on specific folds
    try:
        calibration_method_default = str(cfg.get('layer3_calibration_method', 'isotonic'))
    except Exception:
        calibration_method_default = 'isotonic'
    if calibration_method_default not in {'isotonic', 'sigmoid'}:
        calibration_method_default = 'isotonic'

    try:
        calibration_cv_splits = int(cfg.get('layer3_calibration_cv_splits', 3))
    except Exception:
        calibration_cv_splits = 3
    calibration_cv_splits = int(max(3, min(5, calibration_cv_splits)))  # More robust calibration

    def _infer_target_mode(y_arr: np.ndarray) -> str:
        try:
            forced = cfg.get('layer3_target_mode', 'auto')
        except Exception:
            forced = 'auto'
        forced = str(forced).strip().lower()
        if forced in {'binary', 'soft', 'continuous'}:
            return 'soft' if forced in {'soft', 'continuous'} else 'binary'

        yv = np.asarray(y_arr, dtype=float).reshape(-1)
        m = np.isfinite(yv)
        if not bool(np.any(m)):
            return 'binary'
        yv = yv[m]
        u = np.unique(yv)
        if u.size <= 0:
            return 'binary'
        if np.all(np.isclose(u, 0.0)) or np.all(np.isclose(u, 1.0)):
            return 'binary'
        if np.all(np.isin(u, [0.0, 1.0])):
            return 'binary'

        # Only treat as soft if explicitly allowed and values are within [0,1].
        try:
            allow_soft = bool(cfg.get('layer3_allow_soft_targets', True))
        except Exception:
            allow_soft = True
        if allow_soft and float(np.nanmin(yv)) >= 0.0 - 1e-12 and float(np.nanmax(yv)) <= 1.0 + 1e-12:
            return 'soft'
        return 'binary'

    def evaluate_scheme(name, w_vec, fold_list, calibration_method: Optional[str] = None):
        oof_probs = np.full(len(df), np.nan)
        fold_metrics = []
        try:
            target_mode = _infer_target_mode(y_values)
            is_continuous = bool(target_mode == 'soft')

            try:
                clip_low = float(cfg.get('layer3_prob_clip_low', 0.0025)) if isinstance(cfg, dict) else 0.0025
            except Exception:
                clip_low = 0.0025
            try:
                clip_high = float(cfg.get('layer3_prob_clip_high', 0.9975)) if isinstance(cfg, dict) else 0.9975
            except Exception:
                clip_high = 0.9975

            try:
                walkforward_only = bool(cfg.get('layer3_walkforward_only', True)) if isinstance(cfg, dict) else True
            except Exception:
                walkforward_only = True

            try:
                calib_tail_frac = float(cfg.get('layer3_calibration_tail_frac', 0.20)) if isinstance(cfg, dict) else 0.20
            except Exception:
                calib_tail_frac = 0.20
            if (not np.isfinite(calib_tail_frac)) or calib_tail_frac <= 0.05 or calib_tail_frac >= 0.5:
                calib_tail_frac = 0.20

            try:
                temp_scaling = bool(cfg.get('layer3_temperature_scaling_enabled', True)) if isinstance(cfg, dict) else True
            except Exception:
                temp_scaling = True
            
            for fold_idx in fold_list:
                train_idx, test_idx = fold_indices[fold_idx]
                train_idx = np.asarray(train_idx, dtype=int)
                test_idx = np.asarray(test_idx, dtype=int)
                train_idx = np.sort(train_idx)
                test_idx = np.sort(test_idx)

                if walkforward_only and test_idx.size > 0:
                    cutoff = int(test_idx.min())
                    train_idx = train_idx[train_idx < cutoff]

                if train_idx.size < 50 or test_idx.size < 10:
                    continue

                X_train_full = X_values[train_idx]
                X_test = X_values[test_idx]
                y_train_full = y_values[train_idx]
                w_train_full = finalize_sample_weights(w_vec[train_idx])

                n_train = int(len(train_idx))
                calib_n = int(max(30, min(int(np.floor(float(calib_tail_frac) * float(n_train))), n_train - 30)))
                fit_n = int(max(30, n_train - calib_n))

                X_fit = X_train_full[:fit_n]
                y_fit = y_train_full[:fit_n]
                w_fit = w_train_full[:fit_n]

                X_cal = X_train_full[fit_n:]
                y_cal = y_train_full[fit_n:]
                w_cal = w_train_full[fit_n:]

                if is_continuous:
                    # Use Regressor for soft labels (minimizing MSE/Brier score or similar)
                    # objective='regression' (l2) or 'binary' (logloss)? 
                    # If soft labels are probs, regression (MSE) is robust.
                    reg = lgb.LGBMRegressor(**lgbm_params)
                    reg.fit(X_fit, y_fit, sample_weight=w_fit)
                    probs = reg.predict(X_test)
                    probs = _clip_probs(probs, clip_low, clip_high)
                else:
                    # Discrete labels: fit on earlier slice, calibrate on tail slice strictly before test.
                    calib_method_to_use = str(calibration_method or calibration_method_default)
                    if calib_method_to_use not in {'isotonic', 'sigmoid'}:
                        calib_method_to_use = calibration_method_default

                    try:
                        y_fit = (np.asarray(y_fit, dtype=float) >= 0.5).astype(int)
                        y_cal_bin = (np.asarray(y_cal, dtype=float) >= 0.5).astype(int)
                    except Exception:
                        y_fit = np.asarray(y_fit)
                        y_cal_bin = np.asarray(y_cal)

                    base_est = lgb.LGBMClassifier(**lgbm_params)
                    base_est.fit(X_fit, y_fit, sample_weight=w_fit)

                    p_test_raw = base_est.predict_proba(X_test)[:, 1]
                    p_cal_raw = base_est.predict_proba(X_cal)[:, 1]

                    p_test_raw = _clip_probs(p_test_raw, clip_low, clip_high)
                    p_cal_raw = _clip_probs(p_cal_raw, clip_low, clip_high)

                    p_test_cal = p_test_raw
                    p_cal_cal = p_cal_raw

                    if calib_method_to_use == 'isotonic':
                        try:
                            iso = IsotonicRegression(out_of_bounds='clip')
                            iso.fit(p_cal_raw, y_cal_bin.astype(float), sample_weight=w_cal)
                            p_test_cal = iso.predict(p_test_raw)
                            p_cal_cal = iso.predict(p_cal_raw)
                        except Exception:
                            p_test_cal = p_test_raw
                            p_cal_cal = p_cal_raw
                    else:
                        try:
                            z_cal = _logit(p_cal_raw).reshape(-1, 1)
                            lr = LogisticRegression(solver='lbfgs', max_iter=200)
                            lr.fit(z_cal, y_cal_bin.astype(int), sample_weight=w_cal)
                            p_test_cal = lr.predict_proba(_logit(p_test_raw).reshape(-1, 1))[:, 1]
                            p_cal_cal = lr.predict_proba(z_cal)[:, 1]
                        except Exception:
                            p_test_cal = p_test_raw
                            p_cal_cal = p_cal_raw

                    p_test_cal = _clip_probs(p_test_cal, clip_low, clip_high)
                    p_cal_cal = _clip_probs(p_cal_cal, clip_low, clip_high)

                    if temp_scaling:
                        try:
                            t_hat = _fit_temperature(y_cal_bin.astype(int), p_cal_cal, sample_weight=w_cal)
                        except Exception:
                            t_hat = 1.0
                        p_test_cal = _apply_temperature(p_test_cal, t_hat)
                        p_test_cal = _clip_probs(p_test_cal, clip_low, clip_high)

                    probs = p_test_cal
                
                oof_probs[test_idx] = probs

                try:
                    y_fold_true = y_values[test_idx]
                    y_fold_prob = np.asarray(probs, dtype=float)
                    mask_f = np.isfinite(y_fold_true) & np.isfinite(y_fold_prob)
                    if bool(np.any(mask_f)):
                        y_fold_true = y_fold_true[mask_f]
                        y_fold_prob = y_fold_prob[mask_f]
                        if is_continuous:
                            y_fold_bin = (y_fold_true > 0.5).astype(int)
                        else:
                            y_fold_bin = y_fold_true
                        try:
                            pr_auc_f = float(average_precision_score(y_fold_bin, y_fold_prob)) if int(np.unique(y_fold_bin).size) >= 2 else float('nan')
                        except Exception:
                            pr_auc_f = float('nan')
                        try:
                            auc_f = float(roc_auc_score(y_fold_bin, y_fold_prob)) if int(np.unique(y_fold_bin).size) >= 2 else float('nan')
                        except Exception:
                            auc_f = float('nan')
                        try:
                            ll_f = float(log_loss(y_fold_bin, y_fold_prob))
                        except Exception:
                            ll_f = float('nan')
                        try:
                            ece_f = float(_fast_expected_calibration_error(y_fold_bin, y_fold_prob, n_bins=10))
                        except Exception:
                            ece_f = float('nan')
                        fold_metrics.append({"fold": int(fold_idx), "auc": auc_f, "pr_auc": pr_auc_f, "logloss": ll_f, "ece": ece_f})
                except Exception:
                    pass

            mask = ~np.isnan(oof_probs)
            y_true_eval = y_values[mask]
            y_prob_eval = oof_probs[mask]

            if len(y_true_eval) == 0:
                raise ValueError("No valid predictions generated.")
            
            # For soft labels, we might want to bin y_true for AUC? 
            # Or just calculate AUC treating y_true as continuous (ranking).
            # roc_auc_score supports continuous y_true? Yes, it treats them as probabilistic reference?
            # Actually, standard AUC needs binary y_true.
            # If y_true is continuous, we might need to threshold it for binary metrics or rely on LogLoss/IC.
            
            if is_continuous:
                 # Threshold for binary metrics
                 y_true_binary = (y_true_eval > 0.5).astype(int)
            else:
                 y_true_binary = y_true_eval

            try:
                allow_flip = bool(cfg.get('layer3_auto_flip_probabilities', True)) if isinstance(cfg, dict) else True
            except Exception:
                allow_flip = True

            if (not is_continuous) and allow_flip:
                try:
                    auc_raw = float(roc_auc_score(y_true_binary, y_prob_eval))
                    if np.isfinite(auc_raw) and auc_raw < 0.5:
                        oof_probs[mask] = 1.0 - oof_probs[mask]
                        y_prob_eval = 1.0 - y_prob_eval
                except Exception:
                    pass

            try:
                auc = roc_auc_score(y_true_binary, y_prob_eval)
            except ValueError:
                auc = 0.5 # Handle single class edge case
                
            try:
                pr_auc = float(average_precision_score(y_true_binary, y_prob_eval)) if int(np.unique(y_true_binary).size) >= 2 else float('nan')
            except Exception:
                pr_auc = float('nan')

            pos_rate = float(np.mean(y_true_binary)) if y_true_binary.size > 0 else float('nan')
            try:
                pr_auc_w = float(cfg.get('layer3_pr_auc_weight', 50.0)) if isinstance(cfg, dict) else 50.0
            except Exception:
                pr_auc_w = 50.0
            if not np.isfinite(pr_auc_w):
                pr_auc_w = 50.0

            ll = log_loss(y_true_binary, y_prob_eval)
            ece = _fast_expected_calibration_error(y_true_binary, y_prob_eval, n_bins=10)
            pr_delta = (pr_auc - pos_rate) if (np.isfinite(pr_auc) and np.isfinite(pos_rate)) else 0.0
            score = 100 * (auc - 0.5) + 50 * (0.693 - ll) - 200 * ece + float(pr_auc_w) * float(pr_delta)

            try:
                brier = float(brier_score_loss(y_true_binary, y_prob_eval))
            except Exception:
                brier = float('nan')

            # --- Top 30% Quantile Metrics ---
            top30_tpd = float('nan')
            top30_wr = float('nan')
            try:
                if len(y_prob_eval) > 0:
                    # Calculate threshold (70th percentile)
                    thr_70 = np.percentile(y_prob_eval, 70)
                    mask_top30 = y_prob_eval >= thr_70

                    n_top30 = np.sum(mask_top30)

                    # Win Rate
                    if n_top30 > 0:
                        top30_wr = float(np.mean(y_true_binary[mask_top30]))

                    # Trades Per Day
                    # Use full df time range for normalization
                    if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 1:
                        total_seconds = (df.index[-1] - df.index[0]).total_seconds()
                        n_days = total_seconds / 86400.0
                        if n_days > 0:
                            top30_tpd = float(n_top30 / n_days)
            except Exception:
                pass

            # Interpretability Rating (raised thresholds for meaningful classification)
            if score < 0: rating = "Toxic"
            elif score < 0.2: rating = "Weak"
            elif score < 0.4: rating = "Good"
            else: rating = "Excellent"

            def _fold_stats(key: str):
                vals = [m.get(key) for m in fold_metrics if isinstance(m, dict)]
                vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
                if len(vals) == 0:
                    return {
                        "mean": float('nan'),
                        "std": float('nan'),
                        "min": float('nan'),
                        "max": float('nan'),
                        "n": 0,
                    }
                arr = np.asarray(vals, dtype=float)
                return {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "n": int(arr.size),
                }

            auc_stats = _fold_stats("auc")
            pr_stats = _fold_stats("pr_auc")
            ll_stats = _fold_stats("logloss")
            ece_stats = _fold_stats("ece")

            wv = np.asarray(w_vec, dtype=float).reshape(-1)
            wv = wv[np.isfinite(wv)]
            ess_ratio = float('nan')
            try:
                if wv.size > 0:
                    ess = (np.sum(wv) ** 2) / (np.sum(wv * wv) + 1e-12)
                    ess_ratio = float(ess / float(wv.size))
            except Exception:
                pass

            return {
                "Scheme": name,
                "Score": score,
                "AUC": auc,
                "PR_AUC": pr_auc,
                "LogLoss": ll,
                "ECE": ece,
                "Brier": brier,
                "Weight_ESS_Ratio": ess_ratio,
                "Top30_TPD": top30_tpd,
                "Top30_Win": top30_wr,
                "FoldAUC_mean": auc_stats["mean"],
                "FoldAUC_std": auc_stats["std"],
                "FoldAUC_min": auc_stats["min"],
                "FoldAUC_max": auc_stats["max"],
                "FoldAUC_n": auc_stats["n"],
                "FoldPRAUC_mean": pr_stats["mean"],
                "FoldPRAUC_std": pr_stats["std"],
                "FoldPRAUC_min": pr_stats["min"],
                "FoldPRAUC_max": pr_stats["max"],
                "FoldPRAUC_n": pr_stats["n"],
                "FoldLogLoss_mean": ll_stats["mean"],
                "FoldLogLoss_std": ll_stats["std"],
                "FoldLogLoss_min": ll_stats["min"],
                "FoldLogLoss_max": ll_stats["max"],
                "FoldLogLoss_n": ll_stats["n"],
                "FoldECE_mean": ece_stats["mean"],
                "FoldECE_std": ece_stats["std"],
                "FoldECE_min": ece_stats["min"],
                "FoldECE_max": ece_stats["max"],
                "FoldECE_n": ece_stats["n"],
                "Rating": rating,
                "oof_probs": oof_probs,
                "w_vec": w_vec
            }
        except Exception as e:
            print(f"⚠️ Scheme {name} failed: {e}")
            return {
                "Scheme": name,
                "Score": -999,
                "AUC": 0, "PR_AUC": float('nan'), "LogLoss": 99, "ECE": 99, "Rating": "Failed",
                "Top30_TPD": float('nan'), "Top30_Win": float('nan'),
                "FoldAUC_mean": float('nan'),
                "FoldAUC_std": float('nan'),
                "FoldAUC_min": float('nan'),
                "FoldAUC_max": float('nan'),
                "FoldAUC_n": 0,
                "FoldPRAUC_mean": float('nan'),
                "FoldPRAUC_std": float('nan'),
                "FoldPRAUC_min": float('nan'),
                "FoldPRAUC_max": float('nan'),
                "FoldPRAUC_n": 0,
                "FoldLogLoss_mean": float('nan'),
                "FoldLogLoss_std": float('nan'),
                "FoldLogLoss_min": float('nan'),
                "FoldLogLoss_max": float('nan'),
                "FoldLogLoss_n": 0,
                "FoldECE_mean": float('nan'),
                "FoldECE_std": float('nan'),
                "FoldECE_min": float('nan'),
                "FoldECE_max": float('nan'),
                "FoldECE_n": 0,
                "oof_probs": None, "w_vec": w_vec
            }

    # Phase 1: Quick screening on 2 folds (first + last) for stability
    phase1_results = []
    try:
        screen_folds = [0, int(max(0, len(fold_indices) - 1))]
        screen_folds = list(dict.fromkeys([int(f) for f in screen_folds if 0 <= int(f) < int(len(fold_indices))]))
        if not screen_folds:
            screen_folds = [0]
    except Exception:
        screen_folds = [0]
    for name, w_vec in schemes.items():
        print(f"   Screening {name}...")
        result = evaluate_scheme(name, w_vec, screen_folds, calibration_method=calibration_method_default)
        phase1_results.append(result)

    # Sort by score and take top 3 for full evaluation
    phase1_results.sort(key=lambda x: x["Score"], reverse=True)
    top_schemes = phase1_results[:3]
    top_scheme_names = [r["Scheme"] for r in top_schemes]

    print(f"\n>> Phase 2: Full Evaluation (Top 3: {top_scheme_names})...")

    # Phase 2: Full 5-fold evaluation for top 3 schemes
    for name in top_scheme_names:
        print(f"   Full evaluation: {name}...")
        result = evaluate_scheme(name, schemes[name], list(range(len(fold_indices))), calibration_method=calibration_method_default)  # All folds
        results.append(result)

        if result["Score"] > best_score:
            best_score = result["Score"]
            best_scheme_name = name
            best_model_artifacts = {
                "oof_probs": result["oof_probs"],
                "w_vec": result["w_vec"]
            }

    # ---------------------------------------------------------
    # 4. Reporting & Selection
    # ---------------------------------------------------------
    results_df = pd.DataFrame(results).sort_values("Score", ascending=False)

    try:
        ts = None
        try:
            if isinstance(cfg, dict):
                ts = cfg.get('run_timestamp')
        except Exception:
            ts = None
        ts = str(ts or datetime.utcnow().strftime('%Y%m%d_%H%M%S'))
    except Exception:
        ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')

    try:
        symbol = str(cfg.get('symbol', '')) if isinstance(cfg, dict) else ''
    except Exception:
        symbol = ''
    try:
        timeframe = str(cfg.get('timeframe', '')) if isinstance(cfg, dict) else ''
    except Exception:
        timeframe = ''

    try:
        outcomes_dir = Path('outcomes')
        outcomes_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        outcomes_dir = Path('outcomes')

    try:
        scheme_csv = outcomes_dir / f"layer3_scheme_comparison_{symbol}_{timeframe}_{ts}.csv"
        export_cols = [
            c
            for c in [
                'Scheme', 'Score', 'AUC', 'PR_AUC', 'LogLoss', 'ECE', 'Brier', 'Weight_ESS_Ratio', 'Rating',
                'Top30_TPD', 'Top30_Win',
                'FoldAUC_mean', 'FoldAUC_std', 'FoldAUC_min', 'FoldAUC_max', 'FoldAUC_n',
                'FoldPRAUC_mean', 'FoldPRAUC_std', 'FoldPRAUC_min', 'FoldPRAUC_max', 'FoldPRAUC_n',
                'FoldLogLoss_mean', 'FoldLogLoss_std', 'FoldLogLoss_min', 'FoldLogLoss_max', 'FoldLogLoss_n',
                'FoldECE_mean', 'FoldECE_std', 'FoldECE_min', 'FoldECE_max', 'FoldECE_n',
            ]
            if c in results_df.columns
        ]
        results_df[export_cols].to_csv(scheme_csv, index=False)
    except Exception:
        pass

    try:
        _dump_layer3_feature_inventory(
            df=df,
            feature_cols=meta_features,
            target_col=target_col,
            outcomes_dir=outcomes_dir,
            symbol=symbol,
            timeframe=timeframe,
            ts=ts,
            stage='pre_oof',
            cfg=cfg,
            meta_prob_col='meta_prob',
        )
    except Exception:
        pass

    print("\n" + "="*85)
    print("   LAYER 3 WEIGHTING SCHEME COMPARISON")
    print("="*85)
    print(f"{'Scheme':<15} | {'Score':<8} | {'AUC':<6} | {'PR_AUC':<7} | {'LogLoss':<8} | {'ECE':<6} | {'T30_TPD':<7} | {'T30_Win%':<8} | {'Rating'}")
    print("-" * 100)
    for row in results_df.itertuples(index=False):
        # Handle formatting safely
        tpd_s = f"{row.Top30_TPD:.1f}" if np.isfinite(row.Top30_TPD) else "nan"
        win_s = f"{row.Top30_Win:.3f}" if np.isfinite(row.Top30_Win) else "nan"
        pr_s = f"{row.PR_AUC:.4f}" if hasattr(row, 'PR_AUC') and np.isfinite(row.PR_AUC) else "nan"
        print(f"{row.Scheme:<15} | {row.Score:>8.4f} | {row.AUC:>6.4f} | {pr_s:>7} | {row.LogLoss:>8.4f} | {row.ECE:>6.4f} | {tpd_s:>7} | {win_s:>8} | {row.Rating}")
    print("-" * 100)

    print(f"\n🏆 WINNER: {best_scheme_name} (Score: {best_score:.4f})")

    if best_model_artifacts is None:
        print("❌ Critical Failure: No schemes succeeded.")
        # Fallback to simple unweighted
        return df, None

    # ---------------------------------------------------------
    # 5. Final Model Training (Production) using WINNER
    # ---------------------------------------------------------
    print(f">> Training Final Production Model using {best_scheme_name}...")

    df['meta_prob'] = best_model_artifacts['oof_probs']
    w_best_raw = best_model_artifacts['w_vec']
    w_best = finalize_sample_weights(w_best_raw)

    try:
        _dump_layer3_feature_inventory(
            df=df,
            feature_cols=meta_features,
            target_col=target_col,
            outcomes_dir=outcomes_dir,
            symbol=symbol,
            timeframe=timeframe,
            ts=ts,
            stage='post_oof',
            cfg=cfg,
            meta_prob_col='meta_prob',
        )
    except Exception:
        pass

    honest_auc = float('nan')
    honest_pr_auc = float('nan')
    honest_logloss = float('nan')
    honest_ece = float('nan')
    honest_brier = float('nan')
    honest_temperature = float('nan')
    honest_prob_clip_low = float('nan')
    honest_prob_clip_high = float('nan')
    honest_n_train = 0
    honest_n_test = 0
    honest_holdout_start = None

    try:
        holdout_n = cfg.get('layer3_honest_holdout_n') if isinstance(cfg, dict) else None
        holdout_n = int(holdout_n) if holdout_n is not None else 0
    except Exception:
        holdout_n = 0

    try:
        holdout_frac = cfg.get('layer3_honest_holdout_frac', 0.15) if isinstance(cfg, dict) else 0.15
        holdout_frac = float(holdout_frac)
    except Exception:
        holdout_frac = 0.15
    if (not np.isfinite(holdout_frac)) or holdout_frac <= 0.0 or holdout_frac >= 0.5:
        holdout_frac = 0.15

    try:
        n_total = int(len(df))
        if n_total >= 200:
            if holdout_n > 0:
                holdout_n = int(min(max(50, holdout_n), max(50, n_total // 2)))
                holdout_start = int(max(0, n_total - holdout_n))
            else:
                holdout_start = int(max(0, int(np.floor(n_total * (1.0 - holdout_frac)))))
                holdout_start = int(min(max(50, holdout_start), max(50, n_total - 50)))

            honest_holdout_start = holdout_start
            honest_n_test = int(n_total - holdout_start)

            # Respect purge around holdout boundary (avoid adjacent label overlap)
            try:
                purge_bars_int = int(purge_bars)
            except Exception:
                purge_bars_int = 0
            purge_bars_int = int(max(0, purge_bars_int))

            train_end = int(max(0, holdout_start - purge_bars_int))
            honest_n_train = int(train_end)

            if honest_n_train >= 50 and honest_n_test >= 50:
                X_arr = X.to_numpy(copy=False)
                y_arr = pd.to_numeric(df[target_col], errors='coerce').astype(float).to_numpy(copy=False)
                w_arr = np.asarray(w_best, dtype=float).reshape(-1)

                X_train = X_arr[:train_end]
                y_train = y_arr[:train_end]
                w_train = w_arr[:train_end]
                X_test = X_arr[holdout_start:]
                y_test = y_arr[holdout_start:]

                mask_tr = np.isfinite(y_train) & np.all(np.isfinite(X_train), axis=1) & np.isfinite(w_train)
                mask_te = np.isfinite(y_test) & np.all(np.isfinite(X_test), axis=1)

                X_train = X_train[mask_tr]
                y_train = y_train[mask_tr]
                w_train = w_train[mask_tr]
                X_test = X_test[mask_te]
                y_test = y_test[mask_te]

                if len(y_train) >= 50 and len(y_test) >= 50:
                    try:
                        clip_low = float(cfg.get('layer3_prob_clip_low', 0.0025)) if isinstance(cfg, dict) else 0.0025
                    except Exception:
                        clip_low = 0.0025
                    try:
                        clip_high = float(cfg.get('layer3_prob_clip_high', 0.9975)) if isinstance(cfg, dict) else 0.9975
                    except Exception:
                        clip_high = 0.9975

                    try:
                        calib_tail_frac = float(cfg.get('layer3_calibration_tail_frac', 0.20)) if isinstance(cfg, dict) else 0.20
                    except Exception:
                        calib_tail_frac = 0.20
                    if (not np.isfinite(calib_tail_frac)) or calib_tail_frac <= 0.05 or calib_tail_frac >= 0.5:
                        calib_tail_frac = 0.20

                    try:
                        temp_scaling = bool(cfg.get('layer3_temperature_scaling_enabled', True)) if isinstance(cfg, dict) else True
                    except Exception:
                        temp_scaling = True

                    n_tr = int(len(y_train))
                    calib_n = int(max(30, min(int(np.floor(float(calib_tail_frac) * float(n_tr))), n_tr - 30)))
                    fit_n = int(max(30, n_tr - calib_n))

                    X_fit = X_train[:fit_n]
                    y_fit = y_train[:fit_n]
                    w_fit = w_train[:fit_n]

                    X_cal = X_train[fit_n:]
                    y_cal = y_train[fit_n:]
                    w_cal = w_train[fit_n:]

                    base_est = lgb.LGBMClassifier(**lgbm_params)
                    base_est.fit(X_fit, y_fit.astype(int), sample_weight=w_fit)

                    p_test_raw = base_est.predict_proba(X_test)[:, 1]
                    p_cal_raw = base_est.predict_proba(X_cal)[:, 1]

                    p_test_raw = _clip_probs(p_test_raw, clip_low, clip_high)
                    p_cal_raw = _clip_probs(p_cal_raw, clip_low, clip_high)

                    p_test_cal = p_test_raw
                    p_cal_cal = p_cal_raw

                    try:
                        iso = IsotonicRegression(out_of_bounds='clip')
                        iso.fit(p_cal_raw, y_cal.astype(float), sample_weight=w_cal)
                        p_test_cal = iso.predict(p_test_raw)
                        p_cal_cal = iso.predict(p_cal_raw)
                    except Exception:
                        p_test_cal = p_test_raw
                        p_cal_cal = p_cal_raw

                    p_test_cal = _clip_probs(p_test_cal, clip_low, clip_high)
                    p_cal_cal = _clip_probs(p_cal_cal, clip_low, clip_high)

                    t_hat = 1.0
                    if temp_scaling:
                        try:
                            t_hat = _fit_temperature(y_cal.astype(int), p_cal_cal, sample_weight=w_cal)
                        except Exception:
                            t_hat = 1.0
                        p_test_cal = _apply_temperature(p_test_cal, t_hat)
                        p_test_cal = _clip_probs(p_test_cal, clip_low, clip_high)

                    p_test = p_test_cal
                    honest_temperature = float(t_hat)
                    honest_prob_clip_low = float(clip_low)
                    honest_prob_clip_high = float(clip_high)

                    y_bin = y_test.astype(int)
                    if int(np.unique(y_bin).size) >= 2:
                        honest_auc = float(roc_auc_score(y_bin, p_test))
                        try:
                            honest_pr_auc = float(average_precision_score(y_bin, p_test))
                        except Exception:
                            honest_pr_auc = float('nan')
                    else:
                        honest_auc = float('nan')
                        honest_pr_auc = float('nan')
                    honest_logloss = float(log_loss(y_bin, p_test))
                    honest_ece = float(_fast_expected_calibration_error(y_bin, p_test, n_bins=10))
                    try:
                        honest_brier = float(brier_score_loss(y_bin, p_test))
                    except Exception:
                        honest_brier = float('nan')
    except Exception:
        pass

    # Detect continuous again for final training (should match above)
    try:
        target_mode_final = _infer_target_mode(y_values)
    except Exception:
        target_mode_final = 'binary'
    is_continuous = bool(target_mode_final == 'soft')

    try:
        if is_continuous:
             final_model = lgb.LGBMRegressor(**lgbm_params)
             final_model.fit(X, y, sample_weight=w_best)
        else:
            final_base = lgb.LGBMClassifier(**lgbm_params)
            final_tscv = TimeSeriesSplit(n_splits=calibration_cv_splits)
            final_model = CalibratedClassifierCV(
                estimator=final_base,
                method='isotonic',
                cv=final_tscv
            )
            final_model.fit(X, y, sample_weight=w_best)
            
    except Exception as e:
        print(f"⚠️ Final model training failed: {e}")
        final_model = None

    # ---------------------------------------------------------
    # 6. Final Diagnostics (on Best OOF)
    # ---------------------------------------------------------
    # Just reusing the print layout from before for consistency
    meta_prob_numeric = pd.to_numeric(df['meta_prob'], errors='coerce')
    mask = ~meta_prob_numeric.isna()
    y_true = y[mask]
    y_prob = df.loc[mask, 'meta_prob']

    score_logloss = float('nan')
    score_auc = float('nan')
    score_ic = float('nan')
    score_mce = float('nan')
    score_brier = float('nan')
    score_ece = float('nan')

    if len(y_true) > 0:
        # Handle continuous targets for metrics
        if is_continuous:
            y_true_metrics = (y_true > 0.5).astype(int)
        else:
            y_true_metrics = y_true

        score_logloss = log_loss(y_true_metrics, y_prob)
        try: score_auc = roc_auc_score(y_true_metrics, y_prob)
        except: score_auc = 0.5
        score_ic, _ = spearmanr(y_prob, y_true) # Spearman works fine with continuous
        if np.isnan(score_ic): score_ic = 0.0

        prob_true, prob_pred = calibration_curve(y_true_metrics, y_prob, n_bins=10)
        score_mce = np.max(np.abs(prob_true - prob_pred)) if len(prob_true) > 0 else 0.0

        score_ece = _fast_expected_calibration_error(
            np.asarray(y_true_metrics, dtype=float),
            np.asarray(y_prob, dtype=float),
            n_bins=10,
        )

        if is_continuous:
             try:
                 y_true_arr = np.asarray(y_true, dtype=float)
                 y_prob_arr = np.asarray(y_prob, dtype=float)
                 m = np.isfinite(y_true_arr) & np.isfinite(y_prob_arr)
                 score_brier = float(np.mean((y_true_arr[m] - y_prob_arr[m]) ** 2)) if bool(np.any(m)) else float('nan')
             except Exception:
                 score_brier = float('nan')
        else:
             try:
                 score_brier = brier_score_loss(y_true, y_prob)
             except ValueError:
                 try:
                     y_true_arr = np.asarray(y_true, dtype=float)
                     y_prob_arr = np.asarray(y_prob, dtype=float)
                     m = np.isfinite(y_true_arr) & np.isfinite(y_prob_arr)
                     score_brier = float(np.mean((y_true_arr[m] - y_prob_arr[m]) ** 2)) if bool(np.any(m)) else float('nan')
                 except Exception:
                     score_brier = float('nan')

    pr_auc_oof = float('nan')
    try:
        if len(y_true_metrics) > 0 and int(np.unique(y_true_metrics).size) >= 2:
            pr_auc_oof = float(average_precision_score(y_true_metrics, y_prob))
    except Exception:
        pr_auc_oof = float('nan')

    metrics = {
        "Log Loss": f"{score_logloss:.5f}",
        "AUC":      f"{score_auc:.5f}",
        "PR AUC":   f"{pr_auc_oof:.5f}" if np.isfinite(pr_auc_oof) else "nan",
        "IC":       f"{score_ic:.5f}",
        "ECE":      f"{score_ece:.5f}",
        "MCE":      f"{score_mce:.5f}",
        "Brier":    f"{score_brier:.5f}"
    }

    try:
        md_path = outcomes_dir / f"layer3_report_{symbol}_{timeframe}_{ts}.md"
        lines = [
            "# Layer3 Report\n",
            f"- timestamp: {ts}\n",
            f"- symbol: {symbol}\n",
            f"- timeframe: {timeframe}\n",
            f"- n_rows_input: {int(len(oof_df))}\n",
            f"- n_rows_after_target_dropna: {int(len(df))}\n",
            f"- n_base_models: {int(len(base_model_cols or []))}\n",
            f"- winner_scheme: {best_scheme_name}\n",
            f"- winner_score: {float(best_score) if best_score is not None else float('nan')}\n",
            "\n## Winner Metrics (OOF)\n",
        ]
        for k in ['AUC', 'PR AUC', 'Log Loss', 'ECE', 'IC', 'MCE', 'Brier']:
            if k in metrics:
                lines.append(f"- {k}: {metrics[k]}\n")
        lines.append("\n## Honest Holdout Metrics (Forward Tail)\n")
        lines.append(f"- n_train: {int(honest_n_train)}\n")
        lines.append(f"- n_holdout: {int(honest_n_test)}\n")
        lines.append(f"- honest_auc: {float(honest_auc) if np.isfinite(honest_auc) else float('nan')}\n")
        lines.append(f"- honest_pr_auc: {float(honest_pr_auc) if np.isfinite(honest_pr_auc) else float('nan')}\n")
        lines.append(f"- honest_logloss: {float(honest_logloss) if np.isfinite(honest_logloss) else float('nan')}\n")
        lines.append(f"- honest_ece: {float(honest_ece) if np.isfinite(honest_ece) else float('nan')}\n")
        lines.append(f"- honest_brier: {float(honest_brier) if np.isfinite(honest_brier) else float('nan')}\n")
        lines.append(f"- honest_temperature: {float(honest_temperature) if np.isfinite(honest_temperature) else float('nan')}\n")
        lines.append(f"- honest_prob_clip_low: {float(honest_prob_clip_low) if np.isfinite(honest_prob_clip_low) else float('nan')}\n")
        lines.append(f"- honest_prob_clip_high: {float(honest_prob_clip_high) if np.isfinite(honest_prob_clip_high) else float('nan')}\n")

        # Add Weighting Scheme Comparison Table
        lines.append("\n## Weighting Scheme Comparison\n")

        # Markdown table header
        table_cols = ['Scheme', 'Score', 'AUC', 'PR_AUC', 'LogLoss', 'ECE', 'Top30_TPD', 'Top30_Win', 'Rating']
        header = "| " + " | ".join(table_cols) + " |"
        separator = "| " + " | ".join(["---"] * len(table_cols)) + " |"
        lines.append(header + "\n")
        lines.append(separator + "\n")

        for _, row in results_df.iterrows():
            row_str = "|"
            for col in table_cols:
                val = row.get(col, float('nan'))
                if isinstance(val, float):
                    if col == 'Top30_TPD':
                         row_str += f" {val:.1f} |"
                    else:
                         row_str += f" {val:.4f} |"
                else:
                    row_str += f" {val} |"
            lines.append(row_str + "\n")

        md_path.write_text(''.join(lines))
    except Exception:
        pass

    # --- Additional diagnostics: rolling AUC + regime breakdown ---
    try:
        diag_rows = []
        idx = df.index
        if isinstance(idx, pd.DatetimeIndex) and int(len(idx)) >= 200:
            y_all = pd.to_numeric(df[target_col], errors='coerce').astype(float)
            p_all = pd.to_numeric(df['meta_prob'], errors='coerce').astype(float)
            m = y_all.notna() & p_all.notna()
            if int(m.sum()) >= 200:
                y_bin_all = (y_all[m].values >= 0.5).astype(int)
                p_bin_all = np.asarray(p_all[m].values, dtype=float)
                # Rolling windows by time (10 equal bins)
                n = int(len(p_bin_all))
                edges = np.linspace(0, n, 11, dtype=int)
                for i in range(10):
                    a = int(edges[i])
                    b = int(edges[i + 1])
                    if b - a < 50:
                        continue
                    yy = y_bin_all[a:b]
                    pp = p_bin_all[a:b]
                    if int(np.unique(yy).size) < 2:
                        continue
                    diag_rows.append({
                        'slice': f'rolling_decile_{i}',
                        'n': int(b - a),
                        'auc': float(roc_auc_score(yy, pp)),
                        'pr_auc': float(average_precision_score(yy, pp)),
                        'ece': float(_fast_expected_calibration_error(yy, pp, n_bins=10)),
                    })

        # Regime breakdown if available
        for reg_col in ['trend_regime', 'vol_regime']:
            if reg_col in df.columns:
                reg = df[reg_col].astype(str)
                for val in sorted(reg.dropna().unique()):
                    mask_r = (reg == val)
                    y_r = pd.to_numeric(df.loc[mask_r, target_col], errors='coerce').astype(float)
                    p_r = pd.to_numeric(df.loc[mask_r, 'meta_prob'], errors='coerce').astype(float)
                    mm = y_r.notna() & p_r.notna()
                    if int(mm.sum()) < 50:
                        continue
                    yy = (y_r[mm].values >= 0.5).astype(int)
                    pp = np.asarray(p_r[mm].values, dtype=float)
                    if int(np.unique(yy).size) < 2:
                        continue
                    diag_rows.append({
                        'slice': f'{reg_col}={val}',
                        'n': int(mm.sum()),
                        'auc': float(roc_auc_score(yy, pp)),
                        'pr_auc': float(average_precision_score(yy, pp)),
                        'ece': float(_fast_expected_calibration_error(yy, pp, n_bins=10)),
                    })

        if diag_rows:
            diag_path = outcomes_dir / f"layer3_temporal_regime_diagnostics_{symbol}_{timeframe}_{ts}.csv"
            pd.DataFrame(diag_rows).to_csv(diag_path, index=False)
    except Exception:
        pass

    try:
        summary_row = {
            'timestamp': ts,
            'symbol': symbol,
            'timeframe': timeframe,
            'n_rows_input': int(len(oof_df)),
            'n_rows_after_target_dropna': int(len(df)),
            'n_base_models': int(len(base_model_cols or [])),
            'winner_scheme': str(best_scheme_name),
            'winner_score': float(best_score) if best_score is not None else float('nan'),
            'auc': float(score_auc),
            'pr_auc': float(pr_auc_oof) if np.isfinite(pr_auc_oof) else float('nan'),
            'logloss': float(score_logloss),
            'ece': float(score_ece),
            'ic': float(score_ic),
            'mce': float(score_mce),
            'brier': float(score_brier),
            'honest_auc': float(honest_auc) if np.isfinite(honest_auc) else float('nan'),
            'honest_pr_auc': float(honest_pr_auc) if np.isfinite(honest_pr_auc) else float('nan'),
            'honest_logloss': float(honest_logloss) if np.isfinite(honest_logloss) else float('nan'),
            'honest_ece': float(honest_ece) if np.isfinite(honest_ece) else float('nan'),
            'honest_brier': float(honest_brier) if np.isfinite(honest_brier) else float('nan'),
            'honest_n_train': int(honest_n_train),
            'honest_n_holdout': int(honest_n_test),
        }
        pd.DataFrame([summary_row]).to_csv(
            outcomes_dir / f"layer3_summary_{symbol}_{timeframe}_{ts}.csv",
            index=False,
        )
    except Exception:
        pass

    print("\n   WINNER PERFORMANCE (OOF)")
    for k, v in metrics.items():
        print(f"   {k:<10} : {v}")
    print("")

    try:
        if honest_holdout_start is not None and int(honest_n_train) > 0 and int(honest_n_test) > 0:
            print("   HONEST HOLDOUT (Forward Tail)")
            print(f"   n_train   : {int(honest_n_train)}")
            print(f"   n_holdout : {int(honest_n_test)}")
            print(f"   AUC       : {float(honest_auc):.5f}" if np.isfinite(honest_auc) else "   AUC       : nan")
            print(f"   Log Loss  : {float(honest_logloss):.5f}" if np.isfinite(honest_logloss) else "   Log Loss  : nan")
            print(f"   ECE       : {float(honest_ece):.5f}" if np.isfinite(honest_ece) else "   ECE       : nan")
            print(f"   Brier     : {float(honest_brier):.5f}" if np.isfinite(honest_brier) else "   Brier     : nan")
            print("")
    except Exception:
        pass

    if enable_timing and t0_all is not None:
        dt = time.perf_counter() - t0_all
        print(f"Layer3 timing: total_seconds={dt:.3f}")

    # ---------------------------------------------------------
    # 7. SHAP Analysis
    # ---------------------------------------------------------
    _run_shap_analysis(final_model, X, outcomes_dir, symbol, timeframe, ts, md_path)

    # Return full dataframe with predictions + final model
    return df, final_model

def _run_shap_analysis(model, X, output_dir, symbol, timeframe, ts, md_path):
    """
    Computes SHAP values for the final model and saves a summary plot.
    Appends results to the markdown report.
    """
    print("\n>> Running SHAP Analysis on Production Model...")
    try:
        if model is None:
            return

        # Sample data for SHAP (max 1000 rows)
        n_sample = min(1000, len(X))
        if n_sample <= 0:
            return

        # Use random sampling for representativeness (or could use tail)
        # Using tail is better for "current regime" explanation, random for global.
        # Let's use random with fixed seed.
        X_sample = X.sample(n=n_sample, random_state=42)

        shap_values_list = []
        estimators = []

        # Extract estimators
        if isinstance(model, CalibratedClassifierCV):
            if hasattr(model, 'calibrated_classifiers_'):
                for cc in model.calibrated_classifiers_:
                    est = getattr(cc, 'estimator', None) or getattr(cc, 'base_estimator', None)
                    if est:
                        estimators.append(est)
        else:
            # Assume it's a direct LGBMRegressor or Classifier
            estimators.append(model)

        if not estimators:
            print("⚠️ SHAP: Could not extract base estimators from model.")
            return

        print(f"   Aggregating SHAP values from {len(estimators)} estimators...")

        # Calculate SHAP values
        for est in estimators:
            try:
                explainer = shap.TreeExplainer(est)
                vals = explainer.shap_values(X_sample)

                # Handle binary classification output (list of arrays)
                if isinstance(vals, list):
                    # Usually index 1 is positive class
                    if len(vals) == 2:
                        vals = vals[1]
                    else:
                        vals = vals[0] # Fallback

                shap_values_list.append(vals)
            except Exception as e:
                print(f"   ⚠️ Estimator SHAP failed: {e}")

        if not shap_values_list:
            return

        # Average SHAP values
        avg_shap_values = np.mean(shap_values_list, axis=0)

        # 1. Summary Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(avg_shap_values, X_sample, show=False, plot_size=(10, 8))

        plot_filename = f"layer3_shap_{symbol}_{timeframe}_{ts}.png"
        plot_path = output_dir / plot_filename
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"   SHAP plot saved to: {plot_path}")

        # 2. Text Summary (Top features)
        # Calculate mean absolute SHAP value per feature
        feature_importance = pd.DataFrame(
            list(zip(X_sample.columns, np.abs(avg_shap_values).mean(0))),
            columns=['feature', 'importance']
        )
        feature_importance.sort_values(by='importance', ascending=False, inplace=True)
        top_20 = feature_importance.head(20)

        print("\n   TOP 20 FEATURES BY SHAP IMPORTANCE:")
        print(top_20.to_string(index=False))

        # 3. Append to Markdown Report
        if md_path and md_path.exists():
            with open(md_path, 'a') as f:
                f.write("\n\n## SHAP Feature Importance (Global)\n")
                f.write(f"![SHAP Summary]({plot_filename})\n\n")

                f.write("### Top 20 Features\n")
                f.write("| Feature | Mean |SHAP| |\n")
                f.write("| --- | --- |\n")
                for _, row in top_20.iterrows():
                    f.write(f"| {row['feature']} | {row['importance']:.6f} |\n")
                f.write("\n")

    except Exception as e:
        print(f"⚠️ SHAP Analysis failed: {e}")
        import traceback
        traceback.print_exc()

# ---------------------------------------------------------
# Helper: Advanced Diagnostic Plot (Unchanged)
# ---------------------------------------------------------
def plot_diagnostics(y_true, y_prob, output_path=None):
    """
    Plots Reliability Diagram (Calibration) AND Probability Density (Resolution).
    """
    try:
        # Remove NaNs with robust numeric casting
        y_prob_numeric = pd.to_numeric(y_prob, errors='coerce')
        y_true_numeric = pd.to_numeric(y_true, errors='coerce')
        mask = ~y_prob_numeric.isna() & ~y_true_numeric.isna()
        y_true = y_true_numeric[mask]
        y_prob = y_prob_numeric[mask]

        if len(y_true) == 0:
            return

        fig, ax = plt.subplots(1, 2, figsize=(14, 6))

        # 1. Reliability Diagram
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        ax[0].plot(prob_pred, prob_true, marker='o', linewidth=2, label='Meta-Model')
        ax[0].plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.5, label='Perfect')
        ax[0].set_xlabel('Predicted Probability')
        ax[0].set_ylabel('Actual Win Rate')
        ax[0].set_title('Calibration (Reliability)')
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)

        # 2. Probability Density (Histogram)
        sns.histplot(y_prob, bins=20, kde=True, ax=ax[1], color='purple', alpha=0.6)
        ax[1].set_xlim(0, 1)
        ax[1].set_xlabel('Predicted Probability')
        ax[1].set_title('Resolution (Confidence Distribution)')
        ax[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path)
            print(f"Diagnostics plot saved to {output_path}")
        else:
            pass
        plt.close(fig)
    except Exception as e:
        print(f"Failed to generate plots: {e}")
