"""
Layer 1: Sample Weighting Optimization.

This module contains logic for optimizing sample weighting parameters (Layer 1)
of the meta-labeling HPO pipeline. It optimizes parameters for magnitude,
uniqueness, and other weighting components to maximize the information content
and stability of the training set.

Enhanced to use Layer0-optimized unified prices for better signal quality.
"""

from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    from scipy.stats import entropy as scipy_entropy, spearmanr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    scipy_entropy = None
    spearmanr = None

from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning
from src.utils.ml_common.transaction_costs import DEFAULT_TRANSACTION_COST
from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
    BayesianTPEOptimizer,
    OptimizationConfig,
)
from src.training.steps.labeling.generate_weights_per_label import (
    generate_weights_per_label,
    compute_multi_horizon_consistency,
    compute_uniqueness_weights,
    _coerce_numeric_array,
)

# Import Layer0 unified price generation
try:
    from src.training.steps.labeling.unified_price_layer2 import load_layer0_params, generate_unified_layer2_price, apply_hampel_filter
    LAYER0_AVAILABLE = True
except ImportError:
    LAYER0_AVAILABLE = False
    load_layer0_params = None
    generate_unified_layer2_price = None
    apply_hampel_filter = None

# Import Optimized Wavelet Decomposition
try:
    from src.training.steps.labeling.optimized_wavelet_decomposition import OptimizedWaveletDecomposition
except ImportError:
    OptimizedWaveletDecomposition = None


def build_layer1_probe_features(
    close_series: pd.Series,
    market_data: pd.DataFrame,
    t_events: pd.DatetimeIndex,
    event_consistency: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Construct the shared probe feature matrix used by both the main (predictive CV)
    probe and the secondary confident-learning probe.

    Feature set (≈13 columns):
        - Momentum: ret1, ret3, ret6
        - Volatility: vol20, rel_vol (vol20/vol100)
        - Oscillators: rsi14, stochastic %K
        - Trend/Deviation: Bollinger %B, trend deviation, efficiency ratio
        - Wavelet noise proxies: wavelet_d1_vol, wavelet_d2_vol
        - Consistency proxy: event_consistency (if available)
    """

    close = close_series.astype(float)

    ret1 = close.pct_change(1).reindex(t_events)
    ret3 = close.pct_change(3).reindex(t_events)
    ret6 = close.pct_change(6).reindex(t_events)

    vol20_raw = close.pct_change().rolling(20).std()
    vol100_raw = close.pct_change().rolling(100).std()
    vol20 = vol20_raw.reindex(t_events)
    rel_vol = (vol20_raw / (vol100_raw + 1e-9)).reindex(t_events)

    rsi_feat = pd.Series(50.0, index=t_events)
    try:
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi_feat = rsi.reindex(t_events)
    except Exception:
        pass

    bb_pct_b_feat = pd.Series(0.0, index=t_events)
    try:
        bb_ma = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        bb_pct_b = (close - bb_ma) / (2 * bb_std + 1e-9)
        bb_pct_b_feat = bb_pct_b.reindex(t_events)
    except Exception:
        pass

    stoch_k_feat = pd.Series(0.5, index=t_events)
    try:
        if 'high' in market_data.columns and 'low' in market_data.columns:
            high_roll = market_data['high'].rolling(14).max()
            low_roll = market_data['low'].rolling(14).min()
        else:
            high_roll = close.rolling(14).max()
            low_roll = close.rolling(14).min()
        stoch_k = (close - low_roll) / (high_roll - low_roll + 1e-9)
        stoch_k_feat = stoch_k.reindex(t_events)
    except Exception:
        pass

    er_feat = pd.Series(0.5, index=t_events)
    try:
        change = close.diff(10).abs()
        path = close.diff(1).abs().rolling(10).sum()
        er = change / (path + 1e-9)
        er_feat = er.reindex(t_events)
    except Exception:
        pass

    trend_feat = pd.Series(0.0, index=t_events)
    try:
        sma50 = close.rolling(window=50).mean()
        trend_dev = (close - sma50) / (sma50 + 1e-9)
        trend_feat = trend_dev.reindex(t_events)
    except Exception:
        pass

    wavelet_d1_feat = pd.Series(0.0, index=t_events)
    wavelet_d2_feat = pd.Series(0.0, index=t_events)
    if OptimizedWaveletDecomposition is not None:
        try:
            # Use Strictly Causal Mode
            wavelet_engine = OptimizedWaveletDecomposition(
                wavelet='db4',
                scales=['d1', 'd2'],
                max_level=2,
                verbose=False,
                causal=True
            )
            decomp = wavelet_engine.decompose_signal_vectorized(close.values)
            if 'd1' in decomp:
                d1_series = pd.Series(decomp['d1'], index=close.index)
                d1_vol = d1_series.rolling(window=20).std()
                wavelet_d1_feat = d1_vol.reindex(t_events).fillna(0.0)
            if 'd2' in decomp:
                d2_series = pd.Series(decomp['d2'], index=close.index)
                d2_vol = d2_series.rolling(window=20).std()
                wavelet_d2_feat = d2_vol.reindex(t_events).fillna(0.0)
        except Exception as e_wav:
            tprint_warning(f"⚠️ Wavelet feature generation failed: {e_wav}")

    if event_consistency is not None and len(event_consistency) == len(t_events):
        cons_feat = pd.Series(event_consistency, index=t_events).fillna(0.5)
    else:
        cons_feat = pd.Series(0.5, index=t_events)

    feature_df = (
        pd.DataFrame(
            {
                "ret1": ret1,
                "ret3": ret3,
                "ret6": ret6,
                "vol20": vol20,
                "rel_vol": rel_vol,
                "rsi14": rsi_feat,
                "bb_pct_b": bb_pct_b_feat,
                "stoch_k": stoch_k_feat,
                "er": er_feat,
                "trend_dev": trend_feat,
                "wavelet_d1_vol": wavelet_d1_feat,
                "wavelet_d2_vol": wavelet_d2_feat,
                "consistency": cons_feat,
            },
            index=t_events,
        )
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    return feature_df


def safe_layer1_objective(
    weights: np.ndarray,
    returns: np.ndarray,
    concurrency: np.ndarray,
    volatility: np.ndarray,
    noise_threshold: float = 0.001,
    quality_scores: Optional[np.ndarray] = None,
) -> float:
    """
    Calculate a safe objective score for Layer 1 weighting optimization.

    The score rewards:
    - Correlation with magnitude (MAS)
    - Entropy (WES)
    - Correlation with quality scores (QAS) if available

    And penalizes:
    - Weight on noise (NWP)
    - Correlation with uniqueness/concurrency (UOP)
    - Correlation with volatility (VDP)
    """
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return -10.0
    if not np.isfinite(w).all():
        return -10.0

    total = float(w.sum())
    if total <= 0:
        return -10.0

    w_norm = w / total

    r = np.asarray(returns, dtype=float)
    if r.size != w_norm.size:
        return -10.0

    n = w_norm.size
    if n < 2:
        return -10.0

    abs_returns = np.abs(r)

    def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.size < 2 or y.size < 2:
            return 0.0
        if not np.isfinite(x).any() or not np.isfinite(y).any():
            return 0.0
        try:
            if spearmanr is not None:
                corr, _ = spearmanr(x, y)
                if corr is None or not np.isfinite(corr):
                    return 0.0
                return float(corr)
        except Exception:
            pass

        try:
            x_center = x - np.nanmean(x)
            y_center = y - np.nanmean(y)
            denom = (
                np.sqrt(np.nanmean(x_center ** 2))
                * np.sqrt(np.nanmean(y_center ** 2))
            )
            if denom <= 0:
                return 0.0
            return float(np.nanmean(x_center * y_center) / denom)
        except Exception:
            return 0.0

    mas = _safe_corr(w, abs_returns)
    mas = max(0.0, mas)

    try:
        if SCIPY_AVAILABLE and scipy_entropy is not None:
            wes = float(scipy_entropy(w_norm) / np.log(float(n)))
        else:
            ent = -float(np.sum(w_norm * np.log(w_norm + 1e-12)))
            wes = ent / np.log(float(n)) if n > 1 else 0.0
    except Exception:
        wes = 0.0

    noise_mask = abs_returns < noise_threshold
    nwp = float(w_norm[noise_mask].sum()) if noise_mask.any() else 0.0

    concurrency_arr = np.asarray(concurrency, dtype=float)
    uop_corr = _safe_corr(w, concurrency_arr)
    uop_penalty = max(0.0, uop_corr)

    vol_arr = np.asarray(volatility, dtype=float)
    vdp_corr = _safe_corr(w, vol_arr)
    vdp_penalty = max(0.0, vdp_corr - 0.6)

    qas_reward = 0.0
    if quality_scores is not None:
        q_arr = np.asarray(quality_scores, dtype=float)
        # We reward weights that correlate with quality (clean labels)
        qas_corr = _safe_corr(w, q_arr)
        qas_reward = max(0.0, qas_corr)

    try:
        w_sorted = np.sort(w_norm)[::-1]
        k10 = int(max(1, round(0.10 * float(n))))
        top10_share = float(np.sum(w_sorted[:k10]))
    except Exception:
        top10_share = 0.0

    try:
        w_median = float(np.median(w))
        max_to_median = float(np.max(w) / (w_median + 1e-12)) if w_median > 0 else float(np.max(w))
    except Exception:
        max_to_median = 0.0

    concentration_penalty = 0.0
    concentration_penalty += 2.0 * max(0.0, top10_share - 0.25)
    concentration_penalty += 0.05 * max(0.0, max_to_median - 5.0)

    score = (
        1.0 * mas
        + 1.5 * wes
        + 1.0 * qas_reward
        - 2.0 * nwp
        - 1.0 * uop_penalty
        - 1.0 * vdp_penalty
        - 1.0 * concentration_penalty
    )

    if not np.isfinite(score):
        return -10.0
    return float(score)


def run_layer1_optimization(
    symbol: str,
    timeframe: str,
    market_data: pd.DataFrame,
    labels: pd.Series,
    committee_agreement_scores: Optional[pd.Series] = None,
    committee_mag_factors: Optional[pd.Series] = None,
    n_trials: int = 60,
    objective_mode: str = "proxy",
    transaction_cost: float = DEFAULT_TRANSACTION_COST,
    uniqueness_horizon_bars: int = 24,
    use_layer0_prices: bool = True,
) -> Dict[str, Any]:
    """
    Run Layer 1 weighting optimization with optional Layer0 price integration.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        market_data: Market data DataFrame
        labels: Label series
        committee_agreement_scores: Committee agreement scores
        committee_mag_factors: Committee magnitude factors
        n_trials: Number of optimization trials
        objective_mode: Objective mode ('proxy', 'predictive_cv')
        transaction_cost: Transaction cost
        uniqueness_horizon_bars: Uniqueness horizon in bars
        use_layer0_prices: Whether to use Layer0-optimized prices
        
    Returns:
        Best weighting parameters
    """
    tprint_info(f"⚙️ Running Layer 1 Weighting Optimization for {symbol} {timeframe}...")
    
    # Load Layer0 parameters if available and requested
    layer0_params = None
    if use_layer0_prices and LAYER0_AVAILABLE:
        try:
            layer0_params = load_layer0_params()
            tprint_info(f"✅ Loaded Layer0 params: Q={layer0_params.get('kalman_Q', 'N/A')}, R={layer0_params.get('kalman_R', 'N/A')}")
        except Exception as e:
            tprint_warning(f"⚠️ Failed to load Layer0 params: {e}")
            layer0_params = None
    
    # Use Layer0-optimized prices if available
    if use_layer0_prices and layer0_params is not None:
        try:
            # Generate Layer0-optimized unified price
            optimized_price = generate_unified_layer2_price(market_data, layer0_params)
            
            # Replace raw close with optimized price for better signal quality
            enhanced_market_data = market_data.copy()
            enhanced_market_data['unified_price'] = optimized_price
            enhanced_market_data['close'] = optimized_price  # Replace raw close
            
            tprint_info(f"✅ Using Layer0-optimized unified price for Layer1 optimization")
            
            # Update close series for volatility calculation
            close_series = enhanced_market_data['close']
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to generate Layer0 unified price: {e}, using raw close")
            close_series = market_data['close']
    else:
        close_series = market_data['close']
    
    # Rest of the function remains the same...
    returns_series = labels.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns_series) < 50:
        tprint_warning(
            f"⚠️ Layer 1: insufficient events for optimization (n={len(returns_series)}). Using defaults.",
        )
        return {
            'mag_compression': 0.8,
            'learn_slope': 0.0,
            'learn_center': 0.5,
            'uniq_intensity': 1.0,
            'quality_intensity': 0.0,
            'quality_floor': 0.2,
            'exp_mag': 1.0,
            'exp_learn': 1.0,
            'exp_uniq': 1.0,
            'exp_cross': 1.0,
            'downside_multiplier': 1.0,
            'time_decay_halflife': 0.0,
            'committee_agreement_alpha': 0.5,
            'committee_mag_clip': 5.0,
        }

    if 'close' not in market_data.columns:
        tprint_warning("⚠️ Layer 1: market_data missing 'close' column. Using defaults.")
        return {
            'mag_compression': 0.8,
            'learn_slope': 0.0,
            'learn_center': 0.5,
            'uniq_intensity': 1.0,
            'quality_intensity': 0.0,
            'quality_floor': 0.2,
            'exp_mag': 1.0,
            'exp_learn': 1.0,
            'exp_uniq': 1.0,
            'exp_cross': 1.0,
            'downside_multiplier': 1.0,
            'time_decay_halflife': 0.0,
            'committee_agreement_alpha': 0.5,
            'committee_mag_clip': 5.0,
        }

    close_series = market_data['close'].astype(float)

    # Simple volatility proxy: rolling standard deviation of returns
    close_ret = close_series.pct_change()
    vol_series = close_ret.rolling(20).std().replace([np.inf, -np.inf], np.nan)

    t_events = returns_series.index
    vol_proxy = vol_series.reindex(t_events).astype(float).values
    if not np.isfinite(vol_proxy).any():
        vol_proxy = None

    returns_arr = returns_series.values.astype(float)

    n_samples = int(len(returns_arr))

    # Confident-learning style per-event label quality (out-of-sample probabilities)
    # This is used as an additional multiplicative component inside generate_weights_per_label.
    cl_quality_scores = None
    try:
        from src.training.steps.labeling.confident_learning import (
            get_cross_val_pred_probs,
            compute_label_quality_scores,
        )

        y_cl = (np.asarray(returns_arr, dtype=float) > 0.0).astype(int)
        if int(np.unique(y_cl).size) >= 2 and shared_probe_features is not None:
            feat_df = shared_probe_features.reindex(t_events).replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0.0)

            pred_probs = get_cross_val_pred_probs(
                feat_df.values,
                y_cl,
                model=None,
                n_splits=3,
                random_state=42,
            )
            cl_quality_scores = compute_label_quality_scores(
                y_cl, pred_probs, method="self_confidence"
            )
            cl_quality_scores = np.asarray(cl_quality_scores, dtype=float)
            cl_quality_scores = np.where(np.isfinite(cl_quality_scores), cl_quality_scores, 1.0)
            cl_quality_scores = np.clip(cl_quality_scores, 0.0, 1.0)
    except Exception:
        cl_quality_scores = None

    # Heuristic floor for "small" returns (used in objective)
    finite_abs = np.abs(returns_arr[np.isfinite(returns_arr)])
    if finite_abs.size:
        small_ret_thr = float(np.quantile(finite_abs, 0.25))
    else:
        small_ret_thr = 0.0

    # Build per-event volatility for the objective
    if vol_proxy is None:
        event_volatility = np.zeros_like(returns_arr)
    else:
        event_volatility = np.asarray(vol_proxy, dtype=float)
        if event_volatility.shape[0] != returns_arr.shape[0]:
            event_volatility = np.resize(event_volatility, returns_arr.shape[0])
        non_finite_vol = ~np.isfinite(event_volatility)
        if non_finite_vol.all():
            event_volatility = np.zeros_like(returns_arr)
        else:
            median_vol = float(
                np.nanmedian(event_volatility[~non_finite_vol])
            )
            event_volatility = np.where(
                non_finite_vol, median_vol, event_volatility
            )

    # Approximate per-event concurrency using local event density
    horizon_bars = 12
    idx = market_data.index
    if len(idx) >= 2:
        try:
            bar_deltas = idx.to_series().diff().dropna()
            bar_delta = bar_deltas.median()
        except Exception:
            bar_delta = None
    else:
        bar_delta = None

    if not isinstance(bar_delta, pd.Timedelta) or bar_delta <= pd.Timedelta(0):
        bar_delta = pd.Timedelta(minutes=15)

    window_span = horizon_bars * bar_delta
    t_events_arr = t_events.to_numpy()
    event_concurrency = np.ones(len(t_events_arr), dtype=float)
    try:
        window = window_span.to_timedelta64()
        order = np.argsort(t_events_arr)
        t_sorted = t_events_arr[order]
        left_bounds = t_sorted - window
        right_bounds = t_sorted + window
        left_idx = np.searchsorted(t_sorted, left_bounds, side='left')
        right_idx = np.searchsorted(t_sorted, right_bounds, side='right')
        concurrency_sorted = (right_idx - left_idx).astype(float)
        event_concurrency[order] = concurrency_sorted
    except Exception:
        event_concurrency = np.zeros(len(t_events_arr), dtype=float)
        for i, ts in enumerate(t_events_arr):
            left_ts = ts - window_span
            right_ts = ts + window_span
            mask = (t_events_arr >= left_ts) & (t_events_arr <= right_ts)
            event_concurrency[i] = float(mask.sum())

    # Convert concurrency into a simple uniqueness proxy (1 / concurrency)
    event_uniqueness = 1.0 / np.maximum(1.0, event_concurrency)
    try:
        uniq_h = int(uniqueness_horizon_bars)
    except Exception:
        uniq_h = 24
    uniq_h = int(max(1, min(uniq_h, 500)))

    try:
        pos = idx.searchsorted(t_events)
        pos = np.asarray(pos, dtype=int)
        pos_end = np.clip(pos + uniq_h, 0, len(idx) - 1)
        t1 = pd.Series(idx[pos_end], index=t_events)
        uniq_series = compute_uniqueness_weights(t1, t_events, idx)
        uniq_arr = uniq_series.reindex(t_events).astype(float).values
        if uniq_arr.shape[0] == returns_arr.shape[0] and np.isfinite(uniq_arr).any():
            event_uniqueness = np.where(np.isfinite(uniq_arr) & (uniq_arr > 0.0), uniq_arr, event_uniqueness)
    except Exception:
        pass

    try:
        cons_series = compute_multi_horizon_consistency(close_series, horizons=[12, 48])
        event_consistency = (
            cons_series.reindex(t_events)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.5)
            .values
        )
        event_consistency = np.asarray(event_consistency, dtype=float)
        if event_consistency.shape[0] != returns_arr.shape[0]:
            event_consistency = None
    except Exception:
        event_consistency = None

    shared_probe_features: Optional[pd.DataFrame] = None
    try:
        shared_probe_features = build_layer1_probe_features(
            close_series=close_series,
            market_data=market_data,
            t_events=t_events,
            event_consistency=event_consistency,
        )
    except Exception as e_probe:
        tprint_warning(f"⚠️ Layer 1 probe feature generation failed: {e_probe}")
        shared_probe_features = None

    try:
        objective_mode_local = str(objective_mode or "predictive_cv").strip().lower()
    except Exception:
        objective_mode_local = "predictive_cv"

    # Pre-compute features for predictive CV objective
    X_proxy_arr: Optional[np.ndarray] = None
    y_proxy_arr: Optional[np.ndarray] = None

    if objective_mode_local == "predictive_cv":
        try:
            y_proxy_arr = (np.asarray(returns_arr, dtype=float) > 0.0).astype(int)

            if shared_probe_features is None:
                shared_probe_features = build_layer1_probe_features(
                    close_series=close_series,
                    market_data=market_data,
                    t_events=t_events,
                    event_consistency=event_consistency,
                )

            X_df_proxy = (
                shared_probe_features.reindex(t_events)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )
            X_proxy_arr = X_df_proxy.values.astype(float)
        except Exception as e_feat:
            tprint_warning(f"⚠️ Failed to generate features for Layer 1 predictive CV: {e_feat}")
            X_proxy_arr = None

    def _predictive_cv_score(
        weights: np.ndarray,
        X_in: np.ndarray,
        y_in: np.ndarray,
        n_splits: int = 3,
    ) -> float:
        try:
            if X_in is None or y_in is None:
                return -10.0

            if int(np.unique(y_in).size) < 2:
                return -10.0

            w = np.asarray(weights, dtype=float)
            w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
            if float(np.sum(w)) <= 0.0:
                w = np.ones_like(w, dtype=float)

            n_samples_local = int(len(y))
            if n_samples_local < 80:
                n_splits = 2
            n_splits = int(max(2, min(int(n_splits), 5)))
            if n_samples_local < (n_splits + 1) * 10:
                return -10.0

            # For predictive CV in HPO, simple TimeSeriesSplit is sufficient and robust
            tscv = TimeSeriesSplit(n_splits=n_splits)
            splits = tscv.split(X_in)

            fold_aucs: List[float] = []
            fold_prs: List[float] = []

            for tr_idx, te_idx in splits:
                y_tr = y_in[tr_idx]
                y_te = y_in[te_idx]
                if int(np.unique(y_tr).size) < 2 or int(np.unique(y_te).size) < 2:
                    continue

                model = LogisticRegression(
                    solver="lbfgs",
                    max_iter=500,
                    n_jobs=1,
                )
                model.fit(X_in[tr_idx], y_tr, sample_weight=w[tr_idx])
                p_te = model.predict_proba(X_in[te_idx])[:, 1]

                sw_te = w[te_idx]
                try:
                    auc = float(roc_auc_score(y_te, p_te, sample_weight=sw_te))
                except Exception:
                    auc = float(roc_auc_score(y_te, p_te))

                try:
                    pr = float(average_precision_score(y_te, p_te, sample_weight=sw_te))
                except Exception:
                    pr = float(average_precision_score(y_te, p_te))

                if np.isfinite(auc):
                    fold_aucs.append(auc)
                if np.isfinite(pr):
                    fold_prs.append(pr)

            if len(fold_aucs) < 1:
                return -10.0

            mean_auc = float(np.mean(fold_aucs))
            mean_pr = float(np.mean(fold_prs)) if len(fold_prs) else 0.0
            score = mean_auc + 0.50 * mean_pr
            if not np.isfinite(score):
                return -10.0
            return float(score)
        except Exception:
            return -10.0

    committee_components_available = False
    committee_agree_arr = _coerce_numeric_array(
        committee_agreement_scores,
        n_samples,
        "committee_agreement_scores",
        fill_value=0.0,
        allow_negative=False,
    )
    committee_mag_arr = _coerce_numeric_array(
        committee_mag_factors,
        n_samples,
        "committee_mag_factors",
        fill_value=1.0,
        allow_negative=False,
    )

    if committee_agree_arr is not None or committee_mag_arr is not None:
        committee_components_available = True
        if committee_agree_arr is None:
            committee_agree_arr = np.zeros(n_samples, dtype=float)
        if committee_mag_arr is None:
            committee_mag_arr = np.ones(n_samples, dtype=float)
        committee_agree_arr = np.where(np.isfinite(committee_agree_arr), committee_agree_arr, 0.0)
        committee_agree_arr = np.clip(committee_agree_arr, 0.0, 1.0)
        committee_mag_arr = np.where(np.isfinite(committee_mag_arr) & (committee_mag_arr > 0.0), committee_mag_arr, 1.0)

    if committee_components_available and committee_agree_arr is not None and committee_mag_arr is not None:
        try:
            def _safe_corr(x_arr: np.ndarray, y_arr: np.ndarray) -> float:
                x_arr = np.asarray(x_arr, dtype=float)
                y_arr = np.asarray(y_arr, dtype=float)
                if x_arr.size < 2 or y_arr.size < 2:
                    return 0.0
                if not np.isfinite(x_arr).any() or not np.isfinite(y_arr).any():
                    return 0.0
                try:
                    if spearmanr is not None:
                        corr, _ = spearmanr(x_arr, y_arr)
                        if corr is None or not np.isfinite(corr):
                            return 0.0
                        return float(corr)
                except Exception:
                    pass

                try:
                    x_center = x_arr - np.nanmean(x_arr)
                    y_center = y_arr - np.nanmean(y_arr)
                    denom = (
                        np.sqrt(np.nanmean(x_center ** 2))
                        * np.sqrt(np.nanmean(y_center ** 2))
                    )
                    if denom <= 0:
                        return 0.0
                    return float(np.nanmean(x_center * y_center) / denom)
                except Exception:
                    return 0.0

            agree_v = np.asarray(committee_agree_arr, dtype=float)
            mag_v = np.asarray(committee_mag_arr, dtype=float)
            abs_ret_v = np.abs(np.asarray(returns_arr, dtype=float))

            agree_std = float(np.nanstd(agree_v)) if np.isfinite(agree_v).any() else 0.0
            tprint_info(
                "   [Layer1 committee] agreement stats: "
                f"mean={float(np.nanmean(agree_v)):.4f}, std={agree_std:.4f}, "
                f"min={float(np.nanmin(agree_v)):.4f}, max={float(np.nanmax(agree_v)):.4f}"
            )
            tprint_info(
                "   [Layer1 committee] magnitude stats: "
                f"mean={float(np.nanmean(mag_v)):.4f}, std={float(np.nanstd(mag_v)):.4f}, "
                f"min={float(np.nanmin(mag_v)):.4f}, max={float(np.nanmax(mag_v)):.4f}"
            )
            tprint_info(
                "   [Layer1 committee] correlations (Spearman if available): "
                f"corr(agree,|ret|)={_safe_corr(agree_v, abs_ret_v):.4f}, "
                f"corr(mag,|ret|)={_safe_corr(mag_v, abs_ret_v):.4f}, "
                f"corr(agree,concurrency)={_safe_corr(agree_v, event_concurrency):.4f}, "
                f"corr(agree,vol)={_safe_corr(agree_v, event_volatility):.4f}"
            )
            if agree_std < 1e-3:
                tprint_warning(
                )
        except Exception:
            pass

    default_params = {
        'mag_compression': 1.0,
        'learn_slope': 1.0,
        'learn_center': 0.5,
        'uniq_intensity': 1.0,
        'quality_intensity': 1.0,
        'quality_floor': 0.0,
        'exp_mag': 1.0,
        'exp_learn': 1.0,
        'exp_uniq': 1.0,
        'exp_cross': 1.0,
        'downside_multiplier': 1.0,
        'mag_clip_pct': 0.99,
        'time_decay_halflife': 1.0,
        'committee_agreement_alpha': 1.0,
        'committee_mag_clip': 3.0,
    }

    search_space: Dict[str, Dict[str, Any]] = {
        # --- A. INFORMATION HANDLING (Magnitude & Uniqueness) ---
        # How much do we reward high returns?
        # 0.5 = Sqrt (Conservative), 1.0 = Linear, 1.5 = Convex
        'mag_compression': {
            'type': 'float',
            'low': 0.90,
            'high': 1.20,
            # step used only by grid utilities; TPE ignores it but it's harmless
            'step': 0.05,
            'log': False,
        },
        # How strictly do we punish concurrent/overlapping events?
        'uniq_intensity': {
            'type': 'float',
            'low': 1.00,
            'high': 3.00,
            'log': False,
        },

        # Confident-learning quality weight (optional; intensity 0 disables)
        'quality_intensity': {
            'type': 'float',
            'low': 0.0,
            'high': 5.0,
            'log': False,
        },
        'quality_floor': {
            'type': 'float',
            'low': 0.05,
            'high': 0.60,
            'log': False,
        },

        # --- C. COMPONENT MIXING (Exponents / Power Law) ---
        'exp_mag': {
            'type': 'float',
            'low': 1.0,
            'high': 1.5,
            'log': True,
        },
        'exp_learn': {
            'type': 'float',
            'low': 0.0,
            'high': 1.5,
            'log': False,
        },
        'exp_uniq': {
            'type': 'float',
            'low': 1.0,
            'high': 1.5,
            'log': True,
        },
        'exp_cross': {
            'type': 'float',
            'low': 0.5,
            'high': 2.0,
            'log': False,
        },

        # --- D. ASYMMETRY (Risk Management) ---
        'downside_multiplier': {
            'type': 'float',
            'low': 1.0,
            'high': 1.4,
            'log': False,
        },

        # --- E. NOISE CLIPPING ---
        'mag_clip_pct': {
            'type': 'float',
            'low': 0.95,
            'high': 0.99,
            'log': False,
        },
        'time_decay_halflife': {
            'type': 'float',
            'low': 0.0,
            'high': 2.0,
            'log': False,
        },
    }

    if committee_components_available:
        search_space.update(
            {
                'committee_agreement_alpha': {
                    'type': 'float',
                    'low': 0.0,
                    'high': 2.0,
                    'log': False,
                },
                'committee_mag_clip': {
                    'type': 'float',
                    'low': 1.0,
                    'high': 10.0,
                    'log': False,
                },
            }
        )

    def objective(params: Dict[str, Any]) -> float:
        try:
            weights = generate_weights_per_label(
                returns=returns_arr,
                t_events=t_events,
                close_series=close_series,
                consistency_scores=event_consistency,
                label_quality_scores=cl_quality_scores,
                uniqueness_scores=event_uniqueness,
                vol_proxy=vol_proxy,
                transaction_cost=float(transaction_cost),
                mag_compression=float(params.get('mag_compression', default_params['mag_compression'])),
                learn_slope=float(default_params.get('learn_slope', 0.0)),
                learn_center=float(default_params.get('learn_center', 0.5)),
                uniq_intensity=float(params.get('uniq_intensity', default_params['uniq_intensity'])),
                quality_intensity=float(params.get('quality_intensity', default_params['quality_intensity'])),
                quality_floor=float(params.get('quality_floor', default_params['quality_floor'])),
                exp_mag=float(params.get('exp_mag', default_params['exp_mag'])),
                exp_learn=float(params.get('exp_learn', default_params.get('exp_learn', 1.0))),
                exp_uniq=float(params.get('exp_uniq', default_params['exp_uniq'])),
                exp_cross=float(params.get('exp_cross', default_params.get('exp_cross', 1.0))),
                downside_multiplier=float(params.get('downside_multiplier', default_params['downside_multiplier'])),
                mag_clip_pct=float(params.get('mag_clip_pct', 0.99)),
                time_decay_halflife=float(params.get('time_decay_halflife', 0.0)),
            )
            if not np.isfinite(weights).all() or weights.sum() <= 0:
                return -10.0

            if committee_components_available and committee_agree_arr is not None and committee_mag_arr is not None:
                try:
                    alpha = float(
                        params.get(
                            'committee_agreement_alpha',
                            default_params.get('committee_agreement_alpha', 0.5),
                        )
                    )
                except Exception:
                    alpha = float(default_params.get('committee_agreement_alpha', 0.5))

                try:
                    mag_clip = float(
                        params.get(
                            'committee_mag_clip',
                            default_params.get('committee_mag_clip', 5.0),
                        )
                    )
                except Exception:
                    mag_clip = float(default_params.get('committee_mag_clip', 5.0))

                alpha = float(np.clip(alpha, 0.0, 10.0))
                mag_clip = float(np.clip(mag_clip, 0.5, 50.0))

                cf = (1.0 + alpha * committee_agree_arr) * np.clip(committee_mag_arr, 0.0, mag_clip)
                cf = np.where(np.isfinite(cf) & (cf > 0.0), cf, 1.0)
                cf_mean = float(np.mean(cf)) if cf.size else 1.0
                if np.isfinite(cf_mean) and cf_mean > 0:
                    cf = cf / cf_mean
                else:
                    cf = np.ones_like(cf, dtype=float)

                weights = np.asarray(weights, dtype=float) * cf
                w_sum = float(np.sum(weights)) if weights.size else 0.0
                if np.isfinite(w_sum) and w_sum > 0:
                    weights = weights * (len(weights) / w_sum)
                else:
                    weights = np.ones(len(returns_arr), dtype=float)

            if objective_mode_local == "predictive_cv" and X_proxy_arr is not None and y_proxy_arr is not None:
                score = _predictive_cv_score(
                    weights=np.asarray(weights, dtype=float),
                    X_in=X_proxy_arr,
                    y_in=y_proxy_arr,
                    n_splits=3,
                )
            else:
                score = safe_layer1_objective(
                    weights=weights,
                    returns=returns_arr,
                    concurrency=event_concurrency,
                    volatility=event_volatility,
                    noise_threshold=float(small_ret_thr) if small_ret_thr > 0 else 0.001,
                    quality_scores=cl_quality_scores,
                )
            return float(score)
        except Exception as e:
            tprint_warning(f"⚠️ Layer 1 objective failure: {e}")
            return -10.0

    try:
        n_trials_i = int(n_trials)
    except Exception:
        n_trials_i = 60
    n_trials_i = max(5, min(n_trials_i, 250))

    opt_config = OptimizationConfig(
        n_trials=n_trials_i,
        execution_mode="light",
        direction="maximize",
        seed=42,
        enable_staged_optimization=False,
        coarse_grid_trials=0,
        fine_grid_trials=0,
        tpe_trials=n_trials_i,
    )

    optimizer = BayesianTPEOptimizer(config=opt_config)
    result = optimizer.optimize(objective=objective, search_space=search_space)

    best_params_raw = result.get('best_params') or {}
    best_value = result.get('best_value')

    best_params: Dict[str, Any] = default_params.copy()
    for key in default_params.keys():
        if key in best_params_raw:
            try:
                best_params[key] = float(best_params_raw[key])
            except Exception:
                continue

    if not committee_components_available:
        best_params.pop('committee_agreement_alpha', None)
        best_params.pop('committee_mag_clip', None)

    if 'mag_clip_pct' in best_params_raw:
        try:
            best_params['mag_clip_pct'] = float(best_params_raw['mag_clip_pct'])
        except Exception:
            pass

    if best_value is not None and np.isfinite(best_value):
        tprint_success(
            f"✅ Layer 1 optimization complete. Best score={best_value:.4f}",
        )
    else:
        tprint_success("✅ Layer 1 optimization complete.")

    tprint_info(f"   Best weighting params: {best_params}")

    # Persist Layer 1 trial metrics for correlation analysis
    try:
        def _compute_l1_metrics(params: Dict[str, Any]) -> Dict[str, Any]:
            try:
                weights = generate_weights_per_label(
                    returns=returns_arr,
                    t_events=t_events,
                    close_series=close_series,
                    consistency_scores=event_consistency,
                    label_quality_scores=cl_quality_scores,
                    uniqueness_scores=event_uniqueness,
                    vol_proxy=vol_proxy,
                    transaction_cost=float(transaction_cost),
                    mag_compression=float(params.get('mag_compression', default_params['mag_compression'])),
                    learn_slope=float(params.get('learn_slope', default_params['learn_slope'])),
                    learn_center=float(params.get('learn_center', default_params['learn_center'])),
                    uniq_intensity=float(params.get('uniq_intensity', default_params['uniq_intensity'])),
                    quality_intensity=float(params.get('quality_intensity', default_params['quality_intensity'])),
                    quality_floor=float(params.get('quality_floor', default_params['quality_floor'])),
                    exp_mag=float(params.get('exp_mag', default_params['exp_mag'])),
                    exp_learn=float(params.get('exp_learn', default_params.get('exp_learn', 1.0))),
                    exp_uniq=float(params.get('exp_uniq', default_params['exp_uniq'])),
                    exp_cross=float(params.get('exp_cross', default_params.get('exp_cross', 1.0))),
                    downside_multiplier=float(params.get('downside_multiplier', default_params['downside_multiplier'])),
                    mag_clip_pct=float(params.get('mag_clip_pct', 0.99)),
                    time_decay_halflife=float(params.get('time_decay_halflife', 0.0)),
                )
                if not np.isfinite(weights).all() or weights.sum() <= 0:
                    return {
                        "score": -10.0,
                        "weights_mean": np.nan,
                        "weights_min": np.nan,
                        "weights_max": np.nan,
                        "weights_entropy": np.nan,
                        "weights_entropy_norm": np.nan,
                        "mas": np.nan,
                        "wes": np.nan,
                        "nwp": np.nan,
                        "uop_penalty": np.nan,
                        "vdp_penalty": np.nan,
                        "committee_alpha": np.nan,
                        "committee_mag_clip": np.nan,
                        "committee_factor_mean": np.nan,
                        "committee_factor_min": np.nan,
                        "committee_factor_max": np.nan,
                        "n_events": int(len(returns_arr)),
                    }
            except Exception:
                return {
                    "score": -10.0,
                    "weights_mean": np.nan,
                    "weights_min": np.nan,
                    "weights_max": np.nan,
                    "weights_entropy": np.nan,
                    "weights_entropy_norm": np.nan,
                    "mas": np.nan,
                    "wes": np.nan,
                    "nwp": np.nan,
                    "uop_penalty": np.nan,
                    "vdp_penalty": np.nan,
                    "committee_alpha": np.nan,
                    "committee_mag_clip": np.nan,
                    "committee_factor_mean": np.nan,
                    "committee_factor_min": np.nan,
                    "committee_factor_max": np.nan,
                    "n_events": int(len(returns_arr)),
                }

            committee_alpha_val = np.nan
            committee_mag_clip_val = np.nan
            committee_factor_mean = np.nan
            committee_factor_min = np.nan
            committee_factor_max = np.nan

            if committee_components_available and committee_agree_arr is not None and committee_mag_arr is not None:
                try:
                    committee_alpha_val = float(
                        params.get(
                            'committee_agreement_alpha',
                            default_params.get('committee_agreement_alpha', 0.5),
                        )
                    )
                except Exception:
                    committee_alpha_val = float(default_params.get('committee_agreement_alpha', 0.5))

                try:
                    committee_mag_clip_val = float(
                        params.get(
                            'committee_mag_clip',
                            default_params.get('committee_mag_clip', 5.0),
                        )
                    )
                except Exception:
                    committee_mag_clip_val = float(default_params.get('committee_mag_clip', 5.0))

                committee_alpha_val = float(np.clip(committee_alpha_val, 0.0, 10.0))
                committee_mag_clip_val = float(np.clip(committee_mag_clip_val, 0.5, 50.0))

                cf = (1.0 + committee_alpha_val * committee_agree_arr) * np.clip(
                    committee_mag_arr, 0.0, committee_mag_clip_val
                )
                cf = np.where(np.isfinite(cf) & (cf > 0.0), cf, 1.0)
                cf_mean = float(np.mean(cf)) if cf.size else 1.0
                if np.isfinite(cf_mean) and cf_mean > 0:
                    cf = cf / cf_mean
                else:
                    cf = np.ones_like(cf, dtype=float)

                try:
                    committee_factor_mean = float(np.mean(cf)) if cf.size else np.nan
                    committee_factor_min = float(np.min(cf)) if cf.size else np.nan
                    committee_factor_max = float(np.max(cf)) if cf.size else np.nan
                except Exception:
                    committee_factor_mean = np.nan
                    committee_factor_min = np.nan
                    committee_factor_max = np.nan

                weights = np.asarray(weights, dtype=float) * cf
                w_sum = float(np.sum(weights)) if weights.size else 0.0
                if np.isfinite(w_sum) and w_sum > 0:
                    weights = weights * (len(weights) / w_sum)
                else:
                    weights = np.ones(len(returns_arr), dtype=float)

            # Recompute objective components for transparency
            w = np.asarray(weights, dtype=float)
            total = float(w.sum())
            w_norm = w / total if total > 0 else w

            r = np.asarray(returns_arr, dtype=float)
            abs_returns = np.abs(r)

            def _safe_corr(x_arr: np.ndarray, y_arr: np.ndarray) -> float:
                x_arr = np.asarray(x_arr, dtype=float)
                y_arr = np.asarray(y_arr, dtype=float)
                if x_arr.size < 2 or y_arr.size < 2:
                    return 0.0
                if not np.isfinite(x_arr).any() or not np.isfinite(y_arr).any():
                    return 0.0
                try:
                    if spearmanr is not None:
                        corr, _ = spearmanr(x_arr, y_arr)
                        if corr is None or not np.isfinite(corr):
                            return 0.0
                        return float(corr)
                except Exception:
                    pass
                try:
                    x_center = x_arr - np.nanmean(x_arr)
                    y_center = y_arr - np.nanmean(y_arr)
                    denom = (
                        np.sqrt(np.nanmean(x_center ** 2))
                        * np.sqrt(np.nanmean(y_center ** 2))
                    )
                    if denom <= 0:
                        return 0.0
                    return float(np.nanmean(x_center * y_center) / denom)
                except Exception:
                    return 0.0

            mas = max(0.0, _safe_corr(w, abs_returns))

            try:
                if SCIPY_AVAILABLE and scipy_entropy is not None:
                    wes = float(scipy_entropy(w_norm) / np.log(float(len(w_norm))))
                else:
                    ent = -float(np.sum(w_norm * np.log(w_norm + 1e-12)))
                    wes = ent / np.log(float(len(w_norm))) if len(w_norm) > 1 else 0.0
            except Exception:
                wes = 0.0

            noise_mask = abs_returns < (float(small_ret_thr) if small_ret_thr > 0 else 0.001)
            nwp = float(w_norm[noise_mask].sum()) if noise_mask.any() else 0.0

            concurrency_arr = np.asarray(event_concurrency, dtype=float)
            uop_penalty = max(0.0, _safe_corr(w, concurrency_arr))

            vol_arr = np.asarray(event_volatility, dtype=float)
            vdp_penalty = max(0.0, _safe_corr(w, vol_arr) - 0.6)

            score = (
                1.0 * mas
                + 1.5 * wes
                - 2.0 * nwp
                - 1.0 * uop_penalty
                - 1.0 * vdp_penalty
            )

            w_valid = weights[np.isfinite(weights)]
            weights_mean = float(w_valid.mean()) if w_valid.size else np.nan
            weights_min = float(w_valid.min()) if w_valid.size else np.nan
            weights_max = float(w_valid.max()) if w_valid.size else np.nan
            weights_entropy = np.nan
            weights_entropy_norm = np.nan
            if w_valid.size > 1:
                w_sum = float(w_valid.sum())
                if w_sum > 0:
                    w_norm = w_valid / w_sum
                    entropy_val = -float(np.sum(w_norm * np.log(w_norm + 1e-12)))
                    max_entropy = np.log(float(len(w_norm)))
                    weights_entropy = entropy_val
                    weights_entropy_norm = float(entropy_val / max_entropy) if max_entropy > 0 else np.nan

            return {
                "score": float(score),
                "weights_mean": weights_mean,
                "weights_min": weights_min,
                "weights_max": weights_max,
                "weights_entropy": weights_entropy,
                "weights_entropy_norm": weights_entropy_norm,
                "mas": float(mas),
                "wes": float(wes),
                "nwp": float(nwp),
                "uop_penalty": float(uop_penalty),
                "vdp_penalty": float(vdp_penalty),
                "committee_alpha": committee_alpha_val,
                "committee_mag_clip": committee_mag_clip_val,
                "committee_factor_mean": committee_factor_mean,
                "committee_factor_min": committee_factor_min,
                "committee_factor_max": committee_factor_max,
                "n_events": int(len(returns_arr)),
            }

        # Log best-score decomposition for the final best_params
        try:
            best_metrics = _compute_l1_metrics(best_params)
            tprint_info(
                "   Layer 1 best score components: "
                f"score={best_metrics.get('score', float('nan')):.4f}, "
                f"MAS={best_metrics.get('mas', float('nan')):.4f}, "
                f"WES={best_metrics.get('wes', float('nan')):.4f}, "
                f"NWP={best_metrics.get('nwp', float('nan')):.4f}, "
                f"UOP_penalty={best_metrics.get('uop_penalty', float('nan')):.4f}, "
                f"VDP_penalty={best_metrics.get('vdp_penalty', float('nan')):.4f}, "
                f"committee_alpha={best_metrics.get('committee_alpha', float('nan'))}, "
                f"committee_mag_clip={best_metrics.get('committee_mag_clip', float('nan'))}, "
                f"committee_factor_min={best_metrics.get('committee_factor_min', float('nan'))}, "
                f"committee_factor_max={best_metrics.get('committee_factor_max', float('nan'))}"
            )
        except Exception:
            pass

        trial_rows = []
        for trial in result.get("history", []):
            params = trial.get("params", {}) if isinstance(trial, dict) else {}
            metrics = _compute_l1_metrics(params)
            row = {
                **metrics,
            }
            for k, v in params.items():
                row[f"param_{k}"] = v
            trial_rows.append(row)

        if trial_rows:
            ts_l1 = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            l1_trials_path = Path("outcomes") / f"hpo_layer1_trials_{symbol}_{timeframe}_{ts_l1}.csv"
            pd.DataFrame(trial_rows).to_csv(l1_trials_path, index=False)
            tprint_info(f"   💾 Saved Layer 1 trial metrics to {l1_trials_path}")
    except Exception as l1_trials_exc:
        tprint_warning(f"   ⚠️ Failed to save Layer 1 trial metrics: {l1_trials_exc}")

    return best_params



def execute_layer1_step(
    symbol: str,
    timeframe: str,
    market_data: pd.DataFrame,
    baseline_t_events: pd.DatetimeIndex,
    baseline_returns_clean: pd.Series,
    committee_agreement_scores_l1: Optional[np.ndarray],
    committee_mag_factors_l1: Optional[np.ndarray],
    config: Dict[str, Any],
    start_rank: int,
    layer1_rank: int,
    loaded_params: Optional[Dict[str, Any]] = None,
    loaded_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Execute the Layer 1 (Weight Optimization) step of the pipeline.

    - Checks if the step should be skipped based on `start_rank`.
    - If skipped, returns `loaded_params`.
    - Otherwise, runs `run_layer1_optimization`.
    - Handles defaults on failure or insufficient data.
    - Persists results to JSON.

    Enhanced to use Layer0-optimized prices for better signal quality.
    """
    layer1_loaded_from: Optional[str] = None
    best_weighting_params: Dict[str, Any]

    if layer1_rank < start_rank:
        best_weighting_params = dict(loaded_params or {})
        layer1_loaded_from = str(loaded_path) if loaded_path is not None else None
        tprint_info(
            f"♻️ Layer 1 skipped (start_rank={start_rank}); loaded best params from {layer1_loaded_from}"
        )
    else:
        if len(baseline_t_events) < 50:
            tprint_warning(f"⚠️ Too few baseline events ({len(baseline_t_events)}) for Layer 1. Using defaults.")
            best_weighting_params = {
                'mag_compression': 0.8, 'learn_slope': 10.0, 'learn_center': 0.4,
                'uniq_intensity': 2.0, 'exp_mag': 1.5, 'exp_learn': 1.0,
                'exp_uniq': 1.5, 'exp_cross': 1.0, 'downside_multiplier': 1.0
            }
        else:
            try:
                try:
                    tx_cost = float(config.get("transaction_cost", DEFAULT_TRANSACTION_COST))
                except Exception:
                    tx_cost = float(DEFAULT_TRANSACTION_COST)

                try:
                    uniq_h = int(config.get("layer1_uniqueness_horizon_bars", 24))
                except Exception:
                    uniq_h = 24

                # Check if Layer0 price integration is enabled
                use_layer0_prices = bool(config.get("layer1_use_layer0_prices", True))
                if use_layer0_prices and not LAYER0_AVAILABLE:
                    tprint_warning("⚠️ Layer0 price integration requested but not available, using raw prices")
                    use_layer0_prices = False

                best_weighting_params = run_layer1_optimization(
                    symbol=symbol,
                    timeframe=timeframe,
                    market_data=market_data,
                    labels=baseline_returns_clean,
                    committee_agreement_scores=committee_agreement_scores_l1,
                    committee_mag_factors=committee_mag_factors_l1,
                    n_trials=int(config.get("layer1_n_trials", 60)),
                    objective_mode=str(config.get("layer1_objective_mode", "proxy")),
                    transaction_cost=tx_cost,
                    uniqueness_horizon_bars=uniq_h,
                    use_layer0_prices=use_layer0_prices,
                )
            except Exception as e:
                tprint_warning(f"⚠️ Layer 1 optimization failed: {e}. Using defaults.")
                best_weighting_params = {
                    'mag_compression': 0.8, 'learn_slope': 10.0, 'learn_center': 0.4,
                    'uniq_intensity': 2.0, 'exp_mag': 1.5, 'exp_learn': 1.0,
                    'exp_uniq': 1.5, 'exp_cross': 1.0, 'downside_multiplier': 1.0
                }

    tprint_success(f"✅ Layer 1 Complete. Best Weighting Params: {best_weighting_params}")

    # Persist Layer 1 params immediately
    l1_path: Optional[Path] = None
    try:
        ts = config.get("run_timestamp") or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        l1_path = Path("outcomes") / f"hpo_layer1_best_params_{symbol}_{timeframe}_{ts}.json"
        l1_payload = {
            "best_params": best_weighting_params,
            "timestamp": ts,
            "symbol": symbol,
            "timeframe": timeframe,
            "source": "optimization" if layer1_loaded_from is None else "loaded",
            "loaded_from": layer1_loaded_from,
            "layer0_prices_used": bool(config.get("layer1_use_layer0_prices", True)) and LAYER0_AVAILABLE,
        }
        with open(l1_path, "w") as f:
            json.dump(l1_payload, f, indent=2, default=str)
        tprint_info(f"💾 Saved Layer 1 best params to {l1_path}")
    except Exception as e:
        tprint_warning(f"⚠️ Failed to save Layer 1 best params: {e}")
    try:
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        outcomes_dir = Path("outcomes")

    try:
        ts_report = str(config.get("run_timestamp") or datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
    except Exception:
        ts_report = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    try:
        ret_arr = pd.to_numeric(baseline_returns_clean, errors="coerce").astype(float)
        n_evt = int(ret_arr.notna().sum())
        pos_rate = float((ret_arr > 0.0).mean()) if n_evt > 0 else float("nan")
        mean_ret = float(ret_arr.mean()) if n_evt > 0 else float("nan")
        std_ret = float(ret_arr.std()) if n_evt > 1 else float("nan")
    except Exception:
        n_evt, pos_rate, mean_ret, std_ret = 0, float("nan"), float("nan"), float("nan")

    try:
        md_path = outcomes_dir / f"layer1_report_{symbol}_{timeframe}_{ts_report}.md"
        lines = [
            "# Layer1 Report\n",
            f"- timestamp: {ts_report}\n",
            f"- symbol: {symbol}\n",
            f"- timeframe: {timeframe}\n",
            f"- n_events: {n_evt}\n",
            f"- pos_rate: {pos_rate}\n",
            f"- mean_return: {mean_ret}\n",
            f"- std_return: {std_ret}\n",
            f"- loaded_from: {str(layer1_loaded_from) if layer1_loaded_from else ''}\n",
            f"- saved_best_params: {str(l1_path) if l1_path else ''}\n",
            f"- layer0_prices_used: {bool(config.get('layer1_use_layer0_prices', True)) and LAYER0_AVAILABLE}\n",
            "\n## Best Params\n",
        ]
        for k in sorted(best_weighting_params.keys()):
            try:
                lines.append(f"- {k}: {best_weighting_params.get(k)}\n")
            except Exception:
                continue
        md_path.write_text("".join(lines))
    except Exception:
        pass

    try:
        summary_row: Dict[str, Any] = {
            "timestamp": ts_report,
            "symbol": symbol,
            "timeframe": timeframe,
            "n_events": int(n_evt),
            "pos_rate": pos_rate,
            "mean_return": mean_ret,
            "std_return": std_ret,
            "loaded_from": str(layer1_loaded_from) if layer1_loaded_from else "",
            "saved_best_params": str(l1_path) if l1_path else "",
            "layer0_prices_used": bool(config.get("layer1_use_layer0_prices", True)) and LAYER0_AVAILABLE,
        }
        for k, v in (best_weighting_params or {}).items():
            summary_row[f"param_{k}"] = v
        csv_path = outcomes_dir / f"layer1_summary_{symbol}_{timeframe}_{ts_report}.csv"
        pd.DataFrame([summary_row]).to_csv(csv_path, index=False)
    except Exception:
        pass

    return best_weighting_params
