"""
Advanced Gating Logic for Meta-Labeling HPO

Production-ready implementations for:
A. Regime-conditional barrier geometry (dynamic TP/SL/horizon)
B. Learned meta-gate (LGBM-based gating function)
C. Per-expert confidence calibration (isotonic/Platt, per-regime)
D. Abstention-aware voting (coverage-gated consensus)
E. Expert specialization scores and diversity regularization

All functions are designed to be non-leaky (train-fold safe) and backtest-ready.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any, Union
import warnings

try:
    from src.utils.tprint import tprint_info, tprint_warning, tprint_success, tprint_error
except ImportError:
    def tprint_info(*args, **kwargs): print(*args)
    def tprint_warning(*args, **kwargs): print(*args)
    def tprint_success(*args, **kwargs): print(*args)
    def tprint_error(*args, **kwargs): print(*args)

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import TimeSeriesSplit
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    IsotonicRegression = None
    LogisticRegression = None
    TimeSeriesSplit = None

try:
    import lightgbm as lgb
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False
    lgb = None


# =============================================================================
# A. REGIME-CONDITIONAL BARRIER GEOMETRY
# =============================================================================

class RegimeBarrierConfig:
    """
    Defines barrier geometry (TP/SL/horizon) per regime.
    
    Regimes: trend, chop, vol_spike, transition, default
    """
    
    # Default multipliers relative to base config
    REGIME_DEFAULTS = {
        "trend": {
            "tp_mult": 1.5,      # wider TP in trend
            "sl_mult": 1.2,      # slightly wider SL
            "horizon_mult": 1.3, # longer horizon
            "trail_mult": 1.0,   # enable trailing
        },
        "chop": {
            "tp_mult": 0.7,      # tighter TP in chop (mean-reversion)
            "sl_mult": 0.8,      # tighter SL
            "horizon_mult": 0.7, # shorter horizon
            "trail_mult": 0.0,   # no trailing in chop
        },
        "vol_spike": {
            "tp_mult": 1.3,      # wider TP during vol
            "sl_mult": 1.5,      # much wider SL (avoid whipsaws)
            "horizon_mult": 0.8, # shorter horizon (vol mean-reverts)
            "trail_mult": 0.5,   # partial trailing
        },
        "transition": {
            "tp_mult": 1.0,      # neutral
            "sl_mult": 1.0,
            "horizon_mult": 1.0,
            "trail_mult": 0.3,
        },
        "default": {
            "tp_mult": 1.0,
            "sl_mult": 1.0,
            "horizon_mult": 1.0,
            "trail_mult": 0.0,
        },
    }
    
    def __init__(self, custom_config: Optional[Dict[str, Dict[str, float]]] = None):
        """
        Args:
            custom_config: Override defaults with custom regime multipliers.
        """
        self.config = {k: dict(v) for k, v in self.REGIME_DEFAULTS.items()}
        if custom_config:
            for regime, params in custom_config.items():
                if regime in self.config:
                    self.config[regime].update(params)
                else:
                    self.config[regime] = params
    
    def get_multipliers(self, regime: str) -> Dict[str, float]:
        """Get multipliers for a given regime."""
        return self.config.get(regime, self.config["default"])


def compute_regime_labels_for_events(
    market_data: pd.DataFrame,
    event_idx: pd.DatetimeIndex,
    adx_col: str = "adx",
    vol_ratio_col: str = "vol_ratio",
    choppiness_col: str = "choppiness",
    trend_threshold: float = 25.0,
    chop_threshold: float = 20.0,
    vol_spike_threshold: float = 1.5,
    transition_window: int = 5,
) -> pd.Series:
    """
    Assign regime labels to events based on market state at event time.
    
    Regimes: 'trend', 'chop', 'vol_spike', 'transition', 'default'
    
    Args:
        market_data: DataFrame with regime indicators.
        event_idx: DatetimeIndex of events.
        adx_col: Column name for ADX.
        vol_ratio_col: Column name for volatility ratio.
        choppiness_col: Column name for choppiness index.
        trend_threshold: ADX above this = trend regime.
        chop_threshold: ADX below this = chop regime.
        vol_spike_threshold: vol_ratio above this = vol_spike.
        transition_window: Bars to look back for regime change detection.
        
    Returns:
        Series with regime labels indexed by event_idx.
    """
    n_events = len(event_idx)
    regimes = pd.Series("default", index=event_idx, dtype=object)
    
    if market_data is None or market_data.empty:
        return regimes

    def _resolve_col(primary: str, candidates: List[str]) -> Optional[str]:
        try:
            if primary and primary in market_data.columns:
                return primary
        except Exception:
            pass
        for cand in candidates:
            try:
                if cand in market_data.columns:
                    return cand
            except Exception:
                continue
        return None

    adx_col_resolved = _resolve_col(adx_col, ["reg_res_adx_14", "adx_14", "ADX_14", "regime_adx"]) 
    vol_ratio_col_resolved = _resolve_col(vol_ratio_col, ["reg_ohlcv__vol_ratio_5", "vol_ratio_5", "vol_ratio", "volume_ratio"]) 
    choppiness_col_resolved = _resolve_col(
        choppiness_col,
        [
            "reg_ohlcv__choppiness_w14",
            "reg_ohlcv__choppiness_w20",
            "reg_ohlcv__choppiness_w10",
            "choppiness",
        ],
    )
    
    # Extract indicators at event times
    adx = (
        market_data[adx_col_resolved].reindex(event_idx).fillna(20.0)
        if adx_col_resolved is not None
        else pd.Series(20.0, index=event_idx)
    )
    vol_ratio = (
        market_data[vol_ratio_col_resolved].reindex(event_idx).fillna(1.0)
        if vol_ratio_col_resolved is not None
        else pd.Series(1.0, index=event_idx)
    )
    choppiness = (
        market_data[choppiness_col_resolved].reindex(event_idx).fillna(50.0)
        if choppiness_col_resolved is not None
        else pd.Series(50.0, index=event_idx)
    )
    
    # Detect regime transitions (ADX crossing thresholds recently)
    if adx_col_resolved is not None:
        adx_full = market_data[adx_col_resolved].astype(float)
        adx_diff = adx_full.diff(transition_window).reindex(event_idx).fillna(0.0)
        transition_mask = np.abs(adx_diff) > 5.0  # Significant ADX change
    else:
        transition_mask = pd.Series(False, index=event_idx)
    
    # Assign regimes (priority order: vol_spike > transition > trend > chop > default)
    vol_spike_mask = vol_ratio > vol_spike_threshold
    trend_mask = (adx > trend_threshold) & ~vol_spike_mask
    chop_mask = (adx < chop_threshold) | (choppiness > 60.0)
    chop_mask = chop_mask & ~vol_spike_mask & ~trend_mask
    
    regimes[vol_spike_mask] = "vol_spike"
    regimes[transition_mask & ~vol_spike_mask] = "transition"
    regimes[trend_mask] = "trend"
    regimes[chop_mask] = "chop"
    
    return regimes


def apply_regime_barrier_geometry(
    base_tp: float,
    base_sl: float,
    base_horizon: int,
    base_trail: float,
    regime_labels: pd.Series,
    barrier_config: Optional[RegimeBarrierConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply regime-conditional multipliers to barrier geometry.
    
    Args:
        base_tp: Base take-profit distance (e.g., ATR multiplier).
        base_sl: Base stop-loss distance.
        base_horizon: Base horizon in bars.
        base_trail: Base trailing stop distance (0 = disabled).
        regime_labels: Series of regime labels per event.
        barrier_config: RegimeBarrierConfig instance.
        
    Returns:
        (tp_arr, sl_arr, horizon_arr, trail_arr): Arrays of adjusted values per event.
    """
    if barrier_config is None:
        barrier_config = RegimeBarrierConfig()
    
    n_events = len(regime_labels)
    tp_arr = np.full(n_events, base_tp, dtype=float)
    sl_arr = np.full(n_events, base_sl, dtype=float)
    horizon_arr = np.full(n_events, base_horizon, dtype=float)
    trail_arr = np.full(n_events, base_trail, dtype=float)
    
    for regime in regime_labels.unique():
        mask = (regime_labels == regime).values
        if not np.any(mask):
            continue
        
        mults = barrier_config.get_multipliers(regime)
        tp_arr[mask] = base_tp * mults.get("tp_mult", 1.0)
        sl_arr[mask] = base_sl * mults.get("sl_mult", 1.0)
        horizon_arr[mask] = base_horizon * mults.get("horizon_mult", 1.0)
        trail_arr[mask] = base_trail * mults.get("trail_mult", 1.0)
    
    # Ensure valid ranges
    tp_arr = np.clip(tp_arr, 0.001, 10.0)
    sl_arr = np.clip(sl_arr, 0.001, 10.0)
    horizon_arr = np.clip(horizon_arr, 1, 100).astype(int)
    trail_arr = np.clip(trail_arr, 0.0, 5.0)
    
    return tp_arr, sl_arr, horizon_arr, trail_arr


# =============================================================================
# B. LEARNED META-GATE (LGBM)
# =============================================================================

class LearnedMetaGate:
    """
    LGBM-based meta-gate that learns to predict optimal expert weights
    or take/no-take decisions based on market state features.
    
    Train on train folds only, apply to test folds (no leakage).
    """
    
    def __init__(
        self,
        n_experts: int = 9,
        mode: str = "weights",  # "weights" or "take_prob"
        lgbm_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            n_experts: Number of experts in the committee.
            mode: "weights" to predict expert weights, "take_prob" to predict take probability.
            lgbm_params: LightGBM parameters.
        """
        self.n_experts = n_experts
        self.mode = mode
        self.lgbm_params = lgbm_params or {
            "objective": "regression" if mode == "weights" else "binary",
            "metric": "rmse" if mode == "weights" else "auc",
            "num_leaves": 15,
            "max_depth": 4,
            "learning_rate": 0.05,
            "n_estimators": 100,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbose": -1,
            "random_state": 42,
        }
        self.models: List[Any] = []  # One model per expert (for weights mode) or single model (for take_prob)
        self.is_fitted = False
        self.feature_names: List[str] = []
    
    def _build_features(
        self,
        market_data: pd.DataFrame,
        event_idx: pd.DatetimeIndex,
        consensus_scores: Optional[np.ndarray] = None,
        expert_confidences: Optional[np.ndarray] = None,
        regime_labels: Optional[pd.Series] = None,
        nn_embeddings: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Build feature matrix for meta-gate model.
        
        Features (all ex-ante, no leakage):
        - Regime indicators (ADX, vol_ratio, choppiness, etc.)
        - Consensus score and confidence stats
        - Expert agreement/disagreement metrics
        - Recent price action features
        - NN embeddings (if available from Layer3 cache)
        """
        features = pd.DataFrame(index=event_idx)
        
        # Regime features from market_data
        regime_sources: Dict[str, List[str]] = {
            "adx": ["adx", "reg_res_adx_14", "adx_14", "ADX_14"],
            "vol_ratio": ["vol_ratio", "reg_ohlcv__vol_ratio_5", "vol_ratio_5"],
            "choppiness": [
                "choppiness",
                "reg_ohlcv__choppiness_w14",
                "reg_ohlcv__choppiness_w20",
                "reg_ohlcv__choppiness_w10",
            ],
            "bb_squeeze": ["bb_squeeze", "bb_squeeze_flag"],
            "atr_pct": ["atr_pct", "atr_frac"],
            "rsi": ["rsi", "rsi_14"],
            "macd_hist": ["macd_hist", "macd", "macd_long"],
        }

        for canonical, candidates in regime_sources.items():
            chosen = None
            for cand in candidates:
                if cand in market_data.columns:
                    chosen = cand
                    break
            if chosen is not None:
                features[f"regime_{canonical}"] = (
                    market_data[chosen].reindex(event_idx).replace([np.inf, -np.inf], np.nan).fillna(0.0).values
                )
        
        # Consensus features
        if consensus_scores is not None:
            cs = np.asarray(consensus_scores, dtype=float)
            features["consensus_score"] = cs
            features["consensus_abs"] = np.abs(cs)
            features["consensus_sign"] = np.sign(cs)
        
        # Expert confidence features
        if expert_confidences is not None:
            conf = np.asarray(expert_confidences, dtype=float)
            if conf.ndim == 2:
                features["conf_mean"] = np.nanmean(conf, axis=1)
                features["conf_std"] = np.nanstd(conf, axis=1)
                features["conf_max"] = np.nanmax(conf, axis=1)
                features["conf_min"] = np.nanmin(conf, axis=1)
                # Per-expert confidence (top 3)
                for i in range(min(3, conf.shape[1])):
                    features[f"conf_expert_{i}"] = conf[:, i]
        
        # Regime labels as categorical
        if regime_labels is not None:
            regime_map = {"trend": 0, "chop": 1, "vol_spike": 2, "transition": 3, "default": 4}
            features["regime_cat"] = regime_labels.map(regime_map).fillna(4).astype(int).values
        
        # Recent returns features (ex-ante)
        if "close" in market_data.columns:
            close = market_data["close"].astype(float)
            ret_1 = close.pct_change(1).reindex(event_idx).fillna(0.0).values
            ret_5 = close.pct_change(5).reindex(event_idx).fillna(0.0).values
            ret_20 = close.pct_change(20).reindex(event_idx).fillna(0.0).values
            features["ret_1"] = ret_1
            features["ret_5"] = ret_5
            features["ret_20"] = ret_20
            features["ret_vol_5"] = close.pct_change().rolling(5).std().reindex(event_idx).fillna(0.0).values
        
        # NN embeddings from Layer3 cache (if available)
        # These are pre-computed by short_nn_sequence_template.py and cached
        # Adding them here is computationally cheap (just a lookup)
        if nn_embeddings is not None and not nn_embeddings.empty:
            nn_cols = [c for c in nn_embeddings.columns if c.startswith("nn_embed_")]
            if nn_cols:
                # Limit to first 8 embedding dimensions to avoid feature explosion
                nn_cols = nn_cols[:8]
                nn_aligned = nn_embeddings[nn_cols].reindex(event_idx).fillna(0.0)
                for col in nn_cols:
                    features[col] = nn_aligned[col].values
                tprint_info(f"   [meta_gate] Added {len(nn_cols)} NN embedding features")
        
        # Fill NaNs
        features = features.fillna(0.0)
        self.feature_names = list(features.columns)
        
        return features
    
    def fit(
        self,
        market_data: pd.DataFrame,
        event_idx: pd.DatetimeIndex,
        expert_returns: np.ndarray,
        expert_labels: np.ndarray,
        expert_confidences: Optional[np.ndarray] = None,
        consensus_scores: Optional[np.ndarray] = None,
        regime_labels: Optional[pd.Series] = None,
        sample_weights: Optional[np.ndarray] = None,
        nn_embeddings: Optional[pd.DataFrame] = None,
    ) -> "LearnedMetaGate":
        """
        Fit the meta-gate model on training data.
        
        Args:
            market_data: Full market data DataFrame.
            event_idx: Event indices for training.
            expert_returns: (n_events, n_experts) realized returns per expert.
            expert_labels: (n_events, n_experts) labels (+1/-1/0) per expert.
            expert_confidences: (n_events, n_experts) confidence scores.
            consensus_scores: (n_events,) consensus scores.
            regime_labels: Series of regime labels.
            sample_weights: Optional sample weights.
            nn_embeddings: Optional NN embeddings from Layer3 cache.
            
        Returns:
            self
        """
        if not _LGBM_AVAILABLE:
            tprint_warning("[meta_gate] LightGBM not available; meta-gate disabled")
            return self
        
        X = self._build_features(
            market_data, event_idx, consensus_scores, expert_confidences, regime_labels, nn_embeddings
        )
        
        n_events = len(event_idx)
        expert_returns = np.asarray(expert_returns, dtype=float)
        expert_labels = np.asarray(expert_labels, dtype=float)
        
        if self.mode == "weights":
            # Train one model per expert to predict optimal weight
            # Target: expert's edge (return when fired, 0 otherwise)
            self.models = []
            for j in range(self.n_experts):
                if j >= expert_returns.shape[1]:
                    self.models.append(None)
                    continue
                
                # Target: positive return indicator (did this expert make money?)
                fired = expert_labels[:, j] != 0
                y = np.zeros(n_events, dtype=float)
                y[fired] = (expert_returns[:, j][fired] > 0).astype(float)
                
                # Only train on events where expert fired
                train_mask = fired & np.isfinite(y)
                if train_mask.sum() < 50:
                    self.models.append(None)
                    continue
                
                model = lgb.LGBMClassifier(
                    **{**self.lgbm_params, "objective": "binary", "metric": "auc"}
                )

                sw = None
                try:
                    sw_base = sample_weights[train_mask] if sample_weights is not None else None
                    ret_abs = np.abs(np.asarray(expert_returns[:, j], dtype=float)[train_mask])
                    ret_abs = np.where(np.isfinite(ret_abs), ret_abs, 0.0)
                    scale = float(np.median(ret_abs[ret_abs > 0])) if np.any(ret_abs > 0) else 0.0
                    if (not np.isfinite(scale)) or scale <= 0.0:
                        scale = 1.0
                    ret_w = np.clip(ret_abs / (scale + 1e-12), 0.0, 5.0)
                    if sw_base is None:
                        sw = ret_w
                    else:
                        sw = np.asarray(sw_base, dtype=float) * (0.5 + 0.5 * ret_w)
                except Exception:
                    sw = sample_weights[train_mask] if sample_weights is not None else None

                sw_full = None
                try:
                    if sw is not None:
                        sw_full = np.zeros(n_events, dtype=float)
                        sw_full[train_mask] = np.asarray(sw, dtype=float)
                except Exception:
                    sw_full = None

                try:
                    idx_train = np.flatnonzero(train_mask)
                    if int(idx_train.size) >= 200:
                        split = int(max(50, 0.8 * float(idx_train.size)))
                        tr_idx = idx_train[:split]
                        va_idx = idx_train[split:]
                        callbacks = None
                        try:
                            callbacks = [lgb.early_stopping(20, verbose=False)]
                        except Exception:
                            callbacks = None
                        model.fit(
                            X.iloc[tr_idx],
                            y[tr_idx],
                            sample_weight=(sw_full[tr_idx] if sw_full is not None else None),
                            eval_set=[(X.iloc[va_idx], y[va_idx])],
                            eval_sample_weight=[(sw_full[va_idx] if sw_full is not None else None)],
                            callbacks=callbacks,
                        )
                    else:
                        model.fit(X.iloc[train_mask], y[train_mask], sample_weight=sw)
                except Exception:
                    model.fit(X.iloc[train_mask], y[train_mask], sample_weight=sw)
                self.models.append(model)
            
            tprint_info(f"[meta_gate] Fitted {sum(1 for m in self.models if m is not None)}/{self.n_experts} expert weight models")
        
        else:  # take_prob mode
            # Single model to predict P(profitable trade | take)
            # Target: was the consensus-weighted return positive?
            if consensus_scores is not None:
                weighted_ret = np.sum(expert_returns * np.abs(expert_labels), axis=1) / (np.sum(np.abs(expert_labels), axis=1) + 1e-8)
            else:
                weighted_ret = np.nanmean(expert_returns, axis=1)
            
            y = (weighted_ret > 0).astype(int)
            valid = np.isfinite(weighted_ret)
            
            if valid.sum() < 100:
                tprint_warning("[meta_gate] Insufficient data for take_prob model")
                return self
            
            model = lgb.LGBMClassifier(**{**self.lgbm_params, "objective": "binary", "metric": "auc"})
            sw = sample_weights[valid] if sample_weights is not None else None
            sw_full = None
            try:
                if sw is not None:
                    sw_full = np.zeros(n_events, dtype=float)
                    sw_full[valid] = np.asarray(sw, dtype=float)
            except Exception:
                sw_full = None
            try:
                idx_valid = np.flatnonzero(valid)
                if int(idx_valid.size) >= 400:
                    split = int(max(100, 0.8 * float(idx_valid.size)))
                    tr_idx = idx_valid[:split]
                    va_idx = idx_valid[split:]
                    callbacks = None
                    try:
                        callbacks = [lgb.early_stopping(20, verbose=False)]
                    except Exception:
                        callbacks = None
                    sw_tr = sw_full[tr_idx] if sw_full is not None else None
                    sw_va = sw_full[va_idx] if sw_full is not None else None
                    model.fit(
                        X.iloc[tr_idx],
                        y[tr_idx],
                        sample_weight=sw_tr,
                        eval_set=[(X.iloc[va_idx], y[va_idx])],
                        eval_sample_weight=[sw_va],
                        callbacks=callbacks,
                    )
                else:
                    model.fit(X.iloc[valid], y[valid], sample_weight=sw)
            except Exception:
                model.fit(X.iloc[valid], y[valid], sample_weight=sw)
            self.models = [model]
            tprint_info("[meta_gate] Fitted take_prob model")
        
        self.is_fitted = True
        return self
    
    def predict_weights(
        self,
        market_data: pd.DataFrame,
        event_idx: pd.DatetimeIndex,
        consensus_scores: Optional[np.ndarray] = None,
        expert_confidences: Optional[np.ndarray] = None,
        regime_labels: Optional[pd.Series] = None,
        base_weights: Optional[np.ndarray] = None,
        nn_embeddings: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """
        Predict expert weights for events.
        
        Args:
            market_data: Market data DataFrame.
            event_idx: Event indices.
            consensus_scores: Consensus scores.
            expert_confidences: Expert confidence matrix.
            regime_labels: Regime labels.
            base_weights: Base weights to blend with predictions.
            nn_embeddings: Optional NN embeddings from Layer3 cache.
            
        Returns:
            (n_events, n_experts) weight matrix.
        """
        n_events = len(event_idx)
        
        if not self.is_fitted or not self.models:
            # Return uniform weights
            if base_weights is not None:
                return np.broadcast_to(base_weights, (n_events, self.n_experts)).copy()
            return np.ones((n_events, self.n_experts), dtype=float) / self.n_experts
        
        X = self._build_features(
            market_data, event_idx, consensus_scores, expert_confidences, regime_labels, nn_embeddings
        )
        
        weights = np.ones((n_events, self.n_experts), dtype=float)
        
        if self.mode == "weights":
            for j, model in enumerate(self.models):
                if model is None:
                    continue
                try:
                    # Predict probability that expert will be profitable
                    prob = model.predict_proba(X)[:, 1]
                    weights[:, j] = prob
                except Exception:
                    pass
        
        else:  # take_prob mode - return uniform weights scaled by take probability
            if self.models and self.models[0] is not None:
                try:
                    take_prob = self.models[0].predict_proba(X)[:, 1]
                    weights = weights * take_prob.reshape(-1, 1)
                except Exception:
                    pass
        
        # Blend with base weights if provided
        if base_weights is not None:
            base = np.broadcast_to(base_weights, (n_events, self.n_experts))
            weights = 0.5 * weights + 0.5 * base
        
        # Normalize per event
        row_sums = weights.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 1e-8, row_sums, 1.0)
        weights = weights / row_sums
        
        return weights
    
    def predict_take_prob(
        self,
        market_data: pd.DataFrame,
        event_idx: pd.DatetimeIndex,
        consensus_scores: Optional[np.ndarray] = None,
        expert_confidences: Optional[np.ndarray] = None,
        regime_labels: Optional[pd.Series] = None,
        nn_embeddings: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """
        Predict take probability for events.
        
        Returns:
            (n_events,) array of take probabilities.
        """
        n_events = len(event_idx)
        
        if not self.is_fitted or not self.models or self.models[0] is None:
            return np.ones(n_events, dtype=float) * 0.5
        
        X = self._build_features(
            market_data, event_idx, consensus_scores, expert_confidences, regime_labels, nn_embeddings
        )
        
        try:
            return self.models[0].predict_proba(X)[:, 1]
        except Exception:
            return np.ones(n_events, dtype=float) * 0.5
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get aggregated feature importance across models."""
        if not self.is_fitted or not self.models:
            return {}
        
        importance = {}
        n_models = 0
        
        for model in self.models:
            if model is None:
                continue
            try:
                imp = model.feature_importances_
                for i, name in enumerate(self.feature_names):
                    importance[name] = importance.get(name, 0.0) + imp[i]
                n_models += 1
            except Exception:
                pass
        
        if n_models > 0:
            importance = {k: v / n_models for k, v in importance.items()}
        
        return dict(sorted(importance.items(), key=lambda x: -x[1]))


# =============================================================================
# C. PER-EXPERT CONFIDENCE CALIBRATION
# =============================================================================

class ExpertConfidenceCalibrator:
    """
    Calibrates expert confidence scores using isotonic regression or Platt scaling.
    Can be regime-conditional for better calibration in different market states.
    """
    
    def __init__(
        self,
        n_experts: int = 9,
        method: str = "isotonic",  # "isotonic" or "platt"
        per_regime: bool = True,
    ):
        """
        Args:
            n_experts: Number of experts.
            method: Calibration method.
            per_regime: Whether to fit separate calibrators per regime.
        """
        self.n_experts = n_experts
        self.method = method
        self.per_regime = per_regime
        self.calibrators: Dict[str, List[Any]] = {}  # regime -> [calibrator per expert]
        self.is_fitted = False
    
    def _create_calibrator(self):
        """Create a new calibrator instance."""
        if self.method == "isotonic":
            return IsotonicRegression(out_of_bounds="clip") if _SKLEARN_AVAILABLE else None
        else:  # platt
            return LogisticRegression(solver="lbfgs", max_iter=1000) if _SKLEARN_AVAILABLE else None
    
    def fit(
        self,
        expert_confidences: np.ndarray,
        expert_returns: np.ndarray,
        expert_labels: np.ndarray,
        regime_labels: Optional[pd.Series] = None,
    ) -> "ExpertConfidenceCalibrator":
        """
        Fit calibrators on training data.
        
        Args:
            expert_confidences: (n_events, n_experts) confidence scores.
            expert_returns: (n_events, n_experts) realized returns.
            expert_labels: (n_events, n_experts) labels.
            regime_labels: Optional regime labels for per-regime calibration.
            
        Returns:
            self
        """
        if not _SKLEARN_AVAILABLE:
            tprint_warning("[calibrator] sklearn not available; calibration disabled")
            return self
        
        conf = np.asarray(expert_confidences, dtype=float)
        ret = np.asarray(expert_returns, dtype=float)
        lbl = np.asarray(expert_labels, dtype=float)
        
        n_events = conf.shape[0]
        
        # Target: did the expert make money when it fired?
        # y = 1 if return > 0, 0 otherwise (only for fired events)
        
        regimes = ["all"]
        if self.per_regime and regime_labels is not None:
            regimes = list(regime_labels.unique()) + ["all"]
        
        for regime in regimes:
            if regime == "all":
                mask = np.ones(n_events, dtype=bool)
            else:
                mask = (regime_labels == regime).values if regime_labels is not None else np.ones(n_events, dtype=bool)
            
            calibrators = []
            for j in range(self.n_experts):
                if j >= conf.shape[1]:
                    calibrators.append(None)
                    continue
                
                # Only calibrate on events where expert fired
                fired = (lbl[:, j] != 0) & mask
                if fired.sum() < 30:
                    calibrators.append(None)
                    continue
                
                x = conf[fired, j]
                y = (ret[fired, j] > 0).astype(float)
                
                # Need variance in both x and y
                if np.std(x) < 1e-6 or np.std(y) < 1e-6:
                    calibrators.append(None)
                    continue
                
                try:
                    cal = self._create_calibrator()
                    if cal is None:
                        calibrators.append(None)
                        continue
                    
                    if self.method == "isotonic":
                        cal.fit(x, y)
                    else:  # platt
                        cal.fit(x.reshape(-1, 1), y)
                    
                    calibrators.append(cal)
                except Exception:
                    calibrators.append(None)
            
            self.calibrators[regime] = calibrators
        
        n_fitted = sum(1 for cals in self.calibrators.values() for c in cals if c is not None)
        tprint_info(f"[calibrator] Fitted {n_fitted} calibrators across {len(self.calibrators)} regimes")
        self.is_fitted = True
        return self
    
    def calibrate(
        self,
        expert_confidences: np.ndarray,
        regime_labels: Optional[pd.Series] = None,
    ) -> np.ndarray:
        """
        Apply calibration to confidence scores.
        
        Args:
            expert_confidences: (n_events, n_experts) raw confidence scores.
            regime_labels: Optional regime labels for per-regime calibration.
            
        Returns:
            (n_events, n_experts) calibrated confidence scores.
        """
        conf = np.asarray(expert_confidences, dtype=float)
        calibrated = conf.copy()
        
        if not self.is_fitted:
            return calibrated
        
        n_events = conf.shape[0]
        
        for i in range(n_events):
            # Determine regime for this event
            if self.per_regime and regime_labels is not None:
                regime = regime_labels.iloc[i] if hasattr(regime_labels, "iloc") else regime_labels[i]
            else:
                regime = "all"
            
            # Get calibrators for this regime (fallback to "all")
            cals = self.calibrators.get(regime, self.calibrators.get("all", []))
            
            for j in range(min(len(cals), conf.shape[1])):
                cal = cals[j] if j < len(cals) else None
                if cal is None:
                    continue
                
                try:
                    x = conf[i, j]
                    if self.method == "isotonic":
                        calibrated[i, j] = cal.predict([x])[0]
                    else:  # platt
                        calibrated[i, j] = cal.predict_proba([[x]])[0, 1]
                except Exception:
                    pass
        
        # Ensure valid range
        calibrated = np.clip(calibrated, 0.0, 1.0)
        return calibrated


# =============================================================================
# D. ABSTENTION-AWARE VOTING
# =============================================================================

def compute_abstention_aware_consensus(
    expert_labels: np.ndarray,
    expert_confidences: np.ndarray,
    expert_weights: np.ndarray,
    coverage_min: float = 0.3,
    consensus_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute consensus score with abstention awareness.
    
    Key insight: A single expert firing shouldn't yield high confidence.
    We gate on coverage (fraction of weighted experts that fired).
    
    Args:
        expert_labels: (n_events, n_experts) labels (+1/-1/0).
        expert_confidences: (n_events, n_experts) confidence scores.
        expert_weights: (n_events, n_experts) or (n_experts,) weights.
        coverage_min: Minimum coverage to consider consensus valid.
        consensus_threshold: Threshold for take decision.
        
    Returns:
        (consensus_scores, coverage, take_mask)
    """
    lbl = np.asarray(expert_labels, dtype=float)
    conf = np.asarray(expert_confidences, dtype=float)
    
    # Handle weights broadcasting
    if expert_weights.ndim == 1:
        w = np.broadcast_to(expert_weights, lbl.shape)
    else:
        w = np.asarray(expert_weights, dtype=float)
    
    n_events = lbl.shape[0]
    
    # Fired mask
    fired = lbl != 0
    
    # Weighted firing
    fired_weighted = fired.astype(float) * w
    total_weight = np.sum(w, axis=1) + 1e-8
    coverage = np.sum(fired_weighted, axis=1) / total_weight
    
    # Weighted consensus (only from fired experts)
    sign_weighted = np.where(fired, np.sign(lbl), 0.0) * conf * w
    denom = np.sum(fired_weighted * conf, axis=1) + 1e-8
    consensus_raw = np.sum(sign_weighted, axis=1) / denom
    
    # Apply coverage damping: low coverage → reduce consensus magnitude
    # sqrt(coverage) is a soft penalty
    coverage_factor = np.sqrt(np.clip(coverage, 0.0, 1.0))
    consensus_scores = consensus_raw * coverage_factor
    consensus_scores = np.clip(consensus_scores, -1.0, 1.0)
    
    # Take mask: require both coverage and consensus threshold
    take_mask = (coverage >= coverage_min) & (np.abs(consensus_scores) >= consensus_threshold)
    
    return consensus_scores, coverage, take_mask


def compute_coverage_gated_weights(
    expert_labels: np.ndarray,
    base_weights: np.ndarray,
    coverage_min: float = 0.2,
    coverage_boost_max: float = 1.5,
) -> np.ndarray:
    """
    Adjust weights based on expert coverage (how many fired).
    
    When few experts fire, boost the weights of those that did fire
    (but only up to a limit to avoid over-concentration).
    
    Args:
        expert_labels: (n_events, n_experts) labels.
        base_weights: (n_experts,) or (n_events, n_experts) base weights.
        coverage_min: Below this coverage, apply boost.
        coverage_boost_max: Maximum boost factor.
        
    Returns:
        (n_events, n_experts) adjusted weights.
    """
    lbl = np.asarray(expert_labels, dtype=float)
    n_events, n_experts = lbl.shape
    
    if base_weights.ndim == 1:
        w = np.broadcast_to(base_weights, (n_events, n_experts)).copy()
    else:
        w = np.asarray(base_weights, dtype=float).copy()
    
    fired = lbl != 0
    n_fired = np.sum(fired, axis=1)
    coverage = n_fired / n_experts
    
    # Boost factor: higher when coverage is low
    # boost = 1 + (coverage_boost_max - 1) * (1 - coverage / coverage_min) when coverage < coverage_min
    boost = np.ones(n_events, dtype=float)
    low_cov = coverage < coverage_min
    if np.any(low_cov):
        boost[low_cov] = 1.0 + (coverage_boost_max - 1.0) * (1.0 - coverage[low_cov] / coverage_min)
        boost = np.clip(boost, 1.0, coverage_boost_max)
    
    # Apply boost only to fired experts
    w = w * (1.0 + (boost.reshape(-1, 1) - 1.0) * fired.astype(float))
    
    # Renormalize
    row_sums = w.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 1e-8, row_sums, 1.0)
    w = w / row_sums
    
    return w


# =============================================================================
# E. EXPERT SPECIALIZATION SCORES AND DIVERSITY REGULARIZATION
# =============================================================================

def compute_expert_specialization_scores(
    expert_returns: np.ndarray,
    expert_labels: np.ndarray,
    regime_labels: pd.Series,
) -> Dict[str, np.ndarray]:
    """
    Compute per-expert, per-regime specialization scores.
    
    Specialization = edge (mean return when fired) in each regime.
    Higher specialization → expert is better suited for that regime.
    
    Args:
        expert_returns: (n_events, n_experts) returns.
        expert_labels: (n_events, n_experts) labels.
        regime_labels: Series of regime labels.
        
    Returns:
        Dict mapping regime -> (n_experts,) specialization scores.
    """
    ret = np.asarray(expert_returns, dtype=float)
    lbl = np.asarray(expert_labels, dtype=float)
    n_experts = ret.shape[1]
    
    specialization = {}
    
    for regime in regime_labels.unique():
        mask = (regime_labels == regime).values
        scores = np.zeros(n_experts, dtype=float)
        
        for j in range(n_experts):
            fired = (lbl[:, j] != 0) & mask
            if fired.sum() < 10:
                scores[j] = 0.0
                continue
            
            r = ret[fired, j]
            r = r[np.isfinite(r)]
            if len(r) < 10:
                scores[j] = 0.0
                continue
            
            # Edge = mean return, normalized by std for comparability
            mean_r = np.mean(r)
            std_r = np.std(r) + 1e-8
            scores[j] = mean_r / std_r  # Sharpe-like
        
        specialization[regime] = scores
    
    return specialization


def apply_specialization_weights(
    base_weights: np.ndarray,
    regime_labels: pd.Series,
    specialization_scores: Dict[str, np.ndarray],
    specialization_strength: float = 0.5,
) -> np.ndarray:
    """
    Adjust weights based on expert specialization in current regime.
    
    Args:
        base_weights: (n_experts,) or (n_events, n_experts) base weights.
        regime_labels: Series of regime labels per event.
        specialization_scores: Dict from compute_expert_specialization_scores.
        specialization_strength: How much to weight specialization (0-1).
        
    Returns:
        (n_events, n_experts) adjusted weights.
    """
    n_events = len(regime_labels)
    
    if base_weights.ndim == 1:
        n_experts = len(base_weights)
        w = np.broadcast_to(base_weights, (n_events, n_experts)).copy()
    else:
        w = np.asarray(base_weights, dtype=float).copy()
        n_experts = w.shape[1]
    
    for i, regime in enumerate(regime_labels):
        if regime not in specialization_scores:
            continue
        
        spec = specialization_scores[regime]
        if len(spec) != n_experts:
            continue
        
        # Convert specialization to multiplicative factor
        # Positive spec → boost, negative → penalize
        # Use sigmoid-like transform to bound the factor
        factor = 1.0 + specialization_strength * np.tanh(spec)
        factor = np.clip(factor, 0.1, 3.0)
        
        w[i, :] = w[i, :] * factor
    
    # Renormalize
    row_sums = w.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 1e-8, row_sums, 1.0)
    w = w / row_sums
    
    return w


def compute_diversity_penalty(
    expert_labels: np.ndarray,
    expert_weights: np.ndarray,
    lambda_diversity: float = 0.1,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute diversity penalty based on expert overlap/correlation.
    
    High overlap between selected experts → penalty.
    
    Args:
        expert_labels: (n_events, n_experts) labels.
        expert_weights: (n_experts,) weights.
        lambda_diversity: Penalty strength.
        
    Returns:
        (penalty, diagnostics_dict)
    """
    lbl = np.asarray(expert_labels, dtype=float)
    w = np.asarray(expert_weights, dtype=float).flatten()
    n_experts = lbl.shape[1]
    
    # Compute pairwise Jaccard overlap
    fired = lbl != 0
    overlaps = []
    
    for i in range(n_experts):
        for j in range(i + 1, n_experts):
            fi = fired[:, i]
            fj = fired[:, j]
            inter = np.sum(fi & fj)
            union = np.sum(fi | fj)
            if union > 0:
                jacc = inter / union
                # Weight by product of expert weights
                weighted_jacc = jacc * w[i] * w[j]
                overlaps.append(weighted_jacc)
    
    if not overlaps:
        return 0.0, {"mean_overlap": 0.0, "max_overlap": 0.0}
    
    mean_overlap = np.mean(overlaps)
    max_overlap = np.max(overlaps)
    
    # Penalty: higher overlap → higher penalty
    penalty = lambda_diversity * mean_overlap
    
    return penalty, {
        "mean_overlap": float(mean_overlap),
        "max_overlap": float(max_overlap),
        "n_pairs": len(overlaps),
    }


def compute_diversity_regularized_utility(
    base_utility: float,
    expert_labels: np.ndarray,
    expert_weights: np.ndarray,
    lambda_diversity: float = 0.1,
) -> Tuple[float, Dict[str, float]]:
    """
    Apply diversity regularization to utility score.
    
    Args:
        base_utility: Original utility score.
        expert_labels: (n_events, n_experts) labels.
        expert_weights: (n_experts,) weights.
        lambda_diversity: Penalty strength.
        
    Returns:
        (regularized_utility, diagnostics)
    """
    penalty, diag = compute_diversity_penalty(expert_labels, expert_weights, lambda_diversity)
    regularized = base_utility - penalty
    diag["base_utility"] = float(base_utility)
    diag["penalty"] = float(penalty)
    diag["regularized_utility"] = float(regularized)
    return regularized, diag


# =============================================================================
# INTEGRATED ADVANCED GATING PIPELINE
# =============================================================================

class AdvancedGatingPipeline:
    """
    Production-ready pipeline combining all advanced gating features:
    - Regime-conditional barrier geometry
    - Learned meta-gate
    - Confidence calibration
    - Abstention-aware voting
    - Specialization and diversity
    """
    
    def __init__(
        self,
        n_experts: int = 9,
        enable_regime_barriers: bool = True,
        enable_meta_gate: bool = True,
        enable_calibration: bool = True,
        enable_abstention_aware: bool = True,
        enable_specialization: bool = True,
        enable_diversity: bool = True,
        meta_gate_mode: str = "weights",
        calibration_method: str = "isotonic",
        coverage_min: float = 0.3,
        consensus_threshold: float = 0.5,
        specialization_strength: float = 0.5,
        diversity_lambda: float = 0.1,
        barrier_config: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.n_experts = n_experts
        self.enable_regime_barriers = enable_regime_barriers
        self.enable_meta_gate = enable_meta_gate
        self.enable_calibration = enable_calibration
        self.enable_abstention_aware = enable_abstention_aware
        self.enable_specialization = enable_specialization
        self.enable_diversity = enable_diversity
        
        self.meta_gate_mode = meta_gate_mode
        self.calibration_method = calibration_method
        self.coverage_min = coverage_min
        self.consensus_threshold = consensus_threshold
        self.specialization_strength = specialization_strength
        self.diversity_lambda = diversity_lambda
        
        # Components
        self.barrier_config = RegimeBarrierConfig(barrier_config)
        self.meta_gate = LearnedMetaGate(n_experts, meta_gate_mode) if enable_meta_gate else None
        self.calibrator = ExpertConfidenceCalibrator(n_experts, calibration_method) if enable_calibration else None
        self.specialization_scores: Dict[str, np.ndarray] = {}
        
        self.is_fitted = False
    
    def fit(
        self,
        market_data: pd.DataFrame,
        event_idx: pd.DatetimeIndex,
        expert_returns: np.ndarray,
        expert_labels: np.ndarray,
        expert_confidences: np.ndarray,
        consensus_scores: Optional[np.ndarray] = None,
        regime_labels: Optional[pd.Series] = None,
        sample_weights: Optional[np.ndarray] = None,
        nn_embeddings: Optional[pd.DataFrame] = None,
    ) -> "AdvancedGatingPipeline":
        """
        Fit all pipeline components on training data.
        
        Args:
            nn_embeddings: Optional NN embeddings from Layer3 cache for meta-gate features.
        """
        tprint_info("[adv_gating] Fitting advanced gating pipeline...")
        
        # Store nn_embeddings for apply phase
        self._nn_embeddings = nn_embeddings
        
        # Compute regime labels if not provided
        if regime_labels is None:
            regime_labels = compute_regime_labels_for_events(market_data, event_idx)
        
        # Fit meta-gate
        if self.meta_gate is not None:
            self.meta_gate.fit(
                market_data, event_idx, expert_returns, expert_labels,
                expert_confidences, consensus_scores, regime_labels, sample_weights, nn_embeddings
            )
        
        # Fit calibrator
        if self.calibrator is not None:
            self.calibrator.fit(expert_confidences, expert_returns, expert_labels, regime_labels)
        
        # Compute specialization scores
        if self.enable_specialization:
            self.specialization_scores = compute_expert_specialization_scores(
                expert_returns, expert_labels, regime_labels
            )
            tprint_info(f"[adv_gating] Computed specialization for {len(self.specialization_scores)} regimes")
        
        self.is_fitted = True
        tprint_success("[adv_gating] Pipeline fitted successfully")
        return self
    
    def apply(
        self,
        market_data: pd.DataFrame,
        event_idx: pd.DatetimeIndex,
        expert_labels: np.ndarray,
        expert_confidences: np.ndarray,
        base_weights: np.ndarray,
        base_tp: float,
        base_sl: float,
        base_horizon: int,
        base_trail: float = 0.0,
        regime_labels: Optional[pd.Series] = None,
        consensus_scores: Optional[np.ndarray] = None,
        nn_embeddings: Optional[pd.DataFrame] = None,
        coverage_min_override: Optional[float] = None,
        consensus_threshold_override: Optional[float] = None,
        specialization_strength_override: Optional[float] = None,
        diversity_lambda_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Apply the full gating pipeline.
        
        Args:
            nn_embeddings: Optional NN embeddings for meta-gate features.
        
        Returns:
            Dict with:
            - weights: (n_events, n_experts) adjusted weights
            - consensus_scores: (n_events,) consensus scores
            - coverage: (n_events,) coverage values
            - take_mask: (n_events,) boolean take decisions
            - tp_arr, sl_arr, horizon_arr, trail_arr: regime-adjusted barrier geometry
            - calibrated_conf: (n_events, n_experts) calibrated confidences
            - diagnostics: dict with debug info
        """
        n_events = len(event_idx)
        
        # Use stored nn_embeddings if not provided
        if nn_embeddings is None:
            nn_embeddings = getattr(self, "_nn_embeddings", None)

        # Allow per-call overrides (so HPO can tune these per-trial without mutating shared state).
        coverage_min_used = self.coverage_min
        consensus_threshold_used = self.consensus_threshold
        specialization_strength_used = self.specialization_strength
        diversity_lambda_used = self.diversity_lambda
        try:
            if coverage_min_override is not None and np.isfinite(float(coverage_min_override)):
                coverage_min_used = float(np.clip(float(coverage_min_override), 0.0, 1.0))
        except Exception:
            pass
        try:
            if consensus_threshold_override is not None and np.isfinite(float(consensus_threshold_override)):
                consensus_threshold_used = float(np.clip(float(consensus_threshold_override), 0.0, 1.0))
        except Exception:
            pass
        try:
            if specialization_strength_override is not None and np.isfinite(float(specialization_strength_override)):
                specialization_strength_used = float(np.clip(float(specialization_strength_override), 0.0, 1.0))
        except Exception:
            pass
        try:
            if diversity_lambda_override is not None and np.isfinite(float(diversity_lambda_override)):
                diversity_lambda_used = float(max(0.0, float(diversity_lambda_override)))
        except Exception:
            pass
        
        # Compute regime labels if not provided
        if regime_labels is None:
            regime_labels = compute_regime_labels_for_events(market_data, event_idx)
        
        # 1. Regime-conditional barrier geometry
        if self.enable_regime_barriers:
            tp_arr, sl_arr, horizon_arr, trail_arr = apply_regime_barrier_geometry(
                base_tp, base_sl, base_horizon, base_trail,
                regime_labels, self.barrier_config
            )
        else:
            tp_arr = np.full(n_events, base_tp)
            sl_arr = np.full(n_events, base_sl)
            horizon_arr = np.full(n_events, base_horizon, dtype=int)
            trail_arr = np.full(n_events, base_trail)
        
        # 2. Calibrate confidences
        if self.calibrator is not None and self.calibrator.is_fitted:
            calibrated_conf = self.calibrator.calibrate(expert_confidences, regime_labels)
        else:
            calibrated_conf = expert_confidences.copy()
        
        # 3. Get weights from meta-gate or use base
        if self.meta_gate is not None and self.meta_gate.is_fitted:
            weights = self.meta_gate.predict_weights(
                market_data, event_idx, consensus_scores, calibrated_conf, regime_labels, base_weights, nn_embeddings
            )
        else:
            if base_weights.ndim == 1:
                weights = np.broadcast_to(base_weights, (n_events, self.n_experts)).copy()
            else:
                weights = base_weights.copy()
        
        # 4. Apply specialization
        if self.enable_specialization and self.specialization_scores:
            weights = apply_specialization_weights(
                weights, regime_labels, self.specialization_scores, specialization_strength_used
            )
        
        # 5. Abstention-aware consensus
        if self.enable_abstention_aware:
            consensus, coverage, take_mask = compute_abstention_aware_consensus(
                expert_labels, calibrated_conf, weights,
                coverage_min_used, consensus_threshold_used
            )
        else:
            # Simple weighted consensus
            fired = expert_labels != 0
            sign_w = np.where(fired, np.sign(expert_labels), 0.0) * calibrated_conf * weights
            denom = np.sum(fired.astype(float) * calibrated_conf * weights, axis=1) + 1e-8
            consensus = np.sum(sign_w, axis=1) / denom
            coverage = np.sum(fired.astype(float) * weights, axis=1) / (np.sum(weights, axis=1) + 1e-8)
            take_mask = np.abs(consensus) >= consensus_threshold_used
        
        # 6. Diversity diagnostics
        diversity_diag = {}
        if self.enable_diversity:
            _, diversity_diag = compute_diversity_penalty(
                expert_labels, weights.mean(axis=0), diversity_lambda_used
            )
        
        # Build diagnostics
        diagnostics = {
            "regime_counts": {r: int((regime_labels == r).sum()) for r in regime_labels.unique()},
            "coverage_mean": float(np.mean(coverage)),
            "coverage_std": float(np.std(coverage)),
            "take_rate": float(np.mean(take_mask)),
            "consensus_mean": float(np.mean(consensus)),
            "consensus_std": float(np.std(consensus)),
            "diversity": diversity_diag,
            "coverage_min_used": float(coverage_min_used),
            "consensus_threshold_used": float(consensus_threshold_used),
            "specialization_strength_used": float(specialization_strength_used),
            "diversity_lambda_used": float(diversity_lambda_used),
            "meta_gate_fitted": self.meta_gate.is_fitted if self.meta_gate else False,
            "calibrator_fitted": self.calibrator.is_fitted if self.calibrator else False,
        }
        
        return {
            "weights": weights,
            "consensus_scores": consensus,
            "coverage": coverage,
            "take_mask": take_mask,
            "tp_arr": tp_arr,
            "sl_arr": sl_arr,
            "horizon_arr": horizon_arr,
            "trail_arr": trail_arr,
            "calibrated_conf": calibrated_conf,
            "regime_labels": regime_labels,
            "diagnostics": diagnostics,
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get serializable state for caching."""
        return {
            "n_experts": self.n_experts,
            "is_fitted": self.is_fitted,
            "specialization_scores": {k: v.tolist() for k, v in self.specialization_scores.items()},
            "config": {
                "enable_regime_barriers": self.enable_regime_barriers,
                "enable_meta_gate": self.enable_meta_gate,
                "enable_calibration": self.enable_calibration,
                "enable_abstention_aware": self.enable_abstention_aware,
                "enable_specialization": self.enable_specialization,
                "enable_diversity": self.enable_diversity,
                "coverage_min": self.coverage_min,
                "consensus_threshold": self.consensus_threshold,
                "specialization_strength": self.specialization_strength,
                "diversity_lambda": self.diversity_lambda,
            },
        }
