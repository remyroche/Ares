"""
position_sizer_v2.py

Refactored position-sizing and policy optimization stack.
Layer A: Predictive Models (Edge, Downside, Uncertainty)
Layer B: Policy Optimization (Entry/Exit Geometry & Selection Boundary)
Layer C: Execution Optimization (Sizing Mapping & Limit Order Offset)

Use a single regularized Ridge-family setup by default unless another loss/model
is materially better suited (example: HuberRegressor for robust regression).
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.preprocessing import StandardScaler

from extreme_price_movements.config import (
    POSITION_SIZER_V2_FEATURE_CONFIG,
    POSITION_SIZER_V2_FEATURE_SELECTION_CONFIG,
)
from extreme_price_movements.elasticnet_feature_selection_v2 import (
    select_features_via_elasticnet,
    select_features_via_staged_en_rfe,
)
from extreme_price_movements.feature_views import get_feature_view
from extreme_price_movements.features import (
    assemble_feature_matrix,
    build_position_sizer_feature_frame,
)
from extreme_price_movements.label_policy_optimizer import (
    LabelPolicy,
    _simulate_policy_batch,
)
from extreme_price_movements.metrics import _stable_equity_and_drawdown
from extreme_price_movements.periods_symbols_management import (
    EventSchema,
    SlicePlanner,
    SlicePlannerConfig,
)
from extreme_price_movements.position_sizer_v2_metrics import (
    compute_bucket_monotonicity,
    compute_false_safe_rate,
    compute_top_slice_metrics,
    compute_uncertainty_calibration,
)
from extreme_price_movements.utils import tprint

# ============================================================
# Shared Configuration & Utilities
# ============================================================
logger = logging.getLogger(__name__)


class PredictionScaler:
    """Scales predictions handling NaNs explicitly with train-fold medians."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.medians_ = None

    def _clean(self, X: np.ndarray, fit: bool) -> np.ndarray:
        X_clean = X.copy()
        X_clean[np.isinf(X_clean)] = np.nan
        if fit:
            self.medians_ = np.nanmedian(X_clean, axis=0)
            self.medians_[np.isnan(self.medians_)] = 0.0

        inds = np.where(np.isnan(X_clean))
        X_clean[inds] = np.take(self.medians_, inds[1])
        return X_clean

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.fit_transform(self._clean(X, fit=True))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(self._clean(X, fit=False))


def make_temporal_splits(
    timestamps: Optional[np.ndarray],
    n_samples: int,
    n_splits: int = 3,
    purge_units: int = 43200,
    embargo_units: int = 43200,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], int]:
    """Build temporal splits via periods_symbols_management planner."""
    try:
        ts = pd.to_datetime(pd.Series(timestamps), unit="s", utc=True, errors="coerce")
        if ts.isna().all():
            ts = pd.to_datetime(pd.Series(timestamps), utc=True, errors="coerce")
        ts = ts.ffill().bfill()
        events = pd.DataFrame(
            {
                "event_id": np.arange(n_samples, dtype=np.int64),
                "symbol": np.repeat("ALL", n_samples),
                "t0": ts.to_numpy(),
                "t1": (ts + pd.Timedelta(seconds=1)).to_numpy(),
            }
        )
        cfg = SlicePlannerConfig.fast_defaults(schema=EventSchema())
        cfg = cfg.__class__(
            **{
                **cfg.__dict__,
                "preset": cfg.preset.__class__(
                    preset_name=cfg.preset.preset_name,
                    outer=cfg.preset.outer,
                    inner=cfg.preset.inner.__class__(n_splits=max(1, int(n_splits))),
                    sampling=cfg.preset.sampling,
                    symbol_policy=cfg.preset.symbol_policy,
                    purge_policy=cfg.preset.purge_policy,
                ),
                "silent": True,
                "min_rows_per_fold": 1,
                "min_symbols_per_fold": 1,
            }
        )
        bundle = SlicePlanner(cfg).build(events)
        plans = bundle["consumer_plans"]["ridge_sizer_fit"]
        splits: List[Tuple[np.ndarray, np.ndarray]] = []
        for plan in plans:
            if plan.tag != "predict_outer_test":
                continue
            tr = np.asarray(plan.fit_idx, dtype=np.int64)
            te = np.asarray(plan.predict_idx, dtype=np.int64)
            if tr.size > 0 and te.size > 0:
                splits.append((tr, te))
        if splits:
            return splits, len(splits)
    except Exception as e:
        logger.warning(f"SlicePlanner exception: {e}")

    raise ValueError(
        f"SlicePlanner failed to generate {n_splits} temporal splits for ridge sizer. "
        "Ensure timestamps are valid and sufficient data exists."
    )


def build_log_clipped_target(returns: np.ndarray, clip_L: float = 0.02) -> np.ndarray:
    """T1 = log_clipped_winsorized_net with soft-plus transform.

    Uses symmetric soft-plus transformation with scaling to preserve relative ordering
    and magnitude information for extreme returns.

    For values within [-clip_L, clip_L], applies log1p transform directly.
    For extreme values (outside [-clip_L, clip_L]), applies a soft-tanh-like scaling
    that preserves relative ordering and    and compresses the tail without destroying information.
    """
    returns = np.asarray(returns, dtype=float)
    if len(returns) == 0:
        return np.array([], dtype=float)
    if np.all(returns == 0):
        return np.zeros_like(returns)

    p01, p99 = np.percentile(returns, [1, 99])
    scale = max(np.std(returns), 1e-9)

    inner = np.abs(returns) <= clip_L
    outer_lo = returns < -clip_L
    outer_hi = returns > clip_L

    result = np.empty_like(returns)
    result[inner] = np.sign(returns[inner]) * np.log1p(np.abs(returns[inner]) / scale)

    result[outer_lo] = (
        -np.log1p(np.abs(-clip_L) / scale)
        - np.log1p((np.abs(returns[outer_lo]) - clip_L) / scale)
    )
    result[outer_hi] = (
        np.log1p(np.abs(clip_L) / scale)
        + np.log1p((np.abs(returns[outer_hi]) - clip_L) / scale)
    )

    return result.astype(np.float32)


def _soft_winsorize_downside(
    y: np.ndarray, lower_pct: float = 0.0, upper_pct: float = 98.0, softness: float = 0.3
) -> np.ndarray:
    """Soft winsorization that preserves relative ordering while compressing tails.
    
    Args:
        y: Input array
        lower_pct: Lower percentile bound (default 0.0)
        upper_pct: Upper percentile bound (default 98.0)
        softness: Compression factor for tail values (default 0.3)
    
    Returns:
        Transformed array with soft-compressed tails
    """
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return y
    p_lo = np.percentile(y, lower_pct)
    p_hi = np.percentile(y, upper_pct)
    scale = max(p_hi - p_lo, 1e-9)
    result = y.copy()
    outer_lo = y < p_lo
    outer_hi = y > p_hi
    result[outer_lo] = p_lo - scale * softness * np.tanh((p_lo - y[outer_lo]) / scale)
    result[outer_hi] = p_hi + scale * softness * np.tanh((y[outer_hi] - p_hi) / scale)
    return result.astype(np.float32)


def _soft_clip_sample_weights(
    weights: np.ndarray, lower: float = 0.01, upper_pct: float = 99.0, softness: float = 0.5
) -> np.ndarray:
    """Soft clip sample weights to preserve relative ordering while bounding extreme values.
    
    Args:
        weights: Sample weight array
        lower: Hard lower bound (default 0.01)
        upper_pct: Upper percentile for soft bound (default 99.0)
        softness: Compression factor for upper tail (default 0.5)
    
    Returns:
        Transformed weights with soft upper bound
    """
    weights = np.asarray(weights, dtype=float)
    if len(weights) == 0:
        return weights
    p_hi = np.percentile(weights, upper_pct)
    scale = max(p_hi - lower, 1e-9)
    result = weights.copy()
    result = np.maximum(result, lower)
    outer_hi = weights > p_hi
    result[outer_hi] = p_hi + scale * softness * np.tanh((weights[outer_hi] - p_hi) / scale)
    return result.astype(np.float32)


def _soft_clip_offset(
    offset: np.ndarray, offset_min: float, offset_max: float, softness: float = 0.3
) -> np.ndarray:
    """Soft clip offset targets to preserve relative ordering while bounding values.
    
    Args:
        offset: Offset array
        offset_min: Minimum offset bound
        offset_max: Maximum offset bound
        softness: Compression factor for bounds (default 0.3)
    
    Returns:
        Transformed offsets with soft bounds
    """
    offset = np.asarray(offset, dtype=float)
    if len(offset) == 0:
        return offset
    scale = max(offset_max - offset_min, 1e-9)
    result = offset.copy()
    outer_lo = offset < offset_min
    outer_hi = offset > offset_max
    result[outer_lo] = offset_min - scale * softness * np.tanh((offset_min - offset[outer_lo]) / scale)
    result[outer_hi] = offset_max + scale * softness * np.tanh((offset[outer_hi] - offset_max) / scale)
    return result.astype(np.float32)


def build_rank_target(
    returns: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    mode: str = "fold_local",
) -> np.ndarray:
    """T2 = rank_style_target. Always rank over the provided vector."""
    returns = np.asarray(returns)
    order = np.argsort(returns)
    ranks = np.empty_like(returns, dtype=np.float32)
    ranks[order] = (np.arange(len(returns)) + 0.5) / max(1, len(returns))
    return (ranks * 2.0) - 1.0


def _pad_series(series: np.ndarray, max_b: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(series)
    res = np.full((n, max_b), np.nan, dtype=np.float32)
    lens = np.zeros(n, dtype=np.int32)
    for i, arr in enumerate(series):
        use = min(len(arr), max_b)
        if use > 0:
            res[i, :use] = arr[:use]
            lens[i] = use
    return res, lens


def _precompute_padded_paths(
    trade_outcomes: pd.DataFrame, max_b: int = 24
) -> Dict[str, np.ndarray]:
    opens, _ = _pad_series(trade_outcomes["future_opens"].values, max_b)
    highs, _ = _pad_series(trade_outcomes["future_highs"].values, max_b)
    lows, _ = _pad_series(trade_outcomes["future_lows"].values, max_b)
    closes, path_lens = _pad_series(trade_outcomes["future_closes"].values, max_b)
    return {
        "opens_2d": opens,
        "highs_2d": highs,
        "lows_2d": lows,
        "closes_2d": closes,
        "path_lens": path_lens,
    }


def build_robust_utility_target(
    trade_outcomes: pd.DataFrame,
    cost_pct: float = 0.002,
    padded_paths: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    max_b = 24
    n = len(trade_outcomes)

    if padded_paths is None:
        padded_paths = _precompute_padded_paths(trade_outcomes, max_b)

    opens = padded_paths["opens_2d"]
    highs = padded_paths["highs_2d"]
    lows = padded_paths["lows_2d"]
    closes = padded_paths["closes_2d"]
    path_lens = padded_paths["path_lens"]

    entry_px = trade_outcomes["entry_price"].values
    atr = trade_outcomes["atr_12_15m"].values
    is_long = trade_outcomes["is_long"].values

    geometries = [
        (1.0, 1.5),
        (1.0, 2.0),
        (1.5, 1.5),
        (1.5, 2.0),
    ]

    u_acc = np.zeros(n, dtype=np.float32)
    for sl, tp_ratio in geometries:
        pol = LabelPolicy(
            sl_atr_mult=sl,
            tp_sl_ratio=tp_ratio,
            max_hold_bars=max_b,
            trail_activate_atr=1e9,  # disabled
            giveback_pct=0.0,
            early_exit_deadline_bars=0,
            early_exit_mfe_atr=0.0,
        )
        u, _ = _simulate_policy_batch(
            entry_px, atr, is_long, opens, highs, lows, closes, path_lens, pol, cost_pct
        )
        u_acc += u

    return u_acc / len(geometries)


def build_volatility_normalized_target(
    returns: np.ndarray, atr_values: np.ndarray, entry_prices: np.ndarray
) -> np.ndarray:
    """
    T4 = volatility_normalized_target
    raw_return / max(atr_like_scale, eps)
    """
    atr_pct = atr_values / np.maximum(entry_prices, 1e-9)
    return returns / np.maximum(atr_pct, 1e-6)


# ============================================================
# Layer A: Predictive models
# ============================================================


class Model1Edge:
    def __init__(self, target_name: str = "log_clipped_winsorized_net"):
        self.target_name = target_name
        self.model = Ridge(alpha=1.0)
        self.scaler = PredictionScaler()
        self.is_fitted = False
        self.model_type_ = "Ridge"

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y, sample_weight=sample_weight)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model1Edge not fitted")
        return self.model.predict(self.scaler.transform(X))


class Model2Downside:
    def __init__(self):
        self.model = Ridge(alpha=1.0)
        self.scaler = PredictionScaler()
        self.is_fitted = False
        self.model_type_ = "Ridge"
        self.target_transform_ = "log1p(abs(residuals))"

    def fit(
        self,
        X: np.ndarray,
        y_mae_atr: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y_mae_atr, sample_weight=sample_weight)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model2Downside not fitted")
        return self.model.predict(self.scaler.transform(X))


class Model3Uncertainty:
    def __init__(self):
        self.model = Ridge(alpha=1.0)
        self.scaler = PredictionScaler()
        self.is_fitted = False
        self.model_type_ = "Ridge"
        self.target_transform_ = "log1p(abs(residuals))"

    def fit(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ):
        y_target = np.log1p(np.abs(residuals))
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y_target, sample_weight=sample_weight)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model3Uncertainty not fitted")
        pred_log = self.model.predict(self.scaler.transform(X))
        return np.expm1(pred_log)


class LayerAPredictor:
    """
    Layer A Orchestrator:
    - Runs target race for Model 1 via temporal OOF.
    - Runs temporal OOF for Model 2.
    - Fits Uncertainty strictly on Model 1 OOF residuals.
    - Fits final models.
    """

    def __init__(self, lambda_downside: float = 0.5, eta_uncertainty: float = 0.5, config: dict = None):
        self.edge_model = Model1Edge()
        self.downside_model = Model2Downside()
        self.uncertainty_model = Model3Uncertainty()

        self.lambda_downside = lambda_downside
        self.eta_uncertainty = eta_uncertainty
        self.is_fitted = False
        self.config = config or {}

        # Component Scaling Artifacts
        self.score_blend_mode = self.config.get("score_blend_mode", "legacy_raw")

        # Ensure boolean parsing for config flags
        use_m3 = self.config.get("use_model3_uncertainty", True)
        if isinstance(use_m3, str):
            use_m3 = use_m3.lower() in ("true", "1", "yes", "y")
        self.use_model3_uncertainty = bool(use_m3)

        self.model1_target_mode = self.config.get("model1_target_mode", "race")
        self.fixed_model1_target_name = self.config.get("fixed_model1_target_name", "robust_utility_target")

        self.scaler_edge_mean_ = 0.0
        self.scaler_edge_scale_ = 1.0
        self.scaler_downside_mean_ = 0.0
        self.scaler_downside_scale_ = 1.0
        self.scaler_uncertainty_mean_ = 0.0
        self.scaler_uncertainty_scale_ = 1.0

        # Artifacts
        self.model1_target_race_results_ = None
        self.model1_best_target_name_ = None
        self.model1_oof_pred_ = None
        self.model1_oof_target_used_ = None  # Crucial for fold-consistent Model 3 training

        self.model2_eval_results_ = None
        self.model2_oof_pred_ = None

        self.model3_eval_results_ = None
        self.model3_oof_pred_ = None

        # Feature Selection Artifacts
        self.final_selected_feature_idx_edge_ = None
        self.final_selected_feature_idx_downside_ = None
        self.final_selected_feature_idx_uncertainty_ = None
        self.feature_selection_results_edge_ = None
        self.feature_selection_results_downside_ = None
        self.feature_selection_results_uncertainty_ = None

    def _run_model1_target_race(
        self,
        X: np.ndarray,
        trade_outcomes: pd.DataFrame,
        raw_returns: np.ndarray,
        timestamps: Optional[np.ndarray],
        sample_weight: Optional[np.ndarray],
    ):
        """Evaluates T1, T2, T3, T4. Picks winner via temporal OOF edge target score or uses fixed target."""
        splits, actual_n_splits = make_temporal_splits(
            timestamps, n_samples=len(X), n_splits=2
        )

        candidates = {
            "log_clipped_winsorized_net": build_log_clipped_target(raw_returns),
            "rank_style_target": build_rank_target(
                raw_returns, timestamps, mode="fold_local"
            ),
            "robust_utility_target": build_robust_utility_target(
                trade_outcomes, padded_paths=_precompute_padded_paths(trade_outcomes)
            ),
            "volatility_normalized_target": build_volatility_normalized_target(
                raw_returns,
                trade_outcomes["atr_12_15m"].values,
                trade_outcomes["entry_price"].values,
            ),
        }

        race_results = []
        best_score = -1e9
        best_name = "log_clipped_winsorized_net"
        best_oof = np.full(len(X), np.nan)
        best_oof_target = np.full(len(X), np.nan) # Store the fold-local target

        # If ablation mode is 'fixed', only evaluate that specific target
        cands_to_eval = candidates.keys()
        if self.model1_target_mode == "fixed":
            best_name = self.fixed_model1_target_name
            if best_name not in candidates:
                best_name = "robust_utility_target" # Fallback
            cands_to_eval = [best_name]

        # Pull FS Config
        fs_cfg = POSITION_SIZER_V2_FEATURE_SELECTION_CONFIG
        do_fs = fs_cfg.get("enabled", True)

        # Choose grid size based on n_samples (small bucket relaxation)
        n_samples = len(X)
        if n_samples < 1200:
            alpha_grid = np.logspace(-3, 0, 8)
            l1_grid = fs_cfg.get("l1_ratio_grid_small", [0.10, 0.25, 0.50])
            inner_cv = fs_cfg.get("inner_n_splits_default", 3)
        else:
            alpha_grid = np.logspace(-3, 0.5, 10)
            l1_grid = fs_cfg.get("l1_ratio_grid_large", [0.15, 0.40, 0.70])
            inner_cv = fs_cfg.get("inner_n_splits_default", 3)

        feature_names = get_feature_view(
            POSITION_SIZER_V2_FEATURE_CONFIG["model1_edge_feature_keys"], "X_linear"
        )

        for name in cands_to_eval:
            y_cand = candidates[name]
            oof_preds = np.full(len(X), np.nan)
            oof_targets = np.full(len(X), np.nan)
            fold_nets_10 = []
            fold_nets_20 = []

            for tr_idx, val_idx in splits:
                if name == "rank_style_target":
                    # Generate targets fold-locally
                    y_tr = build_rank_target(raw_returns[tr_idx], mode="fold_local")
                    y_val = build_rank_target(raw_returns[val_idx], mode="fold_local")
                else:
                    y_tr = y_cand[tr_idx]
                    y_val = y_cand[val_idx]

                # Store the exact target we are trying to predict
                oof_targets[val_idx] = y_val

                w_tr = sample_weight[tr_idx] if sample_weight is not None else None

                sel_idx = np.arange(X.shape[1])

                m = Model1Edge(target_name=name)
                m.fit(X[tr_idx][:, sel_idx], y_tr, w_tr)

                preds_val = m.predict(X[val_idx][:, sel_idx])
                oof_preds[val_idx] = preds_val

                top_mets = compute_top_slice_metrics(
                    preds_val, raw_returns[val_idx], top_fracs=(0.1, 0.2)
                )
                fold_nets_10.append(top_mets["top_10_mean_net"])
                fold_nets_20.append(top_mets["top_20_mean_net"])

            mean_net = float(np.mean(fold_nets_10)) if fold_nets_10 else 0.0
            std_net = float(np.std(fold_nets_10)) if len(fold_nets_10) > 1 else 0.0
            score = mean_net - 0.5 * std_net

            spear, _ = spearmanr(
                oof_preds[np.isfinite(oof_preds)],
                raw_returns[np.isfinite(oof_preds)],
                nan_policy="omit",
            )
            monot = compute_bucket_monotonicity(
                oof_preds[np.isfinite(oof_preds)], raw_returns[np.isfinite(oof_preds)]
            )

            full_top = compute_top_slice_metrics(
                oof_preds[np.isfinite(oof_preds)],
                raw_returns[np.isfinite(oof_preds)],
                (0.1,),
            )

            race_results.append(
                {
                    "target": name,
                    "score": score,
                    "spearman_ic_mean": float(spear) if pd.notna(spear) else 0.0,
                    "top_decile_realized_net_mean": mean_net,
                    "top_quintile_realized_net_mean": float(np.mean(fold_nets_20))
                    if fold_nets_20
                    else 0.0,
                    "top_decile_hit_rate_mean": full_top["top_10_hit_rate"],
                    "score_bucket_monotonicity": monot,
                    "fold_stability_std": std_net,
                }
            )

            if self.model1_target_mode == "fixed" or score > best_score:
                best_score = score
                best_name = name
                best_oof = oof_preds
                best_oof_target = oof_targets

        self.model1_target_race_results_ = pd.DataFrame(race_results)
        self.model1_best_target_name_ = best_name
        self.actual_n_splits_used_ = actual_n_splits
        self.model1_oof_pred_ = best_oof
        self.model1_oof_target_used_ = best_oof_target

        # Fit final Edge
        self.edge_model = Model1Edge(target_name=best_name)
        if best_name == "rank_style_target" and timestamps is None:
            y_final = build_rank_target(raw_returns, mode="fold_local")
        else:
            y_final = candidates[best_name]
        self.edge_model.fit(X, y_final, sample_weight)

        # We keep y_final, but it should NO LONGER be used for Model 3 residuals.
        self.model1_y_final_ = y_final

        # Perform Final Feature Selection for Model 1
        if do_fs:
            final_fs_res = select_features_via_elasticnet(
                X_train=X,
                y_train=y_final,
                timestamps_train=timestamps,
                model_kind="edge",
                feature_names=feature_names,
                alpha_grid=alpha_grid,
                l1_ratio_grid=l1_grid,
                sample_weight_train=sample_weight,
                inner_n_splits=inner_cv,
                max_features_cap=fs_cfg["max_features_cap"]["edge"],
                min_features_floor=fs_cfg["min_features_floor"]["edge"],
                sparsity_penalty=fs_cfg["sparsity_penalty"]["edge"],
            )
            self.final_selected_feature_idx_edge_ = final_fs_res["selected_idx"]
            self.feature_selection_results_edge_ = final_fs_res
        else:
            self.final_selected_feature_idx_edge_ = np.arange(X.shape[1])
            self.feature_selection_results_edge_ = None

        self.edge_model.fit(
            X[:, self.final_selected_feature_idx_edge_], y_final, sample_weight
        )

    def transform_model2_downside_target(self, y_downside: np.ndarray) -> np.ndarray:
        """Centralized helper for Downside target transform to ensure OOF and final fit match."""
        return _soft_winsorize_downside(y_downside, lower_pct=0.0, upper_pct=98.0, softness=0.3)

    def _run_model2_oof_eval(
        self,
        X: np.ndarray,
        y_downside: np.ndarray,
        timestamps: Optional[np.ndarray],
        sample_weight: Optional[np.ndarray],
    ):
        """OOF eval for Model 2 Downside."""
        splits, actual_n_splits = make_temporal_splits(
            timestamps, n_samples=len(X), n_splits=2
        )
        oof_preds = np.full(len(X), np.nan)

        fs_cfg = POSITION_SIZER_V2_FEATURE_SELECTION_CONFIG
        do_fs = fs_cfg.get("enabled", True)
        n_samples = len(X)
        if n_samples < 1200:
            alpha_grid = np.logspace(-3, 0, 8)
            l1_grid = fs_cfg.get("l1_ratio_grid_small", [0.10, 0.25, 0.50])
            inner_cv = fs_cfg.get("inner_n_splits_default", 3)
        else:
            alpha_grid = np.logspace(-3, 0.5, 10)
            l1_grid = fs_cfg.get("l1_ratio_grid_large", [0.15, 0.40, 0.70])
            inner_cv = fs_cfg.get("inner_n_splits_default", 3)

        feature_names = get_feature_view(
            POSITION_SIZER_V2_FEATURE_CONFIG["model2_downside_feature_keys"], "X_linear"
        )

        for tr_idx, val_idx in splits:
            w_tr = sample_weight[tr_idx] if sample_weight is not None else None
            y_tr_down = self.transform_model2_downside_target(y_downside[tr_idx])

            sel_idx = np.arange(X.shape[1])

            m = Model2Downside()
            m.fit(X[tr_idx][:, sel_idx], y_tr_down, w_tr)
            oof_preds[val_idx] = m.predict(X[val_idx][:, sel_idx])

        valid = np.isfinite(oof_preds)
        if np.any(valid):
            mae = float(np.mean(np.abs(oof_preds[valid] - y_downside[valid])))
            monot = compute_bucket_monotonicity(oof_preds[valid], y_downside[valid])
            fsr = compute_false_safe_rate(
                oof_preds[valid], y_downside[valid], low_q=0.2, high_q=0.8
            )

            # Simple manual Huber eval
            err = np.abs(oof_preds[valid] - y_downside[valid])
            delta = 1.35
            quad = np.minimum(err, delta)
            lin = err - quad
            hloss = float(np.mean(0.5 * quad**2 + delta * lin))
        else:
            mae = hloss = monot = fsr = 0.0

        self.model2_eval_results_ = {
            "oof_mae": mae,
            "oof_huber_loss": hloss,
            "downside_decile_monotonicity": monot,
            "false_safe_rate": fsr,
        }
        self.model2_oof_pred_ = oof_preds

        # Fit final Downside model with consistent transform
        y_final = self.transform_model2_downside_target(y_downside)

        if do_fs:
            final_fs_res = select_features_via_elasticnet(
                X_train=X,
                y_train=y_final,
                timestamps_train=timestamps,
                model_kind="downside",
                feature_names=feature_names,
                alpha_grid=alpha_grid,
                l1_ratio_grid=l1_grid,
                sample_weight_train=sample_weight,
                inner_n_splits=inner_cv,
                max_features_cap=fs_cfg["max_features_cap"]["downside"],
                min_features_floor=fs_cfg["min_features_floor"]["downside"],
                sparsity_penalty=fs_cfg["sparsity_penalty"]["downside"],
            )
            self.final_selected_feature_idx_downside_ = final_fs_res["selected_idx"]
            self.feature_selection_results_downside_ = final_fs_res
        else:
            self.final_selected_feature_idx_downside_ = np.arange(X.shape[1])
            self.feature_selection_results_downside_ = None

        self.downside_model.fit(
            X[:, self.final_selected_feature_idx_downside_], y_final, sample_weight
        )

    def build_model3_residual_target_from_model1_oof(self) -> np.ndarray:
        """Constructs Model 3 residual targets strictly from fold-consistent Model 1 targets."""
        valid_oof = np.isfinite(self.model1_oof_pred_) & np.isfinite(self.model1_oof_target_used_)
        residuals = np.full(len(self.model1_oof_pred_), np.nan)
        if np.any(valid_oof):
            residuals[valid_oof] = self.model1_oof_target_used_[valid_oof] - self.model1_oof_pred_[valid_oof]
        return residuals

    def _run_model3_oof_eval(
        self,
        X: np.ndarray,
        timestamps: Optional[np.ndarray],
        sample_weight: Optional[np.ndarray],
    ):
        """OOF eval for Uncertainty. Fits on Model 1 OOF residuals strictly from paired OOF target artifacts."""
        all_residuals = self.build_model3_residual_target_from_model1_oof()
        valid_res = np.isfinite(all_residuals)

        if not np.any(valid_res):
            logger.warning("No valid Model 3 residuals. Skipping Model 3 Uncertainty training.")
            return

        residuals = all_residuals[valid_res]
        X_res = X[valid_res]
        w_res = sample_weight[valid_res] if sample_weight is not None else None

        splits, actual_n_splits = make_temporal_splits(
            timestamps[valid_oof] if timestamps is not None else None,
            n_samples=len(X_res),
            n_splits=2,
        )
        oof_preds = np.full(len(X_res), np.nan)

        fs_cfg = POSITION_SIZER_V2_FEATURE_SELECTION_CONFIG
        do_fs = fs_cfg.get("enabled", True)
        n_samples = len(X_res)
        if n_samples < 1200:
            alpha_grid = np.logspace(-3, 0, 8)
            l1_grid = fs_cfg.get("l1_ratio_grid_small", [0.10, 0.25, 0.50])
            inner_cv = fs_cfg.get("inner_n_splits_default", 3)
        else:
            alpha_grid = np.logspace(-3, 0.5, 10)
            l1_grid = fs_cfg.get("l1_ratio_grid_large", [0.15, 0.40, 0.70])
            inner_cv = fs_cfg.get("inner_n_splits_default", 3)

        feature_names = get_feature_view(
            POSITION_SIZER_V2_FEATURE_CONFIG["model3_uncertainty_feature_keys"],
            "X_linear",
        )

        for tr_idx, val_idx in splits:
            w_tr = w_res[tr_idx] if w_res is not None else None
            sel_idx = np.arange(X_res.shape[1])

            m = Model3Uncertainty()
            m.fit(X_res[tr_idx][:, sel_idx], residuals[tr_idx], w_tr)
            oof_preds[val_idx] = m.predict(X_res[val_idx][:, sel_idx])

        valid2 = np.isfinite(oof_preds)
        realized_abs = np.abs(residuals)

        if np.any(valid2):
            calib = compute_uncertainty_calibration(
                oof_preds[valid2], realized_abs[valid2]
            )
        else:
            calib = {}

        self.model3_eval_results_ = calib
        self.model3_oof_pred_ = oof_preds

        # Fit final Uncertainty model
        if do_fs:
            final_fs_res = select_features_via_elasticnet(
                X_train=X_res,
                y_train=residuals,
                timestamps_train=timestamps[valid_oof]
                if timestamps is not None
                else None,
                model_kind="uncertainty",
                feature_names=feature_names,
                alpha_grid=alpha_grid,
                l1_ratio_grid=l1_grid,
                sample_weight_train=w_res,
                inner_n_splits=inner_cv,
                max_features_cap=fs_cfg["max_features_cap"]["uncertainty"],
                min_features_floor=fs_cfg["min_features_floor"]["uncertainty"],
                sparsity_penalty=fs_cfg["sparsity_penalty"]["uncertainty"],
            )
            self.final_selected_feature_idx_uncertainty_ = final_fs_res["selected_idx"]
            self.feature_selection_results_uncertainty_ = final_fs_res
        else:
            self.final_selected_feature_idx_uncertainty_ = np.arange(X_res.shape[1])
            self.feature_selection_results_uncertainty_ = None

        self.uncertainty_model.fit(
            X_res[:, self.final_selected_feature_idx_uncertainty_], residuals, w_res
        )

    def fit(
        self,
        feature_dict: Dict[str, np.ndarray],
        trade_outcomes: pd.DataFrame,
        y_raw_net_return: np.ndarray,
        y_downside: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ):
        # 0. Feature Diagnostics
        self.feature_coverage_report_ = {
            "model1_requested": get_feature_view(
                POSITION_SIZER_V2_FEATURE_CONFIG["model1_edge_feature_keys"], "X_linear"
            ),
            "model2_requested": get_feature_view(
                POSITION_SIZER_V2_FEATURE_CONFIG["model2_downside_feature_keys"],
                "X_linear",
            ),
            "model3_requested": get_feature_view(
                POSITION_SIZER_V2_FEATURE_CONFIG["model3_uncertainty_feature_keys"],
                "X_linear",
            ),
            "available_in_dict": list(feature_dict.keys()),
        }

        def _get_missing(requested):
            return [k for k in requested if k not in feature_dict]

        self.feature_coverage_report_["model1_missing"] = _get_missing(
            get_feature_view(
                POSITION_SIZER_V2_FEATURE_CONFIG["model1_edge_feature_keys"], "X_linear"
            )
        )
        self.feature_coverage_report_["model2_missing"] = _get_missing(
            get_feature_view(
                POSITION_SIZER_V2_FEATURE_CONFIG["model2_downside_feature_keys"],
                "X_linear",
            )
        )
        self.feature_coverage_report_["model3_missing"] = [
            k
            for k in get_feature_view(
                POSITION_SIZER_V2_FEATURE_CONFIG["model3_uncertainty_feature_keys"],
                "X_linear",
            )
            if k not in feature_dict
            and k
            not in [
                "edge_pred",
                "downside_pred",
                "edge_minus_downside",
                "abs_edge_pred",
            ]
        ]

        # 1. Assemble X1, X2
        X1 = assemble_feature_matrix(
            feature_dict,
            get_feature_view(
                POSITION_SIZER_V2_FEATURE_CONFIG["model1_edge_feature_keys"], "X_linear"
            ),
        )
        X2 = assemble_feature_matrix(
            feature_dict,
            get_feature_view(
                POSITION_SIZER_V2_FEATURE_CONFIG["model2_downside_feature_keys"],
                "X_linear",
            ),
        )

        self.feature_coverage_report_["X1_shape"] = X1.shape
        self.feature_coverage_report_["X2_shape"] = X2.shape

        # Soft-bound sample weights to prevent explosive leverage while preserving ordering
        if sample_weight is not None:
            sw_finite = sample_weight[np.isfinite(sample_weight) & (sample_weight > 0)]
            if len(sw_finite) > 0:
                sample_weight = _soft_clip_sample_weights(sample_weight, lower=0.01, upper_pct=99.0, softness=0.5)
                sample_weight[~np.isfinite(sample_weight)] = 0.01
            else:
                sample_weight = None

        # Orchestration steps as required:
        # a) run Model 1 target race with OOF preds, c) fits final model 1
        self._run_model1_target_race(
            X1, trade_outcomes, y_raw_net_return, timestamps, sample_weight
        )

        # b) run Model 2 OOF eval, d) fits final model 2
        self._run_model2_oof_eval(X2, y_downside, timestamps, sample_weight)

        # Assemble X3 using OOF predictions
        fd3 = feature_dict.copy()
        fd3["edge_pred"] = self.model1_oof_pred_
        fd3["downside_pred"] = self.model2_oof_pred_
        valid12 = np.isfinite(self.model1_oof_pred_) & np.isfinite(
            self.model2_oof_pred_
        )
        fd3["edge_minus_downside"] = np.where(
            valid12,
            self.model1_oof_pred_ - self.lambda_downside * self.model2_oof_pred_,
            0.0,
        )
        fd3["abs_edge_pred"] = np.where(
            np.isfinite(self.model1_oof_pred_), np.abs(self.model1_oof_pred_), 0.0
        )

        # Pull required OOF inputs from feature dict explicitly to avoid key errors in Model 3 assembly
        if "oof_asym_hat" in feature_dict:
            fd3["oof_asym_hat"] = feature_dict["oof_asym_hat"]
        else:
            # Fallback if missing upstream
            fd3["oof_asym_hat"] = np.zeros(len(self.model1_oof_pred_))

        X3 = assemble_feature_matrix(
            fd3,
            get_feature_view(
                POSITION_SIZER_V2_FEATURE_CONFIG["model3_uncertainty_feature_keys"],
                "X_linear",
            ),
        )
        self.feature_coverage_report_["X3_shape"] = X3.shape

        # e) fit final Model 3 on OOF residual target using the dimensionally accurate winning target
        if self.use_model3_uncertainty:
            self._run_model3_oof_eval(X3, timestamps, sample_weight)
        else:
            self.model3_oof_pred_ = np.zeros(len(X3))
            self.model3_eval_results_ = {}
            self.final_selected_feature_idx_uncertainty_ = np.arange(X3.shape[1])
            self.feature_selection_results_uncertainty_ = None

        self.fit_layerA_component_scalers(self.model1_oof_pred_, self.model2_oof_pred_, self.model3_oof_pred_)

        self.log_layerA_diagnostics()

        self.is_fitted = True
        return self

    def log_layerA_diagnostics(self):
        """Logs detailed component semantics, dominance, and target race outcomes."""
        logger.info("=" * 80)
        logger.info("LAYER A DIAGNOSTICS")
        logger.info("=" * 80)

        # Model 1 Diagnostics
        logger.info(f"Model 1 (Edge) Target Mode: {self.model1_target_mode}")
        if self.model1_target_race_results_ is not None and not self.model1_target_race_results_.empty:
            race_df = self.model1_target_race_results_.sort_values(by="score", ascending=False)
            logger.info("Target Race Standings:")
            for i, row in race_df.iterrows():
                logger.info(f"  {row['target']}: Score={row['score']:.4f} (Win Rate={row['top_decile_hit_rate_mean']:.2%})")
            if len(race_df) > 1:
                winner = race_df.iloc[0]
                runner_up = race_df.iloc[1]
                logger.info(f"  Winner Margin: {winner['score'] - runner_up['score']:.4f}")

        logger.info(f"Selected Target: {self.model1_best_target_name_}")

        def _stats(arr, name):
            if arr is None or not np.any(np.isfinite(arr)):
                logger.info(f"{name} Stats: N/A")
                return
            arr_f = arr[np.isfinite(arr)]
            logger.info(f"{name} Stats: Mean={np.mean(arr_f):.4f}, Std={np.std(arr_f):.4f}, Min={np.min(arr_f):.4f}, Max={np.max(arr_f):.4f}")

        _stats(self.model1_oof_pred_, "Model 1 OOF Pred (Raw)")

        # Model 2 Diagnostics
        logger.info("-" * 40)
        _stats(self.model2_oof_pred_, "Model 2 OOF Pred (Raw)")

        # Model 3 Diagnostics
        logger.info("-" * 40)
        logger.info(f"Model 3 Enabled: {self.use_model3_uncertainty}")
        if self.use_model3_uncertainty:
            residuals = self.build_model3_residual_target_from_model1_oof()
            _stats(residuals, "Model 3 Residual Target")
            _stats(self.model3_oof_pred_, "Model 3 OOF Pred (Raw)")

        # Blending Diagnostics
        logger.info("-" * 40)
        logger.info(f"Score Blend Mode: {self.score_blend_mode}")

        scaled = self.transform_layerA_components(self.model1_oof_pred_, self.model2_oof_pred_, self.model3_oof_pred_)
        _stats(scaled["edge"], "Model 1 Scaled Component")
        _stats(scaled["downside"], "Model 2 Scaled Component")
        _stats(scaled["uncertainty"], "Model 3 Scaled Component")

        # Correlations
        valid = np.isfinite(self.model1_oof_pred_) & np.isfinite(self.model2_oof_pred_) & np.isfinite(self.model3_oof_pred_)
        if np.sum(valid) > 2:
            df = pd.DataFrame({
                "Edge": scaled["edge"][valid],
                "Downside": scaled["downside"][valid],
                "Uncertainty": scaled["uncertainty"][valid]
            })
            corr = df.corr(method="spearman")
            logger.info("Spearman Correlations between Scaled Components:")
            logger.info(f"  Edge vs Downside:   {corr.loc['Edge', 'Downside']:.4f}")
            logger.info(f"  Edge vs Uncertainty: {corr.loc['Edge', 'Uncertainty']:.4f}")
            logger.info(f"  Downside vs Uncertainty: {corr.loc['Downside', 'Uncertainty']:.4f}")

        logger.info("=" * 80)

    def predict_components(
        self, feature_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("LayerAPredictor not fitted")

        X1 = assemble_feature_matrix(
            feature_dict,
            get_feature_view(
                POSITION_SIZER_V2_FEATURE_CONFIG["model1_edge_feature_keys"], "X_linear"
            ),
        )
        X2 = assemble_feature_matrix(
            feature_dict,
            get_feature_view(
                POSITION_SIZER_V2_FEATURE_CONFIG["model2_downside_feature_keys"],
                "X_linear",
            ),
        )

        edge_p = self.edge_model.predict(X1[:, self.final_selected_feature_idx_edge_])
        downside_p = self.downside_model.predict(
            X2[:, self.final_selected_feature_idx_downside_]
        )

        fd3 = feature_dict.copy()
        fd3["edge_pred"] = edge_p
        fd3["downside_pred"] = downside_p
        fd3["edge_minus_downside"] = edge_p - self.lambda_downside * downside_p
        fd3["abs_edge_pred"] = np.abs(edge_p)

        if "oof_asym_hat" in feature_dict:
            fd3["oof_asym_hat"] = feature_dict["oof_asym_hat"]
        else:
            fd3["oof_asym_hat"] = np.zeros_like(edge_p)

        X3 = assemble_feature_matrix(
            fd3,
            get_feature_view(
                POSITION_SIZER_V2_FEATURE_CONFIG["model3_uncertainty_feature_keys"],
                "X_linear",
            ),
        )

        if self.use_model3_uncertainty:
            uncertainty_p = self.uncertainty_model.predict(X3[:, self.final_selected_feature_idx_uncertainty_])
        else:
            uncertainty_p = np.zeros_like(edge_p)

        return {
            "edge": edge_p,
            "downside": downside_p,
            "uncertainty": uncertainty_p,
        }

    def fit_layerA_component_scalers(
        self, edge_pred: np.ndarray, downside_pred: np.ndarray, uncertainty_pred: np.ndarray
    ):
        """Fits robust standard scalers on the OOF predictions (training fold equivalents)."""
        def _get_stats(arr):
            valid = np.isfinite(arr)
            if np.sum(valid) > 1:
                mu = np.median(arr[valid])
                p_75, p_25 = np.percentile(arr[valid], [75, 25])
                scale = max((p_75 - p_25) / 1.349, 1e-9)  # IQR to std mapping
                return mu, scale
            return 0.0, 1.0

        self.scaler_edge_mean_, self.scaler_edge_scale_ = _get_stats(edge_pred)
        self.scaler_downside_mean_, self.scaler_downside_scale_ = _get_stats(downside_pred)
        self.scaler_uncertainty_mean_, self.scaler_uncertainty_scale_ = _get_stats(uncertainty_pred)

    def transform_layerA_components(
        self, edge_pred: np.ndarray, downside_pred: np.ndarray, uncertainty_pred: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Applies the fitted robust scalers to predictions."""
        e_scaled = (edge_pred - self.scaler_edge_mean_) / self.scaler_edge_scale_
        d_scaled = (downside_pred - self.scaler_downside_mean_) / self.scaler_downside_scale_
        u_scaled = (uncertainty_pred - self.scaler_uncertainty_mean_) / self.scaler_uncertainty_scale_

        # Soft-clip to prevent crazy tails from destroying blend
        def _clip(x):
            return np.clip(x, -5.0, 5.0)

        return {
            "edge": _clip(e_scaled),
            "downside": _clip(d_scaled),
            "uncertainty": _clip(u_scaled)
        }

    def combine_layerA_score(self, comps: Dict[str, np.ndarray]) -> np.ndarray:
        """Blends components according to configured mode."""
        if self.score_blend_mode == "train_scaled_components":
            scaled_comps = self.transform_layerA_components(
                comps["edge"], comps["downside"], comps["uncertainty"]
            )
            return (
                scaled_comps["edge"]
                - (self.lambda_downside * scaled_comps["downside"])
                - (self.eta_uncertainty * scaled_comps["uncertainty"])
            )
        else: # "legacy_raw"
            return (
                comps["edge"]
                - (self.lambda_downside * comps["downside"])
                - (self.eta_uncertainty * comps["uncertainty"])
            )

    def predict_score(self, feature_dict: Dict[str, np.ndarray]) -> np.ndarray:
        comps = self.predict_components(feature_dict)
        return self.combine_layerA_score(comps)

    def initial_sizing(
        self, score: np.ndarray, threshold: float = 0.0, base_size: float = 0.05
    ) -> np.ndarray:
        active = score > threshold
        sizes = np.zeros_like(score)
        if np.any(active):
            s_active = score[active]
            min_s, max_s = s_active.min(), s_active.max()
            if max_s > min_s:
                sizes[active] = base_size * (1.0 + (s_active - min_s) / (max_s - min_s))
            else:
                sizes[active] = base_size
        return np.clip(sizes, 0.0, 1.0)


# ============================================================
# Layer B: Policy Optimization
# ============================================================


def _standardize(x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    if len(x_arr) < 2:
        return np.zeros_like(x_arr)
    std = np.std(x_arr)
    if std < 1e-9:
        return np.zeros_like(x_arr)
    return (x_arr - np.mean(x_arr)) / std


class LayerBPolicyOptimizer:
    """
    Layer B: Moves utility optimization out of labels and into simulation.
    Optimizes exit geometry, selection boundaries, and time management.

    Standardized empty fold behavior:
    no active trades -> pnl_day = 0, sortino = 0, maxDD = 1, timeout_rate = 1
    """

    def __init__(
        self,
        cost_pct: float = 0.002,
        lambda_penalty: float = 0.5,
        annualization_factor: Optional[float] = None,
    ):
        self.cost_pct = cost_pct
        self.lambda_penalty = lambda_penalty
        self.annualization_factor = (
            annualization_factor if annualization_factor is not None else 1.0
        )  # default no annualization
        self.best_policy = {}
        self.best_j = -1e9

        self.obj_a = 1.0  # Sortino weight
        self.obj_b = 1.0  # MaxDD weight
        self.obj_c = 1.0  # Instability weight

        self.layer_b_candidate_table_ = None
        self.layer_b_selected_objective_components_ = None

    def _eval_policy_over_folds(
        self,
        pol: LabelPolicy,
        trade_outcomes: pd.DataFrame,
        scores: np.ndarray,
        score_quantile_fraction: float,
        splits: List[Tuple[np.ndarray, np.ndarray]],
        timestamps: np.ndarray,
        padded_paths: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate a single LabelPolicy over temporal validation folds.
        _simulate_policy_batch returns log-return-like utility per trade (u).
        """
        TIMEOUT_REASON_IDX = 3
        fold_pnl_days = []
        fold_sortinos = []
        fold_maxdds = []
        fold_timeout_rates = []

        min_days_floor = (
            1.0 / 24.0
        )  # Fix: lower floor for short intraday evaluation bounds

        for _, val_idx in splits:
            if len(val_idx) == 0:
                continue

            scores_val = scores[val_idx]
            fold_thresh = np.percentile(scores_val, score_quantile_fraction * 100)
            active_mask = scores_val >= fold_thresh
            active_val_idx = val_idx[active_mask]

            if len(active_val_idx) == 0:
                fold_pnl_days.append(0.0)
                fold_sortinos.append(0.0)
                fold_maxdds.append(1.0)
                fold_timeout_rates.append(1.0)
                continue

            entry_prices = trade_outcomes["entry_price"].values[active_val_idx]
            atr_entries = trade_outcomes["atr_12_15m"].values[active_val_idx]
            is_longs = trade_outcomes["is_long"].values[active_val_idx]

            max_b = min(48, pol.max_hold_bars)
            if padded_paths is not None:
                opens_2d = padded_paths["opens_2d"][active_val_idx, :max_b]
                highs_2d = padded_paths["highs_2d"][active_val_idx, :max_b]
                lows_2d = padded_paths["lows_2d"][active_val_idx, :max_b]
                closes_2d = padded_paths["closes_2d"][active_val_idx, :max_b]
                path_lens = np.minimum(padded_paths["path_lens"][active_val_idx], max_b)
            else:
                tmp_paths = _precompute_padded_paths(
                    trade_outcomes.iloc[active_val_idx], max_b
                )
                opens_2d = tmp_paths["opens_2d"]
                highs_2d = tmp_paths["highs_2d"]
                lows_2d = tmp_paths["lows_2d"]
                closes_2d = tmp_paths["closes_2d"]
                path_lens = tmp_paths["path_lens"]

            # u is log-return-like utility returned by simulator
            u, reason_counts = _simulate_policy_batch(
                entry_prices=entry_prices,
                atr_entries=atr_entries,
                is_longs=is_longs,
                opens_2d=opens_2d,
                highs_2d=highs_2d,
                lows_2d=lows_2d,
                closes_2d=closes_2d,
                path_lengths=path_lens,
                policy=pol,
                cost_pct=self.cost_pct,
            )

            if len(u) == 0:
                fold_pnl_days.append(0.0)
                fold_sortinos.append(0.0)
                fold_maxdds.append(1.0)
                fold_timeout_rates.append(1.0)
                continue

            ret = np.expm1(u)
            pnl = float(np.sum(ret))

            ts = pd.to_datetime(timestamps[active_val_idx])
            days = max((ts.max() - ts.min()).total_seconds() / 86400.0, min_days_floor)
            pnl_day = pnl / days

            # Defensive check for stable_equity function mapping (already globally imported)
            _, dd = _stable_equity_and_drawdown(ret)
            maxDD = float(np.max(dd)) if len(dd) > 0 else 1.0

            neg_rets = ret[ret < 0]
            dd_dev = (
                float(np.sqrt(np.mean(neg_rets**2))) if len(neg_rets) > 0 else 1e-3
            )
            sortino = float(np.mean(ret) / (dd_dev + 1e-9) * self.annualization_factor)

            fold_pnl_days.append(pnl_day)
            fold_sortinos.append(sortino)
            fold_maxdds.append(maxDD)

            # Defensive index access for timeout reason code
            to_idx = min(TIMEOUT_REASON_IDX, len(reason_counts) - 1)
            fold_timeout_rates.append(reason_counts[to_idx] / len(u))

        return {
            "net_pnl_day": float(np.mean(fold_pnl_days)),
            "sortino": float(np.mean(fold_sortinos)),
            "maxDD": float(np.mean(fold_maxdds)),
            "instability": float(np.std(fold_pnl_days)),
            "timeout_rate": float(np.mean(fold_timeout_rates)),
        }

    def _score_candidates(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not results:
            return []

        pnl_arr = np.array([r["net_pnl_day"] for r in results])
        sortino_arr = np.array([r["sortino"] for r in results])
        maxdd_arr = np.array([r["maxDD"] for r in results])
        instab_arr = np.array([r["instability"] for r in results])

        z_pnl = _standardize(pnl_arr)
        z_sortino = _standardize(sortino_arr)
        z_maxdd = _standardize(maxdd_arr)
        z_instab = _standardize(instab_arr)

        for i, r in enumerate(results):
            u = (
                z_pnl[i]
                + self.obj_a * z_sortino[i]
                - self.obj_b * z_maxdd[i]
                - self.obj_c * z_instab[i]
            )
            r["utility"] = float(u)

        return sorted(results, key=lambda x: x["utility"], reverse=True)

    def optimize(
        self, scores: np.ndarray, trade_outcomes: pd.DataFrame, timestamps: np.ndarray
    ):
        splits, actual_n_splits = make_temporal_splits(
            timestamps, n_samples=len(scores), n_splits=3
        )
        candidate_table = []

        # --- Step 1: Exit Geometry (SL, Trailing, Giveback) ---
        sl_cands = [1.0, 1.5, 2.0]
        tp_cands = [1.5, 2.0, 2.5]
        trail_cands = [0.8, 1.0, 1.5]
        giveback_cands = [0.35, 0.50]

        tprint("Layer B Step 1: Optimizing exit geometry over temporal folds...")
        step1_results = []
        for sl, tp, trail, gb in itertools.product(
            sl_cands, tp_cands, trail_cands, giveback_cands
        ):
            pol = LabelPolicy(
                sl_atr_mult=sl,
                tp_sl_ratio=tp,
                max_hold_bars=24,
                trail_activate_atr=trail,
                giveback_pct=gb,
                early_exit_deadline_bars=0,
                early_exit_mfe_atr=0.0,
            )
            res = self._eval_policy_over_folds(
                pol,
                trade_outcomes,
                scores,
                score_quantile_fraction=0.60,
                splits=splits,
                timestamps=timestamps,
                padded_paths=padded_paths,
            )
            res["step"] = 1
            res["policy_obj"] = pol
            res["threshold_q"] = 0.60
            step1_results.append(res)

        step1_scored = self._score_candidates(step1_results)
        candidate_table.extend(step1_scored)
        best_geom = step1_scored[0]["policy_obj"]

        # Staged pruning: top 2
        top_k_step1 = step1_scored[:2]

        # --- Step 2: Selection Boundary ---
        tprint("Layer B Step 2: Optimizing selection boundary...")
        step2_results = []
        for geom_cand in top_k_step1:
            best_geom_cand = geom_cand["policy_obj"]
            for q in [0.70, 0.50]:
                res = self._eval_policy_over_folds(
                    best_geom_cand,
                    trade_outcomes,
                    scores,
                    score_quantile_fraction=q,
                    splits=splits,
                    timestamps=timestamps,
                    padded_paths=padded_paths,
                )
                res["step"] = 2
                res["policy_obj"] = best_geom_cand
                res["score_quantile_fraction"] = q
                step2_results.append(res)

        step2_scored = self._score_candidates(step2_results)
        candidate_table.extend(step2_scored)

        top_k_step2 = step2_scored[:2]

        # --- Step 3: Time Management ---
        tprint("Layer B Step 3: Optimizing time management...")
        timeout_cands = [16, 24]
        early_deadline_cands = [0, 8]
        early_mfe_cands = [0.25]

        step3_results = []
        for cand2 in top_k_step2:
            best_geom2 = cand2["policy_obj"]
            best_q = cand2["score_quantile_fraction"]
            for t_out, ed, em in itertools.product(
                timeout_cands, early_deadline_cands, early_mfe_cands
            ):
                if ed == 0:
                    em = 0.0
                pol = LabelPolicy(
                    sl_atr_mult=best_geom2.sl_atr_mult,
                    tp_sl_ratio=best_geom2.tp_sl_ratio,
                    max_hold_bars=t_out,
                    trail_activate_atr=best_geom2.trail_activate_atr,
                    giveback_pct=best_geom2.giveback_pct,
                    early_exit_deadline_bars=ed,
                    early_exit_mfe_atr=em,
                )
                res = self._eval_policy_over_folds(
                    pol,
                    trade_outcomes,
                    scores,
                    score_quantile_fraction=best_q,
                    splits=splits,
                    timestamps=timestamps,
                    padded_paths=padded_paths,
                )
                res["step"] = 3
                res["policy_obj"] = pol
                res["score_quantile_fraction"] = best_q
                step3_results.append(res)

        step3_scored = self._score_candidates(step3_results)
        candidate_table.extend(step3_scored)
        best_pol = step3_scored[0]["policy_obj"]

        # Artifact storage ensuring fully reconstructable records
        self.layer_b_candidate_table_ = pd.DataFrame(
            [
                {
                    "step": r["step"],
                    "score_quantile_fraction": r.get("score_quantile_fraction", 0.0),
                    "sl_atr_mult": r["policy_obj"].sl_atr_mult,
                    "tp_sl_ratio": r["policy_obj"].tp_sl_ratio,
                    "trail_activate_atr": r["policy_obj"].trail_activate_atr,
                    "giveback_pct": r["policy_obj"].giveback_pct,
                    "max_hold_bars": r["policy_obj"].max_hold_bars,
                    "early_exit_deadline_bars": r[
                        "policy_obj"
                    ].early_exit_deadline_bars,
                    "early_exit_mfe_atr": r["policy_obj"].early_exit_mfe_atr,
                    "net_pnl_day": r["net_pnl_day"],
                    "sortino": r["sortino"],
                    "maxDD": r["maxDD"],
                    "instability": r["instability"],
                    "utility": r["utility"],
                }
                for r in candidate_table
            ]
        )

        self.layer_b_selected_objective_components_ = step3_scored[0]
        self.best_j = step3_scored[0]["utility"]

        best_q_final = step3_scored[0].get("score_quantile_fraction", 0.0)
        final_thresh_val = np.percentile(scores, best_q_final * 100)
        self.best_policy = {
            "sl_atr_mult": best_pol.sl_atr_mult,
            "tp_sl_ratio": best_pol.tp_sl_ratio,
            "giveback_pct": best_pol.giveback_pct,
            "trail_activate_atr": best_pol.trail_activate_atr,
            "score_threshold": float(final_thresh_val),
            "max_hold_bars": best_pol.max_hold_bars,
            "early_exit_deadline_bars": best_pol.early_exit_deadline_bars,
            "early_exit_mfe_atr": best_pol.early_exit_mfe_atr,
        }
        return self.best_policy


# ============================================================
# Layer C: Execution Optimization
# ============================================================


def fit_sizing_normalizer(
    scores: np.ndarray,
    threshold: float,
    mode: str = "train_distribution_absolute",
) -> Dict[str, Any]:
    active = scores[scores >= threshold]
    state = {"sizing_norm_mode": mode, "lower_anchor": threshold, "upper_anchor": threshold}

    if mode == "legacy_batch_minmax":
        # Anchors will be overwritten dynamically at transform time in legacy mode,
        # but we capture the threshold as a fallback.
        pass
    elif mode == "train_distribution_absolute":
        if len(active) > 0:
            upper = np.percentile(active, 95)
            if upper > threshold:
                state["upper_anchor"] = float(upper)
            else:
                state["upper_anchor"] = threshold
    else:
        raise ValueError(f"Unknown sizing_norm_mode: {mode}")

    return state


def transform_scores_to_sizing_input(
    scores: np.ndarray,
    normalizer_state: Dict[str, Any],
    fallback_threshold: float = 0.0,
) -> np.ndarray:
    mode = normalizer_state.get("sizing_norm_mode", "legacy_batch_minmax")
    s_norm = np.zeros_like(scores, dtype=float)

    if mode == "legacy_batch_minmax":
        # Legacy behavior: dynamically normalizes based on the passed batch.
        active_mask = scores >= fallback_threshold
        if np.any(active_mask):
            s_act = scores[active_mask]
            s_min, s_max = s_act.min(), s_act.max()
            if s_max > s_min:
                s_norm[active_mask] = (s_act - s_min) / (s_max - s_min)
            else:
                s_norm[active_mask] = 0.0
    elif mode == "train_distribution_absolute":
        lower = normalizer_state["lower_anchor"]
        upper = normalizer_state["upper_anchor"]
        if upper > lower:
            s_norm = np.clip((scores - lower) / (upper - lower), 0.0, 1.0)
        else:
            s_norm = np.zeros_like(scores, dtype=float)

    return s_norm


def apply_sizing_curve(
    s_norm: np.ndarray,
    base_size: float,
    mode: str,
    max_size: Optional[float] = None,
) -> np.ndarray:
    # If max_size is None, assume the original effective cap of 2 * base_size
    multiplier = (max_size - base_size) if max_size is not None else base_size
    sizes = np.full_like(s_norm, base_size)

    if mode == "linear":
        sizes = base_size + multiplier * s_norm
    elif mode == "convex":
        sizes = base_size + multiplier * (s_norm**2)
    elif mode == "concave":
        sizes = base_size + multiplier * np.sqrt(s_norm)
    else:
        raise ValueError(f"Unknown sizing curve mode: {mode}")

    return np.clip(sizes, 0.0, 1.0)


class LayerCExecutionOptimizer:
    """
    Layer C: Sizing mapping and Ridge Limit offset optimizer.
    Limit offset semantic bounds: 5.0 to 50.0 explicitly in basis points to match operational logic.
    """

    def __init__(
        self,
        start_equity: float = 100000.0,
        offset_min: float = 5.0,
        offset_max: float = 50.0,
        offset_unit: str = "bps",
        annualization_factor: Optional[float] = None,
        sizing_norm_mode: str = "train_distribution_absolute",
        max_size_cap: Optional[float] = None,
    ):
        self.sizing_mode = "linear"
        self.limit_model = Ridge(alpha=1.0)
        self.scaler = PredictionScaler()
        self.is_fitted = False
        self.start_equity = start_equity
        self.offset_min = offset_min
        self.offset_max = offset_max
        self.offset_unit = offset_unit
        self.annualization_factor = (
            annualization_factor if annualization_factor is not None else 1.0
        )
        self.sizing_norm_mode = sizing_norm_mode
        self.max_size_cap = max_size_cap
        self.normalizer_state_ = {}

        self.obj_a = 1.0  # Sortino
        self.obj_b = 1.0  # MaxDD
        self.obj_c = 1.0  # Instability

        self.layer_c_candidate_table_ = None

    def _score_candidates(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not results:
            return []
        z_pnl = _standardize([r["net_pnl_day"] for r in results])
        z_sortino = _standardize([r["sortino"] for r in results])
        z_maxdd = _standardize([r["maxDD"] for r in results])
        z_instab = _standardize([r["instability"] for r in results])

        for i, r in enumerate(results):
            u = (
                z_pnl[i]
                + self.obj_a * z_sortino[i]
                - self.obj_b * z_maxdd[i]
                - self.obj_c * z_instab[i]
            )
            r["utility"] = float(u)
        return sorted(results, key=lambda x: x["utility"], reverse=True)

    def optimize_sizing(
        self,
        scores: np.ndarray,
        returns: np.ndarray,
        threshold: float,
        timestamps: np.ndarray,
        base_size: float = 0.05,
    ):
        splits, actual_n_splits = make_temporal_splits(
            timestamps, n_samples=len(scores), n_splits=3
        )
        modes = ["linear", "convex", "concave"]
        mode_results = {
            m: {"fold_pnl_day": [], "fold_sortino": [], "fold_maxdd": []} for m in modes
        }

        diagnostics = []

        min_days_floor = 1.0 / 24.0  # Fix floor logic

        for fold_idx, (tr_idx, val_idx) in enumerate(splits):
            if len(val_idx) == 0:
                continue

            # 1. Fit normalizer strictly on the train fold
            tr_scores = scores[tr_idx]
            fold_normalizer = fit_sizing_normalizer(
                tr_scores,
                threshold,
                mode=self.sizing_norm_mode,
            )

            # 2. Evaluate on the validation fold
            s_val = scores[val_idx]
            r_val = returns[val_idx]
            ts_val = pd.to_datetime(timestamps[val_idx])

            active = s_val >= threshold

            # Record diagnostics
            fold_diag = {
                "fold": fold_idx,
                "sizing_norm_mode": self.sizing_norm_mode,
                "lower_anchor": fold_normalizer["lower_anchor"],
                "upper_anchor": fold_normalizer["upper_anchor"],
                "count_active_train": int(np.sum(tr_scores >= threshold)),
                "count_active_val": int(np.sum(active)),
            }

            if not np.any(active):
                fold_diag.update({
                    "normalization_fallback_triggered": True,
                    "active_score_min": 0.0,
                    "active_score_max": 0.0,
                    "active_score_std": 0.0,
                    "active_size_min": 0.0,
                    "active_size_max": 0.0,
                    "active_size_std": 0.0,
                    "top1_size_share": 0.0,
                    "top5_size_share": 0.0,
                    "base_size_fraction": 0.0,
                })
                diagnostics.append(fold_diag)
                for m in modes:
                    mode_results[m]["fold_pnl_day"].append(0.0)
                    mode_results[m]["fold_sortino"].append(0.0)
                    mode_results[m]["fold_maxdd"].append(1.0)
                continue

            s_act = s_val[active]
            r_act = r_val[active]
            days = max(
                (ts_val.max() - ts_val.min()).total_seconds() / 86400.0, min_days_floor
            )

            # Transform scores using fitted normalizer
            s_norm_act = transform_scores_to_sizing_input(
                s_act, fold_normalizer, fallback_threshold=threshold
            )

            fold_diag.update({
                "normalization_fallback_triggered": (fold_normalizer["upper_anchor"] <= fold_normalizer["lower_anchor"]),
                "active_score_min": float(s_act.min()),
                "active_score_max": float(s_act.max()),
                "active_score_std": float(np.std(s_act)),
            })

            # Calculate sizes for base/linear for diag
            sample_sizes = apply_sizing_curve(s_norm_act, base_size, "linear", self.max_size_cap)
            fold_diag.update({
                "active_size_min": float(sample_sizes.min()),
                "active_size_max": float(sample_sizes.max()),
                "active_size_std": float(np.std(sample_sizes)),
                "top1_size_share": float(np.max(sample_sizes) / np.sum(sample_sizes)),
                "top5_size_share": float(np.sum(np.sort(sample_sizes)[-5:]) / np.sum(sample_sizes)) if len(sample_sizes) >= 5 else 1.0,
                "base_size_fraction": float(np.mean(sample_sizes == base_size)),
            })
            diagnostics.append(fold_diag)

            for m in modes:
                sizes = apply_sizing_curve(s_norm_act, base_size, m, self.max_size_cap)
                pnl_series = self.start_equity * sizes * r_act
                pnl_total = float(np.sum(pnl_series))
                pnl_day = pnl_total / days

                neg = pnl_series[pnl_series < 0]
                dd_dev = float(np.sqrt(np.mean(neg**2))) if len(neg) > 0 else 1e-3
                sortino = float(
                    np.mean(pnl_series) / (dd_dev + 1e-9) * self.annualization_factor
                )

                _, dd = _stable_equity_and_drawdown(pnl_series)
                mdd = float(np.max(dd)) if len(dd) > 0 else 1.0

                mode_results[m]["fold_pnl_day"].append(pnl_day)
                mode_results[m]["fold_sortino"].append(sortino)
                mode_results[m]["fold_maxdd"].append(mdd)

        eval_rows = []
        for m in modes:
            mr = mode_results[m]
            eval_rows.append(
                {
                    "mode": m,
                    "net_pnl_day": float(np.mean(mr["fold_pnl_day"])),
                    "sortino": float(np.mean(mr["fold_sortino"])),
                    "maxDD": float(np.mean(mr["fold_maxdd"])),
                    "instability": float(np.std(mr["fold_pnl_day"])),
                    "threshold": threshold,
                    "base_size": base_size,
                }
            )

        scored = self._score_candidates(eval_rows)
        self.layer_c_candidate_table_ = pd.DataFrame(scored)
        self.sizing_mode = scored[0]["mode"]

        # Fit final normalizer state on ALL scores
        self.normalizer_state_ = fit_sizing_normalizer(
            scores, threshold, mode=self.sizing_norm_mode
        )
        self.diagnostics_ = pd.DataFrame(diagnostics)

        tprint(
            f"Layer C: Selected sizing mode '{self.sizing_mode}' with Utility={scored[0]['utility']:.4f} "
            f"(Norm={self.sizing_norm_mode}, Anchor={self.normalizer_state_.get('upper_anchor', 0):.4f})"
        )
        return self.sizing_mode

    def fit_limit_offset(
        self,
        feature_dict: Dict[str, np.ndarray],
        y_offset: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        cfg: Optional[Dict] = None,
    ):
        cfg = cfg or {}
        offset_mode = cfg.get("limit_offset_mode", "heuristic")
        if offset_mode != "ml":
            self.is_fitted = False
            self.offset_n_samples_ = 0
            return self

        target_mode = cfg.get("limit_offset_target_mode", "undefined")
        if target_mode == "undefined":
            raise ValueError(
                "limit_offset_mode='ml' requires a valid limit_offset_target_mode "
                "(e.g., 'utility_grid_search', 'simulated_fill_tradeoff'). "
                "Hindsight-biased max excursion targets are invalid."
            )

        X = assemble_feature_matrix(
            feature_dict, get_feature_view(POSITION_SIZER_V2_FEATURE_CONFIG.get("limit_offset_sizer", POSITION_SIZER_V2_FEATURE_CONFIG["shared_feature_keys"]), "X_linear")
        )
        valid = np.isfinite(y_offset)
        if not np.any(valid):
            self.is_fitted = False
            self.offset_n_samples_ = 0
            return self

        y_train = _soft_clip_offset(y_offset[valid], self.offset_min, self.offset_max, softness=0.3)
        X_train = X[valid]
        w_train = sample_weight[valid] if sample_weight is not None else None

        X_scaled = self.scaler.fit_transform(X_train)
        self.limit_model.fit(X_scaled, y_train, sample_weight=w_train)
        self.is_fitted = True
        self.offset_n_samples_ = len(y_train)
        return self

    def predict_size_and_offset(
        self,
        feature_dict: Dict[str, np.ndarray],
        score: np.ndarray,
        threshold: float = 0.0,
        base_size: float = 0.05,
        cfg: Optional[Dict] = None,
    ):
        cfg = cfg or {}
        X = assemble_feature_matrix(
            feature_dict, get_feature_view(POSITION_SIZER_V2_FEATURE_CONFIG.get("limit_offset_sizer", POSITION_SIZER_V2_FEATURE_CONFIG["shared_feature_keys"]), "X_linear")
        )

        offset_mode = cfg.get("limit_offset_mode", "heuristic")
        offset = np.zeros_like(score)

        if offset_mode == "ml" and self.is_fitted:
            offset = self.limit_model.predict(self.scaler.transform(X))
            offset = _soft_clip_offset(offset, self.offset_min, self.offset_max, softness=0.3)
        elif offset_mode == "heuristic" and "entry_offset_bps" in cfg:
            offset = np.full_like(score, cfg["entry_offset_bps"])

        sizes = np.zeros_like(score)
        active = score >= threshold

        if np.any(active):
            s_act = score[active]
            s_norm = transform_scores_to_sizing_input(
                s_act, self.normalizer_state_, fallback_threshold=threshold
            )
            sizes[active] = apply_sizing_curve(s_norm, base_size, self.sizing_mode, self.max_size_cap)

        return np.clip(sizes, 0.0, 1.0), offset


def run_experiment_comparison(
    baseline_returns: np.ndarray,
    baseline_sizes: np.ndarray,
    v2_sizes_no_offset: np.ndarray,
    v2_sizes_with_offset: np.ndarray,
    v2_returns_no_offset: np.ndarray,
    v2_returns_with_offset: np.ndarray,
    start_equity: float = 100000.0,
    timestamps: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Produce final comparison report between Original setup,
    V2 (No offset), and V2 (With limit offset).
    """
    tprint("=" * 80)
    tprint("FINAL EXPERIMENT COMPARISON")
    tprint("=" * 80)

    rows = []

    def _eval(sz, rets, name):
        pnl_series = start_equity * sz * rets
        pnl = np.sum(pnl_series)
        neg = pnl_series[pnl_series < 0]
        pos = pnl_series[pnl_series > 0]

        dd_dev = float(np.sqrt(np.mean(neg**2))) if len(neg) > 0 else 1e-3
        sortino = float(
            np.mean(pnl_series) / (dd_dev + 1e-9)
        )  # Note: No annualization factor available here, default to basic sortino.

        _, dd = _stable_equity_and_drawdown(pnl_series)
        mdd = float(np.max(dd)) if len(dd) > 0 else 1.0

        hit_rate = float(np.mean(pnl_series > 0)) if len(pnl_series) > 0 else 0.0
        pf = (
            float(np.sum(pos) / abs(np.sum(neg)))
            if len(neg) > 0 and abs(np.sum(neg)) > 1e-9
            else float("inf")
            if len(pos) > 0
            else 0.0
        )

        avg_win = float(np.mean(pos)) if len(pos) > 0 else 0.0
        avg_loss = float(np.mean(neg)) if len(neg) > 0 else 0.0

        trades = np.sum(sz > 0)
        days = (
            max(
                1.0,
                (
                    pd.to_datetime(timestamps).max() - pd.to_datetime(timestamps).min()
                ).total_seconds()
                / 86400.0,
            )
            if timestamps is not None
            else 1.0
        )
        trades_per_day = float(trades / days)

        row = {
            "setup": name,
            "net_pnl": pnl,
            "sortino": sortino,
            "maxDD": mdd,
            "hit_rate": hit_rate,
            "profit_factor": pf,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "trades_per_day": trades_per_day,
        }
        rows.append(row)

        tprint(f"  {name}:")
        tprint(f"    Net PnL:       ${pnl:.2f}")
        tprint(f"    Sortino:       {sortino:.3f}")
        tprint(f"    Max DD:        {mdd:.4%}")
        tprint(f"    Hit Rate:      {hit_rate:.2%}")
        tprint(f"    Profit Factor: {pf:.3f}")
        tprint(f"    Trades/Day:    {trades_per_day:.2f}")
        tprint(f"    Avg Win/Loss:  ${avg_win:.2f} / ${avg_loss:.2f}")

    _eval(baseline_sizes, baseline_returns, "Baseline (Original)")
    _eval(
        v2_sizes_no_offset,
        v2_returns_no_offset,
        "V2 (Layer A + Layer B policy + best Sizing)",
    )
    _eval(
        v2_sizes_with_offset,
        v2_returns_with_offset,
        "V2 (Full) + Limit Offset Optimizer",
    )
    tprint("=" * 80)

    return pd.DataFrame(rows)


# Layer B reporting expansion
def generate_layer_b_deliverables(
    best_policy: dict,
    j_score: float,
    trade_outcomes: pd.DataFrame,
    start_equity: float = 100000.0,
):
    tprint("=" * 80)
    tprint("LAYER B DELIVERABLES: POLICY OPTIMIZATION")
    tprint("=" * 80)
    tprint("  Best Layer B Policy:")
    for k, v in best_policy.items():
        if isinstance(v, float):
            tprint(f"    {k}: {v:.4f}")
        else:
            tprint(f"    {k}: {v}")

    tprint(f"  Composite Objective (Utility): {j_score:.4f}")
    tprint("=" * 80)


# Target race expansion helper
def generate_target_race_report(
    target_candidates: Dict[str, np.ndarray],
    y_oos_dict: Dict[str, np.ndarray],
    base_returns: np.ndarray,
):
    tprint("=" * 80)
    tprint("MODEL 1 EDGE TARGET RACE DIAGNOSTICS")
    tprint("=" * 80)
    for name, pred in y_oos_dict.items():
        score = np.nan_to_num(pred)
        if np.all(score == 0):
            continue

        n_trades = len(score)
        k_10 = max(1, int(n_trades * 0.10))
        k_20 = max(1, int(n_trades * 0.20))

        idx_10 = np.argpartition(score, -k_10)[-k_10:]
        idx_20 = np.argpartition(score, -k_20)[-k_20:]

        ret_10 = base_returns[idx_10]
        ret_20 = base_returns[idx_20]

        win_10 = np.mean(ret_10 > 0)
        win_20 = np.mean(ret_20 > 0)

        pnl_10 = np.sum(ret_10)
        pnl_20 = np.sum(ret_20)

        spear, _ = spearmanr(score, base_returns, nan_policy="omit")

        tprint(f"  Target: {name}")
        tprint(f"    Spearman IC:     {spear:.4f}")
        tprint(f"    Top 10% WinRate: {win_10:.2%}")
        tprint(f"    Top 10% Net PnL: {pnl_10:.4f}")
        tprint(f"    Top 20% Net PnL: {pnl_20:.4f}")

    tprint("=" * 80)


# ============================================================
# Bucketed Orchestrator
# ============================================================


def run_bucketed_position_sizer_v2(
    feature_dict: Dict[str, np.ndarray],
    trade_outcomes: pd.DataFrame,
    y_raw_net_return: np.ndarray,
    y_downside: np.ndarray,
    bucket_labels: np.ndarray,
    timestamps: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    lambda_downside: float = 0.5,
    eta_uncertainty: float = 0.5,
    cost_pct: float = 0.002,
    start_equity: float = 100000.0,
) -> Dict[str, Any]:
    from extreme_price_movements.config import (
        POSITION_SIZER_V2_BUCKET_CONFIG,
        POSITION_SIZER_V2_BUCKETS,
    )

    results = {}
    summary_rows = []

    min_samples = POSITION_SIZER_V2_BUCKET_CONFIG.get("min_samples_total", 500)

    for bucket in POSITION_SIZER_V2_BUCKETS:
        tprint("=" * 80)
        tprint(f"PROCESSING BUCKET: {bucket}")
        tprint("=" * 80)

        mask = bucket_labels == bucket
        n_bucket = int(np.sum(mask))

        if n_bucket < min_samples:
            tprint(
                f"  Skipping bucket {bucket} due to insufficient samples ({n_bucket} < {min_samples})"
            )
            summary_rows.append(
                {
                    "bucket": bucket,
                    "n_samples": n_bucket,
                    "status": "skipped_insufficient_samples",
                    "model1_n_features": 0,
                    "model2_n_features": 0,
                    "model3_n_features": 0,
                    "model1_feature_stability": 0.0,
                    "model2_feature_stability": 0.0,
                    "model3_feature_stability": 0.0,
                    "final_policy_utility": 0.0,
                    "final_sizing_mode": "none",
                }
            )
            continue

        def _slice_dict(d, m):
            return {k: v[m] for k, v in d.items()}

        fd_bucket = _slice_dict(feature_dict, mask)
        outcomes_bucket = trade_outcomes.iloc[mask].reset_index(drop=True)
        y_ret_bucket = y_raw_net_return[mask]
        y_down_bucket = y_downside[mask]
        ts_bucket = timestamps[mask]
        sw_bucket = sample_weight[mask] if sample_weight is not None else None

        # --- Layer A ---
        from extreme_price_movements.config import CFG
        predictor = LayerAPredictor(
            lambda_downside=lambda_downside, eta_uncertainty=eta_uncertainty, config=CFG
        )
        predictor.fit(
            feature_dict=fd_bucket,
            trade_outcomes=outcomes_bucket,
            y_raw_net_return=y_ret_bucket,
            y_downside=y_down_bucket,
            timestamps=ts_bucket,
            sample_weight=sw_bucket,
        )

        scores = predictor.predict_score(fd_bucket)

        # Extract stability stats
        fs1 = predictor.feature_selection_results_edge_
        fs2 = predictor.feature_selection_results_downside_
        fs3 = predictor.feature_selection_results_uncertainty_

        # --- Layer B ---
        b_opt = LayerBPolicyOptimizer(cost_pct=cost_pct)
        best_policy = b_opt.optimize(scores, outcomes_bucket, ts_bucket)

        # --- Layer C ---
        c_opt = LayerCExecutionOptimizer(start_equity=start_equity)
        sizing_mode = c_opt.optimize_sizing(
            scores,
            y_ret_bucket,
            threshold=best_policy["score_threshold"],
            timestamps=ts_bucket,
        )

        # For full implementation, limit offset regression requires its target.
        # Left as placeholder fit for orchestrator architecture mapping.
        c_opt.sizing_mode = sizing_mode

        summary_rows.append(
            {
                "bucket": bucket,
                "n_samples": n_bucket,
                "status": "success",
                "model1_n_features": len(predictor.final_selected_feature_idx_edge_)
                if fs1
                else 0,
                "model2_n_features": len(predictor.final_selected_feature_idx_downside_)
                if fs2
                else 0,
                "model3_n_features": len(
                    predictor.final_selected_feature_idx_uncertainty_
                )
                if fs3
                else 0,
                "model1_feature_stability": float(fs1.get("stability_score", 0.0))
                if isinstance(fs1, dict)
                else 0.0,
                "model2_feature_stability": float(fs2.get("stability_score", 0.0))
                if isinstance(fs2, dict)
                else 0.0,
                "model3_feature_stability": float(fs3.get("stability_score", 0.0))
                if isinstance(fs3, dict)
                else 0.0,
                "model1_type": predictor.edge_model.model_type_,
                "model2_type": predictor.downside_model.model_type_,
                "model3_type": predictor.uncertainty_model.model_type_,
                "model3_target_transform": getattr(
                    predictor.uncertainty_model, "target_transform_", None
                ),
                "actual_n_splits_used": predictor.actual_n_splits_used_,
                "final_policy_utility": float(b_opt.best_j),
                "final_sizing_mode": str(sizing_mode),
            }
        )

        results[bucket] = {
            "layer_a": predictor,
            "layer_b": b_opt,
            "layer_c": c_opt,
            "best_policy": best_policy,
            "scores": scores,
            "feature_selection_results_edge": fs1,
            "feature_selection_results_downside": fs2,
            "feature_selection_results_uncertainty": fs3,
            "final_selected_feature_names_edge": fs1.get("selected_names", [])
            if isinstance(fs1, dict)
            else [],
            "final_selected_feature_names_downside": fs2.get("selected_names", [])
            if isinstance(fs2, dict)
            else [],
            "final_selected_feature_names_uncertainty": fs3.get("selected_names", [])
            if isinstance(fs3, dict)
            else [],
        }

    summary_df = pd.DataFrame(summary_rows)
    return {"bucket_results": results, "bucket_summary_table_": summary_df}
