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

import logging
import itertools
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

from extreme_price_movements.purged_cv import PurgedKFold
from extreme_price_movements.metrics import _stable_equity_and_drawdown
from extreme_price_movements.utils import tprint
from extreme_price_movements.label_policy_optimizer import LabelPolicy, _simulate_policy_batch
from extreme_price_movements.position_sizer_v2_metrics import (
    compute_top_slice_metrics,
    compute_bucket_monotonicity,
    compute_false_safe_rate,
    compute_uncertainty_calibration
)

# ============================================================
# Shared Configuration & Utilities
# ============================================================
logger = logging.getLogger(__name__)

class PredictionScaler:
    """Scales predictions handling NaNs explicitly."""
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.fit_transform(np.nan_to_num(X))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(np.nan_to_num(X))

def make_temporal_splits(
    timestamps: Optional[np.ndarray],
    n_samples: int,
    n_splits: int = 3,
    purge_units: int = 43200,
    embargo_units: int = 43200
) -> List[Tuple[np.ndarray, np.ndarray]]:
    cv = PurgedKFold(n_splits=n_splits, purge=purge_units, embargo=embargo_units, times=timestamps)
    dummy_X = np.empty((n_samples, 1))
    return list(cv.split(dummy_X))

def build_log_clipped_target(returns: np.ndarray, clip_L: float = 0.02) -> np.ndarray:
    """T1 = log_clipped_winsorized_net"""
    clipped = np.clip(returns, -clip_L, clip_L)
    scale = np.std(clipped) if len(clipped) > 1 and np.std(clipped) > 1e-9 else 1.0
    return np.sign(clipped) * np.log1p(np.abs(clipped) / scale)

def build_rank_target(returns: np.ndarray, timestamps: Optional[np.ndarray] = None, mode: str = "fold_local") -> np.ndarray:
    """T2 = rank_style_target"""
    returns = np.asarray(returns)
    if mode == "per_timestamp" and timestamps is not None:
        ranks = np.empty_like(returns, dtype=float)
        df = pd.DataFrame({"ret": returns, "ts": timestamps})
        for ts, group in df.groupby("ts"):
            idx = group.index
            local_rets = group["ret"].values
            order = np.argsort(local_rets)
            local_ranks = np.empty_like(local_rets, dtype=float)
            local_ranks[order] = (np.arange(len(local_rets)) + 0.5) / max(1, len(local_rets))
            ranks[idx] = local_ranks
        return (ranks * 2.0) - 1.0
    elif mode == "fold_local":
        order = np.argsort(returns)
        ranks = np.empty_like(returns, dtype=float)
        ranks[order] = (np.arange(len(returns)) + 0.5) / max(1, len(returns))
        return (ranks * 2.0) - 1.0
    else:
        raise ValueError(f"Unknown or unsupported rank target mode: {mode}")

def build_robust_utility_target(
    trade_outcomes: pd.DataFrame,
    cost_pct: float = 0.002
) -> np.ndarray:
    """
    T3 = robust_utility_target_simple_tbm
    Fixed simple TBM geometry: SL in {1.0, 1.5} ATR, TP in {1.5, 2.0} * SL
    No trailing, no early exit.
    """
    # Defensive path extraction
    max_b = 24

    n = len(trade_outcomes)
    def _pad(series):
        res = np.full((n, max_b), np.nan, dtype=np.float64)
        lens = np.zeros(n, dtype=np.int64)
        for i, arr in enumerate(series):
            use = min(len(arr), max_b)
            if use > 0:
                res[i, :use] = arr[:use]
                lens[i] = use
        return res, lens

    opens, _ = _pad(trade_outcomes["future_opens"].values)
    highs, _ = _pad(trade_outcomes["future_highs"].values)
    lows, _ = _pad(trade_outcomes["future_lows"].values)
    closes, path_lens = _pad(trade_outcomes["future_closes"].values)

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
            trail_activate_atr=1e9, # disabled
            giveback_pct=0.0,
            early_exit_deadline_bars=0,
            early_exit_mfe_atr=0.0
        )
        u, _ = _simulate_policy_batch(
            entry_px, atr, is_long, opens, highs, lows, closes, path_lens, pol, cost_pct
        )
        u_acc += u

    # Mean utility across the fixed simple geometries
    return u_acc / len(geometries)

def build_volatility_normalized_target(returns: np.ndarray, atr_values: np.ndarray, entry_prices: np.ndarray) -> np.ndarray:
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
        self.model = HuberRegressor(epsilon=1.35, max_iter=2000)
        self.scaler = PredictionScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        X_scaled = self.scaler.fit_transform(X)
        try:
            self.model.fit(X_scaled, y, sample_weight=sample_weight)
        except Exception:
            self.model = Ridge(alpha=1.0)
            self.model.fit(X_scaled, y, sample_weight=sample_weight)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model1Edge not fitted")
        return self.model.predict(self.scaler.transform(X))


class Model2Downside:
    def __init__(self):
        self.model = HuberRegressor(epsilon=1.35, max_iter=2000)
        self.scaler = PredictionScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray, y_mae_atr: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        X_scaled = self.scaler.fit_transform(X)
        try:
            self.model.fit(X_scaled, y_mae_atr, sample_weight=sample_weight)
        except Exception:
            self.model = Ridge(alpha=1.0)
            self.model.fit(X_scaled, y_mae_atr, sample_weight=sample_weight)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model2Downside not fitted")
        return self.model.predict(self.scaler.transform(X))


class Model3Uncertainty:
    def __init__(self):
        self.model = HuberRegressor(epsilon=1.35, max_iter=2000)
        self.scaler = PredictionScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray, residuals: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        y_target = np.log1p(np.abs(residuals))
        X_scaled = self.scaler.fit_transform(X)
        try:
            self.model.fit(X_scaled, y_target, sample_weight=sample_weight)
        except Exception:
            self.model = Ridge(alpha=1.0)
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
    def __init__(self, lambda_downside: float = 0.5, eta_uncertainty: float = 0.5):
        self.edge_model = Model1Edge()
        self.downside_model = Model2Downside()
        self.uncertainty_model = Model3Uncertainty()

        self.lambda_downside = lambda_downside
        self.eta_uncertainty = eta_uncertainty
        self.is_fitted = False

        # Artifacts
        self.model1_target_race_results_ = None
        self.model1_best_target_name_ = None
        self.model1_oof_pred_ = None

        self.model2_eval_results_ = None
        self.model2_oof_pred_ = None

        self.model3_eval_results_ = None
        self.model3_oof_pred_ = None

    def _run_model1_target_race(
        self,
        X: np.ndarray,
        trade_outcomes: pd.DataFrame,
        raw_returns: np.ndarray,
        timestamps: Optional[np.ndarray],
        sample_weight: Optional[np.ndarray]
    ):
        """Evaluates T1, T2, T3, T4. Picks winner via temporal OOF edge target score."""
        splits = make_temporal_splits(timestamps, n_samples=len(X), n_splits=3)

        candidates = {
            "log_clipped_winsorized_net": build_log_clipped_target(raw_returns),
            "rank_style_target": build_rank_target(
                raw_returns,
                timestamps,
                mode="per_timestamp" if timestamps is not None else "fold_local"
            ),
            "robust_utility_target": build_robust_utility_target(trade_outcomes),
            "volatility_normalized_target": build_volatility_normalized_target(
                raw_returns,
                trade_outcomes["atr_12_15m"].values,
                trade_outcomes["entry_price"].values
            )
        }

        race_results = []
        best_score = -1e9
        best_name = "log_clipped_winsorized_net"
        best_oof = np.zeros(len(X))

        for name, y_cand in candidates.items():
            oof_preds = np.full(len(X), np.nan)
            fold_nets_10 = []
            fold_nets_20 = []

            for tr_idx, val_idx in splits:
                m = Model1Edge(target_name=name)
                # fold local for rank
                if name == "rank_style_target" and timestamps is None:
                    y_tr = build_rank_target(raw_returns[tr_idx], mode="fold_local")
                else:
                    y_tr = y_cand[tr_idx]

                w_tr = sample_weight[tr_idx] if sample_weight is not None else None
                m.fit(X[tr_idx], y_tr, w_tr)

                preds_val = m.predict(X[val_idx])
                oof_preds[val_idx] = preds_val

                top_mets = compute_top_slice_metrics(preds_val, raw_returns[val_idx], top_fracs=(0.1, 0.2))
                fold_nets_10.append(top_mets["top_10_mean_net"])
                fold_nets_20.append(top_mets["top_20_mean_net"])

            mean_net = float(np.mean(fold_nets_10)) if fold_nets_10 else 0.0
            std_net = float(np.std(fold_nets_10)) if len(fold_nets_10) > 1 else 0.0
            score = mean_net - 0.5 * std_net

            spear, _ = spearmanr(oof_preds[np.isfinite(oof_preds)], raw_returns[np.isfinite(oof_preds)], nan_policy="omit")
            monot = compute_bucket_monotonicity(oof_preds[np.isfinite(oof_preds)], raw_returns[np.isfinite(oof_preds)])

            full_top = compute_top_slice_metrics(oof_preds[np.isfinite(oof_preds)], raw_returns[np.isfinite(oof_preds)], (0.1,))

            race_results.append({
                "target": name,
                "score": score,
                "spearman_ic_mean": float(spear) if pd.notna(spear) else 0.0,
                "top_decile_realized_net_mean": mean_net,
                "top_quintile_realized_net_mean": float(np.mean(fold_nets_20)) if fold_nets_20 else 0.0,
                "top_decile_hit_rate_mean": full_top["top_10_hit_rate"],
                "score_bucket_monotonicity": monot,
                "fold_stability_std": std_net
            })

            if score > best_score:
                best_score = score
                best_name = name
                best_oof = oof_preds

        self.model1_target_race_results_ = pd.DataFrame(race_results)
        self.model1_best_target_name_ = best_name
        self.model1_oof_pred_ = best_oof

        # Fit final Edge
        self.edge_model = Model1Edge(target_name=best_name)
        if best_name == "rank_style_target" and timestamps is None:
            y_final = build_rank_target(raw_returns, mode="fold_local")
        else:
            y_final = candidates[best_name]
        self.edge_model.fit(X, y_final, sample_weight)

    def _run_model2_oof_eval(
        self,
        X: np.ndarray,
        y_downside: np.ndarray,
        timestamps: Optional[np.ndarray],
        sample_weight: Optional[np.ndarray]
    ):
        """OOF eval for Model 2 Downside."""
        splits = make_temporal_splits(timestamps, n_samples=len(X), n_splits=3)
        oof_preds = np.full(len(X), np.nan)

        for tr_idx, val_idx in splits:
            m = Model2Downside()
            w_tr = sample_weight[tr_idx] if sample_weight is not None else None
            # Robustness: winzorize extreme tails in downside target before fit
            y_tr_down = np.clip(y_downside[tr_idx], 0.0, np.percentile(y_downside[tr_idx], 98))
            m.fit(X[tr_idx], y_tr_down, w_tr)
            oof_preds[val_idx] = m.predict(X[val_idx])

        valid = np.isfinite(oof_preds)
        if np.any(valid):
            mae = float(np.mean(np.abs(oof_preds[valid] - y_downside[valid])))
            monot = compute_bucket_monotonicity(oof_preds[valid], y_downside[valid])
            fsr = compute_false_safe_rate(oof_preds[valid], y_downside[valid], low_q=0.2, high_q=0.8)

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
            "false_safe_rate": fsr
        }
        self.model2_oof_pred_ = oof_preds

        # Fit final Downside model
        y_final = np.clip(y_downside, 0.0, np.percentile(y_downside, 98))
        self.downside_model.fit(X, y_final, sample_weight)

    def _run_model3_oof_eval(
        self,
        X: np.ndarray,
        raw_returns: np.ndarray,
        timestamps: Optional[np.ndarray],
        sample_weight: Optional[np.ndarray]
    ):
        """OOF eval for Uncertainty. Fits on Model 1 OOF residuals."""
        valid_oof = np.isfinite(self.model1_oof_pred_)
        if not np.any(valid_oof):
            return

        y_true_for_res = raw_returns[valid_oof]
        residuals = y_true_for_res - self.model1_oof_pred_[valid_oof]
        X_res = X[valid_oof]
        w_res = sample_weight[valid_oof] if sample_weight is not None else None

        splits = make_temporal_splits(timestamps[valid_oof] if timestamps is not None else None, n_samples=len(X_res), n_splits=3)
        oof_preds = np.full(len(X_res), np.nan)

        for tr_idx, val_idx in splits:
            m = Model3Uncertainty()
            w_tr = w_res[tr_idx] if w_res is not None else None
            m.fit(X_res[tr_idx], residuals[tr_idx], w_tr)
            oof_preds[val_idx] = m.predict(X_res[val_idx])

        valid2 = np.isfinite(oof_preds)
        realized_abs = np.abs(residuals)

        if np.any(valid2):
            calib = compute_uncertainty_calibration(oof_preds[valid2], realized_abs[valid2])
        else:
            calib = {}

        self.model3_eval_results_ = calib
        self.model3_oof_pred_ = oof_preds

        # Fit final Uncertainty model
        self.uncertainty_model.fit(X_res, residuals, w_res)

    def fit(self, X: np.ndarray, trade_outcomes: pd.DataFrame, y_raw_net_return: np.ndarray, y_downside: np.ndarray,
            timestamps: Optional[np.ndarray] = None,
            sample_weight: Optional[np.ndarray] = None):

        # Bound/clip sample weights to prevent explosive leverage
        if sample_weight is not None:
            sample_weight = np.clip(sample_weight, 0.01, np.percentile(sample_weight, 99))

        # Orchestration steps as required:
        # a) run Model 1 target race with OOF preds, c) fits final model 1
        self._run_model1_target_race(X, trade_outcomes, y_raw_net_return, timestamps, sample_weight)

        # b) run Model 2 OOF eval, d) fits final model 2
        self._run_model2_oof_eval(X, y_downside, timestamps, sample_weight)

        # e) fit final Model 3 on OOF residual target (includes its own OOF eval)
        self._run_model3_oof_eval(X, y_raw_net_return, timestamps, sample_weight)

        self.is_fitted = True
        return self

    def predict_components(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("LayerAPredictor not fitted")
        return {
            "edge": self.edge_model.predict(X),
            "downside": self.downside_model.predict(X),
            "uncertainty": self.uncertainty_model.predict(X)
        }

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        comps = self.predict_components(X)
        return comps["edge"] - (self.lambda_downside * comps["downside"]) - (self.eta_uncertainty * comps["uncertainty"])

    def initial_sizing(self, score: np.ndarray, threshold: float = 0.0, base_size: float = 0.05) -> np.ndarray:
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
    """
    def __init__(self, cost_pct: float = 0.002, lambda_penalty: float = 0.5):
        self.cost_pct = cost_pct
        self.lambda_penalty = lambda_penalty
        self.best_policy = {}
        self.best_j = -1e9

        self.obj_a = 1.0 # Sortino weight
        self.obj_b = 1.0 # MaxDD weight
        self.obj_c = 1.0 # Instability weight

        self.layer_b_candidate_table_ = None
        self.layer_b_selected_objective_components_ = None

    def _eval_policy_over_folds(
        self,
        pol: LabelPolicy,
        trade_outcomes: pd.DataFrame,
        scores: np.ndarray,
        threshold_q: float,
        splits: List[Tuple[np.ndarray, np.ndarray]],
        timestamps: np.ndarray
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

        min_days_floor = 1.0 / 24.0 # Fix: lower floor for short intraday evaluation bounds

        for _, val_idx in splits:
            if len(val_idx) == 0:
                continue

            scores_val = scores[val_idx]
            fold_thresh = np.percentile(scores_val, threshold_q * 100)
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

            def _pad(series, max_len):
                res = np.full((len(series), max_len), np.nan, dtype=np.float64)
                lens = np.zeros(len(series), dtype=np.int64)
                for i, arr in enumerate(series):
                    use = min(len(arr), max_len)
                    if use > 0:
                        res[i, :use] = arr[:use]
                        lens[i] = use
                return res, lens

            opens_2d, _ = _pad(trade_outcomes["future_opens"].values[active_val_idx], max_b)
            highs_2d, _ = _pad(trade_outcomes["future_highs"].values[active_val_idx], max_b)
            lows_2d, _ = _pad(trade_outcomes["future_lows"].values[active_val_idx], max_b)
            closes_2d, path_lens = _pad(trade_outcomes["future_closes"].values[active_val_idx], max_b)

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
                cost_pct=self.cost_pct
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
            dd_dev = float(np.sqrt(np.mean(neg_rets**2))) if len(neg_rets) > 0 else 1e-3
            sortino = float(np.mean(ret) / (dd_dev + 1e-9) * np.sqrt(365))

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
            "timeout_rate": float(np.mean(fold_timeout_rates))
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
            u = z_pnl[i] + self.obj_a * z_sortino[i] - self.obj_b * z_maxdd[i] - self.obj_c * z_instab[i]
            r["utility"] = float(u)

        return sorted(results, key=lambda x: x["utility"], reverse=True)

    def optimize(self, scores: np.ndarray, trade_outcomes: pd.DataFrame, timestamps: np.ndarray):
        splits = make_temporal_splits(timestamps, n_samples=len(scores), n_splits=3)
        candidate_table = []

        # --- Step 1: Exit Geometry (SL, Trailing, Giveback) ---
        sl_cands = [1.0, 1.5, 2.0]
        tp_cands = [1.5, 2.0, 2.5]
        trail_cands = [0.8, 1.0, 1.5]
        giveback_cands = [0.35, 0.50]

        tprint("Layer B Step 1: Optimizing exit geometry over temporal folds...")
        step1_results = []
        for sl, tp, trail, gb in itertools.product(sl_cands, tp_cands, trail_cands, giveback_cands):
            pol = LabelPolicy(
                sl_atr_mult=sl, tp_sl_ratio=tp, max_hold_bars=24,
                trail_activate_atr=trail, giveback_pct=gb,
                early_exit_deadline_bars=0, early_exit_mfe_atr=0.0
            )
            res = self._eval_policy_over_folds(pol, trade_outcomes, scores, threshold_q=0.60, splits=splits, timestamps=timestamps)
            res["step"] = 1
            res["policy_obj"] = pol
            res["threshold_q"] = 0.60
            step1_results.append(res)

        step1_scored = self._score_candidates(step1_results)
        candidate_table.extend(step1_scored)
        best_geom = step1_scored[0]["policy_obj"]

        # --- Step 2: Selection Boundary ---
        tprint("Layer B Step 2: Optimizing selection boundary...")
        step2_results = []
        for q in [0.70, 0.60, 0.50, 0.30]:
            res = self._eval_policy_over_folds(best_geom, trade_outcomes, scores, threshold_q=q, splits=splits, timestamps=timestamps)
            res["step"] = 2
            res["policy_obj"] = best_geom
            res["threshold_q"] = q
            step2_results.append(res)

        step2_scored = self._score_candidates(step2_results)
        candidate_table.extend(step2_scored)
        best_q = step2_scored[0]["threshold_q"]

        # --- Step 3: Time Management ---
        tprint("Layer B Step 3: Optimizing time management...")
        timeout_cands = [16, 24, 32]
        early_deadline_cands = [0, 8, 12]
        early_mfe_cands = [0.25, 0.50]

        step3_results = []
        for t_out, ed, em in itertools.product(timeout_cands, early_deadline_cands, early_mfe_cands):
            if ed == 0: em = 0.0
            pol = LabelPolicy(
                sl_atr_mult=best_geom.sl_atr_mult, tp_sl_ratio=best_geom.tp_sl_ratio,
                max_hold_bars=t_out, trail_activate_atr=best_geom.trail_activate_atr,
                giveback_pct=best_geom.giveback_pct, early_exit_deadline_bars=ed, early_exit_mfe_atr=em
            )
            res = self._eval_policy_over_folds(pol, trade_outcomes, scores, threshold_q=best_q, splits=splits, timestamps=timestamps)
            res["step"] = 3
            res["policy_obj"] = pol
            res["threshold_q"] = best_q
            step3_results.append(res)

        step3_scored = self._score_candidates(step3_results)
        candidate_table.extend(step3_scored)
        best_pol = step3_scored[0]["policy_obj"]

        # Artifact storage ensuring fully reconstructable records
        self.layer_b_candidate_table_ = pd.DataFrame([{
            "step": r["step"],
            "threshold_q": r["threshold_q"],
            "sl_atr_mult": r["policy_obj"].sl_atr_mult,
            "tp_sl_ratio": r["policy_obj"].tp_sl_ratio,
            "trail_activate_atr": r["policy_obj"].trail_activate_atr,
            "giveback_pct": r["policy_obj"].giveback_pct,
            "max_hold_bars": r["policy_obj"].max_hold_bars,
            "early_exit_deadline_bars": r["policy_obj"].early_exit_deadline_bars,
            "early_exit_mfe_atr": r["policy_obj"].early_exit_mfe_atr,
            "net_pnl_day": r["net_pnl_day"],
            "sortino": r["sortino"],
            "maxDD": r["maxDD"],
            "instability": r["instability"],
            "utility": r["utility"]
        } for r in candidate_table])

        self.layer_b_selected_objective_components_ = step3_scored[0]
        self.best_j = step3_scored[0]["utility"]

        final_thresh_val = np.percentile(scores, best_q * 100)
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

class LayerCExecutionOptimizer:
    """
    Layer C: Sizing mapping and Ridge Limit offset optimizer.
    Limit offset semantic bounds: 0.0 to 5.0 explicitly in percentage points or ticks based on input.
    """
    def __init__(self, start_equity: float = 100000.0,
                 offset_min: float = 0.0, offset_max: float = 5.0, offset_unit: str = "ticks"):
        self.sizing_mode = "linear"
        self.limit_model = Ridge(alpha=1.0)
        self.scaler = PredictionScaler()
        self.is_fitted = False
        self.start_equity = start_equity
        self.offset_min = offset_min
        self.offset_max = offset_max
        self.offset_unit = offset_unit

        self.obj_a = 1.0 # Sortino
        self.obj_b = 1.0 # MaxDD
        self.obj_c = 1.0 # Instability

        self.layer_c_candidate_table_ = None

    def _score_candidates(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not results:
            return []
        z_pnl = _standardize([r["net_pnl_day"] for r in results])
        z_sortino = _standardize([r["sortino"] for r in results])
        z_maxdd = _standardize([r["maxDD"] for r in results])
        z_instab = _standardize([r["instability"] for r in results])

        for i, r in enumerate(results):
            u = z_pnl[i] + self.obj_a * z_sortino[i] - self.obj_b * z_maxdd[i] - self.obj_c * z_instab[i]
            r["utility"] = float(u)
        return sorted(results, key=lambda x: x["utility"], reverse=True)

    def optimize_sizing(
        self,
        scores: np.ndarray,
        returns: np.ndarray,
        threshold: float,
        timestamps: np.ndarray,
        base_size: float = 0.05
    ):
        splits = make_temporal_splits(timestamps, n_samples=len(scores), n_splits=3)
        modes = ["linear", "convex", "concave"]
        mode_results = {m: {"fold_pnl_day": [], "fold_sortino": [], "fold_maxdd": []} for m in modes}

        min_days_floor = 1.0 / 24.0 # Fix floor logic

        for _, val_idx in splits:
            if len(val_idx) == 0:
                continue

            s_val = scores[val_idx]
            r_val = returns[val_idx]
            ts_val = pd.to_datetime(timestamps[val_idx])

            active = s_val >= threshold
            if not np.any(active):
                for m in modes:
                    mode_results[m]["fold_pnl_day"].append(0.0)
                    mode_results[m]["fold_sortino"].append(0.0)
                    mode_results[m]["fold_maxdd"].append(1.0)
                continue

            s_act = s_val[active]
            r_act = r_val[active]
            days = max((ts_val.max() - ts_val.min()).total_seconds() / 86400.0, min_days_floor)

            s_min, s_max = s_act.min(), s_act.max()
            s_norm = (s_act - s_min) / max(s_max - s_min, 1e-9)

            candidates = {
                "linear": base_size * (1.0 + s_norm),
                "convex": base_size * (1.0 + s_norm**2),
                "concave": base_size * (1.0 + np.sqrt(s_norm))
            }

            for m in modes:
                sizes = np.clip(candidates[m], 0.0, 1.0)
                pnl_series = self.start_equity * sizes * r_act
                pnl_total = float(np.sum(pnl_series))
                pnl_day = pnl_total / days

                neg = pnl_series[pnl_series < 0]
                dd_dev = float(np.sqrt(np.mean(neg**2))) if len(neg) > 0 else 1e-3
                sortino = float(np.mean(pnl_series) / (dd_dev + 1e-9) * np.sqrt(365))

                _, dd = _stable_equity_and_drawdown(pnl_series)
                mdd = float(np.max(dd)) if len(dd) > 0 else 1.0

                mode_results[m]["fold_pnl_day"].append(pnl_day)
                mode_results[m]["fold_sortino"].append(sortino)
                mode_results[m]["fold_maxdd"].append(mdd)

        eval_rows = []
        for m in modes:
            mr = mode_results[m]
            eval_rows.append({
                "mode": m,
                "net_pnl_day": float(np.mean(mr["fold_pnl_day"])),
                "sortino": float(np.mean(mr["fold_sortino"])),
                "maxDD": float(np.mean(mr["fold_maxdd"])),
                "instability": float(np.std(mr["fold_pnl_day"])),
                "threshold": threshold,
                "base_size": base_size
            })

        scored = self._score_candidates(eval_rows)
        self.layer_c_candidate_table_ = pd.DataFrame(scored)
        self.sizing_mode = scored[0]["mode"]

        tprint(f"Layer C: Selected sizing mode '{self.sizing_mode}' with Utility={scored[0]['utility']:.4f}")
        return self.sizing_mode

    def fit_limit_offset(self, X: np.ndarray, y_offset: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        X_scaled = self.scaler.fit_transform(X)
        self.limit_model.fit(X_scaled, y_offset, sample_weight=sample_weight)
        self.is_fitted = True
        return self

    def predict_size_and_offset(self, X: np.ndarray, score: np.ndarray, threshold: float = 0.0, base_size: float = 0.05):
        offset = np.zeros_like(score)
        if self.is_fitted:
            offset = self.limit_model.predict(self.scaler.transform(X))
            offset = np.clip(offset, self.offset_min, self.offset_max)

        sizes = np.zeros_like(score)
        active = score >= threshold
        if np.any(active):
            s_val = score[active]
            s_min, s_max = s_val.min(), s_val.max()
            if s_max > s_min:
                s_norm = (s_val - s_min) / (s_max - s_min)
                if self.sizing_mode == "linear":
                    sizes[active] = base_size * (1.0 + s_norm)
                elif self.sizing_mode == "concave":
                    sizes[active] = base_size * (1.0 + np.sqrt(s_norm))
                elif self.sizing_mode == "convex":
                    sizes[active] = base_size * (1.0 + s_norm**2)
            else:
                sizes[active] = base_size

        return np.clip(sizes, 0.0, 1.0), offset


def run_experiment_comparison(
    baseline_returns: np.ndarray,
    baseline_sizes: np.ndarray,
    v2_sizes_no_offset: np.ndarray,
    v2_sizes_with_offset: np.ndarray,
    v2_returns_no_offset: np.ndarray,
    v2_returns_with_offset: np.ndarray,
    start_equity: float = 100000.0,
    timestamps: Optional[np.ndarray] = None
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
        sortino = float(np.mean(pnl_series) / (dd_dev + 1e-9) * np.sqrt(365))

        _, dd = _stable_equity_and_drawdown(pnl_series)
        mdd = float(np.max(dd)) if len(dd) > 0 else 1.0

        hit_rate = float(np.mean(pnl_series > 0)) if len(pnl_series) > 0 else 0.0
        pf = float(np.sum(pos) / abs(np.sum(neg))) if len(neg) > 0 and abs(np.sum(neg)) > 1e-9 else float('inf') if len(pos) > 0 else 0.0

        avg_win = float(np.mean(pos)) if len(pos) > 0 else 0.0
        avg_loss = float(np.mean(neg)) if len(neg) > 0 else 0.0

        trades = np.sum(sz > 0)
        days = max(1.0, (pd.to_datetime(timestamps).max() - pd.to_datetime(timestamps).min()).total_seconds() / 86400.0) if timestamps is not None else 1.0
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
            "trades_per_day": trades_per_day
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
    _eval(v2_sizes_no_offset, v2_returns_no_offset, "V2 (Layer A + Layer B policy + best Sizing)")
    _eval(v2_sizes_with_offset, v2_returns_with_offset, "V2 (Full) + Limit Offset Optimizer")
    tprint("=" * 80)

    return pd.DataFrame(rows)

# Layer B reporting expansion
def generate_layer_b_deliverables(
    best_policy: dict,
    j_score: float,
    trade_outcomes: pd.DataFrame,
    start_equity: float = 100000.0
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
