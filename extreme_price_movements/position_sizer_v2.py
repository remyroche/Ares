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


# ============================================================
# Shared Configuration & Utilities
# ============================================================
logger = logging.getLogger(__name__)

class PredictionScaler:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.fit_transform(np.nan_to_num(X))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(np.nan_to_num(X))

def _robust_scale(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 1.0
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return 1.4826 * mad if mad > 1e-6 else 1.0

def build_winsorized_target(returns: np.ndarray, clip_L: float = 0.02) -> np.ndarray:
    """T1 = log_clipped_winsorized_net"""
    return np.clip(returns, -clip_L, clip_L)

def build_rank_target(returns: np.ndarray, window: int = 2000) -> np.ndarray:
    """T2 = rank_style_target"""
    # Simplified cross-sectional/rolling rank
    order = np.argsort(returns)
    ranks = np.empty_like(returns, dtype=float)
    ranks[order] = (np.arange(len(returns)) + 0.5) / len(returns)
    return (ranks * 2.0) - 1.0

# ============================================================
# Layer A: Predictive models
# ============================================================

class Model1Edge:
    """
    Model 1: Edge model
    Predict edge / expected return / expected utility proxy / rank score.
    Uses HuberRegressor (or Ridge fallback) to maintain robustness.
    """
    def __init__(self, target_name: str = "winsorized_net"):
        self.target_name = target_name
        self.model = HuberRegressor(epsilon=1.35, max_iter=2000)
        self.scaler = PredictionScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        X_scaled = self.scaler.fit_transform(X)
        try:
            self.model.fit(X_scaled, y, sample_weight=sample_weight)
        except Exception as e:
            tprint(f"HuberRegressor failed for Edge model: {e}. Falling back to Ridge.")
            self.model = Ridge(alpha=1.0)
            self.model.fit(X_scaled, y, sample_weight=sample_weight)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model1Edge not fitted")
        return self.model.predict(self.scaler.transform(X))


class Model2Downside:
    """
    Model 2: Downside model
    Predicts path downside using MAE_x / ATR as primary target using HuberRegressor.
    """
    def __init__(self):
        self.model = HuberRegressor(epsilon=1.35, max_iter=2000)
        self.scaler = PredictionScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray, y_mae_atr: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        """Fits downside model to MAE_x / ATR target."""
        X_scaled = self.scaler.fit_transform(X)
        try:
            self.model.fit(X_scaled, y_mae_atr, sample_weight=sample_weight)
        except Exception as e:
            tprint(f"HuberRegressor failed for Downside model: {e}. Falling back to Ridge.")
            self.model = Ridge(alpha=1.0)
            self.model.fit(X_scaled, y_mae_atr, sample_weight=sample_weight)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model2Downside not fitted")
        return self.model.predict(self.scaler.transform(X))


class Model3Uncertainty:
    """
    Model 3: Uncertainty model
    Predicts future unreliability of Model 1.
    Target is transformed regression target: log1p(abs(residual))
    """
    def __init__(self):
        self.model = HuberRegressor(epsilon=1.35, max_iter=2000)
        self.scaler = PredictionScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray, residuals: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        # Transformed regression target: log1p(abs_residual)
        y_target = np.log1p(np.abs(residuals))
        X_scaled = self.scaler.fit_transform(X)
        try:
            self.model.fit(X_scaled, y_target, sample_weight=sample_weight)
        except Exception as e:
            tprint(f"HuberRegressor failed for Uncertainty model: {e}. Falling back to Ridge.")
            self.model = Ridge(alpha=1.0)
            self.model.fit(X_scaled, y_target, sample_weight=sample_weight)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model3Uncertainty not fitted")
        pred_log = self.model.predict(self.scaler.transform(X))
        # Inverse transform
        return np.expm1(pred_log)


class LayerAPredictor:
    """
    Layer A Combined Score & Initial Sizing
    score = edge_net_of_cost - lambda_downside * downside - eta_uncertainty * uncertainty
    """
    def __init__(self, lambda_downside: float = 0.5, eta_uncertainty: float = 0.5):
        self.edge_model = Model1Edge()
        self.downside_model = Model2Downside()
        self.uncertainty_model = Model3Uncertainty()

        self.lambda_downside = lambda_downside
        self.eta_uncertainty = eta_uncertainty
        self.is_fitted = False

    def fit(self, X: np.ndarray, y_edge: np.ndarray, y_downside: np.ndarray,
            timestamps: Optional[np.ndarray] = None,
            sample_weight: Optional[np.ndarray] = None):

        # In a real CV pipeline, these would be fit sequentially inside temporal folds
        # so that Model 3 fits on OOF residuals.
        # For simplicity here, we fit on full X and assume residuals are from the same dataset.

        self.edge_model.fit(X, y_edge, sample_weight)

        # Simulate OOF residual extraction for Uncertainty model
        y_edge_pred = self.edge_model.predict(X)
        residuals = y_edge - y_edge_pred

        self.downside_model.fit(X, y_downside, sample_weight)
        self.uncertainty_model.fit(X, residuals, sample_weight)

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
        """
        Initial decision rule: select if score > threshold, size with linear map.
        size = base_size * linear_map(score)
        """
        active = score > threshold
        sizes = np.zeros_like(score)
        if np.any(active):
            # Simple linear mapping scaled to [1.0, 2.0] multiple of base size
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

    def _eval_policy(
        self,
        pol: LabelPolicy,
        trade_outcomes: pd.DataFrame,
        active_mask: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate a single LabelPolicy over active trades."""
        active_idx = np.where(active_mask)[0]
        if len(active_idx) == 0:
            return {"j_stable": -1e9, "net_pnl_day": 0.0, "sortino": 0.0, "maxDD": 1.0}

        # Extract path features
        # Assuming trade_outcomes has future_opens, future_highs, future_lows, future_closes
        entry_prices = trade_outcomes["entry_price"].values[active_idx]
        atr_entries = trade_outcomes["atr_12_15m"].values[active_idx]
        is_longs = trade_outcomes["is_long"].values[active_idx]

        # Max bars logic (cap at 24/48 depending on data)
        max_b = min(48, pol.max_hold_bars)

        # Extract variable-length paths into 2D arrays
        def _pad(series, max_len):
            res = np.full((len(series), max_len), np.nan, dtype=np.float64)
            lens = np.zeros(len(series), dtype=np.int64)
            for i, arr in enumerate(series):
                use = min(len(arr), max_len)
                if use > 0:
                    res[i, :use] = arr[:use]
                    lens[i] = use
            return res, lens

        opens_2d, _ = _pad(trade_outcomes["future_opens"].values[active_idx], max_b)
        highs_2d, _ = _pad(trade_outcomes["future_highs"].values[active_idx], max_b)
        lows_2d, _ = _pad(trade_outcomes["future_lows"].values[active_idx], max_b)
        closes_2d, path_lens = _pad(trade_outcomes["future_closes"].values[active_idx], max_b)

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

        # Compute metrics
        if len(u) == 0:
             return {"j_stable": -1e9, "net_pnl_day": 0.0, "sortino": 0.0, "maxDD": 1.0}

        # Simple net PnL logic
        ret = np.expm1(u)
        pnl = float(np.sum(ret))

        # Time components
        ts = pd.to_datetime(trade_outcomes.get("timestamp", trade_outcomes.index)[active_idx])
        days = max(1, (ts.max() - ts.min()).days if not pd.isna(ts.max()) else 1)
        pnl_day = pnl / days

        from extreme_price_movements.metrics import _stable_equity_and_drawdown
        _, dd = _stable_equity_and_drawdown(ret)
        maxDD = float(np.max(dd)) if len(dd) > 0 else 1.0

        # Sortino
        neg_rets = ret[ret < 0]
        dd_dev = float(np.sqrt(np.mean(neg_rets**2))) if len(neg_rets) > 0 else 1e-3
        sortino = float(np.mean(ret) / (dd_dev + 1e-9) * np.sqrt(365))

        # Objective: utility = z(net_pnl_day) + a * z(sortino) - b * z(maxDD) - c * z(instability)
        # To avoid relying on cross-policy z-scoring in a dynamic loop, we use a heuristic linear combo
        j_stable = pnl_day + 0.01 * sortino - 5.0 * maxDD

        return {
            "j_stable": j_stable,
            "net_pnl_day": pnl_day,
            "sortino": sortino,
            "maxDD": maxDD,
            "timeout_rate": reason_counts[3] / len(u)
        }

    def optimize(self, scores: np.ndarray, trade_outcomes: pd.DataFrame, timestamps: np.ndarray):
        """
        1. Optimize exit geometry on top 40% trades
        2. Optimize selection boundary
        3. Optimize time management
        """
        # --- Step 1: Exit Geometry (SL, Trailing, Giveback) ---
        top40_thresh = np.percentile(scores, 60)
        active_mask = scores >= top40_thresh

        sl_cands = [1.0, 1.5, 2.0]
        tp_cands = [1.5, 2.0, 2.5]
        trail_cands = [0.8, 1.0, 1.5]
        giveback_cands = [0.35, 0.50]

        best_j = -1e9
        best_geom = None

        tprint(f"Layer B Step 1: Optimizing exit geometry over top 40% ({np.sum(active_mask)} trades)...")
        for sl, tp, trail, gb in itertools.product(sl_cands, tp_cands, trail_cands, giveback_cands):
            # Enforce spec constraint: trailing activation must make sense
            if trail < 2.0 * gb:
                continue

            pol = LabelPolicy(
                sl_atr_mult=sl,
                tp_sl_ratio=tp,
                max_hold_bars=24, # Fixed for step 1
                trail_activate_atr=trail,
                giveback_pct=gb,
                early_exit_deadline_bars=0, # Disabled for step 1
                early_exit_mfe_atr=0.0
            )
            res = self._eval_policy(pol, trade_outcomes, active_mask)
            if res["j_stable"] > best_j:
                best_j = res["j_stable"]
                best_geom = pol

        tprint(f"Layer B Step 1 Winner: SL={best_geom.sl_atr_mult}, TP/SL={best_geom.tp_sl_ratio}, J={best_j:.4f}")

        # --- Step 2: Selection Boundary ---
        tprint("Layer B Step 2: Optimizing selection boundary...")
        best_thresh_j = -1e9
        best_thresh = top40_thresh
        for q in [0.70, 0.60, 0.50, 0.30]:
            thresh = np.percentile(scores, q * 100)
            mask = scores >= thresh
            res = self._eval_policy(best_geom, trade_outcomes, mask)
            if res["j_stable"] > best_thresh_j:
                best_thresh_j = res["j_stable"]
                best_thresh = thresh

        tprint(f"Layer B Step 2 Winner: Threshold={best_thresh:.4f}, J={best_thresh_j:.4f}")

        # --- Step 3: Time Management ---
        tprint("Layer B Step 3: Optimizing time management...")
        active_mask = scores >= best_thresh
        timeout_cands = [16, 24, 32]
        early_deadline_cands = [0, 8, 12]
        early_mfe_cands = [0.25, 0.50]

        best_time_j = -1e9
        best_pol = best_geom

        for t_out, ed, em in itertools.product(timeout_cands, early_deadline_cands, early_mfe_cands):
            if ed == 0:
                em = 0.0 # Clear MFE if no early exit
            pol = LabelPolicy(
                sl_atr_mult=best_geom.sl_atr_mult,
                tp_sl_ratio=best_geom.tp_sl_ratio,
                max_hold_bars=t_out,
                trail_activate_atr=best_geom.trail_activate_atr,
                giveback_pct=best_geom.giveback_pct,
                early_exit_deadline_bars=ed,
                early_exit_mfe_atr=em
            )
            res = self._eval_policy(pol, trade_outcomes, active_mask)
            if res["j_stable"] > best_time_j:
                best_time_j = res["j_stable"]
                best_pol = pol

        tprint(f"Layer B Step 3 Winner: Timeout={best_pol.max_hold_bars}, EarlyExit={best_pol.early_exit_deadline_bars} bars @ {best_pol.early_exit_mfe_atr} ATR, J={best_time_j:.4f}")

        self.best_j = best_time_j
        self.best_policy = {
            "sl_atr_mult": best_pol.sl_atr_mult,
            "tp_sl_ratio": best_pol.tp_sl_ratio,
            "giveback_pct": best_pol.giveback_pct,
            "trail_activate_atr": best_pol.trail_activate_atr,
            "score_threshold": float(best_thresh),
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
    """
    def __init__(self, start_equity: float = 100000.0):
        self.sizing_mode = "linear"
        self.limit_model = Ridge(alpha=1.0)
        self.scaler = PredictionScaler()
        self.is_fitted = False
        self.start_equity = start_equity

    def optimize_sizing(self, scores: np.ndarray, returns: np.ndarray, threshold: float, base_size: float = 0.05):
        """
        Compare linear, convex, concave sizing curves based on PnL & Sortino.
        Default to linear or concave.
        """
        active = scores >= threshold
        if not np.any(active):
            return "linear"

        s_act = scores[active]
        r_act = returns[active]
        s_min, s_max = s_act.min(), s_act.max()
        if s_max <= s_min:
            return "linear"

        s_norm = (s_act - s_min) / (s_max - s_min)

        candidates = {
            "linear": base_size * (1.0 + s_norm),
            "convex": base_size * (1.0 + s_norm**2),
            "concave": base_size * (1.0 + np.sqrt(s_norm))
        }

        best_mode = "linear"
        best_score = -1e9

        for mode, sizes in candidates.items():
            sizes = np.clip(sizes, 0.0, 1.0)
            pnl_series = self.start_equity * sizes * r_act
            pnl_total = float(np.sum(pnl_series))

            neg = pnl_series[pnl_series < 0]
            dd_dev = float(np.sqrt(np.mean(neg**2))) if len(neg) > 0 else 1e-3
            sortino = float(np.mean(pnl_series) / (dd_dev + 1e-9) * np.sqrt(365))

            # Simple scoring: purely looking for which sizing maps the tail best
            score = pnl_total + 1000.0 * sortino
            if score > best_score:
                best_score = score
                best_mode = mode

        self.sizing_mode = best_mode
        tprint(f"Layer C: Selected sizing mode '{self.sizing_mode}'")
        return self.sizing_mode

    def fit_limit_offset(self, X: np.ndarray, y_offset: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        """Train passive limit offset regression."""
        X_scaled = self.scaler.fit_transform(X)
        self.limit_model.fit(X_scaled, y_offset, sample_weight=sample_weight)
        self.is_fitted = True
        return self

    def predict_size_and_offset(self, X: np.ndarray, score: np.ndarray, threshold: float = 0.0, base_size: float = 0.05):
        offset = np.zeros_like(score)
        if self.is_fitted:
            offset = self.limit_model.predict(self.scaler.transform(X))
            offset = np.clip(offset, 0.0, 5.0)

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
):
    """
    Produce final comparison report between Original setup,
    V2 (No offset), and V2 (With limit offset).
    """
    tprint("=" * 80)
    tprint("FINAL EXPERIMENT COMPARISON")
    tprint("=" * 80)

    def _eval(sz, rets, name):
        pnl_series = start_equity * sz * rets
        pnl = np.sum(pnl_series)
        neg = pnl_series[pnl_series < 0]
        dd_dev = float(np.sqrt(np.mean(neg**2))) if len(neg) > 0 else 1e-3
        sortino = float(np.mean(pnl_series) / (dd_dev + 1e-9) * np.sqrt(365))

        _, dd = _stable_equity_and_drawdown(pnl_series)
        mdd = float(np.max(dd)) if len(dd) > 0 else 1.0

        tprint(f"  {name}:")
        tprint(f"    Net PnL: ${pnl:.2f}")
        tprint(f"    Sortino: {sortino:.3f}")
        tprint(f"    Max DD:  {mdd:.4%}")

    _eval(baseline_sizes, baseline_returns, "Baseline (Original)")
    _eval(v2_sizes_no_offset, v2_returns_no_offset, "V2 (Layer A + Layer B policy + best Sizing)")
    _eval(v2_sizes_with_offset, v2_returns_with_offset, "V2 (Full) + Limit Offset Optimizer")
    tprint("=" * 80)

# Layer B reporting expansion
def generate_layer_b_deliverables(
    best_policy: dict,
    j_score: float,
    trade_outcomes: pd.DataFrame,
    start_equity: float = 100000.0
):
    """
    Produce Layer B specific artifacts and financial metrics per spec.
    """
    tprint("=" * 80)
    tprint("LAYER B DELIVERABLES: POLICY OPTIMIZATION")
    tprint("=" * 80)
    tprint("  Best Layer B Policy:")
    for k, v in best_policy.items():
        if isinstance(v, float):
            tprint(f"    {k}: {v:.4f}")
        else:
            tprint(f"    {k}: {v}")

    tprint(f"  Composite Objective (J): {j_score:.4f}")
    tprint("=" * 80)

# Target race expansion helper
def generate_target_race_report(
    target_candidates: Dict[str, np.ndarray],
    y_oos_dict: Dict[str, np.ndarray],
    base_returns: np.ndarray,
):
    """
    Evaluate target proxies (e.g. Huber vs Winsorized) based on top-decile performance.
    """
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
