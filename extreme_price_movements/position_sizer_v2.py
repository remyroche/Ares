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
    """Scales predictions handling NaNs explicitly."""
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        # Note: np.nan_to_num silently maps NaNs to zero before scaling.
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
    """
    Creates temporal splits using PurgedKFold.
    If timestamps are provided, purge/embargo are interpreted as seconds.
    If timestamps are None, purge/embargo are interpreted as number of samples.
    """
    cv = PurgedKFold(n_splits=n_splits, purge=purge_units, embargo=embargo_units, times=timestamps)
    dummy_X = np.empty((n_samples, 1))
    return list(cv.split(dummy_X))


def build_log_clipped_target(returns: np.ndarray, clip_L: float = 0.02) -> np.ndarray:
    """
    T1 = log_clipped_winsorized_net
    Signed-log clipped target as specified in Option A.
    """
    clipped = np.clip(returns, -clip_L, clip_L)
    scale = np.std(clipped) if len(clipped) > 1 and np.std(clipped) > 1e-9 else 1.0
    return np.sign(clipped) * np.log1p(np.abs(clipped) / scale)


def build_rank_target(returns: np.ndarray, timestamps: Optional[np.ndarray] = None, mode: str = "fold_local") -> np.ndarray:
    """
    T2 = rank_style_target
    Generates rank-based targets based on explicitly defined modes.

    Modes:
      - 'per_timestamp': Cross-sectional rank within each timestamp.
      - 'fold_local': Rank the provided subset globally (assumes it's already a fold subset).
    """
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


# ============================================================
# Layer A: Predictive models
# ============================================================

class Model1Edge:
    """
    Model 1: Edge model
    Predict edge / expected return / expected utility proxy / rank score.
    Uses HuberRegressor (or Ridge fallback) to maintain robustness.
    """
    def __init__(self, target_name: str = "log_clipped_winsorized_net"):
        self.target_name = target_name
        self.model = HuberRegressor(epsilon=1.35, max_iter=2000)
        self.scaler = PredictionScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        X_scaled = self.scaler.fit_transform(X)
        try:
            self.model.fit(X_scaled, y, sample_weight=sample_weight)
        except Exception as e:
            # Silencing fallback log spam when used inside loops, but conceptually falls back
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
        except Exception:
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
    Layer A Orchestrator:
    - Runs target race for Model 1 via temporal OOF.
    - Fits Downside via temporal OOF.
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
        self.model3_eval_results_ = None

    def _run_model1_target_race(
        self,
        X: np.ndarray,
        raw_returns: np.ndarray,
        timestamps: Optional[np.ndarray],
        sample_weight: Optional[np.ndarray]
    ):
        """Runs temporal OOF for multiple target candidates, evaluates metrics, picks best."""
        splits = make_temporal_splits(timestamps, n_samples=len(X), n_splits=3)

        candidates = {
            "log_clipped_winsorized_net": build_log_clipped_target(raw_returns),
            "rank_style_target": build_rank_target(
                raw_returns,
                timestamps,
                mode="per_timestamp" if timestamps is not None else "fold_local"
            )
        }

        race_results = []
        best_score = -1e9
        best_name = "log_clipped_winsorized_net"
        best_oof = np.zeros(len(X))

        for name, y_cand in candidates.items():
            oof_preds = np.full(len(X), np.nan)
            fold_nets = []

            for tr_idx, val_idx in splits:
                m = Model1Edge(target_name=name)
                # Ensure rank target is fold-local if requested
                if name == "rank_style_target" and timestamps is None:
                    y_tr = build_rank_target(raw_returns[tr_idx], mode="fold_local")
                else:
                    y_tr = y_cand[tr_idx]

                w_tr = sample_weight[tr_idx] if sample_weight is not None else None
                m.fit(X[tr_idx], y_tr, w_tr)

                preds_val = m.predict(X[val_idx])
                oof_preds[val_idx] = preds_val

                # Evaluate fold: top-decile realized net return
                k = max(1, int(0.10 * len(preds_val)))
                top_idx = np.argpartition(preds_val, -k)[-k:]
                top_ret = raw_returns[val_idx][top_idx]
                fold_nets.append(float(np.mean(top_ret)))

            mean_net = float(np.mean(fold_nets)) if fold_nets else 0.0
            std_net = float(np.std(fold_nets)) if len(fold_nets) > 1 else 0.0
            score = mean_net - 0.5 * std_net

            race_results.append({
                "target": name,
                "score": score,
                "mean_top_decile_net": mean_net,
                "std_top_decile_net": std_net
            })

            if score > best_score:
                best_score = score
                best_name = name
                best_oof = oof_preds

        self.model1_target_race_results_ = race_results
        self.model1_best_target_name_ = best_name
        self.model1_oof_pred_ = best_oof

        # Refit final Edge model
        self.edge_model = Model1Edge(target_name=best_name)
        if best_name == "rank_style_target" and timestamps is None:
            y_final = build_rank_target(raw_returns, mode="fold_local")
        else:
            y_final = candidates[best_name]

        self.edge_model.fit(X, y_final, sample_weight)

    def fit(self, X: np.ndarray, y_raw_net_return: np.ndarray, y_downside: np.ndarray,
            timestamps: Optional[np.ndarray] = None,
            sample_weight: Optional[np.ndarray] = None):

        # 1. Edge Target Race (generates self.model1_oof_pred_)
        self._run_model1_target_race(X, y_raw_net_return, timestamps, sample_weight)

        # 2. Downside Model
        self.downside_model.fit(X, y_downside, sample_weight)

        # 3. Uncertainty Model (Strict OOF residuals)
        # Handle cases where OOF has NaNs due to unpredicted boundary edges in Purged CV
        valid_oof = np.isfinite(self.model1_oof_pred_)
        if np.any(valid_oof):
            y_true_for_res = y_raw_net_return[valid_oof]
            residuals = y_true_for_res - self.model1_oof_pred_[valid_oof]
            self.uncertainty_model.fit(X[valid_oof], residuals,
                                       sample_weight[valid_oof] if sample_weight is not None else None)
        else:
            # Fallback if CV completely fails
            residuals = y_raw_net_return - self.edge_model.predict(X)
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

        # Fixed composite objective coefficients
        self.obj_a = 1.0 # Sortino weight
        self.obj_b = 1.0 # MaxDD weight
        self.obj_c = 1.0 # Instability weight

        # Artifacts
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
        Computes the active mask locally per-fold based on the threshold percentile.
        """
        TIMEOUT_REASON_IDX = 3
        fold_pnl_days = []
        fold_sortinos = []
        fold_maxdds = []
        fold_timeout_rates = []

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

            # U is log-return-like utility.
            ret = np.expm1(u)
            pnl = float(np.sum(ret))

            # Intraday exact fractional day count
            ts = pd.to_datetime(timestamps[active_val_idx])
            days = max((ts.max() - ts.min()).total_seconds() / 86400.0, 1.0)
            pnl_day = pnl / days

            from extreme_price_movements.metrics import _stable_equity_and_drawdown
            _, dd = _stable_equity_and_drawdown(ret)
            maxDD = float(np.max(dd)) if len(dd) > 0 else 1.0

            neg_rets = ret[ret < 0]
            dd_dev = float(np.sqrt(np.mean(neg_rets**2))) if len(neg_rets) > 0 else 1e-3
            sortino = float(np.mean(ret) / (dd_dev + 1e-9) * np.sqrt(365))

            fold_pnl_days.append(pnl_day)
            fold_sortinos.append(sortino)
            fold_maxdds.append(maxDD)
            fold_timeout_rates.append(reason_counts[TIMEOUT_REASON_IDX] / len(u))

        return {
            "net_pnl_day": float(np.mean(fold_pnl_days)),
            "sortino": float(np.mean(fold_sortinos)),
            "maxDD": float(np.mean(fold_maxdds)),
            "instability": float(np.std(fold_pnl_days)),
            "timeout_rate": float(np.mean(fold_timeout_rates))
        }

    def _score_candidates(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Applies standardized composite scoring within the candidate set.
        utility = z(net_pnl_day) + a*z(sortino) - b*z(maxDD) - c*z(instability)
        """
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
        """
        1. Optimize exit geometry on top 40% trades (q=0.60)
        2. Optimize selection boundary
        3. Optimize time management
        """
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
            # Removed arbitrary trail < 2.0*gb constraint per Issue 8
            pol = LabelPolicy(
                sl_atr_mult=sl, tp_sl_ratio=tp, max_hold_bars=24,
                trail_activate_atr=trail, giveback_pct=gb,
                early_exit_deadline_bars=0, early_exit_mfe_atr=0.0
            )
            res = self._eval_policy_over_folds(pol, trade_outcomes, scores, threshold_q=0.60, splits=splits, timestamps=timestamps)
            res["step"] = 1
            res["policy_obj"] = pol
            step1_results.append(res)

        step1_scored = self._score_candidates(step1_results)
        candidate_table.extend(step1_scored)
        best_geom = step1_scored[0]["policy_obj"]
        tprint(f"Layer B Step 1 Winner: SL={best_geom.sl_atr_mult}, TP/SL={best_geom.tp_sl_ratio}, Utility={step1_scored[0]['utility']:.4f}")

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
        tprint(f"Layer B Step 2 Winner: Threshold Q={best_q:.2f}, Utility={step2_scored[0]['utility']:.4f}")

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
            step3_results.append(res)

        step3_scored = self._score_candidates(step3_results)
        candidate_table.extend(step3_scored)
        best_pol = step3_scored[0]["policy_obj"]
        tprint(f"Layer B Step 3 Winner: Timeout={best_pol.max_hold_bars}, EarlyExit={best_pol.early_exit_deadline_bars} bars @ {best_pol.early_exit_mfe_atr} ATR, Utility={step3_scored[0]['utility']:.4f}")

        # Artifact storage
        self.layer_b_candidate_table_ = pd.DataFrame([{
            "step": r["step"], "sl": r["policy_obj"].sl_atr_mult, "tp_ratio": r["policy_obj"].tp_sl_ratio,
            "max_hold": r["policy_obj"].max_hold_bars, "net_pnl_day": r["net_pnl_day"],
            "sortino": r["sortino"], "maxDD": r["maxDD"], "instability": r["instability"],
            "utility": r["utility"]
        } for r in candidate_table])

        self.layer_b_selected_objective_components_ = step3_scored[0]
        self.best_j = step3_scored[0]["utility"]

        # Save threshold mapping to absolute score domain based on final full run
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
        """Same standardized objective as Layer B."""
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
        """
        Compare linear, convex, concave sizing curves over temporal folds based on Utility score.
        """
        splits = make_temporal_splits(timestamps, n_samples=len(scores), n_splits=3)
        modes = ["linear", "convex", "concave"]
        mode_results = {m: {"fold_pnl_day": [], "fold_sortino": [], "fold_maxdd": []} for m in modes}

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
            days = max((ts_val.max() - ts_val.min()).total_seconds() / 86400.0, 1.0)

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

        # Aggregate
        eval_rows = []
        for m in modes:
            mr = mode_results[m]
            eval_rows.append({
                "mode": m,
                "net_pnl_day": float(np.mean(mr["fold_pnl_day"])),
                "sortino": float(np.mean(mr["fold_sortino"])),
                "maxDD": float(np.mean(mr["fold_maxdd"])),
                "instability": float(np.std(mr["fold_pnl_day"]))
            })

        scored = self._score_candidates(eval_rows)
        self.layer_c_candidate_table_ = pd.DataFrame(scored)
        self.sizing_mode = scored[0]["mode"]

        tprint(f"Layer C: Selected sizing mode '{self.sizing_mode}' with Utility={scored[0]['utility']:.4f}")
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

    tprint(f"  Composite Objective (Utility): {j_score:.4f}")
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
