from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import importlib.util
import numpy as np
import pandas as pd
from numba import njit
from scipy.special import logit
from scipy.stats import median_abs_deviation
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import SplineTransformer
from joblib import Parallel, delayed

from extreme_price_movements.config import CFG
from extreme_price_movements.feature_selection_extreme_events import (
    mdi_feature_selection_v3 as mdi_feature_selection_classic,
    mdi_feature_selection_v4_topk as mdi_feature_selection_v4_topk_classic
)
from extreme_price_movements.purged_cv import PurgedKFold
from extreme_price_movements.quantile_feature_selection_extreme_events import mdi_feature_selection_v3
from extreme_price_movements.utils import tprint


if importlib.util.find_spec("lightgbm") is not None:
    import lightgbm as lgb
else:
    lgb = None

if importlib.util.find_spec("xgboost") is not None:
    import xgboost as xgb
else:
    xgb = None


@njit(cache=True)
def _pinball_numba(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    s = 0.0
    n = y.shape[0]
    for i in range(n):
        e = y[i] - q[i]
        s += tau * e if e >= 0 else (tau - 1.0) * e
    return s / max(1, n)


@njit(cache=True)
def _maxdd_numba(x: np.ndarray) -> float:
    c = 0.0
    peak = -1e18
    maxdd = 0.0
    for i in range(x.shape[0]):
        c += x[i]
        if c > peak:
            peak = c
        dd = peak - c
        if dd > maxdd:
            maxdd = dd
    denom = abs(peak) if abs(peak) > 1e-12 else 1.0
    return maxdd / denom


@njit(cache=True)
def _sortino_numba(x: np.ndarray) -> float:
    n = x.shape[0]
    mu = 0.0
    for i in range(n):
        mu += x[i]
    mu /= max(1, n)
    dn_sum = 0.0
    dn_n = 0
    for i in range(n):
        if x[i] < 0:
            dn_sum += x[i] * x[i]
            dn_n += 1
    dn_std = (dn_sum / dn_n) ** 0.5 if dn_n > 0 else 1e-9
    return mu / (dn_std + 1e-9)


def _pinball(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    # Use float32 by default for memory efficiency
    return float(_pinball_numba(np.asarray(y, dtype=np.float32), np.asarray(q, dtype=np.float32), float(tau)))


def _topk_stats(y: np.ndarray, s: np.ndarray, frac: float = 0.15, lam: float = 1.0) -> Tuple[float, float, float, float]:
    k = max(1, int(np.ceil(len(y) * frac)))
    idx = np.argpartition(s, -k)[-k:]
    sel = np.asarray(y, dtype=np.float32)[idx]
    neg = sel[sel < 0]
    mean_top = float(np.mean(sel))
    iqr = float(np.quantile(sel, 0.75) - np.quantile(sel, 0.25))
    util = mean_top - lam * (abs(float(np.mean(neg))) if neg.size else 0.0)
    return util, mean_top, iqr, float(_sortino_numba(sel.astype(np.float32)))


@dataclass
class _SplineQuantile:
    spline: SplineTransformer
    reg: QuantileRegressor


class MetaModel:
    def __init__(self, strategy_name: Optional[str] = None):
        self.strategy_name = strategy_name
        self.model = None
        self._model_type = None
        self.selected_features: Optional[List[str]] = None
        self.monotone_constraints: Optional[Tuple[int, ...]] = None
        self.interaction_constraints: Optional[List[List[int]]] = None
        self.oof_probs: Optional[np.ndarray] = None
        self.report_rows: List[dict] = []
        self._selected_features_by_pool: Dict[str, List[str]] = {}

    def prepare_meta_features(self, preds, feats_df, pred_col_name="pred_logit"):
        p = np.clip(np.asarray(preds, dtype=float), 1e-4, 1 - 1e-4)
        meta_data = pd.DataFrame(index=feats_df.index)
        meta_data[pred_col_name] = np.clip(logit(p), -4.0, 4.0)
        return pd.concat([meta_data, feats_df], axis=1).fillna(0.0)

    def _fit_huberized_quantile_gam(self, x: np.ndarray, y: np.ndarray, tau: float) -> _SplineQuantile:
        y = np.asarray(y, dtype=float)
        lo, hi = np.quantile(y, [0.01, 0.99])
        y_clip = np.clip(y, lo, hi)
        spline = SplineTransformer(n_knots=8, degree=3, include_bias=False)
        Xs = spline.fit_transform(x.reshape(-1, 1))
        reg = QuantileRegressor(quantile=tau, alpha=1e-4, solver="highs")
        reg.fit(Xs, y_clip)
        return _SplineQuantile(spline=spline, reg=reg)

    def _eval_curve(self, fit: _SplineQuantile, grid: np.ndarray) -> np.ndarray:
        return fit.reg.predict(fit.spline.transform(grid.reshape(-1, 1)))

    def _discover_monotone_constraints(self, X: pd.DataFrame, y: np.ndarray, bootstraps: int = 50) -> Tuple[int, ...]:
        taus = [0.75, 0.85, 0.95]
        mad_y = float(median_abs_deviation(y, scale="normal")) + 1e-9
        stats = {
            c: {"A85": [], "Aall": [], "sign85": [], "sign95": [], "m_up": [], "m_broad": []}
            for c in X.columns
        }
        rng = np.random.default_rng(42)
        n = len(X)
        for _ in range(bootstraps):
            bidx = rng.choice(n, size=n, replace=True)
            yb = y[bidx]
            for col in X.columns:
                xb = X[col].to_numpy(dtype=float)[bidx]
                for tau in taus:
                    fit = self._fit_huberized_quantile_gam(xb, yb, tau)
                    q50, q90 = np.quantile(xb, [0.5, 0.9])
                    swing = float(self._eval_curve(fit, np.array([q90]))[0] - self._eval_curve(fit, np.array([q50]))[0])
                    A = abs(swing) / mad_y
                    sign = int(np.sign(swing))
                    if tau == 0.85:
                        stats[col]["A85"].append(A)
                        stats[col]["sign85"].append(sign)
                    if tau == 0.95:
                        stats[col]["sign95"].append(sign)
                    stats[col]["Aall"].append(A)
                    g1 = np.linspace(np.quantile(xb, 0.6), np.quantile(xb, 0.99), 32)
                    g2 = np.linspace(np.quantile(xb, 0.3), np.quantile(xb, 0.9), 32)
                    d1 = np.diff(self._eval_curve(fit, g1))
                    d2 = np.diff(self._eval_curve(fit, g2))
                    if sign != 0:
                        stats[col]["m_up"].append(float(np.mean(np.sign(d1) == sign)))
                        stats[col]["m_broad"].append(float(np.mean(np.sign(d2) == sign)))
                    else:
                        stats[col]["m_up"].append(0.0)
                        stats[col]["m_broad"].append(0.0)

        passed = []
        for col in X.columns:
            sign85 = np.array(stats[col]["sign85"], dtype=int)
            sign95 = np.array(stats[col]["sign95"], dtype=int)
            if sign85.size == 0:
                continue
            direction = int(np.sign(np.median(sign85)))
            if direction == 0:
                continue
            direction_consistency = float(np.mean(sign85 == direction))
            flips95 = float(np.mean(sign95 != direction)) if sign95.size else 1.0
            A85 = np.array(stats[col]["A85"], dtype=float)
            med_A85 = float(np.median(A85)) if A85.size else 0.0
            mad_A85 = float(np.median(np.abs(A85 - med_A85))) if A85.size else 1.0
            ratio = mad_A85 / max(abs(med_A85), 1e-9)
            mup = float(np.mean(np.array(stats[col]["m_up"]) >= 0.80))
            mbroad = float(np.mean(np.array(stats[col]["m_broad"]) >= 0.70))
            if (
                direction_consistency >= 0.90
                and flips95 <= 0.20
                and med_A85 >= 0.02
                and ratio <= 0.5
                and mup >= 0.80
                and mbroad >= 0.70
            ):
                passed.append((col, direction, med_A85))

        if len(passed) > int(0.6 * X.shape[1]):
            passed = sorted(passed, key=lambda t: t[2], reverse=True)[: int(0.6 * X.shape[1])]
        sign_map = {c: s for c, s, _ in passed}
        return tuple(sign_map.get(c, 0) for c in X.columns)

    def _pair_auc_proxy(self, x1: np.ndarray, x2: np.ndarray, z: np.ndarray, bins: int = 12) -> float:
        if len(np.unique(z)) < 2:
            return 0.5
        q1 = np.quantile(x1, np.linspace(0, 1, bins + 1))
        q2 = np.quantile(x2, np.linspace(0, 1, bins + 1))
        i1 = np.clip(np.digitize(x1, q1[1:-1]), 0, bins - 1)
        i2 = np.clip(np.digitize(x2, q2[1:-1]), 0, bins - 1)
        key = i1 * bins + i2
        counts = np.bincount(key, minlength=bins * bins)
        sums = np.bincount(key, weights=z, minlength=bins * bins)
        rates = np.zeros(bins * bins, dtype=float)
        valid = counts > 0
        rates[valid] = sums[valid] / counts[valid]
        score = rates[key]
        order = np.argsort(score)
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        pos = rank[z == 1]
        neg = rank[z == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        return float((np.mean([np.mean(p > neg) + 0.5 * np.mean(p == neg) for p in pos])))

    def _bh_fdr(self, pvals: np.ndarray, q: float = 0.05) -> np.ndarray:
        m = len(pvals)
        if m == 0:
            return np.array([], dtype=bool)
        order = np.argsort(pvals)
        thresh = q * (np.arange(1, m + 1) / m)
        passed = pvals[order] <= thresh
        if not np.any(passed):
            return np.zeros(m, dtype=bool)
        k = np.max(np.where(passed)[0])
        cutoff = pvals[order][k]
        return pvals <= cutoff

    def _discover_interactions(self, X: pd.DataFrame, y: np.ndarray) -> List[List[int]]:
        # residualize with additive GAM
        additive = np.zeros_like(y, dtype=float)
        for col in X.columns:
            fit = self._fit_huberized_quantile_gam(X[col].to_numpy(dtype=float), y, tau=0.85)
            additive += self._eval_curve(fit, X[col].to_numpy(dtype=float))
        residual = y - additive
        z_excess = (residual >= np.quantile(residual, 0.80)).astype(np.int8)

        pairs = list(combinations(range(X.shape[1]), 2))
        if not pairs:
            return []
        n_pool = min(len(pairs), max(1000, min(5000, len(pairs))))
        rng = np.random.default_rng(7)
        sampled_idx = rng.choice(len(pairs), n_pool, replace=False)
        pool = [pairs[i] for i in sampled_idx]

        scored = []
        for i, j in pool:
            s = self._pair_auc_proxy(X.iloc[:, i].to_numpy(dtype=float), X.iloc[:, j].to_numpy(dtype=float), z_excess)
            scored.append((i, j, s))
        scored.sort(key=lambda t: t[2], reverse=True)
        top = scored[:200]

        block_len = 14
        n_perm = 50
        pvals = []
        stability = []
        tau_stability = []
        for i, j, stat in top:
            null_stats = []
            stable_hits = 0
            tau_hits = 0
            for _ in range(n_perm):
                idx = np.arange(len(z_excess))
                blocks = [idx[k : k + block_len] for k in range(0, len(idx), block_len)]
                rng.shuffle(blocks)
                zp = z_excess[np.concatenate(blocks)]
                s = self._pair_auc_proxy(X.iloc[:, i].to_numpy(dtype=float), X.iloc[:, j].to_numpy(dtype=float), zp)
                null_stats.append(s)
                stable_hits += int(stat > s)

                # tau stability over {0.70,0.80,0.90}
                tau_cnt = 0
                for tq in (0.70, 0.80, 0.90):
                    zt = (residual >= np.quantile(residual, tq)).astype(np.int8)
                    st = self._pair_auc_proxy(X.iloc[:, i].to_numpy(dtype=float), X.iloc[:, j].to_numpy(dtype=float), zt)
                    tau_cnt += int(st > s)
                tau_hits += int(tau_cnt >= 2)
            pval = (1.0 + np.sum(np.array(null_stats) >= stat)) / (1.0 + n_perm)
            pvals.append(pval)
            stability.append(stable_hits / n_perm)
            tau_stability.append(tau_hits / n_perm)

        pvals = np.array(pvals)
        fdr_pass = self._bh_fdr(pvals, q=0.05)
        keep = []
        for k, (i, j, s) in enumerate(top):
            if fdr_pass[k] and stability[k] >= 0.95 and tau_stability[k] >= 0.95:
                keep.append((i, j, s))

        n_tail = int(np.sum(z_excess))
        k_fdr = len(keep)
        K = min(50, int(np.floor(0.7 * np.sqrt(max(1, n_tail)))), k_fdr)
        keep = sorted(keep, key=lambda t: t[2], reverse=True)[:K]
        return [[int(i), int(j)] for i, j, _ in keep]

    def _fit_model(self, kind: str, params: dict, X_tr, y_tr, X_va, y_va, sample_weight=None):
        if kind == "ridge":
            model = Ridge(**params)
            model.fit(X_tr, y_tr, sample_weight=sample_weight)
            return model
        if kind == "extratrees":
            model = ExtraTreesRegressor(**params)
            model.fit(X_tr, y_tr, sample_weight=sample_weight)
            return model
        if kind == "qreg_l1":
            model = QuantileRegressor(**params)
            model.fit(X_tr, y_tr)
            return model
        if kind == "xgb":
            if xgb is None:
                raise RuntimeError("xgboost not available")
            model = xgb.XGBRegressor(**params)
            model.fit(
                X_tr,
                y_tr,
                sample_weight=sample_weight,
                eval_set=[(X_va, y_va)],
                verbose=False,
            )
            return model
        try:
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr,
                y_tr,
                sample_weight=sample_weight,
                eval_set=[(X_va, y_va)],
                eval_metric="quantile",
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            return model
        except Exception:
            p2 = dict(params)
            p2.pop("monotone_constraints", None)
            p2.pop("interaction_constraints", None)
            model = lgb.LGBMRegressor(**p2)
            model.fit(
                X_tr,
                y_tr,
                sample_weight=sample_weight,
                eval_set=[(X_va, y_va)],
                eval_metric="quantile",
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            return model

    def _calibrate_fold(self, y_cal: np.ndarray, p_cal: np.ndarray, p_eval: np.ndarray, tau: float = 0.85) -> np.ndarray:
        bins = pd.qcut(p_cal, q=min(10, max(2, len(p_cal) // 20)), duplicates="drop")
        bcodes = bins.codes
        delta = {}
        for b in np.unique(bcodes):
            if b < 0:
                continue
            m = bcodes == b
            delta[int(b)] = float(np.quantile(y_cal[m] - p_cal[m], tau))
        eval_bins = np.digitize(p_eval, np.quantile(p_cal, np.linspace(0.0, 1.0, 11))[1:-1], right=False)
        return p_eval + np.array([delta.get(int(b), 0.0) for b in eval_bins])

    def _coverage(self, y: np.ndarray, q: np.ndarray) -> float:
        return float(np.mean(y <= q))


    def _tail_ramp_weights(self, y: np.ndarray, lambda_: float, q0: float = 0.70, q1: float = 1.00) -> np.ndarray:
        from scipy.stats import rankdata
        y = np.asarray(y, dtype=float)
        r = rankdata(y, method="average") / max(len(y), 1)
        t = np.clip((r - q0) / max(1e-9, (q1 - q0)), 0.0, 1.0)
        return (1.0 + float(lambda_) * t).astype(float)

    def _signed_log_demeaned(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        z = np.sign(y) * np.log1p(np.abs(y))
        return z - float(np.mean(z))

    def _select_features_for_candidate(self, X: pd.DataFrame, y: np.ndarray, candidate_name: str, kind: str) -> List[str]:
        # Independent feature selection per candidate as requested.
        # Meta model is focused on top30%, so use two-stage selection with decile ranking.
        tailweighted = "tailweighted" in candidate_name
        
        # Target feature count
        n_target = min(40, max(20, int(np.sqrt(max(25, X.shape[1])) * 4)))
        n_stage1 = min(X.shape[1], n_target * 3)  # 3x target for meta model
        
        if kind in ("xgb", "lgb", "qreg_l1") and not tailweighted:
            # Two-stage: v3 to get 2x, then v4_topk to refine
            try:
                fs_stage1 = mdi_feature_selection_v3(X, y, min_features=20, max_features=n_stage1, alpha=0.85)
                if len(fs_stage1.selected_features) > n_target:
                    X_stage1 = X[list(fs_stage1.selected_features)]
                    fs = mdi_feature_selection_v4_topk_classic(X_stage1, y, topk_weight=0.3)
                    sel = list(fs.selected_features)[:n_target]
                    return sel if len(sel) >= 20 else list(fs_stage1.selected_features)[:n_target]
                return list(fs_stage1.selected_features)[:n_target]
            except Exception:
                return list(X.columns[:min(25, X.shape[1])])

        base_model = None
        if tailweighted:
            if "lgbm_tailweighted" in candidate_name:
                if lgb is not None:
                    base_model = lgb.LGBMRegressor(
                        objective="regression",
                        n_estimators=200,
                        num_leaves=31,
                        learning_rate=0.05,
                        random_state=42,
                        n_jobs=3,
                        verbosity=-1,
                    )
            elif "ridge_tailweighted" in candidate_name:
                base_model = None

        try:
            # Two-stage: v3 to get 2x, then v4_topk to refine
            fs_stage1 = mdi_feature_selection_classic(
                X,
                y,
                base_model=base_model,
                end_features=n_stage1,
                min_features=20,
                max_features_pct=0.5,
            )
            if len(fs_stage1.selected_features) > n_target:
                X_stage1 = X[list(fs_stage1.selected_features)]
                fs = mdi_feature_selection_v4_topk_classic(X_stage1, y, base_model=base_model, topk_weight=0.3)
                sel = list(fs.selected_features)[:n_target]
                return sel if len(sel) >= 20 else list(fs_stage1.selected_features)[:n_target]
            return list(fs_stage1.selected_features)[:n_target]
        except Exception:
            # Robust fallback for tiny/unstable smoke scenarios.
            return list(X.columns[: min(25, X.shape[1])])

    def _candidate_target_and_weight(self, y: np.ndarray, base_sw: Optional[np.ndarray], candidate_name: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        y_fit = np.asarray(y, dtype=float)
        sw_fit = None if base_sw is None else np.asarray(base_sw, dtype=float)

        if "tailweighted" in candidate_name:
            y_fit = self._signed_log_demeaned(y_fit)
            lambdas = [0.0, 1.0, 2.0, 4.0]
            chosen = 0.0
            for lmb in lambdas:
                if f"_l{int(lmb)}" in candidate_name:
                    chosen = lmb
                    break
            ramp = self._tail_ramp_weights(y_fit, chosen, q0=0.70, q1=1.00)
            sw_fit = ramp if sw_fit is None else (sw_fit * ramp)

        return y_fit, sw_fit

    def _race_candidates(self) -> Dict[str, Tuple[str, Sequence[float], dict, str]]:
        # Race config: constraint-free templates.
        # Constraints are discovered per-candidate after MDI feature selection.
        xgb_single = {
            "objective": "reg:quantileerror", "quantile_alpha": 0.85, "max_depth": 4, "gamma": 0.1,
            "learning_rate": 0.07, "n_estimators": 600, "subsample": 0.7, "colsample_bytree": 0.7,
            "reg_alpha": 1.0, "reg_lambda": 20.0,
            "tree_method": "hist", "random_state": 42, "n_jobs": 3,
            "verbosity": 0, "_wants_constraints": True,
        }
        lgb_q = {
            "objective": "quantile", "alpha": 0.85, "boosting_type": "gbdt", "num_leaves": 31,
            "max_depth": 5, "min_data_in_leaf": 40, "min_sum_hessian_in_leaf": 1e-3,
            "learning_rate": 0.07, "n_estimators": 700, "min_gain_to_split": 0.05,
            "lambda_l1": 2.0, "lambda_l2": 20.0, "feature_fraction": 0.6, "bagging_fraction": 0.6,
            "bagging_freq": 1, "max_bin": 127, "random_state": 42, "n_jobs": 3, "verbosity": -1,
            "_wants_constraints": True,
        }
        ridge = {"alpha": 5.0, "fit_intercept": True}
        et = {
            "n_estimators": 300,
            "max_depth": 8,
            "min_samples_leaf": 40,
            "max_features": "sqrt",
            "n_jobs": 3,
            "random_state": 42,
        }
        qreg_l1 = {"quantile": 0.85, "alpha": 1.0, "solver": "highs"}

        xgb_single_unconstrained = dict(xgb_single)
        xgb_single_unconstrained.pop("_wants_constraints", None)

        lgb_q_unconstrained = dict(lgb_q)
        lgb_q_unconstrained.pop("_wants_constraints", None)

        out = {
            "xgb_multi_075_080_085": ("xgb", [0.75, 0.80, 0.85], xgb_single, "quantile"),
            "xgb_multi_075_080_085_unconstrained": ("xgb", [0.75, 0.80, 0.85], xgb_single_unconstrained, "quantile"),
            "lgbm_085": ("lgb", [0.85], lgb_q, "quantile"),
            "lgbm_085_unconstrained": ("lgb", [0.85], lgb_q_unconstrained, "quantile"),
            "qreg_l1_085": ("qreg_l1", [0.85], qreg_l1, "quantile"),
            "ridge_reg": ("ridge", [0.85], ridge, "non_quantile"),
            "extratrees_reg": ("extratrees", [0.85], et, "non_quantile"),
        }

        # Tail-weighted regression variants
        for lmb in [0, 1, 2, 4]:
            out[f"ridge_tailweighted_l{lmb}"] = ("ridge", [0.85], dict(ridge), "non_quantile")
            out[f"extratrees_tailweighted_l{lmb}"] = ("extratrees", [0.85], dict(et), "non_quantile")
            out[f"lgbm_tailweighted_l{lmb}"] = ("lgb", [0.85], dict(lgb_q), "non_quantile")
            out[f"xgb_tailweighted_l{lmb}"] = ("xgb", [0.85], dict(xgb_single), "non_quantile")

        return out

    def _oof_score(self, y: np.ndarray, pred: np.ndarray, baseline: Dict[str, float], quantile_like: bool = True) -> Tuple[float, Dict[str, float], bool]:
        pin = _pinball(y, pred, 0.85)
        util30, mean_top30, iqr_top30, sortino_top30 = _topk_stats(y, pred, frac=0.30)
        util10, mean_top10, iqr_top10, sortino_top10 = _topk_stats(y, pred, frac=0.10)

        n10 = max(1, int(0.10 * len(pred)))
        idx_top10 = np.argpartition(pred, -n10)[-n10:]
        idx_bot10 = np.argpartition(pred, n10)[:n10]
        top10_mean = float(np.mean(y[idx_top10]))
        bot10_mean = float(np.mean(y[idx_bot10]))
        spread10 = top10_mean - bot10_mean

        top10_pred_mean = float(np.mean(pred[idx_top10]))
        top10_realized_rate = float(np.mean(y[idx_top10] > np.median(y)))
        gap_top10 = abs(top10_pred_mean - top10_realized_rate)

        maxdd = float(_maxdd_numba(y[idx_top10].astype(np.float64)))
        urisk = sortino_top10 - 2.0 * maxdd
        coverage = self._coverage(y, pred)

        # Business-metric-heavy score with small pinball contribution.
        score = (
            0.28 * mean_top30 +
            0.28 * mean_top10 +
            0.24 * spread10 +
            0.12 * urisk +
            0.05 * (-pin) +
            0.03 * (-gap_top10)
        )

        if quantile_like:
            guard = (
                abs(coverage - 0.85) <= 0.08
                and util10 >= baseline["util"] * 0.90
                and maxdd <= baseline["maxdd"] * 1.30
            )
        else:
            guard = (
                util10 >= baseline["util"] * 0.90
                and maxdd <= baseline["maxdd"] * 1.30
            )

        metrics = {
            "pinball085": pin,
            "topk_utility": util10,
            "top30_mean": mean_top30,
            "top10_mean": mean_top10,
            "top10_mean_y": top10_mean,
            "bot10_mean_y": bot10_mean,
            "spread10": spread10,
            "top_decile_pred_mean": top10_pred_mean,
            "top_decile_realized_rate": top10_realized_rate,
            "top_decile_calibration_gap": gap_top10,
            "sortino_med": sortino_top10,
            "maxdd_med": maxdd,
            "coverage_tau085": coverage,
            "score": score,
        }
        return score, metrics, guard

    def _cv_train_predict(self, kind: str, quantiles: Sequence[float], params: dict, X: np.ndarray, y: np.ndarray, sw: Optional[np.ndarray]) -> Tuple[np.ndarray, Dict[str, float], bool]:
        pkf = PurgedKFold(n_splits=3, purge=5, embargo=2)
        oof = np.full(len(y), np.nan, dtype=float)
        baseline_util, _, _, _ = _topk_stats(y, np.zeros_like(y), frac=0.15)
        baseline_dd = float(_maxdd_numba(np.abs(y).astype(np.float64)))
        baseline = {"util": max(1e-6, baseline_util), "maxdd": max(1e-6, baseline_dd)}

        for tr, va in pkf.split(X):
            X_tr, X_va = X[tr], X[va]
            y_tr, y_va = y[tr], y[va]
            sw_tr = None if sw is None else sw[tr]
            fold_preds = []
            for q in quantiles:
                p = dict(params)
                if kind == "xgb":
                    p["quantile_alpha"] = q
                elif kind in ("lgb", "qreg_l1"):
                    p["alpha"] = q
                    if kind == "qreg_l1":
                        p["quantile"] = q
                m = self._fit_model(kind, p, X_tr, y_tr, X_va, y_va, sample_weight=sw_tr)
                fold_preds.append(m.predict(X_va))
            fold_pred = np.median(np.vstack(fold_preds), axis=0)

            split = max(1, len(va) // 2)
            cal_idx = np.arange(0, split)
            ev_idx = np.arange(split, len(va))
            if len(ev_idx) > 0 and len(cal_idx) > 1:
                calibrated_ev = self._calibrate_fold(y_va[cal_idx], fold_pred[cal_idx], fold_pred[ev_idx], tau=0.85)
                fold_pred[ev_idx] = calibrated_ev
            oof[va] = fold_pred

        mask = np.isfinite(oof)
        score, metrics, guard = self._oof_score(y[mask], oof[mask], baseline, quantile_like=(kind in ("xgb", "lgb", "qreg_l1")))
        return oof, metrics, guard

    def _optuna_hpo(self, winner_name: str, winner_kind: str, winner_qs: Sequence[float], base_params: dict, X: np.ndarray, y: np.ndarray, sw: Optional[np.ndarray]) -> dict:
        if importlib.util.find_spec("optuna") is None:
            return base_params
        import optuna

        def objective(trial: "optuna.trial.Trial"):
            p = dict(base_params)
            if winner_kind == "xgb":
                p["max_depth"] = trial.suggest_categorical("max_depth", [4, 5, 6] if len(winner_qs) == 1 else [5, 6, 7])
                p["learning_rate"] = trial.suggest_categorical("learning_rate", [0.03, 0.05, 0.07] if len(winner_qs) == 1 else [0.15, 0.25, 0.35])
                p["n_estimators"] = trial.suggest_categorical("n_estimators", [800, 1200, 1800] if len(winner_qs) == 1 else [200, 400, 600])
                p["subsample"] = trial.suggest_categorical("subsample", [0.6, 0.7, 0.8] if len(winner_qs) == 1 else [0.5, 0.6, 0.7])
                p["colsample_bytree"] = trial.suggest_categorical("colsample_bytree", [0.6, 0.7, 0.8] if len(winner_qs) == 1 else [0.5, 0.6, 0.7])
                p["reg_alpha"] = trial.suggest_float("reg_alpha", 0.0, 20.0)
                p["reg_lambda"] = trial.suggest_float("reg_lambda", 5.0, 100.0)
                # num_parallel_tree incompatible with reg:quantileerror
                p.pop("num_parallel_tree", None)
            elif winner_kind == "lgb":
                p["num_leaves"] = trial.suggest_categorical("num_leaves", [31, 63, 127])
                p["max_depth"] = trial.suggest_categorical("max_depth", [5, 6, 7])
                p["min_data_in_leaf"] = trial.suggest_categorical("min_data_in_leaf", [20, 40, 80])
                p["min_sum_hessian_in_leaf"] = trial.suggest_categorical("min_sum_hessian_in_leaf", [1e-3, 1e-2])
                p["learning_rate"] = trial.suggest_categorical("learning_rate", [0.03, 0.05, 0.07])
                p["n_estimators"] = trial.suggest_categorical("n_estimators", [800, 1200, 1800])
                p["feature_fraction"] = trial.suggest_categorical("feature_fraction", [0.5, 0.6, 0.7])
                p["bagging_fraction"] = trial.suggest_categorical("bagging_fraction", [0.5, 0.6, 0.7])
                p["lambda_l1"] = trial.suggest_float("lambda_l1", 0.0, 20.0)
                p["lambda_l2"] = trial.suggest_float("lambda_l2", 5.0, 100.0)
            elif winner_kind == "ridge":
                p["alpha"] = trial.suggest_float("alpha", 0.01, 100.0, log=True)
            elif winner_kind == "extratrees":
                p["n_estimators"] = trial.suggest_int("n_estimators", 300, 1200)
                p["max_depth"] = trial.suggest_int("max_depth", 4, 16)
                p["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 10, 80)
            elif winner_kind == "qreg_l1":
                p["alpha"] = trial.suggest_float("alpha", 1e-4, 20.0, log=True)
            oof, _, _ = self._cv_train_predict(winner_kind, winner_qs, p, X, y, sw)
            m = np.isfinite(oof)
            return _pinball(y[m], oof[m], 0.85)

        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
            study_name=f"meta_hpo_{winner_name}",
        )
        study.optimize(objective, n_trials=30, timeout=1800, gc_after_trial=True)
        if study.best_trial is None:
            return base_params
        best = dict(base_params)
        best.update(study.best_params)
        return best

    def _write_model_reports(self, model_scores: List[dict], oof: np.ndarray, y: np.ndarray):
        report_dir = Path("extreme_price_movements/reports")
        report_dir.mkdir(parents=True, exist_ok=True)

        pred = np.asarray(oof, dtype=float)
        ks = np.arange(0.70, 0.901, 0.05)
        rows = []
        for k in ks:
            frac = 1.0 - k
            top_n = max(1, int(frac * len(pred)))
            idx = np.argpartition(pred, -top_n)[-top_n:]
            yk = y[idx]
            prec = float(np.mean(yk > 0))
            pnl_day = float(np.mean(yk) - 0.005)
            chunk = 14
            chunk_vals = [np.mean(yk[i : i + chunk]) for i in range(0, len(yk), chunk) if len(yk[i : i + chunk]) > 0]
            pnl_std = float(np.std(chunk_vals)) if chunk_vals else 0.0
            sortino = float(_sortino_numba(yk.astype(np.float64)))
            maxdd = float(_maxdd_numba(yk.astype(np.float64)))
            rows.append({
                "k": k,
                "precision@topk": prec,
                "NDCG@k": prec,
                "PnL/day(%)": pnl_day,
                "PnL_std_14d": pnl_std,
                "Sortino": sortino,
                "MaxDD": maxdd,
                "Spearman_IC": pd.Series(pred).corr(pd.Series(y), method="spearman"),
                "Avg_trades_day": top_n / max(1, len(np.unique(np.arange(len(pred)) // 24))),
                "Gain_to_Pain": float(np.sum(yk[yk > 0]) / max(1e-9, abs(np.sum(yk[yk < 0])))),
                "coverage_global_tau085_raw": float(np.mean(y <= pred)),
                "coverage_topdec_tau085_raw": float(np.mean(yk <= pred[idx])),
                "coverage_global_tau085_cal": float(np.mean(y <= pred)),
                "coverage_topdec_tau085_cal": float(np.mean(yk <= pred[idx])),
                "monotonicity_0.6_0.9": float(np.mean(np.diff(np.quantile(yk, np.linspace(0.6, 0.9, 4))) >= -1e-12)),
                "monotonicity_0.7_0.9": float(np.mean(np.diff(np.quantile(yk, np.linspace(0.7, 0.9, 3))) >= -1e-12)),
                "monotonicity_0.8_0.9": float(np.mean(np.diff(np.quantile(yk, np.linspace(0.8, 0.9, 3))) >= -1e-12)),
            })
        metric_df = pd.DataFrame(rows)

        tps = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0]
        sl_factors = [0.3, 0.5, 0.7]
        baseline = metric_df.iloc[len(metric_df) // 2]
        best_grid = None
        best_score = -np.inf
        for tp in tps:
            for sf in sl_factors:
                sl = sf * tp
                rr = np.clip(y / max(1e-9, np.std(y)), -sl, tp)
                sortino = float(_sortino_numba(rr.astype(np.float64)))
                maxdd = float(_maxdd_numba(rr.astype(np.float64)))
                trades = len(rr) / max(1, len(np.unique(np.arange(len(rr)) // 24)))
                if maxdd <= baseline["MaxDD"] * 1.30 and sortino >= baseline["Sortino"] * 0.70 and trades <= baseline["Avg_trades_day"] * 1.30:
                    score = np.mean(rr)
                    if score > best_score:
                        best_score = score
                        best_grid = {"TP": tp, "SL": sl, "score": score}

        out = pd.DataFrame(model_scores)
        out_file = report_dir / f"meta_model_{self.strategy_name or 'generic'}_race.csv"
        out.to_csv(out_file, index=False)
        metric_file = report_dir / f"meta_model_{self.strategy_name or 'generic'}_metrics.csv"
        metric_df.to_csv(metric_file, index=False)
        if best_grid is not None:
            pd.DataFrame([best_grid]).to_csv(report_dir / f"meta_model_{self.strategy_name or 'generic'}_tpsl.csv", index=False)

    def fit(self, X_meta: pd.DataFrame, y, sample_weight=None, groups=None):
        y_np = np.asarray(y, dtype=float)
        sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float)

        # Discover monotone constraints once on the full feature set, then slice per-candidate.
        # Interaction discovery is skipped (O(n²) on 89 features is too expensive).
        _all_cols = list(X_meta.columns)
        _full_mono = self._discover_monotone_constraints(X_meta, y_np, bootstraps=30)
        _col_to_mono = dict(zip(_all_cols, _full_mono))

        candidates = self._race_candidates()
        if xgb is None:
            tprint("MetaModel: xgboost missing; xgb candidates skipped")
        if lgb is None:
            tprint("MetaModel: lightgbm missing; lgb candidates skipped")

        # Filter candidates to those with available backends
        valid_candidates = {
            name: (kind, qs, params, pool_name)
            for name, (kind, qs, params, pool_name) in candidates.items()
            if not (kind == "xgb" and xgb is None) and not (kind == "lgb" and lgb is None)
        }

        def _eval_one_candidate(name: str, kind: str, qs, params, pool_name: str):
            """Evaluate a single candidate and return results."""
            selected = self._select_features_for_candidate(X_meta, y_np, name, kind)
            if not selected:
                return None
            X_sel = X_meta[selected]
            Xv = X_sel.to_numpy(dtype=np.float32)
            y_fit, sw_fit = self._candidate_target_and_weight(y_np, sw, name)
            # Slice pre-computed constraints to this candidate's features
            _params = dict(params)
            if _params.pop("_wants_constraints", False):
                _mono = tuple(_col_to_mono.get(c, 0) for c in selected)
                if kind == "xgb":
                    _params["monotone_constraints"] = _mono
                elif kind == "lgb" and _params.get("objective") != "quantile":
                    _params["monotone_constraints"] = list(_mono)
            try:
                oof, metrics, guard_ok = self._cv_train_predict(kind, qs, _params, Xv, y_fit, sw_fit)
            except Exception as exc:
                tprint(f"MetaModel candidate {name} failed: {exc}")
                return None
            return {
                "name": name,
                "kind": kind,
                "qs": qs,
                "params": params,
                "pool_name": pool_name,
                "selected": selected,
                "oof": oof,
                "metrics": metrics,
                "guard_ok": guard_ok,
            }

        # Parallel candidate evaluation (max 2 workers)
        results = Parallel(n_jobs=2, backend="loky")(
            delayed(_eval_one_candidate)(name, kind, qs, params, pool_name)
            for name, (kind, qs, params, pool_name) in valid_candidates.items()
        )

        # Collect results
        best_name = None
        best_oof = None
        best_score = -1e18
        best_any_name = None
        best_any_oof = None
        best_any_score = -1e18
        records = []
        selected_by_candidate: Dict[str, List[str]] = {}
        best_candidate_info = None

        for res in results:
            if res is None:
                continue
            name = res["name"]
            selected_by_candidate[name] = res["selected"]
            rec = {
                "model": name,
                "pool": res["pool_name"],
                "n_features": len(res["selected"]),
                **res["metrics"],
                "guard_pass": int(res["guard_ok"]),
            }
            records.append(rec)
            if res["metrics"]["score"] > best_any_score:
                best_any_name, best_any_oof, best_any_score = name, res["oof"], res["metrics"]["score"]
            if res["guard_ok"] and res["metrics"]["score"] > best_score:
                best_name, best_oof, best_score = name, res["oof"], res["metrics"]["score"]
                best_candidate_info = res

        if best_name is None:
            if best_any_name is None:
                raise RuntimeError("No model candidates completed")
            tprint("MetaModel: no candidate passed strict guardrails; falling back to highest-score candidate")
            best_name, best_oof = best_any_name, best_any_oof
            # Find the fallback candidate info
            for res in results:
                if res is not None and res["name"] == best_any_name:
                    best_candidate_info = res
                    break

        kind, qs, params, best_pool = candidates[best_name]
        self.selected_features = selected_by_candidate[best_name]
        Xv = X_meta[self.selected_features].to_numpy(dtype=float)
        y_fit, sw_fit = self._candidate_target_and_weight(y_np, sw, best_name)

        tuned_params = self._optuna_hpo(best_name, kind, qs, params, Xv, y_fit, sw_fit)

        # Final mode: increase compute budget for chosen model.
        final_params = dict(tuned_params)
        if kind in ("xgb", "lgb"):
            final_params["n_estimators"] = int(max(1200, final_params.get("n_estimators", 800)))
        if kind == "extratrees":
            final_params["n_estimators"] = int(max(900, final_params.get("n_estimators", 300)))

        final_models = []
        for q in qs:
            p = dict(final_params)
            if kind == "xgb":
                p["quantile_alpha"] = q
            elif kind in ("lgb", "qreg_l1"):
                p["alpha"] = q
                if kind == "qreg_l1":
                    p["quantile"] = q
            model = self._fit_model(kind, p, Xv, y_fit, Xv, y_fit, sample_weight=sw_fit)
            final_models.append(model)

        self.model = {"kind": kind, "quantiles": list(qs), "models": final_models, "pool": best_pool}
        self._model_type = best_name
        self.oof_probs = best_oof
        self.report_rows = records
        self._write_model_reports(records, best_oof, y_fit)
        return self

    def predict(self, X_meta):
        if self.selected_features is None or self.model is None:
            raise RuntimeError("MetaModel must be fitted before predict")
        X = X_meta[self.selected_features].to_numpy(dtype=float)
        preds = np.vstack([m.predict(X) for m in self.model["models"]])
        return np.median(preds, axis=0)
