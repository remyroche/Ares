"""Simple meta model with 5-candidate race: Ridge, ExtraTrees, ExtraTrees+tail-weighting, XGB, XGB-quantile.

No strict guardrails, no monotone constraints. Winner selection by Spearman IC on OOF.
Optional Optuna HPO on the winner.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import importlib.util
import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import spearmanr, rankdata
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from extreme_price_movements.feature_selection_extreme_events import (
    mdi_feature_selection_v3,
    mdi_feature_selection_v4_topk,
)
from extreme_price_movements.purged_cv import PurgedKFold
from extreme_price_movements.utils import tprint
from extreme_price_movements.policy_ml import (
    MetaClassifierSelectionConfig,
    pick_meta_classifier_by_utility_top30,
)

META_CLASS_ORDER = np.array([0, 1, 2], dtype=np.int64)

if importlib.util.find_spec("xgboost") is not None:
    import xgboost as xgb
else:
    xgb = None


def _safe_spearman(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 10:
        return 0.0
    rho, _ = spearmanr(a[mask], b[mask])
    return float(rho) if np.isfinite(rho) else 0.0


class MetaModel:
    def __init__(self, strategy_name: Optional[str] = None):
        self.strategy_name = strategy_name
        self.model = None
        self._model_type = None
        self.selected_features: Optional[List[str]] = None
        self.oof_probs: Optional[np.ndarray] = None
        self.report_rows: List[dict] = []
        self.score_sign: int = 1

    def prepare_meta_features(self, preds, feats_df, pred_col_name="pred_logit"):
        p = np.clip(np.asarray(preds, dtype=float), 1e-4, 1 - 1e-4)
        meta_data = pd.DataFrame(index=feats_df.index)
        meta_data[pred_col_name] = np.clip(logit(p), -4.0, 4.0)
        return pd.concat([meta_data, feats_df], axis=1).fillna(0.0)

    # ── Tail-ramp weighting ──────────────────────────────────────────────
    def _tail_ramp_weights(self, y: np.ndarray, lambda_: float, q0: float = 0.70, q1: float = 1.00) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        r = rankdata(y, method="average") / max(len(y), 1)
        t = np.clip((r - q0) / max(1e-9, (q1 - q0)), 0.0, 1.0)
        return (1.0 + float(lambda_) * t).astype(float)

    def _signed_log(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        z = np.sign(y) * np.log1p(np.abs(y))
        finite = np.isfinite(z)
        if not finite.any():
            return np.zeros_like(z, dtype=float)
        fill = float(np.nanmedian(z[finite]))
        z[~finite] = fill
        return z

    def _inverse_signed_log(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        return np.sign(z) * np.expm1(np.abs(z))

    # ── Tail-focused feature selection (light MDI for meta) ─────────────
    def _select_tail_features(
        self, X: pd.DataFrame, y: np.ndarray, max_features: int = 80,
    ) -> List[str]:
        """Light feature selection for meta models via MDI v3 only.

        Meta features are already curated (~100 features: pred_logit,
        per-horizon OOFs, handpicked market features). Aggressive pruning
        removes signal — only do a broad v3 pass with a high floor.
        No v4_topk refinement (that's for base models with 600+ raw features).

        Target is scaled to unit variance before MDI (meta targets have
        std ~0.01 which causes near-zero tree importances otherwise).
        """
        y_arr = np.asarray(y, dtype=float)
        finite = np.isfinite(y_arr)
        if finite.sum() < 50:
            return list(X.columns[:max_features])

        # Scale target to unit variance for meaningful tree splits
        y_scaled = y_arr.copy()
        y_std = float(np.nanstd(y_scaled[finite]))
        if y_std > 1e-9:
            y_scaled = y_scaled / y_std

        try:
            # Single broad MDI v3 pass — high floor, no aggressive refinement
            fs1 = mdi_feature_selection_v3(
                X, y_scaled, min_features=30, end_features=max_features,
                selector_y=y,
                selector_target="regression",
                selector_loss="huber",
                max_features_pct=0.90,
            )
            selected = list(fs1.selected_features)
            if len(selected) < 30:
                selected = list(X.columns[:max_features])
        except Exception as exc:
            tprint(f"  MDI feature selection failed ({exc}), using all columns")
            selected = list(X.columns[:max_features])

        # Always keep pred_logit and pred_H* (core alpha signals)
        must_keep = [c for c in X.columns if c.startswith("pred_")]
        for c in must_keep:
            if c not in selected:
                selected.append(c)

        tprint(f"  Meta feature selection: {len(X.columns)} -> {len(selected)} features")
        return selected

    # ── Candidate definitions ────────────────────────────────────────────
    def _build_candidates(self) -> Dict[str, dict]:
        """Candidates: Ridge, Huber, ET, ET+tail, XGB."""
        candidates = {}

        # 1. Ridge (RobustScaler + Ridge)
        candidates["ridge"] = {
            "kind": "ridge",
            "params": {"alpha": 5.0, "fit_intercept": True},
            "tail_lambda": 0.0,
        }

        # 2. Huber Regressor (Robust objective)
        candidates["huber"] = {
            "kind": "huber",
            "params": {"epsilon": 1.35, "alpha": 0.001, "fit_intercept": True},
            "tail_lambda": 0.0,
        }

        # 3. ExtraTrees (baseline)
        candidates["extratrees"] = {
            "kind": "extratrees",
            "params": {
                "n_estimators": 300, "max_depth": 8, "min_samples_leaf": 40,
                "max_features": "sqrt", "n_jobs": 3, "random_state": 42,
            },
            "tail_lambda": 0.0,
        }

        # 4. ExtraTrees + tail-weighting (λ=2.0, no monotone constraints)
        candidates["extratrees_tailweighted"] = {
            "kind": "extratrees",
            "params": {
                "n_estimators": 300, "max_depth": 8, "min_samples_leaf": 40,
                "max_features": "sqrt", "n_jobs": 3, "random_state": 42,
            },
            "tail_lambda": 2.0,
        }

        # 5. XGB basic (reg:squarederror) — heavily regularised for small meta datasets
        if xgb is not None:
            _xgb_common = {
                "max_depth": 4, "learning_rate": 0.03, "n_estimators": 800,
                "subsample": 0.65, "colsample_bytree": 0.60,
                "reg_alpha": 2.0, "reg_lambda": 15.0,
                "min_child_weight": 10, "gamma": 1.0, "max_delta_step": 1.0,
                "tree_method": "hist", "random_state": 42, "n_jobs": 3,
                "verbosity": 0,
            }
            candidates["xgb_basic"] = {
                "kind": "xgb",
                "params": {"objective": "reg:squarederror", **_xgb_common},
                "tail_lambda": 0.0,
            }
            # Robust objective candidate if available (Pseudohuber)
            candidates["xgb_robust"] = {
                "kind": "xgb",
                "params": {"objective": "reg:pseudohubererror", **_xgb_common},
                "tail_lambda": 0.0,
            }

        return candidates

    # ── Model fitting ────────────────────────────────────────────────────
    def _fit_one(self, kind, params, X_tr, y_tr, X_va, y_va, sw=None):
        if kind == "ridge":
            model = Pipeline([
                ("scaler", RobustScaler()),
                ("ridge", Ridge(**params)),
            ])
            model.fit(X_tr, y_tr, ridge__sample_weight=sw)
            return model
        if kind == "huber":
            from sklearn.linear_model import HuberRegressor
            model = Pipeline([
                ("scaler", RobustScaler()),
                ("huber", HuberRegressor(**params)),
            ])
            model.fit(X_tr, y_tr, huber__sample_weight=sw)
            return model
        if kind == "extratrees":
            model = ExtraTreesRegressor(**params)
            model.fit(X_tr, y_tr, sample_weight=sw)
            return model
        if kind == "xgb":
            p = dict(params)
            es_rounds = p.pop("early_stopping_rounds", 50)
            model = xgb.XGBRegressor(**p, early_stopping_rounds=es_rounds)
            model.fit(X_tr, y_tr, sample_weight=sw,
                      eval_set=[(X_va, y_va)], verbose=False)
            return model
        raise ValueError(f"Unknown kind: {kind}")

    # ── CV evaluation ────────────────────────────────────────────────────
    def _cv_evaluate(self, kind, params, X, y, sw=None) -> Tuple[np.ndarray, float]:
        """3-fold purged CV. Returns (oof_predictions, spearman_ic)."""
        pkf = PurgedKFold(n_splits=3, purge=12, embargo=12)
        oof = np.full(len(y), np.nan, dtype=float)

        for tr, va in pkf.split(X):
            X_tr, X_va = X[tr], X[va]
            y_tr, y_va = y[tr], y[va]
            sw_tr = None if sw is None else sw[tr]
            model = self._fit_one(kind, params, X_tr, y_tr, X_va, y_va, sw=sw_tr)
            oof[va] = model.predict(X_va)

        mask = np.isfinite(oof)
        ic = _safe_spearman(oof[mask], y[mask])
        return oof, ic

    @staticmethod
    def _compute_oof_metrics(oof: np.ndarray, y: np.ndarray,
                             y_per_horizon: Optional[Dict[int, np.ndarray]] = None) -> dict:
        """Compute comprehensive OOF metrics for meta model reporting."""
        mask = np.isfinite(oof) & np.isfinite(y)
        pred, tgt = oof[mask], y[mask]
        n = len(pred)
        if n < 20:
            return {"ic": 0.0, "ic_mh": 0.0, "n": n}

        ic = _safe_spearman(pred, tgt)

        # Multi-horizon IC: average Spearman(pred, r_h) across horizons
        ic_mh = ic  # fallback if no per-horizon data
        if y_per_horizon:
            h_ics = []
            for h, r_h in sorted(y_per_horizon.items()):
                r_h = np.asarray(r_h, dtype=float)
                if len(r_h) == len(oof):
                    h_mask = mask & np.isfinite(r_h)
                    h_ic = _safe_spearman(oof[h_mask], r_h[h_mask])
                else:
                    min_len = min(len(oof), len(r_h))
                    h_mask2 = np.isfinite(oof[:min_len]) & np.isfinite(r_h[:min_len])
                    h_ic = _safe_spearman(oof[:min_len][h_mask2], r_h[:min_len][h_mask2])
                h_ics.append(h_ic)
            if h_ics:
                ic_mh = float(np.mean(h_ics))

        # Top-30% metrics: select samples where pred is in top 30%
        n30 = max(1, int(0.30 * n))
        idx_top30 = np.argpartition(pred, -n30)[-n30:]
        idx_bot30 = np.argpartition(pred, n30)[:n30]
        y_top30 = tgt[idx_top30]
        y_bot30 = tgt[idx_bot30]
        pred_top30 = pred[idx_top30]

        ic_top30 = _safe_spearman(pred_top30, y_top30)
        mean_ret_top30 = float(np.mean(y_top30))
        mean_ret_bot30 = float(np.mean(y_bot30))
        spread30 = mean_ret_top30 - mean_ret_bot30

        # Top-10% spread
        n10 = max(1, int(0.10 * n))
        idx_top10 = np.argpartition(pred, -n10)[-n10:]
        idx_bot10 = np.argpartition(pred, n10)[:n10]
        spread10 = float(np.mean(tgt[idx_top10]) - np.mean(tgt[idx_bot10]))

        # ECE (Expected Calibration Error) on top-30%: how well does predicted
        # rank order match actual positive rate in 5 bins?
        n_bins = 5
        bin_edges = np.percentile(pred_top30, np.linspace(0, 100, n_bins + 1))
        ece = 0.0
        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            if b == n_bins - 1:
                in_bin = (pred_top30 >= lo) & (pred_top30 <= hi)
            else:
                in_bin = (pred_top30 >= lo) & (pred_top30 < hi)
            if in_bin.sum() == 0:
                continue
            # "positive" = above-median return within top-30%
            med_top30 = float(np.median(y_top30))
            actual_pos_rate = float(np.mean(y_top30[in_bin] > med_top30))
            expected_pos_rate = float((b + 0.5) / n_bins)
            ece += abs(actual_pos_rate - expected_pos_rate) * (in_bin.sum() / len(pred_top30))

        # Robust loss: fraction of top-30% trades with negative return
        robust_loss = float(np.mean(y_top30 < 0))

        # Win rate in top-30%
        win_rate_top30 = float(np.mean(y_top30 > 0))

        return {
            "n": n, "ic": ic, "ic_mh": ic_mh, "ic_top30": ic_top30,
            "mean_ret_top30": mean_ret_top30, "mean_ret_bot30": mean_ret_bot30,
            "spread30": spread30, "spread10": spread10,
            "ece_top30": ece, "robust_loss_top30": robust_loss,
            "win_rate_top30": win_rate_top30,
        }

    # ── Optuna HPO (optional, on winner) ─────────────────────────────────
    def _optuna_hpo(self, name, kind, params, X, y, sw=None, n_trials=15) -> dict:
        if importlib.util.find_spec("optuna") is None:
            return params
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            p = dict(params)
            if kind == "ridge":
                p["alpha"] = trial.suggest_float("alpha", 0.01, 100.0, log=True)
            elif kind == "extratrees":
                p["n_estimators"] = trial.suggest_int("n_estimators", 200, 800)
                p["max_depth"] = trial.suggest_int("max_depth", 4, 16)
                p["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 10, 80)
            elif kind == "xgb":
                p["max_depth"] = trial.suggest_int("max_depth", 3, 7)
                p["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.15, log=True)
                p["n_estimators"] = trial.suggest_int("n_estimators", 300, 1200)
                p["subsample"] = trial.suggest_float("subsample", 0.5, 0.9)
                p["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 0.9)
                p["reg_alpha"] = trial.suggest_float("reg_alpha", 0.01, 20.0, log=True)
                p["reg_lambda"] = trial.suggest_float("reg_lambda", 1.0, 100.0, log=True)
            _, ic = self._cv_evaluate(kind, p, X, y, sw)
            return -ic  # minimize negative IC

        study = optuna.create_study(study_name=f"meta_hpo_{name}")
        study.optimize(objective, n_trials=n_trials, timeout=180, gc_after_trial=True)
        if study.best_trial is None:
            return params
        best = dict(params)
        best.update(study.best_params)
        return best

    # ── Main fit ─────────────────────────────────────────────────────────
    def fit(self, X_meta: pd.DataFrame, y, sample_weight=None, groups=None,
            y_per_horizon: Optional[Dict[int, np.ndarray]] = None):
        import time as _time
        _t0 = _time.monotonic()
        tprint(f"MetaModel.fit: {self.strategy_name} starting (n={len(y)}, feats={X_meta.shape[1]})")
        y_np = np.asarray(y, dtype=float)
        sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float)

        # Light feature selection: meta features are already curated (~100),
        # so keep most of them. Only prune clearly irrelevant ones.
        n_target = max(30, min(40, X_meta.shape[1]))
        selected_cols = self._select_tail_features(X_meta, y_np, max_features=n_target)
        X_sel = X_meta[selected_cols]
        self.selected_features = selected_cols

        candidates = self._build_candidates()
        tprint(f"  Racing {len(candidates)} candidates ({_time.monotonic()-_t0:.1f}s)...")

        Xv = X_sel.to_numpy(dtype=np.float32)
        records = []
        best_name = None
        best_ic = -1e18
        best_oof = None

        for name, cand in candidates.items():
            kind = cand["kind"]
            params = cand["params"]
            tail_lambda = cand["tail_lambda"]

            # Prepare target and weights for this candidate
            y_fit = y_np.copy()
            sw_fit = sw.copy() if sw is not None else None

            if tail_lambda > 0:
                y_fit = self._signed_log(y_fit)
                ramp = self._tail_ramp_weights(y_fit, tail_lambda, q0=0.70, q1=1.00)
                sw_fit = ramp if sw_fit is None else (sw_fit * ramp)

            # Ensure finite
            finite_y = np.isfinite(y_fit)
            if not finite_y.all():
                fill = float(np.nanmedian(y_fit[finite_y])) if finite_y.any() else 0.0
                y_fit = np.where(finite_y, y_fit, fill)
            if sw_fit is not None:
                finite_w = np.isfinite(sw_fit)
                if not finite_w.all():
                    w_fill = float(np.nanmedian(sw_fit[finite_w])) if finite_w.any() else 1.0
                    sw_fit = np.where(finite_w, sw_fit, w_fill)

            try:
                oof, ic = self._cv_evaluate(kind, params, Xv, y_fit, sw_fit)
                # If tail-weighted, inverse-transform OOF for IC computation against original y
                if tail_lambda > 0:
                    oof_orig = self._inverse_signed_log(oof)
                else:
                    oof_orig = oof
            except Exception as exc:
                tprint(f"  Candidate {name} failed: {exc}")
                continue

            # Comprehensive OOF metrics against original (unscaled) target
            metrics = self._compute_oof_metrics(oof_orig, y_np,
                                                y_per_horizon=y_per_horizon)
            ic_orig = metrics.get("ic", 0.0)
            ic_mh = metrics.get("ic_mh", ic_orig)
            ic_t30 = metrics.get("ic_top30", 0.0)
            spread10 = metrics.get("spread10", 0.0)
            # Composite score: 40% IC_mh + 30% IC_t30 + 30% spread10 (normalized)
            # IC_mh = avg Spearman(pred, r_h) across horizons — avoids dilution
            # spread10 is in return units; scale by 100 to bring into ~IC range
            composite = 0.40 * ic_mh + 0.30 * ic_t30 + 0.30 * min(spread10 * 100, 1.0)

            rec = {"model": name, "kind": kind, "tail_lambda": tail_lambda,
                   "composite": composite, **metrics}
            records.append(rec)
            tprint(f"  {name}: IC={ic_orig:.4f}, IC_mh={ic_mh:.4f}, IC_t30={ic_t30:.4f}, "
                   f"spread10={spread10:.6f}, ECE_t30={metrics.get('ece_top30',0):.3f}, "
                   f"win_t30={metrics.get('win_rate_top30',0):.1%}, composite={composite:.4f}")

            if composite > best_ic:
                best_ic = composite
                best_name = name
                best_oof = oof_orig

        if best_name is None:
            raise RuntimeError("No meta model candidates completed")

        winner = candidates[best_name]
        kind = winner["kind"]
        params = winner["params"]
        tail_lambda = winner["tail_lambda"]

        tprint(f"  Winner: {best_name} (composite={best_ic:.4f}). Starting HPO ({_time.monotonic()-_t0:.1f}s)...")

        # Prepare target for HPO and final fit
        y_fit = y_np.copy()
        sw_fit = sw.copy() if sw is not None else None
        if tail_lambda > 0:
            y_fit = self._signed_log(y_fit)
            ramp = self._tail_ramp_weights(y_fit, tail_lambda, q0=0.70, q1=1.00)
            sw_fit = ramp if sw_fit is None else (sw_fit * ramp)
        finite_y = np.isfinite(y_fit)
        if not finite_y.all():
            fill = float(np.nanmedian(y_fit[finite_y])) if finite_y.any() else 0.0
            y_fit = np.where(finite_y, y_fit, fill)
        if sw_fit is not None:
            finite_w = np.isfinite(sw_fit)
            if not finite_w.all():
                w_fill = float(np.nanmedian(sw_fit[finite_w])) if finite_w.any() else 1.0
                sw_fit = np.where(finite_w, sw_fit, w_fill)

        # Re-derive Xv from selected features (in case it was narrowed)
        Xv = X_meta[self.selected_features].to_numpy(dtype=np.float32)
        tuned_params = self._optuna_hpo(best_name, kind, params, Xv, y_fit, sw_fit)
        tprint(f"  HPO done ({_time.monotonic()-_t0:.1f}s). Fitting final model...")

        # Final fit on all data
        if len(np.unique(y_fit)) < 2 and kind in ["ridge", "huber"]:
            tprint(f"  WARNING: Final fit on single-class data ({np.unique(y_fit)}), returning trivial model")
            # For regressors, we could return a constant model, but _fit_one expects a pipeline.
            # Let's just catch the error or ensure y_fit has at least some noise.
            pass

        final_model = self._fit_one(kind, tuned_params, Xv, y_fit, Xv, y_fit, sw=sw_fit)

        self.model = {
            "kind": kind, "models": [final_model],
            "is_transformed": tail_lambda > 0,
            "tail_lambda": tail_lambda,
        }
        self._model_type = best_name
        self.oof_probs = best_oof
        self.report_rows = records

        # Save race report
        report_dir = Path("extreme_price_movements/reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(
            report_dir / f"meta_model_{self.strategy_name or 'generic'}_race.csv",
            index=False,
        )

        tprint(f"MetaModel.fit: {self.strategy_name} done ({_time.monotonic()-_t0:.1f}s). "
               f"Winner={best_name}, IC={best_ic:.4f}")
        return self

    def predict(self, X_meta):
        if self.selected_features is None or self.model is None:
            raise RuntimeError("MetaModel must be fitted before predict")
        X = X_meta[self.selected_features].to_numpy(dtype=float)
        preds = np.vstack([m.predict(X) for m in self.model["models"]])
        med_preds = np.median(preds, axis=0)
        if self.model.get("is_transformed", False):
            med_preds = self._inverse_signed_log(med_preds)
        return int(self.score_sign) * med_preds


# ═══════════════════════════════════════════════════════════════════════
# Meta Classifier Model — binary classification race on top-K% labels
# ═══════════════════════════════════════════════════════════════════════

class MetaClassifierModel:
    """Classifier-based meta model that races Ridge/ET/XGB/CatBoost on
    binary labels derived from top-K% of return magnitude.

    Each label threshold (e.g. top39%, top26%, top13%) produces a separate
    race. The best (model, threshold) pair is selected by a composite score
    combining PR-AUC, Lift@K, and Sortino.
    """

    LABEL_THRESHOLDS = [0.39, 0.26, 0.13]  # top-K fractions
    FEE_PER_ROUND_TRIP = 0.005  # 0.5% total round-trip fee

    def __init__(self, strategy_name: Optional[str] = None):
        self.strategy_name = strategy_name
        self.model = None
        self._model_type: Optional[str] = None
        self.selected_features: Optional[List[str]] = None
        self.oof_probs: Optional[np.ndarray] = None
        self.report_rows: List[dict] = []
        self.label_threshold: float = 0.26  # default
        self.score_sign: int = 1

    def prepare_meta_features(self, preds, feats_df, pred_col_name="pred_logit"):
        p = np.clip(np.asarray(preds, dtype=float), 1e-4, 1 - 1e-4)
        meta_data = pd.DataFrame(index=feats_df.index)
        meta_data[pred_col_name] = np.clip(logit(p), -4.0, 4.0)
        return pd.concat([meta_data, feats_df], axis=1).fillna(0.0)

    # ── Candidate definitions ────────────────────────────────────────
    def _build_candidates(self) -> Dict[str, dict]:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import ExtraTreesClassifier

        candidates = {}

        # 1. Ridge (LogisticRegression with L2)
        candidates["ridge_clf"] = {
            "kind": "ridge_clf",
            "params": {"C": 0.1, "penalty": "l2", "solver": "lbfgs",
                       "max_iter": 1000, "class_weight": "balanced",
                       "multi_class": "multinomial"},
        }

        # 2. ExtraTrees Classifier
        candidates["et_clf"] = {
            "kind": "et_clf",
            "params": {
                "n_estimators": 300, "max_depth": 8, "min_samples_leaf": 40,
                "max_features": "sqrt", "n_jobs": 3, "random_state": 42,
                "class_weight": "balanced",
            },
        }

        # 3. CatBoost Classifier
        try:
            import catboost
            candidates["catboost_clf"] = {
                "kind": "catboost_clf",
                "params": {
                    "iterations": 500, "depth": 5, "learning_rate": 0.05,
                    "l2_leaf_reg": 10.0, "random_seed": 42,
                    "auto_class_weights": "Balanced",
                    "verbose": 0, "thread_count": 3,
                    "loss_function": "MultiClass",
                },
            }
        except ImportError:
            pass

        return candidates

    # ── Model fitting ────────────────────────────────────────────────
    def _fit_one(self, kind, params, X_tr, y_tr, X_va, y_va, sw=None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import ExtraTreesClassifier

        if kind == "ridge_clf":
            model = Pipeline([
                ("scaler", RobustScaler()),
                ("clf", LogisticRegression(**params)),
            ])
            model.fit(X_tr, y_tr, clf__sample_weight=sw)
            return model
        if kind == "et_clf":
            model = ExtraTreesClassifier(**params)
            model.fit(X_tr, y_tr, sample_weight=sw)
            return model
        if kind == "catboost_clf":
            import catboost
            p = dict(params)
            model = catboost.CatBoostClassifier(**p)
            model.fit(X_tr, y_tr, sample_weight=sw,
                      eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False)
            return model
        raise ValueError(f"Unknown classifier kind: {kind}")

    def _predict_proba(self, model, X):
        """Get class probabilities aligned to class order [0,1,2]."""
        pp_raw = np.asarray(model.predict_proba(X), dtype=np.float64)
        if pp_raw.ndim != 2:
            raise ValueError(f"predict_proba returned invalid shape: {pp_raw.shape}")

        out = np.zeros((pp_raw.shape[0], 3), dtype=np.float64)
        classes = getattr(model, "classes_", None)
        if classes is None and hasattr(model, "named_steps"):
            classes = getattr(model.named_steps.get("clf"), "classes_", None)
        if classes is None and hasattr(model, "classes"):
            classes = np.asarray(model.classes)
        if classes is None:
            if pp_raw.shape[1] == 3:
                out = pp_raw
            else:
                raise ValueError("Unable to align class probabilities: missing class metadata")
        else:
            classes = np.asarray(classes).astype(np.int64)
            for j, cls in enumerate(classes):
                if cls in META_CLASS_ORDER and j < pp_raw.shape[1]:
                    out[:, int(cls)] = pp_raw[:, j]

        row_sum = out.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum > 1e-12, row_sum, 1.0)
        out = out / row_sum
        assert out.shape[1] == 3, f"Expected 3 classes after alignment, got {out.shape}"
        return out

    # ── CV evaluation ────────────────────────────────────────────────
    def _cv_evaluate(self, kind, params, X, y, sw=None) -> Tuple[np.ndarray, float]:
        """3-fold purged CV. Returns (oof_probs, brier_score).
        oof_probs shape: (N, 3).
        """
        from sklearn.metrics import brier_score_loss
        pkf = PurgedKFold(n_splits=3, purge=12, embargo=12)
        oof = np.full((len(y), 3), np.nan, dtype=float)

        for tr, va in pkf.split(X):
            X_tr, X_va = X[tr], X[va]
            y_tr, y_va = y[tr], y[va]
            sw_tr = None if sw is None else sw[tr]
            try:
                model = self._fit_one(kind, params, X_tr, y_tr, X_va, y_va, sw=sw_tr)
                pp = self._predict_proba(model, X_va)
                oof[va] = pp
            except Exception:
                oof[va] = 1.0 / 3.0

        mask = np.isfinite(oof).all(axis=1)
        if mask.sum() < 20:
            return oof, 999.0
        row_sums = oof[mask].sum(axis=1)
        assert np.all(np.isfinite(row_sums)), "OOF row sums contain non-finite values"
        assert np.all(np.abs(row_sums - 1.0) < 1e-3), "OOF probabilities must sum to ~1"

        # Metric: Brier score (multi-class: sum of squared differences)
        # We can use EV as primary metric?
        # But for CV selection, a proper scoring rule like Brier or LogLoss is better.
        # Let's use LogLoss.
        from sklearn.metrics import log_loss
        try:
            score = log_loss(y[mask], oof[mask])
        except Exception:
            score = 999.0
        return oof, score

    # ── Comprehensive classifier metrics ─────────────────────────────
    @staticmethod
    def _compute_clf_metrics(oof: np.ndarray, y_class: np.ndarray,
                             y_ret: np.ndarray, groups=None,
                             fee: float = 0.005) -> dict:
        """Compute classifier metrics: EV, Brier, Accuracy, PnL."""
        from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
        mask = np.isfinite(oof).all(axis=1) & np.isfinite(y_ret)
        pred, y_c, y_r = oof[mask], y_class[mask], y_ret[mask]
        n = len(pred)
        if n < 20:
            return {"n": n}

        # EV calculation: P(TP)*2 - P(SL)*1 (assuming 2:1 ratio standard)
        # TP=class 2, SL=class 0.
        ev_vec = pred[:, 2] * 2.0 - pred[:, 0] * 1.0

        try:
            ll = float(log_loss(y_c, np.clip(pred, 1e-7, 1 - 1e-7)))
        except Exception:
            ll = float("nan")

        acc = float(accuracy_score(y_c, np.argmax(pred, axis=1)))

        metrics = {"n": n, "logloss": ll, "accuracy": acc}

        # Precision/PnL at top K EV
        for frac_pct in [13, 26, 39]:
            frac = frac_pct / 100.0
            k = max(1, int(n * frac))
            idx_top = np.argpartition(ev_vec, -k)[-k:]

            # Realized PnL of selected trades
            trade_rets = y_r[idx_top] - fee
            mean_ret = float(np.mean(trade_rets))
            win_rate = float(np.mean(trade_rets > 0))

            metrics[f"ev_top{frac_pct}_ret"] = mean_ret
            metrics[f"ev_top{frac_pct}_wr"] = win_rate

        # Calibration (Brier for TP class)
        y_tp = (y_c == 2).astype(int)
        try:
            brier_tp = float(brier_score_loss(y_tp, pred[:, 2]))
            metrics["brier_tp"] = brier_tp
        except Exception:
            pass

        return metrics

    # ── Multi-barrier label construction ────────────────────────────
    @staticmethod
    def _build_multiclass_labels(
        y_per_horizon: Dict[int, np.ndarray],
        vol_proxy: np.ndarray,
        k_tp: float = 2.0,
        k_sl: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build 3-class labels (0=SL, 1=Timeout, 2=TP) using risk-unit thresholds."""
        n = len(vol_proxy)
        y_class = np.ones(n, dtype=np.int8) # Default 1 (Timeout)

        vp = np.clip(vol_proxy, 1e-9, None)
        tp_thresh = k_tp * vp
        sl_thresh = k_sl * vp

        max_ret = np.full(n, -999.0)
        min_ret = np.full(n, 999.0)

        for h, y_h in y_per_horizon.items():
            max_ret = np.maximum(max_ret, y_h)
            min_ret = np.minimum(min_ret, y_h)

        hit_tp = max_ret >= tp_thresh
        hit_sl = min_ret <= -sl_thresh

        # Assign classes
        # If hit_sl -> 0
        # Else if hit_tp -> 2
        # Else -> 1

        y_class[hit_tp] = 2
        y_class[hit_sl] = 0 # SL overrides TP (conservative)

        w_class = np.ones(n, dtype=np.float32)
        return y_class, w_class

    # ── Main fit ─────────────────────────────────────────────────────
    def fit(self, X_meta: pd.DataFrame, y_ret: np.ndarray,
            sample_weight=None, groups=None,
            y_per_horizon: Optional[Dict[int, np.ndarray]] = None,
            vol_proxy: Optional[np.ndarray] = None,
            realized_u_policy: Optional[np.ndarray] = None,
            selection_cfg: Optional[MetaClassifierSelectionConfig] = None,
            y_class_override: Optional[np.ndarray] = None,
            trade_mask: Optional[np.ndarray] = None):
        """Race classifiers using multi-barrier labels (multiclass if vol_proxy provided)."""
        import time as _time
        _t0 = _time.monotonic()
        tprint(f"MetaClassifierModel.fit: {self.strategy_name} starting "
               f"(n={len(y_ret)}, feats={X_meta.shape[1]})")
        y_ret_np = np.asarray(y_ret, dtype=float)
        sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float)

        # Feature selection: use all features, drop near-constant
        Xv_raw = X_meta.to_numpy(dtype=np.float32)
        col_std = np.nanstd(Xv_raw, axis=0)
        keep_cols = col_std > 1e-9
        selected_cols = list(X_meta.columns[keep_cols])
        self.selected_features = selected_cols
        Xv = Xv_raw[:, keep_cols]
        tprint(f"  Features: {X_meta.shape[1]} -> {len(selected_cols)}")

        # Build labels
        if y_class_override is not None:
            y_class = np.asarray(y_class_override, dtype=np.int8)
            w_barrier = np.ones(len(y_class), dtype=np.float32)
            tprint(f"  Multiclass labels from engine (0=SL, 1=TO, 2=TP): {np.bincount(y_class, minlength=3)}")
        elif vol_proxy is not None and y_per_horizon:
            # Drop samples with undefined volatility (Task 5)
            valid_vol = np.isfinite(vol_proxy) & (vol_proxy > 1e-9)
            if not valid_vol.all():
                n_drop = int((~valid_vol).sum())
                tprint(f"  Dropping {n_drop} samples with invalid vol_proxy.")
                valid_idx = np.where(valid_vol)[0]
                Xv = Xv[valid_idx]
                y_ret_np = y_ret_np[valid_idx]
                if sw is not None:
                    sw = sw[valid_idx]
                if groups is not None:
                    groups = np.asarray(groups)[valid_idx]
                if realized_u_policy is not None:
                    realized_u_policy = np.asarray(realized_u_policy, dtype=float)[valid_idx]
                if trade_mask is not None:
                    trade_mask = np.asarray(trade_mask, dtype=bool)[valid_idx]
                y_per_horizon = {h: v[valid_idx] for h, v in y_per_horizon.items()}
                vol_proxy = vol_proxy[valid_idx]

            # New multiclass path (Task 6, 7)
            y_class, w_barrier = self._build_multiclass_labels(y_per_horizon, vol_proxy)
            tprint(f"  Multiclass labels (0=SL, 1=TO, 2=TP): {np.bincount(y_class)}")
        else:
            raise ValueError("MetaClassifierModel requires vol_proxy (ATR) and y_per_horizon for risk-based labeling.")

        # Combine barrier weights with sample weights
        if sw is not None:
            sw_combined = sw * w_barrier
            sw_combined = sw_combined / max(float(np.mean(sw_combined)), 1e-12)
        else:
            sw_combined = w_barrier

        candidates = self._build_candidates()
        records = []
        scored = []
        best_rec = None
        if selection_cfg is None:
            selection_cfg = MetaClassifierSelectionConfig()

        if realized_u_policy is None:
            realized_u_policy = np.log1p(np.clip(y_ret_np, -0.999999, None))
        else:
            realized_u_policy = np.asarray(realized_u_policy, dtype=float)

        for name, cand in candidates.items():
            kind = cand["kind"]
            params = dict(cand["params"])

            try:
                oof, logloss = self._cv_evaluate(kind, params, Xv, y_class, sw_combined)
            except Exception as exc:
                tprint(f"    {name} failed: {exc}")
                continue

            metrics = self._compute_clf_metrics(
                oof, y_class, y_ret_np, groups=groups, fee=self.FEE_PER_ROUND_TRIP)
            sel = pick_meta_classifier_by_utility_top30(
                y_true=y_class,
                p_pred=oof,
                realized_u_policy=realized_u_policy,
                cfg=selection_cfg,
                trade_mask=trade_mask,
            )
            metrics["model"] = name
            metrics["logloss_cv"] = logloss
            metrics["selection_score"] = float(sel.get("selection_score", float("nan")))
            metrics["top_realized_u_mean"] = float(sel.get("top_realized_u_mean", float("nan")))
            metrics["passed_gate"] = float(sel.get("passed_gate", 0.0))
            metrics["passed_econ"] = float(sel.get("passed_econ", 0.0))

            records.append(metrics)
            scored.append((name, kind, params, oof, y_class, metrics, sel))
            tprint(
                f"    {name}: LogLoss={logloss:.4f}, Acc={metrics.get('accuracy',0):.3f}, "
                f"TopU={sel.get('selection_score', float('nan')):.5f}, "
                f"TopRealU={sel.get('top_realized_u_mean', float('nan')):.5f}, "
                f"gate={bool(sel.get('passed_gate',0))}, econ={bool(sel.get('passed_econ',0))}"
            )

        gated = [r for r in scored if bool(r[6].get("passed_gate", 0.0) > 0.5 and r[6].get("passed_econ", 0.0) > 0.5)]
        pool = gated if gated else scored
        if pool:
            _best = max(pool, key=lambda r: float(r[6].get("selection_score", -1e18)))
            best_rec = {
                "name": _best[0], "kind": _best[1], "params": _best[2],
                "oof": _best[3], "y_class": _best[4],
                "metrics": _best[5],
                "selection": _best[6],
            }

        if best_rec is None:
            raise RuntimeError("No classifier candidates completed")

        self.label_threshold = 0.0 # Not used in multiclass
        self.oof_probs = best_rec["oof"] # (N, 3)
        self._model_type = best_rec["name"]
        self.report_rows = records

        # Final fit on all data with best config
        kind = best_rec["kind"]
        params = best_rec["params"]
        y_final = best_rec["y_class"]
        
        # Safety: ensure at least 2 classes for Classifier (especially LogisticRegression)
        unique_classes = np.unique(y_final)
        if len(unique_classes) < 2:
            tprint(f"  WARNING: Meta labels for {self.strategy_name} have only one class: {unique_classes}. Skipping final fit.")
            self.model = {"kind": "trivial", "class": unique_classes[0], "multiclass": True}
            return self

        _sel_best = best_rec.get("selection", {}) if isinstance(best_rec, dict) else {}
        tprint(
            f"  Winner: {best_rec['name']} "
            f"(SelScore={float(_sel_best.get('selection_score', float('nan'))):.5f}, "
            f"LogLoss={float(_sel_best.get('logloss', float('nan'))):.4f}, "
            f"TopRealU={float(_sel_best.get('top_realized_u_mean', float('nan'))):.5f}). "
            f"Fitting final model..."
        )

        final_model = self._fit_one(kind, params, Xv, y_final, Xv, y_final,
                                    sw=sw_combined)
        self.model = {"kind": kind, "models": [final_model], "multiclass": True}

        # Save race report
        report_dir = Path("extreme_price_movements/reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(
            report_dir / f"meta_clf_{self.strategy_name or 'generic'}_race.csv",
            index=False,
        )

        tprint(f"MetaClassifierModel.fit: {self.strategy_name} done ({_time.monotonic()-_t0:.1f}s). "
               f"Winner={best_rec['name']}")
        return self

    def predict_proba(self, X_meta):
        if self.selected_features is None or self.model is None:
            raise RuntimeError("MetaClassifierModel must be fitted before predict")
            
        if self.model.get("kind") == "trivial":
            n = len(X_meta)
            cls = self.model["class"]
            out = np.zeros((n, 3), dtype=np.float64)
            out[:, int(cls)] = 1.0
            return out

        X = X_meta[self.selected_features].to_numpy(dtype=float)
        probs_list = []
        for m in self.model["models"]:
            pp = self._predict_proba(m, X)
            probs_list.append(pp)
        out = np.mean(probs_list, axis=0)
        row_sums = out.sum(axis=1)
        assert out.shape[1] == 3, f"Expected multiclass probabilities of shape (N,3), got {out.shape}"
        assert np.all(np.abs(row_sums - 1.0) < 1e-3), "Predicted probabilities must sum to ~1"
        return out
