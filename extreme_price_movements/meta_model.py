"""Simple meta model with candidate race: Ridge, ExtraTrees, XGB variants.

No strict guardrails, no monotone constraints. Winner selection by Spearman IC on OOF.
Optional Optuna HPO on the winner.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple
from pathlib import Path

import importlib.util
import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import spearmanr, rankdata
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, brier_score_loss

from extreme_price_movements.feature_selection_extreme_events import (
    mdi_feature_selection_v3,
    mdi_feature_selection_v4_topk,
)
from extreme_price_movements.purged_cv import PurgedKFold
from extreme_price_movements.utils import tprint
from extreme_price_movements.path_utils import resolve_reports_dir
from extreme_price_movements.policy_ml import (
    MetaClassifierSelectionConfig,
    MetaMoveSelectionConfig,
    pick_meta_classifier_by_utility_top30,
    pick_meta_move_by_topq,
)
from extreme_price_movements.training_utils import robust_sigma

if importlib.util.find_spec("optuna") is not None:
    import optuna
    from optuna.pruners import SuccessiveHalvingPruner
else:
    optuna = None

META_CLASS_ORDER = np.array([0, 1, 2], dtype=np.int64)


# ---------------------------
# Running statistics for z-score normalization across trials (meta HPO)
# ---------------------------
class _RunningStats:
    """Online mean and variance tracking for z-score computation."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float):
        """Update with new value using Welford's algorithm."""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def get_mean(self) -> float:
        return self.mean if self.n > 0 else 0.0

    def get_std(self) -> float:
        return (self.m2 / self.n) ** 0.5 if self.n > 0 else 1.0

    def zscore(self, x: float) -> float:
        """Compute z-score for a value."""
        std = self.get_std()
        if std < 1e-9:
            return 0.0
        return (x - self.get_mean()) / std


# Global running stats for meta HPO metrics (populated across trials)
_running_stats_meta = {
    "rank_metric": _RunningStats(),
    "top_bucket_lift": _RunningStats(),
    "top_bucket_mean": _RunningStats(),
    "rank_metric_std": _RunningStats(),
    "error_metric": _RunningStats(),
}


# ---------------------------
# Custom Early Stopping Callback for Optuna HPO
# ---------------------------
def _make_early_stopping_callback(
    min_trials_before_check: int = 50,
    patience: int = 30,
    magnitude_patience: int = 40,
    improvement_threshold: float = 1.005
):
    """Create Optuna early stopping callback with dual rules.

    Early stopping ONLY activates after min_trials_before_check trials.

    Rule 1 (standard): Stop if no improvement for `patience` trials.
    Rule 2 (magnitude): Stop if Score_new > Score_old * 1.005 for
                       `magnitude_patience` consecutive trials.

    The two rules are complementary - either can trigger early stopping.
    """
    def early_stopping_callback(study, trial):
        # Early stopping ONLY after min_trials_before_check
        if len(study.trials) < min_trials_before_check:
            return

        # Rule 1: Standard patience-based early stopping
        # Check last `patience` trials for no improvement
        recent_trials = study.trials[-patience:]
        best_value = study.best_value
        no_improvement = all(
            t.value is None or t.value < best_value * 0.9999
            for t in recent_trials
        )
        if no_improvement:
            tprint(f"  Meta HPO early stop (patience={patience}): no improvement for {patience} trials")
            study.stop()
            return

        # Rule 2: Magnitude-based improvement threshold
        # Check if all recent trials exceed threshold over baseline
        if len(study.trials) < magnitude_patience + 1:
            return

        # Get the best value from before the recent window
        trials_before_window = study.trials[:-magnitude_patience]
        if not trials_before_window:
            return

        best_before = max(
            (t.value for t in trials_before_window if t.value is not None),
            default=None
        )
        if best_before is None:
            return

        # Check if all recent trials exceed threshold
        threshold = best_before * improvement_threshold
        all_above_threshold = all(
            t.value is not None and t.value > threshold
            for t in study.trials[-magnitude_patience:]
        )
        if all_above_threshold:
            tprint(f"  Meta HPO early stop (magnitude): {magnitude_patience} trials > {improvement_threshold:.4f}x baseline")
            study.stop()

    return early_stopping_callback


if importlib.util.find_spec("xgboost") is not None:
    import xgboost as xgb
else:
    xgb = None


def _safe_spearman(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 10:
        return 0.0
    va, vb = a[mask], b[mask]
    if np.std(va) < 1e-12 or np.std(vb) < 1e-12:
        return 0.0
    rho, _ = spearmanr(va, vb)
    return float(rho) if np.isfinite(rho) else 0.0


def _xgb_staged_dispersion(model, X, *, kind: str, max_checkpoints: int = 16):
    """Estimate uncertainty from staged XGB predictions.

    We sample cumulative predictions at several boosting checkpoints and use
    their dispersion as a proxy for tree-wise uncertainty.
    """
    if xgb is None:
        n = len(X)
        nan_vec = np.full(n, np.nan, dtype=np.float32)
        return nan_vec, nan_vec

    try:
        n_estimators = int(getattr(model, "n_estimators", 0) or 0)
    except Exception:
        n_estimators = 0
    if n_estimators <= 1:
        n = len(X)
        nan_vec = np.full(n, np.nan, dtype=np.float32)
        return nan_vec, nan_vec

    checkpoints = np.unique(
        np.clip(
            np.round(np.linspace(1, n_estimators, num=min(max_checkpoints, n_estimators))),
            1,
            n_estimators,
        ).astype(int)
    )
    preds = []
    for ck in checkpoints:
        try:
            if kind == "classifier" and hasattr(model, "predict_proba"):
                pp = np.asarray(
                    model.predict_proba(X, iteration_range=(0, int(ck))),
                    dtype=np.float64,
                )
                if pp.ndim == 2 and pp.shape[1] >= 2:
                    pred = pp[:, 1]
                elif pp.ndim == 1:
                    pred = pp
                else:
                    pred = pp.reshape(-1)
            else:
                pred = np.asarray(
                    model.predict(X, iteration_range=(0, int(ck))),
                    dtype=np.float64,
                ).reshape(-1)
        except Exception:
            try:
                pred = np.asarray(model.predict(X), dtype=np.float64).reshape(-1)
            except Exception:
                continue
        pred = np.where(np.isfinite(pred), pred, np.nan)
        preds.append(pred)

    if len(preds) < 2:
        n = len(X)
        nan_vec = np.full(n, np.nan, dtype=np.float32)
        return nan_vec, nan_vec

    mat = np.stack(preds, axis=1)
    sigma = np.asarray(np.nanstd(mat, axis=1), dtype=np.float32)
    r_sigma = np.asarray(robust_sigma(mat), dtype=np.float32)
    return sigma, r_sigma


def discover_monotone_constraints(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: List[str],
    max_per_bucket: int = 5000,
    max_frac: float = 0.10,
    min_same_sign: float = 0.875,
    min_abs_mean_rho: float = 0.03,
    max_median_p: float = 0.03,
) -> Dict[str, int]:
    n, p = len(y), len(feature_names)
    if n < 200 or p == 0:
        return {}

    y_arr = np.asarray(y, dtype=np.float64)
    X_arr = np.asarray(X[feature_names], dtype=np.float64)

    trend_col = None
    for c in ("trend_strength_percentile", "trend"):
        if c in X.columns:
            trend_col = c
            break
    if trend_col is not None:
        trend_raw = np.asarray(X[trend_col], dtype=np.float64)
    else:
        if p >= 3:
            pc1 = X_arr[:, 2]
        else:
            pc1 = np.zeros(n, dtype=np.float64)
        trend_raw = pc1

    vol_col = None
    for c in ("realized_volatility_24h", "volatility_zscore", "rv_8h"):
        if c in X.columns:
            vol_col = c
            break
    if vol_col is not None:
        vol_raw = np.asarray(X[vol_col], dtype=np.float64)
    else:
        vol_raw = np.std(X_arr, axis=1)

    t_med = np.nanmedian(trend_raw)
    t_std = max(np.nanstd(trend_raw), 1e-9)
    trend_bucket = np.where(
        trend_raw > t_med + 0.5 * t_std, 1,
        np.where(trend_raw < t_med - 0.5 * t_std, -1, 0)
    ).astype(np.int8)

    v33 = np.nanpercentile(vol_raw, 33)
    v66 = np.nanpercentile(vol_raw, 66)
    vol_bucket = np.where(
        vol_raw <= v33, 0,
        np.where(vol_raw <= v66, 1, 2)
    ).astype(np.int8)

    bucket_ids = trend_bucket.astype(np.int32) * 3 + vol_bucket.astype(np.int32)
    unique_buckets = np.unique(bucket_ids)
    n_buckets = len(unique_buckets)

    finite_mask = np.isfinite(y_arr)
    rng = np.random.RandomState(42)
    bucket_indices: List[np.ndarray] = []
    for b in unique_buckets:
        bm = (bucket_ids == b) & finite_mask
        idx = np.where(bm)[0]
        if len(idx) > max_per_bucket:
            idx = rng.choice(idx, size=max_per_bucket, replace=False)
        if len(idx) >= 50:
            bucket_indices.append(idx)

    if len(bucket_indices) < 2:
        return {}

    rho_matrix = np.zeros((len(bucket_indices), p), dtype=np.float64)
    pval_matrix = np.ones((len(bucket_indices), p), dtype=np.float64)

    for bi, idx in enumerate(bucket_indices):
        yb = y_arr[idx]
        Xb = X_arr[idx]
        fm = np.isfinite(yb)
        for fm_col in range(p):
            col_vals = Xb[fm, fm_col]
            cm = fm.copy()
            valid = np.isfinite(col_vals)
            cm = fm & valid
            if cm.sum() < 20:
                continue
            r, pv = spearmanr(Xb[cm, fm_col], yb[cm])
            if np.isfinite(r):
                rho_matrix[bi, fm_col] = r
                pval_matrix[bi, fm_col] = pv

    signs = np.sign(rho_matrix)
    pos_frac = np.mean(signs > 0, axis=0)
    neg_frac = np.mean(signs < 0, axis=0)
    same_sign_ratio = np.maximum(pos_frac, neg_frac)
    dominant_sign = np.where(pos_frac >= neg_frac, 1.0, -1.0)

    mean_rho = np.mean(rho_matrix, axis=0)
    abs_mean_rho = np.abs(mean_rho)
    std_rho = np.std(rho_matrix, axis=0, ddof=1)
    se_rho = std_rho / np.sqrt(max(n_buckets, 1))
    mean_abs_rho = np.mean(np.abs(rho_matrix), axis=0)
    median_bucket_p = np.median(pval_matrix, axis=0)

    abs_rho_q70 = np.nanpercentile(mean_abs_rho, 70)

    t_stat_ok = abs_mean_rho > 2.0 * se_rho
    pass_gate = (
        (same_sign_ratio >= min_same_sign)
        & t_stat_ok
        & (median_bucket_p < max_median_p)
        & (abs_mean_rho >= min_abs_mean_rho)
        & (mean_abs_rho >= abs_rho_q70)
    )

    safe_p = np.maximum(median_bucket_p, 1e-6)
    safe_se = np.maximum(se_rho, 1e-6)
    score = abs_mean_rho * (-np.log(safe_p)) / safe_se
    score[~pass_gate] = -np.inf

    max_features = max(1, int(p * max_frac))
    ranked = np.argsort(-score)
    constrained_features: Dict[str, int] = {}
    count = 0
    for fi in ranked:
        if count >= max_features:
            break
        if not pass_gate[fi]:
            break
        fname = feature_names[fi]
        constrained_features[fname] = int(np.sign(mean_rho[fi]))
        count += 1

    if constrained_features:
        tprint(
            f"  Monotone constraints: {count}/{p} features "
            f"(buckets={len(bucket_indices)}/{n_buckets})"
        )
    return constrained_features


_META_HPO_JSON = "best_meta_params.json"


def _meta_hpo_scope_suffix(scope_key: Optional[str]) -> str:
    if not scope_key:
        return ""
    safe = str(scope_key).strip().replace(os.sep, "_").replace(" ", "_")
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in safe)
    return f"__{safe}" if safe else ""


def _meta_hpo_json_path(
    out_dir: str,
    scope_key: Optional[str] = None,
    kind: Optional[str] = None,
) -> str:
    suffix = _meta_hpo_scope_suffix(scope_key)
    if kind:
        kind_safe = "".join(
            ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(kind)
        )
        suffix = f"{suffix}__{kind_safe}" if suffix else f"__{kind_safe}"
    stem, ext = os.path.splitext(_META_HPO_JSON)
    return os.path.join(out_dir, f"{stem}{suffix}{ext}")


def _load_meta_hpo_json(
    out_dir: str,
    scope_key: Optional[str] = None,
    kind: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    path = _meta_hpo_json_path(out_dir, scope_key=scope_key, kind=kind)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_meta_hpo_json(
    out_dir: str,
    payload: Dict[str, Any],
    scope_key: Optional[str] = None,
    kind: Optional[str] = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = _meta_hpo_json_path(out_dir, scope_key=scope_key, kind=kind)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def load_best_meta_params(
    out_dir: str = "./hpo_out",
    scope_key: Optional[str] = None,
    kind: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Load previously saved best meta params for warm-start reuse."""
    data = _load_meta_hpo_json(out_dir, scope_key=scope_key, kind=kind)
    if data is None:
        return None
    return data.get("best_params")


class MetaModel:
    def __init__(self, strategy_name: Optional[str] = None, reports_dir: str | Path | None = None):
        self.strategy_name = strategy_name
        self.model = None
        self._model_type = None
        self.selected_features: Optional[List[str]] = None
        self.oof_probs: Optional[np.ndarray] = None
        self.oof_sigma: Optional[np.ndarray] = None
        self.oof_robust_sigma: Optional[np.ndarray] = None
        self.report_rows: List[dict] = []
        self.score_sign: int = 1
        self._reports_dir = resolve_reports_dir(reports_dir)
        self.selector_cfg: Dict[str, object] = {}
        self.selector_report_dir: Optional[str] = None
        self.selector_prev_selected: Optional[Sequence[str]] = None
        self.selector_family_map: Dict[str, str] = {}
        self.candidate_mode: str = "race"
        self.selector_target_override: Optional[str] = None
        self.selector_loss_override: Optional[str] = None
        self.disable_hpo: bool = False
        self.hpo_out_dir: Optional[str] = None
        self.xgb_parallel_forest_params: Optional[Dict[str, object]] = None
        self.monotone_constraints: Optional[Dict[str, int]] = None

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

    def _prepare_fit_arrays(
        self,
        y: np.ndarray,
        sw: Optional[np.ndarray],
        tail_lambda: float,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        y_fit = np.asarray(y, dtype=float).copy()
        sw_fit = None if sw is None else np.asarray(sw, dtype=float).copy()

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

        return y_fit, sw_fit

    @staticmethod
    def _top_slice_pnl(pred: np.ndarray, y: np.ndarray, frac: float = 0.30) -> float:
        mask = np.isfinite(pred) & np.isfinite(y)
        if mask.sum() == 0:
            return float("-inf")
        pred_f = np.asarray(pred[mask], dtype=float)
        y_f = np.asarray(y[mask], dtype=float)
        k = max(1, int(np.ceil(float(frac) * len(pred_f))))
        idx = np.argpartition(pred_f, -k)[-k:]
        return float(np.sum(y_f[idx]))

    @staticmethod
    def _gate_mask_from_scores(anchor_scores: np.ndarray, gate_name: str) -> np.ndarray:
        scores = np.asarray(anchor_scores, dtype=float)
        valid = np.isfinite(scores)
        if gate_name == "all":
            return valid.copy()
        if not gate_name.startswith("top"):
            raise ValueError(f"Unknown gate name: {gate_name}")
        top_pct = int(gate_name[3:])
        if valid.sum() == 0:
            return np.zeros(len(scores), dtype=bool)
        top_pct = max(1, min(100, top_pct))
        q = 100.0 - float(top_pct)
        thr = float(np.nanpercentile(scores[valid], q))
        mask = valid & (scores >= thr)
        if mask.sum() == 0:
            best_idx = np.where(valid)[0][np.argmax(scores[valid])]
            mask[best_idx] = True
        return mask

    @staticmethod
    def _reject_floor(scores: np.ndarray) -> float:
        pred = np.asarray(scores, dtype=float)
        valid = pred[np.isfinite(pred)]
        if valid.size == 0:
            return -1e9
        sigma = float(np.nanstd(valid))
        return float(np.nanmin(valid) - max(sigma, 1e-6))

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
            _sel_cfg = dict(self.selector_cfg or {})
            # Single broad MDI v3 pass — high floor, no aggressive refinement
            fs1 = mdi_feature_selection_v3(
                X, y_scaled, min_features=30, end_features=max_features,
                selector_y=y,
                selector_target=str(self.selector_target_override or "regression"),
                selector_loss=str(self.selector_loss_override or "huber"),
                selector_head_name=f"meta_{self.strategy_name or 'default'}",
                selector_report_dir=self.selector_report_dir,
                selector_prev_selected=list(self.selector_prev_selected) if self.selector_prev_selected is not None else None,
                selector_family_map=dict(self.selector_family_map or {}),
                selector_focus_top_frac=float(_sel_cfg.get("selector_focus_top_frac", 1.0)),
                selector_top_metric=_sel_cfg.get("selector_top_metric", "ic"),
                selector_frequency_hit_mode=str(_sel_cfg.get("selector_frequency_hit_mode", "relative")),
                selector_frequency_hit_quantile=float(_sel_cfg.get("selector_frequency_hit_quantile", 0.80)),
                selector_frequency_hit_abs=float(_sel_cfg.get("selector_frequency_hit_abs", 1e-6)),
                selector_interaction_mode=str(_sel_cfg.get("selector_interaction_mode", "tree_path_lift")),
                selector_interaction_topk_pairs=int(_sel_cfg.get("selector_interaction_topk_pairs", 100)),
                selector_interaction_max_pairs_per_feature=int(_sel_cfg.get("selector_interaction_max_pairs_per_feature", 8)),
                selector_interaction_corr_penalty=bool(_sel_cfg.get("selector_interaction_corr_penalty", True)),
                selector_family_penalty=bool(_sel_cfg.get("selector_family_penalty", True)),
                selector_emit_report=bool(_sel_cfg.get("selector_emit_report", True)),
                analysis_n_estimators=int(_sel_cfg.get("analysis_n_estimators", 192)),
                analysis_max_samples=int(_sel_cfg.get("analysis_max_samples", 3000)),
                min_samples_leaf_pct=float(_sel_cfg.get("min_samples_leaf_pct", 0.015)),
                selector_max_missing_frac=float(_sel_cfg.get("selector_max_missing_frac", 0.15)),
                selector_near_constant_dominance=float(
                    _sel_cfg.get("selector_near_constant_dominance", 0.999)
                ),
                selector_hysteresis_margin=float(_sel_cfg.get("selector_hysteresis_margin", 0.05)),
                selector_min_overlap=float(_sel_cfg.get("selector_min_overlap", 0.70)),
                composite_weights={
                    "top30": float(_sel_cfg.get("top30", 0.20)),
                    "global": float(_sel_cfg.get("global", 0.35)),
                    "stability": float(_sel_cfg.get("stability", 0.25)),
                    "frequency": float(_sel_cfg.get("frequency", 0.10)),
                    "interaction": float(_sel_cfg.get("interaction", 0.10)),
                },
                max_features_pct=0.90,
            )
            selected_stage1 = list(fs1.selected_features)
            if len(selected_stage1) < 30:
                selected = list(X.columns[:max_features])
            else:
                # Stage 2: v4_topk refinement for redundancy removal
                n_stage2_target = max(30, min(max_features, len(selected_stage1) - 10))
                if len(selected_stage1) > n_stage2_target:
                    try:
                        X_stage1 = X[selected_stage1]
                        fs2 = mdi_feature_selection_v4_topk(
                            X_stage1, y_scaled,
                            topk_weight=0.3,
                            base_model=None,
                        )
                        selected = list(fs2.selected_features)[:n_stage2_target]
                        tprint(f"  Meta v4_topk refinement: {len(selected_stage1)} -> {len(selected)} features")
                    except Exception as exc2:
                        tprint(f"  v4_topk refinement failed ({exc2}), using stage1 selection")
                        selected = selected_stage1[:max_features]
                else:
                    selected = selected_stage1[:max_features]
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
        """Candidates: Ridge, ExtraTrees, XGB variants."""
        if self.candidate_mode == "xgb_parallel_forest":
            if xgb is None:
                raise RuntimeError("xgboost is required for MetaModel candidate_mode='xgb_parallel_forest'")
            _params = dict(self.xgb_parallel_forest_params or {})
            if not _params:
                _params = {
                    "objective": "reg:squarederror",
                    "n_estimators": 400,
                    "num_parallel_tree": 5,
                    "max_depth": 5,
                    "learning_rate": 0.05,
                    "subsample": 0.75,
                    "colsample_bytree": 0.75,
                    "reg_alpha": 2.0,
                    "reg_lambda": 15.0,
                    "min_child_weight": 40.0,
                    "gamma": 1.5,
                    "tree_method": "hist",
                    "random_state": 42,
                    "n_jobs": 3,
                    "verbosity": 0,
                    "early_stopping_rounds": 20,
                }
            return {
                "xgb_parallel_forest": {
                    "kind": "xgb",
                    "params": _params,
                    "tail_lambda": 0.0,
                }
            }
        candidates = {}

        # Legacy fallback for ad hoc MetaModel usage.
        # Main training paths set candidate_mode explicitly and do not rely on this.
        candidates["ridge"] = {
            "kind": "ridge",
            "params": {"alpha": 5.0, "fit_intercept": True},
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
        if kind == "extratrees":
            model = ExtraTreesRegressor(**params)
            model.fit(X_tr, y_tr, sample_weight=sw)
            return model
        if kind == "lgbm":
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(**params)
            model.fit(X_tr, y_tr, sample_weight=sw,
                      eval_set=[(X_va, y_va)], callbacks=[])
            return model
        if kind == "xgb":
            p = dict(params)
            es_rounds = p.pop("early_stopping_rounds", 50)
            mc = getattr(self, "monotone_constraints", None)
            if mc and isinstance(mc, dict) and self.selected_features:
                p["monotone_constraints"] = tuple(
                    mc.get(f, 0) for f in self.selected_features
                )
            model = xgb.XGBRegressor(**p, early_stopping_rounds=es_rounds)
            model.fit(X_tr, y_tr, sample_weight=sw,
                      eval_set=[(X_va, y_va)], verbose=False)
            return model
        raise ValueError(f"Unknown kind: {kind}")

    # ── CV evaluation ────────────────────────────────────────────────────
    def _cv_evaluate(
        self,
        kind,
        params,
        X,
        y,
        sw=None,
        feature_max_features: Optional[int] = None,
    ) -> Tuple[np.ndarray, float, Dict[str, Any], Dict[str, np.ndarray]]:
        """3-fold purged CV. Returns OOF predictions plus uncertainty proxies."""
        pkf = PurgedKFold(n_splits=3, purge=12, embargo=12)
        oof = np.full(len(y), np.nan, dtype=float)
        oof_sigma = np.full(len(y), np.nan, dtype=np.float32)
        oof_robust_sigma = np.full(len(y), np.nan, dtype=np.float32)

        if isinstance(X, pd.DataFrame):
            X_frame = X.reset_index(drop=True)
        else:
            X_frame = pd.DataFrame(
                np.asarray(X, dtype=np.float32),
                columns=[f"f_{i}" for i in range(np.asarray(X).shape[1])],
            )
        max_features = int(
            feature_max_features if feature_max_features is not None else max(30, min(50, X_frame.shape[1]))
        )

        # Per-fold metrics for composite scoring
        fold_spearmans: List[float] = []
        fold_top_decile_lifts: List[float] = []
        fold_top_decile_means: List[float] = []
        fold_rmses: List[float] = []

        for tr, va in pkf.split(X_frame):
            if getattr(self, "_dynamic_labels", None) is not None:
                # Dynamic soft labels evaluated on train set to prevent leakage
                dyn_tr = self._dynamic_labels[tr]
                dyn_va = self._dynamic_labels[va]
                retained = dyn_tr.get_retained_geometries()
                y_tr = dyn_tr.build_soft_labels_with(retained).astype(np.float32)
                y_va = dyn_va.build_soft_labels_with(retained).astype(np.float32)
            else:
                y_tr, y_va = y[tr], y[va]
            sw_tr = None if sw is None else sw[tr]
            X_tr_df = X_frame.iloc[tr].reset_index(drop=True)
            X_va_df = X_frame.iloc[va].reset_index(drop=True)
            selected_cols = self._select_tail_features(
                X_tr_df, np.asarray(y_tr, dtype=float), max_features=max_features
            )
            X_tr = X_tr_df[selected_cols].to_numpy(dtype=np.float32)
            X_va = X_va_df[selected_cols].to_numpy(dtype=np.float32)
            model = self._fit_one(kind, params, X_tr, y_tr, X_va, y_va, sw=sw_tr)
            pred_va = model.predict(X_va)
            oof[va] = pred_va
            if kind == "xgb":
                sigma_fold, sigma_robust_fold = _xgb_staged_dispersion(
                    model,
                    X_va,
                    kind="regression",
                )
                oof_sigma[va] = sigma_fold
                oof_robust_sigma[va] = sigma_robust_fold

            # Compute per-fold metrics
            mask_va = np.isfinite(pred_va) & np.isfinite(y_va)
            if mask_va.sum() >= 10:
                # Spearman correlation (rank metric)
                fold_ic = _safe_spearman(pred_va[mask_va], y_va[mask_va])
                fold_spearmans.append(fold_ic)

                # Top decile lift and mean
                pred_masked = pred_va[mask_va]
                y_masked = y_va[mask_va]
                n = len(pred_masked)
                if n >= 10:
                    n10 = max(1, int(0.10 * n))
                    idx_top10 = np.argpartition(pred_masked, -n10)[-n10:]
                    top10_mean = float(np.mean(y_masked[idx_top10]))
                    overall_mean = float(np.mean(y_masked))
                    lift = (top10_mean - overall_mean) / max(abs(overall_mean), 1e-9)
                    fold_top_decile_lifts.append(lift)
                    fold_top_decile_means.append(top10_mean)

                # RMSE (error metric for regressors)
                try:
                    rmse = float(np.sqrt(mean_squared_error(y_va[mask_va], pred_va[mask_va])))
                    fold_rmses.append(rmse)
                except Exception:
                    pass

        missing = ~np.isfinite(oof)
        if np.any(missing):
            fitted = np.isfinite(oof)
            if np.any(fitted):
                fill = float(np.nanmedian(oof[fitted]))
            else:
                fill = float(np.nanmedian(np.asarray(y, dtype=float))) if len(y) else 0.0
            oof[missing] = fill

        mask = np.isfinite(oof)
        ic = _safe_spearman(oof[mask], y[mask])

        # Compile detailed metrics
        detailed_metrics = {
            "fold_spearmans": fold_spearmans,
            "fold_top_decile_lifts": fold_top_decile_lifts,
            "fold_top_decile_means": fold_top_decile_means,
            "fold_rmses": fold_rmses,
            "median_fold_spearman": float(np.median(fold_spearmans)) if fold_spearmans else 0.0,
            "std_fold_spearman": float(np.std(fold_spearmans)) if len(fold_spearmans) > 1 else 0.0,
            "mean_top_decile_lift": float(np.mean(fold_top_decile_lifts)) if fold_top_decile_lifts else 0.0,
            "mean_top_decile_mean": float(np.mean(fold_top_decile_means)) if fold_top_decile_means else 0.0,
            "mean_rmse": float(np.mean(fold_rmses)) if fold_rmses else 1.0,
        }

        return oof, ic, detailed_metrics, {
            "oof_sigma": oof_sigma,
            "oof_robust_sigma": oof_robust_sigma,
        }

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
    def _optuna_hpo(self, name, kind, params, X, y, sw=None, n_trials=150) -> dict:
        if importlib.util.find_spec("optuna") is None:
            return params
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        out_dir = str(self.hpo_out_dir or os.path.join("data", "hpo_out"))
        scope_key = str(self.strategy_name or "default")
        prev_blob = _load_meta_hpo_json(out_dir, scope_key=scope_key, kind=kind)
        prev_params = (
            dict(prev_blob.get("best_params", {}))
            if isinstance(prev_blob, dict) and prev_blob.get("best_params")
            else None
        )

        def objective(trial):
            p = dict(params)
            if kind == "ridge":
                p["alpha"] = trial.suggest_float("alpha", 0.01, 100.0, log=True)
            elif kind == "extratrees":
                p["n_estimators"] = trial.suggest_int("n_estimators", 200, 800)
                p["max_depth"] = trial.suggest_int("max_depth", 4, 16)
                p["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 10, 80)
            elif kind == "lgbm":
                p["num_leaves"] = trial.suggest_int("num_leaves", 20, 60)
                p["max_depth"] = trial.suggest_int("max_depth", 4, 10)
                p["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.15, log=True)
                p["n_estimators"] = trial.suggest_int("n_estimators", 300, 1000)
                p["feature_fraction"] = trial.suggest_float("feature_fraction", 0.6, 0.9)
                p["bagging_fraction"] = trial.suggest_float("bagging_fraction", 0.6, 0.9)
                p["reg_alpha"] = trial.suggest_float("reg_alpha", 0.1, 10.0, log=True)
                p["reg_lambda"] = trial.suggest_float("reg_lambda", 1.0, 50.0, log=True)
            elif kind == "xgb":
                p["max_depth"] = trial.suggest_int("max_depth", 3, 7)
                p["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.15, log=True)
                p["n_estimators"] = trial.suggest_int("n_estimators", 300, 1200)
                p["subsample"] = trial.suggest_float("subsample", 0.5, 0.9)
                p["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 0.9)
                p["reg_alpha"] = trial.suggest_float("reg_alpha", 0.01, 20.0, log=True)
                p["reg_lambda"] = trial.suggest_float("reg_lambda", 1.0, 100.0, log=True)
            try:
                _, ic, metrics, _ = self._cv_evaluate(kind, p, X, y, sw)
            except Exception as exc:
                tprint(f"  Meta HPO trial failed[{scope_key}/{kind}]: {exc}")
                return -1e9

            # Compute composite score with z-scored metrics for regressors
            # score = 0.45*z(rank_metric) + 0.30*z(top_bucket_lift) + 0.20*z(top_bucket_mean)
            #         - 0.20*z(rank_metric_std) - 0.10*z(error_metric)
            rank_metric = metrics.get("median_fold_spearman", 0.0)
            top_bucket_lift = metrics.get("mean_top_decile_lift", 0.0)
            top_bucket_mean = metrics.get("mean_top_decile_mean", 0.0)
            rank_metric_std = metrics.get("std_fold_spearman", 0.0)
            error_metric = metrics.get("mean_rmse", 1.0)

            # Get running stats
            stats_rank = _running_stats_meta["rank_metric"]
            stats_lift = _running_stats_meta["top_bucket_lift"]
            stats_mean = _running_stats_meta["top_bucket_mean"]
            stats_std = _running_stats_meta["rank_metric_std"]
            stats_err = _running_stats_meta["error_metric"]

            # Compute z-scores (use scaled values if not enough samples yet)
            z_rank = stats_rank.zscore(rank_metric) if stats_rank.n >= 10 else rank_metric * 5.0
            z_lift = stats_lift.zscore(top_bucket_lift) if stats_lift.n >= 10 else top_bucket_lift * 5.0
            z_mean = stats_mean.zscore(top_bucket_mean) if stats_mean.n >= 10 else top_bucket_mean * 5.0
            z_std = stats_std.zscore(rank_metric_std) if stats_std.n >= 10 else rank_metric_std * 5.0
            z_err = stats_err.zscore(error_metric) if stats_err.n >= 10 else error_metric * 5.0

            # Update running stats
            stats_rank.update(rank_metric)
            stats_lift.update(top_bucket_lift)
            stats_mean.update(top_bucket_mean)
            stats_std.update(rank_metric_std)
            stats_err.update(error_metric)

            # Composite score
            composite = (
                0.45 * z_rank
                + 0.30 * z_lift
                + 0.20 * z_mean
                - 0.20 * z_std  # Penalize high std
                - 0.10 * z_err  # Penalize high error
            )

            # Store metrics for analysis
            trial.set_user_attr("rank_metric", rank_metric)
            trial.set_user_attr("top_bucket_lift", top_bucket_lift)
            trial.set_user_attr("top_bucket_mean", top_bucket_mean)
            trial.set_user_attr("rank_metric_std", rank_metric_std)
            trial.set_user_attr("error_metric", error_metric)
            trial.set_user_attr("z_rank", z_rank)
            trial.set_user_attr("z_lift", z_lift)
            trial.set_user_attr("z_mean", z_mean)
            trial.set_user_attr("z_std", z_std)
            trial.set_user_attr("z_err", z_err)

            return float(composite)

        study = optuna.create_study(
            study_name=f"meta_hpo_{name}",
            direction="maximize",
            pruner=SuccessiveHalvingPruner(
                min_resource=1,
                reduction_factor=2,
                min_early_stopping_rate=0,
                bootstrap_count=5,
            ),
        )
        if prev_params:
            try:
                study.enqueue_trial(prev_params)
                tprint(
                    f"  Meta HPO warm-start[{scope_key}/{kind}]: loaded previous params from "
                    f"{_meta_hpo_json_path(out_dir, scope_key=scope_key, kind=kind)}"
                )
            except Exception as exc:
                tprint(f"  Meta HPO warm-start[{scope_key}/{kind}] failed: {exc}")
        study.optimize(objective, n_trials=n_trials, timeout=None, callbacks=[_make_early_stopping_callback()], gc_after_trial=True)
        if study.best_trial is None:
            return params
        best = dict(params)
        best.update(study.best_params)
        payload = {
            "scope_key": scope_key,
            "kind": kind,
            "best_params": dict(study.best_params),
            "n_trials_completed": len(study.trials),
        }
        try:
            _save_meta_hpo_json(out_dir, payload, scope_key=scope_key, kind=kind)
            tprint(
                f"  Meta HPO saved params[{scope_key}/{kind}] to "
                f"{_meta_hpo_json_path(out_dir, scope_key=scope_key, kind=kind)}"
            )
        except Exception as exc:
            tprint(f"  Meta HPO save[{scope_key}/{kind}] failed: {exc}")
        return best

    # ── Main fit ─────────────────────────────────────────────────────────
    def fit(self, X_meta: pd.DataFrame, y, sample_weight=None, groups=None,
            y_per_horizon: Optional[Dict[int, np.ndarray]] = None):
        import time as _time
        _t0 = _time.monotonic()
        tprint(f"MetaModel.fit: {self.strategy_name} starting (n={len(y)}, feats={X_meta.shape[1]})")
        y_np = np.asarray(y, dtype=float)
        sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        n_target = max(30, min(50, X_meta.shape[1]))
        self._feature_selection_max_features = n_target

        candidates = self._build_candidates()
        tprint(f"  Racing {len(candidates)} candidates ({_time.monotonic()-_t0:.1f}s)...")

        records = []
        best_name = None
        best_ic = -1e18
        best_oof = None
        best_aux = None
        best_gate_name = "all"
        best_gate_mask = np.isfinite(y_np)
        if not best_gate_mask.any():
            raise RuntimeError("MetaModel fit has no finite targets")

        gate_horizons = None
        if y_per_horizon:
            gate_horizons = {h: np.asarray(v, dtype=float)[best_gate_mask] for h, v in y_per_horizon.items()}

        X_gate_df = X_meta.iloc[np.flatnonzero(best_gate_mask)].reset_index(drop=True)

        for name, cand in candidates.items():
            kind = cand["kind"]
            params = cand["params"]
            tail_lambda = cand["tail_lambda"]

            y_fit_all, sw_fit_all = self._prepare_fit_arrays(y_np, sw, tail_lambda)
            y_fit = y_fit_all[best_gate_mask]
            sw_fit = None if sw_fit_all is None else sw_fit_all[best_gate_mask]

            try:
                oof_gate, _, _, aux_gate = self._cv_evaluate(
                    kind,
                    params,
                    X_gate_df,
                    y_fit,
                    sw_fit,
                    feature_max_features=n_target,
                )
                oof_orig_gate = self._inverse_signed_log(oof_gate) if tail_lambda > 0 else oof_gate
            except Exception as exc:
                tprint(f"  Candidate {name} failed: {exc}")
                continue

            metrics = self._compute_oof_metrics(
                oof_orig_gate,
                y_np[best_gate_mask],
                y_per_horizon=gate_horizons,
            )
            ic_orig = metrics.get("ic", 0.0)
            ic_mh = metrics.get("ic_mh", ic_orig)
            ic_t30 = metrics.get("ic_top30", 0.0)
            spread10 = metrics.get("spread10", 0.0)
            # Neutral optimization across all deciles
            composite = ic_mh
            pnl_top30 = self._top_slice_pnl(oof_orig_gate, y_np[best_gate_mask], frac=0.30)

            oof_full = np.full(len(y_np), np.nan, dtype=float)
            oof_full[best_gate_mask] = oof_orig_gate

            rec = {
                "model": name,
                "kind": kind,
                "tail_lambda": tail_lambda,
                "gate": best_gate_name,
                "n_gate": int(best_gate_mask.sum()),
                "pnl_top30": pnl_top30,
                "composite": composite,
                **metrics,
            }
            records.append(rec)
            tprint(f"  {name}@{best_gate_name}: IC={ic_orig:.4f}, IC_mh={ic_mh:.4f}, IC_t30={ic_t30:.4f}, "
                   f"spread10={spread10:.6f}, pnl_top30={pnl_top30:.6f}, "
                   f"ECE_t30={metrics.get('ece_top30',0):.3f}, "
                   f"win_t30={metrics.get('win_rate_top30',0):.1%}, composite={composite:.4f}")

            if composite > best_ic:
                best_ic = composite
                best_name = name
                best_oof = oof_full
                best_aux = aux_gate

        if best_name is None:
            raise RuntimeError("No meta model candidates completed")

        winner = candidates[best_name]
        kind = winner["kind"]
        params = winner["params"]
        tail_lambda = winner["tail_lambda"]

        tprint(f"  Winner: {best_name} (composite={best_ic:.4f}). Starting HPO ({_time.monotonic()-_t0:.1f}s)...")

        y_fit_all, sw_fit_all = self._prepare_fit_arrays(y_np, sw, tail_lambda)
        y_fit = y_fit_all[best_gate_mask]
        sw_fit = None if sw_fit_all is None else sw_fit_all[best_gate_mask]

        self.selected_features = self._select_tail_features(
            X_gate_df, y_fit, max_features=n_target
        )
        self.monotone_constraints = discover_monotone_constraints(
            X_gate_df, y_fit, self.selected_features
        )
        Xv_gate = X_gate_df[self.selected_features].to_numpy(dtype=np.float32)
        tuned_params = (
            dict(params)
            if self.disable_hpo
            else self._optuna_hpo(best_name, kind, params, Xv_gate, y_fit, sw_fit)
        )
        tprint(f"  HPO done ({_time.monotonic()-_t0:.1f}s). Fitting final model...")

        # Final fit on all data
        if len(np.unique(y_fit)) < 2 and kind in ["ridge"]:
            tprint(f"  WARNING: Final fit on single-class data ({np.unique(y_fit)}), returning trivial model")
            # For regressors, we could return a constant model, but _fit_one expects a pipeline.
            # Let's just catch the error or ensure y_fit has at least some noise.
            pass

        final_model = self._fit_one(kind, tuned_params, Xv_gate, y_fit, Xv_gate, y_fit, sw=sw_fit)

        self.model = {
            "kind": kind, "models": [final_model],
            "is_transformed": tail_lambda > 0,
            "tail_lambda": tail_lambda,
        }
        self._model_type = best_name
        self.oof_probs = best_oof
        if best_aux is not None:
            self.oof_sigma = np.asarray(best_aux.get("oof_sigma"), dtype=np.float32).reshape(-1)
            self.oof_robust_sigma = np.asarray(
                best_aux.get("oof_robust_sigma"), dtype=np.float32
            ).reshape(-1)
        else:
            self.oof_sigma = np.full(len(self.oof_probs), np.nan, dtype=np.float32)
            self.oof_robust_sigma = np.full(len(self.oof_probs), np.nan, dtype=np.float32)
        self.report_rows = records

        # Save race report
        report_dir = self._reports_dir
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
# Meta Classifier Model — binary "magnitude move" head
# ═══════════════════════════════════════════════════════════════════════

class MetaClassifierModel:
    """Binary meta model for p_move = E[y_move_soft | X_t]."""

    FEE_PER_ROUND_TRIP = 0.005  # 0.5% total round-trip fee

    def __init__(self, strategy_name: Optional[str] = None, reports_dir: str | Path | None = None):
        self.strategy_name = strategy_name
        self.model = None
        self._model_type: Optional[str] = None
        self.selected_features: Optional[List[str]] = None
        self.oof_probs: Optional[np.ndarray] = None
        self.oof_probs_raw: Optional[np.ndarray] = None
        self.oof_sigma: Optional[np.ndarray] = None
        self.oof_robust_sigma: Optional[np.ndarray] = None
        self.report_rows: List[dict] = []
        self.label_threshold: float = 1.25  # move threshold multiplier
        self.score_sign: int = 1
        self._reports_dir = resolve_reports_dir(reports_dir)
        self.candidate_mode: str = "race"
        self.selector_cfg: Dict[str, object] = {}
        self.selector_report_dir: Optional[str] = None
        self.selector_prev_selected: Optional[Sequence[str]] = None
        self.selector_family_map: Dict[str, str] = {}
        self.selector_target_override: Optional[str] = None
        self.selector_loss_override: Optional[str] = None
        self.xgb_parallel_forest_params: Optional[Dict[str, object]] = None
        self.disable_hpo: bool = False
        self.hpo_out_dir: Optional[str] = None
        self.move_k: float = 1.25
        self.move_k_by_h: Dict[int, float] = {}
        self.move_thresholds: Tuple[float, ...] = (1.0, 1.25, 1.5)
        self.move_weights: Tuple[float, ...] = (0.45, 0.35, 0.20)
        self.use_class_weight_multiplier: bool = True
        self.max_class_weight: float = 10.0
        self.use_calibration: bool = True
        self.move_horizon: Optional[int] = None
        self._calibrator = None
        self.y_move: Optional[np.ndarray] = None
        self.y_move_soft: Optional[np.ndarray] = None
        self.move_threshold: Optional[np.ndarray] = None
        self.monotone_constraints: Optional[Dict[str, int]] = None

    def prepare_meta_features(self, preds, feats_df, pred_col_name="pred_logit"):
        p = np.clip(np.asarray(preds, dtype=float), 1e-4, 1 - 1e-4)
        meta_data = pd.DataFrame(index=feats_df.index)
        meta_data[pred_col_name] = np.clip(logit(p), -4.0, 4.0)
        return pd.concat([meta_data, feats_df], axis=1).fillna(0.0)

    def _select_tail_features(
        self, X: pd.DataFrame, y: np.ndarray, max_features: int = 80
    ) -> List[str]:
        return MetaModel._select_tail_features(self, X, y, max_features=max_features)

    # ── Candidate definitions ────────────────────────────────────────
    def _build_candidates(self) -> Dict[str, dict]:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import ExtraTreesClassifier

        if self.candidate_mode == "xgb_parallel_forest":
            if xgb is None:
                raise RuntimeError("xgboost is required for MetaClassifierModel candidate_mode='xgb_parallel_forest'")
            _params = dict(self.xgb_parallel_forest_params or {})
            if not _params:
                _params = {
                    "objective": "binary:logistic",
                    "n_estimators": 100,
                    "num_parallel_tree": 20,
                    "max_depth": 5,
                    "learning_rate": 0.05,
                    "subsample": 0.75,
                    "colsample_bytree": 0.75,
                    "reg_alpha": 2.0,
                    "reg_lambda": 15.0,
                    "min_child_weight": 40.0,
                    "gamma": 1.5,
                    "tree_method": "hist",
                    "random_state": 42,
                    "n_jobs": 3,
                    "verbosity": 0,
                    "eval_metric": "logloss",
                    "early_stopping_rounds": 20,
                }
            return {
                "xgb_parallel_forest_clf": {
                    "kind": "xgb_clf",
                    "params": _params,
                }
            }
        candidates = {}

        # 1. Ridge (LogisticRegression with L2)
        candidates["ridge_clf"] = {
            "kind": "ridge_clf",
            "params": {"C": 0.1, "penalty": "l2", "solver": "lbfgs",
                       "max_iter": 35000, "tol": 1e-4, "class_weight": "balanced"},
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
                    "loss_function": "Logloss",
                },
            }
        except ImportError:
            pass

        return candidates

    # ── Model fitting ────────────────────────────────────────────────
    def _fit_one(self, kind, params, X_tr, y_tr, X_va, y_va, sw=None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import ExtraTreesClassifier
        y_tr_arr = np.asarray(y_tr, dtype=float)
        y_va_arr = np.asarray(y_va, dtype=float)
        y_tr_bin = (y_tr_arr >= 0.5).astype(np.int8)
        y_va_bin = (y_va_arr >= 0.5).astype(np.int8)

        if kind == "ridge_clf":
            if y_tr.ndim == 2:
                raise ValueError("ridge_clf does not support proper multiclass cross-entropy for soft labels. Use xgb_clf instead.")
            else:
                model = Pipeline([
                    ("scaler", RobustScaler()),
                    ("clf", LogisticRegression(**params)),
                ])
                model.fit(X_tr, y_tr_bin, clf__sample_weight=sw)
                return model
        if kind == "et_clf":
            if y_tr.ndim == 2:
                raise ValueError("et_clf does not support proper multiclass cross-entropy for soft labels. Use xgb_clf instead.")
            else:
                model = ExtraTreesClassifier(**params)
                model.fit(X_tr, y_tr_bin, sample_weight=sw)
                return model
        if kind == "catboost_clf":
            import catboost
            p = dict(params)
            if y_tr.ndim == 2:
                raise ValueError("catboost_clf does not support proper multiclass cross-entropy for soft labels. Use xgb_clf instead.")
            else:
                model = catboost.CatBoostClassifier(**p)
                model.fit(
                    X_tr,
                    y_tr_bin,
                    sample_weight=sw,
                    eval_set=(X_va, y_va_bin),
                    early_stopping_rounds=50,
                    verbose=False,
                )
                return model
        if kind == "xgb_clf":
            p = dict(params)
            if y_tr.ndim == 2:
                # Proper Soft-Target Multiclass Cross-Entropy for XGBoost
                # Replace standard objective with a custom one
                if "objective" in p:
                    del p["objective"]
                if "eval_metric" in p:
                    del p["eval_metric"]
                p["disable_default_eval_metric"] = 1

                mc = getattr(self, "monotone_constraints", None)
                if mc and isinstance(mc, dict) and self.selected_features:
                    p["monotone_constraints"] = tuple(
                        mc.get(f, 0) for f in self.selected_features
                    )

                # Number of classes is assumed to be y_tr.shape[1] (which is 3)
                n_classes = y_tr.shape[1]
                p["num_class"] = n_classes

                class TrueSoftXGBWrapper:
                    def __init__(self, p_dict):
                        self.params = p_dict
                        self.bst = None

                    def _soft_crossentropy_obj(self, y_soft):
                        def obj(preds, dtrain):
                            if preds.ndim == 1:
                                preds = preds.reshape(-1, n_classes)
                            # softmax
                            preds_shifted = preds - np.max(preds, axis=1, keepdims=True)
                            exp_preds = np.exp(preds_shifted)
                            prob = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)

                            # grad and hess for softmax cross-entropy
                            grad = prob - y_soft
                            hess = prob * (1.0 - prob)
                            return grad, hess
                        return obj

                    def fit(self, X, y, sample_weight=None, eval_set=None, verbose=False):
                        # Ensure inputs are valid
                        # Create dummy DMatrix label because XGB custom obj handles soft labels directly
                        dummy_y_tr = np.zeros(len(y))
                        dtrain = xgb.DMatrix(X, label=dummy_y_tr, weight=sample_weight)

                        evals = []
                        if eval_set:
                            X_va, y_va = eval_set[0]
                            dummy_y_va = np.zeros(len(y_va))
                            dvalid = xgb.DMatrix(X_va, label=dummy_y_va)
                            evals = [(dtrain, "train"), (dvalid, "valid")]

                        self.bst = xgb.train(
                            self.params,
                            dtrain,
                            num_boost_round=self.params.get("n_estimators", 100),
                            evals=evals,
                            obj=self._soft_crossentropy_obj(y),
                            verbose_eval=False
                        )
                        return self

                    def predict_proba(self, X):
                        dtest = xgb.DMatrix(X)
                        preds = self.bst.predict(dtest, output_margin=True)
                        if preds.ndim == 1:
                            preds = preds.reshape(-1, n_classes)
                        preds_shifted = preds - np.max(preds, axis=1, keepdims=True)
                        exp_preds = np.exp(preds_shifted)
                        prob = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
                        return prob

                    @property
                    def classes_(self):
                        return np.array([0, 1, 2])

                model = TrueSoftXGBWrapper(p)
                model.fit(X_tr, y_tr, sample_weight=sw, eval_set=[(X_va, y_va)], verbose=False)
                return model
            else:
                model = xgb.XGBRegressor(**p)
                model.fit(
                    X_tr,
                    y_tr_arr,
                    sample_weight=sw,
                    eval_set=[(X_va, y_va_arr)],
                    verbose=False,
                )
                return model
        raise ValueError(f"Unknown classifier kind: {kind}")

    def _predict_proba(self, model, X):
        """Get positive-class probabilities for the binary move head."""
        if hasattr(model, "predict_proba"):
            pp_raw = np.asarray(model.predict_proba(X), dtype=np.float64)
        else:
            pp_raw = np.asarray(model.predict(X), dtype=np.float64)
        if pp_raw.ndim == 1:
            out = pp_raw
        elif pp_raw.ndim == 2 and pp_raw.shape[1] == 2:
            out = pp_raw[:, 1]
        elif pp_raw.ndim == 2 and pp_raw.shape[1] == 1:
            out = pp_raw[:, 0]
        else:
            raise ValueError(f"predict_proba returned invalid shape: {pp_raw.shape}")

        out = np.asarray(out, dtype=np.float64).reshape(-1)
        out = np.where(np.isfinite(out), out, 0.5)
        out = np.clip(out, 0.0, 1.0)
        return out

    @staticmethod
    def _build_soft_move_target(
        abs_ret: np.ndarray,
        vol_proxy: np.ndarray,
        thresholds: Sequence[float],
        weights: Sequence[float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        abs_ret = np.asarray(abs_ret, dtype=np.float64).reshape(-1)
        vol_proxy = np.asarray(vol_proxy, dtype=np.float64).reshape(-1)
        thresh = np.asarray(list(thresholds), dtype=np.float64).reshape(-1)
        w = np.asarray(list(weights), dtype=np.float64).reshape(-1)
        if thresh.size == 0:
            thresh = np.asarray([1.0, 1.25, 1.5], dtype=np.float64)
        if w.size == 0:
            w = np.asarray([0.45, 0.35, 0.20], dtype=np.float64)
        if thresh.size != w.size:
            n = min(thresh.size, w.size)
            thresh = thresh[:n]
            w = w[:n]
        if thresh.size == 0:
            thresh = np.asarray([1.0, 1.25, 1.5], dtype=np.float64)
            w = np.asarray([0.45, 0.35, 0.20], dtype=np.float64)
        if not np.isfinite(np.sum(w)) or float(np.sum(w)) <= 0.0:
            w = np.ones_like(thresh, dtype=np.float64)
        w = w / max(float(np.sum(w)), 1e-12)
        vp = np.clip(vol_proxy, 1e-9, None)
        ladder = []
        for k in thresh:
            ladder.append((abs_ret > (float(k) * vp)).astype(np.float32))
        stack = np.vstack(ladder).astype(np.float32)
        soft = np.dot(w.astype(np.float32), stack).astype(np.float32)
        hard = (soft >= 0.5).astype(np.int8)
        middle_idx = 1 if len(thresh) > 1 else 0
        move_thr = (float(thresh[middle_idx]) * vp).astype(np.float32)
        return soft, hard, move_thr

    # ── CV evaluation ────────────────────────────────────────────────
    def _cv_evaluate(
        self,
        kind,
        params,
        X,
        y,
        sw=None,
        feature_max_features: Optional[int] = None,
    ) -> Tuple[np.ndarray, float, Dict[str, Any], Dict[str, np.ndarray]]:
        """3-fold purged CV for the binary move head."""
        pkf = PurgedKFold(n_splits=3, purge=12, embargo=12)
        oof = np.full(len(y), np.nan, dtype=float)
        oof_sigma = np.full(len(y), np.nan, dtype=np.float32)
        oof_robust_sigma = np.full(len(y), np.nan, dtype=np.float32)

        if isinstance(X, pd.DataFrame):
            X_frame = X.reset_index(drop=True)
        else:
            X_frame = pd.DataFrame(
                np.asarray(X, dtype=np.float32),
                columns=[f"f_{i}" for i in range(np.asarray(X).shape[1])],
            )
        max_features = int(
            feature_max_features if feature_max_features is not None else max(30, min(60, X_frame.shape[1]))
        )

        y_arr = np.asarray(y, dtype=float).reshape(-1)
        y_hard = (y_arr >= 0.5).astype(np.int8)

        for tr, va in pkf.split(X_frame):
            y_tr_soft = np.asarray(y_arr[tr], dtype=float)
            y_va_soft = np.asarray(y_arr[va], dtype=float)
            y_tr = (y_tr_soft >= 0.5).astype(np.int8)
            y_va = (y_va_soft >= 0.5).astype(np.int8)
            sw_tr = None if sw is None else np.asarray(sw[tr], dtype=float)
            if y_tr.size == 0 or y_va.size == 0:
                continue
            n_pos = int(np.sum(y_tr == 1))
            n_neg = int(np.sum(y_tr == 0))
            if n_pos < 1 or n_neg < 1:
                continue
            fold_sw = None
            pos_weight = float(n_neg) / max(float(n_pos), 1.0)
            pos_weight = float(np.clip(pos_weight, 1.0, max(float(self.max_class_weight), 1.0)))
            if sw_tr is not None:
                fold_sw = sw_tr.copy()
                fold_sw[y_tr == 1] *= pos_weight
            else:
                fold_sw = np.ones(len(y_tr), dtype=float)
                fold_sw[y_tr == 1] *= pos_weight
            fold_sw = fold_sw / max(float(np.mean(fold_sw)), 1e-12)
            X_tr_df = X_frame.iloc[tr].reset_index(drop=True)
            X_va_df = X_frame.iloc[va].reset_index(drop=True)
            selected_cols = self._select_tail_features(
                X_tr_df, np.asarray(y_tr_soft, dtype=float), max_features=max_features
            )
            X_tr = X_tr_df[selected_cols].to_numpy(dtype=np.float32)
            X_va = X_va_df[selected_cols].to_numpy(dtype=np.float32)
            try:
                model = self._fit_one(kind, params, X_tr, y_tr_soft, X_va, y_va_soft, sw=fold_sw)
                pp = self._predict_proba(model, X_va)
                oof[va] = pp
                sigma_fold, sigma_robust_fold = _xgb_staged_dispersion(
                    model,
                    X_va,
                    kind="classifier" if kind == "xgb" else "regression",
                )
                oof_sigma[va] = sigma_fold
                oof_robust_sigma[va] = sigma_robust_fold
            except Exception as e:
                tprint(f"Warning: Fold evaluation failed ({e}). Returning 1/3 prior.")
                oof[va] = 0.5

        # Purged/embargo CV may leave boundary rows without OOF predictions.
        # Fill with a safe prior so downstream selection metrics stay finite.
        missing = ~np.isfinite(oof)
        if np.any(missing):
            fitted = np.isfinite(oof)
            if np.any(fitted):
                prior = float(np.nanmean(oof[fitted]))
            else:
                prior = float(np.mean(np.asarray(y, dtype=float))) if len(y) else 0.5
            if not np.isfinite(prior):
                prior = 0.5
            oof[missing] = prior

        mask = np.isfinite(oof)
        if mask.sum() < 20:
            return oof, 999.0, {}, {
                "oof_sigma": oof_sigma,
                "oof_robust_sigma": oof_robust_sigma,
            }
        try:
            from sklearn.metrics import log_loss

            score = float(
                log_loss(
                    y_hard[mask],
                    np.clip(oof[mask], 1e-6, 1 - 1e-6),
                    labels=[0, 1],
                )
            )
        except Exception:
            score = 999.0
        return oof, score, {}, {
            "oof_sigma": oof_sigma,
            "oof_robust_sigma": oof_robust_sigma,
        }

    # ── Comprehensive classifier metrics ─────────────────────────────
    @staticmethod
    def _compute_clf_metrics(
        oof: np.ndarray,
        y_move: np.ndarray,
        realized_abs_return: np.ndarray,
        groups=None,
        fee: float = 0.005,
    ) -> dict:
        """Compute binary move-head metrics and calibration diagnostics."""
        from sklearn.metrics import (
            average_precision_score,
            balanced_accuracy_score,
            brier_score_loss,
            log_loss,
            roc_auc_score,
            roc_curve,
        )
        from sklearn.linear_model import LogisticRegression

        mask = np.isfinite(oof) & np.isfinite(y_move) & np.isfinite(realized_abs_return)
        pred = np.asarray(oof, dtype=float)[mask]
        y = np.asarray(y_move, dtype=int)[mask]
        r = np.asarray(realized_abs_return, dtype=float)[mask]
        n = len(pred)
        if n < 20:
            return {"n": n}

        pred = np.clip(pred, 1e-6, 1 - 1e-6)
        try:
            ll = float(log_loss(y, pred, labels=[0, 1]))
        except Exception:
            ll = float("nan")
        try:
            roc = float(roc_auc_score(y, pred))
        except Exception:
            roc = float("nan")
        try:
            pr_auc = float(average_precision_score(y, pred))
        except Exception:
            pr_auc = float("nan")
        try:
            brier = float(brier_score_loss(y, pred))
        except Exception:
            brier = float("nan")

        pred_05 = (pred >= 0.5).astype(int)
        bal_05 = float(balanced_accuracy_score(y, pred_05))
        try:
            fpr, tpr, thr = roc_curve(y, pred)
            youden = tpr - fpr
            best_idx = int(np.argmax(youden))
            best_thr = float(thr[best_idx])
        except Exception:
            best_thr = 0.5
        pred_best = (pred >= best_thr).astype(int)
        bal_best = float(balanced_accuracy_score(y, pred_best))

        # Spearman IC between p_move and realized absolute return
        try:
            ic = float(spearmanr(pred, r).correlation)
            if not np.isfinite(ic):
                ic = 0.0
        except Exception:
            ic = 0.0

        slope = float("nan")
        intercept = float("nan")
        try:
            logits = np.log(
                np.clip(pred, 1e-6, 1 - 1e-6) / np.clip(1.0 - pred, 1e-6, 1 - 1e-6)
            )
            _lr = LogisticRegression(solver="lbfgs", max_iter=1000)
            _lr.fit(logits.reshape(-1, 1), y)
            slope = float(_lr.coef_[0][0])
            intercept = float(_lr.intercept_[0])
        except Exception:
            pass

        def _top_slice_metrics(frac: float) -> Tuple[float, float, float, float]:
            k = max(1, int(np.ceil(float(frac) * n)))
            idx = np.argsort(-pred)[:k]
            absret_mean = float(np.mean(r[idx]))
            lift = absret_mean - float(np.mean(r))
            precision = float(np.mean(y[idx]))
            recall = float(np.sum(y[idx]) / max(np.sum(y), 1))
            return absret_mean, lift, precision, recall

        top10_absret, top10_lift, top10_prec, top10_rec = _top_slice_metrics(0.10)
        metrics = {
            "n": n,
            "logloss": ll,
            "roc_auc": roc,
            "pr_auc": pr_auc,
            "pr_auc_base_rate": float(np.mean(y)),
            "pr_auc_vs_base_rate": float(pr_auc - np.mean(y)) if np.isfinite(pr_auc) else float("nan"),
            "pr_auc_lift_vs_base_rate": float(pr_auc / max(float(np.mean(y)), 1e-9))
            if np.isfinite(pr_auc)
            else float("nan"),
            "brier": brier,
            "brier_score": brier,
            "balanced_accuracy_0p5": bal_05,
            "balanced_accuracy_best": bal_best,
            "best_threshold": best_thr,
            "move_ic": ic,
            "calibration_slope": slope,
            "calibration_intercept": intercept,
            "base_rate": float(np.mean(y)),
            "move_base_rate": float(np.mean(y)),
            "absret_mean": float(np.mean(r)),
            "top10_absret_mean": top10_absret,
            "top10_lift": top10_lift,
            "top10_hit_rate": top10_prec,
            "top_decile_precision": top10_prec,
            "top_decile_recall": top10_rec,
            "top_decile_absret_mean": top10_absret,
            "top_decile_absret_lift": top10_lift,
            "top_decile_absret_hit_rate": top10_prec,
        }
        for frac in (0.05, 0.10, 0.20):
            absret_mean, lift, precision, recall = _top_slice_metrics(frac)
            key = int(round(frac * 100))
            metrics[f"top{key}_absret_mean"] = absret_mean
            metrics[f"top{key}_lift"] = lift
            metrics[f"top{key}_precision"] = precision
            metrics[f"top{key}_recall"] = recall

        # Optional calibration curve diagnostic.
        edges = np.percentile(pred, np.linspace(0, 100, min(10, n) + 1))
        def _fixed_bin_ece(n_bins: int) -> tuple[float, list[dict[str, float]]]:
            edges = np.linspace(0.0, 1.0, n_bins + 1)
            curve: list[dict[str, float]] = []
            ece = 0.0
            for b in range(n_bins):
                lo, hi = float(edges[b]), float(edges[b + 1])
                if b == n_bins - 1:
                    in_bin = (pred >= lo) & (pred <= hi)
                else:
                    in_bin = (pred >= lo) & (pred < hi)
                if not np.any(in_bin):
                    continue
                pred_mean = float(np.mean(pred[in_bin]))
                true_mean = float(np.mean(y[in_bin]))
                frac = float(np.mean(in_bin))
                ece += abs(pred_mean - true_mean) * frac
                curve.append(
                    {
                        "bin": int(b),
                        "lo": lo,
                        "hi": hi,
                        "n": int(np.sum(in_bin)),
                        "pred_mean": pred_mean,
                        "true_mean": true_mean,
                        "gap": float(pred_mean - true_mean),
                    }
                )
            return float(ece), curve

        ece_10, curve_10 = _fixed_bin_ece(10)
        ece_15, curve_15 = _fixed_bin_ece(15)
        metrics["calibration_curve_10"] = curve_10
        metrics["calibration_curve_15"] = curve_15
        metrics["ece_10"] = ece_10
        metrics["ece_15"] = ece_15
        metrics["ece"] = ece_10
        return metrics

    @staticmethod
    def _compute_soft_label_diagnostics(
        pred: np.ndarray,
        y_true: np.ndarray,
        n_bins: int = 10,
    ) -> dict:
        """Soft-label monitoring: simplex violation, entropy, KL divergence,
        hard-label agreement, and P(TP) calibration."""
        pred = np.asarray(pred, dtype=np.float64)
        if pred.ndim != 2 or pred.shape[1] != 3:
            return {}
        n = pred.shape[0]
        if n < 20:
            return {}

        row_sums = pred.sum(axis=1)
        simplex_violation_rate = float(np.mean(np.abs(row_sums - 1.0) > 1e-4))

        eps = 1e-12
        p_safe = np.clip(pred, eps, 1.0 - eps)
        entropy = -np.sum(p_safe * np.log(p_safe), axis=1)
        max_entropy = np.log(3.0)

        y_true_arr = np.asarray(y_true, dtype=np.float64)
        is_soft = y_true_arr.ndim == 2 and y_true_arr.shape[1] == 3

        if is_soft:
            y_soft = np.clip(y_true_arr, eps, 1.0 - eps)
            kl_per_row = np.sum(y_soft * (np.log(y_soft) - np.log(p_safe)), axis=1)
            kl_divergence = float(np.mean(kl_per_row))

            argmax_true = np.argmax(y_soft, axis=1)
            argmax_pred = np.argmax(pred, axis=1)
            hard_agreement_rate = float(np.mean(argmax_true == argmax_pred))

            y_tp_true = y_soft[:, 2]
        else:
            kl_divergence = float("nan")
            y_hard = np.asarray(y_true, dtype=np.int64)
            argmax_pred = np.argmax(pred, axis=1)
            hard_agreement_rate = float(np.mean(y_hard == argmax_pred))
            y_tp_true = (y_hard == 2).astype(np.float64)

        pred_tp = pred[:, 2]
        n_ece = min(n_bins, n)
        bin_edges = np.percentile(pred_tp, np.linspace(0, 100, n_ece + 1))
        bin_edges[0] -= 1e-6
        bin_edges[-1] += 1e-6
        ece_tp = 0.0
        for b in range(n_ece):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            in_bin = (pred_tp >= lo) & (pred_tp < hi) if b < n_ece - 1 else (pred_tp >= lo) & (pred_tp <= hi)
            n_bin = int(in_bin.sum())
            if n_bin == 0:
                continue
            mean_pred = float(np.mean(pred_tp[in_bin]))
            mean_true = float(np.mean(y_tp_true[in_bin]))
            ece_tp += abs(mean_pred - mean_true) * (n_bin / n)

        return {
            "simplex_violation_rate": simplex_violation_rate,
            "entropy_median": float(np.median(entropy)),
            "entropy_p10": float(np.percentile(entropy, 10)),
            "entropy_p90": float(np.percentile(entropy, 90)),
            "entropy_max_possible": float(max_entropy),
            "kl_divergence_mean": kl_divergence,
            "hard_agreement_rate": hard_agreement_rate,
            "ece_tp": float(ece_tp),
        }

    def _optuna_hpo(
        self,
        name: str,
        kind: str,
        params: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        sw: Optional[np.ndarray] = None,
        n_trials: int = 150,
    ) -> Dict[str, Any]:
        if importlib.util.find_spec("optuna") is None:
            return params
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        out_dir = str(self.hpo_out_dir or os.path.join("data", "hpo_out"))
        scope_key = str(self.strategy_name or "default")
        prev_blob = _load_meta_hpo_json(out_dir, scope_key=scope_key, kind=kind)
        prev_params = (
            dict(prev_blob.get("best_params", {}))
            if isinstance(prev_blob, dict) and prev_blob.get("best_params")
            else None
        )

        def objective(trial: optuna.Trial) -> float:
            p = dict(params)
            if kind == "ridge_clf":
                p["C"] = trial.suggest_float("C", 0.01, 25.0, log=True)
            elif kind == "et_clf":
                p["n_estimators"] = trial.suggest_int("n_estimators", 200, 900, step=100)
                p["max_depth"] = trial.suggest_int("max_depth", 4, 12)
                p["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 10, 120)
                p["min_samples_split"] = trial.suggest_int("min_samples_split", 20, 240)
                p["max_features"] = trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", 0.25, 0.33, 0.5]
                )
                p["ccp_alpha"] = trial.suggest_float("ccp_alpha", 1e-6, 1e-3, log=True)
            elif kind == "catboost_clf":
                p["iterations"] = trial.suggest_int("iterations", 300, 1200, step=100)
                p["depth"] = trial.suggest_int("depth", 3, 7)
                p["learning_rate"] = trial.suggest_float(
                    "learning_rate", 0.01, 0.15, log=True
                )
                p["l2_leaf_reg"] = trial.suggest_float(
                    "l2_leaf_reg", 1.0, 50.0, log=True
                )
                p["bagging_temperature"] = trial.suggest_float(
                    "bagging_temperature", 0.0, 2.0
                )
                p["rsm"] = trial.suggest_float("rsm", 0.5, 0.95)
            elif kind == "xgb_clf":
                p["num_parallel_tree"] = 5
                p["n_estimators"] = trial.suggest_int("n_estimators", 200, 400, step=50)
                p["max_depth"] = trial.suggest_int("max_depth", 3, 7)
                p["learning_rate"] = trial.suggest_float(
                    "learning_rate", 0.01, 0.15, log=True
                )
                p["subsample"] = trial.suggest_float("subsample", 0.5, 0.9)
                p["colsample_bytree"] = trial.suggest_float(
                    "colsample_bytree", 0.5, 0.9
                )
                p["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-6, 15.0, log=True)
                p["reg_lambda"] = trial.suggest_float("reg_lambda", 1.0, 60.0, log=True)
                p["min_child_weight"] = trial.suggest_float(
                    "min_child_weight", 10.0, 120.0, log=True
                )
                p["gamma"] = trial.suggest_float("gamma", 0.0, 5.0)
            else:
                raise ValueError(f"Unsupported classifier kind for HPO: {kind}")

            try:
                _, score, metrics, _ = self._cv_evaluate(kind, p, X, y, sw)
            except Exception as exc:
                tprint(f"  Meta clf HPO trial failed[{scope_key}/{kind}]: {exc}")
                return -1e9

            # For classifiers, use composite score with classifier-specific metrics
            # rank_metric = median_fold_aucpr_lift (computed as selection metric)
            # top_bucket_lift = top_decile_target_lift
            # top_bucket_mean = top_decile_target_mean
            # rank_metric_std = std_fold_aucpr_lift
            # error_metric = Brier score
            rank_metric = metrics.get("median_fold_spearman", 0.0)  # Proxy for aucpr_lift
            top_bucket_lift = metrics.get("mean_top_decile_lift", 0.0)
            top_bucket_mean = metrics.get("mean_top_decile_mean", 0.0)
            rank_metric_std = metrics.get("std_fold_spearman", 0.0)
            error_metric = metrics.get("mean_brier", 0.5)  # Brier score

            # Get running stats
            stats_rank = _running_stats_meta["rank_metric"]
            stats_lift = _running_stats_meta["top_bucket_lift"]
            stats_mean = _running_stats_meta["top_bucket_mean"]
            stats_std = _running_stats_meta["rank_metric_std"]
            stats_err = _running_stats_meta["error_metric"]

            # Compute z-scores
            z_rank = stats_rank.zscore(rank_metric) if stats_rank.n >= 10 else rank_metric * 5.0
            z_lift = stats_lift.zscore(top_bucket_lift) if stats_lift.n >= 10 else top_bucket_lift * 5.0
            z_mean = stats_mean.zscore(top_bucket_mean) if stats_mean.n >= 10 else top_bucket_mean * 5.0
            z_std = stats_std.zscore(rank_metric_std) if stats_std.n >= 10 else rank_metric_std * 5.0
            z_err = stats_err.zscore(error_metric) if stats_err.n >= 10 else (error_metric - 0.25) * 5.0

            # Update running stats
            stats_rank.update(rank_metric)
            stats_lift.update(top_bucket_lift)
            stats_mean.update(top_bucket_mean)
            stats_std.update(rank_metric_std)
            stats_err.update(error_metric)

            # Composite score (same weights as regressors)
            composite = (
                0.45 * z_rank
                + 0.30 * z_lift
                + 0.20 * z_mean
                - 0.20 * z_std
                - 0.10 * z_err
            )

            # Store metrics
            trial.set_user_attr("rank_metric", rank_metric)
            trial.set_user_attr("top_bucket_lift", top_bucket_lift)
            trial.set_user_attr("top_bucket_mean", top_bucket_mean)
            trial.set_user_attr("rank_metric_std", rank_metric_std)
            trial.set_user_attr("error_metric", error_metric)

            return float(composite)

        study = optuna.create_study(
            direction="maximize",
            study_name=f"meta_clf_hpo_{name}",
            pruner=SuccessiveHalvingPruner(
                min_resource=1,
                reduction_factor=2,
                min_early_stopping_rate=0,
                bootstrap_count=5,
            ),
        )
        if prev_params:
            try:
                study.enqueue_trial(prev_params)
                tprint(
                    f"  Meta clf HPO warm-start[{scope_key}/{kind}]: loaded previous params from "
                    f"{_meta_hpo_json_path(out_dir, scope_key=scope_key, kind=kind)}"
                )
            except Exception as exc:
                tprint(f"  Meta clf HPO warm-start[{scope_key}/{kind}] failed: {exc}")

        study.optimize(objective, n_trials=n_trials, timeout=None, callbacks=[_make_early_stopping_callback()], gc_after_trial=True)
        if study.best_trial is None:
            return params

        tuned = dict(params)
        tuned.update(study.best_params)
        payload = {
            "scope_key": scope_key,
            "kind": kind,
            "best_params": dict(study.best_params),
            "n_trials_completed": len(study.trials),
        }
        try:
            _save_meta_hpo_json(out_dir, payload, scope_key=scope_key, kind=kind)
            tprint(
                f"  Meta clf HPO saved params[{scope_key}/{kind}] to "
                f"{_meta_hpo_json_path(out_dir, scope_key=scope_key, kind=kind)}"
            )
        except Exception as exc:
            tprint(f"  Meta clf HPO save[{scope_key}/{kind}] failed: {exc}")
        return tuned

    # ── Multi-barrier label construction ────────────────────────────
    @staticmethod
    def _build_multiclass_labels(
        y_per_horizon: Dict[int, np.ndarray],
        vol_proxy: np.ndarray,
        k_tp: float = 2.0,
        k_sl: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build 3-class labels (0=SL, 1=Timeout, 2=TP) using risk-unit thresholds.

        Conflict resolution: when both TP and SL are hit, uses first-hit-wins
        based on time-to-barrier information (approximated from return sign at
        each horizon). If barrier timing is unavailable, defaults to TP-wins
        (optimistic but avoids the previous conservative SL-override bug).
        """
        n = len(vol_proxy)
        y_class = np.ones(n, dtype=np.int8)

        vp = np.clip(vol_proxy, 1e-9, None)
        tp_thresh = k_tp * vp
        sl_thresh = k_sl * vp

        max_ret = np.full(n, -999.0)
        min_ret = np.full(n, 999.0)
        first_tp_h = np.full(n, np.inf)
        first_sl_h = np.full(n, np.inf)

        for h, y_h in y_per_horizon.items():
            y_h = np.asarray(y_h, dtype=np.float64)
            max_ret = np.maximum(max_ret, y_h)
            min_ret = np.minimum(min_ret, y_h)
            hit_tp_h = y_h >= tp_thresh
            hit_sl_h = y_h <= -sl_thresh
            first_tp_h = np.where(hit_tp_h & (h < first_tp_h), h, first_tp_h)
            first_sl_h = np.where(hit_sl_h & (h < first_sl_h), h, first_sl_h)

        hit_tp = max_ret >= tp_thresh
        hit_sl = min_ret <= -sl_thresh
        hit_both = hit_tp & hit_sl

        y_class[hit_tp & ~hit_sl] = 2
        y_class[hit_sl & ~hit_tp] = 0

        first_tp_wins = hit_both & (first_tp_h <= first_sl_h)
        first_sl_wins = hit_both & (first_tp_h > first_sl_h)
        y_class[first_tp_wins] = 2
        y_class[first_sl_wins] = 0

        w_class = np.ones(n, dtype=np.float32)
        return y_class, w_class

    # ── Main fit ─────────────────────────────────────────────────────
    def fit(
        self,
        X_meta: pd.DataFrame,
        y_ret: np.ndarray,
        sample_weight=None,
        groups=None,
        y_per_horizon: Optional[Dict[int, np.ndarray]] = None,
        vol_proxy: Optional[np.ndarray] = None,
        realized_u_policy: Optional[np.ndarray] = None,
        selection_cfg: Optional[MetaMoveSelectionConfig] = None,
        y_move_override: Optional[np.ndarray] = None,
        y_class_override: Optional[np.ndarray] = None,
        trade_mask: Optional[np.ndarray] = None,
        move_k: Optional[float] = None,
        move_k_by_h: Optional[Dict[int, float]] = None,
        move_thresholds: Optional[Sequence[float]] = None,
        move_weights: Optional[Sequence[float]] = None,
        use_class_weight_multiplier: Optional[bool] = None,
        max_class_weight: Optional[float] = None,
        use_calibration: Optional[bool] = None,
        move_horizon: Optional[int] = None,
    ):
        """Fit a binary move classifier on a soft move ladder."""
        import time as _time
        from sklearn.isotonic import IsotonicRegression

        _t0 = _time.monotonic()
        tprint(
            f"MetaClassifierModel.fit: {self.strategy_name} starting "
            f"(n={len(y_ret)}, feats={X_meta.shape[1]})"
        )
        self.move_horizon = move_horizon
        self.move_k = float(move_k if move_k is not None else self.move_k)
        self.move_k_by_h = dict(move_k_by_h or {})
        self.move_thresholds = tuple(float(x) for x in (move_thresholds or self.move_thresholds))
        self.move_weights = tuple(float(x) for x in (move_weights or self.move_weights))
        self.use_class_weight_multiplier = bool(
            self.use_class_weight_multiplier if use_class_weight_multiplier is None else use_class_weight_multiplier
        )
        self.max_class_weight = float(self.max_class_weight if max_class_weight is None else max_class_weight)
        self.use_calibration = bool(self.use_calibration if use_calibration is None else use_calibration)

        y_ret_np = np.asarray(y_ret, dtype=float)
        sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        X_meta_work = X_meta.reset_index(drop=True) if isinstance(X_meta, pd.DataFrame) else pd.DataFrame(X_meta)

        if vol_proxy is None:
            raise ValueError("MetaClassifierModel requires a causal vol_proxy for binary move labeling.")
        vol_proxy = np.asarray(vol_proxy, dtype=float)
        valid_vol = np.isfinite(vol_proxy) & (vol_proxy > 1e-9)
        if not valid_vol.all():
            n_drop = int((~valid_vol).sum())
            tprint(f"  Dropping {n_drop} samples with invalid vol_proxy.")
            valid_idx = np.where(valid_vol)[0]
            X_meta_work = X_meta_work.iloc[valid_idx].reset_index(drop=True)
            y_ret_np = y_ret_np[valid_idx]
            vol_proxy = vol_proxy[valid_idx]
            if sw is not None:
                sw = sw[valid_idx]
            if groups is not None:
                groups = np.asarray(groups)[valid_idx]
            if trade_mask is not None:
                trade_mask = np.asarray(trade_mask, dtype=bool)[valid_idx]
            if y_per_horizon is not None:
                y_per_horizon = {h: np.asarray(v)[valid_idx] for h, v in y_per_horizon.items()}

        move_threshold_arr = None
        if y_move_override is not None or y_class_override is not None:
            raw_override = y_move_override if y_move_override is not None else y_class_override
            y_soft = np.asarray(raw_override, dtype=float).reshape(-1)
            y_soft = y_soft[: len(X_meta_work)]
            y_soft = np.clip(np.where(np.isfinite(y_soft), y_soft, 0.0), 0.0, 1.0).astype(np.float32)
            y_hard = (y_soft >= 0.5).astype(np.int8)
            tprint(
                f"  META MOVE target: using override labels; base_rate={float(np.mean(y_hard)):.4f} "
                f"soft_mean={float(np.mean(y_soft)):.4f}"
            )
        else:
            y_soft, y_hard, move_threshold_arr = self._build_soft_move_target(
                abs_ret=np.abs(y_ret_np),
                vol_proxy=vol_proxy,
                thresholds=self.move_thresholds,
                weights=self.move_weights,
            )
            y_soft = y_soft[: len(X_meta_work)]
            y_hard = y_hard[: len(X_meta_work)]
            move_threshold_arr = move_threshold_arr[: len(X_meta_work)]
            tprint(
                f"  META MOVE target: horizon={move_horizon if move_horizon is not None else 'n/a'} "
                f"thresholds={list(self.move_thresholds)} weights={list(self.move_weights)} "
                f"base_rate={float(np.mean(y_hard)):.4f} soft_mean={float(np.mean(y_soft)):.4f} "
                f"threshold_mean={float(np.mean(move_threshold_arr)):.6f}"
            )

        if trade_mask is None:
            trade_mask = np.ones(len(y_soft), dtype=bool)
        else:
            trade_mask = np.asarray(trade_mask, dtype=bool)[: len(y_soft)]

        if sw is None:
            sw = np.ones(len(y_soft), dtype=float)
        sw = np.asarray(sw, dtype=float)
        sw = np.where(np.isfinite(sw), sw, 1.0)
        sw = sw / max(float(np.mean(sw)), 1e-12)

        if self.use_class_weight_multiplier:
            n_pos = int(np.sum(y_hard == 1))
            n_neg = int(np.sum(y_hard == 0))
            pos_weight = float(n_neg) / max(float(n_pos), 1.0)
            pos_weight = float(np.clip(pos_weight, 1.0, max(float(self.max_class_weight), 1.0)))
            class_mult = np.ones(len(y_hard), dtype=float)
            class_mult[y_hard == 1] *= pos_weight
            sw = sw * class_mult
            sw = sw / max(float(np.mean(sw)), 1e-12)
            tprint(
                f"  Class weights: n_pos={n_pos} n_neg={n_neg} pos_weight={pos_weight:.3f} "
                f"effective_mean={float(np.mean(sw)):.3f}"
            )

        self._feature_selection_max_features = max(30, min(60, X_meta_work.shape[1]))
        self.selected_features = list(X_meta_work.columns)
        tprint(
            f"  Features: {X_meta.shape[1]} -> {len(self.selected_features)} "
            "(fold-local selection handled inside CV)"
        )

        candidates = self._build_candidates()
        records = []
        scored = []
        best_rec = None
        best_aux = None
        if selection_cfg is None:
            selection_cfg = MetaMoveSelectionConfig()

        realized_abs_return = np.abs(y_ret_np)

        for name, cand in candidates.items():
            kind = cand["kind"]
            params = dict(cand["params"])
            try:
                oof_raw, logloss, _, aux = self._cv_evaluate(
                    kind,
                    params,
                    X_meta_work,
                    y_soft,
                    sw,
                    feature_max_features=self._feature_selection_max_features,
                )
            except Exception as exc:
                tprint(f"    {name} failed: {exc}")
                continue

            metrics = self._compute_clf_metrics(
                oof_raw, y_hard, realized_abs_return, groups=groups, fee=self.FEE_PER_ROUND_TRIP
            )
            sel = pick_meta_move_by_topq(
                y_true=y_hard,
                p_pred=oof_raw,
                realized_abs_return=realized_abs_return,
                cfg=selection_cfg,
                trade_mask=trade_mask,
            )
            metrics["model"] = name
            metrics["logloss_cv"] = logloss
            metrics["selection_score"] = float(sel.get("selection_score", float("nan")))
            metrics["top_decile_absret_mean"] = float(sel.get("top_decile_absret_mean", float("nan")))
            metrics["top_decile_lift"] = float(sel.get("top_decile_lift", float("nan")))
            metrics["top_decile_hit_rate"] = float(sel.get("top_decile_hit_rate", float("nan")))
            metrics["passed_gate"] = float(sel.get("passed_gate", 0.0))
            metrics["passed_econ"] = float(sel.get("passed_econ", 0.0))

            records.append(metrics)
            scored.append((name, kind, params, oof_raw, y_soft, y_hard, metrics, sel, aux))
            tprint(
                f"    {name}: LogLoss={logloss:.4f}, AUC={metrics.get('roc_auc',0):.3f}, "
                f"PR={metrics.get('pr_auc',0):.3f}, Brier={metrics.get('brier',0):.4f}, "
                f"BalAcc@0.5={metrics.get('balanced_accuracy_0p5',0):.3f}, "
                f"BalAcc*={metrics.get('balanced_accuracy_best',0):.3f}, "
                f"ECE10={metrics.get('ece_10', metrics.get('ece',0)):.3f}, "
                f"Slope={metrics.get('calibration_slope', float('nan')):.3f}, "
                f"Int={metrics.get('calibration_intercept', float('nan')):.3f}, "
                f"BaseR={metrics.get('base_rate',0):.3f}, "
                f"IC={metrics.get('move_ic',0):.3f}, "
                f"Top10Lift={sel.get('top_decile_lift', float('nan')):.5f}, "
                f"Top10P={metrics.get('top_decile_precision', sel.get('top_decile_hit_rate', float('nan'))):.3f}, "
                f"Top10R={metrics.get('top_decile_recall', float('nan')):.3f}, "
                f"gate={bool(sel.get('passed_gate',0))}, econ={bool(sel.get('passed_econ',0))}"
            )

        gated = [
            r for r in scored if bool(r[6].get("passed_gate", 0.0) > 0.5 and r[6].get("passed_econ", 0.0) > 0.5)
        ]
        pool = gated if gated else scored
        if pool:
            def _rank_key(r):
                sel_score = float(r[7].get("selection_score", -1e18))
                auc = float(r[6].get("roc_auc", -1e18))
                pr = float(r[6].get("pr_auc", -1e18))
                return (sel_score, auc, pr)

            _best = max(pool, key=_rank_key)
            best_rec = {
                "name": _best[0],
                "kind": _best[1],
                "params": _best[2],
                "oof": _best[3],
                "y_move_soft": _best[4],
                "y_move": _best[5],
                "metrics": _best[6],
                "selection": _best[7],
            }
            best_aux = _best[8]

        if best_rec is None:
            raise RuntimeError("No classifier candidates completed")

        self.label_threshold = float(self.move_thresholds[1] if len(self.move_thresholds) > 1 else self.move_thresholds[0])
        self.y_move = np.asarray(best_rec["y_move"], dtype=np.int8).reshape(-1).copy()
        self.y_move_soft = np.asarray(best_rec["y_move_soft"], dtype=np.float32).reshape(-1).copy()
        self.move_threshold = None
        if move_threshold_arr is not None:
            self.move_threshold = np.asarray(move_threshold_arr, dtype=np.float32).reshape(-1).copy()
        self.oof_probs_raw = np.asarray(best_rec["oof"], dtype=float).reshape(-1)
        if best_aux is not None:
            self.oof_sigma = np.asarray(best_aux.get("oof_sigma"), dtype=np.float32).reshape(-1)
            self.oof_robust_sigma = np.asarray(
                best_aux.get("oof_robust_sigma"), dtype=np.float32
            ).reshape(-1)
        else:
            self.oof_sigma = np.full(len(self.oof_probs_raw), np.nan, dtype=np.float32)
            self.oof_robust_sigma = np.full(len(self.oof_probs_raw), np.nan, dtype=np.float32)
        self._model_type = best_rec["name"]
        self.report_rows = records

        kind = best_rec["kind"]
        params = best_rec["params"]
        y_final_soft = np.asarray(best_rec["y_move_soft"], dtype=np.float32).reshape(-1)
        y_final = np.asarray(best_rec["y_move"], dtype=np.int8)

        unique_classes = np.unique(y_final)
        if len(unique_classes) < 2:
            tprint(
                f"  WARNING: Meta move labels for {self.strategy_name} have only one class: {unique_classes}. Skipping final fit."
            )
            self.model = {"kind": "trivial", "class": int(unique_classes[0]), "binary": True}
            self.oof_probs = np.full(len(y_final), float(unique_classes[0]), dtype=float)
            return self

        _sel_best = best_rec.get("selection", {}) if isinstance(best_rec, dict) else {}
        tprint(
            f"  Winner: {best_rec['name']} "
            f"(Score={float(_sel_best.get('selection_score', float('nan'))):.5f}, "
            f"AUC={float(best_rec['metrics'].get('roc_auc', float('nan'))):.4f}, "
            f"PR={float(best_rec['metrics'].get('pr_auc', float('nan'))):.4f}, "
            f"Top10Lift={float(_sel_best.get('top_decile_lift', float('nan'))):.5f}). "
            f"Fitting final model..."
        )

        tuned_params = (
            dict(params)
            if self.disable_hpo
            else self._optuna_hpo(best_rec["name"], kind, params, X_meta_work, y_final_soft, sw=sw)
        )

        self.selected_features = self._select_tail_features(
            X_meta_work, y_final_soft, max_features=self._feature_selection_max_features
        )
        self.monotone_constraints = discover_monotone_constraints(
            X_meta_work, y_final_soft, self.selected_features
        )
        Xv = X_meta_work[self.selected_features].to_numpy(dtype=np.float32)
        final_model = self._fit_one(
            kind,
            tuned_params,
            Xv,
            y_final_soft,
            Xv,
            y_final_soft,
            sw=sw,
        )
        self.model = {"kind": kind, "models": [final_model], "binary": True}

        if self.use_calibration:
            try:
                calib_mask = np.isfinite(self.oof_probs_raw) & np.isfinite(y_final)
                if int(np.sum(calib_mask)) >= 50 and len(np.unique(y_final[calib_mask])) > 1:
                    self._calibrator = IsotonicRegression(out_of_bounds="clip")
                    self._calibrator.fit(self.oof_probs_raw[calib_mask], y_final[calib_mask])
                    self.oof_probs = np.asarray(
                        self._calibrator.transform(self.oof_probs_raw),
                        dtype=float,
                    ).reshape(-1)
                else:
                    self._calibrator = None
                    self.oof_probs = self.oof_probs_raw.copy()
            except Exception as exc:
                tprint(f"  Calibration skipped for {self.strategy_name}: {exc}")
                self._calibrator = None
                self.oof_probs = self.oof_probs_raw.copy()
        else:
            self.oof_probs = self.oof_probs_raw.copy()

        report_dir = self._reports_dir
        report_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(
            report_dir / f"meta_clf_{self.strategy_name or 'generic'}_race.csv",
            index=False,
        )

        tprint(
            f"MetaClassifierModel.fit: {self.strategy_name} done ({_time.monotonic()-_t0:.1f}s). "
            f"Winner={best_rec['name']}"
        )
        return self

    def predict_proba(self, X_meta):
        if self.selected_features is None or self.model is None:
            raise RuntimeError("MetaClassifierModel must be fitted before predict")
            
        if self.model.get("kind") == "trivial":
            n = len(X_meta)
            cls = self.model["class"]
            return np.full(n, float(cls), dtype=np.float64)

        X = X_meta[self.selected_features].to_numpy(dtype=float)
        probs_list = []
        for m in self.model["models"]:
            pp = self._predict_proba(m, X)
            probs_list.append(pp)
        out = np.mean(probs_list, axis=0)
        out = np.asarray(out, dtype=np.float64).reshape(-1)
        if self._calibrator is not None:
            try:
                out = np.asarray(self._calibrator.transform(out), dtype=np.float64).reshape(-1)
            except Exception:
                pass
        return np.clip(out, 0.0, 1.0)
