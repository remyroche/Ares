from __future__ import annotations

import gc
import inspect
import json
import logging
import multiprocessing as mp
import os
import queue
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.stats import rankdata, spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from .en_uncertainty_ebm import (
    EBMUncertaintyState,
    ENUncertaintyAdjuster,
    compute_uncertainty_features,
    fit_en_uncertainty_adjuster,
    fit_uncertainty_state,
    uncertainty_weighted_prediction,
)
from .ridge_on_lgbm import _compute_weight_distillation
from .utils import tprint

EBM_CV_SPLITS = 2
EBM_RACE_MAX_ROWS = 60000
EBM_RACE_EVAL_FRACTION = 1.0 / 3.0
EBM_FOLD_SUBSAMPLE_ROWS = 5000
EBM_MIN_FEATURES = 40
EBM_SELECTED_FEATURES_MIN = int(os.environ.get("EPM_EBM_SELECTED_FEATURES_MIN", "100"))
EBM_SELECTED_FEATURES_MAX = int(os.environ.get("EPM_EBM_SELECTED_FEATURES_MAX", "350"))
EBM_MAX_ROUNDS = int(os.environ.get("EPM_EBM_MAX_ROUNDS", "10"))
EBM_PRESCREEN_MAX_FEATURES: int | None = None
EBM_PRESCREEN_BASE_FEATURES = 200
EBM_PRESCREEN_FEATURE_FRACTION = 0.25
EBM_TREE_FEATURE_CAP = 1200
EBM_TREE_TARGET_RANK_CAP = 2000
EBM_TREE_CORR_PRUNE_THRESHOLD = 0.96
EBM_TREE_CORR_PRUNE_MIN_FEATURES = 1000
EBM_TREE_LGBM_MAX_FIT_ROWS = 30000
EBM_TREE_LGBM_EARLY_STOPPING_ROUNDS = 25
EBM_FINAL_MODEL_COUNT = int(os.environ.get("EPM_EBM_FINAL_MODEL_COUNT", "3"))
EBM_HONEST_EVAL_MIN_MODELS = int(os.environ.get("EPM_EBM_HONEST_EVAL_MIN_MODELS", "3"))
EBM_HONEST_EVAL_MAX_J_STD = float(
    os.environ.get("EPM_EBM_HONEST_EVAL_MAX_J_STD", "0.015")
)
EBM_HONEST_EVAL_MAX_PRED_DISAGREE = float(
    os.environ.get("EPM_EBM_HONEST_EVAL_MAX_PRED_DISAGREE", "0.020")
)
EBM_SALG_LIFT_COEF = float(os.environ.get("EPM_EBM_SALG_LIFT_COEF", "0.38"))
EBM_J_SALG_WEIGHT = float(os.environ.get("EPM_EBM_J_SALG_WEIGHT", "0.75"))
EBM_J_NDCG_WEIGHT = float(os.environ.get("EPM_EBM_J_NDCG_WEIGHT", "0.25"))
EBM_J_ROBUST_IQR_PENALTY = float(os.environ.get("EPM_EBM_J_ROBUST_IQR_PENALTY", "0.50"))
EBM_TOP20_SALG_TOL = float(os.environ.get("EPM_EBM_TOP20_SALG_TOL", "0.05"))
# Reserved for future explicit stability floor/tolerance tuning.
EBM_TOP20_STABILITY_TOL = float(os.environ.get("EPM_EBM_TOP20_STABILITY_TOL", "0.10"))
if EBM_SELECTED_FEATURES_MIN > EBM_SELECTED_FEATURES_MAX:
    raise ValueError(
        "EPM_EBM_SELECTED_FEATURES_MIN must be <= EPM_EBM_SELECTED_FEATURES_MAX "
        f"({EBM_SELECTED_FEATURES_MIN} > {EBM_SELECTED_FEATURES_MAX})."
    )
if EBM_SELECTED_FEATURES_MIN < EBM_MIN_FEATURES:
    tprint(
        "EBMOnLGBM: EPM_EBM_SELECTED_FEATURES_MIN is below EBM_MIN_FEATURES; "
        "EBM_MIN_FEATURES will dominate the hard floor."
    )
if EBM_FINAL_MODEL_COUNT < 1:
    raise ValueError("EPM_EBM_FINAL_MODEL_COUNT must be >= 1.")
if EBM_HONEST_EVAL_MIN_MODELS > EBM_FINAL_MODEL_COUNT:
    tprint(
        "EBMOnLGBM: clamping EBM_HONEST_EVAL_MIN_MODELS to " "EBM_FINAL_MODEL_COUNT."
    )
    EBM_HONEST_EVAL_MIN_MODELS = EBM_FINAL_MODEL_COUNT
_j_weight_sum = EBM_J_SALG_WEIGHT + EBM_J_NDCG_WEIGHT
if _j_weight_sum <= 0.0:
    raise ValueError("EPM_EBM_J_SALG_WEIGHT + EPM_EBM_J_NDCG_WEIGHT must be > 0.")
if abs(_j_weight_sum - 1.0) > 1e-6:
    tprint(
        "EBMOnLGBM: J weights do not sum to 1.0 "
        f"(sum={_j_weight_sum:.6f}); using them as provided."
    )
EBM_SPEC_N_JOBS = 1
EBM_HPO_TRIALS = 200
EBM_HPO_EARLY_STOP_PATIENCE = 50
EBM_HPO_N_JOBS = 1
EBM_FIT_N_JOBS = 1
EBM_HPO_FOLD_TIMEOUT_SEC = float(
    os.environ.get("EPM_EBM_HPO_FOLD_TIMEOUT_SEC", "300")
)
EBM_HPO_MAX_FOLD_TIMEOUTS = int(os.environ.get("EPM_EBM_HPO_MAX_FOLD_TIMEOUTS", "5"))
EBM_HPO_MAX_ROUNDS = int(os.environ.get("EPM_EBM_HPO_MAX_ROUNDS", "1000"))
EBM_HPO_VALIDATION_SIZE = float(os.environ.get("EPM_EBM_HPO_VALIDATION_SIZE", "0.15"))
EBM_HPO_SUBPROCESS_TIMEOUT = (
    os.environ.get("EPM_EBM_HPO_SUBPROCESS_TIMEOUT", "0").strip().lower()
    in {"1", "true", "yes", "on"}
)
EBM_FINAL_STAGE_OOF_OUTER_BAGS = int(
    os.environ.get("EPM_EBM_FINAL_STAGE_OOF_OUTER_BAGS", "1")
)
EBM_OOF_DISTILLATION_PASSES = int(
    os.environ.get("EPM_EBM_OOF_DISTILLATION_PASSES", "1")
)
EBM_FOLD_PRESENCE_MIN_START = float(
    os.environ.get("EPM_EBM_FOLD_PRESENCE_MIN_START", "0.20")
)
EBM_FOLD_PRESENCE_MIN_STEP = float(
    os.environ.get("EPM_EBM_FOLD_PRESENCE_MIN_STEP", "0.05")
)
EBM_FOLD_PRESENCE_MIN_CAP = float(
    os.environ.get("EPM_EBM_FOLD_PRESENCE_MIN_CAP", "0.80")
)
EBM_SIGN_CONSISTENCY_MIN_START = float(
    os.environ.get("EPM_EBM_SIGN_CONSISTENCY_MIN_START", "0.60")
)
EBM_SIGN_CONSISTENCY_MIN_STEP = float(
    os.environ.get("EPM_EBM_SIGN_CONSISTENCY_MIN_STEP", "0.05")
)
EBM_SIGN_CONSISTENCY_MIN_CAP = float(
    os.environ.get("EPM_EBM_SIGN_CONSISTENCY_MIN_CAP", "0.90")
)
EBM_TOP_SLICE_TIEBREAKER_WEIGHT = float(
    os.environ.get("EPM_EBM_TOP_SLICE_TIEBREAKER_WEIGHT", "0.05")
)
EBM_TOP_ECON_TIEBREAKER_WEIGHT = float(
    os.environ.get("EPM_EBM_TOP_ECON_TIEBREAKER_WEIGHT", "0.05")
)
EBM_TOP_CONTRIB_TIEBREAKER_WEIGHT = float(
    os.environ.get("EPM_EBM_TOP_CONTRIB_TIEBREAKER_WEIGHT", "0.04")
)
EBM_TOP_CONTRIB_FRACTION = 0.20
EBM_STABILITY_TIEBREAKER_WEIGHT = float(
    os.environ.get("EPM_EBM_STABILITY_TIEBREAKER_WEIGHT", "0.03")
)
EBM_TOP_BUCKET_MULTIPLIER_STRENGTH = float(
    os.environ.get("EPM_EBM_TOP_BUCKET_MULTIPLIER_STRENGTH", "0.45")
)
EBM_TOP_BUCKET_MULTIPLIER_MAX = float(
    os.environ.get("EPM_EBM_TOP_BUCKET_MULTIPLIER_MAX", "1.75")
)
EBM_TOP_BUCKET_FALSE_POSITIVE_HARD_MAX = float(
    os.environ.get("EPM_EBM_TOP_BUCKET_FALSE_POSITIVE_HARD_MAX", "0.65")
)
EBM_TOP_BUCKET_FALSE_POSITIVE_MULTIPLIER_MIN = float(
    os.environ.get("EPM_EBM_TOP_BUCKET_FALSE_POSITIVE_MULTIPLIER_MIN", "0.65")
)
EBM_CONTRIB_REDUNDANCY_CORR_START = float(
    os.environ.get("EPM_EBM_CONTRIB_REDUNDANCY_CORR_START", "0.85")
)
EBM_CONTRIB_REDUNDANCY_CORR_FULL = float(
    os.environ.get("EPM_EBM_CONTRIB_REDUNDANCY_CORR_FULL", "0.96")
)
EBM_CONTRIB_REDUNDANCY_MULTIPLIER_MIN = float(
    os.environ.get("EPM_EBM_CONTRIB_REDUNDANCY_MULTIPLIER_MIN", "0.75")
)
EBM_CONTRIB_REDUNDANCY_MAX_ROWS = int(
    os.environ.get("EPM_EBM_CONTRIB_REDUNDANCY_MAX_ROWS", "5000")
)
EBM_CONTRIB_REDUNDANCY_MAX_FEATURES = int(
    os.environ.get("EPM_EBM_CONTRIB_REDUNDANCY_MAX_FEATURES", "2000")
)
EBM_PRUNE_FOLD_N_JOBS = int(os.environ.get("EPM_EBM_PRUNE_FOLD_N_JOBS", "2"))
EBM_HPO_MIN_LEAF_PCT_LO = 0.02
EBM_HPO_MIN_LEAF_PCT_HI = 0.08
EBM_STAGE_LGBM_PRUNE_FRACTION = 0.35
EBM_STAGE_HPO_FRACTION = 0.10
EBM_STAGE_FIT_OOF_FRACTION = 0.55
TREE_BLOCKS: tuple[tuple[int, int], ...] = (
    (0, 10),
    (10, 25),
    (25, 50),
    (50, 100),
    (100, 200),
)
EBM_METRIC_TARGET_FRACTION = float(
    os.environ.get("EPM_EBM_METRIC_TARGET_FRACTION", "0.15")
)
EBM_FINAL_FIT_METRIC_MAX_ROWS = int(
    os.environ.get("EPM_EBM_FINAL_FIT_METRIC_MAX_ROWS", "50000")
)
EBM_FINAL_UNCERTAINTY_MAX_ROWS = int(
    os.environ.get("EPM_EBM_FINAL_UNCERTAINTY_MAX_ROWS", "50000")
)
EBM_FINAL_FIT_MAX_ROWS = int(os.environ.get("EPM_EBM_FINAL_FIT_MAX_ROWS", "200000"))


def _candidate_gate_thresholds() -> tuple[float, float]:
    min_lift = float(
        os.environ.get(
            "EPM_EBM_CANDIDATE_MIN_LIFT15",
            os.environ.get("EPM_EBM_CANDIDATE_MIN_LIFT30", 1.0),
        )
    )
    min_stability = float(
        os.environ.get(
            "EPM_EBM_CANDIDATE_MIN_STABILITY15",
            os.environ.get("EPM_EBM_CANDIDATE_MIN_STABILITY30", 0.50),
        )
    )
    return min_lift, min_stability


def _quiet_interpret_logging() -> None:
    """Keep Interpret's native boosting internals from flooding long pipeline logs."""
    for name in (
        "interpret",
        "interpret.utils",
        "interpret.utils._native",
        "interpret.utils._compressed_dataset",
        "interpret.glassbox",
        "interpret.glassbox._ebm",
        "interpret.glassbox._ebm._boost",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


_quiet_interpret_logging()


@dataclass
class ShapeAudit:
    shape_type: str
    bend_count: int
    wiggle_score: float
    penalty: float
    recommend_monotone: bool
    reason: str


def _smooth_shape(y: np.ndarray, window: int = 3) -> np.ndarray:
    if window <= 1:
        return np.asarray(y, dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(np.asarray(y, dtype=np.float64), kernel, mode="same")


def _shape_sign_changes(dy: np.ndarray, eps: float) -> int:
    signs = np.sign(np.where(np.abs(dy) < eps, 0.0, dy))
    signs = signs[signs != 0]
    if len(signs) <= 1:
        return 0
    return int(np.sum(signs[1:] != signs[:-1]))


def audit_feature_shape(
    x_bins: np.ndarray,
    contrib: np.ndarray,
    fold_contribs: list[np.ndarray] | None = None,
    bin_counts: np.ndarray | None = None,
    spearman_corr: float | None = None,
    top_quartile: bool = False,
    economic_monotone_prior: bool = False,
    amplitude_eps: float = 0.05,
    bend_eps: float = 0.01,
    min_bin_frac: float = 0.02,
    max_penalty: float = 0.70,
) -> ShapeAudit:
    x = np.asarray(x_bins)
    y = np.asarray(contrib, dtype=np.float64)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if len(y) <= 1 or not np.any(np.isfinite(y)):
        return ShapeAudit("pure_wiggle", 0, 1.0, max_penalty, False, "Invalid shape.")

    y_s = _smooth_shape(np.nan_to_num(y, nan=0.0), window=3)
    dy = np.diff(y_s)
    amplitude = float(np.nanmax(y_s) - np.nanmin(y_s))
    eps = max(float(bend_eps), float(amplitude_eps) * amplitude)
    bend_count = _shape_sign_changes(dy, eps)

    raw_bends = _shape_sign_changes(np.diff(y), bend_eps)
    low_amp_osc = min(1.0, max(0.0, raw_bends - bend_count) / 5.0)

    low_support_score = 0.0
    if bin_counts is not None:
        counts = np.asarray(bin_counts, dtype=np.float64)[order]
        denom = max(float(np.sum(counts)), 1.0)
        low_support_score = float(np.mean((counts / denom) < min_bin_frac))

    shape_instability = 0.0
    if fold_contribs is not None and len(fold_contribs) >= 2:
        fold_bends: list[int] = []
        fold_dirs: list[float] = []
        for fc in fold_contribs:
            fc_arr = np.asarray(fc, dtype=np.float64)[order]
            fc_s = _smooth_shape(np.nan_to_num(fc_arr, nan=0.0), window=3)
            f_amp = float(np.nanmax(fc_s) - np.nanmin(fc_s))
            f_eps = max(float(bend_eps), float(amplitude_eps) * f_amp)
            fold_bends.append(_shape_sign_changes(np.diff(fc_s), f_eps))
            fold_dirs.append(float(np.sign(fc_s[-1] - fc_s[0])))
        bend_cv = float(np.std(fold_bends) / (1.0 + np.mean(fold_bends)))
        direction_disagreement = float(1.0 - abs(np.mean(fold_dirs)))
        shape_instability = float(
            np.clip(0.6 * bend_cv + 0.4 * direction_disagreement, 0.0, 1.0)
        )

    bend_norm = min(1.0, bend_count / 6.0)
    wiggle_score = float(
        0.35 * bend_norm
        + 0.35 * shape_instability
        + 0.20 * low_support_score
        + 0.10 * low_amp_osc
    )
    penalty = min(float(max_penalty), wiggle_score)

    if bend_count <= 1 and wiggle_score < 0.25:
        shape_type = "monotonic_noise"
        reason = "Mostly directional with only minor local reversals."
    elif 1 <= bend_count <= 2 and shape_instability < 0.35 and amplitude > eps * 3:
        shape_type = "legitimate_convexity"
        reason = "Stable low-bend nonlinear shape."
    elif bend_count <= 3 and low_support_score < 0.25 and shape_instability < 0.40:
        shape_type = "regime_discontinuity"
        reason = "Moderate bends with acceptable support and fold stability."
    else:
        shape_type = "pure_wiggle"
        reason = "Too many or unstable bends; likely overfit/noise."

    recommend_monotone = False
    if economic_monotone_prior:
        if shape_type == "monotonic_noise":
            recommend_monotone = True
        elif top_quartile and spearman_corr is not None:
            recommend_monotone = (
                abs(float(spearman_corr)) > 0.10 and wiggle_score < 0.35
            )
    if shape_type == "pure_wiggle":
        recommend_monotone = False

    return ShapeAudit(
        shape_type=shape_type,
        bend_count=int(bend_count),
        wiggle_score=wiggle_score,
        penalty=float(penalty),
        recommend_monotone=bool(recommend_monotone),
        reason=reason,
    )


def pure_wiggle_action(
    bend_count: int,
    fold_stability: float,
    oos_j_score: float,
    raw_j_score: float,
    is_tree_leaf: bool = True,
) -> str:
    oos_decay = 1.0 - (float(oos_j_score) / max(float(raw_j_score), 1e-9))
    if bend_count >= 6 or fold_stability < 0.50 or oos_decay > 0.50 or is_tree_leaf:
        return "delete"
    return "smooth"


def _log_selected_features(stage: str, features: list[str]) -> None:
    tprint(
        f"EBMOnLGBM: {stage} selected features "
        f"(n={len(features)}): {json.dumps(features, separators=(',', ':'))}"
    )


def _compute_ece(y_true: np.ndarray, pred: np.ndarray, n_bins: int = 10) -> float:
    from .ridge_on_lgbm import _expected_calibration_error

    return float(_expected_calibration_error(y_true, pred, n_bins))


@dataclass
class SplinePostProcessor:
    mode: str = "classifier"
    min_knots: int = 4
    spline: Optional[Any] = None
    isotonic: Optional[Any] = None
    identity: bool = True
    calibration_method: str = "identity"
    clip_lo: float = 1e-4
    clip_hi: float = 1.0 - 1e-4
    k: float = 1.0
    _w_shape_warned: bool = False

    def fit(
        self,
        raw_pred: np.ndarray,
        y: np.ndarray,
        n_bins: int = 20,
        use_dynamic_smoothing: bool = False,
    ) -> "SplinePostProcessor":
        x = np.asarray(raw_pred, dtype=np.float64)
        yy = np.asarray(y, dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(yy)
        if int(np.sum(mask)) < max(20, self.min_knots * 3):
            self.identity = True
            return self
        x = x[mask]
        yy = yy[mask]
        ranks = rankdata(x, method="average") / max(float(len(x)), 1.0)
        edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
        knots_x: list[float] = []
        knots_y: list[float] = []
        for i in range(int(n_bins)):
            if i < int(n_bins) - 1:
                m = (ranks >= edges[i]) & (ranks < edges[i + 1])
            else:
                m = (ranks >= edges[i]) & (ranks <= edges[i + 1])
            if int(np.sum(m)) < 3:
                continue
            knots_x.append(float(np.median(x[m])))
            knots_y.append(float(np.mean(yy[m])))
        if len(knots_x) < self.min_knots or len(np.unique(knots_x)) < self.min_knots:
            self.identity = True
            return self
        order = np.argsort(np.asarray(knots_x))
        xs = np.asarray(knots_x, dtype=np.float64)[order]
        ys = np.asarray(knots_y, dtype=np.float64)[order]
        uniq_x, uniq_idx = np.unique(xs, return_index=True)
        if len(uniq_x) < self.min_knots:
            self.identity = True
            return self
        uniq_y = ys[uniq_idx]
        if self.mode == "classifier":
            uniq_y = np.clip(uniq_y, self.clip_lo, self.clip_hi)

        if use_dynamic_smoothing:
            y_range = float(np.max(uniq_y) - np.min(uniq_y))
            if y_range < 1e-12:
                s_val = 0.0
            else:
                tv = float(np.sum(np.abs(np.diff(uniq_y))))
                ratio = tv / y_range
                s_val = self.k * (ratio - 1.0) ** 2
        else:
            s_val = 0.5

        # Do not smooth binary features
        if len(np.unique(raw_pred)) <= 2:
            s_val = 0.0

        if self.mode == "classifier":
            self.spline = UnivariateSpline(uniq_x, uniq_y, s=s_val)
            self._check_w_shape(uniq_x)
            self.identity = False
            base_pred = self._predict_spline_only(x)
            if (
                int(np.sum(mask)) >= max(50, self.min_knots * 5)
                and len(np.unique(np.round(base_pred, 8))) >= self.min_knots
                and len(np.unique((yy >= 0.5).astype(np.int8))) > 1
            ):
                iso = IsotonicRegression(
                    y_min=self.clip_lo,
                    y_max=self.clip_hi,
                    out_of_bounds="clip",
                )
                iso.fit(base_pred, yy)
                self.isotonic = iso
                self.calibration_method = "spline_isotonic"
            else:
                self.calibration_method = "spline"
            return self

        self.spline = UnivariateSpline(uniq_x, uniq_y, s=s_val)
        self._check_w_shape(uniq_x)
        self.identity = False
        self.calibration_method = "spline"
        return self

    def _check_w_shape(self, x_eval: np.ndarray) -> None:
        if self.spline is None:
            return
        x_grid = np.linspace(float(x_eval[0]), float(x_eval[-1]), 200)
        y_grid = np.asarray(self.spline(x_grid), dtype=np.float64)
        dy = np.diff(y_grid)
        sign_changes = int(np.sum(np.diff(np.sign(dy)) != 0))
        if sign_changes >= 4:
            tprint(
                f"  WARNING: SplinePostProcessor shape has {sign_changes} inflection "
                f"points (W-shape). Consider increasing min_samples_leaf."
            )
            self._w_shape_warned = True

    def _predict_spline_only(self, raw_pred: np.ndarray) -> np.ndarray:
        x = np.asarray(raw_pred, dtype=np.float64)
        if self.identity or self.spline is None:
            if self.mode == "classifier":
                return np.clip(x, self.clip_lo, self.clip_hi).astype(np.float32)
            return x.astype(np.float32)
        out = np.asarray(self.spline(x), dtype=np.float64)
        if self.mode == "classifier":
            out = np.clip(out, self.clip_lo, self.clip_hi)
        return np.nan_to_num(out, nan=0.5 if self.mode == "classifier" else 0.0).astype(
            np.float32
        )

    def predict(self, raw_pred: np.ndarray) -> np.ndarray:
        out = self._predict_spline_only(raw_pred)
        if self.mode == "classifier" and self.isotonic is not None:
            out = np.asarray(self.isotonic.predict(out), dtype=np.float32)
            out = np.clip(out, self.clip_lo, self.clip_hi)
        return out.astype(np.float32)


class EBMOnLGBMModel:
    def __init__(self, mode: str = "classifier") -> None:
        self.mode = mode
        self.models: list[Any] = []
        self.postprocessors: list[Any] = []
        self.selected_features: list[str] = []
        self.raw_selected_features: list[str] = []
        self.tree_feature_names: list[str] = []
        self.tree_models: list[Any] = []
        self.tree_feature_config: dict[str, Any] = {}
        self.tree_feature_scales: np.ndarray | None = None
        self.selected_indices: np.ndarray = np.array([], dtype=np.int32)
        self.oof_probs: Optional[np.ndarray] = None
        self.oof_probs_raw_ebm: Optional[np.ndarray] = None
        self.oof_probs_en: Optional[np.ndarray] = None
        self.oof_probs_uncertainty_weighted: Optional[np.ndarray] = None
        self.oof_uncertainty_features: dict[str, np.ndarray] = {}
        self.uncertainty_state: EBMUncertaintyState | None = None
        self.en_adjuster: ENUncertaintyAdjuster | None = None
        self.metrics: dict[str, Any] = {}
        self.pruning_history: list[dict[str, Any]] = []

    def _frame(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X)
        X_df = X_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        if self.raw_selected_features:
            raw_names = [str(c) for c in self.raw_selected_features]
            synthetic_raw_contract = all(
                re.fullmatch(r"f\d+", name) is not None for name in raw_names
            )
            missing_raw = sum(1 for name in raw_names if name not in X_df.columns)
            if (
                synthetic_raw_contract
                and missing_raw == len(raw_names)
                and X_df.shape[1] >= len(raw_names)
            ):
                renamed = X_df.iloc[:, : len(raw_names)].copy()
                renamed.columns = raw_names
                extra = X_df.iloc[:, len(raw_names) :]
                if not extra.empty:
                    X_df = pd.concat([renamed, extra], axis=1)
                else:
                    X_df = renamed
                if not getattr(self, "_synthetic_contract_warned", False):
                    tprint(
                        "EBMOnLGBM: mapped positional inference features onto "
                        "synthetic fN training contract."
                    )
                    self._synthetic_contract_warned = True
        if self.tree_models and self.raw_selected_features:
            raw_df = X_df.reindex(columns=self.raw_selected_features, fill_value=0.0)
            if self.tree_feature_config.get("oof_tree_features"):
                tree_df = _compute_oof_bundle_tree_frame(
                    self.tree_feature_config,
                    raw_df,
                    selected_tree_names=self.tree_feature_names,
                )
            else:
                x_raw = raw_df.to_numpy(dtype=np.float32)
                X_tree, out_names, _ = _compute_soft_tree_features_ebm(
                    self.tree_models,
                    x_raw,
                    self.tree_feature_scales,
                    selected_names=set(self.tree_feature_names),
                )
                tree_df = pd.DataFrame(X_tree, columns=out_names, index=raw_df.index)
            missing = [c for c in self.tree_feature_names if c not in tree_df.columns]
            for m in missing:
                tree_df[m] = 0.0
            tree_df = tree_df.reindex(columns=self.tree_feature_names, fill_value=0.0)
            X_df = pd.concat([raw_df, tree_df], axis=1)
        return (
            X_df.reindex(columns=self.selected_features, fill_value=0.0)
            .replace([np.inf, -np.inf], 0.0)
            .fillna(0.0)
        )

    def predict(self, X: Any) -> np.ndarray:
        X_df = self._frame(X)
        if not self.models:
            fill = 0.5 if self.mode == "classifier" else 0.0
            return np.full(len(X_df), fill, dtype=np.float32)
        preds: list[np.ndarray] = []
        raw_preds: list[np.ndarray] = []
        for model, pp in zip(self.models, self.postprocessors):
            raw = _predict_raw_ebm(model, X_df, self.mode)
            raw_preds.append(raw)
            preds.append(pp.predict(raw))
        out = np.mean(np.vstack(preds), axis=0).astype(np.float32)
        if (
            self.mode == "classifier"
            and len(out) > 1
            and raw_preds
            and float(np.nanstd(out)) < 1e-10
        ):
            raw_out = np.mean(np.vstack(raw_preds), axis=0).astype(np.float32)
            if float(np.nanstd(raw_out)) > 1e-6:
                tprint(
                    "EBMOnLGBM: postprocessed predictions collapsed to a constant "
                    "on live/inference batch; using raw EBM probabilities to "
                    "preserve cross-symbol ranking."
                )
                out = raw_out
        if self.mode == "classifier":
            out = np.clip(out, 1e-4, 1.0 - 1e-4)
        return out.astype(np.float32)

    def predict_proba(self, X: Any) -> np.ndarray:
        p = self.predict(X)
        if self.mode != "classifier":
            return np.column_stack([p, p]).astype(np.float32)
        return np.column_stack([1.0 - p, p]).astype(np.float32)

    def predict_uncertainty_features(self, X: Any) -> dict[str, np.ndarray]:
        X_df = self._frame(X)
        feats = compute_uncertainty_features(
            X_df,
            self.models,
            self.mode,
            _predict_raw_ebm,
            state=self.uncertainty_state,
        )
        pred = self.predict(X)
        out = {c: feats[c].to_numpy(dtype=np.float32) for c in feats.columns}
        out["pred_mean"] = pred.astype(np.float32)
        out["confidence_norm"] = (
            np.abs(pred - 0.5) * 2.0
            if self.mode == "classifier"
            else np.abs(pred) / max(float(np.nanpercentile(np.abs(pred), 95)), 1e-6)
        ).astype(np.float32)
        if "ebm_unc_logodds_var" in out:
            out["pred_std"] = out["ebm_unc_logodds_var"].astype(np.float32)
        return out

    def predict_with_uncertainty(self, X: Any) -> dict[str, Any]:
        X_df = self._frame(X)
        raw_pred = self.predict(X)
        features = compute_uncertainty_features(
            X_df,
            self.models,
            self.mode,
            _predict_raw_ebm,
            state=self.uncertainty_state,
        )
        en_pred = (
            self.en_adjuster.predict(raw_pred, features)
            if self.en_adjuster is not None and self.mode == "classifier"
            else raw_pred.copy()
        ).astype(np.float32)
        weighted = uncertainty_weighted_prediction(raw_pred, features, en_pred)
        self._log_inference_diagnostics(X, X_df, raw_pred, en_pred, weighted)
        return {
            "raw_ebm_pred": raw_pred.astype(np.float32),
            "en_pred": en_pred.astype(np.float32),
            "uncertainty_weighted_pred": weighted.astype(np.float32),
            "features": features,
        }

    def _log_inference_diagnostics(
        self,
        X_in: Any,
        X_frame: pd.DataFrame,
        base_pred: np.ndarray,
        meta_pred: np.ndarray,
        final_pred: np.ndarray,
    ) -> None:
        def _stats(name: str, arr: np.ndarray) -> None:
            a = np.asarray(arr, dtype=np.float32)
            if len(a) == 0:
                return
            q99 = float(np.quantile(a, 0.99))
            q95 = float(np.quantile(a, 0.95))
            q90 = float(np.quantile(a, 0.90))
            tprint(
                f"EBM inference [{name}] n={len(a)} mean={float(np.mean(a)):.6f} "
                f"std={float(np.std(a)):.6f} min={float(np.min(a)):.6f} max={float(np.max(a)):.6f} "
                f"top1={int(np.sum(a >= q99))} top5={int(np.sum(a >= q95))} top10={int(np.sum(a >= q90))}."
            )

        _stats("base", base_pred)
        _stats("meta", meta_pred)
        _stats("final", final_pred)
        non_finite = int(np.sum(~np.isfinite(np.asarray(final_pred))))
        if non_finite > 0:
            tprint(f"EBM inference: non-fatal issue non_finite_preds={non_finite}.")
        tprint(
            f"EBM inference: generated_features={X_frame.shape[1]} rows={X_frame.shape[0]} "
            f"missing_values_after_frame={int(np.sum(~np.isfinite(X_frame.to_numpy(dtype=np.float32))))}."
        )
        if isinstance(X_in, pd.DataFrame) and "symbol" in X_in.columns:
            symbols = X_in["symbol"].astype(str)
            fin_mask = np.isfinite(np.asarray(final_pred, dtype=np.float32))
            kept_symbols = sorted(set(symbols[fin_mask]))
            tprint(
                f"EBM inference: symbols passing finite-pred mask={len(kept_symbols)} "
                f"sample={kept_symbols[:10]}."
            )
        ts: pd.Series | None = None
        if isinstance(X_in, pd.DataFrame):
            if isinstance(X_in.index, pd.DatetimeIndex):
                ts = pd.Series(X_in.index, index=X_in.index)
            elif "timestamp" in X_in.columns:
                ts = pd.to_datetime(X_in["timestamp"], errors="coerce")
        if ts is not None and len(ts) == len(final_pred):
            hour = pd.to_datetime(ts, errors="coerce").dt.floor("h")
            top10_thr = float(np.quantile(final_pred, 0.90))
            pos = pd.DataFrame(
                {"hour": hour, "flag": np.asarray(final_pred) >= top10_thr}
            )
            pos = pos.dropna(subset=["hour"])
            if not pos.empty:
                per_hour = pos.groupby("hour")["flag"].sum()
                tprint(
                    "EBM inference: concurrent top10 positions/hour "
                    f"mean={float(per_hour.mean()):.2f} std={float(per_hour.std(ddof=0)):.2f} "
                    f"max={int(per_hour.max())} current={int(per_hour.iloc[-1])}."
                )


def _compute_soft_tree_features_ebm(
    models: list[Any],
    X_raw: np.ndarray,
    scales: np.ndarray | None = None,
    selected_names: set[str] | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    n_instances, n_features = X_raw.shape

    if scales is None:
        import pandas as pd

        df = pd.DataFrame(X_raw)
        # Expanding standard deviation for OOB-like look-ahead prevention
        std_2d = df.expanding(min_periods=20).std().to_numpy(dtype=np.float32)
        global_std = np.std(X_raw, axis=0)
        global_std[global_std < 1e-8] = 1.0
        for j in range(n_features):
            mask = np.isnan(std_2d[:, j]) | (std_2d[:, j] < 1e-8)
            std_2d[mask, j] = global_std[j]
        scales_for_calc = std_2d
        final_scales = std_2d[-1, :].copy()
    else:
        scales_for_calc = np.asarray(scales).reshape(1, -1)
        final_scales = scales

    all_features = []
    out_names = []
    forced_tree_value_count = 0
    selected_names_str = (
        {str(name) for name in selected_names} if selected_names is not None else None
    )

    def emit_feature(name: str, values: np.ndarray, *, force: bool = False) -> bool:
        out_name = str(name)
        if (
            selected_names_str is not None
            and out_name not in selected_names_str
            and not force
        ):
            return False
        all_features.append(np.asarray(values).reshape(-1, 1))
        out_names.append(out_name)
        return True

    model_prefixes = [
        str(getattr(model, "tree_feature_prefix", f"lgbm_model{mi}"))
        for mi, model in enumerate(models)
    ]
    selected_by_tree: dict[tuple[str, int], set[str]] = {}
    if selected_names_str is not None:
        selected_tree_feature_names = list(selected_names_str)
        for model_prefix in model_prefixes:
            for name in selected_tree_feature_names:
                if not name.startswith(f"{model_prefix}_tree"):
                    continue
                parsed = _parse_tree_feature_name(name)
                if parsed is None:
                    continue
                tree_idx, _ = parsed
                selected_by_tree.setdefault((model_prefix, tree_idx), set()).add(name)

    for mi, model in enumerate(models):
        booster = getattr(model, "booster_", model)
        model_prefix = model_prefixes[mi]
        dump = booster.dump_model()
        trees = dump.get("tree_info", [])

        if len(trees) == 0:
            continue

        # Keep all trees instead of sampling
        tree_indices = list(range(len(trees)))

        for ti in tree_indices:
            tree_struct = trees[ti]["tree_structure"]
            splits = {}

            def traverse(node):
                if "leaf_index" in node or "split_feature" not in node:
                    return
                splits[node["split_index"]] = {
                    "feature": node["split_feature"],
                    "threshold": node["threshold"],
                }
                left = node.get("left_child")
                right = node.get("right_child")
                if left is not None:
                    traverse(left)
                if right is not None:
                    traverse(right)

            traverse(tree_struct)

            split_probs = {}
            for s_idx, s_data in splits.items():
                f_idx = s_data["feature"]
                thresh = s_data["threshold"]

                if scales_for_calc.ndim == 2:
                    s_val = scales_for_calc[:, f_idx]
                else:
                    s_val = scales_for_calc[0, f_idx]

                z = (thresh - X_raw[:, f_idx]) / s_val
                z_clipped = np.clip(z, -20.0, 20.0)
                split_probs[s_idx] = 1.0 / (1.0 + np.exp(-z_clipped))

            paths = {}

            def get_leaf_probs(node, current_prob):
                if "leaf_index" in node:
                    paths[node["leaf_index"]] = {
                        "prob": current_prob,
                        "value": node["leaf_value"],
                    }
                    return
                if "split_index" not in node or node["split_index"] not in split_probs:
                    leaf_idx = int(node.get("leaf_index", len(paths)))
                    paths[leaf_idx] = {
                        "prob": current_prob,
                        "value": float(
                            node.get("leaf_value", node.get("internal_value", 0.0))
                        ),
                    }
                    return
                s_idx = node["split_index"]
                p_left = split_probs[s_idx]
                p_right = 1.0 - p_left

                # Soft-Min aggregation
                left = node.get("left_child")
                right = node.get("right_child")
                if left is not None:
                    get_leaf_probs(left, np.minimum(current_prob, p_left))
                if right is not None:
                    get_leaf_probs(right, np.minimum(current_prob, p_right))

            get_leaf_probs(tree_struct, np.ones(n_instances, dtype=np.float32))

            n_leaves = len(paths)
            if n_leaves == 0:
                continue
            soft_leaves = np.zeros((n_instances, n_leaves), dtype=np.float32)
            leaf_values = np.zeros(n_leaves, dtype=np.float32)

            for leaf_idx, data in paths.items():
                soft_leaves[:, leaf_idx] = data["prob"]
                leaf_values[leaf_idx] = data["value"]

            value_name = f"{model_prefix}_tree{ti}_value"
            tree_pred = np.sum(soft_leaves * leaf_values, axis=1)
            n_before_tree = len(out_names)
            emit_feature(value_name, tree_pred)

            for leaf_idx in range(n_leaves):
                leaf_name = f"{model_prefix}_tree{ti}_leaf{leaf_idx}_soft"
                emit_feature(leaf_name, soft_leaves[:, leaf_idx])

            if (
                selected_names_str is not None
                and len(out_names) == n_before_tree
                and (model_prefix, ti) in selected_by_tree
            ):
                # Preserve selected-tree coverage without admitting every
                # unselected LGBM tree into the final EBM contract.
                if emit_feature(value_name, tree_pred, force=True):
                    forced_tree_value_count += 1

    if selected_names_str is not None:
        if forced_tree_value_count:
            tprint(
                "EBMOnLGBM: added forced per-tree value features for selected "
                f"soft-tree generation ({forced_tree_value_count} trees without "
                "selected leaf/value features)."
            )
        requested_tree_names = set()
        for name in selected_names_str:
            if _parse_tree_feature_name(name) is None:
                continue
            if any(name.startswith(f"{prefix}_tree") for prefix in model_prefixes):
                requested_tree_names.add(name)
        emitted = set(out_names)
        missing = requested_tree_names - emitted
        if missing:
            tprint(
                "EBMOnLGBM: missing selected tree features during soft-tree generation "
                f"({len(missing)} missing, {len(emitted)} emitted)."
            )

    if not all_features:
        return np.zeros((n_instances, 0), dtype=np.float32), [], final_scales

    X_out = np.hstack(all_features).astype(np.float32)
    return X_out, out_names, final_scales


def _fit_lgbm_tree_feature_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray | None,
    y_eval: np.ndarray | None,
    params: dict[str, Any],
    random_state: int,
    sample_weight: np.ndarray | None = None,
) -> Any:
    import lightgbm as lgb

    x_train = np.nan_to_num(
        np.asarray(x_train, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    y_train = np.nan_to_num(
        np.asarray(y_train, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    if x_eval is not None:
        x_eval = np.nan_to_num(
            np.asarray(x_eval, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
    if y_eval is not None:
        y_eval = np.nan_to_num(
            np.asarray(y_eval, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )

    sw = None
    if sample_weight is not None:
        sw = np.nan_to_num(
            np.asarray(sample_weight, dtype=np.float32), nan=1.0, posinf=1.0, neginf=1.0
        )
    train_set = lgb.Dataset(x_train, label=y_train, weight=sw, free_raw_data=False)
    train_kwargs: dict[str, Any] = {}
    if x_eval is not None and y_eval is not None and len(y_eval) > 1:
        train_kwargs["valid_sets"] = [
            lgb.Dataset(x_eval, label=y_eval, reference=train_set)
        ]
        train_kwargs["callbacks"] = [
            lgb.early_stopping(
                EBM_TREE_LGBM_EARLY_STOPPING_ROUNDS,
                verbose=False,
            )
        ]
    booster = lgb.train(
        {
            "objective": "huber",
            "metric": "l2",
            "max_depth": int(params.get("max_depth", 3)),
            "min_child_samples": int(params.get("min_child_samples", 100)),
            "learning_rate": 0.04,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "seed": int(random_state),
            "num_threads": 1,
            "verbosity": -1,
        },
        train_set,
        num_boost_round=500,
        **train_kwargs,
    )
    depth = int(params.get("max_depth", 3))
    min_samples = int(params.get("min_child_samples", 100))
    min_pct = params.get("min_child_pct")
    if min_pct is None:
        min_pct_label = f"minobs{min_samples}"
    else:
        min_pct_label = f"minpct{int(round(float(min_pct) * 10000)):04d}"
    return SimpleNamespace(
        booster_=booster,
        tree_feature_prefix=f"lgbm_depth{depth}_{min_pct_label}",
    )


def _leaf_path_features(
    tree_struct: dict[str, Any], feature_names: list[str] | None = None
) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    names = feature_names or []

    def walk(node: dict[str, Any], path: list[dict[str, Any]]) -> None:
        if "leaf_index" in node:
            out[int(node["leaf_index"])] = list(path)
            return
        if "split_feature" not in node:
            return
        fidx = int(node.get("split_feature", -1))
        fname = names[fidx] if 0 <= fidx < len(names) else f"f{fidx}"
        thr = float(node.get("threshold", 0.0))
        left = node.get("left_child")
        right = node.get("right_child")
        if left is not None:
            walk(
                left,
                path
                + [{"split_feature_name": fname, "threshold": thr, "direction": "<="}],
            )
        if right is not None:
            walk(
                right,
                path
                + [{"split_feature_name": fname, "threshold": thr, "direction": ">"}],
            )

    walk(tree_struct, [])
    return out


def _family_signature_from_path(path_features: list[dict[str, Any]]) -> str:
    toks: set[str] = set()
    for step in path_features:
        nm = str(step.get("split_feature_name", "")).lower()
        for tok in ("price", "volume", "cross_asset", "orderbook_wall", "funding"):
            if tok in nm:
                toks.add(tok)
    return "+".join(sorted(toks)) if toks else "unknown"


def _subsample_tree_fit_rows(
    x: np.ndarray,
    y: np.ndarray,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(y) <= EBM_TREE_LGBM_MAX_FIT_ROWS:
        return x, y
    y_arr = np.asarray(y)
    classifier_like = _looks_classifier_target(y_arr)
    sub = _stratified_subsample_indices(
        y_arr,
        max_n=EBM_TREE_LGBM_MAX_FIT_ROWS,
        random_state=random_state,
        classifier=bool(classifier_like),
    )
    tprint(
        "EBMOnLGBM tree features: capped LGBM fit rows "
        f"to {len(sub)} from {len(y_arr)}."
    )
    return x[sub], y_arr[sub]


def _inner_tree_fit_split(
    x: np.ndarray,
    y: np.ndarray,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    if len(y) < 50:
        return x, y, None, None
    y_arr = np.asarray(y)
    stratify = None
    if _looks_classifier_target(y_arr):
        labels = np.asarray(y_arr >= 0.5, dtype=np.int8)
        if np.min(np.bincount(labels, minlength=2)) >= 2:
            stratify = labels
    tr, va = train_test_split(
        np.arange(len(y_arr)),
        test_size=0.15,
        random_state=random_state,
        stratify=stratify,
    )
    return x[tr], y_arr[tr], x[va], y_arr[va]


def _load_ebm_classes() -> tuple[Any, Any] | tuple[None, None]:
    try:
        _quiet_interpret_logging()
        from interpret.glassbox import (  # type: ignore
            ExplainableBoostingClassifier,
            ExplainableBoostingRegressor,
        )

        return ExplainableBoostingClassifier, ExplainableBoostingRegressor
    except Exception as exc:  # pragma: no cover - depends on optional package
        tprint(f"EBMOnLGBM: interpret unavailable, skipping candidate ({exc})")
        return None, None


def _filter_model_kwargs(cls: Any, params: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(cls)
    except Exception:
        return dict(params)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(params)
    return {k: v for k, v in params.items() if k in sig.parameters}


def _make_ebm(cls: Any, params: dict[str, Any]) -> Any:
    _quiet_interpret_logging()
    return cls(**_filter_model_kwargs(cls, params))


def _hpo_fit_predict_worker(
    out_q: Any,
    mode: str,
    params: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    x_valid: pd.DataFrame,
) -> None:
    """Fit one HPO fold in a child process so native EBM stalls are killable."""
    try:
        import traceback

        ebm_clf, ebm_reg = _load_ebm_classes()
        cls = ebm_clf if mode == "classifier" else ebm_reg
        if cls is None:
            raise RuntimeError("interpret EBM class unavailable in HPO worker")
        model = _fit_one_ebm(
            cls,
            x_train,
            np.asarray(y_train),
            np.asarray(sample_weight, dtype=np.float32),
            dict(params),
        )
        pred = (
            model.predict_proba(_coerce_ebm_feature_types(x_valid))[:, 1]
            if mode == "classifier"
            else model.predict(_coerce_ebm_feature_types(x_valid))
        )
        out_q.put(("ok", np.asarray(pred, dtype=np.float32), ""))
    except BaseException as exc:  # pragma: no cover - child-process safety net
        out_q.put(("err", None, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))


def _fit_predict_hpo_fold_with_timeout(
    *,
    mode: str,
    params: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    x_valid: pd.DataFrame,
    timeout_sec: float,
) -> np.ndarray:
    """Return fold predictions, terminating the child if native EBM fit stalls."""
    timeout_sec = float(timeout_sec)
    if timeout_sec <= 0.0:
        ebm_clf, ebm_reg = _load_ebm_classes()
        cls = ebm_clf if mode == "classifier" else ebm_reg
        if cls is None:
            raise RuntimeError("interpret EBM class unavailable")
        model = _fit_one_ebm(cls, x_train, y_train, sample_weight, params)
        pred = (
            model.predict_proba(_coerce_ebm_feature_types(x_valid))[:, 1]
            if mode == "classifier"
            else model.predict(_coerce_ebm_feature_types(x_valid))
        )
        return np.asarray(pred, dtype=np.float32)

    start_method = os.environ.get("EPM_EBM_HPO_TIMEOUT_START_METHOD", "spawn")
    try:
        ctx = mp.get_context(start_method)
    except ValueError:
        ctx = mp.get_context("spawn")
    out_q = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_hpo_fit_predict_worker,
        args=(
            out_q,
            mode,
            dict(params),
            x_train,
            np.asarray(y_train),
            np.asarray(sample_weight, dtype=np.float32),
            x_valid,
        ),
    )
    proc.daemon = True
    proc.start()
    proc.join(timeout_sec)
    if proc.is_alive():
        proc.terminate()
        proc.join(5.0)
        if proc.is_alive() and hasattr(proc, "kill"):
            proc.kill()
            proc.join(2.0)
        try:
            out_q.close()
        except Exception:
            pass
        raise TimeoutError(f"HPO EBM fold exceeded {timeout_sec:.1f}s")

    try:
        status, payload, err = out_q.get_nowait()
    except queue.Empty as exc:
        raise RuntimeError(
            f"HPO EBM worker exited without predictions (exitcode={proc.exitcode})"
        ) from exc
    finally:
        try:
            out_q.close()
        except Exception:
            pass
    if status != "ok":
        raise RuntimeError(str(err))
    return np.asarray(payload, dtype=np.float32)


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    m = np.isfinite(aa) & np.isfinite(bb)
    if int(np.sum(m)) < 8 or np.nanstd(aa[m]) < 1e-12 or np.nanstd(bb[m]) < 1e-12:
        return 0.0
    val = spearmanr(aa[m], bb[m]).correlation
    return float(val) if val is not None and np.isfinite(val) else 0.0


def _target_top_fraction() -> float:
    return float(np.clip(EBM_METRIC_TARGET_FRACTION, 0.001, 0.5))


def _top_idx(order: np.ndarray, frac: float, n: int) -> np.ndarray:
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    k = max(1, int(np.ceil(float(frac) * n)))
    return np.asarray(order[-k:], dtype=np.int64)


def _ndcg_at_k(y_true: np.ndarray, pred: np.ndarray, k: int = 10) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(p)
    if int(np.sum(mask)) == 0:
        return 0.0
    y = y[mask]
    p = p[mask]
    n = len(y)
    if n <= 1:
        return 0.0
    k = max(1, int(min(k, n)))
    order = np.argsort(p)
    top = order[-k:][::-1]
    y_top = y[top]
    y_sorted = np.sort(y)[::-1]
    idcg = np.sum((2.0 ** y_sorted[:k] - 1.0) / np.log2(np.arange(2, k + 2)))
    if idcg <= 0.0:
        return 0.0
    dcg = np.sum((2.0**y_top - 1.0) / np.log2(np.arange(2, k + 2)))
    return float(dcg / idcg)


def _metric_salg(lift: float, stability: float) -> float:
    """Model Stability-Adjusted Lift Gain: Stability + coef * (Lift - 1)."""
    if not (np.isfinite(lift) and np.isfinite(stability)):
        return float("nan")
    return float(stability + EBM_SALG_LIFT_COEF * (lift - 1.0))


def _metric_j_from_lift_stability_ndcg(
    lift: float,
    stability: float,
    ndcg: float,
) -> float:
    """Top-20 model comparison J from SALG and NDCG."""
    salg = _metric_salg(lift, stability)
    if not (np.isfinite(salg) and np.isfinite(ndcg)):
        return float("nan")
    return float(EBM_J_SALG_WEIGHT * salg + EBM_J_NDCG_WEIGHT * ndcg)


def _metric_pack(
    y_true: np.ndarray,
    pred: np.ndarray,
    classifier: bool,
    groups: Any = None,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    m = np.isfinite(y) & np.isfinite(p)
    if isinstance(groups, dict):
        grp = {
            str(k): np.asarray(v, dtype=object)[m]
            for k, v in groups.items()
            if len(np.asarray(v)) == len(m)
        }
    else:
        grp = np.asarray(groups, dtype=object)[m] if groups is not None else None
    y = y[m]
    p = p[m]
    if len(y) < 8:
        return {
            "lift15": 1.0,
            "lift30": 1.0,
            "lift20": 1.0,
            "lift10": 1.0,
            "lift5": 1.0,
            "precision15": 0.0,
            "hit_rate10": 0.0,
            "hit_rate5": 0.0,
            "hit_rate20": 0.0,
            "hit_rate30": 0.0,
            "hit_rate15": 0.0,
            "precision10": 0.0,
            "precision5": 0.0,
            "precision20": 0.0,
            "precision30": 0.0,
            "precision01": 0.0,
            "precision005": 0.0,
            "ndcg_at_10": 0.0,
            "ndcg_at_20": 0.0,
            "ndcg20": 0.0,
            "ndcg@10": 0.0,
            "ndcg@20": 0.0,
            "salg15": 0.0,
            "salg20": 0.0,
            "salg30": 0.0,
            "J20": 0.0,
            "j20": 0.0,
            "auc_correct_15": 0.5,
            "auc_correct_30": 0.5,
            "stability30_proxy": 0.0,
            "stability30": 0.0,
            "stability15_proxy": 0.0,
            "stability15": 0.0,
            "stability20": 0.0,
            "stability15_n_groups": 0.0,
            "stability15_group_mean": 0.0,
            "stability15_group_std": 0.0,
            "stability30_n_groups": 0.0,
            "stability30_group_mean": 0.0,
            "stability30_group_std": 0.0,
            "auc": 0.5,
            "pr_auc": 0.0,
            "brier": 1.0,
            "ece": _compute_ece(y, p),
            "ece_top20": float("nan"),
            "ece_at_20": float("nan"),
            "oof_std": 0.0,
        }
    order = np.argsort(p)

    def _top_idx_frac(frac: float) -> np.ndarray:
        return _top_idx(order, frac, len(y))

    frac_obj = _target_top_fraction()
    top15 = _top_idx_frac(frac_obj)
    top30 = _top_idx_frac(0.30)
    top20 = _top_idx_frac(0.20)
    top10 = _top_idx_frac(0.10)
    top5 = _top_idx_frac(0.05)
    top1 = _top_idx_frac(0.01)
    top05 = _top_idx_frac(0.005)
    if classifier:
        yb = (y >= 0.5).astype(np.int8)
        base = float(np.mean(yb))
        top_rate15 = float(np.mean(yb[top15])) if len(top15) else 0.0
        top_rate30 = float(np.mean(yb[top30])) if len(top30) else 0.0
        hit_rate20 = float(np.mean(yb[top20])) if len(top20) else 0.0
        hit_rate10 = float(np.mean(yb[top10])) if len(top10) else 0.0
        hit_rate5 = float(np.mean(yb[top5])) if len(top5) else 0.0
        hit_rate01 = float(np.mean(yb[top1])) if len(top1) else 0.0
        hit_rate005 = float(np.mean(yb[top05])) if len(top05) else 0.0
        hit_rate15 = top_rate15
        hit_rate30 = top_rate30
        lift20 = hit_rate20 / max(base, 1e-6)
        lift10 = hit_rate10 / max(base, 1e-6)
        lift5 = hit_rate5 / max(base, 1e-6)
        lift15 = hit_rate15 / max(base, 1e-6)
        lift30 = hit_rate30 / max(base, 1e-6)
        auc15 = 0.5
        if len(np.unique(yb[top15])) > 1:
            try:
                auc15 = float(roc_auc_score(yb[top15], p[top15]))
            except Exception:
                auc15 = 0.5
        auc = float(roc_auc_score(yb, p)) if len(np.unique(yb)) > 1 else 0.5
        pr = float(average_precision_score(yb, p)) if len(np.unique(yb)) > 1 else base
        p_clip = np.clip(p, 1e-6, 1.0 - 1e-6)
        brier = float(brier_score_loss(yb, p_clip))
        ece = _compute_ece(yb, p_clip)
        ece_top20 = (
            _compute_ece(yb[top20], p_clip[top20]) if len(top20) else float("nan")
        )
    else:
        denom = float(np.mean(np.abs(y)) + 1e-6)
        hit_rate5 = float(np.mean(y[top5])) if len(top5) else 0.0
        hit_rate10 = float(np.mean(y[top10])) if len(top10) else 0.0
        hit_rate20 = float(np.mean(y[top20])) if len(top20) else 0.0
        hit_rate15 = float(np.mean(y[top15])) if len(top15) else 0.0
        hit_rate30 = float(np.mean(y[top30])) if len(top30) else 0.0
        lift5 = float(hit_rate5 / denom)
        lift10 = float(hit_rate10 / denom)
        lift20 = float(hit_rate20 / denom)
        lift15 = float(hit_rate15 / denom)
        lift30 = float(hit_rate30 / denom)
        auc15 = max(0.0, _safe_spearman(p[top15], y[top15]))
        hit_rate01 = 0.0
        hit_rate005 = 0.0
        auc = max(0.0, _safe_spearman(p, y))
        pr = auc
        brier = float(np.mean((p - y) ** 2))
        ece = float("nan")
        ece_top20 = float("nan")
    stab_proxy = _top_stability(y, p, frac_obj)
    default_groups = grp.get("week") if isinstance(grp, dict) else grp
    stab_metrics = _grouped_top_stability(
        y, p, frac_obj, classifier=classifier, groups=default_groups
    )
    stab_proxy30 = _top_stability(y, p, 0.30)
    stab_metrics30 = _grouped_top_stability(
        y, p, 0.30, classifier=classifier, groups=default_groups
    )
    stab_proxy20 = _top_stability(y, p, 0.20)
    stab_metrics20 = _grouped_top_stability(
        y, p, 0.20, classifier=classifier, groups=default_groups
    )
    stab_week30 = stab_metrics30
    stab_month30 = None
    stab_week20 = stab_metrics20
    stab_month20 = None
    stab_week15 = stab_metrics
    stab_month15 = None
    if isinstance(grp, dict):
        week_groups = grp.get("week")
        month_groups = grp.get("month")
        stab_week30 = _grouped_top_stability(
            y, p, 0.30, classifier=classifier, groups=week_groups
        )
        stab_month30 = _grouped_top_stability(
            y, p, 0.30, classifier=classifier, groups=month_groups
        )
        stab_week20 = _grouped_top_stability(
            y, p, 0.20, classifier=classifier, groups=week_groups
        )
        stab_month20 = _grouped_top_stability(
            y, p, 0.20, classifier=classifier, groups=month_groups
        )
        stab_week15 = _grouped_top_stability(
            y, p, frac_obj, classifier=classifier, groups=week_groups
        )
        stab_month15 = _grouped_top_stability(
            y, p, frac_obj, classifier=classifier, groups=month_groups
        )

    if stab_month30 is not None and stab_week30["n_groups"] >= 3:
        weekly_std30 = float(stab_week30["group_std"])
        monthly_std30 = (
            float(stab_month30["group_std"])
            if stab_month30["n_groups"] >= 3
            else weekly_std30
        )
        stability30_std_blend = 0.7 * weekly_std30 + 0.3 * monthly_std30
        stability30 = float(1.0 / (1.0 + max(stability30_std_blend, 0.0)))
    else:
        weekly_std30 = float(stab_metrics30["group_std"])
        monthly_std30 = float("nan")
        stability30_std_blend = weekly_std30
        stability30 = (
            stab_metrics30["stability"]
            if stab_metrics30["n_groups"] >= 3
            else stab_proxy30
        )

    if stab_month20 is not None and stab_week20["n_groups"] >= 3:
        weekly_std20 = float(stab_week20["group_std"])
        monthly_std20 = (
            float(stab_month20["group_std"])
            if stab_month20["n_groups"] >= 3
            else weekly_std20
        )
        stability20_std_blend = 0.7 * weekly_std20 + 0.3 * monthly_std20
        stability20 = float(1.0 / (1.0 + max(stability20_std_blend, 0.0)))
    else:
        weekly_std20 = float(stab_metrics20["group_std"])
        monthly_std20 = float("nan")
        stability20_std_blend = weekly_std20
        stability20 = (
            stab_metrics20["stability"]
            if stab_metrics20["n_groups"] >= 3
            else stab_proxy20
        )

    if stab_month15 is not None and stab_week15["n_groups"] >= 3:
        weekly_std15 = float(stab_week15["group_std"])
        monthly_std15 = (
            float(stab_month15["group_std"])
            if stab_month15["n_groups"] >= 3
            else weekly_std15
        )
        stability15_std_blend = 0.7 * weekly_std15 + 0.3 * monthly_std15
        stability15 = float(1.0 / (1.0 + max(stability15_std_blend, 0.0)))
    else:
        weekly_std15 = float(stab_metrics["group_std"])
        monthly_std15 = float("nan")
        stability15_std_blend = weekly_std15
        stability15 = (
            stab_metrics["stability"] if stab_metrics["n_groups"] >= 3 else stab_proxy
        )
    ic_metrics = _rank_ic_metrics(y, p, frac_obj)
    ic_metrics30 = _rank_ic_metrics(y, p, 0.30)
    ic15 = float(ic_metrics["ic_top"])
    ndcg10 = float(_ndcg_at_k(y, p, k=10))
    ndcg20 = float(_ndcg_at_k(y, p, k=20))
    # Tradable-calibrated ranking diagnostics over execution-like top slices.
    ndcg_top20pct = (
        float(_ndcg_at_k(y[top20], p[top20], k=min(20, len(top20))))
        if len(top20) > 0
        else 0.0
    )
    ndcg_top10pct = (
        float(_ndcg_at_k(y[top10], p[top10], k=min(10, len(top10))))
        if len(top10) > 0
        else 0.0
    )
    salg15 = _metric_salg(float(lift15), float(stability15))
    salg20 = _metric_salg(float(lift20), float(stability20))
    salg30 = _metric_salg(float(lift30), float(stability30))
    j20 = _metric_j_from_lift_stability_ndcg(
        float(lift20),
        float(stability20),
        float(ndcg20),
    )
    return {
        "lift15": float(lift15),
        "lift30": float(lift30),
        "lift20": float(lift20),
        "lift10": float(lift10),
        "lift5": float(lift5),
        "hit_rate5": float(hit_rate5),
        "hit_rate10": float(hit_rate10),
        "hit_rate20": float(hit_rate20),
        "hit_rate15": float(hit_rate15),
        "hit_rate30": float(hit_rate30),
        "hit_rate01": float(hit_rate01),
        "hit_rate005": float(hit_rate005),
        "precision5": float(hit_rate5),
        "precision10": float(hit_rate10),
        "precision20": float(hit_rate20),
        "precision30": float(hit_rate30),
        "precision15": float(hit_rate15),
        "precision01": float(hit_rate01),
        "precision005": float(hit_rate005),
        "ndcg@10": ndcg10,
        "ndcg_at_10": ndcg10,
        "ndcg@20": ndcg20,
        "ndcg_at_20": ndcg20,
        "ndcg20": ndcg20,
        "salg15": float(salg15),
        "salg20": float(salg20),
        "salg30": float(salg30),
        "J20": float(j20),
        "j20": float(j20),
        "ndcg_tradable_top20pct": ndcg_top20pct,
        "ndcg_tradable_top10pct": ndcg_top10pct,
        "auc_correct_15": float(auc15),
        "auc_correct_30": (
            float(
                max(0.0, _safe_spearman(p[top30], yb[top30]))
                if classifier
                else max(0.0, _safe_spearman(p[top30], y[top30]))
            )
            if len(top30) > 1
            else 0.5
        ),
        "stability15_proxy": float(stab_proxy),
        "stability15": float(stability15),
        "stability20_proxy": float(stab_proxy20),
        "stability20": float(stability20),
        "stability20_n_groups": float(stab_metrics20["n_groups"]),
        "stability20_group_mean": float(stab_metrics20["group_mean"]),
        "stability20_group_std": float(stab_metrics20["group_std"]),
        "stability20_weekly_std": float(weekly_std20),
        "stability20_monthly_std": float(monthly_std20),
        "stability20_std_blend": float(stability20_std_blend),
        "stability15_n_groups": float(stab_metrics["n_groups"]),
        "stability15_group_mean": float(stab_metrics["group_mean"]),
        "stability15_group_std": float(stab_metrics["group_std"]),
        "stability15_weekly_std": float(weekly_std15),
        "stability15_monthly_std": float(monthly_std15),
        "stability15_std_blend": float(stability15_std_blend),
        "stability30_proxy": float(stab_proxy30),
        "stability30": float(stability30),
        "stability30_n_groups": float(stab_metrics30["n_groups"]),
        "stability30_group_mean": float(stab_metrics30["group_mean"]),
        "stability30_group_std": float(stab_metrics30["group_std"]),
        "stability30_weekly_std": float(weekly_std30),
        "stability30_monthly_std": float(monthly_std30),
        "stability30_std_blend": float(stability30_std_blend),
        "ic_total": float(ic_metrics["ic_total"]),
        "ic_top15": float(ic15),
        "ic_top30": float(ic_metrics30["ic_top30"]),
        "auc": float(auc),
        "pr_auc": float(pr),
        "brier": float(brier),
        "ece": float(ece),
        "ece_top20": float(ece_top20),
        "ece_at_20": float(ece_top20),
        "oof_std": float(np.std(p)),
    }


def _fold_j(m: dict[str, float]) -> float:
    lift20 = float(m.get("lift20", m.get("lift15", m.get("lift30", np.nan))))
    stability20 = float(
        m.get("stability20", m.get("stability15", m.get("stability30", np.nan)))
    )
    ndcg20 = float(m.get("ndcg20", m.get("ndcg_at_20", m.get("ndcg@20", np.nan))))

    j = _metric_j_from_lift_stability_ndcg(lift20, stability20, ndcg20)
    if np.isfinite(j):
        return float(j)

    return float("nan")


def _mean_metric(
    fold_metrics: list[dict[str, float]], *keys: str, default: float = 0.0
) -> float:
    vals: list[float] = []
    for m in fold_metrics:
        for key in keys:
            if key not in m:
                continue
            try:
                v = float(m[key])
            except Exception:
                continue
            if np.isfinite(v):
                vals.append(v)
                break
    return float(np.mean(vals)) if vals else float(default)


def _aggregate_j(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    if not fold_metrics:
        return {
            "J_final": -999.0,
            "J_mean": -999.0,
            "J_std": 0.0,
            "J_robust": -999.0,
            "J_median": -999.0,
            "J_iqr": 0.0,
            "J_se": 0.0,
        }

    j = np.asarray([_fold_j(m) for m in fold_metrics], dtype=np.float64)
    j = j[np.isfinite(j)]
    if len(j) == 0:
        return {
            "J_final": -999.0,
            "J_mean": -999.0,
            "J_std": 0.0,
            "J_robust": -999.0,
            "J_median": -999.0,
            "J_iqr": 0.0,
            "J_se": 0.0,
        }

    q25, q50, q75 = np.percentile(j, [25, 50, 75])
    j_iqr = float(q75 - q25)
    j_median = float(q50)
    j_mean = float(np.mean(j))
    j_std = float(np.std(j, ddof=1)) if len(j) > 1 else 0.0
    j_se = float(j_std / np.sqrt(max(len(j), 1)))
    j_robust = float(j_median - EBM_J_ROBUST_IQR_PENALTY * j_iqr)

    lift30 = _mean_metric(fold_metrics, "lift30")
    lift20 = _mean_metric(fold_metrics, "lift20")
    lift10 = _mean_metric(fold_metrics, "lift10")
    lift5 = _mean_metric(fold_metrics, "lift5")
    lift15 = _mean_metric(fold_metrics, "lift15", "lift30", "lift20")
    hit_rate5 = _mean_metric(fold_metrics, "hit_rate5")
    hit_rate10 = _mean_metric(fold_metrics, "hit_rate10")
    hit_rate20 = _mean_metric(fold_metrics, "hit_rate20")
    hit_rate15 = _mean_metric(fold_metrics, "hit_rate15", "hit_rate30")
    hit_rate30 = _mean_metric(fold_metrics, "hit_rate30", "hit_rate15", "hit_rate20")
    precision01 = _mean_metric(fold_metrics, "precision01", "hit_rate01")
    precision005 = _mean_metric(fold_metrics, "precision005", "hit_rate005")
    ndcg_at_10 = _mean_metric(fold_metrics, "ndcg_at_10", "ndcg@10")
    ndcg_at_20 = _mean_metric(fold_metrics, "ndcg20", "ndcg_at_20", "ndcg@20")
    ece_at_20 = _mean_metric(fold_metrics, "ece_top20", "ece_at_20", default=np.nan)
    ndcg_tradable_top20pct = _mean_metric(fold_metrics, "ndcg_tradable_top20pct")
    ndcg_tradable_top10pct = _mean_metric(fold_metrics, "ndcg_tradable_top10pct")
    _ndcg20_fold = np.asarray(
        [
            float(m.get("ndcg20", m.get("ndcg_at_20", m.get("ndcg@20", np.nan))))
            for m in fold_metrics
        ],
        dtype=np.float64,
    )
    _ndcg20_fold = _ndcg20_fold[np.isfinite(_ndcg20_fold)]
    ndcg20_std = float(np.std(_ndcg20_fold, ddof=1)) if len(_ndcg20_fold) > 1 else 0.0
    ndcg20_p25 = (
        float(np.percentile(_ndcg20_fold, 25.0)) if len(_ndcg20_fold) > 0 else 0.0
    )
    auc30 = _mean_metric(fold_metrics, "auc_correct_30", "auc_correct_15")

    stability15 = _mean_metric(fold_metrics, "stability15", default=np.nan)
    stability20 = _mean_metric(fold_metrics, "stability20", default=np.nan)
    stability30 = _mean_metric(fold_metrics, "stability30", default=np.nan)

    if not np.isfinite(stability20):
        if np.isfinite(stability15) and np.isfinite(stability30):
            stability20 = float(
                np.clip(
                    (2.0 / 3.0) * stability15 + (1.0 / 3.0) * stability30,
                    0.0,
                    1.0,
                )
            )
        elif np.isfinite(stability15):
            stability20 = float(stability15)
        elif np.isfinite(stability30):
            stability20 = float(stability30)
        else:
            stability20 = 0.0

    if not np.isfinite(stability15):
        stability15 = stability20
    if not np.isfinite(stability30):
        stability30 = stability20

    stability15 = float(np.clip(stability15, 0.0, 1.0))
    stability20 = float(np.clip(stability20, 0.0, 1.0))
    stability30 = float(np.clip(stability30, 0.0, 1.0))

    salg15 = _metric_salg(lift15, stability15)
    salg20 = _metric_salg(lift20, stability20)
    salg30 = _metric_salg(lift30, stability30)
    j20_mean_metrics = _metric_j_from_lift_stability_ndcg(
        lift20,
        stability20,
        ndcg_at_20,
    )

    return {
        "lift15": lift15,
        "lift30": lift30,
        "lift20": lift20,
        "lift10": lift10,
        "lift5": lift5,
        "hit_rate5": hit_rate5,
        "hit_rate10": hit_rate10,
        "hit_rate20": hit_rate20,
        "hit_rate15": hit_rate15,
        "hit_rate30": hit_rate30,
        "precision5": hit_rate5,
        "precision10": hit_rate10,
        "precision20": hit_rate20,
        "precision15": hit_rate15,
        "precision30": hit_rate30,
        "precision1": precision01,
        "precision01": precision01,
        "precision005": precision005,
        "ndcg@10": ndcg_at_10,
        "ndcg_at_10": ndcg_at_10,
        "ndcg20": ndcg_at_20,
        "ndcg@20": ndcg_at_20,
        "ndcg_at_20": ndcg_at_20,
        "ece_top20": ece_at_20,
        "ece_at_20": ece_at_20,
        "ndcg_tradable_top20pct": ndcg_tradable_top20pct,
        "ndcg_tradable_top10pct": ndcg_tradable_top10pct,
        "ndcg20_std": ndcg20_std,
        "ndcg20_p25": ndcg20_p25,
        "auc_correct_30": auc30,
        "stability20": stability20,
        "stability30": stability30,
        "stability15": stability15,
        "stability30_n_groups": float(
            np.sum([m.get("stability30_n_groups", 0.0) for m in fold_metrics])
        ),
        "stability15_n_groups": float(
            np.sum([m.get("stability15_n_groups", 0.0) for m in fold_metrics])
        ),
        "stability15_group_mean": _mean_metric(fold_metrics, "stability15_group_mean"),
        "stability15_group_std": _mean_metric(fold_metrics, "stability15_group_std"),
        "salg15": float(salg15),
        "salg20": float(salg20),
        "salg30": float(salg30),
        "J20": float(j20_mean_metrics),
        "j20": float(j20_mean_metrics),
        "J20_mean_metrics": float(j20_mean_metrics),
        "J_mean": j_mean,
        "J_median": j_median,
        "J_std": j_std,
        "J_se": j_se,
        "J_iqr": j_iqr,
        "J_robust": j_robust,
        "J_final": j_robust,
    }


def _hpo_objective_from_aggregate(
    agg: dict[str, float], objective_mode: str = "base"
) -> float:
    """HPO score aligned to robust top-20 SALG/NDCG model comparison."""
    base_score = float(agg.get("J_robust", agg.get("J_final", float("nan"))))
    if not np.isfinite(base_score):
        base_score = float(agg.get("J20", agg.get("j20", float("nan"))))
    if not np.isfinite(base_score):
        lift20 = float(agg.get("lift20", agg.get("lift15", agg.get("lift30", np.nan))))
        stability20 = float(
            agg.get(
                "stability20", agg.get("stability15", agg.get("stability30", np.nan))
            )
        )
        ndcg20 = float(
            agg.get("ndcg20", agg.get("ndcg_at_20", agg.get("ndcg@20", np.nan)))
        )
        base_score = _metric_j_from_lift_stability_ndcg(lift20, stability20, ndcg20)
    if not np.isfinite(base_score):
        return -999.0

    ndcg20 = float(agg.get("ndcg20", agg.get("ndcg_at_20", agg.get("ndcg@20", np.nan))))
    ndcg20_p25 = float(agg.get("ndcg20_p25", float("nan")))
    ndcg20_floor = float(agg.get("pareto_ndcg20_floor", 0.0))
    precision_floor = float(agg.get("pareto_precision_floor", 0.0))
    precision15 = float(agg.get("precision15", agg.get("hit_rate15", float("nan"))))
    lift20 = float(agg.get("lift20", agg.get("lift15", agg.get("lift30", np.nan))))
    salg20 = float(agg.get("salg20", float("nan")))

    penalty = 0.0
    if np.isfinite(ndcg20) and ndcg20 < 0.05:
        penalty += 0.05 - ndcg20
    if np.isfinite(ndcg20) and np.isfinite(ndcg20_floor) and ndcg20 < ndcg20_floor:
        penalty += ndcg20_floor - ndcg20
    if np.isfinite(ndcg20_p25) and ndcg20_p25 < 0.03:
        penalty += 0.03 - ndcg20_p25
    if np.isfinite(precision15) and precision15 < precision_floor:
        penalty += precision_floor - precision15
    if np.isfinite(lift20) and lift20 < 1.0:
        penalty += 0.10 * (1.0 - lift20)
    if np.isfinite(salg20) and salg20 < 0.0:
        penalty += 0.10 * abs(salg20)

    if str(objective_mode).lower() == "meta":
        ndcg10 = float(agg.get("ndcg_at_10", agg.get("ndcg@10", 0.0)))
        tradable_ndcg = float(
            agg.get("ndcg_tradable_top20pct", agg.get("ndcg_tradable_top10pct", 0.0))
        )
        base_score += 0.02 * ndcg10 + 0.02 * tradable_ndcg

    return float(base_score - penalty)


def _ebm_hpo_warm_start_path() -> Path:
    raw = os.environ.get("EPM_EBM_HPO_WARM_START_PATH")
    if raw:
        return Path(raw).expanduser()
    return Path("data/artifacts/ebm_on_lgbm_hpo_best_params.json")


def _clip_float_param(value: Any, lo: float, hi: float) -> float:
    return float(np.clip(float(value), lo, hi))


def _clip_int_param(value: Any, lo: int, hi: int) -> int:
    return int(np.clip(int(round(float(value))), lo, hi))


def _quantize_hpo_learning_rate(value: Any) -> float:
    return float(np.clip(round(float(value) / 0.01) * 0.01, 0.01, 0.05))


def _start_stage_heartbeat(label: str, *, interval_sec: float = 60.0) -> Callable[[], None]:
    """Emit coarse progress while a blocking native fit is running."""
    stop_event = threading.Event()
    start = time.monotonic()

    def _run() -> None:
        while not stop_event.wait(float(max(5.0, interval_sec))):
            tprint(f"EBMOnLGBM: {label} still running ({time.monotonic() - start:.0f}s).")

    thread = threading.Thread(target=_run, name=f"ebm-{label}-heartbeat", daemon=True)
    thread.start()

    def _stop() -> None:
        stop_event.set()
        thread.join(timeout=1.0)

    return _stop


def _sanitize_ebm_hpo_trial_params(params: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    if "learning_rate" in params:
        sanitized["learning_rate"] = _quantize_hpo_learning_rate(
            params["learning_rate"]
        )
    if "max_bins" in params:
        sanitized["max_bins"] = _clip_int_param(params["max_bins"], 16, 48)
    if "smoothing_rounds" in params:
        sanitized["smoothing_rounds"] = _clip_int_param(
            params["smoothing_rounds"], 100, 100
        )
    if "max_leaves" in params:
        sanitized["max_leaves"] = _clip_int_param(params["max_leaves"], 2, 3)
    if "reg_alpha" in params:
        sanitized["reg_alpha"] = _clip_float_param(params["reg_alpha"], 0.01, 2.0)
    if "reg_lambda" in params:
        sanitized["reg_lambda"] = _clip_float_param(params["reg_lambda"], 0.5, 10.0)
    if "greedy_ratio" in params:
        sanitized["greedy_ratio"] = _clip_float_param(params["greedy_ratio"], 0.1, 20.0)
    required = {
        "learning_rate",
        "max_bins",
        "smoothing_rounds",
        "max_leaves",
        "reg_alpha",
        "reg_lambda",
        "greedy_ratio",
    }
    return sanitized if required.issubset(sanitized) else {}


def _load_ebm_hpo_warm_start() -> tuple[dict[str, Any], Path] | tuple[None, Path]:
    path = _ebm_hpo_warm_start_path()
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except FileNotFoundError:
        return None, path
    except Exception as exc:
        tprint(f"  EBMOnLGBM: ignoring unreadable HPO warm-start cache {path}: {exc}")
        return None, path
    if not isinstance(payload, dict):
        return None, path
    params = payload.get("best_params", payload)
    if not isinstance(params, dict):
        return None, path
    sanitized = _sanitize_ebm_hpo_trial_params(params)
    return (sanitized or None), path


def _save_ebm_hpo_warm_start(
    *,
    best_params: dict[str, Any],
    best_value: float,
    best_trial_number: int,
    best_trial_attrs: dict[str, Any],
    leaf_min_pct: float,
) -> None:
    path = _ebm_hpo_warm_start_path()
    params = _sanitize_ebm_hpo_trial_params(best_params)
    if not params:
        return
    payload = {
        "schema_version": 1,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_params": params,
        "min_samples_leaf_pct": float(leaf_min_pct),
        "best_value": float(best_value),
        "best_trial_number": int(best_trial_number),
        "metrics": {
            "lift30": float(best_trial_attrs.get("lift30", np.nan)),
            "lift15": float(
                best_trial_attrs.get("lift15", best_trial_attrs.get("lift30", np.nan))
            ),
            "stability30": float(best_trial_attrs.get("stability30", np.nan)),
            "stability15": float(
                best_trial_attrs.get(
                    "stability15", best_trial_attrs.get("stability30", np.nan)
                )
            ),
            "precision15": float(best_trial_attrs.get("precision15", np.nan)),
            "precision01": float(best_trial_attrs.get("precision01", np.nan)),
            "precision005": float(best_trial_attrs.get("precision005", np.nan)),
            "ndcg_at_10": float(
                best_trial_attrs.get(
                    "ndcg_at_10", best_trial_attrs.get("ndcg@10", np.nan)
                )
            ),
            "ndcg_at_20": float(
                best_trial_attrs.get(
                    "ndcg_at_20", best_trial_attrs.get("ndcg@20", np.nan)
                )
            ),
            "ece_top20": float(
                best_trial_attrs.get(
                    "ece_top20", best_trial_attrs.get("ece_at_20", np.nan)
                )
            ),
            "hpo_objective": float(best_trial_attrs.get("hpo_objective", best_value)),
        },
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp_path, path)
    except Exception as exc:
        tprint(f"  EBMOnLGBM: failed to persist HPO warm-start cache {path}: {exc}")


def _hpo_leaf_min_pct_formula(
    *,
    max_leaves: int,
    max_bins: int,
    smoothing_rounds: int,
    n_features: int,
) -> float:
    """Couple EBM leaf support to shape complexity instead of tuning it directly."""
    leaves = float(max(2, int(max_leaves)))
    bins = float(max(16, int(max_bins)))
    smoothing = float(max(0, int(smoothing_rounds)))
    n_feat = float(max(1, int(n_features)))

    leaf_term = 0.0125 * max(0.0, leaves - 2.0)
    bin_term = 0.0125 * np.clip((bins - 16.0) / 48.0, 0.0, 1.0)
    low_smoothing_term = 0.0100 * np.clip((250.0 - smoothing) / 250.0, 0.0, 1.0)
    feature_term = 0.0075 * np.clip(
        (np.log1p(n_feat) - np.log1p(40.0)) / (np.log1p(800.0) - np.log1p(40.0)),
        0.0,
        1.0,
    )
    pct = (
        EBM_HPO_MIN_LEAF_PCT_LO
        + leaf_term
        + bin_term
        + low_smoothing_term
        + feature_term
    )
    return float(np.clip(pct, EBM_HPO_MIN_LEAF_PCT_LO, EBM_HPO_MIN_LEAF_PCT_HI))


def _top_stability(y: np.ndarray, pred: np.ndarray, frac: float) -> float:
    if len(pred) < 20:
        return 0.0
    _frac = float(np.clip(float(frac), 0.001, 1.0))
    k = max(1, int(np.ceil(_frac * len(pred))))
    idx = np.argsort(pred)[-k:]
    s = np.asarray(pred[idx], dtype=np.float64)
    yy = np.asarray(y[idx], dtype=np.float64)
    if len(s) == 0:
        return 0.0
    q = np.quantile(s, np.linspace(0.0, 1.0, 6))
    vals: list[float] = []
    for i in range(5):
        m = (s >= q[i]) & (s < q[i + 1] if i < 4 else s <= q[i + 1])
        if np.any(m):
            vals.append(float(np.mean(yy[m])))
    return float(1.0 / (1.0 + np.std(vals))) if vals else 0.0


def _top30_stability(y: np.ndarray, pred: np.ndarray) -> float:
    return _top_stability(y, pred, 0.30)


def _grouped_top30_stability(
    y: np.ndarray,
    pred: np.ndarray,
    classifier: bool,
    groups: Any = None,
    min_groups: int = 3,
    min_group_n: int = 20,
) -> dict[str, float]:
    out = _grouped_top_stability(
        y,
        pred,
        0.30,
        classifier=classifier,
        groups=groups,
        min_groups=min_groups,
        min_group_n=min_group_n,
    )
    return {
        "stability30": float(out["stability"]),
        "stability30_n_groups": float(out["n_groups"]),
        "stability30_group_mean": float(out["group_mean"]),
        "stability30_group_std": float(out["group_std"]),
    }


def _grouped_top_stability(
    y: np.ndarray,
    pred: np.ndarray,
    frac: float,
    classifier: bool,
    groups: Any = None,
    min_groups: int = 3,
    min_group_n: int = 20,
) -> dict[str, float]:
    if groups is None:
        return {
            "stability": 0.0,
            "n_groups": 0.0,
            "group_mean": 0.0,
            "group_std": 0.0,
        }
    yy = np.asarray(y, dtype=np.float64)
    pp = np.asarray(pred, dtype=np.float64)
    gg = np.asarray(groups, dtype=object)
    _frac = float(np.clip(float(frac), 0.001, 1.0))
    n = min(len(yy), len(pp), len(gg))
    yy = yy[:n]
    pp = pp[:n]
    gg = gg[:n]
    m = np.isfinite(yy) & np.isfinite(pp)
    yy = yy[m]
    pp = pp[m]
    gg = gg[m]
    vals: list[float] = []
    for g in pd.unique(pd.Series(gg)):
        gm = gg == g
        if int(np.sum(gm)) < int(min_group_n):
            continue
        yg = yy[gm]
        pg = pp[gm]
        k = max(1, int(np.ceil(_frac * len(pg))))
        if k >= len(pg):
            continue
        top = np.argsort(pg)[-k:]
        if classifier:
            yb = (yg >= 0.5).astype(np.int8)
            base = float(np.mean(yb))
            if base <= 1e-6:
                continue
            vals.append(float(np.mean(yb[top]) / base))
        else:
            denom = float(np.mean(np.abs(yg))) + 1e-6
            vals.append(float(np.mean(yg[top]) / denom))
    if len(vals) < int(min_groups):
        return {
            "stability": 0.0,
            "n_groups": float(len(vals)),
            "group_mean": float(np.mean(vals)) if vals else 0.0,
            "group_std": float(np.std(vals)) if vals else 0.0,
        }
    arr = np.asarray(vals, dtype=np.float64)
    mean_v = float(np.mean(arr))
    std_v = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    cv_v = float(std_v / (abs(mean_v) + 1e-6))
    stability_v = float(1.0 / (1.0 + cv_v))
    return {
        "stability": float(np.clip(stability_v, 0.0, 1.0)),
        "n_groups": float(len(arr)),
        "group_mean": mean_v,
        "group_std": std_v,
    }


def _stability_group_labels(
    n: int,
    timestamps: Any = None,
    assets: Any = None,
) -> np.ndarray | None:
    if n <= 0:
        return None
    if timestamps is None or len(np.asarray(timestamps)) != n:
        if assets is None or len(np.asarray(assets)) != n:
            return None
        return np.asarray(assets).astype(str)
    ts = pd.to_datetime(np.asarray(timestamps), utc=True, errors="coerce")
    week = (
        pd.Series(ts)
        .dt.tz_localize(None)
        .dt.to_period("W")
        .astype(str)
        .to_numpy(dtype=object)
    )
    week = np.where(pd.isna(week), "__unknown_week__", week).astype(object)
    if assets is None or len(np.asarray(assets)) != n:
        return week.astype(str)
    asset_arr = np.asarray(assets).astype(str)
    counts = pd.Series(asset_arr).value_counts()
    common = set(counts[counts >= 20].index.astype(str))
    asset_bucket = np.asarray(
        [a if a in common else "__rare_asset__" for a in asset_arr], dtype=object
    )
    combined = np.asarray(
        [f"{w}|{a}" for w, a in zip(week.astype(str), asset_bucket.astype(str))],
        dtype=object,
    )
    if pd.Series(combined).value_counts().ge(20).sum() >= 3:
        return combined.astype(str)
    return week.astype(str)


def _monthly_stability_group_labels(
    n: int,
    timestamps: Any = None,
    assets: Any = None,
) -> np.ndarray | None:
    if n <= 0 or timestamps is None or len(np.asarray(timestamps)) != n:
        return None
    ts = pd.to_datetime(np.asarray(timestamps), utc=True, errors="coerce")
    month = (
        pd.Series(ts)
        .dt.tz_localize(None)
        .dt.to_period("M")
        .astype(str)
        .to_numpy(dtype=object)
    )
    month = np.where(pd.isna(month), "__unknown_month__", month).astype(object)
    if assets is None or len(np.asarray(assets)) != n:
        return month.astype(str)
    asset_arr = np.asarray(assets).astype(str)
    counts = pd.Series(asset_arr).value_counts()
    common = set(counts[counts >= 20].index.astype(str))
    asset_bucket = np.asarray(
        [a if a in common else "__rare_asset__" for a in asset_arr], dtype=object
    )
    combined = np.asarray(
        [f"{m}|{a}" for m, a in zip(month.astype(str), asset_bucket.astype(str))],
        dtype=object,
    )
    if pd.Series(combined).value_counts().ge(20).sum() >= 3:
        return combined.astype(str)
    return month.astype(str)


def _calendar_month_group_labels(n: int, timestamps: Any = None) -> np.ndarray | None:
    if n <= 0 or timestamps is None or len(np.asarray(timestamps)) != n:
        return None
    ts = pd.to_datetime(np.asarray(timestamps), utc=True, errors="coerce")
    month = (
        pd.Series(ts)
        .dt.tz_localize(None)
        .dt.to_period("M")
        .astype(str)
        .to_numpy(dtype=object)
    )
    return np.where(pd.isna(month), "__unknown_month__", month).astype(str)


def _asset_stability_group_labels(n: int, assets: Any = None) -> np.ndarray | None:
    if n <= 0 or assets is None or len(np.asarray(assets)) != n:
        return None
    asset_arr = np.asarray(assets).astype(str)
    counts = pd.Series(asset_arr).value_counts()
    common = set(counts[counts >= 20].index.astype(str))
    return np.asarray(
        [a if a in common else "__rare_asset__" for a in asset_arr], dtype=object
    ).astype(str)


def _stability_group_bundle(
    n: int,
    timestamps: Any = None,
    assets: Any = None,
) -> dict[str, np.ndarray] | np.ndarray | None:
    week = _stability_group_labels(n, timestamps=timestamps, assets=assets)
    month = _monthly_stability_group_labels(n, timestamps=timestamps, assets=assets)
    calendar_month = _calendar_month_group_labels(n, timestamps=timestamps)
    asset = _asset_stability_group_labels(n, assets=assets)
    if week is None:
        return None
    out = {"week": week}
    if month is not None:
        out["month"] = month
    if calendar_month is not None:
        out["calendar_month"] = calendar_month
    if asset is not None:
        out["asset"] = asset
    return out


def _groups_take(groups: Any, idx: Any) -> Any:
    if groups is None:
        return None
    if isinstance(groups, dict):
        return {k: np.asarray(v, dtype=object)[idx] for k, v in groups.items()}
    return np.asarray(groups, dtype=object)[idx]


def _groups_primary(groups: Any) -> Any:
    if isinstance(groups, dict):
        return groups.get("week")
    return groups


def _rank_ic_metrics(
    y: np.ndarray, pred: np.ndarray, frac: float = 0.30
) -> dict[str, float]:
    yy = np.asarray(y, dtype=np.float64)
    pp = np.asarray(pred, dtype=np.float64)
    m = np.isfinite(yy) & np.isfinite(pp)
    yy = yy[m]
    pp = pp[m]
    if len(yy) < 8:
        return {"ic_total": 0.0, "ic_top30": 0.0, "ic_top": 0.0}
    ic_total = _safe_spearman(pp, yy)
    k = max(1, int(np.ceil(float(np.clip(float(frac), 0.001, 1.0)) * len(pp))))
    top = np.argsort(pp)[-k:]
    ic_top = _safe_spearman(pp[top], yy[top]) if len(top) >= 8 else 0.0
    result = {"ic_total": float(ic_total), "ic_top": float(ic_top), "ic_top30": 0.0}
    if abs(float(frac) - 0.30) <= 0.001:
        result["ic_top30"] = float(ic_top)
    return result


def _stratified_subsample_indices(
    y: np.ndarray, max_n: int, random_state: int, classifier: bool
) -> np.ndarray:
    n = len(y)
    if n <= max_n:
        return np.arange(n, dtype=np.int32)
    rng = np.random.default_rng(random_state)
    if classifier:
        strata = np.asarray(y >= 0.5, dtype=np.int8)
    else:
        ranks = pd.Series(np.asarray(y, dtype=np.float32)).rank(pct=True).to_numpy()
        strata = np.clip((ranks * 5).astype(np.int32), 0, 4)
    out: list[np.ndarray] = []
    for s in np.unique(strata):
        ids = np.where(strata == s)[0]
        take = max(1, int(round(max_n * len(ids) / n)))
        take = min(take, len(ids))
        out.append(rng.choice(ids, size=take, replace=False))
    idx = np.sort(np.concatenate(out).astype(np.int32))
    if len(idx) > max_n:
        idx = np.sort(rng.choice(idx, size=max_n, replace=False).astype(np.int32))
    return idx


def _stage_partition_indices(
    y: np.ndarray,
    *,
    timestamps: Any = None,
    assets: Any = None,
    random_state: int,
) -> dict[str, np.ndarray]:
    """Split current train subset into interwoven LGBM/prune, HPO, and fit/OOF stages."""
    y_arr = np.asarray(y)
    n = len(y_arr)
    if n == 0:
        empty = np.array([], dtype=np.int32)
        return {"lgbm_prune": empty, "hpo": empty, "fit_oof": empty}

    if _looks_classifier_target(y_arr):
        y_bucket = np.asarray(y_arr >= 0.5, dtype=np.int8).astype(str)
    else:
        ranks = pd.Series(np.asarray(y_arr, dtype=np.float32)).rank(pct=True).to_numpy()
        y_bucket = np.clip((ranks * 5).astype(np.int32), 0, 4).astype(str)

    if assets is not None and len(np.asarray(assets)) == n:
        asset_arr = np.asarray(assets).astype(str)
        counts = pd.Series(asset_arr).value_counts()
        common = set(counts[counts >= 20].index.astype(str))
        asset_bucket = np.asarray(
            [a if a in common else "__rare_asset__" for a in asset_arr], dtype=object
        )
    else:
        asset_bucket = np.asarray(["__all_assets__"] * n, dtype=object)

    if timestamps is not None and len(np.asarray(timestamps)) == n:
        ts = pd.to_datetime(np.asarray(timestamps), utc=True, errors="coerce")
        if bool(pd.Series(ts).notna().any()):
            week = (
                pd.Series(ts)
                .dt.tz_localize(None)
                .dt.to_period("W")
                .astype(str)
                .to_numpy()
            )
            week_rank = pd.Series(week).rank(method="dense").to_numpy(dtype=np.int32)
        else:
            week_rank = np.arange(n, dtype=np.int32)
    else:
        week_rank = np.arange(n, dtype=np.int32)

    strata = np.asarray(
        [f"{yb}|{ab}" for yb, ab in zip(y_bucket, asset_bucket)], dtype=object
    )
    rng = np.random.default_rng(random_state)
    pattern = np.asarray(["lgbm_prune"] * 7 + ["hpo"] * 2 + ["fit_oof"] * 11)
    out = {"lgbm_prune": [], "hpo": [], "fit_oof": []}
    for stratum in np.unique(strata):
        ids = np.where(strata == stratum)[0]
        if len(ids) == 0:
            continue
        jitter = rng.random(len(ids)) * 1e-6
        order = np.lexsort((jitter, np.arange(len(ids)) % 997, week_rank[ids]))
        ordered = ids[order]
        offset = int(rng.integers(0, len(pattern)))
        labels = pattern[(np.arange(len(ordered)) + offset) % len(pattern)]
        for key in out:
            out[key].extend(ordered[labels == key].tolist())

    result = {
        key: np.asarray(sorted(vals), dtype=np.int32) for key, vals in out.items()
    }
    assigned = np.concatenate([v for v in result.values() if len(v)])
    if len(np.unique(assigned)) != n:
        missing = np.setdiff1d(
            np.arange(n, dtype=np.int32), assigned, assume_unique=False
        )
        result["fit_oof"] = np.asarray(
            sorted(np.concatenate([result["fit_oof"], missing]).tolist()),
            dtype=np.int32,
        )
    tprint(
        "EBMOnLGBM stage split: "
        f"lgbm_prune={len(result['lgbm_prune'])}/{n} "
        f"({len(result['lgbm_prune']) / max(n, 1):.1%}), "
        f"hpo={len(result['hpo'])}/{n} ({len(result['hpo']) / max(n, 1):.1%}), "
        f"fit_oof={len(result['fit_oof'])}/{n} "
        f"({len(result['fit_oof']) / max(n, 1):.1%})."
    )
    return result


def _looks_classifier_target(y: np.ndarray) -> bool:
    yy = np.asarray(y)
    unique = np.unique(yy[np.isfinite(yy)])
    return bool(len(unique) <= 20 and np.all(np.isclose(unique, np.round(unique))))


def _rank_cols_2d(X: np.ndarray) -> np.ndarray:
    x = np.asarray(X, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return pd.DataFrame(x).rank(pct=True).to_numpy(dtype=np.float32)


def _vectorized_spearman_scores(
    X: np.ndarray, y: np.ndarray, *, signed: bool = False
) -> np.ndarray:
    if X.shape[1] == 0:
        return np.zeros(0, dtype=np.float32)
    xr = _rank_cols_2d(X).astype(np.float64, copy=False)
    yr = (
        pd.Series(np.asarray(y, dtype=np.float64))
        .rank(pct=True)
        .to_numpy(dtype=np.float64)
    )
    xr = xr - np.nanmean(xr, axis=0)
    yr = yr - np.nanmean(yr)
    x_std = np.sqrt(np.nanmean(xr * xr, axis=0))
    y_std = float(np.sqrt(np.nanmean(yr * yr)))
    denom = np.maximum(x_std * max(y_std, 1e-12), 1e-12)
    corr = np.nanmean(xr * yr[:, None], axis=0) / denom
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    if not signed:
        corr = np.abs(corr)
    return corr.astype(np.float32)


def _lift_at_30(feature: np.ndarray, target: np.ndarray) -> float:
    x = np.asarray(feature, dtype=np.float32)
    y = np.asarray(target, dtype=np.float32)
    m = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(m)) < 20:
        return 0.0
    x = x[m]
    y = y[m]
    base = float(np.mean(y))
    if abs(base) < 1e-9 or float(np.nanstd(x)) < 1e-12:
        return 0.0
    threshold = float(np.quantile(x, 0.70))
    top = y[x >= threshold]
    if len(top) == 0:
        return 0.0
    lift = float(np.mean(top) / base)
    return float(np.clip(lift - 1.0, -1.0, 1.0))


def _tree_feature_target_scores(
    X: np.ndarray,
    y: np.ndarray,
    classifier: bool,
    random_state: int,
) -> np.ndarray:
    n, p = X.shape
    if p == 0:
        return np.zeros(0, dtype=np.float32)
    splitter = (
        StratifiedKFold(n_splits=EBM_CV_SPLITS, shuffle=True, random_state=random_state)
        if classifier
        else KFold(n_splits=EBM_CV_SPLITS, shuffle=True, random_state=random_state)
    )
    y_split = (
        np.asarray(y >= 0.5, dtype=np.int8)
        if classifier
        else np.asarray(y, dtype=np.float32)
    )
    ic_folds: list[np.ndarray] = []
    lift_folds: list[np.ndarray] = []
    for _, test_idx in splitter.split(np.zeros(n), y_split):
        X_te = np.asarray(X[test_idx], dtype=np.float32)
        y_te = np.asarray(y[test_idx], dtype=np.float32)
        ic = _vectorized_spearman_scores(X_te, y_te, signed=True)
        lift = np.asarray([_lift_at_30(X_te[:, j], y_te) for j in range(p)])
        ic_folds.append(ic)
        lift_folds.append(lift.astype(np.float32))
    ic_mat = np.vstack(ic_folds).astype(np.float32)
    lift_mat = np.vstack(lift_folds).astype(np.float32)
    spearman_ic = np.nanmean(ic_mat, axis=0)
    lift_score = np.nanmean(lift_mat, axis=0)
    fold_stability = 1.0 - np.nanstd(ic_mat, axis=0)
    fold_stability = np.clip(fold_stability, 0.0, 1.0)
    coverage = np.mean(np.asarray(X, dtype=np.float32) != 0.0, axis=0)
    score = (
        0.50 * np.abs(spearman_ic)
        + 0.25 * np.maximum(lift_score, 0.0)
        + 0.15 * fold_stability
        + 0.10 * coverage
    )
    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _target_aware_tree_feature_cap(
    X_tree: np.ndarray,
    X_eval_tree: np.ndarray,
    tree_names: list[str],
    y: np.ndarray,
    classifier: bool,
    random_state: int,
    cap: int = EBM_TREE_FEATURE_CAP,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if X_tree.shape[1] == 0:
        return X_tree, X_eval_tree, tree_names
    X = np.nan_to_num(
        np.asarray(X_tree, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    X_ev = np.nan_to_num(
        np.asarray(X_eval_tree, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    names = list(tree_names)

    var = np.var(X, axis=0)
    coverage = np.mean(X != 0.0, axis=0)
    hygiene = (var > 1e-8) & (coverage >= 0.005)
    X = X[:, hygiene]
    X_ev = X_ev[:, hygiene]
    names = [n for n, k in zip(names, hygiene) if k]
    if X.shape[1] == 0:
        return X, X_ev, names

    dup_sub_n = min(len(X), 2000)
    rng = np.random.default_rng(random_state + 617)
    dup_sub = (
        rng.choice(len(X), size=dup_sub_n, replace=False)
        if len(X) > dup_sub_n
        else np.arange(len(X))
    )
    signatures: dict[bytes, int] = {}
    keep_dup = np.ones(X.shape[1], dtype=bool)
    var = np.var(X, axis=0)
    for j in range(X.shape[1]):
        sig = np.round(X[dup_sub, j], 6).astype(np.float32).tobytes()
        prior = signatures.get(sig)
        if prior is None:
            signatures[sig] = j
            continue
        if var[j] > var[prior]:
            keep_dup[prior] = False
            signatures[sig] = j
        else:
            keep_dup[j] = False
    if not np.all(keep_dup):
        X = X[:, keep_dup]
        X_ev = X_ev[:, keep_dup]
        names = [n for n, k in zip(names, keep_dup) if k]

    del y, classifier
    score = _lgbm_leaf_screen_scores(X, names)

    if X.shape[1] > EBM_TREE_TARGET_RANK_CAP:
        keep_rank = np.argsort(score)[-EBM_TREE_TARGET_RANK_CAP:]
        keep_rank = np.sort(keep_rank)
        X = X[:, keep_rank]
        X_ev = X_ev[:, keep_rank]
        score = score[keep_rank]
        names = [names[i] for i in keep_rank]

    if X.shape[1] <= 1 or len(X) == 0:
        tprint(
            "EBMOnLGBM tree cap: hygiene/ranking kept "
            f"{X.shape[1]} features; correlation pruning not applicable."
        )
        return X, X_ev, names
    if X.shape[1] <= EBM_TREE_CORR_PRUNE_MIN_FEATURES:
        tprint(
            "EBMOnLGBM tree cap: hygiene/ranking kept "
            f"{X.shape[1]} features; below corr-prune threshold "
            f"{EBM_TREE_CORR_PRUNE_MIN_FEATURES}."
        )
        return X, X_ev, names

    sub_n = min(len(X), 5000)
    rng = np.random.default_rng(random_state + 871)
    sub = (
        rng.choice(len(X), size=sub_n, replace=False)
        if len(X) > sub_n
        else np.arange(len(X))
    )
    Xs = X[sub].astype(np.float32, copy=False)
    corr = np.abs(np.corrcoef(Xs.T))
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    leaf_value_strength = np.mean(np.abs(X), axis=0)
    order = np.argsort(score)[::-1]
    keep_mask = np.ones(X.shape[1], dtype=bool)
    for i in order:
        if not keep_mask[i]:
            continue
        dup = np.where((corr[i] > EBM_TREE_CORR_PRUNE_THRESHOLD) & keep_mask)[0]
        for j in dup:
            if j == i:
                continue
            if leaf_value_strength[j] <= leaf_value_strength[i]:
                keep_mask[j] = False
            else:
                keep_mask[i] = False
                break
    kept = np.where(keep_mask)[0]
    if len(kept) > cap:
        kept = kept[np.argsort(score[kept])[-cap:]]
    kept = np.sort(kept)
    tprint(
        "EBMOnLGBM tree cap: "
        f"hygiene={len(names)}, ranked_cap={min(len(names), EBM_TREE_TARGET_RANK_CAP)}, "
        f"kept={len(kept)} by target-aware corr pruning "
        f"(thr={EBM_TREE_CORR_PRUNE_THRESHOLD:.2f})."
    )
    return X[:, kept], X_ev[:, kept], [names[i] for i in kept]


def max_leaves_for_tree(tree_idx: int) -> int:
    if tree_idx < 10:
        return 8
    if tree_idx < 25:
        return 6
    if tree_idx < 50:
        return 4
    if tree_idx < 100:
        return 2
    if tree_idx < 200:
        return 1
    return 0


def _parse_tree_feature_name(name: str) -> tuple[int, int] | None:
    m = re.search(r"_tree(\d+)_leaf(\d+)_soft$", str(name))
    if m is not None:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"_tree(\d+)_value$", str(name))
    if m is not None:
        return int(m.group(1)), -1
    return None


def _is_lgbm_tree_feature_name(name: str) -> bool:
    s = str(name).lower()
    return (
        s.startswith("lgbm_")
        or re.search(r"_tree\d+_(?:leaf\d+_soft|value)$", s) is not None
    )


def _safe_zscore(x: np.ndarray) -> np.ndarray:
    vals = np.asarray(x, dtype=np.float64)
    mask = np.isfinite(vals)
    if int(np.sum(mask)) < 2:
        return np.zeros(len(vals), dtype=np.float32)
    mu = float(np.nanmean(vals[mask]))
    sd = float(np.nanstd(vals[mask]))
    out = (vals - mu) / (sd + 1e-8)
    out[~mask] = 0.0
    return out.astype(np.float32)


def _lgbm_leaf_screen_scores(X: np.ndarray, names: list[str]) -> np.ndarray:
    n, p = X.shape
    if p == 0:
        return np.zeros(0, dtype=np.float32)
    arr = np.nan_to_num(
        np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    model_contrib = np.sum(np.abs(arr), axis=1)
    ranks = pd.Series(model_contrib).rank(pct=True).to_numpy(dtype=np.float32)
    top15 = ranks >= 0.85
    eps = 1e-8
    support_all = np.mean(np.abs(arr), axis=0)
    support_top15 = (
        np.mean(np.abs(arr[top15]), axis=0)
        if np.any(top15)
        else np.zeros(p, dtype=np.float32)
    )
    contrib_all = np.mean(np.abs(arr), axis=0)
    contrib_top15 = (
        np.mean(np.abs(arr[top15]), axis=0)
        if np.any(top15)
        else np.zeros(p, dtype=np.float32)
    )
    share_top15_num = (
        np.sum(np.abs(arr[top15]), axis=0)
        if np.any(top15)
        else np.zeros(p, dtype=np.float32)
    )
    share_top15 = share_top15_num / (float(np.sum(share_top15_num)) + eps)
    a15 = np.log((support_top15 + eps) / (support_all + eps))
    c15 = np.log((contrib_top15 + eps) / (contrib_all + eps))
    raw_score = np.zeros(p, dtype=np.float32)
    tree_idx = np.array(
        [
            _parse_tree_feature_name(nm)[0] if _parse_tree_feature_name(nm) else -1
            for nm in names
        ],
        dtype=np.int32,
    )
    for lo, hi in TREE_BLOCKS:
        mask = (tree_idx >= lo) & (tree_idx < hi)
        if not np.any(mask):
            continue
        z1 = _safe_zscore(share_top15[mask])
        z2 = _safe_zscore(a15[mask])
        z3 = _safe_zscore(c15[mask])
        block_score = 0.50 * z1 + 0.30 * z2 + 0.20 * z3
        raw_score[mask] = block_score
    keep = np.zeros(p, dtype=bool)
    per_tree: dict[int, list[int]] = {}
    for j, nm in enumerate(names):
        parsed = _parse_tree_feature_name(nm)
        if parsed is None:
            keep[j] = True
            continue
        ti, _ = parsed
        if ti >= 200:
            continue
        per_tree.setdefault(ti, []).append(j)
    for ti, idxs in per_tree.items():
        k = max_leaves_for_tree(ti)
        if k <= 0:
            continue
        adjusted = [
            (ix, float(raw_score[ix] + _diversity_bonus_for_leaf_name(names[ix])))
            for ix in idxs
        ]
        order = [ix for ix, _ in sorted(adjusted, key=lambda x: x[1], reverse=True)[:k]]
        keep[np.asarray(order, dtype=np.int32)] = True
    out = np.where(keep, raw_score, -np.inf).astype(np.float32)
    return out


def _select_leaf_names_from_score_matrix(
    X_score: np.ndarray, names: list[str], cap: int = EBM_TREE_FEATURE_CAP
) -> list[str]:
    score = _lgbm_leaf_screen_scores(X_score, names)
    order = np.argsort(score)[::-1]
    chosen = [names[i] for i in order if np.isfinite(score[i])]
    if len(chosen) > cap:
        chosen = chosen[:cap]
    return chosen


def _diversity_bonus_for_leaf_name(name: str) -> float:
    parsed = _parse_tree_feature_name(name)
    if parsed is None:
        return 0.0
    toks = re.findall(
        r"(cross_asset|funding|orderbook_wall|price|volume)", name.lower()
    )
    return 0.03 * float(len(set(toks)) >= 2)


def _target_scores(X: np.ndarray, y: np.ndarray, cols: list[str]) -> np.ndarray:
    del cols
    return _vectorized_spearman_scores(np.asarray(X, dtype=np.float32), y, signed=False)


def _corr_prune(
    X: np.ndarray, y: np.ndarray, active: np.ndarray, thr: float, random_state: int
) -> np.ndarray:
    if len(active) <= 2:
        return active
    sub_n = min(len(X), 5000)
    rng = np.random.default_rng(random_state)
    sub = (
        rng.choice(len(X), size=sub_n, replace=False)
        if len(X) > sub_n
        else np.arange(len(X))
    )
    Xa = X[sub][:, active].astype(np.float32)
    Xa = pd.DataFrame(Xa).rank(pct=True).to_numpy(dtype=np.float32)
    corr = np.abs(np.corrcoef(Xa, rowvar=False))
    corr = np.nan_to_num(corr, nan=0.0)
    scores = _target_scores(X[:, active], y, [str(i) for i in active])
    order = np.argsort(scores)[::-1]
    keep = np.ones(len(active), dtype=bool)
    for pos, i in enumerate(order):
        if not keep[i]:
            continue
        related = np.where(corr[i] > thr)[0]
        for j in related:
            if j != i and scores[j] <= scores[i]:
                keep[j] = False
        keep[i] = True
    return active[keep]


def _spearman_target_screen(
    X: np.ndarray, y: np.ndarray, active: np.ndarray, max_features: int | None
) -> np.ndarray:
    if max_features is None:
        max_features = int(
            EBM_PRESCREEN_BASE_FEATURES + EBM_PRESCREEN_FEATURE_FRACTION * len(active)
        )
    if max_features <= 0:
        return active
    if len(active) <= max_features:
        return active
    scores = _target_scores(X[:, active], y, [str(i) for i in active])
    order = np.argsort(scores)[::-1][:max_features]
    return active[np.sort(order)]


def _spearman_instability_screen(
    X: np.ndarray,
    y: np.ndarray,
    active: np.ndarray,
    classifier: bool,
    random_state: int,
    min_consistency: float = 0.6,
) -> np.ndarray:
    if len(active) <= EBM_MIN_FEATURES:
        return active
    splitter: Any
    y_split = (
        np.asarray(y >= 0.5, dtype=np.int8)
        if classifier
        else np.asarray(y, dtype=np.float32)
    )
    splitter = (
        StratifiedKFold(n_splits=EBM_CV_SPLITS, shuffle=True, random_state=random_state)
        if classifier
        else KFold(n_splits=EBM_CV_SPLITS, shuffle=True, random_state=random_state)
    )
    signs = np.zeros((EBM_CV_SPLITS, len(active)), dtype=np.int8)
    mags = np.zeros((EBM_CV_SPLITS, len(active)), dtype=np.float32)
    for fold_i, (_, va) in enumerate(splitter.split(np.zeros(len(y_split)), y_split)):
        fold_corr = _vectorized_spearman_scores(X[va][:, active], y[va], signed=True)
        signs[fold_i] = np.where(fold_corr > 0, 1, np.where(fold_corr < 0, -1, 0))
        mags[fold_i] = np.abs(fold_corr)
    pos = np.sum(signs > 0, axis=0)
    neg = np.sum(signs < 0, axis=0)
    consistency = np.maximum(pos, neg) / float(EBM_CV_SPLITS)
    score = np.mean(mags, axis=0) - 0.5 * np.std(mags, axis=0)
    keep = (consistency >= min_consistency) & (score > 0.0)
    if int(np.sum(keep)) < EBM_MIN_FEATURES:
        keep = np.zeros(len(active), dtype=bool)
        keep[np.argsort(score)[-min(EBM_MIN_FEATURES, len(active)) :]] = True
    return active[keep]


def _prescreen_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    classifier: bool,
    random_state: int,
) -> np.ndarray:
    active = np.arange(len(feature_names), dtype=np.int32)
    tprint(f"EBMOnLGBM prescreen: start with {len(active)} features.")
    tprint("EBMOnLGBM prescreen: EN soft-stage disabled.")
    active = _spearman_target_screen(X, y, active, EBM_PRESCREEN_MAX_FEATURES)
    tprint(f"EBMOnLGBM prescreen: Spearman target -> {len(active)}")
    active = _corr_prune(X, y, active, thr=0.98, random_state=random_state)
    tprint(f"EBMOnLGBM prescreen: corr prune #1 -> {len(active)}")
    active = _spearman_instability_screen(
        X, y, active, classifier=classifier, random_state=random_state
    )
    tprint(f"EBMOnLGBM prescreen: Spearman instability -> {len(active)}")
    active = _corr_prune(X, y, active, thr=0.95, random_state=random_state + 1)
    tprint(f"EBMOnLGBM prescreen: corr prune #2 -> {len(active)}")
    return active.astype(np.int32)


def _ebm_specs(pruning: bool, random_state: int) -> list[dict[str, Any]]:
    if pruning:
        pairs = [(3.0, 0.1), (3.0, 1.0), (6.0, 0.1), (6.0, 1.0)]
        outer_bags = 1
        max_bins = 32
        learning_rate = 0.05
        early_stopping_rounds = 25
        smoothing_rounds = 10
    else:
        pairs = [(5.0, 0.01), (3.0, 0.1), (6.0, 0.1), (3.0, 1.0), (6.0, 1.0)]
        outer_bags = 10
        max_bins = 64
        learning_rate = 0.01
        early_stopping_rounds = 25
        smoothing_rounds = 400
    specs = []
    for i, (l2, l1) in enumerate(pairs):
        specs.append(
            {
                "max_bins": max_bins,
                "outer_bags": outer_bags,
                "learning_rate": learning_rate,
                "early_stopping_rounds": early_stopping_rounds,
                "interactions": 0,
                "min_samples_leaf": 2,
                "reg_lambda": float(l2),
                "reg_alpha": float(l1),
                "smoothing_rounds": smoothing_rounds,
                "random_state": int(random_state + i * 17),
                "n_jobs": 1,
                "binning": "uniform" if pruning else "quantile",
            }
        )
    return specs


def _predict_raw_ebm(model: Any, X: pd.DataFrame, mode: str) -> np.ndarray:
    X_pred = _coerce_ebm_feature_types(X)
    if mode == "classifier" and hasattr(model, "predict_proba"):
        p = np.asarray(model.predict_proba(X_pred), dtype=np.float64)
        if p.ndim == 2 and p.shape[1] > 1:
            return np.clip(p[:, 1], 1e-4, 1.0 - 1e-4).astype(np.float32)
        return np.clip(p.reshape(-1), 1e-4, 1.0 - 1e-4).astype(np.float32)
    return np.asarray(model.predict(X_pred), dtype=np.float32).reshape(-1)


def _binary_feature_mask(X: pd.DataFrame) -> np.ndarray:
    mask = np.zeros(X.shape[1], dtype=bool)
    for i, col in enumerate(X.columns):
        vals = pd.unique(X[col].dropna())
        if len(vals) <= 2:
            mask[i] = True
    return mask


def _coerce_ebm_feature_types(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    binary_mask = _binary_feature_mask(out)
    for col in out.columns[binary_mask]:
        out[col] = np.rint(out[col].to_numpy(dtype=np.float32)).astype(np.int8)
    return out


def _fit_one_ebm(
    cls: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray,
    params: dict[str, Any],
) -> Any:
    p = dict(params)
    p["n_jobs"] = 1
    X_fit = _coerce_ebm_feature_types(X)

    if "min_samples_leaf_pct" in p:
        p["min_samples_leaf"] = max(1, int(p.pop("min_samples_leaf_pct") * len(X)))
    else:
        p["min_samples_leaf"] = max(
            int(p.get("min_samples_leaf", 2)), int(0.03 * len(X))
        )

    if "monotone_constraints" in p:
        mc = p.pop("monotone_constraints")
        if len(mc) < len(X.columns):
            mc = list(mc) + [0] * (len(X.columns) - len(mc))
        p["monotone_constraints"] = mc

    model = _make_ebm(cls, p)
    fit_params = {}
    try:
        sig = inspect.signature(model.fit)
        if "sample_weight" in sig.parameters:
            fit_params["sample_weight"] = sample_weight
    except Exception:
        fit_params["sample_weight"] = sample_weight
    _quiet_interpret_logging()
    try:
        model.fit(X_fit, y, **fit_params)
    except (PermissionError, OSError) as exc:
        if int(p.get("n_jobs", 1)) <= 1:
            raise
        tprint(
            "EBMOnLGBM: EBM fit with n_jobs="
            f"{int(p.get('n_jobs', 1))} failed ({type(exc).__name__}: {exc}); "
            "retrying with n_jobs=1."
        )
        p["n_jobs"] = 1
        model = _make_ebm(cls, p)
        model.fit(X_fit, y, **fit_params)
    _quiet_interpret_logging()
    return model


def _shape_map(model: Any, feature_names: list[str]) -> dict[str, np.ndarray]:
    scores = getattr(model, "term_scores_", None)
    term_features = getattr(model, "term_features_", None)
    out: dict[str, np.ndarray] = {}
    if scores is None:
        return out
    if term_features is None:
        term_features = [(i,) for i in range(min(len(scores), len(feature_names)))]
    for term, shape in zip(term_features, scores):
        if len(term) != 1:
            continue
        idx = int(term[0])
        if 0 <= idx < len(feature_names):
            arr = np.asarray(shape, dtype=np.float64).ravel()
            if len(arr) > 1 and np.any(np.isfinite(arr)):
                out[feature_names[idx]] = np.nan_to_num(arr, nan=0.0)
    return out


def _resample_shape(shape: np.ndarray, n: int = 64) -> np.ndarray:
    arr = np.asarray(shape, dtype=np.float64).ravel()
    if len(arr) == 0:
        return np.zeros(n, dtype=np.float32)
    xp = np.linspace(0.0, 1.0, len(arr))
    xq = np.linspace(0.0, 1.0, n)
    return np.interp(xq, xp, arr).astype(np.float32)


def _feature_shape_score_components(
    models: list[Any], feature_names: list[str], binary_mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    maps = [_shape_map(m, feature_names) for m in models]
    n_models = len(maps)
    n_features = len(feature_names)
    if n_models == 0 or n_features == 0:
        return (
            np.zeros(n_features, dtype=np.float32),
            np.zeros(n_features, dtype=np.float32),
            np.zeros(n_features, dtype=bool),
            np.zeros(n_features, dtype=np.float32),
        )
    shape_tensor = np.zeros((n_models, n_features, 64), dtype=np.float32)
    present = np.zeros((n_models, n_features), dtype=bool)
    bends_tensor = np.zeros((n_models, n_features), dtype=np.float32)

    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    for mi, mp in enumerate(maps):
        if not mp:
            continue
        all_shapes = list(mp.values())
        mas_all = [np.mean(np.abs(s)) for s in all_shapes]
        max_mas = max(mas_all) if len(mas_all) > 0 and max(mas_all) > 0 else 1.0
        Tmax = (
            max([np.max(s) - np.min(s) for s in all_shapes])
            if len(all_shapes) > 0
            else 1.0
        )
        Tmin = (
            min([np.max(s) - np.min(s) for s in all_shapes])
            if len(all_shapes) > 0
            else 0.0
        )

        for name, shape in mp.items():
            idx = name_to_idx.get(name)
            if idx is None:
                continue
            shape_tensor[mi, idx] = _resample_shape(shape)
            present[mi, idx] = True

            if len(shape) <= 2:
                bends_tensor[mi, idx] = 0
                continue

            mas_i = np.mean(np.abs(shape))
            T = Tmax - (Tmax - Tmin) * (mas_i / max_mas)

            filtered_score = [shape[0]]
            for val in shape[1:]:
                if abs(val - filtered_score[-1]) >= T:
                    filtered_score.append(val)

            diffs = np.diff(filtered_score)
            flips = 0
            for k in range(1, len(diffs)):
                if (
                    np.sign(diffs[k]) != np.sign(diffs[k - 1])
                    and diffs[k] != 0
                    and diffs[k - 1] != 0
                ):
                    flips += 1
            bends_tensor[mi, idx] = flips

    counts = np.sum(present, axis=0)
    valid = counts > 0
    scores = np.zeros(n_features, dtype=np.float32)
    if not np.any(valid):
        return (
            scores,
            np.zeros(n_features, dtype=np.float32),
            np.zeros(n_features, dtype=bool),
            np.zeros(n_features, dtype=np.float32),
        )

    denom = np.maximum(counts[:, None], 1.0).astype(np.float32)
    masked_abs = np.abs(shape_tensor) * present[:, :, None]
    mas = np.sum(masked_abs, axis=(0, 2)) / (denom[:, 0] * shape_tensor.shape[2])
    mean_shape = np.sum(shape_tensor * present[:, :, None], axis=0) / denom
    shape_range = np.max(mean_shape, axis=1) - np.min(mean_shape, axis=1)
    shape_iqr = np.percentile(mean_shape, 75, axis=1) - np.percentile(
        mean_shape, 25, axis=1
    )
    flatness_ratio = shape_iqr / (shape_range + 1e-8)
    lo = np.percentile(mean_shape, 10, axis=1)
    hi = np.percentile(mean_shape, 90, axis=1)
    tail_div = np.maximum(hi - lo, 0.0)
    if binary_mask is not None and len(binary_mask) == n_features:
        tail_div = tail_div.copy()
        tail_div[np.asarray(binary_mask, dtype=bool)] = np.maximum(
            shape_range[np.asarray(binary_mask, dtype=bool)], 0.0
        )

    shape_corr = np.ones(n_features, dtype=np.float32)
    pair_sum = np.zeros(n_features, dtype=np.float32)
    pair_count = np.zeros(n_features, dtype=np.float32)
    rank_tensor = np.empty_like(shape_tensor, dtype=np.float64)
    rank_std = np.empty((n_models, n_features), dtype=np.float64)
    for a in range(n_models):
        ranks = np.apply_along_axis(rankdata, 1, shape_tensor[a].astype(np.float64))
        ranks -= np.mean(ranks, axis=1, keepdims=True)
        rank_tensor[a] = ranks
        rank_std[a] = np.sqrt(np.mean(ranks * ranks, axis=1))
    for a in range(n_models):
        xa_rank = rank_tensor[a]
        for b in range(a + 1, n_models):
            mask = present[a] & present[b]
            if not np.any(mask):
                continue
            xb_rank = rank_tensor[b]
            denom_corr = np.maximum(rank_std[a] * rank_std[b], 1e-12)
            corr = np.mean(xa_rank * xb_rank, axis=1) / denom_corr
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
            pair_sum[mask] += np.maximum(corr[mask], 0.0).astype(np.float32)
            pair_count[mask] += 1.0
    multi = pair_count > 0
    shape_corr[multi] = pair_sum[multi] / pair_count[multi]
    mean_bends = np.sum(bends_tensor * present, axis=0) / denom[:, 0]
    mas_threshold = float(np.percentile(mas[valid], 75)) if np.any(valid) else 0.0
    score_multiplier = np.ones(n_features, dtype=np.float32)
    delete_mask = np.zeros(n_features, dtype=bool)
    pure_wiggle_deletes = 0
    smooth_penalties = 0
    spline_adjustments = 0
    for fi, name in enumerate(feature_names):
        if not valid[fi]:
            continue
        folds = [
            shape_tensor[mi, fi].copy() for mi in range(n_models) if present[mi, fi]
        ]
        audit = audit_feature_shape(
            x_bins=np.linspace(0.0, 1.0, shape_tensor.shape[2]),
            contrib=mean_shape[fi],
            fold_contribs=folds,
            top_quartile=bool(mas[fi] >= mas_threshold),
            economic_monotone_prior=False,
        )
        mean_bends[fi] = float(audit.bend_count)
        if audit.shape_type == "pure_wiggle":
            action = pure_wiggle_action(
                bend_count=audit.bend_count,
                fold_stability=float(shape_corr[fi]),
                oos_j_score=1.0,
                raw_j_score=1.0,
                is_tree_leaf=("leaf" in name and name.endswith("_soft")),
            )
            if action == "delete":
                delete_mask[fi] = True
                pure_wiggle_deletes += 1
            else:
                for mi in range(n_models):
                    if present[mi, fi]:
                        shape_tensor[mi, fi] = _smooth_shape(
                            shape_tensor[mi, fi], window=7
                        )
                score_multiplier[fi] *= max(0.0, 1.0 - audit.penalty)
                smooth_penalties += 1
                spline_adjustments += 1
        else:
            if audit.shape_type == "monotonic_noise":
                for mi in range(n_models):
                    if present[mi, fi]:
                        shape_tensor[mi, fi] = _smooth_shape(
                            shape_tensor[mi, fi], window=3
                        )
                spline_adjustments += 1
            score_multiplier[fi] *= max(0.0, 1.0 - audit.penalty)

    masked_abs = np.abs(shape_tensor) * present[:, :, None]
    mas = np.sum(masked_abs, axis=(0, 2)) / (denom[:, 0] * shape_tensor.shape[2])
    mean_shape = np.sum(shape_tensor * present[:, :, None], axis=0) / denom
    shape_range = np.max(mean_shape, axis=1) - np.min(mean_shape, axis=1)
    shape_iqr = np.percentile(mean_shape, 75, axis=1) - np.percentile(
        mean_shape, 25, axis=1
    )
    flatness_ratio = shape_iqr / (shape_range + 1e-8)
    lo = np.percentile(mean_shape, 10, axis=1)
    hi = np.percentile(mean_shape, 90, axis=1)
    tail_div = np.maximum(hi - lo, 0.0)
    if binary_mask is not None and len(binary_mask) == n_features:
        tail_div = tail_div.copy()
        tail_div[np.asarray(binary_mask, dtype=bool)] = np.maximum(
            shape_range[np.asarray(binary_mask, dtype=bool)], 0.0
        )
    scores[valid] = (
        mas[valid]
        * shape_corr[valid]
        * flatness_ratio[valid]
        * np.sqrt(tail_div[valid] + 1e-9)
        * score_multiplier[valid]
    )
    scores[delete_mask] = 0.0

    if pure_wiggle_deletes or smooth_penalties or spline_adjustments:
        tprint(
            "EBMOnLGBM shape audit: "
            f"pure_wiggle_deleted={pure_wiggle_deletes}, "
            f"pure_wiggle_smoothed={smooth_penalties}, "
            f"spline_adjusted={spline_adjustments}."
        )
    is_cont = np.array(
        [
            (binary_mask[i] == False) if binary_mask is not None else True
            for i in range(n_features)
        ]
    )

    return (
        np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        mean_bends,
        is_cont,
        mas,
    )


def _feature_shape_scores(
    models: list[Any], feature_names: list[str], binary_mask: np.ndarray | None = None
) -> np.ndarray:
    scores, _bends, _is_cont, _mas = _feature_shape_score_components(
        models, feature_names, binary_mask=binary_mask
    )
    return scores


def _post_hpo_manage_features(
    cls: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray,
    feature_names: list[str],
    spec: dict[str, Any],
    mode: str,
) -> tuple[list[str], dict[str, int], dict[str, float]]:
    """Audit post-HPO shapes without changing the selected feature contract.

    Feature generation and EBM pruning happen once in the candidate race. HPO may
    change model parameters and this audit may smooth unstable learned shapes,
    but it must not remove or replace selected raw/LGBM-derived features.
    """
    if len(feature_names) <= EBM_MIN_FEATURES:
        return (
            feature_names,
            {},
            {
                "post_hpo_shape_dropped": 0.0,
                "post_hpo_shape_smoothed": 0.0,
                "post_hpo_shape_features": float(len(feature_names)),
            },
        )
    tprint(
        "EBMOnLGBM: post-HPO shape management audit "
        f"on {len(feature_names)} features."
    )
    try:
        X_audit = X[feature_names].reset_index(drop=True)
        audit_spec = dict(spec)
        audit_spec["outer_bags"] = 1
        audit_spec["n_jobs"] = 1
        audit_model = _fit_one_ebm(cls, X_audit, y, sample_weight, audit_spec)
        binary_mask = _binary_feature_mask(X_audit)
        scores, _bends, _is_cont, mas = _feature_shape_score_components(
            [audit_model],
            feature_names,
            binary_mask=binary_mask,
        )
        shape_map = _shape_map(audit_model, feature_names)
        mas_valid = np.isfinite(mas) & (mas > 0.0)
        mas_threshold = (
            float(np.percentile(mas[mas_valid], 75)) if np.any(mas_valid) else 0.0
        )
        smooth_policy: dict[str, int] = {}
        for i, name in enumerate(feature_names):
            shape = shape_map.get(name)
            if shape is None or len(shape) <= 1:
                continue
            audit = audit_feature_shape(
                x_bins=np.linspace(0.0, 1.0, len(shape)),
                contrib=np.asarray(shape, dtype=np.float64),
                spearman_corr=None,
                top_quartile=bool(mas[i] >= mas_threshold),
                economic_monotone_prior=False,
            )
            if audit.shape_type == "pure_wiggle":
                smooth_policy[name] = 7
            elif audit.shape_type == "monotonic_noise":
                smooth_policy[name] = 3
            scores[i] *= max(0.0, 1.0 - audit.penalty)

        tprint(
            "EBMOnLGBM: post-HPO shape management locked "
            f"{len(feature_names)}/{len(feature_names)} selected features, "
            "dropped=0, "
            f"smoothed={len(smooth_policy)}."
        )
        return (
            feature_names,
            smooth_policy,
            {
                "post_hpo_shape_dropped": 0.0,
                "post_hpo_shape_smoothed": float(len(smooth_policy)),
                "post_hpo_shape_features": float(len(feature_names)),
            },
        )
    except Exception as exc:
        tprint(f"EBMOnLGBM: post-HPO shape management skipped ({exc}).")
        return (
            feature_names,
            {},
            {
                "post_hpo_shape_dropped": 0.0,
                "post_hpo_shape_smoothed": 0.0,
                "post_hpo_shape_features": float(len(feature_names)),
            },
        )


def _final_stage_oof_predictions(
    cls: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray,
    fit_idx: np.ndarray,
    spec: dict[str, Any],
    shape_smoothing_policy: dict[str, int],
    mode: str,
    random_state: int,
    use_spec_outer_bags: bool = False,
) -> np.ndarray:
    n = len(y)
    out = np.full(n, np.nan, dtype=np.float32)
    fit_idx = np.asarray(fit_idx, dtype=np.int32)
    fit_idx = fit_idx[(fit_idx >= 0) & (fit_idx < n)]
    if len(fit_idx) < max(20, EBM_CV_SPLITS * 5):
        return out
    classifier = mode == "classifier"
    y_fit_split = (
        np.asarray(y[fit_idx] >= 0.5, dtype=np.int8)
        if classifier
        else np.asarray(y[fit_idx], dtype=np.float32)
    )
    splitter: Any = (
        StratifiedKFold(n_splits=EBM_CV_SPLITS, shuffle=True, random_state=random_state)
        if classifier and len(np.unique(y_fit_split)) > 1
        else KFold(n_splits=EBM_CV_SPLITS, shuffle=True, random_state=random_state)
    )
    heldout_idx = np.setdiff1d(
        np.arange(n, dtype=np.int32), fit_idx, assume_unique=False
    )
    heldout_preds: list[np.ndarray] = []
    feature_names = list(X.columns)
    for fold_i, (tr_local, va_local) in enumerate(
        splitter.split(np.zeros(len(fit_idx)), y_fit_split), start=1
    ):
        tr = fit_idx[tr_local]
        va = fit_idx[va_local]
        fold_spec = dict(spec)
        if use_spec_outer_bags:
            fold_spec["outer_bags"] = max(1, int(fold_spec.get("outer_bags", 1)))
        else:
            fold_spec["outer_bags"] = min(
                int(fold_spec.get("outer_bags", 1)),
                max(1, int(EBM_FINAL_STAGE_OOF_OUTER_BAGS)),
            )
        t0 = time.perf_counter()
        model = _fit_one_ebm(
            cls,
            X.iloc[tr].reset_index(drop=True),
            y[tr],
            sample_weight[tr],
            fold_spec,
        )
        _apply_shape_smoothing_policy(
            model, feature_names, shape_smoothing_policy, log=False
        )
        raw_tr = _predict_raw_ebm(model, X.iloc[tr].reset_index(drop=True), mode)
        pp = SplinePostProcessor(mode).fit(raw_tr, y[tr], use_dynamic_smoothing=True)
        raw_va = _predict_raw_ebm(model, X.iloc[va].reset_index(drop=True), mode)
        out[va] = pp.predict(raw_va)
        if len(heldout_idx) > 0:
            raw_heldout = _predict_raw_ebm(
                model, X.iloc[heldout_idx].reset_index(drop=True), mode
            )
            heldout_preds.append(pp.predict(raw_heldout))
        tprint(
            "EBMOnLGBM: final-stage OOF fold "
            f"{fold_i}/{EBM_CV_SPLITS} fit in {time.perf_counter() - t0:.1f}s."
        )
    if heldout_preds and len(heldout_idx) > 0:
        out[heldout_idx] = np.mean(np.vstack(heldout_preds), axis=0).astype(np.float32)
    fill = 0.5 if classifier else 0.0
    out = np.nan_to_num(out, nan=fill, posinf=fill, neginf=fill).astype(np.float32)
    tprint(
        "EBMOnLGBM: final-stage OOF predictions generated "
        f"for {int(np.sum(np.isfinite(out)))}/{n} rows "
        f"(fit_oof_rows={len(fit_idx)}, heldout_rows={len(heldout_idx)})."
    )
    return out


def _oof_distilled_sample_weights(
    cls: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    base_weight: np.ndarray,
    fit_idx: np.ndarray,
    spec: dict[str, Any],
    shape_smoothing_policy: dict[str, int],
    mode: str,
    random_state: int,
    passes: int = 2,
    label: str = "distill",
) -> tuple[np.ndarray, np.ndarray]:
    classifier = mode == "classifier"
    base = np.nan_to_num(
        np.asarray(base_weight, dtype=np.float32), nan=1.0, posinf=1.0, neginf=1.0
    )
    base = base / max(float(np.mean(base)), 1e-6)
    current = base.copy()
    prev_oof: np.ndarray | None = None
    last_oof = np.full(
        len(y), 0.5 if classifier else float(np.nanmean(y)), dtype=np.float32
    )
    for pass_i in range(1, max(1, int(passes)) + 1):
        t0 = time.perf_counter()
        last_oof = _final_stage_oof_predictions(
            cls,
            X,
            y,
            current,
            fit_idx,
            spec,
            shape_smoothing_policy,
            mode,
            random_state=random_state + pass_i * 7919,
        )
        multiplier = _compute_weight_distillation(
            y,
            last_oof,
            prev_oof if prev_oof is not None else last_oof,
            is_classifier=classifier,
            include_false_positive_focus=False,
        )
        fp_avoid = _false_positive_avoidance_weight(
            y,
            last_oof,
            classifier=classifier,
            trade_top_frac=_target_top_fraction(),
        )
        current, ess = _normalize_rank_based_weights(base * multiplier * fp_avoid)
        prev_oof = last_oof.copy()
        tprint(
            "EBMOnLGBM: OOF-only distilled weights "
            f"{label} pass {pass_i}/{max(1, int(passes))} "
            f"in {time.perf_counter() - t0:.1f}s "
            f"(mean={float(np.mean(current)):.3f}, "
            f"p90={float(np.percentile(current, 90)):.3f}, ess={ess:.1f})."
        )
    return current.astype(np.float32), last_oof.astype(np.float32)


def _false_positive_avoidance_weight(
    y_true: np.ndarray,
    pred: np.ndarray,
    classifier: bool,
    threshold: float = 0.80,
    trade_top_frac: float | None = None,
    positive_recall_top_frac_multiplier: float = 1.5,
    fp_upweight: float = 1.60,
    top_positive_upweight: float = 1.25,
    max_weight: float = 4.0,
) -> np.ndarray:
    if not classifier:
        return np.ones(len(pred), dtype=np.float32)
    yb = np.asarray(y_true, dtype=np.float32)
    pp = np.nan_to_num(np.asarray(pred, dtype=np.float32), nan=-np.inf)
    if len(pp) == 0:
        return np.ones(0, dtype=np.float32)
    if trade_top_frac is None:
        trade_top_frac = 1.0 - float(threshold)
    trade_top_frac = float(np.clip(trade_top_frac, 1e-3, 0.95))
    pos_top_frac = float(
        np.clip(
            positive_recall_top_frac_multiplier * trade_top_frac,
            trade_top_frac,
            0.95,
        )
    )
    rank_pct = pd.Series(pp).rank(method="average", pct=True).to_numpy(dtype=np.float32)
    trade_top_mask = rank_pct >= 1.0 - trade_top_frac
    pos_support_mask = rank_pct >= 1.0 - pos_top_frac
    w = np.ones(len(pp), dtype=np.float32)
    fp_mask = (yb < 0.5) & trade_top_mask
    pos_mask = (yb >= 0.5) & pos_support_mask
    w[fp_mask] = fp_upweight
    w[pos_mask] = np.maximum(w[pos_mask], top_positive_upweight)
    return np.clip(w, 0.25, max_weight).astype(np.float32)


def _normalize_rank_based_weights(
    weights: np.ndarray,
    *,
    min_weight: float = 0.25,
    max_weight: float = 4.0,
) -> tuple[np.ndarray, float]:
    w = np.nan_to_num(
        np.asarray(weights, dtype=np.float32),
        nan=1.0,
        posinf=max_weight,
        neginf=min_weight,
    )
    w = np.clip(w, min_weight, max_weight)
    w = w / max(float(np.mean(w)), 1e-6)
    ess = float((w.sum() ** 2) / max(np.sum(w**2), 1e-6))
    return w.astype(np.float32), ess


def _apply_shape_smoothing_policy(
    model: Any,
    feature_names: list[str],
    smooth_policy: dict[str, int],
    *,
    log: bool = False,
) -> None:
    if not smooth_policy:
        return
    scores = getattr(model, "term_scores_", None)
    term_features = getattr(model, "term_features_", None)
    if scores is None:
        return
    if term_features is None:
        term_features = [(i,) for i in range(min(len(scores), len(feature_names)))]
    changed = 0
    for ti, term in enumerate(term_features):
        if ti >= len(scores) or len(term) != 1:
            continue
        idx = int(term[0])
        if not (0 <= idx < len(feature_names)):
            continue
        window = int(smooth_policy.get(feature_names[idx], 0))
        if window <= 1:
            continue
        arr = np.asarray(scores[ti], dtype=np.float64)
        if arr.ndim == 0 or arr.size <= 2:
            continue
        original_mean = float(np.nanmean(arr))
        smoothed = _smooth_shape(arr.ravel(), window=window).reshape(arr.shape)
        smoothed += original_mean - float(np.nanmean(smoothed))
        try:
            scores[ti][...] = smoothed.astype(np.asarray(scores[ti]).dtype, copy=False)
            changed += 1
        except Exception:
            scores[ti] = smoothed.astype(np.float64, copy=False)
            changed += 1
    if changed and log:
        tprint(
            "EBMOnLGBM: applied saved post-HPO shape smoothing policy "
            f"to {changed} terms."
        )


def _contribution_correlation_prune(
    cls: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    X_val: pd.DataFrame,
    selected_features: list[str],
    mode: str,
    random_state: int,
    threshold: float = 0.95,
) -> list[str]:
    if len(selected_features) <= EBM_MIN_FEATURES:
        return selected_features
    X_tr = X_train[selected_features].reset_index(drop=True)
    X_va = X_val[selected_features].reset_index(drop=True)
    spec = _ebm_specs(pruning=True, random_state=random_state)[0]
    spec["outer_bags"] = 1
    spec["min_samples_leaf_pct"] = 0.02
    tprint(
        "EBMOnLGBM: contribution-correlation pruning "
        f"on {len(selected_features)} selected features (threshold={threshold:.2f})."
    )
    try:
        model = _fit_one_ebm(cls, X_tr, y_train, sample_weight, spec)
        contrib = getattr(model, "eval_terms", None)
        if contrib is None:
            tprint("EBMOnLGBM: eval_terms unavailable; skipping contribution prune.")
            return selected_features
        terms = np.asarray(contrib(_coerce_ebm_feature_types(X_va)), dtype=np.float32)
        term_features = getattr(model, "term_features_", None)
        if term_features is None:
            term_features = [
                (i,) for i in range(min(terms.shape[1], len(selected_features)))
            ]
        cols: list[np.ndarray] = []
        names: list[str] = []
        for ti, term in enumerate(term_features):
            if ti >= terms.shape[1] or len(term) != 1:
                continue
            idx = int(term[0])
            if 0 <= idx < len(selected_features):
                vals = terms[:, ti]
                if np.isfinite(vals).any() and float(np.nanstd(vals)) > 1e-12:
                    cols.append(np.nan_to_num(vals, nan=0.0).astype(np.float32))
                    names.append(selected_features[idx])
        if len(cols) <= 1:
            return selected_features
        contrib_mat = np.column_stack(cols).astype(np.float32)
        ranks = pd.DataFrame(contrib_mat).rank(pct=True).to_numpy(dtype=np.float32)
        corr = np.abs(np.corrcoef(ranks, rowvar=False))
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        _scores, _bends, _is_cont, mas = _feature_shape_score_components(
            [model], selected_features, binary_mask=_binary_feature_mask(X_tr)
        )
        mas_map = {name: float(mas[i]) for i, name in enumerate(selected_features)}
        drop: set[str] = set()
        for i in range(len(names)):
            if names[i] in drop:
                continue
            for j in range(i + 1, len(names)):
                if names[j] in drop or corr[i, j] <= threshold:
                    continue
                loser = (
                    names[i]
                    if mas_map.get(names[i], 0.0) < mas_map.get(names[j], 0.0)
                    else names[j]
                )
                drop.add(loser)
        if not drop:
            tprint("EBMOnLGBM: contribution-correlation pruning dropped 0 features.")
            return selected_features
        pruned = [f for f in selected_features if f not in drop]
        if len(pruned) < EBM_MIN_FEATURES:
            ordered = sorted(selected_features, key=lambda f: mas_map.get(f, 0.0))
            protected = set(ordered[-EBM_MIN_FEATURES:])
            pruned = [f for f in selected_features if f not in drop or f in protected]
        tprint(
            "EBMOnLGBM: contribution-correlation pruning dropped "
            f"{len(selected_features) - len(pruned)} features; kept {len(pruned)}."
        )
        return pruned
    except Exception as exc:
        tprint(f"EBMOnLGBM: contribution-correlation pruning skipped: {exc}")
        return selected_features


def _history_float(
    row: dict[str, Any], keys: tuple[str, ...], default: float = 0.0
) -> float:
    """Read the first finite float-like value from a history row."""
    for k in keys:
        if k not in row:
            continue
        try:
            v = float(row[k])
        except Exception:
            continue
        if np.isfinite(v):
            return v
    return float(default)


def _history_fold_values(row: dict[str, Any], keys: tuple[str, ...]) -> list[float]:
    """
    Extract fold-level values from a history row.

    Prefer explicit fold-J containers over fold metric dictionaries so records
    that store both do not double-count the same folds.
    """
    for container_key in ("fold_J", "J_folds", "fold_j", "j_folds"):
        raw = row.get(container_key)
        if raw is None:
            continue
        try:
            arr = np.asarray(raw, dtype=float).reshape(-1)
        except Exception:
            continue
        vals = [float(v) for v in arr if np.isfinite(v)]
        if vals:
            return vals

    vals: list[float] = []
    fold_metrics = row.get("fold_metrics")
    if isinstance(fold_metrics, list):
        for fm in fold_metrics:
            if not isinstance(fm, dict):
                continue
            for k in keys:
                if k not in fm:
                    continue
                try:
                    v = float(fm[k])
                except Exception:
                    continue
                if np.isfinite(v):
                    vals.append(v)
                    break

    return vals


def _robust_round_score_from_history(
    row: dict[str, Any], iqr_penalty: float = EBM_J_ROBUST_IQR_PENALTY
) -> tuple[float, float, float]:
    """
    Return median_j, iqr_j, robust_j for a pruning/history row.

    If fold-level J values exist, use:
        robust_j = median(fold_J) - iqr_penalty * IQR(fold_J)

    If fold-level values do not exist, fall back to:
        robust_j = J_final - J_se
    """
    fold_js = _history_fold_values(
        row,
        keys=("J20", "j20", "J", "j", "J_final", "j_final", "score", "objective"),
    )

    if len(fold_js) >= 2:
        arr = np.asarray(fold_js, dtype=float)
        q25, q50, q75 = np.percentile(arr, [25, 50, 75])
        iqr = float(q75 - q25)
        median = float(q50)
        robust = float(median - float(iqr_penalty) * iqr)
        return median, iqr, robust

    j_final = _history_float(
        row, ("J_final", "j_final", "J", "j", "score", "objective"), 0.0
    )
    j_se = _history_float(row, ("J_se", "j_se", "se", "std", "J_std", "j_std"), 0.0)
    return float(j_final), float(j_se), float(j_final - j_se)


def _top20_round_quality(row: dict[str, Any]) -> float:
    """Top20-focused SALG/NDCG quality score used as a tie-breaker."""
    salg20 = _round_salg20(row)
    ndcg20 = _round_ndcg20(row)
    if not np.isfinite(salg20):
        salg20 = 0.0
    if not np.isfinite(ndcg20):
        ndcg20 = 0.0
    return float(EBM_J_SALG_WEIGHT * salg20 + EBM_J_NDCG_WEIGHT * ndcg20)


def _feature_bucket(n: int, bucket_size: int = 5) -> int:
    """Bucket feature counts so tiny differences do not dominate selection."""
    return int(np.ceil(int(n) / float(bucket_size)) * bucket_size)


def _feature_efficiency_score(
    robust_j: float, n_features: int, sparsity_lambda: float = 0.003
) -> float:
    """
    Mild feature-efficiency score.

    This is only a late tie-breaker, not the primary selector.
    """
    return float(robust_j - sparsity_lambda * np.log1p(max(int(n_features), 1)))


def _feature_count_in_target_range(n_features: int) -> bool:
    n = int(n_features)
    return EBM_SELECTED_FEATURES_MIN <= n <= EBM_SELECTED_FEATURES_MAX


def _dynamic_round_tolerance(best_robust: float) -> float:
    """
    Dynamic tolerance around the best robust round score.

    Wider than the old very-tight one-SE behavior so smaller acceptable rounds
    can win when performance is close.
    """
    return float(min(0.05, max(0.01, 0.05 * abs(float(best_robust)))))


def _pareto_rounds(rounds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Return pruning rounds that are not dominated on:
      - higher robust_j is better
      - lower n_features_end is better

    A round is dominated if another round has:
      - robust_j >= this robust_j
      - n_features_end <= this n_features_end
      - and is strictly better on at least one of those two dimensions
    """
    front: list[dict[str, Any]] = []

    for r in rounds:
        r_j = float(r["_robust_j"])
        r_n = int(r["_n_features_end"])

        dominated = False
        for q in rounds:
            if q is r:
                continue

            q_j = float(q["_robust_j"])
            q_n = int(q["_n_features_end"])

            if q_j >= r_j and q_n <= r_n and (q_j > r_j or q_n < r_n):
                dominated = True
                break

        if not dominated:
            front.append(r)

    return front


def _round_lift20(row: dict[str, Any], default: float = 1.0) -> float:
    return _history_float(
        row, ("lift20", "Lift20", "lift_top20", "Lift@20"), float(default)
    )


def _round_stability20(row: dict[str, Any], default: float = 0.0) -> float:
    weekly_std = _history_float(
        row,
        (
            "stability20_weekly_std",
            "Stability20_weekly_std",
            "lift20_weekly_std",
            "Lift20_weekly_std",
        ),
        float("nan"),
    )
    monthly_std = _history_float(
        row,
        (
            "stability20_monthly_std",
            "Stability20_monthly_std",
            "lift20_monthly_std",
            "Lift20_monthly_std",
        ),
        float("nan"),
    )
    if np.isfinite(weekly_std):
        if not np.isfinite(monthly_std):
            monthly_std = weekly_std
        std_blend = 0.7 * float(weekly_std) + 0.3 * float(monthly_std)
        return float(1.0 / (1.0 + max(std_blend, 0.0)))

    return _history_float(
        row,
        ("stability20", "Stability20", "stability_top20", "Stability@20"),
        float(default),
    )


def _round_ndcg20(row: dict[str, Any], default: float = 0.0) -> float:
    return _history_float(
        row,
        (
            "ndcg20",
            "ndcg_at_20",
            "ndcg@20",
            "NDCG20",
            "NDCG@20",
            "ndcg_top20",
        ),
        float(default),
    )


def _round_salg20(row: dict[str, Any]) -> float:
    return _metric_salg(_round_lift20(row), _round_stability20(row))


def _round_lift_stab_adjusted(row: dict[str, Any]) -> float:
    """Backward-compatible alias for the SALG top-20 metric."""
    return _round_salg20(row)


def _salg_acceptance_threshold(best_salg: float) -> float:
    if not np.isfinite(best_salg):
        return float("nan")
    if best_salg > 0.0:
        return float(0.75 * best_salg)
    return float(best_salg - EBM_TOP20_SALG_TOL)


def _reference_quality_key(row: dict[str, Any]) -> float:
    for key in ("_robust_j", "J_robust", "J_final"):
        try:
            value = float(row.get(key, float("nan")))
        except Exception:
            value = float("nan")
        if np.isfinite(value):
            return value
    return _top20_round_quality(row)


def _has_finite_top20_metrics(row: dict[str, Any]) -> bool:
    return bool(
        np.isfinite(_round_lift20(row, default=float("nan")))
        and np.isfinite(_round_stability20(row, default=float("nan")))
        and np.isfinite(_round_ndcg20(row, default=float("nan")))
    )


def _top20_acceptance_thresholds(
    reference_models: list[dict[str, Any]],
) -> dict[str, float]:
    reference_models = [r for r in reference_models if _has_finite_top20_metrics(r)]
    if not reference_models:
        return {
            "min_lift20": float("nan"),
            "min_stability20": float("nan"),
            "min_salg20": float("nan"),
            "min_lift_stab_adjusted20": float("nan"),
            "min_ndcg20": float("nan"),
            "best_lift20": float("nan"),
            "best_stability20": float("nan"),
            "best_salg20": float("nan"),
            "best_lift_stab_adjusted20": float("nan"),
            "best_ndcg20": float("nan"),
        }

    ref = max(reference_models, key=_reference_quality_key)
    best_lift = _round_lift20(ref)
    best_stability = _round_stability20(ref)
    best_salg = _round_salg20(ref)
    best_ndcg = _round_ndcg20(ref)
    min_salg = _salg_acceptance_threshold(best_salg)
    return {
        "min_lift20": float(max(1.0, 1.0 + 0.66 * (best_lift - 1.0))),
        "min_stability20": 0.0,
        "min_salg20": float(min_salg),
        "min_lift_stab_adjusted20": float(min_salg),
        "min_ndcg20": float(max(0.0, best_ndcg - 0.10)),
        "best_lift20": float(best_lift),
        "best_stability20": float(best_stability),
        "best_salg20": float(best_salg),
        "best_lift_stab_adjusted20": float(best_salg),
        "best_ndcg20": float(best_ndcg),
    }


def _passes_top20_acceptance(row: dict[str, Any], thresholds: dict[str, float]) -> bool:
    min_lift = float(thresholds.get("min_lift20", float("nan")))
    min_stability = float(thresholds.get("min_stability20", float("nan")))
    min_salg = float(
        thresholds.get(
            "min_salg20", thresholds.get("min_lift_stab_adjusted20", float("nan"))
        )
    )
    min_ndcg = float(thresholds.get("min_ndcg20", float("nan")))

    if not (
        np.isfinite(min_lift)
        and np.isfinite(min_stability)
        and np.isfinite(min_salg)
        and np.isfinite(min_ndcg)
    ):
        return False

    lift20 = _round_lift20(row, default=float("nan"))
    stability20 = _round_stability20(row, default=float("nan"))
    salg20 = _metric_salg(lift20, stability20)
    ndcg20 = _round_ndcg20(row, default=float("nan"))

    if not (
        np.isfinite(lift20)
        and np.isfinite(stability20)
        and np.isfinite(salg20)
        and np.isfinite(ndcg20)
    ):
        return False

    return bool(
        lift20 >= min_lift
        and stability20 >= min_stability
        and salg20 >= min_salg
        and ndcg20 >= min_ndcg
    )


def _top20_metric_acceptable_models(
    models: list[dict[str, Any]],
    reference_models: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    thresholds = _top20_acceptance_thresholds(reference_models)
    return [r for r in models if _passes_top20_acceptance(r, thresholds)]


def _round_robust_value(row: dict[str, Any]) -> float:
    try:
        return float(
            row.get("_robust_j", row.get("J_robust", row.get("J_final", np.nan)))
        )
    except Exception:
        return float("nan")


def _top20_reference_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "J_final": float(row.get("J_final", np.nan)),
        "J_robust": float(row.get("J_robust", row.get("J_final", np.nan))),
        "lift20": _round_lift20(row, default=float("nan")),
        "stability20": _round_stability20(row, default=float("nan")),
        "salg20": _round_salg20(row),
        "ndcg20": _round_ndcg20(row, default=float("nan")),
        "round": int(row.get("round", -1)) if "round" in row else -1,
    }


def _finite_fold_j_values(fold_metrics: list[Any]) -> list[float]:
    fold_j_values: list[float] = []
    for m in fold_metrics:
        if not isinstance(m, dict):
            continue
        v = float(_fold_j(m))
        if np.isfinite(v):
            fold_j_values.append(v)
    return fold_j_values


def _top20_metric_acceptable_rounds(
    rounds: list[dict[str, Any]],
    best_robust: float,
    best_tolerance: float,
    reference_models: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not rounds:
        return []
    thresholds = _top20_acceptance_thresholds(
        rounds if reference_models is None else reference_models
    )
    acceptable: list[dict[str, Any]] = []
    for r in rounds:
        robust_value = _round_robust_value(r)
        if (
            np.isfinite(robust_value)
            and robust_value >= best_robust - best_tolerance
            and _passes_top20_acceptance(r, thresholds)
        ):
            acceptable.append(r)
    return acceptable


def _select_pareto_knee_round(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Legacy helper retained for focused geometry tests; not used for pruning selection."""
    if not candidates:
        raise ValueError("Cannot select knee from empty candidate list.")

    candidates = sorted(
        candidates,
        key=lambda r: (
            int(r["_n_features_end"]),
            float(r["_robust_j"]),
        ),
    )

    if len(candidates) <= 2:
        return min(
            candidates,
            key=lambda r: (
                _feature_bucket(int(r["_n_features_end"])),
                -float(r["_robust_j"]),
            ),
        )

    n = np.asarray([int(r["_n_features_end"]) for r in candidates], dtype=float)
    j = np.asarray([float(r["_robust_j"]) for r in candidates], dtype=float)

    n_span = max(float(np.max(n) - np.min(n)), 1e-12)
    j_span = max(float(np.max(j) - np.min(j)), 1e-12)

    # Normalize so feature count and score are comparable.
    # x: 0 = fewest features, 1 = most features
    # y: 0 = worst robust score, 1 = best robust score
    x = (n - np.min(n)) / n_span
    y = (j - np.min(j)) / j_span

    start_idx = int(np.argmin(n))
    end_idx = int(np.argmax(j))

    if end_idx == start_idx:
        return candidates[start_idx]

    x0, y0 = float(x[start_idx]), float(y[start_idx])
    x1, y1 = float(x[end_idx]), float(y[end_idx])

    dx = x1 - x0
    dy = y1 - y0
    denom = max(float(np.sqrt(dx * dx + dy * dy)), 1e-12)

    distances = (dx * (y - y0) - dy * (x - x0)) / denom
    in_span = (x >= min(x0, x1)) & (x <= max(x0, x1))
    distances = np.where(in_span, distances, -np.inf)

    feature_penalty = 0.02 * x
    knee_score = distances - feature_penalty

    return candidates[int(np.argmax(knee_score))]


def _select_robust_pruning_round(
    history: list[dict[str, Any]],
    *,
    iqr_penalty: float = EBM_J_ROBUST_IQR_PENALTY,
    max_iqr_multiplier: float = 1.50,
) -> dict[str, Any]:
    """
    Select an EBM pruning round with hard quality/efficiency gates.

    Policy:
      1. Use the highest-robust-J candidate as the reference model.
      2. Apply top-20 hard gates for Lift, SALG, and NDCG; prefer candidates
         in the configured 100-350 feature range, but if fewer than five pass
         that range gate keep all quality-passing candidates.
      3. Require robust J to be at least 80% of the reference robust J.
      4. Require feature-value efficiency gates:
            robust_j / n_features >= median
            robust_j / sqrt(n_features) >= q33
      5. Keep the Pareto front on higher robust J and lower feature count.
      6. Choose the smallest Pareto candidate within max(J_se, dynamic_tol)
         of the best robust J, unless its top-20 quality is materially worse.
    """
    valid = [
        h
        for h in history
        if ("active_indices" not in h or h.get("active_indices") is not None)
        and bool(h.get("candidate_gate_passed", True))
    ]
    if not valid:
        tprint("EBMOnLGBM: no pruning candidate passed candidate-level gates.")
        return {}

    enriched: list[dict[str, Any]] = []
    for row in valid:
        r = dict(row)
        median_j, iqr_j, robust_j = _robust_round_score_from_history(
            r,
            iqr_penalty=iqr_penalty,
        )
        r["_median_j"] = float(median_j)
        r["_iqr_j"] = float(iqr_j)
        r["_robust_j"] = float(robust_j)
        r["_top20_quality"] = float(_top20_round_quality(r))
        r["_n_features_end"] = int(
            _history_float(
                r,
                ("n_features_end", "n_features", "feature_count", "n_keep"),
                10**9,
            )
        )
        n_features = max(int(r["_n_features_end"]), 1)
        r["_selection_j_eff"] = 0.0
        r["_feature_value"] = 0.0
        r["_sqrt_feature_value"] = 0.0
        r["_feature_efficiency_j"] = _feature_efficiency_score(
            float(r["_robust_j"]),
            n_features,
        )
        enriched.append(r)

    hard_min_features = 50
    hard_max_features = 500
    target_max_features = int(EBM_SELECTED_FEATURES_MAX)
    # Pruning proceeds in coarse fractional drops. A round can land just above
    # the target cap (for example 504 vs 500) while still being the best honest
    # candidate. Keep near-cap rounds eligible and trim them deterministically
    # after selection instead of rejecting the whole EBM candidate.
    near_cap_max_features = int(np.ceil(float(hard_max_features) * 1.05))
    hard_count_pool = [
        r
        for r in enriched
        if hard_min_features <= int(r["_n_features_end"]) <= near_cap_max_features
    ]
    if not hard_count_pool:
        tprint(
            "EBMOnLGBM: no pruning candidate satisfied the hard "
            f"{hard_min_features}-{near_cap_max_features} feature-count gate; "
            "falling back to quality-first selection with post-selection cap."
        )
        hard_count_pool = [
            r
            for r in enriched
            if hard_min_features <= int(r["_n_features_end"])
        ]
        if not hard_count_pool:
            tprint(
                "EBMOnLGBM: no pruning candidate satisfied the minimum "
                f"{hard_min_features} feature-count gate."
            )
            return {}

    reference_pool = [r for r in hard_count_pool if _has_finite_top20_metrics(r)]
    if not reference_pool:
        tprint("EBMOnLGBM: no pruning candidate had finite top-20 metrics.")
        return {}

    reference = max(reference_pool, key=lambda r: float(r["_robust_j"]))
    best_robust = float(reference["_robust_j"])
    top20_thresholds = _top20_acceptance_thresholds(reference_pool)
    min_lift = float(top20_thresholds["min_lift20"])
    min_salg = float(top20_thresholds["min_salg20"])
    min_ndcg = float(top20_thresholds["min_ndcg20"])

    quality_pass = [
        r for r in hard_count_pool if _passes_top20_acceptance(r, top20_thresholds)
    ]
    if not quality_pass:
        tprint("EBMOnLGBM: no pruning candidate satisfied top-20 quality hard gates.")
        return {}

    target_quality_pass = [
        r
        for r in quality_pass
        if _feature_count_in_target_range(int(r["_n_features_end"]))
    ]
    used_target_range = len(target_quality_pass) > 0
    if not used_target_range:
        tprint(
            "EBMOnLGBM: no top-20 quality-passing pruning candidate is inside "
            f"the target feature range [{EBM_SELECTED_FEATURES_MIN}, "
            f"{EBM_SELECTED_FEATURES_MAX}]; falling back to 50-500 hard range."
        )
    quality_pool = target_quality_pass if used_target_range else quality_pass

    selection_tol = _dynamic_round_tolerance(best_robust)
    robust_threshold = float(best_robust - selection_tol)
    robust_pool = [r for r in quality_pool if float(r["_robust_j"]) >= robust_threshold]
    if not robust_pool and used_target_range:
        tprint(
            "EBMOnLGBM: target feature-range pool failed robust-J gate; "
            "falling back to broader top-20 quality-passing pool."
        )
        used_target_range = False
        quality_pool = quality_pass
        robust_pool = [
            r for r in quality_pool if float(r["_robust_j"]) >= robust_threshold
        ]
    if not robust_pool:
        tprint("EBMOnLGBM: no pruning candidate satisfied the robust-J hard gate.")
        return {}
    n_robust_pass_rounds = len(robust_pool)

    iqr_values = np.asarray([float(r["_iqr_j"]) for r in robust_pool], dtype=float)
    finite_iqr = iqr_values[np.isfinite(iqr_values)]
    median_iqr = float(np.median(finite_iqr)) if len(finite_iqr) else 0.0
    max_allowed_iqr = max(0.005, float(max_iqr_multiplier) * median_iqr)
    stable_pool = [
        r
        for r in robust_pool
        if np.isfinite(float(r["_iqr_j"])) and float(r["_iqr_j"]) <= max_allowed_iqr
    ]
    n_stable_rounds = len(stable_pool)
    if stable_pool:
        robust_pool = stable_pool
    else:
        n_stable_rounds = len(robust_pool)

    j_floor = robust_threshold
    for r in robust_pool:
        n_features = max(int(r["_n_features_end"]), 1)
        j_eff = max(0.0, float(r["_robust_j"]) - j_floor)
        r["_selection_j_eff"] = float(j_eff)
        r["_feature_value"] = float(j_eff / float(n_features))
        r["_sqrt_feature_value"] = float(j_eff / np.sqrt(float(n_features)))

    feature_values = np.asarray(
        [float(r["_feature_value"]) for r in robust_pool], dtype=float
    )
    sqrt_feature_values = np.asarray(
        [float(r["_sqrt_feature_value"]) for r in robust_pool], dtype=float
    )
    finite_feature_values = feature_values[np.isfinite(feature_values)]
    finite_sqrt_feature_values = sqrt_feature_values[np.isfinite(sqrt_feature_values)]
    feature_value_min = (
        float(np.median(finite_feature_values))
        if len(finite_feature_values)
        else -np.inf
    )
    sqrt_feature_value_min = (
        float(np.percentile(finite_sqrt_feature_values, 33.0))
        if len(finite_sqrt_feature_values)
        else -np.inf
    )
    efficiency_pool = [
        r
        for r in robust_pool
        if float(r["_feature_value"]) >= feature_value_min
        and float(r["_sqrt_feature_value"]) >= sqrt_feature_value_min
    ]
    if not efficiency_pool:
        tprint("EBMOnLGBM: no pruning candidate satisfied feature-efficiency gates.")
        return {}

    pareto = _pareto_rounds(efficiency_pool)
    if not pareto:
        pareto = efficiency_pool

    best_pareto = max(pareto, key=lambda r: float(r["_robust_j"]))
    best_pareto_robust = float(best_pareto["_robust_j"])
    best_pareto_j_se = _history_float(
        best_pareto,
        ("J_se", "j_se", "se", "J_std", "j_std"),
        0.0,
    )
    if not np.isfinite(best_pareto_j_se):
        best_pareto_j_se = 0.0
    pareto_robust_tol = float(
        max(float(best_pareto_j_se), _dynamic_round_tolerance(best_pareto_robust))
    )
    pareto_robust_floor = float(best_pareto_robust - pareto_robust_tol)
    robust_close_pareto = [
        r for r in pareto if float(r["_robust_j"]) >= pareto_robust_floor
    ]
    if not robust_close_pareto:
        robust_close_pareto = [best_pareto]

    best_close_top20_quality = max(
        float(r["_top20_quality"]) for r in robust_close_pareto
    )
    top20_quality_tol = float(
        max(EBM_TOP20_SALG_TOL, 0.05 * abs(best_close_top20_quality))
    )
    top20_quality_floor = float(best_close_top20_quality - top20_quality_tol)
    conservative_pareto = [
        r
        for r in robust_close_pareto
        if float(r["_top20_quality"]) >= top20_quality_floor
    ]
    if not conservative_pareto:
        conservative_pareto = robust_close_pareto

    chosen = min(
        conservative_pareto,
        key=lambda r: (
            _feature_bucket(int(r["_n_features_end"])),
            -float(r["_top20_quality"]),
            float(r["_iqr_j"]),
            -float(r["_robust_j"]),
        ),
    )

    tprint(
        "EBMOnLGBM: robust pruning selection "
        f"hard_count={len(hard_count_pool)}, quality_pass={len(quality_pass)}, "
        f"target_pass={len(target_quality_pass)}, robust_pass={n_robust_pass_rounds}, "
        f"stable_pass={n_stable_rounds}, efficiency_pass={len(efficiency_pool)}, "
        f"pareto={len(pareto)}, pareto_close={len(robust_close_pareto)}, "
        f"pareto_conservative={len(conservative_pareto)}, "
        f"used_target_range={used_target_range}; "
        f"min_lift20={min_lift:.4f}, min_salg20={min_salg:.4f}, "
        f"min_ndcg20={min_ndcg:.4f}, min_robust_j={robust_threshold:.4f}; "
        f"pareto_best_robust_j={best_pareto_robust:.4f}, "
        f"pareto_robust_tol={pareto_robust_tol:.4f}, "
        f"pareto_top20_quality_floor={top20_quality_floor:.4f}; "
        f"chosen_round={int(chosen.get('round', -1))}, "
        f"chosen_features={int(chosen['_n_features_end'])}, "
        f"chosen_robust_j={float(chosen['_robust_j']):.4f}."
    )

    out = dict(chosen)
    chosen_feature_count = int(out.get("_n_features_end", out.get("n_features_end", 0)))
    if chosen_feature_count > target_max_features:
        active_idx = np.asarray(out.get("active_indices"), dtype=np.int32)
        active_scores = np.asarray(out.get("active_scores", []), dtype=np.float32)
        if len(active_idx) == chosen_feature_count and len(active_scores) == len(
            active_idx
        ):
            keep_pos = np.argsort(active_scores)[-target_max_features:]
            capped = np.sort(active_idx[keep_pos]).astype(np.int32)
        else:
            capped = np.sort(active_idx)[:target_max_features].astype(np.int32)
        out["active_indices"] = capped
        out["_n_features_end"] = int(len(capped))
        out["n_features_end"] = int(len(capped))
        out["selection_feature_cap_applied"] = 1.0
        out["selection_feature_count_before_cap"] = float(chosen_feature_count)
        tprint(
            "EBMOnLGBM: capped selected pruning round features "
            f"{chosen_feature_count} -> {len(capped)} to satisfy "
            f"target_max_features={target_max_features}."
        )
    else:
        out["selection_feature_cap_applied"] = 0.0
    out["selection_policy"] = "quality_efficiency_pareto_close_j_min_features"
    out["selection_best_robust_j"] = float(best_robust)
    out["selection_tol"] = float(selection_tol)
    out["selection_median_j"] = float(chosen["_median_j"])
    out["selection_iqr_j"] = float(chosen["_iqr_j"])
    out["selection_robust_j"] = float(chosen["_robust_j"])
    out["selection_top20_quality"] = float(chosen["_top20_quality"])
    out["selection_n_acceptable_rounds"] = float(len(quality_pool))
    out["selection_n_quality_pool_rounds"] = float(len(quality_pool))
    out["selection_n_quality_pass_rounds"] = float(len(quality_pass))
    out["selection_n_robust_pass_rounds"] = float(n_robust_pass_rounds)
    out["selection_n_stable_rounds"] = float(n_stable_rounds)
    out["selection_n_efficiency_pass_rounds"] = float(len(efficiency_pool))
    out["selection_n_enriched_rounds"] = float(len(enriched))
    out["selection_n_pareto_rounds"] = float(len(pareto))
    out["selection_n_pareto_close_rounds"] = float(len(robust_close_pareto))
    out["selection_n_pareto_conservative_rounds"] = float(len(conservative_pareto))
    out["selection_n_target_acceptable_rounds"] = float(len(target_quality_pass))
    out["selection_target_min_features"] = float(EBM_SELECTED_FEATURES_MIN)
    out["selection_target_max_features"] = float(EBM_SELECTED_FEATURES_MAX)
    out["selection_used_target_feature_range"] = float(used_target_range)
    out["selection_feature_efficiency_j"] = float(chosen["_feature_efficiency_j"])
    out["selection_feature_value"] = float(chosen["_feature_value"])
    out["selection_sqrt_feature_value"] = float(chosen["_sqrt_feature_value"])
    out["selection_j_eff"] = float(chosen["_selection_j_eff"])
    out["selection_n_hard_count_pass_rounds"] = float(len(hard_count_pool))
    out["selection_min_lift20"] = float(min_lift)
    out["selection_min_salg20"] = float(min_salg)
    out["selection_min_ndcg20"] = float(min_ndcg)
    out["selection_min_robust_j"] = float(robust_threshold)
    out["selection_pareto_best_robust_j"] = float(best_pareto_robust)
    out["selection_pareto_best_j_se"] = float(best_pareto_j_se)
    out["selection_pareto_robust_tol"] = float(pareto_robust_tol)
    out["selection_pareto_robust_floor"] = float(pareto_robust_floor)
    out["selection_pareto_top20_quality_floor"] = float(top20_quality_floor)
    out["selection_pareto_top20_quality_tol"] = float(top20_quality_tol)
    out["selection_min_feature_value"] = float(feature_value_min)
    out["selection_min_sqrt_feature_value"] = float(sqrt_feature_value_min)
    out["selection_reference_round"] = float(reference.get("round", -1))
    return out


def _select_smallest_within_one_se(
    history: list[dict[str, Any]],
    *,
    iqr_penalty: float = EBM_J_ROBUST_IQR_PENALTY,
    max_iqr_multiplier: float = 1.50,
) -> dict[str, Any]:
    """Backward-compatible alias for the robust quality/efficiency selector."""
    return _select_robust_pruning_round(
        history,
        iqr_penalty=iqr_penalty,
        max_iqr_multiplier=max_iqr_multiplier,
    )


def _feature_pruning_candidate_gate(rec: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    min_lift20, min_stability20 = _candidate_gate_thresholds()
    lift20 = _round_lift20(rec, default=float("nan"))
    stability20 = _round_stability20(rec, default=float("nan"))
    salg20 = _round_salg20(rec)
    ndcg20 = _round_ndcg20(rec, default=float("nan"))
    j20 = _metric_j_from_lift_stability_ndcg(lift20, stability20, ndcg20)

    lift15 = float(rec.get("lift15", rec.get("lift30", np.nan)))
    stability15 = float(rec.get("stability15", rec.get("stability30", np.nan)))
    lift30 = float(rec.get("lift30", np.nan))
    stability30 = float(rec.get("stability30", np.nan))
    j_final = float(rec.get("J_final", np.nan))
    n_features = int(rec.get("n_features_end", rec.get("n_features", 0)))
    checks = {
        "finite_score": bool(np.isfinite(j_final)),
        "finite_j20": bool(np.isfinite(j20)),
        "finite_salg20": bool(np.isfinite(salg20)),
        "min_lift20": bool(np.isfinite(lift20) and lift20 >= min_lift20),
        "min_stability20": bool(
            np.isfinite(stability20) and stability20 >= min_stability20
        ),
        "finite_lift15": bool(np.isfinite(lift15)),
        "finite_stability15": bool(np.isfinite(stability15)),
        "legacy_min_lift30": bool(np.isfinite(lift30)),
        "legacy_min_stability30": bool(np.isfinite(stability30)),
        "min_features": bool(n_features >= EBM_MIN_FEATURES),
    }
    details: dict[str, Any] = {
        "candidate_gate_passed": bool(all(checks.values())),
        "candidate_gate_min_lift15": float(min_lift20),
        "candidate_gate_min_stability15": float(min_stability20),
        "candidate_gate_min_lift20": float(min_lift20),
        "candidate_gate_min_stability20": float(min_stability20),
        "candidate_gate_min_lift30": float(min_lift20),
        "candidate_gate_min_stability30": float(min_stability20),
        "candidate_gate_lift15": float(lift15),
        "candidate_gate_stability15": float(stability15),
        "candidate_gate_lift20": float(lift20),
        "candidate_gate_stability20": float(stability20),
        "candidate_gate_salg20": float(salg20),
        "candidate_gate_ndcg20": float(ndcg20),
        "candidate_gate_j20": float(j20),
        "candidate_gate_lift30": float(lift30),
        "candidate_gate_stability30": float(stability30),
    }
    details.update({f"candidate_gate_{k}": bool(v) for k, v in checks.items()})
    return bool(details["candidate_gate_passed"]), details


def _round_drop_fraction(round_id: int) -> float:
    schedule = [0.30, 0.20, 0.20, 0.20, 0.20, 0.20]
    if round_id <= len(schedule):
        return schedule[round_id - 1]
    return 0.10


def _round_presence_min(round_id: int) -> float:
    first_round_relief = 0.10 if int(round_id) <= 1 else 0.0
    threshold = float(
        np.clip(
            EBM_FOLD_PRESENCE_MIN_START
            + EBM_FOLD_PRESENCE_MIN_STEP * max(0, int(round_id) - 1),
            0.0,
            EBM_FOLD_PRESENCE_MIN_CAP,
        )
    )
    return float(max(0.0, threshold - first_round_relief))


def _round_sign_consistency_min(round_id: int) -> float:
    first_round_relief = 0.10 if int(round_id) <= 1 else 0.0
    threshold = float(
        np.clip(
            EBM_SIGN_CONSISTENCY_MIN_START
            + EBM_SIGN_CONSISTENCY_MIN_STEP * max(0, int(round_id) - 1),
            0.0,
            EBM_SIGN_CONSISTENCY_MIN_CAP,
        )
    )
    return float(max(0.0, threshold - first_round_relief))


def _term_contrib_matrix(model: Any, X: pd.DataFrame) -> np.ndarray | None:
    eval_terms = getattr(model, "eval_terms", None)
    if not callable(eval_terms):
        return None
    try:
        contrib = eval_terms(_coerce_ebm_feature_types(X))
    except Exception:
        return None
    arr = np.asarray(contrib, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != len(X):
        return None
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _contribution_redundancy_stats(
    contrib: np.ndarray,
    term_features: Any,
    finite_mask: np.ndarray,
    n_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    max_corr = np.zeros(n_features, dtype=np.float32)
    multiplier = np.ones(n_features, dtype=np.float32)
    if EBM_CONTRIB_REDUNDANCY_MAX_FEATURES <= 1:
        return max_corr, multiplier

    finite = np.asarray(finite_mask, dtype=bool)
    if contrib.ndim != 2 or contrib.shape[0] != len(finite):
        return max_corr, multiplier

    cols: list[np.ndarray] = []
    feature_indices: list[int] = []
    for ti, term in enumerate(term_features):
        if ti >= contrib.shape[1] or len(term) != 1:
            continue
        fi = int(term[0])
        if fi < 0 or fi >= n_features:
            continue
        vals = np.asarray(contrib[:, ti][finite], dtype=np.float32)
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        if vals.size < 8 or float(np.std(vals)) <= 1e-12:
            continue
        cols.append(vals)
        feature_indices.append(fi)

    n_cols = len(cols)
    if n_cols <= 1 or n_cols > EBM_CONTRIB_REDUNDANCY_MAX_FEATURES:
        return max_corr, multiplier

    mat = np.column_stack(cols).astype(np.float32, copy=False)
    if mat.shape[0] > EBM_CONTRIB_REDUNDANCY_MAX_ROWS > 0:
        idx = np.linspace(
            0,
            mat.shape[0] - 1,
            int(EBM_CONTRIB_REDUNDANCY_MAX_ROWS),
            dtype=np.int32,
        )
        mat = mat[idx]

    ranks = pd.DataFrame(mat).rank(pct=True).to_numpy(dtype=np.float32)
    ranks = np.nan_to_num(ranks, nan=0.0, posinf=0.0, neginf=0.0)
    ranks -= np.mean(ranks, axis=0, keepdims=True)
    denom = np.sqrt(np.sum(ranks * ranks, axis=0, keepdims=True))
    valid_cols = denom.ravel() > 1e-12
    if int(np.sum(valid_cols)) <= 1:
        return max_corr, multiplier

    ranks = ranks[:, valid_cols] / np.maximum(denom[:, valid_cols], 1e-12)
    valid_feature_indices = np.asarray(feature_indices, dtype=np.int32)[valid_cols]
    corr = np.abs(np.matmul(ranks.T, ranks))
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)
    local_max = np.max(corr, axis=1).astype(np.float32)

    start = float(np.clip(EBM_CONTRIB_REDUNDANCY_CORR_START, 0.0, 0.999))
    full = float(np.clip(EBM_CONTRIB_REDUNDANCY_CORR_FULL, start + 1e-6, 0.999))
    min_mult = float(np.clip(EBM_CONTRIB_REDUNDANCY_MULTIPLIER_MIN, 0.0, 1.0))
    excess = np.clip((local_max - start) / max(full - start, 1e-6), 0.0, 1.0)
    local_mult = 1.0 - (1.0 - min_mult) * excess
    max_corr[valid_feature_indices] = local_max
    multiplier[valid_feature_indices] = local_mult.astype(np.float32)
    return max_corr, multiplier


def _feature_eval_term_stats(
    model: Any,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    pred_va: np.ndarray,
    feature_names: list[str],
    classifier: bool,
    groups: Any = None,
) -> dict[str, np.ndarray] | None:
    contrib = _term_contrib_matrix(model, X_va)
    if contrib is None:
        return None
    term_features = getattr(model, "term_features_", None)
    if term_features is None:
        term_features = [(i,) for i in range(min(contrib.shape[1], len(feature_names)))]
    n_features = len(feature_names)
    presence = np.zeros(n_features, dtype=np.float32)
    sign = np.zeros(n_features, dtype=np.float32)
    top_use = np.zeros(n_features, dtype=np.float32)
    top_econ = np.zeros(n_features, dtype=np.float32)
    top_model_contrib_abs = np.zeros(n_features, dtype=np.float32)
    top_model_contrib_enrichment = np.zeros(n_features, dtype=np.float32)
    top_model_contrib_signed = np.zeros(n_features, dtype=np.float32)
    top_model_contrib_push = np.zeros(n_features, dtype=np.float32)
    top_model_contrib_push_enrichment = np.zeros(n_features, dtype=np.float32)
    top_model_fp_push_share = np.zeros(n_features, dtype=np.float32)
    top_model_fp_multiplier = np.ones(n_features, dtype=np.float32)
    asset_edge_std = np.zeros(n_features, dtype=np.float32)
    month_edge_std = np.zeros(n_features, dtype=np.float32)
    asset_stability = np.ones(n_features, dtype=np.float32)
    month_stability = np.ones(n_features, dtype=np.float32)
    contrib_redundancy_corr = np.zeros(n_features, dtype=np.float32)
    contrib_redundancy_multiplier = np.ones(n_features, dtype=np.float32)
    yy = np.asarray(y_va, dtype=np.float32)
    pp = np.asarray(pred_va, dtype=np.float32)
    finite = np.isfinite(yy) & np.isfinite(pp)
    if int(np.sum(finite)) < 8:
        return None
    grp_asset = None
    grp_month = None
    if isinstance(groups, dict):
        if "asset" in groups and len(np.asarray(groups["asset"])) == len(finite):
            grp_asset = np.asarray(groups["asset"], dtype=object)[finite]
        month_key = "calendar_month" if "calendar_month" in groups else "month"
        if month_key in groups and len(np.asarray(groups[month_key])) == len(finite):
            grp_month = np.asarray(groups[month_key], dtype=object)[finite]
    elif groups is not None and len(np.asarray(groups)) == len(finite):
        grp_asset = np.asarray(groups, dtype=object)[finite]
    yy = yy[finite]
    pp = pp[finite]
    order = np.argsort(pp)
    top20 = _top_idx(order, 0.20, len(pp))
    top30 = _top_idx(order, 0.30, len(pp))
    top_contrib_idx = _top_idx(order, EBM_TOP_CONTRIB_FRACTION, len(pp))
    if classifier:
        y_eval = (yy >= 0.5).astype(np.float32)
    else:
        y_eval = yy.astype(np.float32)
    contrib_redundancy_corr, contrib_redundancy_multiplier = (
        _contribution_redundancy_stats(contrib, term_features, finite, n_features)
    )
    for ti, term in enumerate(term_features):
        if ti >= contrib.shape[1] or len(term) != 1:
            continue
        fi = int(term[0])
        if fi < 0 or fi >= n_features:
            continue
        c = contrib[:, ti][finite].astype(np.float32)
        c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
        amp = float(np.mean(np.abs(c)))
        spread = float(np.std(c))
        if amp <= 1e-7 and spread <= 1e-7:
            continue
        presence[fi] = 1.0
        corr_total = _safe_spearman(c, y_eval)
        if abs(corr_total) <= 1e-12:
            lift_dir = float(np.mean(y_eval[c >= np.median(c)])) - float(
                np.mean(y_eval[c < np.median(c)])
            )
            corr_total = lift_dir
        direction = float(np.sign(corr_total))
        sign[fi] = direction
        score20 = (
            abs(_safe_spearman(c[top20], y_eval[top20])) if len(top20) > 2 else 0.0
        )
        score30 = (
            abs(_safe_spearman(c[top30], y_eval[top30])) if len(top30) > 2 else 0.0
        )
        top_use[fi] = float(0.75 * score20 + 0.25 * score30)
        if len(top_contrib_idx) >= 3:
            c_top = c[top_contrib_idx]
            y_top = y_eval[top_contrib_idx]
            positive_push = np.maximum(c_top, 0.0)
            top_push_sum = float(np.sum(positive_push))
            top_push = float(np.mean(positive_push))
            top_abs = float(np.mean(np.abs(c_top)))
            all_abs = float(np.mean(np.abs(c))) + 1e-8
            all_push = float(np.mean(np.maximum(c, 0.0))) + 1e-8
            top_signed = float(np.mean(c_top))
            top_model_contrib_abs[fi] = top_abs
            top_model_contrib_enrichment[fi] = float(top_abs / all_abs)
            top_model_contrib_signed[fi] = top_signed
            top_model_contrib_push[fi] = top_push
            top_model_contrib_push_enrichment[fi] = float(top_push / all_push)
            if top_push_sum > 1e-12:
                if classifier:
                    false_positive_mask = y_top < 0.5
                else:
                    false_positive_mask = y_top <= 0.0
                fp_push = float(np.sum(positive_push[false_positive_mask]))
                fp_share = float(np.clip(fp_push / top_push_sum, 0.0, 1.0))
                top_model_fp_push_share[fi] = fp_share
                fp_mult_min = float(
                    np.clip(EBM_TOP_BUCKET_FALSE_POSITIVE_MULTIPLIER_MIN, 0.0, 1.0)
                )
                top_model_fp_multiplier[fi] = float(
                    np.clip(
                        1.0 - (1.0 - fp_mult_min) * fp_share,
                        fp_mult_min,
                        1.0,
                    )
                )
        if direction == 0.0:
            direction = 1.0
        feature_rank_score = c * float(direction)
        feature_top = _top_idx(np.argsort(feature_rank_score), 0.20, len(y_eval))
        if len(feature_top) >= 3:
            if classifier:
                base_rate = float(np.mean(y_eval >= 0.5))
                top_rate = float(np.mean(y_eval[feature_top] >= 0.5))
                top_econ[fi] = float(max(0.0, top_rate / max(base_rate, 1e-6) - 1.0))
            else:
                denom = float(np.mean(np.abs(y_eval)) + 1e-6)
                top_econ[fi] = float(max(0.0, np.mean(y_eval[feature_top]) / denom))
        asset_std, asset_mult = _feature_group_edge_stability(
            c,
            y_eval,
            direction=direction,
            classifier=classifier,
            groups=grp_asset,
        )
        month_std, month_mult = _feature_group_edge_stability(
            c,
            y_eval,
            direction=direction,
            classifier=classifier,
            groups=grp_month,
        )
        asset_edge_std[fi] = asset_std
        month_edge_std[fi] = month_std
        asset_stability[fi] = asset_mult
        month_stability[fi] = month_mult
    return {
        "presence": presence,
        "sign": sign,
        "top_usefulness": top_use,
        "top_economic_usefulness": top_econ,
        "top_model_contribution_abs": top_model_contrib_abs,
        "top_model_contribution_enrichment": top_model_contrib_enrichment,
        "top_model_contribution_signed": top_model_contrib_signed,
        "top_model_contribution_push": top_model_contrib_push,
        "top_model_contribution_push_enrichment": top_model_contrib_push_enrichment,
        "top_model_false_positive_push_share": top_model_fp_push_share,
        "top_model_false_positive_multiplier": top_model_fp_multiplier,
        "asset_edge_std": asset_edge_std,
        "month_edge_std": month_edge_std,
        "asset_stability_multiplier": asset_stability,
        "month_stability_multiplier": month_stability,
        "contribution_redundancy_corr": contrib_redundancy_corr,
        "contribution_redundancy_multiplier": contrib_redundancy_multiplier,
    }


def _feature_group_edge_stability(
    contribution: np.ndarray,
    y_eval: np.ndarray,
    *,
    direction: float,
    classifier: bool,
    groups: np.ndarray | None,
    top_frac: float = 0.20,
) -> tuple[float, float]:
    if groups is None:
        return 0.0, 1.0
    c = np.asarray(contribution, dtype=np.float32)
    y = np.asarray(y_eval, dtype=np.float32)
    g = np.asarray(groups, dtype=object)
    if len(c) != len(y) or len(g) != len(y):
        return 0.0, 1.0
    score = np.nan_to_num(c * float(direction), nan=0.0, posinf=0.0, neginf=0.0)
    edges: list[float] = []
    for group in pd.unique(g):
        mask = g == group
        n = int(np.sum(mask))
        if n < 20:
            continue
        yy = y[mask]
        ss = score[mask]
        finite = np.isfinite(yy) & np.isfinite(ss)
        if int(np.sum(finite)) < 20 or float(np.nanstd(ss[finite])) <= 1e-12:
            continue
        yy = yy[finite]
        ss = ss[finite]
        order = np.argsort(ss)
        top = _top_idx(order, top_frac, len(yy))
        if len(top) < 3:
            continue
        if classifier:
            base = float(np.mean(yy >= 0.5))
            edge = float(np.mean(yy[top] >= 0.5) - base)
        else:
            edge = float(np.mean(yy[top]))
        if np.isfinite(edge):
            edges.append(edge)
    if len(edges) < 3:
        return 0.0, 1.0
    arr = np.asarray(edges, dtype=np.float32)
    std = float(np.std(arr, ddof=1))
    mean_abs = float(abs(np.mean(arr)))
    cv = std / max(mean_abs, 0.05)
    multiplier = float(np.clip(1.0 / (1.0 + cv), 0.65, 1.0))
    return std, multiplier


def _aggregate_feature_eval_stats(
    fold_stats: list[dict[str, np.ndarray]],
    n_features: int,
) -> dict[str, np.ndarray]:
    if not fold_stats:
        zeros = np.zeros(n_features, dtype=np.float32)
        return {
            "presence_rate": zeros,
            "sign_consistency": zeros,
            "top_usefulness": zeros,
            "top_economic_usefulness": zeros,
            "top_model_contribution_abs": zeros,
            "top_model_contribution_enrichment": zeros,
            "top_model_contribution_signed": zeros,
            "top_model_contribution_push": zeros,
            "top_model_contribution_push_enrichment": zeros,
            "top_model_false_positive_push_share": zeros,
            "top_model_false_positive_multiplier": np.ones(
                n_features, dtype=np.float32
            ),
            "stability_tiebreaker": zeros,
            "asset_edge_std": zeros,
            "month_edge_std": zeros,
            "asset_stability_multiplier": np.ones(n_features, dtype=np.float32),
            "month_stability_multiplier": np.ones(n_features, dtype=np.float32),
            "group_stability_multiplier": np.ones(n_features, dtype=np.float32),
            "contribution_redundancy_corr": zeros,
            "contribution_redundancy_multiplier": np.ones(n_features, dtype=np.float32),
            "n_eval_folds": zeros,
        }
    presence = np.vstack([s["presence"] for s in fold_stats]).astype(np.float32)
    signs = np.vstack([s["sign"] for s in fold_stats]).astype(np.float32)
    top_use = np.vstack([s["top_usefulness"] for s in fold_stats]).astype(np.float32)
    top_econ = np.vstack(
        [
            s.get("top_economic_usefulness", np.zeros(n_features, dtype=np.float32))
            for s in fold_stats
        ]
    ).astype(np.float32)
    top_contrib_abs = np.vstack(
        [
            s.get("top_model_contribution_abs", np.zeros(n_features, dtype=np.float32))
            for s in fold_stats
        ]
    ).astype(np.float32)
    top_contrib_enrichment = np.vstack(
        [
            s.get(
                "top_model_contribution_enrichment",
                np.zeros(n_features, dtype=np.float32),
            )
            for s in fold_stats
        ]
    ).astype(np.float32)
    top_contrib_signed = np.vstack(
        [
            s.get(
                "top_model_contribution_signed",
                np.zeros(n_features, dtype=np.float32),
            )
            for s in fold_stats
        ]
    ).astype(np.float32)
    top_contrib_push = np.vstack(
        [
            s.get(
                "top_model_contribution_push",
                np.zeros(n_features, dtype=np.float32),
            )
            for s in fold_stats
        ]
    ).astype(np.float32)
    top_contrib_push_enrichment = np.vstack(
        [
            s.get(
                "top_model_contribution_push_enrichment",
                np.zeros(n_features, dtype=np.float32),
            )
            for s in fold_stats
        ]
    ).astype(np.float32)
    top_fp_push_share = np.vstack(
        [
            s.get(
                "top_model_false_positive_push_share",
                np.zeros(n_features, dtype=np.float32),
            )
            for s in fold_stats
        ]
    ).astype(np.float32)
    top_fp_multiplier = np.vstack(
        [
            s.get(
                "top_model_false_positive_multiplier",
                np.ones(n_features, dtype=np.float32),
            )
            for s in fold_stats
        ]
    ).astype(np.float32)
    presence_rate = np.mean(presence > 0.0, axis=0).astype(np.float32)
    pos_rate = np.mean(signs > 0.0, axis=0)
    neg_rate = np.mean(signs < 0.0, axis=0)
    sign_consistency = np.maximum(pos_rate, neg_rate).astype(np.float32)
    sign_consistency[presence_rate <= 0.0] = 0.0
    mean_top = np.mean(top_use, axis=0)
    std_top = np.std(top_use, axis=0)
    top_usefulness = mean_top.astype(np.float32)
    top_economic_usefulness = np.mean(top_econ, axis=0).astype(np.float32)
    top_model_contribution_abs = np.mean(top_contrib_abs, axis=0).astype(np.float32)
    top_model_contribution_enrichment = np.mean(top_contrib_enrichment, axis=0).astype(
        np.float32
    )
    top_model_contribution_signed = np.mean(top_contrib_signed, axis=0).astype(
        np.float32
    )
    top_model_contribution_push = np.mean(top_contrib_push, axis=0).astype(np.float32)
    top_model_contribution_push_enrichment = np.mean(
        top_contrib_push_enrichment, axis=0
    ).astype(np.float32)
    top_model_false_positive_push_share = np.mean(top_fp_push_share, axis=0).astype(
        np.float32
    )
    top_model_false_positive_multiplier = np.mean(top_fp_multiplier, axis=0).astype(
        np.float32
    )
    stability_tiebreaker = (1.0 / (1.0 + std_top)).astype(np.float32)
    asset_std = np.vstack(
        [
            s.get("asset_edge_std", np.zeros(n_features, dtype=np.float32))
            for s in fold_stats
        ]
    ).astype(np.float32)
    month_std = np.vstack(
        [
            s.get("month_edge_std", np.zeros(n_features, dtype=np.float32))
            for s in fold_stats
        ]
    ).astype(np.float32)
    asset_mult = np.vstack(
        [
            s.get(
                "asset_stability_multiplier",
                np.ones(n_features, dtype=np.float32),
            )
            for s in fold_stats
        ]
    ).astype(np.float32)
    month_mult = np.vstack(
        [
            s.get(
                "month_stability_multiplier",
                np.ones(n_features, dtype=np.float32),
            )
            for s in fold_stats
        ]
    ).astype(np.float32)
    contrib_redundancy_corr = np.vstack(
        [
            s.get(
                "contribution_redundancy_corr",
                np.zeros(n_features, dtype=np.float32),
            )
            for s in fold_stats
        ]
    ).astype(np.float32)
    contrib_redundancy_multiplier = np.vstack(
        [
            s.get(
                "contribution_redundancy_multiplier",
                np.ones(n_features, dtype=np.float32),
            )
            for s in fold_stats
        ]
    ).astype(np.float32)
    asset_stability_multiplier = np.mean(asset_mult, axis=0).astype(np.float32)
    month_stability_multiplier = np.mean(month_mult, axis=0).astype(np.float32)
    group_stability_multiplier = np.sqrt(
        np.clip(asset_stability_multiplier, 0.65, 1.0)
        * np.clip(month_stability_multiplier, 0.65, 1.0)
    ).astype(np.float32)
    return {
        "presence_rate": presence_rate,
        "sign_consistency": sign_consistency,
        "top_usefulness": top_usefulness,
        "top_economic_usefulness": top_economic_usefulness,
        "top_model_contribution_abs": top_model_contribution_abs,
        "top_model_contribution_enrichment": top_model_contribution_enrichment,
        "top_model_contribution_signed": top_model_contribution_signed,
        "top_model_contribution_push": top_model_contribution_push,
        "top_model_contribution_push_enrichment": top_model_contribution_push_enrichment,
        "top_model_false_positive_push_share": top_model_false_positive_push_share,
        "top_model_false_positive_multiplier": top_model_false_positive_multiplier,
        "stability_tiebreaker": stability_tiebreaker,
        "asset_edge_std": np.mean(asset_std, axis=0).astype(np.float32),
        "month_edge_std": np.mean(month_std, axis=0).astype(np.float32),
        "asset_stability_multiplier": asset_stability_multiplier,
        "month_stability_multiplier": month_stability_multiplier,
        "group_stability_multiplier": group_stability_multiplier,
        "contribution_redundancy_corr": np.mean(contrib_redundancy_corr, axis=0).astype(
            np.float32
        ),
        "contribution_redundancy_multiplier": np.mean(
            contrib_redundancy_multiplier, axis=0
        ).astype(np.float32),
        "n_eval_folds": np.full(n_features, len(fold_stats), dtype=np.float32),
    }


def _combine_ebm_feature_scores(
    base_scores: np.ndarray,
    eval_stats: dict[str, np.ndarray] | None,
    round_id: int,
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    scores = np.nan_to_num(
        np.asarray(base_scores, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    if eval_stats is None:
        return scores, {
            "presence_min": float("nan"),
            "sign_consistency_min": float("nan"),
            "presence_fail_count": 0.0,
            "sign_fail_count": 0.0,
            "lgbm_presence_min": float("nan"),
            "lgbm_sign_consistency_min": float("nan"),
            "lgbm_presence_fail_count": 0.0,
            "lgbm_sign_fail_count": 0.0,
            "raw_presence_fail_count": 0.0,
            "raw_sign_fail_count": 0.0,
            "top_usefulness_mean": 0.0,
            "top_economic_usefulness_mean": 0.0,
            "top_model_contribution_abs_mean": 0.0,
            "top_model_contribution_enrichment_mean": 0.0,
            "top_model_contribution_push_mean": 0.0,
            "top_model_contribution_push_enrichment_mean": 0.0,
            "top_model_false_positive_push_share_mean": 0.0,
            "top_model_false_positive_multiplier_mean": 1.0,
            "top_bucket_multiplier_mean": 1.0,
            "top_bucket_false_positive_hard_max": float(
                EBM_TOP_BUCKET_FALSE_POSITIVE_HARD_MAX
            ),
            "top_bucket_false_positive_hard_fail_count": 0.0,
            "top_bucket_false_positive_multiplier_min": float(
                EBM_TOP_BUCKET_FALSE_POSITIVE_MULTIPLIER_MIN
            ),
            "stability_tiebreaker_mean": 0.0,
            "asset_edge_std_mean": 0.0,
            "month_edge_std_mean": 0.0,
            "asset_stability_multiplier_mean": 1.0,
            "month_stability_multiplier_mean": 1.0,
            "group_stability_multiplier_mean": 1.0,
            "contribution_redundancy_corr_mean": 0.0,
            "contribution_redundancy_multiplier_mean": 1.0,
        }
    n_eval_folds = np.asarray(
        eval_stats.get("n_eval_folds", np.zeros_like(scores)), dtype=np.float32
    )
    if n_eval_folds.size == 0 or float(np.max(n_eval_folds)) <= 0.0:
        return scores, {
            "presence_min": float("nan"),
            "sign_consistency_min": float("nan"),
            "presence_fail_count": 0.0,
            "sign_fail_count": 0.0,
            "lgbm_presence_min": float("nan"),
            "lgbm_sign_consistency_min": float("nan"),
            "lgbm_presence_fail_count": 0.0,
            "lgbm_sign_fail_count": 0.0,
            "raw_presence_fail_count": 0.0,
            "raw_sign_fail_count": 0.0,
            "top_usefulness_mean": 0.0,
            "top_economic_usefulness_mean": 0.0,
            "top_model_contribution_abs_mean": 0.0,
            "top_model_contribution_enrichment_mean": 0.0,
            "top_model_contribution_push_mean": 0.0,
            "top_model_contribution_push_enrichment_mean": 0.0,
            "top_model_false_positive_push_share_mean": 0.0,
            "top_model_false_positive_multiplier_mean": 1.0,
            "top_bucket_multiplier_mean": 1.0,
            "top_bucket_false_positive_hard_max": float(
                EBM_TOP_BUCKET_FALSE_POSITIVE_HARD_MAX
            ),
            "top_bucket_false_positive_hard_fail_count": 0.0,
            "top_bucket_false_positive_multiplier_min": float(
                EBM_TOP_BUCKET_FALSE_POSITIVE_MULTIPLIER_MIN
            ),
            "stability_tiebreaker_mean": 0.0,
            "asset_edge_std_mean": 0.0,
            "month_edge_std_mean": 0.0,
            "asset_stability_multiplier_mean": 1.0,
            "month_stability_multiplier_mean": 1.0,
            "group_stability_multiplier_mean": 1.0,
            "contribution_redundancy_corr_mean": 0.0,
            "contribution_redundancy_multiplier_mean": 1.0,
        }
    presence = np.asarray(eval_stats["presence_rate"], dtype=np.float32)
    sign_consistency = np.asarray(eval_stats["sign_consistency"], dtype=np.float32)
    top_use = np.asarray(eval_stats["top_usefulness"], dtype=np.float32)
    top_econ = np.asarray(
        eval_stats.get("top_economic_usefulness", np.zeros_like(scores)),
        dtype=np.float32,
    )
    top_contrib_abs = np.asarray(
        eval_stats.get("top_model_contribution_abs", np.zeros_like(scores)),
        dtype=np.float32,
    )
    top_contrib_enrichment = np.asarray(
        eval_stats.get("top_model_contribution_enrichment", np.zeros_like(scores)),
        dtype=np.float32,
    )
    top_contrib_signed = np.asarray(
        eval_stats.get("top_model_contribution_signed", np.zeros_like(scores)),
        dtype=np.float32,
    )
    top_contrib_push = np.asarray(
        eval_stats.get("top_model_contribution_push", np.zeros_like(scores)),
        dtype=np.float32,
    )
    top_contrib_push_enrichment = np.asarray(
        eval_stats.get("top_model_contribution_push_enrichment", np.zeros_like(scores)),
        dtype=np.float32,
    )
    top_fp_push_share = np.asarray(
        eval_stats.get("top_model_false_positive_push_share", np.zeros_like(scores)),
        dtype=np.float32,
    )
    top_fp_multiplier = np.asarray(
        eval_stats.get("top_model_false_positive_multiplier", np.ones_like(scores)),
        dtype=np.float32,
    )
    stability = np.asarray(eval_stats["stability_tiebreaker"], dtype=np.float32)
    asset_edge_std = np.asarray(
        eval_stats.get("asset_edge_std", np.zeros_like(scores)), dtype=np.float32
    )
    month_edge_std = np.asarray(
        eval_stats.get("month_edge_std", np.zeros_like(scores)), dtype=np.float32
    )
    asset_stability_multiplier = np.asarray(
        eval_stats.get("asset_stability_multiplier", np.ones_like(scores)),
        dtype=np.float32,
    )
    month_stability_multiplier = np.asarray(
        eval_stats.get("month_stability_multiplier", np.ones_like(scores)),
        dtype=np.float32,
    )
    group_stability_multiplier = np.asarray(
        eval_stats.get("group_stability_multiplier", np.ones_like(scores)),
        dtype=np.float32,
    )
    contribution_redundancy_corr = np.asarray(
        eval_stats.get("contribution_redundancy_corr", np.zeros_like(scores)),
        dtype=np.float32,
    )
    contribution_redundancy_multiplier = np.asarray(
        eval_stats.get("contribution_redundancy_multiplier", np.ones_like(scores)),
        dtype=np.float32,
    )
    expected_n = scores.shape[0]
    if any(
        arr.shape[0] != expected_n
        for arr in (
            presence,
            sign_consistency,
            top_use,
            top_econ,
            top_contrib_abs,
            top_contrib_enrichment,
            top_contrib_signed,
            top_contrib_push,
            top_contrib_push_enrichment,
            top_fp_push_share,
            top_fp_multiplier,
            stability,
            asset_edge_std,
            month_edge_std,
            asset_stability_multiplier,
            month_stability_multiplier,
            group_stability_multiplier,
            contribution_redundancy_corr,
            contribution_redundancy_multiplier,
        )
    ):
        tprint(
            "EBMOnLGBM: feature eval stats shape mismatch; "
            "falling back to ungated feature scores."
        )
        return scores, {
            "presence_min": float("nan"),
            "sign_consistency_min": float("nan"),
            "presence_fail_count": 0.0,
            "sign_fail_count": 0.0,
            "lgbm_presence_min": float("nan"),
            "lgbm_sign_consistency_min": float("nan"),
            "lgbm_presence_fail_count": 0.0,
            "lgbm_sign_fail_count": 0.0,
            "raw_presence_fail_count": 0.0,
            "raw_sign_fail_count": 0.0,
            "top_usefulness_mean": 0.0,
            "top_economic_usefulness_mean": 0.0,
            "top_model_contribution_abs_mean": 0.0,
            "top_model_contribution_enrichment_mean": 0.0,
            "top_model_contribution_push_mean": 0.0,
            "top_model_contribution_push_enrichment_mean": 0.0,
            "top_model_false_positive_push_share_mean": 0.0,
            "top_model_false_positive_multiplier_mean": 1.0,
            "top_bucket_multiplier_mean": 1.0,
            "top_bucket_false_positive_hard_max": float(
                EBM_TOP_BUCKET_FALSE_POSITIVE_HARD_MAX
            ),
            "top_bucket_false_positive_hard_fail_count": 0.0,
            "top_bucket_false_positive_multiplier_min": float(
                EBM_TOP_BUCKET_FALSE_POSITIVE_MULTIPLIER_MIN
            ),
            "stability_tiebreaker_mean": 0.0,
            "asset_edge_std_mean": 0.0,
            "month_edge_std_mean": 0.0,
            "asset_stability_multiplier_mean": 1.0,
            "month_stability_multiplier_mean": 1.0,
            "group_stability_multiplier_mean": 1.0,
            "contribution_redundancy_corr_mean": 0.0,
            "contribution_redundancy_multiplier_mean": 1.0,
            "eval_stats_shape_mismatch": 1.0,
        }
    presence_min = _round_presence_min(round_id)
    sign_min = _round_sign_consistency_min(round_id)
    presence_threshold = np.full_like(presence, presence_min, dtype=np.float32)
    sign_threshold = np.full_like(sign_consistency, sign_min, dtype=np.float32)
    lgbm_mask = np.zeros_like(presence_threshold, dtype=bool)
    if feature_names is not None:
        candidate_lgbm_mask = np.asarray(
            [_is_lgbm_tree_feature_name(name) for name in feature_names],
            dtype=bool,
        )
        if candidate_lgbm_mask.shape[0] == presence_threshold.shape[0]:
            lgbm_mask = candidate_lgbm_mask
            presence_threshold[lgbm_mask] = np.minimum(
                presence_threshold[lgbm_mask] + 0.10, 1.0
            )
            sign_threshold[lgbm_mask] = np.minimum(
                sign_threshold[lgbm_mask] + 0.10, 1.0
            )
    raw_mask = ~lgbm_mask
    presence_fail = presence < presence_threshold
    sign_fail = sign_consistency < sign_threshold
    top_fp_hard_fail = (top_contrib_push > 1e-8) & (
        top_fp_push_share > EBM_TOP_BUCKET_FALSE_POSITIVE_HARD_MAX
    )
    gated = scores.copy()
    gated[presence_fail | sign_fail | top_fp_hard_fail] = 0.0
    top_norm = top_use / max(float(np.nanpercentile(top_use, 90.0)), 1e-6)
    top_econ_norm = top_econ / max(float(np.nanpercentile(top_econ, 90.0)), 1e-6)
    top_contrib_push_norm = top_contrib_push / max(
        float(np.nanpercentile(top_contrib_push, 90.0)), 1e-6
    )
    top_contrib_push_enrichment_norm = top_contrib_push_enrichment / max(
        float(np.nanpercentile(top_contrib_push_enrichment, 90.0)), 1e-6
    )
    stability_norm = stability / max(float(np.nanpercentile(stability, 90.0)), 1e-6)
    fp_mult_min = float(np.clip(EBM_TOP_BUCKET_FALSE_POSITIVE_MULTIPLIER_MIN, 0.0, 1.0))
    top_fp_score_mult = np.clip(top_fp_multiplier, fp_mult_min, 1.0).astype(np.float32)
    top_contrib_signal = np.where(
        top_contrib_push > 1e-8,
        np.clip(
            0.5 * top_contrib_push_norm + 0.5 * top_contrib_push_enrichment_norm,
            0.0,
            3.0,
        )
        * top_fp_score_mult,
        0.0,
    ).astype(np.float32)
    top_bucket_weight_sum = max(
        EBM_TOP_SLICE_TIEBREAKER_WEIGHT
        + EBM_TOP_ECON_TIEBREAKER_WEIGHT
        + EBM_TOP_CONTRIB_TIEBREAKER_WEIGHT
        + EBM_STABILITY_TIEBREAKER_WEIGHT,
        1e-6,
    )
    top_bucket_signal = (
        EBM_TOP_SLICE_TIEBREAKER_WEIGHT * np.clip(top_norm, 0.0, 3.0)
        + EBM_TOP_ECON_TIEBREAKER_WEIGHT * np.clip(top_econ_norm, 0.0, 3.0)
        + EBM_TOP_CONTRIB_TIEBREAKER_WEIGHT * top_contrib_signal
        + EBM_STABILITY_TIEBREAKER_WEIGHT * np.clip(stability_norm, 0.0, 3.0)
    ) / top_bucket_weight_sum
    top_bucket_multiplier = np.clip(
        1.0 + EBM_TOP_BUCKET_MULTIPLIER_STRENGTH * top_bucket_signal,
        1.0,
        max(1.0, EBM_TOP_BUCKET_MULTIPLIER_MAX),
    ).astype(np.float32)
    group_mult = np.sqrt(np.clip(group_stability_multiplier, 0.65, 1.0)).astype(
        np.float32
    )
    contribution_redundancy_mult = np.clip(
        contribution_redundancy_multiplier,
        EBM_CONTRIB_REDUNDANCY_MULTIPLIER_MIN,
        1.0,
    ).astype(np.float32)
    combined = (
        gated
        * group_mult
        * top_fp_score_mult
        * top_bucket_multiplier
        * contribution_redundancy_mult
    )
    return (
        np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        {
            "presence_min": float(presence_min),
            "sign_consistency_min": float(sign_min),
            "presence_fail_count": float(np.sum(presence_fail)),
            "sign_fail_count": float(np.sum(sign_fail)),
            "lgbm_presence_min": float(min(presence_min + 0.10, 1.0)),
            "lgbm_sign_consistency_min": float(min(sign_min + 0.10, 1.0)),
            "lgbm_presence_fail_count": float(np.sum(presence_fail & lgbm_mask)),
            "lgbm_sign_fail_count": float(np.sum(sign_fail & lgbm_mask)),
            "raw_presence_fail_count": float(np.sum(presence_fail & raw_mask)),
            "raw_sign_fail_count": float(np.sum(sign_fail & raw_mask)),
            "top_usefulness_mean": float(np.mean(top_use)),
            "top_economic_usefulness_mean": float(np.mean(top_econ)),
            "top_model_contribution_abs_mean": float(np.mean(top_contrib_abs)),
            "top_model_contribution_enrichment_mean": float(
                np.mean(top_contrib_enrichment)
            ),
            "top_model_contribution_signed_mean": float(np.mean(top_contrib_signed)),
            "top_model_contribution_push_mean": float(np.mean(top_contrib_push)),
            "top_model_contribution_push_enrichment_mean": float(
                np.mean(top_contrib_push_enrichment)
            ),
            "top_model_false_positive_push_share_mean": float(
                np.mean(top_fp_push_share)
            ),
            "top_model_false_positive_multiplier_mean": float(
                np.mean(top_fp_score_mult)
            ),
            "top_bucket_multiplier_mean": float(np.mean(top_bucket_multiplier)),
            "top_bucket_false_positive_hard_max": float(
                EBM_TOP_BUCKET_FALSE_POSITIVE_HARD_MAX
            ),
            "top_bucket_false_positive_hard_fail_count": float(
                np.sum(top_fp_hard_fail)
            ),
            "top_bucket_false_positive_multiplier_min": float(fp_mult_min),
            "stability_tiebreaker_mean": float(np.mean(stability)),
            "asset_edge_std_mean": float(np.mean(asset_edge_std)),
            "month_edge_std_mean": float(np.mean(month_edge_std)),
            "asset_stability_multiplier_mean": float(
                np.mean(asset_stability_multiplier)
            ),
            "month_stability_multiplier_mean": float(
                np.mean(month_stability_multiplier)
            ),
            "group_stability_multiplier_mean": float(np.mean(group_mult)),
            "contribution_redundancy_corr_mean": float(
                np.mean(contribution_redundancy_corr)
            ),
            "contribution_redundancy_multiplier_mean": float(
                np.mean(contribution_redundancy_mult)
            ),
        },
    )


def _top_prediction_feature_contributors(
    eval_stats: dict[str, np.ndarray] | None,
    feature_names: list[str],
    *,
    top_n: int = 10,
) -> list[dict[str, float | str]]:
    if not eval_stats or not feature_names:
        return []
    top_abs = np.asarray(
        eval_stats.get("top_model_contribution_abs", np.array([], dtype=np.float32)),
        dtype=np.float32,
    )
    enrichment = np.asarray(
        eval_stats.get(
            "top_model_contribution_enrichment", np.array([], dtype=np.float32)
        ),
        dtype=np.float32,
    )
    signed = np.asarray(
        eval_stats.get("top_model_contribution_signed", np.array([], dtype=np.float32)),
        dtype=np.float32,
    )
    push = np.asarray(
        eval_stats.get("top_model_contribution_push", np.array([], dtype=np.float32)),
        dtype=np.float32,
    )
    push_enrichment = np.asarray(
        eval_stats.get(
            "top_model_contribution_push_enrichment",
            np.array([], dtype=np.float32),
        ),
        dtype=np.float32,
    )
    fp_push_share = np.asarray(
        eval_stats.get(
            "top_model_false_positive_push_share",
            np.array([], dtype=np.float32),
        ),
        dtype=np.float32,
    )
    fp_multiplier = np.asarray(
        eval_stats.get(
            "top_model_false_positive_multiplier",
            np.ones_like(push),
        ),
        dtype=np.float32,
    )
    top_econ = np.asarray(
        eval_stats.get("top_economic_usefulness", np.array([], dtype=np.float32)),
        dtype=np.float32,
    )
    group_mult = np.asarray(
        eval_stats.get("group_stability_multiplier", np.ones_like(top_abs)),
        dtype=np.float32,
    )
    contribution_redundancy_corr = np.asarray(
        eval_stats.get("contribution_redundancy_corr", np.zeros_like(top_abs)),
        dtype=np.float32,
    )
    contribution_redundancy_multiplier = np.asarray(
        eval_stats.get("contribution_redundancy_multiplier", np.ones_like(top_abs)),
        dtype=np.float32,
    )
    n = min(
        len(feature_names),
        len(top_abs),
        len(enrichment),
        len(signed),
        len(push),
        len(push_enrichment),
        len(contribution_redundancy_corr),
        len(contribution_redundancy_multiplier),
    )
    if n <= 0:
        return []
    top_econ = top_econ[:n] if len(top_econ) >= n else np.zeros(n, dtype=np.float32)
    group_mult = (
        group_mult[:n] if len(group_mult) >= n else np.ones(n, dtype=np.float32)
    )
    fp_push_share = (
        fp_push_share[:n] if len(fp_push_share) >= n else np.zeros(n, dtype=np.float32)
    )
    fp_multiplier = (
        fp_multiplier[:n] if len(fp_multiplier) >= n else np.ones(n, dtype=np.float32)
    )
    fp_mult_min = float(np.clip(EBM_TOP_BUCKET_FALSE_POSITIVE_MULTIPLIER_MIN, 0.0, 1.0))
    fp_multiplier = np.clip(fp_multiplier, fp_mult_min, 1.0)
    contribution_redundancy_corr = contribution_redundancy_corr[:n]
    contribution_redundancy_multiplier = np.clip(
        contribution_redundancy_multiplier[:n],
        EBM_CONTRIB_REDUNDANCY_MULTIPLIER_MIN,
        1.0,
    )
    contributor_group_mult = np.sqrt(np.clip(group_mult, 0.65, 1.0))
    score = np.nan_to_num(
        push[:n]
        * np.maximum(push_enrichment[:n], 0.0)
        * np.maximum(contributor_group_mult, 0.0)
        * fp_multiplier
        * contribution_redundancy_multiplier,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    order = np.argsort(score)[::-1]
    out: list[dict[str, float | str]] = []
    for i in order[: max(0, int(top_n))]:
        if not np.isfinite(score[i]) or float(score[i]) <= 0.0:
            continue
        out.append(
            {
                "feature": str(feature_names[i]),
                "top_contribution_score": float(score[i]),
                "top_contribution_abs": float(top_abs[i]),
                "top_contribution_enrichment": float(enrichment[i]),
                "top_contribution_signed": float(signed[i]),
                "top_contribution_push": float(push[i]),
                "top_contribution_push_enrichment": float(push_enrichment[i]),
                "top_false_positive_push_share": float(fp_push_share[i]),
                "top_false_positive_multiplier": float(fp_multiplier[i]),
                "top_economic_usefulness": float(top_econ[i]),
                "group_stability_multiplier": float(group_mult[i]),
                "contribution_redundancy_corr": float(contribution_redundancy_corr[i]),
                "contribution_redundancy_multiplier": float(
                    contribution_redundancy_multiplier[i]
                ),
            }
        )
    return out


def _build_tree_features_fold(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_eval_raw: np.ndarray,
    feature_names: list[str],
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    model = _fit_lgbm_tree_feature_model(
        X_train_raw,
        y_train,
        None,
        None,
        {"max_depth": 3, "min_child_samples": 100},
        random_state,
    )

    tr_tree, tree_names, scales = _compute_soft_tree_features_ebm(
        [model], X_train_raw, None
    )
    ev_tree, _, _ = _compute_soft_tree_features_ebm([model], X_eval_raw, scales)

    tr_tree, ev_tree, tree_names = _target_aware_tree_feature_cap(
        tr_tree,
        ev_tree,
        tree_names,
        y_train,
        classifier=_looks_classifier_target(y_train),
        random_state=random_state,
    )

    return (
        pd.DataFrame(tr_tree, columns=tree_names),
        pd.DataFrame(ev_tree, columns=tree_names),
        tree_names,
    )


def _augment_with_tree_features(
    X_train_raw: pd.DataFrame,
    y_train: np.ndarray,
    X_eval_raw: pd.DataFrame,
    random_state: int,
    y_eval: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    x_tr = np.nan_to_num(
        X_train_raw.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    x_ev = np.nan_to_num(
        X_eval_raw.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )

    if y_eval is None:
        from sklearn.model_selection import train_test_split

        idx_tr, idx_va = train_test_split(
            np.arange(len(y_train)), test_size=0.15, random_state=random_state
        )
        x_tr_fit = x_tr[idx_tr]
        y_tr_fit = y_train[idx_tr]
        x_ev_fit = x_tr[idx_va]
        y_ev_fit = y_train[idx_va]
    else:
        x_tr_fit = x_tr
        y_tr_fit = y_train
        x_ev_fit = x_ev
        y_ev_fit = y_eval

    x_tr_fit, y_tr_fit = _subsample_tree_fit_rows(
        x_tr_fit,
        np.asarray(y_tr_fit),
        random_state=random_state + 991,
    )

    N = len(y_train)
    grid = [
        {
            "max_depth": 3,
            "min_child_pct": 0.02,
            "min_child_samples": max(2, int(0.02 * N)),
        },
        {
            "max_depth": 3,
            "min_child_pct": 0.05,
            "min_child_samples": max(2, int(0.05 * N)),
        },
        {
            "max_depth": 3,
            "min_child_pct": 0.08,
            "min_child_samples": max(2, int(0.08 * N)),
        },
        {
            "max_depth": 4,
            "min_child_pct": 0.05,
            "min_child_samples": max(2, int(0.05 * N)),
        },
        {
            "max_depth": 4,
            "min_child_pct": 0.10,
            "min_child_samples": max(2, int(0.10 * N)),
        },
        {
            "max_depth": 4,
            "min_child_pct": 0.15,
            "min_child_samples": max(2, int(0.15 * N)),
        },
    ]

    models = []
    for i, params in enumerate(grid):
        tprint(
            "EBMOnLGBM tree features: fitting native LGBM bundle "
            f"{i + 1}/{len(grid)} on {len(y_tr_fit)} rows "
            f"(early_stopping_rounds={EBM_TREE_LGBM_EARLY_STOPPING_ROUNDS})."
        )
        model = _fit_lgbm_tree_feature_model(
            x_tr_fit,
            y_tr_fit,
            x_ev_fit,
            y_ev_fit,
            params,
            random_state + i,
        )
        models.append(model)

    tr_tree, tree_names, scales = _compute_soft_tree_features_ebm(models, x_tr, None)
    ev_tree, _, _ = _compute_soft_tree_features_ebm(models, x_ev, scales)

    tree_generated = int(tr_tree.shape[1])
    tr_tree, ev_tree, tree_names = _target_aware_tree_feature_cap(
        tr_tree,
        ev_tree,
        tree_names,
        np.asarray(y_train),
        classifier=_looks_classifier_target(y_train),
        random_state=random_state,
    )
    tprint(
        "EBMOnLGBM tree features: "
        f"generated={tree_generated}, kept={tr_tree.shape[1]}, "
        f"cap={EBM_TREE_FEATURE_CAP}."
    )

    train_tree = pd.DataFrame(tr_tree, columns=tree_names, index=X_train_raw.index)
    eval_tree = pd.DataFrame(ev_tree, columns=tree_names, index=X_eval_raw.index)

    train_aug = pd.concat([X_train_raw, train_tree], axis=1)
    eval_aug = pd.concat([X_eval_raw, eval_tree], axis=1)

    bundle = {
        "models": models,
        "tree_feature_config": {},
        "tree_feature_names": tree_names,
        "tree_feature_scales": scales,
    }
    return train_aug, eval_aug, bundle


def _augment_with_oof_tree_features(
    X_select_raw: pd.DataFrame,
    y_select: np.ndarray,
    X_eval_raw: pd.DataFrame,
    random_state: int,
    classifier: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    splitter = (
        StratifiedKFold(n_splits=EBM_CV_SPLITS, shuffle=True, random_state=random_state)
        if classifier
        else KFold(n_splits=EBM_CV_SPLITS, shuffle=True, random_state=random_state)
    )
    y_split = (
        np.asarray(y_select >= 0.5, dtype=np.int8)
        if classifier
        else np.asarray(y_select, dtype=np.float32)
    )
    select_parts: list[pd.DataFrame] = []
    eval_parts: list[pd.DataFrame] = []
    fold_models: list[Any] = []
    fold_models_by_fold: list[list[Any]] = []
    fold_tree_names: list[list[str]] = []
    fold_scales: list[np.ndarray] = []
    leaf_screen_diagnostics: list[dict[str, Any]] = []

    for fold_i, (tr, va) in enumerate(
        splitter.split(np.zeros(len(y_split)), y_split), start=1
    ):
        tprint(
            "EBMOnLGBM tree features: cross-fitting OOF LightGBM "
            f"fold {fold_i}/{EBM_CV_SPLITS}."
        )
        X_tr_raw = X_select_raw.iloc[tr].reset_index(drop=True)
        X_va_raw = X_select_raw.iloc[va].reset_index(drop=True)
        x_tr = np.nan_to_num(
            X_tr_raw.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        x_va = np.nan_to_num(
            X_va_raw.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        x_ev = np.nan_to_num(
            X_eval_raw.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        x_fit, y_fit = _subsample_tree_fit_rows(
            x_tr,
            np.asarray(y_select[tr]),
            random_state=random_state + fold_i * 3011,
        )
        x_fit_tr, y_fit_tr, x_fit_va, y_fit_va = _inner_tree_fit_split(
            x_fit,
            y_fit,
            random_state=random_state + fold_i * 4013,
        )
        n_fold = len(y_fit_tr)
        grid = [
            {
                "max_depth": 3,
                "min_child_pct": 0.02,
                "min_child_samples": max(2, int(0.02 * n_fold)),
            },
            {
                "max_depth": 3,
                "min_child_pct": 0.05,
                "min_child_samples": max(2, int(0.05 * n_fold)),
            },
            {
                "max_depth": 3,
                "min_child_pct": 0.08,
                "min_child_samples": max(2, int(0.08 * n_fold)),
            },
            {
                "max_depth": 4,
                "min_child_pct": 0.05,
                "min_child_samples": max(2, int(0.05 * n_fold)),
            },
            {
                "max_depth": 4,
                "min_child_pct": 0.10,
                "min_child_samples": max(2, int(0.10 * n_fold)),
            },
            {
                "max_depth": 4,
                "min_child_pct": 0.15,
                "min_child_samples": max(2, int(0.15 * n_fold)),
            },
        ]
        models = []
        for i, params in enumerate(grid):
            tprint(
                "EBMOnLGBM tree features: fitting native LGBM bundle "
                f"{i + 1}/{len(grid)} on OOF fold {fold_i}/{EBM_CV_SPLITS}."
            )
            model_seed = random_state + fold_i * 1009 + i
            model0 = _fit_lgbm_tree_feature_model(
                x_fit_tr,
                y_fit_tr,
                x_fit_va,
                y_fit_va,
                params,
                model_seed,
            )
            pred_fit = np.full(len(y_fit_tr), np.nan, dtype=np.float32)
            if classifier:
                split_target = np.asarray(y_fit_tr >= 0.5, dtype=np.int8)
                class_counts = np.bincount(split_target, minlength=2)
                nested_splits = int(min(3, np.min(class_counts)))
                nested_splits = max(nested_splits, 0)
            else:
                split_target = np.asarray(y_fit_tr, dtype=np.float32)
                nested_splits = (
                    3 if len(y_fit_tr) >= 120 else 2 if len(y_fit_tr) >= 40 else 0
                )
            if nested_splits >= 2:
                nested_splitter = (
                    StratifiedKFold(
                        n_splits=nested_splits,
                        shuffle=True,
                        random_state=model_seed + 11,
                    )
                    if classifier
                    else KFold(
                        n_splits=nested_splits,
                        shuffle=True,
                        random_state=model_seed + 11,
                    )
                )
                for nested_tr, nested_va in nested_splitter.split(
                    np.zeros(len(y_fit_tr)), split_target
                ):
                    nested_model = _fit_lgbm_tree_feature_model(
                        x_fit_tr[nested_tr],
                        y_fit_tr[nested_tr],
                        x_fit_tr[nested_va],
                        y_fit_tr[nested_va],
                        params,
                        model_seed + 101 + int(nested_va[0]),
                    )
                    pred_fit[nested_va] = np.asarray(
                        nested_model.booster_.predict(x_fit_tr[nested_va]),
                        dtype=np.float32,
                    )
            if not np.all(np.isfinite(pred_fit)):
                pred_fit = np.asarray(
                    model0.booster_.predict(x_fit_tr), dtype=np.float32
                )
            distill_w = _compute_weight_distillation(
                np.asarray(y_fit_tr, dtype=np.float32),
                pred_fit,
                None,
                is_classifier=classifier,
                include_false_positive_focus=False,
            )
            fp_avoid_w = _false_positive_avoidance_weight(
                np.asarray(y_fit_tr, dtype=np.float32),
                pred_fit,
                classifier=classifier,
                trade_top_frac=_target_top_fraction(),
            )
            fit_sample_weight, fit_ess = _normalize_rank_based_weights(
                distill_w * fp_avoid_w
            )
            model = _fit_lgbm_tree_feature_model(
                x_fit_tr,
                y_fit_tr,
                x_fit_va,
                y_fit_va,
                params,
                model_seed + 17,
                sample_weight=fit_sample_weight,
            )
            tprint(
                "EBMOnLGBM tree features: nested OOF weighting "
                f"fold {fold_i}/{EBM_CV_SPLITS} bundle {i + 1}/{len(grid)} "
                f"(ess={fit_ess:.1f})."
            )
            models.append(model)
        fit_arr, fit_names, scales = _compute_soft_tree_features_ebm(
            models, x_fit_tr, None
        )
        va_arr, tree_names, _ = _compute_soft_tree_features_ebm(models, x_va, scales)
        ev_arr, _, _ = _compute_soft_tree_features_ebm(models, x_ev, scales)
        generated_leaf_names = [n for n in tree_names if "_leaf" in n]
        kept_tree_names = _select_leaf_names_from_score_matrix(
            fit_arr,
            fit_names,
            cap=EBM_TREE_FEATURE_CAP,
        )
        keep_cols = [i for i, n in enumerate(tree_names) if n in set(kept_tree_names)]
        if keep_cols:
            va_arr = va_arr[:, keep_cols]
            ev_arr = ev_arr[:, keep_cols]
            tree_names = [tree_names[i] for i in keep_cols]
        fold_models.extend(models)
        fold_models_by_fold.append(list(models))
        fold_tree_names.append(tree_names)
        fold_scales.append(np.asarray(scales, dtype=np.float32))
        leaf_screen_diagnostics.append(
            {
                "fold": int(fold_i),
                "total_leaves_generated": int(len(generated_leaf_names)),
                "total_leaves_retained": int(
                    len([n for n in tree_names if "_leaf" in n])
                ),
                "total_leaf_features_dropped": int(
                    len(generated_leaf_names)
                    - len([n for n in tree_names if "_leaf" in n])
                ),
                "retention_frac": float(
                    len([n for n in tree_names if "_leaf" in n])
                    / max(len(generated_leaf_names), 1)
                ),
            }
        )
        va_tree = pd.DataFrame(va_arr, columns=tree_names, dtype=np.float32)
        va_tree.index = X_select_raw.index[va]
        select_parts.append(va_tree)
        eval_parts.append(pd.DataFrame(ev_arr, columns=tree_names, dtype=np.float32))

    if not select_parts:
        return _augment_with_tree_features(
            X_select_raw,
            y_select,
            X_eval_raw,
            random_state=random_state,
        )

    all_tree_names = sorted(set().union(*(set(df.columns) for df in select_parts)))
    select_tree = pd.DataFrame(
        0.0, index=X_select_raw.index, columns=all_tree_names, dtype=np.float32
    )
    for part in select_parts:
        select_tree.loc[part.index, part.columns] = part.astype(np.float32)

    eval_tree_acc = pd.DataFrame(
        0.0, index=X_eval_raw.index, columns=all_tree_names, dtype=np.float32
    )
    eval_count = pd.Series(0.0, index=all_tree_names, dtype=np.float32)
    for part in eval_parts:
        cols = list(part.columns)
        eval_tree_acc.loc[:, cols] += part.set_index(X_eval_raw.index).astype(
            np.float32
        )
        eval_count.loc[cols] += 1.0
    for col in all_tree_names:
        c = float(eval_count.loc[col])
        if c > 0:
            eval_tree_acc[col] = eval_tree_acc[col] / c

    if select_tree.shape[1] > EBM_TREE_FEATURE_CAP:
        sel_arr, eval_arr, keep_cols = _target_aware_tree_feature_cap(
            select_tree.to_numpy(dtype=np.float32),
            eval_tree_acc.to_numpy(dtype=np.float32),
            list(select_tree.columns),
            np.asarray(y_select),
            classifier=classifier,
            random_state=random_state + 9041,
        )
        select_tree = pd.DataFrame(
            sel_arr, index=X_select_raw.index, columns=keep_cols, dtype=np.float32
        )
        eval_tree_acc = pd.DataFrame(
            eval_arr, index=X_eval_raw.index, columns=keep_cols, dtype=np.float32
        )

    select_aug = pd.concat([X_select_raw, select_tree], axis=1)
    eval_aug = pd.concat([X_eval_raw, eval_tree_acc], axis=1)
    tprint(
        "EBMOnLGBM tree features: OOF generated "
        f"{len(all_tree_names)}, kept={select_tree.shape[1]}, cap={EBM_TREE_FEATURE_CAP}."
    )
    return (
        select_aug,
        eval_aug,
        {
            "models": fold_models,
            "models_by_fold": fold_models_by_fold,
            "tree_feature_names_by_fold": fold_tree_names,
            "tree_feature_scales_by_fold": fold_scales,
            "tree_feature_names": list(select_tree.columns),
            "oof_tree_features": True,
            "leaf_screen_diagnostics": leaf_screen_diagnostics,
        },
    )


def _compute_oof_bundle_tree_frame(
    tree_bundle: dict[str, Any],
    X_raw: pd.DataFrame,
    selected_tree_names: list[str] | None = None,
) -> pd.DataFrame:
    """Compute inference/final-fit tree features from pruning OOF LGBM folds."""
    models_by_fold = tree_bundle.get("models_by_fold") or []
    scales_by_fold = tree_bundle.get("tree_feature_scales_by_fold") or []
    names_by_fold = tree_bundle.get("tree_feature_names_by_fold") or []
    if not models_by_fold:
        models = list(tree_bundle.get("models", []))
        scales = tree_bundle.get("tree_feature_scales")
        arr, names, _ = _compute_soft_tree_features_ebm(
            models,
            np.nan_to_num(
                X_raw.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
            ),
            scales,
            selected_names=set(selected_tree_names) if selected_tree_names else None,
        )
        return pd.DataFrame(arr, columns=names, index=X_raw.index, dtype=np.float32)

    target_names = (
        [str(c) for c in selected_tree_names]
        if selected_tree_names
        else sorted(set().union(*(set(map(str, names)) for names in names_by_fold)))
    )
    out = pd.DataFrame(0.0, index=X_raw.index, columns=target_names, dtype=np.float32)
    counts = pd.Series(0.0, index=target_names, dtype=np.float32)
    x_raw = np.nan_to_num(
        X_raw.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    forced_tree_value_names: set[str] = set()
    for fold_i, models in enumerate(models_by_fold):
        scales = scales_by_fold[fold_i] if fold_i < len(scales_by_fold) else None
        arr, names, _ = _compute_soft_tree_features_ebm(
            list(models), x_raw, scales, selected_names=set(target_names)
        )
        if arr.shape[1] == 0:
            continue
        frame = pd.DataFrame(arr, columns=names, index=X_raw.index, dtype=np.float32)
        allowed = (
            set(map(str, names_by_fold[fold_i]))
            if fold_i < len(names_by_fold)
            else set(frame.columns)
        )
        forced_cols = [
            c
            for c in frame.columns
            if c.endswith("_value") and c not in allowed and c in target_names
        ]
        if forced_cols:
            forced_tree_value_names.update(map(str, forced_cols))
            for c in forced_cols:
                out[c] = np.float32(0.0)
                counts.loc[c] = np.float32(0.0)
        cols = [
            c
            for c in frame.columns
            if (c in target_names and c in allowed) or c in forced_cols
        ]
        if not cols:
            continue
        out.loc[:, cols] += frame.loc[:, cols].astype(np.float32)
        counts.loc[cols] += 1.0
    for col in list(out.columns):
        c = float(counts.loc[col])
        if c > 0.0:
            out[col] = out[col] / c
    if forced_tree_value_names:
        tprint(
            "EBMOnLGBM: OOF tree bundle retained forced per-tree value features "
            f"({len(forced_tree_value_names)} additional tree-state columns)."
        )
    return out.astype(np.float32)


def _build_ebm_fold_cache(
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray,
    groups: Any,
    y_split: np.ndarray,
    splitter: Any,
    classifier: bool,
    random_state: int,
    round_id: int,
    build_tree_features: bool,
    active_features: list[str] | None = None,
) -> list[dict[str, Any]]:
    fold_cache: list[dict[str, Any]] = []
    raw_cols = list(X.columns)
    for fold_i, (tr, va) in enumerate(
        splitter.split(np.zeros(len(y_split)), y_split), start=1
    ):
        sub_local = _stratified_subsample_indices(
            y_split[tr] if classifier else y[tr],
            max_n=EBM_FOLD_SUBSAMPLE_ROWS,
            random_state=random_state + round_id * 1000 + fold_i,
            classifier=classifier,
        )
        sub = tr[sub_local]
        if build_tree_features:
            X_tr_raw = X.iloc[tr][raw_cols].reset_index(drop=True)
            X_va_raw = X.iloc[va][raw_cols].reset_index(drop=True)
            X_tr_aug, X_va_df, _ = _augment_with_tree_features(
                X_tr_raw,
                y_split[tr] if classifier else y[tr],
                X_va_raw,
                random_state=random_state + fold_i * 100,
                y_eval=y_split[va] if classifier else y[va],
            )
            if active_features is not None:
                tree_cols = [
                    c
                    for c in X_tr_aug.columns
                    if c.startswith("tree_") or c.startswith("lgbm_")
                ]
                keep_cols = list(active_features) + tree_cols
                X_tr_aug = X_tr_aug[keep_cols]
                X_va_df = X_va_df[keep_cols]
            X_sub = X_tr_aug.iloc[sub_local].reset_index(drop=True)
        else:
            if active_features is not None:
                X_sub = X.iloc[sub][active_features].reset_index(drop=True)
                X_va_df = X.iloc[va][active_features].reset_index(drop=True)
            else:
                X_sub = X.iloc[sub].reset_index(drop=True)
                X_va_df = X.iloc[va].reset_index(drop=True)
        fold_cache.append(
            {
                "fold_i": fold_i,
                "sub": sub,
                "va": va,
                "X_sub": X_sub,
                "X_va": X_va_df,
                "y_sub": y[sub],
                "sw_sub": sample_weight[sub],
                "y_va": y[va],
                "groups_va": _groups_take(groups, va),
            }
        )
    return fold_cache


def _fit_ebm_spec_on_cache(
    cls: Any,
    spec_i: int,
    spec: dict[str, Any],
    fold_cache: list[dict[str, Any]],
    n_rows: int,
    classifier: bool,
    round_id: int,
) -> Optional[dict[str, Any]]:
    oof = np.zeros(n_rows, dtype=np.float32)
    fold_metrics: list[dict[str, float]] = []
    shape_models: list[Any] = []
    fold_feature_stats: list[dict[str, np.ndarray]] = []
    mode = "classifier" if classifier else "regressor"

    def _fit_fold(fold: dict[str, Any]) -> dict[str, Any]:
        model = _fit_one_ebm(
            cls,
            fold["X_sub"],
            fold["y_sub"],
            fold["sw_sub"],
            spec,
        )
        raw_tr = _predict_raw_ebm(model, fold["X_sub"], mode)
        pp = SplinePostProcessor(mode).fit(raw_tr, fold["y_sub"])
        raw_va = _predict_raw_ebm(model, fold["X_va"], mode)
        pred_va = pp.predict(raw_va)
        metrics = _metric_pack(
            fold["y_va"],
            pred_va,
            classifier=classifier,
            groups=fold.get("groups_va"),
        )
        stats = _feature_eval_term_stats(
            model,
            fold["X_va"],
            fold["y_va"],
            pred_va,
            list(fold["X_va"].columns),
            classifier=classifier,
            groups=fold.get("groups_va"),
        )
        return {
            "fold_i": int(fold.get("fold_i", -1)),
            "va": fold["va"],
            "pred_va": pred_va,
            "metrics": metrics,
            "model": model,
            "stats": stats,
        }

    fold_jobs = max(1, min(int(EBM_PRUNE_FOLD_N_JOBS), len(fold_cache)))
    if fold_jobs > 1:
        tprint(
            f"      spec {spec_i}: fitting {len(fold_cache)} EBM folds "
            f"with fold_n_jobs={fold_jobs} (each EBM n_jobs=1)."
        )
    if fold_jobs > 1:
        try:
            with ThreadPoolExecutor(max_workers=fold_jobs) as pool:
                futures = {pool.submit(_fit_fold, fold): fold for fold in fold_cache}
                fold_results = []
                for fut in as_completed(futures):
                    fold_results.append(fut.result())
            fold_results.sort(key=lambda r: int(r.get("fold_i", -1)))
        except Exception as exc:
            tprint(
                f"      EBM fold parallel fit failed: round={round_id} "
                f"spec={spec_i} err={exc}; retrying folds sequentially."
            )
            fold_results = [_fit_fold(fold) for fold in fold_cache]
    else:
        try:
            fold_results = [_fit_fold(fold) for fold in fold_cache]
        except Exception as exc:
            tprint(f"      EBM fold failed: round={round_id} spec={spec_i} err={exc}")
            return None

    for res in fold_results:
        oof[np.asarray(res["va"], dtype=np.int32)] = np.asarray(
            res["pred_va"], dtype=np.float32
        )
        fold_metrics.append(res["metrics"])
        shape_models.append(res["model"])
        if res["stats"] is not None:
            fold_feature_stats.append(res["stats"])
    if not fold_metrics:
        return None
    agg = _aggregate_j(fold_metrics)
    agg["J_se"] = float(agg.get("J_std", 0.0) / max(np.sqrt(len(fold_metrics)), 1.0))
    feature_eval_stats = _aggregate_feature_eval_stats(
        fold_feature_stats,
        len(fold_cache[0]["X_va"].columns) if fold_cache else 0,
    )
    return {
        "spec": spec,
        "spec_i": int(spec_i),
        "oof": oof,
        "fold_metrics": fold_metrics,
        "shape_models": shape_models,
        "feature_eval_stats": feature_eval_stats,
        **agg,
    }


def _fit_round_oof(
    cls: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray,
    groups: np.ndarray | None,
    classifier: bool,
    random_state: int,
    round_id: int,
    active_features: list[str] | None = None,
    build_tree_features: bool = False,
    min_samples_leaf_pct: float = 0.02,
    monotone_constraints: list[int] | None = None,
) -> dict[str, Any]:
    splitter = (
        StratifiedKFold(
            n_splits=EBM_CV_SPLITS, shuffle=True, random_state=random_state + round_id
        )
        if classifier
        else KFold(
            n_splits=EBM_CV_SPLITS, shuffle=True, random_state=random_state + round_id
        )
    )
    y_split = (
        np.asarray(y >= 0.5, dtype=np.int8)
        if classifier
        else np.asarray(y, dtype=np.float32)
    )
    specs = _ebm_specs(pruning=True, random_state=random_state + round_id * 100)
    for s in specs:
        s["min_samples_leaf_pct"] = min_samples_leaf_pct
        if monotone_constraints is not None:
            s["monotone_constraints"] = monotone_constraints
    tprint(
        f"    EBM round {round_id}: caching {EBM_CV_SPLITS} folds "
        f"(subsample={EBM_FOLD_SUBSAMPLE_ROWS}, sequential specs, "
        "weights update after each spec)"
    )
    fold_cache = _build_ebm_fold_cache(
        X=X,
        y=y,
        sample_weight=sample_weight,
        groups=groups,
        y_split=y_split,
        splitter=splitter,
        classifier=classifier,
        random_state=random_state,
        round_id=round_id,
        build_tree_features=build_tree_features,
        active_features=active_features,
    )
    records: list[dict[str, Any]] = []
    spec_weights = np.asarray(sample_weight, dtype=np.float32).copy()
    prev_spec_oof = np.full(len(y), float(np.mean(y)), dtype=np.float32)
    for spec_i, spec in enumerate(specs, start=1):
        for fold in fold_cache:
            fold["sw_sub"] = spec_weights[fold["sub"]]
        tprint(
            f"    EBM round {round_id}: spec {spec_i}/{len(specs)} "
            f"l1={spec.get('reg_alpha')} l2={spec.get('reg_lambda')} "
            "using current model-distilled weights"
        )
        rec = _fit_ebm_spec_on_cache(
            cls,
            spec_i,
            spec,
            fold_cache,
            len(y),
            classifier,
            round_id,
        )
        if rec is None:
            continue
        records.append(rec)
        spec_oof = np.asarray(rec["oof"], dtype=np.float32)
        spec_weights = sample_weight * _compute_weight_distillation(
            y, spec_oof, prev_spec_oof, is_classifier=classifier
        )
        spec_weights = np.nan_to_num(
            spec_weights, nan=1.0, posinf=1.0, neginf=1.0
        ).astype(np.float32)
        spec_weights = spec_weights / max(float(np.mean(spec_weights)), 1e-6)
        prev_spec_oof = spec_oof
        tprint(
            f"      spec {spec_i}: updated next-spec weights "
            f"(mean={float(np.mean(spec_weights)):.3f}, "
            f"p90={float(np.percentile(spec_weights, 90)):.3f})"
        )
    all_shape_models: list[Any] = []
    for rec in records:
        all_shape_models.extend(rec.get("shape_models", []))
        tprint(
            f"      spec {rec['spec_i']}: J={rec['J_final']:.4f} "
            f"lift30={rec.get('lift30', 0.0):.3f} "
            f"ndcg20={rec.get('ndcg_at_20', rec.get('ndcg@20', 0.0)):.4f} "
            f"ece20={rec.get('ece_top20', rec.get('ece_at_20', np.nan)):.4f} "
            f"se={rec['J_se']:.4f}"
        )
    if not records:
        return {"ok": False}
    best = max(records, key=lambda r: float(r["J_final"]))
    return {
        "ok": True,
        "best": best,
        "records": records,
        "shape_models": all_shape_models,
    }


def _fit_final_model(
    cls: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray,
    selected_features: list[str],
    selected_feature_names: list[str] | None,
    mode: str,
    random_state: int,
    pruning_history: list[dict[str, Any]],
    oof_probs: np.ndarray,
    metrics: dict[str, Any],
    stage_indices: dict[str, np.ndarray] | None = None,
    timestamps: Any = None,
    assets: Any = None,
    tree_feature_bundle: dict[str, Any] | None = None,
    hpo_objective_mode: str = "base",
    hpo_trials_override: int | None = None,
    hpo_patience_override: int | None = None,
) -> EBMOnLGBMModel:
    tprint(
        f"EBMOnLGBM: full-data fit with {len(selected_features)} raw features "
        "plus tree-state features."
    )
    model = EBMOnLGBMModel(mode=mode)
    X_raw = X.reset_index(drop=True)
    n_all = len(X_raw)
    if stage_indices is None:
        all_idx = np.arange(n_all, dtype=np.int32)
        stage_indices = {"lgbm_prune": all_idx, "hpo": all_idx, "fit_oof": all_idx}
    lgbm_idx = np.asarray(stage_indices.get("lgbm_prune", []), dtype=np.int32)
    hpo_idx = np.asarray(stage_indices.get("hpo", []), dtype=np.int32)
    fit_idx = np.asarray(stage_indices.get("fit_oof", []), dtype=np.int32)
    lgbm_idx = lgbm_idx[(lgbm_idx >= 0) & (lgbm_idx < n_all)]
    hpo_idx = hpo_idx[(hpo_idx >= 0) & (hpo_idx < n_all)]
    fit_idx = fit_idx[(fit_idx >= 0) & (fit_idx < n_all)]
    if len(lgbm_idx) == 0:
        lgbm_idx = np.arange(n_all, dtype=np.int32)
    if len(hpo_idx) == 0:
        hpo_idx = np.arange(n_all, dtype=np.int32)
    if len(fit_idx) == 0:
        fit_idx = np.arange(n_all, dtype=np.int32)
    if EBM_FINAL_FIT_MAX_ROWS > 0 and len(fit_idx) > EBM_FINAL_FIT_MAX_ROWS:
        fit_local = _stratified_subsample_indices(
            np.asarray(y, dtype=np.float32)[fit_idx],
            max_n=EBM_FINAL_FIT_MAX_ROWS,
            random_state=random_state + 40711,
            classifier=(mode == "classifier"),
        )
        fit_idx = np.sort(fit_idx[fit_local].astype(np.int32))
        tprint(
            "EBMOnLGBM final fit: stratified capped fit_oof rows "
            f"to {len(fit_idx)}/{n_all} (cap={EBM_FINAL_FIT_MAX_ROWS})."
        )
    else:
        cap_label = EBM_FINAL_FIT_MAX_ROWS if EBM_FINAL_FIT_MAX_ROWS > 0 else "off"
        tprint(
            "EBMOnLGBM final fit: using "
            f"{len(fit_idx)}/{n_all} fit_oof rows (cap={cap_label})."
        )
    stability_groups = _stability_group_bundle(
        n_all, timestamps=timestamps, assets=assets
    )

    if tree_feature_bundle and tree_feature_bundle.get("oof_tree_features"):
        tree_bundle = dict(tree_feature_bundle)
        selected_tree_names = [
            str(c) for c in (selected_feature_names or []) if str(c).startswith("lgbm_")
        ]
        tree_df = _compute_oof_bundle_tree_frame(
            tree_bundle,
            X_raw,
            selected_tree_names=selected_tree_names
            or list(tree_bundle.get("tree_feature_names", [])),
        )
        Xs = pd.concat([X_raw, tree_df], axis=1)
        tprint(
            "EBMOnLGBM: full-data fit reusing pruning-stage OOF LGBM "
            f"bundle with {tree_df.shape[1]} tree-state features."
        )
    else:
        _, Xs, tree_bundle = _augment_with_tree_features(
            X_raw.iloc[lgbm_idx].reset_index(drop=True),
            y[lgbm_idx],
            X_raw,
            random_state=random_state + 9901,
        )
    if selected_feature_names:
        wanted = [str(c) for c in selected_feature_names]
        missing = [c for c in wanted if c not in Xs.columns]
        for col in missing:
            Xs[col] = 0.0
        final_active_cols = list(wanted)
        n_raw = int(sum(not c.startswith("lgbm_") for c in final_active_cols))
        n_tree = int(len(final_active_cols) - n_raw)
        if missing:
            tprint(
                "EBMOnLGBM: full-data fit could not regenerate "
                f"{len(missing)}/{len(wanted)} candidate-selected features; "
                "keeping the selected feature contract with zero-filled missing "
                "raw/tree features."
            )
        tprint(
            "EBMOnLGBM: full-data fit using candidate-selected feature set "
            f"before final contribution prune: {n_raw} raw + {n_tree} tree "
            f"features ({len(final_active_cols)} total)."
        )
    else:
        final_active_cols = list(Xs.columns)
    _log_selected_features("full-data candidate-selected", final_active_cols)
    Xs = Xs[final_active_cols]

    try:
        import optuna
        from optuna.pruners import MedianPruner

        HAS_OPTUNA = True
    except ImportError:
        HAS_OPTUNA = False

    final_spec = _ebm_specs(pruning=False, random_state=random_state)[0]
    hpo_selected_params: dict[str, Any] = dict(final_spec)
    final_base_sample_weight = np.asarray(sample_weight, dtype=np.float32).copy()
    hpo_sample_weight = np.asarray(sample_weight, dtype=np.float32).copy()
    weight_transfer_metrics: dict[str, float | str] = {}
    try:
        pre_hpo_spec = dict(final_spec)
        pre_hpo_spec["max_bins"] = min(int(pre_hpo_spec.get("max_bins", 48)), 48)
        pre_hpo_spec["outer_bags"] = 1
        pre_hpo_spec["smoothing_rounds"] = 100
        hpo_sample_weight, _pre_hpo_oof = _oof_distilled_sample_weights(
            cls,
            Xs[final_active_cols],
            y,
            sample_weight,
            fit_idx,
            pre_hpo_spec,
            {},
            mode,
            random_state=random_state + 22103,
            passes=EBM_OOF_DISTILLATION_PASSES,
            label="pre-HPO",
        )
        weight_transfer_metrics["hpo_weight_source"] = "fit_oof_oof_distilled"
        weight_transfer_metrics["hpo_weight_mean"] = float(np.mean(hpo_sample_weight))
        weight_transfer_metrics["hpo_weight_p90"] = float(
            np.percentile(hpo_sample_weight, 90)
        )
    except Exception as exc:
        hpo_sample_weight = np.asarray(sample_weight, dtype=np.float32).copy()
        weight_transfer_metrics["hpo_weight_source"] = "base_weight_fallback"
        tprint(f"EBMOnLGBM: pre-HPO OOF-only weight distillation skipped ({exc}).")

    post_hpo_shape_metrics: dict[str, float] = {}
    shape_smoothing_policy: dict[str, int] = {}

    if HAS_OPTUNA:
        hpo_trials = (
            int(hpo_trials_override)
            if hpo_trials_override is not None
            else int(os.environ.get("EPM_EBM_HPO_TRIALS", EBM_HPO_TRIALS))
        )
        hpo_patience = (
            int(hpo_patience_override)
            if hpo_patience_override is not None
            else int(
                os.environ.get(
                    "EPM_EBM_HPO_EARLY_STOP_PATIENCE", EBM_HPO_EARLY_STOP_PATIENCE
                )
            )
        )
        hpo_n_jobs = int(os.environ.get("EPM_EBM_HPO_N_JOBS", EBM_HPO_N_JOBS))
        hpo_folds = int(os.environ.get("EPM_EBM_HPO_FOLDS", 3))
        hpo_pruner_startup = int(os.environ.get("EPM_EBM_HPO_MEDIAN_STARTUP", 10))
        hpo_pruner_min_trials = int(os.environ.get("EPM_EBM_HPO_MEDIAN_MIN_TRIALS", 5))
        hpo_fold_timeout_sec = float(
            os.environ.get("EPM_EBM_HPO_FOLD_TIMEOUT_SEC", EBM_HPO_FOLD_TIMEOUT_SEC)
        )
        hpo_max_fold_timeouts = int(
            os.environ.get("EPM_EBM_HPO_MAX_FOLD_TIMEOUTS", EBM_HPO_MAX_FOLD_TIMEOUTS)
        )
        hpo_max_rounds = int(os.environ.get("EPM_EBM_HPO_MAX_ROUNDS", EBM_HPO_MAX_ROUNDS))
        hpo_validation_size = float(
            os.environ.get("EPM_EBM_HPO_VALIDATION_SIZE", EBM_HPO_VALIDATION_SIZE)
        )
        hpo_subprocess_timeout = (
            os.environ.get(
                "EPM_EBM_HPO_SUBPROCESS_TIMEOUT",
                "1" if EBM_HPO_SUBPROCESS_TIMEOUT else "0",
            )
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
        )
        hpo_trials = max(0, hpo_trials)
        hpo_patience = max(1, hpo_patience)
        hpo_n_jobs = min(1, max(1, hpo_n_jobs))
        hpo_folds = min(3, max(2, hpo_folds))
        hpo_pruner_startup = max(0, hpo_pruner_startup)
        hpo_pruner_min_trials = max(1, hpo_pruner_min_trials)
        hpo_fold_timeout_sec = max(0.0, hpo_fold_timeout_sec)
        hpo_max_fold_timeouts = max(1, hpo_max_fold_timeouts)
        hpo_max_rounds = int(np.clip(hpo_max_rounds, 100, 10000))
        hpo_validation_size = float(np.clip(hpo_validation_size, 0.05, 0.25))
        tprint(
            "  EBMOnLGBM: running Optuna HPO with Bends Analysis "
            f"(trials={hpo_trials}, early_stop_patience={hpo_patience}, "
            f"subsample=5000, folds={hpo_folds}, outer_bags=1, "
            f"inner_bags=0, validation_size={hpo_validation_size:.2f}, "
            f"max_rounds={hpo_max_rounds}, "
            f"trial_n_jobs={hpo_n_jobs}, fit_n_jobs=1, "
            f"fold_timeout={hpo_fold_timeout_sec:.0f}s, "
            f"subprocess_timeout={hpo_subprocess_timeout}, "
            f"max_fold_timeouts={hpo_max_fold_timeouts}, "
            "pruner=Median"
            f"(startup={hpo_pruner_startup}, min_trials={hpo_pruner_min_trials}))."
        )
        _quiet_interpret_logging()
        from interpret.glassbox import (
            ExplainableBoostingClassifier,
            ExplainableBoostingRegressor,
        )

        EBMCls = (
            ExplainableBoostingClassifier
            if mode == "classifier"
            else ExplainableBoostingRegressor
        )

        hpo_pool = np.asarray(hpo_idx, dtype=np.int32)
        n_sub = min(5000, len(hpo_pool))
        rng_hpo = np.random.default_rng(random_state + 44017)
        idx_sub = (
            rng_hpo.choice(hpo_pool, n_sub, replace=False)
            if len(hpo_pool) > n_sub
            else hpo_pool
        )
        Xs_sub = Xs.iloc[idx_sub].reset_index(drop=True)
        y_sub = y[idx_sub]
        sw_sub = (
            hpo_sample_weight[idx_sub]
            if hpo_sample_weight is not None
            else np.ones(n_sub)
        )

        from sklearn.model_selection import KFold

        kf = KFold(n_splits=hpo_folds, shuffle=True, random_state=random_state)

        global_hp_state = {
            "feature_j_scores": {c: 1.0 for c in Xs.columns},
        }
        hpo_runtime_state = {"n_jobs": hpo_n_jobs}
        hpo_log_interval = float(
            os.environ.get("EPM_EBM_HPO_PROGRESS_INTERVAL_SEC", "120")
        )
        hpo_log_interval = float(max(10.0, hpo_log_interval))
        hpo_progress_state = {
            "last_log_ts": 0.0,
            "started_trials": 0,
            "finished_trials": 0,
            "fold_timeouts": 0,
        }
        best_ndcg20 = float("-inf")
        best_ndcg20_se = 0.0
        best_precision15 = float("-inf")

        def _hpo_heartbeat(message: str, *, force: bool = False) -> None:
            now = time.monotonic()
            if force or (
                now - float(hpo_progress_state["last_log_ts"]) >= hpo_log_interval
            ):
                hpo_progress_state["last_log_ts"] = now
                tprint(f"  EBMOnLGBM: HPO {message}")

        def objective(trial):
            nonlocal best_ndcg20, best_ndcg20_se, best_precision15
            trial_start = time.monotonic()
            hpo_progress_state["started_trials"] = (
                int(hpo_progress_state["started_trials"]) + 1
            )
            lr = trial.suggest_categorical(
                "learning_rate", [0.01, 0.02, 0.03, 0.04, 0.05]
            )
            max_bins = trial.suggest_int("max_bins", 16, 48)
            smoothing_rounds = 100
            max_leaves = trial.suggest_int("max_leaves", 2, 3)
            reg_alpha = trial.suggest_float("reg_alpha", 0.01, 2.0, log=True)
            reg_lambda = trial.suggest_float("reg_lambda", 0.5, 10.0, log=True)
            greedy_ratio = trial.suggest_float("greedy_ratio", 0.1, 20.0, log=True)
            outer_bags = 1
            active_cols = [
                c
                for c in Xs_sub.columns
                if global_hp_state["feature_j_scores"][c] > 0.1
            ]
            if not active_cols:
                _hpo_heartbeat(
                    f"trial {trial.number} pruned before fit: no active features.",
                    force=True,
                )
                raise optuna.TrialPruned()
            leaf_min_pct = _hpo_leaf_min_pct_formula(
                max_leaves=max_leaves,
                max_bins=max_bins,
                smoothing_rounds=smoothing_rounds,
                n_features=len(active_cols),
            )
            trial.set_user_attr("leaf_min_pct", float(leaf_min_pct))
            trial.set_user_attr(
                "min_samples_leaf_sub", int(max(1, int(leaf_min_pct * n_sub)))
            )
            trial.set_user_attr("active_feature_count", int(len(active_cols)))

            spec = {
                "learning_rate": lr,
                "max_bins": max_bins,
                "smoothing_rounds": smoothing_rounds,
                "max_leaves": max_leaves,
                "reg_alpha": reg_alpha,
                "reg_lambda": reg_lambda,
                "greedy_ratio": greedy_ratio,
                "outer_bags": outer_bags,
                "inner_bags": 0,
                "validation_size": hpo_validation_size,
                "max_rounds": hpo_max_rounds,
                "min_samples_leaf": max(1, int(leaf_min_pct * n_sub)),
                "early_stopping_rounds": 30,
                "interactions": 0,
                "n_jobs": 1,
            }
            _hpo_heartbeat(
                f"trial {trial.number} start "
                f"(started={hpo_progress_state['started_trials']}, "
                f"finished={hpo_progress_state['finished_trials']}, "
                f"active_features={len(active_cols)}, "
                f"leaf_min_pct={leaf_min_pct:.4f}, "
                f"min_leaf={spec['min_samples_leaf']}, lr={lr:.4f}, "
                f"bins={max_bins}, leaves={max_leaves}, "
                f"smooth={smoothing_rounds}, l1={reg_alpha:.4g}, "
                f"l2={reg_lambda:.4g}, greedy={greedy_ratio:.4g}).",
                force=trial.number < 3,
            )

            fold_metrics = []
            for step, (tr, va) in enumerate(kf.split(Xs_sub)):
                fold_start = time.monotonic()
                _hpo_heartbeat(
                    f"trial {trial.number} fold {step + 1}/{hpo_folds} fitting "
                    f"(train={len(tr)}, valid={len(va)}).",
                    force=trial.number < 3,
                )
                try:
                    if hpo_subprocess_timeout:
                        pred = _fit_predict_hpo_fold_with_timeout(
                            mode=mode,
                            params=spec,
                            x_train=Xs_sub.iloc[tr][active_cols].reset_index(
                                drop=True
                            ),
                            y_train=y_sub[tr],
                            sample_weight=sw_sub[tr],
                            x_valid=Xs_sub.iloc[va][active_cols].reset_index(
                                drop=True
                            ),
                            timeout_sec=hpo_fold_timeout_sec,
                        )
                    else:
                        model = _fit_one_ebm(
                            EBMCls,
                            Xs_sub.iloc[tr][active_cols].reset_index(drop=True),
                            y_sub[tr],
                            np.asarray(sw_sub[tr], dtype=np.float32),
                            spec,
                        )
                        pred = (
                            model.predict_proba(
                                _coerce_ebm_feature_types(
                                    Xs_sub.iloc[va][active_cols].reset_index(
                                        drop=True
                                    )
                                )
                            )[:, 1]
                            if mode == "classifier"
                            else model.predict(
                                _coerce_ebm_feature_types(
                                    Xs_sub.iloc[va][active_cols].reset_index(
                                        drop=True
                                    )
                                )
                            )
                        )
                        pred = np.asarray(pred, dtype=np.float32)
                except TimeoutError as exc:
                    hpo_progress_state["fold_timeouts"] = (
                        int(hpo_progress_state["fold_timeouts"]) + 1
                    )
                    timeout_count = int(hpo_progress_state["fold_timeouts"])
                    tprint(
                        "  EBMOnLGBM: HPO fold timeout "
                        f"{timeout_count}/{hpo_max_fold_timeouts} on trial "
                        f"{trial.number} fold {step + 1}/{hpo_folds}: {exc}"
                    )
                    if timeout_count >= hpo_max_fold_timeouts:
                        tprint(
                            "  EBMOnLGBM: stopping HPO after repeated native EBM "
                            "fold timeouts; keeping best completed parameters."
                        )
                        study.stop()
                    raise optuna.TrialPruned()
                except PermissionError as exc:
                    if int(spec.get("n_jobs", 1)) <= 1:
                        raise
                    tprint(
                        "  EBMOnLGBM: HPO EBM parallel fit hit macOS semaphore "
                        f"limit ({exc}); retrying this fold with n_jobs=1."
                    )
                    spec["n_jobs"] = 1
                    hpo_runtime_state["n_jobs"] = 1
                    m = EBMCls(
                        **{k: v for k, v in spec.items() if k in EBMCls().get_params()}
                    )
                    m.fit(
                        Xs_sub.iloc[tr][active_cols],
                        y_sub[tr],
                        sample_weight=sw_sub[tr],
                    )
                    pred = (
                        m.predict_proba(Xs_sub.iloc[va][active_cols])[:, 1]
                        if mode == "classifier"
                        else m.predict(Xs_sub.iloc[va][active_cols])
                    )

                metrics = _metric_pack(
                    y_sub[va],
                    pred,
                    classifier=(mode == "classifier"),
                    groups=_groups_take(_groups_take(stability_groups, idx_sub), va),
                )
                fold_metrics.append(metrics)
                agg = _aggregate_j(fold_metrics)
                agg["pareto_ndcg20_floor"] = float(
                    max(0.0, best_ndcg20 - 0.75 * best_ndcg20_se)
                )
                agg["pareto_precision_floor"] = float(max(0.0, best_precision15))
                score = _hpo_objective_from_aggregate(agg, hpo_objective_mode)

                trial.report(score, step)
                if trial.should_prune():
                    _hpo_heartbeat(
                        f"trial {trial.number} pruned after fold "
                        f"{step + 1}/{hpo_folds} "
                        f"(score={score:.4f}, "
                        f"elapsed={time.monotonic() - trial_start:.1f}s).",
                        force=True,
                    )
                    raise optuna.TrialPruned()
                _hpo_heartbeat(
                    f"trial {trial.number} fold {step + 1}/{hpo_folds} done "
                    f"(score={score:.4f}, "
                    f"fold_time={time.monotonic() - fold_start:.1f}s, "
                    f"elapsed={time.monotonic() - trial_start:.1f}s).",
                    force=trial.number < 3,
                )

            agg = _aggregate_j(fold_metrics)
            _ndcg20_curr = float(agg.get("ndcg_at_20", agg.get("ndcg@20", 0.0)))
            _n_folds_curr = max(1, int(len(fold_metrics)))
            _se_curr = float(agg.get("ndcg20_std", 0.0)) / float(np.sqrt(_n_folds_curr))
            if np.isfinite(_ndcg20_curr) and (
                _ndcg20_curr > best_ndcg20
                or (np.isclose(_ndcg20_curr, best_ndcg20) and _se_curr < best_ndcg20_se)
            ):
                best_ndcg20 = _ndcg20_curr
                best_ndcg20_se = _se_curr
            _precision15_curr = float(
                agg.get("precision15", agg.get("hit_rate15", float("nan")))
            )
            if np.isfinite(_precision15_curr):
                best_precision15 = max(best_precision15, _precision15_curr)
            agg["pareto_ndcg20_floor"] = float(
                max(0.0, best_ndcg20 - 0.75 * best_ndcg20_se)
            )
            agg["pareto_precision_floor"] = float(max(0.0, best_precision15))
            hpo_objective = _hpo_objective_from_aggregate(agg, hpo_objective_mode)
            trial.set_user_attr("lift15", float(agg.get("lift15", np.nan)))
            trial.set_user_attr("lift30", float(agg.get("lift30", np.nan)))
            trial.set_user_attr("stability15", float(agg.get("stability15", np.nan)))
            trial.set_user_attr("stability30", float(agg.get("stability30", np.nan)))
            trial.set_user_attr(
                "precision15",
                float(agg.get("precision15", agg.get("hit_rate15", np.nan))),
            )
            trial.set_user_attr(
                "precision01",
                float(agg.get("precision01", agg.get("precision1", np.nan))),
            )
            trial.set_user_attr("precision005", float(agg.get("precision005", np.nan)))
            trial.set_user_attr(
                "ndcg_at_10", float(agg.get("ndcg_at_10", agg.get("ndcg@10", np.nan)))
            )
            trial.set_user_attr(
                "ndcg_at_20", float(agg.get("ndcg_at_20", agg.get("ndcg@20", np.nan)))
            )
            trial.set_user_attr(
                "ece_top20", float(agg.get("ece_top20", agg.get("ece_at_20", np.nan)))
            )
            trial.set_user_attr(
                "ece_at_20", float(agg.get("ece_at_20", agg.get("ece_top20", np.nan)))
            )
            trial.set_user_attr(
                "ndcg_tradable_top20pct",
                float(agg.get("ndcg_tradable_top20pct", np.nan)),
            )
            trial.set_user_attr(
                "ndcg_tradable_top10pct",
                float(agg.get("ndcg_tradable_top10pct", np.nan)),
            )
            trial.set_user_attr("ndcg20_std", float(agg.get("ndcg20_std", np.nan)))
            trial.set_user_attr("ndcg20_p25", float(agg.get("ndcg20_p25", np.nan)))
            trial.set_user_attr(
                "pareto_ndcg20_floor",
                float(agg.get("pareto_ndcg20_floor", np.nan)),
            )
            trial.set_user_attr(
                "pareto_precision_floor",
                float(agg.get("pareto_precision_floor", np.nan)),
            )
            trial.set_user_attr("J_final", float(agg.get("J_final", np.nan)))
            trial.set_user_attr("hpo_objective", float(hpo_objective))
            hpo_progress_state["finished_trials"] = (
                int(hpo_progress_state["finished_trials"]) + 1
            )
            _hpo_heartbeat(
                f"trial {trial.number} complete "
                f"(value={hpo_objective:.4f}, "
                f"finished={hpo_progress_state['finished_trials']}/{hpo_trials}, "
                f"elapsed={time.monotonic() - trial_start:.1f}s).",
                force=trial.number < 3,
            )
            return float(hpo_objective)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize",
            pruner=MedianPruner(
                n_startup_trials=hpo_pruner_startup,
                n_warmup_steps=0,
                interval_steps=1,
                n_min_trials=hpo_pruner_min_trials,
            ),
        )
        enqueued_hpo_params: list[dict[str, Any]] = []

        def _enqueue_hpo_trial(params: dict[str, Any], label: str) -> None:
            sanitized = _sanitize_ebm_hpo_trial_params(params)
            if not sanitized:
                return
            if any(sanitized == existing for existing in enqueued_hpo_params):
                return
            study.enqueue_trial(sanitized)
            enqueued_hpo_params.append(sanitized)
            if label:
                tprint(f"  EBMOnLGBM: enqueued HPO warm-start trial ({label}).")

        warm_params, warm_path = _load_ebm_hpo_warm_start()
        if hpo_trials > 0 and warm_params is not None:
            _enqueue_hpo_trial(warm_params, str(warm_path))
        if hpo_trials > 0:
            _enqueue_hpo_trial(
                {
                    "learning_rate": 0.05,
                    "max_bins": 32,
                    "smoothing_rounds": 100,
                    "max_leaves": 3,
                    "reg_alpha": 0.1,
                    "reg_lambda": 3.0,
                    "greedy_ratio": 0.1,
                },
                "",
            )
        if hpo_trials > 1:
            _enqueue_hpo_trial(
                {
                    "learning_rate": 0.01,
                    "max_bins": 48,
                    "smoothing_rounds": 100,
                    "max_leaves": 3,
                    "reg_alpha": 1.0,
                    "reg_lambda": 6.0,
                    "greedy_ratio": 1.0,
                },
                "",
            )

        hpo_state = {
            "best_value": -np.inf,
            "best_trial": -1,
            "completed_since_improvement": 0,
        }

        def _log_hpo_progress(study, trial) -> None:
            from optuna.trial import TrialState

            if trial.state == TrialState.COMPLETE and trial.value is not None:
                value = float(trial.value)
                if value > float(hpo_state["best_value"]) + 1e-12:
                    hpo_state["best_value"] = value
                    hpo_state["best_trial"] = int(trial.number)
            if int(hpo_state["best_trial"]) >= 0:
                hpo_state["completed_since_improvement"] = max(
                    0, int(trial.number) - int(hpo_state["best_trial"])
                )
            n_done = len(study.trials)
            if n_done == 1 or n_done % 5 == 0 or n_done >= hpo_trials:
                best = study.best_trial if len(study.trials) else None
                best_msg = (
                    f"best_trial={best.number} best_value={best.value:.4f}"
                    if best is not None and best.value is not None
                    else "best_trial=none"
                )
                tprint(
                    "  EBMOnLGBM: HPO progress "
                    f"{n_done}/{hpo_trials} trials, {best_msg}, "
                    "completed_since_improvement="
                    f"{hpo_state['completed_since_improvement']}."
                )
            if int(hpo_state["completed_since_improvement"]) >= hpo_patience:
                tprint(
                    "  EBMOnLGBM: stopping HPO after "
                    f"{hpo_state['completed_since_improvement']} completed trials "
                    "without improvement."
                )
                study.stop()

        if hpo_trials <= 0:
            tprint("  EBMOnLGBM: Optuna HPO disabled by EPM_EBM_HPO_TRIALS=0.")
            HAS_OPTUNA = False
        else:
            study.optimize(
                objective,
                n_trials=hpo_trials,
                n_jobs=hpo_n_jobs,
                callbacks=[_log_hpo_progress],
            )

        if HAS_OPTUNA:
            from optuna.trial import TrialState

            hpo_state_counts = {
                state.name.lower(): int(
                    sum(1 for trial in study.trials if trial.state == state)
                )
                for state in (
                    TrialState.COMPLETE,
                    TrialState.PRUNED,
                    TrialState.FAIL,
                )
            }
            completed_trials = [
                trial
                for trial in study.trials
                if trial.state == TrialState.COMPLETE and trial.value is not None
            ]
            if not completed_trials:
                tprint(
                    "  EBMOnLGBM: HPO produced no completed trials "
                    f"(trial_states={hpo_state_counts}); keeping default final "
                    f"params={final_spec}."
                )
                best_trial = None
            else:
                best_trial = study.best_trial
            if best_trial is None:
                hpo_selected_params = dict(final_spec)
            else:
                best_spec = dict(best_trial.params)
                best_leaf_min_pct = float(
                    best_trial.user_attrs.get(
                        "leaf_min_pct",
                        _hpo_leaf_min_pct_formula(
                            max_leaves=int(best_spec.get("max_leaves", 3)),
                            max_bins=int(best_spec.get("max_bins", 48)),
                            smoothing_rounds=int(best_spec.get("smoothing_rounds", 25)),
                            n_features=len(Xs.columns),
                        ),
                    ),
                )
                best_spec["min_samples_leaf"] = max(
                    1, int(best_leaf_min_pct * len(fit_idx))
                )
                best_spec["min_samples_leaf_pct"] = best_leaf_min_pct
                best_spec["outer_bags"] = 10
                best_spec["smoothing_rounds"] = 400
                best_spec["early_stopping_rounds"] = 30
                best_spec["interactions"] = 0
                best_spec["n_jobs"] = 1
                final_spec = best_spec
                hpo_selected_params = dict(best_spec)
                _save_ebm_hpo_warm_start(
                    best_params=best_spec,
                    best_value=float(best_trial.value),
                    best_trial_number=int(best_trial.number),
                    best_trial_attrs=dict(best_trial.user_attrs),
                    leaf_min_pct=best_leaf_min_pct,
                )
                tprint(
                    "  EBMOnLGBM: HPO complete. "
                    "trial_states="
                    f"{hpo_state_counts}, "
                    f"Best trial={best_trial.number}, value={best_trial.value:.4f}, "
                    f"lift30={float(best_trial.user_attrs.get('lift30', np.nan)):.4f}, "
                    "stability30="
                    f"{float(best_trial.user_attrs.get('stability30', np.nan)):.4f}, "
                    f"ndcg20={float(best_trial.user_attrs.get('ndcg_at_20', np.nan)):.4f}, "
                    f"ece20={float(best_trial.user_attrs.get('ece_top20', np.nan)):.4f}, "
                    f"objective_mode={hpo_objective_mode}, "
                    f"leaf_min_pct={best_leaf_min_pct:.4f}, final params={best_spec}."
                )

    (
        final_active_cols,
        shape_smoothing_policy,
        post_hpo_shape_metrics,
    ) = _post_hpo_manage_features(
        cls,
        Xs,
        y,
        hpo_sample_weight,
        final_active_cols,
        final_spec,
        mode,
    )

    try:
        final_sample_weight, _ = _oof_distilled_sample_weights(
            cls,
            Xs[final_active_cols],
            y,
            sample_weight,
            fit_idx,
            final_spec,
            shape_smoothing_policy,
            mode,
            random_state=random_state + 33107,
            passes=EBM_OOF_DISTILLATION_PASSES,
            label="final",
        )
        weight_transfer_metrics["final_weight_source"] = "fit_oof_oof_distilled"
        weight_transfer_metrics["final_weight_mean"] = float(
            np.mean(final_sample_weight)
        )
        weight_transfer_metrics["final_weight_p90"] = float(
            np.percentile(final_sample_weight, 90)
        )
    except Exception as exc:
        final_sample_weight = np.asarray(sample_weight, dtype=np.float32).copy()
        weight_transfer_metrics["final_weight_source"] = "base_weight_fallback"
        tprint(f"EBMOnLGBM: final OOF-only weight distillation skipped ({exc}).")

    for i in range(1):
        t0 = time.perf_counter()
        final_fit_rows = int(len(fit_idx))
        final_fit_features = int(len(final_active_cols))
        tprint(
            "EBMOnLGBM: starting final EBM fit "
            f"(rows={final_fit_rows}, features={final_fit_features}, "
            f"outer_bags={int(final_spec.get('outer_bags', 1))}, "
            f"smoothing_rounds={int(final_spec.get('smoothing_rounds', 0))}, "
            f"max_rounds={int(final_spec.get('max_rounds', 0))})."
        )
        stop_heartbeat = _start_stage_heartbeat("final EBM fit", interval_sec=60.0)
        try:
            ebm = _fit_one_ebm(
                cls,
                Xs.iloc[fit_idx][final_active_cols].reset_index(drop=True),
                y[fit_idx],
                final_sample_weight[fit_idx],
                final_spec,
            )
        finally:
            stop_heartbeat()
        _apply_shape_smoothing_policy(
            ebm, final_active_cols, shape_smoothing_policy, log=True
        )
        X_fit_final = Xs.iloc[fit_idx][final_active_cols].reset_index(drop=True)
        raw = _predict_raw_ebm(ebm, X_fit_final, mode)
        pp = SplinePostProcessor(mode).fit(raw, y[fit_idx], use_dynamic_smoothing=True)
        model.models.append(ebm)
        model.postprocessors.append(pp)
        tprint(f"  Final EBM fit in {time.perf_counter() - t0:.1f}s")

    model.raw_selected_features = list(X_raw.columns)
    model.tree_models = list(tree_bundle.get("models", []))
    model.tree_feature_config = dict(tree_bundle.get("tree_feature_config", {}))
    for key in (
        "oof_tree_features",
        "models_by_fold",
        "tree_feature_names_by_fold",
        "tree_feature_scales_by_fold",
    ):
        if key in tree_bundle:
            model.tree_feature_config[key] = tree_bundle[key]
    model.tree_feature_scales = tree_bundle.get("tree_feature_scales")
    model.tree_feature_names = [
        str(c) for c in final_active_cols if str(c).startswith("lgbm_")
    ]
    model.selected_features = final_active_cols
    model.selected_indices = np.array([], dtype=np.int32)
    model.oof_probs = _final_stage_oof_predictions(
        cls,
        Xs[final_active_cols],
        y,
        final_sample_weight,
        fit_idx,
        final_spec,
        shape_smoothing_policy,
        mode,
        random_state=random_state + 11701,
        use_spec_outer_bags=True,
    )
    if model.oof_probs is None or len(model.oof_probs) != len(y):
        model.oof_probs = np.asarray(oof_probs, dtype=np.float32)
    pre_distill_oof_probs = _final_stage_oof_predictions(
        cls,
        Xs[final_active_cols],
        y,
        final_base_sample_weight,
        fit_idx,
        final_spec,
        shape_smoothing_policy,
        mode,
        random_state=random_state + 11701,
    )
    uncertainty_metrics: dict[str, float] = {}
    try:
        model.oof_probs_raw_ebm = np.asarray(model.oof_probs, dtype=np.float32).copy()
        if (
            EBM_FINAL_UNCERTAINTY_MAX_ROWS > 0
            and len(y) > EBM_FINAL_UNCERTAINTY_MAX_ROWS
        ):
            model.oof_probs_en = model.oof_probs_raw_ebm.copy()
            model.oof_probs_uncertainty_weighted = model.oof_probs_raw_ebm.copy()
            uncertainty_metrics["uncertainty_oof_skipped_large_n"] = 1.0
            uncertainty_metrics["uncertainty_oof_n"] = float(len(y))
            uncertainty_metrics["uncertainty_oof_cap"] = float(
                EBM_FINAL_UNCERTAINTY_MAX_ROWS
            )
            tprint(
                "EBMOnLGBM: uncertainty OOF feature generation skipped for "
                f"large final fit n={len(y)} > cap={EBM_FINAL_UNCERTAINTY_MAX_ROWS}; "
                "using raw EBM OOF probabilities for uncertainty-adjusted outputs."
            )
        else:
            model.uncertainty_state = fit_uncertainty_state(
                Xs.iloc[fit_idx][final_active_cols].reset_index(drop=True),
                final_active_cols,
                max_bins=32,
            )
            unc_features = compute_uncertainty_features(
                Xs[final_active_cols],
                model.models,
                mode,
                _predict_raw_ebm,
                state=model.uncertainty_state,
            )
            model.oof_uncertainty_features = {
                c: unc_features[c].to_numpy(dtype=np.float32)
                for c in unc_features.columns
            }
            if mode == "classifier":
                model.en_adjuster = fit_en_uncertainty_adjuster(
                    model.oof_probs_raw_ebm,
                    y,
                    unc_features,
                    groups=_groups_primary(stability_groups),
                    random_state=random_state + 55021,
                    n_trials=30,
                )
                if model.en_adjuster is not None:
                    model.oof_probs_en = model.en_adjuster.predict(
                        model.oof_probs_raw_ebm, unc_features
                    )
                else:
                    model.oof_probs_en = model.oof_probs_raw_ebm.copy()
                model.oof_probs_uncertainty_weighted = uncertainty_weighted_prediction(
                    model.oof_probs_raw_ebm,
                    unc_features,
                    model.oof_probs_en,
                )
            else:
                model.oof_probs_en = model.oof_probs_raw_ebm.copy()
                model.oof_probs_uncertainty_weighted = model.oof_probs_raw_ebm.copy()
            uncertainty_metrics["uncertainty_oof_skipped_large_n"] = 0.0
            uncertainty_metrics["uncertainty_oof_n"] = float(len(y))
            tprint(
                "EBMOnLGBM: generated uncertainty OOF features "
                f"({len(model.oof_uncertainty_features)} columns)."
            )
    except Exception as exc:
        tprint(f"EBMOnLGBM: uncertainty feature generation skipped ({exc}).")
    model.metrics = dict(metrics)
    model.metrics.update(uncertainty_metrics)
    model.metrics.update(post_hpo_shape_metrics)
    model.metrics.update(weight_transfer_metrics)
    for key in (
        "J_final_oof",
        "J_Score",
        "lift30",
        "stability30",
        "auc",
        "brier",
        "ece",
        "ndcg_at_20",
        "ece_top20",
        "ic_total",
        "ic_top30",
    ):
        if key in metrics:
            model.metrics[f"candidate_prune_{key}"] = metrics[key]
    try:
        pre_distill_fit_oof_pred = np.asarray(pre_distill_oof_probs, dtype=np.float32)[
            fit_idx
        ]
        fit_oof_y = np.asarray(y, dtype=np.float32)[fit_idx]
        pre_distill_mask = np.isfinite(pre_distill_fit_oof_pred) & np.isfinite(
            fit_oof_y
        )
        if int(np.sum(pre_distill_mask)) >= 8:
            pre_distill_fit_oof_metrics = _metric_pack(
                fit_oof_y[pre_distill_mask],
                pre_distill_fit_oof_pred[pre_distill_mask],
                classifier=(mode == "classifier"),
                groups=_groups_take(
                    _groups_take(stability_groups, fit_idx), pre_distill_mask
                ),
            )
            pre_distill_fit_oof_metrics.update(
                _aggregate_j([pre_distill_fit_oof_metrics])
            )
            for key, value in pre_distill_fit_oof_metrics.items():
                model.metrics[f"fit_oof_pre_distill_{key}"] = float(value)

        fit_oof_pred = np.asarray(model.oof_probs, dtype=np.float32)[fit_idx]
        fit_oof_mask = np.isfinite(fit_oof_pred) & np.isfinite(fit_oof_y)
        if int(np.sum(fit_oof_mask)) >= 8:
            fit_oof_metrics = _metric_pack(
                fit_oof_y[fit_oof_mask],
                fit_oof_pred[fit_oof_mask],
                classifier=(mode == "classifier"),
                groups=_groups_take(
                    _groups_take(stability_groups, fit_idx), fit_oof_mask
                ),
            )
            fit_oof_metrics.update(_aggregate_j([fit_oof_metrics]))
            for key, value in fit_oof_metrics.items():
                model.metrics[key] = value
                model.metrics[f"fit_oof_{key}"] = value
            model.metrics["J_final_oof"] = float(
                fit_oof_metrics.get("J_final", fit_oof_metrics.get("lift30", 0.0))
            )
            model.metrics["J_Score"] = float(model.metrics["J_final_oof"])
            model.metrics["metrics_assessment_slice"] = "fit_oof"
            model.metrics["metrics_assessment_n"] = int(np.sum(fit_oof_mask))
        else:
            model.metrics["metrics_assessment_slice"] = "candidate_prune_eval"
            model.metrics["metrics_assessment_n"] = int(np.sum(fit_oof_mask))
    except Exception as exc:
        model.metrics["metrics_assessment_slice"] = "candidate_prune_eval"
        tprint(f"EBMOnLGBM: fit_oof metrics skipped ({exc}).")
    for key in (
        "J_final_oof",
        "lift30",
        "stability30",
        "auc",
        "brier",
        "ece",
        "ndcg_at_20",
        "ece_top20",
        "ic_total",
        "ic_top30",
    ):
        before = model.metrics.get(f"candidate_prune_{key}")
        after = model.metrics.get(key)
        if before is None or after is None:
            continue
        try:
            before_f = float(before)
            after_f = float(after)
        except Exception:
            continue
        model.metrics[f"metric_stage_prune_{key}"] = before_f
        model.metrics[f"metric_stage_fit_oof_{key}"] = after_f
        model.metrics[f"metric_stage_delta_{key}"] = after_f - before_f
    for key in (
        "J_final_oof",
        "lift30",
        "stability30",
        "auc",
        "brier",
        "ece",
        "ndcg_at_20",
        "ece_top20",
        "ic_total",
        "ic_top30",
    ):
        before = model.metrics.get(f"fit_oof_pre_distill_{key}")
        after = model.metrics.get(f"fit_oof_{key}")
        if before is None or after is None:
            continue
        try:
            before_f = float(before)
            after_f = float(after)
        except Exception:
            continue
        model.metrics[f"metric_fit_oof_pre_distill_{key}"] = before_f
        model.metrics[f"metric_fit_oof_after_distill_{key}"] = after_f
        model.metrics[f"metric_fit_oof_distill_delta_{key}"] = after_f - before_f
    if "metric_stage_prune_lift30" in model.metrics:
        stage_metrics = (
            ("lift30", "lift30", "lift30"),
            ("stability30", "stability30", "stability30"),
            ("ndcg_at_20", "ndcg@20", "ndcg20"),
            ("ece_top20", "ece@20", "ece20"),
            ("auc", "AUC", "auc"),
            ("brier", "Brier", "brier"),
            ("ece", "ECE", "ece"),
            ("J_final_oof", "J_final_oof", "J_final"),
            ("ic_total", "IC_total", "IC"),
            ("ic_top30", "IC_top30", "IC_top30"),
            ("J_Score", "J_Score", "J_Score"),
        )
        rows: list[tuple[str, float, float, float]] = []
        for metric_key, label, _ in stage_metrics:
            before_key = f"metric_stage_prune_{metric_key}"
            after_key = f"metric_stage_fit_oof_{metric_key}"
            before = model.metrics.get(before_key)
            after = model.metrics.get(after_key)
            if before is None or after is None:
                continue
            try:
                before_v = float(before)
                after_v = float(after)
            except Exception:
                continue
            rows.append((label, before_v, after_v, after_v - before_v))
        if rows:
            metric_name_w = max(len("Metric"), max(len(r[0]) for r in rows))
            val_w = 14
            lines = [
                "EBMOnLGBM stage metric comparison (prune -> fit_oof)",
                (
                    f"{'Metric':<{metric_name_w}} | "
                    f"{'Before':>{val_w}} | "
                    f"{'After':>{val_w}} | "
                    f"{'Delta':>{val_w}}"
                ),
                "-" * (metric_name_w + val_w * 3 + 9),
            ]
            for metric_name, before_v, after_v, delta_v in rows:
                lines.append(
                    f"{metric_name:<{metric_name_w}} | "
                    f"{before_v:>{val_w}.4f} | "
                    f"{after_v:>{val_w}.4f} | "
                    f"{delta_v:>{val_w}.4f}"
                )
            for line in lines:
                tprint(line)
    if "metric_fit_oof_pre_distill_lift30" in model.metrics:
        distill_metrics = (
            ("lift30", "lift30", "lift30"),
            ("stability30", "stability30", "stability30"),
            ("ndcg_at_20", "ndcg@20", "ndcg20"),
            ("ece_top20", "ece@20", "ece20"),
            ("auc", "AUC", "auc"),
            ("brier", "Brier", "brier"),
            ("ece", "ECE", "ece"),
            ("J_final_oof", "J_final_oof", "J_final"),
            ("ic_total", "IC_total", "IC"),
            ("ic_top30", "IC_top30", "IC_top30"),
            ("J_Score", "J_Score", "J_Score"),
        )
        distill_rows: list[tuple[str, float, float, float]] = []
        for metric_key, label, _ in distill_metrics:
            before_key = f"metric_fit_oof_pre_distill_{metric_key}"
            after_key = f"metric_fit_oof_after_distill_{metric_key}"
            before = model.metrics.get(before_key)
            after = model.metrics.get(after_key)
            if before is None or after is None:
                continue
            try:
                before_v = float(before)
                after_v = float(after)
            except Exception:
                continue
            distill_rows.append((label, before_v, after_v, after_v - before_v))
        if distill_rows:
            metric_name_w = max(len("Metric"), max(len(r[0]) for r in distill_rows))
            val_w = 14
            lines = [
                "EBMOnLGBM fit_oof metric comparison (before self-distillation -> after self-distillation)",
                (
                    f"{'Metric':<{metric_name_w}} | "
                    f"{'Before':>{val_w}} | "
                    f"{'After':>{val_w}} | "
                    f"{'Delta':>{val_w}}"
                ),
                "-" * (metric_name_w + val_w * 3 + 9),
            ]
            for metric_name, before_v, after_v, delta_v in distill_rows:
                lines.append(
                    f"{metric_name:<{metric_name_w}} | "
                    f"{before_v:>{val_w}.4f} | "
                    f"{after_v:>{val_w}.4f} | "
                    f"{delta_v:>{val_w}.4f}"
                )
            for line in lines:
                tprint(line)
    model.metrics["feature_count"] = int(len(final_active_cols))
    model.metrics["n_raw_features_kept"] = int(
        sum(not c.startswith("lgbm_") for c in final_active_cols)
    )
    model.metrics["n_leaf_features_kept"] = int(
        sum(c.startswith("lgbm_") for c in final_active_cols)
    )
    model.metrics["final_fit_row_cap"] = int(EBM_FINAL_FIT_MAX_ROWS)
    model.metrics["final_fit_train_rows"] = int(len(fit_idx))
    model.metrics["final_fit_train_rows_total"] = int(n_all)
    model.metrics["final_outer_bags"] = int(final_spec.get("outer_bags", 0))
    model.metrics["best_params"] = dict(final_spec)
    model.metrics["final_fit_params"] = dict(final_spec)
    model.metrics["hpo_selected_params"] = dict(hpo_selected_params)
    model.metrics["final_fit_used_hpo_params"] = (
        model.metrics["final_fit_params"] == model.metrics["hpo_selected_params"]
    )
    tprint(
        "EBMOnLGBM final fit: "
        f"used_hpo_params={model.metrics['final_fit_used_hpo_params']} "
        f"params={json.dumps(model.metrics['final_fit_params'], sort_keys=True)}"
    )
    model.metrics["shape_smoothing_policy"] = dict(shape_smoothing_policy)
    model.metrics["postprocessor_calibration_method"] = (
        model.postprocessors[0].calibration_method
        if model.postprocessors
        else "identity"
    )
    if model.en_adjuster is not None:
        model.metrics["en_uncertainty_alpha"] = float(model.en_adjuster.alpha)
        model.metrics["en_uncertainty_l1_ratio"] = float(model.en_adjuster.l1_ratio)
        model.metrics["en_uncertainty_blend"] = float(model.en_adjuster.blend)
        for key, value in model.en_adjuster.metrics.items():
            model.metrics[f"en_uncertainty_{key}"] = float(value)
    try:
        final_metric_idx: np.ndarray | slice
        if len(y) > EBM_FINAL_FIT_METRIC_MAX_ROWS > 0:
            rng = np.random.default_rng(random_state + 19191)
            final_metric_idx = np.sort(
                rng.choice(
                    len(y),
                    size=EBM_FINAL_FIT_METRIC_MAX_ROWS,
                    replace=False,
                ).astype(np.int32)
            )
            tprint(
                "EBMOnLGBM: final-fit diagnostic metrics using deterministic "
                f"sample {len(final_metric_idx)}/{len(y)} rows."
            )
        else:
            final_metric_idx = slice(None)
            tprint(
                "EBMOnLGBM: final-fit diagnostic metrics using full " f"{len(y)} rows."
            )
        if isinstance(final_metric_idx, slice):
            X_final_metric = X_raw
            y_final_metric = y
            groups_final_metric = stability_groups
        else:
            X_final_metric = X_raw.iloc[final_metric_idx].reset_index(drop=True)
            y_final_metric = y[final_metric_idx]
            groups_final_metric = _groups_take(stability_groups, final_metric_idx)
        final_fit_pred = model.predict(X_final_metric)
        final_fit_metrics = _metric_pack(
            y_final_metric,
            final_fit_pred,
            classifier=(mode == "classifier"),
            groups=groups_final_metric,
        )
        final_fit_metrics.update(_aggregate_j([final_fit_metrics]))
        for key, value in final_fit_metrics.items():
            model.metrics[f"final_fit_{key}"] = value
        model.metrics["final_fit_metric_n"] = int(len(y_final_metric))
        model.metrics["final_fit_metric_sampled"] = bool(
            not isinstance(final_metric_idx, slice)
        )
    except Exception as exc:
        tprint(f"EBMOnLGBM: final-fit metrics skipped ({exc}).")
    model.pruning_history = list(pruning_history)
    return model


def train_ebm_on_lgbm_candidate(
    X: Any,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    random_state: int = 42,
    mode: str = "classifier",
    timestamps: Any = None,
    assets: Any = None,
    hpo_objective_mode: str = "base",
) -> Optional[dict[str, Any]]:
    tprint("EBMOnLGBM: starting EBM candidate training.")
    t0_total = time.perf_counter()
    classifier = mode == "classifier"
    ebm_clf, ebm_reg = _load_ebm_classes()
    cls = ebm_clf if classifier else ebm_reg
    if cls is None:
        return None

    X_df = (
        (X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X))
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
    )
    X_df.columns = [str(c) for c in X_df.columns]
    y_arr = (
        np.asarray(y >= 0.5, dtype=np.int8)
        if classifier
        else np.asarray(y, dtype=np.float32)
    )
    n = len(y_arr)
    if n < 200 or X_df.shape[1] < 2:
        tprint("EBMOnLGBM: skipping, not enough rows/features.")
        return None

    sw = (
        np.ones(n, dtype=np.float32)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float32)
    )
    sw = np.nan_to_num(sw, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32)
    sw = sw / max(float(np.mean(sw)), 1e-6)

    stage_indices = _stage_partition_indices(
        y_arr,
        timestamps=timestamps,
        assets=assets,
        random_state=random_state + 701,
    )
    race_idx = np.asarray(stage_indices["lgbm_prune"], dtype=np.int32)
    if len(race_idx) < 200:
        race_idx = _stratified_subsample_indices(
            y_arr,
            max_n=min(EBM_RACE_MAX_ROWS, n),
            random_state=random_state + 701,
            classifier=classifier,
        )
        stage_indices["lgbm_prune"] = race_idx
    X_race = X_df.iloc[race_idx].reset_index(drop=True)
    y_race = y_arr[race_idx]
    sw_race = sw[race_idx]
    tprint(
        f"EBMOnLGBM: race subsample {len(race_idx)} rows, "
        f"{X_race.shape[1]} raw features before tree augmentation."
    )

    local_idx = np.arange(len(y_race), dtype=np.int32)
    if classifier:
        split_strata = y_race
    else:
        split_rank = pd.Series(y_race).rank(pct=True).to_numpy(dtype=np.float32)
        split_strata = np.clip((split_rank * 5).astype(np.int32), 0, 4)
    select_local, eval_local = train_test_split(
        local_idx,
        test_size=EBM_RACE_EVAL_FRACTION,
        stratify=split_strata,
        random_state=random_state + 1701,
    )
    select_local = np.asarray(select_local, dtype=np.int32)
    eval_local = np.asarray(eval_local, dtype=np.int32)
    X_select = X_race.iloc[select_local].reset_index(drop=True)
    y_select = y_race[select_local]
    sw_select = sw_race[select_local]
    X_eval = X_race.iloc[eval_local].reset_index(drop=True)
    y_eval = y_race[eval_local]
    race_groups = _stability_group_bundle(
        len(race_idx),
        timestamps=(
            np.asarray(timestamps)[race_idx]
            if timestamps is not None and len(np.asarray(timestamps)) == n
            else None
        ),
        assets=(
            np.asarray(assets)[race_idx]
            if assets is not None and len(np.asarray(assets)) == n
            else None
        ),
    )
    select_groups = _groups_take(race_groups, select_local)
    eval_groups = _groups_take(race_groups, eval_local)

    raw_feature_names = list(X_select.columns)
    tprint(
        "EBMOnLGBM: augmenting with tree features BEFORE raw-feature prescreening..."
    )
    X_select, X_eval, tree_feature_bundle = _augment_with_oof_tree_features(
        X_select,
        y_select,
        X_eval,
        random_state=random_state,
        classifier=classifier,
    )
    tprint(
        "EBMOnLGBM: pre-screen matrix after tree augmentation: "
        f"select={X_select.shape}, eval={X_eval.shape}."
    )

    x_select_np = X_select.to_numpy(dtype=np.float32)
    tprint(
        "EBMOnLGBM: honest prune/eval split: "
        f"select_train={len(select_local)}, eval={len(eval_local)}. "
        f"Each pruning round uses one frozen feature set across {EBM_CV_SPLITS} CV folds."
    )

    raw_feature_names = [c for c in raw_feature_names if c in X_select.columns]
    raw_active = _prescreen_features(
        X_select[raw_feature_names].to_numpy(dtype=np.float32),
        y_select,
        raw_feature_names,
        classifier=classifier,
        random_state=random_state,
    )
    full_col_pos = {str(c): i for i, c in enumerate(X_select.columns)}
    raw_active_full = np.asarray(
        [full_col_pos[str(raw_feature_names[i])] for i in raw_active],
        dtype=np.int32,
    )
    raw_feature_set = set(map(str, raw_feature_names))
    tree_active_full = np.asarray(
        [i for i, c in enumerate(X_select.columns) if str(c) not in raw_feature_set],
        dtype=np.int32,
    )
    active = np.unique(np.concatenate([raw_active_full, tree_active_full])).astype(
        np.int32
    )
    tprint(
        "EBMOnLGBM prescreen: raw Spearman/correlation/stability kept "
        f"{len(raw_active_full)}/{len(raw_feature_names)} raw features; "
        f"adding {len(tree_active_full)} pre-screened LGBM tree features for EBM rounds "
        f"({len(active)} active total)."
    )
    history: list[dict[str, Any]] = []
    prior_pruning_records: list[dict[str, Any]] = []
    best_seen: Optional[dict[str, Any]] = None
    pruning_stop_reason = "max_rounds"
    current_weights = sw_select.copy()
    current_oof = np.full(len(y_select), float(np.mean(y_select)), dtype=np.float32)

    current_leaf_min_pct = 0.02
    current_constraints = None

    for round_id in range(1, EBM_MAX_ROUNDS + 1):
        if len(active) <= EBM_MIN_FEATURES:
            tprint(
                f"EBMOnLGBM round {round_id}: feature floor reached ({len(active)})."
            )
            pruning_stop_reason = "feature_floor"
            break
        round_features = [X_select.columns[i] for i in active]
        tprint(
            f"EBMOnLGBM round {round_id}: {len(round_features)} active features "
            f"(frozen for all {EBM_CV_SPLITS} folds and specs in this round)."
        )
        round_res = _fit_round_oof(
            cls,
            X_select,
            y_select,
            current_weights,
            select_groups,
            classifier=classifier,
            random_state=random_state,
            round_id=round_id,
            active_features=round_features,
            build_tree_features=False,
            min_samples_leaf_pct=current_leaf_min_pct,
            monotone_constraints=current_constraints,
        )
        if not round_res.get("ok"):
            tprint(f"EBMOnLGBM round {round_id}: no valid EBM records, stopping.")
            pruning_stop_reason = "no_valid_records"
            break
        round_records = list(round_res.get("records", []))
        original_best = dict(round_res["best"])
        best = original_best
        top20_thresholds: dict[str, float] | None = None
        acceptable_round_records: list[dict[str, Any]] = []
        if round_id >= 3 and prior_pruning_records:
            top20_thresholds = _top20_acceptance_thresholds(prior_pruning_records)
            acceptable_round_records = [
                r
                for r in round_records
                if _passes_top20_acceptance(r, top20_thresholds)
            ]
            if not acceptable_round_records:
                best_j = float(original_best["J_final"])
                best_se = float(original_best.get("J_se", 0.0))
                fold_metrics = list(original_best.get("fold_metrics", []))
                fold_j_values = _finite_fold_j_values(fold_metrics)
                rec = {
                    "round": round_id,
                    "n_features_start": int(len(active)),
                    "J_final": best_j,
                    "J_mean": float(original_best.get("J_mean", 0.0)),
                    "J_std": float(original_best.get("J_std", 0.0)),
                    "J_se": best_se,
                    "J_robust": float(original_best.get("J_robust", best_j)),
                    "J_iqr": float(original_best.get("J_iqr", np.nan)),
                    "J_median": float(original_best.get("J_median", np.nan)),
                    "lift15": float(original_best.get("lift15", np.nan)),
                    "stability15": float(original_best.get("stability15", np.nan)),
                    "lift20": float(original_best.get("lift20", np.nan)),
                    "stability20": float(
                        original_best.get(
                            "stability20", original_best.get("stability30", np.nan)
                        )
                    ),
                    "salg20": float(original_best.get("salg20", np.nan)),
                    "J20": float(
                        original_best.get("J20", original_best.get("j20", np.nan))
                    ),
                    "j20": float(
                        original_best.get("j20", original_best.get("J20", np.nan))
                    ),
                    "ndcg20": float(
                        original_best.get(
                            "ndcg20",
                            original_best.get(
                                "ndcg_at_20", original_best.get("ndcg@20", np.nan)
                            ),
                        )
                    ),
                    "J_folds": fold_j_values,
                    "fold_metrics": fold_metrics,
                    "lift30": float(original_best.get("lift30", 0.0)),
                    "stability30": float(original_best.get("stability30", 0.0)),
                    "stopped_by_top20_acceptability": True,
                    "candidate_gate_passed": False,
                    "n_features_end": int(len(active)),
                    "active_indices": active.copy(),
                    "n_top20_acceptable_specs": 0,
                    "top20_acceptance_min_lift20": float(
                        top20_thresholds["min_lift20"]
                    ),
                    "top20_acceptance_min_salg20": float(
                        top20_thresholds["min_salg20"]
                    ),
                    "top20_acceptance_min_stability20": float(
                        top20_thresholds["min_stability20"]
                    ),
                    "top20_acceptance_min_ndcg20": float(
                        top20_thresholds["min_ndcg20"]
                    ),
                    "top20_acceptance_best_lift20": float(
                        top20_thresholds["best_lift20"]
                    ),
                    "top20_acceptance_best_salg20": float(
                        top20_thresholds["best_salg20"]
                    ),
                    "top20_acceptance_best_stability20": float(
                        top20_thresholds["best_stability20"]
                    ),
                    "top20_acceptance_best_ndcg20": float(
                        top20_thresholds["best_ndcg20"]
                    ),
                }
                history.append(rec)
                tprint(
                    f"EBMOnLGBM round {round_id}: stop, no full-round EBM spec "
                    "met prior best-so-far @20 acceptance gates "
                    f"(min_lift20={top20_thresholds['min_lift20']:.4f}, "
                    f"min_stability20={top20_thresholds['min_stability20']:.4f}, "
                    f"min_salg20={top20_thresholds['min_salg20']:.4f}, "
                    f"min_ndcg20={top20_thresholds['min_ndcg20']:.4f})."
                )
                pruning_stop_reason = "no_top20_acceptable_specs_vs_best_so_far"
                break
            best = dict(
                max(
                    acceptable_round_records,
                    key=lambda r: float(r.get("J_final", -np.inf)),
                )
            )

        best_j = float(best["J_final"])
        best_se = float(best.get("J_se", 0.0))
        fold_metrics = list(best.get("fold_metrics", []))
        fold_j_values = _finite_fold_j_values(fold_metrics)
        rec = {
            "round": round_id,
            "n_features_start": int(len(active)),
            "J_final": best_j,
            "J_mean": float(best.get("J_mean", 0.0)),
            "J_std": float(best.get("J_std", 0.0)),
            "J_se": best_se,
            "J_robust": float(best.get("J_robust", best_j)),
            "J_iqr": float(best.get("J_iqr", np.nan)),
            "J_median": float(best.get("J_median", np.nan)),
            "lift15": float(best.get("lift15", np.nan)),
            "stability15": float(best.get("stability15", np.nan)),
            "lift20": float(best.get("lift20", np.nan)),
            "stability20": float(
                best.get("stability20", best.get("stability30", np.nan))
            ),
            "salg20": float(best.get("salg20", np.nan)),
            "J20": float(best.get("J20", best.get("j20", np.nan))),
            "j20": float(best.get("j20", best.get("J20", np.nan))),
            "ndcg20": float(
                best.get("ndcg20", best.get("ndcg_at_20", best.get("ndcg@20", np.nan)))
            ),
            "J_folds": fold_j_values,
            "fold_metrics": fold_metrics,
            "lift30": float(best.get("lift30", 0.0)),
            "stability30": float(best.get("stability30", 0.0)),
        }
        if top20_thresholds is not None:
            rec["n_top20_acceptable_specs"] = int(len(acceptable_round_records))

        binary_mask = _binary_feature_mask(X_select[round_features])
        shape_models_for_scoring = best.get("shape_models", round_res["shape_models"])
        shape_scores, bends, is_cont, mas = _feature_shape_score_components(
            shape_models_for_scoring, round_features, binary_mask=binary_mask
        )

        cont_bends = bends[is_cont]
        if len(cont_bends) > 0:
            tprint(
                "  -> ShapeAudit stats: "
                f"avg_bends={float(np.mean(cont_bends)):.2f}, "
                f"count_pure_wiggle_proxy={int(np.sum(cont_bends >= 4))}; "
                "monotone constraints disabled for alpha/tree-state features."
            )
        current_constraints = None

        score_source = "shape"
        if not np.any(shape_scores > 0):
            shape_scores = _target_scores(
                x_select_np[:, active], y_select, round_features
            )
            score_source = "target_fallback"
        shape_scores, eval_score_details = _combine_ebm_feature_scores(
            shape_scores,
            best.get("feature_eval_stats"),
            round_id,
            round_features,
        )
        top_contributors = _top_prediction_feature_contributors(
            best.get("feature_eval_stats"), round_features, top_n=10
        )
        if top_contributors:
            rec["top_prediction_feature_contributors"] = top_contributors
            tprint(
                "  -> Top-prediction feature contributors: "
                + ", ".join(
                    f"{row['feature']}:{row['top_contribution_score']:.3g}"
                    for row in top_contributors[:5]
                )
            )
        if not np.any(shape_scores > 0.0):
            tprint(
                "EBMOnLGBM: all combined feature scores are zero after "
                "stability gates; falling back to ungated target scores for "
                "this pruning round."
            )
            shape_scores = _target_scores(
                x_select_np[:, active],
                y_select,
                round_features,
            )
            score_source = "ungated_target_fallback"
            eval_score_details["used_ungated_score_fallback"] = 1.0
            if not np.any(shape_scores > 0.0):
                start_features = [str(X_select.columns[i]) for i in active]
                n_tree_start = int(sum(c.startswith("lgbm_") for c in start_features))
                rec["stopped_by_zero_feature_scores"] = True
                rec["candidate_gate_passed"] = False
                rec["n_features_end"] = int(len(active))
                rec["n_raw_start"] = int(len(active) - n_tree_start)
                rec["n_tree_start"] = n_tree_start
                rec["n_raw_end"] = int(len(active) - n_tree_start)
                rec["n_tree_end"] = n_tree_start
                rec["score_source"] = score_source
                rec.update(eval_score_details)
                rec["feature_score_mean"] = 0.0
                rec["feature_score_p10"] = 0.0
                rec["feature_score_p50"] = 0.0
                rec["feature_score_p90"] = 0.0
                rec["active_indices"] = active.copy()
                history.append(rec)
                pruning_stop_reason = "all_feature_scores_zero"
                break
        else:
            eval_score_details["used_ungated_score_fallback"] = 0.0
        drop_frac = _round_drop_fraction(round_id)
        keep_n = max(EBM_MIN_FEATURES, int(np.ceil(len(active) * (1.0 - drop_frac))))
        keep_n = min(keep_n, len(active))
        keep_local = np.argsort(shape_scores)[-keep_n:]
        next_active = active[np.sort(keep_local)]
        start_features = [str(X_select.columns[i]) for i in active]
        end_features = [str(X_select.columns[i]) for i in next_active]
        n_tree_start = int(sum(c.startswith("lgbm_") for c in start_features))
        n_tree_end = int(sum(c.startswith("lgbm_") for c in end_features))
        rec["n_features_end"] = int(len(next_active))
        rec["n_raw_start"] = int(len(active) - n_tree_start)
        rec["n_tree_start"] = n_tree_start
        rec["n_raw_end"] = int(len(next_active) - n_tree_end)
        rec["n_tree_end"] = n_tree_end
        rec["drop_fraction"] = float(drop_frac)
        rec["score_source"] = score_source
        rec.update(eval_score_details)
        rec["feature_score_mean"] = float(np.mean(shape_scores))
        rec["feature_score_p10"] = float(np.percentile(shape_scores, 10.0))
        rec["feature_score_p50"] = float(np.percentile(shape_scores, 50.0))
        rec["feature_score_p90"] = float(np.percentile(shape_scores, 90.0))
        rec["active_indices"] = next_active.copy()
        rec["active_scores"] = np.asarray(
            shape_scores[np.sort(keep_local)], dtype=np.float32
        )
        if best_seen is None or best_j > float(best_seen["J_final"]):
            best_seen = {
                "J_final": best_j,
                "J_se": best_se,
                "active": next_active.copy(),
                "oof": np.asarray(best["oof"], dtype=np.float32).copy(),
                "round": round_id,
            }
        gate_passed, gate_details = _feature_pruning_candidate_gate(rec)
        rec.update(gate_details)
        history.append(rec)
        gate_msg = "passed" if gate_passed else "failed"
        tprint(
            f"EBMOnLGBM round {round_id}: J={best_j:.4f}, SE={best_se:.4f}, "
            f"pruned {len(active)} -> {len(next_active)}; "
            f"raw {rec['n_raw_start']} -> {rec['n_raw_end']}, "
            f"tree-state {rec['n_tree_start']} -> {rec['n_tree_end']}; "
            f"score_source={score_source}; "
            f"presence_min={rec['presence_min']:.2f}, "
            f"sign_min={rec['sign_consistency_min']:.2f}, "
            f"lgbm_presence_min={rec['lgbm_presence_min']:.2f}, "
            f"lgbm_sign_min={rec['lgbm_sign_consistency_min']:.2f}, "
            f"presence_fail={int(rec['presence_fail_count'])}, "
            f"sign_fail={int(rec['sign_fail_count'])}, "
            f"lgbm_presence_fail={int(rec['lgbm_presence_fail_count'])}, "
            f"lgbm_sign_fail={int(rec['lgbm_sign_fail_count'])}; "
            f"candidate gates {gate_msg} "
            f"(lift20={rec.get('lift20', np.nan):.4f}, "
            f"stability20={rec.get('stability20', np.nan):.4f}, "
            f"salg20={rec.get('salg20', np.nan):.4f}, "
            f"ndcg20={rec.get('ndcg20', np.nan):.4f}, "
            f"J={rec.get('J_final', np.nan):.4f})."
        )

        current_oof = np.asarray(best["oof"], dtype=np.float32)
        current_weights = sw_select * _compute_weight_distillation(
            y_select,
            current_oof,
            current_oof,
            is_classifier=classifier,
            include_false_positive_focus=False,
        )
        current_weights = current_weights * _false_positive_avoidance_weight(
            y_select,
            current_oof,
            classifier=classifier,
            trade_top_frac=_target_top_fraction(),
        )
        current_weights, _current_ess = _normalize_rank_based_weights(current_weights)
        active = next_active
        if _has_finite_top20_metrics(best):
            snap = _top20_reference_snapshot(best)
            snap["round"] = int(round_id)
            prior_pruning_records.append(snap)
        gc.collect()

    chosen_round = _select_robust_pruning_round(history) if history else {}
    tprint(
        "EBMOnLGBM pruning stopped: "
        f"reason={pruning_stop_reason}, rounds_completed={len(history)}, "
        f"best_round={int(best_seen.get('round', -1)) if best_seen else -1}."
    )
    if not chosen_round or chosen_round.get("active_indices") is None:
        tprint(
            "EBMOnLGBM: no pruning round passed hard selection guardrails; "
            "rejecting candidate."
        )
        return None

    selected_active = np.asarray(chosen_round["active_indices"], dtype=np.int32)
    tprint(
        "EBMOnLGBM: selected robust quality-efficiency feature candidate "
        f"(round={int(chosen_round.get('round', -1))}, "
        f"features={len(selected_active)}, "
        f"J={float(chosen_round.get('J_final', np.nan)):.4f}, "
        f"SE={float(chosen_round.get('J_se', np.nan)):.4f})."
    )

    selected_features = [X_select.columns[i] for i in selected_active]
    if best_seen is None:
        tprint("EBMOnLGBM: no successful pruning round.")
        return None

    eval_preds: list[np.ndarray] = []
    final_weights = sw_select * _compute_weight_distillation(
        y_select,
        np.asarray(best_seen["oof"], dtype=np.float32),
        np.asarray(best_seen["oof"], dtype=np.float32),
        is_classifier=classifier,
        include_false_positive_focus=False,
    )
    final_weights = final_weights * _false_positive_avoidance_weight(
        y_select,
        np.asarray(best_seen["oof"], dtype=np.float32),
        classifier=classifier,
        trade_top_frac=_target_top_fraction(),
    )
    final_weights, _final_ess = _normalize_rank_based_weights(final_weights)
    mode_name = "classifier" if classifier else "regressor"
    specs = _ebm_specs(pruning=False, random_state=random_state)[:EBM_FINAL_MODEL_COUNT]
    for spec in specs:
        spec["outer_bags"] = 1
    selected_features = _contribution_correlation_prune(
        cls,
        X_select,
        y_select,
        final_weights,
        X_eval,
        selected_features,
        mode_name,
        random_state=random_state + 7601,
    )
    _log_selected_features("post-pruning candidate", selected_features)
    min_honest_models = max(1, min(len(specs), int(EBM_HONEST_EVAL_MIN_MODELS)))
    tprint(
        f"EBMOnLGBM: fitting up to {len(specs)} honest eval EBMs on selected features "
        f"(adaptive_min={min_honest_models}, "
        f"max_j_std={EBM_HONEST_EVAL_MAX_J_STD:.4f}, "
        f"max_pred_disagree={EBM_HONEST_EVAL_MAX_PRED_DISAGREE:.4f})."
    )
    X_select_final = X_select[selected_features].reset_index(drop=True)
    X_eval_final = X_eval[selected_features].reset_index(drop=True)
    n_tree_selected = int(sum(name.startswith("lgbm_") for name in selected_features))
    n_raw_selected = int(len(selected_features) - n_tree_selected)
    tprint(
        "EBMOnLGBM: honest eval matrix includes "
        f"{n_raw_selected} raw + {n_tree_selected} selected tree-state features."
    )
    honest_eval_trace: list[dict[str, float]] = []
    honest_eval_stop_reason = "max_models"
    for i, spec in enumerate(specs, start=1):
        t0_eval_fit = time.perf_counter()
        ebm = _fit_one_ebm(cls, X_select_final, y_select, final_weights, spec)
        raw_select = _predict_raw_ebm(ebm, X_select_final, mode_name)
        pp = SplinePostProcessor(mode_name).fit(
            raw_select, y_select, use_dynamic_smoothing=True
        )
        raw_eval = _predict_raw_ebm(ebm, X_eval_final, mode_name)
        eval_preds.append(pp.predict(raw_eval))
        ensemble_pred_i = np.mean(np.vstack(eval_preds), axis=0).astype(np.float32)
        metrics_i = _metric_pack(
            y_eval,
            ensemble_pred_i,
            classifier=classifier,
            groups=eval_groups,
        )
        metrics_i.update(_aggregate_j([metrics_i]))
        pred_disagree = (
            float(np.mean(np.std(np.vstack(eval_preds), axis=0)))
            if len(eval_preds) >= 2
            else float("inf")
        )
        j_vals = np.asarray(
            [row["J_final"] for row in honest_eval_trace] + [metrics_i["J_final"]],
            dtype=np.float32,
        )
        j_std = float(np.std(j_vals, ddof=0)) if len(j_vals) >= 2 else float("inf")
        honest_eval_trace.append(
            {
                "model": float(i),
                "J_final": float(metrics_i["J_final"]),
                "lift30": float(metrics_i.get("lift30", np.nan)),
                "stability30": float(metrics_i.get("stability30", np.nan)),
                "pred_disagree": float(pred_disagree),
                "j_std": float(j_std),
            }
        )
        tprint(
            f"  Honest eval EBM {i}/{len(specs)} fit in "
            f"{time.perf_counter() - t0_eval_fit:.1f}s "
            f"(ensemble_J={float(metrics_i['J_final']):.4f}, "
            f"j_std={j_std:.4f}, pred_disagree={pred_disagree:.4f})"
        )
        if (
            i >= min_honest_models
            and j_std <= EBM_HONEST_EVAL_MAX_J_STD
            and pred_disagree <= EBM_HONEST_EVAL_MAX_PRED_DISAGREE
        ):
            honest_eval_stop_reason = "stable_adaptive_stop"
            tprint(
                "EBMOnLGBM: honest eval adaptive stop after "
                f"{i}/{len(specs)} models "
                f"(j_std={j_std:.4f}, pred_disagree={pred_disagree:.4f})."
            )
            break
    eval_pred = np.mean(np.vstack(eval_preds), axis=0).astype(np.float32)
    oof_race = np.full(len(y_race), np.nan, dtype=np.float32)
    oof_race[eval_local] = eval_pred
    metrics = _metric_pack(
        y_eval,
        eval_pred,
        classifier=classifier,
        groups=eval_groups,
    )
    metrics.update(_aggregate_j([metrics]))
    metrics["J_final_oof"] = float(metrics.get("J_final", 0.0))
    metrics["J_Score"] = float(metrics["J_final_oof"])
    metrics["feature_count"] = int(len(selected_features))
    metrics["honest_eval_models_fit"] = int(len(eval_preds))
    metrics["honest_eval_max_models"] = int(len(specs))
    metrics["honest_eval_stop_reason"] = honest_eval_stop_reason
    metrics["honest_eval_trace"] = list(honest_eval_trace)
    metrics["feature_pruning_stop_reason"] = pruning_stop_reason
    metrics["feature_pruning_rounds_completed"] = int(len(history))
    metrics["n_raw_features_kept"] = int(
        sum(not str(name).startswith("lgbm_") for name in selected_features)
    )
    metrics["n_leaf_features_kept"] = int(
        sum(str(name).startswith("lgbm_") for name in selected_features)
    )
    metrics["race_n"] = int(len(race_idx))
    metrics["stage_lgbm_prune_n"] = int(len(stage_indices["lgbm_prune"]))
    metrics["stage_hpo_n"] = int(len(stage_indices["hpo"]))
    metrics["stage_fit_oof_n"] = int(len(stage_indices["fit_oof"]))
    oof_full = np.full(n, np.nan, dtype=np.float32)
    oof_full[race_idx] = oof_race
    oof_for_full_fit = np.where(
        np.isfinite(oof_full), oof_full, float(np.mean(y_arr))
    ).astype(np.float32)
    metrics["oof_coverage"] = float(
        np.sum(np.isfinite(oof_full)) / max(metrics["race_n"], 1)
    )

    tprint(
        f"EBMOnLGBM: race done J={metrics.get('J_final_oof', 0.0):.4f}, "
        f"lift30={metrics.get('lift30', 0.0):.3f}, "
        f"ndcg20={metrics.get('ndcg_at_20', metrics.get('ndcg@20', np.nan)):.4f}, "
        f"ece20={metrics.get('ece_top20', metrics.get('ece_at_20', np.nan)):.4f}, "
        f"features={len(selected_features)}."
    )
    for key in (
        "J_final_oof",
        "lift30",
        "ndcg_at_20",
        "auc_correct_30",
        "stability30",
        "auc",
        "brier",
        "ece",
        "ece_top20",
        "top30_correctness_rate",
        "overall_correctness_rate",
        "feature_count",
        "oof_coverage",
    ):
        if key in metrics:
            tprint(f"    {key}: {metrics[key]}")
    tprint(f"EBMOnLGBM: total candidate time {time.perf_counter() - t0_total:.1f}s.")
    selected_raw_indices = [
        X_df.columns.get_loc(c) for c in selected_features if c in X_df.columns
    ]
    return {
        "model": None,
        "metrics": metrics,
        "oof_probs": oof_full,
        "pruning_history": history,
        "selected_features_from_cv": np.asarray(selected_raw_indices, dtype=np.int32),
        "selected_feature_names": selected_features,
        "full_fit_needed": True,
        "race_idx": race_idx,
        "oof_race": oof_for_full_fit,
        "stage_indices": {
            k: np.asarray(v, dtype=np.int32) for k, v in stage_indices.items()
        },
        "tree_feature_bundle": tree_feature_bundle,
        "mode": mode,
        "hpo_objective_mode": hpo_objective_mode,
    }


def fit_ebm_on_lgbm_full_model(
    X: Any,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray],
    selected_features_from_cv: np.ndarray,
    random_state: int = 42,
    mode: str = "classifier",
    oof_probs: Optional[np.ndarray] = None,
    metrics: Optional[dict[str, Any]] = None,
    pruning_history: Optional[list[dict[str, Any]]] = None,
    selected_feature_names: Optional[list[str]] = None,
    stage_indices: Optional[dict[str, np.ndarray]] = None,
    timestamps: Any = None,
    assets: Any = None,
    tree_feature_bundle: Optional[dict[str, Any]] = None,
    hpo_objective_mode: str = "base",
    hpo_trials_override: int | None = None,
    hpo_patience_override: int | None = None,
) -> Optional[EBMOnLGBMModel]:
    classifier = mode == "classifier"
    ebm_clf, ebm_reg = _load_ebm_classes()
    cls = ebm_clf if classifier else ebm_reg
    if cls is None:
        return None
    X_df = (
        (X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X))
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
    )
    X_df.columns = [str(c) for c in X_df.columns]
    idx = np.asarray(selected_features_from_cv, dtype=np.int32)
    idx = idx[(idx >= 0) & (idx < X_df.shape[1])]
    selected_features = [X_df.columns[i] for i in idx]
    if len(selected_features) == 0 and not selected_feature_names:
        return None
    y_arr = (
        np.asarray(y >= 0.5, dtype=np.int8)
        if classifier
        else np.asarray(y, dtype=np.float32)
    )
    sw = (
        np.ones(len(y_arr), dtype=np.float32)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float32)
    )
    sw = np.nan_to_num(sw, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32)
    sw = sw / max(float(np.mean(sw)), 1e-6)
    return _fit_final_model(
        cls,
        X_df,
        y_arr,
        sw,
        selected_features,
        selected_feature_names,
        mode,
        random_state,
        pruning_history or [],
        np.asarray(
            oof_probs if oof_probs is not None else np.zeros(len(y_arr)),
            dtype=np.float32,
        ),
        metrics or {},
        stage_indices=stage_indices,
        timestamps=timestamps,
        assets=assets,
        tree_feature_bundle=tree_feature_bundle,
        hpo_objective_mode=hpo_objective_mode,
        hpo_trials_override=hpo_trials_override,
        hpo_patience_override=hpo_patience_override,
    )
