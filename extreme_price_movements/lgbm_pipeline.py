from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

try:
    from .ridge_on_lgbm import _compute_weight_distillation
except Exception:  # pragma: no cover - standalone fallback
    def _compute_weight_distillation(
        y_true: np.ndarray,
        pred: np.ndarray,
        prev_pred: np.ndarray | None = None,
        *,
        is_classifier: bool = True,
        include_false_positive_focus: bool = False,
    ) -> np.ndarray:
        del prev_pred, include_false_positive_focus
        y = np.asarray(y_true, dtype=np.float32)
        p = np.asarray(pred, dtype=np.float32)
        if len(y) == 0:
            return np.ones(0, dtype=np.float32)
        if is_classifier:
            yb = (y >= 0.5).astype(np.float32)
            err = np.abs(yb - np.clip(p, 0.0, 1.0))
        else:
            scale = float(np.nanpercentile(np.abs(y - np.nanmedian(y)), 75.0) + 1e-6)
            err = np.abs(y - p) / scale
        rank = pd.Series(np.nan_to_num(err, nan=0.0)).rank(pct=True).to_numpy(dtype=np.float32)
        return np.clip(0.75 + 1.50 * rank, 0.25, 4.0).astype(np.float32)

try:
    from .utils import tprint
except Exception:  # pragma: no cover - standalone fallback
    def tprint(message: str) -> None:
        print(message, flush=True)


LGBM_CV_SPLITS = int(os.environ.get("EPM_LGBM_CV_SPLITS", "3"))
LGBM_RACE_MAX_ROWS = int(os.environ.get("EPM_LGBM_RACE_MAX_ROWS", "120000"))
LGBM_RACE_EVAL_FRACTION = float(os.environ.get("EPM_LGBM_RACE_EVAL_FRACTION", "0.3333333333"))
LGBM_MIN_FEATURES = int(os.environ.get("EPM_LGBM_MIN_FEATURES", "40"))
LGBM_SELECTED_FEATURES_MIN = int(os.environ.get("EPM_LGBM_SELECTED_FEATURES_MIN", "100"))
LGBM_SELECTED_FEATURES_MAX = int(os.environ.get("EPM_LGBM_SELECTED_FEATURES_MAX", "350"))
LGBM_MAX_ROUNDS = int(os.environ.get("EPM_LGBM_MAX_ROUNDS", "10"))
LGBM_ROW_SUBSAMPLE_FRAC = float(os.environ.get("EPM_LGBM_ROW_SUBSAMPLE_FRAC", "1.0"))
LGBM_UNIVARIATE_MAX_ROWS = int(os.environ.get("EPM_LGBM_UNIVARIATE_MAX_ROWS", "20000"))
LGBM_UNIVARIATE_ROW_SUBSAMPLE_FRAC = float(os.environ.get("EPM_LGBM_UNIVARIATE_ROW_SUBSAMPLE_FRAC", os.environ.get("EPM_LGBM_ROW_SUBSAMPLE_FRAC", "1.0")))
LGBM_RELIEF_ENABLED = os.environ.get("EPM_LGBM_RELIEF_ENABLED", "1") != "0"
LGBM_RELIEF_REPEATS = int(os.environ.get("EPM_LGBM_RELIEF_REPEATS", "4"))
LGBM_RELIEF_PRESENCE_MIN = float(os.environ.get("EPM_LGBM_RELIEF_PRESENCE_MIN", "0.35"))
LGBM_RELIEF_RESCUE_MAX = int(os.environ.get("EPM_LGBM_RELIEF_RESCUE_MAX", "80"))
LGBM_RELIEF_RESCUE_MIN = int(os.environ.get("EPM_LGBM_RELIEF_RESCUE_MIN", "20"))
LGBM_RELIEF_RESCUE_FRAC = float(os.environ.get("EPM_LGBM_RELIEF_RESCUE_FRAC", "0.25"))
LGBM_RELIEF_ANCHOR_MAX_ROWS = int(os.environ.get("EPM_LGBM_RELIEF_ANCHOR_MAX_ROWS", "768"))
LGBM_RELIEF_NEIGHBOR_CANDIDATES = int(os.environ.get("EPM_LGBM_RELIEF_NEIGHBOR_CANDIDATES", "2048"))
LGBM_RELIEF_NEIGHBORS = int(os.environ.get("EPM_LGBM_RELIEF_NEIGHBORS", "8"))
LGBM_HPO_MAX_ROWS = int(os.environ.get("EPM_LGBM_HPO_MAX_ROWS", "10000"))
LGBM_HPO_ROW_SUBSAMPLE_FRAC = float(os.environ.get("EPM_LGBM_HPO_ROW_SUBSAMPLE_FRAC", os.environ.get("EPM_LGBM_ROW_SUBSAMPLE_FRAC", "1.0")))
LGBM_HPO_TRIALS = int(os.environ.get("EPM_LGBM_HPO_TRIALS", "200"))
LGBM_HPO_EARLY_STOP_PATIENCE = int(os.environ.get("EPM_LGBM_HPO_EARLY_STOP_PATIENCE", "50"))
LGBM_HPO_FINAL_MIN_ESTIMATORS = int(os.environ.get("EPM_LGBM_HPO_FINAL_MIN_ESTIMATORS", "800"))
LGBM_FINAL_MODEL_COUNT = int(os.environ.get("EPM_LGBM_FINAL_MODEL_COUNT", "3"))
LGBM_OOF_DISTILLATION_PASSES = int(os.environ.get("EPM_LGBM_OOF_DISTILLATION_PASSES", "1"))
LGBM_MIN_OOF_DISTILLATION_PASSES = int(os.environ.get("EPM_LGBM_MIN_OOF_DISTILLATION_PASSES", "2"))
LGBM_META_MIN_OOF_DISTILLATION_PASSES = int(os.environ.get("EPM_LGBM_META_MIN_OOF_DISTILLATION_PASSES", "2"))
LGBM_FINAL_FIT_USE_ALL_ROWS = os.environ.get("EPM_LGBM_FINAL_FIT_USE_ALL_ROWS", "1") == "1"
LGBM_META_RANK_BINS = int(os.environ.get("EPM_LGBM_META_RANK_BINS", "10"))
LGBM_DIRECTION_STABILITY_MIN = float(os.environ.get("EPM_LGBM_DIRECTION_STABILITY_MIN", "0.75"))
LGBM_DIRECTION_MAX_ROWS = int(os.environ.get("EPM_LGBM_DIRECTION_MAX_ROWS", "5000"))
LGBM_STABILITY_CONFIGS = int(os.environ.get("EPM_LGBM_STABILITY_CONFIGS", "4"))
LGBM_SELECTION_SE_MULT = float(os.environ.get("EPM_LGBM_SELECTION_SE_MULT", "0.75"))
LGBM_POSITIVE_PERM_RATE_MIN = float(os.environ.get("EPM_LGBM_POSITIVE_PERM_RATE_MIN", "0.50"))
LGBM_LOW_PRESENCE_RATE = float(os.environ.get("EPM_LGBM_LOW_PRESENCE_RATE", "0.20"))
LGBM_FEATURE_RECENT_MIN_COVERAGE = float(os.environ.get("EPM_LGBM_FEATURE_RECENT_MIN_COVERAGE", "0.85"))
LGBM_FEATURE_RECENT_TAIL_MONTHS = int(os.environ.get("EPM_LGBM_FEATURE_RECENT_TAIL_MONTHS", "4"))
LGBM_REDUNDANCY_CORR_THRESHOLD = float(os.environ.get("EPM_LGBM_REDUNDANCY_CORR_THRESHOLD", "0.90"))
LGBM_REDUNDANCY_PENALTY_START = float(os.environ.get("EPM_LGBM_REDUNDANCY_PENALTY_START", "0.85"))
LGBM_UNIVARIATE_MONOTONICITY_MIN = float(os.environ.get("EPM_LGBM_UNIVARIATE_MONOTONICITY_MIN", "0.95"))
LGBM_PERMUTATION_REPEATS = int(os.environ.get("EPM_LGBM_PERMUTATION_REPEATS", "2"))
LGBM_PERMUTATION_EPS = float(os.environ.get("EPM_LGBM_PERMUTATION_EPS", "1e-5"))
LGBM_PERMUTATION_MAX_FEATURES = int(os.environ.get("EPM_LGBM_PERMUTATION_MAX_FEATURES", "50"))
LGBM_PERMUTATION_MAX_ROWS = int(os.environ.get("EPM_LGBM_PERMUTATION_MAX_ROWS", "5000"))
LGBM_PERMUTATION_TOP_CONFIGS = int(os.environ.get("EPM_LGBM_PERMUTATION_TOP_CONFIGS", "2"))
LGBM_PERMUTATION_SKIP_STRONG_TOP_FRAC = float(os.environ.get("EPM_LGBM_PERMUTATION_SKIP_STRONG_TOP_FRAC", "0.10"))
LGBM_PERMUTATION_SKIP_WEAK_BOTTOM_FRAC = float(os.environ.get("EPM_LGBM_PERMUTATION_SKIP_WEAK_BOTTOM_FRAC", "0.50"))
LGBM_OVERFIT_GAP_PENALTY = float(os.environ.get("EPM_LGBM_OVERFIT_GAP_PENALTY", "0.0"))
LGBM_OVERFIT_GAP_DEADBAND = float(os.environ.get("EPM_LGBM_OVERFIT_GAP_DEADBAND", "0.02"))
LGBM_OVERFIT_GAP_CAP = float(os.environ.get("EPM_LGBM_OVERFIT_GAP_CAP", "0.50"))
LGBM_IMPORTANCE_INSTABILITY_ENABLE = os.environ.get("EPM_LGBM_IMPORTANCE_INSTABILITY_ENABLE", "0") == "1"
LGBM_IMPORTANCE_INSTABILITY_PENALTY = float(os.environ.get("EPM_LGBM_IMPORTANCE_INSTABILITY_PENALTY", "0.15"))
LGBM_IMPORTANCE_INSTABILITY_GAIN_WEIGHT = float(os.environ.get("EPM_LGBM_IMPORTANCE_INSTABILITY_GAIN_WEIGHT", "0.70"))
LGBM_IMPORTANCE_INSTABILITY_SPLIT_WEIGHT = float(os.environ.get("EPM_LGBM_IMPORTANCE_INSTABILITY_SPLIT_WEIGHT", "0.30"))
LGBM_IMPORTANCE_TOPK_CONTRIB_BLEND = float(os.environ.get("EPM_LGBM_IMPORTANCE_TOPK_CONTRIB_BLEND", "0.30"))
LGBM_IMPORTANCE_INSTABILITY_PRIOR_STRENGTH = float(os.environ.get("EPM_LGBM_IMPORTANCE_INSTABILITY_PRIOR_STRENGTH", "0.10"))
LGBM_IMPORTANCE_INSTABILITY_MATERIAL_TOP_FRAC = float(os.environ.get("EPM_LGBM_IMPORTANCE_INSTABILITY_MATERIAL_TOP_FRAC", "0.20"))
LGBM_IMPORTANCE_INSTABILITY_CV_CAP = float(os.environ.get("EPM_LGBM_IMPORTANCE_INSTABILITY_CV_CAP", "3.0"))
LGBM_IMPORTANCE_TOPK_FOCUS_SOFTNESS = float(os.environ.get("EPM_LGBM_IMPORTANCE_TOPK_FOCUS_SOFTNESS", "0.50"))
LGBM_FINAL_FIT_MAX_ROWS = int(os.environ.get("EPM_LGBM_FINAL_FIT_MAX_ROWS", "0"))
LGBM_HPO_LEARNING_RATE = float(os.environ.get("EPM_LGBM_HPO_LEARNING_RATE", "0.02"))
LGBM_FINAL_LEARNING_RATE = float(os.environ.get("EPM_LGBM_FINAL_LEARNING_RATE", "0.02"))
LGBM_HPO_PATH_SMOOTH_MAX = float(os.environ.get("EPM_LGBM_HPO_PATH_SMOOTH_MAX", "10.0"))
LGBM_BASE_METRIC_TARGET_FRACTION = float(os.environ.get("EPM_LGBM_BASE_METRIC_TARGET_FRACTION", "0.30"))
LGBM_META_METRIC_TARGET_FRACTION = float(os.environ.get("EPM_LGBM_META_METRIC_TARGET_FRACTION", os.environ.get("EPM_LGBM_METRIC_TARGET_FRACTION", "0.15")))
LGBM_SALG_LIFT_COEF = float(os.environ.get("EPM_LGBM_SALG_LIFT_COEF", "0.38"))
LGBM_J_SALG_NORM_DENOM = float(os.environ.get("EPM_LGBM_J_SALG_NORM_DENOM", "1.50"))
LGBM_OBJECTIVE = str(os.environ.get("EPM_LGBM_OBJECTIVE", "default")).strip().lower()
LGBM_HPO_PARAM_SET = str(os.environ.get("EPM_LGBM_HPO_PARAM_SET", "full")).strip().lower()
LGBM_FEATURE_SELECTION_OBJECTIVE = str(
    os.environ.get("EPM_LGBM_FEATURE_SELECTION_OBJECTIVE", "")
).strip().lower()
LGBM_TAIL_WEEK_MIN_ROWS = int(os.environ.get("EPM_LGBM_TAIL_WEEK_MIN_ROWS", "20"))
LGBM_TAIL_ASSET_MIN_ROWS = int(os.environ.get("EPM_LGBM_TAIL_ASSET_MIN_ROWS", "20"))
LGBM_TAIL_ROLLING_ROWS = int(os.environ.get("EPM_LGBM_TAIL_ROLLING_ROWS", "1000"))
LGBM_TAIL_LIFT_NORM_DENOM = float(os.environ.get("EPM_LGBM_TAIL_LIFT_NORM_DENOM", "2.0"))
LGBM_TAIL_WORST_FEATURE_PENALTY = float(os.environ.get("EPM_LGBM_TAIL_WORST_FEATURE_PENALTY", "0.05"))
LGBM_N_JOBS = int(os.environ.get("EPM_LGBM_N_JOBS", "3"))

LGBM_CV_SPLITS = max(2, int(LGBM_CV_SPLITS))
LGBM_RACE_EVAL_FRACTION = float(np.clip(LGBM_RACE_EVAL_FRACTION, 0.10, 0.50))
LGBM_ROW_SUBSAMPLE_FRAC = float(np.clip(LGBM_ROW_SUBSAMPLE_FRAC, 0.01, 1.0))
LGBM_RELIEF_REPEATS = max(1, int(LGBM_RELIEF_REPEATS))
LGBM_RELIEF_PRESENCE_MIN = float(np.clip(LGBM_RELIEF_PRESENCE_MIN, 0.0, 1.0))
LGBM_RELIEF_RESCUE_MAX = max(1, int(LGBM_RELIEF_RESCUE_MAX))
LGBM_RELIEF_RESCUE_MIN = max(1, int(LGBM_RELIEF_RESCUE_MIN))
LGBM_RELIEF_RESCUE_FRAC = float(np.clip(LGBM_RELIEF_RESCUE_FRAC, 0.01, 1.0))
LGBM_RELIEF_ANCHOR_MAX_ROWS = max(1, int(LGBM_RELIEF_ANCHOR_MAX_ROWS))
LGBM_RELIEF_NEIGHBOR_CANDIDATES = max(2, int(LGBM_RELIEF_NEIGHBOR_CANDIDATES))
LGBM_RELIEF_NEIGHBORS = max(1, int(LGBM_RELIEF_NEIGHBORS))
LGBM_STABILITY_CONFIGS = max(1, int(LGBM_STABILITY_CONFIGS))
LGBM_OVERFIT_GAP_PENALTY = float(np.clip(LGBM_OVERFIT_GAP_PENALTY, 0.0, 2.0))
LGBM_OVERFIT_GAP_DEADBAND = float(np.clip(LGBM_OVERFIT_GAP_DEADBAND, 0.0, 1.0))
LGBM_OVERFIT_GAP_CAP = float(np.clip(LGBM_OVERFIT_GAP_CAP, 0.0, 2.0))
LGBM_IMPORTANCE_INSTABILITY_PENALTY = float(np.clip(LGBM_IMPORTANCE_INSTABILITY_PENALTY, 0.0, 2.0))
LGBM_IMPORTANCE_INSTABILITY_GAIN_WEIGHT = max(0.0, float(LGBM_IMPORTANCE_INSTABILITY_GAIN_WEIGHT))
LGBM_IMPORTANCE_INSTABILITY_SPLIT_WEIGHT = max(0.0, float(LGBM_IMPORTANCE_INSTABILITY_SPLIT_WEIGHT))
LGBM_IMPORTANCE_TOPK_CONTRIB_BLEND = float(np.clip(LGBM_IMPORTANCE_TOPK_CONTRIB_BLEND, 0.0, 1.0))
LGBM_IMPORTANCE_INSTABILITY_PRIOR_STRENGTH = float(np.clip(LGBM_IMPORTANCE_INSTABILITY_PRIOR_STRENGTH, 0.0, 0.90))
LGBM_IMPORTANCE_INSTABILITY_MATERIAL_TOP_FRAC = float(np.clip(LGBM_IMPORTANCE_INSTABILITY_MATERIAL_TOP_FRAC, 0.01, 1.0))
LGBM_IMPORTANCE_INSTABILITY_CV_CAP = max(1e-6, float(LGBM_IMPORTANCE_INSTABILITY_CV_CAP))
LGBM_IMPORTANCE_TOPK_FOCUS_SOFTNESS = max(1e-6, float(LGBM_IMPORTANCE_TOPK_FOCUS_SOFTNESS))
LGBM_FINAL_MODEL_COUNT = max(1, int(LGBM_FINAL_MODEL_COUNT))
LGBM_HPO_FINAL_MIN_ESTIMATORS = max(1, int(LGBM_HPO_FINAL_MIN_ESTIMATORS))
LGBM_HPO_PATH_SMOOTH_MAX = float(np.clip(LGBM_HPO_PATH_SMOOTH_MAX, 0.0, 10.0))
LGBM_OOF_DISTILLATION_PASSES = max(0, int(LGBM_OOF_DISTILLATION_PASSES))
LGBM_MIN_OOF_DISTILLATION_PASSES = max(0, int(LGBM_MIN_OOF_DISTILLATION_PASSES))
LGBM_META_MIN_OOF_DISTILLATION_PASSES = max(0, int(LGBM_META_MIN_OOF_DISTILLATION_PASSES))
LGBM_META_RANK_BINS = max(2, int(LGBM_META_RANK_BINS))
LGBM_BASE_METRIC_TARGET_FRACTION = float(np.clip(LGBM_BASE_METRIC_TARGET_FRACTION, 0.001, 0.5))
LGBM_META_METRIC_TARGET_FRACTION = float(np.clip(LGBM_META_METRIC_TARGET_FRACTION, 0.001, 0.5))
LGBM_J_SALG_NORM_DENOM = max(1e-6, float(LGBM_J_SALG_NORM_DENOM))
if LGBM_OBJECTIVE not in {"default", "tail_control"}:
    LGBM_OBJECTIVE = "default"
if LGBM_HPO_PARAM_SET not in {"full", "reduced"}:
    LGBM_HPO_PARAM_SET = "full"
LGBM_TAIL_WEEK_MIN_ROWS = max(1, int(LGBM_TAIL_WEEK_MIN_ROWS))
LGBM_TAIL_ASSET_MIN_ROWS = max(1, int(LGBM_TAIL_ASSET_MIN_ROWS))
LGBM_TAIL_ROLLING_ROWS = max(8, int(LGBM_TAIL_ROLLING_ROWS))
LGBM_TAIL_LIFT_NORM_DENOM = max(1e-6, float(LGBM_TAIL_LIFT_NORM_DENOM))
LGBM_TAIL_WORST_FEATURE_PENALTY = float(np.clip(LGBM_TAIL_WORST_FEATURE_PENALTY, 0.0, 1.0))

LGBM_META_FEATURE_NAMES = [
    "lgbm_prob",
    "lgbm_raw_score",
    "abs_raw_score",
    "entropy",
    "variance_proxy",
    "rank_pct",
    "score_margin_top10",
    "score_margin_top20",
    "score_margin_top30",
    "rank_margin_top10",
    "rank_margin_top20",
    "leaf_count_p10",
    "leaf_count_min",
    "rare_leaf_fraction",
    "leaf_weight_p10",
    "leaf_depth_mean",
    "leaf_depth_max",
    "leaf_value_abs_mean",
    "leaf_value_abs_max",
    "large_leaf_value_fraction",
    "contrib_top1_abs_share",
    "contrib_top3_abs_share",
    "contrib_entropy",
    "contrib_balance",
    "num_material_contrib_features",
    "score_100_minus_50",
    "score_100_minus_75",
    "score_path_std",
    "rank_100_minus_50",
    "rank_path_std",
    "rank_bin_win_rate_oof",
    "rank_bin_lift_oof",
    "rank_bin_net_ret_oof",
    "rank_bin_se_oof",
    "regime_centroid_similarity_train",
    "feature_drift_psi_core",
    "feature_drift_cov_shift",
]


def _is_lgbm_model_derived_meta_feature(name: str) -> bool:
    """Return true for meta diagnostics unavailable in the raw coverage slice."""
    key = str(name)
    if key in LGBM_META_FEATURE_NAMES:
        return True
    if key in {
        "pred_logit",
        "regime_centroid_similarity_train",
        "feature_drift_psi_core",
        "feature_drift_cov_shift",
    }:
        return True
    return key.startswith(
        (
            "pred_",
            "base_H",
        )
    )


@dataclass
class FeatureSelectionResult:
    feature_names: list[str]
    selected_features: list[str]
    history: list[dict[str, Any]]
    stats: pd.DataFrame
    oof_pred: np.ndarray
    metrics: dict[str, Any]
    stage_indices: dict[str, np.ndarray]


@dataclass
class LGBMStabilityModel:
    mode: str = "classifier"
    models: list[Any] = field(default_factory=list)
    selected_features: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    pruning_history: list[dict[str, Any]] = field(default_factory=list)
    oof_probs: Optional[np.ndarray] = None
    best_params: dict[str, Any] = field(default_factory=dict)
    meta_feature_names: list[str] = field(default_factory=lambda: list(LGBM_META_FEATURE_NAMES))
    meta_oof_features: Optional[pd.DataFrame] = None
    rank_bin_stats_oof: pd.DataFrame = field(default_factory=pd.DataFrame)
    allow_missing_features_at_inference: bool = False
    feature_stats_train: dict[str, dict[str, float]] = field(default_factory=dict)
    input_feature_names: list[str] = field(default_factory=list)
    drift_reference: dict[str, Any] = field(default_factory=dict)

    def _frame(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X)
        X_df.columns = [str(c) for c in X_df.columns]
        selected = [str(c) for c in self.selected_features]
        input_features = [str(c) for c in getattr(self, "input_feature_names", []) or []]
        use_input_aliases = len(input_features) == len(selected) and input_features != selected
        contract_features = input_features if use_input_aliases else selected
        missing = [c for c in contract_features if c not in X_df.columns]
        if missing:
            preview = missing[:20]
            raise ValueError(
                "LGBM inference feature contract violation: "
                f"{len(missing)}/{len(contract_features)} contracted features are missing. "
                f"Examples: {preview}"
            )
        try:
            out = X_df.loc[:, contract_features].astype(np.float32, copy=False)
        except Exception as exc:
            raise ValueError(
                "LGBM inference feature contract violation: contracted features "
                f"cannot be cast to float32: {exc}"
            ) from exc
        values = out.to_numpy(dtype=np.float32, copy=False)
        finite_mask = np.isfinite(values)
        if not finite_mask.all():
            bad_cols = [
                str(col)
                for col in out.columns
                if not np.isfinite(out[col].to_numpy(dtype=np.float32, copy=False)).all()
            ]
            bad_rows = int((~finite_mask.all(axis=1)).sum())
            raise ValueError(
                "LGBM inference feature contract violation: "
                f"{bad_rows}/{len(out)} rows contain non-finite contracted features. "
                f"Examples: {bad_cols[:20]}"
            )
        if use_input_aliases:
            out = out.copy()
            out.columns = selected
        return out

    def inference_schema_diagnostics(self, X: Any) -> dict[str, Any]:
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X)
        X_df.columns = [str(c) for c in X_df.columns]
        live_cols = set(map(str, X_df.columns))
        selected = list(map(str, self.selected_features))
        input_features = [
            str(c) for c in getattr(self, "input_feature_names", []) or []
        ]
        contract = (
            input_features
            if len(input_features) == len(selected) and input_features != selected
            else selected
        )
        missing = [c for c in contract if c not in live_cols]
        overlap = len(set(contract) & live_cols)
        nonfinite: list[str] = []
        if not missing and contract:
            try:
                values = X_df.loc[:, contract].astype(np.float32, copy=False)
                nonfinite = [
                    str(col)
                    for col in values.columns
                    if not np.isfinite(values[col].to_numpy(dtype=np.float32, copy=False)).all()
                ]
            except Exception:
                nonfinite = list(contract)
        return {
            "selected_features_count": int(len(contract)),
            "provided_features_count": int(X_df.shape[1]),
            "matched_selected_features_count": int(overlap),
            "missing_selected_features_count": int(len(missing)),
            "missing_selected_features_fraction": float(len(missing) / max(len(contract), 1)),
            "missing_selected_features_preview": missing[:20],
            "nonfinite_selected_features_count": int(len(nonfinite)),
            "nonfinite_selected_features_preview": nonfinite[:20],
            "selected_features_preview": contract[:50],
            "model_feature_names_preview": selected[:50],
        }

    def prediction_diagnostics(self, X: Any | None = None, pred: np.ndarray | None = None) -> dict[str, float]:
        if pred is None:
            if X is None:
                raise ValueError("Either X or pred must be provided for prediction diagnostics")
            pred = self.predict(X)
        return _prediction_diagnostics(pred)

    def feature_distribution_diagnostics(self, X: Any) -> dict[str, Any]:
        X_df = self._frame(X)
        return _feature_distribution_diagnostics(X_df, self.feature_stats_train)

    def predict(self, X: Any) -> np.ndarray:
        X_df = self._frame(X)
        if not self.models:
            fill = 0.5 if self.mode == "classifier" else 0.0
            out = np.full(len(X_df), fill, dtype=np.float32)
        else:
            preds = [_predict_lgbm_raw(model, X_df, self.mode) for model in self.models]
            out = np.mean(np.vstack(preds), axis=0).astype(np.float32)
            if self.mode == "classifier":
                out = np.clip(out, 1e-5, 1.0 - 1e-5)
        diagnostics = _prediction_diagnostics(out)
        if _prediction_ranking_collapsed(diagnostics):
            tprint(
                "WARNING: LGBM prediction ranking may be collapsed: "
                f"std={diagnostics.get('std', 0.0):.6g}, "
                f"unique_rounded_6={diagnostics.get('unique_rounded_6', 0.0):.0f}, "
                f"top1_spread={diagnostics.get('top1_spread', 0.0):.6g}"
            )
        return out.astype(np.float32)

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.mode != "classifier":
            raise ValueError("predict_proba is only valid for classifier LGBMStabilityModel instances; use predict() for ranking scores.")
        p = self.predict(X)
        return np.column_stack([1.0 - p, p]).astype(np.float32)

    def transform_meta_features(self, X: Any) -> pd.DataFrame:
        X_df = self._frame(X)
        features = _lgbm_meta_features_from_models(
            self.models,
            X_df,
            mode=self.mode,
            rank_bin_stats=self.rank_bin_stats_oof,
        )
        _append_feature_drift_meta_features(
            features,
            X_df,
            self.drift_reference,
        )
        return features.reindex(columns=self.meta_feature_names, fill_value=0.0).astype(np.float32)

    def get_training_meta_features(self) -> pd.DataFrame:
        if self.meta_oof_features is None:
            return pd.DataFrame(columns=self.meta_feature_names, dtype=np.float32)
        return self.meta_oof_features.reindex(columns=self.meta_feature_names, fill_value=0.0).copy()


def _frame(X: Any) -> pd.DataFrame:
    X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    X_df = X_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    X_df.columns = [str(c) for c in X_df.columns]
    numeric_cols = X_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols):
        X_df[numeric_cols] = X_df[numeric_cols].astype(np.float32, copy=False)
    return X_df


def _validate_input_lengths(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    sample_weight: Any = None,
    timestamps: Any = None,
    assets: Any = None,
    returns: Any = None,
) -> None:
    n = len(y)
    if len(X) != n:
        raise ValueError(f"X and y must have the same length; got len(X)={len(X)} and len(y)={n}")
    for name, arr in (("sample_weight", sample_weight), ("timestamps", timestamps), ("assets", assets), ("returns", returns)):
        if arr is not None and len(np.asarray(arr)) != n:
            raise ValueError(f"{name} must have the same length as y; got len({name})={len(np.asarray(arr))} and len(y)={n}")


def _feature_selection_oi_present_mask(
    X_raw: Any,
    n: int,
) -> tuple[np.ndarray | None, dict[str, Any], set[str]]:
    diagnostics: dict[str, Any] = {
        "feature_selection_oi_filter_source": "unavailable_assumed_upstream",
        "feature_selection_oi_filter_enforced": False,
        "feature_selection_oi_present_rows_total": None,
        "feature_selection_oi_absent_rows_total": None,
    }
    if not isinstance(X_raw, pd.DataFrame) or len(X_raw) != int(n):
        return None, diagnostics, set()
    lower_to_name = {str(c).lower(): str(c) for c in X_raw.columns}
    availability_names = (
        "__oi_available__",
        "__open_interest_available__",
        "__open_interest_present__",
        "__has_open_interest__",
        "oi_available",
        "open_interest_available",
        "open_interest_present",
        "has_open_interest",
    )
    raw_names = (
        "__open_interest__",
        "open_interest",
        "open_interest_native",
        "native_open_interest",
    )
    for name in availability_names:
        col = lower_to_name.get(name)
        if col is None:
            continue
        values = X_raw[col]
        if pd.api.types.is_bool_dtype(values):
            mask = values.fillna(False).to_numpy(dtype=bool)
        else:
            arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float32)
            mask = np.isfinite(arr) & (arr > 0.0)
        diagnostics.update(
            {
                "feature_selection_oi_filter_source": col,
                "feature_selection_oi_filter_enforced": True,
                "feature_selection_oi_present_rows_total": int(np.sum(mask)),
                "feature_selection_oi_absent_rows_total": int(len(mask) - np.sum(mask)),
            }
        )
        return mask.astype(bool, copy=False), diagnostics, {col}
    for name in raw_names:
        col = lower_to_name.get(name)
        if col is None:
            continue
        arr = pd.to_numeric(X_raw[col], errors="coerce").to_numpy(dtype=np.float32)
        mask = np.isfinite(arr)
        diagnostics.update(
            {
                "feature_selection_oi_filter_source": col,
                "feature_selection_oi_filter_enforced": True,
                "feature_selection_oi_present_rows_total": int(np.sum(mask)),
                "feature_selection_oi_absent_rows_total": int(len(mask) - np.sum(mask)),
            }
        )
        return mask.astype(bool, copy=False), diagnostics, set()
    return None, diagnostics, set()


def _recent_feature_coverage_survivors(
    X_raw: pd.DataFrame,
    timestamps: Any,
    exempt_features: Optional[set[str]] = None,
) -> tuple[list[str], dict[str, Any]]:
    all_cols = [str(c) for c in X_raw.columns]
    exempt = {str(c) for c in (exempt_features or set()) if str(c) in set(all_cols)}
    cols = [c for c in all_cols if c not in exempt]
    diagnostics: dict[str, Any] = {
        "feature_recent_min_coverage": float(LGBM_FEATURE_RECENT_MIN_COVERAGE),
        "feature_recent_tail_months": int(LGBM_FEATURE_RECENT_TAIL_MONTHS),
        "feature_recent_input_count": int(len(all_cols)),
        "feature_recent_coverage_input_count": int(len(cols)),
        "feature_recent_exempt_model_derived_count": int(len(exempt)),
        "feature_recent_exempt_model_derived_preview": sorted(exempt)[:25],
    }
    if X_raw.empty:
        diagnostics["feature_recent_survivor_count"] = int(len(all_cols))
        return all_cols, diagnostics
    if not cols:
        diagnostics.update(
            {
                "feature_recent_row_count": int(len(X_raw)),
                "feature_recent_survivor_count": int(len(all_cols)),
                "feature_recent_removed_count": 0,
                "feature_recent_removed_iterative_count": 0,
                "feature_recent_removed_group_count": 0,
                "feature_recent_removed_groups": [],
                "feature_recent_joint_coverage": 1.0,
                "feature_recent_stopped_no_gain": False,
                "feature_recent_removed_lowest": [],
            }
        )
        return all_cols, diagnostics
    row_mask = np.ones(len(X_raw), dtype=bool)
    if timestamps is not None:
        ts = pd.to_datetime(np.asarray(timestamps), utc=True, errors="coerce")
        if len(ts) == len(X_raw) and pd.notna(ts).any():
            end_ts = ts.max()
            try:
                start_ts = end_ts - pd.DateOffset(
                    months=max(1, LGBM_FEATURE_RECENT_TAIL_MONTHS)
                )
            except Exception:
                start_ts = end_ts - pd.Timedelta(
                    days=30 * max(1, LGBM_FEATURE_RECENT_TAIL_MONTHS)
                )
            candidate = (ts >= start_ts) & (ts <= end_ts)
            if int(candidate.sum()) >= min(200, len(X_raw)):
                row_mask = np.asarray(candidate, dtype=bool)
                diagnostics["feature_recent_window_start"] = str(start_ts)
                diagnostics["feature_recent_window_end"] = str(end_ts)
    price_candidates = (
        "close",
        "price",
        "mark_price",
        "index_price",
        "perp_close",
        "spot_close",
        "__close__",
    )
    price_masks: list[np.ndarray] = []
    price_sources: list[str] = []
    for pc in price_candidates:
        if pc not in X_raw.columns:
            continue
        arr = pd.to_numeric(X_raw[pc], errors="coerce").to_numpy(
            dtype=np.float64, copy=False
        )
        mask = np.isfinite(arr) & (arr > 0.0)
        if int((row_mask & mask).sum()) >= min(200, int(row_mask.sum())):
            price_masks.append(mask)
            price_sources.append(pc)
    if price_masks:
        price_mask = np.logical_or.reduce(price_masks)
        priced_rows = row_mask & price_mask
        if int(priced_rows.sum()) >= min(200, int(row_mask.sum())):
            row_mask = priced_rows
            diagnostics["feature_recent_price_sources"] = price_sources
    sample = X_raw.iloc[np.flatnonzero(row_mask)][cols].apply(
        pd.to_numeric, errors="coerce"
    )
    arr = sample.to_numpy(dtype=np.float32, copy=False)
    finite = np.isfinite(arr)
    feature_coverage = finite.mean(axis=0)
    missing = ~finite
    active = np.ones(len(cols), dtype=bool)
    missing_count = missing.sum(axis=1).astype(np.int32, copy=False)
    joint_coverage = float((missing_count == 0).mean()) if cols else 1.0
    removed_iter: list[tuple[str, float]] = []
    removed_group_iter: list[dict[str, Any]] = []
    stopped_no_gain = False
    n_rows = max(int(len(finite)), 1)

    def _best_group_removal() -> tuple[np.ndarray | None, float, int]:
        """Find a correlated missingness group whose removal unlocks rows."""
        active_idx_inner = np.flatnonzero(active)
        blocker_rows = missing_count > 0
        if len(active_idx_inner) == 0 or not bool(blocker_rows.any()):
            return None, 0.0, 0
        active_missing = missing[np.ix_(blocker_rows, active_idx_inner)]
        if active_missing.size == 0:
            return None, 0.0, 0
        packed = np.packbits(active_missing, axis=1)
        _, first_idx, counts = np.unique(
            packed, axis=0, return_index=True, return_counts=True
        )
        order = np.argsort(counts)[::-1][:256]
        best_group: np.ndarray | None = None
        best_gain = 0.0
        best_score = -1.0
        best_size = 0
        for pos in order:
            local_mask = np.asarray(active_missing[int(first_idx[pos])], dtype=bool)
            group_size = int(local_mask.sum())
            if group_size <= 0:
                continue
            gain = float(counts[pos]) / float(n_rows)
            score = gain / float(group_size)
            if (
                score > best_score
                or (np.isclose(score, best_score) and gain > best_gain)
                or (
                    np.isclose(score, best_score)
                    and np.isclose(gain, best_gain)
                    and (best_size == 0 or group_size < best_size)
                )
            ):
                best_group = active_idx_inner[local_mask]
                best_gain = gain
                best_score = score
                best_size = group_size
        return best_group, best_gain, best_size

    while int(active.sum()) > 0 and joint_coverage < LGBM_FEATURE_RECENT_MIN_COVERAGE:
        active_idx = np.flatnonzero(active)
        single_blocker = missing_count == 1
        if bool(single_blocker.any()):
            removal_gain = (
                missing[np.ix_(single_blocker, active_idx)]
                .sum(axis=0)
                .astype(np.float64)
                / float(len(finite))
            )
        else:
            removal_gain = np.zeros(len(active_idx), dtype=np.float64)
        active_cov = feature_coverage[active_idx]
        best_gain = float(np.max(removal_gain))
        if best_gain <= 0.0:
            group_idx, group_gain, group_size = _best_group_removal()
            if group_idx is None or group_gain <= 0.0:
                stopped_no_gain = True
                break
            missing_rates = 1.0 - feature_coverage[group_idx]
            order = np.argsort(missing_rates)[::-1]
            group_idx = group_idx[order]
            removed_group_iter.append(
                {
                    "features": [cols[int(i)] for i in group_idx[:25]],
                    "feature_count": int(group_size),
                    "gain": float(group_gain),
                }
            )
            for idx in group_idx:
                removed_iter.append((cols[int(idx)], float(1.0 - feature_coverage[int(idx)])))
            active[group_idx] = False
            missing_count -= missing[:, group_idx].sum(axis=1).astype(np.int32, copy=False)
            joint_coverage = float((missing_count == 0).mean()) if active.any() else 1.0
            continue
        else:
            best_local = np.flatnonzero(np.isclose(removal_gain, best_gain))
            worst_local = int(best_local[np.argmin(active_cov[best_local])])
        worst_idx = int(active_idx[worst_local])
        removed_iter.append((cols[worst_idx], float(1.0 - active_cov[worst_local])))
        active[worst_idx] = False
        missing_count -= missing[:, worst_idx].astype(np.int32, copy=False)
        joint_coverage = float((missing_count == 0).mean()) if active.any() else 1.0
    active_survivors = {c for c, is_active in zip(cols, active) if bool(is_active)}
    survivors = [c for c in all_cols if c in exempt or c in active_survivors]
    removed_by_feature = sorted(
        (
            (c, float(1.0 - rate))
            for c, rate in zip(cols, feature_coverage)
            if float(rate) < LGBM_FEATURE_RECENT_MIN_COVERAGE
        ),
        key=lambda item: (-item[1], item[0]),
    )
    removed = removed_iter
    diagnostics.update(
        {
            "feature_recent_row_count": int(row_mask.sum()),
            "feature_recent_survivor_count": int(len(survivors)),
            "feature_recent_removed_count": int(len(removed)),
            "feature_recent_removed_iterative_count": int(len(removed_iter)),
            "feature_recent_removed_group_count": int(len(removed_group_iter)),
            "feature_recent_removed_groups": removed_group_iter[:10],
            "feature_recent_joint_coverage": float(joint_coverage),
            "feature_recent_stopped_no_gain": bool(stopped_no_gain),
            "feature_recent_removed_lowest": removed[:25],
        }
    )
    return survivors, diagnostics


def _prediction_diagnostics(pred: np.ndarray) -> dict[str, float]:
    p = np.asarray(pred, dtype=np.float32)
    finite = p[np.isfinite(p)]
    if len(finite) == 0:
        return {
            "n": float(len(p)),
            "finite_n": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "q01": 0.0,
            "q05": 0.0,
            "q50": 0.0,
            "q95": 0.0,
            "q99": 0.0,
            "unique_rounded_6": 0.0,
            "top5_spread": 0.0,
            "top1_spread": 0.0,
        }
    q = np.quantile(finite, [0.01, 0.05, 0.50, 0.95, 0.99])
    return {
        "n": float(len(p)),
        "finite_n": float(len(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "q01": float(q[0]),
        "q05": float(q[1]),
        "q50": float(q[2]),
        "q95": float(q[3]),
        "q99": float(q[4]),
        "unique_rounded_6": float(len(np.unique(np.round(finite, 6)))),
        "top5_spread": float(q[3] - q[2]),
        "top1_spread": float(q[4] - q[2]),
    }


def _prediction_ranking_collapsed(diagnostics: dict[str, float]) -> bool:
    n = int(diagnostics.get("n", 0.0))
    if n <= 1:
        return False
    return bool(
        float(diagnostics.get("std", 0.0)) < 1e-5
        or float(diagnostics.get("unique_rounded_6", 0.0)) < max(5.0, 0.01 * float(n))
        or float(diagnostics.get("top1_spread", 0.0)) < 1e-4
    )


def _feature_stats_frame(X: pd.DataFrame, features: list[str]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for col in features:
        if col not in X.columns:
            continue
        vals = pd.to_numeric(X[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        finite = vals.dropna()
        if len(finite) == 0:
            stats[col] = {"mean": 0.0, "std": 0.0, "p01": 0.0, "p50": 0.0, "p99": 0.0, "zero_rate": 1.0, "missing_rate": 1.0}
            continue
        stats[col] = {
            "mean": float(finite.mean()),
            "std": float(finite.std(ddof=0)),
            "p01": float(finite.quantile(0.01)),
            "p50": float(finite.quantile(0.50)),
            "p99": float(finite.quantile(0.99)),
            "zero_rate": float((finite == 0.0).mean()),
            "missing_rate": float(1.0 - len(finite) / max(len(vals), 1)),
        }
    return stats


def _feature_distribution_diagnostics(X: pd.DataFrame, train_stats: dict[str, dict[str, float]]) -> dict[str, Any]:
    if not train_stats:
        return {"available": False, "feature_count": 0, "drifted_feature_count": 0, "drifted_features_preview": []}
    live_stats = _feature_stats_frame(X, [c for c in train_stats if c in X.columns])
    drifted: list[dict[str, float | str]] = []
    for col, train in train_stats.items():
        live = live_stats.get(col)
        if live is None:
            drifted.append({"feature": col, "reason": "missing_live_feature"})
            continue
        train_std = max(float(train.get("std", 0.0)), 1e-6)
        mean_z = abs(float(live.get("mean", 0.0)) - float(train.get("mean", 0.0))) / train_std
        live_p99 = float(live.get("p99", 0.0))
        live_p01 = float(live.get("p01", 0.0))
        train_p99 = float(train.get("p99", 0.0))
        train_p01 = float(train.get("p01", 0.0))
        zero_delta = abs(float(live.get("zero_rate", 0.0)) - float(train.get("zero_rate", 0.0)))
        outside = bool(live_p99 > train_p99 + 3.0 * train_std or live_p01 < train_p01 - 3.0 * train_std)
        if mean_z > 5.0 or zero_delta > 0.50 or outside:
            drifted.append(
                {
                    "feature": col,
                    "mean_z": float(mean_z),
                    "zero_rate_delta": float(zero_delta),
                    "outside_train_range": float(outside),
                }
            )
    return {
        "available": True,
        "feature_count": int(len(train_stats)),
        "drifted_feature_count": int(len(drifted)),
        "drifted_feature_fraction": float(len(drifted) / max(len(train_stats), 1)),
        "drifted_features_preview": drifted[:20],
    }


def score_for_trading(model: "LGBMStabilityModel", X_live: pd.DataFrame, group: Any = None) -> np.ndarray:
    if not isinstance(X_live, pd.DataFrame):
        raise ValueError("score_for_trading requires a pandas DataFrame so live feature names can be validated.")
    schema = model.inference_schema_diagnostics(X_live)
    if int(schema["missing_selected_features_count"]):
        raise ValueError(
            "Missing live features: "
            f"{schema['missing_selected_features_count']} examples={schema['missing_selected_features_preview']}"
        )
    if int(schema.get("nonfinite_selected_features_count", 0)):
        raise ValueError(
            "Non-finite live features: "
            f"{schema['nonfinite_selected_features_count']} "
            f"examples={schema.get('nonfinite_selected_features_preview', [])}"
        )
    raw_score = model.predict(X_live)
    if not np.all(np.isfinite(raw_score)):
        raise ValueError("Non-finite model predictions.")
    diagnostics = _prediction_diagnostics(raw_score)
    if _prediction_ranking_collapsed(diagnostics):
        raise ValueError(f"Prediction ranking collapsed or low-spread: {diagnostics}")
    if group is None:
        return pd.Series(raw_score).rank(pct=True).to_numpy(dtype=np.float32)
    if len(np.asarray(group)) != len(raw_score):
        raise ValueError("group must have the same length as X_live")
    return (
        pd.DataFrame({"score": raw_score, "group": np.asarray(group, dtype=object)})
        .groupby("group")["score"]
        .rank(pct=True)
        .to_numpy(dtype=np.float32)
    )


def _normalize_objective_mode(objective_mode: str | None) -> str:
    mode = str(objective_mode or "train_base").lower()
    if mode not in {"train_base", "train_meta"}:
        raise ValueError('hpo_objective_mode must be either "train_base" or "train_meta"')
    return mode


def _distillation_passes_for_objective(objective_mode: str | None = "train_base") -> int:
    passes = max(
        int(LGBM_OOF_DISTILLATION_PASSES),
        int(LGBM_MIN_OOF_DISTILLATION_PASSES),
    )
    if _normalize_objective_mode(objective_mode) == "train_meta":
        passes = max(passes, int(LGBM_META_MIN_OOF_DISTILLATION_PASSES))
    return max(0, passes)


def _objective_value(metrics: dict[str, float], objective_mode: str | None = "train_base") -> float:
    mode = _normalize_objective_mode(objective_mode)
    if LGBM_OBJECTIVE == "tail_control":
        prefix = "meta" if mode == "train_meta" else "base"
        value = metrics.get(
            f"{prefix}_tail_control_score",
            metrics.get("tail_control_score", metrics.get("J_final", np.nan)),
        )
        if not np.isfinite(float(value)):
            return -999.0
        return float(value)
    key = "J_meta" if mode == "train_meta" else "J_base"
    value = metrics.get(key, metrics.get("J_final", np.nan))
    if not np.isfinite(float(value)):
        return -999.0
    return float(value)


def _apply_overfit_gap_penalty(
    train_metrics: dict[str, float],
    valid_metrics: dict[str, float],
    *,
    objective_mode: str | None = "train_base",
    penalty: float = LGBM_OVERFIT_GAP_PENALTY,
    deadband: float = LGBM_OVERFIT_GAP_DEADBAND,
    gap_cap: float = LGBM_OVERFIT_GAP_CAP,
) -> dict[str, float]:
    """Validation-first J with train/valid overfit penalty."""
    out = dict(valid_metrics)
    j_train = _objective_value(train_metrics, objective_mode)
    j_valid = _objective_value(valid_metrics, objective_mode)
    out["J_train"] = float(j_train) if np.isfinite(j_train) else np.nan
    out["J_valid_raw"] = float(j_valid) if np.isfinite(j_valid) else np.nan
    if not np.isfinite(j_train) or not np.isfinite(j_valid):
        out["J_overfit_gap_raw"] = 0.0
        out["J_overfit_gap"] = 0.0
        out["J_overfit_penalty"] = 0.0
        return out
    raw_gap = max(0.0, float(j_train) - float(j_valid))
    gap = max(0.0, raw_gap - float(deadband))
    gap = min(gap, float(gap_cap))
    penalty_value = float(penalty) * gap
    j_penalized = float(j_valid) - penalty_value
    out["J_overfit_gap_raw"] = float(raw_gap)
    out["J_overfit_gap"] = float(gap)
    out["J_overfit_penalty"] = float(penalty_value)
    out["J_final"] = float(j_penalized)
    out["selected_objective"] = float(j_penalized)
    mode = _normalize_objective_mode(objective_mode)
    if LGBM_OBJECTIVE == "tail_control":
        prefix = "meta" if mode == "train_meta" else "base"
        out[f"{prefix}_tail_control_score_raw"] = float(j_valid)
        out[f"{prefix}_tail_control_score"] = float(j_penalized)
        out["tail_control_score_raw"] = float(j_valid)
        out["tail_control_score"] = float(j_penalized)
    if mode == "train_meta":
        out["J_meta"] = float(j_penalized)
    else:
        out["J_base"] = float(j_penalized)
    return out



def train_base(metrics: dict[str, float]) -> float:
    return _objective_value(metrics, "train_base")


def train_meta(metrics: dict[str, float]) -> float:
    return _objective_value(metrics, "train_meta")


def _looks_classifier_target(y: np.ndarray) -> bool:
    yy = np.asarray(y)
    finite = yy[np.isfinite(yy)]
    if len(finite) == 0:
        return True
    unique = np.unique(finite)
    return bool(len(unique) <= 20 and np.all(np.isclose(unique, np.round(unique))))


def _coerce_target(y: np.ndarray, classifier: bool) -> np.ndarray:
    if classifier:
        return np.asarray(y >= 0.5, dtype=np.int8)
    return np.asarray(y, dtype=np.float32)


def _as_returns(y: np.ndarray, returns: Any = None) -> np.ndarray:
    if returns is None:
        return np.asarray(y, dtype=np.float32)
    arr = np.asarray(returns, dtype=np.float32)
    if len(arr) != len(y):
        raise ValueError("returns must have the same length as y")
    return arr


def _rank01(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(vals)
    out = np.zeros(len(vals), dtype=np.float32)
    if int(np.sum(finite)) <= 1:
        return out
    finite_vals = vals[finite]
    span = float(np.nanmax(finite_vals) - np.nanmin(finite_vals))
    if span <= 1e-12:
        return out
    out[finite] = pd.Series(finite_vals).rank(pct=True).to_numpy(dtype=np.float32)
    return out


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    m = np.isfinite(aa) & np.isfinite(bb)
    if int(np.sum(m)) < 8:
        return 0.0
    if float(np.nanstd(aa[m])) <= 1e-12 or float(np.nanstd(bb[m])) <= 1e-12:
        return 0.0
    val = spearmanr(aa[m], bb[m]).correlation
    return float(val) if val is not None and np.isfinite(val) else 0.0


def _top_idx(order: np.ndarray, frac: float, n: int) -> np.ndarray:
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    k = max(1, int(np.ceil(float(frac) * n)))
    return np.asarray(order[-k:], dtype=np.int64)


def _metric_salg(lift: float, stability: float) -> float:
    if not (np.isfinite(lift) and np.isfinite(stability)):
        return float("nan")
    return float(stability + LGBM_SALG_LIFT_COEF * (lift - 1.0))


def _unit_interval(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _normalize_salg_for_objective(salg: float) -> float:
    return _unit_interval(float(salg) / LGBM_J_SALG_NORM_DENOM)


def _target_top_fraction(objective_mode: str | None = "train_meta") -> float:
    mode = _normalize_objective_mode(objective_mode)
    if mode == "train_base":
        return float(LGBM_BASE_METRIC_TARGET_FRACTION)
    return float(LGBM_META_METRIC_TARGET_FRACTION)


def _normalize_precision(precision: float, baseline: float) -> float:
    if not (np.isfinite(precision) and np.isfinite(baseline)):
        return 0.0
    return _unit_interval((float(precision) - float(baseline)) / max(1.0 - float(baseline), 1e-6))


def _normalize_lift(lift: float) -> float:
    if not np.isfinite(lift):
        return 0.0
    return _unit_interval((float(lift) - 1.0) / LGBM_TAIL_LIFT_NORM_DENOM)


def _normalize_return(value: float, scale: float) -> float:
    if not np.isfinite(value):
        return 0.0
    scale = max(float(scale), 1e-6)
    return _unit_interval(0.5 + 0.5 * np.tanh(float(value) / scale))


def _ndcg_at_frac(y_true: np.ndarray, pred: np.ndarray, frac: float) -> float:
    n = len(np.asarray(y_true))
    k = max(1, int(np.ceil(float(frac) * max(n, 1))))
    return _ndcg_at_k(y_true, pred, k=k)


def _ndcg_at_k(y_true: np.ndarray, pred: np.ndarray, k: int) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    m = np.isfinite(y) & np.isfinite(p)
    y = y[m]
    p = p[m]
    if len(y) <= 1:
        return 0.0
    k = max(1, min(int(k), len(y)))
    order = np.argsort(p)[-k:][::-1]
    ideal = np.argsort(y)[-k:][::-1]
    gains = np.maximum(y, 0.0)
    dcg = float(np.sum(gains[order] / np.log2(np.arange(2, k + 2))))
    idcg = float(np.sum(gains[ideal] / np.log2(np.arange(2, k + 2))))
    if idcg <= 1e-12:
        return 0.0
    return float(np.clip(dcg / idcg, 0.0, 1.0))


def _bucket_monotonicity_score(y_win: np.ndarray, order: np.ndarray, *, top_frac: float = 0.20, n_buckets: int = 4) -> dict[str, float]:
    y = np.asarray(y_win, dtype=np.float64)
    bucket_count = max(1, int(n_buckets))
    top_frac_f = float(np.clip(top_frac, 0.001, 0.95))
    empty = {
        "rank_bucket_monotonicity": 0.0,
        "rank_bucket_monotonicity_violation": 1.0,
    }
    for i in range(bucket_count):
        lo = top_frac_f * i / bucket_count
        hi = top_frac_f * (i + 1) / bucket_count
        empty[f"rank_bucket_win_rate_{lo:.4f}_{hi:.4f}"] = 0.0
    if len(y) == 0:
        return empty
    sorted_y = y[np.asarray(order, dtype=np.int64)[::-1]]
    n = len(sorted_y)
    bounds_f = np.linspace(0.0, top_frac_f, bucket_count + 1)
    bounds = np.asarray([0] + [max(1, int(np.ceil(frac * n))) for frac in bounds_f[1:]], dtype=np.int64)
    bounds = np.maximum.accumulate(np.clip(bounds, 0, n))
    sums = np.concatenate([[0.0], np.cumsum(sorted_y, dtype=np.float64)])
    bucket_n = np.maximum(bounds[1:] - bounds[:-1], 1)
    rates = (sums[bounds[1:]] - sums[bounds[:-1]]) / bucket_n
    diffs = rates[1:] - rates[:-1]
    violation = float(np.mean(np.maximum(diffs, 0.0))) if len(diffs) else 0.0
    out = {
        "rank_bucket_monotonicity": float(np.clip(1.0 - violation, 0.0, 1.0)),
        "rank_bucket_monotonicity_violation": violation,
    }
    for i, rate in enumerate(rates):
        lo = top_frac_f * i / bucket_count
        hi = top_frac_f * (i + 1) / bucket_count
        out[f"rank_bucket_win_rate_{lo:.4f}_{hi:.4f}"] = float(rate)
    return out


def _grouped_top_stability(
    y: np.ndarray,
    pred: np.ndarray,
    classifier: bool,
    groups: Any = None,
    frac: float = 0.20,
    min_groups: int = 3,
    min_group_n: int = 20,
) -> dict[str, float]:
    if groups is None:
        return {"stability": 0.0, "n_groups": 0.0, "group_mean": 0.0, "group_std": 0.0}
    yy = np.asarray(y, dtype=np.float64)
    pp = np.asarray(pred, dtype=np.float64)
    gg = np.asarray(groups, dtype=object)
    n = min(len(yy), len(pp), len(gg))
    yy = yy[:n]
    pp = pp[:n]
    gg = gg[:n]
    m = np.isfinite(yy) & np.isfinite(pp)
    yy = yy[m]
    pp = pp[m]
    gg = gg[m]
    vals: list[float] = []
    for group in pd.unique(pd.Series(gg)):
        mask = gg == group
        if int(np.sum(mask)) < int(min_group_n):
            continue
        yg = yy[mask]
        pg = pp[mask]
        k = max(1, int(np.ceil(float(frac) * len(pg))))
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
    cv_v = std_v / (abs(mean_v) + 1e-6)
    return {
        "stability": float(np.clip(1.0 / (1.0 + cv_v), 0.0, 1.0)),
        "n_groups": float(len(arr)),
        "group_mean": mean_v,
        "group_std": std_v,
    }


def _stability_group_bundle(n: int, timestamps: Any = None, assets: Any = None) -> dict[str, np.ndarray] | None:
    if n <= 0:
        return None
    out: dict[str, np.ndarray] = {}
    if timestamps is not None and len(np.asarray(timestamps)) == n:
        ts = pd.to_datetime(np.asarray(timestamps), utc=True, errors="coerce")
        week = pd.Series(ts).dt.tz_localize(None).dt.to_period("W").astype(str).to_numpy(dtype=object)
        month = pd.Series(ts).dt.tz_localize(None).dt.to_period("M").astype(str).to_numpy(dtype=object)
        out["week"] = np.where(pd.isna(week), "__unknown_week__", week).astype(str)
        out["month"] = np.where(pd.isna(month), "__unknown_month__", month).astype(str)
    if assets is not None and len(np.asarray(assets)) == n:
        asset_arr = np.asarray(assets).astype(str)
        counts = pd.Series(asset_arr).value_counts()
        common = set(counts[counts >= 20].index.astype(str))
        out["asset"] = np.asarray([a if a in common else "__rare_asset__" for a in asset_arr], dtype=object).astype(str)
    if not out:
        return None
    if "week" not in out:
        out["week"] = out.get("asset", np.asarray(["__all__"] * n, dtype=str))
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



def _slice_objective_components(
    y: np.ndarray,
    y_win: np.ndarray,
    pred: np.ndarray,
    ret: np.ndarray,
    order: np.ndarray,
    *,
    classifier: bool,
    baseline: float,
    groups: Any,
    target_frac: float,
    prefix: str,
) -> dict[str, float]:
    target = float(np.clip(target_frac, 0.001, 0.95))
    precision_fracs = tuple(float(np.clip(target * mult, 0.001, 0.95)) for mult in (1.0 / 3.0, 2.0 / 3.0, 1.0))
    top_indices = [_top_idx(order, frac, len(y)) for frac in precision_fracs]
    precision_vals = [float(np.mean(y_win[idx])) if len(idx) else 0.0 for idx in top_indices]
    precision_norms = [_normalize_precision(v, baseline) for v in precision_vals]
    target_top = top_indices[-1]
    if classifier:
        lift_target = precision_vals[-1] / max(baseline, 1e-6)
    else:
        denom = float(np.mean(np.abs(y))) + 1e-6
        lift_target = float(np.mean(y[target_top]) / denom) if len(target_top) else 0.0
    stability = _grouped_top_stability(y, pred, classifier, groups=_groups_primary(groups), frac=target)
    stability_target = float(stability["stability"])
    if stability_target <= 0.0:
        top_vals = y_win[target_top] if len(target_top) else np.asarray([], dtype=np.float64)
        stability_target = float(1.0 / (1.0 + np.std(top_vals))) if len(top_vals) else 0.0
    mono = _bucket_monotonicity_score(y_win, order, top_frac=target, n_buckets=4)
    salg_target = _metric_salg(float(lift_target), stability_target)
    normalized_salg_target = _normalize_salg_for_objective(salg_target)
    precision_blend = 0.40 * precision_norms[0] + 0.35 * precision_norms[1] + 0.25 * precision_norms[2]
    objective = float(
        0.25 * normalized_salg_target
        + 0.20 * precision_blend
        + 0.30 * float(mono["rank_bucket_monotonicity"])
        + 0.25 * stability_target
    )
    ndcg_target = _ndcg_at_frac(ret, pred, frac=target)
    out = {
        f"{prefix}_target_frac": target,
        f"{prefix}_precision_frac_1": precision_fracs[0],
        f"{prefix}_precision_frac_2": precision_fracs[1],
        f"{prefix}_precision_frac_3": precision_fracs[2],
        f"{prefix}_precision_1": precision_vals[0],
        f"{prefix}_precision_2": precision_vals[1],
        f"{prefix}_precision_3": precision_vals[2],
        f"{prefix}_precision_norm_1": precision_norms[0],
        f"{prefix}_precision_norm_2": precision_norms[1],
        f"{prefix}_precision_norm_3": precision_norms[2],
        f"{prefix}_precision_blend": float(precision_blend),
        f"{prefix}_lift": float(lift_target),
        f"{prefix}_stability": float(stability_target),
        f"{prefix}_stability_n_groups": float(stability.get("n_groups", 0.0)),
        f"{prefix}_salg": float(salg_target),
        f"{prefix}_normalized_salg": float(normalized_salg_target),
        f"{prefix}_rank_bucket_monotonicity": float(mono["rank_bucket_monotonicity"]),
        f"{prefix}_rank_bucket_monotonicity_violation": float(mono["rank_bucket_monotonicity_violation"]),
        f"{prefix}_ndcg_at_target": float(ndcg_target),
        f"{prefix}_J": objective,
    }
    for key, value in mono.items():
        if key.startswith("rank_bucket_win_rate_"):
            out[f"{prefix}_{key}"] = float(value)
    return out


def _tail_control_slice_metrics(
    y_win: np.ndarray,
    pred: np.ndarray,
    *,
    baseline: float,
    target_frac: float,
) -> dict[str, float]:
    yy = np.asarray(y_win, dtype=np.float64)
    pp = np.asarray(pred, dtype=np.float64)
    m = np.isfinite(yy) & np.isfinite(pp)
    yy = yy[m]
    pp = pp[m]
    if len(yy) < 8:
        return {
            "precision_blend": 0.0,
            "rank_bucket_monotonicity": 0.0,
            "hit_rate_at_target": 0.0,
            "lift_at_target": 0.0,
            "lift_at_target_norm": 0.0,
            "score_at_k": 0.0,
            "n_rows": float(len(yy)),
        }
    target = float(np.clip(target_frac, 0.001, 0.95))
    order = np.argsort(pp)
    precision_fracs = tuple(
        float(np.clip(target * mult, 0.001, 0.95))
        for mult in (1.0 / 3.0, 2.0 / 3.0, 1.0)
    )
    top_indices = [_top_idx(order, frac, len(yy)) for frac in precision_fracs]
    precision_vals = [float(np.mean(yy[idx])) if len(idx) else 0.0 for idx in top_indices]
    precision_norms = [_normalize_precision(v, baseline) for v in precision_vals]
    hit_rate = precision_vals[-1]
    hit_norm = precision_norms[-1]
    lift = float(hit_rate / max(float(baseline), 1e-6))
    lift_norm = _normalize_lift(lift)
    mono = _bucket_monotonicity_score(yy, order, top_frac=target, n_buckets=4)
    precision_blend = float(
        0.40 * precision_norms[0]
        + 0.35 * precision_norms[1]
        + 0.25 * precision_norms[2]
    )
    rank_mono = float(mono["rank_bucket_monotonicity"])
    return {
        "precision_blend": precision_blend,
        "rank_bucket_monotonicity": rank_mono,
        "hit_rate_at_target": float(hit_norm),
        "hit_rate_at_target_raw": float(hit_rate),
        "lift_at_target": float(lift_norm),
        "lift_at_target_raw": float(lift),
        "lift_at_target_norm": float(lift_norm),
        "score_at_k": float(precision_blend + rank_mono),
        "n_rows": float(len(yy)),
    }


def _tail_group_values(
    y_win: np.ndarray,
    pred: np.ndarray,
    group_values: Any,
    *,
    baseline: float,
    target_frac: float,
    min_rows: int,
) -> list[dict[str, float]]:
    groups = np.asarray(group_values, dtype=object)
    yy = np.asarray(y_win, dtype=np.float64)
    pp = np.asarray(pred, dtype=np.float64)
    n = min(len(yy), len(pp), len(groups))
    if n <= 0:
        return []
    yy = yy[:n]
    pp = pp[:n]
    groups = groups[:n]
    out: list[dict[str, float]] = []
    for group in pd.unique(pd.Series(groups).astype(str)):
        mask = groups.astype(str) == str(group)
        if int(np.sum(mask)) < int(min_rows):
            continue
        metrics = _tail_control_slice_metrics(
            yy[mask],
            pp[mask],
            baseline=baseline,
            target_frac=target_frac,
        )
        metrics["group"] = str(group)  # type: ignore[assignment]
        out.append(metrics)
    return out


def _tail_rolling_values(
    y_win: np.ndarray,
    pred: np.ndarray,
    *,
    baseline: float,
    target_frac: float,
    window_rows: int = LGBM_TAIL_ROLLING_ROWS,
) -> list[dict[str, float]]:
    yy = np.asarray(y_win, dtype=np.float64)
    pp = np.asarray(pred, dtype=np.float64)
    n = min(len(yy), len(pp))
    if n < 8:
        return []
    window = max(8, min(int(window_rows), n))
    out: list[dict[str, float]] = []
    start = 0
    bucket = 0
    while start < n:
        end = min(n, start + window)
        if end - start >= 8:
            metrics = _tail_control_slice_metrics(
                yy[start:end],
                pp[start:end],
                baseline=baseline,
                target_frac=target_frac,
            )
            metrics["group"] = f"rolling_{bucket:04d}"  # type: ignore[assignment]
            out.append(metrics)
        start = end
        bucket += 1
    return out


def _bottom_tail_mean(records: list[dict[str, float]], key: str = "score_at_k") -> float:
    vals = np.asarray(
        [float(r.get(key, np.nan)) for r in records if np.isfinite(float(r.get(key, np.nan)))],
        dtype=np.float64,
    )
    if len(vals) == 0:
        return 0.0
    n_tail = max(1, int(np.ceil(0.20 * len(vals))))
    return float(np.mean(np.sort(vals)[:n_tail]))


def _tail_control_metrics(
    y_win: np.ndarray,
    pred: np.ndarray,
    *,
    baseline: float,
    groups: Any,
    target_frac: float,
) -> dict[str, float]:
    global_metrics = _tail_control_slice_metrics(
        y_win,
        pred,
        baseline=baseline,
        target_frac=target_frac,
    )
    week_records: list[dict[str, float]] = []
    asset_records: list[dict[str, float]] = []
    if isinstance(groups, dict):
        if groups.get("week") is not None:
            week_records = _tail_group_values(
                y_win,
                pred,
                groups["week"],
                baseline=baseline,
                target_frac=target_frac,
                min_rows=LGBM_TAIL_WEEK_MIN_ROWS,
            )
        if groups.get("asset") is not None:
            asset_records = _tail_group_values(
                y_win,
                pred,
                groups["asset"],
                baseline=baseline,
                target_frac=target_frac,
                min_rows=LGBM_TAIL_ASSET_MIN_ROWS,
            )
    if not week_records:
        week_records = _tail_rolling_values(
            y_win,
            pred,
            baseline=baseline,
            target_frac=target_frac,
            window_rows=LGBM_TAIL_ROLLING_ROWS,
        )
    mean_score = float(
        0.40 * float(global_metrics["precision_blend"])
        + 0.20 * float(global_metrics["hit_rate_at_target"])
        + 0.20 * float(global_metrics["lift_at_target"])
        + 0.20 * float(global_metrics["rank_bucket_monotonicity"])
    )
    tail_week = _bottom_tail_mean(week_records)
    tail_asset = _bottom_tail_mean(asset_records) if asset_records else float(global_metrics["score_at_k"])
    robust_tail = float(0.70 * mean_score + 0.15 * tail_week + 0.15 * tail_asset)
    return {
        "tail_precision_blend": float(global_metrics["precision_blend"]),
        "tail_rank_bucket_monotonicity": float(global_metrics["rank_bucket_monotonicity"]),
        "hit_rate_at_target": float(global_metrics["hit_rate_at_target"]),
        "hit_rate_at_target_raw": float(global_metrics.get("hit_rate_at_target_raw", 0.0)),
        "lift_at_target": float(global_metrics["lift_at_target"]),
        "lift_at_target_raw": float(global_metrics.get("lift_at_target_raw", 0.0)),
        "weekly_score_at_k_mean": float(
            np.mean([float(r["score_at_k"]) for r in week_records]) if week_records else 0.0
        ),
        "asset_score_at_k_mean": float(
            np.mean([float(r["score_at_k"]) for r in asset_records]) if asset_records else float(global_metrics["score_at_k"])
        ),
        "mean_score": mean_score,
        "tail_week_20_score": tail_week,
        "tail_asset_20_score": tail_asset,
        "robust_tail_score": robust_tail,
        "tail_control_score": robust_tail,
        "tail_week_group_count": float(len(week_records)),
        "tail_asset_group_count": float(len(asset_records)),
    }


def _metric_pack(
    y_true: np.ndarray,
    pred: np.ndarray,
    *,
    classifier: bool,
    groups: Any = None,
    returns: Any = None,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    ret = _as_returns(y, returns).astype(np.float64)
    m = np.isfinite(y) & np.isfinite(p) & np.isfinite(ret)
    if isinstance(groups, dict):
        grp = {k: np.asarray(v, dtype=object)[m] for k, v in groups.items() if len(np.asarray(v)) == len(m)}
    else:
        grp = np.asarray(groups, dtype=object)[m] if groups is not None and len(np.asarray(groups)) == len(m) else None
    y = y[m]
    p = p[m]
    ret = ret[m]
    if len(y) < 8:
        return {
            "J_base": 0.0,
            "J_meta": 0.0,
            "J_final": 0.0,
            "J_Score": 0.0,
            "precision10_norm": 0.0,
            "precision20_norm": 0.0,
            "precision15": 0.0,
            "precision30": 0.0,
            "precision15_pct": 0.0,
            "precision30_pct": 0.0,
            "ndcg_at_10": 0.0,
            "ndcg_at_20": 0.0,
            "ndcg_at_15": 0.0,
            "ndcg_at_30": 0.0,
            "ndcg_at_15_pct": 0.0,
            "ndcg_at_30_pct": 0.0,
            "rank_bucket_monotonicity": 0.0,
            "stability20": 0.0,
            "lift20": 1.0,
        }
    order = np.argsort(p)
    top10 = _top_idx(order, 0.10, len(y))
    top15 = _top_idx(order, 0.15, len(y))
    top20 = _top_idx(order, 0.20, len(y))
    top30 = _top_idx(order, 0.30, len(y))
    if classifier:
        y_win = (y >= 0.5).astype(np.float64)
        baseline = float(np.mean(y_win))
        precision10 = float(np.mean(y_win[top10])) if len(top10) else 0.0
        precision15 = float(np.mean(y_win[top15])) if len(top15) else 0.0
        precision20 = float(np.mean(y_win[top20])) if len(top20) else 0.0
        precision30 = float(np.mean(y_win[top30])) if len(top30) else 0.0
        lift10 = precision10 / max(baseline, 1e-6)
        lift20 = precision20 / max(baseline, 1e-6)
        lift30 = precision30 / max(baseline, 1e-6)
        auc = float(roc_auc_score(y_win, p)) if len(np.unique(y_win)) > 1 else 0.5
        pr_auc = float(average_precision_score(y_win, p)) if len(np.unique(y_win)) > 1 else baseline
        brier = float(brier_score_loss(y_win, np.clip(p, 1e-6, 1.0 - 1e-6)))
    else:
        y_win = (y > 0.0).astype(np.float64)
        baseline = float(np.mean(y_win))
        precision10 = float(np.mean(y_win[top10])) if len(top10) else 0.0
        precision15 = float(np.mean(y_win[top15])) if len(top15) else 0.0
        precision20 = float(np.mean(y_win[top20])) if len(top20) else 0.0
        precision30 = float(np.mean(y_win[top30])) if len(top30) else 0.0
        denom = float(np.mean(np.abs(y))) + 1e-6
        lift10 = float(np.mean(y[top10]) / denom) if len(top10) else 0.0
        lift20 = float(np.mean(y[top20]) / denom) if len(top20) else 0.0
        lift30 = float(np.mean(y[top30]) / denom) if len(top30) else 0.0
        auc = max(0.0, _safe_spearman(y, p))
        pr_auc = auc
        brier = float(np.mean((p - y) ** 2))
    ret_scale = float(np.nanpercentile(np.abs(ret), 75.0) + 1e-6)
    mean_ret10 = float(np.mean(ret[top10])) if len(top10) else 0.0
    mean_ret20 = float(np.mean(ret[top20])) if len(top20) else 0.0
    norm_ret10 = _normalize_return(mean_ret10, ret_scale)
    norm_ret20 = _normalize_return(mean_ret20, ret_scale)
    precision10_norm = _normalize_precision(precision10, baseline)
    precision15_norm = _normalize_precision(precision15, baseline)
    precision20_norm = _normalize_precision(precision20, baseline)
    precision30_norm = _normalize_precision(precision30, baseline)
    ndcg10 = _ndcg_at_frac(ret, p, frac=0.10)
    ndcg15 = _ndcg_at_frac(ret, p, frac=0.15)
    ndcg20 = _ndcg_at_frac(ret, p, frac=0.20)
    ndcg30 = _ndcg_at_frac(ret, p, frac=0.30)
    mono = _bucket_monotonicity_score(y_win, order, top_frac=0.20)
    stability = _grouped_top_stability(y, p, classifier, groups=_groups_primary(grp), frac=0.20)
    stability20 = float(stability["stability"])
    if stability20 <= 0.0:
        top_vals = y_win[top20] if len(top20) else np.asarray([], dtype=np.float64)
        stability20 = float(1.0 / (1.0 + np.std(top_vals))) if len(top_vals) else 0.0
    net_return_blend = 0.60 * norm_ret10 + 0.40 * norm_ret20
    precision_blend = 0.60 * precision10_norm + 0.40 * precision20_norm
    precision_blend_top10_20_30 = 0.40 * precision10_norm + 0.35 * precision20_norm + 0.25 * precision30_norm
    ndcg_blend = 0.60 * ndcg10 + 0.40 * ndcg20
    salg20 = _metric_salg(float(lift20), stability20)
    normalized_salg20 = _normalize_salg_for_objective(salg20)
    base_components = _slice_objective_components(
        y,
        y_win,
        p,
        ret,
        order,
        classifier=classifier,
        baseline=baseline,
        groups=grp,
        target_frac=_target_top_fraction("train_base"),
        prefix="base",
    )
    meta_components = _slice_objective_components(
        y,
        y_win,
        p,
        ret,
        order,
        classifier=classifier,
        baseline=baseline,
        groups=grp,
        target_frac=_target_top_fraction("train_meta"),
        prefix="meta",
    )
    tail_base_components = _tail_control_metrics(
        y_win,
        p,
        baseline=baseline,
        groups=grp,
        target_frac=_target_top_fraction("train_base"),
    )
    tail_meta_components = _tail_control_metrics(
        y_win,
        p,
        baseline=baseline,
        groups=grp,
        target_frac=_target_top_fraction("train_meta"),
    )
    j_base = float(base_components["base_J"])
    j_meta = float(meta_components["meta_J"])
    out = {
        "J_base": j_base,
        "J_meta": j_meta,
        "J_final": j_base,
        "J_Score": j_base,
        "selected_objective": j_base,
        "base_target_frac": float(base_components["base_target_frac"]),
        "meta_target_frac": float(meta_components["meta_target_frac"]),
        "net_return_blend": float(net_return_blend),
        "normalized_net_mean_ret10": float(norm_ret10),
        "normalized_net_mean_ret20": float(norm_ret20),
        "mean_ret10": float(mean_ret10),
        "mean_ret20": float(mean_ret20),
        "precision_blend": float(precision_blend),
        "precision_blend_top10_20_30": float(precision_blend_top10_20_30),
        "salg20": float(salg20),
        "normalized_salg20": float(normalized_salg20),
        "precision10": float(precision10),
        "precision15": float(precision15),
        "precision20": float(precision20),
        "precision30": float(precision30),
        "precision15_pct": float(precision15),
        "precision30_pct": float(precision30),
        "precision10_norm": float(precision10_norm),
        "precision15_norm": float(precision15_norm),
        "precision20_norm": float(precision20_norm),
        "precision30_norm": float(precision30_norm),
        "NDCG_blend": float(ndcg_blend),
        "ndcg_at_10": float(ndcg10),
        "ndcg_at_15": float(ndcg15),
        "ndcg_at_20": float(ndcg20),
        "ndcg_at_30": float(ndcg30),
        "ndcg_at_15_pct": float(ndcg15),
        "ndcg_at_30_pct": float(ndcg30),
        "lift10": float(lift10),
        "lift20": float(lift20),
        "lift30": float(lift30),
        "baseline_win_rate": float(baseline),
        "stability20": float(stability20),
        "stability20_n_groups": float(stability.get("n_groups", 0.0)),
        "auc": float(auc),
        "pr_auc": float(pr_auc),
        "brier": float(brier),
        "oof_std": float(np.std(p)),
    }
    out.update(base_components)
    out.update(meta_components)
    out.update({f"base_{k}": float(v) for k, v in tail_base_components.items()})
    out.update({f"meta_{k}": float(v) for k, v in tail_meta_components.items()})
    mode_for_tail = _normalize_objective_mode("train_meta" if _target_top_fraction("train_meta") < _target_top_fraction("train_base") else "train_base")
    selected_tail = tail_meta_components if mode_for_tail == "train_meta" else tail_base_components
    out.update({k: float(v) for k, v in selected_tail.items()})
    out.update(mono)
    return out


def _aggregate_j(fold_metrics: list[dict[str, float]], objective_mode: str | None = "train_base") -> dict[str, Any]:
    if not fold_metrics:
        return {"J_final": -999.0, "J_mean": -999.0, "J_std": 0.0, "J_se": 0.0, "J_median": -999.0, "J_iqr": 0.0, "J_robust": -999.0}
    vals = np.asarray([_objective_value(m, objective_mode) for m in fold_metrics], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"J_final": -999.0, "J_mean": -999.0, "J_std": 0.0, "J_se": 0.0, "J_median": -999.0, "J_iqr": 0.0, "J_robust": -999.0}
    q25, q50, q75 = np.percentile(vals, [25, 50, 75])
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    se = float(std / np.sqrt(max(len(vals), 1)))
    robust = float(q50 - 0.50 * (q75 - q25))
    means: dict[str, float] = {}
    for key in sorted(set().union(*(m.keys() for m in fold_metrics))):
        arr = np.asarray([float(m.get(key, np.nan)) for m in fold_metrics], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if len(arr):
            means[key] = float(np.mean(arr))
    means.update({"J_final": robust, "selected_objective": robust, "selected_objective_mode": _normalize_objective_mode(objective_mode), "J_mean": float(np.mean(vals)), "J_std": std, "J_se": se, "J_median": float(q50), "J_iqr": float(q75 - q25), "J_robust": robust})
    return means



def _record_lgbm_stage_metric_comparison(
    metrics_out: dict[str, Any],
    *,
    candidate_metrics: dict[str, Any],
    fit_oof_metrics: dict[str, Any] | None,
    post_distill_metrics: dict[str, Any],
) -> None:
    metric_keys = (
        "J_final",
        "J_meta",
        "J_base",
        "J_Score",
        "base_target_frac",
        "meta_target_frac",
        "base_salg",
        "meta_salg",
        "base_normalized_salg",
        "meta_normalized_salg",
        "base_precision_blend",
        "meta_precision_blend",
        "base_rank_bucket_monotonicity",
        "meta_rank_bucket_monotonicity",
        "base_stability",
        "meta_stability",
        "salg20",
        "normalized_salg20",
        "precision_blend_top10_20_30",
        "precision10_norm",
        "precision20_norm",
        "precision30_norm",
        "lift20",
        "lift30",
        "lift10",
        "mean_return10_gross",
        "mean_return30_gross",
        "hit_tp2_sl1_top10",
        "hit_tp2_sl1_top30",
        "hit_tp3_sl15_top10",
        "hit_tp3_sl15_top30",
        "stability20",
        "rank_bucket_monotonicity",
        "ndcg_at_20",
        "auc",
        "brier",
    )
    stages = (
        ("candidate_prune", candidate_metrics),
        ("fit_oof", fit_oof_metrics or {}),
        ("post_distill", post_distill_metrics),
    )
    rows: list[tuple[str, float | None, float | None, float | None]] = []
    for key in metric_keys:
        vals: list[float | None] = []
        for stage_name, stage_metrics in stages:
            value = stage_metrics.get(key)
            try:
                value_f = float(value)
            except Exception:
                value_f = float("nan")
            if np.isfinite(value_f):
                metrics_out[f"metric_stage_{stage_name}_{key}"] = value_f
                vals.append(value_f)
            else:
                vals.append(None)
        if vals[0] is not None and vals[1] is not None:
            metrics_out[f"metric_stage_delta_candidate_prune_to_fit_oof_{key}"] = vals[1] - vals[0]
        if vals[1] is not None and vals[2] is not None:
            metrics_out[f"metric_stage_delta_fit_oof_to_post_distill_{key}"] = vals[2] - vals[1]
        if any(v is not None for v in vals):
            rows.append((key, vals[0], vals[1], vals[2]))
    if not rows:
        return
    metric_w = max(len("Metric"), max(len(r[0]) for r in rows))
    val_w = 14
    lines = [
        "LGBM stage metric comparison (candidate/prune -> fit_oof -> post-distill)",
        (
            f"{'Metric':<{metric_w}} | "
            f"{'Cand/prune':>{val_w}} | "
            f"{'Fit OOF':>{val_w}} | "
            f"{'Post-distill':>{val_w}}"
        ),
        "-" * (metric_w + val_w * 3 + 9),
    ]
    for key, cand, fit, post in rows:
        def _fmt(v: float | None) -> str:
            return "-" if v is None else f"{v:.4f}"

        lines.append(
            f"{key:<{metric_w}} | "
            f"{_fmt(cand):>{val_w}} | "
            f"{_fmt(fit):>{val_w}} | "
            f"{_fmt(post):>{val_w}}"
        )
    for line in lines:
        tprint(line)

def _stratified_subsample_indices(y: np.ndarray, max_n: int, random_state: int, classifier: bool) -> np.ndarray:
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


def _evenly_spaced_take(ids: np.ndarray, take: int) -> np.ndarray:
    ids_arr = np.asarray(ids, dtype=np.int32)
    if take <= 0 or len(ids_arr) == 0:
        return np.array([], dtype=np.int32)
    if len(ids_arr) <= take:
        return np.sort(ids_arr.astype(np.int32, copy=False))
    positions = np.linspace(0, len(ids_arr) - 1, int(take), dtype=np.int32)
    return np.sort(ids_arr[positions].astype(np.int32, copy=False))


def _stratified_spread_subsample_indices(
    y: np.ndarray,
    max_n: int,
    random_state: int,
    classifier: bool,
) -> np.ndarray:
    """Stratified cap that spans the full ordered sample instead of a local block."""
    n = len(y)
    if n <= max_n:
        return np.arange(n, dtype=np.int32)
    if max_n <= 0:
        return np.array([], dtype=np.int32)
    if classifier:
        strata = np.asarray(y >= 0.5, dtype=np.int8)
    else:
        ranks = pd.Series(np.asarray(y, dtype=np.float32)).rank(pct=True).to_numpy()
        strata = np.clip((ranks * 5).astype(np.int32), 0, 4)
    out: list[np.ndarray] = []
    for s in np.unique(strata):
        ids = np.where(strata == s)[0].astype(np.int32)
        take = max(1, int(round(max_n * len(ids) / n)))
        take = min(take, len(ids))
        out.append(_evenly_spaced_take(ids, take))
    idx = np.sort(np.concatenate(out).astype(np.int32)) if out else np.array([], dtype=np.int32)
    if len(idx) > max_n:
        idx = _evenly_spaced_take(idx, int(max_n))
    if len(idx) < min(max_n, n):
        missing = np.setdiff1d(np.arange(n, dtype=np.int32), idx, assume_unique=False)
        fill = _evenly_spaced_take(missing, min(max_n, n) - len(idx))
        idx = np.sort(np.concatenate([idx, fill]).astype(np.int32))
    return idx


def _stage_partition_indices(y: np.ndarray, *, timestamps: Any = None, assets: Any = None, random_state: int) -> dict[str, np.ndarray]:
    y_arr = np.asarray(y)
    n = len(y_arr)
    if n == 0:
        empty = np.array([], dtype=np.int32)
        return {"lgbm_select": empty, "hpo": empty, "fit_oof": empty}
    classifier = _looks_classifier_target(y_arr)
    if classifier:
        y_bucket = np.asarray(y_arr >= 0.5, dtype=np.int8).astype(str)
    else:
        ranks = pd.Series(np.asarray(y_arr, dtype=np.float32)).rank(pct=True).to_numpy()
        y_bucket = np.clip((ranks * 5).astype(np.int32), 0, 4).astype(str)
    if assets is not None and len(np.asarray(assets)) == n:
        asset_arr = np.asarray(assets).astype(str)
        counts = pd.Series(asset_arr).value_counts()
        common = set(counts[counts >= 20].index.astype(str))
        asset_bucket = np.asarray([a if a in common else "__rare_asset__" for a in asset_arr], dtype=object)
    else:
        asset_bucket = np.asarray(["__all_assets__"] * n, dtype=object)
    if timestamps is not None and len(np.asarray(timestamps)) == n:
        ts = pd.to_datetime(np.asarray(timestamps), utc=True, errors="coerce")
        if bool(pd.Series(ts).notna().any()):
            week = pd.Series(ts).dt.tz_localize(None).dt.to_period("W").astype(str).to_numpy()
            week_rank = pd.Series(week).rank(method="dense").to_numpy(dtype=np.int32)
        else:
            week_rank = np.arange(n, dtype=np.int32)
    else:
        week_rank = np.arange(n, dtype=np.int32)
    strata = np.asarray([f"{yb}|{ab}" for yb, ab in zip(y_bucket, asset_bucket)], dtype=object)
    rng = np.random.default_rng(random_state)
    pattern = np.asarray(["lgbm_select"] * 7 + ["hpo"] * 2 + ["fit_oof"] * 11)
    out: dict[str, list[int]] = {"lgbm_select": [], "hpo": [], "fit_oof": []}
    for stratum in np.unique(strata):
        ids = np.where(strata == stratum)[0]
        jitter = rng.random(len(ids)) * 1e-6
        order = np.lexsort((jitter, np.arange(len(ids)) % 997, week_rank[ids]))
        ordered = ids[order]
        offset = int(rng.integers(0, len(pattern)))
        labels = pattern[(np.arange(len(ordered)) + offset) % len(pattern)]
        for key in out:
            out[key].extend(ordered[labels == key].tolist())
    result = {key: np.asarray(sorted(vals), dtype=np.int32) for key, vals in out.items()}
    assigned = np.concatenate([v for v in result.values() if len(v)])
    missing = np.setdiff1d(np.arange(n, dtype=np.int32), assigned, assume_unique=False)
    if len(missing):
        result["fit_oof"] = np.asarray(sorted(np.concatenate([result["fit_oof"], missing]).tolist()), dtype=np.int32)
    tprint(
        "LGBM stage split: "
        f"select={len(result['lgbm_select'])}/{n}, hpo={len(result['hpo'])}/{n}, "
        f"fit_oof={len(result['fit_oof'])}/{n}."
    )
    return result


def _subsample_stage_indices(stage_indices: dict[str, np.ndarray], y: np.ndarray, *, max_fraction: float, random_state: int, classifier: bool) -> dict[str, np.ndarray]:
    frac = float(np.clip(float(max_fraction), 0.01, 1.0))
    if frac >= 0.999:
        return stage_indices
    n = len(y)
    cap = max(1, int(np.ceil(frac * max(n, 1))))
    out = dict(stage_indices)
    for offset, stage_key in enumerate(("lgbm_select", "hpo", "fit_oof"), start=1):
        idx = np.asarray(out.get(stage_key, []), dtype=np.int32)
        if len(idx) <= cap:
            continue
        sampler = (
            _stratified_spread_subsample_indices
            if stage_key == "lgbm_select"
            else _stratified_subsample_indices
        )
        keep_local = sampler(np.asarray(y, dtype=np.float32)[idx], max_n=cap, random_state=int(random_state) + offset * 10007, classifier=classifier)
        out[stage_key] = np.sort(idx[keep_local].astype(np.int32))
    return out


def _cap_stage_and_move_unused_to_fit_oof(stage_indices: dict[str, np.ndarray], y: np.ndarray, *, stage_key: str, cap: int, random_state: int, classifier: bool, spread: bool = False) -> dict[str, np.ndarray]:
    if cap <= 0:
        return stage_indices
    idx = np.asarray(stage_indices.get(stage_key, []), dtype=np.int32)
    if len(idx) <= cap:
        return stage_indices
    sampler = _stratified_spread_subsample_indices if spread else _stratified_subsample_indices
    keep_local = sampler(np.asarray(y, dtype=np.float32)[idx], max_n=int(cap), random_state=int(random_state), classifier=classifier)
    keep = np.sort(idx[keep_local].astype(np.int32))
    unused = np.setdiff1d(idx, keep, assume_unique=False).astype(np.int32)
    out = dict(stage_indices)
    out[stage_key] = keep
    out["fit_oof"] = np.asarray(sorted(np.unique(np.concatenate([out.get("fit_oof", np.array([], dtype=np.int32)), unused])).tolist()), dtype=np.int32)
    return out


def _splitter(y: np.ndarray, classifier: bool, random_state: int, n_splits: int = LGBM_CV_SPLITS) -> Any:
    y_split = np.asarray(y >= 0.5, dtype=np.int8) if classifier else np.asarray(y, dtype=np.float32)
    if classifier and len(np.unique(y_split)) > 1 and np.min(np.bincount(y_split, minlength=2)) >= n_splits:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state), y_split
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state), y_split


class _PrecomputedSplitter:
    def __init__(self, folds: list[tuple[np.ndarray, np.ndarray]]) -> None:
        self._folds = folds

    def split(self, X: Any, y: Any = None, groups: Any = None) -> Any:
        del X, y, groups
        for tr, va in self._folds:
            yield tr, va


def _interleaved_spread_splitter(
    y: np.ndarray,
    classifier: bool,
    n_splits: int = LGBM_CV_SPLITS,
) -> tuple[Any, np.ndarray]:
    """Build HPO folds whose validation rows span the full ordered sample.

    Unlike shuffled KFold, this preserves the ordered sample axis and assigns
    every nth row within each target stratum to the same validation fold. The
    result is non-consecutive validation coverage across early/middle/late rows.
    """
    n = len(y)
    n_splits_local = max(2, min(int(n_splits), max(2, n)))
    if classifier:
        y_split = np.asarray(y >= 0.5, dtype=np.int8)
        strata = y_split.copy()
    else:
        y_split = np.asarray(y, dtype=np.float32)
        ranks = pd.Series(y_split).rank(pct=True).to_numpy()
        strata = np.clip((ranks * 5).astype(np.int32), 0, 4)

    fold_ids = np.full(n, -1, dtype=np.int32)
    for s in np.unique(strata):
        ids = np.where(strata == s)[0].astype(np.int32)
        if len(ids) == 0:
            continue
        fold_ids[ids] = np.arange(len(ids), dtype=np.int32) % n_splits_local
    if np.any(fold_ids < 0):
        missing = np.where(fold_ids < 0)[0].astype(np.int32)
        fold_ids[missing] = np.arange(len(missing), dtype=np.int32) % n_splits_local

    all_idx = np.arange(n, dtype=np.int32)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_i in range(n_splits_local):
        va = all_idx[fold_ids == fold_i]
        if len(va) == 0:
            continue
        tr = all_idx[fold_ids != fold_i]
        folds.append((tr.astype(np.int32, copy=False), va.astype(np.int32, copy=False)))
    if len(folds) < 2:
        return _splitter(y, classifier, random_state=0, n_splits=n_splits_local)
    return _PrecomputedSplitter(folds), y_split


def _direction_score_for_feature(
    x: np.ndarray,
    y: np.ndarray,
    *,
    classifier: bool,
    groups: Any = None,
    returns: Any = None,
    objective_mode: str | None = "train_base",
) -> tuple[float, float, int, float]:
    x_arr = np.asarray(x, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    m = np.isfinite(x_arr) & np.isfinite(y_arr)
    if returns is not None and len(np.asarray(returns)) == len(y_arr):
        ret_arr = np.asarray(returns, dtype=np.float32)
        m &= np.isfinite(ret_arr)
    else:
        ret_arr = y_arr
    x_arr = x_arr[m]
    y_arr = y_arr[m]
    ret_arr = ret_arr[m] if len(ret_arr) == len(m) else ret_arr
    if LGBM_DIRECTION_MAX_ROWS > 0 and len(x_arr) > LGBM_DIRECTION_MAX_ROWS:
        pre_sample_n = len(x_arr)
        idx = np.linspace(0, pre_sample_n - 1, int(LGBM_DIRECTION_MAX_ROWS), dtype=np.int32)
        x_arr = x_arr[idx]
        y_arr = y_arr[idx]
        ret_arr = ret_arr[idx] if len(ret_arr) == pre_sample_n else ret_arr
    if len(x_arr) < 32 or float(np.nanstd(x_arr)) <= 1e-12:
        return 0.0, 0.0, 0, 0.0
    try:
        q = min(10, max(3, int(np.sqrt(len(x_arr) // 20))))
        bins = pd.qcut(pd.Series(x_arr), q=q, labels=False, duplicates="drop").to_numpy()
    except Exception:
        ranks = pd.Series(x_arr).rank(pct=True).to_numpy()
        bins = np.clip((ranks * 10).astype(np.int32), 0, 9)
    bins = np.asarray(bins, dtype=np.float32)
    finite = np.isfinite(bins)
    bins = bins[finite].astype(np.int32)
    y_arr = y_arr[finite]
    ret_arr = ret_arr[finite] if len(ret_arr) == len(finite) else ret_arr
    if len(bins) < 32 or len(np.unique(bins)) < 2:
        return 0.0, 0.0, 0, 0.0
    y_bin = (y_arr >= 0.5).astype(np.int32) if classifier else (np.asarray(ret_arr) > 0.0).astype(np.int32)
    baseline = float(np.mean(y_bin))
    if baseline <= 1e-6 or baseline >= 1.0 - 1e-6:
        return 0.0, 0.0, 0, 0.0
    levels = np.unique(bins)
    low_level = int(levels[0])
    high_level = int(levels[-1])
    low = bins == low_level
    high = bins == high_level
    low_rate = float(np.mean(y_bin[low])) if int(np.sum(low)) else baseline
    high_rate = float(np.mean(y_bin[high])) if int(np.sum(high)) else baseline
    lift_delta = high_rate - low_rate

    joint = np.zeros((len(levels), 2), dtype=np.float64)
    level_to_i = {int(v): i for i, v in enumerate(levels)}
    for b, yy in zip(bins, y_bin):
        joint[level_to_i[int(b)], int(yy)] += 1.0
    joint /= max(float(np.sum(joint)), 1.0)
    px = np.sum(joint, axis=1, keepdims=True)
    py = np.sum(joint, axis=0, keepdims=True)
    denom = px @ py
    nz = (joint > 0.0) & (denom > 0.0)
    mi = float(np.sum(joint[nz] * np.log(joint[nz] / denom[nz])))
    direction = 1 if lift_delta >= 0.0 else -1
    margin = float(abs(lift_delta) * np.sqrt(max(mi, 0.0) + 1e-12))
    pos_score = max(lift_delta, 0.0) + mi
    neg_score = max(-lift_delta, 0.0) + mi
    return float(pos_score), float(neg_score), int(direction), float(margin)


def _sample_direction_indices(n: int, random_state: int) -> np.ndarray:
    if LGBM_DIRECTION_MAX_ROWS <= 0 or n <= LGBM_DIRECTION_MAX_ROWS:
        return np.arange(n, dtype=np.int32)
    return _evenly_spaced_take(np.arange(n, dtype=np.int32), int(LGBM_DIRECTION_MAX_ROWS))


def _direction_vectors_binned_mi(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    classifier: bool,
    groups: Any = None,
    returns: Any = None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    idx = _sample_direction_indices(len(X), random_state)
    y_s = np.asarray(y)[idx]
    ret_s = (
        np.asarray(returns)[idx]
        if returns is not None and len(np.asarray(returns)) == len(X)
        else returns
    )
    dirs = np.zeros(X.shape[1], dtype=np.float32)
    margins = np.zeros(X.shape[1], dtype=np.float32)
    for j in range(X.shape[1]):
        _jp, _jn, direction, margin = _direction_score_for_feature(
            X.iloc[idx, j].to_numpy(dtype=np.float32),
            y_s,
            classifier=classifier,
            groups=None,
            returns=ret_s,
        )
        dirs[j] = float(direction)
        margins[j] = float(margin)
    return dirs, margins


def _weighted_direction_stability(directions: np.ndarray, margins: np.ndarray) -> float:
    d = np.asarray(directions, dtype=np.float64)
    w = np.asarray(margins, dtype=np.float64)
    m = np.isfinite(d) & np.isfinite(w) & (w > 0)
    if int(np.sum(m)) == 0 or float(np.sum(w[m])) <= 1e-12:
        return 0.0
    return float(abs(np.sum(d[m] * w[m]) / np.sum(w[m])))


def _univariate_directional_filter(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    classifier: bool,
    groups: Any = None,
    returns: Any = None,
    random_state: int,
    objective_mode: str | None = "train_base",
) -> tuple[list[str], pd.DataFrame]:
    names = list(X.columns)
    y_arr = np.asarray(y)
    ret_arr_all = _as_returns(y_arr, returns)
    uni_frac = float(np.clip(LGBM_UNIVARIATE_ROW_SUBSAMPLE_FRAC, 0.01, 1.0))
    uni_cap = int(len(y_arr))
    if uni_frac < 0.999:
        uni_cap = min(uni_cap, max(1, int(np.ceil(uni_frac * len(y_arr)))))
    if LGBM_UNIVARIATE_MAX_ROWS > 0:
        uni_cap = min(uni_cap, int(LGBM_UNIVARIATE_MAX_ROWS))
    if len(y_arr) > uni_cap:
        uni_idx = _stratified_spread_subsample_indices(
            y_arr,
            uni_cap,
            random_state + 593,
            classifier,
        )
        X_work = X.iloc[uni_idx].reset_index(drop=True)
        y_work = y_arr[uni_idx]
        ret_work = ret_arr_all[uni_idx]
        groups_work = _groups_take(groups, uni_idx)
    else:
        X_work = X.reset_index(drop=True)
        y_work = y_arr
        ret_work = ret_arr_all
        groups_work = groups
    t0 = time.perf_counter()
    tprint(
        "LGBM univariate filter started: "
        f"rows={len(X_work)}/{len(X)}, features={len(names)}, cv_splits={LGBM_CV_SPLITS}, "
        f"row_subsample_frac={uni_frac:.3f}, max_rows={int(LGBM_UNIVARIATE_MAX_ROWS)}, "
        "sample_policy=stratified_spread, "
        f"objective={_normalize_objective_mode(objective_mode)}."
    )
    splitter, y_split = _splitter(y_work, classifier, random_state, n_splits=LGBM_CV_SPLITS)
    records: list[dict[str, Any]] = []
    for fi, name in enumerate(names):
        if fi > 0 and (fi % 25 == 0 or fi == len(names) - 1):
            passed_so_far = sum(1 for r in records if bool(r.get("passed", False)))
            tprint(
                f"LGBM univariate filter progress: {fi}/{len(names)} features "
                f"processed, passed_so_far={passed_so_far}, "
                f"elapsed={time.perf_counter() - t0:.1f}s."
            )
        x = X_work[name].to_numpy(dtype=np.float32)
        if float(np.nanstd(x)) <= 1e-12:
            records.append({"feature": name, "passed": False, "univariate_j": 0.0, "direction_stability": 0.0})
            continue
        j_pos_vals: list[float] = []
        j_neg_vals: list[float] = []
        p20_norm_vals: list[float] = []
        lift20_vals: list[float] = []
        mono_vals: list[float] = []
        dirs: list[int] = []
        margins: list[float] = []
        for _tr, va in splitter.split(np.zeros(len(y_split)), y_split):
            x_va = x[va]
            y_va = y_work[va]
            grp_va = _groups_take(groups_work, va)
            ret_va = ret_work[va]
            j_pos, j_neg, direction, margin = _direction_score_for_feature(x_va, y_va, classifier=classifier, groups=grp_va, returns=ret_va, objective_mode=objective_mode)
            pred = x_va if direction >= 0 else -x_va
            metrics = _metric_pack(y_va, pred, classifier=classifier, groups=grp_va, returns=ret_va)
            j_pos_vals.append(j_pos)
            j_neg_vals.append(j_neg)
            p20_norm_vals.append(float(metrics.get("precision20_norm", 0.0)))
            lift20_vals.append(float(metrics.get("lift20", 1.0)))
            mono_vals.append(float(metrics.get("rank_bucket_monotonicity", 0.0)))
            dirs.append(direction)
            margins.append(margin)
        j_pos_med = float(np.median(j_pos_vals)) if j_pos_vals else 0.0
        j_neg_med = float(np.median(j_neg_vals)) if j_neg_vals else 0.0
        direction = 1 if j_pos_med >= j_neg_med else -1
        direction_stability = _weighted_direction_stability(np.asarray(dirs), np.asarray(margins))
        univariate_j = max(j_pos_med, j_neg_med)
        precision_pass = float(np.median(p20_norm_vals)) > 0.0 if p20_norm_vals else False
        lift_pass = float(np.median(lift20_vals)) > 1.0 if lift20_vals else False
        mono_pass = float(np.median(mono_vals)) >= LGBM_UNIVARIATE_MONOTONICITY_MIN if mono_vals else False
        passed = bool((precision_pass or lift_pass or mono_pass) and direction_stability >= LGBM_DIRECTION_STABILITY_MIN)
        records.append(
            {
                "feature": name,
                "feature_index": int(fi),
                "passed": passed,
                "univariate_j": float(univariate_j),
                "J_pos_median": j_pos_med,
                "J_neg_median": j_neg_med,
                "direction": int(direction),
                "direction_stability": float(direction_stability),
                "direction_margin_median": float(np.median(margins)) if margins else 0.0,
                "precision20_norm_median": float(np.median(p20_norm_vals)) if p20_norm_vals else 0.0,
                "lift20_median": float(np.median(lift20_vals)) if lift20_vals else 1.0,
                "monotonicity_median": float(np.median(mono_vals)) if mono_vals else 0.0,
                "pass_precision": bool(precision_pass),
                "pass_lift": bool(lift_pass),
                "pass_monotonicity": bool(mono_pass),
            }
        )
    stats = pd.DataFrame(records)
    selected = stats.loc[stats["passed"].astype(bool), "feature"].astype(str).tolist()
    if len(selected) < min(LGBM_MIN_FEATURES, len(names)):
        eligible = stats[stats["direction_stability"].astype(float) >= LGBM_DIRECTION_STABILITY_MIN]
        rescue = eligible.sort_values("univariate_j", ascending=False)["feature"].astype(str).head(min(LGBM_MIN_FEATURES, len(names))).tolist()
        selected = sorted(set(selected).union(rescue), key=lambda c: names.index(c))
    tprint(
        f"LGBM univariate filter complete: {len(names)} -> {len(selected)} "
        f"features in {time.perf_counter() - t0:.1f}s."
    )
    return selected, stats


def _relief_target_labels(y: np.ndarray, *, classifier: bool) -> np.ndarray:
    y_arr = np.asarray(y, dtype=np.float32)
    if classifier:
        return (y_arr >= 0.5).astype(np.int8)
    ranks = pd.Series(y_arr).rank(pct=True).to_numpy(dtype=np.float32)
    return np.clip((ranks * 5).astype(np.int8), 0, 4)


def _relief_work_indices(
    y: np.ndarray,
    *,
    classifier: bool,
    random_state: int,
) -> np.ndarray:
    cap = int(len(y))
    uni_frac = float(np.clip(LGBM_UNIVARIATE_ROW_SUBSAMPLE_FRAC, 0.01, 1.0))
    if uni_frac < 0.999:
        cap = min(cap, max(1, int(np.ceil(uni_frac * len(y)))))
    if LGBM_UNIVARIATE_MAX_ROWS > 0:
        cap = min(cap, int(LGBM_UNIVARIATE_MAX_ROWS))
    if len(y) > cap:
        return _stratified_spread_subsample_indices(
            np.asarray(y),
            cap,
            random_state,
            classifier,
        )
    return np.arange(len(y), dtype=np.int32)


def _standardized_relief_matrix(X: pd.DataFrame) -> np.ndarray:
    arr = X.to_numpy(dtype=np.float32, copy=True)
    med = np.nanmedian(arr, axis=0).astype(np.float32, copy=False)
    bad_med = ~np.isfinite(med)
    if np.any(bad_med):
        med[bad_med] = 0.0
    arr = np.where(np.isfinite(arr), arr, med[None, :]).astype(np.float32, copy=False)
    q25 = np.nanpercentile(arr, 25.0, axis=0).astype(np.float32, copy=False)
    q75 = np.nanpercentile(arr, 75.0, axis=0).astype(np.float32, copy=False)
    scale = q75 - q25
    std = np.nanstd(arr, axis=0).astype(np.float32, copy=False)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, std)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    arr = (arr - med[None, :]) / scale[None, :]
    return np.clip(arr, -8.0, 8.0).astype(np.float32, copy=False)


def _approx_relief_scores_once(
    arr: np.ndarray,
    labels: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    n, p = arr.shape
    if n < 4 or p == 0 or len(np.unique(labels)) < 2:
        return np.zeros(p, dtype=np.float32)
    anchor_n = min(n, int(LGBM_RELIEF_ANCHOR_MAX_ROWS))
    anchor_ids = (
        rng.choice(n, size=anchor_n, replace=False)
        if n > anchor_n
        else np.arange(n, dtype=np.int32)
    )
    cand_n = min(n, max(int(LGBM_RELIEF_NEIGHBOR_CANDIDATES), anchor_n))
    candidate_ids = (
        rng.choice(n, size=cand_n, replace=False)
        if n > cand_n
        else np.arange(n, dtype=np.int32)
    )
    candidate_ids = np.unique(
        np.concatenate([candidate_ids.astype(np.int32), anchor_ids.astype(np.int32)])
    ).astype(np.int32)
    cand = arr[candidate_ids]
    cand_labels = labels[candidate_ids]
    cand_norm = np.einsum("ij,ij->i", cand, cand).astype(np.float32, copy=False)
    scores = np.zeros(p, dtype=np.float64)
    used = 0
    k = max(1, int(LGBM_RELIEF_NEIGHBORS))
    for anchor_id in anchor_ids:
        a = arr[int(anchor_id)]
        d = cand_norm + float(np.dot(a, a)) - 2.0 * np.dot(cand, a)
        d = np.maximum(d, 0.0)
        same_self = candidate_ids == int(anchor_id)
        if np.any(same_self):
            d[same_self] = np.inf
        same = cand_labels == labels[int(anchor_id)]
        miss = ~same
        hit_idx = np.flatnonzero(same & np.isfinite(d))
        miss_idx = np.flatnonzero(miss & np.isfinite(d))
        if len(hit_idx) == 0 or len(miss_idx) == 0:
            continue
        hit_take = hit_idx[np.argsort(d[hit_idx])[: min(k, len(hit_idx))]]
        miss_take = miss_idx[np.argsort(d[miss_idx])[: min(k, len(miss_idx))]]
        hit_diff = np.mean(np.abs(cand[hit_take] - a[None, :]), axis=0)
        miss_diff = np.mean(np.abs(cand[miss_take] - a[None, :]), axis=0)
        scores += (miss_diff - hit_diff).astype(np.float64, copy=False)
        used += 1
    if used <= 0:
        return np.zeros(p, dtype=np.float32)
    return (scores / float(used)).astype(np.float32, copy=False)


def _relief_rescue_filter(
    X: pd.DataFrame,
    y: np.ndarray,
    uni_features: list[str],
    *,
    classifier: bool,
    random_state: int,
) -> tuple[list[str], pd.DataFrame]:
    names = list(map(str, X.columns))
    rescue_n = min(
        int(LGBM_RELIEF_RESCUE_MAX),
        max(
            int(LGBM_RELIEF_RESCUE_MIN),
            int(LGBM_RELIEF_RESCUE_FRAC * len(uni_features)),
        ),
    )
    if not LGBM_RELIEF_ENABLED or not names or rescue_n <= 0:
        return [], pd.DataFrame(columns=["feature", "relief_score", "relief_presence"])
    t0 = time.perf_counter()
    y_arr = np.asarray(y)
    min_present_runs = int(np.ceil(float(LGBM_RELIEF_PRESENCE_MIN) * LGBM_RELIEF_REPEATS))
    min_present_runs = max(1, min(int(LGBM_RELIEF_REPEATS), min_present_runs))
    tprint(
        "LGBM ReliefF rescue started: "
        f"features={len(names)}, repeats={LGBM_RELIEF_REPEATS}, "
        f"rescue_n={rescue_n}, min_present_runs={min_present_runs}, "
        f"row_cap={int(LGBM_UNIVARIATE_MAX_ROWS)}."
    )
    per_run_scores: list[np.ndarray] = []
    per_run_top: list[set[str]] = []
    for rep in range(int(LGBM_RELIEF_REPEATS)):
        idx = _relief_work_indices(
            y_arr,
            classifier=classifier,
            random_state=random_state + 1009 + rep * 37,
        )
        X_rep = X.iloc[idx].reset_index(drop=True)
        y_rep = y_arr[idx]
        labels = _relief_target_labels(y_rep, classifier=classifier)
        arr = _standardized_relief_matrix(X_rep)
        rng = np.random.default_rng(random_state + 2017 + rep * 53)
        scores = _approx_relief_scores_once(arr, labels, rng=rng)
        per_run_scores.append(scores)
        top_idx = np.argsort(scores)[::-1][: min(rescue_n, len(names))]
        per_run_top.append(
            {names[int(i)] for i in top_idx if np.isfinite(scores[int(i)])}
        )
        tprint(
            "LGBM ReliefF rescue repeat complete: "
            f"{rep + 1}/{LGBM_RELIEF_REPEATS}, rows={len(idx)}, "
            f"top_score={float(np.nanmax(scores)) if len(scores) else 0.0:.6f}."
        )
    score_mat = (
        np.vstack(per_run_scores) if per_run_scores else np.zeros((0, len(names)))
    )
    mean_score = (
        np.nanmean(score_mat, axis=0).astype(np.float32, copy=False)
        if len(score_mat)
        else np.zeros(len(names), dtype=np.float32)
    )
    presence = np.zeros(len(names), dtype=np.int16)
    for j, name in enumerate(names):
        presence[j] = sum(1 for top in per_run_top if name in top)
    denom = max(1, int(LGBM_RELIEF_REPEATS))
    relief_presence = presence.astype(np.float32) / float(denom)
    stats = pd.DataFrame(
        {
            "feature": names,
            "relief_score": mean_score,
            "relief_presence": relief_presence,
            "relief_present_runs": presence.astype(np.int16),
        }
    )
    eligible = stats[stats["relief_present_runs"].astype(int) >= min_present_runs]
    rescue = (
        eligible.sort_values(["relief_score", "relief_presence"], ascending=False)[
            "feature"
        ]
        .astype(str)
        .head(min(rescue_n, len(eligible)))
        .tolist()
    )
    uni_set = set(uni_features)
    new_rescue = [f for f in rescue if f not in uni_set]
    stats["relief_selected"] = stats["feature"].astype(str).isin(rescue)
    stats["relief_rescued"] = stats["feature"].astype(str).isin(new_rescue)
    tprint(
        "LGBM ReliefF rescue complete: "
        f"eligible={len(eligible)}, selected={len(rescue)}, new_rescues={len(new_rescue)}, "
        f"elapsed={time.perf_counter() - t0:.1f}s."
    )
    return new_rescue, stats


def _redundancy_cluster_filter(
    X: pd.DataFrame,
    features: list[str],
    score_map: dict[str, float],
    *,
    random_state: int,
    corr_threshold: float = LGBM_REDUNDANCY_CORR_THRESHOLD,
) -> list[str]:
    if len(features) <= 2:
        return list(features)
    t0 = time.perf_counter()
    tprint(
        "LGBM redundancy clustering started: "
        f"rows={len(X)}, features={len(features)}, corr_threshold={corr_threshold:.3f}."
    )
    rng = np.random.default_rng(random_state)
    sub_n = min(len(X), 5000)
    sub = rng.choice(len(X), size=sub_n, replace=False) if len(X) > sub_n else np.arange(len(X))
    arr = X.iloc[sub][features].to_numpy(dtype=np.float32)
    ranks = pd.DataFrame(arr).rank(pct=True).to_numpy(dtype=np.float32)
    corr = np.abs(np.corrcoef(ranks, rowvar=False))
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    if corr.shape[0] != len(features):
        return list(features)
    dist = 1.0 - np.clip(corr, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    if not np.any(np.isfinite(condensed)):
        return list(features)
    z = linkage(condensed, method="average")
    labels = fcluster(z, t=1.0 - float(corr_threshold), criterion="distance")
    keep: list[str] = []
    for lab in sorted(set(labels)):
        members = [features[i] for i in np.where(labels == lab)[0]]
        members_sorted = sorted(members, key=lambda f: float(score_map.get(f, 0.0)), reverse=True)
        keep_n = min(3, max(1, int(np.ceil(0.25 * len(members_sorted)))))
        keep.extend(members_sorted[:keep_n])
    keep_ordered = [f for f in features if f in set(keep)]
    tprint(
        f"LGBM redundancy clustering complete: {len(features)} -> {len(keep_ordered)} "
        f"features in {time.perf_counter() - t0:.1f}s."
    )
    return keep_ordered


def _base_lgbm_params(seed: int, *, classifier: bool, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    params: dict[str, Any] = {
        "objective": "binary" if classifier else "regression",
        "n_estimators": 800,
        "learning_rate": 0.02,
        "max_depth": 4,
        "num_leaves": 16,
        "min_child_samples": 300,
        "subsample": 0.80,
        "subsample_freq": 1,
        "colsample_bytree": 0.60,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "min_split_gain": 0.01,
        "max_bin": 63,
        "random_state": int(seed),
        "n_jobs": LGBM_N_JOBS,
        "verbosity": -1,
    }
    if overrides:
        params.update(overrides)
    depth = int(params.get("max_depth", 4))
    if "num_leaves" not in params or params.get("num_leaves") is None:
        params["num_leaves"] = int(2 ** depth)
    params["num_leaves"] = int(min(int(params["num_leaves"]), 2 ** max(depth, 1)))
    return params


def _make_lgbm_model(params: dict[str, Any], classifier: bool) -> Any:
    import lightgbm as lgb

    if classifier:
        return lgb.LGBMClassifier(**params)
    return lgb.LGBMRegressor(**params)


def _fit_lgbm_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None,
    *,
    classifier: bool,
    params: dict[str, Any],
    X_valid: pd.DataFrame | None = None,
    y_valid: np.ndarray | None = None,
    early_stopping_rounds: int | None = None,
) -> Any:
    import lightgbm as lgb

    model = _make_lgbm_model(dict(params), classifier)
    callbacks = []
    eval_set = None
    if X_valid is not None and y_valid is not None and early_stopping_rounds and len(y_valid) > 10:
        callbacks.append(lgb.early_stopping(int(early_stopping_rounds), verbose=False))
        eval_set = [(X_valid, y_valid)]
    fit_kwargs: dict[str, Any] = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = np.asarray(sample_weight, dtype=np.float32)
    if eval_set is not None:
        fit_kwargs["eval_set"] = eval_set
        fit_kwargs["callbacks"] = callbacks
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def _predict_lgbm_raw(model: Any, X: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "classifier" and hasattr(model, "predict_proba"):
        p = np.asarray(model.predict_proba(X), dtype=np.float64)
        if p.ndim == 2 and p.shape[1] > 1:
            return np.clip(p[:, 1], 1e-6, 1.0 - 1e-6).astype(np.float32)
        return np.clip(p.reshape(-1), 1e-6, 1.0 - 1e-6).astype(np.float32)
    return np.asarray(model.predict(X), dtype=np.float32).reshape(-1)


def _predict_lgbm_raw_batched(
    model: Any,
    X: pd.DataFrame,
    mode: str,
    *,
    batch_size: int = 200000,
) -> np.ndarray:
    n = len(X)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    bs = max(1, int(batch_size))
    if n <= bs:
        return _predict_lgbm_raw(model, X, mode).astype(np.float32)
    out = np.empty(n, dtype=np.float32)
    for start in range(0, n, bs):
        stop = min(n, start + bs)
        out[start:stop] = _predict_lgbm_raw(model, X.iloc[start:stop], mode)
    return out


def _predict_lgbm_raw_score(model: Any, X: pd.DataFrame, mode: str, num_iteration: int | None = None) -> np.ndarray:
    predict_kwargs: dict[str, Any] = {}
    if num_iteration is not None and int(num_iteration) > 0:
        predict_kwargs["num_iteration"] = int(num_iteration)
    try:
        raw = model.predict(X, raw_score=True, **predict_kwargs)
        return np.asarray(raw, dtype=np.float32).reshape(len(X), -1)[:, -1]
    except Exception:
        pred = _predict_lgbm_raw(model, X, mode)
        if mode == "classifier":
            pred = np.clip(pred, 1e-6, 1.0 - 1e-6)
            return np.log(pred / (1.0 - pred)).astype(np.float32)
        return pred.astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(x, dtype=np.float32), -50.0, 50.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def _safe_rank_pct(values: np.ndarray) -> np.ndarray:
    vals = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if len(vals) == 0:
        return np.zeros(0, dtype=np.float32)
    return pd.Series(vals).rank(method="average", pct=True).to_numpy(dtype=np.float32)


def _score_margin(score: np.ndarray, frac: float) -> np.ndarray:
    vals = np.nan_to_num(np.asarray(score, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if len(vals) == 0:
        return vals
    threshold = float(np.nanquantile(vals, 1.0 - float(frac)))
    return (vals - threshold).astype(np.float32)


def _binary_entropy(prob: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(prob, dtype=np.float32), 1e-6, 1.0 - 1e-6)
    return (-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))).astype(np.float32)


def _model_num_iterations(model: Any) -> int:
    for attr in ("best_iteration_", "n_estimators_"):
        value = getattr(model, attr, None)
        if value is not None:
            try:
                ivalue = int(value)
                if ivalue > 0:
                    return ivalue
            except Exception:
                pass
    booster = getattr(model, "booster_", None)
    if booster is not None:
        for name in ("current_iteration", "num_trees"):
            fn = getattr(booster, name, None)
            if callable(fn):
                try:
                    ivalue = int(fn())
                    if ivalue > 0:
                        return ivalue
                except Exception:
                    pass
    return 0


def _walk_lgbm_tree(node: dict[str, Any], depth: int, out: dict[int, tuple[float, float, float, float]]) -> None:
    if "leaf_index" in node:
        leaf_index = int(node.get("leaf_index", 0))
        out[leaf_index] = (
            float(node.get("leaf_count", 0.0) or 0.0),
            float(node.get("leaf_weight", 0.0) or 0.0),
            float(depth),
            float(node.get("leaf_value", 0.0) or 0.0),
        )
        return
    left = node.get("left_child")
    right = node.get("right_child")
    if isinstance(left, dict):
        _walk_lgbm_tree(left, depth + 1, out)
    if isinstance(right, dict):
        _walk_lgbm_tree(right, depth + 1, out)


def _leaf_metadata(model: Any) -> list[dict[int, tuple[float, float, float, float]]]:
    booster = getattr(model, "booster_", None)
    if booster is None:
        return []
    try:
        dumped = booster.dump_model()
    except Exception:
        return []
    meta: list[dict[int, tuple[float, float, float, float]]] = []
    for tree in dumped.get("tree_info", []):
        leaves: dict[int, tuple[float, float, float, float]] = {}
        root = tree.get("tree_structure", {}) if isinstance(tree, dict) else {}
        if isinstance(root, dict):
            _walk_lgbm_tree(root, 0, leaves)
        meta.append(leaves)
    return meta


def _append_leaf_diagnostics(frames: dict[str, np.ndarray], models: list[Any], X: pd.DataFrame) -> None:
    n = len(X)
    counts: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    abs_values: list[np.ndarray] = []
    for model in models:
        meta = _leaf_metadata(model)
        if not meta:
            continue
        try:
            leaves = np.asarray(model.predict(X, pred_leaf=True), dtype=np.int64)
        except Exception:
            continue
        if leaves.ndim == 1:
            leaves = leaves.reshape(n, 1)
        tree_n = min(leaves.shape[1], len(meta))
        for tree_i in range(tree_n):
            tree_meta = meta[tree_i]
            c = np.zeros(n, dtype=np.float32)
            w = np.zeros(n, dtype=np.float32)
            d = np.zeros(n, dtype=np.float32)
            v = np.zeros(n, dtype=np.float32)
            for row_i, leaf_id in enumerate(leaves[:, tree_i]):
                vals = tree_meta.get(int(leaf_id))
                if vals is None:
                    continue
                c[row_i], w[row_i], d[row_i], leaf_value = vals
                v[row_i] = abs(float(leaf_value))
            counts.append(c)
            weights.append(w)
            depths.append(d)
            abs_values.append(v)
    if not counts:
        for name in (
            "leaf_count_p10", "leaf_count_min", "rare_leaf_fraction", "leaf_weight_p10",
            "leaf_depth_mean", "leaf_depth_max", "leaf_value_abs_mean", "leaf_value_abs_max",
            "large_leaf_value_fraction",
        ):
            frames[name] = np.zeros(n, dtype=np.float32)
        return
    count_mat = np.vstack(counts).T.astype(np.float32)
    weight_mat = np.vstack(weights).T.astype(np.float32)
    depth_mat = np.vstack(depths).T.astype(np.float32)
    value_mat = np.vstack(abs_values).T.astype(np.float32)
    global_count_p10 = float(np.nanpercentile(count_mat, 10.0)) if count_mat.size else 0.0
    global_value_p90 = float(np.nanpercentile(value_mat, 90.0)) if value_mat.size else 0.0
    frames["leaf_count_p10"] = np.nanpercentile(count_mat, 10.0, axis=1).astype(np.float32)
    frames["leaf_count_min"] = np.nanmin(count_mat, axis=1).astype(np.float32)
    frames["rare_leaf_fraction"] = np.mean(count_mat <= max(global_count_p10, 1.0), axis=1).astype(np.float32)
    frames["leaf_weight_p10"] = np.nanpercentile(weight_mat, 10.0, axis=1).astype(np.float32)
    frames["leaf_depth_mean"] = np.nanmean(depth_mat, axis=1).astype(np.float32)
    frames["leaf_depth_max"] = np.nanmax(depth_mat, axis=1).astype(np.float32)
    frames["leaf_value_abs_mean"] = np.nanmean(value_mat, axis=1).astype(np.float32)
    frames["leaf_value_abs_max"] = np.nanmax(value_mat, axis=1).astype(np.float32)
    frames["large_leaf_value_fraction"] = np.mean(value_mat >= max(global_value_p90, 1e-12), axis=1).astype(np.float32)


def _predict_contrib_matrix(model: Any, X: pd.DataFrame, n_features: int) -> np.ndarray | None:
    try:
        contrib = np.asarray(model.predict(X, pred_contrib=True), dtype=np.float32)
    except Exception:
        return None
    if contrib.ndim == 1:
        contrib = contrib.reshape(len(X), -1)
    if contrib.ndim == 3:
        if contrib.shape[0] == len(X):
            class_i = 1 if contrib.shape[1] > 1 else 0
            contrib = contrib[:, class_i, :]
        else:
            class_i = 1 if contrib.shape[0] > 1 else 0
            contrib = contrib[class_i, :, :]
    block = n_features + 1
    if contrib.shape[1] == block:
        return contrib[:, :n_features]
    if block > 0 and contrib.shape[1] % block == 0:
        n_classes = contrib.shape[1] // block
        reshaped = contrib.reshape(len(X), n_classes, block)
        class_i = 1 if n_classes > 1 else 0
        return reshaped[:, class_i, :n_features]
    if contrib.shape[1] > n_features:
        return contrib[:, :n_features]
    return contrib


def _append_contrib_diagnostics(frames: dict[str, np.ndarray], models: list[Any], X: pd.DataFrame) -> None:
    n = len(X)
    mats = []
    for model in models:
        mat = _predict_contrib_matrix(model, X, X.shape[1])
        if mat is not None and mat.size:
            mats.append(mat)
    if not mats:
        for name in ("contrib_top1_abs_share", "contrib_top3_abs_share", "contrib_entropy", "contrib_balance", "num_material_contrib_features"):
            frames[name] = np.zeros(n, dtype=np.float32)
        return
    contrib = np.mean(np.stack(mats, axis=0), axis=0).astype(np.float32)
    abs_c = np.abs(contrib)
    total_abs = np.sum(abs_c, axis=1) + 1e-12
    sorted_abs = np.sort(abs_c, axis=1)[:, ::-1]
    top3 = np.sum(sorted_abs[:, : min(3, sorted_abs.shape[1])], axis=1)
    share = abs_c / total_abs[:, None]
    entropy = -np.sum(np.where(share > 0.0, share * np.log(share + 1e-12), 0.0), axis=1)
    entropy = entropy / max(np.log(max(abs_c.shape[1], 2)), 1e-12)
    frames["contrib_top1_abs_share"] = (sorted_abs[:, 0] / total_abs).astype(np.float32)
    frames["contrib_top3_abs_share"] = (top3 / total_abs).astype(np.float32)
    frames["contrib_entropy"] = entropy.astype(np.float32)
    frames["contrib_balance"] = np.clip(np.sum(contrib, axis=1) / total_abs, -1.0, 1.0).astype(np.float32)
    frames["num_material_contrib_features"] = np.sum(share >= 0.01, axis=1).astype(np.float32)


def _rank_bin_indices(rank_pct: np.ndarray, n_bins: int | None = None) -> np.ndarray:
    bin_count = max(2, int(n_bins or LGBM_META_RANK_BINS))
    bins = np.floor(np.clip(np.asarray(rank_pct, dtype=np.float32), 0.0, 1.0 - 1e-7) * bin_count).astype(np.int32)
    return np.clip(bins, 0, bin_count - 1)


def _fit_rank_bin_stats_oof(y: np.ndarray, rank_pct: np.ndarray, *, classifier: bool, returns: Any = None) -> pd.DataFrame:
    y_arr = np.asarray(y, dtype=np.float32)
    ret = _as_returns(y_arr, returns)
    bins = _rank_bin_indices(rank_pct)
    win = (y_arr >= 0.5).astype(np.float32) if classifier else (y_arr > 0.0).astype(np.float32)
    baseline = float(np.mean(win)) if len(win) else 0.0
    rows: list[dict[str, float]] = []
    for b in range(LGBM_META_RANK_BINS):
        mask = bins == b
        n_b = int(np.sum(mask))
        if n_b == 0:
            win_rate = baseline
            net_ret = 0.0
            se = 0.0
        else:
            win_rate = float(np.mean(win[mask]))
            net_ret = float(np.mean(ret[mask]))
            se = float(np.sqrt(max(win_rate * (1.0 - win_rate), 0.0) / max(n_b, 1)))
        rows.append(
            {
                "rank_bin": float(b),
                "rank_bin_win_rate_oof": win_rate,
                "rank_bin_lift_oof": win_rate / max(baseline, 1e-6),
                "rank_bin_net_ret_oof": net_ret,
                "rank_bin_se_oof": se,
            }
        )
    return pd.DataFrame(rows)


def _append_rank_bin_oof_features(frames: dict[str, np.ndarray], rank_pct: np.ndarray, rank_bin_stats: pd.DataFrame | None) -> None:
    n = len(rank_pct)
    n_bins = len(rank_bin_stats) if rank_bin_stats is not None and len(rank_bin_stats) else LGBM_META_RANK_BINS
    bins = _rank_bin_indices(rank_pct, n_bins=n_bins)
    if rank_bin_stats is None or len(rank_bin_stats) == 0:
        for name in ("rank_bin_win_rate_oof", "rank_bin_lift_oof", "rank_bin_net_ret_oof", "rank_bin_se_oof"):
            frames[name] = np.zeros(n, dtype=np.float32)
        return
    stats = rank_bin_stats.set_index("rank_bin") if "rank_bin" in rank_bin_stats.columns else rank_bin_stats.copy()
    for name in ("rank_bin_win_rate_oof", "rank_bin_lift_oof", "rank_bin_net_ret_oof", "rank_bin_se_oof"):
        mapping = stats[name].to_dict() if name in stats.columns else {}
        frames[name] = np.asarray([float(mapping.get(float(b), mapping.get(int(b), 0.0))) for b in bins], dtype=np.float32)


def _lgbm_meta_features_from_models(
    models: list[Any],
    X: pd.DataFrame,
    *,
    mode: str,
    rank_bin_stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    n = len(X)
    frames: dict[str, np.ndarray] = {}
    if not models:
        raw_score = np.zeros(n, dtype=np.float32)
        prob = np.full(n, 0.5 if mode == "classifier" else 0.0, dtype=np.float32)
        raw_mat = raw_score.reshape(1, -1)
        prob_mat = prob.reshape(1, -1)
    else:
        raw_mat = np.vstack([_predict_lgbm_raw_score(model, X, mode) for model in models]).astype(np.float32)
        raw_score = np.mean(raw_mat, axis=0).astype(np.float32)
        prob_mat = _sigmoid(raw_mat) if mode == "classifier" else raw_mat
        prob = np.mean(prob_mat, axis=0).astype(np.float32)
    rank_pct = _safe_rank_pct(prob)
    frames["lgbm_prob"] = prob
    frames["lgbm_raw_score"] = raw_score
    frames["abs_raw_score"] = np.abs(raw_score).astype(np.float32)
    frames["entropy"] = _binary_entropy(prob) if mode == "classifier" else np.zeros(n, dtype=np.float32)
    frames["variance_proxy"] = np.var(prob_mat, axis=0).astype(np.float32) if prob_mat.shape[0] > 1 else np.zeros(n, dtype=np.float32)
    frames["rank_pct"] = rank_pct
    for frac in (0.10, 0.20, 0.30):
        pct = int(round(frac * 100))
        frames[f"score_margin_top{pct}"] = _score_margin(prob, frac)
    frames["rank_margin_top10"] = (rank_pct - 0.90).astype(np.float32)
    frames["rank_margin_top20"] = (rank_pct - 0.80).astype(np.float32)
    _append_leaf_diagnostics(frames, models, X)
    _append_contrib_diagnostics(frames, models, X)
    path_scores: list[np.ndarray] = []
    for frac in (0.50, 0.75, 1.00):
        per_model = []
        for model in models:
            n_iter = _model_num_iterations(model)
            num_iter = int(np.ceil(frac * n_iter)) if n_iter > 0 and frac < 1.0 else None
            raw = _predict_lgbm_raw_score(model, X, mode, num_iteration=num_iter)
            per_model.append(_sigmoid(raw) if mode == "classifier" else raw)
        path_scores.append(np.mean(np.vstack(per_model), axis=0).astype(np.float32) if per_model else prob.copy())
    score50, score75, score100 = path_scores
    frames["score_100_minus_50"] = (score100 - score50).astype(np.float32)
    frames["score_100_minus_75"] = (score100 - score75).astype(np.float32)
    frames["score_path_std"] = np.std(np.vstack(path_scores), axis=0).astype(np.float32)
    rank_paths = [_safe_rank_pct(score) for score in path_scores]
    frames["rank_100_minus_50"] = (rank_paths[2] - rank_paths[0]).astype(np.float32)
    frames["rank_path_std"] = np.std(np.vstack(rank_paths), axis=0).astype(np.float32)
    _append_rank_bin_oof_features(frames, rank_pct, rank_bin_stats)
    out = pd.DataFrame({name: np.nan_to_num(frames.get(name, np.zeros(n, dtype=np.float32)), nan=0.0, posinf=0.0, neginf=0.0) for name in LGBM_META_FEATURE_NAMES})
    return out.astype(np.float32)


def _save_lgbm_meta_features(model: LGBMStabilityModel, output_path: str | os.PathLike[str] | None) -> None:
    if not output_path or model.meta_oof_features is None:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    saved_path = path
    try:
        model.meta_oof_features.to_parquet(path)
    except Exception as exc:
        saved_path = path.with_suffix(path.suffix + ".pkl")
        model.meta_oof_features.to_pickle(saved_path)
        tprint(f"WARNING: failed to write LGBM meta features as parquet ({exc}); wrote pickle to {saved_path}.")
    schema_path = Path(str(path) + ".schema.json")
    schema = {
        "meta_feature_names": list(model.meta_feature_names),
        "rank_bin_stats_oof": model.rank_bin_stats_oof.to_dict(orient="records"),
        "selected_features": list(model.selected_features),
        "selected_features_count": int(len(model.selected_features)),
        "feature_stats_train": model.feature_stats_train,
        "meta_features_path": str(saved_path),
    }
    schema_path.write_text(json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8")
    model.metrics["lgbm_meta_feature_output_path"] = str(saved_path)
    model.metrics["lgbm_meta_feature_schema_path"] = str(schema_path)


def _json_sanitize(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [_json_sanitize(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def _summary_stats(values: Any) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(arr.size)
    finite = np.isfinite(arr)
    finite_arr = arr[finite]
    out: dict[str, Any] = {
        "n": n,
        "finite_n": int(finite_arr.size),
        "nonfinite_n": int(n - finite_arr.size),
        "nonfinite_rate": float((n - finite_arr.size) / max(n, 1)),
    }
    if finite_arr.size == 0:
        out.update(
            {
                "mean": None,
                "std": None,
                "min": None,
                "q01": None,
                "q05": None,
                "q10": None,
                "q25": None,
                "q50": None,
                "q75": None,
                "q90": None,
                "q95": None,
                "q99": None,
                "max": None,
            }
        )
        return out
    qs = np.quantile(
        finite_arr,
        [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
    )
    out.update(
        {
            "mean": float(np.mean(finite_arr)),
            "std": float(np.std(finite_arr)),
            "min": float(np.min(finite_arr)),
            "q01": float(qs[0]),
            "q05": float(qs[1]),
            "q10": float(qs[2]),
            "q25": float(qs[3]),
            "q50": float(qs[4]),
            "q75": float(qs[5]),
            "q90": float(qs[6]),
            "q95": float(qs[7]),
            "q99": float(qs[8]),
            "max": float(np.max(finite_arr)),
        }
    )
    return out


def _feature_distribution_summary(
    X: pd.DataFrame,
    feature_names: list[str],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for feature in feature_names:
        if feature not in X.columns:
            continue
        out[str(feature)] = _summary_stats(X[feature].to_numpy(dtype=np.float64, copy=False))
    return out


def _compact_covariance_reference(X: pd.DataFrame, feature_names: list[str]) -> dict[str, Any]:
    cols = [str(c) for c in feature_names if str(c) in X.columns]
    if not cols:
        return {"features": [], "feature_count": 0}
    frame = X.loc[:, cols].astype(np.float64, copy=False)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    finite_rate = np.isfinite(frame.to_numpy(dtype=np.float64, copy=False)).mean(axis=0)
    frame = frame.fillna(frame.median(numeric_only=True)).fillna(0.0)
    if len(frame) < 2 or len(cols) < 2:
        return {
            "features": cols,
            "feature_count": int(len(cols)),
            "finite_rate": {c: float(finite_rate[i]) for i, c in enumerate(cols)},
        }
    corr = frame.corr().to_numpy(dtype=np.float64)
    upper = corr[np.triu_indices_from(corr, k=1)]
    return {
        "features": cols,
        "feature_count": int(len(cols)),
        "finite_rate": {c: float(finite_rate[i]) for i, c in enumerate(cols)},
        "abs_corr_upper_summary": _summary_stats(np.abs(upper)),
        "corr_upper_summary": _summary_stats(upper),
    }


def _reference_sample_frame(
    *,
    timestamps: Any,
    assets: Any,
    oof_probs: np.ndarray | None,
    meta_oof_features: pd.DataFrame | None,
    returns: np.ndarray | None,
    y_metric: np.ndarray,
    max_rows: int,
) -> pd.DataFrame:
    n = len(y_metric)
    frame = pd.DataFrame(index=np.arange(n))
    if timestamps is not None and len(timestamps) == n:
        frame["timestamp"] = np.asarray(timestamps)
    if assets is not None and len(assets) == n:
        frame["asset"] = np.asarray(assets).astype(str)
    if oof_probs is not None and len(oof_probs) == n:
        score = np.asarray(oof_probs, dtype=np.float32)
        frame["score"] = score
        frame["rank_pct"] = _safe_rank_pct(score).astype(np.float32)
        frame["raw_logit"] = np.log(
            np.clip(score.astype(np.float64), 1e-7, 1.0 - 1e-7)
            / np.clip(1.0 - score.astype(np.float64), 1e-7, 1.0)
        ).astype(np.float32)
    frame["target"] = np.asarray(y_metric, dtype=np.float32)
    if returns is not None and len(returns) == n:
        frame["return"] = np.asarray(returns, dtype=np.float32)
    diag_cols = (
        "prob_uncertainty",
        "rare_leaf_fraction",
        "leaf_count_p10",
        "leaf_count_min",
        "leaf_weight_p10",
        "contrib_top1_abs_share",
        "contrib_top3_abs_share",
        "contrib_entropy",
        "regime_centroid_similarity_train",
        "feature_drift_psi_core",
        "feature_drift_cov_shift",
    )
    if meta_oof_features is not None:
        for col in diag_cols:
            if col in meta_oof_features.columns:
                frame[col] = np.asarray(meta_oof_features[col], dtype=np.float32)
    if max_rows > 0 and len(frame) > max_rows:
        idx = np.linspace(0, len(frame) - 1, num=max_rows, dtype=np.int64)
        frame = frame.iloc[idx].reset_index(drop=True)
    return frame.reset_index(drop=True)


def _save_lgbm_reference_artifacts(
    model: LGBMStabilityModel,
    output_dir: str | os.PathLike[str] | None,
    *,
    X_reference: pd.DataFrame,
    split_importance_sum: np.ndarray,
    gain_importance_sum: np.ndarray,
    y_metric: np.ndarray,
    returns: np.ndarray | None,
    timestamps: Any,
    assets: Any,
    objective_mode: str,
    mode: str,
) -> None:
    if not output_dir:
        return
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    selected_features = list(map(str, model.selected_features))
    core_50 = _top_cumulative_importance_feature_names(
        selected_features,
        gain_importance_sum,
        split_importance_sum,
        cumulative_fraction=0.50,
    )
    core_80 = _top_cumulative_importance_feature_names(
        selected_features,
        gain_importance_sum,
        split_importance_sum,
        cumulative_fraction=0.80,
    )
    oof = np.asarray(model.oof_probs, dtype=np.float32) if model.oof_probs is not None else None
    raw_logit = None
    rank_pct = None
    if oof is not None and len(oof):
        clipped = np.clip(oof.astype(np.float64), 1e-7, 1.0 - 1e-7)
        raw_logit = np.log(clipped / (1.0 - clipped))
        rank_pct = _safe_rank_pct(oof)
    meta_oof = model.meta_oof_features
    uncertainty_score = None
    if meta_oof is not None and oof is not None and len(meta_oof) == len(oof):
        prob_uncertainty = 1.0 - (2.0 * np.abs(oof.astype(np.float64) - 0.5))
        leaf_uncertainty = np.zeros(len(meta_oof), dtype=np.float64)
        if "rare_leaf_fraction" in meta_oof.columns:
            leaf_uncertainty += np.asarray(meta_oof["rare_leaf_fraction"], dtype=np.float64)
        if "leaf_count_p10" in meta_oof.columns:
            support = np.asarray(meta_oof["leaf_count_p10"], dtype=np.float64)
            finite = np.isfinite(support)
            scale = float(np.nanpercentile(support[finite], 75.0)) if np.any(finite) else 1.0
            if not np.isfinite(scale) or scale <= 1e-6:
                scale = 1.0
            leaf_uncertainty += np.clip(1.0 - support / scale, 0.0, 1.0)
        contrib_uncertainty = np.zeros(len(meta_oof), dtype=np.float64)
        if "contrib_entropy" in meta_oof.columns:
            contrib_uncertainty += np.asarray(meta_oof["contrib_entropy"], dtype=np.float64)
        if "contrib_top1_abs_share" in meta_oof.columns:
            contrib_uncertainty += 1.0 - np.asarray(meta_oof["contrib_top1_abs_share"], dtype=np.float64)
        regime_distance = np.zeros(len(meta_oof), dtype=np.float64)
        if "regime_centroid_similarity_train" in meta_oof.columns:
            regime_distance = 1.0 - np.asarray(meta_oof["regime_centroid_similarity_train"], dtype=np.float64)
        uncertainty_score = (
            0.35 * prob_uncertainty
            + 0.35 * leaf_uncertainty
            + 0.20 * contrib_uncertainty
            + 0.10 * regime_distance
        )
    summary = {
        "schema_version": 1,
        "objective_mode": str(objective_mode),
        "mode": str(mode),
        "selected_features_count": int(len(selected_features)),
        "selected_features": selected_features,
        "core_50_features": list(core_50),
        "core_80_features": list(core_80),
        "split_importance": {f: float(v) for f, v in zip(selected_features, split_importance_sum)},
        "gain_importance": {f: float(v) for f, v in zip(selected_features, gain_importance_sum)},
        "feature_distribution_core_50": _feature_distribution_summary(X_reference, list(core_50)),
        "feature_distribution_core_80": _feature_distribution_summary(X_reference, list(core_80)),
        "feature_cov_shift_reference": _compact_covariance_reference(X_reference, list(core_50)),
        "prediction_reference": {
            "score": _summary_stats(oof) if oof is not None else {},
            "raw_logit": _summary_stats(raw_logit) if raw_logit is not None else {},
            "rank_pct": _summary_stats(rank_pct) if rank_pct is not None else {},
            "topq_thresholds": {
                "top10": float(np.nanquantile(oof, 0.90)) if oof is not None and len(oof) else None,
                "top30": float(np.nanquantile(oof, 0.70)) if oof is not None and len(oof) else None,
                "top40": float(np.nanquantile(oof, 0.60)) if oof is not None and len(oof) else None,
            },
        },
        "leaf_reference": {
            col: _summary_stats(meta_oof[col].to_numpy(dtype=np.float64, copy=False))
            for col in (
                "rare_leaf_fraction",
                "leaf_count_p10",
                "leaf_count_min",
                "leaf_weight_p10",
            )
            if meta_oof is not None and col in meta_oof.columns
        },
        "contribution_reference": {
            col: _summary_stats(meta_oof[col].to_numpy(dtype=np.float64, copy=False))
            for col in (
                "contrib_top1_abs_share",
                "contrib_top3_abs_share",
                "contrib_entropy",
                "contrib_balance",
                "num_material_contrib_features",
            )
            if meta_oof is not None and col in meta_oof.columns
        },
        "regime_reference": {
            col: _summary_stats(meta_oof[col].to_numpy(dtype=np.float64, copy=False))
            for col in ("regime_centroid_similarity_train",)
            if meta_oof is not None and col in meta_oof.columns
        },
        "uncertainty_reference": {
            "prob_uncertainty": _summary_stats(
                1.0 - 2.0 * np.abs(oof.astype(np.float64) - 0.5)
            )
            if oof is not None
            else {},
            "uncertainty_score": _summary_stats(uncertainty_score)
            if uncertainty_score is not None
            else {},
        },
    }
    summary_path = path / "lgbm_reference_summary.json"
    summary_path.write_text(
        json.dumps(_json_sanitize(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    sample_max = int(os.environ.get("EPM_LGBM_REFERENCE_SAMPLE_MAX_ROWS", "50000"))
    sample = _reference_sample_frame(
        timestamps=timestamps,
        assets=assets,
        oof_probs=oof,
        meta_oof_features=meta_oof,
        returns=returns,
        y_metric=y_metric,
        max_rows=sample_max,
    )
    sample_path = path / "lgbm_reference_sample.parquet"
    saved_sample_path = sample_path
    try:
        sample.to_parquet(sample_path)
    except Exception as exc:
        saved_sample_path = sample_path.with_suffix(".pkl")
        sample.to_pickle(saved_sample_path)
        tprint(f"WARNING: failed to write LGBM reference sample as parquet ({exc}); wrote pickle to {saved_sample_path}.")
    manifest = {
        "schema_version": 1,
        "objective_mode": str(objective_mode),
        "summary_path": str(summary_path),
        "sample_path": str(saved_sample_path),
        "selected_features_count": int(len(selected_features)),
        "core_50_feature_count": int(len(core_50)),
        "core_80_feature_count": int(len(core_80)),
        "oof_rows": int(len(oof)) if oof is not None else 0,
        "reference_rows": int(len(X_reference)),
    }
    manifest_path = path / "manifest.json"
    manifest_path.write_text(
        json.dumps(_json_sanitize(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    model.metrics["lgbm_reference_artifact_dir"] = str(path)
    model.metrics["lgbm_reference_summary_path"] = str(summary_path)
    model.metrics["lgbm_reference_sample_path"] = str(saved_sample_path)
    tprint(
        "LGBM reference artifacts saved: "
        f"{path} (core50={len(core_50)}, core80={len(core_80)}, sample_rows={len(sample)})."
    )


def _false_positive_avoidance_weight(
    y_true: np.ndarray,
    pred: np.ndarray,
    *,
    classifier: bool,
    top_frac: float = 0.20,
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
    rank_pct = pd.Series(pp).rank(method="average", pct=True).to_numpy(dtype=np.float32)
    top_mask = rank_pct >= 1.0 - float(np.clip(top_frac, 0.001, 0.95))
    support_mask = rank_pct >= 1.0 - float(np.clip(1.5 * top_frac, top_frac, 0.95))
    w = np.ones(len(pp), dtype=np.float32)
    w[(yb < 0.5) & top_mask] = fp_upweight
    w[(yb >= 0.5) & support_mask] = np.maximum(w[(yb >= 0.5) & support_mask], top_positive_upweight)
    return np.clip(w, 0.25, max_weight).astype(np.float32)


def _normalize_weights(weights: np.ndarray, *, min_weight: float = 0.25, max_weight: float = 4.0) -> tuple[np.ndarray, float]:
    w = np.nan_to_num(np.asarray(weights, dtype=np.float32), nan=1.0, posinf=max_weight, neginf=min_weight)
    w = np.clip(w, min_weight, max_weight)
    w = w / max(float(np.mean(w)), 1e-6)
    ess = float((w.sum() ** 2) / max(np.sum(w**2), 1e-6))
    return w.astype(np.float32), ess


def _drop_fraction(n_features: int) -> float:
    n = int(n_features)
    if n > 150:
        return 0.25
    if n > 100:
        return 0.20
    if n > 70:
        return 0.15
    return 0.05


def _importance_rank_scores(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    out = np.zeros(len(vals), dtype=np.float32)
    positive = np.isfinite(vals) & (vals > 0.0)
    if int(np.sum(positive)) <= 1:
        out[positive] = 1.0
        return out
    ranks = pd.Series(vals[positive]).rank(pct=True).to_numpy(dtype=np.float32)
    out[positive] = ranks
    return out


def _feature_importances(model: Any, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    gain = np.zeros(n_features, dtype=np.float32)
    split = np.zeros(n_features, dtype=np.float32)
    try:
        booster = getattr(model, "booster_", None)
        if booster is not None:
            gain_v = np.asarray(booster.feature_importance(importance_type="gain"), dtype=np.float32)
            split_v = np.asarray(booster.feature_importance(importance_type="split"), dtype=np.float32)
        else:
            gain_v = np.asarray(model.feature_importances_, dtype=np.float32)
            split_v = gain_v.copy()
        gain[: min(n_features, len(gain_v))] = gain_v[:n_features]
        split[: min(n_features, len(split_v))] = split_v[:n_features]
    except Exception:
        pass
    return gain, split


def _top_cumulative_importance_feature_names(
    feature_names: list[str],
    gain_importance: np.ndarray,
    split_importance: np.ndarray | None = None,
    *,
    cumulative_fraction: float = 0.50,
) -> list[str]:
    names = [str(name) for name in feature_names]
    if not names:
        return []
    gain = np.asarray(gain_importance, dtype=np.float64)
    split = (
        np.zeros(len(names), dtype=np.float64)
        if split_importance is None
        else np.asarray(split_importance, dtype=np.float64)
    )
    if gain.shape[0] < len(names):
        gain = np.pad(gain, (0, len(names) - gain.shape[0]))
    if split.shape[0] < len(names):
        split = np.pad(split, (0, len(names) - split.shape[0]))
    score = np.nan_to_num(gain[: len(names)], nan=0.0, posinf=0.0, neginf=0.0)
    if float(score.sum()) <= 0.0:
        score = np.nan_to_num(split[: len(names)], nan=0.0, posinf=0.0, neginf=0.0)
    if float(score.sum()) <= 0.0:
        return names[: min(len(names), max(1, int(np.ceil(0.10 * len(names)))))]
    order = np.argsort(score)[::-1]
    cum = np.cumsum(score[order])
    total = float(cum[-1])
    threshold = float(np.clip(cumulative_fraction, 0.01, 1.0)) * total
    keep_count = int(np.searchsorted(cum, threshold, side="left") + 1)
    keep_count = min(len(names), max(1, keep_count))
    selected_idx = set(int(i) for i in order[:keep_count])
    return [name for i, name in enumerate(names) if i in selected_idx]


def _safe_corr_matrix(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if arr.shape[1] == 1:
        return np.ones((1, 1), dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.shape[0] < 2:
        return np.eye(arr.shape[1], dtype=np.float32)
    corr = np.corrcoef(arr, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    if corr.ndim != 2:
        return np.eye(arr.shape[1], dtype=np.float32)
    np.fill_diagonal(corr, 1.0)
    return corr.astype(np.float32)


def _fit_feature_drift_reference(
    X_ref: pd.DataFrame,
    feature_names: list[str],
    *,
    bins: int = 10,
) -> dict[str, Any]:
    features = [str(c) for c in feature_names if str(c) in X_ref.columns]
    if not features:
        return {}
    ref = X_ref.loc[:, features].astype(np.float32, copy=False)
    values = ref.to_numpy(dtype=np.float32, copy=False)
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype(np.float32)
    z = (values - mean) / std
    centroid_raw = np.nanmean(values / std, axis=0)
    centroid_raw = np.nan_to_num(centroid_raw, nan=0.0, posinf=0.0, neginf=0.0)
    centroid_norm = float(np.linalg.norm(centroid_raw))
    centroid = (
        centroid_raw / centroid_norm
        if centroid_norm > 1e-12
        else np.zeros(len(features), dtype=np.float32)
    )
    edges_by_feature: dict[str, list[float]] = {}
    ref_props_by_feature: dict[str, list[float]] = {}
    bins = max(2, int(bins))
    for feature in features:
        series = pd.to_numeric(ref[feature], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        vals = series.dropna().to_numpy(dtype=np.float64)
        if vals.size < max(10, bins):
            continue
        quantiles = np.linspace(0.0, 1.0, bins + 1)
        edges = np.unique(np.nanquantile(vals, quantiles))
        if edges.size < 3:
            continue
        edges[0] = -np.inf
        edges[-1] = np.inf
        counts, _ = np.histogram(vals, bins=edges)
        props = counts.astype(np.float64) / max(float(counts.sum()), 1.0)
        edges_by_feature[feature] = [float(v) for v in edges]
        ref_props_by_feature[feature] = [float(v) for v in props]
    return {
        "feature_names": features,
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "centroid": centroid.astype(np.float32),
        "corr": _safe_corr_matrix(z),
        "psi_edges": edges_by_feature,
        "psi_ref_props": ref_props_by_feature,
        "bins": int(bins),
        "fit_rows": int(len(ref)),
    }


def _feature_drift_values(
    X_current: pd.DataFrame,
    reference: dict[str, Any] | None,
) -> dict[str, float]:
    if not reference:
        return {
            "regime_centroid_similarity_train": 1.0,
            "feature_drift_psi_core": 0.0,
            "feature_drift_cov_shift": 0.0,
        }
    ref_features = [str(c) for c in reference.get("feature_names", [])]
    features = [name for name in ref_features if name in X_current.columns]
    if not features:
        return {
            "regime_centroid_similarity_train": 0.0,
            "feature_drift_psi_core": 0.0,
            "feature_drift_cov_shift": 0.0,
        }
    Xc = X_current.loc[:, features].astype(np.float32, copy=False)
    values = Xc.to_numpy(dtype=np.float32, copy=False)
    ref_pos = {name: i for i, name in enumerate(ref_features)}
    idx = np.asarray([ref_pos[name] for name in features], dtype=np.int32)
    mean = np.asarray(reference.get("mean", np.zeros(len(ref_features))), dtype=np.float32)[idx]
    std = np.asarray(reference.get("std", np.ones(len(ref_features))), dtype=np.float32)[idx]
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype(np.float32)
    z = (values - mean) / std
    cur_centroid = np.nanmean(values / std, axis=0)
    cur_centroid = np.nan_to_num(cur_centroid, nan=0.0, posinf=0.0, neginf=0.0)
    cur_norm = float(np.linalg.norm(cur_centroid))
    ref_centroid = np.asarray(reference.get("centroid", np.zeros(len(ref_features))), dtype=np.float32)[idx]
    ref_norm = float(np.linalg.norm(ref_centroid))
    if cur_norm > 1e-12 and ref_norm > 1e-12:
        centroid_similarity = float(
            np.clip(np.dot(cur_centroid / cur_norm, ref_centroid / ref_norm), -1.0, 1.0)
        )
    else:
        centroid_similarity = 0.0
    psi_vals: list[float] = []
    eps = 1e-6
    edges_by_feature = reference.get("psi_edges", {}) or {}
    ref_props_by_feature = reference.get("psi_ref_props", {}) or {}
    for feature in features:
        edges = edges_by_feature.get(feature)
        ref_props = ref_props_by_feature.get(feature)
        if not edges or not ref_props:
            continue
        vals = pd.to_numeric(Xc[feature], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).dropna().to_numpy(dtype=np.float64)
        if vals.size == 0:
            continue
        counts, _ = np.histogram(vals, bins=np.asarray(edges, dtype=np.float64))
        cur_props = counts.astype(np.float64) / max(float(counts.sum()), 1.0)
        ref_arr = np.asarray(ref_props, dtype=np.float64)
        m = min(len(ref_arr), len(cur_props))
        if m == 0:
            continue
        psi = np.sum((cur_props[:m] - ref_arr[:m]) * np.log((cur_props[:m] + eps) / (ref_arr[:m] + eps)))
        if np.isfinite(psi):
            psi_vals.append(float(psi))
    psi_core = float(np.nanmean(psi_vals)) if psi_vals else 0.0
    cur_corr = _safe_corr_matrix(z)
    ref_corr_full = np.asarray(reference.get("corr", np.eye(len(ref_features))), dtype=np.float32)
    ref_corr = (
        ref_corr_full[np.ix_(idx, idx)]
        if ref_corr_full.ndim == 2 and ref_corr_full.shape[0] >= len(ref_features)
        else np.eye(len(features), dtype=np.float32)
    )
    if cur_corr.shape == ref_corr.shape and cur_corr.size:
        cov_shift = float(np.linalg.norm(cur_corr - ref_corr, ord="fro") / max(cur_corr.shape[0], 1))
    else:
        cov_shift = 0.0
    return {
        "regime_centroid_similarity_train": centroid_similarity,
        "feature_drift_psi_core": psi_core,
        "feature_drift_cov_shift": cov_shift,
    }


def _append_feature_drift_meta_features(
    meta_features: pd.DataFrame,
    X_current: pd.DataFrame,
    reference: dict[str, Any] | None,
) -> None:
    values = _feature_drift_values(X_current, reference)
    n = len(meta_features)
    for name in (
        "regime_centroid_similarity_train",
        "feature_drift_psi_core",
        "feature_drift_cov_shift",
    ):
        meta_features[name] = np.full(n, float(values.get(name, 0.0)), dtype=np.float32)


def _normalize_importance_vector(
    values: np.ndarray,
    *,
    prior_strength: float = LGBM_IMPORTANCE_INSTABILITY_PRIOR_STRENGTH,
    eps: float = 1e-12,
) -> np.ndarray:
    """Normalize a non-negative importance vector with shrinkage toward uniform."""
    v = np.nan_to_num(
        np.asarray(values, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    v = np.maximum(v, 0.0)
    n = int(len(v))
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    total = float(np.sum(v))
    if total <= eps:
        out = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        out = v / total
    prior_strength = float(np.clip(prior_strength, 0.0, 0.90))
    uniform = np.full(n, 1.0 / n, dtype=np.float64)
    out = (1.0 - prior_strength) * out + prior_strength * uniform
    out = out / max(float(np.sum(out)), eps)
    return out.astype(np.float32)


def _topk_oof_focus_weights(
    pred: np.ndarray,
    *,
    top_frac: float,
    softness: float = LGBM_IMPORTANCE_TOPK_FOCUS_SOFTNESS,
    eps: float = 1e-12,
) -> np.ndarray:
    """Smooth top-k row weights from validation/OOF prediction ranks."""
    p = np.asarray(pred, dtype=np.float32)
    n = int(len(p))
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    rank = _safe_rank_pct(p)
    top_frac = float(np.clip(top_frac, 0.001, 0.95))
    cutoff = 1.0 - top_frac
    softness = max(float(softness), eps)
    denom = max(top_frac * softness, eps)
    w = np.clip((rank - cutoff) / denom, 0.0, 1.0).astype(np.float32)
    if float(np.sum(w)) <= eps:
        k = max(1, int(np.ceil(top_frac * n)))
        order = np.argsort(p)
        w = np.zeros(n, dtype=np.float32)
        w[order[-k:]] = 1.0
    w = w / max(float(np.sum(w)), eps)
    return w.astype(np.float32)


def _topk_focused_gain_split_importance(
    model: Any,
    X_valid: pd.DataFrame,
    pred_valid: np.ndarray,
    gain: np.ndarray,
    split: np.ndarray,
    *,
    objective_mode: str | None,
    contrib_blend: float = LGBM_IMPORTANCE_TOPK_CONTRIB_BLEND,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Return top-k-focused gain and split vectors from validation rows."""
    n_features = int(X_valid.shape[1])
    top_frac = _target_top_fraction(objective_mode)
    w = _topk_oof_focus_weights(pred_valid, top_frac=top_frac)
    gain_norm = _normalize_importance_vector(gain)
    split_norm = _normalize_importance_vector(split)
    contrib_norm = np.zeros(n_features, dtype=np.float32)
    contrib_available = False
    try:
        contrib = _predict_contrib_matrix(model, X_valid, n_features)
        if contrib is not None and np.asarray(contrib).size:
            contrib = np.asarray(contrib, dtype=np.float32)[:, :n_features]
            abs_contrib = np.abs(
                np.nan_to_num(contrib, nan=0.0, posinf=0.0, neginf=0.0)
            )
            topk_contrib = np.sum(abs_contrib * w[:, None], axis=0)
            contrib_norm = _normalize_importance_vector(topk_contrib)
            contrib_available = True
    except Exception:
        contrib_available = False
    blend = float(np.clip(contrib_blend, 0.0, 1.0))
    if contrib_available:
        gain_focus = _normalize_importance_vector(
            (1.0 - blend) * gain_norm + blend * contrib_norm
        )
        split_focus = _normalize_importance_vector(
            (1.0 - blend) * split_norm + blend * contrib_norm
        )
    else:
        gain_focus = gain_norm
        split_focus = split_norm
    diag = {
        "topk_focus_frac": float(top_frac),
        "topk_focus_rows_effective": float(1.0 / max(float(np.sum(w * w)), 1e-12)),
        "topk_contrib_available": float(contrib_available),
    }
    return gain_focus.astype(np.float32), split_focus.astype(np.float32), diag


def _bounded_importance_instability_from_matrix(
    mat: np.ndarray,
    *,
    material_top_frac: float = LGBM_IMPORTANCE_INSTABILITY_MATERIAL_TOP_FRAC,
    cv_cap: float = LGBM_IMPORTANCE_INSTABILITY_CV_CAP,
    prior_strength: float = LGBM_IMPORTANCE_INSTABILITY_PRIOR_STRENGTH,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Bounded robust importance instability in [0, 1]."""
    arr = np.asarray(mat, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] == 0:
        return {
            "instability": 0.0,
            "instability_raw_cv": 0.0,
            "material_feature_count": 0.0,
        }
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.maximum(arr, 0.0)
    row_sum = arr.sum(axis=1, keepdims=True)
    arr = arr / np.maximum(row_sum, eps)
    n_fits, n_features = arr.shape
    del n_fits
    prior_strength = float(np.clip(prior_strength, 0.0, 0.90))
    uniform = np.full_like(arr, 1.0 / n_features)
    arr = (1.0 - prior_strength) * arr + prior_strength * uniform
    arr = arr / np.maximum(arr.sum(axis=1, keepdims=True), eps)
    mean_imp = arr.mean(axis=0)
    k = max(1, int(np.ceil(float(material_top_frac) * n_features)))
    k = min(k, n_features)
    material_idx = np.argsort(mean_imp)[-k:]
    sub = arr[:, material_idx]
    sub_mean = mean_imp[material_idx]
    q25 = np.percentile(sub, 25, axis=0)
    q75 = np.percentile(sub, 75, axis=0)
    robust_std = (q75 - q25) / 1.349
    prior = prior_strength / max(n_features, 1)
    cv = robust_std / np.maximum(sub_mean + prior, eps)
    cv = np.nan_to_num(cv, nan=0.0, posinf=cv_cap, neginf=0.0)
    cv_clipped = np.clip(cv, 0.0, cv_cap)
    weights = np.sqrt(np.maximum(sub_mean, eps))
    weights = weights / max(float(weights.sum()), eps)
    raw_cv = float(np.sum(weights * cv))
    bounded = float(np.sum(weights * cv_clipped) / max(cv_cap, eps))
    return {
        "instability": float(np.clip(bounded, 0.0, 1.0)),
        "instability_raw_cv": raw_cv,
        "material_feature_count": float(len(material_idx)),
    }


def _combined_gain_split_instability(
    gain_focus_runs: list[np.ndarray],
    split_focus_runs: list[np.ndarray],
) -> dict[str, float]:
    """Combine top-k-focused gain/split instability."""
    if not LGBM_IMPORTANCE_INSTABILITY_ENABLE:
        return {
            "importance_instability": 0.0,
            "gain_instability": 0.0,
            "split_instability": 0.0,
            "gain_instability_raw_cv": 0.0,
            "split_instability_raw_cv": 0.0,
            "importance_material_feature_count": 0.0,
        }
    gain_mat = np.vstack(gain_focus_runs) if gain_focus_runs else np.empty((0, 0))
    split_mat = np.vstack(split_focus_runs) if split_focus_runs else np.empty((0, 0))
    gain_info = _bounded_importance_instability_from_matrix(gain_mat)
    split_info = _bounded_importance_instability_from_matrix(split_mat)
    gain_w = max(0.0, float(LGBM_IMPORTANCE_INSTABILITY_GAIN_WEIGHT))
    split_w = max(0.0, float(LGBM_IMPORTANCE_INSTABILITY_SPLIT_WEIGHT))
    denom = max(gain_w + split_w, 1e-12)
    gain_w /= denom
    split_w /= denom
    combined = (
        gain_w * float(gain_info["instability"])
        + split_w * float(split_info["instability"])
    )
    return {
        "importance_instability": float(np.clip(combined, 0.0, 1.0)),
        "gain_instability": float(gain_info["instability"]),
        "split_instability": float(split_info["instability"]),
        "gain_instability_raw_cv": float(gain_info["instability_raw_cv"]),
        "split_instability_raw_cv": float(split_info["instability_raw_cv"]),
        "importance_material_feature_count": float(
            max(
                gain_info["material_feature_count"],
                split_info["material_feature_count"],
            )
        ),
    }


def _permutation_delta_j(
    model: Any,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    *,
    base_pred: np.ndarray,
    classifier: bool,
    groups_valid: Any,
    returns_valid: Any,
    rng: np.random.Generator,
    feature_indices: np.ndarray,
    objective_mode: str | None = "train_base",
) -> np.ndarray:
    n_features = X_valid.shape[1]
    out = np.zeros(n_features, dtype=np.float32)
    if LGBM_PERMUTATION_MAX_ROWS > 0 and len(X_valid) > LGBM_PERMUTATION_MAX_ROWS:
        sample_idx = np.sort(
            rng.choice(len(X_valid), size=int(LGBM_PERMUTATION_MAX_ROWS), replace=False)
        )
        X_perm_base = X_valid.iloc[sample_idx].reset_index(drop=True)
        y_perm = np.asarray(y_valid)[sample_idx]
        pred_perm_base = np.asarray(base_pred)[sample_idx]
        returns_perm = (
            np.asarray(returns_valid)[sample_idx]
            if returns_valid is not None and len(np.asarray(returns_valid)) == len(X_valid)
            else returns_valid
        )
        groups_perm = _groups_take(groups_valid, sample_idx)
        base_j_perm = _objective_value(
            _metric_pack(
                y_perm,
                pred_perm_base,
                classifier=classifier,
                groups=groups_perm,
                returns=returns_perm,
            ),
            objective_mode,
        )
        if np.isfinite(base_j_perm):
            base_j = base_j_perm
    else:
        X_perm_base = X_valid
        y_perm = np.asarray(y_valid)
        pred_perm_base = np.asarray(base_pred)
        returns_perm = returns_valid
        groups_perm = groups_valid
    base_j = _objective_value(
        _metric_pack(
            y_perm,
            pred_perm_base,
            classifier=classifier,
            groups=groups_perm,
            returns=returns_perm,
        ),
        objective_mode,
    )
    if not np.isfinite(base_j):
        return out
    Xp = X_perm_base.copy()
    for j in feature_indices:
        vals = Xp.iloc[:, int(j)].to_numpy(copy=True)
        deltas: list[float] = []
        repeats = 1 if n_features > 200 else max(1, LGBM_PERMUTATION_REPEATS)
        for _ in range(repeats):
            Xp.iloc[:, int(j)] = rng.permutation(vals)
            pred_perm = _predict_lgbm_raw(model, Xp, "classifier" if classifier else "regressor")
            perm_j = _objective_value(_metric_pack(y_perm, pred_perm, classifier=classifier, groups=groups_perm, returns=returns_perm), objective_mode)
            deltas.append(base_j - perm_j)
        Xp.iloc[:, int(j)] = vals
        out[int(j)] = float(np.median(deltas)) if deltas else 0.0
    return out


def _redundancy_penalty(X: pd.DataFrame, features: list[str], quality: np.ndarray, *, random_state: int) -> np.ndarray:
    p = len(features)
    out = np.zeros(p, dtype=np.float32)
    if p <= 1:
        return out
    rng = np.random.default_rng(random_state)
    sub_n = min(len(X), 5000)
    sub = rng.choice(len(X), size=sub_n, replace=False) if len(X) > sub_n else np.arange(len(X))
    ranks = pd.DataFrame(X.iloc[sub][features].to_numpy(dtype=np.float32)).rank(pct=True).to_numpy(dtype=np.float32)
    corr = np.abs(np.corrcoef(ranks, rowvar=False))
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(np.asarray(quality, dtype=np.float32))[::-1]
    seen: list[int] = []
    for idx in order:
        if seen:
            max_corr = float(np.max(corr[int(idx), np.asarray(seen, dtype=np.int32)]))
            out[int(idx)] = max(0.0, (max_corr - LGBM_REDUNDANCY_PENALTY_START) / max(1.0 - LGBM_REDUNDANCY_PENALTY_START, 1e-6))
        seen.append(int(idx))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _permutation_candidate_indices(gain: np.ndarray, split: np.ndarray, n_features: int) -> np.ndarray:
    quality = _importance_rank_scores(gain) + _importance_rank_scores(split)
    if LGBM_PERMUTATION_MAX_FEATURES > 0:
        cap = min(int(LGBM_PERMUTATION_MAX_FEATURES), int(n_features))
    elif n_features > 200:
        cap = min(200, max(40, int(np.ceil(0.35 * int(n_features)))))
    else:
        cap = int(n_features)
    candidate_mask = np.ones(int(n_features), dtype=bool)
    finite_quality = quality[np.isfinite(quality)]
    if int(n_features) >= 20 and finite_quality.size:
        weak_frac = float(np.clip(LGBM_PERMUTATION_SKIP_WEAK_BOTTOM_FRAC, 0.0, 0.95))
        strong_frac = float(np.clip(LGBM_PERMUTATION_SKIP_STRONG_TOP_FRAC, 0.0, 0.95))
        if weak_frac > 0.0:
            weak_cut = float(np.quantile(finite_quality, weak_frac))
            candidate_mask &= quality > weak_cut
        if strong_frac > 0.0:
            strong_cut = float(np.quantile(finite_quality, 1.0 - strong_frac))
            candidate_mask &= quality < strong_cut
    pool = np.where(candidate_mask & np.isfinite(quality))[0]
    if len(pool) == 0:
        pool = np.where(np.isfinite(quality))[0]
    if len(pool) == 0:
        return np.array([], dtype=np.int32)
    idx = pool[np.argsort(quality[pool])[-min(cap, len(pool)):]]
    return np.asarray(np.sort(idx), dtype=np.int32)


def _lgbm_stability_selection_pass(
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray,
    features: list[str],
    *,
    classifier: bool,
    groups: Any = None,
    returns: Any = None,
    metric_y: np.ndarray | None = None,
    random_state: int,
    seeds: list[int] | None = None,
    objective_mode: str | None = "train_base",
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    t0 = time.perf_counter()
    if seeds is None:
        seeds = [int(random_state)]
    Xf = X[features].reset_index(drop=True)
    y_arr = np.asarray(y)
    y_metric = np.asarray(metric_y if metric_y is not None else y_arr)
    ret_arr = _as_returns(y_metric, returns)
    p = len(features)
    configs = []
    for depth in (4, 5):
        for l2 in (5.0, 15.0):
            configs.append({"max_depth": depth, "num_leaves": 2 ** depth, "reg_lambda": l2})
    configs = configs[: min(len(configs), int(LGBM_STABILITY_CONFIGS))]
    n_fits = 0
    used_count = np.zeros(p, dtype=np.float32)
    top_used_count = np.zeros(p, dtype=np.float32)
    gain_rank_sum = np.zeros(p, dtype=np.float32)
    split_rank_sum = np.zeros(p, dtype=np.float32)
    perm_values: list[np.ndarray] = []
    direction_values: list[np.ndarray] = []
    margin_values: list[np.ndarray] = []
    fit_scores: list[float] = []
    fold_metrics_all: list[dict[str, float]] = []
    best_oof = np.full(len(y_arr), np.nan, dtype=np.float32)
    best_score = -np.inf
    base_weight = np.asarray(sample_weight, dtype=np.float32)
    current_weight = base_weight.copy()
    prev_oof: np.ndarray | None = None
    stability_splits = max(2, int(LGBM_CV_SPLITS))
    splitter, y_split = _splitter(y_metric, classifier, random_state, n_splits=stability_splits)
    dir_t0 = time.perf_counter()
    round_direction, round_margin = _direction_vectors_binned_mi(
        Xf,
        y_metric,
        classifier=classifier,
        groups=groups,
        returns=ret_arr,
        random_state=random_state + 509,
    )
    tprint(
        "LGBM direction proxy complete: "
        f"features={p}, max_rows={LGBM_DIRECTION_MAX_ROWS}, "
        f"elapsed={time.perf_counter() - dir_t0:.1f}s."
    )
    rng_perm = np.random.default_rng(random_state + 107)
    all_fit_usage: list[np.ndarray] = []
    all_fit_gain_rank: list[np.ndarray] = []
    all_fit_split_rank: list[np.ndarray] = []
    all_fit_perm: list[np.ndarray] = []
    all_fit_perm_evaluated: list[np.ndarray] = []
    all_fit_direction: list[np.ndarray] = []
    all_fit_margin: list[np.ndarray] = []
    all_fit_gain_focus: list[np.ndarray] = []
    all_fit_split_focus: list[np.ndarray] = []
    topk_contrib_available_flags: list[float] = []
    topk_effective_rows: list[float] = []
    config_records: list[dict[str, Any]] = []
    total_fits = len(seeds) * len(configs) * stability_splits
    tprint(
        "LGBM stability selection pass started: "
        f"rows={len(y_arr)}, features={p}, seeds={len(seeds)}, "
        f"configs={len(configs)}, expected_fits={total_fits}, "
        f"objective={_normalize_objective_mode(objective_mode)}."
    )
    for seed in seeds:
        for cfg_i, cfg in enumerate(configs, start=1):
            cfg_t0 = time.perf_counter()
            cfg_oof = np.full(len(y_arr), np.nan, dtype=np.float32)
            cfg_metrics: list[dict[str, float]] = []
            cfg_fold_records: list[dict[str, Any]] = []
            for fold_i, (tr, va) in enumerate(splitter.split(np.zeros(len(y_split)), y_split), start=1):
                fold_t0 = time.perf_counter()
                tprint(
                    "LGBM stability fit started: "
                    f"seed={seed}, config={cfg_i}/{len(configs)}, fold={fold_i}/{stability_splits}, "
                    f"train_rows={len(tr)}, valid_rows={len(va)}, features={p}."
                )
                params = _base_lgbm_params(int(seed) + cfg_i * 1000 + fold_i, classifier=classifier, overrides=cfg)
                model = _fit_lgbm_model(
                    Xf.iloc[tr].reset_index(drop=True),
                    y_arr[tr],
                    current_weight[tr],
                    classifier=classifier,
                    params=params,
                )
                tprint(
                    "LGBM stability fit model complete: "
                    f"seed={seed}, config={cfg_i}/{len(configs)}, fold={fold_i}/{stability_splits}, "
                    f"model_fit_elapsed={time.perf_counter() - fold_t0:.1f}s."
                )
                mode_name = "classifier" if classifier else "regressor"
                X_tr_fold = Xf.iloc[tr].reset_index(drop=True)
                X_va_fold = Xf.iloc[va].reset_index(drop=True)
                pred = _predict_lgbm_raw(model, X_va_fold, mode_name)
                cfg_oof[va] = pred
                pred_train = _predict_lgbm_raw(model, X_tr_fold, mode_name)
                fold_groups = _groups_take(groups, va)
                fold_returns = ret_arr[va]
                train_groups = _groups_take(groups, tr)
                train_returns = ret_arr[tr]
                train_metrics = _metric_pack(
                    y_metric[tr],
                    pred_train,
                    classifier=classifier,
                    groups=train_groups,
                    returns=train_returns,
                )
                valid_metrics = _metric_pack(
                    y_metric[va],
                    pred,
                    classifier=classifier,
                    groups=fold_groups,
                    returns=fold_returns,
                )
                metrics = _apply_overfit_gap_penalty(
                    train_metrics,
                    valid_metrics,
                    objective_mode=objective_mode,
                )
                cfg_metrics.append(metrics)
                fit_scores.append(_objective_value(metrics, objective_mode))
                gain, split = _feature_importances(model, p)
                gain_focus, split_focus, focus_diag = _topk_focused_gain_split_importance(
                    model,
                    X_va_fold,
                    pred,
                    gain,
                    split,
                    objective_mode=objective_mode,
                )
                all_fit_gain_focus.append(gain_focus)
                all_fit_split_focus.append(split_focus)
                topk_contrib_available_flags.append(
                    float(focus_diag["topk_contrib_available"])
                )
                topk_effective_rows.append(float(focus_diag["topk_focus_rows_effective"]))
                used = (split > 0).astype(np.float32)
                all_fit_usage.append(used)
                all_fit_gain_rank.append(_importance_rank_scores(gain))
                all_fit_split_rank.append(_importance_rank_scores(split))
                fit_id = len(all_fit_perm)
                all_fit_perm.append(np.zeros(p, dtype=np.float32))
                evaluated = np.zeros(p, dtype=bool)
                all_fit_perm_evaluated.append(evaluated)
                cfg_fold_records.append(
                    {
                        "fit_id": int(fit_id),
                        "seed": int(seed),
                        "cfg_i": int(cfg_i),
                        "fold_i": int(fold_i),
                        "model": model,
                        "va": np.asarray(va, dtype=np.int32),
                        "pred": np.asarray(pred, dtype=np.float32),
                        "gain": np.asarray(gain, dtype=np.float32),
                        "split": np.asarray(split, dtype=np.float32),
                    }
                )
                all_fit_direction.append(round_direction.copy())
                all_fit_margin.append(round_margin.copy())
                n_fits += 1
            agg = _aggregate_j(cfg_metrics, objective_mode=objective_mode)
            cfg_score = float(agg.get("J_final", -np.inf))
            config_records.append(
                {
                    "score": cfg_score,
                    "seed": int(seed),
                    "cfg_i": int(cfg_i),
                    "folds": cfg_fold_records,
                }
            )
            fold_metrics_all.extend(cfg_metrics)
            if cfg_score > best_score:
                best_score = cfg_score
                best_oof = cfg_oof.copy()
            metric_fill = float(np.mean(y_metric))
            distill = _compute_weight_distillation(y_metric, np.nan_to_num(cfg_oof, nan=float(metric_fill)), prev_oof, is_classifier=classifier, include_false_positive_focus=False)
            fp_weight = _false_positive_avoidance_weight(
                y_metric,
                np.nan_to_num(cfg_oof, nan=float(metric_fill)),
                classifier=classifier,
                top_frac=_target_top_fraction(objective_mode),
            )
            current_weight, ess = _normalize_weights(base_weight * distill * fp_weight)
            prev_oof = np.nan_to_num(cfg_oof, nan=float(metric_fill)).astype(np.float32)
            tprint(
                f"LGBM stability grid complete: seed={seed} config={cfg_i}/{len(configs)} "
                f"score={cfg_score:.4f} ess={ess:.1f} "
                f"elapsed={time.perf_counter() - cfg_t0:.1f}s."
            )
    if n_fits == 0:
        raise RuntimeError("No LGBM stability fits completed")
    perm_t0 = time.perf_counter()
    perm_config_cap = max(0, int(LGBM_PERMUTATION_TOP_CONFIGS))
    eligible_configs = [
        rec for rec in config_records if np.isfinite(float(rec.get("score", np.nan)))
    ]
    eligible_configs = sorted(
        eligible_configs,
        key=lambda rec: float(rec.get("score", -np.inf)),
        reverse=True,
    )[:perm_config_cap]
    tprint(
        "LGBM stability permutation audit started: "
        f"top_configs={len(eligible_configs)}/{len(config_records)}, "
        f"max_features={LGBM_PERMUTATION_MAX_FEATURES}, "
        f"max_rows={LGBM_PERMUTATION_MAX_ROWS}."
    )
    permuted_folds = 0
    permuted_features = 0
    for rec in eligible_configs:
        cfg_i = int(rec.get("cfg_i", 0))
        seed = int(rec.get("seed", random_state))
        for fold_rec in rec.get("folds", []):
            va = np.asarray(fold_rec["va"], dtype=np.int32)
            candidate_idx = _permutation_candidate_indices(
                np.asarray(fold_rec["gain"], dtype=np.float32),
                np.asarray(fold_rec["split"], dtype=np.float32),
                p,
            )
            fit_id = int(fold_rec["fit_id"])
            fold_i = int(fold_rec.get("fold_i", 0))
            tprint(
                "LGBM stability permutation started: "
                f"seed={seed}, config={cfg_i}/{len(configs)}, fold={fold_i}/{stability_splits}, "
                f"candidate_features={len(candidate_idx)}."
            )
            perm = _permutation_delta_j(
                fold_rec["model"],
                Xf.iloc[va].reset_index(drop=True),
                y_metric[va],
                base_pred=np.asarray(fold_rec["pred"], dtype=np.float32),
                classifier=classifier,
                groups_valid=_groups_take(groups, va),
                returns_valid=ret_arr[va],
                rng=rng_perm,
                feature_indices=np.asarray(candidate_idx, dtype=np.int32),
                objective_mode=objective_mode,
            )
            all_fit_perm[fit_id] = perm
            evaluated = np.zeros(p, dtype=bool)
            evaluated[np.asarray(candidate_idx, dtype=np.int32)] = True
            all_fit_perm_evaluated[fit_id] = evaluated
            permuted_folds += 1
            permuted_features += int(len(candidate_idx))
            tprint(
                "LGBM stability permutation complete: "
                f"seed={seed}, config={cfg_i}/{len(configs)}, fold={fold_i}/{stability_splits}."
            )
    tprint(
        "LGBM stability permutation audit complete: "
        f"folds={permuted_folds}, feature_evals={permuted_features}, "
        f"elapsed={time.perf_counter() - perm_t0:.1f}s."
    )
    tprint(
        "LGBM stability selection pass aggregating feature statistics: "
        f"completed_fits={n_fits}, elapsed={time.perf_counter() - t0:.1f}s."
    )
    usage = np.vstack(all_fit_usage).astype(np.float32)
    gain_rank = np.vstack(all_fit_gain_rank).astype(np.float32)
    split_rank = np.vstack(all_fit_split_rank).astype(np.float32)
    perm_mat = np.vstack(all_fit_perm).astype(np.float32)
    perm_evaluated = np.vstack(all_fit_perm_evaluated).astype(bool)
    dirs = np.vstack(all_fit_direction).astype(np.float32)
    margins = np.vstack(all_fit_margin).astype(np.float32)
    gain_focus_mat = (
        np.vstack(all_fit_gain_focus).astype(np.float32)
        if all_fit_gain_focus
        else np.zeros((0, p), dtype=np.float32)
    )
    split_focus_mat = (
        np.vstack(all_fit_split_focus).astype(np.float32)
        if all_fit_split_focus
        else np.zeros((0, p), dtype=np.float32)
    )
    if gain_focus_mat.shape[0] >= 2:
        gain_focus_mean = gain_focus_mat.mean(axis=0).astype(np.float32)
        gain_focus_std = gain_focus_mat.std(axis=0).astype(np.float32)
    else:
        gain_focus_mean = np.zeros(p, dtype=np.float32)
        gain_focus_std = np.zeros(p, dtype=np.float32)
    if split_focus_mat.shape[0] >= 2:
        split_focus_mean = split_focus_mat.mean(axis=0).astype(np.float32)
        split_focus_std = split_focus_mat.std(axis=0).astype(np.float32)
    else:
        split_focus_mean = np.zeros(p, dtype=np.float32)
        split_focus_std = np.zeros(p, dtype=np.float32)
    fit_scores_arr = np.asarray(fit_scores, dtype=np.float32)
    top_threshold = float(np.nanmedian(fit_scores_arr)) if len(fit_scores_arr) else -np.inf
    top_mask = np.isfinite(fit_scores_arr) & (fit_scores_arr >= top_threshold)
    presence_rate = np.mean(usage > 0.0, axis=0).astype(np.float32)
    top_count = np.sum(usage[top_mask] > 0.0, axis=0).astype(np.float32) if np.any(top_mask) else np.zeros(p, dtype=np.float32)
    gain_rank_score = np.mean(gain_rank, axis=0).astype(np.float32)
    split_rank_score = np.mean(split_rank, axis=0).astype(np.float32)
    perm_eval_count = np.sum(perm_evaluated, axis=0).astype(np.float32)
    permutation_evaluated_rate = np.mean(perm_evaluated, axis=0).astype(np.float32)
    positive_perm_rate = np.divide(
        np.sum((perm_mat > LGBM_PERMUTATION_EPS) & perm_evaluated, axis=0),
        np.maximum(perm_eval_count, 1.0),
    ).astype(np.float32)
    median_perm = np.zeros(p, dtype=np.float32)
    for j in range(p):
        vals = perm_mat[perm_evaluated[:, j], j]
        median_perm[j] = float(np.median(vals)) if len(vals) else 0.0
    direction_stability = np.asarray([_weighted_direction_stability(dirs[:, j], margins[:, j]) for j in range(p)], dtype=np.float32)
    direction = np.asarray([1 if np.sum(dirs[:, j] * margins[:, j]) >= 0 else -1 for j in range(p)], dtype=np.int8)
    norm_perm = _rank01(np.maximum(median_perm, 0.0))
    prelim_quality = 0.50 * norm_perm + 0.25 * positive_perm_rate + 0.15 * presence_rate + 0.10 * direction_stability
    redundancy = _redundancy_penalty(Xf, features, prelim_quality, random_state=random_state + 677)
    worst_context_activity_penalty = np.zeros(p, dtype=np.float32)
    if LGBM_FEATURE_SELECTION_OBJECTIVE == "tail_control" and len(all_fit_gain_focus) >= 2:
        fit_quality = np.asarray(fit_scores, dtype=np.float64)
        valid_fit = np.isfinite(fit_quality)
        if np.any(valid_fit):
            n_tail = max(1, int(np.ceil(0.20 * int(np.sum(valid_fit)))))
            valid_idx = np.where(valid_fit)[0]
            worst_idx = valid_idx[np.argsort(fit_quality[valid_idx])[:n_tail]]
            gain_focus_all = np.vstack(all_fit_gain_focus).astype(np.float32)
            split_focus_all = np.vstack(all_fit_split_focus).astype(np.float32)
            focus_all = 0.5 * gain_focus_all + 0.5 * split_focus_all
            all_mean = np.mean(focus_all[valid_idx], axis=0)
            worst_mean = np.mean(focus_all[worst_idx], axis=0)
            excess = np.maximum(worst_mean - all_mean, 0.0)
            worst_context_activity_penalty = _rank01(excess).astype(np.float32)
    feature_score = (
        0.40 * norm_perm
        + 0.20 * positive_perm_rate
        + 0.15 * presence_rate
        + 0.10 * direction_stability
        + 0.075 * gain_rank_score
        + 0.075 * split_rank_score
        - 0.10 * redundancy
        - float(LGBM_TAIL_WORST_FEATURE_PENALTY) * worst_context_activity_penalty
    ).astype(np.float32)
    perm_filter_applies = permutation_evaluated_rate > 0.0
    hard_drop = (
        (direction_stability < LGBM_DIRECTION_STABILITY_MIN)
        | (perm_filter_applies & (positive_perm_rate < LGBM_POSITIVE_PERM_RATE_MIN))
        | (perm_filter_applies & (median_perm < -LGBM_PERMUTATION_EPS) & (presence_rate < LGBM_LOW_PRESENCE_RATE))
    )
    rescue = np.zeros(p, dtype=bool)
    feature_score = np.where(hard_drop, -1.0, feature_score).astype(np.float32)
    stats = pd.DataFrame(
        {
            "feature": features,
            "feature_score": feature_score,
            "normalized_permutation_delta_J": norm_perm,
            "median_permutation_delta_J": median_perm,
            "positive_permutation_rate": positive_perm_rate,
            "permutation_evaluated_rate": permutation_evaluated_rate,
            "presence_rate": presence_rate,
            "direction": direction,
            "direction_stability": direction_stability,
            "gain_rank_score": gain_rank_score,
            "split_rank_score": split_rank_score,
            "topk_gain_focus_mean": gain_focus_mean,
            "topk_gain_focus_std": gain_focus_std,
            "topk_split_focus_mean": split_focus_mean,
            "topk_split_focus_std": split_focus_std,
            "worst_context_activity_penalty": worst_context_activity_penalty,
            "selected_in_top_model_count": top_count,
            "redundancy_penalty": redundancy,
            "hard_drop": hard_drop.astype(bool),
            "rescue": rescue.astype(bool),
        }
    )
    agg_all = _aggregate_j(fold_metrics_all, objective_mode=objective_mode)
    instability_info = _combined_gain_split_instability(
        all_fit_gain_focus,
        all_fit_split_focus,
    )
    raw_j_final = float(agg_all.get("J_final", -999.0))
    importance_instability = float(instability_info["importance_instability"])
    importance_penalty = (
        float(LGBM_IMPORTANCE_INSTABILITY_PENALTY) * importance_instability
        if LGBM_IMPORTANCE_INSTABILITY_ENABLE
        and LGBM_FEATURE_SELECTION_OBJECTIVE != "tail_control"
        else 0.0
    )
    agg_all["J_final_pre_importance_instability_penalty"] = raw_j_final
    agg_all["importance_instability_penalty"] = float(importance_penalty)
    agg_all.update(instability_info)
    agg_all["topk_contrib_available_rate"] = (
        float(np.mean(topk_contrib_available_flags))
        if topk_contrib_available_flags
        else 0.0
    )
    agg_all["topk_focus_effective_rows_mean"] = (
        float(np.mean(topk_effective_rows)) if topk_effective_rows else 0.0
    )
    agg_all["feature_selection_objective"] = (
        LGBM_FEATURE_SELECTION_OBJECTIVE or "default"
    )
    agg_all["worst_context_activity_penalty_mean"] = float(
        np.mean(worst_context_activity_penalty) if len(worst_context_activity_penalty) else 0.0
    )
    agg_all["J_final"] = float(raw_j_final - importance_penalty)
    agg_all["selected_objective"] = agg_all["J_final"]
    mode_obj = _normalize_objective_mode(objective_mode)
    if mode_obj == "train_meta":
        agg_all["J_meta"] = agg_all["J_final"]
    else:
        agg_all["J_base"] = agg_all["J_final"]
    tprint(
        "LGBM stability selection pass complete: "
        f"J={float(agg_all.get('J_final', -999.0)):.4f}, "
        f"hard_drop={int(np.sum(hard_drop))}/{p}, "
        f"elapsed={time.perf_counter() - t0:.1f}s."
    )
    return stats, np.nan_to_num(best_oof, nan=float(np.mean(y_arr))).astype(np.float32), agg_all


def _select_smallest_within_one_se(history: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [h for h in history if h.get("active_features_end", h.get("active_features"))]
    if not valid:
        return {}
    best = max(valid, key=lambda h: float(h.get("J_final", -np.inf)))
    best_score = float(best.get("J_final", -np.inf))
    one_se = float(best.get("J_se", 0.0))
    floor = best_score - max(float(LGBM_SELECTION_SE_MULT) * one_se, 0.0)
    close = [h for h in valid if float(h.get("J_final", -np.inf)) >= floor]
    if not close:
        close = [best]
    chosen = min(close, key=lambda h: (int(h.get("n_features_end", h.get("n_features", 10**9))), -float(h.get("J_final", -np.inf))))
    out = dict(chosen)
    out["selection_best_J"] = best_score
    out["selection_one_se"] = one_se
    out["selection_se_mult"] = float(LGBM_SELECTION_SE_MULT)
    out["selection_floor"] = floor
    out["selection_policy"] = "smallest_within_fractional_se"
    return out


def _iterative_feature_prune(
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray,
    initial_features: list[str],
    *,
    classifier: bool,
    groups: Any = None,
    returns: Any = None,
    metric_y: np.ndarray | None = None,
    random_state: int,
    objective_mode: str | None = "train_base",
) -> tuple[list[str], list[dict[str, Any]], pd.DataFrame, np.ndarray, dict[str, Any]]:
    t0 = time.perf_counter()
    active = list(initial_features)
    y_metric = np.asarray(metric_y if metric_y is not None else y)
    history: list[dict[str, Any]] = []
    last_stats = pd.DataFrame()
    last_oof = np.full(len(y), float(np.mean(y)), dtype=np.float32)
    last_metrics: dict[str, Any] = {}
    for round_id in range(1, LGBM_MAX_ROUNDS + 1):
        if len(active) <= LGBM_MIN_FEATURES:
            tprint(
                f"LGBM prune stopped before round {round_id}: "
                f"active_features={len(active)} <= min_features={LGBM_MIN_FEATURES}."
            )
            break
        round_t0 = time.perf_counter()
        tprint(
            f"LGBM prune round {round_id} started: evaluating {len(active)} "
            f"features, max_rounds={LGBM_MAX_ROUNDS}."
        )
        stats, oof, metrics = _lgbm_stability_selection_pass(
            X,
            y,
            sample_weight,
            active,
            classifier=classifier,
            groups=groups,
            returns=returns,
            metric_y=y_metric,
            random_state=random_state + round_id * 1009,
            seeds=[random_state],
            objective_mode=objective_mode,
        )
        rec = {
            "round": int(round_id),
            "n_features": int(len(active)),
            "active_features": list(active),
            "J_final": float(metrics.get("J_final", metrics.get("J_meta", -999.0))),
            "J_mean": float(metrics.get("J_mean", metrics.get("J_final", -999.0))),
            "J_std": float(metrics.get("J_std", 0.0)),
            "J_se": float(metrics.get("J_se", 0.0)),
            "J_median": float(metrics.get("J_median", metrics.get("J_final", -999.0))),
            "J_iqr": float(metrics.get("J_iqr", 0.0)),
            "J_robust": float(metrics.get("J_robust", metrics.get("J_final", -999.0))),
            "lift20": float(metrics.get("lift20", np.nan)),
            "precision20_norm": float(metrics.get("precision20_norm", np.nan)),
            "rank_bucket_monotonicity": float(metrics.get("rank_bucket_monotonicity", np.nan)),
            "ndcg_at_20": float(metrics.get("ndcg_at_20", np.nan)),
        }
        last_stats = stats.copy()
        last_oof = oof.copy()
        last_metrics = dict(metrics)
        hard_kept = stats.loc[~stats["hard_drop"].astype(bool)].copy()
        if len(hard_kept) == 0:
            tprint("LGBM prune stopped: hard filters removed every active feature.")
            break
        drop_frac = _drop_fraction(len(active))
        keep_n = max(LGBM_MIN_FEATURES, int(np.ceil(len(active) * (1.0 - drop_frac))))
        keep_n = min(keep_n, len(hard_kept), len(active))
        next_active = hard_kept.sort_values("feature_score", ascending=False)["feature"].astype(str).head(keep_n).tolist()
        next_active = [f for f in active if f in set(next_active)]
        rec.update(
            {
                "n_features_start": int(len(active)),
                "active_features_start": list(active),
                "n_features": int(len(next_active)),
                "n_features_end": int(len(next_active)),
                "active_features": list(next_active),
                "active_features_end": list(next_active),
            }
        )
        history.append(rec)
        tprint(
            f"LGBM prune round {round_id}: J={rec['J_final']:.4f}, "
            f"SE={rec['J_se']:.4f}, {len(active)} -> {len(next_active)}, "
            f"elapsed={time.perf_counter() - round_t0:.1f}s."
        )
        if len(next_active) >= len(active) or len(next_active) <= LGBM_MIN_FEATURES:
            active = next_active
            break
        active = next_active
        gc.collect()
    chosen = _select_smallest_within_one_se(history)
    selected = list(chosen.get("active_features_end", chosen.get("active_features", active))) if chosen else active
    if len(selected) > LGBM_SELECTED_FEATURES_MAX > 0 and not last_stats.empty:
        selected_set = set(selected)
        selected = (
            last_stats[last_stats["feature"].astype(str).isin(selected_set)]
            .sort_values("feature_score", ascending=False)["feature"]
            .astype(str)
            .head(LGBM_SELECTED_FEATURES_MAX)
            .tolist()
        )
    tprint(
        "LGBM prune complete: "
        f"initial={len(initial_features)}, selected={len(selected)}, "
        f"rounds={len(history)}, elapsed={time.perf_counter() - t0:.1f}s."
    )
    return selected, history, last_stats, last_oof, last_metrics


def _cross_val_oof_lgbm(
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray,
    features: list[str],
    *,
    classifier: bool,
    params: dict[str, Any],
    groups: Any = None,
    returns: Any = None,
    metric_y: np.ndarray | None = None,
    random_state: int,
    n_splits: int = LGBM_CV_SPLITS,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    t0 = time.perf_counter()
    Xf = X[features].reset_index(drop=True)
    y_arr = np.asarray(y)
    y_metric = np.asarray(metric_y if metric_y is not None else y_arr)
    ret_arr = _as_returns(y_metric, returns)
    splitter, y_split = _splitter(y_metric, classifier, random_state, n_splits=n_splits)
    oof = np.full(len(y_arr), np.nan, dtype=np.float32)
    metrics: list[dict[str, float]] = []
    tprint(
        "LGBM OOF CV started: "
        f"rows={len(y_arr)}, features={len(features)}, splits={n_splits}."
    )
    for fold_i, (tr, va) in enumerate(splitter.split(np.zeros(len(y_split)), y_split), start=1):
        fold_t0 = time.perf_counter()
        tprint(
            f"LGBM OOF CV fold {fold_i}/{n_splits} started: "
            f"train_rows={len(tr)}, valid_rows={len(va)}."
        )
        fold_params = dict(params)
        fold_params["random_state"] = int(random_state + fold_i * 1009)
        model = _fit_lgbm_model(
            Xf.iloc[tr].reset_index(drop=True),
            y_arr[tr],
            sample_weight[tr],
            classifier=classifier,
            params=fold_params,
            X_valid=Xf.iloc[va].reset_index(drop=True),
            y_valid=y_arr[va],
            early_stopping_rounds=50,
        )
        pred = _predict_lgbm_raw(model, Xf.iloc[va].reset_index(drop=True), "classifier" if classifier else "regressor")
        oof[va] = pred
        metrics.append(_metric_pack(y_metric[va], pred, classifier=classifier, groups=_groups_take(groups, va), returns=ret_arr[va]))
        tprint(
            f"LGBM OOF CV fold {fold_i}/{n_splits} complete: "
            f"elapsed={time.perf_counter() - fold_t0:.1f}s."
        )
    fill = float(np.nanmean(oof)) if np.isfinite(oof).any() else float(np.mean(y_arr))
    tprint(f"LGBM OOF CV complete in {time.perf_counter() - t0:.1f}s.")
    return np.nan_to_num(oof, nan=fill).astype(np.float32), metrics


def _cross_val_oof_lgbm_with_meta_features(
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray,
    features: list[str],
    *,
    classifier: bool,
    params: dict[str, Any],
    groups: Any = None,
    returns: Any = None,
    metric_y: np.ndarray | None = None,
    random_state: int,
    n_splits: int = LGBM_CV_SPLITS,
) -> tuple[np.ndarray, list[dict[str, float]], pd.DataFrame]:
    t0 = time.perf_counter()
    Xf = X[features].reset_index(drop=True)
    y_arr = np.asarray(y)
    y_metric = np.asarray(metric_y if metric_y is not None else y_arr)
    ret_arr = _as_returns(y_metric, returns)
    splitter, y_split = _splitter(y_metric, classifier, random_state, n_splits=n_splits)
    oof = np.full(len(y_arr), np.nan, dtype=np.float32)
    metrics: list[dict[str, float]] = []
    meta_features = pd.DataFrame(index=np.arange(len(y_arr)), columns=LGBM_META_FEATURE_NAMES, dtype=np.float32)
    mode = "classifier" if classifier else "regressor"
    tprint(
        "LGBM final OOF/meta CV started: "
        f"rows={len(y_arr)}, features={len(features)}, splits={n_splits}."
    )
    for fold_i, (tr, va) in enumerate(splitter.split(np.zeros(len(y_split)), y_split), start=1):
        fold_t0 = time.perf_counter()
        tprint(
            f"LGBM final OOF/meta fold {fold_i}/{n_splits} started: "
            f"train_rows={len(tr)}, valid_rows={len(va)}."
        )
        fold_params = dict(params)
        fold_params["random_state"] = int(random_state + fold_i * 1009)
        X_tr = Xf.iloc[tr].reset_index(drop=True)
        X_va = Xf.iloc[va].reset_index(drop=True)
        model = _fit_lgbm_model(
            X_tr,
            y_arr[tr],
            sample_weight[tr],
            classifier=classifier,
            params=fold_params,
            X_valid=X_va,
            y_valid=y_arr[va],
            early_stopping_rounds=50,
        )
        pred = _predict_lgbm_raw(model, X_va, mode)
        oof[va] = pred
        metrics.append(_metric_pack(y_metric[va], pred, classifier=classifier, groups=_groups_take(groups, va), returns=ret_arr[va]))
        fold_meta = _lgbm_meta_features_from_models([model], X_va, mode=mode, rank_bin_stats=None)
        gain_imp, split_imp = _feature_importances(model, len(features))
        drift_features = _top_cumulative_importance_feature_names(
            features,
            gain_imp,
            split_imp,
            cumulative_fraction=0.50,
        )
        fold_drift_reference = _fit_feature_drift_reference(
            X_tr,
            drift_features,
        )
        _append_feature_drift_meta_features(
            fold_meta,
            X_va,
            fold_drift_reference,
        )
        meta_features.iloc[va] = fold_meta.to_numpy(dtype=np.float32)
        tprint(
            f"LGBM final OOF/meta fold {fold_i}/{n_splits} complete: "
            f"elapsed={time.perf_counter() - fold_t0:.1f}s."
        )
    fill = float(np.nanmean(oof)) if np.isfinite(oof).any() else float(np.mean(y_arr))
    oof = np.nan_to_num(oof, nan=fill).astype(np.float32)
    full_rank = _safe_rank_pct(oof)
    meta_features["lgbm_prob"] = oof
    meta_features["rank_pct"] = full_rank
    meta_features["score_margin_top10"] = _score_margin(oof, 0.10)
    meta_features["score_margin_top20"] = _score_margin(oof, 0.20)
    meta_features["score_margin_top30"] = _score_margin(oof, 0.30)
    meta_features["rank_margin_top10"] = (full_rank - 0.90).astype(np.float32)
    meta_features["rank_margin_top20"] = (full_rank - 0.80).astype(np.float32)
    rank_stats = _fit_rank_bin_stats_oof(y_metric, full_rank, classifier=classifier, returns=ret_arr)
    rank_cols = ["rank_bin_win_rate_oof", "rank_bin_lift_oof", "rank_bin_net_ret_oof", "rank_bin_se_oof"]
    rank_frames: dict[str, np.ndarray] = {}
    _append_rank_bin_oof_features(rank_frames, full_rank, rank_stats)
    for col in rank_cols:
        meta_features[col] = rank_frames[col]
    tprint(f"LGBM final OOF/meta CV complete in {time.perf_counter() - t0:.1f}s.")
    return oof, metrics, meta_features.reindex(columns=LGBM_META_FEATURE_NAMES, fill_value=0.0).astype(np.float32)


def _oof_distilled_sample_weights_lgbm(
    X: pd.DataFrame,
    y: np.ndarray,
    base_weight: np.ndarray,
    features: list[str],
    *,
    classifier: bool,
    params: dict[str, Any],
    groups: Any = None,
    returns: Any = None,
    metric_y: np.ndarray | None = None,
    random_state: int,
    passes: int,
    label: str,
    objective_mode: str | None = "train_base",
) -> tuple[np.ndarray, np.ndarray]:
    base, _ = _normalize_weights(base_weight)
    current = base.copy()
    y_metric = np.asarray(metric_y if metric_y is not None else y)
    prev_oof: np.ndarray | None = None
    last_oof = np.full(len(y), float(np.mean(y)), dtype=np.float32)
    if int(passes) <= 0:
        tprint(f"LGBM OOF distilled weights {label}: passes=0, returning base weights without fitting.")
        return current.astype(np.float32), last_oof.astype(np.float32)
    for pass_i in range(1, int(passes) + 1):
        start = time.perf_counter()
        last_oof, _fold_metrics = _cross_val_oof_lgbm(
            X,
            y,
            current,
            features,
            classifier=classifier,
            params=params,
            groups=groups,
            returns=returns,
            metric_y=y_metric,
            random_state=random_state + pass_i * 7919,
        )
        distill = _compute_weight_distillation(y_metric, last_oof, prev_oof, is_classifier=classifier, include_false_positive_focus=False)
        fp_weight = _false_positive_avoidance_weight(
            y_metric,
            last_oof,
            classifier=classifier,
            top_frac=_target_top_fraction(objective_mode),
        )
        current, ess = _normalize_weights(base * distill * fp_weight)
        prev_oof = last_oof.copy()
        tprint(
            f"LGBM OOF distilled weights {label} pass {pass_i}/{int(passes)} "
            f"in {time.perf_counter() - start:.1f}s, ess={ess:.1f}."
        )
    return current.astype(np.float32), last_oof.astype(np.float32)


def _default_hpo_params(seed: int, classifier: bool) -> dict[str, Any]:
    return _base_lgbm_params(
        seed,
        classifier=classifier,
        overrides={
            "n_estimators": 1200,
            "learning_rate": LGBM_HPO_LEARNING_RATE,
            "max_depth": 4,
            "num_leaves": 16,
            "min_child_samples": 300,
            "min_child_weight": 40.0,
            "min_split_gain": 0.01,
            "reg_alpha": 1.0,
            "reg_lambda": 8.0,
            "subsample": 0.75,
            "colsample_bytree": 0.70,
        },
    )


def _run_lgbm_hpo(
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray,
    features: list[str],
    *,
    classifier: bool,
    groups: Any = None,
    returns: Any = None,
    metric_y: np.ndarray | None = None,
    random_state: int,
    max_trials: int = LGBM_HPO_TRIALS,
    patience: int = LGBM_HPO_EARLY_STOP_PATIENCE,
    objective_mode: str | None = "train_base",
) -> tuple[dict[str, Any], dict[str, Any]]:
    t0 = time.perf_counter()
    try:
        import optuna
        from optuna.pruners import MedianPruner
        from optuna.trial import TrialState
    except Exception as exc:
        tprint(f"LGBM HPO skipped, Optuna unavailable ({exc}).")
        params = _default_hpo_params(random_state, classifier)
        return params, {"hpo_available": False, "hpo_best_value": np.nan, "hpo_objective_mode": _normalize_objective_mode(objective_mode)}
    y_arr = np.asarray(y)
    y_metric = np.asarray(metric_y if metric_y is not None else y_arr)
    hpo_frac = float(np.clip(LGBM_HPO_ROW_SUBSAMPLE_FRAC, 0.01, 1.0))
    hpo_cap = int(len(y_arr))
    if hpo_frac < 0.999:
        hpo_cap = min(hpo_cap, max(1, int(np.ceil(hpo_frac * len(y_arr)))))
    if LGBM_HPO_MAX_ROWS > 0:
        hpo_cap = min(hpo_cap, int(LGBM_HPO_MAX_ROWS))
    if len(y_arr) > hpo_cap:
        idx = _stratified_spread_subsample_indices(y_metric, hpo_cap, random_state + 71, classifier)
    else:
        idx = np.arange(len(y_arr), dtype=np.int32)
    X_sub = X.iloc[idx][features].reset_index(drop=True)
    y_sub = y_arr[idx]
    y_metric_sub = y_metric[idx]
    sw_sub = sample_weight[idx]
    ret_sub = _as_returns(y_metric, returns)[idx]
    groups_sub = _groups_take(groups, idx)
    splitter, y_split = _interleaved_spread_splitter(
        y_metric_sub,
        classifier,
        n_splits=max(2, int(LGBM_CV_SPLITS)),
    )
    best_seen = {"value": -np.inf, "trial": -1}
    tprint(
        "LGBM HPO started: "
        f"rows={len(y_sub)}/{len(y_arr)}, features={len(features)}, trials={int(max_trials)}, "
        f"row_subsample_frac={hpo_frac:.3f}, max_rows={int(LGBM_HPO_MAX_ROWS)}, "
        f"patience={int(patience)}, objective={_normalize_objective_mode(objective_mode)}, "
        f"param_set={LGBM_HPO_PARAM_SET}, "
        "fold_mode=interleaved_spread, cegb_enabled=False, "
        f"path_smooth_max={float(LGBM_HPO_PATH_SMOOTH_MAX):.3g}, "
        f"final_min_estimators={int(LGBM_HPO_FINAL_MIN_ESTIMATORS)}."
    )

    def objective(trial: Any) -> float:
        trial_t0 = time.perf_counter()
        depth = trial.suggest_int("max_depth", 3, 6)
        subsample = trial.suggest_float("subsample", 0.60, 0.80)
        path_smooth = (
            trial.suggest_float("path_smooth", 0.0, LGBM_HPO_PATH_SMOOTH_MAX)
            if LGBM_HPO_PATH_SMOOTH_MAX > 0.0
            else 0.0
        )
        bynode_fraction = (
            trial.suggest_float("feature_fraction_bynode", 0.5, 1.0)
            if LGBM_HPO_PARAM_SET == "full"
            else 1.0
        )
        max_delta_step = (
            trial.suggest_float("max_delta_step", 0.0, 5.0)
            if LGBM_HPO_PARAM_SET == "full"
            else 0.0
        )
        params = _base_lgbm_params(
            random_state + trial.number * 101,
            classifier=classifier,
            overrides={
                "boosting_type": "gbdt",
                "n_estimators": 1600,
                "learning_rate": LGBM_HPO_LEARNING_RATE,
                "max_depth": depth,
                "num_leaves": int(2 ** depth),
                "max_bin": 63,
                "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 5.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 2.0, 100.0, log=True),
                "min_child_samples": max(2, int(trial.suggest_float("min_child_samples_pct", 0.02, 0.07) * len(y_sub))),
                "min_child_weight": trial.suggest_float("min_child_weight", 20.0, 70.0),
                "min_data_in_bin": trial.suggest_int("min_data_in_bin", 5, 100),
                "min_split_gain": trial.suggest_float("min_split_gain", 1e-4, 3e-2, log=True),
                "subsample": subsample,
                "subsample_freq": 1,
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.40, 0.80),
                "feature_fraction_bynode": bynode_fraction,
                "extra_trees": False,
                "path_smooth": path_smooth,
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.25, 4.0, log=True),
                "max_delta_step": max_delta_step,
            },
        )
        fold_metrics: list[dict[str, float]] = []
        fold_best_iterations: list[int] = []
        trial_gain_focus_runs: list[np.ndarray] = []
        trial_split_focus_runs: list[np.ndarray] = []
        trial_topk_contrib_available_flags: list[float] = []
        trial_topk_effective_rows: list[float] = []
        for step, (tr, va) in enumerate(splitter.split(np.zeros(len(y_split)), y_split)):
            fold_t0 = time.perf_counter()
            tprint(
                f"LGBM HPO trial {trial.number} fold {step + 1}/3 started: "
                f"train_rows={len(tr)}, valid_rows={len(va)}, depth={depth}."
            )
            model = _fit_lgbm_model(
                X_sub.iloc[tr].reset_index(drop=True),
                y_sub[tr],
                sw_sub[tr],
                classifier=classifier,
                params=params,
                X_valid=X_sub.iloc[va].reset_index(drop=True),
                y_valid=y_sub[va],
                early_stopping_rounds=75,
            )
            best_iter = _model_num_iterations(model)
            if best_iter > 0:
                fold_best_iterations.append(int(best_iter))
            mode_name = "classifier" if classifier else "regressor"
            X_tr_fold = X_sub.iloc[tr].reset_index(drop=True)
            X_va_fold = X_sub.iloc[va].reset_index(drop=True)
            pred = _predict_lgbm_raw(model, X_va_fold, mode_name)
            pred_train = _predict_lgbm_raw(model, X_tr_fold, mode_name)
            train_metrics = _metric_pack(
                y_metric_sub[tr],
                pred_train,
                classifier=classifier,
                groups=_groups_take(groups_sub, tr),
                returns=ret_sub[tr],
            )
            valid_metrics = _metric_pack(
                y_metric_sub[va],
                pred,
                classifier=classifier,
                groups=_groups_take(groups_sub, va),
                returns=ret_sub[va],
            )
            fold_metric = _apply_overfit_gap_penalty(
                train_metrics,
                valid_metrics,
                objective_mode=objective_mode,
            )
            fold_metrics.append(fold_metric)
            gain, split = _feature_importances(model, len(features))
            gain_focus, split_focus, focus_diag = _topk_focused_gain_split_importance(
                model,
                X_va_fold,
                pred,
                gain,
                split,
                objective_mode=objective_mode,
            )
            trial_gain_focus_runs.append(gain_focus)
            trial_split_focus_runs.append(split_focus)
            trial_topk_contrib_available_flags.append(
                float(focus_diag["topk_contrib_available"])
            )
            trial_topk_effective_rows.append(
                float(focus_diag["topk_focus_rows_effective"])
            )
            agg_step = _aggregate_j(fold_metrics, objective_mode=objective_mode)
            value_step_raw = float(agg_step.get("J_final", -999.0))
            if len(trial_gain_focus_runs) >= 2:
                step_instability = _combined_gain_split_instability(
                    trial_gain_focus_runs,
                    trial_split_focus_runs,
                )
                step_penalty = (
                    float(LGBM_IMPORTANCE_INSTABILITY_PENALTY)
                    * float(step_instability["importance_instability"])
                    if LGBM_IMPORTANCE_INSTABILITY_ENABLE
                    else 0.0
                )
                value_step = float(value_step_raw - step_penalty)
            else:
                value_step = value_step_raw
            trial.report(value_step, step)
            tprint(
                f"LGBM HPO trial {trial.number} fold {step + 1}/3 complete: "
                f"partial_J={value_step:.4f}, elapsed={time.perf_counter() - fold_t0:.1f}s."
            )
            if trial.should_prune():
                tprint(
                    f"LGBM HPO trial {trial.number} pruned at fold {step + 1}/3 "
                    f"(partial_J={value_step:.4f})."
                )
                raise optuna.TrialPruned()
        agg = _aggregate_j(fold_metrics, objective_mode=objective_mode)
        instability_info = _combined_gain_split_instability(
            trial_gain_focus_runs,
            trial_split_focus_runs,
        )
        importance_instability = float(instability_info["importance_instability"])
        importance_penalty = (
            float(LGBM_IMPORTANCE_INSTABILITY_PENALTY) * importance_instability
            if LGBM_IMPORTANCE_INSTABILITY_ENABLE
            else 0.0
        )
        raw_value = float(agg.get("J_final", -999.0))
        penalized_value = raw_value - importance_penalty
        trial.set_user_attr("J_final_pre_importance_instability_penalty", raw_value)
        trial.set_user_attr("importance_instability", importance_instability)
        trial.set_user_attr("gain_instability", float(instability_info["gain_instability"]))
        trial.set_user_attr("split_instability", float(instability_info["split_instability"]))
        trial.set_user_attr(
            "gain_instability_raw_cv",
            float(instability_info["gain_instability_raw_cv"]),
        )
        trial.set_user_attr(
            "split_instability_raw_cv",
            float(instability_info["split_instability_raw_cv"]),
        )
        trial.set_user_attr("importance_instability_penalty", float(importance_penalty))
        trial.set_user_attr(
            "topk_contrib_available_rate",
            (
                float(np.mean(trial_topk_contrib_available_flags))
                if trial_topk_contrib_available_flags
                else 0.0
            ),
        )
        trial.set_user_attr(
            "topk_focus_effective_rows_mean",
            (
                float(np.mean(trial_topk_effective_rows))
                if trial_topk_effective_rows
                else 0.0
            ),
        )
        for key, value in agg.items():
            try:
                trial.set_user_attr(key, float(value))
            except Exception:
                pass
        trial.set_user_attr("J_final", float(penalized_value))
        trial.set_user_attr("selected_objective", float(penalized_value))
        mode_obj = _normalize_objective_mode(objective_mode)
        trial.set_user_attr(
            "J_meta" if mode_obj == "train_meta" else "J_base",
            float(penalized_value),
        )
        if fold_best_iterations:
            trial.set_user_attr("hpo_fold_best_iterations", [int(v) for v in fold_best_iterations])
            trial.set_user_attr("hpo_best_iteration_median", float(np.median(fold_best_iterations)))
            trial.set_user_attr("hpo_best_iteration_p75", float(np.percentile(fold_best_iterations, 75)))
        tprint(
            f"LGBM HPO trial {trial.number} complete: "
            f"J={float(penalized_value):.4f}, raw_J={raw_value:.4f}, "
            f"elapsed={time.perf_counter() - trial_t0:.1f}s."
        )
        return float(penalized_value)

    def early_stop_callback(study: Any, trial: Any) -> None:
        if trial.state != TrialState.COMPLETE or trial.value is None:
            return
        if float(trial.value) > float(best_seen["value"]):
            best_seen["value"] = float(trial.value)
            best_seen["trial"] = int(trial.number)
        elif int(trial.number) - int(best_seen["trial"]) >= int(patience):
            study.stop()

    study = optuna.create_study(direction="maximize", pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=1, interval_steps=1))
    study.optimize(objective, n_trials=max(0, int(max_trials)), callbacks=[early_stop_callback], n_jobs=1, show_progress_bar=False)
    complete = [t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None]
    if not complete:
        params = _default_hpo_params(random_state, classifier)
        return params, {"hpo_available": True, "hpo_completed_trials": 0, "hpo_best_value": np.nan, "hpo_objective_mode": _normalize_objective_mode(objective_mode)}
    best = study.best_trial
    best_iterations = [int(v) for v in best.user_attrs.get("hpo_fold_best_iterations", []) if int(v) > 0]
    raw_final_n_estimators = int(np.percentile(best_iterations, 75)) if best_iterations else 1600
    final_n_estimators = max(int(LGBM_HPO_FINAL_MIN_ESTIMATORS), raw_final_n_estimators)
    if raw_final_n_estimators < int(LGBM_HPO_FINAL_MIN_ESTIMATORS):
        tprint(
            "LGBM HPO final estimator guard: "
            f"best_trial={best.number} requested n_estimators={raw_final_n_estimators}; "
            f"using min={int(LGBM_HPO_FINAL_MIN_ESTIMATORS)}."
        )
    depth = int(best.params.get("max_depth", 4))
    best_params = _base_lgbm_params(
        random_state + 191,
        classifier=classifier,
        overrides={
            "boosting_type": "gbdt",
            "n_estimators": final_n_estimators,
            "learning_rate": LGBM_FINAL_LEARNING_RATE,
            "max_depth": depth,
            "num_leaves": int(2 ** depth),
            "max_bin": 63,
            "reg_alpha": float(best.params.get("reg_alpha", 1.0)),
            "reg_lambda": float(best.params.get("reg_lambda", 8.0)),
            "min_child_samples": max(2, int(float(best.params.get("min_child_samples_pct", 0.03)) * max(1, len(y)))),
            "min_child_weight": float(best.params.get("min_child_weight", 40.0)),
            "min_data_in_bin": int(best.params.get("min_data_in_bin", 20)),
            "min_split_gain": float(best.params.get("min_split_gain", 0.01)),
            "subsample": float(best.params.get("subsample", 0.75)),
            "subsample_freq": 1,
            "colsample_bytree": float(best.params.get("colsample_bytree", 0.70)),
            "extra_trees": False,
            "path_smooth": min(
                float(best.params.get("path_smooth", 0.0)),
                float(LGBM_HPO_PATH_SMOOTH_MAX),
            ),
            "scale_pos_weight": float(best.params.get("scale_pos_weight", 1.0)),
        },
    )
    if LGBM_HPO_PARAM_SET == "full":
        best_params["feature_fraction_bynode"] = float(
            best.params.get("feature_fraction_bynode", 1.0)
        )
        best_params["max_delta_step"] = float(best.params.get("max_delta_step", 0.0))
    attrs = dict(best.user_attrs)
    attrs.update(
        {
            "hpo_available": True,
            "hpo_completed_trials": int(len(complete)),
            "hpo_best_trial": int(best.number),
            "hpo_best_value": float(best.value),
            "hpo_best_params": dict(best_params),
            "hpo_objective_mode": _normalize_objective_mode(objective_mode),
            "hpo_param_set": LGBM_HPO_PARAM_SET,
            "hpo_final_n_estimators": int(final_n_estimators),
        }
    )
    tprint(
        f"LGBM HPO complete: best_trial={best.number}, "
        f"value={float(best.value):.4f}, completed={len(complete)}, "
        f"elapsed={time.perf_counter() - t0:.1f}s, "
        f"params={json.dumps(best_params, sort_keys=True)}"
    )
    return best_params, attrs


def train_lgbm_stability_candidate(
    X: Any,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    random_state: int = 42,
    mode: str = "classifier",
    timestamps: Any = None,
    assets: Any = None,
    returns: Any = None,
    hard_labels: Optional[np.ndarray] = None,
    hpo_objective_mode: str = "train_base",
    preset_feature_names: Optional[list[str]] = None,
    preset_best_params: Optional[dict[str, Any]] = None,
    preset_source: str | None = None,
) -> Optional[dict[str, Any]]:
    objective_mode = _normalize_objective_mode(hpo_objective_mode)
    distill_passes = _distillation_passes_for_objective(objective_mode)
    tprint(f"LGBM stability candidate training started (objective={objective_mode}).")
    t0 = time.perf_counter()
    classifier = mode == "classifier"
    X_raw_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    X_raw_df.columns = [str(c) for c in X_raw_df.columns]
    X_df = _frame(X)
    y_arr = _coerce_target(y, classifier)
    y_metric = _coerce_target(hard_labels, classifier) if hard_labels is not None else y_arr
    _validate_input_lengths(X_df, y_arr, sample_weight=sample_weight, timestamps=timestamps, assets=assets, returns=returns)
    if len(y_metric) != len(y_arr):
        raise ValueError(f"hard_labels length {len(y_metric)} != target length {len(y_arr)}")
    ret_arr = _as_returns(y_metric, returns)
    n = len(y_arr)
    if n < 200 or X_df.shape[1] < 2:
        tprint("LGBM stability candidate skipped: not enough rows or features.")
        return None
    oi_present_mask, oi_diagnostics, oi_metadata_cols = _feature_selection_oi_present_mask(X, n)
    if oi_metadata_cols:
        X_df = X_df.drop(columns=[c for c in oi_metadata_cols if c in X_df.columns], errors="ignore")
        X_raw_df = X_raw_df.drop(columns=[c for c in oi_metadata_cols if c in X_raw_df.columns], errors="ignore")
    coverage_exempt_features = (
        {
            str(c)
            for c in X_df.columns
            if _is_lgbm_model_derived_meta_feature(str(c))
        }
        if objective_mode == "train_meta"
        else set()
    )
    coverage_survivors, coverage_diagnostics = _recent_feature_coverage_survivors(
        X_raw_df.reindex(columns=list(X_df.columns)),
        timestamps,
        exempt_features=coverage_exempt_features,
    )
    if len(coverage_survivors) < len(X_df.columns):
        removed_preview = coverage_diagnostics.get("feature_recent_removed_lowest", [])[:10]
        tprint(
            "LGBM recent complete-case feature availability filter "
            "(in-memory post-upstream feature set): "
            f"{len(X_df.columns)} -> {len(coverage_survivors)} features "
            f"(target>={LGBM_FEATURE_RECENT_MIN_COVERAGE:.0%}, "
            f"joint={float(coverage_diagnostics.get('feature_recent_joint_coverage', float('nan'))):.1%}, "
            f"rows={int(coverage_diagnostics.get('feature_recent_row_count', 0))}, "
            f"exempt_model_derived={int(coverage_diagnostics.get('feature_recent_exempt_model_derived_count', 0))}, "
            f"greedy_removed={int(coverage_diagnostics.get('feature_recent_removed_iterative_count', 0))}, "
            f"stopped_no_gain={bool(coverage_diagnostics.get('feature_recent_stopped_no_gain', False))}); "
            f"lowest removed={removed_preview}"
        )
        X_df = X_df.reindex(columns=coverage_survivors)
    else:
        tprint(
            "LGBM recent complete-case feature availability filter "
            "(in-memory post-upstream feature set): "
            f"kept all {X_df.shape[1]} features "
            f"(target>={LGBM_FEATURE_RECENT_MIN_COVERAGE:.0%}, "
            f"joint={float(coverage_diagnostics.get('feature_recent_joint_coverage', float('nan'))):.1%}, "
            f"rows={int(coverage_diagnostics.get('feature_recent_row_count', 0))}, "
            f"exempt_model_derived={int(coverage_diagnostics.get('feature_recent_exempt_model_derived_count', 0))}, "
            f"window={coverage_diagnostics.get('feature_recent_window_start', 'all')} -> "
            f"{coverage_diagnostics.get('feature_recent_window_end', 'all')}, "
            f"price_sources={coverage_diagnostics.get('feature_recent_price_sources', [])}, "
            f"stopped_no_gain={bool(coverage_diagnostics.get('feature_recent_stopped_no_gain', False))})"
        )
    if X_df.shape[1] < 2:
        tprint(
            "LGBM stability candidate skipped: fewer than 2 features meet recent coverage threshold."
        )
        return None
    tprint(
        "LGBM candidate input: "
        f"rows={n}, features={X_df.shape[1]}, classifier={classifier}, "
        f"sample_weight={'yes' if sample_weight is not None else 'no'}, "
        f"returns={'yes' if returns is not None else 'no'}."
    )
    sw = np.ones(n, dtype=np.float32) if sample_weight is None else np.asarray(sample_weight, dtype=np.float32)
    sw, _ = _normalize_weights(sw)
    stage_indices = _stage_partition_indices(y_metric, timestamps=timestamps, assets=assets, random_state=random_state + 701)
    stage_indices = _subsample_stage_indices(stage_indices, y_metric, max_fraction=LGBM_ROW_SUBSAMPLE_FRAC, random_state=random_state + 3701, classifier=classifier)
    if oi_present_mask is not None:
        select_idx = np.asarray(stage_indices.get("lgbm_select", []), dtype=np.int32)
        keep_idx = select_idx[oi_present_mask[select_idx]]
        dropped_idx = np.setdiff1d(select_idx, keep_idx, assume_unique=False).astype(np.int32)
        stage_indices["lgbm_select"] = np.sort(keep_idx.astype(np.int32))
        if len(dropped_idx):
            stage_indices["fit_oof"] = np.asarray(
                sorted(
                    np.unique(
                        np.concatenate(
                            [
                                np.asarray(stage_indices.get("fit_oof", []), dtype=np.int32),
                                dropped_idx,
                            ]
                        )
                    ).tolist()
                ),
                dtype=np.int32,
            )
        oi_diagnostics["feature_selection_oi_absent_rows_excluded"] = int(len(dropped_idx))
        oi_diagnostics["feature_selection_oi_present_rows_after_stage_split"] = int(len(stage_indices["lgbm_select"]))
        tprint(
            "LGBM feature-selection OI filter: "
            f"source={oi_diagnostics.get('feature_selection_oi_filter_source')}, "
            f"select_rows={len(select_idx)}->{len(stage_indices['lgbm_select'])}, "
            f"excluded={len(dropped_idx)}."
        )
    stage_indices = _cap_stage_and_move_unused_to_fit_oof(stage_indices, y_metric, stage_key="lgbm_select", cap=LGBM_RACE_MAX_ROWS, random_state=random_state + 1701, classifier=classifier, spread=True)
    stage_indices = _cap_stage_and_move_unused_to_fit_oof(stage_indices, y_metric, stage_key="hpo", cap=LGBM_HPO_MAX_ROWS, random_state=random_state + 2701, classifier=classifier)
    race_idx = np.asarray(stage_indices["lgbm_select"], dtype=np.int32)
    if len(race_idx) < 200:
        fallback_pool = (
            np.flatnonzero(oi_present_mask).astype(np.int32)
            if oi_present_mask is not None
            else np.arange(n, dtype=np.int32)
        )
        if len(fallback_pool) < 200:
            tprint(
                "LGBM stability candidate skipped: fewer than 200 OI-present rows "
                f"available for feature selection ({len(fallback_pool)})."
            )
            return None
        keep_local = _stratified_spread_subsample_indices(
            y_metric[fallback_pool],
            max_n=min(LGBM_RACE_MAX_ROWS, len(fallback_pool)),
            random_state=random_state + 701,
            classifier=classifier,
        )
        race_idx = np.sort(fallback_pool[keep_local].astype(np.int32))
        stage_indices["lgbm_select"] = race_idx
    X_race = X_df.iloc[race_idx].reset_index(drop=True)
    y_race = y_arr[race_idx]
    y_metric_race = y_metric[race_idx]
    sw_race = sw[race_idx]
    ret_race = ret_arr[race_idx]
    race_groups = _stability_group_bundle(
        len(race_idx),
        timestamps=(np.asarray(timestamps)[race_idx] if timestamps is not None and len(np.asarray(timestamps)) == n else None),
        assets=(np.asarray(assets)[race_idx] if assets is not None and len(np.asarray(assets)) == n else None),
    )
    local_idx = np.arange(len(y_race), dtype=np.int32)
    split_strata = y_metric_race if classifier else np.clip((pd.Series(y_metric_race).rank(pct=True).to_numpy() * 5).astype(np.int32), 0, 4)
    stratify_arg = None
    if classifier and len(np.unique(split_strata)) > 1:
        _, strata_counts = np.unique(split_strata, return_counts=True)
        if int(np.min(strata_counts)) >= 2:
            stratify_arg = split_strata
    select_local, eval_local = train_test_split(local_idx, test_size=LGBM_RACE_EVAL_FRACTION, stratify=stratify_arg, random_state=random_state + 1701)
    select_local = np.asarray(select_local, dtype=np.int32)
    eval_local = np.asarray(eval_local, dtype=np.int32)
    X_select = X_race.iloc[select_local].reset_index(drop=True)
    y_select = y_race[select_local]
    y_metric_select = y_metric_race[select_local]
    sw_select = sw_race[select_local]
    ret_select = ret_race[select_local]
    select_groups = _groups_take(race_groups, select_local)
    X_eval = X_race.iloc[eval_local].reset_index(drop=True)
    y_eval = y_race[eval_local]
    y_metric_eval = y_metric_race[eval_local]
    ret_eval = ret_race[eval_local]
    eval_groups = _groups_take(race_groups, eval_local)
    tprint(f"LGBM candidate split: select={len(y_select)}, eval={len(y_eval)}, features={X_select.shape[1]}.")
    preset_features = [str(c) for c in (preset_feature_names or []) if str(c).strip()]
    if preset_features:
        missing_preset = [c for c in preset_features if c not in X_df.columns]
        if missing_preset:
            tprint(
                "LGBM candidate preset rejected: "
                f"{len(missing_preset)} selected features are missing; "
                f"examples={missing_preset[:20]}."
            )
            return None
        selected_features = list(preset_features)
        uni_features = list(selected_features)
        relief_features = []
        precluster_features = list(selected_features)
        cluster_features = list(selected_features)
        uni_stats = pd.DataFrame(
            {
                "feature": selected_features,
                "univariate_j": np.full(len(selected_features), np.nan),
                "source": "native_preset",
            }
        )
        relief_stats = pd.DataFrame()
        history = [
            {
                "round": 0,
                "source": "native_preset",
                "selected_features": list(selected_features),
                "preset_source": str(preset_source or ""),
            }
        ]
        feature_stats = pd.DataFrame({"feature": selected_features})
        prune_metrics = {
            "feature_selection_source": "native_preset",
            "feature_pruning_rounds_completed": 0,
        }
        tprint(
            "LGBM candidate using native preset features; "
            f"selected={len(selected_features)}, source={preset_source or 'unknown'}."
        )
    else:
        uni_features, uni_stats = _univariate_directional_filter(X_select, y_metric_select, classifier=classifier, groups=select_groups, returns=ret_select, random_state=random_state + 101, objective_mode=objective_mode)
        tprint(f"LGBM candidate after univariate filter: {len(uni_features)} features.")
        score_map = dict(zip(uni_stats["feature"].astype(str), uni_stats["univariate_j"].astype(float)))
        relief_features, relief_stats = _relief_rescue_filter(
            X_select,
            y_metric_select,
            uni_features,
            classifier=classifier,
            random_state=random_state + 151,
        )
        relief_score_map: dict[str, float] = {}
        if not relief_stats.empty:
            ranked = relief_stats["relief_score"].rank(pct=True).to_numpy(dtype=np.float32)
            relief_score_map = dict(zip(relief_stats["feature"].astype(str), ranked))
            for feature in relief_features:
                score_map[feature] = max(
                    float(score_map.get(feature, 0.0)),
                    float(relief_score_map.get(feature, 0.0)),
                )
        feature_order = {str(c): i for i, c in enumerate(X_select.columns)}
        precluster_features = sorted(
            set(uni_features).union(relief_features),
            key=lambda c: feature_order.get(str(c), len(feature_order)),
        )
        tprint(
            "LGBM candidate after ReliefF rescue: "
            f"univariate={len(uni_features)}, rescued={len(relief_features)}, "
            f"precluster={len(precluster_features)} features."
        )
        cluster_features = _redundancy_cluster_filter(
            X_select,
            precluster_features,
            score_map,
            random_state=random_state + 211,
        )
        tprint(f"LGBM candidate after redundancy clustering: {len(cluster_features)} features.")
        selected_features, history, feature_stats, prune_oof, prune_metrics = _iterative_feature_prune(
            X_select,
            y_select,
            sw_select,
            cluster_features,
            classifier=classifier,
            groups=select_groups,
            returns=ret_select,
            metric_y=y_metric_select,
            random_state=random_state + 307,
            objective_mode=objective_mode,
        )
    if not selected_features:
        tprint("LGBM candidate rejected: no selected features.")
        return None
    tprint(
        "LGBM candidate selected features: "
        f"{len(selected_features)} after {len(history)} prune rounds; "
        f"preview={selected_features[:10]}."
    )
    base_params = dict(preset_best_params or _default_hpo_params(random_state + 401, classifier))
    if preset_best_params:
        tprint("LGBM candidate using native preset best_params; HPO is skipped for base preset candidate scoring.")
    final_weights, _ = _oof_distilled_sample_weights_lgbm(
        X_select,
        y_select,
        sw_select,
        selected_features,
        classifier=classifier,
        params=base_params,
        groups=select_groups,
        returns=ret_select,
        metric_y=y_metric_select,
        random_state=random_state + 409,
        passes=distill_passes,
        label="candidate",
        objective_mode=objective_mode,
    )
    eval_preds: list[np.ndarray] = []
    eval_configs = (
        [dict(base_params)]
        if preset_best_params
        else [
            {"max_depth": 4, "reg_lambda": 5.0},
            {"max_depth": 4, "reg_lambda": 15.0},
            {"max_depth": 5, "reg_lambda": 5.0},
            {"max_depth": 5, "reg_lambda": 15.0},
        ][: min(4, int(LGBM_STABILITY_CONFIGS))]
    )
    for i, cfg in enumerate(eval_configs, start=1):
        fit_t0 = time.perf_counter()
        tprint(
            f"LGBM candidate eval ensemble model {i}/{len(eval_configs)} started: "
            f"rows={len(y_select)}, features={len(selected_features)}, cfg={cfg}."
        )
        params = dict(cfg) if preset_best_params else _base_lgbm_params(random_state + 500 + i, classifier=classifier, overrides=cfg)
        model = _fit_lgbm_model(X_select[selected_features], y_select, final_weights, classifier=classifier, params=params)
        eval_preds.append(_predict_lgbm_raw(model, X_eval[selected_features], mode))
        tprint(
            f"LGBM candidate eval ensemble model {i}/{len(eval_configs)} complete: "
            f"elapsed={time.perf_counter() - fit_t0:.1f}s."
        )
    eval_pred = np.mean(np.vstack(eval_preds), axis=0).astype(np.float32)
    metrics = _metric_pack(y_metric_eval, eval_pred, classifier=classifier, groups=eval_groups, returns=ret_eval)
    metrics.update(_aggregate_j([metrics], objective_mode=objective_mode))
    for key, value in prune_metrics.items():
        if key not in metrics:
            metrics[key] = value
        metrics[f"prune_{key}"] = value
    metrics["feature_count"] = int(len(selected_features))
    metrics["n_univariate_features"] = int(len(uni_features))
    metrics["n_relief_rescued_features"] = int(len(relief_features))
    metrics["n_precluster_features"] = int(len(precluster_features))
    metrics["n_cluster_features"] = int(len(cluster_features))
    metrics["feature_pruning_rounds_completed"] = int(len(history))
    metrics["candidate_elapsed_sec"] = float(time.perf_counter() - t0)
    metrics["hpo_objective_mode"] = objective_mode
    metrics["oof_distillation_passes"] = int(distill_passes)
    if preset_features:
        metrics["feature_selection_source"] = "native_preset"
        metrics["native_preset_source"] = str(preset_source or "")
        metrics["native_preset_hpo_reused"] = bool(preset_best_params)
    metrics.update(oi_diagnostics)
    metrics.update(coverage_diagnostics)
    metrics["feature_selection_sample_policy"] = "stratified_spread_across_ordered_rows"
    metrics["race_n"] = int(len(eval_local))
    metrics["race_select_n"] = int(len(select_local))
    metrics["race_total_n"] = int(len(race_idx))
    oof_full = np.full(n, np.nan, dtype=np.float32)
    oof_race = np.full(len(y_race), np.nan, dtype=np.float32)
    oof_race[eval_local] = eval_pred
    oof_full[race_idx] = oof_race
    fill = float(np.mean(y_arr))
    oof_for_fit = np.nan_to_num(oof_full, nan=fill).astype(np.float32)
    tprint(f"LGBM candidate done: J={metrics.get('J_final', 0.0):.4f}, features={len(selected_features)}, elapsed={metrics['candidate_elapsed_sec']:.1f}s.")
    return {
        "model": None,
        "metrics": metrics,
        "oof_probs": oof_full,
        "oof_for_full_fit": oof_for_fit,
        "selected_feature_names": list(selected_features),
        "selected_features_from_cv": np.asarray([X_df.columns.get_loc(c) for c in selected_features if c in X_df.columns], dtype=np.int32),
        "pruning_history": history,
        "univariate_stats": uni_stats,
        "relief_stats": relief_stats,
        "feature_stats": feature_stats,
        "stage_indices": {k: np.asarray(v, dtype=np.int32) for k, v in stage_indices.items()},
        "full_fit_needed": True,
        "mode": mode,
    }


def fit_lgbm_stability_full_model(
    X: Any,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray],
    selected_features_from_cv: np.ndarray | None = None,
    random_state: int = 42,
    mode: str = "classifier",
    oof_probs: Optional[np.ndarray] = None,
    metrics: Optional[dict[str, Any]] = None,
    pruning_history: Optional[list[dict[str, Any]]] = None,
    selected_feature_names: Optional[list[str]] = None,
    stage_indices: Optional[dict[str, np.ndarray]] = None,
    timestamps: Any = None,
    assets: Any = None,
    returns: Any = None,
    hard_labels: Optional[np.ndarray] = None,
    hpo_trials_override: int | None = None,
    hpo_patience_override: int | None = None,
    hpo_objective_mode: str = "train_base",
    meta_feature_output_path: str | os.PathLike[str] | None = None,
    reference_artifact_dir: str | os.PathLike[str] | None = None,
    preset_best_params: Optional[dict[str, Any]] = None,
    preset_source: str | None = None,
) -> Optional[LGBMStabilityModel]:
    t0 = time.perf_counter()
    objective_mode = _normalize_objective_mode(hpo_objective_mode)
    distill_passes = _distillation_passes_for_objective(objective_mode)
    classifier = mode == "classifier"
    X_df = _frame(X)
    y_arr = _coerce_target(y, classifier)
    y_metric = _coerce_target(hard_labels, classifier) if hard_labels is not None else y_arr
    _validate_input_lengths(X_df, y_arr, sample_weight=sample_weight, timestamps=timestamps, assets=assets, returns=returns)
    if len(y_metric) != len(y_arr):
        raise ValueError(f"hard_labels length {len(y_metric)} != target length {len(y_arr)}")
    ret_arr = _as_returns(y_metric, returns)
    n = len(y_arr)
    if selected_feature_names:
        selected_features = [str(c) for c in selected_feature_names]
    else:
        idx = np.asarray(selected_features_from_cv if selected_features_from_cv is not None else [], dtype=np.int32)
        idx = idx[(idx >= 0) & (idx < X_df.shape[1])]
        selected_features = [str(X_df.columns[i]) for i in idx]
    if not selected_features:
        tprint("LGBM full fit skipped: no selected features.")
        return None
    for col in selected_features:
        if col not in X_df.columns:
            X_df[col] = 0.0
    sw = np.ones(n, dtype=np.float32) if sample_weight is None else np.asarray(sample_weight, dtype=np.float32)
    sw, _ = _normalize_weights(sw)
    if stage_indices is None:
        all_idx = np.arange(n, dtype=np.int32)
        stage_indices = {"lgbm_select": all_idx, "hpo": all_idx, "fit_oof": all_idx}
    hpo_idx = np.asarray(stage_indices.get("hpo", np.arange(n)), dtype=np.int32)
    fit_idx = np.asarray(stage_indices.get("fit_oof", np.arange(n)), dtype=np.int32)
    hpo_idx = hpo_idx[(hpo_idx >= 0) & (hpo_idx < n)]
    fit_idx = fit_idx[(fit_idx >= 0) & (fit_idx < n)]
    if len(hpo_idx) == 0:
        hpo_idx = np.arange(n, dtype=np.int32)
    if len(fit_idx) == 0:
        fit_idx = np.arange(n, dtype=np.int32)
    if LGBM_FINAL_FIT_USE_ALL_ROWS:
        if len(fit_idx) != n or not np.array_equal(np.sort(fit_idx), np.arange(n, dtype=np.int32)):
            tprint(
                "LGBM full fit using all rows: "
                f"fit_oof_partition={len(fit_idx)} -> all_rows={n}."
            )
        fit_idx = np.arange(n, dtype=np.int32)
    if LGBM_FINAL_FIT_MAX_ROWS > 0 and len(fit_idx) > LGBM_FINAL_FIT_MAX_ROWS:
        pre_fit_n = len(fit_idx)
        local = _stratified_subsample_indices(y_metric[fit_idx], LGBM_FINAL_FIT_MAX_ROWS, random_state + 40711, classifier)
        fit_idx = np.sort(fit_idx[local].astype(np.int32))
        tprint(
            f"LGBM full fit row cap: fit_oof {pre_fit_n} -> {len(fit_idx)} "
            f"(cap={LGBM_FINAL_FIT_MAX_ROWS})."
        )
    tprint(
        "LGBM full fit started: "
        f"rows={n}, selected_features={len(selected_features)}, "
        f"hpo_rows={len(hpo_idx)}, fit_rows={len(fit_idx)}, "
        f"objective={objective_mode}."
    )
    stability_groups = _stability_group_bundle(n, timestamps=timestamps, assets=assets)
    hpo_groups = _groups_take(stability_groups, hpo_idx)
    hpo_weights, _ = _normalize_weights(sw[hpo_idx])
    if preset_best_params:
        best_params = dict(preset_best_params)
        hpo_metrics = {
            "hpo_available": True,
            "hpo_reused_from_native_preset": True,
            "hpo_objective_mode": objective_mode,
            "hpo_best_params": dict(best_params),
            "native_preset_source": str(preset_source or ""),
        }
        tprint(
            "LGBM full fit using native preset best_params; "
            f"skipping HPO, source={preset_source or 'unknown'}."
        )
    else:
        tprint("LGBM full fit HPO using base sample weights; pre-HPO OOF distillation skipped.")
        best_params, hpo_metrics = _run_lgbm_hpo(
            X_df.iloc[hpo_idx].reset_index(drop=True),
            y_arr[hpo_idx],
            hpo_weights,
            selected_features,
            classifier=classifier,
            groups=hpo_groups,
            returns=ret_arr[hpo_idx],
            metric_y=y_metric[hpo_idx],
            random_state=random_state + 131,
            max_trials=LGBM_HPO_TRIALS if hpo_trials_override is None else int(hpo_trials_override),
            patience=LGBM_HPO_EARLY_STOP_PATIENCE if hpo_patience_override is None else int(hpo_patience_override),
            objective_mode=objective_mode,
        )
    if distill_passes > 0:
        final_weights, pre_final_oof = _oof_distilled_sample_weights_lgbm(
            X_df,
            y_arr,
            sw,
            selected_features,
            classifier=classifier,
            params=best_params,
            groups=stability_groups,
            returns=ret_arr,
            metric_y=y_metric,
            random_state=random_state + 33107,
            passes=distill_passes,
            label="final",
            objective_mode=objective_mode,
        )
    else:
        final_weights = sw.copy()
        pre_final_oof = np.asarray(oof_probs if oof_probs is not None else np.full(n, float(np.mean(y_arr))), dtype=np.float32)
    model = LGBMStabilityModel(mode=mode)
    model.selected_features = list(selected_features)
    model.best_params = dict(best_params)
    X_fit = X_df.iloc[fit_idx][selected_features].reset_index(drop=True)
    X_all_selected = X_df[selected_features].reset_index(drop=True)
    model.feature_stats_train = _feature_stats_frame(X_fit, selected_features)
    y_fit = y_arr[fit_idx]
    sequential_weight_base = final_weights.copy()
    sequential_weights = final_weights.copy()
    prev_ensemble_pred = pre_final_oof.copy() if len(pre_final_oof) == n else None
    running_ensemble_pred: np.ndarray | None = None
    final_ensemble_ess = float("nan")
    for i in range(LGBM_FINAL_MODEL_COUNT):
        fit_t0 = time.perf_counter()
        w_fit = sequential_weights[fit_idx]
        tprint(
            f"LGBM final model {i + 1}/{LGBM_FINAL_MODEL_COUNT} started: "
            f"rows={len(y_fit)}, features={len(selected_features)}, "
            f"sequential_distill=yes."
        )
        params_i = dict(best_params)
        params_i["random_state"] = int(random_state + 7001 + i * 101)
        fitted = _fit_lgbm_model(X_fit, y_fit, w_fit, classifier=classifier, params=params_i)
        model.models.append(fitted)
        model_pred_all = _predict_lgbm_raw_batched(fitted, X_all_selected, mode)
        if running_ensemble_pred is None:
            running_ensemble_pred = model_pred_all.astype(np.float32)
        else:
            running_ensemble_pred = (
                (running_ensemble_pred.astype(np.float32) * float(i)) + model_pred_all.astype(np.float32)
            ) / float(i + 1)
        distill = _compute_weight_distillation(
            y_metric,
            running_ensemble_pred,
            prev_ensemble_pred,
            is_classifier=classifier,
            include_false_positive_focus=False,
        )
        fp_weight = _false_positive_avoidance_weight(
            y_metric,
            running_ensemble_pred,
            classifier=classifier,
            top_frac=_target_top_fraction(objective_mode),
        )
        sequential_weights, final_ensemble_ess = _normalize_weights(sequential_weight_base * distill * fp_weight)
        prev_ensemble_pred = running_ensemble_pred.copy()
        tprint(
            f"LGBM final model {i + 1}/{LGBM_FINAL_MODEL_COUNT} fitted on "
            f"{len(y_fit)} rows in {time.perf_counter() - fit_t0:.1f}s; "
            f"updated all-row sequential weights ess={final_ensemble_ess:.1f}."
        )
    split_importance_sum = np.zeros(len(selected_features), dtype=np.float64)
    gain_importance_sum = np.zeros(len(selected_features), dtype=np.float64)
    for fitted in model.models:
        gain_imp, split_imp = _feature_importances(fitted, len(selected_features))
        gain_importance_sum += np.asarray(gain_imp, dtype=np.float64)
        split_importance_sum += np.asarray(split_imp, dtype=np.float64)
    final_used_feature_count = int(np.sum(split_importance_sum > 0.0))
    final_gain_feature_count = int(np.sum(gain_importance_sum > 0.0))
    drift_feature_names = _top_cumulative_importance_feature_names(
        selected_features,
        gain_importance_sum,
        split_importance_sum,
        cumulative_fraction=0.50,
    )
    model.drift_reference = _fit_feature_drift_reference(
        X_fit,
        drift_feature_names,
    )
    tprint(
        "LGBM final fitted feature usage: "
        f"selected={len(selected_features)}, split_used={final_used_feature_count}, "
        f"gain_used={final_gain_feature_count}."
    )
    if len(selected_features) and final_used_feature_count <= max(5, int(0.05 * len(selected_features))):
        tprint(
            "WARNING: LGBM final fit used very few selected features; "
            f"selected={len(selected_features)}, split_used={final_used_feature_count}, "
            "check CEGB/path_smooth/min_split_gain and final HPO params."
        )
    final_weights = sequential_weights.astype(np.float32)
    final_oof, final_fold_metrics, meta_oof_features = _cross_val_oof_lgbm_with_meta_features(
        X_df,
        y_arr,
        final_weights,
        selected_features,
        classifier=classifier,
        params=best_params,
        groups=stability_groups,
        returns=ret_arr,
        metric_y=y_metric,
        random_state=random_state + 11701,
    )
    model.oof_probs = final_oof.astype(np.float32)
    model.rank_bin_stats_oof = _fit_rank_bin_stats_oof(y_metric, np.asarray(meta_oof_features["rank_pct"], dtype=np.float32), classifier=classifier, returns=ret_arr)
    model.meta_oof_features = meta_oof_features.reindex(columns=model.meta_feature_names, fill_value=0.0).astype(np.float32)
    final_metrics = _metric_pack(y_metric, final_oof, classifier=classifier, groups=stability_groups, returns=ret_arr)
    final_metrics.update(_aggregate_j(final_fold_metrics, objective_mode=objective_mode))
    candidate_metrics = dict(metrics or {})
    model.metrics = dict(candidate_metrics)
    model.metrics.update(hpo_metrics)
    model.metrics.update(final_metrics)
    model.metrics["feature_count"] = int(len(selected_features))
    model.metrics["selected_features_count"] = int(len(selected_features))
    model.metrics["selected_features_preview"] = list(selected_features[:50])
    model.metrics["final_fit_train_rows"] = int(len(fit_idx))
    model.metrics["final_fit_train_rows_total"] = int(n)
    model.metrics["final_fit_used_all_rows"] = bool(len(fit_idx) == n)
    model.metrics["final_fit_split_used_feature_count"] = int(final_used_feature_count)
    model.metrics["final_fit_gain_used_feature_count"] = int(final_gain_feature_count)
    model.metrics["feature_drift_reference_feature_count"] = int(
        len(model.drift_reference.get("feature_names", []))
    )
    model.metrics["feature_drift_reference_fit_rows"] = int(
        model.drift_reference.get("fit_rows", 0)
    )
    model.metrics["feature_drift_reference_features_preview"] = list(
        model.drift_reference.get("feature_names", [])[:50]
    )
    model.metrics["final_model_count"] = int(LGBM_FINAL_MODEL_COUNT)
    model.metrics["final_ensemble_sequential_distillation"] = True
    model.metrics["oof_distillation_passes"] = int(distill_passes)
    model.metrics["min_oof_distillation_passes"] = int(LGBM_MIN_OOF_DISTILLATION_PASSES)
    model.metrics["meta_min_oof_distillation_passes"] = int(LGBM_META_MIN_OOF_DISTILLATION_PASSES)
    model.metrics["final_ensemble_sequential_weight_ess"] = float(final_ensemble_ess)
    model.metrics["best_params"] = dict(best_params)
    model.metrics["hpo_objective_mode"] = objective_mode
    model.metrics["lgbm_meta_feature_names"] = list(model.meta_feature_names)
    model.metrics["lgbm_meta_feature_count"] = int(len(model.meta_feature_names))
    meta_path = meta_feature_output_path or os.environ.get("EPM_LGBM_META_FEATURE_OUTPUT_PATH")
    _save_lgbm_meta_features(model, meta_path)
    ref_dir = reference_artifact_dir or os.environ.get("EPM_LGBM_REFERENCE_ARTIFACT_DIR")
    if ref_dir is None and meta_path:
        meta_path_obj = Path(meta_path)
        ref_dir = meta_path_obj.parent / "lgbm_reference" / meta_path_obj.stem
    _save_lgbm_reference_artifacts(
        model,
        ref_dir,
        X_reference=X_fit,
        split_importance_sum=split_importance_sum,
        gain_importance_sum=gain_importance_sum,
        y_metric=y_metric,
        returns=ret_arr,
        timestamps=timestamps,
        assets=assets,
        objective_mode=objective_mode,
        mode=mode,
    )
    fit_oof_metrics_for_stage: dict[str, Any] | None = None
    if pre_final_oof is not None and len(pre_final_oof) == n:
        pre_metrics = _metric_pack(y_metric, pre_final_oof, classifier=classifier, groups=stability_groups, returns=ret_arr)
        pre_metrics.update(_aggregate_j([pre_metrics], objective_mode=objective_mode))
        if distill_passes > 0:
            fit_oof_metrics_for_stage = dict(pre_metrics)
        for key, value in pre_metrics.items():
            model.metrics[f"pre_final_distill_{key}"] = value
            if key in final_metrics:
                try:
                    model.metrics[f"distill_delta_{key}"] = float(final_metrics[key]) - float(value)
                except (TypeError, ValueError):
                    continue
    _record_lgbm_stage_metric_comparison(
        model.metrics,
        candidate_metrics=candidate_metrics,
        fit_oof_metrics=fit_oof_metrics_for_stage,
        post_distill_metrics=final_metrics,
    )
    try:
        cand_auc = float(candidate_metrics.get("auc", np.nan))
        final_auc = float(final_metrics.get("auc", np.nan))
        cand_lift30 = float(candidate_metrics.get("lift30", np.nan))
        final_lift30 = float(final_metrics.get("lift30", np.nan))
        if (
            (np.isfinite(cand_auc) and np.isfinite(final_auc) and final_auc + 0.05 < cand_auc)
            or (
                np.isfinite(cand_lift30)
                and np.isfinite(final_lift30)
                and final_lift30 + 0.20 < cand_lift30
            )
        ):
            tprint(
                "WARNING: LGBM final/post-distill metrics materially trail pre-final race metrics: "
                f"candidate_auc={cand_auc:.4f}, final_auc={final_auc:.4f}, "
                f"candidate_lift30={cand_lift30:.4f}, final_lift30={final_lift30:.4f}, "
                f"split_used={final_used_feature_count}/{len(selected_features)}."
            )
    except Exception:
        pass
    model.pruning_history = list(pruning_history or [])
    tprint(
        f"LGBM full fit done: J={model.metrics.get('J_final', 0.0):.4f}, "
        f"features={len(selected_features)}, elapsed={time.perf_counter() - t0:.1f}s."
    )
    return model


def train_lgbm_stability_pipeline(
    X: Any,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    random_state: int = 42,
    mode: str = "classifier",
    timestamps: Any = None,
    assets: Any = None,
    returns: Any = None,
    hard_labels: Optional[np.ndarray] = None,
    hpo_trials_override: int | None = None,
    hpo_patience_override: int | None = None,
    hpo_objective_mode: str = "train_base",
    meta_feature_output_path: str | os.PathLike[str] | None = None,
    reference_artifact_dir: str | os.PathLike[str] | None = None,
) -> Optional[LGBMStabilityModel]:
    objective_mode = _normalize_objective_mode(hpo_objective_mode)
    candidate = train_lgbm_stability_candidate(
        X,
        y,
        sample_weight=sample_weight,
        random_state=random_state,
        mode=mode,
        timestamps=timestamps,
        assets=assets,
        returns=returns,
        hard_labels=hard_labels,
        hpo_objective_mode=objective_mode,
    )
    if candidate is None:
        return None
    return fit_lgbm_stability_full_model(
        X,
        y,
        sample_weight,
        selected_features_from_cv=candidate.get("selected_features_from_cv"),
        random_state=random_state,
        mode=mode,
        oof_probs=candidate.get("oof_probs"),
        metrics=candidate.get("metrics"),
        pruning_history=candidate.get("pruning_history"),
        selected_feature_names=candidate.get("selected_feature_names"),
        stage_indices=candidate.get("stage_indices"),
        timestamps=timestamps,
        assets=assets,
        returns=returns,
        hard_labels=hard_labels,
        hpo_trials_override=hpo_trials_override,
        hpo_patience_override=hpo_patience_override,
        hpo_objective_mode=objective_mode,
        meta_feature_output_path=meta_feature_output_path,
        reference_artifact_dir=reference_artifact_dir,
    )


def _first_present(columns: set[str], candidates: tuple[str, ...]) -> str | None:
    for col in candidates:
        if col in columns:
            return col
    return None


def tail_control_frames_from_oof(
    df: pd.DataFrame,
    *,
    model: str,
    layer: str,
    target_frac: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    cols = set(map(str, df.columns))
    score_col = _first_present(
        cols,
        (
            "oof_pred",
            "oof_prob",
            "oof_raw",
            "oof_meta_clf",
            "oof_base_clf",
            "score",
            "pred",
        ),
    )
    y_col = _first_present(cols, ("y_bin", "y_move", "target", "label"))
    ret_col = _first_present(cols, ("y_ret", "return", "ret", "target_ret"))
    if score_col is None or y_col is None:
        raise ValueError(
            f"{model}: cannot compute tail-control metrics; missing score or target column"
        )
    work = df.copy()
    pred = pd.to_numeric(work[score_col], errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(work[y_col], errors="coerce").to_numpy(dtype=np.float64)
    if ret_col is not None:
        ret = pd.to_numeric(work[ret_col], errors="coerce").to_numpy(dtype=np.float64)
    else:
        ret = y
    m = np.isfinite(pred) & np.isfinite(y) & np.isfinite(ret)
    pred = pred[m]
    y = y[m]
    ret = ret[m]
    y_win = (y >= 0.5).astype(np.float64)
    baseline = float(np.mean(y_win)) if len(y_win) else 0.0
    timestamps = (
        pd.to_datetime(work.loc[m, "timestamp"], utc=True, errors="coerce")
        if "timestamp" in work.columns
        else pd.Series([pd.NaT] * int(np.sum(m)))
    )
    assets = (
        work.loc[m, "symbol"].astype(str).to_numpy(dtype=object)
        if "symbol" in work.columns
        else np.asarray(["__all__"] * int(np.sum(m)), dtype=object)
    )
    groups = _stability_group_bundle(len(y_win), timestamps=timestamps, assets=assets)
    summary = _tail_control_metrics(
        y_win,
        pred,
        baseline=baseline,
        groups=groups,
        target_frac=target_frac,
    )
    summary.update(
        {
            "model": str(model),
            "layer": str(layer),
            "target_frac": float(target_frac),
            "n_rows": int(len(y_win)),
            "baseline_win_rate": baseline,
            "score_col": str(score_col),
            "target_col": str(y_col),
            "return_col": str(ret_col or ""),
        }
    )
    week_records: list[dict[str, Any]] = []
    asset_records: list[dict[str, Any]] = []
    if groups and groups.get("week") is not None:
        for rec in _tail_group_values(
            y_win,
            pred,
            groups["week"],
            baseline=baseline,
            target_frac=target_frac,
            min_rows=LGBM_TAIL_WEEK_MIN_ROWS,
        ):
            rec.update({"model": str(model), "layer": str(layer), "period_type": "week"})
            week_records.append(rec)
    if not week_records:
        for rec in _tail_rolling_values(
            y_win,
            pred,
            baseline=baseline,
            target_frac=target_frac,
            window_rows=LGBM_TAIL_ROLLING_ROWS,
        ):
            rec.update({"model": str(model), "layer": str(layer), "period_type": "rolling"})
            week_records.append(rec)
    if groups and groups.get("asset") is not None:
        for rec in _tail_group_values(
            y_win,
            pred,
            groups["asset"],
            baseline=baseline,
            target_frac=target_frac,
            min_rows=LGBM_TAIL_ASSET_MIN_ROWS,
        ):
            rec.update({"model": str(model), "layer": str(layer), "period_type": "asset"})
            asset_records.append(rec)
    return summary, pd.DataFrame(week_records), pd.DataFrame(asset_records)


def export_tail_control_reports(
    data_root: str | os.PathLike[str],
    run_id: str,
    *,
    compare_run_ids: list[str] | None = None,
    target_strategy_id: str | None = None,
) -> dict[str, str]:
    root = Path(data_root) / "artifacts" / str(run_id)
    out_dir = root / "tail_control_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = str(target_strategy_id or os.environ.get("EPM_TAIL_CONTROL_STRATEGY_ID", "")).strip()

    def _include(path: Path) -> bool:
        return not target or target in path.name

    summaries: list[dict[str, Any]] = []
    week_frames: list[pd.DataFrame] = []
    asset_frames: list[pd.DataFrame] = []
    for layer, glob_pat, frac in (
        ("base", "oof/oof_*_H*.parquet", LGBM_BASE_METRIC_TARGET_FRACTION),
        ("meta", "meta_oof/meta_oof_*_clf.parquet", LGBM_META_METRIC_TARGET_FRACTION),
    ):
        for path in sorted(root.glob(glob_pat)):
            if not _include(path):
                continue
            try:
                df = pd.read_parquet(path)
                summary, week_df, asset_df = tail_control_frames_from_oof(
                    df,
                    model=path.stem,
                    layer=layer,
                    target_frac=frac,
                )
                summaries.append(summary)
                if not week_df.empty:
                    week_frames.append(week_df)
                if not asset_df.empty:
                    asset_frames.append(asset_df)
            except Exception as exc:
                summaries.append(
                    {
                        "model": path.stem,
                        "layer": layer,
                        "error": str(exc),
                        "tail_control_score": np.nan,
                    }
                )
    per_model = pd.DataFrame(summaries)
    per_week = pd.concat(week_frames, ignore_index=True) if week_frames else pd.DataFrame()
    per_asset = pd.concat(asset_frames, ignore_index=True) if asset_frames else pd.DataFrame()
    if not per_model.empty:
        per_model.insert(0, "run_id", str(run_id))
    if not per_week.empty:
        per_week.insert(0, "run_id", str(run_id))
    if not per_asset.empty:
        per_asset.insert(0, "run_id", str(run_id))
    summary_rows = []
    if not per_model.empty:
        for layer, part in per_model.groupby("layer", dropna=False):
            vals = pd.to_numeric(part.get("tail_control_score"), errors="coerce")
            summary_rows.append(
                {
                    "run_id": str(run_id),
                    "layer": str(layer),
                    "models": int(len(part)),
                    "mean_tail_control_score": float(vals.mean()) if vals.notna().any() else np.nan,
                    "max_tail_control_score": float(vals.max()) if vals.notna().any() else np.nan,
                }
            )
    trial_summary = pd.DataFrame(summary_rows)
    paths = {
        "per_model_metrics_csv": str(out_dir / "per_model_metrics.csv"),
        "per_model_metrics_json": str(out_dir / "per_model_metrics.json"),
        "per_week_metrics_csv": str(out_dir / "per_week_metrics.csv"),
        "per_asset_metrics_csv": str(out_dir / "per_asset_metrics.csv"),
        "trial_summary_csv": str(out_dir / "trial_summary.csv"),
        "trial_summary_json": str(out_dir / "trial_summary.json"),
    }
    per_model.to_csv(paths["per_model_metrics_csv"], index=False)
    per_model.to_json(paths["per_model_metrics_json"], orient="records", indent=2)
    per_week.to_csv(paths["per_week_metrics_csv"], index=False)
    per_asset.to_csv(paths["per_asset_metrics_csv"], index=False)
    trial_summary.to_csv(paths["trial_summary_csv"], index=False)
    trial_summary.to_json(paths["trial_summary_json"], orient="records", indent=2)

    compare_ids = [str(v).strip() for v in (compare_run_ids or []) if str(v).strip()]
    if compare_ids:
        frames = [per_model]
        for other in compare_ids:
            other_path = (
                Path(data_root)
                / "artifacts"
                / other
                / "tail_control_reports"
                / "per_model_metrics.csv"
            )
            if other_path.exists():
                frames.append(pd.read_csv(other_path))
            else:
                other_root = Path(data_root) / "artifacts" / other
                other_summaries: list[dict[str, Any]] = []
                for layer, glob_pat, frac in (
                    ("base", "oof/oof_*_H*.parquet", LGBM_BASE_METRIC_TARGET_FRACTION),
                    ("meta", "meta_oof/meta_oof_*_clf.parquet", LGBM_META_METRIC_TARGET_FRACTION),
                ):
                    for path in sorted(other_root.glob(glob_pat)):
                        if not _include(path):
                            continue
                        try:
                            df = pd.read_parquet(path)
                            summary, _, _ = tail_control_frames_from_oof(
                                df,
                                model=path.stem,
                                layer=layer,
                                target_frac=frac,
                            )
                            summary["run_id"] = other
                            other_summaries.append(summary)
                        except Exception:
                            continue
                if other_summaries:
                    frames.append(pd.DataFrame(other_summaries))
        comparison = pd.concat(frames, ignore_index=True) if frames else per_model
        comparison_path = out_dir / "comparison_table.csv"
        comparison.to_csv(comparison_path, index=False)
        paths["comparison_table_csv"] = str(comparison_path)
    return paths


__all__ = [
    "LGBMStabilityModel",
    "FeatureSelectionResult",
    "train_lgbm_stability_candidate",
    "fit_lgbm_stability_full_model",
    "train_lgbm_stability_pipeline",
    "tail_control_frames_from_oof",
    "export_tail_control_reports",
    "train_base",
    "train_meta",
    "LGBM_META_FEATURE_NAMES",
    "score_for_trading",
]
