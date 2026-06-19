from __future__ import annotations

import gc
import json
import os
import pickle
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

try:
    from numba import njit as _numba_njit

    _LGBM_LABEL_WEIGHT_NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional acceleration
    _LGBM_LABEL_WEIGHT_NUMBA_AVAILABLE = False

    def _numba_njit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def _wrap(fn):
            return fn

        return _wrap

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
    from .label_weight_optuna import apply_distillation_recipe, load_recipe_from_env_or_cfg
except Exception:  # pragma: no cover - optional experiment hook
    def apply_distillation_recipe(distill, fp_weight, **_kwargs):
        return np.asarray(distill, dtype=np.float32), np.asarray(fp_weight, dtype=np.float32)

    def load_recipe_from_env_or_cfg(_cfg=None, *, scope=None):
        return None

try:
    from .utils import tprint
except Exception:  # pragma: no cover - standalone fallback
    def tprint(message: str) -> None:
        print(message, flush=True)

from .model_drift_features import MODEL_DRIFT_FEATURE_KEYS
from .features_gmm_ae import (
    AE_GMM_FEATURE_COLUMNS,
    AE_GMM_LATENT_FEATURE_COLUMNS,
    fit_ae_gmm_state,
    transform_ae_gmm_features,
)

from .lgbm_archetype_features import (
    ARCHETYPE_FEATURE_NAMES,
    BASE_ERROR_ARCHETYPE_FEATURE_NAMES,
    CONTRIB_ARCHETYPE_FEATURE_NAMES,
    CONTRIB_SUMMARY_FEATURE_NAMES,
    META_RAW_CONTRIB_SVD_FEATURE_NAMES,
    RAW_CONTRIB_FEATURE_PREFIX,
    RAW_STATE_DIAGNOSTIC_FEATURE_NAMES,
    RAW_STATE_SVD_FEATURE_NAMES,
    ContribArchetypeState,
    RawStateArchetypeState,
    ResidualErrorArchetypeState,
    contrib_summary_frame,
    fit_contrib_archetype_state,
    fit_raw_state_archetype_state,
    fit_residual_error_archetype_state,
    is_raw_contrib_feature_name,
    raw_contrib_feature_mapping,
    raw_contrib_frame,
    transform_contrib_archetype_features,
    transform_raw_state_archetype_features,
    transform_residual_error_archetype_features,
)
from .lgbm_recency_hpo import (
    active_recency_hpo_config,
    composite_decay_from_timestamps,
    final_selection_score as recency_hpo_final_selection_score,
    objective_scope as recency_hpo_objective_scope,
    recency_hpo_decay_from_config,
    recency_hpo_grid,
    recency_hpo_train_oos_masks,
    save_recency_hpo_winner,
)
from .model_effectiveness_history import (
    apply_model_effectiveness_history_defaults,
    build_model_effectiveness_history_defaults,
)


LGBM_CV_SPLITS = int(os.environ.get("EPM_LGBM_CV_SPLITS", "3"))
_LGBM_CV_MODE_RAW = str(os.environ.get("EPM_LGBM_CV_MODE", "")).strip().lower()
LGBM_PURGED_CV = os.environ.get("EPM_LGBM_PURGED_CV", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
LGBM_CV_MODE = (
    _LGBM_CV_MODE_RAW
    if _LGBM_CV_MODE_RAW
    else ("purged_time" if LGBM_PURGED_CV else "interleaved_spread")
)
LGBM_PURGE_HOURS = float(os.environ.get("EPM_LGBM_PURGE_HOURS", "10"))
LGBM_RECENCY_WEIGHTING = os.environ.get("EPM_LGBM_RECENCY_WEIGHTING", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
LGBM_BASE_RECENCY_HALF_LIFE_DAYS = float(
    os.environ.get("EPM_LGBM_BASE_RECENCY_HALF_LIFE_DAYS", "365")
)
LGBM_META_RECENCY_HALF_LIFE_DAYS = float(
    os.environ.get("EPM_LGBM_META_RECENCY_HALF_LIFE_DAYS", "182.5")
)
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
LGBM_HPO_EARLY_STOP_PATIENCE = int(os.environ.get("EPM_LGBM_HPO_EARLY_STOP_PATIENCE", "40"))
LGBM_EARLY_STOPPING_ROUNDS = int(os.environ.get("EPM_LGBM_EARLY_STOPPING_ROUNDS", "40"))
LGBM_BASE_LABEL_WEIGHT_HPO_ENABLED = os.environ.get(
    "EPM_LGBM_BASE_LABEL_WEIGHT_HPO",
    "1",
).strip().lower() not in {"0", "false", "no", "n", "off"}
LGBM_LABEL_WEIGHT_HPO_LAYER1_TRIALS = int(
    os.environ.get("EPM_LGBM_LABEL_WEIGHT_HPO_LAYER1_TRIALS", "300")
)
LGBM_LABEL_WEIGHT_HPO_LAYER1_PATIENCE = int(
    os.environ.get("EPM_LGBM_LABEL_WEIGHT_HPO_LAYER1_PATIENCE", "40")
)
LGBM_LABEL_WEIGHT_HPO_LAYER2_TRIALS = int(
    os.environ.get("EPM_LGBM_LABEL_WEIGHT_HPO_LAYER2_TRIALS", "150")
)
LGBM_LABEL_WEIGHT_HPO_LAYER2_PATIENCE = int(
    os.environ.get("EPM_LGBM_LABEL_WEIGHT_HPO_LAYER2_PATIENCE", "30")
)
LGBM_LABEL_WEIGHT_HPO_MAX_ROWS = int(
    os.environ.get("EPM_LGBM_LABEL_WEIGHT_HPO_MAX_ROWS", str(LGBM_HPO_MAX_ROWS))
)
LGBM_LABEL_WEIGHT_HPO_ELECTION_MAX_ROWS = int(
    os.environ.get("EPM_LGBM_LABEL_WEIGHT_HPO_ELECTION_MAX_ROWS", "50000")
)
LGBM_LABEL_WEIGHT_HPO_CV_SPLITS = int(
    os.environ.get("EPM_LGBM_LABEL_WEIGHT_HPO_CV_SPLITS", "2")
)
LGBM_LABEL_WEIGHT_HPO_N_ESTIMATORS_CAP = int(
    os.environ.get("EPM_LGBM_LABEL_WEIGHT_HPO_N_ESTIMATORS_CAP", "300")
)
LGBM_LABEL_WEIGHT_HPO_EARLY_STOPPING_ROUNDS = int(
    os.environ.get("EPM_LGBM_LABEL_WEIGHT_HPO_EARLY_STOPPING_ROUNDS", "20")
)
LGBM_LABEL_WEIGHT_HPO_MIN_ROWS = int(
    os.environ.get("EPM_LGBM_LABEL_WEIGHT_HPO_MIN_ROWS", "1000")
)
LGBM_LABEL_WEIGHT_HPO_NUMBA = os.environ.get(
    "EPM_LGBM_LABEL_WEIGHT_HPO_NUMBA",
    "1",
).strip().lower() not in {"0", "false", "no", "n", "off"}
LGBM_HPO_FINAL_MIN_ESTIMATORS = int(os.environ.get("EPM_LGBM_HPO_FINAL_MIN_ESTIMATORS", "800"))
LGBM_N_ESTIMATORS_CAP = int(os.environ.get("EPM_LGBM_N_ESTIMATORS_CAP", "0"))
LGBM_FINAL_MODEL_COUNT = int(os.environ.get("EPM_LGBM_FINAL_MODEL_COUNT", "3"))
LGBM_OOF_DISTILLATION_PASSES = int(os.environ.get("EPM_LGBM_OOF_DISTILLATION_PASSES", "1"))
LGBM_MIN_OOF_DISTILLATION_PASSES = int(os.environ.get("EPM_LGBM_MIN_OOF_DISTILLATION_PASSES", "2"))
LGBM_META_MIN_OOF_DISTILLATION_PASSES = int(os.environ.get("EPM_LGBM_META_MIN_OOF_DISTILLATION_PASSES", "2"))
LGBM_DISABLE_SELF_DISTILLATION = os.environ.get("EPM_LGBM_DISABLE_SELF_DISTILLATION", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
LGBM_SKIP_FINAL_OOF_META_CV = os.environ.get("EPM_LGBM_SKIP_FINAL_OOF_META_CV", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
LGBM_SKIP_REFERENCE_ARTIFACTS = os.environ.get("EPM_LGBM_SKIP_REFERENCE_ARTIFACTS", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
LGBM_ARCHETYPE_FEATURES = os.environ.get("EPM_LGBM_ARCHETYPE_FEATURES", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "n",
    "off",
}
LGBM_RAW_CONTRIB_OOF_EXPORT = os.environ.get("EPM_LGBM_RAW_CONTRIB_OOF_EXPORT", "0").strip().lower() not in {
    "0",
    "false",
    "no",
    "n",
    "off",
}
LGBM_AE_GMM_FEATURES = os.environ.get("EPM_LGBM_AE_GMM_FEATURES", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "n",
    "off",
}
LGBM_AE_GMM_MAX_TRAIN_ROWS = int(os.environ.get("EPM_LGBM_AE_GMM_MAX_TRAIN_ROWS", "5000"))
LGBM_AE_GMM_MAX_ITER = int(os.environ.get("EPM_LGBM_AE_GMM_MAX_ITER", "80"))
LGBM_FINAL_MODEL_CHECKPOINT_DIR = os.environ.get("EPM_LGBM_FINAL_MODEL_CHECKPOINT_DIR", "").strip()
LGBM_META_LEAF_DIAGNOSTICS = os.environ.get("EPM_LGBM_META_LEAF_DIAGNOSTICS", "0").strip().lower() not in {
    "0",
    "false",
    "no",
    "n",
    "off",
}
LGBM_META_LEAF_LITE_DIAGNOSTICS = os.environ.get(
    "EPM_LGBM_META_LEAF_LITE_DIAGNOSTICS",
    "1",
).strip().lower() not in {
    "0",
    "false",
    "no",
    "n",
    "off",
}
LGBM_META_LEAF_SUPPORT_DIAGNOSTICS = os.environ.get(
    "EPM_LGBM_META_LEAF_SUPPORT_DIAGNOSTICS",
    "1",
).strip().lower() not in {
    "0",
    "false",
    "no",
    "n",
    "off",
}
LGBM_META_LEAF_TARGET_DIAGNOSTICS = os.environ.get(
    "EPM_LGBM_META_LEAF_TARGET_DIAGNOSTICS",
    "1",
).strip().lower() not in {
    "0",
    "false",
    "no",
    "n",
    "off",
}
LGBM_META_LEAF_CENTROID_DIAGNOSTICS = os.environ.get(
    "EPM_LGBM_META_LEAF_CENTROID_DIAGNOSTICS",
    "1" if LGBM_META_LEAF_DIAGNOSTICS else "0",
).strip().lower() not in {
    "0",
    "false",
    "no",
    "n",
    "off",
}
LGBM_META_LEAF_MAX_TREES = int(os.environ.get("EPM_LGBM_META_LEAF_MAX_TREES", "64"))
LGBM_META_CONTRIB_DIAGNOSTICS = os.environ.get("EPM_LGBM_META_CONTRIB_DIAGNOSTICS", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "n",
    "off",
}
LGBM_META_CONTRIB_METHOD = str(os.environ.get("EPM_LGBM_META_CONTRIB_METHOD", "path")).strip().lower()
if LGBM_META_CONTRIB_METHOD in {"lightgbm", "shap", "pred_contrib", "full"}:
    LGBM_META_CONTRIB_METHOD = "shap"
elif LGBM_META_CONTRIB_METHOD not in {"path", "saabas"}:
    LGBM_META_CONTRIB_METHOD = "path"
LGBM_META_SCORE_PATH_DIAGNOSTICS = os.environ.get("EPM_LGBM_META_SCORE_PATH_DIAGNOSTICS", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "n",
    "off",
}
LGBM_META_SCORE_PATH_MAX_TREES = int(
    os.environ.get("EPM_LGBM_META_SCORE_PATH_MAX_TREES", str(LGBM_META_LEAF_MAX_TREES))
)
LGBM_META_DRIFT_FEATURES = os.environ.get("EPM_LGBM_META_DRIFT_FEATURES", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "n",
    "off",
}
LGBM_META_DRIFT_MAX_ROWS = int(os.environ.get("EPM_LGBM_META_DRIFT_MAX_ROWS", "100000"))
LGBM_META_DRIFT_MAX_FEATURES = int(os.environ.get("EPM_LGBM_META_DRIFT_MAX_FEATURES", "32"))
LGBM_FINAL_OOF_CONTEXT_FEATURES = os.environ.get("EPM_LGBM_FINAL_OOF_CONTEXT_FEATURES", "0").strip().lower() not in {
    "0",
    "false",
    "no",
    "n",
    "off",
}
LGBM_FINAL_OOF_CONTRIB_CONTEXT_FEATURES = os.environ.get(
    "EPM_LGBM_FINAL_OOF_CONTRIB_CONTEXT_FEATURES",
    "1" if LGBM_FINAL_OOF_CONTEXT_FEATURES else "0",
).strip().lower() not in {"0", "false", "no", "n", "off"}
LGBM_FINAL_OOF_RAW_STATE_CONTEXT_FEATURES = os.environ.get(
    "EPM_LGBM_FINAL_OOF_RAW_STATE_CONTEXT_FEATURES",
    "1" if LGBM_FINAL_OOF_CONTEXT_FEATURES else "0",
).strip().lower() not in {"0", "false", "no", "n", "off"}
LGBM_FINAL_OOF_BASE_ERROR_CONTEXT_FEATURES = os.environ.get(
    "EPM_LGBM_FINAL_OOF_BASE_ERROR_CONTEXT_FEATURES",
    "1" if LGBM_FINAL_OOF_CONTEXT_FEATURES else "0",
).strip().lower() not in {"0", "false", "no", "n", "off"}
LGBM_CONTRIB_PRED_BATCH_ROWS = int(os.environ.get("EPM_LGBM_CONTRIB_PRED_BATCH_ROWS", "50000"))
LGBM_ARCHETYPE_FIT_MAX_ROWS = int(os.environ.get("EPM_LGBM_ARCHETYPE_FIT_MAX_ROWS", "50000"))
LGBM_OPTUNA_CANDIDATE_ONLY = os.environ.get("EPM_LGBM_OPTUNA_CANDIDATE_ONLY", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
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
LGBM_HPO_MAX_DEPTH_MAX = int(os.environ.get("EPM_LGBM_HPO_MAX_DEPTH_MAX", "7"))
LGBM_HPO_SUBSAMPLE_MIN = float(os.environ.get("EPM_LGBM_HPO_SUBSAMPLE_MIN", "0.55"))
LGBM_HPO_SUBSAMPLE_MAX = float(os.environ.get("EPM_LGBM_HPO_SUBSAMPLE_MAX", "0.90"))
LGBM_HPO_MIN_CHILD_WEIGHT_MIN = float(os.environ.get("EPM_LGBM_HPO_MIN_CHILD_WEIGHT_MIN", "15.0"))
LGBM_HPO_MIN_CHILD_WEIGHT_MAX = float(os.environ.get("EPM_LGBM_HPO_MIN_CHILD_WEIGHT_MAX", "90.0"))
LGBM_HPO_PATH_SMOOTH_MAX = float(os.environ.get("EPM_LGBM_HPO_PATH_SMOOTH_MAX", "12.0"))
LGBM_FINAL_MIN_CHILD_TRANSFER_ALPHA = float(os.environ.get("EPM_LGBM_FINAL_MIN_CHILD_TRANSFER_ALPHA", "0.5"))
LGBM_FINAL_MIN_CHILD_SAMPLES_CAP_PCT = float(os.environ.get("EPM_LGBM_FINAL_MIN_CHILD_SAMPLES_CAP_PCT", "0.01"))
LGBM_FINAL_MIN_CHILD_SAMPLES_BASE_ABS = int(os.environ.get("EPM_LGBM_FINAL_MIN_CHILD_SAMPLES_BASE_ABS", "300"))
LGBM_FINAL_MIN_CHILD_SAMPLES_META_ABS = int(os.environ.get("EPM_LGBM_FINAL_MIN_CHILD_SAMPLES_META_ABS", "200"))
LGBM_FINAL_MIN_CHILD_SAMPLES_CAP = int(os.environ.get("EPM_LGBM_FINAL_MIN_CHILD_SAMPLES_CAP", "0"))
LGBM_FINAL_LEAF_FLOOR_PRESET = os.environ.get("EPM_LGBM_FINAL_LEAF_FLOOR_PRESET", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
LGBM_FEATURE_SELECTION_N_ESTIMATORS = 300
LGBM_BASE_METRIC_TARGET_FRACTION = float(os.environ.get("EPM_LGBM_BASE_METRIC_TARGET_FRACTION", "0.30"))
LGBM_META_METRIC_TARGET_FRACTION = float(os.environ.get("EPM_LGBM_META_METRIC_TARGET_FRACTION", os.environ.get("EPM_LGBM_METRIC_TARGET_FRACTION", "0.15")))
LGBM_SALG_LIFT_COEF = float(os.environ.get("EPM_LGBM_SALG_LIFT_COEF", "0.38"))
LGBM_J_SALG_NORM_DENOM = float(os.environ.get("EPM_LGBM_J_SALG_NORM_DENOM", "1.50"))
LGBM_OBJECTIVE = str(os.environ.get("EPM_LGBM_OBJECTIVE", "default")).strip().lower()
LGBM_TRUE_SOFT_LABELS = os.environ.get("EPM_LGBM_TRUE_SOFT_LABELS", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
LGBM_REBALANCE_EFFECTIVE_CLASSES = os.environ.get(
    "EPM_LGBM_REBALANCE_EFFECTIVE_CLASSES", "0"
).strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
LGBM_REBALANCE_POS_MASS_MIN = float(os.environ.get("EPM_LGBM_REBALANCE_POS_MASS_MIN", "0.25"))
LGBM_REBALANCE_POS_MASS_MAX = float(os.environ.get("EPM_LGBM_REBALANCE_POS_MASS_MAX", "0.55"))
LGBM_REBALANCE_MAX_MULTIPLIER = float(os.environ.get("EPM_LGBM_REBALANCE_MAX_MULTIPLIER", "2.0"))
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
if LGBM_CV_MODE not in {"", "shuffled", "purged_time", "interleaved", "interleaved_spread"}:
    LGBM_CV_MODE = "shuffled"
LGBM_PURGE_HOURS = max(0.0, float(LGBM_PURGE_HOURS))
LGBM_BASE_RECENCY_HALF_LIFE_DAYS = max(1e-6, float(LGBM_BASE_RECENCY_HALF_LIFE_DAYS))
LGBM_META_RECENCY_HALF_LIFE_DAYS = max(1e-6, float(LGBM_META_RECENCY_HALF_LIFE_DAYS))
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
LGBM_EARLY_STOPPING_ROUNDS = max(1, int(LGBM_EARLY_STOPPING_ROUNDS))
LGBM_LABEL_WEIGHT_HPO_LAYER1_TRIALS = max(0, int(LGBM_LABEL_WEIGHT_HPO_LAYER1_TRIALS))
LGBM_LABEL_WEIGHT_HPO_LAYER1_PATIENCE = max(1, int(LGBM_LABEL_WEIGHT_HPO_LAYER1_PATIENCE))
LGBM_LABEL_WEIGHT_HPO_LAYER2_TRIALS = max(0, int(LGBM_LABEL_WEIGHT_HPO_LAYER2_TRIALS))
LGBM_LABEL_WEIGHT_HPO_LAYER2_PATIENCE = max(1, int(LGBM_LABEL_WEIGHT_HPO_LAYER2_PATIENCE))
LGBM_LABEL_WEIGHT_HPO_MAX_ROWS = max(0, int(LGBM_LABEL_WEIGHT_HPO_MAX_ROWS))
LGBM_LABEL_WEIGHT_HPO_ELECTION_MAX_ROWS = max(0, int(LGBM_LABEL_WEIGHT_HPO_ELECTION_MAX_ROWS))
LGBM_LABEL_WEIGHT_HPO_CV_SPLITS = max(2, int(LGBM_LABEL_WEIGHT_HPO_CV_SPLITS))
LGBM_LABEL_WEIGHT_HPO_N_ESTIMATORS_CAP = max(0, int(LGBM_LABEL_WEIGHT_HPO_N_ESTIMATORS_CAP))
LGBM_LABEL_WEIGHT_HPO_EARLY_STOPPING_ROUNDS = max(0, int(LGBM_LABEL_WEIGHT_HPO_EARLY_STOPPING_ROUNDS))
LGBM_LABEL_WEIGHT_HPO_MIN_ROWS = max(100, int(LGBM_LABEL_WEIGHT_HPO_MIN_ROWS))
LGBM_HPO_FINAL_MIN_ESTIMATORS = max(1, int(LGBM_HPO_FINAL_MIN_ESTIMATORS))
LGBM_N_ESTIMATORS_CAP = max(0, int(LGBM_N_ESTIMATORS_CAP))
LGBM_HPO_MAX_DEPTH_MAX = int(np.clip(LGBM_HPO_MAX_DEPTH_MAX, 4, 8))
LGBM_HPO_SUBSAMPLE_MIN = float(np.clip(LGBM_HPO_SUBSAMPLE_MIN, 0.30, 1.0))
LGBM_HPO_SUBSAMPLE_MAX = float(
    np.clip(max(LGBM_HPO_SUBSAMPLE_MAX, LGBM_HPO_SUBSAMPLE_MIN), 0.30, 1.0)
)
LGBM_HPO_MIN_CHILD_WEIGHT_MIN = max(0.0, float(LGBM_HPO_MIN_CHILD_WEIGHT_MIN))
LGBM_HPO_MIN_CHILD_WEIGHT_MAX = max(
    LGBM_HPO_MIN_CHILD_WEIGHT_MIN,
    float(LGBM_HPO_MIN_CHILD_WEIGHT_MAX),
)
LGBM_HPO_PATH_SMOOTH_MAX = float(np.clip(LGBM_HPO_PATH_SMOOTH_MAX, 0.0, 15.0))
LGBM_FINAL_MIN_CHILD_TRANSFER_ALPHA = float(np.clip(LGBM_FINAL_MIN_CHILD_TRANSFER_ALPHA, 0.0, 1.0))
LGBM_FINAL_MIN_CHILD_SAMPLES_CAP_PCT = float(np.clip(LGBM_FINAL_MIN_CHILD_SAMPLES_CAP_PCT, 0.0, 0.10))
LGBM_FINAL_MIN_CHILD_SAMPLES_BASE_ABS = max(0, int(LGBM_FINAL_MIN_CHILD_SAMPLES_BASE_ABS))
LGBM_FINAL_MIN_CHILD_SAMPLES_META_ABS = max(0, int(LGBM_FINAL_MIN_CHILD_SAMPLES_META_ABS))
LGBM_FINAL_MIN_CHILD_SAMPLES_CAP = max(0, int(LGBM_FINAL_MIN_CHILD_SAMPLES_CAP))
LGBM_META_LEAF_MAX_TREES = max(0, int(LGBM_META_LEAF_MAX_TREES))
LGBM_META_SCORE_PATH_MAX_TREES = max(0, int(LGBM_META_SCORE_PATH_MAX_TREES))
LGBM_META_DRIFT_MAX_ROWS = max(0, int(LGBM_META_DRIFT_MAX_ROWS))
LGBM_META_DRIFT_MAX_FEATURES = max(1, int(LGBM_META_DRIFT_MAX_FEATURES))
LGBM_OOF_DISTILLATION_PASSES = max(0, int(LGBM_OOF_DISTILLATION_PASSES))
LGBM_MIN_OOF_DISTILLATION_PASSES = max(0, int(LGBM_MIN_OOF_DISTILLATION_PASSES))
LGBM_META_MIN_OOF_DISTILLATION_PASSES = max(0, int(LGBM_META_MIN_OOF_DISTILLATION_PASSES))
LGBM_META_RANK_BINS = max(2, int(LGBM_META_RANK_BINS))
LGBM_BASE_METRIC_TARGET_FRACTION = float(np.clip(LGBM_BASE_METRIC_TARGET_FRACTION, 0.001, 0.5))
LGBM_META_METRIC_TARGET_FRACTION = float(np.clip(LGBM_META_METRIC_TARGET_FRACTION, 0.001, 0.5))
LGBM_J_SALG_NORM_DENOM = max(1e-6, float(LGBM_J_SALG_NORM_DENOM))
LGBM_REBALANCE_POS_MASS_MIN = float(np.clip(LGBM_REBALANCE_POS_MASS_MIN, 0.01, 0.95))
LGBM_REBALANCE_POS_MASS_MAX = float(
    np.clip(max(LGBM_REBALANCE_POS_MASS_MAX, LGBM_REBALANCE_POS_MASS_MIN), 0.01, 0.95)
)
LGBM_REBALANCE_MAX_MULTIPLIER = max(1.0, float(LGBM_REBALANCE_MAX_MULTIPLIER))
if LGBM_OBJECTIVE not in {"default", "tail_control"}:
    LGBM_OBJECTIVE = "default"
if LGBM_HPO_PARAM_SET not in {"full", "reduced"}:
    LGBM_HPO_PARAM_SET = "full"
LGBM_TAIL_WEEK_MIN_ROWS = max(1, int(LGBM_TAIL_WEEK_MIN_ROWS))
LGBM_TAIL_ASSET_MIN_ROWS = max(1, int(LGBM_TAIL_ASSET_MIN_ROWS))
LGBM_TAIL_ROLLING_ROWS = max(8, int(LGBM_TAIL_ROLLING_ROWS))
LGBM_TAIL_LIFT_NORM_DENOM = max(1e-6, float(LGBM_TAIL_LIFT_NORM_DENOM))
LGBM_TAIL_WORST_FEATURE_PENALTY = float(np.clip(LGBM_TAIL_WORST_FEATURE_PENALTY, 0.0, 1.0))

LGBM_META_CORE_FEATURE_NAMES = [
    "lgbm_prob",
    "lgbm_raw_score",
    "abs_raw_score",
    "model_count",
    "tree_count_mean",
    "tree_count_min",
    "tree_count_max",
    "prob_mean",
    "prob_std",
    "prob_min",
    "prob_max",
    "prob_range",
    "raw_score_mean",
    "raw_score_std",
    "raw_score_min",
    "raw_score_max",
    "raw_score_range",
    "margin_from_neutral",
    "prob_uncertainty",
    "entropy",
    "variance_proxy",
    "rank_pct",
    "score_margin_top10",
    "score_margin_top20",
    "score_margin_top30",
    "rank_margin_top10",
    "rank_margin_top20",
    "rank_margin_top30",
    "rank_bin_win_rate_oof",
    "rank_bin_lift_oof",
    "rank_bin_net_ret_oof",
    "rank_bin_se_oof",
]

LGBM_META_LEAF_LITE_FEATURE_NAMES = [
    "leaf_count_mean",
    "leaf_count_median",
    "leaf_count_q25",
    "leaf_count_p10",
    "leaf_count_min",
    "rare_leaf_fraction",
    "leaf_weight_mean",
    "leaf_weight_p10",
    "leaf_depth_mean",
    "leaf_depth_std",
    "leaf_depth_max",
    "leaf_value_abs_mean",
    "leaf_value_abs_std",
    "leaf_value_abs_p90",
    "leaf_value_abs_max",
    "large_leaf_value_fraction",
]

LGBM_META_LEAF_SUPPORT_FEATURE_NAMES = [
    "leaf_train_freq_mean",
    "leaf_train_freq_p90",
    "leaf_train_freq_p10",
    "leaf_train_freq_min",
    "leaf_train_freq_max",
    "leaf_train_freq_std",
    "leaf_surprisal_mean",
    "leaf_surprisal_p90",
    "leaf_surprisal_max",
    "leaf_low_freq_fraction",
    "leaf_proximity_mean",
    "leaf_proximity_p90",
    "leaf_proximity_max",
    "leaf_model_space_distance_mean",
    "leaf_model_space_distance_p10",
]

LGBM_META_LEAF_TARGET_FEATURE_NAMES = [
    "leaf_target_mean_mean",
    "leaf_target_mean_std",
    "leaf_target_mean_min",
    "leaf_target_mean_max",
    "leaf_target_std_mean",
    "leaf_target_iqr_mean",
    "leaf_target_range_mean",
    "leaf_target_abs_mean",
    "leaf_target_positive_fraction",
    "leaf_hit_rate_avg",
    "leaf_target_dispersion",
    "support_gap",
    "leaf_pred_mean_mean",
    "leaf_error_mean_mean",
]

LGBM_META_LEAF_CENTROID_FEATURE_NAMES = [
    "leaf_centroid_radius_mean",
    "leaf_centroid_dist_mean",
    "leaf_centroid_dist_median",
    "leaf_centroid_dist_std",
    "leaf_centroid_dist_p90",
    "leaf_centroid_dist_max",
    "leaf_centroid_dist_cv",
    "leaf_centroid_dist_rel_mean",
    "leaf_centroid_dist_rel_std",
    "leaf_centroid_dist_norm_mean",
    "leaf_centroid_dist_norm_p90",
    "leaf_centroid_dist_norm_max",
]
LGBM_META_LEAF_FULL_EXTRA_FEATURE_NAMES = (
    list(LGBM_META_LEAF_SUPPORT_FEATURE_NAMES)
    + list(LGBM_META_LEAF_TARGET_FEATURE_NAMES)
    + list(LGBM_META_LEAF_CENTROID_FEATURE_NAMES)
)
LGBM_META_LEAF_FEATURE_NAMES = (
    list(LGBM_META_LEAF_LITE_FEATURE_NAMES)
    + list(LGBM_META_LEAF_FULL_EXTRA_FEATURE_NAMES)
)

LGBM_META_CONTRIB_FEATURE_NAMES = [
    *CONTRIB_SUMMARY_FEATURE_NAMES,
    "contrib_top1_abs_share",
    "contrib_top3_abs_share",
    "contrib_entropy",
    "contrib_balance",
    "num_material_contrib_features",
]

LGBM_META_SCORE_PATH_FEATURE_NAMES = [
    "score_final",
    "score_early_10pct",
    "score_early_25pct",
    "score_early_50pct",
    "score_100_minus_50",
    "score_100_minus_75",
    "score_path_std",
    "score_path_volatility",
    "score_path_min",
    "score_path_max",
    "score_path_drawdown",
    "score_reversal_count",
    "positive_tree_frac",
    "negative_tree_frac",
    "mean_tree_contribution",
    "max_tree_contribution",
    "top_tree_contribution_share",
    "rank_100_minus_50",
    "rank_path_std",
]

LGBM_META_DRIFT_FEATURE_NAMES = [
    "regime_centroid_similarity_train",
    "feature_drift_psi_core",
    "feature_drift_ks_core",
    "feature_drift_cov_shift",
]

LGBM_META_CONTRIB_CONTEXT_FEATURE_NAMES = [name for name in CONTRIB_ARCHETYPE_FEATURE_NAMES]
LGBM_META_RAW_STATE_CONTEXT_FEATURE_NAMES = list(RAW_STATE_SVD_FEATURE_NAMES) + list(
    RAW_STATE_DIAGNOSTIC_FEATURE_NAMES
)
LGBM_META_BASE_ERROR_CONTEXT_FEATURE_NAMES = list(BASE_ERROR_ARCHETYPE_FEATURE_NAMES)
LGBM_META_AE_GMM_FEATURE_NAMES = list(AE_GMM_FEATURE_COLUMNS)
LGBM_META_CONTEXT_FEATURE_NAMES = (
    list(LGBM_META_CONTRIB_CONTEXT_FEATURE_NAMES)
    + list(LGBM_META_RAW_STATE_CONTEXT_FEATURE_NAMES)
    + list(LGBM_META_BASE_ERROR_CONTEXT_FEATURE_NAMES)
    + list(LGBM_META_AE_GMM_FEATURE_NAMES)
)

LGBM_META_FEATURE_NAMES = list(
    dict.fromkeys(
        list(LGBM_META_CORE_FEATURE_NAMES)
        + (
            list(LGBM_META_LEAF_FEATURE_NAMES)
            if LGBM_META_LEAF_DIAGNOSTICS
            else (
                (
                    list(LGBM_META_LEAF_LITE_FEATURE_NAMES)
                    if LGBM_META_LEAF_LITE_DIAGNOSTICS
                    else []
                )
                + (
                    list(LGBM_META_LEAF_SUPPORT_FEATURE_NAMES)
                    if LGBM_META_LEAF_SUPPORT_DIAGNOSTICS
                    else []
                )
                + (
                    list(LGBM_META_LEAF_TARGET_FEATURE_NAMES)
                    if LGBM_META_LEAF_TARGET_DIAGNOSTICS
                    else []
                )
                + (
                    list(LGBM_META_LEAF_CENTROID_FEATURE_NAMES)
                    if LGBM_META_LEAF_CENTROID_DIAGNOSTICS
                    else []
                )
            )
        )
        + (list(LGBM_META_CONTRIB_FEATURE_NAMES) if LGBM_META_CONTRIB_DIAGNOSTICS else [])
        + (list(LGBM_META_SCORE_PATH_FEATURE_NAMES) if LGBM_META_SCORE_PATH_DIAGNOSTICS else [])
        + (list(LGBM_META_DRIFT_FEATURE_NAMES) if LGBM_META_DRIFT_FEATURES else [])
        + (
            list(LGBM_META_CONTRIB_CONTEXT_FEATURE_NAMES)
            if LGBM_FINAL_OOF_CONTRIB_CONTEXT_FEATURES
            else []
        )
        + (
            list(LGBM_META_RAW_STATE_CONTEXT_FEATURE_NAMES)
            if LGBM_FINAL_OOF_RAW_STATE_CONTEXT_FEATURES
            else []
        )
        + (
            list(LGBM_META_BASE_ERROR_CONTEXT_FEATURE_NAMES)
            if LGBM_FINAL_OOF_BASE_ERROR_CONTEXT_FEATURES
            else []
        )
        + (list(LGBM_META_AE_GMM_FEATURE_NAMES) if LGBM_AE_GMM_FEATURES else [])
    )
)

LGBM_INTERNAL_METRIC_FEATURE_NAMES = list(dict.fromkeys(LGBM_META_FEATURE_NAMES))


def _is_lgbm_model_derived_meta_feature(name: str) -> bool:
    """Return true for meta diagnostics unavailable in the raw coverage slice."""
    key = str(name)
    if key in LGBM_META_FEATURE_NAMES:
        return True
    if key in {
        "pred_logit",
        "prob_error",
        "signed_prediction_error",
        "negative_log_likelihood",
        "surprise_error_z",
        "wrong_confident",
        "regime_centroid_similarity_train",
        "feature_drift_psi_core",
        "feature_drift_ks_core",
        "feature_drift_cov_shift",
        "inference_drift_score",
        "uncertainty_score",
        "rare_leaf_low_support_score",
        "contribution_drift_score",
    }:
        return True
    return key.startswith(
        (
            "pred_",
            "base_H",
            "base_lgbm_",
            "meta_lgbm_",
            "base_med_",
            "base_prob_",
            "base_model_",
            "base_error_",
            "shap_archetype_",
            "archetype_",
            "raw_state_",
            "state_",
            "feature_drift_",
            "recent_",
            "drift_",
            "leaf_",
            "reg_leaf_",
            "reg_rare_leaf_",
            "contrib_",
            "rank_bin_",
            "score_path_",
            "score_margin_top",
            "rank_margin_top",
            "distance_to_archetype_",
            "distance_to_bad_archetype",
            "distance_to_good_archetype",
            "distance_to_nearest_bad_archetype",
            "top_1_contrib_abs",
            "top_3_contrib_abs_sum",
            "positive_contrib_sum",
            "negative_contrib_sum",
            "support_gap",
            "mahalanobis_mean_shift",
            "frobenius_corr_shift",
            "recent_prob_error_",
            "recent_hit_rate_",
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
    contrib_archetype_state: ContribArchetypeState | None = None
    raw_state_archetype_state: RawStateArchetypeState | None = None
    base_error_archetype_state: ResidualErrorArchetypeState | None = None
    oof_uncertainty_features: dict[str, np.ndarray] = field(default_factory=dict)
    raw_contrib_oof_features: Optional[pd.DataFrame] = None
    raw_contrib_oof_feature_names: list[str] = field(default_factory=list)
    raw_contrib_feature_mapping: dict[str, str] = field(default_factory=dict)
    raw_contrib_input_features: list[str] = field(default_factory=list)
    raw_contrib_passthrough_features: list[str] = field(default_factory=list)
    raw_contrib_transformed_feature_names: list[str] = field(default_factory=list)
    raw_contrib_input_state: ContribArchetypeState | None = None
    ae_gmm_input_features: list[str] = field(default_factory=list)
    ae_gmm_feature_names: list[str] = field(default_factory=list)
    ae_gmm_context_feature_names: list[str] = field(default_factory=list)
    ae_gmm_state: dict[str, Any] = field(default_factory=dict)
    model_effectiveness_history_defaults_: dict[str, float] = field(default_factory=dict)
    model_effectiveness_history_default_sources_: dict[str, str] = field(default_factory=dict)
    meta_leaf_lite_diagnostics_enabled: bool = False
    meta_leaf_support_diagnostics_enabled: bool = False
    meta_leaf_target_diagnostics_enabled: bool = False
    meta_leaf_centroid_diagnostics_enabled: bool = False
    meta_leaf_diagnostics_enabled: bool = False
    meta_contrib_diagnostics_enabled: bool = False
    meta_contrib_method: str = "path"
    meta_score_path_diagnostics_enabled: bool = True
    meta_drift_features_enabled: bool = True
    meta_context_features_enabled: bool = False
    meta_contrib_context_features_enabled: bool = False
    meta_raw_state_context_features_enabled: bool = False
    meta_base_error_context_features_enabled: bool = False
    label_weight_hpo_report_: dict[str, Any] = field(default_factory=dict)
    label_weight_hpo_soft_label_: Optional[np.ndarray] = None
    label_weight_hpo_hard_label_: Optional[np.ndarray] = None
    label_weight_hpo_sample_weight_: Optional[np.ndarray] = None

    def _frame(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X)
        X_df.columns = [str(c) for c in X_df.columns]
        selected = [str(c) for c in self.selected_features]
        ae_gmm_input_features = [
            str(c) for c in getattr(self, "ae_gmm_input_features", []) or []
        ]
        ae_gmm_feature_names = [
            str(c) for c in getattr(self, "ae_gmm_feature_names", []) or []
        ]
        ae_gmm_state = getattr(self, "ae_gmm_state", {}) or {}
        use_ae_gmm = bool(ae_gmm_input_features and ae_gmm_feature_names)
        raw_contrib_inputs = [
            str(c) for c in getattr(self, "raw_contrib_input_features", []) or []
        ]
        if raw_contrib_inputs:
            passthrough = [
                str(c)
                for c in getattr(self, "raw_contrib_passthrough_features", []) or []
            ]
            contract_features = list(dict.fromkeys(passthrough + raw_contrib_inputs))
            missing = [c for c in contract_features if c not in X_df.columns]
            if missing:
                preview = missing[:20]
                raise ValueError(
                    "LGBM inference feature contract violation: "
                    f"{len(missing)}/{len(contract_features)} raw/meta contribution "
                    f"contracted features are missing. Examples: {preview}"
                )
            try:
                passthrough_df = (
                    X_df.loc[:, passthrough].astype(np.float32, copy=False)
                    if passthrough
                    else pd.DataFrame(index=X_df.index)
                )
                contrib_df = _transform_raw_contrib_input_features(
                    X_df,
                    raw_contrib_inputs,
                    getattr(self, "raw_contrib_input_state", None),
                    output_feature_names=(
                        getattr(self, "raw_contrib_transformed_feature_names", None)
                        or META_RAW_CONTRIB_SVD_FEATURE_NAMES
                    ),
                    index=X_df.index,
                )
                out = pd.concat([passthrough_df, contrib_df], axis=1)
                if use_ae_gmm:
                    out = _append_ae_gmm_features_to_model_frame(
                        out,
                        ae_gmm_input_features,
                        ae_gmm_state,
                        selected,
                        index=X_df.index,
                    )
                else:
                    out = out.reindex(columns=selected, fill_value=0.0).astype(
                        np.float32,
                        copy=False,
                    )
            except Exception as exc:
                raise ValueError(
                    "LGBM inference feature contract violation: raw contribution "
                    f"features cannot be transformed: {exc}"
                ) from exc
            _validate_finite_contract_frame(out)
            return out
        input_features = [str(c) for c in getattr(self, "input_feature_names", []) or []]
        if use_ae_gmm:
            contract_features = list(ae_gmm_input_features)
            use_input_aliases = False
        else:
            use_input_aliases = len(input_features) == len(selected) and input_features != selected
            contract_features = input_features if use_input_aliases else selected
        if getattr(self, "model_effectiveness_history_defaults_", None):
            default_features = contract_features
            if use_input_aliases:
                default_features = list(
                    dict.fromkeys(list(contract_features) + list(selected))
                )
            X_df, _added_defaults, _filled_defaults = apply_model_effectiveness_history_defaults(
                X_df,
                default_features,
                getattr(self, "model_effectiveness_history_defaults_", {}) or {},
            )
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
        if getattr(self, "model_effectiveness_history_defaults_", None):
            fill_cols = list(out.columns)
            if use_input_aliases:
                fill_cols = list(dict.fromkeys(fill_cols + selected))
            out, _added_defaults, _filled_defaults = apply_model_effectiveness_history_defaults(
                out,
                fill_cols,
                getattr(self, "model_effectiveness_history_defaults_", {}) or {},
            )
        if use_ae_gmm:
            try:
                out = _append_ae_gmm_features_to_model_frame(
                    out,
                    contract_features,
                    ae_gmm_state,
                    selected,
                    index=X_df.index,
                )
            except Exception as exc:
                raise ValueError(
                    "LGBM inference feature contract violation: AE/GMM selected-feature "
                    f"representation cannot be transformed: {exc}"
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
        raw_contrib_inputs = [
            str(c) for c in getattr(self, "raw_contrib_input_features", []) or []
        ]
        input_features = [
            str(c) for c in getattr(self, "input_feature_names", []) or []
        ]
        ae_gmm_input_features = [
            str(c) for c in getattr(self, "ae_gmm_input_features", []) or []
        ]
        ae_gmm_feature_names = [
            str(c) for c in getattr(self, "ae_gmm_feature_names", []) or []
        ]
        use_ae_gmm = bool(ae_gmm_input_features and ae_gmm_feature_names)
        if raw_contrib_inputs:
            passthrough = [
                str(c)
                for c in getattr(self, "raw_contrib_passthrough_features", []) or []
            ]
            contract = list(dict.fromkeys(passthrough + raw_contrib_inputs))
        elif use_ae_gmm:
            contract = list(ae_gmm_input_features)
        else:
            contract = (
                input_features
                if len(input_features) == len(selected) and input_features != selected
                else selected
            )
        if getattr(self, "model_effectiveness_history_defaults_", None):
            default_features = contract
            if input_features and len(input_features) == len(selected) and input_features != selected:
                default_features = list(dict.fromkeys(list(contract) + list(selected)))
            X_df, _added_defaults, _filled_defaults = apply_model_effectiveness_history_defaults(
                X_df,
                default_features,
                getattr(self, "model_effectiveness_history_defaults_", {}) or {},
            )
            live_cols = set(map(str, X_df.columns))
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
            "raw_contrib_input_features_count": int(len(raw_contrib_inputs)),
            "ae_gmm_enabled": bool(use_ae_gmm),
            "ae_gmm_input_features_count": int(len(ae_gmm_input_features)),
            "ae_gmm_feature_count": int(
                len(getattr(self, "ae_gmm_feature_names", []) or [])
            ),
            "ae_gmm_context_feature_count": int(
                len(getattr(self, "ae_gmm_context_feature_names", []) or [])
            ),
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

    def transform_internal_model_metrics(self, X: Any) -> pd.DataFrame:
        X_df = self._frame(X)
        features = _lgbm_meta_features_from_models(
            self.models,
            X_df,
            mode=self.mode,
            rank_bin_stats=self.rank_bin_stats_oof,
            leaf_diagnostics=(
                getattr(self, "meta_leaf_lite_diagnostics_enabled", False)
                or getattr(self, "meta_leaf_diagnostics_enabled", False)
                or getattr(self, "meta_leaf_support_diagnostics_enabled", False)
                or getattr(self, "meta_leaf_target_diagnostics_enabled", False)
                or getattr(self, "meta_leaf_centroid_diagnostics_enabled", False)
            ),
            leaf_full_diagnostics=getattr(self, "meta_leaf_diagnostics_enabled", False),
            leaf_support_diagnostics=getattr(self, "meta_leaf_support_diagnostics_enabled", False),
            leaf_target_diagnostics=getattr(self, "meta_leaf_target_diagnostics_enabled", False),
            leaf_centroid_diagnostics=getattr(self, "meta_leaf_centroid_diagnostics_enabled", False),
            contrib_diagnostics=getattr(self, "meta_contrib_diagnostics_enabled", False),
            contrib_method=getattr(self, "meta_contrib_method", LGBM_META_CONTRIB_METHOD),
            score_path_diagnostics=getattr(self, "meta_score_path_diagnostics_enabled", True),
        )
        if getattr(self, "meta_drift_features_enabled", True):
            _append_feature_drift_meta_features(
                features,
                X_df,
                self.drift_reference,
            )
        if (
            getattr(self, "meta_contrib_context_features_enabled", False)
            or getattr(self, "meta_raw_state_context_features_enabled", False)
        ):
            _append_archetype_meta_features(
                features,
                self.models,
                X_df,
                contrib_state=getattr(self, "contrib_archetype_state", None),
                raw_state=getattr(self, "raw_state_archetype_state", None),
                contrib_enabled=getattr(self, "meta_contrib_context_features_enabled", False),
                raw_state_enabled=getattr(self, "meta_raw_state_context_features_enabled", False),
            )
            if getattr(self, "meta_contrib_context_features_enabled", False):
                _append_raw_contrib_export_features(
                    features,
                    self.models,
                    X_df,
                    getattr(self, "raw_contrib_feature_mapping", {}) or {},
                )
        if getattr(self, "meta_base_error_context_features_enabled", False):
            _append_base_error_archetype_features(
                features,
                getattr(self, "base_error_archetype_state", None),
            )
        ae_gmm_context_feature_names = [
            str(c)
            for c in (
                getattr(self, "ae_gmm_context_feature_names", [])
                or getattr(self, "ae_gmm_feature_names", [])
                or []
            )
        ]
        ae_gmm_input_features = [
            str(c) for c in getattr(self, "ae_gmm_input_features", []) or []
        ]
        if ae_gmm_context_feature_names and ae_gmm_input_features:
            ae_context = _append_ae_gmm_features_to_model_frame(
                X_df,
                ae_gmm_input_features,
                getattr(self, "ae_gmm_state", {}) or {},
                ae_gmm_context_feature_names,
                index=X_df.index,
            )
            for col in ae_gmm_context_feature_names:
                if col in ae_context.columns:
                    features[col] = ae_context[col].to_numpy(dtype=np.float32, copy=False)
        else:
            for col in ae_gmm_context_feature_names:
                if col in X_df.columns:
                    features[col] = pd.to_numeric(
                        X_df[col],
                        errors="coerce",
                    ).to_numpy(dtype=np.float32, copy=False)
        metric_names = list(
            dict.fromkeys(
                list(LGBM_INTERNAL_METRIC_FEATURE_NAMES)
                + (
                    list(LGBM_META_CONTRIB_CONTEXT_FEATURE_NAMES)
                    if getattr(self, "meta_contrib_context_features_enabled", False)
                    else []
                )
                + (
                    list(LGBM_META_RAW_STATE_CONTEXT_FEATURE_NAMES)
                    if getattr(self, "meta_raw_state_context_features_enabled", False)
                    else []
                )
                + (
                    list(LGBM_META_BASE_ERROR_CONTEXT_FEATURE_NAMES)
                    if getattr(self, "meta_base_error_context_features_enabled", False)
                    else []
                )
                + list(getattr(self, "raw_contrib_oof_feature_names", []) or [])
                + ae_gmm_context_feature_names
            )
        )
        return features.reindex(columns=metric_names, fill_value=0.0).astype(np.float32)

    def transform_meta_features(self, X: Any) -> pd.DataFrame:
        features = self.transform_internal_model_metrics(X)
        return features.reindex(columns=self.meta_feature_names, fill_value=0.0).astype(np.float32)

    def get_training_meta_features(self) -> pd.DataFrame:
        if self.meta_oof_features is None:
            return pd.DataFrame(columns=self.meta_feature_names, dtype=np.float32)
        return self.meta_oof_features.reindex(columns=self.meta_feature_names, fill_value=0.0).copy()

    def predict_tree_uncertainty_features(self, X: Any) -> dict[str, np.ndarray]:
        meta = self.transform_internal_model_metrics(X)
        out: dict[str, np.ndarray] = {}
        wanted = set(LGBM_INTERNAL_METRIC_FEATURE_NAMES)
        wanted.update(MODEL_DRIFT_FEATURE_KEYS)
        wanted.update(ARCHETYPE_FEATURE_NAMES)
        wanted.update(BASE_ERROR_ARCHETYPE_FEATURE_NAMES)
        wanted.update(str(c) for c in getattr(self, "raw_contrib_oof_feature_names", []) or [])
        wanted.update(
            str(c)
            for c in (
                getattr(self, "ae_gmm_context_feature_names", [])
                or getattr(self, "ae_gmm_feature_names", [])
                or []
            )
        )
        for col in meta.columns:
            col_s = str(col)
            if col_s in wanted or col_s.startswith(RAW_CONTRIB_FEATURE_PREFIX):
                out[col_s] = meta[col].to_numpy(dtype=np.float32, copy=True)
        return out


def _frame(X: Any) -> pd.DataFrame:
    X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    X_df = X_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    X_df.columns = [str(c) for c in X_df.columns]
    numeric_cols = X_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols):
        X_df[numeric_cols] = X_df[numeric_cols].astype(np.float32, copy=False)
    return X_df


def _lgbm_ae_gmm_enabled(cfg: dict[str, Any] | None = None) -> bool:
    if cfg is not None and "lgbm_ae_gmm_features_enabled" in cfg:
        return str(cfg.get("lgbm_ae_gmm_features_enabled")).strip().lower() not in {
            "0",
            "false",
            "no",
            "n",
            "off",
        }
    return bool(LGBM_AE_GMM_FEATURES)


def _ae_gmm_state_enabled(state: Any) -> bool:
    return isinstance(state, Mapping) and bool(state.get("enabled", False))


def _ae_gmm_model_feature_names_for_objective(objective_mode: str | None) -> list[str]:
    """Columns from the selected-feature AE/GMM transform that may enter LGBM.

    Selected-feature representations are generated by the current layer and are
    therefore exported only to downstream consumers. Older artifacts may still
    carry generated columns in ``selected_features``; inference keeps supporting
    those through ``ae_gmm_feature_names`` persisted in the artifact.
    """
    return []


def _ae_gmm_context_feature_names_for_objective(objective_mode: str | None) -> list[str]:
    """Columns exported downstream from the AE/GMM state."""
    return list(AE_GMM_FEATURE_COLUMNS)


def _append_ae_gmm_features_to_model_frame(
    base_frame: pd.DataFrame,
    input_features: list[str],
    state: dict[str, Any] | None,
    selected_features: list[str],
    *,
    index: Any = None,
) -> pd.DataFrame:
    input_cols = [str(c) for c in input_features]
    source = base_frame.reindex(columns=input_cols, fill_value=0.0).astype(
        np.float32,
        copy=False,
    )
    generated = transform_ae_gmm_features(
        source,
        state,
        index=source.index if index is None else index,
    )
    out = pd.concat([source, generated], axis=1)
    return out.reindex(columns=[str(c) for c in selected_features], fill_value=0.0).astype(
        np.float32,
        copy=False,
    )


def _fit_ae_gmm_post_selection_state(
    X_model_df: pd.DataFrame,
    input_features: list[str],
    fit_idx: np.ndarray,
    *,
    y_metric: np.ndarray | None = None,
    returns: np.ndarray | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    input_cols = [str(c) for c in input_features]
    idx = np.asarray(fit_idx, dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < len(X_model_df))]
    if len(idx) == 0:
        idx = np.arange(len(X_model_df), dtype=np.int64)
    economic_targets: dict[str, Any] = {}
    if y_metric is not None and len(y_metric) == len(X_model_df):
        economic_targets["target"] = np.asarray(y_metric, dtype=np.float32)[idx]
    if returns is not None and len(returns) == len(X_model_df):
        economic_targets["returns"] = np.asarray(returns, dtype=np.float32)[idx]
    return fit_ae_gmm_state(
        X_model_df.iloc[idx].reindex(columns=input_cols, fill_value=0.0),
        economic_targets=economic_targets,
        random_state=random_state,
        max_train_rows=int(LGBM_AE_GMM_MAX_TRAIN_ROWS),
        ae_max_iter=int(LGBM_AE_GMM_MAX_ITER),
    )


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


def _timestamp_ns(timestamps: Any, n: int) -> np.ndarray | None:
    if timestamps is None:
        return None
    arr = np.asarray(timestamps)
    if len(arr) != int(n):
        return None
    ts = pd.to_datetime(arr, utc=True, errors="coerce")
    valid = pd.Series(ts).notna().to_numpy(dtype=bool)
    if not bool(np.any(valid)):
        return None
    ns = pd.Series(ts).astype("int64").to_numpy(dtype=np.int64)
    ns[~valid] = np.iinfo(np.int64).min
    return ns


def _recency_half_life_days(objective_mode: str | None) -> float:
    recipe = load_recipe_from_env_or_cfg(
        None,
        scope="meta" if _normalize_objective_mode(objective_mode) == "train_meta" else "base",
    )
    if recipe is not None and str(getattr(recipe, "stage", "")).strip().lower() != "__label_weight_noop__":
        base_half_life = max(1e-6, float(recipe.weight.recency_half_life_days))
        if _normalize_objective_mode(objective_mode) == "train_meta":
            return base_half_life * 0.5
        return base_half_life
    return (
        float(LGBM_META_RECENCY_HALF_LIFE_DAYS)
        if _normalize_objective_mode(objective_mode) == "train_meta"
        else float(LGBM_BASE_RECENCY_HALF_LIFE_DAYS)
    )


def _recency_decay_from_timestamps(
    timestamps: Any,
    n: int,
    *,
    objective_mode: str | None,
) -> np.ndarray | None:
    if not bool(LGBM_RECENCY_WEIGHTING):
        return None
    ns = _timestamp_ns(timestamps, n)
    if ns is None:
        return None
    valid = ns != np.iinfo(np.int64).min
    latest_ns = int(np.max(ns[valid]))
    age_days = (float(latest_ns) - ns.astype(np.float64)) / float(24 * 3600 * 1_000_000_000)
    age_days[~valid] = float(np.nanmax(age_days[valid])) if bool(np.any(valid)) else 0.0
    half_life = _recency_half_life_days(objective_mode)
    decay = np.power(0.5, np.maximum(age_days, 0.0) / max(half_life, 1e-6))
    return np.clip(decay, 1e-6, 1.0).astype(np.float32)


def _apply_recency_sample_weight(
    base_weight: np.ndarray,
    timestamps: Any,
    *,
    objective_mode: str | None,
    cfg: dict[str, Any] | None = None,
) -> tuple[np.ndarray, bool]:
    w = np.asarray(base_weight, dtype=np.float32)
    hpo_decay, _hpo_active = recency_hpo_decay_from_config(
        timestamps,
        len(w),
        cfg=cfg,
        objective_mode=objective_mode,
    )
    if hpo_decay is not None:
        out, _ = _normalize_weights(w * hpo_decay)
        return out.astype(np.float32), True
    decay = _recency_decay_from_timestamps(timestamps, len(w), objective_mode=objective_mode)
    if decay is None:
        return w, False
    out, _ = _normalize_weights(w * decay)
    return out.astype(np.float32), True


def _recency_shrink_weight_towards_one(
    weights: np.ndarray,
    timestamps: Any,
    *,
    objective_mode: str | None,
    cfg: dict[str, Any] | None = None,
) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float32)
    hpo_decay, _hpo_active = recency_hpo_decay_from_config(
        timestamps,
        len(w),
        cfg=cfg,
        objective_mode=objective_mode,
    )
    if hpo_decay is not None:
        return (1.0 + (w - 1.0) * hpo_decay).astype(np.float32)
    decay = _recency_decay_from_timestamps(timestamps, len(w), objective_mode=objective_mode)
    if decay is None:
        return w
    return (1.0 + (w - 1.0) * decay).astype(np.float32)


def _cfg_env_bool(
    cfg: dict[str, Any] | None,
    key: str,
    env_key: str,
    default: bool,
) -> bool:
    raw = os.environ.get(env_key)
    if raw is None and isinstance(cfg, dict):
        raw = cfg.get(key)
    if raw is None or str(raw).strip() == "":
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _cfg_env_int(
    cfg: dict[str, Any] | None,
    key: str,
    env_key: str,
    default: int,
) -> int:
    raw = os.environ.get(env_key)
    if raw is None and isinstance(cfg, dict):
        raw = cfg.get(key)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _cfg_env_float(
    cfg: dict[str, Any] | None,
    key: str,
    env_key: str,
    default: float,
) -> float:
    raw = os.environ.get(env_key)
    if raw is None and isinstance(cfg, dict):
        raw = cfg.get(key)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _cfg_value(cfg: dict[str, Any] | None, key: str, default: Any = None) -> Any:
    raw = os.environ.get(f"EPM_{key.upper()}")
    if raw is None and isinstance(cfg, dict):
        raw = cfg.get(key)
    return default if raw is None else raw


def _cfg_bool_value(cfg: dict[str, Any] | None, key: str, default: bool = False) -> bool:
    raw = _cfg_value(cfg, key, default)
    if isinstance(raw, bool):
        return bool(raw)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _cfg_float_value(cfg: dict[str, Any] | None, key: str, default: float) -> float:
    raw = _cfg_value(cfg, key, default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _cfg_int_value(cfg: dict[str, Any] | None, key: str, default: int) -> int:
    raw = _cfg_value(cfg, key, default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _lgbm_regime_specialist_objective_allowed(
    cfg: dict[str, Any] | None,
    objective_mode: str | None,
) -> bool:
    mode = _normalize_objective_mode(objective_mode)
    raw = _cfg_value(cfg, "lgbm_regime_specialist_objectives", ["train_base", "train_meta"])
    if isinstance(raw, str):
        allowed = {part.strip().lower() for part in raw.split(",") if part.strip()}
    else:
        try:
            allowed = {str(part).strip().lower() for part in raw if str(part).strip()}
        except Exception:
            allowed = {"train_base", "train_meta"}
    return mode in allowed


def _lgbm_regime_specialist_enabled(
    cfg: dict[str, Any] | None,
    objective_mode: str | None,
) -> bool:
    if not _cfg_bool_value(cfg, "lgbm_regime_specialist_enabled", False):
        return False
    return _lgbm_regime_specialist_objective_allowed(cfg, objective_mode)


def _lgbm_regime_specialist_feature_engineering_diagnostics_enabled(
    cfg: dict[str, Any] | None,
    objective_mode: str | None,
) -> bool:
    if not _cfg_bool_value(
        cfg,
        "lgbm_regime_specialist_feature_engineering_diagnostics_enabled",
        True,
    ):
        return False
    return _lgbm_regime_specialist_objective_allowed(cfg, objective_mode)


def _lgbm_regime_specialist_should_build_bundle(
    cfg: dict[str, Any] | None,
    objective_mode: str | None,
) -> bool:
    return _lgbm_regime_specialist_enabled(
        cfg,
        objective_mode,
    ) or _lgbm_regime_specialist_feature_engineering_diagnostics_enabled(
        cfg,
        objective_mode,
    )


def _regime_specialist_similarity_config(cfg: dict[str, Any] | None):
    from extreme_price_movements.regime_specialist_similarity import RegimeSimilarityConfig

    kwargs: dict[str, Any] = {}
    defaults = RegimeSimilarityConfig()
    for name in RegimeSimilarityConfig.__dataclass_fields__:
        key = f"lgbm_regime_specialist_{name}"
        default = getattr(defaults, name)
        raw = _cfg_value(cfg, key, default)
        if isinstance(default, bool):
            kwargs[name] = str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}
        elif isinstance(default, int) and not isinstance(default, bool):
            kwargs[name] = _cfg_int_value(cfg, key, int(default))
        elif isinstance(default, float):
            kwargs[name] = _cfg_float_value(cfg, key, float(default))
        else:
            kwargs[name] = raw
    return RegimeSimilarityConfig(**kwargs)


def _regime_specialist_weight_config(cfg: dict[str, Any] | None):
    from extreme_price_movements.regime_specialist_similarity import SpecialistWeightConfig

    defaults = SpecialistWeightConfig()
    kwargs: dict[str, Any] = {}
    alias = {
        "min_weight": "lgbm_regime_specialist_weight_min",
        "max_weight": "lgbm_regime_specialist_weight_max",
    }
    for name in SpecialistWeightConfig.__dataclass_fields__:
        key = alias.get(name, f"lgbm_regime_specialist_{name}")
        default = getattr(defaults, name)
        if isinstance(default, bool):
            kwargs[name] = _cfg_bool_value(cfg, key, bool(default))
        elif isinstance(default, int) and not isinstance(default, bool):
            kwargs[name] = _cfg_int_value(cfg, key, int(default))
        else:
            kwargs[name] = _cfg_float_value(cfg, key, float(default))
    return SpecialistWeightConfig(**kwargs)


_REGIME_SPECIALIST_FE_DIAGNOSTIC_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_REGIME_SPECIALIST_FE_DIAGNOSTIC_CACHE_MAX = 8
_REGIME_SPECIALIST_SCORE_FEATURE_CACHE: dict[tuple[Any, ...], tuple[pd.DataFrame, dict[str, Any]]] = {}
_REGIME_SPECIALIST_SCORE_FEATURE_CACHE_MAX = 4
_REGIME_SPECIALIST_UNSUPERVISED_CACHE: dict[tuple[Any, ...], tuple[Any, dict[str, Any]]] = {}
_REGIME_SPECIALIST_UNSUPERVISED_CACHE_MAX = 4
_LGBM_REGIME_SCORE_FEATURE_NAME = "regime_lgbm_elasticnet_similarity"
_LGBM_REGIME_SCORE_SOURCE_COLUMN = "regime_domain_current_likeness"
_LGBM_REGIME_SCORE_FEATURE_SOURCES: tuple[tuple[str, str], ...] = (
    ("regime_lgbm_current_likeness", "regime_lgbm_current_likeness"),
    ("regime_elasticnet_current_likeness", "regime_elasticnet_current_likeness"),
    (_LGBM_REGIME_SCORE_FEATURE_NAME, _LGBM_REGIME_SCORE_SOURCE_COLUMN),
)


def _unsupervised_regime_learning_cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    from extreme_price_movements.unsupervised_regime_learning.feature_registry import (
        UNSUPERVISED_REGIME_LEARNING_DEFAULTS,
    )

    out: dict[str, Any] = dict(UNSUPERVISED_REGIME_LEARNING_DEFAULTS)
    raw = cfg.get("UNSUPERVISED_REGIME_LEARNING") if isinstance(cfg, dict) else None
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
                nested = dict(out[key])
                nested.update(dict(value))
                out[key] = nested
            else:
                out[key] = value
    regime = dict(out.get("regime_models", {}) or {})
    flat_enabled = _cfg_value(cfg, "lgbm_unsupervised_regime_learning_enabled", None)
    if flat_enabled is not None:
        regime["enabled"] = str(flat_enabled).strip().lower() in {"1", "true", "yes", "y", "on"}
    elif "enabled" in regime and not isinstance(regime.get("enabled"), bool):
        regime["enabled"] = str(regime.get("enabled")).strip().lower() in {"1", "true", "yes", "y", "on"}
    out["regime_models"] = regime
    return out


def _coerce_advanced_regime_config_value(default: Any, value: Any) -> Any:
    if isinstance(default, bool):
        if isinstance(value, bool):
            return bool(value)
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(default, int) and not isinstance(default, bool):
        try:
            return int(value)
        except Exception:
            return int(default)
    if isinstance(default, float):
        try:
            return float(value)
        except Exception:
            return float(default)
    return value


def _lgbm_frame_feature_fingerprint(
    frame: pd.DataFrame,
    features: Sequence[str],
) -> tuple[int, int, int]:
    cols = [col for col in ["timestamp", "symbol"] if col in frame.columns]
    cols.extend(str(feature) for feature in features if str(feature) in frame.columns)
    cols = list(dict.fromkeys(cols))
    if not cols:
        return (int(len(frame)), 0, 0)
    try:
        hashes = pd.util.hash_pandas_object(frame.loc[:, cols], index=True).to_numpy(dtype=np.uint64, copy=False)
        if hashes.size == 0:
            return (0, 0, 0)
        xor_value = int(np.bitwise_xor.reduce(hashes))
        sum_value = int(np.sum(hashes, dtype=np.uint64))
        return (int(hashes.size), xor_value, sum_value)
    except Exception:
        numeric = frame.reindex(columns=[col for col in features if col in frame.columns])
        arr = numeric.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64, copy=False)
        finite = np.nan_to_num(arr, nan=0.0, posinf=1e12, neginf=-1e12)
        return (
            int(finite.size),
            int(np.round(float(np.sum(finite)) * 1e6)),
            int(np.round(float(np.sum(finite * finite)) * 1e6)),
        )


def _lgbm_unsupervised_regime_artifact(
    frame: pd.DataFrame,
    candidate_features: Sequence[str],
    *,
    cfg: dict[str, Any] | None,
    random_state: int,
) -> tuple[Any | None, dict[str, Any]]:
    active = _unsupervised_regime_learning_cfg(cfg)
    regime_cfg = dict(active.get("regime_models", {}) or {})
    if not bool(regime_cfg.get("enabled", False)):
        return None, {"enabled": False, "used": False, "reason": "disabled"}
    features = [str(c) for c in dict.fromkeys(candidate_features) if str(c) in frame.columns]
    if not features:
        return None, {"enabled": True, "used": False, "reason": "no_candidate_features"}
    try:
        from extreme_price_movements.unsupervised_regime_learning.regime_models import (
            AdvancedRegimeLearningConfig,
            fit_advanced_regime_learning,
            regime_artifact_assessment_summary,
            save_advanced_regime_learning_artifact,
        )
    except Exception as exc:
        return None, {"enabled": True, "used": False, "reason": f"import_failed:{exc}"}

    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce") if "timestamp" in frame.columns else pd.Series(pd.NaT, index=frame.index)
    start_val = ts.min()
    end_val = ts.max()
    cfg_fingerprint = tuple(
        sorted(
            (str(k), str(v))
            for k, v in regime_cfg.items()
            if k not in {"artifact_output_dir"}
        )
    )
    cache_key = (
        int(len(frame)),
        int(pd.Timestamp(start_val).value) if pd.notna(start_val) else 0,
        int(pd.Timestamp(end_val).value) if pd.notna(end_val) else 0,
        tuple(features),
        _lgbm_frame_feature_fingerprint(frame, features),
        int(random_state),
        cfg_fingerprint,
    )
    cached = _REGIME_SPECIALIST_UNSUPERVISED_CACHE.get(cache_key)
    if cached is not None:
        artifact, diag = cached
        out = dict(diag)
        out["cache_hit"] = True
        output_dir = str(regime_cfg.get("artifact_output_dir", "") or "").strip()
        if output_dir:
            try:
                from extreme_price_movements.unsupervised_regime_learning.regime_models import (
                    save_advanced_regime_learning_artifact,
                )

                out["saved_paths"] = save_advanced_regime_learning_artifact(artifact, output_dir)
            except Exception as exc:
                out["save_failed"] = str(exc)
        return artifact, out
    try:
        fields = set(AdvancedRegimeLearningConfig.__dataclass_fields__)
        defaults = AdvancedRegimeLearningConfig()
        values = {
            key: _coerce_advanced_regime_config_value(getattr(defaults, key), value)
            for key, value in regime_cfg.items()
            if key in fields
        }
        values["timestamp_col"] = "timestamp"
        values["symbol_col"] = "symbol"
        values["random_state"] = int(random_state)
        artifact = fit_advanced_regime_learning(
            frame,
            features,
            config=AdvancedRegimeLearningConfig(**values),
        )
        output_dir = str(regime_cfg.get("artifact_output_dir", "") or "").strip()
        saved_paths: dict[str, str] = {}
        if output_dir:
            saved_paths = save_advanced_regime_learning_artifact(artifact, output_dir)
        assessment_summary = regime_artifact_assessment_summary(artifact)
        diag = {
            "enabled": True,
            "used": False,
            "reason": "assessment_only_not_injected",
            "candidate_feature_count": int(len(features)),
            "selected_feature_count": int(len(getattr(artifact, "selected_features", []) or [])),
            "specialist_candidate_feature_count": int(len(getattr(artifact, "specialist_candidate_features", []) or [])),
            "kept_methods": list((getattr(artifact, "diagnostics", {}) or {}).get("kept_methods", [])),
            "baseline_score": (getattr(artifact, "diagnostics", {}) or {}).get("baseline_score"),
            "top_method": assessment_summary.get("top_method"),
            "top_total_score": assessment_summary.get("top_total_score"),
            "assessment": assessment_summary.get("assessment", {}),
            "cache_hit": False,
            "saved_paths": saved_paths,
        }
        if len(_REGIME_SPECIALIST_UNSUPERVISED_CACHE) >= _REGIME_SPECIALIST_UNSUPERVISED_CACHE_MAX:
            _REGIME_SPECIALIST_UNSUPERVISED_CACHE.pop(next(iter(_REGIME_SPECIALIST_UNSUPERVISED_CACHE)))
        _REGIME_SPECIALIST_UNSUPERVISED_CACHE[cache_key] = (artifact, dict(diag))
        return artifact, diag
    except Exception as exc:
        return None, {"enabled": True, "used": False, "reason": f"failed:{exc}"}


def _regime_specialist_feature_engineering_config(
    cfg: dict[str, Any] | None,
    *,
    random_state: int,
):
    from extreme_price_movements.regime_specialist_feature_engineering import (
        RegimeFeatureEngineeringConfig,
    )

    defaults = RegimeFeatureEngineeringConfig()
    kwargs: dict[str, Any] = {}
    for name in RegimeFeatureEngineeringConfig.__dataclass_fields__:
        default = getattr(defaults, name)
        key = f"lgbm_regime_specialist_feature_engineering_{name}"
        if name == "random_state":
            kwargs[name] = int(random_state)
        elif isinstance(default, bool):
            kwargs[name] = _cfg_bool_value(cfg, key, bool(default))
        elif isinstance(default, int) and not isinstance(default, bool):
            kwargs[name] = _cfg_int_value(cfg, key, int(default))
        elif isinstance(default, float):
            kwargs[name] = _cfg_float_value(cfg, key, float(default))
        else:
            kwargs[name] = _cfg_value(cfg, key, default)
    return RegimeFeatureEngineeringConfig(**kwargs)


def _top_score_items(score_map: Mapping[str, Any] | None, *, limit: int = 40) -> list[dict[str, Any]]:
    if not isinstance(score_map, Mapping):
        return []
    items: list[tuple[str, float]] = []
    for key, value in score_map.items():
        try:
            score = float(value)
        except Exception:
            continue
        if np.isfinite(score):
            items.append((str(key), score))
    items.sort(key=lambda item: item[1], reverse=True)
    return [{"feature": key, "score": score} for key, score in items[: int(limit)]]


def _feature_report_records(report: Any, *, limit: int = 80) -> list[dict[str, Any]]:
    if not isinstance(report, pd.DataFrame) or report.empty:
        return []
    preferred = [
        "feature",
        "univariate_score",
        "auc_lift_mean",
        "auc_lift_std",
        "ks_mean",
        "ks_std",
        "median_shift_mean",
        "median_shift_std",
        "sign_consistency",
        "fold_pass_rate",
        "selected_univariate",
    ]
    cols = [col for col in preferred if col in report.columns]
    if "univariate_score" in report.columns:
        view = report.sort_values("univariate_score", ascending=False)
    else:
        view = report
    return _json_sanitize(view.loc[:, cols].head(int(limit)).to_dict(orient="records"))


def _regime_specialist_feature_engineering_summary(artifact: Any) -> dict[str, Any]:
    diagnostics = getattr(artifact, "diagnostics", {}) or {}
    row_scores = getattr(artifact, "row_scores", None)
    row_score_summary: dict[str, Any] = {}
    if isinstance(row_scores, pd.DataFrame) and not row_scores.empty:
        for col in row_scores.columns:
            arr = pd.to_numeric(row_scores[col], errors="coerce").to_numpy(dtype=np.float64)
            row_score_summary[str(col)] = {
                "mean": float(np.nanmean(arr)) if arr.size else float("nan"),
                "std": float(np.nanstd(arr)) if arr.size else float("nan"),
                "p10": float(np.nanpercentile(arr, 10.0)) if arr.size else float("nan"),
                "p90": float(np.nanpercentile(arr, 90.0)) if arr.size else float("nan"),
            }
    summary = {
        "enabled": True,
        "used": False,
        "diagnostic_only": True,
        "reason": "diagnostics_only",
        "schema_version": str(getattr(artifact, "schema_version", "")),
        "selected_features": list(getattr(artifact, "selected_features", []) or []),
        "selected_raw_features": list(getattr(artifact, "selected_raw_features", []) or []),
        "selected_pair_features": list(getattr(artifact, "selected_pair_features", []) or []),
        "selected_drift_features": list(getattr(artifact, "selected_drift_features", []) or []),
        "selected_feature_count": int(len(getattr(artifact, "selected_features", []) or [])),
        "selected_raw_feature_count": int(len(getattr(artifact, "selected_raw_features", []) or [])),
        "selected_pair_feature_count": int(len(getattr(artifact, "selected_pair_features", []) or [])),
        "selected_drift_feature_count": int(len(getattr(artifact, "selected_drift_features", []) or [])),
        "lgbm_features": list(getattr(artifact, "lgbm_features", []) or []),
        "elasticnet_features": list(getattr(artifact, "elasticnet_features", []) or []),
        "lgbm_feature_count": int(len(getattr(artifact, "lgbm_features", []) or [])),
        "elasticnet_feature_count": int(len(getattr(artifact, "elasticnet_features", []) or [])),
        "top_final_features": _top_score_items(
            getattr(artifact, "final_feature_scores", {}) or {},
            limit=40,
        ),
        "top_lgbm_features": _top_score_items(
            getattr(artifact, "lgbm_feature_scores", {}) or {},
            limit=40,
        ),
        "top_elasticnet_features": _top_score_items(
            getattr(artifact, "elasticnet_feature_scores", {}) or {},
            limit=40,
        ),
        "feature_report_top": _feature_report_records(
            getattr(artifact, "feature_report", None),
            limit=80,
        ),
        "row_score_summary": row_score_summary,
        "diagnostics": diagnostics,
    }
    return _json_sanitize(summary)


def _regime_specialist_feature_engineering_metric_summary(
    feature_engineering_diag: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(feature_engineering_diag, Mapping):
        return {
            "regime_specialist_feature_engineering_diagnostics_enabled": False,
            "regime_specialist_feature_engineering_reason": "unavailable",
        }
    diagnostics = feature_engineering_diag.get("diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        diagnostics = {}
    lgbm_diag = diagnostics.get("lgbm", {})
    elastic_diag = diagnostics.get("elasticnet", {})
    validation = diagnostics.get("validation", {})
    if not isinstance(lgbm_diag, Mapping):
        lgbm_diag = {}
    if not isinstance(elastic_diag, Mapping):
        elastic_diag = {}
    if not isinstance(validation, Mapping):
        validation = {}

    def _validation_value(block: str, key: str) -> float:
        raw = validation.get(block, {})
        if not isinstance(raw, Mapping):
            return float("nan")
        try:
            return float(raw.get(key, float("nan")))
        except Exception:
            return float("nan")

    def _float_metric(source: Mapping[str, Any], key: str) -> float:
        try:
            return float(source.get(key, float("nan")))
        except Exception:
            return float("nan")

    out = {
        "regime_specialist_feature_engineering_diagnostics_enabled": bool(
            feature_engineering_diag.get("enabled", False)
        ),
        "regime_specialist_feature_engineering_diagnostic_only": bool(
            feature_engineering_diag.get("diagnostic_only", False)
        ),
        "regime_specialist_feature_engineering_used_in_similarity": bool(
            feature_engineering_diag.get("used", False)
        ),
        "regime_specialist_feature_engineering_reason": str(
            feature_engineering_diag.get("reason", "")
        ),
        "regime_specialist_feature_engineering_selected_feature_count": int(
            feature_engineering_diag.get("selected_feature_count", 0) or 0
        ),
        "regime_specialist_feature_engineering_selected_raw_feature_count": int(
            feature_engineering_diag.get("selected_raw_feature_count", 0) or 0
        ),
        "regime_specialist_feature_engineering_selected_pair_feature_count": int(
            feature_engineering_diag.get("selected_pair_feature_count", 0) or 0
        ),
        "regime_specialist_feature_engineering_selected_drift_feature_count": int(
            feature_engineering_diag.get("selected_drift_feature_count", 0) or 0
        ),
        "regime_specialist_feature_engineering_lgbm_feature_count": int(
            len(feature_engineering_diag.get("lgbm_features", []) or [])
        ),
        "regime_specialist_feature_engineering_elasticnet_feature_count": int(
            len(feature_engineering_diag.get("elasticnet_features", []) or [])
        ),
        "regime_specialist_feature_engineering_lgbm_enabled": bool(lgbm_diag.get("enabled", False)),
        "regime_specialist_feature_engineering_lgbm_oof_rows": int(lgbm_diag.get("oof_rows", 0) or 0),
        "regime_specialist_feature_engineering_lgbm_fold_auc_lift_mean": _float_metric(
            lgbm_diag,
            "fold_auc_lift_mean",
        ),
        "regime_specialist_feature_engineering_lgbm_fold_auc_lift_std": _float_metric(
            lgbm_diag,
            "fold_auc_lift_std",
        ),
        "regime_specialist_feature_engineering_elasticnet_enabled": bool(elastic_diag.get("enabled", False)),
        "regime_specialist_feature_engineering_elasticnet_oof_rows": int(elastic_diag.get("oof_rows", 0) or 0),
        "regime_specialist_feature_engineering_elasticnet_fold_auc_lift_mean": _float_metric(
            elastic_diag,
            "fold_auc_lift_mean",
        ),
        "regime_specialist_feature_engineering_elasticnet_fold_auc_lift_std": _float_metric(
            elastic_diag,
            "fold_auc_lift_std",
        ),
        "regime_specialist_feature_engineering_validation_enabled": bool(validation.get("enabled", False)),
        "regime_specialist_feature_engineering_validation_raw_auc_lift_mean": _validation_value("raw", "mean"),
        "regime_specialist_feature_engineering_validation_drift_auc_lift_mean": _validation_value("drift", "mean"),
        "regime_specialist_feature_engineering_validation_raw_plus_drift_auc_lift_mean": _validation_value("raw_plus_drift", "mean"),
    }
    return out


def _lgbm_regime_score_features_enabled(
    cfg: dict[str, Any] | None,
    objective_mode: str | None,
) -> bool:
    if not _cfg_bool_value(cfg, "lgbm_regime_score_features_enabled", False):
        return False
    return _lgbm_regime_specialist_objective_allowed(cfg, objective_mode)


def _is_lgbm_regime_score_feature_name(name: str) -> bool:
    value = str(name)
    score_names = {feature for feature, _source in _LGBM_REGIME_SCORE_FEATURE_SOURCES}
    return value in score_names or any(value.startswith(f"{feature}__") for feature in score_names)


def _lgbm_regime_score_feature_metric_summary(
    diag: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(diag, Mapping):
        return {
            "regime_score_features_enabled": False,
            "regime_score_features_added_count": 0,
            "regime_score_features_reason": "unavailable",
        }
    out = {
        "regime_score_features_enabled": bool(diag.get("enabled", False)),
        "regime_score_features_added_count": int(
            len(diag.get("added_feature_names", []) or [])
        ),
        "regime_score_features_reason": str(diag.get("reason", "")),
        "regime_score_features_source_feature_count": int(
            diag.get("source_feature_count", 0) or 0
        ),
        "regime_score_features_current_rows": int(diag.get("current_rows", 0) or 0),
        "regime_score_features_historical_rows": int(
            diag.get("historical_rows", 0) or 0
        ),
        "regime_score_features_cache_hit": bool(diag.get("cache_hit", False)),
    }
    for feature_name, _source_col in _LGBM_REGIME_SCORE_FEATURE_SOURCES:
        metric_prefix = feature_name.replace("regime_", "regime_score_", 1)
        for key in ("mean", "std", "p10", "p90"):
            raw = diag.get(f"{feature_name}_{key}")
            try:
                out[f"{metric_prefix}_{key}"] = float(raw)
            except Exception:
                out[f"{metric_prefix}_{key}"] = float("nan")
    fe_summary = diag.get("feature_engineering", {})
    if isinstance(fe_summary, Mapping):
        out.update(_regime_specialist_feature_engineering_metric_summary(fe_summary))
        out["regime_specialist_feature_engineering_used_as_model_feature"] = bool(
            diag.get("used_as_model_feature", False)
        )
    return out


def _append_lgbm_regime_score_features(
    X_df: pd.DataFrame,
    selected_features: Sequence[str],
    *,
    timestamps: Any = None,
    assets: Any = None,
    objective_mode: str | None,
    cfg: dict[str, Any] | None,
    random_state: int,
    label: str,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    selected = [str(c) for c in selected_features if str(c).strip()]
    if not _lgbm_regime_score_features_enabled(cfg, objective_mode):
        return X_df, selected, {"enabled": False, "reason": "disabled"}
    n = len(X_df)
    diag_base: dict[str, Any] = {
        "enabled": True,
        "label": str(label),
        "objective_mode": _normalize_objective_mode(objective_mode),
        "target_feature_names": [
            feature for feature, _source in _LGBM_REGIME_SCORE_FEATURE_SOURCES
        ],
        "source_score_columns": [
            source for _feature, source in _LGBM_REGIME_SCORE_FEATURE_SOURCES
        ],
    }
    source_features = [
        feature
        for feature in selected
        if feature in X_df.columns and not _is_lgbm_regime_score_feature_name(feature)
    ]
    if len(source_features) < 2:
        out = dict(diag_base)
        out.update(
            {
                "reason": "insufficient_source_features",
                "source_feature_count": int(len(source_features)),
            }
        )
        return X_df, selected, out
    try:
        from extreme_price_movements.regime_specialist_feature_engineering import (
            build_regime_specialist_feature_engineering_artifact,
            build_regime_specialist_frozen_feature_score_artifact,
        )
    except Exception as exc:
        out = dict(diag_base)
        out.update({"reason": f"import_failed:{exc}", "source_feature_count": int(len(source_features))})
        return X_df, selected, out

    try:
        frame = _lgbm_regime_specialist_build_frame(
            X_df,
            source_features,
            timestamps=timestamps,
            assets=assets,
        )
        if "timestamp" not in frame.columns:
            out = dict(diag_base)
            out.update({"reason": "missing_timestamps", "source_feature_count": int(len(source_features))})
            return X_df, selected, out
        ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        valid_ts = ts.notna()
        if not bool(valid_ts.any()):
            out = dict(diag_base)
            out.update({"reason": "missing_valid_timestamps", "source_feature_count": int(len(source_features))})
            return X_df, selected, out
        current_end_raw = _cfg_value(cfg, "lgbm_regime_specialist_current_end", None)
        end = (
            pd.to_datetime(current_end_raw, utc=True, errors="coerce")
            if current_end_raw is not None
            else ts.max()
        )
        if pd.isna(end):
            end = ts.max()
        current_days = float(
            _cfg_float_value(cfg, "lgbm_regime_specialist_current_window_days", 28.0)
        )
        current_start = end - pd.Timedelta(days=current_days)
        current_mask = (ts >= current_start) & (ts <= end)
        historical_cutoff = current_start - pd.Timedelta(
            days=max(0.0, _cfg_float_value(cfg, "lgbm_regime_specialist_embargo_days", 0.0))
        )
        historical_mask = valid_ts & (ts < historical_cutoff)
        min_current_rows = _cfg_int_value(cfg, "lgbm_regime_specialist_min_current_rows", 24)
        if int(current_mask.sum()) < int(min_current_rows):
            out = dict(diag_base)
            out.update(
                {
                    "reason": "insufficient_current_rows",
                    "source_feature_count": int(len(source_features)),
                    "current_rows": int(current_mask.sum()),
                    "historical_rows": int(historical_mask.sum()),
                }
            )
            return X_df, selected, out
        if int(historical_mask.sum()) < 200:
            out = dict(diag_base)
            out.update(
                {
                    "reason": "insufficient_historical_rows",
                    "source_feature_count": int(len(source_features)),
                    "current_rows": int(current_mask.sum()),
                    "historical_rows": int(historical_mask.sum()),
                }
            )
            return X_df, selected, out
        start_val = ts.loc[valid_ts].min()
        end_val = ts.loc[valid_ts].max()
        cache_key = (
            "score_feature",
            _normalize_objective_mode(objective_mode),
            int(n),
            int(pd.Timestamp(start_val).value) if pd.notna(start_val) else 0,
            int(pd.Timestamp(end_val).value) if pd.notna(end_val) else 0,
            tuple(source_features),
            _lgbm_frame_feature_fingerprint(frame, source_features),
            int(random_state),
            int(_cfg_int_value(cfg, "lgbm_regime_specialist_feature_engineering_max_final_features", 40)),
            bool(_cfg_bool_value(cfg, "lgbm_regime_specialist_feature_engineering_lgbm_enabled", True)),
            bool(_cfg_bool_value(cfg, "lgbm_regime_specialist_feature_engineering_elasticnet_enabled", True)),
            bool(_cfg_bool_value(cfg, "lgbm_regime_score_feature_full_engineering_enabled", False)),
        )
        cached = _REGIME_SPECIALIST_SCORE_FEATURE_CACHE.get(cache_key)
        if cached is None:
            full_feature_engineering = _cfg_bool_value(
                cfg,
                "lgbm_regime_score_feature_full_engineering_enabled",
                False,
            )
            tprint(
                "LGBM regime score feature build started: "
                f"label={label}, objective={_normalize_objective_mode(objective_mode)}, "
                f"rows={n}, source_features={len(source_features)}, "
                f"current_rows={int(current_mask.sum())}, historical_rows={int(historical_mask.sum())}, "
                f"mode={'full_feature_engineering' if full_feature_engineering else 'frozen_feature_scores'}."
            )
            fe_config = _regime_specialist_feature_engineering_config(
                cfg,
                random_state=random_state,
            )
            if full_feature_engineering:
                artifact = build_regime_specialist_feature_engineering_artifact(
                    frame,
                    timestamp_col="timestamp",
                    symbol_col="symbol",
                    candidate_features=source_features,
                    unsupervised_regime_artifact=None,
                    current_mask=current_mask.to_numpy(dtype=bool),
                    historical_mask=historical_mask.to_numpy(dtype=bool),
                    config=fe_config,
                )
            else:
                artifact = build_regime_specialist_frozen_feature_score_artifact(
                    frame,
                    timestamp_col="timestamp",
                    symbol_col="symbol",
                    candidate_features=source_features,
                    current_mask=current_mask.to_numpy(dtype=bool),
                    historical_mask=historical_mask.to_numpy(dtype=bool),
                    config=fe_config,
                )
            row_scores = getattr(artifact, "row_scores", pd.DataFrame())
            available_sources = (
                {
                    source_col
                    for _feature_name, source_col in _LGBM_REGIME_SCORE_FEATURE_SOURCES
                    if isinstance(row_scores, pd.DataFrame) and source_col in row_scores.columns
                }
                if isinstance(row_scores, pd.DataFrame)
                else set()
            )
            if not available_sources:
                out = dict(diag_base)
                out.update(
                    {
                        "reason": "missing_source_score_column",
                        "source_feature_count": int(len(source_features)),
                        "current_rows": int(current_mask.sum()),
                        "historical_rows": int(historical_mask.sum()),
                        "missing_source_score_columns": [
                            source
                            for _feature_name, source in _LGBM_REGIME_SCORE_FEATURE_SOURCES
                        ],
                        "feature_engineering": _regime_specialist_feature_engineering_summary(artifact),
                    }
                )
                return X_df, selected, out
            score_columns: dict[str, np.ndarray] = {}
            missing_score_sources: list[str] = []
            for feature_name, source_col in _LGBM_REGIME_SCORE_FEATURE_SOURCES:
                if source_col not in available_sources:
                    missing_score_sources.append(source_col)
                    continue
                score = pd.to_numeric(
                    row_scores[source_col],
                    errors="coerce",
                ).reindex(frame.index)
                fill_value = (
                    float(np.nanmean(score.to_numpy(dtype=np.float64)))
                    if score.notna().any()
                    else 0.5
                )
                score = score.fillna(fill_value).clip(0.0, 1.0).astype(np.float32)
                score_columns[feature_name] = score.to_numpy(dtype=np.float32)
            scores_df = pd.DataFrame(score_columns, index=X_df.index)
            fe_summary = _regime_specialist_feature_engineering_summary(artifact)
            diag = dict(diag_base)
            diag.update(
                {
                    "reason": "built",
                    "used_as_model_feature": True,
                    "added_feature_names": list(scores_df.columns),
                    "missing_source_score_columns": missing_score_sources,
                    "source_feature_count": int(len(source_features)),
                    "current_rows": int(current_mask.sum()),
                    "historical_rows": int(historical_mask.sum()),
                    "cache_hit": False,
                    "feature_engineering": fe_summary,
                }
            )
            for feature_name in scores_df.columns:
                score_arr = scores_df[feature_name].to_numpy(dtype=np.float64)
                diag.update(
                    {
                        f"{feature_name}_mean": float(np.nanmean(score_arr)),
                        f"{feature_name}_std": float(np.nanstd(score_arr)),
                        f"{feature_name}_p10": float(np.nanpercentile(score_arr, 10.0)),
                        f"{feature_name}_p90": float(np.nanpercentile(score_arr, 90.0)),
                    }
                )
            if len(_REGIME_SPECIALIST_SCORE_FEATURE_CACHE) >= _REGIME_SPECIALIST_SCORE_FEATURE_CACHE_MAX:
                _REGIME_SPECIALIST_SCORE_FEATURE_CACHE.pop(
                    next(iter(_REGIME_SPECIALIST_SCORE_FEATURE_CACHE))
                )
            _REGIME_SPECIALIST_SCORE_FEATURE_CACHE[cache_key] = (scores_df, dict(diag))
            del artifact
            gc.collect()
        else:
            scores_df, cached_diag = cached
            diag = dict(cached_diag)
            diag["cache_hit"] = True
        X_out = X_df.copy(deep=False)
        for col in scores_df.columns:
            X_out[col] = scores_df[col].reindex(X_df.index).fillna(0.5).to_numpy(
                dtype=np.float32,
                copy=False,
            )
        selected_out = list(
            dict.fromkeys(
                [feature for feature in selected if feature in X_out.columns]
                + [str(col) for col in scores_df.columns]
            )
        )
        diag["added_feature_names"] = [str(col) for col in scores_df.columns]
        diag["used_as_model_feature"] = True
        tprint(
            "LGBM regime score feature ready: "
            f"label={label}, features={list(scores_df.columns)}, "
            f"selected_features={len(selected)}->{len(selected_out)}, "
            f"cache_hit={bool(diag.get('cache_hit', False))}."
        )
        return X_out, selected_out, diag
    except Exception as exc:
        out = dict(diag_base)
        out.update(
            {
                "reason": f"failed:{exc}",
                "source_feature_count": int(len(source_features)),
            }
        )
        tprint(f"WARNING: LGBM regime score feature build failed: {exc}")
        return X_df, selected, out


def _lgbm_regime_specialist_feature_engineering_diagnostics(
    X_df: pd.DataFrame,
    selected_features: list[str],
    *,
    timestamps: Any = None,
    assets: Any = None,
    assessment_X_df: pd.DataFrame | None = None,
    assessment_timestamps: Any = None,
    assessment_assets: Any = None,
    objective_mode: str | None,
    cfg: dict[str, Any] | None,
    random_state: int,
) -> dict[str, Any]:
    if not _lgbm_regime_specialist_feature_engineering_diagnostics_enabled(cfg, objective_mode):
        return {"enabled": False, "used": False, "reason": "disabled"}
    try:
        from extreme_price_movements.regime_specialist_similarity import infer_regime_specialist_columns
        from extreme_price_movements.regime_specialist_feature_engineering import (
            build_regime_specialist_feature_engineering_artifact,
        )
    except Exception as exc:
        return {"enabled": True, "used": False, "reason": f"import_failed:{exc}"}

    source_df = assessment_X_df if isinstance(assessment_X_df, pd.DataFrame) and len(assessment_X_df) > len(X_df) else X_df
    source_ts = assessment_timestamps if source_df is assessment_X_df else timestamps
    source_assets = assessment_assets if source_df is assessment_X_df else assets
    try:
        sim_cfg = _regime_specialist_similarity_config(cfg)
        inferred = infer_regime_specialist_columns(
            source_df,
            selected_feature_columns=selected_features,
            config=sim_cfg,
        )
        extra_features = list(
            dict.fromkeys(
                list(inferred.get("market", []))
                + list(inferred.get("covariance", []))
                + list(inferred.get("knn", []))
            )
        )
        frame = _lgbm_regime_specialist_build_frame(
            source_df,
            selected_features,
            extra_features=extra_features,
            timestamps=source_ts,
            assets=source_assets,
        )
        if "timestamp" not in frame.columns:
            return {"enabled": True, "used": False, "reason": "missing_timestamps"}
        ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        valid_ts = ts.notna()
        if not bool(valid_ts.any()):
            return {"enabled": True, "used": False, "reason": "missing_valid_timestamps"}
        current_end_raw = _cfg_value(cfg, "lgbm_regime_specialist_current_end", None)
        end = (
            pd.to_datetime(current_end_raw, utc=True, errors="coerce")
            if current_end_raw is not None
            else ts.max()
        )
        if pd.isna(end):
            end = ts.max()
        current_days = float(_cfg_float_value(cfg, "lgbm_regime_specialist_current_window_days", 28.0))
        current_start = end - pd.Timedelta(days=current_days)
        current_mask = (ts >= current_start) & (ts <= end)
        historical_cutoff = current_start - pd.Timedelta(
            days=max(0.0, _cfg_float_value(cfg, "lgbm_regime_specialist_embargo_days", 0.0))
        )
        historical_mask = valid_ts & (ts < historical_cutoff)
        if int(current_mask.sum()) < _cfg_int_value(cfg, "lgbm_regime_specialist_min_current_rows", 24):
            return {
                "enabled": True,
                "used": False,
                "reason": "insufficient_current_rows",
                "current_rows": int(current_mask.sum()),
            }
        candidate_features = list(
            dict.fromkeys([str(c) for c in list(selected_features or []) + extra_features if str(c) in frame.columns])
        )
        if not candidate_features:
            return {"enabled": True, "used": False, "reason": "no_candidate_features"}
        start_val = ts.loc[valid_ts].min()
        end_val = ts.loc[valid_ts].max()
        cache_key = (
            _normalize_objective_mode(objective_mode),
            int(len(frame)),
            int(pd.Timestamp(start_val).value) if pd.notna(start_val) else 0,
            int(pd.Timestamp(end_val).value) if pd.notna(end_val) else 0,
            tuple(candidate_features),
            int(random_state),
            int(_cfg_int_value(cfg, "lgbm_regime_specialist_feature_engineering_max_final_features", 40)),
            bool(_cfg_bool_value(cfg, "lgbm_regime_specialist_feature_engineering_lgbm_enabled", True)),
            bool(_cfg_bool_value(cfg, "lgbm_regime_specialist_feature_engineering_elasticnet_enabled", True)),
            bool(_cfg_bool_value(cfg, "lgbm_regime_specialist_feature_engineering_run_validation_diagnostics", False)),
            str(_unsupervised_regime_learning_cfg(cfg).get("regime_models", {})),
        )
        cached = _REGIME_SPECIALIST_FE_DIAGNOSTIC_CACHE.get(cache_key)
        if cached is not None:
            out = dict(cached)
            out["cache_hit"] = True
            return out
        unsup_artifact, unsup_diag = _lgbm_unsupervised_regime_artifact(
            frame,
            candidate_features,
            cfg=cfg,
            random_state=random_state,
        )
        artifact = build_regime_specialist_feature_engineering_artifact(
            frame,
            timestamp_col="timestamp",
            symbol_col="symbol",
            candidate_features=candidate_features,
            unsupervised_regime_artifact=unsup_artifact,
            current_mask=current_mask.to_numpy(dtype=bool),
            historical_mask=historical_mask.to_numpy(dtype=bool),
            config=_regime_specialist_feature_engineering_config(
                cfg,
                random_state=random_state,
            ),
        )
        summary = _regime_specialist_feature_engineering_summary(artifact)
        summary.update(
            {
                "assessment_rows": int(len(frame)),
                "assessment_mode": "global" if source_df is assessment_X_df else "local",
                "candidate_feature_count": int(len(candidate_features)),
                "current_rows": int(current_mask.sum()),
                "historical_rows": int(historical_mask.sum()),
                "cache_hit": False,
                "unsupervised_regime_learning": unsup_diag,
            }
        )
        if len(_REGIME_SPECIALIST_FE_DIAGNOSTIC_CACHE) >= _REGIME_SPECIALIST_FE_DIAGNOSTIC_CACHE_MAX:
            _REGIME_SPECIALIST_FE_DIAGNOSTIC_CACHE.pop(next(iter(_REGIME_SPECIALIST_FE_DIAGNOSTIC_CACHE)))
        _REGIME_SPECIALIST_FE_DIAGNOSTIC_CACHE[cache_key] = dict(summary)
        return summary
    except Exception as exc:
        return {"enabled": True, "used": False, "reason": f"failed:{exc}"}


def _lgbm_regime_specialist_build_frame(
    X_df: pd.DataFrame,
    selected_features: list[str],
    *,
    extra_features: Sequence[str] | None = None,
    timestamps: Any = None,
    assets: Any = None,
) -> pd.DataFrame:
    cols = [
        str(c)
        for c in list(selected_features or []) + list(extra_features or [])
        if str(c) in X_df.columns
    ]
    cols = list(dict.fromkeys(cols))
    frame = X_df.reindex(columns=cols, fill_value=0.0).copy(deep=False)
    n = len(frame)
    if timestamps is not None and len(np.asarray(timestamps)) == n:
        frame = frame.copy(deep=False)
        frame["timestamp"] = np.asarray(timestamps)
    elif "timestamp" in X_df.columns:
        frame = frame.copy(deep=False)
        frame["timestamp"] = X_df["timestamp"].to_numpy(copy=False)
    if assets is not None and len(np.asarray(assets)) == n:
        frame = frame.copy(deep=False)
        frame["symbol"] = np.asarray(assets)
    elif "symbol" in X_df.columns:
        frame = frame.copy(deep=False)
        frame["symbol"] = X_df["symbol"].to_numpy(copy=False)
    return frame


def _lgbm_regime_specialist_context_value(
    label_context: Mapping[str, Any] | None,
    *names: str,
) -> Any:
    if not isinstance(label_context, Mapping):
        return None
    for name in names:
        if name in label_context and label_context.get(name) is not None:
            return label_context.get(name)
    return None


def _lgbm_regime_specialist_assessment_inputs(
    assessment_X: Any = None,
    assessment_timestamps: Any = None,
    assessment_assets: Any = None,
    *,
    label_context: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame | None, Any, Any]:
    frame_source = assessment_X
    if frame_source is None:
        frame_source = _lgbm_regime_specialist_context_value(
            label_context,
            "regime_specialist_assessment_frame",
            "regime_specialist_assessment_X",
            "regime_assessment_frame",
            "regime_assessment_X",
        )
    frame = None
    if frame_source is not None:
        try:
            frame = _frame(frame_source)
        except Exception:
            frame = None
    ts = assessment_timestamps
    if ts is None:
        ts = _lgbm_regime_specialist_context_value(
            label_context,
            "regime_specialist_assessment_timestamps",
            "regime_assessment_timestamps",
        )
    asset_values = assessment_assets
    if asset_values is None:
        asset_values = _lgbm_regime_specialist_context_value(
            label_context,
            "regime_specialist_assessment_assets",
            "regime_assessment_assets",
            "regime_specialist_assessment_symbols",
            "regime_assessment_symbols",
        )
    return frame, ts, asset_values


def _build_lgbm_regime_specialist_bundle(
    X_df: pd.DataFrame,
    selected_features: list[str],
    *,
    timestamps: Any = None,
    assets: Any = None,
    assessment_X_df: pd.DataFrame | None = None,
    assessment_timestamps: Any = None,
    assessment_assets: Any = None,
    objective_mode: str | None,
    cfg: dict[str, Any] | None,
    random_state: int,
    label: str,
) -> dict[str, Any]:
    n = len(X_df)
    disabled = {
        "enabled": False,
        "label": str(label),
        "objective_mode": _normalize_objective_mode(objective_mode),
        "weights": np.ones(n, dtype=np.float32),
        "similarity": np.ones(n, dtype=np.float32),
        "diagnostics": {"enabled": False, "reason": "disabled"},
        "metrics": {
            "regime_specialist_enabled": False,
            "regime_specialist_sample_weight_applied": False,
            "regime_specialist_distillation_shrink_enabled": False,
        },
    }
    specialist_enabled = _lgbm_regime_specialist_enabled(cfg, objective_mode)
    diagnostic_enabled = _lgbm_regime_specialist_feature_engineering_diagnostics_enabled(
        cfg,
        objective_mode,
    )
    diagnostic_final_only = _cfg_bool_value(
        cfg,
        "lgbm_regime_specialist_feature_engineering_diagnostics_final_only",
        True,
    )
    diagnostic_label_allowed = (not diagnostic_final_only) or str(label).strip().lower() == "final"
    if not specialist_enabled and not diagnostic_enabled:
        return disabled
    if not specialist_enabled and diagnostic_enabled and not diagnostic_label_allowed:
        out = dict(disabled)
        out["diagnostics"] = {
            "enabled": False,
            "reason": "regime_specialist_disabled_feature_engineering_diagnostics_deferred_to_final",
            "feature_engineering": {
                "enabled": True,
                "used": False,
                "diagnostic_only": True,
                "reason": "deferred_to_final",
            },
        }
        out["metrics"] = dict(disabled["metrics"])
        out["metrics"]["regime_specialist_reason"] = "disabled_feature_engineering_diagnostics_deferred_to_final"
        out["metrics"].update(
            _regime_specialist_feature_engineering_metric_summary(
                out["diagnostics"]["feature_engineering"]
            )
        )
        return out
    if not specialist_enabled and diagnostic_enabled:
        fe_diag = _lgbm_regime_specialist_feature_engineering_diagnostics(
            X_df,
            selected_features,
            timestamps=timestamps,
            assets=assets,
            assessment_X_df=assessment_X_df,
            assessment_timestamps=assessment_timestamps,
            assessment_assets=assessment_assets,
            objective_mode=objective_mode,
            cfg=cfg,
            random_state=random_state,
        )
        out = dict(disabled)
        out["diagnostics"] = {
            "enabled": False,
            "reason": "regime_specialist_disabled_feature_engineering_diagnostics_only",
            "feature_engineering": fe_diag,
        }
        out["metrics"] = dict(disabled["metrics"])
        out["metrics"]["regime_specialist_reason"] = "disabled_feature_engineering_diagnostics_only"
        out["metrics"].update(_regime_specialist_feature_engineering_metric_summary(fe_diag))
        return out
    try:
        from extreme_price_movements.regime_specialist_similarity import (
            REGIME_SPECIALIST_SCHEMA_VERSION,
            build_regime_specialist_training_frame,
            infer_regime_specialist_columns,
        )
    except Exception as exc:
        out = dict(disabled)
        out["enabled"] = False
        out["diagnostics"] = {"enabled": False, "reason": f"import_failed:{exc}"}
        out["metrics"] = dict(disabled["metrics"])
        out["metrics"]["regime_specialist_enabled"] = True
        out["metrics"]["regime_specialist_reason"] = "import_failed"
        return out
    if timestamps is None and "timestamp" not in X_df.columns:
        out = dict(disabled)
        out["enabled"] = False
        out["diagnostics"] = {"enabled": False, "reason": "missing_timestamps"}
        out["metrics"] = dict(disabled["metrics"])
        out["metrics"]["regime_specialist_enabled"] = True
        out["metrics"]["regime_specialist_reason"] = "missing_timestamps"
        return out
    sim_cfg = _regime_specialist_similarity_config(cfg)
    weight_cfg = _regime_specialist_weight_config(cfg)
    specialist_columns = infer_regime_specialist_columns(
        X_df,
        selected_feature_columns=selected_features,
        config=sim_cfg,
    )
    specialist_extra_features = list(
        dict.fromkeys(
            list(specialist_columns.get("market", []))
            + list(specialist_columns.get("drift", []))
            + list(specialist_columns.get("covariance", []))
            + list(specialist_columns.get("knn", []))
        )
    )
    frame = _lgbm_regime_specialist_build_frame(
        X_df,
        selected_features,
        extra_features=specialist_extra_features,
        timestamps=timestamps,
        assets=assets,
    )
    assessment_frame = None
    unsup_source_frame = frame
    unsup_candidate_features = list(dict.fromkeys(list(selected_features or []) + specialist_extra_features))
    if assessment_X_df is not None and isinstance(assessment_X_df, pd.DataFrame) and len(assessment_X_df) > len(X_df):
        assessment_columns = infer_regime_specialist_columns(
            assessment_X_df,
            selected_feature_columns=selected_features,
            config=sim_cfg,
        )
        assessment_extra_features = list(
            dict.fromkeys(
                specialist_extra_features
                + list(assessment_columns.get("market", []))
                + list(assessment_columns.get("drift", []))
                + list(assessment_columns.get("covariance", []))
                + list(assessment_columns.get("knn", []))
            )
        )
        assessment_frame = _lgbm_regime_specialist_build_frame(
            assessment_X_df,
            selected_features,
            extra_features=assessment_extra_features,
            timestamps=assessment_timestamps,
            assets=assessment_assets,
        )
        unsup_source_frame = assessment_frame
        unsup_candidate_features = list(dict.fromkeys(list(selected_features or []) + assessment_extra_features))
    unsup_artifact, unsup_diag = _lgbm_unsupervised_regime_artifact(
        unsup_source_frame,
        unsup_candidate_features,
        cfg=cfg,
        random_state=random_state,
    )
    try:
        generated, diag = build_regime_specialist_training_frame(
            frame,
            selected_feature_columns=selected_features,
            current_end=_cfg_value(cfg, "lgbm_regime_specialist_current_end", None),
            similarity_config=sim_cfg,
            weight_config=weight_cfg,
            market_columns=specialist_columns.get("market", []),
            drift_columns=specialist_columns.get("drift", []),
            covariance_columns=specialist_columns.get("covariance", []),
            knn_columns=specialist_columns.get("knn", []),
            asset_return_col=_cfg_value(cfg, "lgbm_regime_specialist_asset_return_col", None),
            include_input_columns=False,
            assessment_frame=assessment_frame,
            unsupervised_regime_artifact=unsup_artifact,
        )
        if isinstance(diag, dict):
            diag.setdefault("unsupervised_regime_learning_artifact", unsup_diag)
    except Exception as exc:
        out = dict(disabled)
        out["enabled"] = False
        out["diagnostics"] = {"enabled": False, "reason": f"build_failed:{exc}"}
        out["metrics"] = dict(disabled["metrics"])
        out["metrics"]["regime_specialist_enabled"] = True
        out["metrics"]["regime_specialist_reason"] = "build_failed"
        return out
    weight = pd.to_numeric(
        generated.get("regime_specialist_sample_weight", pd.Series(1.0, index=X_df.index)),
        errors="coerce",
    ).fillna(1.0).to_numpy(dtype=np.float32)
    similarity = pd.to_numeric(
        generated.get("similarity_to_current", pd.Series(1.0, index=X_df.index)),
        errors="coerce",
    ).fillna(1.0).clip(0.0, 1.0).to_numpy(dtype=np.float32)
    bucket = (
        generated.get("regime_specialist_bucket", pd.Series("normal", index=X_df.index))
        .astype(str)
        .to_numpy(dtype=object)
    )
    current_recency_weight = pd.to_numeric(
        generated.get("current_regime_recency_weight", pd.Series(0.0, index=X_df.index)),
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float32)
    if len(weight) != n:
        weight = np.ones(n, dtype=np.float32)
    if len(similarity) != n:
        similarity = np.ones(n, dtype=np.float32)
    if len(bucket) != n:
        bucket = np.repeat("normal", n).astype(object)
    if len(current_recency_weight) != n:
        current_recency_weight = np.zeros(n, dtype=np.float32)
    sim_diag = dict((diag or {}).get("similarity", {}) or {})
    weight_diag = dict((diag or {}).get("sample_weight", {}) or {})
    drift_baseline_diag = dict((diag or {}).get("weighted_drift_baseline", {}) or {})
    unsup_artifact_diag = (
        dict((diag or {}).get("unsupervised_regime_learning_artifact", {}) or {})
        if isinstance((diag or {}).get("unsupervised_regime_learning_artifact", {}), Mapping)
        else {}
    )
    drift_baseline_stats = (
        drift_baseline_diag.get("stats", {})
        if isinstance(drift_baseline_diag.get("stats", {}), dict)
        else {}
    )
    drift_weighted_means = np.asarray(
        [
            abs(float(v.get("weighted_mean", np.nan)))
            for v in drift_baseline_stats.values()
            if isinstance(v, dict)
        ],
        dtype=np.float64,
    )
    drift_weighted_medians = np.asarray(
        [
            abs(float(v.get("weighted_median", np.nan)))
            for v in drift_baseline_stats.values()
            if isinstance(v, dict)
        ],
        dtype=np.float64,
    )
    enabled = bool(sim_diag.get("enabled", False))
    should_train = bool(weight_diag.get("should_train_specialist", False))
    assessment_scope = sim_diag.get("assessment_scope", {}) if isinstance(sim_diag.get("assessment_scope", {}), dict) else {}
    feature_engineering_diag = (
        sim_diag.get("feature_engineering", {})
        if isinstance(sim_diag.get("feature_engineering", {}), dict)
        else {}
    )
    if (
        diagnostic_enabled
        and diagnostic_label_allowed
        and not bool(feature_engineering_diag.get("used", False))
        and not bool(feature_engineering_diag.get("diagnostic_only", False))
    ):
        feature_engineering_diag = _lgbm_regime_specialist_feature_engineering_diagnostics(
            X_df,
            selected_features,
            timestamps=timestamps,
            assets=assets,
            assessment_X_df=assessment_X_df,
            assessment_timestamps=assessment_timestamps,
            assessment_assets=assessment_assets,
            objective_mode=objective_mode,
            cfg=cfg,
            random_state=random_state,
        )
        sim_diag["feature_engineering"] = feature_engineering_diag
    shadow_only = _cfg_bool_value(cfg, "lgbm_regime_specialist_shadow_only", True)
    apply_weight = (
        enabled
        and should_train
        and not shadow_only
        and _cfg_bool_value(cfg, "lgbm_regime_specialist_apply_sample_weight", False)
    )
    apply_shrink = (
        enabled
        and should_train
        and not shadow_only
        and _cfg_bool_value(cfg, "lgbm_regime_specialist_apply_distillation_shrink", False)
    )
    metrics = {
        "regime_specialist_enabled": True,
        "regime_specialist_schema_version": REGIME_SPECIALIST_SCHEMA_VERSION,
        "regime_specialist_shadow_only": bool(shadow_only),
        "regime_specialist_similarity_enabled": bool(enabled),
        "regime_specialist_should_train": bool(should_train),
        "regime_specialist_sample_weight_applied": bool(apply_weight),
        "regime_specialist_distillation_shrink_enabled": bool(apply_shrink),
        "regime_specialist_assessment_mode": str(assessment_scope.get("mode", "local")),
        "regime_specialist_assessment_rows": int(assessment_scope.get("assessment_rows", len(frame)) or 0),
        "regime_specialist_local_training_rows": int(assessment_scope.get("local_training_rows", len(frame)) or 0),
        "regime_specialist_assessment_alignment": str(assessment_scope.get("alignment", "local")),
        "regime_specialist_assessment_aligned_fraction": float(assessment_scope.get("aligned_fraction", 1.0) or 0.0),
        "regime_specialist_assessment_alignment_ok": bool(assessment_scope.get("alignment_ok", True)),
        "regime_specialist_similarity_mean": float(np.nanmean(similarity)) if len(similarity) else float("nan"),
        "regime_specialist_similarity_p10": float(np.nanpercentile(similarity, 10.0)) if len(similarity) else float("nan"),
        "regime_specialist_similarity_p90": float(np.nanpercentile(similarity, 90.0)) if len(similarity) else float("nan"),
        "regime_specialist_weight_mean": float(np.nanmean(weight)) if len(weight) else float("nan"),
        "regime_specialist_weight_p10": float(np.nanpercentile(weight, 10.0)) if len(weight) else float("nan"),
        "regime_specialist_weight_p90": float(np.nanpercentile(weight, 90.0)) if len(weight) else float("nan"),
        "regime_specialist_adaptive_reliability": float(weight_diag.get("adaptive_reliability", 0.0) or 0.0),
        "regime_specialist_current_mass": float(weight_diag.get("current_mass", 0.0) or 0.0),
        "regime_specialist_analogue_mass": float(weight_diag.get("analogue_mass", 0.0) or 0.0),
        "regime_specialist_normal_mass": float(weight_diag.get("normal_mass", 0.0) or 0.0),
        "regime_specialist_irrelevant_mass": float(weight_diag.get("irrelevant_mass", 0.0) or 0.0),
        "regime_specialist_less_interesting_mass": float(weight_diag.get("less_interesting_mass", 0.0) or 0.0),
        "regime_specialist_less_interesting_mass_cap": float(weight_diag.get("less_interesting_mass_cap", 0.0) or 0.0),
        "regime_specialist_adaptive_n_eff": float(weight_diag.get("adaptive_n_eff", 0.0) or 0.0),
        "regime_specialist_replay_n_eff": float(weight_diag.get("replay_n_eff", 0.0) or 0.0),
        "regime_specialist_adaptive_n_eff_reliability": float(
            weight_diag.get("adaptive_n_eff_reliability", 0.0) or 0.0
        ),
        "regime_specialist_replay_n_eff_reliability": float(
            weight_diag.get("replay_n_eff_reliability", 0.0) or 0.0
        ),
        "regime_specialist_replay_need": float(weight_diag.get("replay_need", 0.0) or 0.0),
        "regime_specialist_actual_adaptive_weight_mass": float(
            weight_diag.get("actual_current_weight_mass", 0.0) or 0.0
        )
        + float(weight_diag.get("actual_analogue_weight_mass", 0.0) or 0.0),
        "regime_specialist_actual_less_interesting_weight_mass": float(
            weight_diag.get("actual_less_interesting_weight_mass", 0.0) or 0.0
        ),
        "regime_specialist_actual_less_interesting_weight_cap_ok": bool(
            weight_diag.get("actual_less_interesting_weight_cap_ok", True)
        ),
        "regime_specialist_bucket_mass_basis": str(weight_diag.get("bucket_mass_basis", "")),
        "regime_specialist_recency_power": float(weight_diag.get("recency_power", 1.0) or 1.0),
        "regime_specialist_candidate_window_count": int(sim_diag.get("candidate_window_count", 0) or 0),
        "regime_specialist_unsupervised_regime_learning_enabled": bool(unsup_artifact_diag.get("enabled", False)),
        "regime_specialist_unsupervised_regime_learning_used": bool(unsup_artifact_diag.get("used", False)),
        "regime_specialist_unsupervised_regime_learning_selected_feature_count": int(
            unsup_artifact_diag.get("selected_feature_count", 0) or 0
        ),
        "regime_specialist_unsupervised_regime_learning_specialist_candidate_count": int(
            unsup_artifact_diag.get("specialist_candidate_feature_count", 0) or 0
        ),
        "regime_specialist_weighted_drift_baseline_enabled": bool(drift_baseline_diag.get("enabled", False)),
        "regime_specialist_weighted_drift_baseline_feature_count": int(drift_baseline_diag.get("feature_count", 0) or 0),
        "regime_specialist_weighted_drift_baseline_abs_mean": float(np.nanmean(drift_weighted_means)) if drift_weighted_means.size else float("nan"),
        "regime_specialist_weighted_drift_baseline_abs_median": float(np.nanmean(drift_weighted_medians)) if drift_weighted_medians.size else float("nan"),
        "regime_specialist_random_state": int(random_state),
        "regime_specialist_label": str(label),
    }
    metrics.update(_regime_specialist_feature_engineering_metric_summary(feature_engineering_diag))
    return {
        "enabled": True,
        "label": str(label),
        "objective_mode": _normalize_objective_mode(objective_mode),
        "weights": np.nan_to_num(weight, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32),
        "similarity": np.nan_to_num(similarity, nan=1.0, posinf=1.0, neginf=0.0).astype(np.float32),
        "bucket": bucket,
        "current_regime_recency_weight": np.nan_to_num(
            current_recency_weight,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32),
        "diagnostics": _json_sanitize(diag),
        "metrics": metrics,
        "apply_sample_weight": bool(apply_weight),
        "apply_distillation_shrink": bool(apply_shrink),
    }


def _normalize_weights_between(
    weights: np.ndarray,
    *,
    min_weight: float,
    max_weight: float,
) -> tuple[np.ndarray, float]:
    """Normalize to mean one, then compress deviations inside a fixed range."""
    lo = float(min(min_weight, max_weight))
    hi = float(max(min_weight, max_weight))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0.0 or hi <= 0.0 or lo >= hi:
        lo, hi = 0.7, 1.3

    w = np.nan_to_num(np.asarray(weights, dtype=np.float32), nan=1.0, posinf=hi, neginf=lo)
    if len(w) == 0:
        return w.astype(np.float32), 0.0
    w = np.clip(w, 0.0, None)
    mean = float(np.mean(w))
    if not np.isfinite(mean) or mean <= 1e-12:
        out = np.ones(len(w), dtype=np.float32)
        return out, float(len(out))

    unit = w / mean
    delta = unit - 1.0
    shrink = 1.0
    max_positive_delta = float(np.max(delta))
    max_negative_delta = float(np.max(-delta))
    if max_positive_delta > hi - 1.0:
        shrink = min(shrink, (hi - 1.0) / max(max_positive_delta, 1e-12))
    if max_negative_delta > 1.0 - lo:
        shrink = min(shrink, (1.0 - lo) / max(max_negative_delta, 1e-12))

    out = (1.0 + shrink * delta).astype(np.float32)
    out = np.clip(out, lo, hi)
    ess = float((out.sum() ** 2) / max(float(np.sum(out**2)), 1e-6))
    return out.astype(np.float32), ess


def _apply_lgbm_regime_specialist_weights(
    base_weight: np.ndarray,
    bundle: dict[str, Any] | None,
    idx: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    w = np.asarray(base_weight, dtype=np.float32)
    if not bundle or not bool(bundle.get("apply_sample_weight", False)):
        return w, {"applied": False, "reason": "disabled_or_shadow"}
    multiplier = np.asarray(bundle.get("weights", np.ones(0, dtype=np.float32)), dtype=np.float32)
    if idx is not None:
        take = np.asarray(idx, dtype=np.int64)
        if len(multiplier) <= int(np.max(take, initial=-1)):
            return w, {"applied": False, "reason": "index_length_mismatch"}
        multiplier = multiplier[take]
    if len(multiplier) != len(w):
        return w, {"applied": False, "reason": "length_mismatch"}
    w_preconditioned, precondition_ess = _normalize_weights_between(
        w,
        min_weight=0.7,
        max_weight=1.3,
    )
    out, ess = _normalize_weights(w_preconditioned * multiplier)
    return out.astype(np.float32), {
        "applied": True,
        "effective_sample_size": float(ess),
        "base_weight_preconditioned": True,
        "base_weight_preconditioned_policy": "unit_mean_compress_0.7_1.3",
        "base_weight_preconditioned_min": float(np.nanmin(w_preconditioned)) if len(w_preconditioned) else float("nan"),
        "base_weight_preconditioned_max": float(np.nanmax(w_preconditioned)) if len(w_preconditioned) else float("nan"),
        "base_weight_preconditioned_mean": float(np.nanmean(w_preconditioned)) if len(w_preconditioned) else float("nan"),
        "base_weight_preconditioned_effective_sample_size": float(precondition_ess),
        "multiplier_mean": float(np.nanmean(multiplier)) if len(multiplier) else float("nan"),
        "multiplier_p90": float(np.nanpercentile(multiplier, 90.0)) if len(multiplier) else float("nan"),
    }


def _lgbm_regime_specialist_similarity_for_idx(
    bundle: dict[str, Any] | None,
    idx: np.ndarray | None = None,
) -> np.ndarray | None:
    if not bundle or not bool(bundle.get("apply_distillation_shrink", False)):
        return None
    sim = np.asarray(bundle.get("similarity", np.ones(0, dtype=np.float32)), dtype=np.float32)
    if idx is not None:
        take = np.asarray(idx, dtype=np.int64)
        if len(sim) <= int(np.max(take, initial=-1)):
            return None
        sim = sim[take]
    return np.clip(sim, 0.0, 1.0).astype(np.float32)


def _lgbm_regime_specialist_current_metrics(
    y_true: np.ndarray,
    pred: np.ndarray,
    bundle: dict[str, Any] | None,
    *,
    classifier: bool,
    groups: Any = None,
    returns: Any = None,
    idx: np.ndarray | None = None,
    label_context: Mapping[str, Any] | None = None,
    label_context_total_rows: int | None = None,
) -> dict[str, Any]:
    prefix = "current_regime_"
    y = np.asarray(y_true)
    p = np.asarray(pred)
    n = min(len(y), len(p))
    out: dict[str, Any] = {
        f"{prefix}metrics_available": False,
        f"{prefix}metric_rows": 0,
        f"{prefix}metric_row_fraction": 0.0,
        f"{prefix}metric_reason": "disabled",
    }
    if bundle is None or not bool(bundle.get("enabled", False)) or n <= 0:
        return out
    bucket = np.asarray(bundle.get("bucket", np.asarray([], dtype=object)), dtype=object)
    recency = np.asarray(
        bundle.get("current_regime_recency_weight", np.asarray([], dtype=np.float32)),
        dtype=np.float32,
    )
    full_idx: np.ndarray | None = None
    if idx is not None:
        take = np.asarray(idx, dtype=np.int64)
        if len(bucket) <= int(np.max(take, initial=-1)):
            out[f"{prefix}metric_reason"] = "index_length_mismatch"
            return out
        full_idx = take[:n]
        bucket = bucket[full_idx]
        recency = recency[full_idx] if len(recency) > int(np.max(full_idx, initial=-1)) else np.zeros(len(full_idx), dtype=np.float32)
    elif len(bucket) != n:
        out[f"{prefix}metric_reason"] = "length_mismatch"
        return out
    else:
        full_idx = np.arange(n, dtype=np.int64)
    if len(recency) != len(bucket):
        recency = np.zeros(len(bucket), dtype=np.float32)
    bucket_s = pd.Series(bucket).astype(str).str.lower().to_numpy(dtype=object)
    current_mask = (bucket_s == "current") | (recency > 0.0)
    current_mask = current_mask[:n]
    rows = int(np.sum(current_mask))
    out[f"{prefix}metric_rows"] = rows
    out[f"{prefix}metric_row_fraction"] = float(rows / max(n, 1))
    out[f"{prefix}metric_recency_weight_sum"] = float(np.nansum(recency[:n][current_mask])) if rows else 0.0
    if rows < 8:
        out[f"{prefix}metric_reason"] = "insufficient_current_rows"
        return out
    local = np.flatnonzero(current_mask).astype(np.int64)
    ret = None
    if returns is not None:
        ret_arr = np.asarray(returns)
        if len(ret_arr) == n:
            ret = ret_arr[local]
    grp = _groups_take(groups, local) if groups is not None else None
    metrics = _metric_pack(
        y[:n][local],
        p[:n][local],
        classifier=classifier,
        groups=grp,
        returns=ret,
    )
    out.update({f"{prefix}{str(k)}": v for k, v in metrics.items()})
    if label_context is not None and full_idx is not None:
        label_n = int(label_context_total_rows) if label_context_total_rows is not None else n
        ctx = _label_context_take(label_context, full_idx[local], label_n)
        out.update(
            {
                f"{prefix}{str(k)}": v
                for k, v in _vol_normalized_tp_sl_precision_metrics(
                    p[:n][local],
                    ctx,
                ).items()
            }
        )
    out[f"{prefix}metrics_available"] = True
    out[f"{prefix}metric_reason"] = "ok"
    return out


def _regime_specialist_shrink_weight_towards_one(
    weights: np.ndarray,
    similarity: np.ndarray | None,
    *,
    cfg: dict[str, Any] | None = None,
) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float32)
    if similarity is None:
        return w
    sim = np.asarray(similarity, dtype=np.float32)
    if len(sim) != len(w):
        return w
    try:
        from extreme_price_movements.regime_specialist_similarity import (
            shrink_self_distillation_towards_one,
        )

        return shrink_self_distillation_towards_one(
            w,
            sim,
            power=_cfg_float_value(cfg, "lgbm_regime_specialist_distillation_power", 1.0),
        ).astype(np.float32)
    except Exception:
        power = _cfg_float_value(cfg, "lgbm_regime_specialist_distillation_power", 1.0)
        factor = np.power(np.clip(sim, 0.0, 1.0), float(power))
        return (1.0 + (w - 1.0) * factor).astype(np.float32)


def _save_lgbm_regime_specialist_diagnostics(
    reference_artifact_dir: str | os.PathLike[str] | None,
    bundle: dict[str, Any] | None,
    *,
    objective_mode: str,
    label: str,
) -> None:
    if reference_artifact_dir is None or not bundle:
        return
    try:
        out_dir = Path(reference_artifact_dir) / "regime_specialist_similarity"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{str(objective_mode)}_{str(label)}.json"
        payload = {
            "metrics": _json_sanitize(bundle.get("metrics", {})),
            "diagnostics": _json_sanitize(bundle.get("diagnostics", {})),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except Exception as exc:
        tprint(f"WARNING: failed to save LGBM regime specialist diagnostics: {exc}")


def _recency_hpo_candidate_override_cfg(
    cfg: dict[str, Any] | None,
    *,
    scope: str,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    out = dict(cfg or {})
    scope_norm = "meta" if str(scope).strip().lower() == "meta" else "base"
    out[f"recency_hpo_{scope_norm}_half_life_days"] = float(candidate["half_life_days"])
    out[f"recency_hpo_{scope_norm}_half_life_months"] = float(candidate["half_life_months"])
    out[f"recency_hpo_{scope_norm}_composite_weight"] = float(candidate["composite_weight"])
    return out


def _recency_hpo_distillation_confirmation(
    candidate: dict[str, Any],
    *,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    y_metric_train: np.ndarray,
    base_train_weight: np.ndarray,
    ts_train: Any,
    returns_train: np.ndarray,
    X_oos: pd.DataFrame,
    y_oos_metric: np.ndarray,
    ts_oos: Any,
    params_base: dict[str, Any],
    classifier: bool,
    mode: str,
    objective_mode: str,
    cfg: dict[str, Any] | None,
    scope: str,
    random_state: int,
) -> dict[str, Any]:
    distill_passes = _distillation_passes_for_objective(objective_mode)
    out: dict[str, Any] = {
        "trial": int(candidate.get("trial", -1)),
        "scope": str(candidate.get("scope", scope)),
        "scope_key": str(candidate.get("scope_key", "")),
        "half_life_months": float(candidate["half_life_months"]),
        "half_life_days": float(candidate["half_life_days"]),
        "composite_weight": float(candidate["composite_weight"]),
        "no_distillation_final_selection_score": float(
            candidate.get("final_selection_score", float("nan"))
        ),
        "distillation_passes": int(distill_passes),
    }
    if distill_passes <= 0:
        out.update(
            {
                "enabled": False,
                "status": "skipped_self_distillation_disabled",
                "final_selection_score": float("nan"),
            }
        )
        return out
    decay = composite_decay_from_timestamps(
        ts_train,
        len(y_train),
        half_life_days=float(candidate["half_life_days"]),
        composite_weight=float(candidate["composite_weight"]),
    )
    if decay is None:
        raise ValueError("recency_hpo distillation confirmation could not compute train decay weights")
    train_weight, train_ess = _normalize_weights(
        np.asarray(base_train_weight, dtype=np.float32) * np.asarray(decay, dtype=np.float32)
    )
    confirmation_cfg = _recency_hpo_candidate_override_cfg(
        cfg,
        scope=scope,
        candidate=candidate,
    )
    confirm_t0 = time.perf_counter()
    final_weight, confirmation_oof = _oof_distilled_sample_weights_lgbm(
        X_train,
        y_train,
        train_weight,
        list(X_train.columns),
        classifier=classifier,
        params=dict(params_base),
        timestamps=ts_train,
        returns=returns_train,
        metric_y=y_metric_train,
        random_state=int(random_state),
        passes=int(distill_passes),
        label=f"recency_hpo_confirm_trial_{int(candidate.get('trial', -1))}",
        objective_mode=objective_mode,
        cfg=confirmation_cfg,
    )
    final_weight, final_ess = _normalize_weights(final_weight)
    sequential_weight_base = final_weight.copy()
    sequential_weights = final_weight.copy()
    prev_ensemble_pred = (
        confirmation_oof.copy()
        if len(confirmation_oof) == len(y_train)
        else None
    )
    running_train_pred: np.ndarray | None = None
    running_oos_pred: np.ndarray | None = None
    final_ensemble_ess = float(final_ess)
    for model_i in range(int(LGBM_FINAL_MODEL_COUNT)):
        params = dict(params_base)
        params["random_state"] = int(
            random_state
            + 4049
            + int(candidate.get("trial", 0)) * 1009
            + model_i * 101
        )
        model = _fit_lgbm_model(
            X_train,
            y_train,
            sequential_weights,
            classifier=classifier,
            params=params,
        )
        train_pred = _predict_lgbm_raw(model, X_train, mode)
        oos_pred = _predict_lgbm_raw(model, X_oos, mode)
        if running_train_pred is None:
            running_train_pred = train_pred.astype(np.float32)
            running_oos_pred = oos_pred.astype(np.float32)
        else:
            running_train_pred = (
                (running_train_pred.astype(np.float32) * float(model_i))
                + train_pred.astype(np.float32)
            ) / float(model_i + 1)
            running_oos_pred = (
                (running_oos_pred.astype(np.float32) * float(model_i))
                + oos_pred.astype(np.float32)
            ) / float(model_i + 1)
        if not LGBM_DISABLE_SELF_DISTILLATION:
            distill = _compute_weight_distillation(
                y_metric_train,
                running_train_pred,
                prev_ensemble_pred,
                is_classifier=classifier,
                include_false_positive_focus=False,
            )
            fp_weight = _false_positive_avoidance_weight(
                y_metric_train,
                running_train_pred,
                classifier=classifier,
                top_frac=_target_top_fraction(objective_mode),
            )
            distill = _recency_shrink_weight_towards_one(
                distill,
                ts_train,
                objective_mode=objective_mode,
                cfg=confirmation_cfg,
            )
            fp_weight = _recency_shrink_weight_towards_one(
                fp_weight,
                ts_train,
                objective_mode=objective_mode,
                cfg=confirmation_cfg,
            )
            distill, fp_weight = apply_distillation_recipe(
                distill,
                fp_weight,
                y_metric=y_metric_train,
                pred=running_train_pred,
                returns=returns_train,
                timestamps=ts_train,
                objective_mode=objective_mode,
                cfg=confirmation_cfg,
            )
            sequential_weights, final_ensemble_ess = _normalize_weights(
                sequential_weight_base * distill * fp_weight
            )
            prev_ensemble_pred = running_train_pred.copy()
    pred = (
        np.asarray(running_oos_pred, dtype=np.float32)
        if running_oos_pred is not None
        else np.full(len(y_oos_metric), float(np.mean(y_train)), dtype=np.float32)
    )
    score_pack = recency_hpo_final_selection_score(
        y_oos_metric,
        pred,
        ts_oos,
    )
    score = float(score_pack.get("final_selection_score", float("nan")))
    out.update(
        {
            "enabled": True,
            "status": "confirmed",
            "train_effective_sample_size": float(train_ess),
            "distilled_effective_sample_size": float(final_ess),
            "final_ensemble_count": int(LGBM_FINAL_MODEL_COUNT),
            "final_ensemble_sequential_distillation": not bool(LGBM_DISABLE_SELF_DISTILLATION),
            "final_ensemble_effective_sample_size": float(final_ensemble_ess),
            "confirmation_oof_mean": float(np.nanmean(confirmation_oof)),
            "elapsed_sec": float(time.perf_counter() - confirm_t0),
            "final_selection_score": score,
            "score_delta_vs_no_distillation": float(
                score - float(candidate.get("final_selection_score", np.nan))
            ),
            **score_pack,
        }
    )
    return out


def run_lgbm_recency_hpo_fixed_contract(
    X: Any,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray],
    *,
    selected_features: list[str],
    best_params: dict[str, Any],
    timestamps: Any,
    returns: Any = None,
    hard_labels: Optional[np.ndarray] = None,
    random_state: int = 42,
    mode: str = "classifier",
    objective_mode: str = "train_base",
    cfg: dict[str, Any] | None = None,
    scope_key: str | None = None,
    persist_winner: bool | None = None,
) -> dict[str, Any]:
    """Evaluate recency weights with a frozen feature/HPO contract.

    This is intentionally not a general model HPO path. It assumes feature
    selection and LightGBM params were optimized elsewhere for the same scope.
    """
    objective_mode = _normalize_objective_mode(objective_mode)
    scope = recency_hpo_objective_scope(objective_mode)
    classifier = mode == "classifier"
    selected = [str(c) for c in (selected_features or []) if str(c).strip()]
    if not selected:
        raise ValueError("recency_hpo requires frozen selected_features")
    if not best_params:
        raise ValueError("recency_hpo requires frozen best_params")
    X_df = _frame(X)
    missing = [c for c in selected if c not in X_df.columns]
    if missing:
        raise ValueError(
            "recency_hpo selected feature contract missing columns: "
            f"{len(missing)}/{len(selected)} examples={missing[:20]}"
        )
    y_arr = _coerce_target(y, classifier, allow_soft_labels=bool(LGBM_TRUE_SOFT_LABELS))
    y_metric = (
        _coerce_target(hard_labels, classifier)
        if hard_labels is not None
        else _coerce_target(y, classifier)
    )
    n = len(y_arr)
    _validate_input_lengths(X_df, y_arr, sample_weight=sample_weight, timestamps=timestamps, returns=returns)
    if len(y_metric) != n:
        raise ValueError(f"recency_hpo metric target length {len(y_metric)} != {n}")
    train_years = int((cfg or {}).get("recency_hpo_train_years", 3))
    holdout_months = int((cfg or {}).get("recency_hpo_holdout_months", 2))
    train_mask, oos_mask, split_meta = recency_hpo_train_oos_masks(
        timestamps,
        train_years=train_years,
        holdout_months=holdout_months,
    )
    train_idx = np.flatnonzero(train_mask).astype(np.int32)
    oos_idx = np.flatnonzero(oos_mask).astype(np.int32)
    min_train_rows = int((cfg or {}).get("recency_hpo_min_train_rows", 1000))
    min_oos_rows = int((cfg or {}).get("recency_hpo_min_oos_rows", 200))
    if len(train_idx) < min_train_rows or len(oos_idx) < min_oos_rows:
        raise ValueError(
            "recency_hpo insufficient rows after 3y/2m split: "
            f"train={len(train_idx)} min={min_train_rows}, "
            f"oos={len(oos_idx)} min={min_oos_rows}, split={split_meta}"
        )
    X_train = X_df.iloc[train_idx][selected].reset_index(drop=True)
    X_oos = X_df.iloc[oos_idx][selected].reset_index(drop=True)
    y_train = y_arr[train_idx]
    y_metric_train = y_metric[train_idx]
    y_oos_metric = y_metric[oos_idx]
    ts_train = _take_aligned(timestamps, train_idx, n)
    ts_oos = _take_aligned(timestamps, oos_idx, n)
    ret_arr = _as_returns(y_metric, returns)
    ret_train = ret_arr[train_idx]
    base_weight = (
        np.ones(n, dtype=np.float32)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float32)
    )
    base_train_weight, _ = _normalize_weights(base_weight[train_idx])
    params_base = _effective_lgbm_params(dict(best_params), classifier=classifier)
    grid = recency_hpo_grid(scope, cfg=cfg)
    trial_results: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None
    best_score = -np.inf
    tprint(
        "LGBM recency_hpo fixed-contract search started: "
        f"scope={scope}, scope_key={scope_key or ''}, "
        f"features={len(selected)}, train_rows={len(train_idx)}, "
        f"oos_rows={len(oos_idx)}, grid={len(grid)}."
    )
    for trial_i, candidate in enumerate(grid, start=1):
        half_life_days = float(candidate["half_life_days"])
        composite_weight = float(candidate["composite_weight"])
        decay = composite_decay_from_timestamps(
            ts_train,
            len(train_idx),
            half_life_days=half_life_days,
            composite_weight=composite_weight,
        )
        if decay is None:
            raise ValueError("recency_hpo could not compute train decay weights")
        train_weight, ess = _normalize_weights(base_train_weight * decay)
        params = dict(params_base)
        params["random_state"] = int(random_state + 1009 * trial_i)
        fit_t0 = time.perf_counter()
        model = _fit_lgbm_model(
            X_train,
            y_train,
            train_weight,
            classifier=classifier,
            params=params,
        )
        pred = _predict_lgbm_raw(model, X_oos, mode)
        score_pack = recency_hpo_final_selection_score(
            y_oos_metric,
            pred,
            ts_oos,
        )
        score = float(score_pack.get("final_selection_score", float("nan")))
        row: dict[str, Any] = {
            "trial": int(trial_i),
            "scope": scope,
            "scope_key": str(scope_key or ""),
            "half_life_months": float(candidate["half_life_months"]),
            "half_life_days": half_life_days,
            "composite_weight": composite_weight,
            "train_effective_sample_size": float(ess),
            "elapsed_sec": float(time.perf_counter() - fit_t0),
            **score_pack,
        }
        trial_results.append(row)
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_result = row
        tprint(
            "LGBM recency_hpo trial complete: "
            f"{trial_i}/{len(grid)} scope={scope} "
            f"half_life_months={float(candidate['half_life_months']):g} "
            f"composite_weight={composite_weight:.2f} "
            f"final_selection_score={score:.6f}."
        )
    if best_result is None:
        raise RuntimeError("recency_hpo completed without a finite selection score")
    distill_confirmation: dict[str, Any] = {
        "enabled": False,
        "status": "disabled",
        "candidates": [],
    }
    confirm_enabled = _cfg_env_bool(
        cfg,
        "recency_hpo_confirm_with_distillation",
        "EPM_RECENCY_HPO_CONFIRM_WITH_DISTILLATION",
        True,
    )
    if confirm_enabled:
        confirm_top_n = _cfg_env_int(
            cfg,
            "recency_hpo_confirmation_top_n",
            "EPM_RECENCY_HPO_CONFIRMATION_TOP_N",
            3,
        )
        ranked_trials = sorted(
            trial_results,
            key=lambda row: float(row.get("final_selection_score", -np.inf)),
            reverse=True,
        )
        if confirm_top_n <= 0:
            confirm_top_n = len(ranked_trials)
        confirm_top_n = max(1, min(int(confirm_top_n), len(ranked_trials)))
        require_confirmation = _cfg_env_bool(
            cfg,
            "recency_hpo_require_distillation_confirmation",
            "EPM_RECENCY_HPO_REQUIRE_DISTILLATION_CONFIRMATION",
            True,
        )
        tolerance = max(
            0.0,
            _cfg_env_float(
                cfg,
                "recency_hpo_confirmation_score_tolerance",
                "EPM_RECENCY_HPO_CONFIRMATION_SCORE_TOLERANCE",
                1e-9,
            ),
        )
        tprint(
            "LGBM recency_hpo self-distillation confirmation started: "
            f"scope={scope}, scope_key={scope_key or ''}, "
            f"top_n={confirm_top_n}, require={require_confirmation}."
        )
        confirmation_rows: list[dict[str, Any]] = []
        try:
            for confirm_i, candidate in enumerate(ranked_trials[:confirm_top_n], start=1):
                row = _recency_hpo_distillation_confirmation(
                    candidate,
                    X_train=X_train,
                    y_train=y_train,
                    y_metric_train=y_metric_train,
                    base_train_weight=base_train_weight,
                    ts_train=ts_train,
                    returns_train=ret_train,
                    X_oos=X_oos,
                    y_oos_metric=y_oos_metric,
                    ts_oos=ts_oos,
                    params_base=params_base,
                    classifier=classifier,
                    mode=mode,
                    objective_mode=objective_mode,
                    cfg=cfg,
                    scope=scope,
                    random_state=int(random_state + 50021 + confirm_i * 3001),
                )
                confirmation_rows.append(row)
                tprint(
                    "LGBM recency_hpo self-distillation confirmation complete: "
                    f"{confirm_i}/{confirm_top_n} trial={int(row.get('trial', -1))} "
                    f"score={float(row.get('final_selection_score', np.nan)):.6f} "
                    f"delta={float(row.get('score_delta_vs_no_distillation', np.nan)):.6f} "
                    f"status={row.get('status')}."
                )
        except Exception as exc:
            distill_confirmation = {
                "enabled": True,
                "status": "failed",
                "error": str(exc),
                "top_n": int(confirm_top_n),
                "candidates": confirmation_rows,
            }
            if require_confirmation:
                raise RuntimeError(
                    f"recency_hpo self-distillation confirmation failed: {exc}"
                ) from exc
        else:
            all_skipped = bool(confirmation_rows) and all(
                not bool(row.get("enabled", False)) for row in confirmation_rows
            )
            if all_skipped:
                distill_confirmation = {
                    "enabled": False,
                    "status": "skipped_self_distillation_disabled",
                    "top_n": int(confirm_top_n),
                    "require_confirmation": bool(require_confirmation),
                    "winner_trial": int(best_result.get("trial", -1)),
                    "candidates": confirmation_rows,
                }
            else:
                finite_confirmed = [
                    row
                    for row in confirmation_rows
                    if np.isfinite(float(row.get("final_selection_score", np.nan)))
                ]
                best_confirmed = (
                    max(
                        finite_confirmed,
                        key=lambda row: float(row.get("final_selection_score", -np.inf)),
                    )
                    if finite_confirmed
                    else None
                )
                winner_confirmed_row = next(
                    (
                        row
                        for row in confirmation_rows
                        if int(row.get("trial", -1)) == int(best_result.get("trial", -2))
                    ),
                    None,
                )
                winner_confirmed_score = float(
                    (winner_confirmed_row or {}).get("final_selection_score", np.nan)
                )
                best_confirmed_score = float(
                    (best_confirmed or {}).get("final_selection_score", np.nan)
                )
                winner_confirmed_best = bool(
                    winner_confirmed_row is not None
                    and best_confirmed is not None
                    and best_confirmed_score <= winner_confirmed_score + tolerance
                )
                selected_confirmed_raw = next(
                    (
                        row
                        for row in ranked_trials
                        if int(row.get("trial", -1)) == int((best_confirmed or {}).get("trial", -2))
                    ),
                    None,
                )
                if best_confirmed is not None and selected_confirmed_raw is not None:
                    selected_result = dict(selected_confirmed_raw)
                    selected_result["final_selection_score"] = float(best_confirmed_score)
                    selected_result["no_distillation_final_selection_score"] = float(
                        selected_confirmed_raw.get("final_selection_score", np.nan)
                    )
                else:
                    selected_result = best_result
                distill_confirmation = {
                    "enabled": True,
                    "status": "confirmed" if winner_confirmed_best else "confirmed_reranked_winner",
                    "top_n": int(confirm_top_n),
                    "require_confirmation": bool(require_confirmation),
                    "score_tolerance": float(tolerance),
                    "raw_winner_trial": int(best_result.get("trial", -1)),
                    "selected_trial": int(selected_result.get("trial", -1)),
                    "winner_trial": int(selected_result.get("trial", -1)),
                    "winner_confirmed_best_among_confirmed": bool(winner_confirmed_best),
                    "raw_winner_distillation_score": winner_confirmed_score,
                    "winner_distillation_score": best_confirmed_score,
                    "best_confirmed_trial": int((best_confirmed or {}).get("trial", -1)),
                    "best_confirmed_score": best_confirmed_score,
                    "candidates": confirmation_rows,
                }
                best_result = selected_result
    winner = {
        "scope": scope,
        "scope_key": str(scope_key or ""),
        "trial": int(best_result.get("trial", -1)),
        "half_life_months": float(best_result["half_life_months"]),
        "half_life_days": float(best_result["half_life_days"]),
        "composite_weight": float(best_result["composite_weight"]),
        "final_selection_score": float(best_result["final_selection_score"]),
        "source": "recency_hpo",
    }
    if "no_distillation_final_selection_score" in best_result:
        winner["no_distillation_final_selection_score"] = float(
            best_result.get("no_distillation_final_selection_score", np.nan)
        )
    if distill_confirmation.get("enabled"):
        winner["distillation_confirmation_score"] = float(
            distill_confirmation.get("winner_distillation_score", np.nan)
        )
        winner["distillation_confirmation_status"] = str(
            distill_confirmation.get("status", "")
        )
    payload: dict[str, Any] = {
        "winner": winner,
        "objective_mode": objective_mode,
        "scope": scope,
        "scope_key": str(scope_key or ""),
        "selected_feature_count": int(len(selected)),
        "best_params": dict(params_base),
        "selection_metric": {
            "precision_score": "0.25*P@10 + 0.50*P@20 + 1.00*P@30",
            "final_selection_score": "0.50*precision_score_last_4w + 1.00*precision_score_last_8w",
        },
        "split": split_meta,
        "grid_results": trial_results,
        "distillation_confirmation": distill_confirmation,
    }
    if persist_winner is None:
        persist_winner = str(
            os.environ.get(
                "EPM_RECENCY_HPO_PERSIST_WINNER",
                (cfg or {}).get("recency_hpo_persist_winner", True),
            )
        ).strip().lower() not in {"0", "false", "no", "off"}
    if bool(persist_winner):
        winner_path = save_recency_hpo_winner(cfg, scope, payload)
        payload["winner_path"] = str(winner_path)
        tprint(
            "LGBM recency_hpo winner saved: "
            f"scope={scope}, path={winner_path}, "
            f"half_life_months={winner['half_life_months']:g}, "
            f"composite_weight={winner['composite_weight']:.2f}, "
            f"score={winner['final_selection_score']:.6f}."
        )
    return payload


def _take_aligned(values: Any, idx: np.ndarray, n: int) -> Any:
    if values is None:
        return None
    arr = np.asarray(values)
    if len(arr) != int(n):
        return None
    return arr[np.asarray(idx, dtype=np.int32)]


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


def _native_preset_selected_feature_variance_guard(
    X: pd.DataFrame,
    features: list[str],
    *,
    cfg: Mapping[str, Any] | None,
    label: str,
    preset_source: str | None = None,
) -> dict[str, Any]:
    native_required = str(
        os.environ.get(
            "EPM_LGBM_REQUIRE_NATIVE_PRESET",
            (cfg or {}).get("lgbm_require_native_preset", ""),
        )
        or ""
    ).strip().lower() in {"1", "true", "yes", "y", "on"}
    if not native_required:
        return {"enabled": False, "reason": "native_preset_not_required"}
    selected = [str(c) for c in features if str(c) in X.columns]
    missing = [str(c) for c in features if str(c) not in X.columns]
    zero_std: list[str] = []
    nonzero_std: list[str] = []
    all_missing: list[str] = []
    for col in selected:
        vals = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=np.float32, copy=False)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            all_missing.append(col)
            zero_std.append(col)
            continue
        if float(np.nanstd(finite)) <= 1e-12:
            zero_std.append(col)
        else:
            nonzero_std.append(col)
    feature_count = int(len(features))
    zero_std_count = int(len(zero_std) + len(missing))
    nonzero_std_count = int(len(nonzero_std))
    zero_std_fraction = float(zero_std_count / max(feature_count, 1))
    min_nonzero_default = max(5, int(np.ceil(0.05 * max(feature_count, 1))))
    min_nonzero = int(
        os.environ.get(
            "EPM_LGBM_NATIVE_PRESET_MIN_NONZERO_FEATURES",
            str(min_nonzero_default),
        )
        or min_nonzero_default
    )
    max_zero_fraction = float(
        os.environ.get("EPM_LGBM_NATIVE_PRESET_MAX_ZERO_STD_FRACTION", "0.90")
        or 0.90
    )
    diag = {
        "enabled": True,
        "label": str(label),
        "feature_count": feature_count,
        "present_feature_count": int(len(selected)),
        "missing_feature_count": int(len(missing)),
        "zero_std_count": zero_std_count,
        "zero_std_fraction": zero_std_fraction,
        "nonzero_std_count": nonzero_std_count,
        "min_nonzero_required": int(min_nonzero),
        "max_zero_std_fraction": float(max_zero_fraction),
        "zero_std_preview": zero_std[:12],
        "missing_preview": missing[:12],
        "all_missing_preview": all_missing[:12],
        "preset_source": str(preset_source or ""),
    }
    if nonzero_std_count < min_nonzero or zero_std_fraction > max_zero_fraction:
        raise RuntimeError(
            "Native preset selected-feature matrix is effectively constant before "
            "LightGBM fit. This usually indicates failed feature injection or the "
            "wrong feature_source_run_id. "
            f"diagnostics={diag}"
        )
    return diag


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
    if LGBM_DISABLE_SELF_DISTILLATION:
        return 0
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


def _coerce_target(y: np.ndarray, classifier: bool, *, allow_soft_labels: bool = False) -> np.ndarray:
    if classifier:
        if allow_soft_labels:
            return np.clip(np.asarray(y, dtype=np.float32), 0.0, 1.0)
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


_VOL_NORM_TPSL_GEOMETRIES: tuple[tuple[str, float, float], ...] = (
    ("tp3_sl2", 3.0, 2.0),
    ("tp2_sl1", 2.0, 1.0),
)
_VOL_NORM_TPSL_TOP_FRACS: tuple[float, ...] = (0.30, 0.20, 0.10)


def _label_context_array(
    label_context: Mapping[str, Any] | None,
    names: tuple[str, ...],
    n: int,
) -> np.ndarray | None:
    if not isinstance(label_context, Mapping):
        return None
    for name in names:
        if name not in label_context:
            continue
        value = label_context.get(name)
        if value is None:
            continue
        try:
            arr = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if arr.ndim != 1 or len(arr) != n:
            continue
        return arr
    return None


def _label_context_take(
    label_context: Mapping[str, Any] | None,
    idx: Any,
    n: int,
) -> dict[str, np.ndarray] | None:
    if not isinstance(label_context, Mapping):
        return None
    idx_arr = np.asarray(idx, dtype=np.int64)
    if idx_arr.ndim != 1 or len(idx_arr) == 0:
        return None
    if int(np.min(idx_arr)) < 0 or int(np.max(idx_arr)) >= int(n):
        return None
    out: dict[str, np.ndarray] = {}
    for key, value in label_context.items():
        if value is None:
            continue
        try:
            arr = np.asarray(value)
        except (TypeError, ValueError):
            continue
        if arr.ndim != 1 or len(arr) != n:
            continue
        out[str(key)] = arr[idx_arr]
    return out or None


def _vol_normalized_tp_sl_precision_metrics(
    pred: np.ndarray,
    label_context: Mapping[str, Any] | None,
) -> dict[str, float]:
    out: dict[str, float] = {
        "vol_norm_tpsl_metrics_available": 0.0,
        "vol_norm_tpsl_metric_rows": 0.0,
    }
    for geom_name, _, _ in _VOL_NORM_TPSL_GEOMETRIES:
        out[f"baseline_{geom_name}_vol_norm"] = float("nan")
        out[f"support_{geom_name}_vol_norm"] = 0.0
        for frac in _VOL_NORM_TPSL_TOP_FRACS:
            pct = int(round(frac * 100.0))
            out[f"precision_at_{pct}_{geom_name}_vol_norm"] = float("nan")
            out[f"lift_at_{pct}_{geom_name}_vol_norm"] = float("nan")

    p = np.asarray(pred, dtype=np.float64)
    n = int(len(p))
    if n < 8:
        return out
    mfe = _label_context_array(
        label_context,
        ("mfe", "mfe_ret", "__mfe_ret__", "mfe_return", "__mfe__"),
        n,
    )
    mae = _label_context_array(
        label_context,
        ("mae", "mae_ret", "__mae_ret__", "mae_return", "__mae__"),
        n,
    )
    vol = _label_context_array(
        label_context,
        ("atr", "barrier_pct", "__barrier_pct__", "vol", "volatility", "tp", "__tp__"),
        n,
    )
    if mfe is None or mae is None or vol is None:
        return out

    tau_tp = _label_context_array(
        label_context,
        ("tau_tp", "__tau_tp__", "bars_to_tp", "__bars_to_tp__", "time_to_mfe", "__t_mfe__"),
        n,
    )
    tau_sl = _label_context_array(
        label_context,
        ("tau_sl", "__tau_sl__", "bars_to_sl", "__bars_to_sl__", "time_to_mae", "__t_mae__"),
        n,
    )
    valid = np.isfinite(p) & np.isfinite(mfe) & np.isfinite(mae) & np.isfinite(vol) & (vol > 0.0)
    if int(np.sum(valid)) < 8:
        return out
    p_v = p[valid]
    mfe_v = mfe[valid]
    mae_v = np.abs(mae[valid])
    vol_v = vol[valid]
    tau_tp_v = tau_tp[valid] if tau_tp is not None and len(tau_tp) == n else None
    tau_sl_v = tau_sl[valid] if tau_sl is not None and len(tau_sl) == n else None
    out["vol_norm_tpsl_metrics_available"] = 1.0
    out["vol_norm_tpsl_metric_rows"] = float(len(p_v))
    order = np.argsort(p_v)
    for geom_name, tp_mult, sl_mult in _VOL_NORM_TPSL_GEOMETRIES:
        tp_hit = mfe_v >= (float(tp_mult) * vol_v)
        sl_hit = mae_v >= (float(sl_mult) * vol_v)
        if tau_tp_v is not None and tau_sl_v is not None:
            tp_first = np.isfinite(tau_tp_v) & np.isfinite(tau_sl_v) & (tau_tp_v <= tau_sl_v)
            success = tp_hit & ((~sl_hit) | tp_first)
        else:
            success = tp_hit & (~sl_hit)
        y_win = success.astype(np.float64)
        baseline = float(np.mean(y_win)) if len(y_win) else float("nan")
        out[f"baseline_{geom_name}_vol_norm"] = baseline
        out[f"support_{geom_name}_vol_norm"] = float(np.sum(y_win))
        for frac in _VOL_NORM_TPSL_TOP_FRACS:
            pct = int(round(frac * 100.0))
            top_idx = _top_idx(order, frac, len(y_win))
            precision = float(np.mean(y_win[top_idx])) if len(top_idx) else float("nan")
            out[f"precision_at_{pct}_{geom_name}_vol_norm"] = precision
            out[f"lift_at_{pct}_{geom_name}_vol_norm"] = precision / max(baseline, 1e-6) if np.isfinite(precision) else float("nan")
    return out


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
        numeric_values: list[float] = []
        for m in fold_metrics:
            raw_value = m.get(key, np.nan)
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            numeric_values.append(value)
        arr = np.asarray(numeric_values, dtype=np.float64)
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
        "precision_at_30_tp3_sl2_vol_norm",
        "precision_at_20_tp3_sl2_vol_norm",
        "precision_at_10_tp3_sl2_vol_norm",
        "precision_at_30_tp2_sl1_vol_norm",
        "precision_at_20_tp2_sl1_vol_norm",
        "precision_at_10_tp2_sl1_vol_norm",
        "lift_at_30_tp3_sl2_vol_norm",
        "lift_at_20_tp3_sl2_vol_norm",
        "lift_at_10_tp3_sl2_vol_norm",
        "lift_at_30_tp2_sl1_vol_norm",
        "lift_at_20_tp2_sl1_vol_norm",
        "lift_at_10_tp2_sl1_vol_norm",
        "baseline_tp3_sl2_vol_norm",
        "baseline_tp2_sl1_vol_norm",
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


def _lgbm_stability_selection_cap(selection_rows: int) -> int:
    rows = int(max(0, selection_rows))
    if rows <= 0:
        return 0
    cap = rows
    if int(LGBM_RACE_MAX_ROWS) > 0:
        cap = min(cap, int(LGBM_RACE_MAX_ROWS))
    return max(1, min(int(cap), rows))


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


def _purged_time_splitter(
    y: np.ndarray,
    classifier: bool,
    random_state: int,
    *,
    timestamps: Any,
    n_splits: int = LGBM_CV_SPLITS,
    purge_hours: float = LGBM_PURGE_HOURS,
) -> tuple[Any, np.ndarray]:
    y_arr = np.asarray(y)
    n = len(y_arr)
    y_split = np.asarray(y_arr >= 0.5, dtype=np.int8) if classifier else np.asarray(y_arr, dtype=np.float32)
    ns = _timestamp_ns(timestamps, n)
    if ns is None or n < 2:
        return _splitter(y_arr, classifier, random_state, n_splits=n_splits)
    valid = ns != np.iinfo(np.int64).min
    unique_ts = np.asarray(sorted(np.unique(ns[valid]).tolist()), dtype=np.int64)
    n_splits_local = max(2, min(int(n_splits), int(len(unique_ts))))
    if len(unique_ts) < n_splits_local:
        return _splitter(y_arr, classifier, random_state, n_splits=n_splits)
    purge_ns = int(max(0.0, float(purge_hours)) * 3600.0 * 1_000_000_000.0)
    ts_blocks = np.array_split(unique_ts, n_splits_local)
    all_idx = np.arange(n, dtype=np.int32)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for block in ts_blocks:
        if len(block) == 0:
            continue
        lo = int(np.min(block))
        hi = int(np.max(block))
        va_mask = valid & (ns >= lo) & (ns <= hi)
        tr_mask = valid & ((ns < lo - purge_ns) | (ns > hi + purge_ns))
        va = all_idx[va_mask]
        tr = all_idx[tr_mask]
        if len(va) == 0 or len(tr) == 0:
            continue
        if classifier and len(np.unique(y_split[tr])) < 2:
            continue
        folds.append((tr.astype(np.int32, copy=False), va.astype(np.int32, copy=False)))
    if len(folds) < 2:
        return _splitter(y_arr, classifier, random_state, n_splits=n_splits)
    return _PrecomputedSplitter(folds), y_split


def _cv_splitter(
    y: np.ndarray,
    classifier: bool,
    random_state: int,
    *,
    timestamps: Any = None,
    n_splits: int = LGBM_CV_SPLITS,
) -> tuple[Any, np.ndarray]:
    if LGBM_CV_MODE in {"interleaved", "interleaved_spread"}:
        return _interleaved_spread_splitter(
            y,
            classifier,
            n_splits=n_splits,
        )
    if bool(LGBM_PURGED_CV) or LGBM_CV_MODE == "purged_time":
        return _purged_time_splitter(
            y,
            classifier,
            random_state,
            timestamps=timestamps,
            n_splits=n_splits,
            purge_hours=LGBM_PURGE_HOURS,
        )
    return _splitter(y, classifier, random_state, n_splits=n_splits)


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
        "objective": "cross_entropy" if classifier and LGBM_TRUE_SOFT_LABELS else ("binary" if classifier else "regression"),
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
    if classifier and LGBM_TRUE_SOFT_LABELS:
        params["objective"] = "cross_entropy"
        params.pop("scale_pos_weight", None)
        params.pop("is_unbalance", None)
        params.pop("class_weight", None)
    depth = int(params.get("max_depth", 4))
    if "num_leaves" not in params or params.get("num_leaves") is None:
        params["num_leaves"] = int(2 ** depth)
    params["num_leaves"] = int(min(int(params["num_leaves"]), 2 ** max(depth, 1)))
    return params


def _make_lgbm_model(params: dict[str, Any], classifier: bool) -> Any:
    import lightgbm as lgb

    params = _effective_lgbm_params(params, classifier=classifier)
    if classifier and LGBM_TRUE_SOFT_LABELS:
        return lgb.LGBMRegressor(**params)
    if classifier:
        return lgb.LGBMClassifier(**params)
    return lgb.LGBMRegressor(**params)


def _attach_lgbm_leaf_training_diagnostics(
    model: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None = None,
    *,
    include_target_stats: bool = True,
    include_centroid_stats: bool = True,
) -> None:
    """Persist compact train-fold leaf summaries for validation diagnostics."""
    n = int(len(X_train))
    if n == 0:
        return
    try:
        predict_kwargs: dict[str, Any] = {}
        if LGBM_META_LEAF_MAX_TREES > 0:
            predict_kwargs["num_iteration"] = int(LGBM_META_LEAF_MAX_TREES)
        leaves = np.asarray(model.predict(X_train, pred_leaf=True, **predict_kwargs), dtype=np.int64)
    except TypeError:
        try:
            leaves = np.asarray(model.predict(X_train, pred_leaf=True), dtype=np.int64)
        except Exception:
            return
    except Exception:
        return
    if leaves.ndim == 1:
        leaves = leaves.reshape(n, 1)
    if leaves.shape[0] != n or leaves.shape[1] == 0:
        return
    if LGBM_META_LEAF_MAX_TREES > 0:
        leaves = leaves[:, : int(LGBM_META_LEAF_MAX_TREES)]
    X_np: np.ndarray | None = None
    if include_centroid_stats:
        X_np = (
            X_train.replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float32, copy=False)
        )
        if X_np.ndim != 2 or X_np.shape[0] != n:
            X_np = None
            include_centroid_stats = False
    y_np: np.ndarray | None = None
    pred_np: np.ndarray | None = None
    error_np: np.ndarray | None = None
    if include_target_stats:
        y_np = np.asarray(y_train, dtype=np.float32).reshape(-1)
        if len(y_np) != n:
            y_np = None
            include_target_stats = False
        else:
            y_np = np.nan_to_num(y_np, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            try:
                mode = "classifier" if hasattr(model, "predict_proba") else "regressor"
                pred_np = _predict_lgbm_raw_batched(model, X_train, mode, batch_size=200000)
                if len(pred_np) == n:
                    error_np = np.abs(y_np - pred_np).astype(np.float32)
                else:
                    pred_np = None
            except Exception:
                pred_np = None
                error_np = None
    sw_np: np.ndarray | None = None
    if sample_weight is not None:
        try:
            sw_np = np.asarray(sample_weight, dtype=np.float32).reshape(-1)
            if len(sw_np) != n:
                sw_np = None
            else:
                sw_np = np.nan_to_num(sw_np, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32)
        except Exception:
            sw_np = None
    diagnostics: list[dict[str, Any]] = []
    for tree_i in range(leaves.shape[1]):
        leaf_ids, inverse = np.unique(leaves[:, tree_i], return_inverse=True)
        if len(leaf_ids) == 0:
            diagnostics.append({})
            continue
        counts = np.bincount(inverse, minlength=len(leaf_ids)).astype(np.float32)
        denom = np.maximum(counts, 1.0)
        train_freq = (counts / max(float(n), 1.0)).astype(np.float32)
        tree_diag: dict[str, Any] = {
            "leaf_ids": leaf_ids.astype(np.int32, copy=False),
            "train_freq": train_freq,
        }
        if sw_np is not None:
            weight_sum = np.bincount(inverse, weights=sw_np, minlength=len(leaf_ids)).astype(np.float32)
            tree_diag["sample_weight_mean"] = (weight_sum / denom).astype(np.float32)
        if include_centroid_stats and X_np is not None:
            centroids = np.zeros((len(leaf_ids), X_np.shape[1]), dtype=np.float32)
            np.add.at(centroids, inverse, X_np)
            centroids = (centroids / denom[:, None]).astype(np.float32, copy=False)
            row_centroids = centroids[inverse]
            row_centroid_dist = np.sqrt(
                np.mean(np.square(X_np - row_centroids, dtype=np.float32), axis=1)
            ).astype(np.float32)
            centroid_radius_sum = np.bincount(
                inverse,
                weights=row_centroid_dist,
                minlength=len(leaf_ids),
            ).astype(np.float32)
            tree_diag["centroids"] = centroids
            tree_diag["centroid_radius"] = (centroid_radius_sum / denom).astype(np.float32)
        if include_target_stats and y_np is not None:
            target_sum = np.bincount(inverse, weights=y_np, minlength=len(leaf_ids)).astype(np.float32)
            target_sq_sum = np.bincount(
                inverse,
                weights=np.square(y_np, dtype=np.float32),
                minlength=len(leaf_ids),
            ).astype(np.float32)
            target_mean = (target_sum / denom).astype(np.float32)
            target_var = np.maximum(target_sq_sum / denom - np.square(target_mean), 0.0)
            target_std = np.sqrt(target_var).astype(np.float32)
            target_iqr = np.zeros(len(leaf_ids), dtype=np.float32)
            target_range = np.zeros(len(leaf_ids), dtype=np.float32)
            target_abs_mean = np.zeros(len(leaf_ids), dtype=np.float32)
            order = np.argsort(inverse, kind="mergesort")
            sorted_inverse = inverse[order]
            sorted_y = y_np[order]
            bounds = np.flatnonzero(np.diff(sorted_inverse)) + 1
            starts = np.r_[0, bounds]
            stops = np.r_[bounds, len(sorted_inverse)]
            for start, stop in zip(starts, stops):
                idx = int(sorted_inverse[start])
                vals = sorted_y[start:stop]
                if vals.size == 0:
                    continue
                q10, q25, q75, q90 = np.nanpercentile(vals, [10.0, 25.0, 75.0, 90.0])
                target_iqr[idx] = float(q75 - q25)
                target_range[idx] = float(q90 - q10)
                target_abs_mean[idx] = float(np.nanmean(np.abs(vals - target_mean[idx])))
            tree_diag["target_mean"] = target_mean
            tree_diag["target_std"] = target_std
            tree_diag["target_iqr"] = target_iqr
            tree_diag["target_range"] = target_range
            tree_diag["target_abs_mean"] = target_abs_mean
            if pred_np is not None:
                pred_sum = np.bincount(inverse, weights=pred_np, minlength=len(leaf_ids)).astype(np.float32)
                tree_diag["pred_mean"] = (pred_sum / denom).astype(np.float32)
            if error_np is not None:
                err_sum = np.bincount(inverse, weights=error_np, minlength=len(leaf_ids)).astype(np.float32)
                tree_diag["error_mean"] = (err_sum / denom).astype(np.float32)
        diagnostics.append(tree_diag)
    try:
        setattr(model, "_ares_lgbm_leaf_training_diagnostics_", diagnostics)
        setattr(
            model,
            "_ares_lgbm_leaf_training_feature_count_",
            int(X_train.shape[1]) if hasattr(X_train, "shape") else 0,
        )
    except Exception:
        return


def _effective_lgbm_params(params: dict[str, Any], *, classifier: bool) -> dict[str, Any]:
    out = dict(params)
    if LGBM_N_ESTIMATORS_CAP > 0 and "n_estimators" in out:
        out["n_estimators"] = min(
            int(out.get("n_estimators", LGBM_N_ESTIMATORS_CAP)),
            int(LGBM_N_ESTIMATORS_CAP),
        )
    if classifier and LGBM_TRUE_SOFT_LABELS:
        out["objective"] = "cross_entropy"
        out.pop("scale_pos_weight", None)
        out.pop("is_unbalance", None)
        out.pop("class_weight", None)
    return out


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
    attach_leaf_diagnostics: bool = False,
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
    if attach_leaf_diagnostics and (
        LGBM_META_LEAF_DIAGNOSTICS
        or LGBM_META_LEAF_SUPPORT_DIAGNOSTICS
        or LGBM_META_LEAF_TARGET_DIAGNOSTICS
        or LGBM_META_LEAF_CENTROID_DIAGNOSTICS
    ):
        _attach_lgbm_leaf_training_diagnostics(
            model,
            X_train,
            y_train,
            sample_weight=sample_weight,
            include_target_stats=(
                LGBM_META_LEAF_DIAGNOSTICS or LGBM_META_LEAF_TARGET_DIAGNOSTICS
            ),
            include_centroid_stats=(
                LGBM_META_LEAF_DIAGNOSTICS or LGBM_META_LEAF_CENTROID_DIAGNOSTICS
            ),
        )
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


def _score_path_predictions(
    models: list[Any],
    X: pd.DataFrame,
    mode: str,
    *,
    final_pred: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    n = int(len(X))
    if n == 0 or not models:
        return None
    scores: list[np.ndarray] = []
    for frac in (0.50, 0.75):
        per_model: list[np.ndarray] = []
        for model in models:
            n_iter = _model_num_iterations(model)
            num_iter = int(np.ceil(float(frac) * n_iter)) if n_iter > 0 else None
            raw = _predict_lgbm_raw_score(model, X, mode, num_iteration=num_iter)
            per_model.append(_sigmoid(raw) if mode == "classifier" else raw.astype(np.float32))
        scores.append(np.mean(np.vstack(per_model).astype(np.float32, copy=False), axis=0).astype(np.float32))
    if final_pred is None:
        per_model = [_predict_lgbm_raw(model, X, mode) for model in models]
        final_score = np.mean(np.vstack(per_model).astype(np.float32, copy=False), axis=0).astype(np.float32)
    else:
        final_score = np.asarray(final_pred, dtype=np.float32).reshape(-1)
    if len(final_score) != n:
        return None
    scores.append(final_score.astype(np.float32, copy=False))
    return scores[0], scores[1], scores[2]


def _safe_rank_pct(values: np.ndarray) -> np.ndarray:
    vals = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if len(vals) == 0:
        return np.zeros(0, dtype=np.float32)
    return pd.Series(vals).rank(method="average", pct=True).to_numpy(dtype=np.float32)


def _rank_pct_against_reference(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    vals = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if len(vals) == 0:
        return np.zeros(0, dtype=np.float32)
    ref = np.asarray(reference, dtype=np.float32)
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        return _safe_rank_pct(vals)
    ref = np.sort(ref)
    rank_pct = np.searchsorted(ref, vals, side="right").astype(np.float32) / max(float(ref.size), 1.0)
    return np.clip(rank_pct, 0.0, 1.0).astype(np.float32)


def _score_margin(score: np.ndarray, frac: float) -> np.ndarray:
    vals = np.nan_to_num(np.asarray(score, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if len(vals) == 0:
        return vals
    threshold = float(np.nanquantile(vals, 1.0 - float(frac)))
    return (vals - threshold).astype(np.float32)


def _score_margin_against_reference(score: np.ndarray, reference: np.ndarray, frac: float) -> np.ndarray:
    vals = np.nan_to_num(np.asarray(score, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if len(vals) == 0:
        return vals
    ref = np.asarray(reference, dtype=np.float32)
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        return _score_margin(vals, frac)
    threshold = float(np.nanquantile(ref, 1.0 - float(frac)))
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


def _node_prediction_value(node: dict[str, Any], fallback: float = 0.0) -> float:
    if not isinstance(node, dict):
        return float(fallback)
    if "leaf_index" in node:
        value = node.get("leaf_value", fallback)
    else:
        value = node.get("internal_value", fallback)
    try:
        out = float(value)
    except Exception:
        out = float(fallback)
    return out if np.isfinite(out) else float(fallback)


def _path_contrib_summary_from_values(values: list[float]) -> tuple[float, float, float, float, float, float, float, float]:
    if not values:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    vals = np.asarray(values, dtype=np.float32)
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    abs_vals = np.abs(vals)
    abs_sum = float(np.sum(abs_vals))
    if abs_sum <= 1e-12:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sorted_abs = np.sort(abs_vals)[::-1]
    share = abs_vals / abs_sum
    entropy = float(
        -np.sum(np.where(share > 0.0, share * np.log(share + 1e-12), 0.0))
        / max(np.log(max(len(abs_vals), 2)), 1e-12)
    )
    material = float(np.sum(share >= 0.01))
    return (
        abs_sum,
        float(np.sqrt(np.sum(np.square(vals, dtype=np.float32)))),
        entropy,
        float(sorted_abs[0]),
        float(np.sum(sorted_abs[: min(3, len(sorted_abs))])),
        float(np.sum(np.maximum(vals, 0.0))),
        float(np.sum(np.minimum(vals, 0.0))),
        material,
    )


def _walk_lgbm_tree_path_contrib(
    node: dict[str, Any],
    path_contrib: dict[int, float],
    out: dict[int, tuple[float, float, float, float, float, float, float, float]],
) -> None:
    if "leaf_index" in node:
        out[int(node.get("leaf_index", 0))] = _path_contrib_summary_from_values(
            list(path_contrib.values())
        )
        return
    split_feature = int(node.get("split_feature", -1) if node.get("split_feature", -1) is not None else -1)
    parent_value = _node_prediction_value(node, 0.0)
    for child_key in ("left_child", "right_child"):
        child = node.get(child_key)
        if not isinstance(child, dict):
            continue
        child_value = _node_prediction_value(child, parent_value)
        if split_feature >= 0:
            next_path = dict(path_contrib)
            next_path[split_feature] = float(next_path.get(split_feature, 0.0) + child_value - parent_value)
        else:
            next_path = path_contrib
        _walk_lgbm_tree_path_contrib(child, next_path, out)


def _path_contrib_metadata(model: Any) -> list[dict[int, tuple[float, float, float, float, float, float, float, float]]]:
    cached = getattr(model, "_ares_lgbm_path_contrib_metadata_", None)
    if isinstance(cached, list):
        return cached
    booster = getattr(model, "booster_", None)
    if booster is None:
        return []
    try:
        dumped = booster.dump_model()
    except Exception:
        return []
    meta: list[dict[int, tuple[float, float, float, float, float, float, float, float]]] = []
    for tree in dumped.get("tree_info", []):
        tree_meta: dict[int, tuple[float, float, float, float, float, float, float, float]] = {}
        root = tree.get("tree_structure", {}) if isinstance(tree, dict) else {}
        if isinstance(root, dict):
            _walk_lgbm_tree_path_contrib(root, {}, tree_meta)
        meta.append(tree_meta)
    try:
        setattr(model, "_ares_lgbm_path_contrib_metadata_", meta)
    except Exception:
        pass
    return meta


def _leaf_value_matrix(
    models: list[Any],
    X: pd.DataFrame,
    *,
    max_trees: int = 0,
) -> np.ndarray | None:
    n = int(len(X))
    if n == 0 or not models:
        return None
    cols: list[np.ndarray] = []
    model_scale = 1.0 / max(float(len(models)), 1.0)
    for model in models:
        meta = _leaf_metadata(model)
        if not meta:
            continue
        try:
            predict_kwargs: dict[str, Any] = {}
            if max_trees > 0:
                predict_kwargs["num_iteration"] = int(max_trees)
            leaves = np.asarray(model.predict(X, pred_leaf=True, **predict_kwargs), dtype=np.int32)
        except TypeError:
            try:
                leaves = np.asarray(model.predict(X, pred_leaf=True), dtype=np.int32)
            except Exception:
                continue
        except Exception:
            continue
        if leaves.ndim == 1:
            leaves = leaves.reshape(n, 1)
        tree_n = min(leaves.shape[1], len(meta))
        if max_trees > 0:
            tree_n = min(tree_n, int(max_trees))
        for tree_i in range(tree_n):
            tree_meta = meta[tree_i]
            values = np.zeros(n, dtype=np.float32)
            if tree_meta:
                leaf_ids = np.fromiter(tree_meta.keys(), dtype=np.int32, count=len(tree_meta))
                leaf_values = np.asarray(
                    [tree_meta[int(leaf_id)][3] for leaf_id in leaf_ids],
                    dtype=np.float32,
                )
                order = np.argsort(leaf_ids, kind="mergesort")
                leaf_ids_sorted = leaf_ids[order]
                leaf_values_sorted = leaf_values[order]
                leaf_col = np.asarray(leaves[:, tree_i], dtype=np.int32)
                pos = np.searchsorted(leaf_ids_sorted, leaf_col)
                pos_clip = np.clip(pos, 0, max(leaf_ids_sorted.size - 1, 0))
                valid = (pos < leaf_ids_sorted.size) & (leaf_ids_sorted[pos_clip] == leaf_col)
                if np.any(valid):
                    values[valid] = leaf_values_sorted[pos[valid]]
            cols.append((values * model_scale).astype(np.float32, copy=False))
    if not cols:
        return None
    return np.vstack(cols).T.astype(np.float32, copy=False)


def _safe_row_stat(mat: np.ndarray | None, stat: str, *, q: float | None = None) -> np.ndarray:
    if mat is None or mat.size == 0:
        return np.zeros(0, dtype=np.float32)
    n = int(mat.shape[0])
    out = np.zeros(n, dtype=np.float32)
    finite_rows = np.isfinite(mat).any(axis=1)
    if not np.any(finite_rows):
        return out
    sub = mat[finite_rows]
    with np.errstate(invalid="ignore", divide="ignore"):
        if stat == "mean":
            vals = np.nanmean(sub, axis=1)
        elif stat == "median":
            vals = np.nanmedian(sub, axis=1)
        elif stat == "std":
            vals = np.nanstd(sub, axis=1)
        elif stat == "min":
            vals = np.nanmin(sub, axis=1)
        elif stat == "max":
            vals = np.nanmax(sub, axis=1)
        elif stat == "percentile" and q is not None:
            vals = np.nanpercentile(sub, float(q), axis=1)
        else:
            vals = np.zeros(len(sub), dtype=np.float32)
    out[finite_rows] = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return out


def _append_leaf_diagnostics(
    frames: dict[str, np.ndarray],
    models: list[Any],
    X: pd.DataFrame,
    *,
    prediction: np.ndarray | None = None,
    full_diagnostics: bool = LGBM_META_LEAF_DIAGNOSTICS,
    support_diagnostics: bool = LGBM_META_LEAF_SUPPORT_DIAGNOSTICS,
    target_diagnostics: bool = LGBM_META_LEAF_TARGET_DIAGNOSTICS,
    centroid_diagnostics: bool = LGBM_META_LEAF_CENTROID_DIAGNOSTICS,
) -> None:
    n = len(X)
    full_diagnostics = bool(full_diagnostics)
    support_diagnostics = bool(support_diagnostics or full_diagnostics)
    target_diagnostics = bool(target_diagnostics or full_diagnostics)
    centroid_diagnostics = bool(centroid_diagnostics or full_diagnostics)
    pred_arr: np.ndarray | None = None
    if prediction is not None:
        try:
            pred_candidate = np.asarray(prediction, dtype=np.float32).reshape(-1)
            if len(pred_candidate) == n:
                pred_arr = np.nan_to_num(pred_candidate, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            pred_arr = None
    counts: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    abs_values: list[np.ndarray] = []
    train_freqs: list[np.ndarray] = []
    surprisals: list[np.ndarray] = []
    target_means: list[np.ndarray] = []
    target_stds: list[np.ndarray] = []
    target_iqrs: list[np.ndarray] = []
    target_ranges: list[np.ndarray] = []
    target_abs_means: list[np.ndarray] = []
    target_pred_means: list[np.ndarray] = []
    target_error_means: list[np.ndarray] = []
    target_positive_fracs: list[np.ndarray] = []
    centroid_radii: list[np.ndarray] = []
    centroid_dists: list[np.ndarray] = []
    centroid_norm_dists: list[np.ndarray] = []
    X_np: np.ndarray | None = None
    if centroid_diagnostics and any(
        bool(getattr(model, "_ares_lgbm_leaf_training_diagnostics_", []) or [])
        for model in models
    ):
        X_np = (
            X.replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float32, copy=False)
        )
    for model in models:
        meta = _leaf_metadata(model)
        if not meta:
            continue
        train_diag = (
            getattr(model, "_ares_lgbm_leaf_training_diagnostics_", []) or []
            if (support_diagnostics or target_diagnostics or centroid_diagnostics)
            else []
        )
        try:
            predict_kwargs: dict[str, Any] = {}
            if LGBM_META_LEAF_MAX_TREES > 0:
                predict_kwargs["num_iteration"] = int(LGBM_META_LEAF_MAX_TREES)
            leaves = np.asarray(model.predict(X, pred_leaf=True, **predict_kwargs), dtype=np.int32)
        except TypeError:
            try:
                leaves = np.asarray(model.predict(X, pred_leaf=True), dtype=np.int32)
            except Exception:
                continue
        except Exception:
            continue
        if leaves.ndim == 1:
            leaves = leaves.reshape(n, 1)
        tree_n = min(leaves.shape[1], len(meta))
        if LGBM_META_LEAF_MAX_TREES > 0:
            tree_n = min(tree_n, int(LGBM_META_LEAF_MAX_TREES))
        for tree_i in range(tree_n):
            tree_meta = meta[tree_i]
            c = np.zeros(n, dtype=np.float32)
            w = np.zeros(n, dtype=np.float32)
            d = np.zeros(n, dtype=np.float32)
            v = np.zeros(n, dtype=np.float32)
            if tree_meta:
                leaf_ids = np.fromiter(tree_meta.keys(), dtype=np.int32, count=len(tree_meta))
                leaf_vals = np.asarray(
                    [tree_meta[int(leaf_id)] for leaf_id in leaf_ids],
                    dtype=np.float32,
                )
                if leaf_ids.size and leaf_vals.ndim == 2 and leaf_vals.shape[1] >= 4:
                    order = np.argsort(leaf_ids, kind="mergesort")
                    leaf_ids_sorted = leaf_ids[order]
                    leaf_vals_sorted = leaf_vals[order]
                    leaf_col = np.asarray(leaves[:, tree_i], dtype=np.int32)
                    pos = np.searchsorted(leaf_ids_sorted, leaf_col)
                    pos_clipped = np.clip(pos, 0, max(leaf_ids_sorted.size - 1, 0))
                    valid = (pos < leaf_ids_sorted.size) & (leaf_ids_sorted[pos_clipped] == leaf_col)
                    if np.any(valid):
                        matched = leaf_vals_sorted[pos[valid]]
                        c[valid] = matched[:, 0]
                        w[valid] = matched[:, 1]
                        d[valid] = matched[:, 2]
                        v[valid] = np.abs(matched[:, 3])
            counts.append(c)
            weights.append(w)
            depths.append(d)
            abs_values.append(v)
            if not (support_diagnostics or target_diagnostics or centroid_diagnostics):
                continue
            tf = np.full(n, np.nan, dtype=np.float32)
            sp = np.full(n, np.nan, dtype=np.float32)
            tm = np.full(n, np.nan, dtype=np.float32)
            ts = np.full(n, np.nan, dtype=np.float32)
            ti = np.full(n, np.nan, dtype=np.float32)
            tr = np.full(n, np.nan, dtype=np.float32)
            ta = np.full(n, np.nan, dtype=np.float32)
            tpm = np.full(n, np.nan, dtype=np.float32)
            tem = np.full(n, np.nan, dtype=np.float32)
            tpf = np.full(n, np.nan, dtype=np.float32)
            cr = np.full(n, np.nan, dtype=np.float32)
            cd = np.full(n, np.nan, dtype=np.float32)
            cdn = np.full(n, np.nan, dtype=np.float32)
            tree_train_diag = (
                train_diag[tree_i]
                if tree_i < len(train_diag) and isinstance(train_diag[tree_i], dict)
                else {}
            )
            diag_leaf_ids = (
                np.asarray(tree_train_diag.get("leaf_ids"), dtype=np.int32)
                if tree_train_diag and tree_train_diag.get("leaf_ids") is not None
                else np.zeros(0, dtype=np.int32)
            )
            centroids = tree_train_diag.get("centroids") if tree_train_diag else None
            centroids_arr = (
                np.asarray(centroids, dtype=np.float32)
                if centroids is not None
                else None
            )
            centroids_compatible = (
                centroid_diagnostics
                and
                centroids_arr is not None
                and centroids_arr.ndim == 2
                and X_np is not None
                and X_np.shape[1] == centroids_arr.shape[1]
            )
            if tree_train_diag and diag_leaf_ids.size:
                leaf_col = np.asarray(leaves[:, tree_i], dtype=np.int32)
                pos = np.searchsorted(diag_leaf_ids, leaf_col)
                pos_clip = np.clip(pos, 0, max(diag_leaf_ids.size - 1, 0))
                valid = (pos < diag_leaf_ids.size) & (diag_leaf_ids[pos_clip] == leaf_col)
                pos_valid = pos[valid]
                if support_diagnostics:
                    freq_arr = tree_train_diag.get("train_freq")
                    if freq_arr is not None:
                        freq_values = np.asarray(freq_arr, dtype=np.float32)
                        ok = pos_valid < len(freq_values)
                        if np.any(ok):
                            vv = np.flatnonzero(valid)[ok]
                            freq = freq_values[pos_valid[ok]]
                            tf[vv] = freq
                            sp[vv] = -np.log(np.maximum(freq, 1e-12)).astype(np.float32)
                if target_diagnostics:
                    for arr_name, dst in (
                        ("target_mean", tm),
                        ("target_std", ts),
                        ("target_iqr", ti),
                        ("target_range", tr),
                        ("target_abs_mean", ta),
                        ("pred_mean", tpm),
                        ("error_mean", tem),
                    ):
                        arr = tree_train_diag.get(arr_name)
                        if arr is None:
                            continue
                        values = np.asarray(arr, dtype=np.float32)
                        ok = pos_valid < len(values)
                        if np.any(ok):
                            dst[np.flatnonzero(valid)[ok]] = values[pos_valid[ok]]
                    finite_tm = np.isfinite(tm)
                    if np.any(finite_tm):
                        tpf[finite_tm] = (tm[finite_tm] >= 0.5).astype(np.float32)
                if centroid_diagnostics:
                    radius_arr = tree_train_diag.get("centroid_radius")
                    radius = np.full(n, np.nan, dtype=np.float32)
                    if radius_arr is not None:
                        radius_values = np.asarray(radius_arr, dtype=np.float32)
                        ok = pos_valid < len(radius_values)
                        if np.any(ok):
                            vv = np.flatnonzero(valid)[ok]
                            radius[vv] = radius_values[pos_valid[ok]]
                            cr[vv] = radius[vv]
                    if centroids_compatible:
                        ok = pos_valid < len(centroids_arr)
                        if np.any(ok):
                            vv = np.flatnonzero(valid)[ok]
                            diff = X_np[vv] - centroids_arr[pos_valid[ok]]
                            dist = np.sqrt(np.mean(np.square(diff, dtype=np.float32), axis=1)).astype(np.float32)
                            cd[vv] = dist
                            cdn[vv] = dist / np.maximum(radius[vv], 1e-6)
            if tree_train_diag and diag_leaf_ids.size and support_diagnostics:
                train_freqs.append(tf)
                surprisals.append(sp)
            if tree_train_diag and diag_leaf_ids.size and target_diagnostics:
                target_means.append(tm)
                target_stds.append(ts)
                target_iqrs.append(ti)
                target_ranges.append(tr)
                target_abs_means.append(ta)
                target_pred_means.append(tpm)
                target_error_means.append(tem)
                target_positive_fracs.append(tpf)
            if tree_train_diag and diag_leaf_ids.size and centroid_diagnostics:
                centroid_radii.append(cr)
                centroid_dists.append(cd)
                centroid_norm_dists.append(cdn)
    if not counts:
        for name in LGBM_META_LEAF_FEATURE_NAMES:
            frames[name] = np.zeros(n, dtype=np.float32)
        return
    count_mat = np.vstack(counts).T.astype(np.float32)
    weight_mat = np.vstack(weights).T.astype(np.float32)
    depth_mat = np.vstack(depths).T.astype(np.float32)
    value_mat = np.vstack(abs_values).T.astype(np.float32)
    global_count_p10 = float(np.nanpercentile(count_mat, 10.0)) if count_mat.size else 0.0
    global_value_p90 = float(np.nanpercentile(value_mat, 90.0)) if value_mat.size else 0.0
    frames["leaf_count_mean"] = np.nanmean(count_mat, axis=1).astype(np.float32)
    frames["leaf_count_median"] = np.nanmedian(count_mat, axis=1).astype(np.float32)
    frames["leaf_count_q25"] = np.nanpercentile(count_mat, 25.0, axis=1).astype(np.float32)
    frames["leaf_count_p10"] = np.nanpercentile(count_mat, 10.0, axis=1).astype(np.float32)
    frames["leaf_count_min"] = np.nanmin(count_mat, axis=1).astype(np.float32)
    frames["rare_leaf_fraction"] = np.mean(count_mat <= max(global_count_p10, 1.0), axis=1).astype(np.float32)
    frames["leaf_weight_mean"] = np.nanmean(weight_mat, axis=1).astype(np.float32)
    frames["leaf_weight_p10"] = np.nanpercentile(weight_mat, 10.0, axis=1).astype(np.float32)
    frames["leaf_depth_mean"] = np.nanmean(depth_mat, axis=1).astype(np.float32)
    frames["leaf_depth_std"] = np.nanstd(depth_mat, axis=1).astype(np.float32)
    frames["leaf_depth_max"] = np.nanmax(depth_mat, axis=1).astype(np.float32)
    frames["leaf_value_abs_mean"] = np.nanmean(value_mat, axis=1).astype(np.float32)
    frames["leaf_value_abs_std"] = np.nanstd(value_mat, axis=1).astype(np.float32)
    frames["leaf_value_abs_p90"] = np.nanpercentile(value_mat, 90.0, axis=1).astype(np.float32)
    frames["leaf_value_abs_max"] = np.nanmax(value_mat, axis=1).astype(np.float32)
    frames["large_leaf_value_fraction"] = np.mean(value_mat >= max(global_value_p90, 1e-12), axis=1).astype(np.float32)
    if train_freqs:
        train_freq_mat = np.vstack(train_freqs).T.astype(np.float32)
        surprisal_mat = np.vstack(surprisals).T.astype(np.float32)
        frames["leaf_train_freq_mean"] = _safe_row_stat(train_freq_mat, "mean")
        frames["leaf_train_freq_p90"] = _safe_row_stat(train_freq_mat, "percentile", q=90.0)
        frames["leaf_train_freq_p10"] = _safe_row_stat(train_freq_mat, "percentile", q=10.0)
        frames["leaf_train_freq_min"] = _safe_row_stat(train_freq_mat, "min")
        frames["leaf_train_freq_max"] = _safe_row_stat(train_freq_mat, "max")
        frames["leaf_train_freq_std"] = _safe_row_stat(train_freq_mat, "std")
        frames["leaf_surprisal_mean"] = _safe_row_stat(surprisal_mat, "mean")
        frames["leaf_surprisal_p90"] = _safe_row_stat(surprisal_mat, "percentile", q=90.0)
        frames["leaf_surprisal_max"] = _safe_row_stat(surprisal_mat, "max")
        frames["leaf_low_freq_fraction"] = np.nan_to_num(
            np.nanmean(train_freq_mat <= 0.01, axis=1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)
        frames["leaf_proximity_mean"] = frames["leaf_train_freq_mean"].astype(np.float32, copy=False)
        frames["leaf_proximity_p90"] = frames["leaf_train_freq_p90"].astype(np.float32, copy=False)
        frames["leaf_proximity_max"] = frames["leaf_train_freq_max"].astype(np.float32, copy=False)
        frames["leaf_model_space_distance_mean"] = np.clip(
            1.0 - frames["leaf_proximity_mean"],
            0.0,
            1.0,
        ).astype(np.float32)
        frames["leaf_model_space_distance_p10"] = np.clip(
            1.0 - frames["leaf_proximity_p90"],
            0.0,
            1.0,
        ).astype(np.float32)
    else:
        for name in (
            "leaf_train_freq_mean", "leaf_train_freq_p90",
            "leaf_train_freq_p10", "leaf_train_freq_min",
            "leaf_train_freq_max", "leaf_train_freq_std",
            "leaf_surprisal_mean", "leaf_surprisal_p90",
            "leaf_surprisal_max", "leaf_low_freq_fraction",
            "leaf_proximity_mean", "leaf_proximity_p90",
            "leaf_proximity_max", "leaf_model_space_distance_mean",
            "leaf_model_space_distance_p10",
        ):
            frames[name] = np.zeros(n, dtype=np.float32)
    if target_iqrs:
        target_mean_mat = np.vstack(target_means).T.astype(np.float32)
        target_std_mat = np.vstack(target_stds).T.astype(np.float32)
        target_iqr_mat = np.vstack(target_iqrs).T.astype(np.float32)
        target_range_mat = np.vstack(target_ranges).T.astype(np.float32)
        target_abs_mean_mat = np.vstack(target_abs_means).T.astype(np.float32)
        frames["leaf_target_mean_mean"] = _safe_row_stat(target_mean_mat, "mean")
        frames["leaf_target_mean_std"] = _safe_row_stat(target_mean_mat, "std")
        frames["leaf_target_mean_min"] = _safe_row_stat(target_mean_mat, "min")
        frames["leaf_target_mean_max"] = _safe_row_stat(target_mean_mat, "max")
        frames["leaf_target_std_mean"] = _safe_row_stat(target_std_mat, "mean")
        frames["leaf_target_iqr_mean"] = _safe_row_stat(target_iqr_mat, "mean")
        frames["leaf_target_range_mean"] = _safe_row_stat(target_range_mat, "mean")
        frames["leaf_target_abs_mean"] = _safe_row_stat(target_abs_mean_mat, "mean")
        if target_positive_fracs:
            target_positive_mat = np.vstack(target_positive_fracs).T.astype(np.float32)
            frames["leaf_target_positive_fraction"] = _safe_row_stat(target_positive_mat, "mean")
        else:
            frames["leaf_target_positive_fraction"] = np.zeros(n, dtype=np.float32)
        frames["leaf_hit_rate_avg"] = frames["leaf_target_mean_mean"].astype(np.float32, copy=False)
        frames["leaf_target_dispersion"] = frames["leaf_target_mean_std"].astype(np.float32, copy=False)
        if pred_arr is not None:
            frames["support_gap"] = (pred_arr - frames["leaf_target_mean_mean"]).astype(np.float32)
        else:
            frames["support_gap"] = np.zeros(n, dtype=np.float32)
        if target_pred_means:
            target_pred_mat = np.vstack(target_pred_means).T.astype(np.float32)
            frames["leaf_pred_mean_mean"] = _safe_row_stat(target_pred_mat, "mean")
        else:
            frames["leaf_pred_mean_mean"] = np.zeros(n, dtype=np.float32)
        if target_error_means:
            target_error_mat = np.vstack(target_error_means).T.astype(np.float32)
            frames["leaf_error_mean_mean"] = _safe_row_stat(target_error_mat, "mean")
        else:
            frames["leaf_error_mean_mean"] = np.zeros(n, dtype=np.float32)
    else:
        for name in LGBM_META_LEAF_TARGET_FEATURE_NAMES:
            frames[name] = np.zeros(n, dtype=np.float32)
    if centroid_dists:
        centroid_radius_mat = np.vstack(centroid_radii).T.astype(np.float32)
        centroid_dist_mat = np.vstack(centroid_dists).T.astype(np.float32)
        centroid_norm_dist_mat = np.vstack(centroid_norm_dists).T.astype(np.float32)
        frames["leaf_centroid_radius_mean"] = _safe_row_stat(centroid_radius_mat, "mean")
        frames["leaf_centroid_dist_mean"] = _safe_row_stat(centroid_dist_mat, "mean")
        frames["leaf_centroid_dist_median"] = _safe_row_stat(centroid_dist_mat, "median")
        frames["leaf_centroid_dist_std"] = _safe_row_stat(centroid_dist_mat, "std")
        frames["leaf_centroid_dist_p90"] = _safe_row_stat(centroid_dist_mat, "percentile", q=90.0)
        frames["leaf_centroid_dist_max"] = _safe_row_stat(centroid_dist_mat, "max")
        mean_dist = np.maximum(frames["leaf_centroid_dist_mean"], 1e-6)
        frames["leaf_centroid_dist_cv"] = np.clip(
            frames["leaf_centroid_dist_std"] / mean_dist,
            0.0,
            100.0,
        ).astype(np.float32)
        rel = centroid_dist_mat / mean_dist[:, None]
        frames["leaf_centroid_dist_rel_mean"] = _safe_row_stat(rel, "mean")
        frames["leaf_centroid_dist_rel_std"] = _safe_row_stat(rel, "std")
        frames["leaf_centroid_dist_norm_mean"] = _safe_row_stat(centroid_norm_dist_mat, "mean")
        frames["leaf_centroid_dist_norm_p90"] = _safe_row_stat(centroid_norm_dist_mat, "percentile", q=90.0)
        frames["leaf_centroid_dist_norm_max"] = _safe_row_stat(centroid_norm_dist_mat, "max")
    else:
        for name in LGBM_META_LEAF_CENTROID_FEATURE_NAMES:
            frames[name] = np.zeros(n, dtype=np.float32)


def _predict_contrib_matrix(model: Any, X: pd.DataFrame, n_features: int) -> np.ndarray | None:
    n = int(len(X))
    bs = max(1, int(LGBM_CONTRIB_PRED_BATCH_ROWS))
    if n > bs:
        parts: list[np.ndarray] = []
        t0 = time.perf_counter()
        batch_count = int(np.ceil(n / bs))
        tprint(
            "LGBM contribution prediction started: "
            f"rows={n}, features={n_features}, batches={batch_count}, batch_rows={bs}."
        )
        for batch_i, start in enumerate(range(0, n, bs), start=1):
            stop = min(n, start + bs)
            part = _predict_contrib_matrix(model, X.iloc[start:stop], n_features)
            if part is None:
                return None
            parts.append(part)
            if batch_i == 1 or batch_i == batch_count or batch_i % 5 == 0:
                tprint(
                    "LGBM contribution prediction progress: "
                    f"batch={batch_i}/{batch_count}, rows={stop}/{n}, "
                    f"elapsed={time.perf_counter() - t0:.1f}s."
                )
        try:
            return np.vstack(parts).astype(np.float32, copy=False)
        finally:
            parts.clear()
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


def _mean_contrib_matrix(models: list[Any], X: pd.DataFrame) -> np.ndarray | None:
    mats = []
    for model_i, model in enumerate(models, start=1):
        t0 = time.perf_counter()
        if len(X) > int(LGBM_CONTRIB_PRED_BATCH_ROWS):
            tprint(
                "LGBM contribution matrix model started: "
                f"model={model_i}/{len(models)}, rows={len(X)}, features={X.shape[1]}."
            )
        mat = _predict_contrib_matrix(model, X, X.shape[1])
        if mat is not None and mat.size:
            mats.append(np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0))
        if len(X) > int(LGBM_CONTRIB_PRED_BATCH_ROWS):
            tprint(
                "LGBM contribution matrix model complete: "
                f"model={model_i}/{len(models)}, elapsed={time.perf_counter() - t0:.1f}s."
            )
    if not mats:
        return None
    return np.mean(np.stack(mats, axis=0), axis=0).astype(np.float32)


def _append_path_contrib_diagnostics(frames: dict[str, np.ndarray], models: list[Any], X: pd.DataFrame) -> bool:
    n = int(len(X))
    if n == 0 or not models:
        return False
    abs_sum = np.zeros(n, dtype=np.float32)
    l2_sq = np.zeros(n, dtype=np.float32)
    entropy_weighted = np.zeros(n, dtype=np.float32)
    top1 = np.zeros(n, dtype=np.float32)
    top3 = np.zeros(n, dtype=np.float32)
    positive = np.zeros(n, dtype=np.float32)
    negative = np.zeros(n, dtype=np.float32)
    material = np.zeros(n, dtype=np.float32)
    trees_seen = 0
    model_scale = 1.0 / max(float(len(models)), 1.0)
    for model in models:
        meta = _path_contrib_metadata(model)
        if not meta:
            continue
        try:
            predict_kwargs: dict[str, Any] = {}
            if LGBM_META_LEAF_MAX_TREES > 0:
                predict_kwargs["num_iteration"] = int(LGBM_META_LEAF_MAX_TREES)
            leaves = np.asarray(model.predict(X, pred_leaf=True, **predict_kwargs), dtype=np.int32)
        except TypeError:
            try:
                leaves = np.asarray(model.predict(X, pred_leaf=True), dtype=np.int32)
            except Exception:
                continue
        except Exception:
            continue
        if leaves.ndim == 1:
            leaves = leaves.reshape(n, 1)
        tree_n = min(leaves.shape[1], len(meta))
        if LGBM_META_LEAF_MAX_TREES > 0:
            tree_n = min(tree_n, int(LGBM_META_LEAF_MAX_TREES))
        for tree_i in range(tree_n):
            tree_meta = meta[tree_i]
            if not tree_meta:
                continue
            leaf_ids = np.fromiter(tree_meta.keys(), dtype=np.int32, count=len(tree_meta))
            vals = np.asarray(
                [tree_meta[int(leaf_id)] for leaf_id in leaf_ids],
                dtype=np.float32,
            )
            if vals.ndim != 2 or vals.shape[1] < 8:
                continue
            order = np.argsort(leaf_ids, kind="mergesort")
            leaf_ids_sorted = leaf_ids[order]
            vals_sorted = vals[order]
            leaf_col = np.asarray(leaves[:, tree_i], dtype=np.int32)
            pos = np.searchsorted(leaf_ids_sorted, leaf_col)
            pos_clip = np.clip(pos, 0, max(leaf_ids_sorted.size - 1, 0))
            valid = (pos < leaf_ids_sorted.size) & (leaf_ids_sorted[pos_clip] == leaf_col)
            if not np.any(valid):
                continue
            matched = vals_sorted[pos[valid]]
            vv = np.flatnonzero(valid)
            abs_component = np.maximum(matched[:, 0], 0.0) * model_scale
            abs_sum[vv] += abs_component
            l2_sq[vv] += np.square(np.maximum(matched[:, 1], 0.0) * model_scale, dtype=np.float32)
            entropy_weighted[vv] += matched[:, 2] * abs_component
            top1[vv] = np.maximum(top1[vv], np.maximum(matched[:, 3], 0.0) * model_scale)
            top3[vv] += np.maximum(matched[:, 4], 0.0) * model_scale
            positive[vv] += np.maximum(matched[:, 5], 0.0) * model_scale
            negative[vv] += np.minimum(matched[:, 6], 0.0) * model_scale
            material[vv] += np.maximum(matched[:, 7], 0.0)
            trees_seen += 1
    if trees_seen <= 0:
        return False
    total_abs = np.maximum(abs_sum, 1e-12)
    frames["contrib_abs_sum"] = abs_sum.astype(np.float32)
    frames["contrib_l2_norm"] = np.sqrt(np.maximum(l2_sq, 0.0)).astype(np.float32)
    frames["contrib_entropy"] = np.clip(entropy_weighted / total_abs, 0.0, 1.0).astype(np.float32)
    frames["top_1_contrib_abs"] = top1.astype(np.float32)
    frames["top_3_contrib_abs_sum"] = top3.astype(np.float32)
    frames["positive_contrib_sum"] = positive.astype(np.float32)
    frames["negative_contrib_sum"] = negative.astype(np.float32)
    frames["contrib_top1_abs_share"] = np.clip(top1 / total_abs, 0.0, 1.0).astype(np.float32)
    frames["contrib_top3_abs_share"] = np.clip(top3 / total_abs, 0.0, 1.0).astype(np.float32)
    frames["contrib_balance"] = np.clip((positive + negative) / total_abs, -1.0, 1.0).astype(np.float32)
    frames["num_material_contrib_features"] = (material / max(float(trees_seen), 1.0)).astype(np.float32)
    return True


def _append_contrib_diagnostics(
    frames: dict[str, np.ndarray],
    models: list[Any],
    X: pd.DataFrame,
    *,
    method: str = LGBM_META_CONTRIB_METHOD,
) -> None:
    n = len(X)
    method_s = str(method or LGBM_META_CONTRIB_METHOD).strip().lower()
    if method_s in {"path", "saabas"} and _append_path_contrib_diagnostics(frames, models, X):
        return
    contrib = _mean_contrib_matrix(models, X)
    if contrib is None:
        for name in (
            "contrib_top1_abs_share",
            "contrib_top3_abs_share",
            "contrib_entropy",
            "contrib_balance",
            "num_material_contrib_features",
            *CONTRIB_SUMMARY_FEATURE_NAMES,
        ):
            frames[name] = np.zeros(n, dtype=np.float32)
        return
    summary = contrib_summary_frame(contrib)
    for name in CONTRIB_SUMMARY_FEATURE_NAMES:
        frames[name] = summary[name].to_numpy(dtype=np.float32, copy=False)
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


def _append_score_path_tree_diagnostics(
    frames: dict[str, np.ndarray],
    models: list[Any],
    X: pd.DataFrame,
    *,
    mode: str,
    final_pred: np.ndarray | None = None,
) -> bool:
    n = int(len(X))
    margins = _leaf_value_matrix(
        models,
        X,
        max_trees=int(LGBM_META_SCORE_PATH_MAX_TREES or LGBM_META_LEAF_MAX_TREES),
    )
    if margins is None or margins.size == 0:
        return False
    raw_final = None
    if final_pred is not None:
        pred = np.asarray(final_pred, dtype=np.float32).reshape(-1)
        if len(pred) == n:
            if mode == "classifier":
                pred = np.clip(pred, 1e-6, 1.0 - 1e-6)
                raw_final = np.log(pred / (1.0 - pred)).astype(np.float32)
            else:
                raw_final = pred.astype(np.float32)
    if raw_final is None:
        per_model = [_predict_lgbm_raw_score(model, X, mode) for model in models]
        raw_final = np.mean(np.vstack(per_model).astype(np.float32), axis=0).astype(np.float32)
    path = np.cumsum(margins, axis=1, dtype=np.float32)
    if path.shape[1] == 0:
        return False
    bias = (raw_final - path[:, -1]).astype(np.float32)
    path = (path + bias[:, None]).astype(np.float32, copy=False)
    tree_n = int(path.shape[1])

    def _path_col(frac: float) -> np.ndarray:
        idx = int(np.ceil(float(frac) * tree_n)) - 1
        idx = int(np.clip(idx, 0, tree_n - 1))
        return path[:, idx].astype(np.float32, copy=False)

    score10 = _path_col(0.10)
    score25 = _path_col(0.25)
    score50 = _path_col(0.50)
    score75 = _path_col(0.75)
    score100 = raw_final.astype(np.float32, copy=False)
    frames["score_final"] = score100
    frames["score_early_10pct"] = score10
    frames["score_early_25pct"] = score25
    frames["score_early_50pct"] = score50
    frames["score_100_minus_50"] = (score100 - score50).astype(np.float32)
    frames["score_100_minus_75"] = (score100 - score75).astype(np.float32)
    path_std = np.std(path, axis=1).astype(np.float32)
    frames["score_path_std"] = path_std
    frames["score_path_volatility"] = path_std
    frames["score_path_min"] = np.min(path, axis=1).astype(np.float32)
    frames["score_path_max"] = np.max(path, axis=1).astype(np.float32)
    frames["score_path_drawdown"] = (frames["score_path_max"] - score100).astype(np.float32)
    signs = np.sign(margins)
    frames["score_reversal_count"] = np.sum(
        (signs[:, 1:] * signs[:, :-1]) < 0.0,
        axis=1,
    ).astype(np.float32)
    frames["positive_tree_frac"] = np.mean(margins > 0.0, axis=1).astype(np.float32)
    frames["negative_tree_frac"] = np.mean(margins < 0.0, axis=1).astype(np.float32)
    frames["mean_tree_contribution"] = np.mean(margins, axis=1).astype(np.float32)
    abs_margins = np.abs(margins)
    max_tree = np.max(abs_margins, axis=1).astype(np.float32)
    frames["max_tree_contribution"] = max_tree
    frames["top_tree_contribution_share"] = (
        max_tree / np.maximum(np.sum(abs_margins, axis=1), 1e-12)
    ).astype(np.float32)
    rank50 = _safe_rank_pct(score50)
    rank75 = _safe_rank_pct(score75)
    rank100 = _safe_rank_pct(score100)
    frames["rank_100_minus_50"] = (rank100 - rank50).astype(np.float32)
    frames["rank_path_std"] = np.std(
        np.vstack((rank50, rank75, rank100)).astype(np.float32, copy=False),
        axis=0,
    ).astype(np.float32)
    return True


def _fit_lgbm_archetype_states(
    models: list[Any],
    X_train: pd.DataFrame,
    selected_features: list[str],
    *,
    timestamps: Any = None,
    assets: Any = None,
    random_state: int = 42,
) -> tuple[ContribArchetypeState | None, RawStateArchetypeState | None]:
    if not LGBM_ARCHETYPE_FEATURES or not (
        LGBM_FINAL_OOF_CONTRIB_CONTEXT_FEATURES or LGBM_FINAL_OOF_RAW_STATE_CONTEXT_FEATURES
    ):
        return None, None
    n_train = int(len(X_train))
    X_ref = X_train
    ref_timestamps = timestamps
    ref_assets = assets
    cap = int(LGBM_ARCHETYPE_FIT_MAX_ROWS)
    if cap > 0 and n_train > cap:
        idx = np.linspace(0, n_train - 1, cap, dtype=np.int32)
        X_ref = X_train.iloc[idx].reset_index(drop=True)
        ref_timestamps = _take_aligned(timestamps, idx, n_train)
        ref_assets = _take_aligned(assets, idx, n_train)
        tprint(
            "LGBM archetype reference fit sampled: "
            f"rows={n_train}->{len(X_ref)}, features={len(selected_features)}, "
            f"max_rows={cap}."
        )
    else:
        tprint(
            "LGBM archetype reference fit started: "
            f"rows={n_train}, features={len(selected_features)}."
        )
    contrib_state: ContribArchetypeState | None = None
    if LGBM_FINAL_OOF_CONTRIB_CONTEXT_FEATURES:
        t0 = time.perf_counter()
        contrib = _mean_contrib_matrix(models, X_ref)
        if contrib is not None and contrib.size:
            contrib_state = fit_contrib_archetype_state(
                contrib,
                selected_features,
                random_state=random_state,
            )
        tprint(
            "LGBM contribution archetype state fit complete: "
            f"rows={len(X_ref)}, elapsed={time.perf_counter() - t0:.1f}s, "
            f"components={getattr(contrib_state, 'component_count', 0) if contrib_state is not None else 0}."
        )
    else:
        tprint("LGBM contribution archetype state fit skipped.")
    raw_state: RawStateArchetypeState | None = None
    if LGBM_FINAL_OOF_RAW_STATE_CONTEXT_FEATURES:
        t1 = time.perf_counter()
        raw_state = fit_raw_state_archetype_state(
            X_ref,
            selected_features,
            timestamps=ref_timestamps,
            assets=ref_assets,
            random_state=random_state,
        )
        tprint(
            "LGBM raw-state archetype state fit complete: "
            f"rows={len(X_ref)}, elapsed={time.perf_counter() - t1:.1f}s, "
            f"components={getattr(raw_state, 'component_count', 0) if raw_state is not None else 0}."
        )
    else:
        tprint("LGBM raw-state archetype state fit skipped.")
    return contrib_state, raw_state


def _append_archetype_meta_features(
    features: pd.DataFrame,
    models: list[Any],
    X: pd.DataFrame,
    *,
    contrib_state: ContribArchetypeState | None = None,
    raw_state: RawStateArchetypeState | None = None,
    timestamps: Any = None,
    assets: Any = None,
    contrib_enabled: bool | None = None,
    raw_state_enabled: bool | None = None,
) -> None:
    contrib_active = (
        bool(LGBM_FINAL_OOF_CONTRIB_CONTEXT_FEATURES)
        if contrib_enabled is None
        else bool(contrib_enabled)
    )
    raw_state_active = (
        bool(LGBM_FINAL_OOF_RAW_STATE_CONTEXT_FEATURES)
        if raw_state_enabled is None
        else bool(raw_state_enabled)
    )
    if not LGBM_ARCHETYPE_FEATURES or not (contrib_active or raw_state_active):
        return
    n = len(X)
    if contrib_active and contrib_state is not None:
        contrib = _mean_contrib_matrix(models, X)
        if contrib is not None and contrib.size:
            contrib_features = transform_contrib_archetype_features(
                contrib,
                contrib_state,
                index=features.index,
            )
        else:
            contrib_features = pd.DataFrame(
                {
                    name: np.zeros(n, dtype=np.float32)
                    for name in CONTRIB_ARCHETYPE_FEATURE_NAMES
                },
                index=features.index,
            )
        for name in CONTRIB_ARCHETYPE_FEATURE_NAMES:
            features[name] = contrib_features.get(name, 0.0)
    if raw_state_active and raw_state is not None:
        raw_features = transform_raw_state_archetype_features(
            X,
            raw_state,
            timestamps=timestamps,
            assets=assets,
            index=features.index,
        )
        for name in RAW_STATE_SVD_FEATURE_NAMES + RAW_STATE_DIAGNOSTIC_FEATURE_NAMES:
            features[name] = raw_features.get(name, 0.0)


BASE_ERROR_ARCHETYPE_SIGNATURE_TOKENS = (
    "unc",
    "uncertainty",
    "pred_std",
    "pred_cv",
    "prob_std",
    "raw_score_std",
    "vote_",
    "leaf_",
    "rare_leaf",
    "low_support",
    "surprisal",
    "centroid",
    "contrib_",
    "archetype_",
    "raw_state_",
    "state_log_likelihood",
    "mahalanobis",
    "knn",
    "reconstruction",
    "transition",
    "feature_drift",
    "drift",
    "psi",
    "ks",
    "cov_shift",
    "frobenius",
    "regime_centroid",
    "rank_bin",
    "score_margin",
    "rank_margin",
    "score_path",
    "rank_path",
    "entropy",
    "variance_proxy",
)


def _base_error_bad_label(
    y_metric: Any,
    pred: Any,
    *,
    classifier: bool,
) -> np.ndarray:
    y = np.asarray(y_metric, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    n = int(min(len(y), len(p)))
    out = np.zeros(n, dtype=np.int8)
    if n == 0:
        return out
    y = y[:n]
    p = p[:n]
    finite = np.isfinite(y) & np.isfinite(p)
    if int(np.sum(finite)) < 3:
        return out
    if classifier:
        y01 = (y >= 0.5).astype(np.float64)
        p_clip = np.clip(p, 0.0, 1.0)
        residual = np.abs(y01 - p_clip)
        wrong = (p_clip >= 0.5) != (y01 >= 0.5)
        try:
            threshold = float(np.nanquantile(residual[finite], 0.70))
        except Exception:
            threshold = float(np.nanmedian(residual[finite]))
        out[finite] = (wrong[finite] | (residual[finite] >= threshold)).astype(np.int8)
    else:
        residual = np.abs(y - p)
        try:
            threshold = float(np.nanquantile(residual[finite], 0.70))
        except Exception:
            threshold = float(np.nanmedian(residual[finite]))
        out[finite] = (residual[finite] >= threshold).astype(np.int8)
    if len(np.unique(out[finite])) < 2:
        residual = np.abs((y[:n] >= 0.5).astype(np.float64) - np.clip(p[:n], 0.0, 1.0)) if classifier else np.abs(y[:n] - p[:n])
        threshold = float(np.nanmedian(residual[finite]))
        out[finite] = (residual[finite] >= threshold).astype(np.int8)
    return out


def _base_error_signature_feature_names(frame: pd.DataFrame, *, max_features: int = 128) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for col in frame.columns:
        name = str(col)
        low = name.lower()
        if "base_error_" in low:
            continue
        if not any(token in low for token in BASE_ERROR_ARCHETYPE_SIGNATURE_TOKENS):
            continue
        if name in seen:
            continue
        try:
            if not pd.api.types.is_numeric_dtype(frame[col]):
                continue
        except Exception:
            continue
        names.append(name)
        seen.add(name)
        if len(names) >= int(max_features):
            break
    return names


def _fit_base_error_archetype_state_from_meta(
    meta_frame: pd.DataFrame,
    y_metric: Any,
    pred: Any,
    *,
    classifier: bool,
    random_state: int,
) -> ResidualErrorArchetypeState | None:
    if (
        not LGBM_ARCHETYPE_FEATURES
        or not LGBM_FINAL_OOF_BASE_ERROR_CONTEXT_FEATURES
        or meta_frame is None
        or meta_frame.empty
    ):
        return None
    signature_features = _base_error_signature_feature_names(meta_frame)
    if not signature_features:
        return ResidualErrorArchetypeState(
            feature_names=[],
            enabled=False,
            reason="no_base_error_signature_features",
        )
    y_bad = _base_error_bad_label(y_metric, pred, classifier=classifier)
    return fit_residual_error_archetype_state(
        meta_frame.reindex(columns=signature_features, fill_value=0.0),
        y_bad,
        feature_names=signature_features,
        random_state=random_state,
    )


def _append_base_error_archetype_features(
    features: pd.DataFrame,
    state: ResidualErrorArchetypeState | None,
) -> None:
    arch = transform_residual_error_archetype_features(
        features,
        state,
        index=features.index,
    )
    for name in BASE_ERROR_ARCHETYPE_FEATURE_NAMES:
        features[name] = arch.get(name, 0.0)


def _raw_contrib_input_columns(features: list[str]) -> list[str]:
    return [str(c) for c in features if is_raw_contrib_feature_name(str(c))]


def _raw_contrib_passthrough_columns(features: list[str]) -> list[str]:
    return [str(c) for c in features if not is_raw_contrib_feature_name(str(c))]


def _fit_raw_contrib_input_state(
    X: pd.DataFrame,
    raw_contrib_features: list[str],
    *,
    random_state: int,
) -> ContribArchetypeState | None:
    raw_cols = [str(c) for c in raw_contrib_features if str(c) in X.columns]
    if not raw_cols:
        return None
    matrix = (
        X.reindex(columns=raw_cols, fill_value=0.0)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32, copy=False)
    )
    return fit_contrib_archetype_state(
        matrix,
        raw_cols,
        random_state=int(random_state),
    )


def _transform_raw_contrib_input_features(
    X: pd.DataFrame,
    raw_contrib_features: list[str],
    state: ContribArchetypeState | None,
    *,
    output_feature_names: list[str] | tuple[str, ...] = META_RAW_CONTRIB_SVD_FEATURE_NAMES,
    index: Any = None,
) -> pd.DataFrame:
    raw_cols = [str(c) for c in raw_contrib_features]
    n = int(len(X))
    output_names = [str(c) for c in output_feature_names]
    if not raw_cols:
        return pd.DataFrame(index=index if index is not None else X.index)
    matrix = (
        X.reindex(columns=raw_cols, fill_value=0.0)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32, copy=False)
    )
    frame = transform_contrib_archetype_features(matrix, state, index=index)
    if not output_names:
        return pd.DataFrame(index=index if index is not None else X.index)
    rename_map = {
        src: dst
        for src, dst in zip(CONTRIB_ARCHETYPE_FEATURE_NAMES, output_names)
    }
    frame = frame.rename(columns=rename_map)
    return frame.reindex(
        columns=output_names,
        fill_value=0.0,
    ).astype(np.float32, copy=False)


def _raw_contrib_model_input_frame(
    X: pd.DataFrame,
    passthrough_features: list[str],
    raw_contrib_features: list[str],
    state: ContribArchetypeState | None,
    *,
    output_feature_names: list[str] | tuple[str, ...] = META_RAW_CONTRIB_SVD_FEATURE_NAMES,
    index: Any = None,
) -> pd.DataFrame:
    idx = index if index is not None else X.index
    passthrough = [str(c) for c in passthrough_features]
    left = (
        X.reindex(columns=passthrough, fill_value=0.0)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32, copy=False)
        if passthrough
        else pd.DataFrame(index=idx)
    )
    left.index = idx
    contrib = _transform_raw_contrib_input_features(
        X,
        raw_contrib_features,
        state,
        output_feature_names=output_feature_names,
        index=idx,
    )
    return pd.concat([left, contrib], axis=1).astype(np.float32, copy=False)


def _validate_finite_contract_frame(frame: pd.DataFrame) -> None:
    values = frame.to_numpy(dtype=np.float32, copy=False)
    finite_mask = np.isfinite(values)
    if finite_mask.all():
        return
    bad_cols = [
        str(col)
        for col in frame.columns
        if not np.isfinite(frame[col].to_numpy(dtype=np.float32, copy=False)).all()
    ]
    bad_rows = int((~finite_mask.all(axis=1)).sum())
    raise ValueError(
        "LGBM inference feature contract violation: "
        f"{bad_rows}/{len(frame)} rows contain non-finite contracted features. "
        f"Examples: {bad_cols[:20]}"
    )


def _append_raw_contrib_export_features(
    features: pd.DataFrame,
    models: list[Any],
    X: pd.DataFrame,
    mapping: dict[str, str],
) -> None:
    if not LGBM_RAW_CONTRIB_OOF_EXPORT or not mapping:
        return
    contrib = _mean_contrib_matrix(models, X)
    if contrib is None or not contrib.size:
        for name in mapping.keys():
            features[str(name)] = np.zeros(len(X), dtype=np.float32)
        return
    raw = raw_contrib_frame(contrib, list(mapping.values()), index=features.index)
    rename = {
        generated: exported
        for exported, generated in zip(mapping.keys(), raw.columns)
    }
    raw = raw.rename(columns=rename)
    for name in mapping.keys():
        features[str(name)] = raw.get(str(name), 0.0)


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
        values = np.zeros(n_bins, dtype=np.float32)
        for b in range(n_bins):
            values[b] = float(mapping.get(float(b), mapping.get(int(b), 0.0)))
        frames[name] = values[bins].astype(np.float32, copy=False)


def _lgbm_meta_features_from_predictions(
    pred: np.ndarray,
    *,
    mode: str,
    rank_pct: np.ndarray | None = None,
    rank_bin_stats: pd.DataFrame | None = None,
    model_count: int = 1,
    tree_count: int | float = 0,
    score_path_probs: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> pd.DataFrame:
    prob = np.nan_to_num(np.asarray(pred, dtype=np.float32).reshape(-1), nan=0.5, posinf=1.0, neginf=0.0)
    if mode == "classifier":
        prob = np.clip(prob, 1e-6, 1.0 - 1e-6).astype(np.float32, copy=False)
        raw_score = np.log(prob / (1.0 - prob)).astype(np.float32)
    else:
        raw_score = prob.astype(np.float32, copy=False)
    n = int(len(prob))
    rank = (
        np.nan_to_num(np.asarray(rank_pct, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
        if rank_pct is not None and len(rank_pct) == n
        else _safe_rank_pct(prob)
    )
    frames: dict[str, np.ndarray] = {}
    zero = np.zeros(n, dtype=np.float32)
    tree_count_f = float(tree_count or 0.0)
    frames["lgbm_prob"] = prob
    frames["lgbm_raw_score"] = raw_score
    frames["abs_raw_score"] = np.abs(raw_score).astype(np.float32)
    frames["model_count"] = np.full(n, float(max(1, int(model_count))), dtype=np.float32)
    frames["tree_count_mean"] = np.full(n, tree_count_f, dtype=np.float32)
    frames["tree_count_min"] = np.full(n, tree_count_f, dtype=np.float32)
    frames["tree_count_max"] = np.full(n, tree_count_f, dtype=np.float32)
    frames["prob_mean"] = prob
    frames["prob_std"] = zero
    frames["prob_min"] = prob
    frames["prob_max"] = prob
    frames["prob_range"] = zero
    frames["raw_score_mean"] = raw_score
    frames["raw_score_std"] = zero
    frames["raw_score_min"] = raw_score
    frames["raw_score_max"] = raw_score
    frames["raw_score_range"] = zero
    frames["margin_from_neutral"] = (2.0 * np.abs(prob - 0.5)).astype(np.float32)
    frames["entropy"] = _binary_entropy(prob) if mode == "classifier" else zero
    frames["variance_proxy"] = zero
    frames["prob_uncertainty"] = np.clip(
        frames["entropy"] / np.log(2.0),
        0.0,
        2.0,
    ).astype(np.float32)
    frames["rank_pct"] = rank.astype(np.float32, copy=False)
    for frac in (0.10, 0.20, 0.30):
        pct = int(round(frac * 100))
        frames[f"score_margin_top{pct}"] = _score_margin(prob, frac)
    frames["rank_margin_top10"] = (rank - 0.90).astype(np.float32)
    frames["rank_margin_top20"] = (rank - 0.80).astype(np.float32)
    frames["rank_margin_top30"] = (rank - 0.70).astype(np.float32)
    _append_rank_bin_oof_features(frames, rank, rank_bin_stats)
    if score_path_probs is not None and set(LGBM_META_SCORE_PATH_FEATURE_NAMES).intersection(LGBM_META_FEATURE_NAMES):
        score50, score75, score100 = (
            np.nan_to_num(np.asarray(values, dtype=np.float32).reshape(-1), nan=0.5, posinf=1.0, neginf=0.0)
            for values in score_path_probs
        )
        if len(score50) == n and len(score75) == n and len(score100) == n:
            frames["score_100_minus_50"] = (score100 - score50).astype(np.float32)
            frames["score_100_minus_75"] = (score100 - score75).astype(np.float32)
            frames["score_path_std"] = np.std(
                np.vstack((score50, score75, score100)).astype(np.float32, copy=False),
                axis=0,
            ).astype(np.float32)
            rank50 = _safe_rank_pct(score50)
            rank100 = _safe_rank_pct(score100)
            frames["rank_100_minus_50"] = (rank100 - rank50).astype(np.float32)
            frames["rank_path_std"] = np.std(
                np.vstack((rank50, _safe_rank_pct(score75), rank100)).astype(np.float32, copy=False),
                axis=0,
            ).astype(np.float32)
    return pd.DataFrame(
        {
            name: np.nan_to_num(
                frames.get(name, np.zeros(n, dtype=np.float32)),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32, copy=False)
            for name in LGBM_META_FEATURE_NAMES
        }
    )


def _lgbm_meta_features_from_models(
    models: list[Any],
    X: pd.DataFrame,
    *,
    mode: str,
    rank_bin_stats: pd.DataFrame | None = None,
    leaf_diagnostics: bool = (
        LGBM_META_LEAF_LITE_DIAGNOSTICS
        or LGBM_META_LEAF_DIAGNOSTICS
        or LGBM_META_LEAF_SUPPORT_DIAGNOSTICS
        or LGBM_META_LEAF_TARGET_DIAGNOSTICS
        or LGBM_META_LEAF_CENTROID_DIAGNOSTICS
    ),
    leaf_full_diagnostics: bool = LGBM_META_LEAF_DIAGNOSTICS,
    leaf_support_diagnostics: bool = LGBM_META_LEAF_SUPPORT_DIAGNOSTICS,
    leaf_target_diagnostics: bool = LGBM_META_LEAF_TARGET_DIAGNOSTICS,
    leaf_centroid_diagnostics: bool = LGBM_META_LEAF_CENTROID_DIAGNOSTICS,
    contrib_diagnostics: bool = LGBM_META_CONTRIB_DIAGNOSTICS,
    contrib_method: str = LGBM_META_CONTRIB_METHOD,
    score_path_diagnostics: bool = LGBM_META_SCORE_PATH_DIAGNOSTICS,
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
    model_count = int(prob_mat.shape[0]) if getattr(prob_mat, "ndim", 1) == 2 else 1
    tree_counts = np.asarray([max(_model_num_iterations(model), 0) for model in models], dtype=np.float32)
    if tree_counts.size == 0:
        tree_counts = np.zeros(1, dtype=np.float32)
    prob_std = np.std(prob_mat, axis=0).astype(np.float32) if prob_mat.shape[0] > 1 else np.zeros(n, dtype=np.float32)
    prob_min = np.min(prob_mat, axis=0).astype(np.float32)
    prob_max = np.max(prob_mat, axis=0).astype(np.float32)
    raw_std = np.std(raw_mat, axis=0).astype(np.float32) if raw_mat.shape[0] > 1 else np.zeros(n, dtype=np.float32)
    raw_min = np.min(raw_mat, axis=0).astype(np.float32)
    raw_max = np.max(raw_mat, axis=0).astype(np.float32)
    rank_pct = _safe_rank_pct(prob)
    frames["lgbm_prob"] = prob
    frames["lgbm_raw_score"] = raw_score
    frames["abs_raw_score"] = np.abs(raw_score).astype(np.float32)
    frames["model_count"] = np.full(n, float(model_count), dtype=np.float32)
    frames["tree_count_mean"] = np.full(n, float(np.nanmean(tree_counts)), dtype=np.float32)
    frames["tree_count_min"] = np.full(n, float(np.nanmin(tree_counts)), dtype=np.float32)
    frames["tree_count_max"] = np.full(n, float(np.nanmax(tree_counts)), dtype=np.float32)
    frames["prob_mean"] = prob
    frames["prob_std"] = prob_std
    frames["prob_min"] = prob_min
    frames["prob_max"] = prob_max
    frames["prob_range"] = (prob_max - prob_min).astype(np.float32)
    frames["raw_score_mean"] = raw_score
    frames["raw_score_std"] = raw_std
    frames["raw_score_min"] = raw_min
    frames["raw_score_max"] = raw_max
    frames["raw_score_range"] = (raw_max - raw_min).astype(np.float32)
    frames["margin_from_neutral"] = (2.0 * np.abs(prob - 0.5)).astype(np.float32)
    frames["entropy"] = _binary_entropy(prob) if mode == "classifier" else np.zeros(n, dtype=np.float32)
    frames["variance_proxy"] = np.var(prob_mat, axis=0).astype(np.float32) if prob_mat.shape[0] > 1 else np.zeros(n, dtype=np.float32)
    frames["prob_uncertainty"] = np.clip(
        frames["entropy"] / np.log(2.0) + prob_std,
        0.0,
        2.0,
    ).astype(np.float32)
    frames["rank_pct"] = rank_pct
    for frac in (0.10, 0.20, 0.30):
        pct = int(round(frac * 100))
        frames[f"score_margin_top{pct}"] = _score_margin(prob, frac)
    frames["rank_margin_top10"] = (rank_pct - 0.90).astype(np.float32)
    frames["rank_margin_top20"] = (rank_pct - 0.80).astype(np.float32)
    frames["rank_margin_top30"] = (rank_pct - 0.70).astype(np.float32)
    if leaf_diagnostics:
        _append_leaf_diagnostics(
            frames,
            models,
            X,
            prediction=prob,
            full_diagnostics=leaf_full_diagnostics,
            support_diagnostics=leaf_support_diagnostics,
            target_diagnostics=leaf_target_diagnostics,
            centroid_diagnostics=leaf_centroid_diagnostics,
        )
    if contrib_diagnostics:
        _append_contrib_diagnostics(frames, models, X, method=contrib_method)
    if score_path_diagnostics and set(LGBM_META_SCORE_PATH_FEATURE_NAMES).intersection(LGBM_META_FEATURE_NAMES):
        if not _append_score_path_tree_diagnostics(
            frames,
            models,
            X,
            mode=mode,
            final_pred=prob,
        ):
            path_scores = _score_path_predictions(models, X, mode, final_pred=prob)
            if path_scores is not None:
                score50, score75, score100 = path_scores
            else:
                score50 = score75 = score100 = prob.copy()
            frames["score_final"] = score100.astype(np.float32)
            frames["score_early_10pct"] = score50.astype(np.float32)
            frames["score_early_25pct"] = score50.astype(np.float32)
            frames["score_early_50pct"] = score50.astype(np.float32)
            frames["score_100_minus_50"] = (score100 - score50).astype(np.float32)
            frames["score_100_minus_75"] = (score100 - score75).astype(np.float32)
            frames["score_path_std"] = np.std(np.vstack((score50, score75, score100)).astype(np.float32, copy=False), axis=0).astype(np.float32)
            frames["score_path_volatility"] = frames["score_path_std"]
            frames["score_path_min"] = np.min(np.vstack((score50, score75, score100)), axis=0).astype(np.float32)
            frames["score_path_max"] = np.max(np.vstack((score50, score75, score100)), axis=0).astype(np.float32)
            frames["score_path_drawdown"] = (frames["score_path_max"] - score100).astype(np.float32)
            rank_paths = [_safe_rank_pct(score) for score in (score50, score75, score100)]
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
        "internal_metric_feature_names": list(LGBM_INTERNAL_METRIC_FEATURE_NAMES),
        "base_error_archetype_feature_names": list(BASE_ERROR_ARCHETYPE_FEATURE_NAMES),
        "base_error_archetype_signature_feature_names": list(
            getattr(getattr(model, "base_error_archetype_state", None), "feature_names", []) or []
        ),
        "base_error_archetype_enabled": bool(
            getattr(getattr(model, "base_error_archetype_state", None), "enabled", False)
        ),
        "rank_bin_stats_oof": model.rank_bin_stats_oof.to_dict(orient="records"),
        "selected_features": list(model.selected_features),
        "selected_features_count": int(len(model.selected_features)),
        "input_feature_names": list(getattr(model, "input_feature_names", []) or []),
        "raw_contrib_oof_feature_names": list(
            getattr(model, "raw_contrib_oof_feature_names", []) or []
        ),
        "raw_contrib_feature_mapping": dict(
            getattr(model, "raw_contrib_feature_mapping", {}) or {}
        ),
        "raw_contrib_input_features": list(
            getattr(model, "raw_contrib_input_features", []) or []
        ),
        "raw_contrib_transformed_feature_names": list(
            getattr(model, "raw_contrib_transformed_feature_names", []) or []
        ),
        "ae_gmm_input_features": list(
            getattr(model, "ae_gmm_input_features", []) or []
        ),
        "ae_gmm_feature_names": list(
            getattr(model, "ae_gmm_feature_names", []) or []
        ),
        "ae_gmm_context_feature_names": list(
            getattr(model, "ae_gmm_context_feature_names", []) or []
        ),
        "ae_gmm_state_enabled": bool(
            _ae_gmm_state_enabled(getattr(model, "ae_gmm_state", {}) or {})
        ),
        "ae_gmm_selected_config": _json_sanitize(
            (getattr(model, "ae_gmm_state", {}) or {}).get("selected_config", {})
        ),
        "model_effectiveness_history_defaults": dict(
            getattr(model, "model_effectiveness_history_defaults_", {}) or {}
        ),
        "model_effectiveness_history_default_sources": dict(
            getattr(model, "model_effectiveness_history_default_sources_", {}) or {}
        ),
        "meta_leaf_lite_diagnostics_enabled": bool(getattr(model, "meta_leaf_lite_diagnostics_enabled", False)),
        "meta_leaf_support_diagnostics_enabled": bool(getattr(model, "meta_leaf_support_diagnostics_enabled", False)),
        "meta_leaf_target_diagnostics_enabled": bool(getattr(model, "meta_leaf_target_diagnostics_enabled", False)),
        "meta_leaf_centroid_diagnostics_enabled": bool(getattr(model, "meta_leaf_centroid_diagnostics_enabled", False)),
        "meta_leaf_diagnostics_enabled": bool(getattr(model, "meta_leaf_diagnostics_enabled", False)),
        "meta_contrib_diagnostics_enabled": bool(getattr(model, "meta_contrib_diagnostics_enabled", False)),
        "meta_contrib_method": str(getattr(model, "meta_contrib_method", LGBM_META_CONTRIB_METHOD)),
        "meta_score_path_diagnostics_enabled": bool(getattr(model, "meta_score_path_diagnostics_enabled", True)),
        "meta_drift_features_enabled": bool(getattr(model, "meta_drift_features_enabled", True)),
        "meta_drift_max_rows": int(LGBM_META_DRIFT_MAX_ROWS),
        "meta_drift_max_features": int(LGBM_META_DRIFT_MAX_FEATURES),
        "meta_context_features_enabled": bool(getattr(model, "meta_context_features_enabled", False)),
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


def _resolve_lgbm_final_model_checkpoint_dir(
    reference_artifact_dir: str | os.PathLike[str] | None,
) -> Path | None:
    raw = str(LGBM_FINAL_MODEL_CHECKPOINT_DIR or "").strip()
    if raw.lower() in {"0", "false", "no", "n", "off"}:
        return None
    if raw:
        return Path(raw)
    if reference_artifact_dir:
        return Path(reference_artifact_dir) / "final_model_checkpoint"
    return None


def _save_lgbm_final_model_checkpoint(
    model: LGBMStabilityModel,
    output_dir: str | os.PathLike[str] | None,
    *,
    split_importance_sum: np.ndarray,
    gain_importance_sum: np.ndarray,
    final_ensemble_ess: float,
    pre_final_oof: np.ndarray | None,
    final_weights: np.ndarray | None,
) -> None:
    if not output_dir:
        return
    path = Path(output_dir)
    t0 = time.perf_counter()
    path.mkdir(parents=True, exist_ok=True)
    tprint(
        "LGBM final model checkpoint save started before final OOF/meta CV: "
        f"dir={path}, models={len(getattr(model, 'models', []) or [])}."
    )
    model_files: list[dict[str, Any]] = []
    for i, fitted in enumerate(getattr(model, "models", []) or [], start=1):
        booster = getattr(fitted, "booster_", None)
        if booster is not None:
            filename = f"final_model_{i}.lgb"
            booster.save_model(str(path / filename))
            model_files.append({"index": i, "file": filename, "format": "lightgbm_booster"})
            continue
        filename = f"final_model_{i}.pkl"
        with open(path / filename, "wb") as f:
            pickle.dump(fitted, f, protocol=pickle.HIGHEST_PROTOCOL)
        model_files.append({"index": i, "file": filename, "format": "pickle"})
    sidecar = {
        "checkpoint_stage": "after_final_models_before_final_oof_meta_cv",
        "mode": str(model.mode),
        "selected_features": list(map(str, model.selected_features)),
        "input_feature_names": list(map(str, getattr(model, "input_feature_names", []) or [])),
        "best_params": dict(model.best_params or {}),
        "label_weight_hpo_report": dict(getattr(model, "label_weight_hpo_report_", {}) or {}),
        "metrics": dict(model.metrics or {}),
        "pruning_history": list(model.pruning_history or []),
        "feature_stats_train": dict(model.feature_stats_train or {}),
        "drift_reference": dict(model.drift_reference or {}),
        "contrib_archetype_state": getattr(model, "contrib_archetype_state", None),
        "raw_state_archetype_state": getattr(model, "raw_state_archetype_state", None),
        "raw_contrib_input_features": list(getattr(model, "raw_contrib_input_features", []) or []),
        "raw_contrib_passthrough_features": list(getattr(model, "raw_contrib_passthrough_features", []) or []),
        "raw_contrib_transformed_feature_names": list(
            getattr(model, "raw_contrib_transformed_feature_names", []) or []
        ),
        "raw_contrib_input_state": getattr(model, "raw_contrib_input_state", None),
        "ae_gmm_input_features": list(getattr(model, "ae_gmm_input_features", []) or []),
        "ae_gmm_feature_names": list(getattr(model, "ae_gmm_feature_names", []) or []),
        "ae_gmm_context_feature_names": list(
            getattr(model, "ae_gmm_context_feature_names", []) or []
        ),
        "ae_gmm_state": dict(getattr(model, "ae_gmm_state", {}) or {}),
        "model_effectiveness_history_defaults": dict(
            getattr(model, "model_effectiveness_history_defaults_", {}) or {}
        ),
        "model_effectiveness_history_default_sources": dict(
            getattr(model, "model_effectiveness_history_default_sources_", {}) or {}
        ),
        "meta_leaf_lite_diagnostics_enabled": bool(getattr(model, "meta_leaf_lite_diagnostics_enabled", False)),
        "meta_leaf_support_diagnostics_enabled": bool(getattr(model, "meta_leaf_support_diagnostics_enabled", False)),
        "meta_leaf_target_diagnostics_enabled": bool(getattr(model, "meta_leaf_target_diagnostics_enabled", False)),
        "meta_leaf_centroid_diagnostics_enabled": bool(getattr(model, "meta_leaf_centroid_diagnostics_enabled", False)),
        "meta_leaf_diagnostics_enabled": bool(getattr(model, "meta_leaf_diagnostics_enabled", False)),
        "meta_contrib_diagnostics_enabled": bool(getattr(model, "meta_contrib_diagnostics_enabled", False)),
        "meta_contrib_method": str(getattr(model, "meta_contrib_method", LGBM_META_CONTRIB_METHOD)),
        "meta_score_path_diagnostics_enabled": bool(getattr(model, "meta_score_path_diagnostics_enabled", True)),
        "meta_drift_features_enabled": bool(getattr(model, "meta_drift_features_enabled", True)),
        "meta_drift_max_rows": int(LGBM_META_DRIFT_MAX_ROWS),
        "meta_drift_max_features": int(LGBM_META_DRIFT_MAX_FEATURES),
        "meta_context_features_enabled": bool(getattr(model, "meta_context_features_enabled", False)),
        "split_importance_sum": np.asarray(split_importance_sum, dtype=np.float64),
        "gain_importance_sum": np.asarray(gain_importance_sum, dtype=np.float64),
        "final_ensemble_sequential_weight_ess": float(final_ensemble_ess),
        "pre_final_oof": (
            np.asarray(pre_final_oof, dtype=np.float32)
            if pre_final_oof is not None
            else None
        ),
        "final_weights": (
            np.asarray(final_weights, dtype=np.float32)
            if final_weights is not None
            else None
        ),
        "model_files": list(model_files),
    }
    sidecar_path = path / "checkpoint_sidecar.pkl"
    with open(sidecar_path, "wb") as f:
        pickle.dump(sidecar, f, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = {
        "checkpoint_stage": "after_final_models_before_final_oof_meta_cv",
        "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        "model_count": int(len(model_files)),
        "model_files": model_files,
        "selected_features_count": int(len(model.selected_features)),
        "input_feature_names_count": int(len(getattr(model, "input_feature_names", []) or [])),
        "ae_gmm_input_feature_count": int(len(getattr(model, "ae_gmm_input_features", []) or [])),
        "ae_gmm_feature_count": int(len(getattr(model, "ae_gmm_feature_names", []) or [])),
        "ae_gmm_context_feature_count": int(
            len(getattr(model, "ae_gmm_context_feature_names", []) or [])
        ),
        "ae_gmm_enabled": bool(_ae_gmm_state_enabled(getattr(model, "ae_gmm_state", {}) or {})),
        "best_params_keys": sorted(map(str, (model.best_params or {}).keys())),
        "label_weight_hpo_winner": str(
            (getattr(model, "label_weight_hpo_report_", {}) or {}).get("winner", "none")
        ),
        "label_weight_hpo_selected": bool(
            (getattr(model, "label_weight_hpo_report_", {}) or {}).get("selected", False)
        ),
        "sidecar": sidecar_path.name,
        "has_pre_final_oof": bool(pre_final_oof is not None),
        "has_final_weights": bool(final_weights is not None),
        "final_ensemble_sequential_weight_ess": float(final_ensemble_ess),
        "meta_leaf_lite_diagnostics_enabled": bool(getattr(model, "meta_leaf_lite_diagnostics_enabled", False)),
        "meta_leaf_support_diagnostics_enabled": bool(getattr(model, "meta_leaf_support_diagnostics_enabled", False)),
        "meta_leaf_target_diagnostics_enabled": bool(getattr(model, "meta_leaf_target_diagnostics_enabled", False)),
        "meta_leaf_centroid_diagnostics_enabled": bool(getattr(model, "meta_leaf_centroid_diagnostics_enabled", False)),
        "meta_leaf_diagnostics_enabled": bool(getattr(model, "meta_leaf_diagnostics_enabled", False)),
        "meta_contrib_diagnostics_enabled": bool(getattr(model, "meta_contrib_diagnostics_enabled", False)),
        "meta_contrib_method": str(getattr(model, "meta_contrib_method", LGBM_META_CONTRIB_METHOD)),
        "meta_score_path_diagnostics_enabled": bool(getattr(model, "meta_score_path_diagnostics_enabled", True)),
        "meta_drift_features_enabled": bool(getattr(model, "meta_drift_features_enabled", True)),
        "meta_drift_max_rows": int(LGBM_META_DRIFT_MAX_ROWS),
        "meta_drift_max_features": int(LGBM_META_DRIFT_MAX_FEATURES),
        "meta_context_features_enabled": bool(getattr(model, "meta_context_features_enabled", False)),
    }
    manifest_path = path / "manifest.json"
    manifest_path.write_text(json.dumps(_json_sanitize(manifest), indent=2, sort_keys=True), encoding="utf-8")
    columns = {
        "selected_features": list(map(str, model.selected_features)),
        "lgbm_selected_model_features": list(map(str, model.selected_features)),
        "lgbm_selected_input_features": list(map(str, getattr(model, "input_feature_names", []) or [])),
        "raw_contrib_input_features": list(getattr(model, "raw_contrib_input_features", []) or []),
        "checkpoint_incomplete_final_oof_meta": True,
        "label_weight_hpo_winner": str(
            (getattr(model, "label_weight_hpo_report_", {}) or {}).get("winner", "none")
        ),
        "label_weight_hpo_selected": bool(
            (getattr(model, "label_weight_hpo_report_", {}) or {}).get("selected", False)
        ),
        "meta_leaf_lite_diagnostics_enabled": bool(getattr(model, "meta_leaf_lite_diagnostics_enabled", False)),
        "meta_leaf_support_diagnostics_enabled": bool(getattr(model, "meta_leaf_support_diagnostics_enabled", False)),
        "meta_leaf_target_diagnostics_enabled": bool(getattr(model, "meta_leaf_target_diagnostics_enabled", False)),
        "meta_leaf_centroid_diagnostics_enabled": bool(getattr(model, "meta_leaf_centroid_diagnostics_enabled", False)),
        "meta_leaf_diagnostics_enabled": bool(getattr(model, "meta_leaf_diagnostics_enabled", False)),
        "meta_contrib_diagnostics_enabled": bool(getattr(model, "meta_contrib_diagnostics_enabled", False)),
        "meta_contrib_method": str(getattr(model, "meta_contrib_method", LGBM_META_CONTRIB_METHOD)),
        "meta_score_path_diagnostics_enabled": bool(getattr(model, "meta_score_path_diagnostics_enabled", True)),
        "meta_drift_features_enabled": bool(getattr(model, "meta_drift_features_enabled", True)),
        "meta_drift_max_rows": int(LGBM_META_DRIFT_MAX_ROWS),
        "meta_drift_max_features": int(LGBM_META_DRIFT_MAX_FEATURES),
        "meta_context_features_enabled": bool(getattr(model, "meta_context_features_enabled", False)),
    }
    (path / "columns.json").write_text(json.dumps(_json_sanitize(columns), indent=2, sort_keys=True), encoding="utf-8")
    model.metrics["lgbm_final_model_checkpoint_dir"] = str(path)
    model.metrics["lgbm_final_model_checkpoint_stage"] = "after_final_models_before_final_oof_meta_cv"
    tprint(
        "LGBM final model checkpoint saved before final OOF/meta CV: "
        f"dir={path}, models={len(model_files)}, elapsed={time.perf_counter() - t0:.1f}s."
    )


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
        "internal_metric_feature_names": list(LGBM_INTERNAL_METRIC_FEATURE_NAMES),
        "internal_metric_feature_count": int(len(LGBM_INTERNAL_METRIC_FEATURE_NAMES)),
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


def _effective_n(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    if len(w) == 0:
        return 0.0
    denom = float(np.sum(w**2))
    if denom <= 1e-12:
        return 0.0
    return float((np.sum(w) ** 2) / denom)


def _label_weight_diagnostics(
    y_soft: np.ndarray,
    y_hard: np.ndarray,
    weights: np.ndarray,
    *,
    prefix: str,
) -> dict[str, float]:
    ys = np.clip(np.asarray(y_soft, dtype=np.float64), 0.0, 1.0)
    yh = np.asarray(y_hard, dtype=np.float64) >= 0.5
    w = np.nan_to_num(np.asarray(weights, dtype=np.float64), nan=1.0, posinf=1.0, neginf=1.0)
    if len(ys) == 0:
        return {
            f"{prefix}_hard_positive_rate": float("nan"),
            f"{prefix}_soft_mean": float("nan"),
            f"{prefix}_soft_std": float("nan"),
            f"{prefix}_weighted_positive_mass": 0.0,
            f"{prefix}_weighted_negative_mass": 0.0,
            f"{prefix}_weighted_positive_fraction": float("nan"),
            f"{prefix}_n_eff_total": 0.0,
            f"{prefix}_n_eff_positive": 0.0,
            f"{prefix}_n_eff_negative": 0.0,
            f"{prefix}_hard_positive_count": 0.0,
            f"{prefix}_hard_negative_count": 0.0,
        }
    pos_mass = float(np.sum(w * ys))
    neg_mass = float(np.sum(w * (1.0 - ys)))
    total_mass = pos_mass + neg_mass
    pos_w = w[yh]
    neg_w = w[~yh]
    return {
        f"{prefix}_hard_positive_rate": float(np.mean(yh)),
        f"{prefix}_soft_mean": float(np.mean(ys)),
        f"{prefix}_soft_std": float(np.std(ys)),
        f"{prefix}_weighted_positive_mass": pos_mass,
        f"{prefix}_weighted_negative_mass": neg_mass,
        f"{prefix}_weighted_positive_fraction": float(pos_mass / total_mass) if total_mass > 1e-12 else float("nan"),
        f"{prefix}_n_eff_total": float(_effective_n(w)),
        f"{prefix}_n_eff_positive": float(_effective_n(pos_w)),
        f"{prefix}_n_eff_negative": float(_effective_n(neg_w)),
        f"{prefix}_hard_positive_count": float(np.sum(yh)),
        f"{prefix}_hard_negative_count": float(np.sum(~yh)),
    }


def _log_label_weight_diagnostics(
    y_soft: np.ndarray,
    y_hard: np.ndarray,
    weights: np.ndarray,
    *,
    label: str,
) -> dict[str, float]:
    diag = _label_weight_diagnostics(y_soft, y_hard, weights, prefix=label)
    tprint(
        "LGBM label/weight diagnostics "
        f"{label}: hard_pos_rate={diag[f'{label}_hard_positive_rate']:.4f}, "
        f"soft_mean={diag[f'{label}_soft_mean']:.4f}, "
        f"soft_std={diag[f'{label}_soft_std']:.4f}, "
        f"weighted_pos_frac={diag[f'{label}_weighted_positive_fraction']:.4f}, "
        f"weighted_pos_mass={diag[f'{label}_weighted_positive_mass']:.1f}, "
        f"weighted_neg_mass={diag[f'{label}_weighted_negative_mass']:.1f}, "
        f"n_eff_total={diag[f'{label}_n_eff_total']:.1f}, "
        f"n_eff_pos={diag[f'{label}_n_eff_positive']:.1f}, "
        f"n_eff_neg={diag[f'{label}_n_eff_negative']:.1f}."
    )
    return diag


def _rebalance_effective_class_mass(
    y_soft: np.ndarray,
    y_hard: np.ndarray,
    weights: np.ndarray,
    *,
    label: str,
) -> tuple[np.ndarray, dict[str, float]]:
    del y_hard
    w = np.asarray(weights, dtype=np.float32)
    if not bool(LGBM_REBALANCE_EFFECTIVE_CLASSES):
        return w, {f"{label}_class_rebalance_applied": 0.0}
    ys = np.clip(np.asarray(y_soft, dtype=np.float64), 0.0, 1.0)
    if len(ys) != len(w) or len(ys) == 0:
        return w, {f"{label}_class_rebalance_applied": 0.0}
    pos_mass = float(np.sum(w.astype(np.float64) * ys))
    neg_mass = float(np.sum(w.astype(np.float64) * (1.0 - ys)))
    total_mass = pos_mass + neg_mass
    if total_mass <= 1e-12 or pos_mass <= 1e-12 or neg_mass <= 1e-12:
        return w, {f"{label}_class_rebalance_applied": 0.0}
    current_frac = float(pos_mass / total_mass)
    target_frac = float(np.clip(current_frac, LGBM_REBALANCE_POS_MASS_MIN, LGBM_REBALANCE_POS_MASS_MAX))
    if abs(target_frac - current_frac) <= 1e-6:
        return w, {
            f"{label}_class_rebalance_applied": 0.0,
            f"{label}_class_rebalance_pos_fraction_before": current_frac,
            f"{label}_class_rebalance_pos_fraction_target": target_frac,
        }
    pos_factor = float(np.clip((target_frac * total_mass) / pos_mass, 1.0 / LGBM_REBALANCE_MAX_MULTIPLIER, LGBM_REBALANCE_MAX_MULTIPLIER))
    neg_factor = float(np.clip(((1.0 - target_frac) * total_mass) / neg_mass, 1.0 / LGBM_REBALANCE_MAX_MULTIPLIER, LGBM_REBALANCE_MAX_MULTIPLIER))
    row_factor = ys * pos_factor + (1.0 - ys) * neg_factor
    out, ess = _normalize_weights(w.astype(np.float64) * row_factor)
    after = _label_weight_diagnostics(ys, ys >= 0.5, out, prefix=f"{label}_post_rebalance")
    diag = {
        f"{label}_class_rebalance_applied": 1.0,
        f"{label}_class_rebalance_pos_fraction_before": current_frac,
        f"{label}_class_rebalance_pos_fraction_target": target_frac,
        f"{label}_class_rebalance_pos_factor": pos_factor,
        f"{label}_class_rebalance_neg_factor": neg_factor,
        f"{label}_class_rebalance_n_eff_after": float(ess),
    }
    diag.update(after)
    tprint(
        "LGBM effective-class rebalance "
        f"{label}: pos_frac={current_frac:.4f}->{after[f'{label}_post_rebalance_weighted_positive_fraction']:.4f} "
        f"target={target_frac:.4f}, pos_factor={pos_factor:.3f}, neg_factor={neg_factor:.3f}, "
        f"cap={LGBM_REBALANCE_MAX_MULTIPLIER:.2f}, ess={ess:.1f}."
    )
    return out.astype(np.float32), diag


def _label_weight_hpo_array(
    context: dict[str, Any] | None,
    names: tuple[str, ...],
    n: int,
) -> np.ndarray | None:
    if not isinstance(context, dict):
        return None
    for name in names:
        if name not in context or context.get(name) is None:
            continue
        try:
            arr = np.asarray(context.get(name), dtype=np.float32).reshape(-1)
        except Exception:
            continue
        if len(arr) < n:
            continue
        out = arr[:n].astype(np.float32, copy=False)
        if np.isfinite(out).any():
            return out
    return None


def _prepare_label_weight_hpo_context(
    label_context: dict[str, Any] | None,
    n: int,
) -> tuple[dict[str, np.ndarray] | None, dict[str, Any]]:
    diag: dict[str, Any] = {
        "enabled": bool(LGBM_BASE_LABEL_WEIGHT_HPO_ENABLED),
        "required_rows": int(n),
        "approximations": [],
    }
    mfe = _label_weight_hpo_array(
        label_context,
        ("mfe", "mfe_ret", "__mfe_ret__", "mfe_return"),
        n,
    )
    mae = _label_weight_hpo_array(
        label_context,
        ("mae", "mae_ret", "__mae_ret__", "mae_return"),
        n,
    )
    atr = _label_weight_hpo_array(
        label_context,
        ("atr", "barrier_pct", "__barrier_pct__", "tp", "__tp__"),
        n,
    )
    missing = []
    if mfe is None:
        missing.append("mfe")
    if mae is None:
        missing.append("mae")
    if atr is None:
        missing.append("atr_or_barrier_pct")
    if missing:
        diag["enabled"] = False
        diag["reason"] = "missing_required_context:" + ",".join(missing)
        return None, diag
    mfe = np.nan_to_num(mfe.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    mae_abs = np.abs(np.nan_to_num(mae.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0))
    atr = np.asarray(atr, dtype=np.float32)
    finite_atr = atr[np.isfinite(atr) & (atr > 0.0)]
    atr_fill = float(np.nanmedian(finite_atr)) if finite_atr.size else 1e-3
    atr = np.nan_to_num(atr, nan=atr_fill, posinf=atr_fill, neginf=atr_fill)
    atr = np.clip(atr, max(atr_fill * 1e-3, 1e-8), None).astype(np.float32, copy=False)
    t_mfe = _label_weight_hpo_array(
        label_context,
        ("time_to_mfe", "bars_to_mfe", "__bars_to_mfe__", "t_mfe"),
        n,
    )
    t_mae = _label_weight_hpo_array(
        label_context,
        ("time_to_mae", "bars_to_mae", "__bars_to_mae__", "t_mae"),
        n,
    )
    tau_tp = _label_weight_hpo_array(
        label_context,
        ("tau_tp", "tau_TP", "__tau_tp__", "__tau_TP__", "bars_to_tp", "__bars_to_tp__"),
        n,
    )
    tau_sl = _label_weight_hpo_array(
        label_context,
        ("tau_sl", "tau_SL", "__tau_sl__", "__tau_SL__", "bars_to_sl", "__bars_to_sl__"),
        n,
    )
    mae_before_tp = _label_weight_hpo_array(
        label_context,
        ("mae_before_tp", "__mae_before_tp__"),
        n,
    )
    mfe_before_sl = _label_weight_hpo_array(
        label_context,
        ("mfe_before_sl", "__mfe_before_sl__"),
        n,
    )
    approximations = list(diag.get("approximations", []))
    if t_mfe is None:
        t_mfe = np.full(n, np.nan, dtype=np.float32)
        approximations.append("time_to_mfe_missing_time_bonus_zero")
    else:
        t_mfe = np.nan_to_num(t_mfe.astype(np.float32, copy=False), nan=np.inf, posinf=np.inf, neginf=np.inf)
    if t_mae is None:
        t_mae = np.full(n, np.nan, dtype=np.float32)
        approximations.append("time_to_mae_missing_sl_first_conservative")
    else:
        t_mae = np.nan_to_num(t_mae.astype(np.float32, copy=False), nan=np.inf, posinf=np.inf, neginf=np.inf)
    if tau_tp is None:
        tau_tp = t_mfe.copy()
        approximations.append("tau_tp_uses_time_to_mfe_proxy")
    else:
        tau_tp = np.nan_to_num(tau_tp.astype(np.float32, copy=False), nan=np.inf, posinf=np.inf, neginf=np.inf)
    if tau_sl is None:
        tau_sl = t_mae.copy()
        approximations.append("tau_sl_uses_time_to_mae_proxy")
    else:
        tau_sl = np.nan_to_num(tau_sl.astype(np.float32, copy=False), nan=np.inf, posinf=np.inf, neginf=np.inf)
    if mae_before_tp is None:
        mae_before_tp = mae_abs.copy()
        approximations.append("mae_before_tp_uses_full_mae_abs")
    else:
        mae_before_tp = np.abs(
            np.nan_to_num(mae_before_tp.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        )
    if mfe_before_sl is None:
        mfe_before_sl = np.maximum(mfe, 0.0).astype(np.float32, copy=False)
        approximations.append("mfe_before_sl_uses_full_mfe")
    else:
        mfe_before_sl = np.maximum(
            np.nan_to_num(mfe_before_sl.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0),
            0.0,
        ).astype(np.float32, copy=False)
    diag["approximations"] = approximations
    diag["atr_median"] = atr_fill
    diag["mfe_p99"] = float(np.nanpercentile(mfe[np.isfinite(mfe)], 99.0)) if np.isfinite(mfe).any() else 0.0
    diag["mae_abs_p99"] = float(np.nanpercentile(mae_abs[np.isfinite(mae_abs)], 99.0)) if np.isfinite(mae_abs).any() else 0.0
    return {
        "mfe": mfe.astype(np.float32, copy=False),
        "mae_abs": mae_abs.astype(np.float32, copy=False),
        "atr": atr,
        "t_mfe": t_mfe.astype(np.float32, copy=False),
        "t_mae": t_mae.astype(np.float32, copy=False),
        "tau_tp": tau_tp.astype(np.float32, copy=False),
        "tau_sl": tau_sl.astype(np.float32, copy=False),
        "mae_before_tp": mae_before_tp.astype(np.float32, copy=False),
        "mfe_before_sl": mfe_before_sl.astype(np.float32, copy=False),
    }, diag


def _winsorized_positive_ratio(
    numer: np.ndarray,
    denom: np.ndarray,
    *,
    train_idx: np.ndarray | None = None,
) -> np.ndarray:
    val = np.maximum(np.asarray(numer, dtype=np.float32), 0.0)
    if train_idx is not None:
        idx = np.asarray(train_idx, dtype=np.int32)
        idx = idx[(idx >= 0) & (idx < len(val))]
        stat_vals = val[idx] if len(idx) else val
    else:
        stat_vals = val
    finite_pos = stat_vals[np.isfinite(stat_vals) & (stat_vals > 0.0)]
    if finite_pos.size:
        cap = float(np.nanpercentile(finite_pos, 99.0))
        if np.isfinite(cap) and cap > 0.0:
            val = np.minimum(val, cap)
    den = np.maximum(np.asarray(denom, dtype=np.float32), 1e-8)
    return np.clip(val / den, 0.0, 1.0).astype(np.float32)


def _sample_capped_pair(
    trial: Any,
    prefix: str,
    lo: float,
    hi: float,
) -> tuple[float, float]:
    ratio = float(trial.suggest_float(f"{prefix}_ratio", 1.0, 2.0))
    base_lo = max(float(lo), 0.0)
    base_hi = max(base_lo, float(hi) / max(ratio, 1e-6))
    base = float(trial.suggest_float(f"{prefix}_base", base_lo, base_hi))
    high = min(float(hi), base * ratio)
    low = base
    side = str(trial.suggest_categorical(f"{prefix}_larger_side", ["h", "l"]))
    return (high, low) if side == "h" else (low, high)


def _sample_label_barrier_params(trial: Any) -> dict[str, float]:
    base_tp = float(trial.suggest_float("base_tp", 1.0, 3.0))
    base_sl = float(trial.suggest_float("base_sl", 1.0, base_tp))
    alpha_h, alpha_l = _sample_capped_pair(trial, "alpha", 0.1, 0.5)
    beta_h, beta_l = _sample_capped_pair(trial, "beta", 0.1, 0.4)
    delta_h, delta_l = _sample_capped_pair(trial, "delta_time", 0.0, 0.2)
    return {
        "base_tp": base_tp,
        "base_sl": base_sl,
        "atr_power": float(trial.suggest_float("atr_power", 0.6, 1.0)),
        "alpha_h": float(alpha_h),
        "alpha_l": float(alpha_l),
        "beta_h": float(beta_h),
        "beta_l": float(beta_l),
        "delta_h": float(delta_h),
        "delta_l": float(delta_l),
        "k": float(trial.suggest_float("k", 0.5, 3.0)),
        "H": float(trial.suggest_int("H", 4, 12)),
    }


def _positive_train_p99(values: np.ndarray, train_idx: np.ndarray | None) -> float:
    val = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    if train_idx is not None:
        idx = np.asarray(train_idx, dtype=np.int32)
        idx = idx[(idx >= 0) & (idx < len(val))]
        stat_vals = val[idx] if len(idx) else val
    else:
        stat_vals = val
    finite_pos = stat_vals[np.isfinite(stat_vals) & (stat_vals > 0.0)]
    if not finite_pos.size:
        return 0.0
    cap = float(np.nanpercentile(finite_pos, 99.0))
    return cap if np.isfinite(cap) and cap > 0.0 else 0.0


@_numba_njit(cache=True, fastmath=True)
def _label_weight_raw_score_numba_kernel(
    mfe: np.ndarray,
    mae_abs: np.ndarray,
    t_mfe: np.ndarray,
    t_mae: np.ndarray,
    tau_tp: np.ndarray,
    tau_sl: np.ndarray,
    mae_before_tp: np.ndarray,
    mfe_before_sl: np.ndarray,
    tp: np.ndarray,
    sl: np.ndarray,
    base_tp: float,
    base_sl: float,
    alpha_h: float,
    alpha_l: float,
    beta_h: float,
    beta_l: float,
    delta_h: float,
    delta_l: float,
    horizon: float,
    mfe_beyond_cap: float,
    mae_beyond_cap: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = mfe.shape[0]
    raw_score = np.empty(n, dtype=np.float32)
    tp_first_out = np.empty(n, dtype=np.int8)
    sl_first_out = np.empty(n, dtype=np.int8)
    h = horizon if horizon > 1.0 else 1.0
    for i in range(n):
        tp_i = tp[i] if tp[i] > 1e-8 else 1e-8
        sl_i = sl[i] if sl[i] > 1e-8 else 1e-8
        mfe_i = mfe[i]
        mae_i = mae_abs[i]
        tau_tp_i = tau_tp[i]
        tau_sl_i = tau_sl[i]
        tp_hit = mfe_i >= tp_i
        sl_hit = mae_i >= sl_i
        has_tau_tp = np.isfinite(tau_tp_i)
        has_tau_sl = np.isfinite(tau_sl_i)
        tp_first = tp_hit and (
            (not sl_hit)
            or (has_tau_tp and has_tau_sl and tau_tp_i < tau_sl_i)
            or (has_tau_tp and not has_tau_sl)
        )
        sl_first = sl_hit and (
            (not tp_hit)
            or (has_tau_tp and has_tau_sl and tau_sl_i < tau_tp_i)
            or ((not has_tau_tp) and has_tau_sl)
        )
        tp_first_out[i] = 1 if tp_first else 0
        sl_first_out[i] = 1 if sl_first else 0

        tp_beyond_val = mfe_i - tp_i
        if tp_beyond_val < 0.0:
            tp_beyond_val = 0.0
        if mfe_beyond_cap > 0.0 and tp_beyond_val > mfe_beyond_cap:
            tp_beyond_val = mfe_beyond_cap
        tp_beyond = tp_beyond_val / tp_i
        if tp_beyond > 1.0:
            tp_beyond = 1.0

        sl_beyond_val = mae_i - sl_i
        if sl_beyond_val < 0.0:
            sl_beyond_val = 0.0
        if mae_beyond_cap > 0.0 and sl_beyond_val > mae_beyond_cap:
            sl_beyond_val = mae_beyond_cap
        sl_beyond = sl_beyond_val / sl_i
        if sl_beyond > 1.0:
            sl_beyond = 1.0

        less_mae = 1.0 - mae_before_tp[i] / sl_i
        if less_mae < 0.0:
            less_mae = 0.0
        elif less_mae > 1.0:
            less_mae = 1.0
        less_mfe = 1.0 - mfe_before_sl[i] / tp_i
        if less_mfe < 0.0:
            less_mfe = 0.0
        elif less_mfe > 1.0:
            less_mfe = 1.0

        fast_mfe = 0.0
        if np.isfinite(t_mfe[i]):
            fast_mfe = 1.0 - t_mfe[i] / h
            if fast_mfe < 0.0:
                fast_mfe = 0.0
            elif fast_mfe > 1.0:
                fast_mfe = 1.0
        fast_mae = 0.0
        if np.isfinite(t_mae[i]):
            fast_mae = 1.0 - t_mae[i] / h
            if fast_mae < 0.0:
                fast_mae = 0.0
            elif fast_mae > 1.0:
                fast_mae = 1.0

        higher = (
            (base_tp if tp_first else 0.0)
            + alpha_h * tp_beyond
            + beta_h * less_mae
            + delta_h * fast_mfe
        )
        lower = (
            (base_sl if sl_first else 0.0)
            + alpha_l * sl_beyond
            + beta_l * less_mfe
            + delta_l * fast_mae
        )
        raw_score[i] = higher - lower
    return raw_score, tp_first_out, sl_first_out


def _build_label_weight_soft_labels(
    ctx: dict[str, np.ndarray],
    params: dict[str, float],
    *,
    train_idx: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    atr = np.asarray(ctx["atr"], dtype=np.float32)
    if train_idx is not None:
        idx = np.asarray(train_idx, dtype=np.int32)
        idx = idx[(idx >= 0) & (idx < len(atr))]
        atr_train = atr[idx] if len(idx) else atr
    else:
        idx = None
        atr_train = atr
    atr_med = (
        float(np.nanmedian(atr_train[np.isfinite(atr_train) & (atr_train > 0.0)]))
        if np.isfinite(atr_train).any()
        else 1e-3
    )
    atr_power = float(np.clip(params.get("atr_power", 1.0), 0.1, 2.0))
    atr_ratio = np.maximum(atr, 1e-8) / max(atr_med, 1e-8)
    atr_eff = (max(atr_med, 1e-8) * np.power(atr_ratio, atr_power)).astype(np.float32)
    base_tp = float(params.get("base_tp", 2.0))
    base_sl = float(params.get("base_sl", 1.0))
    tp = np.maximum(base_tp * atr_eff, 1e-8).astype(np.float32)
    sl = np.maximum(base_sl * atr_eff, 1e-8).astype(np.float32)
    mfe = np.asarray(ctx["mfe"], dtype=np.float32)
    mae_abs = np.asarray(ctx["mae_abs"], dtype=np.float32)
    t_mfe = np.asarray(ctx["t_mfe"], dtype=np.float32)
    t_mae = np.asarray(ctx["t_mae"], dtype=np.float32)
    tau_tp = np.asarray(ctx.get("tau_tp", t_mfe), dtype=np.float32)
    tau_sl = np.asarray(ctx.get("tau_sl", t_mae), dtype=np.float32)
    h = max(float(params.get("H", 8.0)), 1.0)
    used_numba = False
    if bool(LGBM_LABEL_WEIGHT_HPO_NUMBA) and bool(_LGBM_LABEL_WEIGHT_NUMBA_AVAILABLE):
        mfe_beyond_cap = _positive_train_p99(mfe - tp, idx)
        mae_beyond_cap = _positive_train_p99(mae_abs - sl, idx)
        raw_score, tp_first_i, sl_first_i = _label_weight_raw_score_numba_kernel(
            mfe.astype(np.float32, copy=False),
            mae_abs.astype(np.float32, copy=False),
            t_mfe.astype(np.float32, copy=False),
            t_mae.astype(np.float32, copy=False),
            tau_tp.astype(np.float32, copy=False),
            tau_sl.astype(np.float32, copy=False),
            np.asarray(ctx["mae_before_tp"], dtype=np.float32),
            np.asarray(ctx["mfe_before_sl"], dtype=np.float32),
            tp,
            sl,
            base_tp,
            base_sl,
            float(params.get("alpha_h", 0.25)),
            float(params.get("alpha_l", 0.25)),
            float(params.get("beta_h", 0.20)),
            float(params.get("beta_l", 0.20)),
            float(params.get("delta_h", 0.10)),
            float(params.get("delta_l", 0.10)),
            h,
            float(mfe_beyond_cap),
            float(mae_beyond_cap),
        )
        tp_first = tp_first_i.astype(bool)
        sl_first = sl_first_i.astype(bool)
        raw_score = np.asarray(raw_score, dtype=np.float32)
        used_numba = True
    else:
        tp_hit = mfe >= tp
        sl_hit = mae_abs >= sl
        has_tau_tp = np.isfinite(tau_tp)
        has_tau_sl = np.isfinite(tau_sl)
        tp_first = tp_hit & (
            (~sl_hit)
            | (has_tau_tp & has_tau_sl & (tau_tp < tau_sl))
            | (has_tau_tp & ~has_tau_sl)
        )
        sl_first = sl_hit & (
            (~tp_hit)
            | (has_tau_tp & has_tau_sl & (tau_sl < tau_tp))
            | (~has_tau_tp & has_tau_sl)
        )
        tp_beyond = _winsorized_positive_ratio(mfe - tp, tp, train_idx=idx)
        sl_beyond = _winsorized_positive_ratio(mae_abs - sl, sl, train_idx=idx)
        less_mae_before_tp = 1.0 - np.clip(np.asarray(ctx["mae_before_tp"], dtype=np.float32) / sl, 0.0, 1.0)
        less_mfe_before_sl = 1.0 - np.clip(np.asarray(ctx["mfe_before_sl"], dtype=np.float32) / tp, 0.0, 1.0)
        has_t_mfe = np.isfinite(t_mfe)
        has_t_mae = np.isfinite(t_mae)
        lower_time_to_mfe = np.where(has_t_mfe, 1.0 - np.clip(t_mfe / h, 0.0, 1.0), 0.0).astype(np.float32)
        lower_time_to_mae = np.where(has_t_mae, 1.0 - np.clip(t_mae / h, 0.0, 1.0), 0.0).astype(np.float32)
        higher = (
            tp_first.astype(np.float32) * base_tp
            + float(params.get("alpha_h", 0.25)) * tp_beyond
            + float(params.get("beta_h", 0.20)) * less_mae_before_tp
            + float(params.get("delta_h", 0.10)) * lower_time_to_mfe
        )
        lower = (
            sl_first.astype(np.float32) * base_sl
            + float(params.get("alpha_l", 0.25)) * sl_beyond
            + float(params.get("beta_l", 0.20)) * less_mfe_before_sl
            + float(params.get("delta_l", 0.10)) * lower_time_to_mae
        )
        raw_score = (higher - lower).astype(np.float32)
    stat_raw = raw_score[idx] if idx is not None and len(idx) else raw_score
    finite_raw = stat_raw[np.isfinite(stat_raw)]
    if finite_raw.size:
        center = float(np.nanmedian(finite_raw))
        q75, q25 = np.nanpercentile(finite_raw, [75.0, 25.0])
        scale = float(max(float(q75 - q25), 1e-6))
    else:
        center = 0.0
        scale = 1.0
    raw_score_z = ((raw_score - center) / scale).astype(np.float32)
    soft = _sigmoid(float(params.get("k", 1.0)) * raw_score_z)
    hard = (soft >= 0.5).astype(np.int8)
    stats = {
        "soft_mean": float(np.mean(soft)) if len(soft) else float("nan"),
        "soft_std": float(np.std(soft)) if len(soft) else float("nan"),
        "hard_positive_rate": float(np.mean(hard)) if len(hard) else float("nan"),
        "tp_first_rate": float(np.mean(tp_first)) if len(tp_first) else float("nan"),
        "sl_first_rate": float(np.mean(sl_first)) if len(sl_first) else float("nan"),
        "atr_effective_median": float(np.nanmedian(atr_eff)) if len(atr_eff) else float("nan"),
        "atr_pct_train_median": float(atr_med),
        "raw_score_center": float(center),
        "raw_score_iqr_scale": float(scale),
        "numba_raw_score_kernel": bool(used_numba),
        "numba_available": bool(_LGBM_LABEL_WEIGHT_NUMBA_AVAILABLE),
    }
    return soft.astype(np.float32), hard, stats


def _sample_weight_hpo_params(trial: Any) -> dict[str, float]:
    return {
        "gamma": float(trial.suggest_float("gamma", 1.5, 3.0)),
        "weight_delta": float(trial.suggest_float("weight_delta", 1.0, 3.0)),
        "w_max": float(trial.suggest_float("w_max", 3.0, 5.0)),
        "neutral_band": float(trial.suggest_float("neutral_band", 0.01, 0.20)),
    }


def _build_label_weight_hpo_weights(
    soft_label: np.ndarray,
    base_weight: np.ndarray,
    params: dict[str, float] | None,
    *,
    train_idx: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not params:
        return np.asarray(base_weight, dtype=np.float32), {
            "weight_mode": "baseline",
            "n_eff": float(_effective_n(base_weight)),
        }
    confidence = np.abs(np.asarray(soft_label, dtype=np.float32) - 0.5)
    neutral = float(np.clip(params.get("neutral_band", 0.05), 0.0, 0.5))
    gamma = float(np.clip(params.get("gamma", 2.0), 0.1, 10.0))
    weight_delta = float(np.clip(params.get("weight_delta", params.get("delta", 1.0)), 0.0, 10.0))
    w_min = 0.10
    w_max = float(np.clip(params.get("w_max", 4.0), w_min, 20.0))
    row_weight = w_min + np.power(weight_delta * confidence, gamma)
    row_weight = np.asarray(row_weight, dtype=np.float32)
    row_weight[confidence < neutral] = w_min
    row_weight = np.clip(row_weight, w_min, w_max).astype(np.float32, copy=False)
    if train_idx is not None:
        idx = np.asarray(train_idx, dtype=np.int32)
        idx = idx[(idx >= 0) & (idx < len(row_weight))]
        stat_weight = row_weight[idx] if len(idx) else row_weight
    else:
        idx = None
        stat_weight = row_weight
    row_weight = row_weight / max(float(np.mean(stat_weight)), 1e-6)
    max_combined = max(4.0, w_max)
    combined = np.nan_to_num(
        np.asarray(base_weight, dtype=np.float32) * row_weight,
        nan=1.0,
        posinf=max_combined,
        neginf=0.05,
    )
    combined = np.clip(combined, 0.05, max_combined).astype(np.float32, copy=False)
    stat_combined = combined[idx] if idx is not None and len(idx) else combined
    combined = combined / max(float(np.mean(stat_combined)), 1e-6)
    combined = np.clip(combined, 0.05, max_combined).astype(np.float32, copy=False)
    stat_combined_after = combined[idx] if idx is not None and len(idx) else combined
    ess = float(_effective_n(stat_combined_after))
    return combined.astype(np.float32), {
        "weight_mode": "optimized",
        "gamma": gamma,
        "weight_delta": weight_delta,
        "w_min": w_min,
        "w_max": w_max,
        "neutral_band": neutral,
        "neutral_fraction": float(np.mean(confidence < neutral)) if len(confidence) else float("nan"),
        "row_weight_mean": float(np.mean(row_weight)) if len(row_weight) else float("nan"),
        "row_weight_train_mean": float(np.mean(stat_weight)) if len(stat_weight) else float("nan"),
        "row_weight_p95": float(np.nanpercentile(row_weight, 95.0)) if len(row_weight) else float("nan"),
        "n_eff": float(ess),
    }


def apply_label_weight_hpo_report_to_arrays(
    y_arr: np.ndarray,
    y_metric: np.ndarray,
    sample_weight: np.ndarray | None,
    label_context: dict[str, Any] | None,
    report: dict[str, Any] | None,
    *,
    objective_mode: str,
    classifier: bool,
    label: str = "preset",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply a previous elected train_base label/weight recipe to aligned arrays."""

    y_out = np.asarray(y_arr, dtype=np.float32)
    y_metric_out = np.asarray(y_metric, dtype=np.float32)
    sw = (
        np.ones(len(y_out), dtype=np.float32)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float32)
    )
    sw, _ = _normalize_weights(sw)
    diag: dict[str, Any] = {
        "applied": False,
        "label": str(label),
        "objective_mode": str(objective_mode),
        "classifier": bool(classifier),
    }
    if not classifier or str(objective_mode) != "train_base":
        diag["reason"] = "non_train_base_or_non_classifier"
        return y_out, y_metric_out, sw, diag
    if not isinstance(report, dict) or not report:
        diag["reason"] = "missing_report"
        return y_out, y_metric_out, sw, diag
    winner = str(report.get("winner", "baseline")).strip().lower()
    selected = bool(report.get("selected", False))
    diag["source_winner"] = winner
    diag["source_selected"] = selected
    if winner != "optimized" or not selected:
        diag["reason"] = "source_winner_baseline"
        tprint(
            "LGBM prior label/sample-weight config reuse skipped: "
            f"label={label}, winner={winner or 'baseline'}."
        )
        return y_out, y_metric_out, sw, diag
    best = report.get("best_optimized") or {}
    if not isinstance(best, dict):
        diag["reason"] = "missing_best_optimized"
        return y_out, y_metric_out, sw, diag
    label_params = dict(best.get("label_params") or {})
    weight_params_raw = best.get("weight_params")
    weight_params = dict(weight_params_raw or {}) if isinstance(weight_params_raw, dict) else None
    if not label_params:
        diag["reason"] = "missing_label_params"
        return y_out, y_metric_out, sw, diag
    ctx, ctx_diag = _prepare_label_weight_hpo_context(label_context, len(y_out))
    diag["context"] = ctx_diag
    if ctx is None:
        diag["reason"] = str(ctx_diag.get("reason", "missing_context"))
        return y_out, y_metric_out, sw, diag
    train_idx = np.arange(len(y_out), dtype=np.int32)
    try:
        soft, hard, label_stats = _build_label_weight_soft_labels(
            ctx,
            label_params,
            train_idx=train_idx,
        )
        train_label = soft if bool(LGBM_TRUE_SOFT_LABELS) else hard.astype(np.float32)
        if weight_params:
            sw_reused, weight_stats = _build_label_weight_hpo_weights(
                soft,
                sw,
                weight_params,
                train_idx=train_idx,
            )
        else:
            sw_reused = sw
            weight_stats = {"weight_mode": "baseline"}
    except Exception as exc:
        diag["reason"] = f"reuse_failed:{type(exc).__name__}:{exc}"
        return y_out, y_metric_out, sw, diag
    diag.update(
        {
            "applied": True,
            "reason": "applied",
            "source_objective": float(best.get("objective", float("nan"))),
            "label_params": _json_sanitize(label_params),
            "weight_params": _json_sanitize(weight_params),
            "label_stats": _json_sanitize(label_stats),
            "weight_stats": _json_sanitize(weight_stats),
        }
    )
    tprint(
        "LGBM prior label/sample-weight config reused: "
        f"label={label}, source_objective={float(best.get('objective', float('nan'))):.5f}, "
        f"soft_mean={float(label_stats.get('soft_mean', float('nan'))):.4f}, "
        f"hard_pos={float(label_stats.get('hard_positive_rate', float('nan'))):.4f}, "
        f"weight_mode={weight_stats.get('weight_mode', 'baseline')}."
    )
    return (
        np.asarray(train_label, dtype=np.float32),
        np.asarray(hard, dtype=np.int8),
        np.asarray(sw_reused, dtype=np.float32),
        diag,
    )


def _label_weight_hpo_pair_hit_mask(
    ctx: dict[str, np.ndarray],
    tp_mult: float,
    sl_mult: float,
) -> np.ndarray:
    atr = np.asarray(ctx["atr"], dtype=np.float32)
    tp = np.maximum(float(tp_mult) * atr, 1e-8)
    sl = np.maximum(float(sl_mult) * atr, 1e-8)
    mfe = np.asarray(ctx["mfe"], dtype=np.float32)
    mae_abs = np.asarray(ctx["mae_abs"], dtype=np.float32)
    tau_tp = np.asarray(ctx.get("tau_tp", ctx["t_mfe"]), dtype=np.float32)
    tau_sl = np.asarray(ctx.get("tau_sl", ctx["t_mae"]), dtype=np.float32)
    tp_hit = mfe >= tp
    sl_hit = mae_abs >= sl
    has_t_mfe = np.isfinite(tau_tp)
    has_t_mae = np.isfinite(tau_sl)
    return tp_hit & (
        (~sl_hit)
        | (has_t_mfe & has_t_mae & (tau_tp < tau_sl))
        | (has_t_mfe & ~has_t_mae)
    )


def _label_weight_hpo_objective_from_pred(
    pred: np.ndarray,
    ctx: dict[str, np.ndarray],
) -> dict[str, Any]:
    p = np.asarray(pred, dtype=np.float32).reshape(-1)
    finite = np.isfinite(p)
    if int(np.sum(finite)) < 10:
        return {"objective": float("-inf"), "finite_rows": int(np.sum(finite))}
    pair_scores: list[float] = []
    components: dict[str, float] = {}
    for tp_mult, sl_mult, pair_name in ((3.0, 2.0, "tp3_sl2"), (2.0, 1.0, "tp2_sl1")):
        hit = _label_weight_hpo_pair_hit_mask(ctx, tp_mult, sl_mult).astype(bool)
        order = np.argsort(np.nan_to_num(p[finite], nan=-np.inf))
        hit_f = hit[finite]
        pair_score = 0.0
        for top_pct, weight in ((30, 3.0), (20, 2.0), (10, 1.0), (5, 0.5)):
            top_n = max(1, int(np.ceil((float(top_pct) / 100.0) * len(order))))
            top_idx = order[-top_n:]
            hr = float(np.mean(hit_f[top_idx])) if len(top_idx) else float("nan")
            components[f"{pair_name}_hr_at_{top_pct}"] = hr
            pair_score += float(weight) * hr
        components[f"{pair_name}_weighted_score"] = float(pair_score)
        pair_scores.append(float(pair_score))
    objective = float(np.mean(pair_scores)) if pair_scores else float("-inf")
    components["objective"] = objective
    components["finite_rows"] = int(np.sum(finite))
    return components


def _label_weight_hpo_eval_params(best_params: dict[str, Any], random_state: int) -> dict[str, Any]:
    params = dict(best_params or {})
    params["random_state"] = int(random_state)
    if LGBM_LABEL_WEIGHT_HPO_N_ESTIMATORS_CAP > 0:
        params["n_estimators"] = min(
            int(params.get("n_estimators", LGBM_LABEL_WEIGHT_HPO_N_ESTIMATORS_CAP)),
            int(LGBM_LABEL_WEIGHT_HPO_N_ESTIMATORS_CAP),
        )
    return params


def _evaluate_label_weight_hpo_candidate(
    X_model_df: pd.DataFrame,
    selected_features: list[str],
    y_train_full: np.ndarray,
    y_metric_full: np.ndarray,
    weight_full: np.ndarray,
    hpo_idx: np.ndarray,
    ctx_full: dict[str, np.ndarray],
    best_params: dict[str, Any],
    *,
    classifier: bool,
    timestamps: Any = None,
    random_state: int = 42,
    label_params: dict[str, float] | None = None,
    weight_params: dict[str, float] | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    if not classifier:
        return {"objective": float("-inf"), "reason": "non_classifier"}
    idx = np.asarray(hpo_idx, dtype=np.int32)
    idx = idx[(idx >= 0) & (idx < len(y_train_full))]
    row_cap = int(LGBM_LABEL_WEIGHT_HPO_MAX_ROWS if max_rows is None else max_rows)
    row_cap = max(0, row_cap)
    if row_cap > 0 and len(idx) > row_cap:
        keep_local = _stratified_spread_subsample_indices(
            np.asarray(y_metric_full, dtype=np.float32)[idx],
            int(row_cap),
            int(random_state) + 9029,
            classifier=True,
        )
        idx = np.sort(idx[keep_local].astype(np.int32))
    if len(idx) < LGBM_LABEL_WEIGHT_HPO_MIN_ROWS:
        return {
            "objective": float("-inf"),
            "reason": "insufficient_hpo_rows",
            "hpo_rows_used": int(len(idx)),
            "hpo_row_cap": int(row_cap),
        }
    y_metric_base = np.asarray(y_metric_full, dtype=np.float32)[idx]
    if len(np.unique(y_metric_base >= 0.5)) < 2:
        return {
            "objective": float("-inf"),
            "reason": "single_class_candidate_labels",
            "hpo_rows_used": int(len(idx)),
            "hpo_row_cap": int(row_cap),
        }
    X_eval = X_model_df.iloc[idx][selected_features].reset_index(drop=True)
    y_train_base = np.asarray(y_train_full, dtype=np.float32)[idx]
    y_metric_candidate = y_metric_base
    weights_base = np.asarray(weight_full, dtype=np.float32)[idx]
    ts_eval = _take_aligned(timestamps, idx, len(y_train_full))
    splitter, y_split = _cv_splitter(
        y_metric_base,
        True,
        int(random_state),
        timestamps=ts_eval,
        n_splits=int(LGBM_LABEL_WEIGHT_HPO_CV_SPLITS),
    )
    oof = np.full(len(idx), np.nan, dtype=np.float32)
    fold_count = 0
    params = _label_weight_hpo_eval_params(best_params, int(random_state))
    ctx_eval = {key: np.asarray(val)[idx] for key, val in ctx_full.items()}
    try:
        for fold, (tr, va) in enumerate(splitter.split(X_eval, y_split), start=1):
            tr = np.asarray(tr, dtype=np.int32)
            va = np.asarray(va, dtype=np.int32)
            if len(tr) < 20 or len(va) < 10:
                continue
            if label_params:
                soft_eval, hard_eval, _ = _build_label_weight_soft_labels(
                    ctx_eval,
                    label_params,
                    train_idx=tr,
                )
                y_train_eval = (
                    soft_eval
                    if bool(LGBM_TRUE_SOFT_LABELS)
                    else hard_eval.astype(np.float32)
                )
                y_metric_candidate = hard_eval.astype(np.float32)
                if weight_params:
                    weights_eval, _ = _build_label_weight_hpo_weights(
                        soft_eval,
                        weights_base,
                        weight_params,
                        train_idx=tr,
                    )
                else:
                    weights_eval = weights_base
            else:
                y_train_eval = y_train_base
                y_metric_candidate = y_metric_base
                weights_eval = weights_base
            if len(np.unique((y_metric_candidate[tr] >= 0.5).astype(np.int8))) < 2:
                continue
            model = _fit_lgbm_model(
                X_eval.iloc[tr].reset_index(drop=True),
                y_train_eval[tr],
                weights_eval[tr],
                classifier=True,
                params={**params, "random_state": int(random_state + fold * 1009)},
                X_valid=X_eval.iloc[va].reset_index(drop=True),
                y_valid=y_train_eval[va],
                early_stopping_rounds=(
                    int(LGBM_LABEL_WEIGHT_HPO_EARLY_STOPPING_ROUNDS)
                    if LGBM_LABEL_WEIGHT_HPO_EARLY_STOPPING_ROUNDS > 0
                    else None
                ),
            )
            oof[va] = _predict_lgbm_raw(model, X_eval.iloc[va], "classifier")
            fold_count += 1
    except Exception as exc:
        return {
            "objective": float("-inf"),
            "reason": f"eval_failed:{type(exc).__name__}:{exc}",
            "hpo_rows_used": int(len(idx)),
            "hpo_row_cap": int(row_cap),
        }
    metrics = _label_weight_hpo_objective_from_pred(oof, ctx_eval)
    metrics["hpo_rows_used"] = int(len(idx))
    metrics["hpo_row_cap"] = int(row_cap)
    metrics["fold_count"] = int(fold_count)
    metrics["n_estimators_cap"] = int(LGBM_LABEL_WEIGHT_HPO_N_ESTIMATORS_CAP)
    if fold_count <= 0:
        metrics["objective"] = float("-inf")
        metrics["reason"] = "no_valid_cv_folds"
    return metrics


def _best_completed_trials(study: Any, optuna_module: Any) -> list[Any]:
    complete_state = optuna_module.trial.TrialState.COMPLETE
    trials = [t for t in getattr(study, "trials", []) if getattr(t, "state", None) == complete_state]
    return sorted(trials, key=lambda t: float(t.value), reverse=True)


def _run_base_label_weight_hpo(
    X_model_df: pd.DataFrame,
    y_arr: np.ndarray,
    y_metric: np.ndarray,
    sw: np.ndarray,
    selected_features: list[str],
    hpo_idx: np.ndarray,
    label_context: dict[str, Any] | None,
    best_params: dict[str, Any],
    *,
    classifier: bool,
    objective_mode: str,
    timestamps: Any = None,
    random_state: int = 42,
    reference_artifact_dir: str | os.PathLike[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    report: dict[str, Any] = {
        "enabled": False,
        "objective_mode": str(objective_mode),
        "selected": False,
        "winner": "baseline",
    }
    if (
        not bool(LGBM_BASE_LABEL_WEIGHT_HPO_ENABLED)
        or not classifier
        or str(objective_mode) != "train_base"
        or LGBM_LABEL_WEIGHT_HPO_LAYER1_TRIALS <= 0
    ):
        report["reason"] = (
            "disabled_or_not_train_base"
            if not bool(LGBM_BASE_LABEL_WEIGHT_HPO_ENABLED) or str(objective_mode) != "train_base"
            else "not_classifier_or_no_trials"
        )
        return y_arr, y_metric, sw, report, None, None, None
    ctx, ctx_diag = _prepare_label_weight_hpo_context(label_context, len(y_arr))
    report.update(ctx_diag)
    if ctx is None:
        return y_arr, y_metric, sw, report, None, None, None
    try:
        import optuna
    except Exception as exc:
        report["enabled"] = False
        report["reason"] = f"optuna_unavailable:{type(exc).__name__}:{exc}"
        return y_arr, y_metric, sw, report, None, None, None

    report["enabled"] = True
    report["hpo_rows_requested"] = int(len(hpo_idx))
    report["layer1_trials_configured"] = int(LGBM_LABEL_WEIGHT_HPO_LAYER1_TRIALS)
    report["layer2_trials_configured"] = int(LGBM_LABEL_WEIGHT_HPO_LAYER2_TRIALS)
    report["layer1_patience"] = int(LGBM_LABEL_WEIGHT_HPO_LAYER1_PATIENCE)
    report["layer2_patience"] = int(LGBM_LABEL_WEIGHT_HPO_LAYER2_PATIENCE)
    report["cv_splits"] = int(LGBM_LABEL_WEIGHT_HPO_CV_SPLITS)
    report["max_rows"] = int(LGBM_LABEL_WEIGHT_HPO_MAX_ROWS)
    report["search_max_rows"] = int(LGBM_LABEL_WEIGHT_HPO_MAX_ROWS)
    report["election_max_rows"] = int(LGBM_LABEL_WEIGHT_HPO_ELECTION_MAX_ROWS)
    report["numba_enabled"] = bool(LGBM_LABEL_WEIGHT_HPO_NUMBA)
    report["numba_available"] = bool(_LGBM_LABEL_WEIGHT_NUMBA_AVAILABLE)
    report["optuna_sampler"] = "TPESampler"
    report["optuna_pruner"] = "MedianPruner"
    report["model_eval_early_stopping_rounds"] = int(
        LGBM_LABEL_WEIGHT_HPO_EARLY_STOPPING_ROUNDS
    )
    report["objective"] = "avg(tp3_sl2,tp2_sl1) weighted top-k hit rate: 3*HR@30+2*HR@20+HR@10+0.5*HR@5"
    report["fold_stat_policy"] = (
        "label ATR median, winsor p99, raw-score median/IQR, and weight mean "
        "are computed on each CV train fold only; elected final labels use full final-train statistics."
    )
    baseline_eval = _evaluate_label_weight_hpo_candidate(
        X_model_df,
        selected_features,
        y_arr,
        y_metric,
        sw,
        hpo_idx,
        ctx,
        best_params,
        classifier=classifier,
        timestamps=timestamps,
        random_state=random_state + 12011,
    )
    report["baseline"] = dict(baseline_eval)
    tprint(
        "LGBM label/sample-weight HPO baseline: "
        f"objective={float(baseline_eval.get('objective', float('-inf'))):.5f}, "
        f"rows={int(baseline_eval.get('hpo_rows_used', 0))}, "
        f"folds={int(baseline_eval.get('fold_count', 0))}."
    )

    best_seen = {"value": float("-inf"), "stale": 0}

    def _patience_callback(study: Any, trial: Any) -> None:
        value = float(trial.value) if trial.value is not None else float("-inf")
        if value > best_seen["value"] + 1e-8:
            best_seen["value"] = value
            best_seen["stale"] = 0
        else:
            best_seen["stale"] += 1
        if best_seen["stale"] >= LGBM_LABEL_WEIGHT_HPO_LAYER1_PATIENCE:
            study.stop()

    def _layer1_objective(trial: Any) -> float:
        params = _sample_label_barrier_params(trial)
        _, _, stats = _build_label_weight_soft_labels(ctx, params, train_idx=hpo_idx)
        metrics = _evaluate_label_weight_hpo_candidate(
            X_model_df,
            selected_features,
            y_arr,
            y_metric,
            sw,
            hpo_idx,
            ctx,
            best_params,
            classifier=classifier,
            timestamps=timestamps,
            random_state=random_state + 13007 + int(trial.number) * 13,
            label_params=params,
        )
        trial.set_user_attr("params", _json_sanitize(params))
        trial.set_user_attr("label_stats", _json_sanitize(stats))
        trial.set_user_attr("metrics", _json_sanitize(metrics))
        value = float(metrics.get("objective", float("-inf")))
        if not np.isfinite(value):
            value = -1e9
        if trial.number == 0 or (trial.number + 1) % 25 == 0 or value > best_seen["value"]:
            tprint(
                "LGBM label/sample-weight HPO layer1 "
                f"trial={int(trial.number) + 1} objective={value:.5f} "
                f"best={max(value, best_seen['value']):.5f}."
            )
        return value

    study1 = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=int(random_state) + 221),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=20),
    )
    study1.optimize(
        _layer1_objective,
        n_trials=int(LGBM_LABEL_WEIGHT_HPO_LAYER1_TRIALS),
        callbacks=[_patience_callback],
        show_progress_bar=False,
    )
    complete1 = _best_completed_trials(study1, optuna)
    report["layer1_completed_trials"] = int(len(complete1))
    report["layer1_top3"] = [
        {
            "rank": int(i + 1),
            "objective": float(t.value),
            "params": _json_sanitize(t.user_attrs.get("params", {})),
            "label_stats": _json_sanitize(t.user_attrs.get("label_stats", {})),
            "metrics": _json_sanitize(t.user_attrs.get("metrics", {})),
        }
        for i, t in enumerate(complete1[:3])
    ]
    if not complete1:
        report["reason"] = "no_completed_layer1_trials"
        return y_arr, y_metric, sw, report, None, None, None
    best_label_params = dict(complete1[0].user_attrs.get("params") or {})
    best_soft, best_hard, best_label_stats = _build_label_weight_soft_labels(
        ctx,
        best_label_params,
        train_idx=np.arange(len(y_arr), dtype=np.int32),
    )
    best_layer1_train = best_soft if bool(LGBM_TRUE_SOFT_LABELS) else best_hard.astype(np.float32)
    best_layer1_weight = sw
    best_layer1_eval = dict(complete1[0].user_attrs.get("metrics") or {})
    tprint(
        "LGBM label/sample-weight HPO layer1 complete: "
        f"completed={len(complete1)}, best_objective={float(best_layer1_eval.get('objective', complete1[0].value)):.5f}, "
        f"top3={[round(float(t.value), 5) for t in complete1[:3]]}."
    )

    best_candidate = {
        "name": "optimized_labels",
        "objective": float(best_layer1_eval.get("objective", complete1[0].value)),
        "label_params": best_label_params,
        "weight_params": None,
        "label_stats": best_label_stats,
        "metrics": best_layer1_eval,
        "soft": best_soft,
        "hard": best_hard,
        "train": best_layer1_train,
        "weight": best_layer1_weight,
    }
    complete2: list[Any] = []
    if LGBM_LABEL_WEIGHT_HPO_LAYER2_TRIALS > 0:
        best_seen2 = {"value": float("-inf"), "stale": 0}

        def _patience_callback2(study: Any, trial: Any) -> None:
            value = float(trial.value) if trial.value is not None else float("-inf")
            if value > best_seen2["value"] + 1e-8:
                best_seen2["value"] = value
                best_seen2["stale"] = 0
            else:
                best_seen2["stale"] += 1
            if best_seen2["stale"] >= LGBM_LABEL_WEIGHT_HPO_LAYER2_PATIENCE:
                study.stop()

        def _layer2_objective(trial: Any) -> float:
            weight_params = _sample_weight_hpo_params(trial)
            _, weight_stats = _build_label_weight_hpo_weights(
                best_soft,
                sw,
                weight_params,
                train_idx=np.arange(len(y_arr), dtype=np.int32),
            )
            metrics = _evaluate_label_weight_hpo_candidate(
                X_model_df,
                selected_features,
                y_arr,
                y_metric,
                sw,
                hpo_idx,
                ctx,
                best_params,
                classifier=classifier,
                timestamps=timestamps,
                random_state=random_state + 17011 + int(trial.number) * 17,
                label_params=best_label_params,
                weight_params=weight_params,
            )
            trial.set_user_attr("params", _json_sanitize(weight_params))
            trial.set_user_attr("weight_stats", _json_sanitize(weight_stats))
            trial.set_user_attr("metrics", _json_sanitize(metrics))
            value = float(metrics.get("objective", float("-inf")))
            if not np.isfinite(value):
                value = -1e9
            if trial.number == 0 or (trial.number + 1) % 25 == 0 or value > best_seen2["value"]:
                tprint(
                    "LGBM label/sample-weight HPO layer2 "
                    f"trial={int(trial.number) + 1} objective={value:.5f} "
                    f"best={max(value, best_seen2['value']):.5f}."
                )
            return value

        study2 = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=int(random_state) + 337),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=15),
        )
        study2.optimize(
            _layer2_objective,
            n_trials=int(LGBM_LABEL_WEIGHT_HPO_LAYER2_TRIALS),
            callbacks=[_patience_callback2],
            show_progress_bar=False,
        )
        complete2 = _best_completed_trials(study2, optuna)
        report["layer2_completed_trials"] = int(len(complete2))
        report["layer2_top3"] = [
            {
                "rank": int(i + 1),
                "objective": float(t.value),
                "weight_params": _json_sanitize(t.user_attrs.get("params", {})),
                "weight_stats": _json_sanitize(t.user_attrs.get("weight_stats", {})),
                "metrics": _json_sanitize(t.user_attrs.get("metrics", {})),
            }
            for i, t in enumerate(complete2[:3])
        ]
        if complete2:
            best_w_params = dict(complete2[0].user_attrs.get("params") or {})
            best_w_full, best_w_stats = _build_label_weight_hpo_weights(
                best_soft,
                sw,
                best_w_params,
                train_idx=np.arange(len(y_arr), dtype=np.int32),
            )
            best_candidate.update(
                {
                    "name": "optimized_labels_and_weights",
                    "objective": float(complete2[0].value),
                    "weight_params": best_w_params,
                    "weight_stats": best_w_stats,
                    "metrics": dict(complete2[0].user_attrs.get("metrics") or {}),
                    "weight": best_w_full,
                }
            )
            tprint(
                "LGBM label/sample-weight HPO layer2 complete: "
                f"completed={len(complete2)}, best_objective={float(complete2[0].value):.5f}, "
                f"top3={[round(float(t.value), 5) for t in complete2[:3]]}."
            )
    election_specs: list[dict[str, Any]] = []
    if complete2:
        for i, trial in enumerate(complete2[:3]):
            election_specs.append(
                {
                    "rank": int(i + 1),
                    "name": "optimized_labels_and_weights",
                    "source": "layer2_top3",
                    "search_objective": float(trial.value),
                    "label_params": dict(best_label_params),
                    "weight_params": dict(trial.user_attrs.get("params") or {}),
                    "search_metrics": dict(trial.user_attrs.get("metrics") or {}),
                }
            )
    else:
        for i, trial in enumerate(complete1[:3]):
            election_specs.append(
                {
                    "rank": int(i + 1),
                    "name": "optimized_labels",
                    "source": "layer1_top3",
                    "search_objective": float(trial.value),
                    "label_params": dict(trial.user_attrs.get("params") or {}),
                    "weight_params": None,
                    "search_metrics": dict(trial.user_attrs.get("metrics") or {}),
                }
            )

    baseline_election_eval = _evaluate_label_weight_hpo_candidate(
        X_model_df,
        selected_features,
        y_arr,
        y_metric,
        sw,
        hpo_idx,
        ctx,
        best_params,
        classifier=classifier,
        timestamps=timestamps,
        random_state=random_state + 19001,
        max_rows=int(LGBM_LABEL_WEIGHT_HPO_ELECTION_MAX_ROWS),
    )
    report["election_baseline"] = dict(baseline_election_eval)
    tprint(
        "LGBM label/sample-weight HPO top3 election baseline: "
        f"objective={float(baseline_election_eval.get('objective', float('-inf'))):.5f}, "
        f"rows={int(baseline_election_eval.get('hpo_rows_used', 0))}, "
        f"cap={int(baseline_election_eval.get('hpo_row_cap', LGBM_LABEL_WEIGHT_HPO_ELECTION_MAX_ROWS))}."
    )

    election_results: list[dict[str, Any]] = []
    if election_specs:
        tprint(
            "LGBM label/sample-weight HPO top3 election started: "
            f"candidates={len(election_specs)}, row_cap={int(LGBM_LABEL_WEIGHT_HPO_ELECTION_MAX_ROWS)}."
        )
    for i, spec in enumerate(election_specs):
        metrics = _evaluate_label_weight_hpo_candidate(
            X_model_df,
            selected_features,
            y_arr,
            y_metric,
            sw,
            hpo_idx,
            ctx,
            best_params,
            classifier=classifier,
            timestamps=timestamps,
            random_state=random_state + 20011 + int(i) * 101,
            label_params=spec.get("label_params"),
            weight_params=spec.get("weight_params"),
            max_rows=int(LGBM_LABEL_WEIGHT_HPO_ELECTION_MAX_ROWS),
        )
        result = {
            **spec,
            "election_objective": float(metrics.get("objective", float("-inf"))),
            "election_metrics": dict(metrics),
        }
        election_results.append(result)
        tprint(
            "LGBM label/sample-weight HPO top3 election candidate: "
            f"source={spec.get('source')}, rank={int(spec.get('rank', i + 1))}, "
            f"search_objective={float(spec.get('search_objective', float('-inf'))):.5f}, "
            f"election_objective={float(result['election_objective']):.5f}, "
            f"rows={int(metrics.get('hpo_rows_used', 0))}."
        )
    election_results = sorted(
        election_results,
        key=lambda item: float(item.get("election_objective", float("-inf"))),
        reverse=True,
    )
    report["election_top3"] = [
        {
            "rank": int(i + 1),
            "source_rank": int(item.get("rank", i + 1)),
            "source": str(item.get("source", "")),
            "name": str(item.get("name", "")),
            "search_objective": float(item.get("search_objective", float("-inf"))),
            "election_objective": float(item.get("election_objective", float("-inf"))),
            "label_params": _json_sanitize(item.get("label_params", {})),
            "weight_params": _json_sanitize(item.get("weight_params", None)),
            "search_metrics": _json_sanitize(item.get("search_metrics", {})),
            "election_metrics": _json_sanitize(item.get("election_metrics", {})),
        }
        for i, item in enumerate(election_results[:3])
    ]
    report["top3_winners"] = list(report.get("election_top3") or report.get("layer2_top3") or report.get("layer1_top3") or [])[:3]

    baseline_score = float(baseline_election_eval.get("objective", baseline_eval.get("objective", float("-inf"))))
    if election_results:
        selected_spec = dict(election_results[0])
        best_candidate = {
            "name": str(selected_spec.get("name", "optimized_labels")),
            "objective": float(selected_spec.get("election_objective", float("-inf"))),
            "search_objective": float(selected_spec.get("search_objective", float("-inf"))),
            "label_params": dict(selected_spec.get("label_params") or {}),
            "weight_params": (
                dict(selected_spec.get("weight_params") or {})
                if selected_spec.get("weight_params") is not None
                else None
            ),
            "metrics": dict(selected_spec.get("election_metrics") or {}),
            "search_metrics": dict(selected_spec.get("search_metrics") or {}),
            "election_source": str(selected_spec.get("source", "")),
            "election_source_rank": int(selected_spec.get("rank", 0)),
        }
    best_score = float(best_candidate.get("objective", float("-inf")))
    keep_optimized = bool(np.isfinite(best_score) and best_score > baseline_score)
    if keep_optimized:
        full_train_idx = np.arange(len(y_arr), dtype=np.int32)
        elected_soft, elected_hard, elected_label_stats = _build_label_weight_soft_labels(
            ctx,
            dict(best_candidate.get("label_params") or {}),
            train_idx=full_train_idx,
        )
        elected_train = (
            elected_soft
            if bool(LGBM_TRUE_SOFT_LABELS)
            else elected_hard.astype(np.float32)
        )
        if best_candidate.get("weight_params"):
            elected_weight, elected_weight_stats = _build_label_weight_hpo_weights(
                elected_soft,
                sw,
                dict(best_candidate.get("weight_params") or {}),
                train_idx=full_train_idx,
            )
        else:
            elected_weight = sw
            elected_weight_stats = {"weight_mode": "baseline"}
        best_candidate["label_stats"] = elected_label_stats
        best_candidate["weight_stats"] = elected_weight_stats
    report["best_optimized"] = {
        "name": str(best_candidate.get("name", "optimized_labels")),
        "objective": best_score,
        "search_objective": float(best_candidate.get("search_objective", best_score)),
        "election_source": str(best_candidate.get("election_source", "")),
        "election_source_rank": int(best_candidate.get("election_source_rank", 0)),
        "label_params": _json_sanitize(best_candidate.get("label_params", {})),
        "weight_params": _json_sanitize(best_candidate.get("weight_params", None)),
        "label_stats": _json_sanitize(best_candidate.get("label_stats", {})),
        "weight_stats": _json_sanitize(best_candidate.get("weight_stats", {})),
        "metrics": _json_sanitize(best_candidate.get("metrics", {})),
        "search_metrics": _json_sanitize(best_candidate.get("search_metrics", {})),
    }
    report["winner"] = "optimized" if keep_optimized else "baseline"
    report["selected"] = bool(keep_optimized)
    report["objective_delta_vs_baseline"] = float(best_score - baseline_score)
    if keep_optimized:
        elected_train = np.asarray(elected_train, dtype=np.float32)
        elected_hard = np.asarray(elected_hard, dtype=np.int8)
        elected_weight = np.asarray(elected_weight, dtype=np.float32)
        elected_soft = np.asarray(elected_soft, dtype=np.float32)
    else:
        elected_train = y_arr
        elected_hard = np.asarray(y_metric >= 0.5, dtype=np.int8)
        elected_weight = sw
        elected_soft = None
    tprint(
        "LGBM label/sample-weight HPO election: "
        f"winner={report['winner']}, baseline={baseline_score:.5f}, "
        f"best_optimized={best_score:.5f}, delta={float(best_score - baseline_score):+.5f}."
    )
    if reference_artifact_dir is not None:
        _save_label_weight_hpo_report(report, reference_artifact_dir)
    return (
        elected_train.astype(np.float32, copy=False),
        elected_hard.astype(np.int8, copy=False),
        elected_weight.astype(np.float32, copy=False),
        report,
        elected_soft,
        elected_hard,
        elected_weight,
    )


def _save_label_weight_hpo_report(
    report: dict[str, Any],
    reference_artifact_dir: str | os.PathLike[str] | None,
) -> None:
    if reference_artifact_dir is None:
        return
    try:
        out_dir = Path(reference_artifact_dir) / "label_weight_hpo"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.json").write_text(
            json.dumps(_json_sanitize(report), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except Exception as exc:
        tprint(f"WARNING: failed to save LGBM label/sample-weight HPO report: {exc}")


def _fold_support_diagnostics(
    splitter: Any,
    y_split: np.ndarray,
    y_hard: np.ndarray,
    *,
    prefix: str,
) -> dict[str, float]:
    yh = np.asarray(y_hard, dtype=np.float64) >= 0.5
    fold_pos_train: list[float] = []
    fold_pos_valid: list[float] = []
    fold_neg_train: list[float] = []
    fold_neg_valid: list[float] = []
    try:
        for tr, va in splitter.split(np.zeros(len(y_split)), y_split):
            tr = np.asarray(tr, dtype=np.int32)
            va = np.asarray(va, dtype=np.int32)
            fold_pos_train.append(float(np.sum(yh[tr])))
            fold_pos_valid.append(float(np.sum(yh[va])))
            fold_neg_train.append(float(len(tr) - np.sum(yh[tr])))
            fold_neg_valid.append(float(len(va) - np.sum(yh[va])))
    except Exception:
        return {}
    if not fold_pos_train:
        return {}
    diag = {
        f"{prefix}_fold_count": float(len(fold_pos_train)),
        f"{prefix}_fold_train_pos_min": float(np.min(fold_pos_train)),
        f"{prefix}_fold_train_pos_mean": float(np.mean(fold_pos_train)),
        f"{prefix}_fold_valid_pos_min": float(np.min(fold_pos_valid)),
        f"{prefix}_fold_valid_pos_mean": float(np.mean(fold_pos_valid)),
        f"{prefix}_fold_train_neg_min": float(np.min(fold_neg_train)),
        f"{prefix}_fold_valid_neg_min": float(np.min(fold_neg_valid)),
    }
    tprint(
        "LGBM fold class-support diagnostics "
        f"{prefix}: folds={int(diag[f'{prefix}_fold_count'])}, "
        f"train_pos_min={diag[f'{prefix}_fold_train_pos_min']:.0f}, "
        f"valid_pos_min={diag[f'{prefix}_fold_valid_pos_min']:.0f}, "
        f"train_neg_min={diag[f'{prefix}_fold_train_neg_min']:.0f}, "
        f"valid_neg_min={diag[f'{prefix}_fold_valid_neg_min']:.0f}."
    )
    return diag


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
    if not (LGBM_META_DRIFT_FEATURES or LGBM_FINAL_OOF_BASE_ERROR_CONTEXT_FEATURES):
        return {}
    features = [str(c) for c in feature_names if str(c) in X_ref.columns][:LGBM_META_DRIFT_MAX_FEATURES]
    if not features:
        return {}
    n_ref = int(len(X_ref))
    ref_frame = X_ref
    if LGBM_META_DRIFT_MAX_ROWS > 0 and n_ref > LGBM_META_DRIFT_MAX_ROWS:
        idx = np.linspace(0, n_ref - 1, LGBM_META_DRIFT_MAX_ROWS, dtype=np.int32)
        ref_frame = X_ref.iloc[idx].reset_index(drop=True)
    ref = ref_frame.loc[:, features].astype(np.float32, copy=False)
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
        "source_rows": n_ref,
        "max_rows": int(LGBM_META_DRIFT_MAX_ROWS),
        "max_features": int(LGBM_META_DRIFT_MAX_FEATURES),
    }


def _feature_drift_values(
    X_current: pd.DataFrame,
    reference: dict[str, Any] | None,
) -> dict[str, float]:
    if not reference:
        return {
            "regime_centroid_similarity_train": 1.0,
            "feature_drift_psi_core": 0.0,
            "feature_drift_ks_core": 0.0,
            "feature_drift_cov_shift": 0.0,
        }
    ref_features = [str(c) for c in reference.get("feature_names", [])]
    features = [name for name in ref_features if name in X_current.columns]
    if not features:
        return {
            "regime_centroid_similarity_train": 0.0,
            "feature_drift_psi_core": 0.0,
            "feature_drift_ks_core": 0.0,
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
    ks_vals: list[float] = []
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
        ks = np.max(np.abs(np.cumsum(cur_props[:m]) - np.cumsum(ref_arr[:m])))
        if np.isfinite(ks):
            ks_vals.append(float(ks))
    psi_core = float(np.nanmean(psi_vals)) if psi_vals else 0.0
    ks_core = float(np.nanmean(ks_vals)) if ks_vals else 0.0
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
        "feature_drift_ks_core": ks_core,
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
        "feature_drift_ks_core",
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
    preset_best_params: Optional[dict[str, Any]] = None,
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
    selection_param_source = "base_grid"
    preset_depth = None
    preset_reg_lambda = None
    if preset_best_params:
        base_preset_params = _effective_lgbm_params(
            dict(preset_best_params),
            classifier=classifier,
        )
        preset_depth = max(1, int(base_preset_params.get("max_depth", 4) or 4))
        preset_reg_lambda = max(
            0.0,
            float(base_preset_params.get("reg_lambda", 5.0) or 0.0),
        )
        depth_values = list(
            dict.fromkeys(
                [
                    max(1, preset_depth - 1),
                    max(1, preset_depth + 1),
                ]
            )
        )
        reg_lambda_values = list(
            dict.fromkeys(
                [
                    max(0.0, preset_reg_lambda * 0.5),
                    max(0.0, preset_reg_lambda * 1.5),
                ]
            )
        )
        for depth in depth_values:
            for l2 in reg_lambda_values:
                cfg = dict(base_preset_params)
                cfg.update(
                    {
                        "max_depth": int(depth),
                        "num_leaves": int(2 ** int(depth)),
                        "reg_lambda": float(l2),
                    }
                )
                configs.append(cfg)
        selection_param_source = "native_preset_local_grid"
        tprint(
            "LGBM stability selection using native preset local grid: "
            f"preset_max_depth={preset_depth}, depth_values={depth_values}, "
            f"preset_reg_lambda={preset_reg_lambda:.4f}, "
            f"reg_lambda_values={[round(float(v), 4) for v in reg_lambda_values]}."
        )
    else:
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
                    f"train_rows={len(tr)}, valid_rows={len(va)}, features={p}, "
                    f"param_source={selection_param_source}, "
                    f"max_depth={int(cfg.get('max_depth', 0) or 0)}, "
                    f"reg_lambda={float(cfg.get('reg_lambda', 0.0) or 0.0):.4f}."
                )
                if preset_best_params:
                    params = _effective_lgbm_params(dict(cfg), classifier=classifier)
                    params["random_state"] = int(seed) + cfg_i * 1000 + fold_i
                else:
                    params = _base_lgbm_params(
                        int(seed) + cfg_i * 1000 + fold_i,
                        classifier=classifier,
                        overrides=cfg,
                    )
                params["n_estimators"] = int(LGBM_FEATURE_SELECTION_N_ESTIMATORS)
                params = _effective_lgbm_params(params, classifier=classifier)
                tprint(
                    "LGBM stability fit params: "
                    f"seed={seed}, config={cfg_i}/{len(configs)}, fold={fold_i}/{stability_splits}, "
                    f"n_estimators={int(params.get('n_estimators', 0) or 0)}, "
                    f"param_source={selection_param_source}."
                )
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
    agg_all["feature_selection_param_source"] = str(selection_param_source)
    if preset_depth is not None:
        agg_all["feature_selection_preset_max_depth"] = int(preset_depth)
    if preset_reg_lambda is not None:
        agg_all["feature_selection_preset_reg_lambda"] = float(preset_reg_lambda)
    agg_all["feature_selection_fit_configs"] = [
        {
            "max_depth": int(cfg.get("max_depth", 0) or 0),
            "num_leaves": int(cfg.get("num_leaves", 0) or 0),
            "reg_lambda": float(cfg.get("reg_lambda", 0.0) or 0.0),
        }
        for cfg in configs
    ]
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
    preset_best_params: Optional[dict[str, Any]] = None,
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
            preset_best_params=preset_best_params,
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
    timestamps: Any = None,
    assets: Any = None,
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
    splitter, y_split = _cv_splitter(
        y_metric,
        classifier,
        random_state,
        timestamps=timestamps,
        n_splits=n_splits,
    )
    fold_support_diag = _fold_support_diagnostics(
        splitter,
        y_split,
        y_metric,
        prefix="oof_cv",
    )
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
            early_stopping_rounds=LGBM_EARLY_STOPPING_ROUNDS,
        )
        pred = _predict_lgbm_raw(model, Xf.iloc[va].reset_index(drop=True), "classifier" if classifier else "regressor")
        oof[va] = pred
        fold_metrics = _metric_pack(y_metric[va], pred, classifier=classifier, groups=_groups_take(groups, va), returns=ret_arr[va])
        if fold_support_diag:
            fold_metrics.update(fold_support_diag)
            fold_metrics["oof_cv_fold_train_pos"] = float(np.sum(np.asarray(y_metric)[tr] >= 0.5)) if classifier else float("nan")
            fold_metrics["oof_cv_fold_valid_pos"] = float(np.sum(np.asarray(y_metric)[va] >= 0.5)) if classifier else float("nan")
            fold_metrics["oof_cv_fold_train_neg"] = float(len(tr) - np.sum(np.asarray(y_metric)[tr] >= 0.5)) if classifier else float("nan")
            fold_metrics["oof_cv_fold_valid_neg"] = float(len(va) - np.sum(np.asarray(y_metric)[va] >= 0.5)) if classifier else float("nan")
        metrics.append(fold_metrics)
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
    timestamps: Any = None,
    assets: Any = None,
    returns: Any = None,
    metric_y: np.ndarray | None = None,
    random_state: int,
    n_splits: int = LGBM_CV_SPLITS,
    raw_contrib_input_features: list[str] | None = None,
    raw_contrib_passthrough_features: list[str] | None = None,
    raw_contrib_transformed_feature_names: list[str] | None = None,
    ae_gmm_input_features: list[str] | None = None,
    ae_gmm_feature_names: list[str] | None = None,
    ae_gmm_context_feature_names: list[str] | None = None,
    ae_gmm_enabled: bool = False,
) -> tuple[np.ndarray, list[dict[str, float]], pd.DataFrame, pd.DataFrame]:
    t0 = time.perf_counter()
    raw_input_features = [str(c) for c in (raw_contrib_input_features or [])]
    raw_passthrough = [str(c) for c in (raw_contrib_passthrough_features or [])]
    raw_svd_features = [
        str(c)
        for c in (raw_contrib_transformed_feature_names or META_RAW_CONTRIB_SVD_FEATURE_NAMES)
    ]
    use_raw_contrib_transform = bool(raw_input_features)
    ae_input_features = [str(c) for c in (ae_gmm_input_features or [])]
    ae_feature_names = [
        str(c)
        for c in (
            ae_gmm_feature_names
            if ae_gmm_feature_names is not None
            else []
        )
    ]
    ae_context_feature_names = [
        str(c)
        for c in (
            ae_gmm_context_feature_names
            if ae_gmm_context_feature_names is not None
            else ae_feature_names
        )
    ]
    use_ae_gmm_transform = bool(
        ae_gmm_enabled and ae_input_features and (ae_feature_names or ae_context_feature_names)
    )
    base_model_features = ae_input_features if use_ae_gmm_transform else [str(c) for c in features]
    X_base = X.reset_index(drop=True)
    Xf = (
        X_base.reindex(columns=base_model_features, fill_value=0.0).astype(np.float32, copy=False)
        if not use_raw_contrib_transform
        else None
    )
    y_arr = np.asarray(y)
    y_metric = np.asarray(metric_y if metric_y is not None else y_arr)
    ret_arr = _as_returns(y_metric, returns)
    splitter, y_split = _cv_splitter(
        y_metric,
        classifier,
        random_state,
        timestamps=timestamps,
        n_splits=n_splits,
    )
    fold_support_diag = _fold_support_diagnostics(
        splitter,
        y_split,
        y_metric,
        prefix="final_oof_cv",
    )
    oof = np.full(len(y_arr), np.nan, dtype=np.float32)
    metrics: list[dict[str, float]] = []
    meta_features = pd.DataFrame(index=np.arange(len(y_arr)), columns=LGBM_META_FEATURE_NAMES, dtype=np.float32)
    raw_contrib_feature_map = raw_contrib_feature_mapping(features)
    raw_contrib_features = (
        pd.DataFrame(
            index=np.arange(len(y_arr)),
            columns=list(raw_contrib_feature_map.keys()),
            dtype=np.float32,
        )
        if LGBM_RAW_CONTRIB_OOF_EXPORT and raw_contrib_feature_map
        else pd.DataFrame(index=np.arange(len(y_arr)))
    )
    mode = "classifier" if classifier else "regressor"
    rank_cols = ["rank_bin_win_rate_oof", "rank_bin_lift_oof", "rank_bin_net_ret_oof", "rank_bin_se_oof"]
    tprint(
        "LGBM final OOF/meta CV started: "
        f"rows={len(y_arr)}, features={len(features)}, splits={n_splits}, "
        f"raw_contrib_transform={'yes' if use_raw_contrib_transform else 'no'}, "
        f"ae_gmm_transform={'yes' if use_ae_gmm_transform else 'no'}."
    )
    for fold_i, (tr, va) in enumerate(splitter.split(np.zeros(len(y_split)), y_split), start=1):
        fold_t0 = time.perf_counter()
        tprint(
            f"LGBM final OOF/meta fold {fold_i}/{n_splits} started: "
            f"train_rows={len(tr)}, valid_rows={len(va)}."
        )
        fold_params = dict(params)
        fold_params["random_state"] = int(random_state + fold_i * 1009)
        if use_raw_contrib_transform:
            fold_state = _fit_raw_contrib_input_state(
                X_base.iloc[tr],
                raw_input_features,
                random_state=int(random_state + fold_i * 1301),
            )
            X_tr = _raw_contrib_model_input_frame(
                X_base.iloc[tr].reset_index(drop=True),
                raw_passthrough,
                raw_input_features,
                fold_state,
                output_feature_names=raw_svd_features,
                index=pd.RangeIndex(len(tr)),
            ).reindex(columns=base_model_features, fill_value=0.0)
            X_va = _raw_contrib_model_input_frame(
                X_base.iloc[va].reset_index(drop=True),
                raw_passthrough,
                raw_input_features,
                fold_state,
                output_feature_names=raw_svd_features,
                index=pd.RangeIndex(len(va)),
            ).reindex(columns=base_model_features, fill_value=0.0)
        else:
            X_tr = Xf.iloc[tr].reset_index(drop=True)
            X_va = Xf.iloc[va].reset_index(drop=True)
        ae_state: dict[str, Any] | None = None
        if use_ae_gmm_transform:
            ae_state = _fit_ae_gmm_post_selection_state(
                X_tr,
                base_model_features,
                np.arange(len(X_tr), dtype=np.int32),
                y_metric=y_metric[tr],
                returns=ret_arr[tr],
                random_state=int(random_state + fold_i * 2221),
            )
            if ae_feature_names:
                X_tr = _append_ae_gmm_features_to_model_frame(
                    X_tr,
                    base_model_features,
                    ae_state,
                    features,
                    index=pd.RangeIndex(len(X_tr)),
                )
                X_va = _append_ae_gmm_features_to_model_frame(
                    X_va,
                    base_model_features,
                    ae_state,
                    features,
                    index=pd.RangeIndex(len(X_va)),
                )
            else:
                X_tr = X_tr.reindex(columns=features, fill_value=0.0).astype(np.float32, copy=False)
                X_va = X_va.reindex(columns=features, fill_value=0.0).astype(np.float32, copy=False)
        else:
            X_tr = X_tr.reindex(columns=features, fill_value=0.0).astype(np.float32, copy=False)
            X_va = X_va.reindex(columns=features, fill_value=0.0).astype(np.float32, copy=False)
        tprint(
            f"LGBM final OOF/meta fold {fold_i}/{n_splits}: input frames ready "
            f"train_shape={X_tr.shape}, valid_shape={X_va.shape}."
        )
        step_t0 = time.perf_counter()
        tprint(f"LGBM final OOF/meta fold {fold_i}/{n_splits}: fold model fit started.")
        model = _fit_lgbm_model(
            X_tr,
            y_arr[tr],
            sample_weight[tr],
            classifier=classifier,
            params=fold_params,
            X_valid=X_va,
            y_valid=y_arr[va],
            early_stopping_rounds=LGBM_EARLY_STOPPING_ROUNDS,
            attach_leaf_diagnostics=True,
        )
        tprint(
            f"LGBM final OOF/meta fold {fold_i}/{n_splits}: fold model fit complete "
            f"in {time.perf_counter() - step_t0:.1f}s."
        )
        step_t0 = time.perf_counter()
        tprint(f"LGBM final OOF/meta fold {fold_i}/{n_splits}: prediction started.")
        pred = _predict_lgbm_raw(model, X_va, mode)
        pred_train = _predict_lgbm_raw(model, X_tr, mode)
        tprint(
            f"LGBM final OOF/meta fold {fold_i}/{n_splits}: prediction complete "
            f"in {time.perf_counter() - step_t0:.1f}s."
        )
        oof[va] = pred
        fold_metrics = _metric_pack(y_metric[va], pred, classifier=classifier, groups=_groups_take(groups, va), returns=ret_arr[va])
        if fold_support_diag:
            fold_metrics.update(fold_support_diag)
            fold_metrics["final_oof_cv_fold_train_pos"] = float(np.sum(np.asarray(y_metric)[tr] >= 0.5)) if classifier else float("nan")
            fold_metrics["final_oof_cv_fold_valid_pos"] = float(np.sum(np.asarray(y_metric)[va] >= 0.5)) if classifier else float("nan")
            fold_metrics["final_oof_cv_fold_train_neg"] = float(len(tr) - np.sum(np.asarray(y_metric)[tr] >= 0.5)) if classifier else float("nan")
            fold_metrics["final_oof_cv_fold_valid_neg"] = float(len(va) - np.sum(np.asarray(y_metric)[va] >= 0.5)) if classifier else float("nan")
        metrics.append(fold_metrics)
        step_t0 = time.perf_counter()
        tprint(f"LGBM final OOF/meta fold {fold_i}/{n_splits}: rank-bin features started.")
        train_rank = _safe_rank_pct(pred_train)
        valid_rank = _rank_pct_against_reference(pred, pred_train)
        train_rank_stats = _fit_rank_bin_stats_oof(
            y_metric[tr],
            train_rank,
            classifier=classifier,
            returns=ret_arr[tr],
        )
        tree_count = _model_num_iterations(model)
        score_path_probs = None
        fold_meta = _lgbm_meta_features_from_predictions(
            pred,
            mode=mode,
            rank_pct=valid_rank,
            rank_bin_stats=train_rank_stats,
            model_count=1,
            tree_count=tree_count,
            score_path_probs=score_path_probs,
        )
        for frac in (0.10, 0.20, 0.30):
            pct = int(round(frac * 100))
            fold_meta[f"score_margin_top{pct}"] = _score_margin_against_reference(
                pred,
                pred_train,
                frac,
            )
        if use_ae_gmm_transform and ae_state is not None:
            X_va_ae_context = _append_ae_gmm_features_to_model_frame(
                X_va,
                base_model_features,
                ae_state,
                ae_context_feature_names,
                index=pd.RangeIndex(len(X_va)),
            )
            for col in ae_feature_names:
                if col in X_va.columns:
                    fold_meta[col] = X_va[col].to_numpy(dtype=np.float32, copy=False)
            for col in ae_context_feature_names:
                if col in X_va_ae_context.columns:
                    fold_meta[col] = X_va_ae_context[col].to_numpy(dtype=np.float32, copy=False)
        tprint(
            f"LGBM final OOF/meta fold {fold_i}/{n_splits}: rank-bin features complete "
            f"in {time.perf_counter() - step_t0:.1f}s."
        )
        if LGBM_META_SCORE_PATH_DIAGNOSTICS:
            score_t0 = time.perf_counter()
            tprint(
                f"LGBM final OOF/meta fold {fold_i}/{n_splits}: score-path diagnostics started "
                f"rows={len(va)}, max_trees={int(LGBM_META_SCORE_PATH_MAX_TREES or LGBM_META_LEAF_MAX_TREES)}."
            )
            _append_score_path_tree_diagnostics(
                fold_meta,
                [model],
                X_va,
                mode=mode,
                final_pred=pred,
            )
            tprint(
                f"LGBM final OOF/meta fold {fold_i}/{n_splits}: score-path diagnostics complete "
                f"in {time.perf_counter() - score_t0:.1f}s."
            )
        if (
            LGBM_META_LEAF_LITE_DIAGNOSTICS
            or LGBM_META_LEAF_DIAGNOSTICS
            or LGBM_META_LEAF_SUPPORT_DIAGNOSTICS
            or LGBM_META_LEAF_TARGET_DIAGNOSTICS
            or LGBM_META_LEAF_CENTROID_DIAGNOSTICS
        ):
            step_t0 = time.perf_counter()
            leaf_mode = (
                "full"
                if LGBM_META_LEAF_DIAGNOSTICS
                else (
                    "lite+support/target"
                    if (LGBM_META_LEAF_SUPPORT_DIAGNOSTICS or LGBM_META_LEAF_TARGET_DIAGNOSTICS)
                    else "lite"
                )
            )
            tprint(
                f"LGBM final OOF/meta fold {fold_i}/{n_splits}: leaf-{leaf_mode} diagnostics started "
                f"rows={len(va)}, max_trees={int(LGBM_META_LEAF_MAX_TREES)}."
            )
            _append_leaf_diagnostics(
                fold_meta,
                [model],
                X_va,
                prediction=pred,
                full_diagnostics=LGBM_META_LEAF_DIAGNOSTICS,
                support_diagnostics=LGBM_META_LEAF_SUPPORT_DIAGNOSTICS,
                target_diagnostics=LGBM_META_LEAF_TARGET_DIAGNOSTICS,
                centroid_diagnostics=LGBM_META_LEAF_CENTROID_DIAGNOSTICS,
            )
            tprint(
                f"LGBM final OOF/meta fold {fold_i}/{n_splits}: leaf-{leaf_mode} diagnostics complete "
                f"in {time.perf_counter() - step_t0:.1f}s."
            )
        if LGBM_META_CONTRIB_DIAGNOSTICS:
            step_t0 = time.perf_counter()
            tprint(
                f"LGBM final OOF/meta fold {fold_i}/{n_splits}: "
                f"{LGBM_META_CONTRIB_METHOD} contribution diagnostics started "
                f"rows={len(va)}, max_trees={int(LGBM_META_LEAF_MAX_TREES)}."
            )
            _append_contrib_diagnostics(
                fold_meta,
                [model],
                X_va,
                method=LGBM_META_CONTRIB_METHOD,
            )
            tprint(
                f"LGBM final OOF/meta fold {fold_i}/{n_splits}: "
                f"{LGBM_META_CONTRIB_METHOD} contribution diagnostics complete "
                f"in {time.perf_counter() - step_t0:.1f}s."
            )
        if LGBM_META_DRIFT_FEATURES:
            step_t0 = time.perf_counter()
            tprint(f"LGBM final OOF/meta fold {fold_i}/{n_splits}: capped drift features started.")
            gain_imp, split_imp = _feature_importances(model, len(features))
            drift_features = _top_cumulative_importance_feature_names(
                features,
                gain_imp,
                split_imp,
                cumulative_fraction=0.50,
            )[:LGBM_META_DRIFT_MAX_FEATURES]
            fold_drift_reference = _fit_feature_drift_reference(
                X_tr,
                drift_features,
            )
            _append_feature_drift_meta_features(
                fold_meta,
                X_va,
                fold_drift_reference,
            )
            tprint(
                f"LGBM final OOF/meta fold {fold_i}/{n_splits}: capped drift features complete "
                f"in {time.perf_counter() - step_t0:.1f}s, features={len(drift_features)}."
            )
        if (
            LGBM_FINAL_OOF_CONTRIB_CONTEXT_FEATURES
            or LGBM_FINAL_OOF_RAW_STATE_CONTEXT_FEATURES
            or LGBM_FINAL_OOF_BASE_ERROR_CONTEXT_FEATURES
        ):
            fold_meta_train = None
            if LGBM_FINAL_OOF_BASE_ERROR_CONTEXT_FEATURES:
                fold_meta_train = _lgbm_meta_features_from_predictions(
                    pred_train,
                    mode=mode,
                    rank_pct=train_rank,
                    rank_bin_stats=train_rank_stats,
                    model_count=1,
                    tree_count=tree_count,
                )
                for frac in (0.10, 0.20, 0.30):
                    pct = int(round(frac * 100))
                    fold_meta_train[f"score_margin_top{pct}"] = _score_margin_against_reference(
                        pred_train,
                        pred_train,
                        frac,
                    )
            train_timestamps = _take_aligned(timestamps, tr, len(y_arr))
            valid_timestamps = _take_aligned(timestamps, va, len(y_arr))
            train_assets = _take_aligned(assets, tr, len(y_arr))
            valid_assets = _take_aligned(assets, va, len(y_arr))
            step_t0 = time.perf_counter()
            tprint(
                f"LGBM final OOF/meta fold {fold_i}/{n_splits}: raw/contrib/base-error context started "
                f"train_rows={len(tr)}, valid_rows={len(va)}."
            )
            contrib_state, raw_state = _fit_lgbm_archetype_states(
                [model],
                X_tr,
                list(features),
                timestamps=train_timestamps,
                assets=train_assets,
                random_state=int(random_state + fold_i * 1543),
            )
            if fold_meta_train is not None:
                _append_archetype_meta_features(
                    fold_meta_train,
                    [model],
                    X_tr,
                    contrib_state=contrib_state,
                    raw_state=raw_state,
                    timestamps=train_timestamps,
                    assets=train_assets,
                )
            _append_archetype_meta_features(
                fold_meta,
                [model],
                X_va,
                contrib_state=contrib_state,
                raw_state=raw_state,
                timestamps=valid_timestamps,
                assets=valid_assets,
            )
            if LGBM_FINAL_OOF_BASE_ERROR_CONTEXT_FEATURES and fold_meta_train is not None:
                gain_imp, split_imp = _feature_importances(model, len(features))
                drift_features = _top_cumulative_importance_feature_names(
                    features,
                    gain_imp,
                    split_imp,
                    cumulative_fraction=0.50,
                )[:LGBM_META_DRIFT_MAX_FEATURES]
                fold_drift_reference = _fit_feature_drift_reference(
                    X_tr,
                    drift_features,
                )
                _append_feature_drift_meta_features(
                    fold_meta_train,
                    X_tr,
                    fold_drift_reference,
                )
                _append_feature_drift_meta_features(
                    fold_meta,
                    X_va,
                    fold_drift_reference,
                )
                fold_base_error_state = _fit_base_error_archetype_state_from_meta(
                    fold_meta_train,
                    y_metric[tr],
                    pred_train,
                    classifier=classifier,
                    random_state=int(random_state + fold_i * 1999),
                )
                _append_base_error_archetype_features(fold_meta, fold_base_error_state)
            tprint(
                f"LGBM final OOF/meta fold {fold_i}/{n_splits}: raw/contrib/base-error context complete "
                f"in {time.perf_counter() - step_t0:.1f}s."
            )
        else:
            tprint(
                f"LGBM final OOF/meta fold {fold_i}/{n_splits}: "
                "raw/contrib/base-error context skipped."
            )
        step_t0 = time.perf_counter()
        tprint(f"LGBM final OOF/meta fold {fold_i}/{n_splits}: meta frame assignment started.")
        meta_features.iloc[va] = fold_meta.reindex(
            columns=LGBM_META_FEATURE_NAMES,
            fill_value=0.0,
        ).to_numpy(dtype=np.float32)
        tprint(
            f"LGBM final OOF/meta fold {fold_i}/{n_splits}: meta frame assignment complete "
            f"in {time.perf_counter() - step_t0:.1f}s."
        )
        if not raw_contrib_features.empty:
            step_t0 = time.perf_counter()
            tprint(
                f"LGBM final OOF/meta fold {fold_i}/{n_splits}: raw contribution OOF export started "
                f"rows={len(va)}."
            )
            fold_contrib = _mean_contrib_matrix([model], X_va)
            if fold_contrib is not None and fold_contrib.size:
                raw_frame = raw_contrib_frame(
                    fold_contrib,
                    features,
                    index=np.arange(len(va)),
                )
                raw_frame = raw_frame.rename(
                    columns={
                        generated: exported
                        for exported, generated in zip(
                            raw_contrib_feature_map.keys(),
                            raw_frame.columns,
                        )
                    }
                )
                raw_contrib_features.iloc[va] = raw_frame.reindex(
                    columns=raw_contrib_features.columns,
                    fill_value=0.0,
                ).to_numpy(dtype=np.float32)
            else:
                raw_contrib_features.iloc[va] = 0.0
            tprint(
                f"LGBM final OOF/meta fold {fold_i}/{n_splits}: raw contribution OOF export complete "
                f"in {time.perf_counter() - step_t0:.1f}s."
            )
        tprint(
            f"LGBM final OOF/meta fold {fold_i}/{n_splits} complete: "
            f"elapsed={time.perf_counter() - fold_t0:.1f}s."
        )
    fill = float(np.nanmean(oof)) if np.isfinite(oof).any() else float(np.mean(y_arr))
    oof = np.nan_to_num(oof, nan=fill).astype(np.float32)
    full_rank = _safe_rank_pct(oof)
    meta_features["lgbm_prob"] = oof
    fold_safe_fallbacks: dict[str, np.ndarray] = {
        "rank_pct": full_rank,
        "score_margin_top10": _score_margin(oof, 0.10),
        "score_margin_top20": _score_margin(oof, 0.20),
        "score_margin_top30": _score_margin(oof, 0.30),
        "rank_margin_top10": (full_rank - 0.90).astype(np.float32),
        "rank_margin_top20": (full_rank - 0.80).astype(np.float32),
        "rank_margin_top30": (full_rank - 0.70).astype(np.float32),
    }
    for col, fallback in fold_safe_fallbacks.items():
        existing = pd.to_numeric(meta_features.get(col), errors="coerce") if col in meta_features.columns else None
        if existing is None:
            meta_features[col] = fallback
            continue
        arr = existing.to_numpy(dtype=np.float32, copy=False)
        meta_features[col] = np.where(np.isfinite(arr), arr, fallback).astype(np.float32)
    for col in rank_cols:
        if col not in meta_features.columns:
            meta_features[col] = np.zeros(len(meta_features), dtype=np.float32)
            continue
        arr = pd.to_numeric(meta_features[col], errors="coerce").to_numpy(dtype=np.float32, copy=False)
        meta_features[col] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    tprint(f"LGBM final OOF/meta CV complete in {time.perf_counter() - t0:.1f}s.")
    raw_contrib_features = raw_contrib_features.fillna(0.0).astype(np.float32, copy=False)
    return (
        oof,
        metrics,
        meta_features.reindex(columns=LGBM_META_FEATURE_NAMES, fill_value=0.0).astype(np.float32),
        raw_contrib_features,
    )


def _oof_distilled_sample_weights_lgbm(
    X: pd.DataFrame,
    y: np.ndarray,
    base_weight: np.ndarray,
    features: list[str],
    *,
    classifier: bool,
    params: dict[str, Any],
    groups: Any = None,
    timestamps: Any = None,
    returns: Any = None,
    metric_y: np.ndarray | None = None,
    random_state: int,
    passes: int,
    label: str,
    objective_mode: str | None = "train_base",
    cfg: dict[str, Any] | None = None,
    specialist_similarity: np.ndarray | None = None,
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
            timestamps=timestamps,
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
        distill = _recency_shrink_weight_towards_one(
            distill,
            timestamps,
            objective_mode=objective_mode,
            cfg=cfg,
        )
        fp_weight = _recency_shrink_weight_towards_one(
            fp_weight,
            timestamps,
            objective_mode=objective_mode,
            cfg=cfg,
        )
        distill = _regime_specialist_shrink_weight_towards_one(
            distill,
            specialist_similarity,
            cfg=cfg,
        )
        fp_weight = _regime_specialist_shrink_weight_towards_one(
            fp_weight,
            specialist_similarity,
            cfg=cfg,
        )
        distill, fp_weight = apply_distillation_recipe(
            distill,
            fp_weight,
            y_metric=y_metric,
            pred=last_oof,
            returns=returns,
            timestamps=timestamps,
            objective_mode=objective_mode,
            cfg=cfg,
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


def _final_fit_leaf_floor(
    params: dict[str, Any],
    *,
    fit_rows: int,
    hpo_rows: int,
    hpo_effective_rows: float,
    final_effective_rows: float,
    objective_mode: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    out = dict(params)
    rows = max(0, int(fit_rows))
    hpo_n = max(0, int(hpo_rows))
    hpo_eff = max(1.0, float(hpo_effective_rows or 0.0))
    final_eff = max(1.0, float(final_effective_rows or 0.0))
    mode = _normalize_objective_mode(objective_mode)
    if rows <= 0:
        return out, {"enabled": False, "reason": "no_fit_rows"}
    if mode == "train_meta":
        abs_floor = int(LGBM_FINAL_MIN_CHILD_SAMPLES_META_ABS)
    else:
        abs_floor = int(LGBM_FINAL_MIN_CHILD_SAMPLES_BASE_ABS)
    before = int(out.get("min_child_samples", 0) or 0)
    if before <= 0:
        return out, {
            "enabled": False,
            "reason": "missing_min_child_samples",
            "fit_rows": int(rows),
            "hpo_rows": int(hpo_n),
            "hpo_effective_rows": float(hpo_eff),
            "final_effective_rows": float(final_eff),
            "objective_mode": str(mode),
        }
    alpha = float(LGBM_FINAL_MIN_CHILD_TRANSFER_ALPHA)
    ratio = final_eff / hpo_eff
    transfer_floor = int(np.ceil(float(before) * (ratio ** alpha)))
    pct_cap = int(np.ceil(float(rows) * float(LGBM_FINAL_MIN_CHILD_SAMPLES_CAP_PCT)))
    abs_cap = int(LGBM_FINAL_MIN_CHILD_SAMPLES_CAP)
    if pct_cap > 0 and abs_cap > 0:
        cap = min(pct_cap, abs_cap)
    else:
        cap = max(pct_cap, abs_cap)
    uncapped_after = max(before, abs_floor, transfer_floor)
    after = int(uncapped_after)
    if cap > 0:
        after = max(before, min(after, cap))
    out["min_child_samples"] = int(after)
    return out, {
        "enabled": bool(after != before),
        "before": int(before),
        "after": int(after),
        "transfer_floor": int(transfer_floor),
        "uncapped_after": int(uncapped_after),
        "fit_rows": int(rows),
        "hpo_rows": int(hpo_n),
        "hpo_effective_rows": float(hpo_eff),
        "final_effective_rows": float(final_eff),
        "effective_ratio": float(ratio),
        "alpha": float(alpha),
        "abs_floor": int(abs_floor),
        "cap_pct": float(LGBM_FINAL_MIN_CHILD_SAMPLES_CAP_PCT),
        "cap": int(cap),
        "objective_mode": str(mode),
    }


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
    cfg: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    t0 = time.perf_counter()
    try:
        import optuna
        from optuna.pruners import SuccessiveHalvingPruner
        from optuna.trial import TrialState
    except Exception as exc:
        tprint(f"LGBM HPO skipped, Optuna unavailable ({exc}).")
        params = _effective_lgbm_params(_default_hpo_params(random_state, classifier), classifier=classifier)
        return params, {"hpo_available": False, "hpo_best_value": np.nan, "hpo_objective_mode": _normalize_objective_mode(objective_mode)}
    y_arr = np.asarray(y)
    y_metric = np.asarray(metric_y if metric_y is not None else y_arr)
    hpo_overrides = cfg.get("lgbm_hpo_overrides", {}) if isinstance(cfg, dict) else {}
    if not isinstance(hpo_overrides, dict):
        hpo_overrides = {}
    max_depth_upper = int(
        np.clip(
            int(hpo_overrides.get("max_depth_max", LGBM_HPO_MAX_DEPTH_MAX)),
            3,
            int(LGBM_HPO_MAX_DEPTH_MAX),
        )
    )
    min_child_pct_min = float(
        hpo_overrides.get("min_child_samples_pct_min", 0.02)
    )
    min_child_pct_max = float(
        hpo_overrides.get("min_child_samples_pct_max", 0.07)
    )
    min_child_pct_min = float(np.clip(min_child_pct_min, 0.001, 0.50))
    min_child_pct_max = float(
        np.clip(max(min_child_pct_max, min_child_pct_min), min_child_pct_min, 0.50)
    )
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
        "fold_mode=interleaved_spread, pruner=successive_halving, cegb_enabled=False, "
        f"max_depth_range=[3,{int(max_depth_upper)}], "
        f"subsample_range=[{float(LGBM_HPO_SUBSAMPLE_MIN):.2f},{float(LGBM_HPO_SUBSAMPLE_MAX):.2f}], "
        f"min_child_weight_range=[{float(LGBM_HPO_MIN_CHILD_WEIGHT_MIN):.1f},{float(LGBM_HPO_MIN_CHILD_WEIGHT_MAX):.1f}], "
        f"min_child_samples_pct_range=[{float(min_child_pct_min):.3f},{float(min_child_pct_max):.3f}], "
        f"path_smooth_max={float(LGBM_HPO_PATH_SMOOTH_MAX):.3g}, "
        f"final_min_estimators={int(LGBM_HPO_FINAL_MIN_ESTIMATORS)}, "
        f"n_estimators_cap={int(LGBM_N_ESTIMATORS_CAP)}, "
        f"early_stopping_rounds={int(LGBM_EARLY_STOPPING_ROUNDS)}."
    )

    def objective(trial: Any) -> float:
        trial_t0 = time.perf_counter()
        depth = trial.suggest_int("max_depth", 3, int(max_depth_upper))
        subsample = trial.suggest_float(
            "subsample",
            float(LGBM_HPO_SUBSAMPLE_MIN),
            float(LGBM_HPO_SUBSAMPLE_MAX),
        )
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
                "min_child_samples": max(
                    2,
                    int(
                        trial.suggest_float(
                            "min_child_samples_pct",
                            float(min_child_pct_min),
                            float(min_child_pct_max),
                        )
                        * len(y_sub)
                    ),
                ),
                "min_child_weight": trial.suggest_float(
                    "min_child_weight",
                    float(LGBM_HPO_MIN_CHILD_WEIGHT_MIN),
                    float(LGBM_HPO_MIN_CHILD_WEIGHT_MAX),
                ),
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
                early_stopping_rounds=LGBM_EARLY_STOPPING_ROUNDS,
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

    study = optuna.create_study(
        direction="maximize",
        pruner=SuccessiveHalvingPruner(
            min_resource=1,
            reduction_factor=2,
            min_early_stopping_rate=0,
        ),
    )
    study.optimize(objective, n_trials=max(0, int(max_trials)), callbacks=[early_stop_callback], n_jobs=1, show_progress_bar=False)
    complete = [t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None]
    if not complete:
        params = _effective_lgbm_params(_default_hpo_params(random_state, classifier), classifier=classifier)
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
    depth = min(int(best.params.get("max_depth", 4)), int(max_depth_upper))
    best_params_raw = _base_lgbm_params(
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
        best_params_raw["feature_fraction_bynode"] = float(
            best.params.get("feature_fraction_bynode", 1.0)
        )
        best_params_raw["max_delta_step"] = float(best.params.get("max_delta_step", 0.0))
    best_params = _effective_lgbm_params(best_params_raw, classifier=classifier)
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
            "hpo_final_n_estimators_raw": int(final_n_estimators),
            "hpo_final_n_estimators": int(best_params.get("n_estimators", final_n_estimators)),
            "hpo_n_estimators_cap": int(LGBM_N_ESTIMATORS_CAP),
            "hpo_lgbm_early_stopping_rounds": int(LGBM_EARLY_STOPPING_ROUNDS),
            "hpo_early_stop_patience_trials": int(patience),
            "hpo_pruner": "successive_halving",
            "hpo_search_space": {
                "max_depth": [3, int(max_depth_upper)],
                "subsample": [
                    float(LGBM_HPO_SUBSAMPLE_MIN),
                    float(LGBM_HPO_SUBSAMPLE_MAX),
                ],
                "min_child_weight": [
                    float(LGBM_HPO_MIN_CHILD_WEIGHT_MIN),
                    float(LGBM_HPO_MIN_CHILD_WEIGHT_MAX),
                ],
                "path_smooth": [0.0, float(LGBM_HPO_PATH_SMOOTH_MAX)],
                "min_child_samples_pct": [
                    float(min_child_pct_min),
                    float(min_child_pct_max),
                ],
            },
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
    cfg: dict[str, Any] | None = None,
    label_context: dict[str, Any] | None = None,
    assessment_X: Any = None,
    assessment_timestamps: Any = None,
    assessment_assets: Any = None,
) -> Optional[dict[str, Any]]:
    objective_mode = _normalize_objective_mode(hpo_objective_mode)
    distill_passes = _distillation_passes_for_objective(objective_mode)
    tprint(f"LGBM stability candidate training started (objective={objective_mode}).")
    t0 = time.perf_counter()
    classifier = mode == "classifier"
    X_raw_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    X_raw_df.columns = [str(c) for c in X_raw_df.columns]
    X_df = _frame(X)
    y_arr = _coerce_target(y, classifier, allow_soft_labels=bool(LGBM_TRUE_SOFT_LABELS))
    y_metric = _coerce_target(hard_labels, classifier) if hard_labels is not None else _coerce_target(y, classifier)
    y_hard_diag = _coerce_target(hard_labels, classifier) if hard_labels is not None else _coerce_target(y, classifier)
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
    preset_features = [str(c) for c in (preset_feature_names or []) if str(c).strip()]
    regime_score_feature_diag: dict[str, Any] = {"enabled": False, "reason": "disabled"}
    if preset_features:
        X_df, preset_features, regime_score_feature_diag = _append_lgbm_regime_score_features(
            X_df,
            preset_features,
            timestamps=timestamps,
            assets=assets,
            objective_mode=objective_mode,
            cfg=cfg,
            random_state=random_state + 17431,
            label="candidate",
        )
    tprint(
        "LGBM candidate input: "
        f"rows={n}, features={X_df.shape[1]}, classifier={classifier}, "
        f"sample_weight={'yes' if sample_weight is not None else 'no'}, "
        f"returns={'yes' if returns is not None else 'no'}."
    )
    sw = np.ones(n, dtype=np.float32) if sample_weight is None else np.asarray(sample_weight, dtype=np.float32)
    sw, _ = _normalize_weights(sw)
    sw, recency_applied = _apply_recency_sample_weight(
        sw,
        timestamps,
        objective_mode=objective_mode,
        cfg=cfg,
    )
    if recency_applied:
        _active_recency_hpo = active_recency_hpo_config(cfg, objective_mode)
        if _active_recency_hpo:
            tprint(
                "LGBM candidate recency-HPO sample weighting enabled: "
                f"objective={objective_mode}, "
                f"half_life_days={float(_active_recency_hpo['half_life_days']):.1f}, "
                f"composite_weight={float(_active_recency_hpo['composite_weight']):.3f}, "
                "legacy_recency=disabled, "
                "distillation_multipliers_shrink_to_one=yes."
            )
        else:
            tprint(
                "LGBM candidate recency sample weighting enabled: "
                f"objective={objective_mode}, half_life_days={_recency_half_life_days(objective_mode):.1f}, "
                "distillation_multipliers_shrink_to_one=yes."
            )
    label_diag = _log_label_weight_diagnostics(
        y_arr,
        y_hard_diag,
        sw,
        label=f"candidate_{objective_mode}_pre_rebalance",
    )
    sw, rebalance_diag = _rebalance_effective_class_mass(
        y_arr,
        y_hard_diag,
        sw,
        label=f"candidate_{objective_mode}",
    )
    if bool(rebalance_diag.get(f"candidate_{objective_mode}_class_rebalance_applied", 0.0)):
        label_diag.update(
            _log_label_weight_diagnostics(
                y_arr,
                y_hard_diag,
                sw,
                label=f"candidate_{objective_mode}_post_rebalance",
            )
        )
    label_diag.update(rebalance_diag)
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
    select_rows_before_cap = int(len(np.asarray(stage_indices.get("lgbm_select", []), dtype=np.int32)))
    stability_select_cap = _lgbm_stability_selection_cap(select_rows_before_cap)
    tprint(
        "LGBM stability selection row cap: "
        f"select_rows={select_rows_before_cap}, cap={stability_select_cap}, "
        f"race_max_rows={int(LGBM_RACE_MAX_ROWS)}."
    )
    stage_indices = _cap_stage_and_move_unused_to_fit_oof(stage_indices, y_metric, stage_key="lgbm_select", cap=stability_select_cap, random_state=random_state + 1701, classifier=classifier, spread=True)
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
            max_n=_lgbm_stability_selection_cap(len(fallback_pool)),
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
            preset_best_params=preset_best_params,
        )
    if not selected_features:
        tprint("LGBM candidate rejected: no selected features.")
        return None
    tprint(
        "LGBM candidate selected features: "
        f"{len(selected_features)} after {len(history)} prune rounds; "
        f"preview={selected_features[:10]}."
    )
    _candidate_guard_cols = [c for c in selected_features if c in X_select.columns]
    _native_preset_selected_feature_variance_guard(
        X_select.iloc[:, [X_select.columns.get_loc(c) for c in _candidate_guard_cols]],
        selected_features,
        cfg=cfg,
        label=f"candidate_{objective_mode}",
        preset_source=preset_source,
    )
    base_params = _effective_lgbm_params(
        dict(preset_best_params or _default_hpo_params(random_state + 401, classifier)),
        classifier=classifier,
    )
    if preset_best_params:
        tprint("LGBM candidate using native preset best_params; HPO is skipped for base preset candidate scoring.")
    if _lgbm_regime_specialist_should_build_bundle(cfg, objective_mode):
        assessment_X_df, assessment_ts, assessment_asset_values = _lgbm_regime_specialist_assessment_inputs(
            assessment_X,
            assessment_timestamps,
            assessment_assets,
            label_context=label_context,
        )
    else:
        assessment_X_df, assessment_ts, assessment_asset_values = None, None, None
    regime_specialist_bundle = _build_lgbm_regime_specialist_bundle(
        X_df,
        selected_features,
        timestamps=timestamps,
        assets=assets,
        assessment_X_df=assessment_X_df,
        assessment_timestamps=assessment_ts,
        assessment_assets=assessment_asset_values,
        objective_mode=objective_mode,
        cfg=cfg,
        random_state=random_state + 1549,
        label="candidate",
    )
    candidate_train_idx = race_idx[select_local]
    sw_select, regime_specialist_apply_diag = _apply_lgbm_regime_specialist_weights(
        sw_select,
        regime_specialist_bundle,
        idx=candidate_train_idx,
    )
    if bool(regime_specialist_apply_diag.get("applied", False)):
        tprint(
            "LGBM candidate regime specialist sample weighting applied: "
            f"objective={objective_mode}, "
            f"ess={float(regime_specialist_apply_diag.get('effective_sample_size', float('nan'))):.1f}."
        )
    specialist_similarity_select = _lgbm_regime_specialist_similarity_for_idx(
        regime_specialist_bundle,
        idx=candidate_train_idx,
    )
    final_weights, _ = _oof_distilled_sample_weights_lgbm(
        X_select,
        y_select,
        sw_select,
        selected_features,
        classifier=classifier,
        params=base_params,
        groups=select_groups,
        timestamps=_take_aligned(timestamps, race_idx[select_local], n),
        returns=ret_select,
        metric_y=y_metric_select,
        random_state=random_state + 409,
        passes=distill_passes,
        label="candidate",
        objective_mode=objective_mode,
        cfg=cfg,
        specialist_similarity=specialist_similarity_select,
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
    metrics.update(
        _vol_normalized_tp_sl_precision_metrics(
            eval_pred,
            _label_context_take(label_context, race_idx[eval_local], n),
        )
    )
    metrics.update(
        _lgbm_regime_specialist_current_metrics(
            y_metric_eval,
            eval_pred,
            regime_specialist_bundle,
            classifier=classifier,
            groups=eval_groups,
            returns=ret_eval,
            idx=race_idx[eval_local],
            label_context=label_context,
            label_context_total_rows=n,
        )
    )
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
    metrics.update(_lgbm_regime_score_feature_metric_summary(regime_score_feature_diag))
    metrics.update(regime_specialist_bundle.get("metrics", {}))
    metrics["regime_specialist_apply_reason"] = str(
        regime_specialist_apply_diag.get("reason", "")
    )
    if bool(regime_specialist_apply_diag.get("applied", False)):
        metrics["regime_specialist_applied_effective_sample_size"] = float(
            regime_specialist_apply_diag.get("effective_sample_size", float("nan"))
        )
    if preset_features:
        metrics["feature_selection_source"] = "native_preset"
        metrics["native_preset_source"] = str(preset_source or "")
        metrics["native_preset_hpo_reused"] = bool(preset_best_params)
    _active_recency_hpo_metrics = active_recency_hpo_config(cfg, objective_mode)
    metrics["recency_weighting_enabled"] = bool(LGBM_RECENCY_WEIGHTING) or bool(
        _active_recency_hpo_metrics
    )
    metrics["recency_sample_weight_applied"] = bool(recency_applied)
    metrics["recency_weighting_scheme"] = (
        "composite_hpo"
        if _active_recency_hpo_metrics
        else ("legacy_exp" if bool(LGBM_RECENCY_WEIGHTING) else "none")
    )
    metrics["recency_half_life_days"] = float(
        _active_recency_hpo_metrics["half_life_days"]
        if _active_recency_hpo_metrics
        else _recency_half_life_days(objective_mode)
    )
    if _active_recency_hpo_metrics:
        metrics["recency_hpo_composite_weight"] = float(
            _active_recency_hpo_metrics["composite_weight"]
        )
        metrics["recency_hpo_source"] = str(
            _active_recency_hpo_metrics.get("source", "")
        )
        metrics["recency_hpo_legacy_recency_disabled"] = True
    metrics["true_soft_labels_enabled"] = bool(LGBM_TRUE_SOFT_LABELS)
    metrics["effective_class_rebalance_enabled"] = bool(LGBM_REBALANCE_EFFECTIVE_CLASSES)
    metrics.update(label_diag)
    metrics["cv_mode"] = (
        "interleaved_spread"
        if LGBM_CV_MODE in {"interleaved", "interleaved_spread"}
        else (
            "purged_time"
            if (bool(LGBM_PURGED_CV) or LGBM_CV_MODE == "purged_time")
            else "shuffled"
        )
    )
    metrics["cv_splits"] = int(LGBM_CV_SPLITS)
    metrics["purge_hours"] = float(
        LGBM_PURGE_HOURS
        if (bool(LGBM_PURGED_CV) or LGBM_CV_MODE == "purged_time")
        else 0.0
    )
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
    candidate_model = None
    if LGBM_OPTUNA_CANDIDATE_ONLY:
        fill_oof = float(np.nanmean(oof_full)) if np.isfinite(oof_full).any() else fill
        oof_filled = np.nan_to_num(oof_full, nan=fill_oof).astype(np.float32)
        rank_pct = _safe_rank_pct(oof_filled)
        meta_oof_features = pd.DataFrame(index=np.arange(n), columns=LGBM_META_FEATURE_NAMES, dtype=np.float32)
        meta_oof_features["lgbm_prob"] = oof_filled
        meta_oof_features["rank_pct"] = rank_pct
        candidate_model = LGBMStabilityModel(mode=mode)
        candidate_model.selected_features = list(selected_features)
        candidate_model.input_feature_names = list(selected_features)
        candidate_model.oof_probs = oof_full.astype(np.float32)
        candidate_model.meta_oof_features = meta_oof_features.reindex(
            columns=candidate_model.meta_feature_names,
            fill_value=0.0,
        ).astype(np.float32)
        candidate_model.rank_bin_stats_oof = _fit_rank_bin_stats_oof(
            y_metric,
            rank_pct,
            classifier=classifier,
            returns=ret_arr,
        )
        candidate_model.metrics = dict(metrics)
        candidate_model.metrics["optuna_candidate_only"] = True
        candidate_model.metrics["final_fit_train_rows"] = 0
        candidate_model.metrics["final_fit_train_rows_total"] = int(n)
        candidate_model.metrics["final_model_count"] = 0
        candidate_model.metrics["final_ensemble_sequential_distillation"] = False
        candidate_model.pruning_history = list(history)
        tprint(
            "LGBM Optuna candidate-only model materialized; "
            f"oof_finite={int(np.isfinite(oof_full).sum())}/{len(oof_full)}, "
            f"features={len(selected_features)}."
        )
    return {
        "model": candidate_model,
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
    preset_label_weight_hpo_report: Optional[dict[str, Any]] = None,
    cfg: dict[str, Any] | None = None,
    label_context: dict[str, Any] | None = None,
    assessment_X: Any = None,
    assessment_timestamps: Any = None,
    assessment_assets: Any = None,
) -> Optional[LGBMStabilityModel]:
    t0 = time.perf_counter()
    objective_mode = _normalize_objective_mode(hpo_objective_mode)
    distill_passes = _distillation_passes_for_objective(objective_mode)
    classifier = mode == "classifier"
    X_df = _frame(X)
    y_arr = _coerce_target(y, classifier, allow_soft_labels=bool(LGBM_TRUE_SOFT_LABELS))
    y_metric = _coerce_target(hard_labels, classifier) if hard_labels is not None else _coerce_target(y, classifier)
    y_hard_diag = _coerce_target(hard_labels, classifier) if hard_labels is not None else _coerce_target(y, classifier)
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
    X_df, selected_features, regime_score_feature_diag = _append_lgbm_regime_score_features(
        X_df,
        selected_features,
        timestamps=timestamps,
        assets=assets,
        objective_mode=objective_mode,
        cfg=cfg,
        random_state=random_state + 17431,
        label="final",
    )
    for col in selected_features:
        if col not in X_df.columns:
            X_df[col] = 0.0
    sw = np.ones(n, dtype=np.float32) if sample_weight is None else np.asarray(sample_weight, dtype=np.float32)
    sw, _ = _normalize_weights(sw)
    sw, recency_applied = _apply_recency_sample_weight(
        sw,
        timestamps,
        objective_mode=objective_mode,
        cfg=cfg,
    )
    if recency_applied:
        _active_recency_hpo = active_recency_hpo_config(cfg, objective_mode)
        if _active_recency_hpo:
            tprint(
                "LGBM recency-HPO sample weighting enabled: "
                f"objective={objective_mode}, "
                f"half_life_days={float(_active_recency_hpo['half_life_days']):.1f}, "
                f"composite_weight={float(_active_recency_hpo['composite_weight']):.3f}, "
                "legacy_recency=disabled, "
                "distillation_multipliers_shrink_to_one=yes."
            )
        else:
            tprint(
                "LGBM recency sample weighting enabled: "
                f"objective={objective_mode}, half_life_days={_recency_half_life_days(objective_mode):.1f}, "
                "distillation_multipliers_shrink_to_one=yes."
            )
    label_diag = _log_label_weight_diagnostics(
        y_arr,
        y_hard_diag,
        sw,
        label=f"final_{objective_mode}_pre_rebalance",
    )
    sw, rebalance_diag = _rebalance_effective_class_mass(
        y_arr,
        y_hard_diag,
        sw,
        label=f"final_{objective_mode}",
    )
    if bool(rebalance_diag.get(f"final_{objective_mode}_class_rebalance_applied", 0.0)):
        label_diag.update(
            _log_label_weight_diagnostics(
                y_arr,
                y_hard_diag,
                sw,
                label=f"final_{objective_mode}_post_rebalance",
            )
        )
    label_diag.update(rebalance_diag)
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
    _final_guard_cols = [c for c in selected_features if c in X_df.columns]
    _native_preset_selected_feature_variance_guard(
        X_df.iloc[fit_idx, [X_df.columns.get_loc(c) for c in _final_guard_cols]],
        selected_features,
        cfg=cfg,
        label=f"final_{objective_mode}",
        preset_source=preset_source,
    )
    input_selected_features = list(selected_features)
    raw_contrib_input_features = _raw_contrib_input_columns(input_selected_features)
    raw_contrib_passthrough_features = _raw_contrib_passthrough_columns(input_selected_features)
    raw_contrib_transformed_feature_names: list[str] = []
    raw_contrib_input_state: ContribArchetypeState | None = None
    X_model_df = X_df
    if raw_contrib_input_features:
        raw_contrib_transformed_feature_names = list(META_RAW_CONTRIB_SVD_FEATURE_NAMES)
        raw_contrib_input_state = _fit_raw_contrib_input_state(
            X_df.iloc[fit_idx],
            raw_contrib_input_features,
            random_state=random_state + 88121,
        )
        X_model_df = _raw_contrib_model_input_frame(
            X_df,
            raw_contrib_passthrough_features,
            raw_contrib_input_features,
            raw_contrib_input_state,
            output_feature_names=raw_contrib_transformed_feature_names,
            index=X_df.index,
        )
        selected_features = list(
            dict.fromkeys(
                raw_contrib_passthrough_features + raw_contrib_transformed_feature_names
            )
        )
        for col in selected_features:
            if col not in X_model_df.columns:
                X_model_df[col] = 0.0
        tprint(
            "LGBM raw contribution input transform enabled: "
            f"raw_cols={len(raw_contrib_input_features)}, "
            f"passthrough_cols={len(raw_contrib_passthrough_features)}, "
            f"svd_cols={len(raw_contrib_transformed_feature_names)}."
        )
    ae_gmm_state: dict[str, Any] = {}
    ae_gmm_input_features: list[str] = []
    ae_gmm_feature_names: list[str] = []
    ae_gmm_context_feature_names: list[str] = []
    ae_gmm_metrics: dict[str, Any] = {
        "ae_gmm_features_enabled": False,
        "ae_gmm_feature_count": 0,
        "ae_gmm_context_feature_count": 0,
    }
    X_oof_source_df = X_df if raw_contrib_input_features else X_model_df
    if _lgbm_ae_gmm_enabled(cfg):
        generated_set = {str(c) for c in AE_GMM_FEATURE_COLUMNS}
        ae_gmm_input_features = [str(c) for c in selected_features if str(c) not in generated_set]
        if len(ae_gmm_input_features) >= 2:
            select_idx = np.asarray(stage_indices.get("lgbm_select", hpo_idx), dtype=np.int32)
            select_idx = select_idx[(select_idx >= 0) & (select_idx < n)]
            repr_idx = np.unique(
                np.concatenate(
                    [
                        select_idx if len(select_idx) else hpo_idx,
                        hpo_idx,
                    ]
                )
            ).astype(np.int32)
            ae_gmm_state = _fit_ae_gmm_post_selection_state(
                X_model_df,
                ae_gmm_input_features,
                repr_idx,
                y_metric=y_metric,
                returns=ret_arr,
                random_state=random_state + 90221,
            )
            ae_gmm_feature_names = _ae_gmm_model_feature_names_for_objective(objective_mode)
            ae_gmm_context_feature_names = _ae_gmm_context_feature_names_for_objective(objective_mode)
            selected_features = list(ae_gmm_input_features)
            ae_gmm_metrics = {
                "ae_gmm_features_enabled": bool(_ae_gmm_state_enabled(ae_gmm_state)),
                "ae_gmm_feature_count": int(len(ae_gmm_feature_names)),
                "ae_gmm_context_feature_count": int(len(ae_gmm_context_feature_names)),
                "ae_gmm_input_feature_count": int(len(ae_gmm_input_features)),
                "ae_gmm_fit_rows": int(len(repr_idx)),
                "ae_gmm_reason": str(ae_gmm_state.get("reason", "")),
                "ae_gmm_selected_config": _json_sanitize(
                    ae_gmm_state.get("selected_config", ae_gmm_state.get("report", {}))
                ),
                "ae_gmm_model_feature_policy": "downstream_only",
            }
            tprint(
                "LGBM selected-feature AE/GMM representation "
                f"{'enabled' if _ae_gmm_state_enabled(ae_gmm_state) else 'neutral'}: "
                f"input_features={len(ae_gmm_input_features)}, "
                f"model_features={len(ae_gmm_feature_names)}, "
                f"context_features={len(ae_gmm_context_feature_names)}, "
                f"fit_rows={len(repr_idx)}, "
                f"reason={ae_gmm_state.get('reason', '')}."
            )
        else:
            ae_gmm_metrics["ae_gmm_reason"] = "insufficient_selected_features"
    stability_groups = _stability_group_bundle(n, timestamps=timestamps, assets=assets)
    hpo_groups = _groups_take(stability_groups, hpo_idx)
    hpo_weights, hpo_weight_ess = _normalize_weights(sw[hpo_idx])
    _, final_fit_weight_ess = _normalize_weights(sw[fit_idx])
    if preset_best_params:
        best_params = _effective_lgbm_params(dict(preset_best_params), classifier=classifier)
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
            X_model_df.iloc[hpo_idx].reset_index(drop=True),
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
            cfg=cfg,
        )
        best_params = _effective_lgbm_params(dict(best_params), classifier=classifier)
        hpo_metrics["hpo_best_params"] = dict(best_params)
    hpo_metrics.update(ae_gmm_metrics)
    hpo_metrics.update(_lgbm_regime_score_feature_metric_summary(regime_score_feature_diag))
    if (not preset_best_params) or bool(LGBM_FINAL_LEAF_FLOOR_PRESET):
        params_before_leaf_floor = dict(best_params)
        best_params, leaf_floor_diag = _final_fit_leaf_floor(
            best_params,
            fit_rows=len(fit_idx),
            hpo_rows=len(hpo_idx),
            hpo_effective_rows=hpo_weight_ess,
            final_effective_rows=final_fit_weight_ess,
            objective_mode=objective_mode,
        )
        hpo_metrics["final_fit_leaf_floor"] = dict(leaf_floor_diag)
        hpo_metrics["hpo_best_params"] = dict(best_params)
        if bool(leaf_floor_diag.get("enabled", False)):
            hpo_metrics["hpo_best_params_pre_final_leaf_floor"] = params_before_leaf_floor
            tprint(
                "LGBM final fit min_child_samples transfer applied: "
                f"{int(leaf_floor_diag.get('before', 0))} -> "
                f"{int(leaf_floor_diag.get('after', 0))} "
                f"(fit_rows={int(leaf_floor_diag.get('fit_rows', 0))}, "
                f"hpo_n_eff={float(leaf_floor_diag.get('hpo_effective_rows', 0.0)):.1f}, "
                f"final_n_eff={float(leaf_floor_diag.get('final_effective_rows', 0.0)):.1f}, "
                f"ratio={float(leaf_floor_diag.get('effective_ratio', 0.0)):.3f}, "
                f"alpha={float(leaf_floor_diag.get('alpha', 0.0)):.2f}, "
                f"cap={int(leaf_floor_diag.get('cap', 0))})."
            )
    else:
        hpo_metrics["final_fit_leaf_floor"] = {
            "enabled": False,
            "reason": "native_preset_reuse_exact",
            "preset_opt_in_env": "EPM_LGBM_FINAL_LEAF_FLOOR_PRESET=1",
            "fit_rows": int(len(fit_idx)),
        }
    label_weight_hpo_report: dict[str, Any] = {
        "enabled": False,
        "objective_mode": str(objective_mode),
        "selected": False,
        "winner": "baseline",
    }
    elected_label_soft: np.ndarray | None = None
    elected_label_hard: np.ndarray | None = None
    elected_label_weight: np.ndarray | None = None
    (
        y_arr,
        y_metric,
        sw,
        label_weight_hpo_report,
        elected_label_soft,
        elected_label_hard,
        elected_label_weight,
    ) = _run_base_label_weight_hpo(
        X_model_df,
        y_arr,
        y_metric,
        sw,
        selected_features,
        hpo_idx,
        label_context,
        best_params,
        classifier=classifier,
        objective_mode=objective_mode,
        timestamps=timestamps,
        random_state=random_state + 571,
        reference_artifact_dir=reference_artifact_dir,
    )
    if (
        not bool(label_weight_hpo_report.get("enabled", False))
        and isinstance(preset_label_weight_hpo_report, dict)
        and preset_label_weight_hpo_report
    ):
        y_reuse, y_metric_reuse, sw_reuse, reuse_diag = apply_label_weight_hpo_report_to_arrays(
            y_arr,
            y_metric,
            sw,
            label_context,
            preset_label_weight_hpo_report,
            objective_mode=objective_mode,
            classifier=classifier,
            label=f"final_{objective_mode}_native_preset_fallback",
        )
        if bool(reuse_diag.get("applied", False)):
            y_arr = y_reuse
            y_metric = y_metric_reuse
            sw = sw_reuse
            y_hard_diag = np.asarray(y_metric_reuse, dtype=np.float32)
            ret_arr = _as_returns(y_metric, returns)
            elected_label_hard = np.asarray(y_metric_reuse, dtype=np.int8)
            elected_label_weight = np.asarray(sw_reuse, dtype=np.float32)
            elected_label_soft = (
                np.asarray(y_reuse, dtype=np.float32)
                if bool(LGBM_TRUE_SOFT_LABELS)
                else None
            )
            label_weight_hpo_report = dict(preset_label_weight_hpo_report)
            label_weight_hpo_report["enabled"] = False
            label_weight_hpo_report["selected"] = True
            label_weight_hpo_report["winner"] = "optimized"
            label_weight_hpo_report["reused_from_native_preset"] = True
            label_weight_hpo_report["reason"] = "hpo_skipped_reused_native_preset"
            label_weight_hpo_report["reuse_diag"] = _json_sanitize(reuse_diag)
            tprint(
                "LGBM label/sample-weight HPO skipped; reused native preset "
                "label/sample-weight winner for final fit."
            )
    if bool(label_weight_hpo_report.get("enabled", False)):
        y_hard_diag = np.asarray(y_metric, dtype=np.float32)
        ret_arr = _as_returns(y_metric, returns)
        label_diag.update(
            _log_label_weight_diagnostics(
                y_arr,
                y_hard_diag,
                sw,
                label=f"final_{objective_mode}_post_label_weight_hpo",
            )
        )
        _, final_fit_weight_ess = _normalize_weights(sw[fit_idx])
        hpo_metrics["label_weight_hpo_enabled"] = True
        hpo_metrics["label_weight_hpo_selected"] = bool(
            label_weight_hpo_report.get("selected", False)
        )
        hpo_metrics["label_weight_hpo_winner"] = str(
            label_weight_hpo_report.get("winner", "baseline")
        )
        hpo_metrics["label_weight_hpo_baseline_objective"] = float(
            (label_weight_hpo_report.get("baseline") or {}).get("objective", float("nan"))
        )
        hpo_metrics["label_weight_hpo_best_objective"] = float(
            (label_weight_hpo_report.get("best_optimized") or {}).get("objective", float("nan"))
        )
        hpo_metrics["label_weight_hpo_objective_delta_vs_baseline"] = float(
            label_weight_hpo_report.get("objective_delta_vs_baseline", float("nan"))
        )
    else:
        hpo_metrics["label_weight_hpo_enabled"] = False
        hpo_metrics["label_weight_hpo_reason"] = str(
            label_weight_hpo_report.get("reason", "disabled")
        )
        hpo_metrics["label_weight_hpo_reused_from_native_preset"] = bool(
            label_weight_hpo_report.get("reused_from_native_preset", False)
        )
    if _lgbm_regime_specialist_should_build_bundle(cfg, objective_mode):
        assessment_X_df, assessment_ts, assessment_asset_values = _lgbm_regime_specialist_assessment_inputs(
            assessment_X,
            assessment_timestamps,
            assessment_assets,
            label_context=label_context,
        )
    else:
        assessment_X_df, assessment_ts, assessment_asset_values = None, None, None
    regime_specialist_bundle = _build_lgbm_regime_specialist_bundle(
        X_model_df,
        selected_features,
        timestamps=timestamps,
        assets=assets,
        assessment_X_df=assessment_X_df,
        assessment_timestamps=assessment_ts,
        assessment_assets=assessment_asset_values,
        objective_mode=objective_mode,
        cfg=cfg,
        random_state=random_state + 9157,
        label="final",
    )
    sw, regime_specialist_apply_diag = _apply_lgbm_regime_specialist_weights(
        sw,
        regime_specialist_bundle,
    )
    if bool(regime_specialist_apply_diag.get("applied", False)):
        _, final_fit_weight_ess = _normalize_weights(sw[fit_idx])
        tprint(
            "LGBM final regime specialist sample weighting applied: "
            f"objective={objective_mode}, "
            f"ess={float(regime_specialist_apply_diag.get('effective_sample_size', float('nan'))):.1f}."
        )
    hpo_metrics.update(regime_specialist_bundle.get("metrics", {}))
    hpo_metrics["regime_specialist_apply_reason"] = str(
        regime_specialist_apply_diag.get("reason", "")
    )
    if bool(regime_specialist_apply_diag.get("applied", False)):
        hpo_metrics["regime_specialist_applied_effective_sample_size"] = float(
            regime_specialist_apply_diag.get("effective_sample_size", float("nan"))
        )
    specialist_similarity_all = _lgbm_regime_specialist_similarity_for_idx(
        regime_specialist_bundle,
    )
    _save_lgbm_regime_specialist_diagnostics(
        reference_artifact_dir,
        regime_specialist_bundle,
        objective_mode=objective_mode,
        label="final",
    )
    if distill_passes > 0:
        final_weights, pre_final_oof = _oof_distilled_sample_weights_lgbm(
            X_model_df,
            y_arr,
            sw,
            selected_features,
            classifier=classifier,
            params=best_params,
            groups=stability_groups,
            timestamps=timestamps,
            returns=ret_arr,
            metric_y=y_metric,
            random_state=random_state + 33107,
            passes=distill_passes,
            label="final",
            objective_mode=objective_mode,
            cfg=cfg,
            specialist_similarity=specialist_similarity_all,
        )
    else:
        final_weights = sw.copy()
        pre_final_oof = np.asarray(oof_probs if oof_probs is not None else np.full(n, float(np.mean(y_arr))), dtype=np.float32)
    model = LGBMStabilityModel(mode=mode)
    model.regime_score_feature_diagnostics_ = dict(regime_score_feature_diag or {})
    model.regime_specialist_diagnostics_ = dict(
        regime_specialist_bundle.get("diagnostics", {}) if regime_specialist_bundle else {}
    )
    model.regime_specialist_metrics_ = dict(
        regime_specialist_bundle.get("metrics", {}) if regime_specialist_bundle else {}
    )
    model.label_weight_hpo_report_ = dict(label_weight_hpo_report or {})
    if elected_label_soft is not None:
        model.label_weight_hpo_soft_label_ = np.asarray(elected_label_soft, dtype=np.float32)
    elif bool((label_weight_hpo_report or {}).get("enabled", False)):
        model.label_weight_hpo_soft_label_ = np.asarray(y_arr, dtype=np.float32)
    if elected_label_hard is not None:
        model.label_weight_hpo_hard_label_ = np.asarray(elected_label_hard, dtype=np.int8)
    elif bool((label_weight_hpo_report or {}).get("enabled", False)):
        model.label_weight_hpo_hard_label_ = np.asarray(y_metric >= 0.5, dtype=np.int8)
    if elected_label_weight is not None:
        model.label_weight_hpo_sample_weight_ = np.asarray(elected_label_weight, dtype=np.float32)
    elif bool((label_weight_hpo_report or {}).get("enabled", False)):
        model.label_weight_hpo_sample_weight_ = np.asarray(sw, dtype=np.float32)
    model.selected_features = list(selected_features)
    if ae_gmm_input_features:
        model.input_feature_names = list(ae_gmm_input_features)
        model.ae_gmm_input_features = list(ae_gmm_input_features)
        model.ae_gmm_feature_names = list(ae_gmm_feature_names)
        model.ae_gmm_context_feature_names = list(ae_gmm_context_feature_names)
        model.ae_gmm_state = dict(ae_gmm_state or {})
    model.meta_leaf_lite_diagnostics_enabled = bool(
        LGBM_META_LEAF_LITE_DIAGNOSTICS or LGBM_META_LEAF_DIAGNOSTICS
    )
    model.meta_leaf_support_diagnostics_enabled = bool(
        LGBM_META_LEAF_SUPPORT_DIAGNOSTICS or LGBM_META_LEAF_DIAGNOSTICS
    )
    model.meta_leaf_target_diagnostics_enabled = bool(
        LGBM_META_LEAF_TARGET_DIAGNOSTICS or LGBM_META_LEAF_DIAGNOSTICS
    )
    model.meta_leaf_centroid_diagnostics_enabled = bool(
        LGBM_META_LEAF_CENTROID_DIAGNOSTICS or LGBM_META_LEAF_DIAGNOSTICS
    )
    model.meta_leaf_diagnostics_enabled = bool(LGBM_META_LEAF_DIAGNOSTICS)
    model.meta_contrib_diagnostics_enabled = bool(LGBM_META_CONTRIB_DIAGNOSTICS)
    model.meta_contrib_method = str(LGBM_META_CONTRIB_METHOD)
    model.meta_score_path_diagnostics_enabled = bool(LGBM_META_SCORE_PATH_DIAGNOSTICS)
    model.meta_drift_features_enabled = bool(LGBM_META_DRIFT_FEATURES)
    model.meta_context_features_enabled = bool(LGBM_FINAL_OOF_CONTEXT_FEATURES)
    model.meta_contrib_context_features_enabled = bool(LGBM_FINAL_OOF_CONTRIB_CONTEXT_FEATURES)
    model.meta_raw_state_context_features_enabled = bool(LGBM_FINAL_OOF_RAW_STATE_CONTEXT_FEATURES)
    model.meta_base_error_context_features_enabled = bool(LGBM_FINAL_OOF_BASE_ERROR_CONTEXT_FEATURES)
    if ae_gmm_context_feature_names:
        model.meta_feature_names = list(
            dict.fromkeys(list(model.meta_feature_names) + list(ae_gmm_context_feature_names))
        )
    if raw_contrib_input_features:
        model.input_feature_names = list(input_selected_features)
        model.raw_contrib_input_features = list(raw_contrib_input_features)
        model.raw_contrib_passthrough_features = list(raw_contrib_passthrough_features)
        model.raw_contrib_transformed_feature_names = list(raw_contrib_transformed_feature_names)
        model.raw_contrib_input_state = raw_contrib_input_state
    model.best_params = dict(best_params)
    X_fit = X_model_df.iloc[fit_idx][selected_features].reset_index(drop=True)
    X_all_selected = X_model_df[selected_features].reset_index(drop=True)
    history_defaults = build_model_effectiveness_history_defaults(
        X_model_df[selected_features].reset_index(drop=True),
        selected_features,
        timestamps=timestamps,
    )
    model.model_effectiveness_history_defaults_ = dict(
        history_defaults.get("defaults", {})
    )
    model.model_effectiveness_history_default_sources_ = dict(
        history_defaults.get("sources", {})
    )
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
            f"sequential_distill={'no' if LGBM_DISABLE_SELF_DISTILLATION else 'yes'}."
        )
        params_i = dict(best_params)
        params_i["random_state"] = int(random_state + 7001 + i * 101)
        fitted = _fit_lgbm_model(
            X_fit,
            y_fit,
            w_fit,
            classifier=classifier,
            params=params_i,
            attach_leaf_diagnostics=True,
        )
        model.models.append(fitted)
        model_pred_all = _predict_lgbm_raw_batched(fitted, X_all_selected, mode)
        if running_ensemble_pred is None:
            running_ensemble_pred = model_pred_all.astype(np.float32)
        else:
            running_ensemble_pred = (
                (running_ensemble_pred.astype(np.float32) * float(i)) + model_pred_all.astype(np.float32)
            ) / float(i + 1)
        if not LGBM_DISABLE_SELF_DISTILLATION:
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
            distill = _recency_shrink_weight_towards_one(
                distill,
                timestamps,
                objective_mode=objective_mode,
                cfg=cfg,
            )
            fp_weight = _recency_shrink_weight_towards_one(
                fp_weight,
                timestamps,
                objective_mode=objective_mode,
                cfg=cfg,
            )
            distill = _regime_specialist_shrink_weight_towards_one(
                distill,
                specialist_similarity_all,
                cfg=cfg,
            )
            fp_weight = _regime_specialist_shrink_weight_towards_one(
                fp_weight,
                specialist_similarity_all,
                cfg=cfg,
            )
            distill, fp_weight = apply_distillation_recipe(
                distill,
                fp_weight,
                y_metric=y_metric,
                pred=running_ensemble_pred,
                returns=ret_arr,
                timestamps=timestamps,
                objective_mode=objective_mode,
                cfg=cfg,
            )
            sequential_weights, final_ensemble_ess = _normalize_weights(
                sequential_weight_base * distill * fp_weight
            )
            prev_ensemble_pred = running_ensemble_pred.copy()
        tprint(
            f"LGBM final model {i + 1}/{LGBM_FINAL_MODEL_COUNT} fitted on "
            f"{len(y_fit)} rows in {time.perf_counter() - fit_t0:.1f}s; "
            f"updated all-row sequential weights ess={final_ensemble_ess:.1f}."
        )
    final_weights = sequential_weights.astype(np.float32)
    split_importance_sum = np.zeros(len(selected_features), dtype=np.float64)
    gain_importance_sum = np.zeros(len(selected_features), dtype=np.float64)
    for fitted in model.models:
        gain_imp, split_imp = _feature_importances(fitted, len(selected_features))
        gain_importance_sum += np.asarray(gain_imp, dtype=np.float64)
        split_importance_sum += np.asarray(split_imp, dtype=np.float64)
    final_used_feature_count = int(np.sum(split_importance_sum > 0.0))
    final_gain_feature_count = int(np.sum(gain_importance_sum > 0.0))
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
    checkpoint_ref_dir = reference_artifact_dir or os.environ.get("EPM_LGBM_REFERENCE_ARTIFACT_DIR")
    if checkpoint_ref_dir is None:
        checkpoint_meta_path = meta_feature_output_path or os.environ.get("EPM_LGBM_META_FEATURE_OUTPUT_PATH")
        if checkpoint_meta_path:
            checkpoint_meta_path_obj = Path(checkpoint_meta_path)
            checkpoint_ref_dir = checkpoint_meta_path_obj.parent / "lgbm_reference" / checkpoint_meta_path_obj.stem
    checkpoint_dir = _resolve_lgbm_final_model_checkpoint_dir(checkpoint_ref_dir)
    if checkpoint_dir is not None:
        try:
            _save_lgbm_final_model_checkpoint(
                model,
                checkpoint_dir,
                split_importance_sum=split_importance_sum,
                gain_importance_sum=gain_importance_sum,
                final_ensemble_ess=final_ensemble_ess,
                pre_final_oof=pre_final_oof,
                final_weights=final_weights,
            )
        except Exception as exc:
            tprint(f"WARNING: failed to save LGBM final model checkpoint before final OOF/meta CV: {exc}")
    if LGBM_META_DRIFT_FEATURES or LGBM_FINAL_OOF_BASE_ERROR_CONTEXT_FEATURES:
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
    else:
        model.drift_reference = {}
    model.contrib_archetype_state, model.raw_state_archetype_state = _fit_lgbm_archetype_states(
        model.models,
        X_fit,
        list(selected_features),
        timestamps=_take_aligned(timestamps, fit_idx, n),
        assets=_take_aligned(assets, fit_idx, n),
        random_state=random_state + 99131,
    )
    if LGBM_SKIP_FINAL_OOF_META_CV:
        tprint(
            "LGBM final OOF/meta CV skipped by EPM_LGBM_SKIP_FINAL_OOF_META_CV=1; "
            "using filled pre-final OOF for trial scoring."
        )
        fill = float(np.nanmean(pre_final_oof)) if np.isfinite(pre_final_oof).any() else float(np.mean(y_arr))
        final_oof = np.nan_to_num(pre_final_oof, nan=fill).astype(np.float32)
        skip_metrics = _metric_pack(y_metric, final_oof, classifier=classifier, groups=stability_groups, returns=ret_arr)
        skip_metrics.update(_vol_normalized_tp_sl_precision_metrics(final_oof, label_context))
        skip_metrics.update(
            _lgbm_regime_specialist_current_metrics(
                y_metric,
                final_oof,
                regime_specialist_bundle,
                classifier=classifier,
                groups=stability_groups,
                returns=ret_arr,
                label_context=label_context,
                label_context_total_rows=n,
            )
        )
        final_fold_metrics = [skip_metrics]
        full_rank = _safe_rank_pct(final_oof)
        meta_oof_features = _lgbm_meta_features_from_predictions(
            final_oof,
            mode=mode,
            rank_pct=full_rank,
            rank_bin_stats=None,
            model_count=len(model.models),
            tree_count=float(np.nanmean([_model_num_iterations(m) for m in model.models])) if model.models else 0.0,
            score_path_probs=None,
        )
        if LGBM_META_SCORE_PATH_DIAGNOSTICS:
            step_t0 = time.perf_counter()
            tprint(
                "LGBM final OOF/meta skip path: "
                f"score-path diagnostics started rows={len(X_all_selected)}, "
                f"max_trees={int(LGBM_META_SCORE_PATH_MAX_TREES or LGBM_META_LEAF_MAX_TREES)}."
            )
            _append_score_path_tree_diagnostics(
                meta_oof_features,
                model.models,
                X_all_selected,
                mode=mode,
                final_pred=final_oof,
            )
            tprint(
                "LGBM final OOF/meta skip path: "
                f"score-path diagnostics complete in {time.perf_counter() - step_t0:.1f}s."
            )
        if (
            LGBM_META_LEAF_LITE_DIAGNOSTICS
            or LGBM_META_LEAF_DIAGNOSTICS
            or LGBM_META_LEAF_SUPPORT_DIAGNOSTICS
            or LGBM_META_LEAF_TARGET_DIAGNOSTICS
            or LGBM_META_LEAF_CENTROID_DIAGNOSTICS
        ):
            step_t0 = time.perf_counter()
            leaf_mode = (
                "full"
                if LGBM_META_LEAF_DIAGNOSTICS
                else (
                    "lite+support/target"
                    if (LGBM_META_LEAF_SUPPORT_DIAGNOSTICS or LGBM_META_LEAF_TARGET_DIAGNOSTICS)
                    else "lite"
                )
            )
            tprint(
                "LGBM final OOF/meta skip path: "
                f"leaf-{leaf_mode} diagnostics started rows={len(X_all_selected)}, "
                f"max_trees={int(LGBM_META_LEAF_MAX_TREES)}."
            )
            _append_leaf_diagnostics(
                meta_oof_features,
                model.models,
                X_all_selected,
                prediction=final_oof,
                full_diagnostics=LGBM_META_LEAF_DIAGNOSTICS,
                support_diagnostics=LGBM_META_LEAF_SUPPORT_DIAGNOSTICS,
                target_diagnostics=LGBM_META_LEAF_TARGET_DIAGNOSTICS,
                centroid_diagnostics=LGBM_META_LEAF_CENTROID_DIAGNOSTICS,
            )
            tprint(
                "LGBM final OOF/meta skip path: "
                f"leaf-{leaf_mode} diagnostics complete in {time.perf_counter() - step_t0:.1f}s."
            )
        if LGBM_META_CONTRIB_DIAGNOSTICS:
            step_t0 = time.perf_counter()
            tprint(
                "LGBM final OOF/meta skip path: "
                f"{LGBM_META_CONTRIB_METHOD} contribution diagnostics started "
                f"rows={len(X_all_selected)}, max_trees={int(LGBM_META_LEAF_MAX_TREES)}."
            )
            _append_contrib_diagnostics(
                meta_oof_features,
                model.models,
                X_all_selected,
                method=LGBM_META_CONTRIB_METHOD,
            )
            tprint(
                "LGBM final OOF/meta skip path: "
                f"{LGBM_META_CONTRIB_METHOD} contribution diagnostics complete "
                f"in {time.perf_counter() - step_t0:.1f}s."
            )
        if LGBM_META_DRIFT_FEATURES:
            step_t0 = time.perf_counter()
            tprint("LGBM final OOF/meta skip path: capped drift features started.")
            _append_feature_drift_meta_features(
                meta_oof_features,
                X_all_selected,
                model.drift_reference,
            )
            tprint(
                "LGBM final OOF/meta skip path: capped drift features complete "
                f"in {time.perf_counter() - step_t0:.1f}s."
            )
        if LGBM_FINAL_OOF_CONTRIB_CONTEXT_FEATURES or LGBM_FINAL_OOF_RAW_STATE_CONTEXT_FEATURES:
            step_t0 = time.perf_counter()
            tprint(
                "LGBM final OOF/meta skip path: raw/contrib meta feature transform started "
                f"rows={len(X_all_selected)}."
            )
            _append_archetype_meta_features(
                meta_oof_features,
                model.models,
                X_all_selected,
                contrib_state=model.contrib_archetype_state,
                raw_state=model.raw_state_archetype_state,
                timestamps=timestamps,
                assets=assets,
            )
            tprint(
                "LGBM final OOF/meta skip path: raw/contrib meta feature transform complete "
                f"in {time.perf_counter() - step_t0:.1f}s."
            )
        else:
            tprint("LGBM final OOF/meta skip path: raw/contrib meta feature transform skipped.")
        if ae_gmm_context_feature_names and ae_gmm_input_features:
            ae_context_frame = _append_ae_gmm_features_to_model_frame(
                X_model_df,
                ae_gmm_input_features,
                ae_gmm_state,
                list(ae_gmm_context_feature_names),
                index=X_model_df.index,
            )
            for col in ae_gmm_context_feature_names:
                if col in ae_context_frame.columns:
                    meta_oof_features[col] = ae_context_frame[col].to_numpy(
                        dtype=np.float32,
                        copy=False,
                    )
        raw_contrib_feature_map = raw_contrib_feature_mapping(selected_features)
        step_t0 = time.perf_counter()
        if LGBM_RAW_CONTRIB_OOF_EXPORT and raw_contrib_feature_map:
            tprint(
                "LGBM final OOF/meta skip path: raw contribution OOF export started "
                f"rows={len(X_all_selected)}."
            )
            skip_contrib_matrix = _mean_contrib_matrix(model.models, X_all_selected)
            if skip_contrib_matrix is None:
                skip_contrib_matrix = np.zeros((n, len(selected_features)), dtype=np.float32)
            raw_contrib_oof_features = raw_contrib_frame(
                skip_contrib_matrix,
                selected_features,
                index=np.arange(n),
            ).reindex(columns=list(raw_contrib_feature_map.keys()), fill_value=0.0)
            tprint(
                "LGBM final OOF/meta skip path: raw contribution OOF export complete "
                f"in {time.perf_counter() - step_t0:.1f}s."
            )
        else:
            raw_contrib_oof_features = pd.DataFrame(index=np.arange(n))
            tprint("LGBM final OOF/meta skip path: raw contribution OOF export skipped.")
    else:
        final_oof, final_fold_metrics, meta_oof_features, raw_contrib_oof_features = _cross_val_oof_lgbm_with_meta_features(
            X_oof_source_df,
            y_arr,
            final_weights,
            selected_features,
            classifier=classifier,
            params=best_params,
            groups=stability_groups,
            timestamps=timestamps,
            assets=assets,
            returns=ret_arr,
            metric_y=y_metric,
            random_state=random_state + 11701,
            raw_contrib_input_features=raw_contrib_input_features,
            raw_contrib_passthrough_features=raw_contrib_passthrough_features,
            raw_contrib_transformed_feature_names=raw_contrib_transformed_feature_names,
            ae_gmm_input_features=ae_gmm_input_features,
            ae_gmm_feature_names=ae_gmm_feature_names,
            ae_gmm_context_feature_names=ae_gmm_context_feature_names,
            ae_gmm_enabled=bool(ae_gmm_input_features),
        )
    model.oof_probs = final_oof.astype(np.float32)
    step_t0 = time.perf_counter()
    tprint("LGBM final OOF/meta post-processing: rank-bin OOF stats fit started.")
    model.rank_bin_stats_oof = _fit_rank_bin_stats_oof(y_metric, np.asarray(meta_oof_features["rank_pct"], dtype=np.float32), classifier=classifier, returns=ret_arr)
    tprint(
        "LGBM final OOF/meta post-processing: rank-bin OOF stats fit complete "
        f"in {time.perf_counter() - step_t0:.1f}s."
    )
    base_error_fit_idx = np.asarray([], dtype=int)
    if LGBM_FINAL_OOF_BASE_ERROR_CONTEXT_FEATURES:
        base_error_fit_idx = np.arange(n, dtype=int)
        base_error_cap = int(LGBM_ARCHETYPE_FIT_MAX_ROWS)
        if base_error_cap > 0 and n > base_error_cap:
            base_error_fit_idx = np.linspace(0, n - 1, base_error_cap, dtype=int)
            tprint(
                "LGBM base-error archetype final-state fit sampled: "
                f"rows={n}->{len(base_error_fit_idx)}, max_rows={base_error_cap}."
            )
        X_base_error_fit = X_all_selected.iloc[base_error_fit_idx].reset_index(drop=True)
        step_t0 = time.perf_counter()
        tprint(
            "LGBM final OOF/meta post-processing: base-error internal diagnostics started "
            f"rows={len(base_error_fit_idx)}."
        )
        final_internal_for_base_error = _lgbm_meta_features_from_models(
            model.models,
            X_base_error_fit,
            mode=mode,
            rank_bin_stats=model.rank_bin_stats_oof,
        )
        tprint(
            "LGBM final OOF/meta post-processing: base-error internal diagnostics complete "
            f"in {time.perf_counter() - step_t0:.1f}s."
        )
        step_t0 = time.perf_counter()
        tprint("LGBM final OOF/meta post-processing: base-error drift/archetype transforms started.")
        _append_feature_drift_meta_features(
            final_internal_for_base_error,
            X_base_error_fit,
            model.drift_reference,
        )
        _append_archetype_meta_features(
            final_internal_for_base_error,
            model.models,
            X_base_error_fit,
            contrib_state=model.contrib_archetype_state,
            raw_state=model.raw_state_archetype_state,
            timestamps=_take_aligned(timestamps, base_error_fit_idx, n),
            assets=_take_aligned(assets, base_error_fit_idx, n),
        )
        tprint(
            "LGBM final OOF/meta post-processing: base-error drift/archetype transforms complete "
            f"in {time.perf_counter() - step_t0:.1f}s."
        )
        step_t0 = time.perf_counter()
        tprint("LGBM final OOF/meta post-processing: final base-error archetype state fit started.")
        model.base_error_archetype_state = _fit_base_error_archetype_state_from_meta(
            final_internal_for_base_error,
            y_metric[base_error_fit_idx],
            final_oof[base_error_fit_idx],
            classifier=classifier,
            random_state=random_state + 99317,
        )
        tprint(
            "LGBM final OOF/meta post-processing: final base-error archetype state fit complete "
            f"in {time.perf_counter() - step_t0:.1f}s."
        )
        base_error_oof_present = bool(
            set(BASE_ERROR_ARCHETYPE_FEATURE_NAMES).issubset(set(map(str, meta_oof_features.columns)))
            and meta_oof_features.reindex(columns=BASE_ERROR_ARCHETYPE_FEATURE_NAMES).notna().all().all()
        )
        if not base_error_oof_present:
            step_t0 = time.perf_counter()
            tprint("LGBM final OOF/meta post-processing: OOF base-error feature append started.")
            _append_base_error_archetype_features(
                meta_oof_features,
                model.base_error_archetype_state,
            )
            tprint(
                "LGBM final OOF/meta post-processing: OOF base-error feature append complete "
                f"in {time.perf_counter() - step_t0:.1f}s."
            )
    else:
        model.base_error_archetype_state = None
        tprint("LGBM final OOF/meta post-processing: base-error context skipped.")
    if not raw_contrib_oof_features.empty:
        model.raw_contrib_oof_features = raw_contrib_oof_features.astype(np.float32, copy=False)
        model.raw_contrib_oof_feature_names = [str(c) for c in model.raw_contrib_oof_features.columns]
        model.raw_contrib_feature_mapping = raw_contrib_feature_mapping(selected_features)
        model.meta_feature_names = list(
            dict.fromkeys(list(model.meta_feature_names) + model.raw_contrib_oof_feature_names)
        )
        meta_oof_features = pd.concat(
            [meta_oof_features, model.raw_contrib_oof_features],
            axis=1,
        )
    model.meta_oof_features = meta_oof_features.reindex(columns=model.meta_feature_names, fill_value=0.0).astype(np.float32)
    model.oof_uncertainty_features = {
        str(col): model.meta_oof_features[col].to_numpy(dtype=np.float32, copy=True)
        for col in LGBM_INTERNAL_METRIC_FEATURE_NAMES
        if col in model.meta_oof_features.columns
    }
    for col in ARCHETYPE_FEATURE_NAMES:
        if col in model.meta_oof_features.columns:
            model.oof_uncertainty_features[str(col)] = model.meta_oof_features[col].to_numpy(
                dtype=np.float32,
                copy=True,
            )
    for col in BASE_ERROR_ARCHETYPE_FEATURE_NAMES:
        if col in model.meta_oof_features.columns:
            model.oof_uncertainty_features[str(col)] = model.meta_oof_features[col].to_numpy(
                dtype=np.float32,
                copy=True,
            )
    for col in getattr(model, "raw_contrib_oof_feature_names", []) or []:
        if col in model.meta_oof_features.columns:
            model.oof_uncertainty_features[str(col)] = model.meta_oof_features[col].to_numpy(
                dtype=np.float32,
                copy=True,
            )
    ae_gmm_uncertainty_feature_names = (
        getattr(model, "ae_gmm_context_feature_names", [])
        or getattr(model, "ae_gmm_feature_names", [])
        or []
    )
    for col in ae_gmm_uncertainty_feature_names:
        if col in model.meta_oof_features.columns:
            model.oof_uncertainty_features[str(col)] = model.meta_oof_features[col].to_numpy(
                dtype=np.float32,
                copy=True,
            )
    final_metrics = _metric_pack(y_metric, final_oof, classifier=classifier, groups=stability_groups, returns=ret_arr)
    final_metrics.update(_vol_normalized_tp_sl_precision_metrics(final_oof, label_context))
    final_metrics.update(
        _lgbm_regime_specialist_current_metrics(
            y_metric,
            final_oof,
            regime_specialist_bundle,
            classifier=classifier,
            groups=stability_groups,
            returns=ret_arr,
            label_context=label_context,
            label_context_total_rows=n,
        )
    )
    final_metrics.update(_aggregate_j(final_fold_metrics, objective_mode=objective_mode))
    candidate_metrics = dict(metrics or {})
    model.metrics = dict(candidate_metrics)
    model.metrics.update(hpo_metrics)
    model.metrics.update(final_metrics)
    model.metrics["feature_count"] = int(len(selected_features))
    model.metrics["selected_features_count"] = int(len(selected_features))
    model.metrics["selected_features_preview"] = list(selected_features[:50])
    model.metrics["lgbm_meta_leaf_lite_diagnostics_enabled"] = bool(model.meta_leaf_lite_diagnostics_enabled)
    model.metrics["lgbm_meta_leaf_support_diagnostics_enabled"] = bool(model.meta_leaf_support_diagnostics_enabled)
    model.metrics["lgbm_meta_leaf_target_diagnostics_enabled"] = bool(model.meta_leaf_target_diagnostics_enabled)
    model.metrics["lgbm_meta_leaf_centroid_diagnostics_enabled"] = bool(model.meta_leaf_centroid_diagnostics_enabled)
    model.metrics["lgbm_meta_leaf_diagnostics_enabled"] = bool(model.meta_leaf_diagnostics_enabled)
    model.metrics["lgbm_meta_leaf_max_trees"] = int(LGBM_META_LEAF_MAX_TREES)
    model.metrics["lgbm_meta_contrib_diagnostics_enabled"] = bool(model.meta_contrib_diagnostics_enabled)
    model.metrics["lgbm_meta_contrib_method"] = str(model.meta_contrib_method)
    model.metrics["lgbm_meta_score_path_diagnostics_enabled"] = bool(model.meta_score_path_diagnostics_enabled)
    model.metrics["lgbm_meta_drift_features_enabled"] = bool(model.meta_drift_features_enabled)
    model.metrics["lgbm_meta_drift_max_rows"] = int(LGBM_META_DRIFT_MAX_ROWS)
    model.metrics["lgbm_meta_drift_max_features"] = int(LGBM_META_DRIFT_MAX_FEATURES)
    model.metrics["lgbm_meta_context_features_enabled"] = bool(model.meta_context_features_enabled)
    model.metrics["lgbm_meta_contrib_context_features_enabled"] = bool(
        model.meta_contrib_context_features_enabled
    )
    model.metrics["lgbm_meta_raw_state_context_features_enabled"] = bool(
        model.meta_raw_state_context_features_enabled
    )
    model.metrics["lgbm_meta_base_error_context_features_enabled"] = bool(
        model.meta_base_error_context_features_enabled
    )
    model.metrics["final_fit_train_rows"] = int(len(fit_idx))
    model.metrics["final_fit_train_rows_total"] = int(n)
    model.metrics["final_fit_used_all_rows"] = bool(len(fit_idx) == n)
    model.metrics["final_fit_split_used_feature_count"] = int(final_used_feature_count)
    model.metrics["final_fit_gain_used_feature_count"] = int(final_gain_feature_count)
    model.metrics["feature_drift_reference_feature_count"] = int(
        len(model.drift_reference.get("feature_names", []))
    )
    model.metrics["model_effectiveness_history_default_feature_count"] = int(
        history_defaults.get("feature_count", 0)
    )
    model.metrics["model_effectiveness_history_default_policy"] = str(
        history_defaults.get("policy", "")
    )
    model.metrics["feature_drift_reference_fit_rows"] = int(
        model.drift_reference.get("fit_rows", 0)
    )
    model.metrics["archetype_features_enabled"] = bool(LGBM_ARCHETYPE_FEATURES)
    model.metrics["contrib_archetype_feature_count"] = int(
        len(CONTRIB_ARCHETYPE_FEATURE_NAMES)
        if getattr(model, "contrib_archetype_state", None) is not None
        else 0
    )
    model.metrics["raw_state_archetype_feature_count"] = int(
        len(RAW_STATE_SVD_FEATURE_NAMES) + len(RAW_STATE_DIAGNOSTIC_FEATURE_NAMES)
        if getattr(model, "raw_state_archetype_state", None) is not None
        else 0
    )
    base_error_state = getattr(model, "base_error_archetype_state", None)
    model.metrics["base_error_archetype_features_enabled"] = bool(
        getattr(base_error_state, "enabled", False)
    )
    model.metrics["base_error_archetype_feature_count"] = int(
        len(BASE_ERROR_ARCHETYPE_FEATURE_NAMES)
        if base_error_state is not None
        else 0
    )
    model.metrics["base_error_archetype_signature_feature_count"] = int(
        len(getattr(base_error_state, "feature_names", []) or [])
    )
    model.metrics["base_error_archetype_fit_rows"] = int(len(base_error_fit_idx))
    model.metrics["base_error_archetype_reason"] = str(
        getattr(base_error_state, "reason", "") if base_error_state is not None else "not_fitted"
    )
    model.metrics["raw_contrib_oof_export_enabled"] = bool(LGBM_RAW_CONTRIB_OOF_EXPORT)
    model.metrics["raw_contrib_oof_feature_count"] = int(
        len(getattr(model, "raw_contrib_oof_feature_names", []) or [])
    )
    model.metrics["raw_contrib_input_feature_count"] = int(
        len(getattr(model, "raw_contrib_input_features", []) or [])
    )
    model.metrics["raw_contrib_svd_feature_count"] = int(
        len(getattr(model, "raw_contrib_transformed_feature_names", []) or [])
    )
    model.metrics["ae_gmm_features_enabled"] = bool(
        _ae_gmm_state_enabled(getattr(model, "ae_gmm_state", {}) or {})
    )
    model.metrics["ae_gmm_input_feature_count"] = int(
        len(getattr(model, "ae_gmm_input_features", []) or [])
    )
    model.metrics["ae_gmm_feature_count"] = int(
        len(getattr(model, "ae_gmm_feature_names", []) or [])
    )
    model.metrics["ae_gmm_context_feature_count"] = int(
        len(getattr(model, "ae_gmm_context_feature_names", []) or [])
    )
    model.metrics["ae_gmm_feature_names_preview"] = list(
        (getattr(model, "ae_gmm_feature_names", []) or [])[:50]
    )
    model.metrics["ae_gmm_context_feature_names_preview"] = list(
        (getattr(model, "ae_gmm_context_feature_names", []) or [])[:50]
    )
    model.metrics["ae_gmm_selected_config"] = _json_sanitize(
        (getattr(model, "ae_gmm_state", {}) or {}).get("selected_config", {})
    )
    model.metrics["archetype_feature_names"] = list(ARCHETYPE_FEATURE_NAMES)
    model.metrics["base_error_archetype_feature_names"] = list(BASE_ERROR_ARCHETYPE_FEATURE_NAMES)
    model.metrics["base_error_archetype_signature_features_preview"] = list(
        (getattr(base_error_state, "feature_names", []) or [])[:50]
    )
    model.metrics["raw_contrib_oof_feature_names_preview"] = list(
        (getattr(model, "raw_contrib_oof_feature_names", []) or [])[:50]
    )
    model.metrics["raw_contrib_input_features_preview"] = list(
        (getattr(model, "raw_contrib_input_features", []) or [])[:50]
    )
    model.metrics["feature_drift_reference_features_preview"] = list(
        model.drift_reference.get("feature_names", [])[:50]
    )
    model.metrics["final_model_count"] = int(LGBM_FINAL_MODEL_COUNT)
    model.metrics["final_ensemble_sequential_distillation"] = not bool(LGBM_DISABLE_SELF_DISTILLATION)
    model.metrics["oof_distillation_passes"] = int(distill_passes)
    model.metrics["min_oof_distillation_passes"] = int(LGBM_MIN_OOF_DISTILLATION_PASSES)
    model.metrics["meta_min_oof_distillation_passes"] = int(LGBM_META_MIN_OOF_DISTILLATION_PASSES)
    model.metrics["final_ensemble_sequential_weight_ess"] = float(final_ensemble_ess)
    _active_recency_hpo_metrics = active_recency_hpo_config(cfg, objective_mode)
    model.metrics["recency_weighting_enabled"] = bool(LGBM_RECENCY_WEIGHTING) or bool(
        _active_recency_hpo_metrics
    )
    model.metrics["recency_sample_weight_applied"] = bool(recency_applied)
    model.metrics["recency_weighting_scheme"] = (
        "composite_hpo"
        if _active_recency_hpo_metrics
        else ("legacy_exp" if bool(LGBM_RECENCY_WEIGHTING) else "none")
    )
    model.metrics["recency_half_life_days"] = float(
        _active_recency_hpo_metrics["half_life_days"]
        if _active_recency_hpo_metrics
        else _recency_half_life_days(objective_mode)
    )
    if _active_recency_hpo_metrics:
        model.metrics["recency_hpo_composite_weight"] = float(
            _active_recency_hpo_metrics["composite_weight"]
        )
        model.metrics["recency_hpo_source"] = str(
            _active_recency_hpo_metrics.get("source", "")
        )
        model.metrics["recency_hpo_legacy_recency_disabled"] = True
    model.metrics["true_soft_labels_enabled"] = bool(LGBM_TRUE_SOFT_LABELS)
    model.metrics["effective_class_rebalance_enabled"] = bool(LGBM_REBALANCE_EFFECTIVE_CLASSES)
    model.metrics["label_weight_hpo_report"] = _json_sanitize(label_weight_hpo_report)
    model.metrics.update(label_diag)
    model.metrics["cv_mode"] = (
        "interleaved_spread"
        if LGBM_CV_MODE in {"interleaved", "interleaved_spread"}
        else (
            "purged_time"
            if (bool(LGBM_PURGED_CV) or LGBM_CV_MODE == "purged_time")
            else "shuffled"
        )
    )
    model.metrics["cv_splits"] = int(LGBM_CV_SPLITS)
    model.metrics["purge_hours"] = float(
        LGBM_PURGE_HOURS
        if (bool(LGBM_PURGED_CV) or LGBM_CV_MODE == "purged_time")
        else 0.0
    )
    model.metrics["best_params"] = dict(best_params)
    model.metrics["hpo_objective_mode"] = objective_mode
    model.metrics["lgbm_meta_feature_names"] = list(model.meta_feature_names)
    model.metrics["lgbm_meta_feature_count"] = int(len(model.meta_feature_names))
    model.metrics["lgbm_internal_metric_feature_names"] = list(LGBM_INTERNAL_METRIC_FEATURE_NAMES)
    model.metrics["lgbm_internal_metric_feature_count"] = int(len(LGBM_INTERNAL_METRIC_FEATURE_NAMES))
    meta_path = meta_feature_output_path or os.environ.get("EPM_LGBM_META_FEATURE_OUTPUT_PATH")
    _save_lgbm_meta_features(model, meta_path)
    ref_dir = reference_artifact_dir or os.environ.get("EPM_LGBM_REFERENCE_ARTIFACT_DIR")
    if ref_dir is None and meta_path:
        meta_path_obj = Path(meta_path)
        ref_dir = meta_path_obj.parent / "lgbm_reference" / meta_path_obj.stem
    if bool(label_weight_hpo_report.get("enabled", False)):
        _save_label_weight_hpo_report(label_weight_hpo_report, ref_dir)
    if LGBM_SKIP_REFERENCE_ARTIFACTS:
        tprint("LGBM reference artifacts skipped by EPM_LGBM_SKIP_REFERENCE_ARTIFACTS=1.")
    else:
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
        pre_metrics.update(_vol_normalized_tp_sl_precision_metrics(pre_final_oof, label_context))
        pre_metrics.update(
            _lgbm_regime_specialist_current_metrics(
                y_metric,
                pre_final_oof,
                regime_specialist_bundle,
                classifier=classifier,
                groups=stability_groups,
                returns=ret_arr,
                label_context=label_context,
                label_context_total_rows=n,
            )
        )
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
    cfg: dict[str, Any] | None = None,
    label_context: dict[str, Any] | None = None,
    assessment_X: Any = None,
    assessment_timestamps: Any = None,
    assessment_assets: Any = None,
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
        cfg=cfg,
        label_context=label_context,
        assessment_X=assessment_X,
        assessment_timestamps=assessment_timestamps,
        assessment_assets=assessment_assets,
    )
    if candidate is None:
        return None
    if LGBM_OPTUNA_CANDIDATE_ONLY:
        selected_features = [str(c) for c in candidate.get("selected_feature_names", []) or []]
        oof = np.asarray(candidate.get("oof_probs"), dtype=np.float32)
        if oof.ndim != 1 or len(oof) != len(y):
            raise ValueError(
                "LGBM Optuna candidate-only mode requires 1-D candidate OOF "
                f"with len={len(y)}, got shape={getattr(oof, 'shape', None)}"
            )
        y_metric = _coerce_target(hard_labels, mode == "classifier") if hard_labels is not None else _coerce_target(y, mode == "classifier")
        ret_arr = _as_returns(y_metric, returns)
        fill = float(np.nanmean(oof)) if np.isfinite(oof).any() else float(np.mean(y_metric))
        oof_filled = np.nan_to_num(oof, nan=fill).astype(np.float32)
        rank_pct = _safe_rank_pct(oof_filled)
        meta_oof_features = pd.DataFrame(index=np.arange(len(oof_filled)), columns=LGBM_META_FEATURE_NAMES, dtype=np.float32)
        meta_oof_features["lgbm_prob"] = oof_filled
        meta_oof_features["rank_pct"] = rank_pct
        model = LGBMStabilityModel(mode=mode)
        model.selected_features = selected_features
        model.meta_leaf_lite_diagnostics_enabled = bool(
            LGBM_META_LEAF_LITE_DIAGNOSTICS or LGBM_META_LEAF_DIAGNOSTICS
        )
        model.meta_leaf_support_diagnostics_enabled = bool(
            LGBM_META_LEAF_SUPPORT_DIAGNOSTICS or LGBM_META_LEAF_DIAGNOSTICS
        )
        model.meta_leaf_target_diagnostics_enabled = bool(
            LGBM_META_LEAF_TARGET_DIAGNOSTICS or LGBM_META_LEAF_DIAGNOSTICS
        )
        model.meta_leaf_centroid_diagnostics_enabled = bool(
            LGBM_META_LEAF_CENTROID_DIAGNOSTICS or LGBM_META_LEAF_DIAGNOSTICS
        )
        model.meta_leaf_diagnostics_enabled = bool(LGBM_META_LEAF_DIAGNOSTICS)
        model.meta_contrib_diagnostics_enabled = bool(LGBM_META_CONTRIB_DIAGNOSTICS)
        model.meta_contrib_method = str(LGBM_META_CONTRIB_METHOD)
        model.meta_score_path_diagnostics_enabled = bool(LGBM_META_SCORE_PATH_DIAGNOSTICS)
        model.meta_drift_features_enabled = bool(LGBM_META_DRIFT_FEATURES)
        model.meta_context_features_enabled = bool(LGBM_FINAL_OOF_CONTEXT_FEATURES)
        model.meta_contrib_context_features_enabled = bool(LGBM_FINAL_OOF_CONTRIB_CONTEXT_FEATURES)
        model.meta_raw_state_context_features_enabled = bool(LGBM_FINAL_OOF_RAW_STATE_CONTEXT_FEATURES)
        model.meta_base_error_context_features_enabled = bool(LGBM_FINAL_OOF_BASE_ERROR_CONTEXT_FEATURES)
        model.input_feature_names = selected_features
        model.oof_probs = oof.astype(np.float32)
        model.meta_oof_features = meta_oof_features.reindex(columns=model.meta_feature_names, fill_value=0.0).astype(np.float32)
        model.rank_bin_stats_oof = _fit_rank_bin_stats_oof(y_metric, rank_pct, classifier=(mode == "classifier"), returns=ret_arr)
        model.metrics = dict(candidate.get("metrics") or {})
        model.metrics["feature_count"] = int(len(selected_features))
        model.metrics["selected_features_count"] = int(len(selected_features))
        model.metrics["selected_features_preview"] = list(selected_features[:50])
        model.metrics["lgbm_meta_leaf_lite_diagnostics_enabled"] = bool(model.meta_leaf_lite_diagnostics_enabled)
        model.metrics["lgbm_meta_leaf_support_diagnostics_enabled"] = bool(model.meta_leaf_support_diagnostics_enabled)
        model.metrics["lgbm_meta_leaf_target_diagnostics_enabled"] = bool(model.meta_leaf_target_diagnostics_enabled)
        model.metrics["lgbm_meta_leaf_centroid_diagnostics_enabled"] = bool(model.meta_leaf_centroid_diagnostics_enabled)
        model.metrics["lgbm_meta_leaf_diagnostics_enabled"] = bool(model.meta_leaf_diagnostics_enabled)
        model.metrics["lgbm_meta_leaf_max_trees"] = int(LGBM_META_LEAF_MAX_TREES)
        model.metrics["lgbm_meta_contrib_diagnostics_enabled"] = bool(model.meta_contrib_diagnostics_enabled)
        model.metrics["lgbm_meta_contrib_method"] = str(model.meta_contrib_method)
        model.metrics["lgbm_meta_score_path_diagnostics_enabled"] = bool(model.meta_score_path_diagnostics_enabled)
        model.metrics["lgbm_meta_drift_features_enabled"] = bool(model.meta_drift_features_enabled)
        model.metrics["lgbm_meta_drift_max_rows"] = int(LGBM_META_DRIFT_MAX_ROWS)
        model.metrics["lgbm_meta_drift_max_features"] = int(LGBM_META_DRIFT_MAX_FEATURES)
        model.metrics["lgbm_meta_context_features_enabled"] = bool(model.meta_context_features_enabled)
        model.metrics["lgbm_meta_contrib_context_features_enabled"] = bool(
            model.meta_contrib_context_features_enabled
        )
        model.metrics["lgbm_meta_raw_state_context_features_enabled"] = bool(
            model.meta_raw_state_context_features_enabled
        )
        model.metrics["lgbm_meta_base_error_context_features_enabled"] = bool(
            model.meta_base_error_context_features_enabled
        )
        model.metrics["optuna_candidate_only"] = True
        model.metrics["final_fit_train_rows"] = 0
        model.metrics["final_fit_train_rows_total"] = int(len(y))
        model.metrics["final_model_count"] = 0
        model.metrics["final_ensemble_sequential_distillation"] = False
        model.pruning_history = list(candidate.get("pruning_history") or [])
        tprint(
            "LGBM Optuna candidate-only fast exit enabled; "
            f"skipping full fit/final OOF/reference artifacts with features={len(selected_features)}."
        )
        return model
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
        cfg=cfg,
        label_context=label_context,
        assessment_X=assessment_X,
        assessment_timestamps=assessment_timestamps,
        assessment_assets=assessment_assets,
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
    "apply_label_weight_hpo_report_to_arrays",
    "train_lgbm_stability_pipeline",
    "run_lgbm_recency_hpo_fixed_contract",
    "tail_control_frames_from_oof",
    "export_tail_control_reports",
    "train_base",
    "train_meta",
    "LGBM_META_FEATURE_NAMES",
    "LGBM_INTERNAL_METRIC_FEATURE_NAMES",
    "BASE_ERROR_ARCHETYPE_FEATURE_NAMES",
    "score_for_trading",
]
