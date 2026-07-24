#!/usr/bin/env python3
"""Month-forward train_meta smoke for S52 regime handoff artifacts.

This is a controlled diagnostic, not production train_meta.  It consumes the
row-level ``train_meta_regime_handoff.parquet`` produced by
``report_s52_trailing_regime_meta_handoff.py`` and evaluates whether those
pre-entry regime/source/action features help filter a base candidate frontier.

Leakage contract:

* model inputs come from the handoff artifact plus the base score only;
* outcome columns are joined only as train labels and validation metrics;
* each validation month is scored by models fit on strictly earlier months;
* top-k precision/EV/path metrics are primary, AUC is secondary.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd

from extreme_price_movements.meta_input_contract import (
    materialize_legacy_constant_zeros,
    require_encoded_meta_matrix,
    require_resolved_meta_input_contract,
    resolve_meta_input_contract,
)
from extreme_price_movements.hierarchical_label_weights import (
    TargetStrengthWeightSpec,
    build_target_strength_weights,
)
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from lightgbm import LGBMClassifier, LGBMRegressor

    _LIGHTGBM_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None
    LGBMRegressor = None
    _LIGHTGBM_AVAILABLE = False


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.lgbm_pipeline import (
    LGBM_HPO_SAMPLE_ROWS,
    LGBM_TWO_PHASE_SELECTION_CONTRACT,
    LGBM_TWO_PHASE_FULL_FIT_ROW_CAP,
    LGBM_TWO_PHASE_SELECTION_SAMPLE_ROWS,
    train_lgbm_stability_candidate,  # noqa: E402
    use_canonical_two_phase_feature_selection,
)
from scripts.run_label_weighted_proxy_ablation import (
    _weight_series as _base_weight_series,  # noqa: E402
)
from scripts.materialize_meta_bme_feature_selection_sample import (  # noqa: E402
    materialize_sample as materialize_meta_bme_sample,
)

DEFAULT_REPORT_ROOT = Path(
    "data_perp/reports/s52_trailing_profit_best_pointwise_scored_ledger_20260705_v1"
)
DEFAULT_HANDOFF_DIR = DEFAULT_REPORT_ROOT / "s52_trailing_regime_meta_handoff_v1"
DEFAULT_LEDGER = DEFAULT_HANDOFF_DIR / "s52_trailing_regime_scored_ledger.parquet"
DEFAULT_OUT_DIR = DEFAULT_HANDOFF_DIR / "train_meta_regime_handoff_smoke_v1"
DEFAULT_META_CHAMPION_CONTRACT = (
    ROOT
    / "extreme_price_movements"
    / "config"
    / "meta_v9_anchor_oldparams_residual_backbone_v1.json"
)

KEY_COLUMNS = ("__ts__", "__symbol__", "side_name")
HANDOFF_RANK_SCOPE = "timestamp_side"
HANDOFF_RANK_SCOPE_COLUMN = "candidate_handoff_rank_scope"
BASE_TARGET_CONTRACT_HASH_COLUMN = "base_target_contract_hash"
BASE_SAMPLE_WEIGHT_SPEC_HASH_COLUMN = "base_sample_weight_spec_hash"
HANDOFF_PROVENANCE_COLUMNS = (
    HANDOFF_RANK_SCOPE_COLUMN,
    BASE_TARGET_CONTRACT_HASH_COLUMN,
    BASE_SAMPLE_WEIGHT_SPEC_HASH_COLUMN,
)
TOP_KEEP_FRACTIONS = (1.00, 0.50, 0.30, 0.20, 0.15, 0.10, 0.05)
POLICY_BUDGET_FRACTIONS = (0.30, 0.20, 0.15, 0.10)
BAD_PATH_CAPS = (0.45, 0.50, 0.55, 0.60, 0.65)
CLEAN_EXEC_FLOORS = (0.0, 0.45, 0.55, 0.65)
POSITIVE_MARGIN_FLOORS = (0.0, 0.45, 0.55)
OUTCOME_COLUMNS = (
    "__decision_ts__",
    "__first_path_ts__",
    "__label_path_end_ts__",
    "__first_touch_bar__",
    "__first_touch_target_soft__",
    "__first_touch_policy_soft__",
    "base_model_target_mode",
    "exec_margin",
    "ev_after_1pct",
    "first_touch_gross",
    "first_touch_bad_mae_1r",
    "full_path_bad_mae_1r",
    "timeout",
    "mfe_before_mae_1r",
    "mae_before_mfe_1r",
    "clean_exec",
    "dirty_positive",
    "u_policy_net",
    "ret_net",
    "mae_norm",
    "mfe_norm",
    "first_touch_net",
    "first_touch_full_path_mae_norm",
    "underwater_bars_before_mfe_1r",
    "long_path_full_bad_mae_1r",
    "long_path_time_to_profit_bars",
    "long_path_slow_profit",
    "long_path_post_mfe_drawdown_norm",
    "long_path_post_mfe_bad_drawdown",
    "long_path_clean_exec_label",
    "long_path_dirty_positive_label",
    "long_path_quality_soft",
    "long_bad_path_label",
)
BASE_SOFT_LABEL_COLUMNS = (
    "__first_touch_target_soft__",
    "target_soft",
    "__target_soft__",
)
ALL_META_POST_SELECTION_OOD_FEATURE_NAMES = (
    "meta_sel_ood_abs_z_mean",
    "meta_sel_ood_abs_z_max",
    "meta_sel_ood_abs_z_p95",
    "meta_sel_ood_iqr_exceed_frac",
    "meta_sel_ood_missing_frac",
    "meta_sel_ood_centroid_l2",
)
# Post-selection OOD metrics are derived from the chosen core contract.  They
# must earn admission in a second conditional MDA pass; keeping a hard-coded
# pair here would make the meta contract drift-heavy by construction.
META_POST_SELECTION_OOD_FEATURE_NAMES: tuple[str, ...] = ()
LEDGER_CONTEXT_COLUMNS = (
    "__archetype_label_family__",
    "__archetype_label_source__",
    "__archetype_policy_key__",
    "__archetype_policy_role__",
    "__archetype_policy_confidence__",
    "__archetype_policy_tp_r__",
    "__archetype_policy_sl_r__",
    "__archetype_policy_trail_r__",
    "__archetype_policy_max_bars_to_mfe__",
    "__archetype_policy_max_barrier__",
)
LEDGER_CONTEXT_FEATURE_ALIASES = {
    "__archetype_label_family__": "archetype_label_family",
    "__archetype_label_source__": "archetype_label_source",
    "__archetype_policy_key__": "archetype_policy_key",
    "__archetype_policy_role__": "archetype_policy_role",
    "__archetype_policy_confidence__": "archetype_policy_confidence",
    "__archetype_policy_tp_r__": "archetype_policy_tp_r",
    "__archetype_policy_sl_r__": "archetype_policy_sl_r",
    "__archetype_policy_trail_r__": "archetype_policy_trail_r",
    "__archetype_policy_max_bars_to_mfe__": "archetype_policy_max_bars_to_mfe",
    "__archetype_policy_max_barrier__": "archetype_policy_max_barrier",
}
NEVER_FEATURE_COLUMNS = {
    "__ts__",
    "__symbol__",
    "month",
    "selected_top10",
    "selected_top20",
    "selected_top30",
    *HANDOFF_PROVENANCE_COLUMNS,
    "base_target_contract",
    "base_target_contract_json",
    "base_sample_weight_spec",
    "base_sample_weight_spec_json",
    "base_model_weight_arm",
    "score",
    "__decision_ts__",
    "__first_path_ts__",
    "__label_path_end_ts__",
    "__first_touch_bar__",
}
BASE_PRIOR_NUMERIC_FEATURES = (
    "base_margin_to_cutoff",
    "base_margin_to_cutoff_z",
    "base_signal_zscore_within_archetype",
    "base_score_rank_pct_train_prior",
)
BASE_PRIOR_CATEGORICAL_FEATURES = (
    "base_rank_band",
    "base_margin_band",
)
# Meta must always condition context on the exact base opportunity signal.  These
# are observable at decision time and are not outcome-derived priors.
META_PROTECTED_BASE_FEATURES = (
    # ``score_base`` is the canonical materialized handoff name.  Keep the
    # legacy alias for old ledgers, but never rely on an unaliased ``score``
    # column being present.
    "score_base",
    "base_score_raw",
    "base_rank_pct_by_timestamp_side",
    "base_score_rank_pct_train_prior",
    "base_margin_to_cutoff",
    "base_margin_to_cutoff_z",
    "base_signal_zscore_within_archetype",
)

# The global meta head must know which side it is scoring.  This is a model
# structure input, not a feature-selection hypothesis, so retain its encoded
# dummies. Archetype descriptors instead remain MDA candidates below.
META_STRUCTURAL_FEATURE_PREFIXES = ("side_name_",)

# Conditional/non-monotonic features may be useless on a univariate screen but
# useful through local state interactions. They bypass only univariate, Relief
# and redundancy pre-screens; iterative MDA still decides final admission.
META_PRE_MDA_BYPASS_PREFIXES = (
    "archetype_label_family_",
    "archetype_policy_key_",
    "archetype_policy_role_",
    "source_semantic_family_",
    "source_semantic_family_base_",
    "aegmm_cluster_",
    "side_aegmm_cluster_",
    "reconstruction_bin_",
    "gmm_",
    "dae_",
    "ae_",
    "cluster_",
    "mahalanobis_",
    "expected_mahalanobis",
    "meta_sel_ood_",
    "support_",
    "base_arch_hit_",
    "rel_",
    "leaf_",
    "drift_",
    "state_spectral_",
)


def _with_meta_protected_base_features(
    features: Sequence[str] | None,
    *,
    available: Sequence[str] | None = None,
) -> list[str]:
    """Return a stable meta contract with observable base-score anchors first."""
    available_set = None if available is None else {str(value) for value in available}
    anchors = [
        feature
        for feature in META_PROTECTED_BASE_FEATURES
        if available_set is None or feature in available_set
    ]
    return list(
        dict.fromkeys(
            [*anchors, *(str(feature) for feature in (features or []) if str(feature).strip())]
        )
    )


def _meta_structural_features(columns: Sequence[str]) -> list[str]:
    """Return encoded side columns required by a shared long/short meta head."""
    return [
        str(column)
        for column in columns
        if str(column).startswith(META_STRUCTURAL_FEATURE_PREFIXES)
    ]


def _meta_pre_mda_bypass_features(columns: Sequence[str]) -> list[str]:
    """Keep conditional state/archetype features available for MDA.

    These are deliberately not protected final features. Their admission still
    requires incremental, side-aware MDA support after the base-score anchors
    and side dummies are present.
    """
    return [
        str(column)
        for column in columns
        if str(column).startswith(META_PRE_MDA_BYPASS_PREFIXES)
    ]
RELIABILITY_NUMERIC_FEATURES = (
    "rel_rankband_rows_log1p",
    "rel_rankband_clean_rate",
    "rel_rankband_bad_mae_rate",
    "rel_rankband_timeout_rate",
    "rel_rankband_dirty_positive_rate",
    "rel_rankband_exec_margin_mean",
    "rel_rankband_edge",
    "rel_marginband_rows_log1p",
    "rel_marginband_clean_rate",
    "rel_marginband_bad_mae_rate",
    "rel_marginband_timeout_rate",
    "rel_marginband_dirty_positive_rate",
    "rel_marginband_exec_margin_mean",
    "rel_marginband_edge",
)
SUPPORT_DRIFT_NUMERIC_FEATURES = (
    "support_min_log_count",
    "support_mean_log_count",
    "support_min_frequency",
    "support_mean_frequency",
    "support_unseen_bucket_count",
    "support_unseen_bucket_share",
    "support_rare_bucket_count",
    "support_rare_bucket_share",
)
HIT_SURPRISE_HALFLIFE_DAYS = (3.0, 7.0, 14.0)
HIT_SURPRISE_NUMERIC_FEATURES = tuple(
    name
    for hl in (3, 7, 14)
    for name in (
        f"base_arch_hit_recent_rate_hl{hl}d",
        f"base_arch_hit_expected_rate_hl{hl}d",
        f"base_arch_hit_surprise_hl{hl}d",
        f"base_arch_hit_surprise_z_hl{hl}d",
        f"base_arch_hit_support_log1p_hl{hl}d",
        f"base_arch_hit_effective_n_hl{hl}d",
    )
)
SUPPORT_DRIFT_COLUMNS = (
    "source_tag",
    "source_semantic_family",
    "source_semantic_family_base",
    "long_source_regime_split",
    "aegmm_cluster",
    "side_aegmm_cluster",
    "aegmm_entropy_bin",
    "aegmm_distance_bin",
    "aegmm_expected_distance_bin",
    "reconstruction_bin",
    "dae_reconstruction_bin",
    "latent_speed_bin",
    "regime_lgbm_leaf_bad_mae_k4",
    "regime_lgbm_leaf_exec_margin_k4",
    "regime_first_touch_bad_mae_score_bin",
    "regime_timeout_score_bin",
    "regime_dirty_positive_score_bin",
    "regime_clean_exec_score_bin",
)
DEFAULT_LGBM_CLASSIFIER_PARAMS = {
    "n_estimators": 128,
    "learning_rate": 0.02328311652285508,
    "num_leaves": 23,
    "max_depth": 8,
    "min_child_samples": 87,
    "subsample": 0.8694916518767465,
    "colsample_bytree": 0.5528386863204131,
    "reg_alpha": 1.4263322282358653,
    "reg_lambda": 3.484420244989796,
}
DEFAULT_LGBM_REGRESSOR_PARAMS = dict(DEFAULT_LGBM_CLASSIFIER_PARAMS)
META_HPO_PRESETS = (
    {
        "name": "base_winner_target_soft_w7",
        "classifier": {
            "n_estimators": 147,
            "learning_rate": 0.023186706725487574,
            "num_leaves": 15,
            "max_depth": 4,
            "min_child_samples": 98,
            "subsample": 0.7858416187773574,
            "colsample_bytree": 0.9119108731134247,
            "reg_alpha": 0.027300426078104418,
            "reg_lambda": 0.34080226291305005,
        },
        "regressor": {
            "n_estimators": 147,
            "learning_rate": 0.023186706725487574,
            "num_leaves": 15,
            "max_depth": 4,
            "min_child_samples": 98,
            "subsample": 0.7858416187773574,
            "colsample_bytree": 0.9119108731134247,
            "reg_alpha": 0.027300426078104418,
            "reg_lambda": 0.34080226291305005,
        },
    },
    {
        "name": "baseline",
        "classifier": DEFAULT_LGBM_CLASSIFIER_PARAMS,
        "regressor": DEFAULT_LGBM_REGRESSOR_PARAMS,
    },
    {
        "name": "regularized_slow",
        "classifier": {
            "n_estimators": 260,
            "learning_rate": 0.022,
            "num_leaves": 15,
            "max_depth": 4,
            "min_child_samples": 60,
            "subsample": 0.78,
            "colsample_bytree": 0.70,
            "reg_alpha": 0.25,
            "reg_lambda": 14.0,
        },
        "regressor": {
            "n_estimators": 300,
            "learning_rate": 0.022,
            "num_leaves": 15,
            "max_depth": 4,
            "min_child_samples": 60,
            "subsample": 0.78,
            "colsample_bytree": 0.70,
            "reg_alpha": 0.25,
            "reg_lambda": 16.0,
        },
    },
    {
        "name": "wider_context",
        "classifier": {
            "n_estimators": 220,
            "learning_rate": 0.028,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 45,
            "subsample": 0.82,
            "colsample_bytree": 0.82,
            "reg_alpha": 0.08,
            "reg_lambda": 8.0,
        },
        "regressor": {
            "n_estimators": 260,
            "learning_rate": 0.028,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 45,
            "subsample": 0.82,
            "colsample_bytree": 0.82,
            "reg_alpha": 0.08,
            "reg_lambda": 10.0,
        },
    },
)
_FEATURE_SELECTION_CACHE: dict[
    tuple[Any, ...],
    tuple[list[str], pd.DataFrame, dict[str, list[str]]],
] = {}
_JOINED_FRAME_CACHE: dict[tuple[Any, ...], pd.DataFrame] = {}
_HPO_FOLD_MATRIX_CACHE: dict[
    tuple[Any, ...],
    tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        list[str],
        dict[str, Any],
    ],
] = {}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _safe_artifact_stem(value: Any) -> str:
    text = str(value)
    stem = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)
    return stem.strip("_") or "artifact"


def _feature_contract_hash(feature_names: list[str]) -> str:
    payload = json.dumps(list(feature_names), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _contract_hash(payload: Any) -> str:
    encoded = json.dumps(
        _json_safe(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _handoff_contract_path(handoff_path: Path) -> Path:
    return Path(handoff_path).parent / "train_meta_regime_handoff_contract.json"


def _unique_nonempty_strings(values: Any) -> list[str]:
    series = pd.Series(values, copy=False).dropna().astype(str).str.strip()
    return sorted(value for value in series.unique() if value)


def _target_strength_spec_from_contract(
    contract: dict[str, Any] | None,
) -> TargetStrengthWeightSpec | dict[str, TargetStrengthWeightSpec] | None:
    raw = dict((contract or {}).get("base_sample_weight_spec", {}) or {})
    if str(raw.get("schema", "")) == "base_weight_arm_v1":
        return None
    by_side = raw.get("spec_by_side")
    if isinstance(by_side, dict):
        return {
            side: TargetStrengthWeightSpec(**{
                key: value
                for key, value in dict(by_side.get(side, {})).items()
                if key in TargetStrengthWeightSpec.__dataclass_fields__
            })
            for side in ("long", "short")
        }
    spec_payload = raw.get("spec", raw)
    if not isinstance(spec_payload, dict):
        return TargetStrengthWeightSpec()
    allowed = set(TargetStrengthWeightSpec.__dataclass_fields__)
    kwargs = {key: value for key, value in spec_payload.items() if key in allowed}
    try:
        return TargetStrengthWeightSpec(**kwargs)
    except (TypeError, ValueError):
        return TargetStrengthWeightSpec()


def _base_weight_arm_from_contract(contract: dict[str, Any] | None) -> str | None:
    raw = dict((contract or {}).get("base_sample_weight_spec", {}) or {})
    if str(raw.get("schema", "")) != "base_weight_arm_v1":
        return None
    arm = str(raw.get("weight_arm", "")).strip()
    return arm or None


def _serialize_inherited_weighting(
    inherited_weight_spec: (
        TargetStrengthWeightSpec | dict[str, TargetStrengthWeightSpec] | None
    ),
    inherited_base_weight_arm: str | None,
) -> dict[str, Any]:
    """Serialize the exact inherited base weighting contract without inventing one."""

    if isinstance(inherited_weight_spec, dict):
        spec_payload: dict[str, Any] | None = {
            side: spec.__dict__ for side, spec in inherited_weight_spec.items()
        }
    elif inherited_weight_spec is not None:
        spec_payload = inherited_weight_spec.__dict__
    else:
        spec_payload = None
    return {
        "schema": (
            "base_weight_arm_v1"
            if inherited_weight_spec is None and inherited_base_weight_arm
            else "target_strength_weight_v1"
        ),
        "weight_arm": str(inherited_base_weight_arm) if inherited_base_weight_arm else None,
        "spec": _json_safe(spec_payload),
    }


def _load_and_validate_handoff_contract(
    *,
    handoff_path: Path,
    handoff_rows: pd.DataFrame,
    strict: bool,
) -> dict[str, Any]:
    """Validate rank/target/weight provenance before meta training.

    The full contracts are sidecar metadata.  The row-level hashes prove that
    the exact selected candidate stream has one uniform base provenance.
    """

    path = _handoff_contract_path(handoff_path)
    raw: dict[str, Any] = {}
    label_resolution_contract: dict[str, Any] = {}
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"Invalid handoff contract JSON: {path}")
        label_resolution_contract = dict(
            loaded.get("label_resolution_contract", {}) or {}
        )
        raw = dict(loaded.get("inherited_base_contract", loaded))
    elif strict:
        raise ValueError(f"Strict meta handoff requires contract sidecar: {path}")

    scopes = _unique_nonempty_strings(handoff_rows.get(HANDOFF_RANK_SCOPE_COLUMN))
    if scopes and scopes != [HANDOFF_RANK_SCOPE]:
        raise ValueError(
            "candidate_handoff_rank_scope must be timestamp_side; "
            f"found row values {scopes}"
        )
    declared_scope = str(raw.get(HANDOFF_RANK_SCOPE_COLUMN, "")).strip()
    if declared_scope and declared_scope != HANDOFF_RANK_SCOPE:
        raise ValueError(
            "handoff contract candidate_handoff_rank_scope must be "
            f"timestamp_side, got {declared_scope!r}"
        )

    target_contract = raw.get("base_target_contract")
    weight_contract = raw.get("base_sample_weight_spec")
    expected_target_hash = str(raw.get(BASE_TARGET_CONTRACT_HASH_COLUMN, "")).strip()
    expected_weight_hash = str(
        raw.get(BASE_SAMPLE_WEIGHT_SPEC_HASH_COLUMN, "")
    ).strip()
    target_hashes = _unique_nonempty_strings(
        handoff_rows.get(BASE_TARGET_CONTRACT_HASH_COLUMN)
    )
    weight_hashes = _unique_nonempty_strings(
        handoff_rows.get(BASE_SAMPLE_WEIGHT_SPEC_HASH_COLUMN)
    )
    if strict:
        if scopes != [HANDOFF_RANK_SCOPE] or declared_scope != HANDOFF_RANK_SCOPE:
            raise ValueError(
                "Strict meta handoff requires timestamp_side rank scope on both "
                "the row stream and sidecar."
            )
        if not isinstance(target_contract, dict) or not isinstance(weight_contract, dict):
            raise ValueError(
                "Strict meta handoff requires explicit base target and sample-weight contracts."
            )
        if not expected_target_hash or not expected_weight_hash:
            raise ValueError(
                "Strict meta handoff requires base target and sample-weight contract hashes."
            )
        if target_hashes != [expected_target_hash] or weight_hashes != [expected_weight_hash]:
            raise ValueError(
                "Strict meta handoff found missing or mixed base contract hashes: "
                f"target={target_hashes}, weight={weight_hashes}."
            )
        if _contract_hash(target_contract) != expected_target_hash:
            raise ValueError("Base target contract sidecar hash does not match its payload.")
        if _contract_hash(weight_contract) != expected_weight_hash:
            raise ValueError("Base sample-weight contract sidecar hash does not match its payload.")
        if str(weight_contract.get("schema", "")) not in {
            "target_strength_weight_v1",
            "base_weight_arm_v1",
        }:
            raise ValueError(
                "Strict meta handoff requires target_strength_weight_v1 or "
                "base_weight_arm_v1 sample weights."
            )
        if (
            label_resolution_contract.get("schema")
            != "forward_label_resolution_v1"
            or label_resolution_contract.get("resolution_column")
            != "__label_path_end_ts__"
            or "__label_path_end_ts__" not in handoff_rows.columns
        ):
            raise ValueError(
                "Strict meta handoff requires the forward_label_resolution_v1 "
                "contract and __label_path_end_ts__ rows."
            )
    return {
        "path": str(path),
        "strict": bool(strict),
        "sidecar_present": bool(path.is_file()),
        "candidate_handoff_rank_scope": HANDOFF_RANK_SCOPE,
        "row_rank_scopes": scopes,
        "declared_rank_scope": declared_scope or None,
        "base_target_contract": target_contract if isinstance(target_contract, dict) else {},
        BASE_TARGET_CONTRACT_HASH_COLUMN: expected_target_hash or None,
        "row_target_contract_hashes": target_hashes,
        "base_sample_weight_spec": weight_contract if isinstance(weight_contract, dict) else {},
        BASE_SAMPLE_WEIGHT_SPEC_HASH_COLUMN: expected_weight_hash or None,
        "row_sample_weight_spec_hashes": weight_hashes,
        "validation_status": "strict_pass" if strict else "non_strict",
        "label_resolution_contract": label_resolution_contract,
    }


def _meta_weight_archetypes(train: pd.DataFrame) -> pd.Series:
    side = train.get("side_name", pd.Series("unknown", index=train.index)).astype(str)
    for column in (
        "archetype_label_family",
        "policy_archetype",
        "local_side_archetype",
        "__archetype_label_family__",
        "source_tag",
    ):
        if column in train.columns:
            return side + "__" + train[column].astype(str).fillna("missing")
    return side + "__missing"


def _train_only_target_strength_weights(
    train: pd.DataFrame,
    target: pd.Series,
    *,
    sample_weight_spec: TargetStrengthWeightSpec | dict[str, TargetStrengthWeightSpec],
    strict: bool,
) -> tuple[pd.Series, dict[str, Any]]:
    """Recompute inherited target-strength weights on this fold's top30 rows."""

    candidate_col = _candidate_column("top30")
    if candidate_col in train.columns:
        selected = train[candidate_col].fillna(False).astype(bool)
        if strict and not bool(selected.all()):
            raise ValueError(
                "Strict meta training must receive only top30 candidate rows; "
                f"found {int((~selected).sum())} non-candidates in the train fold."
            )
    elif strict:
        raise ValueError("Strict meta handoff requires selected_top30 on train rows.")
    timestamps = pd.to_datetime(train.get("__ts__"), utc=True, errors="coerce")
    if strict and timestamps.isna().any():
        raise ValueError("Strict meta handoff has invalid candidate timestamps.")
    if isinstance(sample_weight_spec, dict):
        side_names = train.get("side_name", pd.Series("long", index=train.index)).astype(str)
        weights = pd.Series(1.0, index=train.index, dtype=np.float32)
        diagnostics: dict[str, Any] = {"schema": "side_target_strength_weight_v1", "by_side": {}}
        for side in ("long", "short"):
            mask = side_names.eq(side)
            if not bool(mask.any()):
                continue
            local, local_diag = build_target_strength_weights(
                target.loc[mask],
                timestamps=timestamps.loc[mask],
                archetypes=_meta_weight_archetypes(train.loc[mask]),
                spec=sample_weight_spec[side],
            )
            weights.loc[mask] = np.asarray(local, dtype=np.float32)
            diagnostics["by_side"][side] = local_diag
        return weights, diagnostics
    weights, diagnostics = build_target_strength_weights(
        target,
        timestamps=timestamps,
        archetypes=_meta_weight_archetypes(train),
        spec=sample_weight_spec,
    )
    return pd.Series(weights, index=train.index, dtype=np.float32), diagnostics


def _save_meta_fold_models(
    *,
    out_dir: Path,
    fold: str,
    calendar_month: str,
    valid_start: Any,
    valid_end: Any,
    fold_idx: int,
    seed: int,
    models: dict[str, Any],
    feature_names: list[str],
    model_feature_names: dict[str, list[str]] | None,
    input_feature_contract: dict[str, Any] | None,
    classifier_params: dict[str, Any],
    regressor_params: dict[str, Any],
    meta_head_mode: str,
    model_profile_name: str,
    train_rows_available: int,
    train_rows_fit: int,
    valid_rows: int,
    target_columns_used: set[str],
    preprocessing_state: dict[str, Any] | None = None,
    inherited_handoff_contract: dict[str, Any] | None = None,
    target_strength_weight_diagnostics: dict[str, Any] | None = None,
    label_purge_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist fold-fitted meta models and their exact feature contract."""

    fold_dir = out_dir / "models" / _safe_artifact_stem(fold)
    fold_dir.mkdir(parents=True, exist_ok=True)
    saved: list[dict[str, Any]] = []
    for label, model in sorted(models.items()):
        model_path = fold_dir / f"{_safe_artifact_stem(label)}.joblib"
        joblib.dump(model, model_path, compress=3)
        saved.append(
            {
                "label": str(label),
                "path": str(model_path),
                "model_class": type(model).__name__,
                "module": type(model).__module__,
            }
        )
    columns_path = fold_dir / "columns.json"
    columns_payload = {
        "schema": "s52_meta_fold_feature_contract_v1",
        "feature_names": list(feature_names),
        "feature_count": int(len(feature_names)),
        "feature_contract_hash": _feature_contract_hash(list(feature_names)),
        "feature_names_by_model": {
            str(label): list(names)
            for label, names in (model_feature_names or {}).items()
        },
        "feature_contract_hash_by_model": {
            str(label): _feature_contract_hash(list(names))
            for label, names in (model_feature_names or {}).items()
        },
        "input_feature_contract": _json_safe(input_feature_contract or {}),
        "preprocessing_state": _json_safe(preprocessing_state or {}),
    }
    columns_path.write_text(
        json.dumps(_json_safe(columns_payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest = {
        "schema": "s52_meta_saved_fold_models_v1",
        "fold": str(fold),
        "calendar_month": str(calendar_month),
        "valid_start": valid_start,
        "valid_end": valid_end,
        "fold_idx": int(fold_idx),
        "seed": int(seed),
        "model_profile_name": str(model_profile_name),
        "meta_head_mode": str(meta_head_mode),
        "train_rows_available": int(train_rows_available),
        "train_rows_fit": int(train_rows_fit),
        "valid_rows": int(valid_rows),
        "models": saved,
        "columns_path": str(columns_path),
        "feature_count": int(len(feature_names)),
        "feature_contract_hash": columns_payload["feature_contract_hash"],
        "feature_names_by_model": columns_payload["feature_names_by_model"],
        "feature_contract_hash_by_model": columns_payload[
            "feature_contract_hash_by_model"
        ],
        "input_feature_contract": columns_payload["input_feature_contract"],
        "preprocessing_state": columns_payload["preprocessing_state"],
        "classifier_params": _json_safe(classifier_params),
        "regressor_params": _json_safe(regressor_params),
        "target_columns_used": sorted(str(c) for c in target_columns_used),
        "inherited_base_handoff_contract": _json_safe(
            inherited_handoff_contract or {}
        ),
        "target_strength_weight_diagnostics": _json_safe(
            target_strength_weight_diagnostics or {}
        ),
        "label_purge_contract": _json_safe(label_purge_contract or {}),
        "leakage_contract": {
            "fit_scope": "prior_rows_with_complete_forward_labels_before_this_oos_fold",
            "oos_rows": "valid_start <= timestamp < valid_end",
            "feature_contract": "columns.json is the required inference-time feature order",
            "outcome_columns": "used only for training labels and validation metrics, never as OOS features",
        },
    }
    manifest_path = fold_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    return {**manifest, "manifest_path": str(manifest_path), "model_dir": str(fold_dir)}


def _num(
    values: Any, *, index: pd.Index | None = None, default: float = np.nan
) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    if values is None:
        if index is None:
            return pd.Series(dtype=np.float32)
        return pd.Series(default, index=index, dtype=np.float32)
    return pd.to_numeric(pd.Series(values, index=index), errors="coerce")


def _mean(values: Any) -> float:
    arr = _num(values).replace([np.inf, -np.inf], np.nan).dropna()
    return float(arr.mean()) if len(arr) else float("nan")


def _rate(values: Any) -> float:
    arr = _num(values).replace([np.inf, -np.inf], np.nan).dropna()
    return float(arr.clip(0.0, 1.0).mean()) if len(arr) else float("nan")


def _safe_spearman(y_true: pd.Series, score: pd.Series) -> float:
    y = _num(y_true)
    s = _num(score)
    valid = y.notna() & s.notna()
    if (
        int(valid.sum()) < 20
        or y.loc[valid].nunique(dropna=True) < 2
        or s.loc[valid].nunique(dropna=True) < 2
    ):
        return float("nan")
    return float(
        y.loc[valid].rank(method="average").corr(s.loc[valid].rank(method="average"))
    )


def _safe_auc(y_true: pd.Series, score: pd.Series) -> float:
    y = _num(y_true)
    s = _num(score)
    valid = y.notna() & s.notna()
    if int(valid.sum()) < 20 or y.loc[valid].nunique(dropna=True) < 2:
        return float("nan")
    return float(roc_auc_score(y.loc[valid].astype(int), s.loc[valid]))


def _safe_ap(y_true: pd.Series, score: pd.Series) -> float:
    y = _num(y_true)
    s = _num(score)
    valid = y.notna() & s.notna()
    if int(valid.sum()) < 20 or y.loc[valid].nunique(dropna=True) < 2:
        return float("nan")
    return float(average_precision_score(y.loc[valid].astype(int), s.loc[valid]))


def _candidate_column(frontier: str) -> str:
    normalized = str(frontier).lower().replace("top", "")
    return f"selected_top{int(normalized)}"


def _parquet_schema_columns(path: Path) -> set[str]:
    """Read Parquet column names without materializing any data rows."""
    try:
        import pyarrow.parquet as pq

        return set(map(str, pq.read_schema(path).names))
    except Exception as pyarrow_error:
        try:
            import duckdb

            escaped = str(Path(path).resolve()).replace("'", "''")
            relation = (
                f"read_parquet('{escaped}/*.parquet', union_by_name=true)"
                if Path(path).is_dir()
                else f"read_parquet('{escaped}')"
            )
            rows = duckdb.connect().execute(
                f"DESCRIBE SELECT * FROM {relation}"
            ).fetchall()
            return {str(row[0]) for row in rows}
        except Exception as duckdb_error:
            raise RuntimeError(
                f"Could not inspect Parquet schema without reading rows: {path}"
            ) from duckdb_error


def _load_joined_frame(
    handoff_path: Path,
    ledger_path: Path,
    frontier: str,
    *,
    handoff_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load the candidate handoff, optionally pruning unused raw columns.

    Fixed-feature replays do not need to inflate a full feature universe merely
    to materialize a frozen model contract.  Outcome columns remain sourced
    from the separate ledger below; callers can therefore pass only the raw
    pre-entry columns required by their selected features and fold-derived
    context.
    """
    handoff_schema_cols = _parquet_schema_columns(handoff_path)
    if handoff_columns is None:
        handoff = pd.read_parquet(handoff_path)
    else:
        # ``month`` is redundant with the decision timestamp and some frozen
        # candidate ledgers intentionally omit it.  It is reconstructed below
        # from ``__ts__`` when absent; score and frontier membership are the
        # actual handoff requirements.
        required_handoff = set(KEY_COLUMNS) | {
            "score",
            _candidate_column(frontier),
        }
        requested_handoff = required_handoff | set(HANDOFF_PROVENANCE_COLUMNS) | {
            str(col) for col in handoff_columns if str(col).strip()
        }
        read_handoff = sorted(requested_handoff.intersection(handoff_schema_cols))
        missing_required = sorted(required_handoff.difference(read_handoff))
        if missing_required:
            raise ValueError(f"Handoff missing required columns: {missing_required}")
        handoff = pd.read_parquet(handoff_path, columns=read_handoff)
    ledger_cols = list(KEY_COLUMNS) + ["month", "score", _candidate_column(frontier)]
    ledger_cols += [col for col in OUTCOME_COLUMNS if col not in ledger_cols]
    ledger_cols += [col for col in LEDGER_CONTEXT_COLUMNS if col not in ledger_cols]
    ledger_schema_cols = _parquet_schema_columns(ledger_path)
    ledger = pd.read_parquet(
        ledger_path, columns=[col for col in ledger_cols if col in ledger_schema_cols]
    )
    missing = [
        col
        for col in KEY_COLUMNS
        if col not in handoff.columns or col not in ledger.columns
    ]
    if missing:
        raise ValueError(f"Missing join key columns: {missing}")
    aligned_keys = False
    if len(handoff) == len(ledger):
        try:
            aligned_keys = all(
                handoff[col]
                .reset_index(drop=True)
                .equals(ledger[col].reset_index(drop=True))
                for col in KEY_COLUMNS
                if col in handoff.columns and col in ledger.columns
            )
        except Exception:
            aligned_keys = False
    if aligned_keys:
        merged = handoff
        for col in ledger.columns:
            if col in KEY_COLUMNS:
                continue
            ledger_col = f"{col}__ledger"
            if col not in merged.columns:
                merged[col] = ledger[col].to_numpy(copy=False)
            else:
                merged[ledger_col] = ledger[col].to_numpy(copy=False)
    else:
        merged = handoff.merge(
            ledger,
            on=list(KEY_COLUMNS),
            how="left",
            suffixes=("", "__ledger"),
            validate="one_to_one",
        )
    for col in ("month", "score", _candidate_column(frontier), *LEDGER_CONTEXT_COLUMNS):
        ledger_col = f"{col}__ledger"
        if ledger_col in merged.columns:
            if col not in merged.columns:
                merged[col] = merged[ledger_col]
            else:
                merged[col] = merged[col].where(merged[col].notna(), merged[ledger_col])
            merged = merged.drop(columns=[ledger_col])
    if "month" not in merged.columns or merged["month"].isna().all():
        merged["month"] = (
            pd.to_datetime(merged["__ts__"], utc=True, errors="coerce")
            .dt.to_period("M")
            .astype(str)
        )
    merged["month"] = merged["month"].astype(str)
    selected_col = _candidate_column(frontier)
    if selected_col not in merged.columns:
        raise ValueError(f"Missing candidate frontier column {selected_col!r}")
    merged[selected_col] = _num(merged[selected_col]).fillna(0.0).gt(0.5)
    merged["positive_exec_margin"] = (
        _num(merged.get("exec_margin"), index=merged.index).gt(0.0).astype(float)
    )
    merged["clean_exec_label"] = (
        _num(merged.get("clean_exec"), index=merged.index, default=0.0)
        .fillna(0.0)
        .gt(0.5)
        .astype(float)
    )
    merged["bad_path_label"] = (
        _num(merged.get("full_path_bad_mae_1r"), index=merged.index, default=0.0)
        .fillna(0.0)
        .gt(0.5)
        | _num(merged.get("timeout"), index=merged.index, default=0.0)
        .fillna(0.0)
        .gt(0.5)
    ).astype(float)
    merged["path_mfe_before_mae_label"] = (
        _num(merged.get("mfe_before_mae_1r"), index=merged.index, default=0.0)
        .fillna(0.0)
        .gt(0.5)
        .astype(float)
    )
    merged["path_mae_before_mfe_label"] = (
        _num(merged.get("mae_before_mfe_1r"), index=merged.index, default=0.0)
        .fillna(0.0)
        .gt(0.5)
        .astype(float)
    )
    underwater = _num(
        merged.get("underwater_bars_before_mfe_1r"), index=merged.index, default=np.nan
    )
    merged["path_underwater_duration_target"] = np.log1p(
        underwater.clip(lower=0.0)
    ).astype(np.float32)
    is_long = merged["side_name"].astype(str).str.lower().eq("long")
    if "long_path_clean_exec_label" not in merged.columns:
        merged["long_path_clean_exec_label"] = np.where(
            is_long, merged["clean_exec_label"], np.nan
        ).astype(np.float32)
    if "long_path_dirty_positive_label" not in merged.columns:
        merged["long_path_dirty_positive_label"] = np.where(
            is_long,
            _num(merged.get("dirty_positive"), index=merged.index, default=0.0).fillna(
                0.0
            ),
            np.nan,
        ).astype(np.float32)
    if "long_path_post_mfe_bad_drawdown" not in merged.columns:
        merged["long_path_post_mfe_bad_drawdown"] = np.nan
    if "long_path_slow_profit" not in merged.columns:
        merged["long_path_slow_profit"] = np.nan
    merged["long_bad_path_label"] = np.where(
        is_long,
        (
            _num(
                merged.get("long_path_dirty_positive_label"),
                index=merged.index,
                default=0.0,
            )
            .fillna(0.0)
            .gt(0.5)
            | _num(
                merged.get("long_path_post_mfe_bad_drawdown"),
                index=merged.index,
                default=0.0,
            )
            .fillna(0.0)
            .gt(0.5)
            | _num(merged.get("long_path_slow_profit"), index=merged.index, default=0.0)
            .fillna(0.0)
            .gt(0.5)
        ).astype(float),
        np.nan,
    ).astype(np.float32)
    # Pre-entry base-score context available at decision time from the base
    # candidate cross-section. Fold-specific train-prior versions are recomputed
    # inside run_smoke before model fitting to avoid validation-month priors.
    score = _num(merged.get("score"), index=merged.index)
    # ``score`` is also a bookkeeping/output name and some generic selectors
    # exclude it defensively.  Preserve an explicit, observable model-input
    # alias so every meta contract can condition on the direct base prediction.
    merged["base_score_raw"] = score.astype(np.float32)
    ts = (
        pd.to_datetime(merged["__ts__"], utc=True, errors="coerce")
        if "__ts__" in merged.columns
        else None
    )
    if ts is not None:
        merged["base_rank_pct_by_timestamp"] = (
            score.groupby(ts).rank(pct=True).astype(np.float32)
        )
        merged["base_rank_pct_by_timestamp_side"] = (
            score.groupby([ts, merged["side_name"].astype(str)])
            .rank(pct=True)
            .astype(np.float32)
        )
        mean_ts = score.groupby(ts).transform("mean")
        std_ts = score.groupby(ts).transform("std").replace(0.0, np.nan)
        merged["base_score_z_by_timestamp"] = (
            ((score - mean_ts) / std_ts)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(np.float32)
        )
        mean_tss = score.groupby([ts, merged["side_name"].astype(str)]).transform(
            "mean"
        )
        std_tss = (
            score.groupby([ts, merged["side_name"].astype(str)])
            .transform("std")
            .replace(0.0, np.nan)
        )
        merged["base_score_z_by_timestamp_side"] = (
            ((score - mean_tss) / std_tss)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(np.float32)
        )
    for col in BASE_PRIOR_NUMERIC_FEATURES:
        if col not in merged.columns:
            merged[col] = np.float32(0.0)
    for col in BASE_PRIOR_CATEGORICAL_FEATURES:
        if col not in merged.columns:
            merged[col] = "missing"
    for col in RELIABILITY_NUMERIC_FEATURES:
        if col not in merged.columns:
            merged[col] = np.float32(0.0)
    for col in SUPPORT_DRIFT_NUMERIC_FEATURES:
        if col not in merged.columns:
            merged[col] = np.float32(0.0)
    for col in HIT_SURPRISE_NUMERIC_FEATURES:
        if col not in merged.columns:
            merged[col] = np.float32(0.0)
    for source_col, alias_col in LEDGER_CONTEXT_FEATURE_ALIASES.items():
        if source_col in merged.columns and alias_col not in merged.columns:
            merged[alias_col] = merged[source_col]
    return merged


def _resolved_meta_train_mask(
    frame: pd.DataFrame,
    *,
    valid_start: Any,
    strict: bool,
) -> tuple[pd.Series, dict[str, Any]]:
    """Return prior rows whose complete forward label predates validation."""

    timestamps = pd.to_datetime(frame.get("__ts__"), utc=True, errors="coerce")
    prior = timestamps.lt(valid_start)
    resolution_column = "__label_path_end_ts__"
    if resolution_column not in frame.columns:
        if strict:
            raise ValueError(
                "Strict meta OOS requires __label_path_end_ts__ so forward-label "
                "paths can be purged at every validation boundary."
            )
        return prior, {
            "schema": "meta_forward_label_purge_v1",
            "status": "legacy_timestamp_only",
            "resolution_column": None,
            "prior_rows": int(prior.sum()),
            "purged_rows": 0,
            "retained_rows": int(prior.sum()),
        }
    resolved_at = pd.to_datetime(
        frame[resolution_column], utc=True, errors="coerce"
    )
    invalid_prior = prior & resolved_at.isna()
    if strict and invalid_prior.any():
        raise ValueError(
            "Strict meta OOS found prior rows without finite label resolution: "
            f"{int(invalid_prior.sum())}"
        )
    # A label ending exactly at the validation boundary is conservatively
    # excluded: its final close belongs to that boundary instant.
    resolved_before = resolved_at.lt(valid_start)
    mask = prior & resolved_before
    purged = prior & ~resolved_before
    return mask, {
        "schema": "meta_forward_label_purge_v1",
        "status": "strict_purged" if strict else "purged_when_available",
        "resolution_column": resolution_column,
        "valid_start": pd.Timestamp(valid_start),
        "prior_rows": int(prior.sum()),
        "purged_rows": int(purged.sum()),
        "invalid_prior_rows": int(invalid_prior.sum()),
        "retained_rows": int(mask.sum()),
        "max_retained_label_end": (
            resolved_at.loc[mask].max() if bool(mask.any()) else None
        ),
        "min_purged_label_end": (
            resolved_at.loc[purged].min() if bool(purged.any()) else None
        ),
    }


def _attach_materialized_soft_labels(
    frame: pd.DataFrame,
    *,
    handoff_path: Path,
    label_paths: Sequence[Path],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach the canonical base soft target by decision-time row identity.

    Candidate ledgers intentionally omit training-only labels.  A controlled
    meta replay may recover the *same* materialized base target from its label
    artifact, but only for the training loss.  The target remains in the
    outcome contract and is removed before every OOS feature transform.
    """

    paths = [Path(path) for path in label_paths if Path(path).exists()]
    if not paths:
        raise FileNotFoundError("No materialized soft-label parquet files were found")
    if "row_id" not in frame.columns:
        raise ValueError("Soft-label attachment requires row_id in the handoff")
    import duckdb

    sql = """
        WITH candidate AS (
            SELECT row_id, __ts__, __symbol__, lower(side_name) AS side_name
            FROM read_parquet(?)
        ), labels AS (
            SELECT __ts__, __symbol__, lower(side_name) AS side_name,
                   __first_touch_target_soft__
            FROM read_parquet(?)
        )
        SELECT candidate.row_id, labels.__first_touch_target_soft__
        FROM candidate
        LEFT JOIN labels USING (__ts__, __symbol__, side_name)
    """
    conn = duckdb.connect()
    try:
        mapping = conn.execute(
            sql, [str(handoff_path), [str(path) for path in paths]]
        ).fetchdf()
    finally:
        conn.close()
    if mapping["row_id"].duplicated().any():
        raise ValueError("Materialized soft-label join is not one-to-one by row_id")
    target_by_row = pd.Series(
        pd.to_numeric(mapping["__first_touch_target_soft__"], errors="coerce").to_numpy(
            dtype=np.float32, copy=False
        ),
        index=mapping["row_id"].to_numpy(copy=False),
    )
    out = frame.copy(deep=False)
    target = out["row_id"].map(target_by_row).astype(np.float32)
    out["__first_touch_target_soft__"] = target
    return out, {
        "target_column": "__first_touch_target_soft__",
        "label_paths": [str(path) for path in paths],
        "rows": int(len(out)),
        "matched_rows": int(target.notna().sum()),
        "coverage": float(target.notna().mean()),
        "contract": "training-loss target only; stripped before OOS transforms",
    }


def _archetype_key(frame: pd.DataFrame) -> pd.Series:
    if "source_tag" in frame.columns:
        return frame["source_tag"].astype(str)
    if "source_semantic_family" in frame.columns:
        return (
            frame["side_name"].astype(str).str.lower()
            + "__"
            + frame["source_semantic_family"].astype(str)
        )
    return frame["side_name"].astype(str).str.lower() + "__unknown"


def _quantile_labels_from_train(
    train_values: pd.Series,
    values: pd.Series,
    *,
    prefix: str,
    q: int,
) -> pd.Series:
    train_clean = (
        pd.to_numeric(train_values, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    out = pd.Series(f"{prefix}__missing", index=values.index, dtype=object)
    if train_clean.nunique(dropna=True) < 2:
        return out
    probs = np.linspace(0.0, 1.0, int(q) + 1)[1:-1]
    edges = np.unique(train_clean.quantile(probs).to_numpy(dtype=np.float64))
    vals = pd.to_numeric(values, errors="coerce")
    bins = np.searchsorted(edges, vals.to_numpy(dtype=np.float64), side="right")
    labels = np.asarray([f"{prefix}__q{int(i)}" for i in bins], dtype=object)
    out.loc[vals.notna()] = labels[vals.notna().to_numpy()]
    return out


def _empirical_pct_from_train(train_values: pd.Series, values: pd.Series) -> pd.Series:
    train_clean = (
        pd.to_numeric(train_values, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values()
    )
    vals = pd.to_numeric(values, errors="coerce")
    out = pd.Series(np.nan, index=values.index, dtype=np.float32)
    if train_clean.empty:
        return out.fillna(0.5)
    ranks = np.searchsorted(
        train_clean.to_numpy(dtype=np.float64),
        vals.to_numpy(dtype=np.float64),
        side="right",
    )
    pct = ranks / max(len(train_clean), 1)
    out.loc[vals.notna()] = pct[vals.notna().to_numpy()].astype(np.float32)
    return out.fillna(0.5).clip(0.0, 1.0).astype(np.float32)


def _add_fold_base_prior_features(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    selected_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add validation-safe base-score priors fit on the training months only."""

    train = train.copy()
    valid = valid.copy()
    train_score = _num(train.get("score"), index=train.index)
    valid_score = _num(valid.get("score"), index=valid.index)
    train_arch = _archetype_key(train)
    valid_arch = _archetype_key(valid)
    train_keys = pd.DataFrame(
        {"side": train["side_name"].astype(str).str.lower(), "arch": train_arch},
        index=train.index,
    )
    valid_keys = pd.DataFrame(
        {"side": valid["side_name"].astype(str).str.lower(), "arch": valid_arch},
        index=valid.index,
    )
    selected = train.get(selected_col, pd.Series(False, index=train.index)).astype(bool)
    global_cutoff = (
        float(train_score.loc[selected & train_score.notna()].min())
        if bool((selected & train_score.notna()).any())
        else float(train_score.quantile(0.90))
    )
    global_mean = float(train_score.mean()) if train_score.notna().any() else 0.0
    global_std = float(train_score.std()) if train_score.notna().sum() > 1 else 1.0
    if not np.isfinite(global_std) or global_std <= 1e-12:
        global_std = 1.0

    prior_rows: list[dict[str, Any]] = []
    train_tmp = train_keys.copy()
    train_tmp["score"] = train_score
    train_tmp["selected"] = selected
    for (side, arch), group in train_tmp.groupby(["side", "arch"], dropna=False):
        scores = pd.to_numeric(group["score"], errors="coerce").dropna()
        selected_scores = pd.to_numeric(
            group.loc[group["selected"].astype(bool), "score"], errors="coerce"
        ).dropna()
        cutoff = (
            float(selected_scores.min())
            if len(selected_scores)
            else (float(scores.quantile(0.90)) if len(scores) else global_cutoff)
        )
        mean = float(scores.mean()) if len(scores) else global_mean
        std = float(scores.std()) if len(scores) > 1 else global_std
        if not np.isfinite(std) or std <= 1e-12:
            std = global_std
        prior_rows.append(
            {
                "side": side,
                "arch": arch,
                "cutoff": cutoff,
                "mean": mean,
                "std": std,
                "rows": int(len(group)),
            }
        )
    priors = pd.DataFrame(prior_rows)

    def attach(
        frame: pd.DataFrame, keys: pd.DataFrame, score: pd.Series
    ) -> pd.DataFrame:
        joined = keys.merge(priors, on=["side", "arch"], how="left")
        cutoff = pd.to_numeric(joined["cutoff"], errors="coerce").fillna(global_cutoff)
        mean = pd.to_numeric(joined["mean"], errors="coerce").fillna(global_mean)
        std = (
            pd.to_numeric(joined["std"], errors="coerce")
            .replace(0.0, np.nan)
            .fillna(global_std)
        )
        frame["base_margin_to_cutoff"] = (
            (score.reset_index(drop=True) - cutoff).astype(np.float32).to_numpy()
        )
        frame["base_margin_to_cutoff_z"] = (
            ((score.reset_index(drop=True) - cutoff) / std)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(np.float32)
            .to_numpy()
        )
        frame["base_signal_zscore_within_archetype"] = (
            ((score.reset_index(drop=True) - mean) / std)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(np.float32)
            .to_numpy()
        )
        return frame

    train = attach(train, train_keys, train_score)
    valid = attach(valid, valid_keys, valid_score)
    train["base_score_rank_pct_train_prior"] = _empirical_pct_from_train(
        train_score, train_score
    )
    valid["base_score_rank_pct_train_prior"] = _empirical_pct_from_train(
        train_score, valid_score
    )
    train["base_rank_band"] = _quantile_labels_from_train(
        train_score, train_score, prefix="base_rank_band", q=5
    )
    valid["base_rank_band"] = _quantile_labels_from_train(
        train_score, valid_score, prefix="base_rank_band", q=5
    )
    train["base_margin_band"] = _quantile_labels_from_train(
        train["base_margin_to_cutoff"],
        train["base_margin_to_cutoff"],
        prefix="base_margin_band",
        q=5,
    )
    valid["base_margin_band"] = _quantile_labels_from_train(
        train["base_margin_to_cutoff"],
        valid["base_margin_to_cutoff"],
        prefix="base_margin_band",
        q=5,
    )
    return train, valid


def _add_fold_reliability_features(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    shrinkage_k: float = 60.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add train-derived reliability priors by side/archetype/base band.

    Validation rows receive only statistics computed on earlier training
    months. Training rows use leave-one-out group statistics to avoid directly
    injecting their own label into the feature value.
    """

    train = train.copy()
    valid = valid.copy()
    train_arch = _archetype_key(train).astype(str)
    valid_arch = _archetype_key(valid).astype(str)
    train["_rel_side"] = train["side_name"].astype(str).str.lower()
    valid["_rel_side"] = valid["side_name"].astype(str).str.lower()
    train["_rel_arch"] = train_arch
    valid["_rel_arch"] = valid_arch

    clean = (
        _num(train.get("clean_exec_label"), index=train.index, default=0.0)
        .fillna(0.0)
        .clip(0.0, 1.0)
    )
    bad = (
        _num(train.get("full_path_bad_mae_1r"), index=train.index, default=0.0)
        .fillna(0.0)
        .clip(0.0, 1.0)
    )
    timeout = (
        _num(train.get("timeout"), index=train.index, default=0.0)
        .fillna(0.0)
        .clip(0.0, 1.0)
    )
    dirty = (
        _num(train.get("dirty_positive"), index=train.index, default=0.0)
        .fillna(0.0)
        .clip(0.0, 1.0)
    )
    exec_margin = _num(train.get("exec_margin"), index=train.index, default=0.0).fillna(
        0.0
    )
    global_stats = {
        "clean": float(clean.mean()) if len(clean) else 0.0,
        "bad": float(bad.mean()) if len(bad) else 0.0,
        "timeout": float(timeout.mean()) if len(timeout) else 0.0,
        "dirty": float(dirty.mean()) if len(dirty) else 0.0,
        "exec": float(exec_margin.mean()) if len(exec_margin) else 0.0,
    }
    global_stats["edge"] = (
        global_stats["clean"]
        - global_stats["bad"]
        - 0.50 * global_stats["timeout"]
        + global_stats["exec"]
    )
    work = train[["_rel_side", "_rel_arch"]].copy()
    work["_clean"] = clean
    work["_bad"] = bad
    work["_timeout"] = timeout
    work["_dirty"] = dirty
    work["_exec"] = exec_margin

    def attach_for_band(band_col: str, prefix: str) -> None:
        nonlocal train, valid
        if band_col not in train.columns:
            train[band_col] = f"{band_col}__missing"
        if band_col not in valid.columns:
            valid[band_col] = f"{band_col}__missing"
        tmp = work.copy()
        tmp["_band"] = train[band_col].astype(str).fillna("missing")
        grouped = (
            tmp.groupby(["_rel_side", "_rel_arch", "_band"], dropna=False)
            .agg(
                rows=("_clean", "size"),
                clean_sum=("_clean", "sum"),
                bad_sum=("_bad", "sum"),
                timeout_sum=("_timeout", "sum"),
                dirty_sum=("_dirty", "sum"),
                exec_sum=("_exec", "sum"),
            )
            .reset_index()
        )

        def fill_from_stats(frame: pd.DataFrame, *, loo: bool) -> pd.DataFrame:
            keys = pd.DataFrame(
                {
                    "_rel_side": frame["_rel_side"].astype(str),
                    "_rel_arch": frame["_rel_arch"].astype(str),
                    "_band": frame[band_col].astype(str).fillna("missing"),
                },
                index=frame.index,
            )
            merged = keys.merge(
                grouped, on=["_rel_side", "_rel_arch", "_band"], how="left"
            )
            rows = pd.to_numeric(merged["rows"], errors="coerce").fillna(0.0)
            clean_sum = pd.to_numeric(merged["clean_sum"], errors="coerce").fillna(0.0)
            bad_sum = pd.to_numeric(merged["bad_sum"], errors="coerce").fillna(0.0)
            timeout_sum = pd.to_numeric(merged["timeout_sum"], errors="coerce").fillna(
                0.0
            )
            dirty_sum = pd.to_numeric(merged["dirty_sum"], errors="coerce").fillna(0.0)
            exec_sum = pd.to_numeric(merged["exec_sum"], errors="coerce").fillna(0.0)
            if loo:
                rows = (rows - 1.0).clip(lower=0.0)
                clean_sum = (
                    clean_sum
                    - _num(
                        frame.get("clean_exec_label"), index=frame.index, default=0.0
                    )
                    .reset_index(drop=True)
                    .fillna(0.0)
                ).clip(lower=0.0)
                bad_sum = (
                    bad_sum
                    - _num(
                        frame.get("full_path_bad_mae_1r"),
                        index=frame.index,
                        default=0.0,
                    )
                    .reset_index(drop=True)
                    .fillna(0.0)
                ).clip(lower=0.0)
                timeout_sum = (
                    timeout_sum
                    - _num(frame.get("timeout"), index=frame.index, default=0.0)
                    .reset_index(drop=True)
                    .fillna(0.0)
                ).clip(lower=0.0)
                dirty_sum = (
                    dirty_sum
                    - _num(frame.get("dirty_positive"), index=frame.index, default=0.0)
                    .reset_index(drop=True)
                    .fillna(0.0)
                ).clip(lower=0.0)
                exec_sum = exec_sum - _num(
                    frame.get("exec_margin"), index=frame.index, default=0.0
                ).reset_index(drop=True).fillna(0.0)
            denom = rows.replace(0.0, np.nan)
            weight = (rows / (rows + float(shrinkage_k))).clip(0.0, 1.0)
            clean_rate = (clean_sum / denom).fillna(global_stats["clean"])
            bad_rate = (bad_sum / denom).fillna(global_stats["bad"])
            timeout_rate = (timeout_sum / denom).fillna(global_stats["timeout"])
            dirty_rate = (dirty_sum / denom).fillna(global_stats["dirty"])
            exec_mean = (exec_sum / denom).fillna(global_stats["exec"])
            frame[f"{prefix}_rows_log1p"] = np.log1p(rows).astype(np.float32).to_numpy()
            frame[f"{prefix}_clean_rate"] = (
                (weight * clean_rate + (1.0 - weight) * global_stats["clean"])
                .astype(np.float32)
                .to_numpy()
            )
            frame[f"{prefix}_bad_mae_rate"] = (
                (weight * bad_rate + (1.0 - weight) * global_stats["bad"])
                .astype(np.float32)
                .to_numpy()
            )
            frame[f"{prefix}_timeout_rate"] = (
                (weight * timeout_rate + (1.0 - weight) * global_stats["timeout"])
                .astype(np.float32)
                .to_numpy()
            )
            frame[f"{prefix}_dirty_positive_rate"] = (
                (weight * dirty_rate + (1.0 - weight) * global_stats["dirty"])
                .astype(np.float32)
                .to_numpy()
            )
            frame[f"{prefix}_exec_margin_mean"] = (
                (weight * exec_mean + (1.0 - weight) * global_stats["exec"])
                .astype(np.float32)
                .to_numpy()
            )
            frame[f"{prefix}_edge"] = (
                frame[f"{prefix}_clean_rate"]
                - frame[f"{prefix}_bad_mae_rate"]
                - 0.50 * frame[f"{prefix}_timeout_rate"]
                + frame[f"{prefix}_exec_margin_mean"]
            ).astype(np.float32)
            return frame

        train = fill_from_stats(train, loo=True)
        valid = fill_from_stats(valid, loo=False)

    attach_for_band("base_rank_band", "rel_rankband")
    attach_for_band("base_margin_band", "rel_marginband")
    train = train.drop(columns=["_rel_side", "_rel_arch"], errors="ignore")
    valid = valid.drop(columns=["_rel_side", "_rel_arch"], errors="ignore")
    return train, valid


def _add_fold_support_drift_features(
    train: pd.DataFrame, valid: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add row-level train-support features for AE/GMM/context/leaf buckets."""

    train = train.copy()
    valid = valid.copy()
    cols = [
        col
        for col in SUPPORT_DRIFT_COLUMNS
        if col in train.columns and col in valid.columns
    ]
    if not cols:
        for col in SUPPORT_DRIFT_NUMERIC_FEATURES:
            train[col] = np.float32(0.0)
            valid[col] = np.float32(0.0)
        return train, valid

    min_count = 20.0
    min_freq = 0.01
    train_logs: list[np.ndarray] = []
    valid_logs: list[np.ndarray] = []
    train_freqs: list[np.ndarray] = []
    valid_freqs: list[np.ndarray] = []
    train_unseen: list[np.ndarray] = []
    valid_unseen: list[np.ndarray] = []
    train_rare: list[np.ndarray] = []
    valid_rare: list[np.ndarray] = []
    n_train = max(float(len(train)), 1.0)
    for col in cols:
        train_values = train[col].astype(str).fillna("missing")
        valid_values = valid[col].astype(str).fillna("missing")
        counts = train_values.value_counts(dropna=False)
        train_count = train_values.map(counts).fillna(0.0).astype(float).to_numpy()
        valid_count = valid_values.map(counts).fillna(0.0).astype(float).to_numpy()
        train_freq = train_count / n_train
        valid_freq = valid_count / n_train
        train_logs.append(np.log1p(train_count))
        valid_logs.append(np.log1p(valid_count))
        train_freqs.append(train_freq)
        valid_freqs.append(valid_freq)
        train_unseen.append((train_count <= 0.0).astype(float))
        valid_unseen.append((valid_count <= 0.0).astype(float))
        train_rare.append(
            ((train_count < min_count) | (train_freq < min_freq)).astype(float)
        )
        valid_rare.append(
            ((valid_count < min_count) | (valid_freq < min_freq)).astype(float)
        )

    def assign(
        frame: pd.DataFrame,
        logs: list[np.ndarray],
        freqs: list[np.ndarray],
        unseen: list[np.ndarray],
        rare: list[np.ndarray],
    ) -> pd.DataFrame:
        log_mat = np.vstack(logs).T
        freq_mat = np.vstack(freqs).T
        unseen_mat = np.vstack(unseen).T
        rare_mat = np.vstack(rare).T
        frame["support_min_log_count"] = np.nanmin(log_mat, axis=1).astype(np.float32)
        frame["support_mean_log_count"] = np.nanmean(log_mat, axis=1).astype(np.float32)
        frame["support_min_frequency"] = np.nanmin(freq_mat, axis=1).astype(np.float32)
        frame["support_mean_frequency"] = np.nanmean(freq_mat, axis=1).astype(
            np.float32
        )
        frame["support_unseen_bucket_count"] = np.nansum(unseen_mat, axis=1).astype(
            np.float32
        )
        frame["support_unseen_bucket_share"] = (np.nanmean(unseen_mat, axis=1)).astype(
            np.float32
        )
        frame["support_rare_bucket_count"] = np.nansum(rare_mat, axis=1).astype(
            np.float32
        )
        frame["support_rare_bucket_share"] = (np.nanmean(rare_mat, axis=1)).astype(
            np.float32
        )
        return frame

    return (
        assign(train, train_logs, train_freqs, train_unseen, train_rare),
        assign(valid, valid_logs, valid_freqs, valid_unseen, valid_rare),
    )


def _timestamp_days(frame: pd.DataFrame) -> pd.Series:
    if "__ts__" not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=np.float64)
    ts = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    out = pd.Series(np.nan, index=frame.index, dtype=np.float64)
    valid = ts.notna()
    out.loc[valid] = ts.loc[valid].astype("int64").astype(np.float64) / (1e9 * 86400.0)
    return out


def _add_fold_hit_surprise_features(
    train: pd.DataFrame, valid: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add decayed recent clean-hit surprise by side/base archetype.

    Validation rows are encoded from training rows only. Training rows are
    encoded from strictly earlier rows in the same training fold, excluding
    same-timestamp rows to avoid direct path-label leakage.
    """

    train = train.copy()
    valid = valid.copy()
    train_days = _timestamp_days(train)
    valid_days = _timestamp_days(valid)
    hit = (
        _num(train.get("clean_exec_label"), index=train.index, default=0.0)
        .fillna(0.0)
        .clip(0.0, 1.0)
    )
    train_key = (
        train["side_name"].astype(str).str.lower()
        + "__"
        + _archetype_key(train).astype(str)
    )
    valid_key = (
        valid["side_name"].astype(str).str.lower()
        + "__"
        + _archetype_key(valid).astype(str)
    )
    global_hit = float(hit.mean()) if len(hit) else 0.0
    shrinkage_k = 40.0

    source = pd.DataFrame(
        {
            "_key": train_key.astype(str),
            "_day": train_days,
            "_hit": hit.astype(np.float64),
        },
        index=train.index,
    ).replace([np.inf, -np.inf], np.nan)
    source = source[source["_day"].notna()].sort_values(
        ["_key", "_day"], kind="mergesort"
    )
    source_groups: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
    for key, group in source.groupby("_key", sort=False):
        days = group["_day"].to_numpy(dtype=np.float64)
        hits = group["_hit"].to_numpy(dtype=np.float64)
        offset = float(days[0]) if len(days) else 0.0
        source_groups[str(key)] = (days, hits, np.cumsum(hits), offset)

    def assign(
        target: pd.DataFrame, target_days: pd.Series, target_key: pd.Series
    ) -> pd.DataFrame:
        for hl in HIT_SURPRISE_HALFLIFE_DAYS:
            suffix = int(hl)
            recent = np.full(len(target), global_hit, dtype=np.float32)
            expected = np.full(len(target), global_hit, dtype=np.float32)
            surprise = np.zeros(len(target), dtype=np.float32)
            zscore = np.zeros(len(target), dtype=np.float32)
            support_log = np.zeros(len(target), dtype=np.float32)
            effective_n = np.zeros(len(target), dtype=np.float32)
            target_tmp = pd.DataFrame(
                {
                    "_pos": np.arange(len(target), dtype=np.int64),
                    "_key": target_key.astype(str).to_numpy(),
                    "_day": target_days.to_numpy(dtype=np.float64),
                },
                index=target.index,
            )
            for key, group in target_tmp.groupby("_key", sort=False):
                source_tuple = source_groups.get(str(key))
                if source_tuple is None:
                    continue
                src_days, src_hits, src_cumsum, src_offset = source_tuple
                pos = group["_pos"].to_numpy(dtype=np.int64)
                day = group["_day"].to_numpy(dtype=np.float64)
                valid_day = np.isfinite(day)
                if not bool(np.any(valid_day)):
                    continue
                pos = pos[valid_day]
                day = day[valid_day]
                right = np.searchsorted(src_days, day, side="left").astype(np.int64)
                has_prior = right > 0
                if not bool(np.any(has_prior)):
                    continue
                pos = pos[has_prior]
                day = day[has_prior]
                right = right[has_prior]
                prior_hits = src_cumsum[right - 1]
                prior_count = right.astype(np.float64)
                prior_raw = prior_hits / np.maximum(prior_count, 1.0)
                prior_weight = prior_count / (prior_count + shrinkage_k)
                prior = prior_weight * prior_raw + (1.0 - prior_weight) * global_hit
                left = np.searchsorted(
                    src_days, day - 4.0 * float(hl), side="left"
                ).astype(np.int64)
                alpha = math.log(2.0) / float(hl)
                shifted_src_days = src_days - src_offset
                shifted_day = day - src_offset
                exp1 = np.exp(alpha * shifted_src_days)
                exp2 = np.exp(2.0 * alpha * shifted_src_days)
                hit_exp1 = src_hits * exp1
                c_exp1 = np.concatenate([[0.0], np.cumsum(exp1)])
                c_hit_exp1 = np.concatenate([[0.0], np.cumsum(hit_exp1)])
                c_exp2 = np.concatenate([[0.0], np.cumsum(exp2)])
                win_exp1 = c_exp1[right] - c_exp1[left]
                win_hit_exp1 = c_hit_exp1[right] - c_hit_exp1[left]
                win_exp2 = c_exp2[right] - c_exp2[left]
                scale1 = np.exp(-alpha * shifted_day)
                scale2 = np.exp(-2.0 * alpha * shifted_day)
                weight_sum = scale1 * win_exp1
                weighted_hit = np.divide(
                    scale1 * win_hit_exp1,
                    np.maximum(weight_sum, 1e-12),
                    out=prior.copy(),
                    where=weight_sum > 1e-12,
                )
                weight_sq_sum = scale2 * win_exp2
                eff_n = np.divide(
                    weight_sum * weight_sum,
                    np.maximum(weight_sq_sum, 1e-12),
                    out=np.zeros_like(weight_sum),
                    where=weight_sum > 1e-12,
                )
                diff = weighted_hit - prior
                denom = np.sqrt(
                    np.maximum(prior * (1.0 - prior), 1e-6) / np.maximum(eff_n, 1.0)
                )
                recent[pos] = weighted_hit.astype(np.float32)
                expected[pos] = prior.astype(np.float32)
                surprise[pos] = diff.astype(np.float32)
                zscore[pos] = np.clip(diff / denom, -8.0, 8.0).astype(np.float32)
                support_log[pos] = np.log1p(weight_sum).astype(np.float32)
                effective_n[pos] = eff_n.astype(np.float32)
            target[f"base_arch_hit_recent_rate_hl{suffix}d"] = recent
            target[f"base_arch_hit_expected_rate_hl{suffix}d"] = expected
            target[f"base_arch_hit_surprise_hl{suffix}d"] = surprise
            target[f"base_arch_hit_surprise_z_hl{suffix}d"] = zscore
            target[f"base_arch_hit_support_log1p_hl{suffix}d"] = support_log
            target[f"base_arch_hit_effective_n_hl{suffix}d"] = effective_n
        return target

    return assign(train, train_days, train_key), assign(valid, valid_days, valid_key)


def _feature_columns(
    frame: pd.DataFrame,
    *,
    enable_base_prior_features: bool = False,
    enable_reliability_features: bool = False,
    enable_support_drift_features: bool = False,
    enable_hit_surprise_features: bool = False,
) -> tuple[list[str], list[str]]:
    excluded = set(NEVER_FEATURE_COLUMNS)
    excluded.update(OUTCOME_COLUMNS)
    excluded.update(
        {
            "positive_exec_margin",
            "clean_exec_label",
            "bad_path_label",
            "path_mfe_before_mae_label",
            "path_mae_before_mfe_label",
            "path_underwater_duration_target",
            "source_recipe_name",
            "target_soft",
            "target_hard",
            "first_pass_good",
            "first_pass_bad",
        }
    )
    if not enable_base_prior_features:
        excluded.update(BASE_PRIOR_NUMERIC_FEATURES)
        excluded.update(BASE_PRIOR_CATEGORICAL_FEATURES)
    if not enable_reliability_features:
        excluded.update(RELIABILITY_NUMERIC_FEATURES)
    if not enable_support_drift_features:
        excluded.update(SUPPORT_DRIFT_NUMERIC_FEATURES)
    if not enable_hit_surprise_features:
        excluded.update(HIT_SURPRISE_NUMERIC_FEATURES)
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for col in frame.columns:
        if col in excluded or col.endswith("__ledger"):
            continue
        if col.startswith("selected_top"):
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            numeric_cols.append(col)
        elif frame[col].dtype == bool:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    # Side is both categorical and already encoded by descriptors in places; keep it categorical.
    return sorted(numeric_cols), sorted(categorical_cols)


def _make_xy(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    numeric_cols: list[str],
    categorical_cols: list[str],
    selected_features: list[str] | None = None,
    generated_features: Sequence[str] = (),
    legacy_constant_zero_features: Sequence[str] = (),
    preprocessing_state_out: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    selected_contract: dict[str, Any] | None = None
    if selected_features:
        selected_contract = resolve_meta_input_contract(
            selected_features,
            materialized_columns=train.columns,
            categorical_columns=categorical_cols,
            generated_features=generated_features,
            legacy_constant_zero_features=legacy_constant_zero_features,
        )
        require_resolved_meta_input_contract(
            selected_contract, role="meta matrix construction"
        )
    train_parts: list[pd.DataFrame] = []
    valid_parts: list[pd.DataFrame] = []
    numeric_medians: dict[str, float] = {}
    if numeric_cols:
        train_num = (
            train.loc[:, numeric_cols]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        valid_num = (
            valid.loc[:, numeric_cols]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        med = (
            train_num.median(numeric_only=True)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        numeric_medians = {
            str(col): float(value) for col, value in med.items()
        }
        train_parts.append(train_num.fillna(med).fillna(0.0).astype(np.float32))
        valid_parts.append(valid_num.fillna(med).fillna(0.0).astype(np.float32))
    if categorical_cols:
        selected = set(str(c) for c in (selected_features or []))
        # A numeric feature may legitimately share a categorical source prefix
        # (for example ``base_margin_to_cutoff_z``).  The frozen numeric
        # contract is authoritative; do not materialize the same name again as
        # a one-hot category inferred only from its prefix.
        selected.difference_update(str(c) for c in numeric_cols)
        if selected:
            train_cat_parts: list[pd.Series] = []
            valid_cat_parts: list[pd.Series] = []
            for col in categorical_cols:
                prefix = f"{col}_"
                wanted = sorted(feat for feat in selected if feat.startswith(prefix))
                if not wanted:
                    continue
                train_vals = (
                    train[col].astype(str).fillna("missing")
                    if col in train.columns
                    else pd.Series("missing", index=train.index)
                )
                valid_vals = (
                    valid[col].astype(str).fillna("missing")
                    if col in valid.columns
                    else pd.Series("missing", index=valid.index)
                )
                for feat in wanted:
                    category = feat[len(prefix) :]
                    train_cat_parts.append(
                        train_vals.eq(category).astype(np.float32).rename(feat)
                    )
                    valid_cat_parts.append(
                        valid_vals.eq(category).astype(np.float32).rename(feat)
                    )
            if train_cat_parts:
                train_parts.append(pd.concat(train_cat_parts, axis=1))
                valid_parts.append(pd.concat(valid_cat_parts, axis=1))
        else:
            train_cat = pd.get_dummies(
                train.loc[:, categorical_cols].astype(str).fillna("missing"),
                dummy_na=False,
            )
            valid_cat = pd.get_dummies(
                valid.loc[:, categorical_cols].astype(str).fillna("missing"),
                dummy_na=False,
            )
            valid_cat = valid_cat.reindex(columns=train_cat.columns, fill_value=0)
            train_parts.append(train_cat.astype(np.float32))
            valid_parts.append(valid_cat.astype(np.float32))
    if not train_parts:
        raise ValueError("No meta feature columns available.")
    x_train = pd.concat(train_parts, axis=1)
    x_valid = pd.concat(valid_parts, axis=1).reindex(
        columns=x_train.columns, fill_value=0.0
    )
    if selected_contract is not None:
        x_train = materialize_legacy_constant_zeros(x_train, selected_contract)
        x_valid = materialize_legacy_constant_zeros(x_valid, selected_contract)
    if preprocessing_state_out is not None:
        preprocessing_state_out.clear()
        preprocessing_state_out.update(
            {
                "schema": "s52_meta_matrix_preprocessing_v1",
                "numeric_columns": list(map(str, numeric_cols)),
                "numeric_medians": numeric_medians,
                "categorical_source_columns": list(map(str, categorical_cols)),
                "encoded_feature_names": list(map(str, x_train.columns)),
                "encoded_feature_contract_hash": _feature_contract_hash(
                    list(map(str, x_train.columns))
                ),
                "categorical_encoding": "fixed selected one-hot names",
                "nonfinite_policy": "replace inf with nan then train-median fill",
            }
        )
    return x_train, x_valid, list(x_train.columns)


def _feature_source_columns_for_selected(
    *,
    selected_features: list[str] | None,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[list[str], list[str]]:
    """Reduce raw feature materialization once the global feature set is known.

    Feature selection operates on the encoded matrix. Numeric selected features
    map 1:1 to raw numeric columns. Categorical selected features are pandas
    dummy columns named ``<raw_col>_<category>``; keep only raw categorical
    columns whose dummy prefix appears in the selected set.
    """
    if not selected_features:
        return list(numeric_cols), list(categorical_cols)
    selected = {str(c) for c in selected_features}
    selected_numeric = [col for col in numeric_cols if str(col) in selected]
    # Fold-local base priors are generated after the raw handoff is loaded.  A
    # frozen side contract may legitimately request them even though they were
    # absent from the original parquet schema.
    selected_numeric.extend(
        feature
        for feature in BASE_PRIOR_NUMERIC_FEATURES
        if feature in selected and feature not in selected_numeric
    )
    selected_categorical: list[str] = []
    for col in categorical_cols:
        prefix = f"{col}_"
        if any(feat.startswith(prefix) for feat in selected):
            selected_categorical.append(col)
    # Post-selection OOD features are appended after matrix construction and do
    # not require raw source columns here.
    if not selected_numeric and not selected_categorical:
        return list(numeric_cols), list(categorical_cols)
    return selected_numeric, selected_categorical


def _projected_handoff_columns_for_selected(
    handoff_path: Path,
    selected_features: Sequence[str] | None,
) -> list[str] | None:
    """Map encoded selected columns back to their raw handoff sources."""
    selected = [str(col) for col in (selected_features or []) if str(col).strip()]
    if not selected:
        return None
    available = sorted(_parquet_schema_columns(handoff_path))
    available_set = set(available)
    projected: set[str] = {col for col in selected if col in available_set}
    unresolved = [col for col in selected if col not in available_set]
    categorical_candidates = [
        col
        for col in available
        if not col.startswith("selected_top")
        and col not in NEVER_FEATURE_COLUMNS
        and col not in OUTCOME_COLUMNS
    ]
    for encoded in unresolved:
        matches = [col for col in categorical_candidates if encoded.startswith(f"{col}_")]
        if matches:
            projected.add(max(matches, key=len))
    projected.update(
        {
            "source_tag",
            "source_semantic_family",
            "policy_archetype",
            "archetype_policy_key",
            "__archetype_policy_key__",
            "archetype_label_family",
            "__archetype_label_family__",
            "local_side_archetype",
            "source_archetype",
            "row_id",
        }.intersection(available_set)
    )
    return sorted(projected)


def _classification_weights(target: pd.Series, train: pd.DataFrame) -> np.ndarray:
    y = _num(target).fillna(0.0).astype(int)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    weights = np.ones(len(y), dtype=np.float32)
    if pos > 0 and neg > 0:
        weights[y.to_numpy(dtype=bool)] = neg / max(pos, 1)
    weights *= 1.0 + 0.75 * _num(
        train.get("dirty_positive"), index=train.index, default=0.0
    ).fillna(0.0).to_numpy(dtype=np.float32)
    weights *= 1.0 + 0.50 * _num(
        train.get("full_path_bad_mae_1r"), index=train.index, default=0.0
    ).fillna(0.0).to_numpy(dtype=np.float32)
    weights = np.clip(weights, 0.25, 8.0)
    return weights / max(float(weights.mean()), 1e-12)


def _regression_weights(target: pd.Series, train: pd.DataFrame) -> np.ndarray:
    y = _num(target).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    weights = np.ones(len(y), dtype=np.float32)
    weights *= np.where(y.gt(0.0), 2.0, 1.0)
    if len(y) and y.nunique(dropna=True) > 1:
        weights *= np.where(y.ge(float(y.quantile(0.80))), 1.5, 1.0)
    weights *= 1.0 + 0.60 * _num(
        train.get("dirty_positive"), index=train.index, default=0.0
    ).fillna(0.0).to_numpy(dtype=np.float32)
    weights = np.clip(weights, 0.25, 6.0)
    return weights / max(float(weights.mean()), 1e-12)


def _lgbm_params(
    defaults: dict[str, Any], override: dict[str, Any] | None
) -> dict[str, Any]:
    params = dict(defaults)
    if override:
        params.update({str(k): v for k, v in override.items()})
    for key in ("n_estimators", "num_leaves", "max_depth", "min_child_samples"):
        params[key] = int(float(params[key]))
    for key in (
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
    ):
        params[key] = float(params[key])
    return params


def _meta_lgbm_n_jobs() -> int:
    return max(1, min(int(os.environ.get("EPM_META_LGBM_N_JOBS", "4")), 16))


def _time_spread_cap_rows(n_rows: int, max_rows: int) -> np.ndarray:
    if int(max_rows) <= 0 or n_rows <= int(max_rows):
        return np.arange(n_rows, dtype=np.int64)
    n = int(n_rows)
    k = int(max_rows)
    base = k // 3
    rem = k - base * 3
    sizes = [base + (1 if i < rem else 0) for i in range(3)]
    windows = [(0, n // 3), (n // 3, (2 * n) // 3), ((2 * n) // 3, n)]
    parts: list[np.ndarray] = []
    for size, (start, end) in zip(sizes, windows):
        size = min(int(size), max(int(end - start), 0))
        if size <= 0:
            continue
        if size >= end - start:
            parts.append(np.arange(start, end, dtype=np.int64))
        else:
            parts.append(np.linspace(start, end - 1, size, dtype=np.int64))
    if not parts:
        return np.arange(0, min(n, k), dtype=np.int64)
    return np.unique(np.concatenate(parts).astype(np.int64))


def _feature_selection_target(train: pd.DataFrame, target_name: str) -> pd.Series:
    if target_name == "exec_margin":
        return _num(train.get("exec_margin"), index=train.index, default=0.0).fillna(
            0.0
        )
    if target_name == "ev_frontier":
        ev = _num(train.get("ev_after_1pct"), index=train.index)
        if ev.isna().all():
            ev = _num(train.get("exec_margin"), index=train.index, default=0.0)
        clean = (
            _num(train.get("clean_exec_label"), index=train.index, default=0.0)
            .fillna(0.0)
            .clip(0.0, 1.0)
        )
        dirty = (
            _num(train.get("dirty_positive"), index=train.index, default=0.0)
            .fillna(0.0)
            .clip(0.0, 1.0)
        )
        bad = (
            _num(train.get("bad_path_label"), index=train.index, default=0.0)
            .fillna(0.0)
            .clip(0.0, 1.0)
        )
        first_bad = (
            _num(train.get("first_touch_bad_mae_1r"), index=train.index, default=0.0)
            .fillna(0.0)
            .clip(0.0, 1.0)
        )
        full_bad = (
            _num(train.get("full_path_bad_mae_1r"), index=train.index, default=0.0)
            .fillna(0.0)
            .clip(0.0, 1.0)
        )
        bad_or_stop = (
            pd.concat([bad, first_bad, full_bad], axis=1).max(axis=1).fillna(0.0)
        )
        ev_pct = ev.fillna(0.0).clip(-0.05, 0.05) * 100.0
        return (
            (0.45 * ev_pct + 0.20 * ((clean - dirty) + (clean - bad_or_stop)))
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
    if target_name == "clean_minus_bad":
        clean = _num(
            train.get("clean_exec_label"), index=train.index, default=0.0
        ).fillna(0.0)
        bad = _num(train.get("bad_path_label"), index=train.index, default=0.0).fillna(
            0.0
        )
        timeout = _num(train.get("timeout"), index=train.index, default=0.0).fillna(0.0)
        margin = (
            _num(train.get("exec_margin"), index=train.index, default=0.0)
            .fillna(0.0)
            .clip(-0.03, 0.03)
        )
        return clean + 0.50 * margin - 0.70 * bad - 0.30 * timeout
    if target_name == "clean_exec":
        return _num(
            train.get("clean_exec_label"), index=train.index, default=0.0
        ).fillna(0.0)
    raise ValueError(f"Unknown feature-selection target: {target_name}")


def _base_soft_label_target(
    frame: pd.DataFrame,
    *,
    target_contract: dict[str, Any] | None = None,
) -> tuple[pd.Series, str]:
    declared = str((target_contract or {}).get("target_column", "")).strip()
    candidates = list(dict.fromkeys([declared, *BASE_SOFT_LABEL_COLUMNS]))
    for col in candidates:
        if not col:
            continue
        if col in frame.columns:
            target = (
                _num(frame.get(col), index=frame.index)
                .replace([np.inf, -np.inf], np.nan)
                .clip(0.0, 1.0)
            )
            if int(target.notna().sum()) > 0:
                return target, col
    raise ValueError(
        "Single-head meta mode requires the base economic soft label. "
        f"Expected one of: {', '.join(BASE_SOFT_LABEL_COLUMNS)}"
    )


def _feature_selection_label_context(train: pd.DataFrame) -> dict[str, np.ndarray]:
    n = int(len(train))
    side_name = (
        train.get("side_name", pd.Series("unknown", index=train.index))
        .astype(str)
        .str.lower()
    )
    side_num = np.where(side_name.eq("short"), -1.0, 1.0).astype(np.float32)
    family = (
        train.get("archetype_label_family")
        if "archetype_label_family" in train.columns
        else train.get(
            "__archetype_label_family__", train.get("source_semantic_family", "unknown")
        )
    )
    family_ser = pd.Series(family, index=train.index).astype(str).fillna("unknown")
    policy = (
        train.get("archetype_policy_key")
        if "archetype_policy_key" in train.columns
        else train.get("__archetype_policy_key__", train.get("source_tag", "unknown"))
    )
    policy_ser = pd.Series(policy, index=train.index).astype(str).fillna("unknown")
    source_family = (
        pd.Series(train.get("source_semantic_family", family_ser), index=train.index)
        .astype(str)
        .fillna("unknown")
    )
    feature_arch = (side_name + "__" + family_ser + "__" + policy_ser).to_numpy(
        dtype=object
    )
    behavior_arch = (
        side_name
        + "__"
        + source_family
        + "__bad"
        + _num(train.get("first_touch_bad_mae_1r"), index=train.index, default=0.0)
        .fillna(0.0)
        .gt(0.5)
        .astype(int)
        .astype(str)
        + "__to"
        + _num(train.get("timeout"), index=train.index, default=0.0)
        .fillna(0.0)
        .gt(0.5)
        .astype(int)
        .astype(str)
    ).to_numpy(dtype=object)
    return {
        "feature_selection_archetype": feature_arch,
        "archetype": feature_arch,
        "label_archetype": family_ser.to_numpy(dtype=object),
        "source_archetype": source_family.to_numpy(dtype=object),
        "geometry_archetype": policy_ser.to_numpy(dtype=object),
        "side_name": side_name.to_numpy(dtype=object),
        "side": side_num,
        "clean_exec": _num(
            train.get("clean_exec_label"), index=train.index, default=0.0
        )
        .fillna(0.0)
        .to_numpy(dtype=np.float32),
        "bad_mae": _num(
            train.get("first_touch_bad_mae_1r"), index=train.index, default=0.0
        )
        .fillna(0.0)
        .to_numpy(dtype=np.float32),
        "timeout": _num(train.get("timeout"), index=train.index, default=0.0)
        .fillna(0.0)
        .to_numpy(dtype=np.float32),
        "dirty_positive": _num(
            train.get("dirty_positive"), index=train.index, default=0.0
        )
        .fillna(0.0)
        .to_numpy(dtype=np.float32),
        "utility": _num(train.get("exec_margin"), index=train.index, default=0.0)
        .fillna(0.0)
        .to_numpy(dtype=np.float32),
        "behavior_archetype": behavior_arch,
    }


def _append_post_selection_ood_features(
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    selected: list[str],
    *,
    output_features: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    base_features = [str(c) for c in selected if str(c) in x_train.columns]
    if len(base_features) < 3:
        return x_train, x_valid, []
    train_arr = x_train.loc[:, base_features].to_numpy(dtype=np.float32, copy=True)
    valid_arr = x_valid.reindex(columns=base_features, fill_value=np.nan).to_numpy(
        dtype=np.float32, copy=True
    )
    finite_train = np.isfinite(train_arr)
    safe_train = np.where(finite_train, train_arr, np.nan).astype(
        np.float32, copy=False
    )
    mean = np.nanmean(safe_train, axis=0).astype(np.float32)
    std = np.nanstd(safe_train, axis=0).astype(np.float32)
    q25 = np.nanquantile(safe_train, 0.25, axis=0).astype(np.float32)
    q75 = np.nanquantile(safe_train, 0.75, axis=0).astype(np.float32)
    mean = np.nan_to_num(mean, nan=0.0).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype(np.float32)
    q25 = np.nan_to_num(q25, nan=mean).astype(np.float32)
    q75 = np.nan_to_num(q75, nan=mean).astype(np.float32)
    iqr = np.maximum(q75 - q25, 1e-6).astype(np.float32)
    lower = q25 - 1.5 * iqr
    upper = q75 + 1.5 * iqr

    def _metrics(arr: np.ndarray) -> dict[str, np.ndarray]:
        finite = np.isfinite(arr)
        filled = np.where(finite, arr, mean).astype(np.float32, copy=False)
        z = (filled - mean) / std
        abs_z = np.abs(z).astype(np.float32, copy=False)
        exceed = ((filled < lower) | (filled > upper)) & finite
        return {
            "meta_sel_ood_abs_z_mean": np.mean(abs_z, axis=1).astype(np.float32),
            "meta_sel_ood_abs_z_max": np.max(abs_z, axis=1).astype(np.float32),
            "meta_sel_ood_abs_z_p95": np.quantile(abs_z, 0.95, axis=1).astype(
                np.float32
            ),
            "meta_sel_ood_iqr_exceed_frac": np.mean(exceed, axis=1).astype(np.float32),
            "meta_sel_ood_missing_frac": np.mean(~finite, axis=1).astype(np.float32),
            "meta_sel_ood_centroid_l2": np.sqrt(np.mean(z * z, axis=1)).astype(
                np.float32
            ),
        }

    x_train = x_train.copy()
    x_valid = x_valid.copy()
    train_metrics = _metrics(train_arr)
    valid_metrics = _metrics(valid_arr)
    requested = [
        str(name)
        for name in (output_features or META_POST_SELECTION_OOD_FEATURE_NAMES)
        if str(name) in set(ALL_META_POST_SELECTION_OOD_FEATURE_NAMES)
    ]
    for name in requested:
        x_train[name] = train_metrics[name]
        x_valid[name] = valid_metrics[name]
    return x_train, x_valid, requested


def _conditionally_select_post_selection_ood_features(
    selector_x: pd.DataFrame,
    selector_y: np.ndarray,
    selector_train: pd.DataFrame,
    *,
    core_features: Sequence[str],
    sample_weight: np.ndarray,
    timestamps: pd.Series,
    assets: np.ndarray,
    returns: np.ndarray,
    fold: str,
    seed: int,
) -> tuple[list[str], dict[str, list[str]], pd.DataFrame]:
    """Keep derived OOD features only when they add value beyond the core.

    OOD statistics cannot be part of the initial candidate universe because
    their reference distribution must be built from the selected core.  This
    second, train-only MDA pass protects every core feature and asks whether
    any of the six OOD summaries adds incremental top-k value.  It therefore
    cannot displace the base score/rank anchors or become a forced feature.
    """
    core = [str(c) for c in core_features if str(c) in selector_x.columns]
    if len(core) < 3 or len(selector_x) < 500:
        return [], {}, pd.DataFrame()
    ood_x, _, all_ood = _append_post_selection_ood_features(
        selector_x.loc[:, core],
        selector_x.loc[:, core],
        core,
        output_features=ALL_META_POST_SELECTION_OOD_FEATURE_NAMES,
    )
    if not all_ood:
        return [], {}, pd.DataFrame()
    cfg = {
        "protected_features": core,
        "mda_config": {
            "enabled": True,
            "objective": "topk_opportunity_precision",
            "topk_fracs": [0.20, 0.10, 0.15],
            "topk_frac_weights": [0.20, 0.15, 0.20],
            "positive_label": 1,
            "use_sample_weight": True,
            "protected_features": core,
            # The OOD stage is an incremental evidence test. It must not gain
            # admission merely to satisfy broad family coverage.
            "feature_family_coverage": False,
            "group_first_screen_enabled": True,
            "group_first_screen_kind": "feature_family",
            "group_first_drop_null": True,
        },
    }
    candidate = train_lgbm_stability_candidate(
        ood_x,
        selector_y,
        sample_weight=sample_weight,
        random_state=int(seed) + 71_003,
        mode="regressor",
        timestamps=timestamps,
        assets=assets,
        returns=returns,
        hpo_objective_mode="train_meta",
        cfg=cfg,
        label_context=_feature_selection_label_context(selector_train),
        reference_artifact_dir=Path("/private/tmp")
        / f"s52_meta_lgbm_ood_incremental_{abs(hash(str(fold))) % 10_000_000}",
    )
    if not candidate:
        return [], {}, pd.DataFrame()
    selected = [
        str(name)
        for name in candidate.get("selected_feature_names", [])
        if str(name) in set(all_ood)
    ]
    selected_by_side: dict[str, list[str]] = {}
    side_raw = dict(candidate.get("metrics", {}) or {}).get(
        "per_side_feature_selection_selected_features", {}
    )
    if isinstance(side_raw, dict):
        for side in ("long", "short"):
            selected_by_side[side] = [
                str(name)
                for name in side_raw.get(side, []) or []
                if str(name) in set(all_ood)
            ]
    stats = candidate.get("feature_stats")
    if not isinstance(stats, pd.DataFrame) or stats.empty or "feature" not in stats:
        stats = pd.DataFrame({"feature": all_ood})
    else:
        stats = stats.loc[stats["feature"].astype(str).isin(set(all_ood))].copy()
    if not stats.empty:
        stats["conditional_on_core"] = "|".join(core)
        stats["selection_stage"] = "post_selection_ood_incremental"
        stats["selected"] = stats["feature"].astype(str).isin(set(selected))
    return list(dict.fromkeys(selected)), selected_by_side, stats


def _load_fixed_selected_features(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            values = (
                payload.get("selected_features")
                or payload.get("selected_feature_union")
                or payload.get("features")
            )
        else:
            values = payload
        if values is None:
            raise ValueError(f"No selected feature list found in {path}")
        return list(dict.fromkeys(str(v) for v in values if str(v).strip()))
    frame = pd.read_csv(path)
    feature_col = None
    for candidate in ("feature", "selected_feature", "name"):
        if candidate in frame.columns:
            feature_col = candidate
            break
    if feature_col is None:
        raise ValueError(f"Could not find a feature column in {path}")
    if "selected" in frame.columns:
        selected = frame["selected"]
        if selected.dtype == object:
            mask = selected.astype(str).str.lower().isin({"1", "true", "yes"})
        else:
            mask = selected.astype(bool)
        frame = frame.loc[mask].copy()
    if "rank" in frame.columns:
        frame = frame.sort_values("rank", kind="stable")
    return list(
        dict.fromkeys(str(v) for v in frame[feature_col].tolist() if str(v).strip())
    )


def _load_fixed_selected_features_by_side(
    path: Path | None,
) -> dict[str, list[str]] | None:
    if path is None or path.suffix.lower() != ".json" or not path.exists():
        return None
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        return None
    raw = payload.get("selected_features_by_side")
    if not isinstance(raw, dict):
        return None
    result = {
        side: list(dict.fromkeys(str(v) for v in raw.get(side, []) if str(v).strip()))
        for side in ("long", "short")
    }
    return result if all(result.values()) else None


def _load_fixed_model_params(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    if "classifier" in payload or "regressor" in payload:
        return payload
    params: dict[str, Any] = {}
    if isinstance(payload.get("classifier_params"), dict):
        params["classifier"] = dict(payload["classifier_params"])
    if isinstance(payload.get("regressor_params"), dict):
        params["regressor"] = dict(payload["regressor_params"])
    if not params and isinstance(payload.get("model_params"), dict):
        raw = payload["model_params"]
        if "classifier" in raw or "regressor" in raw:
            return raw
        params["classifier"] = dict(raw)
    if not params:
        raise ValueError(f"No fixed model params found in {path}")
    return params


def _select_features_by_lgbm_pipeline(
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    train: pd.DataFrame,
    *,
    target_name: str,
    top_n: int,
    fold: str,
    seed: int,
    target_contract: dict[str, Any] | None = None,
    target_strength_weight_spec: (
        TargetStrengthWeightSpec | dict[str, TargetStrengthWeightSpec] | None
    ) = None,
    base_weight_arm: str | None = None,
    strict_handoff_contract: bool = False,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[str],
    dict[str, list[str]],
    pd.DataFrame,
]:
    y = _feature_selection_target(train, target_name).replace([np.inf, -np.inf], np.nan)
    valid_y = y.notna().to_numpy(dtype=bool)
    if int(valid_y.sum()) < 500 or int(y.loc[valid_y].nunique(dropna=True)) < 3:
        raise ValueError(
            "Canonical lgbm_pipeline feature selection requires at least 500 rows and target variation."
        )
    selector_x = (
        x_train.loc[valid_y].reset_index(drop=True).astype(np.float32, copy=False)
    )
    selector_train = train.loc[valid_y].reset_index(drop=True)
    selector_y = (
        y.loc[valid_y]
        .reset_index(drop=True)
        .astype(np.float32)
        .to_numpy(dtype=np.float32)
    )
    base_soft, _base_soft_col = _base_soft_label_target(
        selector_train, target_contract=target_contract
    )
    if target_strength_weight_spec is None:
        sample_weight = _base_style_weights_for_soft_label(
            selector_train,
            base_soft,
            weight_arm=str(base_weight_arm or "W7_timestamp_balanced"),
        )
    else:
        sample_weight, _ = _train_only_target_strength_weights(
            selector_train,
            base_soft,
            sample_weight_spec=target_strength_weight_spec,
            strict=bool(strict_handoff_contract),
        )
    timestamps = pd.to_datetime(selector_train.get("__ts__"), utc=True, errors="coerce")
    assets = (
        selector_train.get(
            "__symbol__", pd.Series("unknown", index=selector_train.index)
        )
        .astype(str)
        .to_numpy()
    )
    returns = _num(selector_train.get("ev_after_1pct"), index=selector_train.index)
    if returns.isna().all():
        returns = _num(
            selector_train.get("exec_margin"), index=selector_train.index, default=0.0
        )
    protected_features = [
        feature
        for feature in META_PROTECTED_BASE_FEATURES
        if feature in selector_x.columns
    ]
    protected_features = list(
        dict.fromkeys(
            [*protected_features, *_meta_structural_features(selector_x.columns)]
        )
    )
    pre_mda_bypass_features = _meta_pre_mda_bypass_features(selector_x.columns)
    cfg = {
        # Group-level MDA is conditional on the protected base anchors.  It
        # lets AE/GMM, leaf, OOD and drift families demonstrate incremental
        # value rather than forcing any family into the final contract.
        "protected_features": protected_features,
        "pre_mda_bypass_features": pre_mda_bypass_features,
        "mda_config": {
            "enabled": True,
            "objective": "topk_opportunity_precision",
            "topk_fracs": [0.20, 0.10, 0.15],
            "topk_frac_weights": [0.20, 0.15, 0.20],
            "positive_label": 1,
            "use_sample_weight": True,
            "protected_features": protected_features,
            "pre_mda_bypass_features": pre_mda_bypass_features,
            # The meta layer has a deliberately broad context universe. Keep
            # the observable base-score anchors, but require AE/GMM, leaf,
            # drift, reliability and other context families to prove
            # incremental value inside the side x archetype objective. Broad
            # family quotas otherwise turn a coverage audit into forced model
            # inputs and made the recent meta contracts unnecessarily base-like.
            "feature_family_coverage": False,
            "group_first_screen_enabled": True,
            "group_first_screen_kind": "feature_family",
            "group_first_drop_null": True,
            # Archetype awareness belongs in the pre-screen.  The final
            # long/short MDA must give each surviving archetype equal voice
            # rather than carrying row-count or target-strength weights across
            # archetypes; this is the validated Pack-B selector contract.
            "side_tail_across_archetypes_unweighted": True,
        }
    }
    candidate = train_lgbm_stability_candidate(
        selector_x,
        selector_y,
        sample_weight=sample_weight,
        random_state=int(seed),
        mode="regressor",
        timestamps=timestamps,
        assets=assets,
        returns=returns.fillna(0.0).to_numpy(dtype=np.float32),
        hpo_objective_mode="train_meta",
        cfg=cfg,
        label_context=_feature_selection_label_context(selector_train),
        reference_artifact_dir=Path("/private/tmp")
        / f"s52_meta_lgbm_pipeline_fs_{abs(hash(str(fold))) % 10_000_000}",
    )
    if not candidate:
        raise RuntimeError(
            "Canonical lgbm_pipeline feature selection returned no candidate."
        )
    selected = [
        str(c)
        for c in candidate.get("selected_feature_names", [])
        if str(c) in x_train.columns
    ]
    selected = list(dict.fromkeys(protected_features + selected))
    if int(top_n) > 0:
        selected = selected[: int(top_n)]
    if not selected:
        raise RuntimeError(
            "Canonical lgbm_pipeline feature selection returned no selected feature names."
        )
    candidate_metrics = dict(candidate.get("metrics", {}) or {})
    selected_by_side_raw = candidate_metrics.get(
        "per_side_feature_selection_selected_features", {}
    )
    selected_by_side: dict[str, list[str]] = {}
    if isinstance(selected_by_side_raw, dict):
        for side in ("long", "short"):
            side_features = [
                str(c)
                for c in selected_by_side_raw.get(side, []) or []
                if str(c) in x_train.columns
            ]
            if int(top_n) > 0:
                side_features = side_features[: int(top_n)]
            if side_features:
                selected_by_side[side] = list(
                    dict.fromkeys(protected_features + side_features)
                )
    if set(selected_by_side) != {"long", "short"}:
        # A global fallback remains available only when one side lacks enough
        # rows for independent MDA. Normal meta training should persist both
        # side contracts and never silently merge them here.
        selected_by_side = {side: list(selected) for side in ("long", "short")}
    # OOD is derived from the selected core, so it receives its own
    # conditional MDA pass rather than being hard-appended to every contract.
    # Each side can therefore keep a different OOD subset.
    ood_features, ood_by_side, ood_selection_stats = (
        _conditionally_select_post_selection_ood_features(
            selector_x,
            selector_y,
            selector_train,
            core_features=selected,
            sample_weight=sample_weight,
            timestamps=timestamps,
            assets=assets,
            returns=returns.fillna(0.0).to_numpy(dtype=np.float32),
            fold=fold,
            seed=int(seed),
        )
    )
    if ood_features:
        selected = list(dict.fromkeys([*selected, *ood_features]))
        selected_by_side = {
            side: list(
                dict.fromkeys(
                    [*features, *(ood_by_side.get(side, []) or [])]
                )
            )
            for side, features in selected_by_side.items()
        }
    x_train_selected = x_train.loc[:, [c for c in selected if c in x_train.columns]]
    x_valid_selected = x_valid.reindex(columns=x_train_selected.columns, fill_value=0.0)
    if ood_features:
        core = [
            c for c in selected if c not in ALL_META_POST_SELECTION_OOD_FEATURE_NAMES
        ]
        x_train_selected, x_valid_selected, _ = _append_post_selection_ood_features(
            x_train.loc[:, core],
            x_valid.reindex(columns=core, fill_value=0.0),
            core,
            output_features=ood_features,
        )
    stats = candidate.get("feature_stats")
    if isinstance(stats, pd.DataFrame) and "feature" in stats.columns:
        rows = stats.copy()
    else:
        rows = pd.DataFrame({"feature": list(x_train.columns)})
    if "feature" not in rows.columns:
        rows["feature"] = list(x_train.columns[: len(rows)])
    selected_set = set(selected)
    rows["feature"] = rows["feature"].astype(str)
    rows["selected"] = rows["feature"].isin(selected_set)
    rows["meta_structural_feature"] = rows["feature"].isin(
        set(protected_features)
    )
    rows["meta_pre_mda_bypass"] = rows["feature"].isin(
        set(pre_mda_bypass_features)
    )
    if "rank" not in rows.columns:
        rows["_rank_score"] = (
            pd.to_numeric(rows.get("feature_score"), errors="coerce")
            .fillna(pd.to_numeric(rows.get("mda_mean"), errors="coerce"))
            .fillna(pd.to_numeric(rows.get("permutation_score"), errors="coerce"))
            .fillna(0.0)
        )
        rows["rank"] = (
            rows["_rank_score"].rank(method="first", ascending=False).astype(int)
        )
        rows = rows.drop(columns=["_rank_score"], errors="ignore")
    if "score" not in rows.columns:
        score_source = None
        for col in (
            "feature_score",
            "mda_mean",
            "median_perm",
            "permutation_score",
            "gain_rank_score",
        ):
            if col in rows.columns:
                score_source = col
                break
        rows["score"] = (
            pd.to_numeric(rows[score_source], errors="coerce")
            if score_source
            else np.nan
        )
    rows["fold"] = str(fold)
    rows["feature_selection_target"] = str(target_name)
    rows["feature_selection_method"] = "lgbm_pipeline_staged"
    rows["feature_selection_status"] = "ok"
    rows["feature_selection_requested_top_n"] = int(top_n)
    rows["feature_selection_auto_selected_count"] = int(len(selected))
    rows["feature_selection_auto_mode"] = (
        "lgbm_pipeline_iterative_prune" if int(top_n) <= 0 else "explicit_top_n_cap"
    )
    rows["lgbm_pipeline_selected_count"] = int(len(selected))
    rows["lgbm_pipeline_input_feature_count"] = int(x_train.shape[1])
    rows["lgbm_pipeline_selector_rows"] = int(len(selector_x))
    if not ood_selection_stats.empty:
        ood_rows = ood_selection_stats.copy()
        ood_rows["fold"] = str(fold)
        ood_rows["feature_selection_target"] = str(target_name)
        ood_rows["feature_selection_method"] = "lgbm_pipeline_staged"
        ood_rows["feature_selection_status"] = "ok"
        ood_rows["feature_selection_requested_top_n"] = int(top_n)
        ood_rows["feature_selection_auto_selected_count"] = int(len(selected))
        ood_rows["feature_selection_auto_mode"] = "conditional_incremental_ood"
        ood_rows["lgbm_pipeline_selected_count"] = int(len(selected))
        ood_rows["lgbm_pipeline_input_feature_count"] = int(x_train.shape[1])
        ood_rows["lgbm_pipeline_selector_rows"] = int(len(selector_x))
        rows = pd.concat([rows, ood_rows], ignore_index=True, sort=False)
    side_membership = {
        feature: ",".join(
            side
            for side in ("long", "short")
            if feature in set(selected_by_side.get(side, []))
        )
        for feature in selected
    }
    rows["selected_by_sides"] = rows["feature"].astype(str).map(side_membership)
    rows["side_specific_contract"] = True
    return (
        x_train_selected.loc[:, selected],
        x_valid_selected.reindex(columns=selected, fill_value=0.0),
        selected,
        selected_by_side,
        rows,
    )


def _fit_classifier(
    x: pd.DataFrame,
    y: pd.Series,
    train: pd.DataFrame,
    seed: int,
    *,
    lgbm_params: dict[str, Any] | None = None,
) -> Any:
    target = _num(y).fillna(0.0).astype(int)
    if int(target.nunique(dropna=True)) < 2:
        return None
    weights = _classification_weights(target, train)
    if _LIGHTGBM_AVAILABLE and LGBMClassifier is not None:
        params = _lgbm_params(DEFAULT_LGBM_CLASSIFIER_PARAMS, lgbm_params)
        model = LGBMClassifier(
            objective="binary",
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            num_leaves=params["num_leaves"],
            max_depth=params["max_depth"],
            min_child_samples=params["min_child_samples"],
            subsample=params["subsample"],
            subsample_freq=1,
            colsample_bytree=params["colsample_bytree"],
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
            random_state=int(seed),
            n_jobs=_meta_lgbm_n_jobs(),
            verbosity=-1,
        )
        model.fit(x, target, sample_weight=weights)
        return model
    model = ExtraTreesClassifier(
        n_estimators=220,
        max_depth=7,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced",
        random_state=int(seed),
        n_jobs=_meta_lgbm_n_jobs(),
    )
    model.fit(x, target, sample_weight=weights)
    return model


def _base_style_metrics_for_weights(train: pd.DataFrame) -> pd.DataFrame:
    index = train.index
    return pd.DataFrame(
        {
            "u_policy_net": _num(
                train.get("u_policy_net", train.get("exec_margin")),
                index=index,
                default=0.0,
            ).fillna(0.0),
            "mae_norm": _num(
                train.get("mae_norm", train.get("first_touch_full_path_mae_norm")),
                index=index,
                default=0.0,
            ).fillna(0.0),
            "mfe_norm": _num(
                train.get("mfe_norm", train.get("first_touch_full_path_mfe_norm")),
                index=index,
                default=0.0,
            ).fillna(0.0),
            "bars_to_mfe": _num(
                train.get("first_touch_bar"), index=index, default=24.0
            ).fillna(24.0),
            "barrier": _num(
                train.get("__archetype_policy_max_barrier__"), index=index, default=0.0
            ).fillna(0.0),
            "is_timeout": _num(train.get("timeout"), index=index, default=0.0).fillna(
                0.0
            ),
            "side": np.where(
                train.get("side_name", pd.Series("long", index=index))
                .astype(str)
                .str.lower()
                .eq("short"),
                -1.0,
                1.0,
            ),
        },
        index=index,
    )


def _base_style_weights_for_soft_label(
    train: pd.DataFrame,
    target: pd.Series,
    *,
    weight_arm: str = "W7_timestamp_balanced",
) -> pd.Series:
    frame = train.copy(deep=False)
    if "__ts__" in frame.columns:
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    target_frame = pd.DataFrame(
        {"target_soft": _num(target, index=train.index).clip(0.0, 1.0)},
        index=train.index,
    )
    try:
        weights = _base_weight_series(
            frame=frame,
            metrics=_base_style_metrics_for_weights(frame),
            target=target_frame,
            arm=str(weight_arm),
        )
    except Exception:
        return pd.Series(1.0, index=train.index, dtype=np.float32)
    return (
        _num(weights, index=train.index, default=1.0)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
        .astype(np.float32)
    )


def _base_weight_context(train: pd.DataFrame, valid_mask: pd.Series) -> pd.DataFrame:
    cols = [
        "__ts__",
        "u_policy_net",
        "exec_margin",
        "mae_norm",
        "first_touch_full_path_mae_norm",
        "mfe_norm",
        "first_touch_full_path_mfe_norm",
        "first_touch_bar",
        "__archetype_policy_max_barrier__",
        "timeout",
        "side_name",
    ]
    available = [col for col in cols if col in train.columns]
    return train.loc[valid_mask, available].copy()


def _fit_base_soft_label_model(
    x: pd.DataFrame,
    y: pd.Series,
    train: pd.DataFrame,
    seed: int,
    *,
    lgbm_params: dict[str, Any] | None = None,
    sample_weight_multiplier: pd.Series | np.ndarray | None = None,
    target_strength_weight_spec: (
        TargetStrengthWeightSpec | dict[str, TargetStrengthWeightSpec] | None
    ) = None,
    base_weight_arm: str | None = None,
    strict_handoff_contract: bool = False,
    weight_diagnostics_out: dict[str, Any] | None = None,
) -> Any:
    target = _num(y).replace([np.inf, -np.inf], np.nan).clip(0.0, 1.0)
    valid = target.notna()
    if int(valid.sum()) < 50 or float(target.loc[valid].std()) <= 1e-12:
        return None
    train_valid = train.loc[valid].copy(deep=False)
    if target_strength_weight_spec is None:
        # Named incumbent base weights are reproduced exactly on each train fold.
        weight_context = _base_weight_context(train, valid)
        resolved_arm = str(base_weight_arm or "W7_timestamp_balanced")
        weights = _base_style_weights_for_soft_label(
            weight_context, target.loc[valid], weight_arm=resolved_arm
        )
        weight_diagnostics = {
            "schema": "base_weight_arm_v1",
            "weight_arm": resolved_arm,
        }
    else:
        weights, weight_diagnostics = _train_only_target_strength_weights(
            train_valid,
            target.loc[valid],
            sample_weight_spec=target_strength_weight_spec,
            strict=bool(strict_handoff_contract),
        )
    if weight_diagnostics_out is not None:
        weight_diagnostics_out.clear()
        weight_diagnostics_out.update(_json_safe(weight_diagnostics))
    if sample_weight_multiplier is not None:
        multiplier = _num(
            sample_weight_multiplier,
            index=x.index,
            default=1.0,
        ).loc[valid]
        multiplier = (
            multiplier.replace([np.inf, -np.inf], np.nan)
            .fillna(1.0)
            .clip(0.25, 4.0)
            .astype(np.float32)
        )
        weights = weights.mul(multiplier, fill_value=1.0)
        mean_weight = float(weights.mean())
        if math.isfinite(mean_weight) and mean_weight > 1e-8:
            weights = (weights / mean_weight).astype(np.float32)
    if _LIGHTGBM_AVAILABLE and LGBMRegressor is not None:
        params = _lgbm_params(DEFAULT_LGBM_CLASSIFIER_PARAMS, lgbm_params)
        model = LGBMRegressor(
            objective="regression",
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            num_leaves=params["num_leaves"],
            max_depth=params["max_depth"],
            min_child_samples=params["min_child_samples"],
            subsample=params["subsample"],
            subsample_freq=1,
            colsample_bytree=params["colsample_bytree"],
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
            random_state=int(seed),
            n_jobs=_meta_lgbm_n_jobs(),
            verbosity=-1,
        )
        model.fit(
            x.loc[valid], target.loc[valid].astype(np.float32), sample_weight=weights
        )
        return model
    model = ExtraTreesRegressor(
        n_estimators=260,
        max_depth=7,
        min_samples_leaf=20,
        max_features="sqrt",
        random_state=int(seed),
        n_jobs=_meta_lgbm_n_jobs(),
    )
    model.fit(x.loc[valid], target.loc[valid].astype(np.float32), sample_weight=weights)
    return model


def _fit_regressor(
    x: pd.DataFrame,
    y: pd.Series,
    train: pd.DataFrame,
    seed: int,
    *,
    lgbm_params: dict[str, Any] | None = None,
) -> Any:
    target = _num(y).replace([np.inf, -np.inf], np.nan)
    valid = target.notna()
    if int(valid.sum()) < 50 or float(target.loc[valid].std()) <= 1e-12:
        return None
    weights = _regression_weights(target.loc[valid], train.loc[valid])
    if _LIGHTGBM_AVAILABLE and LGBMRegressor is not None:
        params = _lgbm_params(DEFAULT_LGBM_REGRESSOR_PARAMS, lgbm_params)
        model = LGBMRegressor(
            objective="regression",
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            num_leaves=params["num_leaves"],
            max_depth=params["max_depth"],
            min_child_samples=params["min_child_samples"],
            subsample=params["subsample"],
            subsample_freq=1,
            colsample_bytree=params["colsample_bytree"],
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
            random_state=int(seed),
            n_jobs=_meta_lgbm_n_jobs(),
            verbosity=-1,
        )
        model.fit(
            x.loc[valid], target.loc[valid].astype(np.float32), sample_weight=weights
        )
        return model
    model = ExtraTreesRegressor(
        n_estimators=260,
        max_depth=7,
        min_samples_leaf=20,
        max_features="sqrt",
        random_state=int(seed),
        n_jobs=_meta_lgbm_n_jobs(),
    )
    model.fit(x.loc[valid], target.loc[valid].astype(np.float32), sample_weight=weights)
    return model


def _fit_side_classifier(
    x: pd.DataFrame,
    y: pd.Series,
    train: pd.DataFrame,
    *,
    side: str,
    seed: int,
    min_rows: int = 80,
    lgbm_params: dict[str, Any] | None = None,
) -> Any:
    side_mask = (
        train["side_name"].astype(str).str.lower().eq(str(side).lower())
        if "side_name" in train.columns
        else pd.Series(False, index=train.index)
    )
    target = _num(y).replace([np.inf, -np.inf], np.nan)
    valid = side_mask & target.notna()
    if (
        int(valid.sum()) < int(min_rows)
        or int(target.loc[valid].nunique(dropna=True)) < 2
    ):
        return None
    return _fit_classifier(
        x.loc[valid], target.loc[valid], train.loc[valid], seed, lgbm_params=lgbm_params
    )


def _predict(model: Any, x: pd.DataFrame, *, classifier: bool) -> pd.Series:
    if model is None:
        return pd.Series(np.nan, index=x.index, dtype=np.float32)
    if classifier and hasattr(model, "predict_proba"):
        pred = model.predict_proba(x)
        if np.asarray(pred).ndim == 2 and pred.shape[1] >= 2:
            return pd.Series(pred[:, 1], index=x.index, dtype=np.float32)
    return pd.Series(np.asarray(model.predict(x), dtype=np.float32), index=x.index)


def _top_selected(
    frame: pd.DataFrame, score_col: str, keep_frac: float
) -> pd.DataFrame:
    valid = frame[_num(frame[score_col]).notna()].copy()
    if valid.empty:
        return valid
    n = max(1, int(math.ceil(len(valid) * float(keep_frac))))
    return valid.sort_values(score_col, ascending=False, kind="mergesort").head(n)


def _budget_selected(
    frame: pd.DataFrame,
    score_col: str,
    *,
    budget_frac: float,
    eligible: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    valid = frame[_num(frame[score_col]).notna()].copy()
    if valid.empty:
        return valid
    target_rows = max(1, int(math.ceil(len(frame) * float(budget_frac))))
    if eligible is not None:
        eligible_s = (
            pd.Series(eligible, index=frame.index)
            .reindex(valid.index)
            .fillna(False)
            .astype(bool)
        )
        valid = valid[eligible_s]
    if valid.empty:
        return valid
    return valid.sort_values(score_col, ascending=False, kind="mergesort").head(
        target_rows
    )


def _metric_row(
    frame: pd.DataFrame, score_col: str, keep_frac: float
) -> dict[str, Any]:
    selected = _top_selected(frame, score_col, keep_frac)
    tag = f"keep{int(round(keep_frac * 100)):03d}"
    oracle = frame.sort_values("exec_margin", ascending=False, kind="mergesort").head(
        len(selected)
    )
    overlap = (
        len(set(selected.index.tolist()) & set(oracle.index.tolist()))
        if len(selected)
        else 0
    )
    selected_ts = (
        pd.to_datetime(selected.get("__ts__"), utc=True, errors="coerce")
        if len(selected)
        else pd.Series(dtype="datetime64[ns, UTC]")
    )
    selected_ev = (
        _num(selected.get("ev_after_1pct"), index=selected.index)
        if len(selected)
        else pd.Series(dtype=float)
    )
    if len(selected) and selected_ts.notna().any():
        week_key = selected_ts.dt.tz_localize(None).dt.to_period("W").astype(str)
        worst_week_ev = float(selected_ev.groupby(week_key).mean().min())
    else:
        worst_week_ev = float("nan")
    out = {
        f"{tag}_rows": int(len(selected)),
        f"{tag}_exec_margin": _mean(selected.get("exec_margin")),
        f"{tag}_ev_after_1pct": _mean(selected.get("ev_after_1pct")),
        f"{tag}_worst_week_ev_after_1pct": worst_week_ev,
        f"{tag}_ret_net": _mean(selected.get("ret_net")),
        f"{tag}_u_policy_net": _mean(selected.get("u_policy_net")),
        f"{tag}_gross": _mean(selected.get("first_touch_gross")),
        f"{tag}_positive_margin_rate": _rate(_num(selected.get("exec_margin")).gt(0.0)),
        f"{tag}_clean_exec_precision": _rate(selected.get("clean_exec")),
        f"{tag}_dirty_positive_rate": _rate(selected.get("dirty_positive")),
        f"{tag}_first_touch_bad_mae": _rate(selected.get("first_touch_bad_mae_1r")),
        f"{tag}_full_path_bad_mae": _rate(selected.get("full_path_bad_mae_1r")),
        f"{tag}_timeout": _rate(selected.get("timeout")),
        f"{tag}_mfe_before_mae": _rate(selected.get("mfe_before_mae_1r")),
        f"{tag}_mae_before_mfe": _rate(selected.get("mae_before_mfe_1r")),
        f"{tag}_underwater_bars": _mean(selected.get("underwater_bars_before_mfe_1r")),
        f"{tag}_oracle_recall_exec_margin": float(overlap / max(len(oracle), 1))
        if len(selected)
        else float("nan"),
        f"{tag}_long_share": float(selected["side_name"].astype(str).eq("long").mean())
        if len(selected)
        else float("nan"),
        f"{tag}_short_share": float(
            selected["side_name"].astype(str).eq("short").mean()
        )
        if len(selected)
        else float("nan"),
    }
    return out


def _policy_metric_row(
    frame: pd.DataFrame,
    score_col: str,
    *,
    selector: str,
    policy_id: str,
    test_month: str,
    budget_frac: float,
    eligible: pd.Series | np.ndarray | None,
) -> dict[str, Any]:
    selected = _budget_selected(
        frame, score_col, budget_frac=budget_frac, eligible=eligible
    )
    target_rows = (
        max(1, int(math.ceil(len(frame) * float(budget_frac)))) if len(frame) else 0
    )
    oracle = frame.sort_values("exec_margin", ascending=False, kind="mergesort").head(
        len(selected)
    )
    overlap = (
        len(set(selected.index.tolist()) & set(oracle.index.tolist()))
        if len(selected)
        else 0
    )
    selected_ts = (
        pd.to_datetime(selected.get("__ts__"), utc=True, errors="coerce")
        if len(selected)
        else pd.Series(dtype="datetime64[ns, UTC]")
    )
    selected_ev = (
        _num(selected.get("ev_after_1pct"), index=selected.index)
        if len(selected)
        else pd.Series(dtype=float)
    )
    if len(selected) and selected_ts.notna().any():
        week_key = selected_ts.dt.tz_localize(None).dt.to_period("W").astype(str)
        worst_week_ev = float(selected_ev.groupby(week_key).mean().min())
    else:
        worst_week_ev = float("nan")
    return {
        "selector": str(selector),
        "policy_id": str(policy_id),
        "test_month": str(test_month),
        "budget_frac": float(budget_frac),
        "candidate_rows": int(len(frame)),
        "target_rows": int(target_rows),
        "selected_rows": int(len(selected)),
        "fill_rate": float(len(selected) / max(target_rows, 1))
        if target_rows
        else float("nan"),
        "no_trade_rate": float(1.0 - len(selected) / max(target_rows, 1))
        if target_rows
        else float("nan"),
        "exec_margin": _mean(selected.get("exec_margin")),
        "ev_after_1pct": _mean(selected.get("ev_after_1pct")),
        "worst_week_ev_after_1pct": worst_week_ev,
        "positive_margin_rate": _rate(_num(selected.get("exec_margin")).gt(0.0)),
        "clean_exec_precision": _rate(selected.get("clean_exec")),
        "dirty_positive_rate": _rate(selected.get("dirty_positive")),
        "first_touch_bad_mae": _rate(selected.get("first_touch_bad_mae_1r")),
        "full_path_bad_mae": _rate(selected.get("full_path_bad_mae_1r")),
        "timeout": _rate(selected.get("timeout")),
        "mfe_before_mae": _rate(selected.get("mfe_before_mae_1r")),
        "mae_before_mfe": _rate(selected.get("mae_before_mfe_1r")),
        "oracle_recall_exec_margin": float(overlap / max(len(oracle), 1))
        if len(selected)
        else float("nan"),
        "long_share": float(selected["side_name"].astype(str).eq("long").mean())
        if len(selected)
        else float("nan"),
        "short_share": float(selected["side_name"].astype(str).eq("short").mean())
        if len(selected)
        else float("nan"),
    }


def _selector_metrics(
    frame: pd.DataFrame, score_col: str, selector: str, test_month: str
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "selector": selector,
        "test_month": str(test_month),
        "candidate_rows": int(len(frame)),
        "scorable_rows": int(_num(frame[score_col]).notna().sum()),
        "base_candidate_exec_margin": _mean(frame.get("exec_margin")),
        "base_candidate_full_path_bad_mae": _rate(frame.get("full_path_bad_mae_1r")),
        "base_candidate_timeout": _rate(frame.get("timeout")),
        "spearman_exec_margin": _safe_spearman(
            frame.get("exec_margin", pd.Series(dtype=float)), frame[score_col]
        ),
        "spearman_ev_after_1pct": _safe_spearman(
            frame.get("ev_after_1pct", pd.Series(dtype=float)), frame[score_col]
        ),
        "spearman_underwater_bars": _safe_spearman(
            -_num(
                frame.get("underwater_bars_before_mfe_1r", pd.Series(dtype=float))
            ).fillna(0.0),
            frame[score_col],
        ),
        "auc_clean_exec": _safe_auc(
            frame.get("clean_exec", pd.Series(dtype=float)), frame[score_col]
        ),
        "ap_clean_exec": _safe_ap(
            frame.get("clean_exec", pd.Series(dtype=float)), frame[score_col]
        ),
        "auc_mfe_before_mae": _safe_auc(
            frame.get("mfe_before_mae_1r", pd.Series(dtype=float)), frame[score_col]
        ),
        "auc_avoids_mae_before_mfe": _safe_auc(
            1.0
            - _num(frame.get("mae_before_mfe_1r", pd.Series(dtype=float))).fillna(0.0),
            frame[score_col],
        ),
        "auc_positive_margin": _safe_auc(
            frame.get("positive_exec_margin", pd.Series(dtype=float)), frame[score_col]
        ),
        "ap_positive_margin": _safe_ap(
            frame.get("positive_exec_margin", pd.Series(dtype=float)), frame[score_col]
        ),
    }
    for frac in TOP_KEEP_FRACTIONS:
        row.update(_metric_row(frame, score_col, frac))
    return row


def _breakdown_rows(
    frame: pd.DataFrame,
    score_col: str,
    selector: str,
    test_month: str,
    keep_frac: float,
) -> list[dict[str, Any]]:
    selected = _top_selected(frame, score_col, keep_frac)
    rows: list[dict[str, Any]] = []
    if selected.empty:
        return rows
    group_cols = [
        ["side_name"],
        ["source_semantic_family"],
        ["side_name", "source_semantic_family"],
        ["side_name", "source_semantic_family", "aegmm_cluster"],
        ["side_name", "long_source_regime_split"],
        ["side_name", "regime_first_touch_bad_mae_score_bin"],
        ["side_name", "aegmm_expected_distance_bin"],
        ["side_name", "regime_lgbm_leaf_bad_mae_k4"],
    ]
    for cols in group_cols:
        if not all(col in selected.columns for col in cols):
            continue
        for key, group in selected.groupby(cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            record: dict[str, Any] = {
                "selector": selector,
                "test_month": str(test_month),
                "keep_frac": float(keep_frac),
                "grouping": "+".join(cols),
                "rows": int(len(group)),
                "exec_margin": _mean(group.get("exec_margin")),
                "clean_exec_precision": _rate(group.get("clean_exec")),
                "full_path_bad_mae": _rate(group.get("full_path_bad_mae_1r")),
                "timeout": _rate(group.get("timeout")),
                "mfe_before_mae": _rate(group.get("mfe_before_mae_1r")),
            }
            for col, value in zip(cols, key, strict=False):
                record[col] = value
            rows.append(record)
    return rows


def _base_conditioned_diagnostic_rows(
    frame: pd.DataFrame,
    *,
    test_month: str,
    selector: str,
    score_col: str,
    keep_frac: float = 0.30,
) -> list[dict[str, Any]]:
    selected = _top_selected(frame, score_col, keep_frac)
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return rows
    candidate = frame.copy()
    candidate["selection_scope"] = "candidate"
    selected = selected.copy()
    selected["selection_scope"] = f"selected_keep{int(round(keep_frac * 100)):02d}"
    combined = pd.concat([candidate, selected], axis=0, ignore_index=False)
    groupings = [
        ["side_name", "source_tag", "base_score_decile"],
        ["side_name", "source_tag", "base_rank_band"],
        ["side_name", "source_tag", "base_margin_band"],
        ["side_name", "source_semantic_family", "base_score_decile"],
        ["side_name", "source_semantic_family", "base_margin_band"],
    ]
    for cols in groupings:
        if not all(col in combined.columns for col in cols):
            continue
        for key, group in combined.groupby(["selection_scope"] + cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            record: dict[str, Any] = {
                "test_month": str(test_month),
                "selector": str(selector),
                "score_col": str(score_col),
                "grouping": "+".join(cols),
                "selection_scope": str(key[0]),
                "rows": int(len(group)),
                "mean_score": _mean(group.get(score_col)),
                "mean_base_score": _mean(group.get("score")),
                "mean_base_margin_to_cutoff": _mean(group.get("base_margin_to_cutoff")),
                "clean_exec_precision": _rate(group.get("clean_exec")),
                "full_path_bad_mae": _rate(group.get("full_path_bad_mae_1r")),
                "timeout": _rate(group.get("timeout")),
                "mfe_before_mae": _rate(group.get("mfe_before_mae_1r")),
                "mae_before_mfe": _rate(group.get("mae_before_mfe_1r")),
                "underwater_bars": _mean(group.get("underwater_bars_before_mfe_1r")),
                "exec_margin": _mean(group.get("exec_margin")),
            }
            for col, value in zip(cols, key[1:], strict=False):
                record[col] = value
            rows.append(record)
    return rows


def _feature_importance(
    model: Any, feature_names: list[str], label: str, test_month: str
) -> pd.DataFrame:
    if model is None or not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["test_month", "model", "feature", "importance"])
    values = np.asarray(model.feature_importances_, dtype=np.float64)
    if len(values) != len(feature_names):
        return pd.DataFrame(columns=["test_month", "model", "feature", "importance"])
    out = pd.DataFrame({"feature": feature_names, "importance": values})
    out = out[out["importance"] > 0].sort_values("importance", ascending=False).head(80)
    out.insert(0, "model", label)
    out.insert(0, "test_month", str(test_month))
    return out


def _summarize(folds: pd.DataFrame) -> pd.DataFrame:
    if folds.empty:
        return folds
    agg = (
        folds.groupby("selector", dropna=False)
        .agg(
            folds=("test_month", "nunique"),
            mean_candidate_rows=("candidate_rows", "mean"),
            mean_keep100_exec_margin=("keep100_exec_margin", "mean"),
            mean_keep100_full_path_bad_mae=("keep100_full_path_bad_mae", "mean"),
            mean_keep100_timeout=("keep100_timeout", "mean"),
            mean_keep050_exec_margin=("keep050_exec_margin", "mean"),
            mean_keep050_full_path_bad_mae=("keep050_full_path_bad_mae", "mean"),
            mean_keep030_exec_margin=("keep030_exec_margin", "mean"),
            mean_keep030_clean_exec_precision=("keep030_clean_exec_precision", "mean"),
            mean_keep030_full_path_bad_mae=("keep030_full_path_bad_mae", "mean"),
            mean_keep030_timeout=("keep030_timeout", "mean"),
            mean_keep030_oracle_recall=("keep030_oracle_recall_exec_margin", "mean"),
            mean_keep020_exec_margin=("keep020_exec_margin", "mean"),
            mean_keep020_ev_after_1pct=("keep020_ev_after_1pct", "mean"),
            mean_keep020_ret_net=("keep020_ret_net", "mean"),
            mean_keep020_u_policy_net=("keep020_u_policy_net", "mean"),
            mean_keep020_clean_exec_precision=("keep020_clean_exec_precision", "mean"),
            mean_keep020_dirty_positive_rate=("keep020_dirty_positive_rate", "mean"),
            mean_keep020_full_path_bad_mae=("keep020_full_path_bad_mae", "mean"),
            mean_keep020_first_touch_bad_mae=("keep020_first_touch_bad_mae", "mean"),
            mean_keep020_timeout=("keep020_timeout", "mean"),
            mean_keep020_oracle_recall=("keep020_oracle_recall_exec_margin", "mean"),
            mean_keep015_exec_margin=("keep015_exec_margin", "mean"),
            mean_keep015_ev_after_1pct=("keep015_ev_after_1pct", "mean"),
            mean_keep015_clean_exec_precision=("keep015_clean_exec_precision", "mean"),
            mean_keep015_dirty_positive_rate=("keep015_dirty_positive_rate", "mean"),
            mean_keep015_full_path_bad_mae=("keep015_full_path_bad_mae", "mean"),
            mean_keep015_first_touch_bad_mae=("keep015_first_touch_bad_mae", "mean"),
            mean_keep015_timeout=("keep015_timeout", "mean"),
            mean_keep010_exec_margin=("keep010_exec_margin", "mean"),
            mean_keep010_ev_after_1pct=("keep010_ev_after_1pct", "mean"),
            mean_keep010_ret_net=("keep010_ret_net", "mean"),
            mean_keep010_u_policy_net=("keep010_u_policy_net", "mean"),
            worst_keep010_exec_margin=("keep010_exec_margin", "min"),
            worst_keep010_ev_after_1pct=("keep010_ev_after_1pct", "min"),
            worst_week_keep010_ev_after_1pct=(
                "keep010_worst_week_ev_after_1pct",
                "min",
            ),
            mean_keep010_clean_exec_precision=("keep010_clean_exec_precision", "mean"),
            mean_keep010_dirty_positive_rate=("keep010_dirty_positive_rate", "mean"),
            mean_keep010_full_path_bad_mae=("keep010_full_path_bad_mae", "mean"),
            mean_keep010_first_touch_bad_mae=("keep010_first_touch_bad_mae", "mean"),
            mean_keep010_timeout=("keep010_timeout", "mean"),
            mean_keep010_oracle_recall=("keep010_oracle_recall_exec_margin", "mean"),
            mean_keep005_exec_margin=("keep005_exec_margin", "mean"),
            mean_keep005_ret_net=("keep005_ret_net", "mean"),
            mean_keep005_u_policy_net=("keep005_u_policy_net", "mean"),
            worst_keep005_exec_margin=("keep005_exec_margin", "min"),
            mean_keep005_clean_exec_precision=("keep005_clean_exec_precision", "mean"),
            mean_keep005_full_path_bad_mae=("keep005_full_path_bad_mae", "mean"),
            mean_keep005_timeout=("keep005_timeout", "mean"),
            mean_keep005_oracle_recall=("keep005_oracle_recall_exec_margin", "mean"),
            mean_spearman_exec_margin=("spearman_exec_margin", "mean"),
            mean_spearman_underwater_bars=("spearman_underwater_bars", "mean"),
            mean_auc_clean_exec=("auc_clean_exec", "mean"),
            mean_ap_clean_exec=("ap_clean_exec", "mean"),
            mean_auc_mfe_before_mae=("auc_mfe_before_mae", "mean"),
            mean_auc_avoids_mae_before_mfe=("auc_avoids_mae_before_mfe", "mean"),
        )
        .reset_index()
    )
    agg["meta_ev_frontier_objective"] = 100.0 * (
        0.20
        * _num(
            agg.get("mean_keep020_ev_after_1pct"), index=agg.index, default=0.0
        ).fillna(0.0)
        + 0.15
        * _num(
            agg.get("mean_keep010_ev_after_1pct"), index=agg.index, default=0.0
        ).fillna(0.0)
        + 0.10
        * _num(
            agg.get("worst_keep010_ev_after_1pct"), index=agg.index, default=0.0
        ).fillna(0.0)
        + 0.10
        * _num(
            agg.get("worst_week_keep010_ev_after_1pct"), index=agg.index, default=0.0
        ).fillna(0.0)
    ) + 0.20 * (
        (
            _num(
                agg.get("mean_keep015_clean_exec_precision"),
                index=agg.index,
                default=0.0,
            ).fillna(0.0)
            - _num(
                agg.get("mean_keep015_dirty_positive_rate"),
                index=agg.index,
                default=0.0,
            ).fillna(0.0)
        )
        + (
            _num(
                agg.get("mean_keep015_clean_exec_precision"),
                index=agg.index,
                default=0.0,
            ).fillna(0.0)
            - pd.concat(
                [
                    _num(
                        agg.get("mean_keep015_first_touch_bad_mae"),
                        index=agg.index,
                        default=0.0,
                    ),
                    _num(
                        agg.get("mean_keep015_full_path_bad_mae"),
                        index=agg.index,
                        default=0.0,
                    ),
                ],
                axis=1,
            )
            .max(axis=1)
            .fillna(0.0)
        )
    )
    agg["meta_smoke_status"] = np.where(
        (agg["mean_keep030_exec_margin"] > 0.0)
        & (agg["mean_keep030_timeout"] <= 0.12)
        & (
            agg["mean_keep030_full_path_bad_mae"]
            < agg["mean_keep100_full_path_bad_mae"]
        )
        & (agg["mean_keep020_exec_margin"] > 0.0),
        "candidate_for_deeper_meta_eval",
        "diagnostic_or_fail",
    )
    return agg.sort_values(
        [
            "meta_smoke_status",
            "meta_ev_frontier_objective",
            "mean_keep020_ev_after_1pct",
        ],
        ascending=[True, False, False],
    )


def _threshold_policy_rows(
    scored: pd.DataFrame, selector_cols: dict[str, str], test_month: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bad = _num(scored.get("score_meta_bad_path"), index=scored.index).fillna(1.0)
    timeout = _num(scored.get("score_meta_timeout"), index=scored.index).fillna(1.0)
    clean = _num(scored.get("score_meta_clean_exec"), index=scored.index).fillna(0.0)
    side = (
        scored.get("side_name", pd.Series("", index=scored.index))
        .astype(str)
        .str.lower()
    )
    long_clean = _num(scored.get("score_meta_long_clean_exec"), index=scored.index)
    side_clean = clean.where(~side.eq("long"), long_clean.fillna(clean))
    long_bad = _num(scored.get("score_meta_long_bad_path"), index=scored.index)
    side_bad = bad.where(~side.eq("long"), long_bad.fillna(bad))
    pos = _num(scored.get("score_meta_positive_margin"), index=scored.index).fillna(0.0)
    for selector, score_col in selector_cols.items():
        if score_col not in scored.columns:
            continue
        policies: list[tuple[str, pd.Series]] = [
            ("no_cap", pd.Series(True, index=scored.index)),
        ]
        for cap in BAD_PATH_CAPS:
            policies.append((f"bad_path_le_{cap:.2f}", bad.le(float(cap))))
        for cap in (0.55, 0.60, 0.65):
            policies.append(
                (
                    f"bad_path_le_{cap:.2f}_timeout_le_0.50",
                    bad.le(float(cap)) & timeout.le(0.50),
                )
            )
        for floor in CLEAN_EXEC_FLOORS:
            if floor <= 0.0:
                continue
            policies.append((f"clean_ge_{floor:.2f}", clean.ge(float(floor))))
            policies.append((f"side_clean_ge_{floor:.2f}", side_clean.ge(float(floor))))
        for long_floor, short_floor in ((0.55, 0.65), (0.60, 0.65), (0.65, 0.65)):
            policies.append(
                (
                    f"side_clean_ge_long_{long_floor:.2f}_short_{short_floor:.2f}",
                    np.where(
                        side.eq("long"),
                        side_clean.ge(float(long_floor)),
                        clean.ge(float(short_floor)),
                    ),
                )
            )
        for cap in (0.50, 0.55, 0.60):
            policies.append((f"side_bad_path_le_{cap:.2f}", side_bad.le(float(cap))))
            policies.append(
                (
                    f"side_bad_path_le_{cap:.2f}_side_clean_ge_0.55",
                    side_bad.le(float(cap)) & side_clean.ge(0.55),
                )
            )
        for floor in POSITIVE_MARGIN_FLOORS:
            if floor <= 0.0:
                continue
            policies.append((f"positive_margin_ge_{floor:.2f}", pos.ge(float(floor))))
        for cap in (0.55, 0.60, 0.65):
            for clean_floor in (0.45, 0.55):
                policies.append(
                    (
                        f"bad_path_le_{cap:.2f}_clean_ge_{clean_floor:.2f}",
                        bad.le(float(cap)) & clean.ge(float(clean_floor)),
                    )
                )
        for policy_id, eligible in policies:
            for budget_frac in POLICY_BUDGET_FRACTIONS:
                rows.append(
                    _policy_metric_row(
                        scored,
                        score_col,
                        selector=selector,
                        policy_id=policy_id,
                        test_month=test_month,
                        budget_frac=budget_frac,
                        eligible=eligible,
                    )
                )
    return rows


def _summarize_threshold_policies(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    summary = (
        rows.groupby(["selector", "policy_id", "budget_frac"], dropna=False)
        .agg(
            folds=("test_month", "nunique"),
            mean_target_rows=("target_rows", "mean"),
            mean_selected_rows=("selected_rows", "mean"),
            min_selected_rows=("selected_rows", "min"),
            mean_fill_rate=("fill_rate", "mean"),
            mean_no_trade_rate=("no_trade_rate", "mean"),
            mean_exec_margin=("exec_margin", "mean"),
            worst_exec_margin=("exec_margin", "min"),
            mean_ev_after_1pct=("ev_after_1pct", "mean"),
            worst_week_ev_after_1pct=("worst_week_ev_after_1pct", "min"),
            mean_positive_margin_rate=("positive_margin_rate", "mean"),
            mean_clean_exec_precision=("clean_exec_precision", "mean"),
            mean_dirty_positive_rate=("dirty_positive_rate", "mean"),
            mean_first_touch_bad_mae=("first_touch_bad_mae", "mean"),
            mean_full_path_bad_mae=("full_path_bad_mae", "mean"),
            max_full_path_bad_mae=("full_path_bad_mae", "max"),
            mean_timeout=("timeout", "mean"),
            max_timeout=("timeout", "max"),
            mean_mfe_before_mae=("mfe_before_mae", "mean"),
            mean_mae_before_mfe=("mae_before_mfe", "mean"),
            mean_oracle_recall=("oracle_recall_exec_margin", "mean"),
            mean_long_share=("long_share", "mean"),
            mean_short_share=("short_share", "mean"),
        )
        .reset_index()
    )
    summary["threshold_policy_status"] = np.where(
        (summary["mean_exec_margin"] > 0.0)
        & (summary["worst_exec_margin"] > 0.0)
        & (summary["mean_full_path_bad_mae"] <= 0.50)
        & (summary["mean_timeout"] <= 0.12)
        & (summary["min_selected_rows"] >= 20)
        & (summary["mean_fill_rate"] >= 0.20),
        "bad_mae_bar_candidate",
        np.where(
            (summary["mean_exec_margin"] > 0.0)
            & (summary["mean_full_path_bad_mae"] < 0.58)
            & (summary["mean_timeout"] <= 0.12),
            "risk_improved_diagnostic",
            "diagnostic_or_fail",
        ),
    )
    objective_by_policy: dict[tuple[str, str], float] = {}
    for key, group in summary.groupby(["selector", "policy_id"], dropna=False):
        budget = (
            _num(group.get("budget_frac"), index=group.index, default=0.0)
            .fillna(0.0)
            .round(4)
        )

        def pick_metric(budget_value: float, col: str, default: float = 0.0) -> float:
            match = group.loc[budget.eq(round(float(budget_value), 4)), col]
            if match.empty:
                return float(default)
            return _mean(match)

        top20_ev = pick_metric(0.20, "mean_ev_after_1pct")
        top10_ev = pick_metric(0.10, "mean_ev_after_1pct")
        worst_month_top10_ev = pick_metric(0.10, "worst_exec_margin")
        worst_week_top10_ev = pick_metric(0.10, "worst_week_ev_after_1pct")
        top15_clean = pick_metric(0.15, "mean_clean_exec_precision")
        top15_dirty = pick_metric(0.15, "mean_dirty_positive_rate")
        top15_bad_or_stop = max(
            pick_metric(0.15, "mean_first_touch_bad_mae"),
            pick_metric(0.15, "mean_full_path_bad_mae"),
        )
        objective_by_policy[key] = float(
            100.0
            * (
                0.20 * top20_ev
                + 0.15 * top10_ev
                + 0.10 * worst_month_top10_ev
                + 0.10 * worst_week_top10_ev
            )
            + 0.20 * ((top15_clean - top15_dirty) + (top15_clean - top15_bad_or_stop))
        )
    summary["meta_ev_frontier_objective"] = [
        objective_by_policy.get((row.selector, row.policy_id), float("-inf"))
        for row in summary.itertuples(index=False)
    ]
    return summary.sort_values(
        [
            "threshold_policy_status",
            "meta_ev_frontier_objective",
            "mean_ev_after_1pct",
            "mean_fill_rate",
        ],
        ascending=[True, False, False, False],
    )


def _selector_to_score_col(selector: str) -> str:
    selector = str(selector)
    if selector == "base_score":
        return "score_base"
    if selector.startswith("meta_"):
        return f"score_{selector}"
    return selector


def _group_topk_metrics(
    frame: pd.DataFrame,
    *,
    score_col: str,
    selector: str,
    group_cols: list[str],
    keep_fracs: tuple[float, ...] = (0.30, 0.20, 0.10, 0.05),
) -> pd.DataFrame:
    if frame.empty or score_col not in frame.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    work = frame.copy()
    for col in group_cols:
        if col not in work.columns:
            work[col] = "missing"
    for keys, group in work.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        score = _num(group.get(score_col), index=group.index)
        valid = group.loc[score.notna()].copy()
        if valid.empty:
            continue
        ordered = valid.assign(
            __score__=_num(valid.get(score_col), index=valid.index)
        ).sort_values(
            "__score__",
            ascending=False,
            kind="mergesort",
        )
        for frac in keep_fracs:
            selected = ordered.head(max(1, int(math.ceil(len(ordered) * float(frac)))))
            rec: dict[str, Any] = {
                "selector": str(selector),
                "score_col": str(score_col),
                "keep_frac": float(frac),
                "candidate_rows": int(len(valid)),
                "selected_rows": int(len(selected)),
                "exec_margin": _mean(selected.get("exec_margin")),
                "ev_after_1pct": _mean(selected.get("ev_after_1pct")),
                "ret_net": _mean(selected.get("ret_net")),
                "u_policy_net": _mean(selected.get("u_policy_net")),
                "positive_exec_margin_rate": _rate(
                    _num(selected.get("exec_margin")).gt(0.0)
                ),
                "clean_exec_precision": _rate(selected.get("clean_exec")),
                "dirty_positive_rate": _rate(selected.get("dirty_positive")),
                "first_touch_bad_mae_rate": _rate(
                    selected.get("first_touch_bad_mae_1r")
                ),
                "full_path_bad_mae_rate": _rate(selected.get("full_path_bad_mae_1r")),
                "timeout_rate": _rate(selected.get("timeout")),
                "mfe_before_mae_rate": _rate(selected.get("mfe_before_mae_1r")),
                "mae_before_mfe_rate": _rate(selected.get("mae_before_mfe_1r")),
                "mean_underwater_bars": _mean(
                    selected.get("underwater_bars_before_mfe_1r")
                ),
                "mean_score": _mean(selected.get(score_col)),
                "mean_base_score": _mean(selected.get("score_base")),
            }
            for col, value in zip(group_cols, keys, strict=False):
                rec[col] = "missing" if pd.isna(value) else value
            rows.append(rec)
    return pd.DataFrame(rows)


def _write_meta_oos_breakdown_reports(
    *,
    out_dir: Path,
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, Any]:
    report_dir = out_dir / "meta_oos_breakdown"
    report_dir.mkdir(parents=True, exist_ok=True)
    if predictions.empty:
        return {"enabled": False, "reason": "no_predictions"}
    best_selector = "base_score"
    if not summary.empty and "selector" in summary.columns:
        best_selector = str(summary.iloc[0].get("selector", "base_score"))
    selector_map = {
        "base_score": "score_base",
        best_selector: _selector_to_score_col(best_selector),
    }
    selector_map = {k: v for k, v in selector_map.items() if v in predictions.columns}
    if not selector_map:
        return {"enabled": False, "reason": "no_score_columns"}
    work = predictions.copy()
    for source_col, alias_col in LEDGER_CONTEXT_FEATURE_ALIASES.items():
        if source_col in work.columns and alias_col not in work.columns:
            work[alias_col] = work[source_col]
    if "policy_archetype" not in work.columns:
        if "archetype_policy_key" in work.columns:
            work["policy_archetype"] = work["archetype_policy_key"].astype(str)
        elif "__archetype_policy_key__" in work.columns:
            work["policy_archetype"] = work["__archetype_policy_key__"].astype(str)
        elif "source_tag" in work.columns:
            work["policy_archetype"] = work["source_tag"].astype(str)
        else:
            work["policy_archetype"] = "missing"
    if "calendar_month" not in work.columns:
        work["calendar_month"] = (
            pd.to_datetime(work.get("__ts__"), utc=True, errors="coerce")
            .dt.to_period("M")
            .astype(str)
        )
    output_files: dict[str, str] = {}
    groupings = {
        "month_side_archetype": ["calendar_month", "side_name", "policy_archetype"],
        "month_side_family": ["calendar_month", "side_name", "source_semantic_family"],
        "side_archetype": ["side_name", "policy_archetype"],
        "month": ["calendar_month"],
    }
    for selector, score_col in selector_map.items():
        for name, cols in groupings.items():
            report = _group_topk_metrics(
                work, score_col=score_col, selector=selector, group_cols=list(cols)
            )
            path = report_dir / f"{selector}_{name}.csv"
            report.to_csv(path, index=False)
            output_files[f"{selector}_{name}"] = str(path)
    comparison_frames = [
        _group_topk_metrics(
            work,
            score_col=score_col,
            selector=selector,
            group_cols=["calendar_month", "side_name", "policy_archetype"],
        )
        for selector, score_col in selector_map.items()
    ]
    comparison_frames = [df for df in comparison_frames if not df.empty]
    if comparison_frames:
        comparison_path = (
            report_dir / "base_vs_meta_month_side_archetype_comparison.csv"
        )
        pd.concat(comparison_frames, ignore_index=True).to_csv(
            comparison_path, index=False
        )
        output_files["base_vs_meta_month_side_archetype_comparison"] = str(
            comparison_path
        )
    manifest = {
        "enabled": True,
        "schema": "s52_meta_oos_breakdown_v1",
        "best_selector": best_selector,
        "selectors": selector_map,
        "metrics_source": "OOS prediction rows only",
        "objective_step": "8_meta_completion_breakdown_before_simple_policy_optimiser",
        "output_files": output_files,
    }
    (report_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def run_smoke(
    *,
    handoff_dir: Path,
    ledger_path: Path | None,
    handoff_path: Path | None = None,
    out_dir: Path,
    frontier: str,
    seed: int,
    train_scope: str = "selected",
    enable_base_prior_features: bool = False,
    enable_reliability_features: bool = False,
    enable_support_drift_features: bool = False,
    enable_hit_surprise_features: bool = False,
    enable_path_order_heads: bool = False,
    enable_path_order_blends: bool = False,
    feature_selection_top_n: int = 0,
    feature_selection_target: str = "ev_frontier",
    feature_selection_method: str = "auto",
    max_oos_model_age_days: int = 0,
    single_fit_oos_window: bool = False,
    validation_scope: str = "all",
    model_train_max_rows: int = 0,
    feature_selection_max_rows: int = 0,
    model_params: dict[str, Any] | None = None,
    model_profile_name: str = "baseline",
    meta_head_mode: str = "multi",
    minimal_artifacts: bool = False,
    fixed_selected_features: list[str] | None = None,
    fixed_selected_features_by_side: dict[str, list[str]] | None = None,
    side_specific_single_head: bool = True,
    eval_months: list[str] | None = None,
    fold_feature_builder: Any | None = None,
    fold_feature_profile_name: str = "none",
    extra_prediction_columns: list[str] | None = None,
    force_prediction_shards: bool = False,
    combine_prediction_shards: bool = True,
    save_fold_models: bool = False,
    handoff_columns: Sequence[str] | None = None,
    soft_label_paths: Sequence[Path] | None = None,
    ood_reference_features: Sequence[str] | None = None,
    single_head_score_override: Any | None = None,
    legacy_constant_zero_features: Sequence[str] = (),
    strict_handoff_contract: bool = False,
    active_sides: Sequence[str] | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    handoff_path = (
        Path(handoff_path)
        if handoff_path is not None
        else handoff_dir / "train_meta_regime_handoff.parquet"
    )
    if ledger_path is None:
        ledger_path = handoff_path.parent / "s52_trailing_regime_scored_ledger.parquet"
    joined_cache_key = (
        str(handoff_path.resolve()),
        str(Path(ledger_path).resolve()),
        str(frontier),
        tuple(sorted(str(col) for col in (handoff_columns or []))),
    )
    data = _JOINED_FRAME_CACHE.get(joined_cache_key)
    if data is None:
        data = _load_joined_frame(
            handoff_path,
            ledger_path,
            frontier,
            handoff_columns=handoff_columns,
        )
        _JOINED_FRAME_CACHE.clear()
        _JOINED_FRAME_CACHE[joined_cache_key] = data
    active_side_set = {
        str(side).strip().lower()
        for side in (active_sides or ())
        if str(side).strip().lower() in {"long", "short"}
    }
    if active_side_set:
        if "side_name" not in data.columns:
            raise ValueError("--active-sides requires the handoff side_name column")
        data = data.loc[
            data["side_name"].astype(str).str.lower().isin(active_side_set)
        ].copy()
        if data.empty:
            raise ValueError(f"No rows remain after active-side filter: {sorted(active_side_set)}")
    if bool(strict_handoff_contract) and str(frontier).lower() != "top30":
        raise ValueError("Strict meta handoff is defined only for the base top30 stream.")
    inherited_handoff_contract = _load_and_validate_handoff_contract(
        handoff_path=handoff_path,
        handoff_rows=data,
        strict=bool(strict_handoff_contract),
    )
    inherited_target_contract = dict(
        inherited_handoff_contract.get("base_target_contract", {}) or {}
    )
    inherited_weight_spec = _target_strength_spec_from_contract(
        inherited_handoff_contract
    )
    inherited_base_weight_arm = _base_weight_arm_from_contract(
        inherited_handoff_contract
    )
    soft_label_manifest: dict[str, Any] | None = None
    if soft_label_paths:
        data, soft_label_manifest = _attach_materialized_soft_labels(
            data,
            handoff_path=handoff_path,
            label_paths=soft_label_paths,
        )
    selected_col = _candidate_column(frontier)
    if train_scope == "selected":
        selected_mask = data[selected_col].fillna(False).astype(bool)
        if not bool(selected_mask.all()):
            data = data.loc[selected_mask].copy()
    elif train_scope != "all":
        raise ValueError("--train-scope must be selected or all")
    months = sorted(str(m) for m in data["month"].dropna().unique())
    if len(months) < 2:
        raise ValueError(f"Need at least two months, got {months}")
    ts_utc = pd.to_datetime(data["__ts__"], utc=True, errors="coerce")
    validation_windows: list[dict[str, Any]] = []
    eval_month_set = {str(m) for m in (eval_months or []) if str(m).strip()}
    eligible_eval_months = [
        str(month)
        for month in months[1:]
        if not eval_month_set or str(month) in eval_month_set
    ]
    if bool(single_fit_oos_window) and eligible_eval_months:
        periods = sorted(pd.Period(month) for month in eligible_eval_months)
        expected = [periods[0] + i for i in range(len(periods))]
        if periods != expected:
            raise ValueError(
                "--single-fit-oos-window requires contiguous evaluation months; "
                f"got={eligible_eval_months}"
            )
        window_start = pd.Timestamp(periods[0].start_time, tz="UTC")
        window_end = pd.Timestamp((periods[-1] + 1).start_time, tz="UTC")
        validation_windows.append(
            {
                "fold": f"{window_start:%Y-%m-%d}_{window_end:%Y-%m-%d}",
                "month": f"{periods[0]}_to_{periods[-1]}",
                "valid_start": window_start,
                "valid_end": window_end,
            }
        )
    for month in ([] if bool(single_fit_oos_window) else months[1:]):
        if eval_month_set and str(month) not in eval_month_set:
            continue
        month_start = pd.Timestamp(pd.Period(month).start_time, tz="UTC")
        month_end = pd.Timestamp((pd.Period(month) + 1).start_time, tz="UTC")
        if int(max_oos_model_age_days) > 0:
            start = month_start
            step = pd.Timedelta(days=int(max_oos_model_age_days))
            while start < month_end:
                end = min(start + step, month_end)
                validation_windows.append(
                    {
                        "fold": f"{start:%Y-%m-%d}_{end:%Y-%m-%d}",
                        "month": str(month),
                        "valid_start": start,
                        "valid_end": end,
                    }
                )
                start = end
        else:
            validation_windows.append(
                {
                    "fold": str(month),
                    "month": str(month),
                    "valid_start": month_start,
                    "valid_end": month_end,
                }
            )
    eligible_windows: list[dict[str, Any]] = []
    for window in validation_windows:
        train_rows = int(ts_utc.lt(window["valid_start"]).sum())
        valid_rows = int(
            (ts_utc.ge(window["valid_start"]) & ts_utc.lt(window["valid_end"])).sum()
        )
        if train_rows < 100 or valid_rows < 30:
            continue
        enriched = dict(window)
        enriched["train_rows_estimate"] = int(train_rows)
        enriched["valid_rows_estimate"] = int(valid_rows)
        eligible_windows.append(enriched)
    calibration_candidates = [
        w for w in eligible_windows if int(w["valid_rows_estimate"]) >= 1000
    ]
    if not calibration_candidates:
        calibration_candidates = eligible_windows
    calibration_window = (
        max(
            calibration_candidates,
            key=lambda w: (
                int(w["train_rows_estimate"]),
                int(w["valid_rows_estimate"]),
            ),
        )
        if calibration_candidates
        else None
    )
    calibration_fold = (
        str(calibration_window["fold"]) if calibration_window is not None else None
    )
    validation_scope_norm = str(validation_scope).strip().lower()
    if validation_scope_norm == "largest":
        validation_windows = (
            [calibration_window] if calibration_window is not None else []
        )
    elif validation_scope_norm in {"chronological", "chrono"}:
        validation_windows = list(eligible_windows)
    elif calibration_window is not None:
        validation_windows = [calibration_window] + [
            w for w in eligible_windows if str(w["fold"]) != calibration_fold
        ]
    else:
        validation_windows = []
    numeric_cols, categorical_cols = _feature_columns(
        data,
        enable_base_prior_features=enable_base_prior_features,
        enable_reliability_features=enable_reliability_features,
        enable_support_drift_features=enable_support_drift_features,
        enable_hit_surprise_features=enable_hit_surprise_features,
    )
    fold_rows: list[dict[str, Any]] = []
    threshold_policy_rows: list[dict[str, Any]] = []
    breakdown: list[dict[str, Any]] = []
    base_conditioned_diagnostics: list[dict[str, Any]] = []
    importances: list[pd.DataFrame] = []
    feature_selection_frames: list[pd.DataFrame] = []
    selected_features_by_fold: dict[str, list[str]] = {}
    selected_features_by_side_by_fold: dict[str, dict[str, list[str]]] = {}
    fold_feature_metadata: list[dict[str, Any]] = []
    saved_model_manifests: list[dict[str, Any]] = []
    label_purge_by_fold: dict[str, dict[str, Any]] = {}
    input_feature_contracts_by_fold: dict[str, dict[str, Any]] = {}
    prediction_frames: list[pd.DataFrame] = []
    prediction_shard_paths: list[Path] = []
    prediction_shard_dir = out_dir / "prediction_shards"
    classifier_params = dict(
        (model_params or {}).get("classifier", DEFAULT_LGBM_CLASSIFIER_PARAMS)
    )
    regressor_params = dict(
        (model_params or {}).get("regressor", DEFAULT_LGBM_REGRESSOR_PARAMS)
    )
    meta_head_mode = str(meta_head_mode).strip().lower()
    if meta_head_mode not in {"multi", "single_base_soft_label"}:
        raise ValueError("--meta-head-mode must be multi or single_base_soft_label")
    legacy_shared_feature_contract = bool(
        meta_head_mode == "single_base_soft_label"
        and side_specific_single_head
        and fixed_selected_features is not None
        and not fixed_selected_features_by_side
    )
    if legacy_shared_feature_contract:
        # Direct callers must follow the same rule as the CLI: a shared list
        # is historical provenance, not a valid side-local MDA contract.
        print(
            "[feature_selection] shared fixed list is incompatible with the "
            "side-aware meta contract; selecting long/short features anew.",
            flush=True,
        )
        fixed_selected_features = None
    fs_cache_key = (
        str(handoff_path.resolve()),
        str(Path(ledger_path).resolve()) if ledger_path is not None else None,
        str(frontier),
        str(train_scope),
        bool(enable_base_prior_features),
        bool(enable_reliability_features),
        bool(enable_support_drift_features),
        bool(enable_hit_surprise_features),
        int(feature_selection_top_n),
        str(feature_selection_target),
        str(feature_selection_method),
        int(max_oos_model_age_days),
        int(feature_selection_max_rows),
        str(meta_head_mode),
        str(fold_feature_profile_name),
        tuple(sorted(str(col) for col in (handoff_columns or []))),
    )
    cached_feature_selection = _FEATURE_SELECTION_CACHE.get(fs_cache_key)
    global_feature_names: list[str] | None = (
        # A frozen contract is immutable. Protected anchors are applied while
        # selecting a new contract, never appended to an already fitted model.
        list(map(str, fixed_selected_features))
        if fixed_selected_features is not None
        else (list(cached_feature_selection[0]) if cached_feature_selection else None)
    )
    global_feature_selection_df: pd.DataFrame | None = (
        cached_feature_selection[1].copy() if cached_feature_selection else None
    )
    global_feature_names_by_side: dict[str, list[str]] | None = None
    has_frozen_side_feature_contract = bool(fixed_selected_features_by_side)
    if fixed_selected_features_by_side:
        global_feature_names_by_side = {
            side: list(map(str, features))
            for side, features in fixed_selected_features_by_side.items()
            if str(side) in {"long", "short"}
        }
        # A side-local fixed contract is still a fixed contract.  Keep the
        # union solely as the matrix schema; each model below receives only
        # its own side list.  Without this, the legacy ``None`` sentinel
        # accidentally re-entered feature selection for side-aware replays.
        global_feature_names = list(
            dict.fromkeys(
                feature
                for side in ("long", "short")
                for feature in global_feature_names_by_side.get(side, [])
            )
        )
        if not global_feature_names:
            raise RuntimeError(
                "Frozen side-local meta feature contract produced an empty matrix union."
            )
    elif cached_feature_selection and len(cached_feature_selection) >= 3:
        global_feature_names_by_side = {
            str(side): list(features)
            for side, features in (cached_feature_selection[2] or {}).items()
        }
    elif global_feature_names is not None:
        global_feature_names_by_side = {
            side: list(global_feature_names) for side in ("long", "short")
        }
    if fixed_selected_features is not None or fixed_selected_features_by_side:
        global_feature_selection_df = pd.DataFrame(
            {
                "fold": ["fixed_hpo_selected_features"]
                * len(global_feature_names or []),
                "feature": list(global_feature_names or []),
                "rank": np.arange(
                    1, len(global_feature_names or []) + 1, dtype=np.int32
                ),
                "selected": True,
                "feature_selection_target": str(feature_selection_target),
                "feature_selection_method": "fixed_from_hpo",
                "feature_selection_status": "fixed_replay",
                "feature_selection_auto_selected_count": int(
                    len(global_feature_names or [])
                ),
            }
        )
    feature_selection_recorded = False
    meta_target_columns_used: set[str] = set()
    target_strength_weight_diagnostics_by_fold: dict[str, Any] = {}
    for fold_idx, window in enumerate(validation_windows, start=1):
        test_fold = str(window["fold"])
        test_month = str(window["month"])
        safe_fold = "".join(
            ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(test_fold)
        )
        shard_path = (
            prediction_shard_dir / f"predictions_{fold_idx:04d}_{safe_fold}.parquet"
        )
        should_shard_predictions = (
            bool(force_prediction_shards)
            or int(max_oos_model_age_days) > 0
            or len(validation_windows) > 12
        )
        if (
            should_shard_predictions
            and not minimal_artifacts
            and shard_path.exists()
            and shard_path.stat().st_size > 0
        ):
            scored = pd.read_parquet(shard_path)
            selector_cols = {
                "base_score": "score_base",
                "meta_base_soft_label": "score_meta_base_soft_label",
                "meta_clean_exec": "score_meta_clean_exec",
                "meta_positive_margin": "score_meta_positive_margin",
                "meta_exec_margin": "score_meta_exec_margin",
                "meta_clean_minus_risk": "score_meta_clean_minus_risk",
                "meta_exec_margin_risk_blend": "score_meta_exec_margin_risk_blend",
                "meta_context_hint_blend": "score_meta_context_hint_blend",
                "meta_long_aware_clean_minus_risk": "score_meta_long_aware_clean_minus_risk",
                "meta_path_order": "score_meta_path_order",
                "meta_path_order_clean_minus_risk": "score_meta_path_order_clean_minus_risk",
            }
            selector_cols = {
                name: col
                for name, col in selector_cols.items()
                if col in scored.columns
            }
            for selector, score_col in selector_cols.items():
                selector_row = _selector_metrics(scored, score_col, selector, test_fold)
                selector_row.update(
                    {
                        "calendar_month": str(test_month),
                        "valid_start": window["valid_start"],
                        "valid_end": window["valid_end"],
                        "max_oos_model_age_days": int(max_oos_model_age_days),
                    }
                )
                fold_rows.append(selector_row)
                for keep_frac in (0.30, 0.20, 0.10):
                    for row in _breakdown_rows(
                        scored, score_col, selector, test_fold, keep_frac
                    ):
                        row.update(
                            {
                                "calendar_month": str(test_month),
                                "valid_start": window["valid_start"],
                                "valid_end": window["valid_end"],
                                "max_oos_model_age_days": int(max_oos_model_age_days),
                            }
                        )
                        breakdown.append(row)
                if selector in {
                    "base_score",
                    "meta_long_aware_clean_minus_risk",
                    "meta_path_order_clean_minus_risk",
                    "meta_exec_margin_risk_blend",
                }:
                    base_conditioned_diagnostics.extend(
                        {
                            **row,
                            "calendar_month": str(test_month),
                            "valid_start": window["valid_start"],
                            "valid_end": window["valid_end"],
                            "max_oos_model_age_days": int(max_oos_model_age_days),
                        }
                        for row in _base_conditioned_diagnostic_rows(
                            scored,
                            test_month=test_fold,
                            selector=selector,
                            score_col=score_col,
                            keep_frac=0.30,
                        )
                    )
            for row in _threshold_policy_rows(scored, selector_cols, test_fold):
                row.update(
                    {
                        "calendar_month": str(test_month),
                        "valid_start": window["valid_start"],
                        "valid_end": window["valid_end"],
                        "max_oos_model_age_days": int(max_oos_model_age_days),
                    }
                )
                threshold_policy_rows.append(row)
            prediction_shard_paths.append(shard_path)
            print(
                json.dumps(
                    {
                        "event": "s52_train_meta_prediction_shard_resume_hit",
                        "fold": test_fold,
                        "path": str(shard_path),
                        "rows": int(len(scored)),
                        "selectors": sorted(selector_cols),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del scored
            gc.collect()
            continue
        generated_feature_names: list[str] = []
        fold_feature_meta: dict[str, Any] = {}
        train_mask, label_purge = _resolved_meta_train_mask(
            data,
            valid_start=window["valid_start"],
            strict=bool(strict_handoff_contract),
        )
        label_purge_by_fold[test_fold] = label_purge
        train = data.loc[train_mask]
        valid = data.loc[
            ts_utc.ge(window["valid_start"]) & ts_utc.lt(window["valid_end"])
        ]
        if len(train) < 100 or len(valid) < 30:
            continue
        print(
            json.dumps(
                _json_safe({
                    "event": "s52_train_meta_fold_start",
                    "frontier": frontier,
                    "test_fold": test_fold,
                    "test_month": test_month,
                    "valid_start": str(window["valid_start"]),
                    "valid_end": str(window["valid_end"]),
                    "max_oos_model_age_days": int(max_oos_model_age_days),
                    "train_rows": int(len(train)),
                    "valid_rows": int(len(valid)),
                    "label_path_purged_rows": int(label_purge["purged_rows"]),
                    "max_retained_label_end": label_purge.get(
                        "max_retained_label_end"
                    ),
                }),
                sort_keys=True,
            ),
            flush=True,
        )
        fold_matrix_cache_key: tuple[Any, ...] | None = None
        cached_fold_matrix = None
        fold_preprocessing_state: dict[str, Any] = {}
        if (
            int(model_train_max_rows) > 0
            and str(validation_scope).strip().lower() == "largest"
        ):
            fold_matrix_cache_key = (
                "hpo_largest_fold_matrix",
                fs_cache_key,
                str(test_fold),
                str(window["valid_start"]),
                str(window["valid_end"]),
                int(model_train_max_rows),
                tuple(numeric_cols),
                tuple(categorical_cols),
            )
            cached_fold_matrix = _HPO_FOLD_MATRIX_CACHE.get(fold_matrix_cache_key)
        if cached_fold_matrix is not None:
            (
                train_matrix,
                valid,
                x_train,
                x_valid,
                feature_names,
                fold_preprocessing_state,
                matrix_numeric_cols,
                matrix_categorical_cols,
            ) = cached_fold_matrix
            print(
                json.dumps(
                    {
                        "event": "s52_train_meta_hpo_fold_matrix_cache_hit",
                        "fold": str(test_fold),
                        "train_rows": int(len(train_matrix)),
                        "valid_rows": int(len(valid)),
                        "features": int(len(feature_names)),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        else:
            train_feature_source = train
            if int(model_train_max_rows) > 0:
                train_feature_idx = _time_spread_cap_rows(
                    len(train), int(model_train_max_rows)
                )
                train_feature_source = train.iloc[train_feature_idx].reset_index(
                    drop=True
                )
            train, valid = _add_fold_base_prior_features(
                train_feature_source, valid, selected_col=selected_col
            )
            if enable_reliability_features:
                train, valid = _add_fold_reliability_features(train, valid)
            if enable_support_drift_features:
                train, valid = _add_fold_support_drift_features(train, valid)
            if enable_hit_surprise_features:
                train, valid = _add_fold_hit_surprise_features(train, valid)
            if fold_feature_builder is not None:
                built = fold_feature_builder(
                    train=train,
                    valid=valid,
                    fold=str(test_fold),
                    month=str(test_month),
                    valid_start=window["valid_start"],
                    valid_end=window["valid_end"],
                    selected_col=selected_col,
                )
                if isinstance(built, tuple) and len(built) == 4:
                    train, valid, generated_feature_names, fold_feature_meta = built
                elif isinstance(built, tuple) and len(built) == 3:
                    train, valid, generated_feature_names = built
                    fold_feature_meta = {}
                else:
                    raise TypeError(
                        "fold_feature_builder must return (train, valid, feature_names[, metadata])"
                    )
                generated_feature_names = [
                    str(c) for c in generated_feature_names if str(c).strip()
                ]
                fold_feature_meta = dict(fold_feature_meta or {})
            # HPO/feature-selection trials only need the capped, time-spread training
            # sample. Expensive fold-derived features are also fit on that sample
            # above; final full-OOS replay passes model_train_max_rows=0 and uses
            # all prior rows.
            train_matrix = train
            if global_feature_names is None:
                # Fold-local base-score priors are generated above. Recompute
                # the source universe after that step so MDA can evaluate them
                # rather than silently omitting them from the selection matrix.
                matrix_numeric_cols, matrix_categorical_cols = _feature_columns(
                    train_matrix,
                    enable_base_prior_features=enable_base_prior_features,
                    enable_reliability_features=enable_reliability_features,
                    enable_support_drift_features=enable_support_drift_features,
                    enable_hit_surprise_features=enable_hit_surprise_features,
                )
            else:
                matrix_numeric_cols, matrix_categorical_cols = (
                    _feature_source_columns_for_selected(
                        selected_features=global_feature_names,
                        numeric_cols=numeric_cols,
                        categorical_cols=categorical_cols,
                    )
                )
            if generated_feature_names:
                generated_numeric = [
                    col
                    for col in generated_feature_names
                    if col in train_matrix.columns
                    and pd.api.types.is_numeric_dtype(train_matrix[col])
                ]
                generated_categorical = [
                    col
                    for col in generated_feature_names
                    if col in train_matrix.columns and col not in set(generated_numeric)
                ]
                matrix_numeric_cols = sorted(
                    set(matrix_numeric_cols).union(generated_numeric)
                )
                matrix_categorical_cols = sorted(
                    set(matrix_categorical_cols).union(generated_categorical)
                )
            x_train, x_valid, feature_names = _make_xy(
                train_matrix,
                valid,
                numeric_cols=matrix_numeric_cols,
                categorical_cols=matrix_categorical_cols,
                selected_features=global_feature_names,
                generated_features=(
                    *ALL_META_POST_SELECTION_OOD_FEATURE_NAMES,
                    *generated_feature_names,
                ),
                legacy_constant_zero_features=legacy_constant_zero_features,
                preprocessing_state_out=fold_preprocessing_state,
            )
            if fold_matrix_cache_key is not None:
                _HPO_FOLD_MATRIX_CACHE.clear()
                _HPO_FOLD_MATRIX_CACHE[fold_matrix_cache_key] = (
                    train_matrix.copy(deep=False),
                    valid.copy(deep=False),
                    x_train.copy(deep=False),
                    x_valid.copy(deep=False),
                    list(feature_names),
                    dict(fold_preprocessing_state),
                    list(matrix_numeric_cols),
                    list(matrix_categorical_cols),
                )
        if fold_feature_meta:
            fold_feature_metadata.append(
                {
                    "fold": str(test_fold),
                    "calendar_month": str(test_month),
                    **_json_safe(fold_feature_meta),
                }
            )
        if has_frozen_side_feature_contract and global_feature_names is None:
            raise RuntimeError(
                "Frozen side-local meta feature contract attempted to re-enter feature selection."
            )
        if global_feature_names is None:
            fs_fold_name = f"largest_train_before_{window['valid_start']:%Y-%m-%d}"
            method = str(feature_selection_method).strip().lower()
            if method not in {"auto", "lgbm_pipeline", "lgbm_staged"}:
                raise ValueError(
                    "--feature-selection-method now routes through lgbm_pipeline.py only. "
                    "Use auto/lgbm_pipeline/lgbm_staged."
                )
            fs_idx = _time_spread_cap_rows(
                len(x_train), int(feature_selection_max_rows)
            )
            (
                _x_train_selected,
                _x_valid_selected,
                feature_names,
                global_feature_names_by_side,
                feature_selection_df,
            ) = (
                _select_features_by_lgbm_pipeline(
                    x_train.iloc[fs_idx].reset_index(drop=True),
                    x_valid,
                    train_matrix.iloc[fs_idx].reset_index(drop=True),
                    target_name=str(feature_selection_target),
                    top_n=int(feature_selection_top_n),
                    fold=fs_fold_name,
                    seed=int(seed) + fold_idx,
                    target_contract=inherited_target_contract,
                    target_strength_weight_spec=inherited_weight_spec,
                    base_weight_arm=inherited_base_weight_arm,
                    strict_handoff_contract=bool(strict_handoff_contract),
                )
            )
            # The selector may derive and retain a subset of OOD metrics from
            # the sampled core. Rebuild those statistics on every prior row
            # before fitting the actual growing-window side models.
            requested_ood = [
                c
                for c in feature_names
                if c in ALL_META_POST_SELECTION_OOD_FEATURE_NAMES
            ]
            core_feature_names = [
                c for c in feature_names
                if c not in ALL_META_POST_SELECTION_OOD_FEATURE_NAMES
            ]
            if requested_ood:
                x_train, x_valid, _ = _append_post_selection_ood_features(
                    x_train.reindex(columns=core_feature_names, fill_value=0.0),
                    x_valid.reindex(columns=core_feature_names, fill_value=0.0),
                    core_feature_names,
                    output_features=requested_ood,
                )
            else:
                x_train = x_train.reindex(columns=core_feature_names, fill_value=0.0)
                x_valid = x_valid.reindex(columns=core_feature_names, fill_value=0.0)
            global_feature_names = list(feature_names)
            global_feature_names = _with_meta_protected_base_features(
                global_feature_names,
                available=x_train.columns,
            )
            global_feature_names_by_side = {
                side: _with_meta_protected_base_features(
                    features,
                    available=x_train.columns,
                )
                for side, features in (global_feature_names_by_side or {}).items()
            }
            global_feature_selection_df = feature_selection_df.copy()
            _FEATURE_SELECTION_CACHE[fs_cache_key] = (
                list(global_feature_names),
                global_feature_selection_df.copy(),
                {
                    side: list(features)
                    for side, features in (global_feature_names_by_side or {}).items()
                },
            )
            feature_selection_frames.append(feature_selection_df)
            feature_selection_recorded = True
        else:
            requested_ood = [
                c
                for c in global_feature_names
                if c in ALL_META_POST_SELECTION_OOD_FEATURE_NAMES
            ]
            if requested_ood:
                # An ablation may append new regime/state columns. Keep OOD
                # tied to the frozen baseline feature set so the comparison is
                # not confounded by a newly redefined drift detector.
                ood_source = (
                    list(ood_reference_features)
                    if ood_reference_features is not None
                    else list(global_feature_names)
                )
                ood_base_features = [
                    c
                    for c in ood_source
                    if c not in ALL_META_POST_SELECTION_OOD_FEATURE_NAMES
                ]
                x_train_ood = x_train.reindex(columns=ood_base_features, fill_value=0.0)
                x_valid_ood = x_valid.reindex(columns=ood_base_features, fill_value=0.0)
                x_train_ood, x_valid_ood, _ = _append_post_selection_ood_features(
                    x_train_ood,
                    x_valid_ood,
                    ood_base_features,
                    output_features=requested_ood,
                )
                x_train = x_train_ood
                x_valid = x_valid_ood
            global_feature_names = _with_meta_protected_base_features(
                global_feature_names,
                available=x_train.columns,
            )
            global_feature_names_by_side = {
                side: _with_meta_protected_base_features(
                    features,
                    available=x_train.columns,
                )
                for side, features in (global_feature_names_by_side or {}).items()
            }
            x_train = x_train.reindex(columns=global_feature_names, fill_value=0.0)
            x_valid = x_valid.reindex(columns=global_feature_names, fill_value=0.0)
            feature_names = list(global_feature_names)
            if (
                not feature_selection_recorded
                and global_feature_selection_df is not None
            ):
                feature_selection_frames.append(global_feature_selection_df.copy())
                feature_selection_recorded = True
        if global_feature_names_by_side is None:
            global_feature_names_by_side = {
                side: list(feature_names) for side in ("long", "short")
            }
        input_feature_contract = resolve_meta_input_contract(
            feature_names,
            materialized_columns=train_matrix.columns,
            categorical_columns=matrix_categorical_cols,
            generated_features=(
                *ALL_META_POST_SELECTION_OOD_FEATURE_NAMES,
                *generated_feature_names,
            ),
            legacy_constant_zero_features=legacy_constant_zero_features,
        )
        require_resolved_meta_input_contract(
            input_feature_contract, role=f"meta fold {test_fold}"
        )
        fold_preprocessing_state["encoded_feature_names"] = list(feature_names)
        fold_preprocessing_state["encoded_feature_contract_hash"] = (
            _feature_contract_hash(list(feature_names))
        )
        x_train = materialize_legacy_constant_zeros(x_train, input_feature_contract)
        x_valid = materialize_legacy_constant_zeros(x_valid, input_feature_contract)
        x_train = require_encoded_meta_matrix(
            x_train, feature_names=feature_names, role=f"meta train fold {test_fold}"
        )
        x_valid = require_encoded_meta_matrix(
            x_valid, feature_names=feature_names, role=f"meta OOS fold {test_fold}"
        )
        input_feature_contracts_by_fold[str(test_fold)] = input_feature_contract
        feature_names_by_side = {
            side: [c for c in global_feature_names_by_side.get(side, []) if c in x_train.columns]
            for side in ("long", "short")
        }
        for side, names in feature_names_by_side.items():
            if not names:
                raise RuntimeError(
                    f"Meta {side} feature contract is empty after matrix construction."
                )
        selected_features_by_fold[str(test_fold)] = list(feature_names)
        selected_features_by_side_by_fold[str(test_fold)] = {
            side: list(names) for side, names in feature_names_by_side.items()
        }
        fit_idx = _time_spread_cap_rows(len(x_train), int(model_train_max_rows))
        x_train_fit = x_train.iloc[fit_idx].reset_index(drop=True)
        train_fit = train_matrix.iloc[fit_idx].reset_index(drop=True)
        scored = valid.copy()
        scored["score_base"] = _num(scored.get("score"), index=scored.index)
        if meta_head_mode == "single_base_soft_label":
            base_soft_target, base_soft_target_col = _base_soft_label_target(
                train_fit, target_contract=inherited_target_contract
            )
            meta_target_columns_used.add(str(base_soft_target_col))
            models: dict[str, Any] = {}
            model_feature_names: dict[str, list[str]] = {}
            if side_specific_single_head:
                scored["score_meta_base_soft_label"] = np.nan
                train_side = train_fit.get(
                    "side_name", pd.Series("long", index=train_fit.index)
                ).astype(str).str.lower()
                valid_side = scored.get(
                    "side_name", pd.Series("long", index=scored.index)
                ).astype(str).str.lower()
                for side_idx, side in enumerate(("long", "short"), start=1):
                    side_features = list(feature_names_by_side[side])
                    train_mask = train_side.eq(side)
                    valid_mask = valid_side.eq(side)
                    if int(train_mask.sum()) < 50:
                        raise RuntimeError(
                            f"Meta {side} model has only {int(train_mask.sum())} training rows."
                        )
                    model_label = f"base_soft_label_{side}"
                    weight_diagnostics: dict[str, Any] = {}
                    model = _fit_base_soft_label_model(
                        x_train_fit.loc[train_mask, side_features],
                        base_soft_target.loc[train_mask],
                        train_fit.loc[train_mask],
                        seed + fold_idx + side_idx * 10_000,
                        lgbm_params=classifier_params,
                        target_strength_weight_spec=inherited_weight_spec,
                        base_weight_arm=inherited_base_weight_arm,
                        strict_handoff_contract=bool(strict_handoff_contract),
                        weight_diagnostics_out=weight_diagnostics,
                    )
                    target_strength_weight_diagnostics_by_fold[
                        f"{test_fold}:{side}"
                    ] = weight_diagnostics
                    models[model_label] = model
                    model_feature_names[model_label] = side_features
                    if bool(valid_mask.any()):
                        scored.loc[valid_mask, "score_meta_base_soft_label"] = _predict(
                            model,
                            x_valid.loc[valid_mask, side_features],
                            classifier=False,
                        ).to_numpy(dtype=np.float32)
            else:
                weight_diagnostics = {}
                model = _fit_base_soft_label_model(
                    x_train_fit,
                    base_soft_target,
                    train_fit,
                    seed + fold_idx,
                    lgbm_params=classifier_params,
                    target_strength_weight_spec=inherited_weight_spec,
                    base_weight_arm=inherited_base_weight_arm,
                    strict_handoff_contract=bool(strict_handoff_contract),
                    weight_diagnostics_out=weight_diagnostics,
                )
                target_strength_weight_diagnostics_by_fold[str(test_fold)] = (
                    weight_diagnostics
                )
                models["base_soft_label"] = model
                model_feature_names["base_soft_label"] = list(feature_names)
                scored["score_meta_base_soft_label"] = _predict(
                    model, x_valid, classifier=False
                )
            if single_head_score_override is not None:
                override = single_head_score_override(
                    x_train=x_train_fit,
                    train=train_fit,
                    x_valid=x_valid,
                    scored=scored,
                    base_target=base_soft_target,
                    feature_names_by_side=feature_names_by_side,
                    classifier_params=classifier_params,
                    fold=str(test_fold),
                    seed=int(seed + fold_idx),
                )
                if not isinstance(override, tuple) or len(override) != 4:
                    raise TypeError(
                        "single_head_score_override must return "
                        "(scored, models, model_feature_names, metadata)."
                    )
                scored, override_models, override_feature_names, override_meta = override
                models.update(dict(override_models or {}))
                model_feature_names.update(dict(override_feature_names or {}))
                if override_meta:
                    fold_feature_metadata.append(
                        {
                            "fold": str(test_fold),
                            "calendar_month": str(test_month),
                            "single_head_score_override": _json_safe(override_meta),
                        }
                    )
            scored["meta_base_soft_label_target_col"] = base_soft_target_col
            selector_cols = {
                "base_score": "score_base",
                "meta_base_soft_label": "score_meta_base_soft_label",
            }
        else:
            model_feature_names = {}
            models = {
                "clean_exec": _fit_classifier(
                    x_train_fit,
                    train_fit["clean_exec_label"],
                    train_fit,
                    seed + fold_idx,
                    lgbm_params=classifier_params,
                ),
                "positive_margin": _fit_classifier(
                    x_train_fit,
                    train_fit["positive_exec_margin"],
                    train_fit,
                    seed + 11 + fold_idx,
                    lgbm_params=classifier_params,
                ),
                "bad_path": _fit_classifier(
                    x_train_fit,
                    train_fit["bad_path_label"],
                    train_fit,
                    seed + 23 + fold_idx,
                    lgbm_params=classifier_params,
                ),
                "timeout": _fit_classifier(
                    x_train_fit,
                    train_fit["timeout"],
                    train_fit,
                    seed + 31 + fold_idx,
                    lgbm_params=classifier_params,
                ),
                "exec_margin": _fit_regressor(
                    x_train_fit,
                    train_fit["exec_margin"],
                    train_fit,
                    seed + 43 + fold_idx,
                    lgbm_params=regressor_params,
                ),
                "long_clean_exec": _fit_side_classifier(
                    x_train_fit,
                    train_fit["long_path_clean_exec_label"],
                    train_fit,
                    side="long",
                    seed=seed + 53 + fold_idx,
                    lgbm_params=classifier_params,
                ),
                "long_bad_path": _fit_side_classifier(
                    x_train_fit,
                    train_fit["long_bad_path_label"],
                    train_fit,
                    side="long",
                    seed=seed + 61 + fold_idx,
                    lgbm_params=classifier_params,
                ),
            }
            if enable_path_order_heads:
                models.update(
                    {
                        "mfe_before_mae": _fit_classifier(
                            x_train_fit,
                            train_fit["path_mfe_before_mae_label"],
                            train_fit,
                            seed + 47 + fold_idx,
                            lgbm_params=classifier_params,
                        ),
                        "mae_before_mfe": _fit_classifier(
                            x_train_fit,
                            train_fit["path_mae_before_mfe_label"],
                            train_fit,
                            seed + 49 + fold_idx,
                            lgbm_params=classifier_params,
                        ),
                        "underwater_duration": _fit_regressor(
                            x_train_fit,
                            train_fit["path_underwater_duration_target"],
                            train_fit,
                            seed + 51 + fold_idx,
                            lgbm_params=regressor_params,
                        ),
                    }
                )
            scored["score_meta_clean_exec"] = _predict(
                models["clean_exec"], x_valid, classifier=True
            )
            scored["score_meta_positive_margin"] = _predict(
                models["positive_margin"], x_valid, classifier=True
            )
            scored["score_meta_bad_path"] = _predict(
                models["bad_path"], x_valid, classifier=True
            )
            scored["score_meta_timeout"] = _predict(
                models["timeout"], x_valid, classifier=True
            )
            scored["score_meta_exec_margin"] = _predict(
                models["exec_margin"], x_valid, classifier=False
            )
            if enable_path_order_heads:
                scored["score_meta_mfe_before_mae"] = _predict(
                    models["mfe_before_mae"], x_valid, classifier=True
                )
                scored["score_meta_mae_before_mfe"] = _predict(
                    models["mae_before_mfe"], x_valid, classifier=True
                )
                scored["score_meta_underwater_duration"] = _predict(
                    models["underwater_duration"], x_valid, classifier=False
                )
                underwater_penalty = (
                    _num(scored["score_meta_underwater_duration"], index=scored.index)
                    .fillna(0.0)
                    .clip(lower=0.0)
                )
                scored["score_meta_path_order"] = (
                    scored["score_meta_mfe_before_mae"].fillna(0.0)
                    - 0.85 * scored["score_meta_mae_before_mfe"].fillna(1.0)
                    - 0.08 * underwater_penalty
                )
            else:
                underwater_penalty = pd.Series(
                    0.0, index=scored.index, dtype=np.float32
                )
            long_valid = scored["side_name"].astype(str).str.lower().eq("long")
            scored["score_meta_long_clean_exec"] = _predict(
                models["long_clean_exec"], x_valid, classifier=True
            )
            scored.loc[~long_valid, "score_meta_long_clean_exec"] = np.nan
            scored["score_meta_long_bad_path"] = _predict(
                models["long_bad_path"], x_valid, classifier=True
            )
            scored.loc[~long_valid, "score_meta_long_bad_path"] = np.nan
            scored["score_meta_clean_minus_risk"] = (
                scored["score_meta_clean_exec"].fillna(0.0)
                + 0.60 * scored["score_meta_positive_margin"].fillna(0.0)
                - 0.70 * scored["score_meta_bad_path"].fillna(0.0)
                - 0.30 * scored["score_meta_timeout"].fillna(0.0)
            )
            if enable_path_order_heads:
                scored["score_meta_path_order_clean_minus_risk"] = (
                    scored["score_meta_clean_minus_risk"].fillna(0.0)
                    + 0.35 * scored["score_meta_mfe_before_mae"].fillna(0.0)
                    - 0.35 * scored["score_meta_mae_before_mfe"].fillna(1.0)
                    - 0.04 * underwater_penalty
                )
            scored["score_meta_exec_margin_risk_blend"] = (
                scored["score_meta_exec_margin"].fillna(0.0)
                + 0.0030 * scored["score_meta_clean_exec"].fillna(0.0)
                + 0.0020 * scored["score_meta_positive_margin"].fillna(0.0)
                - 0.0040 * scored["score_meta_bad_path"].fillna(0.0)
                - 0.0020 * scored["score_meta_timeout"].fillna(0.0)
            )
            if enable_path_order_blends and enable_path_order_heads:
                scored["score_meta_exec_margin_risk_blend"] = (
                    scored["score_meta_exec_margin_risk_blend"].fillna(0.0)
                    + 0.0015 * scored["score_meta_mfe_before_mae"].fillna(0.0)
                    - 0.0015 * scored["score_meta_mae_before_mfe"].fillna(1.0)
                    - 0.0002 * underwater_penalty
                )
            scored["score_meta_context_hint_blend"] = (
                scored["score_meta_exec_margin_risk_blend"].fillna(0.0)
                + 0.0010
                * _num(
                    scored.get("meta_context_weight_hint"),
                    index=scored.index,
                    default=0.0,
                ).fillna(0.0)
                - 0.0010
                * _num(
                    scored.get("meta_threshold_adjustment_hint"),
                    index=scored.index,
                    default=0.0,
                ).fillna(0.0)
            )
            long_clean_score = scored["score_meta_long_clean_exec"].where(
                scored["score_meta_long_clean_exec"].notna(),
                scored["score_meta_clean_exec"],
            )
            long_bad_score = scored["score_meta_long_bad_path"].where(
                scored["score_meta_long_bad_path"].notna(),
                scored["score_meta_bad_path"],
            )
            scored["score_meta_long_aware_clean_minus_risk"] = scored[
                "score_meta_clean_minus_risk"
            ]
            long_aware = (
                long_clean_score.loc[long_valid].fillna(0.0)
                + 0.55
                * scored.loc[long_valid, "score_meta_positive_margin"].fillna(0.0)
                - 0.80 * long_bad_score.loc[long_valid].fillna(1.0)
                - 0.25 * scored.loc[long_valid, "score_meta_timeout"].fillna(1.0)
            )
            if enable_path_order_blends and enable_path_order_heads:
                long_aware = (
                    long_aware
                    + 0.25
                    * scored.loc[long_valid, "score_meta_mfe_before_mae"].fillna(0.0)
                    - 0.25
                    * scored.loc[long_valid, "score_meta_mae_before_mfe"].fillna(1.0)
                    - 0.03 * underwater_penalty.loc[long_valid]
                )
            scored.loc[long_valid, "score_meta_long_aware_clean_minus_risk"] = (
                long_aware
            )
            selector_cols = {
                "base_score": "score_base",
                "meta_clean_exec": "score_meta_clean_exec",
                "meta_positive_margin": "score_meta_positive_margin",
                "meta_exec_margin": "score_meta_exec_margin",
                "meta_clean_minus_risk": "score_meta_clean_minus_risk",
                "meta_exec_margin_risk_blend": "score_meta_exec_margin_risk_blend",
                "meta_context_hint_blend": "score_meta_context_hint_blend",
                "meta_long_aware_clean_minus_risk": "score_meta_long_aware_clean_minus_risk",
            }
            if enable_path_order_heads:
                selector_cols["meta_path_order"] = "score_meta_path_order"
                selector_cols["meta_path_order_clean_minus_risk"] = (
                    "score_meta_path_order_clean_minus_risk"
                )
        scored["oos_fold"] = str(test_fold)
        scored["calendar_month"] = str(test_month)
        scored["valid_start"] = window["valid_start"]
        scored["valid_end"] = window["valid_end"]
        scored["max_oos_model_age_days"] = int(max_oos_model_age_days)
        for selector, score_col in selector_cols.items():
            selector_row = _selector_metrics(scored, score_col, selector, test_fold)
            selector_row.update(
                {
                    "calendar_month": str(test_month),
                    "valid_start": window["valid_start"],
                    "valid_end": window["valid_end"],
                    "max_oos_model_age_days": int(max_oos_model_age_days),
                }
            )
            fold_rows.append(selector_row)
            if not minimal_artifacts:
                for keep_frac in (0.30, 0.20, 0.10):
                    for row in _breakdown_rows(
                        scored, score_col, selector, test_fold, keep_frac
                    ):
                        row.update(
                            {
                                "calendar_month": str(test_month),
                                "valid_start": window["valid_start"],
                                "valid_end": window["valid_end"],
                                "max_oos_model_age_days": int(max_oos_model_age_days),
                            }
                        )
                        breakdown.append(row)
                if selector in {
                    "base_score",
                    "meta_long_aware_clean_minus_risk",
                    "meta_path_order_clean_minus_risk",
                    "meta_exec_margin_risk_blend",
                }:
                    base_conditioned_diagnostics.extend(
                        {
                            **row,
                            "calendar_month": str(test_month),
                            "valid_start": window["valid_start"],
                            "valid_end": window["valid_end"],
                            "max_oos_model_age_days": int(max_oos_model_age_days),
                        }
                        for row in _base_conditioned_diagnostic_rows(
                            scored,
                            test_month=test_fold,
                            selector=selector,
                            score_col=score_col,
                            keep_frac=0.30,
                        )
                    )
        if not minimal_artifacts:
            for row in _threshold_policy_rows(scored, selector_cols, test_fold):
                row.update(
                    {
                        "calendar_month": str(test_month),
                        "valid_start": window["valid_start"],
                        "valid_end": window["valid_end"],
                        "max_oos_model_age_days": int(max_oos_model_age_days),
                    }
                )
                threshold_policy_rows.append(row)
        if not minimal_artifacts:
            for label, model in models.items():
                importances.append(
                    _feature_importance(
                        model,
                        model_feature_names.get(label, feature_names),
                        label,
                        test_fold,
                    )
                )
            keep_cols = (
                [
                    "__ts__",
                    "__symbol__",
                    "side_name",
                    "month",
                    "oos_fold",
                    "calendar_month",
                    "valid_start",
                    "valid_end",
                    "max_oos_model_age_days",
                    "source_semantic_family",
                    "source_semantic_family_base",
                    "long_source_regime_split",
                    "aegmm_cluster",
                    "side_aegmm_cluster",
                    "aegmm_expected_distance_bin",
                    "reconstruction_bin",
                    "exec_margin",
                    "ev_after_1pct",
                    "ret_net",
                    "u_policy_net",
                    "first_touch_gross",
                    "first_touch_bad_mae_1r",
                    "full_path_bad_mae_1r",
                    "timeout",
                    "mfe_before_mae_1r",
                    "mae_before_mfe_1r",
                    "clean_exec",
                    "dirty_positive",
                    "underwater_bars_before_mfe_1r",
                    "long_path_clean_exec_label",
                    "long_path_dirty_positive_label",
                    "long_path_post_mfe_drawdown_norm",
                    "long_path_time_to_profit_bars",
                    "long_path_slow_profit",
                    "long_path_post_mfe_bad_drawdown",
                    "long_bad_path_label",
                    "__archetype_label_family__",
                    "__archetype_label_source__",
                    "__archetype_policy_key__",
                    "__archetype_policy_role__",
                    "__archetype_policy_confidence__",
                    "__archetype_policy_tp_r__",
                    "__archetype_policy_sl_r__",
                    "__archetype_policy_trail_r__",
                    "__archetype_policy_max_bars_to_mfe__",
                    "__archetype_policy_max_barrier__",
                    *LEDGER_CONTEXT_FEATURE_ALIASES.values(),
                    "score_meta_bad_path",
                    "score_meta_timeout",
                    "score_meta_mfe_before_mae",
                    "score_meta_mae_before_mfe",
                    "score_meta_underwater_duration",
                    "score_meta_path_order",
                    "score_meta_path_order_clean_minus_risk",
                    "score_meta_long_clean_exec",
                    "score_meta_long_bad_path",
                    "base_margin_to_cutoff",
                    "base_margin_to_cutoff_z",
                    "base_signal_zscore_within_archetype",
                    "base_score_rank_pct_train_prior",
                    "base_rank_band",
                    "base_margin_band",
                ]
                + list(RELIABILITY_NUMERIC_FEATURES)
                + list(SUPPORT_DRIFT_NUMERIC_FEATURES)
                + list(HIT_SURPRISE_NUMERIC_FEATURES)
                + list(generated_feature_names)
                + list(extra_prediction_columns or [])
                + list(selector_cols.values())
            )
            export_cols = [
                col for col in dict.fromkeys(keep_cols) if col in scored.columns
            ]
            fold_prediction = scored[export_cols].copy()
            if should_shard_predictions:
                prediction_shard_dir.mkdir(parents=True, exist_ok=True)
                fold_prediction.to_parquet(shard_path, index=False)
                prediction_shard_paths.append(shard_path)
            else:
                prediction_frames.append(fold_prediction)
            if bool(save_fold_models):
                saved_model_manifests.append(
                    _save_meta_fold_models(
                        out_dir=out_dir,
                        fold=str(test_fold),
                        calendar_month=str(test_month),
                        valid_start=window["valid_start"],
                        valid_end=window["valid_end"],
                        fold_idx=int(fold_idx),
                        seed=int(seed),
                        models=models,
                        feature_names=list(feature_names),
                        model_feature_names=model_feature_names,
                        input_feature_contract=input_feature_contract,
                        classifier_params=classifier_params,
                        regressor_params=regressor_params,
                        meta_head_mode=str(meta_head_mode),
                        model_profile_name=str(model_profile_name),
                        train_rows_available=int(len(train_matrix)),
                        train_rows_fit=int(len(x_train_fit)),
                        valid_rows=int(len(valid)),
                        target_columns_used=set(meta_target_columns_used),
                        preprocessing_state=fold_preprocessing_state,
                        inherited_handoff_contract=inherited_handoff_contract,
                        target_strength_weight_diagnostics={
                            key: value
                            for key, value in target_strength_weight_diagnostics_by_fold.items()
                            if str(key).startswith(f"{test_fold}:")
                            or str(key) == str(test_fold)
                        },
                        label_purge_contract=label_purge,
                    )
                )
        del (
            train,
            valid,
            train_matrix,
            x_train,
            x_valid,
            x_train_fit,
            train_fit,
            scored,
            models,
        )
        generated_feature_names = []
        fold_feature_meta = {}
        gc.collect()
    folds = pd.DataFrame(fold_rows)
    summary = _summarize(folds)
    threshold_policies = pd.DataFrame(threshold_policy_rows)
    threshold_summary = _summarize_threshold_policies(threshold_policies)
    breakdown_df = pd.DataFrame(breakdown)
    base_conditioned_df = pd.DataFrame(base_conditioned_diagnostics)
    importance_df = (
        pd.concat([part for part in importances if not part.empty], ignore_index=True)
        if any(not part.empty for part in importances)
        else pd.DataFrame(columns=["test_month", "model", "feature", "importance"])
    )
    feature_selection_all = (
        pd.concat(feature_selection_frames, ignore_index=True)
        if feature_selection_frames
        else pd.DataFrame(
            columns=[
                "fold",
                "feature",
                "score",
                "rank",
                "selected",
                "feature_selection_target",
                "feature_selection_status",
            ]
        )
    )
    if prediction_shard_paths and bool(combine_prediction_shards):
        prediction_frames.extend(
            pd.read_parquet(path) for path in prediction_shard_paths
        )
    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    folds.to_csv(out_dir / "s52_train_meta_regime_handoff_smoke_folds.csv", index=False)
    summary.to_csv(
        out_dir / "s52_train_meta_regime_handoff_smoke_summary.csv", index=False
    )
    threshold_policies.to_csv(
        out_dir / "s52_train_meta_regime_handoff_threshold_policy_folds.csv",
        index=False,
    )
    threshold_summary.to_csv(
        out_dir / "s52_train_meta_regime_handoff_threshold_policy_summary.csv",
        index=False,
    )
    feature_selection_all.to_csv(
        out_dir / "s52_train_meta_feature_selection_by_fold.csv", index=False
    )
    if minimal_artifacts:
        meta_oos_breakdown = {
            "enabled": False,
            "reason": "minimal_hpo_trial",
            "metrics_source": "skipped_for_hpo_speed",
        }
    else:
        breakdown_df.to_csv(
            out_dir / "s52_train_meta_regime_handoff_smoke_breakdown.csv", index=False
        )
        base_conditioned_df.to_csv(
            out_dir / "s52_train_meta_base_conditioned_diagnostics.csv", index=False
        )
        importance_df.to_csv(
            out_dir / "s52_train_meta_regime_handoff_smoke_feature_importance.csv",
            index=False,
        )
        if not predictions.empty:
            predictions.to_parquet(
                out_dir / "s52_train_meta_regime_handoff_smoke_predictions.parquet",
                index=False,
            )
        meta_oos_breakdown = _write_meta_oos_breakdown_reports(
            out_dir=out_dir,
            predictions=predictions,
            summary=summary,
        )
    best = summary.iloc[0].to_dict() if not summary.empty else {}
    best_threshold = (
        threshold_summary.iloc[0].to_dict() if not threshold_summary.empty else {}
    )
    # ``selected_features_by_fold`` stores the union required to build a
    # common matrix for each fold.  It cannot describe long/short overlap: a
    # frozen side-specific contract makes every fold union identical.  Publish
    # the actual side-contract intersection and retain the old fold-stability
    # diagnostic under an explicit name.
    selected_features_by_side = (
        next(iter(selected_features_by_side_by_fold.values()))
        if selected_features_by_side_by_fold
        else {}
    )
    selected_fold_sets = [set(v) for v in selected_features_by_fold.values()]
    selected_side_sets = [
        set(features) for features in selected_features_by_side.values()
    ]
    selected_union = (
        sorted(set().union(*selected_fold_sets)) if selected_fold_sets else []
    )
    selected_intersection = (
        sorted(set.intersection(*selected_side_sets)) if selected_side_sets else []
    )
    selected_fold_intersection = (
        sorted(set.intersection(*selected_fold_sets)) if selected_fold_sets else []
    )
    manifest = {
        "generated_by": "run_s52_train_meta_regime_handoff_smoke",
        "output_dir": str(out_dir),
        "handoff_dir": str(handoff_dir),
        "handoff_path": str(handoff_path),
        "handoff_input_columns": (
            sorted(str(col) for col in handoff_columns)
            if handoff_columns is not None
            else "all"
        ),
        "ledger_path": str(ledger_path),
        "materialized_soft_label": soft_label_manifest,
        "inherited_base_handoff_contract": _json_safe(inherited_handoff_contract),
        "target_strength_weight_diagnostics_by_fold": _json_safe(
            target_strength_weight_diagnostics_by_fold
        ),
        "target_strength_weighting": {
            **_serialize_inherited_weighting(
                inherited_weight_spec,
                inherited_base_weight_arm,
            ),
            "train_rows": "recomputed per OOS fold from train-only top30 rows",
            "target_column": str(
                inherited_target_contract.get("target_column", "")
            )
            or "auto_detected_base_soft_label",
        },
        "ood_reference_features": (
            list(ood_reference_features)
            if ood_reference_features is not None
            else "selected_features"
        ),
        "frontier": str(frontier),
        "train_scope": str(train_scope),
        "active_sides": sorted(active_side_set) if active_side_set else "all",
        "strict_handoff_contract": bool(strict_handoff_contract),
        "forward_label_purge_by_fold": _json_safe(label_purge_by_fold),
        "forward_label_purge_contract": {
            "schema": "meta_forward_label_purge_v1",
            "resolution_column": "__label_path_end_ts__",
            "train_rule": "__ts__ < valid_start and __label_path_end_ts__ < valid_start",
            "strict_missing_resolution": "fail_closed",
        },
        "months": months,
        "rows": int(len(data)),
        "numeric_feature_count": int(len(numeric_cols)),
        "categorical_feature_count": int(len(categorical_cols)),
        "feature_selection_scope": (
            "single_calibration_window_side_local_long_short"
            if meta_head_mode == "single_base_soft_label"
            and side_specific_single_head
            else "single_global_largest_train_window"
        ),
        "feature_selection_calibration_fold": calibration_fold,
        "calibration_min_valid_rows": 1000,
        "validation_scope": str(validation_scope),
        "eval_months": sorted(eval_month_set) if eval_month_set else None,
        "global_feature_selection_fold": (
            str(feature_selection_all["fold"].iloc[0])
            if not feature_selection_all.empty
            else None
        ),
        "feature_selection_method": str(feature_selection_method),
        "feature_selection_execution": (
            "frozen_side_contract_no_selector"
            if has_frozen_side_feature_contract
            else (
                "frozen_shared_contract_no_selector"
                if fixed_selected_features is not None
                else "selector_executed_once"
            )
        ),
        "feature_selection_top_n": int(feature_selection_top_n),
        "feature_selection_target": str(feature_selection_target),
        "protected_base_anchors": list(META_PROTECTED_BASE_FEATURES),
        "post_selection_ood_contract": "conditional_incremental_mda_only",
        "legacy_shared_feature_contract_rejected": bool(
            legacy_shared_feature_contract
        ),
        "meta_head_mode": str(meta_head_mode),
        "meta_target_columns_used": sorted(meta_target_columns_used),
        "single_head_model_contract": (
            (
                "independent long/short LGBMRegressor models on base target_soft with side-local MDA features and W7_timestamp_balanced weights"
                if side_specific_single_head
                else "shared-feature global LGBMRegressor on base target_soft with W7_timestamp_balanced weights"
            )
            if meta_head_mode == "single_base_soft_label"
            else "legacy_multi_head"
        ),
        "side_specific_feature_contract_enabled": bool(
            meta_head_mode == "single_base_soft_label" and side_specific_single_head
        ),
        "max_oos_model_age_days": int(max_oos_model_age_days),
        "single_fit_oos_window": bool(single_fit_oos_window),
        "model_train_max_rows": int(model_train_max_rows),
        "model_train_sampling": (
            "beginning_middle_end_time_spread"
            if int(model_train_max_rows) > 0
            else "full_train_rows"
        ),
        "feature_selection_max_rows": int(feature_selection_max_rows),
        "feature_selection_sampling": (
            "beginning_middle_end_time_spread"
            if int(feature_selection_max_rows) > 0
            else "full_train_rows"
        ),
        "feature_selection_forced_periods": [],
        "feature_selection_period_contract": (
            "beginning_middle_end_time_spread_only; no outcome-selected difficult dates"
        ),
        "minimal_artifacts": bool(minimal_artifacts),
        "force_prediction_shards": bool(force_prediction_shards),
        "combine_prediction_shards": bool(combine_prediction_shards),
        "save_fold_models": bool(save_fold_models),
        "saved_fold_models": _json_safe(saved_model_manifests),
        "fold_windows": [
            {
                "fold": row.get("test_month"),
                "calendar_month": row.get("calendar_month"),
                "valid_start": row.get("valid_start"),
                "valid_end": row.get("valid_end"),
                "max_oos_model_age_days": row.get("max_oos_model_age_days"),
            }
            for row in fold_rows
            if row.get("selector") == "base_score"
        ],
        "selected_features_by_fold": selected_features_by_fold,
        "input_feature_contracts_by_fold": input_feature_contracts_by_fold,
        "legacy_constant_zero_features": list(
            map(str, legacy_constant_zero_features)
        ),
        "selected_features_by_side_by_fold": selected_features_by_side_by_fold,
        "selected_features_by_side": selected_features_by_side,
        "selected_feature_union": selected_union,
        "selected_feature_intersection": selected_intersection,
        "selected_feature_union_count": int(len(selected_union)),
        "selected_feature_intersection_count": int(len(selected_intersection)),
        "selected_feature_fold_intersection": selected_fold_intersection,
        "selected_feature_fold_intersection_count": int(
            len(selected_fold_intersection)
        ),
        "feature_selection_output": str(
            out_dir / "s52_train_meta_feature_selection_by_fold.csv"
        ),
        "prediction_shards": [str(path) for path in prediction_shard_paths],
        "objective_step_8_meta_oos_breakdown": meta_oos_breakdown,
        "model_profile_name": str(model_profile_name),
        "fold_feature_profile_name": str(fold_feature_profile_name),
        "fold_feature_builder_enabled": bool(fold_feature_builder is not None),
        "fold_feature_metadata": fold_feature_metadata,
        "classifier_params": _json_safe(classifier_params),
        "regressor_params": _json_safe(regressor_params),
        "added_path_order_targets": [
            "path_mfe_before_mae_label",
            "path_mae_before_mfe_label",
            "path_underwater_duration_target",
        ],
        "added_base_conditioned_features": list(BASE_PRIOR_NUMERIC_FEATURES)
        + list(BASE_PRIOR_CATEGORICAL_FEATURES),
        "added_reliability_features": list(RELIABILITY_NUMERIC_FEATURES),
        "added_support_drift_features": list(SUPPORT_DRIFT_NUMERIC_FEATURES),
        "added_hit_surprise_features": list(HIT_SURPRISE_NUMERIC_FEATURES),
        "hit_surprise_half_life_days": list(HIT_SURPRISE_HALFLIFE_DAYS),
        "hit_surprise_window_cap": "4x_half_life",
        "hit_surprise_target": "clean_exec_label",
        "support_drift_source_columns": [
            col for col in SUPPORT_DRIFT_COLUMNS if col in data.columns
        ],
        "enable_base_prior_features": bool(enable_base_prior_features),
        "enable_reliability_features": bool(enable_reliability_features),
        "enable_support_drift_features": bool(enable_support_drift_features),
        "enable_hit_surprise_features": bool(enable_hit_surprise_features),
        "enable_path_order_heads": bool(enable_path_order_heads),
        "enable_path_order_blends": bool(enable_path_order_blends),
        "base_conditioned_diagnostics": str(
            out_dir / "s52_train_meta_base_conditioned_diagnostics.csv"
        ),
        "lightgbm_available": bool(_LIGHTGBM_AVAILABLE),
        "best_selector": _json_safe(best),
        "best_threshold_policy": _json_safe(best_threshold),
        "leakage_contract": {
            "feature_source": "train_meta_regime_handoff.parquet",
            "outcomes_joined_for": "training_labels_and_validation_metrics_only",
            "base_candidate_handoff": "timestamp_side top30 only in strict mode; base target/sample-weight hashes are uniform and excluded from model features",
            "sample_weights": "target-strength weights are recomputed from each fold's train-only candidate rows and never materialized as model features",
            "split": (
                "expanding_window_with_oos_age_cap"
                if int(max_oos_model_age_days) > 0
                else "month_forward_train_past_validate_next_month"
            ),
            "max_oos_model_age_days": int(max_oos_model_age_days),
            "primary_metrics": "top-k exec_margin/clean precision/bad-MAE/timeout",
            "threshold_policy_note": "fixed smoke templates evaluated OOS; not validation-optimized production thresholds",
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True)
    )
    if not minimal_artifacts:
        _write_markdown(out_dir, summary, folds, threshold_summary, manifest)
    return manifest


def _fmt_pct(value: Any) -> str:
    try:
        val = float(value)
    except Exception:
        return "nan"
    if not math.isfinite(val):
        return "nan"
    return f"{val * 100:.2f}%"


def _write_markdown(
    out_dir: Path,
    summary: pd.DataFrame,
    folds: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    lines = [
        "# S52 Train-Meta Regime Handoff Smoke",
        "",
        "## Scope",
        "",
        "Controlled month-forward diagnostic over the S52 row-level regime handoff artifact.",
        "This is not production train_meta and is not frozen replay evidence.",
        "",
        "## Contract",
        "",
        f"- frontier: `{manifest.get('frontier')}`",
        f"- train scope: `{manifest.get('train_scope')}`",
        f"- rows: `{manifest.get('rows')}`",
        f"- numeric features: `{manifest.get('numeric_feature_count')}`",
        f"- categorical features: `{manifest.get('categorical_feature_count')}`",
        f"- months: `{', '.join(manifest.get('months', []))}`",
        "",
        "## Best Summary",
        "",
    ]
    if summary.empty:
        lines.append("No valid folds were produced.")
    else:
        display_cols = [
            "selector",
            "meta_smoke_status",
            "mean_keep030_exec_margin",
            "mean_keep030_clean_exec_precision",
            "mean_keep030_full_path_bad_mae",
            "mean_keep030_timeout",
            "mean_keep020_exec_margin",
            "mean_keep020_ret_net",
            "mean_keep010_exec_margin",
            "mean_keep010_ret_net",
            "mean_keep005_exec_margin",
            "mean_keep005_ret_net",
            "worst_keep010_exec_margin",
            "mean_keep030_oracle_recall",
        ]
        head = summary[[col for col in display_cols if col in summary.columns]].head(8)
        lines.append(head.to_markdown(index=False))
        best = summary.iloc[0]
        lines.extend(
            [
                "",
                "## Read",
                "",
                f"- best selector: `{best.get('selector')}`",
                f"- keep30 executable margin: `{_fmt_pct(best.get('mean_keep030_exec_margin'))}`",
                f"- keep30 clean precision: `{_fmt_pct(best.get('mean_keep030_clean_exec_precision'))}`",
                f"- keep30 full-path bad-MAE: `{_fmt_pct(best.get('mean_keep030_full_path_bad_mae'))}`",
                f"- keep30 timeout: `{_fmt_pct(best.get('mean_keep030_timeout'))}`",
                f"- keep20 average net return: `{_fmt_pct(best.get('mean_keep020_ret_net'))}`",
                f"- keep10 average net return: `{_fmt_pct(best.get('mean_keep010_ret_net'))}`",
                f"- keep5 average net return: `{_fmt_pct(best.get('mean_keep005_ret_net'))}`",
                f"- status: `{best.get('meta_smoke_status')}`",
            ]
        )
    if not folds.empty:
        lines.extend(["", "## Fold Rows", "", folds.head(30).to_markdown(index=False)])
    if not threshold_summary.empty:
        display_cols = [
            "selector",
            "policy_id",
            "budget_frac",
            "threshold_policy_status",
            "mean_selected_rows",
            "mean_fill_rate",
            "mean_exec_margin",
            "worst_exec_margin",
            "mean_clean_exec_precision",
            "mean_full_path_bad_mae",
            "mean_timeout",
            "mean_oracle_recall",
        ]
        lines.extend(
            [
                "",
                "## Threshold / Abstention Sweep",
                "",
                "Fixed risk templates evaluated OOS. These are diagnostic threshold candidates, not production-optimized thresholds.",
                "",
                threshold_summary[
                    [col for col in display_cols if col in threshold_summary.columns]
                ]
                .head(15)
                .to_markdown(index=False),
            ]
        )
    (out_dir / "s52_train_meta_regime_handoff_smoke.md").write_text(
        "\n".join(lines) + "\n"
    )


def _preferred_meta_hpo_selectors(manifest: dict[str, Any]) -> list[str]:
    """Return selectors that match the requested meta target architecture.

    ``single_base_soft_label`` is a base-target replay mode.  The production
    meta path is the multi-head base-correctness stack, where the meta model
    estimates whether the base candidate is executable/clean and then combines
    the component heads.  HPO must therefore score the combined multi-head
    selectors, not the base-soft-label replay head.
    """

    mode = str(manifest.get("meta_head_mode", "")).strip().lower()
    if mode == "single_base_soft_label":
        return ["meta_base_soft_label"]
    return [
        "meta_long_aware_clean_minus_risk",
        "meta_path_order_clean_minus_risk",
        "meta_exec_margin_risk_blend",
        "meta_context_hint_blend",
        "meta_clean_minus_risk",
        "meta_exec_margin",
    ]


def _meta_hpo_selector_row(manifest: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(str(manifest.get("output_dir") or ""))
    summary_path = output_dir / "s52_train_meta_regime_handoff_smoke_summary.csv"
    preferred_selectors = _preferred_meta_hpo_selectors(manifest)
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        for selector in preferred_selectors:
            selected = summary.loc[summary["selector"].eq(selector)]
            if not selected.empty:
                return selected.iloc[0].to_dict()
        if not summary.empty:
            return summary.iloc[0].to_dict()
    best = dict(manifest.get("best_selector", {}) or {})
    if str(best.get("selector")) not in set(preferred_selectors):
        return {}
    return best


def _meta_trial_objective(manifest: dict[str, Any]) -> float:
    def f(value: Any, default: float = 0.0) -> float:
        try:
            val = float(value)
        except Exception:
            return float(default)
        return val if math.isfinite(val) else float(default)

    metrics = _meta_trial_metrics(manifest)
    top10_ev = f(metrics.get("top10_ev"))
    top20_ev = f(metrics.get("top20_ev"))
    worst_month = f(metrics.get("worst_month_top10_ev"))
    worst_week = f(metrics.get("worst_week_top10_ev"))
    top15_clean = f(metrics.get("top15_clean"))
    top15_dirty = f(metrics.get("top15_dirty"))
    top15_bad = f(metrics.get("top15_bad"))
    # HPO is EV-first. Path quality remains a weak tie-breaker because it is
    # handled downstream by meta/execution and previously dominated tiny EV
    # differences. Threshold templates are not part of model HPO.
    return float(
        100.0
        * (
            0.50 * top10_ev
            + 0.20 * top20_ev
            + 0.15 * worst_month
            + 0.15 * worst_week
        )
        + 0.05
        * ((top15_clean - top15_dirty) + (top15_clean - top15_bad))
    )


def _meta_trial_metrics(manifest: dict[str, Any]) -> dict[str, float]:
    """Return row-weighted top-k metrics with real month/week downside bars."""
    output_dir = Path(str(manifest.get("output_dir") or ""))
    folds_path = output_dir / "s52_train_meta_regime_handoff_smoke_folds.csv"
    if folds_path.exists():
        folds = pd.read_csv(folds_path)
        preferred_selectors = _preferred_meta_hpo_selectors(manifest)
        selected = pd.DataFrame()
        for selector in preferred_selectors:
            selected = folds.loc[folds["selector"].eq(selector)].copy()
            if not selected.empty:
                break
        folds = selected
        if not folds.empty:
            def weighted(column: str, weight_column: str) -> float:
                value = pd.to_numeric(folds[column], errors="coerce")
                weight = pd.to_numeric(folds[weight_column], errors="coerce")
                valid = value.notna() & weight.notna() & weight.gt(0)
                if not valid.any():
                    return float("nan")
                return float(np.average(value.loc[valid], weights=weight.loc[valid]))

            month_rows: list[float] = []
            for _, group in folds.groupby("calendar_month", sort=False):
                value = pd.to_numeric(group["keep010_ev_after_1pct"], errors="coerce")
                weight = pd.to_numeric(group["keep010_rows"], errors="coerce")
                valid = value.notna() & weight.notna() & weight.gt(0)
                if valid.any():
                    month_rows.append(
                        float(np.average(value.loc[valid], weights=weight.loc[valid]))
                    )
            first_bad = weighted(
                "keep015_first_touch_bad_mae", "keep015_rows"
            )
            full_bad = weighted("keep015_full_path_bad_mae", "keep015_rows")
            worst_week_values = pd.to_numeric(
                folds["keep010_worst_week_ev_after_1pct"], errors="coerce"
            )
            return {
                "top10_ev": weighted("keep010_ev_after_1pct", "keep010_rows"),
                "top20_ev": weighted("keep020_ev_after_1pct", "keep020_rows"),
                "worst_month_top10_ev": min(month_rows) if month_rows else float("nan"),
                "worst_week_top10_ev": float(worst_week_values.min()),
                "top15_clean": weighted(
                    "keep015_clean_exec_precision", "keep015_rows"
                ),
                "top15_dirty": weighted(
                    "keep015_dirty_positive_rate", "keep015_rows"
                ),
                "top15_bad": max(first_bad, full_bad),
            }

    best = _meta_hpo_selector_row(manifest)
    return {
        "top10_ev": best.get("mean_keep010_ev_after_1pct"),
        "top20_ev": best.get("mean_keep020_ev_after_1pct"),
        "worst_month_top10_ev": best.get("worst_keep010_ev_after_1pct"),
        "worst_week_top10_ev": best.get("worst_week_keep010_ev_after_1pct"),
        "top15_clean": best.get("mean_keep015_clean_exec_precision"),
        "top15_dirty": best.get("mean_keep015_dirty_positive_rate"),
        "top15_bad": max(
            float(best.get("mean_keep015_first_touch_bad_mae", 0.0) or 0.0),
            float(best.get("mean_keep015_full_path_bad_mae", 0.0) or 0.0),
        ),
    }


def _hpo_candidate_beats_incumbent(
    candidate_manifest: dict[str, Any],
    incumbent_manifest: dict[str, Any],
) -> tuple[bool, dict[str, float | bool]]:
    candidate = _meta_trial_metrics(candidate_manifest)
    incumbent = _meta_trial_metrics(incumbent_manifest)

    def f(row: dict[str, Any], key: str) -> float:
        try:
            value = float(row.get(key))
        except Exception:
            return float("nan")
        return value if math.isfinite(value) else float("nan")

    candidate_ev = f(candidate, "top10_ev")
    incumbent_ev = f(incumbent, "top10_ev")
    ev_gain = candidate_ev - incumbent_ev
    # A worst-period concession is acceptable only when it is at least five
    # times smaller than the average top-10 EV improvement.
    allowed_degradation = max(ev_gain / 5.0, 0.0)
    candidate_month = f(candidate, "worst_month_top10_ev")
    incumbent_month = f(incumbent, "worst_month_top10_ev")
    candidate_week = f(candidate, "worst_week_top10_ev")
    incumbent_week = f(incumbent, "worst_week_top10_ev")
    candidate_objective = _meta_trial_objective(candidate_manifest)
    incumbent_objective = _meta_trial_objective(incumbent_manifest)
    finite = all(
        math.isfinite(value)
        for value in (
            candidate_ev,
            incumbent_ev,
            candidate_month,
            incumbent_month,
            candidate_week,
            incumbent_week,
            candidate_objective,
            incumbent_objective,
        )
    )
    passed = bool(
        finite
        and ev_gain > 0.0
        and candidate_objective > incumbent_objective
        and candidate_month >= incumbent_month - allowed_degradation
        and candidate_week >= incumbent_week - allowed_degradation
    )
    return passed, {
        "passed": passed,
        "top10_ev_gain": float(ev_gain),
        "allowed_worst_period_degradation": float(allowed_degradation),
        "worst_month_delta": float(candidate_month - incumbent_month),
        "worst_week_delta": float(candidate_week - incumbent_week),
        "objective_delta": float(candidate_objective - incumbent_objective),
    }


def _hpo_param_grid(seed: int, requested_trials: int) -> list[dict[str, Any]]:
    trials = [{**preset, "trial_source": "preset"} for preset in META_HPO_PRESETS]
    rng = np.random.default_rng(int(seed))
    while len(trials) < int(requested_trials):
        classifier = {
            "n_estimators": int(rng.integers(100, 221)),
            "learning_rate": float(np.exp(rng.uniform(np.log(0.015), np.log(0.080)))),
            "num_leaves": int(rng.choice([15, 23, 31, 47, 63])),
            "max_depth": int(rng.choice([-1, 4, 5, 6, 8])),
            "min_child_samples": int(rng.integers(25, 111)),
            "subsample": float(rng.uniform(0.65, 0.95)),
            "colsample_bytree": float(rng.uniform(0.55, 0.95)),
            "reg_alpha": float(np.exp(rng.uniform(np.log(1e-4), np.log(3.0)))),
            "reg_lambda": float(np.exp(rng.uniform(np.log(0.3), np.log(12.0)))),
        }
        regressor = dict(classifier)
        trials.append(
            {
                "name": f"random_{len(trials):03d}",
                "trial_source": "random",
                "classifier": classifier,
                "regressor": regressor,
            }
        )
    return trials[: int(requested_trials)]


def run_hpo_smoke(
    *,
    handoff_dir: Path,
    ledger_path: Path | None,
    out_dir: Path,
    frontier: str,
    seed: int,
    train_scope: str,
    enable_base_prior_features: bool,
    enable_reliability_features: bool,
    enable_support_drift_features: bool,
    enable_hit_surprise_features: bool,
    enable_path_order_heads: bool,
    enable_path_order_blends: bool,
    feature_selection_top_n: int,
    feature_selection_target: str,
    feature_selection_method: str,
    hpo_trials: int,
    hpo_max_train_rows: int,
    max_oos_model_age_days: int = 0,
    single_fit_oos_window: bool = False,
    meta_head_mode: str = "multi",
    hpo_calibration_month: str = "",
    eval_months: list[str] | None = None,
    save_fold_models: bool = False,
    fixed_selected_features: list[str] | None = None,
    fixed_selected_features_by_side: dict[str, list[str]] | None = None,
    side_specific_single_head: bool = True,
    hpo_refit_shortlist_size: int = 8,
    strict_handoff_contract: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    trial_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    resume_selected_features = (
        list(fixed_selected_features) if fixed_selected_features is not None else None
    )
    resume_selected_features_by_side = (
        {side: list(features) for side, features in fixed_selected_features_by_side.items()}
        if fixed_selected_features_by_side
        else None
    )
    completed_manifests = sorted(out_dir.glob("trial_*/manifest.json"))
    if completed_manifests and resume_selected_features is None:
        first_completed = json.loads(
            completed_manifests[0].read_text(encoding="utf-8")
        )
        resume_selected_features = list(
            first_completed.get("selected_feature_union", []) or []
        )
        raw_by_side = first_completed.get("selected_features_by_side", {}) or {}
        if isinstance(raw_by_side, dict):
            resume_selected_features_by_side = {
                str(side): list(features)
                for side, features in raw_by_side.items()
                if str(side) in {"long", "short"}
            }
        if not resume_selected_features:
            raise ValueError(
                "Cannot resume meta HPO without the persisted selected feature contract"
            )
        print(
            json.dumps(
                {
                    "event": "s52_train_meta_hpo_resume",
                    "completed_trials": len(completed_manifests),
                    "selected_features": len(resume_selected_features),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    for trial_idx, profile in enumerate(_hpo_param_grid(seed, hpo_trials)):
        trial_name = str(profile["name"])
        trial_dir = out_dir / f"trial_{trial_idx:03d}_{trial_name}"
        existing_manifest_path = trial_dir / "manifest.json"
        if existing_manifest_path.exists():
            manifest = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        else:
            projected_handoff_columns = _projected_handoff_columns_for_selected(
                handoff_dir / "train_meta_regime_handoff.parquet",
                resume_selected_features,
            )
            manifest = run_smoke(
                handoff_dir=handoff_dir,
                ledger_path=ledger_path,
                out_dir=trial_dir,
                frontier=frontier,
                seed=int(seed) + 10_000 * trial_idx,
                train_scope=train_scope,
                enable_base_prior_features=enable_base_prior_features,
                enable_reliability_features=enable_reliability_features,
                enable_support_drift_features=enable_support_drift_features,
                enable_hit_surprise_features=enable_hit_surprise_features,
                enable_path_order_heads=enable_path_order_heads,
                enable_path_order_blends=enable_path_order_blends,
                feature_selection_top_n=feature_selection_top_n,
                feature_selection_target=feature_selection_target,
                feature_selection_method=feature_selection_method,
                max_oos_model_age_days=int(max_oos_model_age_days),
                single_fit_oos_window=bool(single_fit_oos_window),
                validation_scope="largest",
                model_train_max_rows=int(hpo_max_train_rows),
                model_params={
                    "classifier": profile["classifier"],
                    "regressor": profile["regressor"],
                },
                model_profile_name=trial_name,
                meta_head_mode=str(meta_head_mode),
                minimal_artifacts=True,
                fixed_selected_features=resume_selected_features,
                fixed_selected_features_by_side=resume_selected_features_by_side,
                side_specific_single_head=bool(side_specific_single_head),
                strict_handoff_contract=bool(strict_handoff_contract),
                handoff_columns=projected_handoff_columns,
                eval_months=[str(hpo_calibration_month)]
                if str(hpo_calibration_month).strip()
                else None,
            )
            if resume_selected_features is None:
                resume_selected_features = list(
                    manifest.get("selected_feature_union", []) or []
                )
                raw_by_side = manifest.get("selected_features_by_side", {}) or {}
                resume_selected_features_by_side = {
                    str(side): list(features)
                    for side, features in raw_by_side.items()
                    if str(side) in {"long", "short"}
                }
        objective = _meta_trial_objective(manifest)
        best = _meta_hpo_selector_row(manifest)
        best_threshold = dict(manifest.get("best_threshold_policy", {}) or {})
        trial_rows.append(
            {
                "trial_idx": int(trial_idx),
                "trial_name": trial_name,
                "trial_source": str(profile.get("trial_source", "")),
                "objective": float(objective),
                "trial_dir": str(trial_dir),
                "best_selector": best.get("selector"),
                "best_selector_meta_ev_frontier_objective": best.get(
                    "meta_ev_frontier_objective"
                ),
                "best_keep020_ev_after_1pct": best.get("mean_keep020_ev_after_1pct"),
                "best_keep020_clean_exec_precision": best.get(
                    "mean_keep020_clean_exec_precision"
                ),
                "best_keep020_dirty_positive_rate": best.get(
                    "mean_keep020_dirty_positive_rate"
                ),
                "best_keep010_exec_margin": best.get("mean_keep010_exec_margin"),
                "best_keep010_ev_after_1pct": best.get("mean_keep010_ev_after_1pct"),
                "best_worst_keep010_ev_after_1pct": best.get(
                    "worst_keep010_ev_after_1pct"
                ),
                "best_keep010_ret_net": best.get("mean_keep010_ret_net"),
                "best_keep010_clean_exec_precision": best.get(
                    "mean_keep010_clean_exec_precision"
                ),
                "best_keep010_dirty_positive_rate": best.get(
                    "mean_keep010_dirty_positive_rate"
                ),
                "best_keep010_first_touch_bad_mae": best.get(
                    "mean_keep010_first_touch_bad_mae"
                ),
                "best_keep010_bad_mae": best.get("mean_keep010_full_path_bad_mae"),
                "best_keep010_timeout": best.get("mean_keep010_timeout"),
                "best_threshold_selector": best_threshold.get("selector"),
                "best_threshold_policy": best_threshold.get("policy_id"),
                "best_threshold_budget_frac": best_threshold.get("budget_frac"),
                "best_threshold_meta_ev_frontier_objective": best_threshold.get(
                    "meta_ev_frontier_objective"
                ),
                "best_threshold_ev_after_1pct": best_threshold.get(
                    "mean_ev_after_1pct"
                ),
                "best_threshold_exec_margin": best_threshold.get("mean_exec_margin"),
                "best_threshold_worst_exec_margin": best_threshold.get(
                    "worst_exec_margin"
                ),
                "best_threshold_clean_exec_precision": best_threshold.get(
                    "mean_clean_exec_precision"
                ),
                "best_threshold_dirty_positive_rate": best_threshold.get(
                    "mean_dirty_positive_rate"
                ),
                "best_threshold_first_touch_bad_mae": best_threshold.get(
                    "mean_first_touch_bad_mae"
                ),
                "best_threshold_bad_mae": best_threshold.get("mean_full_path_bad_mae"),
                "best_threshold_timeout": best_threshold.get("mean_timeout"),
                "selected_feature_union_count": manifest.get(
                    "selected_feature_union_count"
                ),
                "selected_feature_intersection_count": manifest.get(
                    "selected_feature_intersection_count"
                ),
                **{f"classifier_{k}": v for k, v in profile["classifier"].items()},
                **{f"regressor_{k}": v for k, v in profile["regressor"].items()},
            }
        )
        manifests.append(manifest)
    trials = (
        pd.DataFrame(trial_rows)
        .sort_values("objective", ascending=False)
        .reset_index(drop=True)
    )
    trials.insert(0, "rank", np.arange(1, len(trials) + 1, dtype=np.int32))
    trials_path = out_dir / "s52_train_meta_hpo_trials.csv"
    trials.to_csv(trials_path, index=False)
    profiles = _hpo_param_grid(seed, hpo_trials)
    shortlist_n = max(1, min(int(hpo_refit_shortlist_size), len(trials)))
    shortlist_indices = [int(v) for v in trials.head(shortlist_n)["trial_idx"].tolist()]
    baseline_idx = next(
        (idx for idx, profile in enumerate(profiles) if profile.get("name") == "baseline"),
        0,
    )
    if baseline_idx not in shortlist_indices:
        shortlist_indices.append(int(baseline_idx))
    fullfit_rows: list[dict[str, Any]] = []
    fullfit_manifests: dict[int, dict[str, Any]] = {}
    fullfit_root = out_dir / "fullfit_validation"
    for trial_idx in shortlist_indices:
        profile = profiles[trial_idx]
        trial_manifest = manifests[trial_idx]
        selected = list(trial_manifest.get("selected_feature_union", []) or [])
        selected_by_side = dict(trial_manifest.get("selected_features_by_side", {}) or {})
        fullfit_dir = fullfit_root / f"trial_{trial_idx:03d}_{profile['name']}"
        fullfit_manifest = run_smoke(
            handoff_dir=handoff_dir,
            ledger_path=ledger_path,
            out_dir=fullfit_dir,
            frontier=frontier,
            seed=int(seed) + 500_000 + trial_idx,
            train_scope=train_scope,
            enable_base_prior_features=enable_base_prior_features,
            enable_reliability_features=enable_reliability_features,
            enable_support_drift_features=enable_support_drift_features,
            enable_hit_surprise_features=enable_hit_surprise_features,
            enable_path_order_heads=enable_path_order_heads,
            enable_path_order_blends=enable_path_order_blends,
            feature_selection_top_n=feature_selection_top_n,
            feature_selection_target=feature_selection_target,
            feature_selection_method=feature_selection_method,
            max_oos_model_age_days=int(max_oos_model_age_days),
            single_fit_oos_window=bool(single_fit_oos_window),
            validation_scope="largest",
            model_train_max_rows=0,
            model_params={
                "classifier": profile["classifier"],
                "regressor": profile["regressor"],
            },
            model_profile_name=f"fullfit_validation_{profile['name']}",
            meta_head_mode=str(meta_head_mode),
            minimal_artifacts=True,
            fixed_selected_features=selected,
            fixed_selected_features_by_side=selected_by_side,
            side_specific_single_head=bool(side_specific_single_head),
            strict_handoff_contract=bool(strict_handoff_contract),
            handoff_columns=_projected_handoff_columns_for_selected(
                handoff_dir / "train_meta_regime_handoff.parquet", selected
            ),
            eval_months=[str(hpo_calibration_month)]
            if str(hpo_calibration_month).strip()
            else None,
        )
        fullfit_manifests[trial_idx] = fullfit_manifest
        fullfit_rows.append(
            {
                "trial_idx": trial_idx,
                "trial_name": profile["name"],
                "fullfit_objective": _meta_trial_objective(fullfit_manifest),
                "fullfit_top10_ev": _meta_hpo_selector_row(fullfit_manifest).get(
                    "mean_keep010_ev_after_1pct"
                ),
                "fullfit_worst_month_top10_ev": _meta_hpo_selector_row(
                    fullfit_manifest
                ).get("worst_keep010_ev_after_1pct"),
                "fullfit_worst_week_top10_ev": _meta_hpo_selector_row(
                    fullfit_manifest
                ).get("worst_week_keep010_ev_after_1pct"),
                "output_dir": fullfit_manifest.get("output_dir"),
            }
        )
    fullfit = pd.DataFrame(fullfit_rows).sort_values(
        "fullfit_objective", ascending=False
    )
    incumbent_manifest = fullfit_manifests[baseline_idx]
    promotion_audits: dict[int, dict[str, float | bool]] = {}
    promotable: list[int] = []
    for trial_idx in shortlist_indices:
        if trial_idx == baseline_idx:
            continue
        passed, audit = _hpo_candidate_beats_incumbent(
            fullfit_manifests[trial_idx], incumbent_manifest
        )
        promotion_audits[trial_idx] = audit
        if passed:
            promotable.append(trial_idx)
    best_idx = (
        max(promotable, key=lambda idx: _meta_trial_objective(fullfit_manifests[idx]))
        if promotable
        else baseline_idx
    )
    best_manifest = fullfit_manifests[best_idx]
    best_profile = profiles[best_idx]
    best_trial = trials.loc[trials["trial_idx"].eq(best_idx)].iloc[0].to_dict()
    best_trial["promotion_status"] = (
        "challenger_promoted" if best_idx != baseline_idx else "incumbent_retained"
    )
    best_trial["fullfit_objective"] = _meta_trial_objective(best_manifest)
    best_trial["promotion_audit"] = promotion_audits.get(best_idx, {})
    fullfit.to_csv(out_dir / "s52_train_meta_hpo_fullfit_shortlist.csv", index=False)
    final_manifest = {}
    incumbent_final_manifest: dict[str, Any] = {}
    if best_profile is not None:
        final_dir = out_dir / "best_full_oos"
        final_selected_features = list(
            best_manifest.get("selected_feature_union", []) or []
        )
        final_selected_features_by_side = dict(
            best_manifest.get("selected_features_by_side", {}) or {}
        )
        final_manifest = run_smoke(
            handoff_dir=handoff_dir,
            ledger_path=ledger_path,
            out_dir=final_dir,
            frontier=frontier,
            seed=int(seed) + 900_000 + best_idx,
            train_scope=train_scope,
            enable_base_prior_features=enable_base_prior_features,
            enable_reliability_features=enable_reliability_features,
            enable_support_drift_features=enable_support_drift_features,
            enable_hit_surprise_features=enable_hit_surprise_features,
            enable_path_order_heads=enable_path_order_heads,
            enable_path_order_blends=enable_path_order_blends,
            feature_selection_top_n=feature_selection_top_n,
            feature_selection_target=feature_selection_target,
            feature_selection_method=feature_selection_method,
            max_oos_model_age_days=int(max_oos_model_age_days),
            single_fit_oos_window=bool(single_fit_oos_window),
            validation_scope="all",
            model_train_max_rows=0,
            model_params={
                "classifier": best_profile["classifier"],
                "regressor": best_profile["regressor"],
            },
            model_profile_name=f"best_full_oos_{best_profile['name']}",
            meta_head_mode=str(meta_head_mode),
            fixed_selected_features=final_selected_features,
            fixed_selected_features_by_side=final_selected_features_by_side,
            side_specific_single_head=bool(side_specific_single_head),
            strict_handoff_contract=bool(strict_handoff_contract),
            handoff_columns=_projected_handoff_columns_for_selected(
                handoff_dir / "train_meta_regime_handoff.parquet",
                final_selected_features,
            ),
            eval_months=eval_months or None,
            save_fold_models=bool(save_fold_models),
        )
    deployment_idx = best_idx
    deployment_manifest = final_manifest
    final_holdout_audit: dict[str, float | bool] = {}
    if best_idx != baseline_idx and final_manifest:
        incumbent_profile = profiles[baseline_idx]
        incumbent_features = list(
            incumbent_manifest.get("selected_feature_union", []) or []
        )
        incumbent_features_by_side = dict(
            incumbent_manifest.get("selected_features_by_side", {}) or {}
        )
        incumbent_final_manifest = run_smoke(
            handoff_dir=handoff_dir,
            ledger_path=ledger_path,
            out_dir=out_dir / "incumbent_full_oos",
            frontier=frontier,
            seed=int(seed) + 950_000 + baseline_idx,
            train_scope=train_scope,
            enable_base_prior_features=enable_base_prior_features,
            enable_reliability_features=enable_reliability_features,
            enable_support_drift_features=enable_support_drift_features,
            enable_hit_surprise_features=enable_hit_surprise_features,
            enable_path_order_heads=enable_path_order_heads,
            enable_path_order_blends=enable_path_order_blends,
            feature_selection_top_n=feature_selection_top_n,
            feature_selection_target=feature_selection_target,
            feature_selection_method=feature_selection_method,
            max_oos_model_age_days=int(max_oos_model_age_days),
            single_fit_oos_window=bool(single_fit_oos_window),
            validation_scope="all",
            model_train_max_rows=0,
            model_params={
                "classifier": incumbent_profile["classifier"],
                "regressor": incumbent_profile["regressor"],
            },
            model_profile_name=f"incumbent_full_oos_{incumbent_profile['name']}",
            meta_head_mode=str(meta_head_mode),
            fixed_selected_features=incumbent_features,
            fixed_selected_features_by_side=incumbent_features_by_side,
            side_specific_single_head=bool(side_specific_single_head),
            strict_handoff_contract=bool(strict_handoff_contract),
            handoff_columns=_projected_handoff_columns_for_selected(
                handoff_dir / "train_meta_regime_handoff.parquet",
                incumbent_features,
            ),
            eval_months=eval_months or None,
            save_fold_models=bool(save_fold_models),
        )
        final_passed, final_holdout_audit = _hpo_candidate_beats_incumbent(
            final_manifest, incumbent_final_manifest
        )
        if not final_passed:
            deployment_idx = baseline_idx
            deployment_manifest = incumbent_final_manifest
    deployment_status = (
        "challenger_promoted_after_holdout"
        if deployment_idx != baseline_idx
        else (
            "incumbent_retained_after_holdout"
            if best_idx != baseline_idx
            else "incumbent_retained_at_calibration"
        )
    )
    best_trial["deployment_status"] = deployment_status
    best_trial["deployment_trial_idx"] = int(deployment_idx)
    best_trial["deployment_trial_name"] = str(profiles[deployment_idx]["name"])
    best_trial["final_holdout_audit"] = final_holdout_audit
    best_params = {
        "trial": _json_safe(best_trial),
        "classifier_params": deployment_manifest.get("classifier_params", {}),
        "regressor_params": deployment_manifest.get("regressor_params", {}),
        "selected_feature_union": deployment_manifest.get(
            "selected_feature_union", []
        ),
        "selected_feature_intersection": deployment_manifest.get(
            "selected_feature_intersection", []
        ),
        "selected_features_by_side": deployment_manifest.get(
            "selected_features_by_side", {}
        ),
        "best_full_oos_manifest": deployment_manifest.get("output_dir"),
        "challenger_full_oos_manifest": final_manifest.get("output_dir"),
        "incumbent_full_oos_manifest": incumbent_final_manifest.get("output_dir"),
    }
    (out_dir / "s52_train_meta_hpo_best.json").write_text(
        json.dumps(_json_safe(best_params), indent=2), encoding="utf-8"
    )
    manifest = {
        "generated_by": "run_s52_train_meta_regime_handoff_smoke_hpo",
        "handoff_dir": str(handoff_dir),
        "ledger_path": str(ledger_path) if ledger_path is not None else None,
        "frontier": str(frontier),
        "train_scope": str(train_scope),
        "hpo_trials": int(hpo_trials),
        "hpo_refit_shortlist_size": int(shortlist_n),
        "hpo_selection_contract": "sampled_screen_then_deployment_scale_refit_then_holdout_incumbent_guard",
        "promotion_status": deployment_status,
        "hpo_scope": "single_largest_train_fold",
        "hpo_sampling": "beginning_middle_end_time_spread_for_lgbm_pipeline_feature_selection_and_model_fit",
        "hpo_max_train_rows": int(hpo_max_train_rows),
        "best_full_oos_model_train_rows": "full_train_rows",
        "feature_selection_top_n": int(feature_selection_top_n),
        "feature_selection_target": str(feature_selection_target),
        "feature_selection_method": str(feature_selection_method),
        "max_oos_model_age_days": int(max_oos_model_age_days),
        "meta_head_mode": str(meta_head_mode),
        "hpo_calibration_month": str(hpo_calibration_month),
        "eval_months": list(eval_months or []),
        "trial_summary": str(trials_path),
        "best_params": str(out_dir / "s52_train_meta_hpo_best.json"),
        "best_trial": _json_safe(best_trial),
        "best_trial_manifest": best_manifest.get("output_dir"),
        "best_full_oos_manifest": deployment_manifest.get("output_dir"),
        "challenger_full_oos_manifest": final_manifest.get("output_dir"),
        "incumbent_full_oos_manifest": incumbent_final_manifest.get("output_dir"),
        "final_holdout_audit": _json_safe(final_holdout_audit),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-dir", type=Path, default=DEFAULT_HANDOFF_DIR)
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--frontier", choices=["top10", "top20", "top30"], default="top30"
    )
    parser.add_argument(
        "--train-scope", choices=["selected", "all"], default="selected"
    )
    parser.add_argument(
        "--strict-handoff-contract",
        action="store_true",
        help=(
            "Fail closed unless the handoff is timestamp_side top30 with one "
            "explicit base target/sample-weight contract hash."
        ),
    )
    parser.add_argument(
        "--enable-base-prior-features",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enable-reliability-features",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enable-support-drift-features",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enable-hit-surprise-features",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--enable-path-order-heads", action="store_true")
    parser.add_argument("--enable-path-order-blends", action="store_true")
    parser.add_argument(
        "--meta-head-mode",
        choices=["multi", "single_base_soft_label"],
        default="single_base_soft_label",
        help="Use the legacy multi-head meta stack or one classifier trained on the base economic soft label.",
    )
    parser.add_argument(
        "--feature-selection-top-n",
        type=int,
        default=0,
        help=(
            "Legacy explicit selected-feature cap. Default 0 lets "
            "lgbm_pipeline.py pick the feature count automatically; positive "
            "values are ignored unless --force-feature-selection-top-n is set."
        ),
    )
    parser.add_argument(
        "--force-feature-selection-top-n",
        action="store_true",
        help="Honor --feature-selection-top-n as an explicit cap instead of MDA auto-count.",
    )
    parser.add_argument(
        "--feature-selection-target",
        choices=["ev_frontier", "clean_minus_bad", "exec_margin", "clean_exec"],
        default="ev_frontier",
    )
    parser.add_argument(
        "--feature-selection-method",
        choices=["auto", "lgbm_pipeline", "lgbm_staged"],
        default="auto",
        help="Canonical staged selector in lgbm_pipeline.py: univariate + ReliefF + redundancy clustering + iterative MDA pruning.",
    )
    parser.add_argument("--hpo-trials", type=int, default=0)
    parser.add_argument(
        "--hpo-refit-shortlist-size",
        type=int,
        default=8,
        help="Refit the top sampled-HPO candidates plus the incumbent on the full calibration train set before promotion.",
    )
    parser.add_argument(
        "--single-head-global-model",
        dest="single_head_global_model",
        action="store_true",
        default=True,
        help=(
            "Fit one shared soft-label head across long and short rows. This is "
            "the promoted V9-recovery architecture; side/archetype context remains "
            "in the feature contract."
        ),
    )
    parser.add_argument(
        "--side-specific-single-head",
        dest="single_head_global_model",
        action="store_false",
        help="Ablation: fit independent long/short soft-label heads.",
    )
    parser.add_argument(
        "--hpo-calibration-month",
        type=str,
        default="",
        help=(
            "When HPO is enabled, use only this chronological validation month "
            "for one-shot feature selection and parameter tuning. Later --eval-months "
            "remain untouched OOS evaluation months."
        ),
    )
    parser.add_argument(
        "--fixed-selected-features-csv",
        type=Path,
        default=DEFAULT_META_CHAMPION_CONTRACT,
        help="Frozen selected-feature contract; defaults to the promoted shared champion.",
    )
    parser.add_argument(
        "--fresh-feature-selection",
        action="store_true",
        help="Ignore the promoted frozen feature contract and rerun chronological MDA.",
    )
    parser.add_argument(
        "--single-phase-wide-feature-selection",
        action="store_true",
        help=(
            "Diagnostic opt-out from the canonical two-phase path. By default, "
            "fresh MDA reads a wide B/M/E sample first and reloads the full "
            "candidate population only after projecting selected columns."
        ),
    )
    parser.add_argument(
        "--fixed-model-params-json",
        type=Path,
        default=DEFAULT_META_CHAMPION_CONTRACT,
        help="Frozen LightGBM parameters; defaults to the promoted shared champion.",
    )
    parser.add_argument(
        "--model-profile-name",
        type=str,
        default="meta_v9_anchor_oldparams_residual_backbone_v1",
        help="Profile name recorded in the replay manifest.",
    )
    parser.add_argument(
        "--eval-months",
        type=str,
        default="",
        help="Comma-separated OOS months to score, e.g. 2026-07. Empty scores all eligible months.",
    )
    parser.add_argument(
        "--feature-selection-sample-rows",
        type=int,
        default=LGBM_TWO_PHASE_SELECTION_SAMPLE_ROWS,
        help=(
            "Rows in the one-shot beginning/middle/end feature-selection sample. "
            "This is independent from the smaller per-trial HPO sample."
        ),
    )
    parser.add_argument(
        "--hpo-max-train-rows",
        type=int,
        default=LGBM_HPO_SAMPLE_ROWS,
        help=(
            "Rows used for one-shot feature selection and HPO trial fitting, "
            "sampled from beginning/middle/end of the calibration fold. A fresh "
            "non-HPO MDA run samples only selection rows; its final side models "
            "still fit all prior rows."
        ),
    )
    parser.add_argument(
        "--max-oos-model-age-days",
        type=int,
        default=0,
        help="Split each OOS month into expanding-window validation slices capped to this many days.",
    )
    parser.add_argument(
        "--single-fit-oos-window",
        action="store_true",
        help=(
            "Fit meta once before the first requested evaluation month and score "
            "the complete contiguous evaluation range without growing refits."
        ),
    )
    parser.add_argument(
        "--save-fold-models",
        action="store_true",
        help="Persist each OOS fold's fitted meta models plus columns.json and leakage manifest.",
    )
    parser.add_argument("--seed", type=int, default=20260705)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    fixed_feature_contract = (
        None if bool(args.fresh_feature_selection) else args.fixed_selected_features_csv
    )
    fixed_selected_features = _load_fixed_selected_features(
        fixed_feature_contract
    )
    fixed_selected_features_by_side = _load_fixed_selected_features_by_side(
        fixed_feature_contract
    )
    if (
        not bool(args.single_head_global_model)
        and fixed_selected_features is not None
        and not fixed_selected_features_by_side
    ):
        # A legacy shared feature list cannot satisfy a side-aware model
        # contract. Treat it only as historical provenance: rerun the
        # train-only, calibration-fold MDA once and persist proper long/short
        # contracts in the new manifest.
        print(
            "[feature_selection] legacy shared meta contract detected; "
            "rerunning side-local MDA for the side-aware model contract.",
            flush=True,
        )
        fixed_selected_features = None
    had_frozen_feature_contract = bool(
        fixed_selected_features is not None or fixed_selected_features_by_side
    )
    diagnostic_single_phase_opt_out = bool(
        not had_frozen_feature_contract and args.single_phase_wide_feature_selection
    )
    fixed_model_params = _load_fixed_model_params(args.fixed_model_params_json)
    fixed_feature_union = list(fixed_selected_features or [])
    for features in (fixed_selected_features_by_side or {}).values():
        fixed_feature_union.extend(features)
    projected_handoff_columns = _projected_handoff_columns_for_selected(
        args.handoff_dir / "train_meta_regime_handoff.parquet",
        list(dict.fromkeys(fixed_feature_union)),
    )
    eval_months = [
        m.strip() for m in str(args.eval_months or "").split(",") if m.strip()
    ]
    feature_selection_top_n = int(args.feature_selection_top_n)
    if feature_selection_top_n > 0 and not bool(args.force_feature_selection_top_n):
        print(
            "[feature_selection] ignoring explicit --feature-selection-top-n="
            f"{feature_selection_top_n}; using MDA auto-count. Pass "
            "--force-feature-selection-top-n to cap intentionally.",
            flush=True,
        )
        feature_selection_top_n = 0
    two_phase_selection_manifest: dict[str, Any] | None = None
    use_two_phase_selection = use_canonical_two_phase_feature_selection(
        has_frozen_feature_contract=had_frozen_feature_contract,
        diagnostic_single_phase=bool(args.single_phase_wide_feature_selection),
    )
    if use_two_phase_selection:
        sample_dir = args.out_dir / "_feature_selection_bme_sample"
        sample_manifest_path = sample_dir / "manifest.json"
        sampled_handoff = sample_dir / "train_meta_regime_handoff.parquet"
        sampled_ledger = sample_dir / "s52_trailing_regime_scored_ledger.parquet"
        if (
            sample_manifest_path.is_file()
            and sampled_handoff.is_file()
            and sampled_ledger.is_file()
        ):
            sample_manifest = json.loads(sample_manifest_path.read_text())
        else:
            source_ledger = args.ledger or (
                args.handoff_dir / "s52_trailing_regime_scored_ledger.parquet"
            )
            sample_manifest = materialize_meta_bme_sample(
                handoff=args.handoff_dir / "train_meta_regime_handoff.parquet",
                ledger=source_ledger,
                out_dir=sample_dir,
                rows=int(args.feature_selection_sample_rows),
                seed=int(args.seed),
            )
        # The B/M/E sample is a row subset of the same base candidate stream.
        # Carry its immutable provenance sidecar so strict selection validates
        # the sampled rows against the identical base target/weight contract.
        source_contract_path = _handoff_contract_path(
            args.handoff_dir / "train_meta_regime_handoff.parquet"
        )
        sampled_contract_path = _handoff_contract_path(sampled_handoff)
        if source_contract_path.is_file() and not sampled_contract_path.is_file():
            sampled_contract_path.write_text(
                source_contract_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
        if bool(args.strict_handoff_contract) and not sampled_contract_path.is_file():
            raise ValueError(
                "Strict two-phase meta selection requires the source handoff "
                f"contract sidecar: {source_contract_path}"
            )
        selection_dir = args.out_dir / "_feature_selection_phase"
        selection_manifest_path = selection_dir / "manifest.json"
        if selection_manifest_path.is_file():
            selection_manifest = json.loads(selection_manifest_path.read_text())
        else:
            selection_manifest = run_smoke(
                handoff_dir=sample_dir,
                ledger_path=sampled_ledger,
                out_dir=selection_dir,
                frontier=args.frontier,
                seed=int(args.seed),
                train_scope=str(args.train_scope),
                enable_base_prior_features=bool(args.enable_base_prior_features),
                enable_reliability_features=bool(args.enable_reliability_features),
                enable_support_drift_features=bool(args.enable_support_drift_features),
                enable_hit_surprise_features=bool(args.enable_hit_surprise_features),
                enable_path_order_heads=bool(args.enable_path_order_heads),
                enable_path_order_blends=bool(args.enable_path_order_blends),
                feature_selection_top_n=int(feature_selection_top_n),
                feature_selection_target=str(args.feature_selection_target),
                feature_selection_method=str(args.feature_selection_method),
                max_oos_model_age_days=int(args.max_oos_model_age_days),
                single_fit_oos_window=bool(args.single_fit_oos_window),
                validation_scope="largest",
                model_train_max_rows=LGBM_TWO_PHASE_FULL_FIT_ROW_CAP,
                feature_selection_max_rows=int(args.feature_selection_sample_rows),
                model_params=fixed_model_params,
                model_profile_name=f"{args.model_profile_name}_selection_phase",
                meta_head_mode=str(args.meta_head_mode),
                minimal_artifacts=True,
                side_specific_single_head=not bool(args.single_head_global_model),
                strict_handoff_contract=bool(args.strict_handoff_contract),
            )
        fixed_selected_features = list(
            selection_manifest.get("selected_feature_union", []) or []
        )
        raw_by_side = selection_manifest.get("selected_features_by_side", {}) or {}
        fixed_selected_features_by_side = {
            str(side): list(features)
            for side, features in raw_by_side.items()
            if str(side) in {"long", "short"} and features
        }
        if not fixed_selected_features:
            raise RuntimeError("Two-phase meta selection produced no feature contract")
        fixed_feature_union = list(fixed_selected_features)
        for features in fixed_selected_features_by_side.values():
            fixed_feature_union.extend(features)
        projected_handoff_columns = _projected_handoff_columns_for_selected(
            args.handoff_dir / "train_meta_regime_handoff.parquet",
            list(dict.fromkeys(fixed_feature_union)),
        )
        two_phase_selection_manifest = {
            "schema": "meta_two_phase_wide_selection_v1",
            "training_contract": LGBM_TWO_PHASE_SELECTION_CONTRACT,
            "sample": sample_manifest,
            "selection_manifest": str(selection_manifest_path),
            "selected_feature_union_count": len(fixed_selected_features),
            "selected_feature_counts_by_side": {
                side: len(features)
                for side, features in fixed_selected_features_by_side.items()
            },
            "full_population_reload": "selected_raw_columns_only",
            "full_population_train_row_cap": LGBM_TWO_PHASE_FULL_FIT_ROW_CAP,
        }
        (args.out_dir / "two_phase_feature_selection.json").write_text(
            json.dumps(_json_safe(two_phase_selection_manifest), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    if int(args.hpo_trials) > 0:
        manifest = run_hpo_smoke(
            handoff_dir=args.handoff_dir,
            ledger_path=args.ledger,
            out_dir=args.out_dir,
            frontier=args.frontier,
            seed=int(args.seed),
            train_scope=str(args.train_scope),
            enable_base_prior_features=bool(args.enable_base_prior_features),
            enable_reliability_features=bool(args.enable_reliability_features),
            enable_support_drift_features=bool(args.enable_support_drift_features),
            enable_hit_surprise_features=bool(args.enable_hit_surprise_features),
            enable_path_order_heads=bool(args.enable_path_order_heads),
            enable_path_order_blends=bool(args.enable_path_order_blends),
            feature_selection_top_n=int(feature_selection_top_n),
            feature_selection_target=str(args.feature_selection_target),
            feature_selection_method=str(args.feature_selection_method),
            hpo_trials=int(args.hpo_trials),
            hpo_max_train_rows=int(args.hpo_max_train_rows),
            max_oos_model_age_days=int(args.max_oos_model_age_days),
            single_fit_oos_window=bool(args.single_fit_oos_window),
            meta_head_mode=str(args.meta_head_mode),
            hpo_calibration_month=str(args.hpo_calibration_month),
            eval_months=eval_months or None,
            save_fold_models=bool(args.save_fold_models),
            fixed_selected_features=fixed_selected_features,
            fixed_selected_features_by_side=fixed_selected_features_by_side,
            side_specific_single_head=not bool(args.single_head_global_model),
            hpo_refit_shortlist_size=int(args.hpo_refit_shortlist_size),
            strict_handoff_contract=bool(args.strict_handoff_contract),
        )
    else:
        manifest = run_smoke(
            handoff_dir=args.handoff_dir,
            ledger_path=args.ledger,
            out_dir=args.out_dir,
            frontier=args.frontier,
            seed=int(args.seed),
            train_scope=str(args.train_scope),
            enable_base_prior_features=bool(args.enable_base_prior_features),
            enable_reliability_features=bool(args.enable_reliability_features),
            enable_support_drift_features=bool(args.enable_support_drift_features),
            enable_hit_surprise_features=bool(args.enable_hit_surprise_features),
            enable_path_order_heads=bool(args.enable_path_order_heads),
            enable_path_order_blends=bool(args.enable_path_order_blends),
            feature_selection_top_n=int(feature_selection_top_n),
            feature_selection_target=str(args.feature_selection_target),
            feature_selection_method=str(args.feature_selection_method),
            max_oos_model_age_days=int(args.max_oos_model_age_days),
            single_fit_oos_window=bool(args.single_fit_oos_window),
            # Feature selection may use a bounded, time-spread sample. The
            # resulting long/short models must always use the complete
            # expanding training window in a normal OOS replay.
            model_train_max_rows=LGBM_TWO_PHASE_FULL_FIT_ROW_CAP,
            feature_selection_max_rows=(
                int(args.feature_selection_sample_rows)
                if fixed_selected_features is None
                else 0
            ),
            model_params=fixed_model_params,
            model_profile_name=str(args.model_profile_name),
            meta_head_mode=str(args.meta_head_mode),
            fixed_selected_features=fixed_selected_features,
            fixed_selected_features_by_side=fixed_selected_features_by_side,
            side_specific_single_head=not bool(args.single_head_global_model),
            eval_months=eval_months or None,
            save_fold_models=bool(args.save_fold_models),
            handoff_columns=projected_handoff_columns,
            strict_handoff_contract=bool(args.strict_handoff_contract),
        )
    if two_phase_selection_manifest is not None:
        manifest["two_phase_feature_selection"] = two_phase_selection_manifest
        manifest_path = args.out_dir / "manifest.json"
        if manifest_path.is_file():
            manifest_path.write_text(
                json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    manifest["feature_materialization_contract"] = {
        "schema": LGBM_TWO_PHASE_SELECTION_CONTRACT,
        "canonical_default": True,
        "fresh_selection_used_two_phase": bool(use_two_phase_selection),
        "diagnostic_single_phase_opt_out": diagnostic_single_phase_opt_out,
        "selection_sample_rows": int(args.feature_selection_sample_rows),
        "full_fit_projection": "selected_raw_columns_only",
        "full_fit_train_row_cap": LGBM_TWO_PHASE_FULL_FIT_ROW_CAP,
        "full_fit_population": "all_prior_rows",
    }
    manifest_path = args.out_dir / "manifest.json"
    if manifest_path.is_file():
        manifest_path.write_text(
            json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            _json_safe({"event": "s52_train_meta_handoff_smoke_done", **manifest}),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
