"""
Model Orchestrator for Inference.

This module orchestrates the full inference chain:
2. Alpha model predictions (long_mr, long_tf, short_mr, short_tf)
3. Compute disagreement features
4. Meta model predictions
5. Ridge position sizing
6. Entry policy (Limit Offset Optimizer)

Returns full prediction chain results for each candidate.
"""

from __future__ import annotations

import os
import re
import resource
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import psutil as _psutil
except Exception:  # pragma: no cover - psutil is an optional runtime aid
    _psutil = None

from extreme_price_movements.engine import _calculate_disagreement_features
from extreme_price_movements.entry_policy import (
    compute_entry_policy_decision,
    flatten_bucket_policy,
)
from extreme_price_movements.inference.feature_generator import (
    get_features_for_candidates,
    is_model_derived_feature_key,
)
from extreme_price_movements.inference.feature_parity import (
    FeatureParityError,
    validate_final_model_matrix,
)
from extreme_price_movements.inference.parity import (
    LIVE_UNAVAILABLE_FEATURES,
    strategy_core_id,
    strategy_id_matches,
    strategy_side,
)
from extreme_price_movements.meta_training.trade_filtering import (
    rolling_asset_percentile,
)
from extreme_price_movements.model_drift_features import (
    MODEL_DRIFT_FEATURE_KEYS,
    transform_model_drift_features,
)
from extreme_price_movements.model_effectiveness_history import (
    apply_model_effectiveness_history_defaults,
    extract_model_effectiveness_history_defaults,
)
from extreme_price_movements.optional_model_features import (
    is_optional_generated_model_feature_key,
)
from extreme_price_movements.drift_monitoring import load_latest_drift_regime_features
from extreme_price_movements.lgbm_archetype_features import (
    BASE_ERROR_ARCHETYPE_FEATURE_NAMES,
    RAW_CONTRIB_FEATURE_PREFIX,
    RAW_STATE_DIAGNOSTIC_FEATURE_NAMES,
    RAW_STATE_SVD_FEATURE_NAMES,
    is_raw_contrib_feature_name,
    transform_residual_error_archetype_features,
    transform_raw_state_archetype_features,
)
try:
    from extreme_price_movements.lgbm_pipeline import LGBM_INTERNAL_METRIC_FEATURE_NAMES
except Exception:  # pragma: no cover - keep inference importable with older bundles
    LGBM_INTERNAL_METRIC_FEATURE_NAMES = ()
from extreme_price_movements.regime_adaptor import (
    apply_regime_adaptor,
    regime_adaptor_inference_enabled,
)
from extreme_price_movements.utils import tprint


DELETED_MODEL_FEATURE_KEYS = {
    "p_exh_lag1",
    "retest_accept",
    "vol_price_diverge",
    "vortex_diff_14",
    "vortex_diff_21",
    "vortex_diff_34",
    "z_breakout_dn_24",
    "z_breakout_up_24",
    "z_slope_change_24",
    "z_sm_momentum_24",
}


def _process_peak_rss_mb() -> float:
    try:
        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return float("nan")
    return rss / (1024.0 * 1024.0) if rss > 10_000_000 else rss / 1024.0


def _process_rss_mb() -> float:
    if _psutil is not None:
        try:
            return float(_psutil.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0)
        except Exception:
            pass
    return _process_peak_rss_mb()


def _process_rss_log_fields() -> str:
    return f"rss={_process_rss_mb():.1f}MB peak_rss={_process_peak_rss_mb():.1f}MB"


ALPHA_MODEL_META_FEATURE_KEYS = {
    *LGBM_INTERNAL_METRIC_FEATURE_NAMES,
    *MODEL_DRIFT_FEATURE_KEYS,
    *BASE_ERROR_ARCHETYPE_FEATURE_NAMES,
    *RAW_STATE_SVD_FEATURE_NAMES,
    *RAW_STATE_DIAGNOSTIC_FEATURE_NAMES,
    "feature_drift_psi_core",
    "feature_drift_ks_core",
}


LGBM_DIAGNOSTIC_LEDGER_KEYS = {
    *LGBM_INTERNAL_METRIC_FEATURE_NAMES,
    *MODEL_DRIFT_FEATURE_KEYS,
    "feature_drift_psi_core",
    "feature_drift_ks_core",
    "rare_leaf_fraction",
    "leaf_count_p10",
    "leaf_count_min",
    "leaf_weight_p10",
    "contrib_top1_abs_share",
    "contrib_top3_abs_share",
    "contrib_entropy",
    "contrib_balance",
    "num_material_contrib_features",
    "prob_uncertainty",
}


def _meta_live_unavailable_neutral_default(feature_name: str) -> float | None:
    """Decision-time neutral value for historical meta diagnostics.

    These features summarize matured historical model-error or predictive-atlas
    context. They are valid training/meta inputs when built causally, but the
    current live row cannot have a same-row outcome. Keep this whitelist narrow;
    ordinary raw/source features still fail strict parity when missing.
    """

    key = str(feature_name).lower()
    if key.startswith(("base_lgbm_predictive_atlas_", "meta_lgbm_predictive_atlas_")):
        if key.endswith(("_hit_rate", "_expected_hit_rate", "_score_mean")):
            return 0.5
        if key.endswith("_score_std"):
            return 0.0
        if key.endswith(("_support_n", "_effective_n", "_support_quality")):
            return 0.0
        if "surprise" in key or key.endswith(("_ic", "_rank_ic")):
            return 0.0
        return 0.0
    if key in {
        "signed_prediction_error",
        "surprise_error_z",
        "wrong_confident",
    }:
        return 0.0
    if key == "negative_log_likelihood":
        return 0.6931471805599453
    if key in {
        "prob_error",
        "recent_prob_error_20",
        "base_model_abs_error_roll20",
    }:
        return 0.5
    if key == "recent_hit_rate_20":
        return 0.5
    return None


def _fill_live_unavailable_meta_contract_features(
    features: pd.DataFrame,
    feature_cols: List[str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if not isinstance(features, pd.DataFrame) or features.empty:
        return features, []
    out = features.copy()
    filled: list[dict[str, Any]] = []
    for col in [str(c) for c in (feature_cols or []) if str(c)]:
        default = _meta_live_unavailable_neutral_default(col)
        if default is None:
            continue
        if col not in out.columns:
            out[col] = np.full(len(out), float(default), dtype=np.float32)
            filled.append(
                {
                    "feature": col,
                    "default": float(default),
                    "missing_column": True,
                    "filled_rows": int(len(out)),
                }
            )
            continue
        vals = pd.to_numeric(out[col], errors="coerce").to_numpy(
            dtype=np.float32,
            copy=True,
        )
        bad = ~np.isfinite(vals)
        bad_n = int(bad.sum())
        if bad_n <= 0:
            continue
        vals[bad] = float(default)
        out[col] = vals
        filled.append(
            {
                "feature": col,
                "default": float(default),
                "missing_column": False,
                "filled_rows": bad_n,
            }
        )
    return out, filled


def _fill_live_sparse_meta_context_features(
    features: pd.DataFrame,
    feature_cols: List[str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    # Sparse live source/context inputs must remain strict: rows with missing
    # trained values are dropped by the meta matrix finite-parity path below.
    # Keep the helper as a no-op so older call sites and diagnostics stay stable
    # without silently substituting raw market/context features at decision time.
    return features, []


def _first_row_diagnostics(frame: Any) -> Dict[str, float]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return {}
    row = frame.iloc[0]
    out: Dict[str, float] = {}
    for key in LGBM_DIAGNOSTIC_LEDGER_KEYS:
        if key not in row.index:
            continue
        try:
            value = float(row[key])
        except Exception:
            continue
        if np.isfinite(value):
            out[key] = value
    return out


def _feature_frame_with_lgbm_diagnostics(
    features: pd.DataFrame,
    *,
    base_diagnostics: Dict[str, float] | None = None,
    meta_diagnostics: Dict[str, float] | None = None,
) -> pd.DataFrame:
    """Expose base/meta LGBM diagnostics as regular downstream feature columns."""
    if not isinstance(features, pd.DataFrame) or features.empty:
        return features
    additions: Dict[str, float] = {}
    for prefix, diagnostics in (
        ("base_lgbm", base_diagnostics or {}),
        ("meta_lgbm", meta_diagnostics or {}),
    ):
        for key, value in diagnostics.items():
            try:
                numeric = float(value)
            except Exception:
                continue
            if not np.isfinite(numeric):
                continue
            col = f"{prefix}_{key}"
            if col not in features.columns:
                additions[col] = numeric
    if not additions:
        return features
    out = features.copy()
    for col, value in additions.items():
        out[col] = np.full(len(out), value, dtype=np.float32)
    return out


def _feature_frame_with_latest_drift_features(
    features: pd.DataFrame,
    cfg: Mapping[str, Any] | None,
) -> pd.DataFrame:
    """Join latest per-asset drift monitoring features into a scoring frame."""
    if not isinstance(features, pd.DataFrame) or features.empty:
        return features
    try:
        latest = load_latest_drift_regime_features(
            live_data_root=(cfg or {}).get("live_data_root"),
            data_root=(cfg or {}).get("data_root"),
        )
    except Exception:
        return features
    if latest is None or latest.empty or "symbol" not in latest.columns:
        return features
    if "symbol" in features.columns:
        symbols = features["symbol"].astype(str)
    else:
        symbols = pd.Series(features.index.astype(str), index=features.index)
    latest = latest.copy()
    latest["symbol"] = latest["symbol"].astype(str)
    latest = latest.drop_duplicates(subset=["symbol"], keep="last").set_index("symbol")
    drift_cols = [str(c) for c in latest.columns if str(c).startswith("drift_")]
    if not drift_cols:
        return features
    out = features.copy()
    aligned = latest.reindex(symbols.to_numpy())
    aligned.index = out.index
    added = 0
    for col in drift_cols:
        if col in out.columns:
            continue
        out[col] = pd.to_numeric(aligned[col], errors="coerce").astype(np.float32)
        added += 1
    return out if added else features


def _regime_adaptor_score_from_applied(
    applied: Dict[str, Any],
    final_preds: np.ndarray,
    regime_weight: np.ndarray,
) -> np.ndarray:
    if "combined_score" in applied and "deployment_score_pre_rank" in applied:
        return np.asarray(applied["deployment_score_pre_rank"], dtype=float)
    if "deployment_score" in applied:
        return np.asarray(applied["deployment_score"], dtype=float)
    if "error_risk_adjusted_score" in applied:
        return np.asarray(applied["error_risk_adjusted_score"], dtype=float)
    return np.asarray(final_preds, dtype=float) * np.clip(
        np.asarray(regime_weight, dtype=float), 0.75, 1.20
    )


def _extract_ebm_contract_model(model: Any) -> Any:
    """Return the EBM contract-bearing model nested in a meta wrapper, if any."""
    if model is None:
        return None
    if model.__class__.__name__ == "EBMOnLGBMModel":
        return model
    best_model = getattr(model, "best_model", None)
    if best_model is not None and best_model.__class__.__name__ == "EBMOnLGBMModel":
        return best_model
    ebm_model = getattr(model, "ebm_model", None)
    if ebm_model is not None and ebm_model.__class__.__name__ == "EBMOnLGBMModel":
        return ebm_model
    return None


def _missing_ebm_raw_contract(model: Any, features: pd.DataFrame) -> list[str]:
    """List required raw EBM features absent from a live inference frame."""
    ebm_model = _extract_ebm_contract_model(model)
    if ebm_model is None or not isinstance(features, pd.DataFrame):
        return []
    raw_features = [
        str(c) for c in (getattr(ebm_model, "raw_selected_features", []) or [])
    ]
    if not raw_features:
        return []
    available = set(map(str, features.columns))
    missing = [name for name in raw_features if name not in available]
    positional_mapping = (
        getattr(ebm_model, "positional_feature_mapping", None)
        or getattr(ebm_model, "meta_positional_feature_mapping_", None)
        or getattr(model, "positional_feature_mapping", None)
        or getattr(model, "meta_positional_feature_mapping_", None)
        or {}
    )
    if missing and isinstance(positional_mapping, dict) and positional_mapping:
        still_missing: list[str] = []
        for raw_name in missing:
            real_name = str(positional_mapping.get(raw_name, ""))
            if not real_name or real_name not in available:
                still_missing.append(real_name or raw_name)
        return still_missing
    return missing


def _effective_selected_feature_contract(model: Any) -> list[str]:
    """Return the feature contract actually consumed by a selected inner model.

    Older ModelRace wrappers persist a broad ``feature_columns`` list from the
    candidate matrix. The winning LGBMStabilityModel then selects a much smaller
    named subset. Inference validation must use the selected inner contract;
    otherwise stale, unused wrapper columns become false hard requirements.
    """
    inner = _selected_feature_owner(model)
    selected = [str(c) for c in (getattr(inner, "selected_features", []) or [])]
    if not selected:
        return []
    input_features = [str(c) for c in (getattr(inner, "input_feature_names", []) or [])]
    raw_contrib_inputs = [
        str(c) for c in (getattr(inner, "raw_contrib_input_features", []) or [])
    ]
    if raw_contrib_inputs and input_features:
        return input_features
    if len(input_features) == len(selected) and input_features != selected:
        return input_features
    return selected


def _model_drift_feature_alias_source(name: str) -> str | None:
    """Return the unprefixed drift/context source for a meta feature alias."""
    name_s = str(name)
    drift_keys = set(MODEL_DRIFT_FEATURE_KEYS)
    drift_keys.update(str(k) for k in LGBM_INTERNAL_METRIC_FEATURE_NAMES)
    drift_keys.update(str(k) for k in BASE_ERROR_ARCHETYPE_FEATURE_NAMES)
    drift_keys.update(str(k) for k in RAW_STATE_SVD_FEATURE_NAMES)
    drift_keys.update(str(k) for k in RAW_STATE_DIAGNOSTIC_FEATURE_NAMES)
    drift_keys.add("feature_drift_psi_core")
    drift_keys.add("feature_drift_ks_core")
    for key in sorted(drift_keys, key=len, reverse=True):
        if name_s == key:
            return key
        if name_s.endswith(f"_{key}") and re.match(
            r"^(?:pred(?:_.*)?_H\d+|base_H\d+|base_lgbm|meta_lgbm)_",
            name_s,
        ):
            return key
    if name_s.endswith("_reg_rare_leaf_low_support_score") and re.match(
        r"^(?:pred(?:_.*)?_H\d+|base_H\d+|base_lgbm|meta_lgbm)_",
        name_s,
    ):
        return "rare_leaf_low_support_score"
    if is_raw_contrib_feature_name(name_s):
        marker = RAW_CONTRIB_FEATURE_PREFIX
        pos = name_s.find(marker)
        if pos == 0:
            return name_s
        if pos > 0 and re.match(r"^(?:pred(?:_.*)?_H\d+|base_H\d+)_", name_s):
            return name_s[pos:]
    return None


def _materialize_model_drift_feature_aliases(
    frame: pd.DataFrame,
    needed: set[str],
    *,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, int]:
    """Populate prefixed meta drift aliases from artifact-backed base columns."""
    if not isinstance(frame, pd.DataFrame) or frame.empty or not needed:
        return frame, 0
    out = frame
    added = 0
    for col in sorted(needed):
        if col in out.columns and not overwrite:
            continue
        src = _model_drift_feature_alias_source(col)
        if src is None:
            continue
        source_candidates = [src]
        if src == "feature_drift_psi_core":
            source_candidates.extend(["feature_drift_psi_core_80", "feature_drift_psi_core_50"])
        elif src == "feature_drift_ks_core":
            source_candidates.extend(["feature_drift_ks_bin_mean", "feature_drift_ks_bin_max"])
        elif src == "row_drift_v1_psi_core":
            source_candidates.extend(["row_drift_v1_psi_core_80", "row_drift_v1_psi_core_50"])
        elif src == "row_drift_v1_ks_core":
            source_candidates.extend(["row_drift_v1_ks_bin_mean", "row_drift_v1_ks_bin_max"])
        if str(col).endswith(src):
            prefix = str(col)[: -len(src)]
            source_candidates.extend(
                f"{prefix}{candidate}"
                for candidate in list(source_candidates)
                if candidate != src
            )
        source_candidates.extend(
            str(existing)
            for existing in out.columns
            if str(existing) != str(col)
            and _model_drift_feature_alias_source(str(existing)) == src
            and str(existing) not in source_candidates
        )
        source_candidates.extend(
            str(existing)
            for existing in out.columns
            if str(existing) != str(col)
            and (
                str(existing) == src
                or str(existing).endswith(f"_{src}")
            )
            and str(existing) not in source_candidates
        )
        for source in source_candidates:
            if source in out.columns:
                out[col] = pd.to_numeric(out[source], errors="coerce").astype(np.float32)
                added += 1
                break
    return out, added


def _iter_model_contract_owners(owner: Any) -> list[Any]:
    """Return a shallow owner list for wrapper/model contract metadata."""
    owners: list[Any] = []
    seen: set[int] = set()

    def _add(candidate: Any) -> None:
        if candidate is None:
            return
        ident = id(candidate)
        if ident in seen:
            return
        seen.add(ident)
        owners.append(candidate)
        if isinstance(candidate, Mapping):
            return
        for attr in ("best_model", "model", "estimator", "clf", "classifier"):
            try:
                nested = getattr(candidate, attr, None)
            except Exception:
                nested = None
            if nested is not candidate:
                _add(nested)

    _add(owner)
    return owners


def _feature_stats_default(owner: Any, feature: str) -> float | None:
    """Return a finite training-stat default for a selected feature."""
    feature_s = str(feature)
    for candidate in _iter_model_contract_owners(owner):
        raw = (
            candidate.get("feature_stats_train")
            if isinstance(candidate, Mapping)
            else getattr(candidate, "feature_stats_train", None)
        )
        if not isinstance(raw, Mapping):
            contract = (
                candidate.get("meta_feature_contract")
                if isinstance(candidate, Mapping)
                else getattr(candidate, "meta_feature_contract_", None)
            )
            if isinstance(contract, Mapping):
                raw = contract.get("feature_stats_train")
        if not isinstance(raw, Mapping):
            continue
        stats = raw.get(feature_s)
        if not isinstance(stats, Mapping):
            continue
        for key in ("p50", "median", "mean"):
            try:
                value = float(stats.get(key))
            except Exception:
                continue
            if np.isfinite(value):
                return value
    return None


def _fill_artifact_context_training_defaults(
    frame: pd.DataFrame,
    features: list[str],
    owner: Any,
    *,
    allowed_sources: set[str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Fill missing artifact-derived context aliases from train stats.

    This is intentionally narrow: live rows can lack artifact transformer state
    for diagnostic drift/context aliases that were persisted as constant or
    otherwise neutral during training. Ordinary source/model inputs still fail
    strict parity when absent.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty or not features:
        return frame, []
    out = frame
    fills: list[dict[str, Any]] = []
    for feature in dict.fromkeys(str(c) for c in features):
        source = _model_drift_feature_alias_source(feature)
        if source not in allowed_sources:
            continue
        value = _feature_stats_default(owner, feature)
        if value is None and source is not None:
            value = _feature_stats_default(owner, source)
        if value is None:
            continue
        if feature not in out.columns:
            if out is frame:
                out = frame.copy()
            out[feature] = np.full(len(out), value, dtype=np.float32)
            fills.append(
                {
                    "feature": feature,
                    "source": source,
                    "default": float(value),
                    "missing_column": True,
                    "filled_rows": int(len(out)),
                }
            )
            continue
        values = pd.to_numeric(out[feature], errors="coerce").to_numpy(dtype=np.float32)
        bad = ~np.isfinite(values)
        if bad.any():
            if out is frame:
                out = frame.copy()
            values[bad] = np.float32(value)
            out[feature] = values.astype(np.float32, copy=False)
            fills.append(
                {
                    "feature": feature,
                    "source": source,
                    "default": float(value),
                    "missing_column": False,
                    "filled_rows": int(bad.sum()),
                }
            )
    return out, fills


def _required_model_drift_sources(needed: set[str]) -> set[str]:
    """Return unprefixed drift/context columns required by direct or aliased keys."""
    sources: set[str] = set()
    for col in needed:
        col_s = str(col)
        if col_s in ALPHA_MODEL_META_FEATURE_KEYS:
            sources.add(col_s)
            if col_s == "feature_drift_psi_core":
                sources.add("feature_drift_psi_core_80")
            if col_s == "feature_drift_ks_core":
                sources.add("feature_drift_ks_bin_mean")
            if col_s == "row_drift_v1_psi_core":
                sources.add("row_drift_v1_psi_core_80")
            if col_s == "row_drift_v1_ks_core":
                sources.add("row_drift_v1_ks_bin_mean")
            continue
        if is_raw_contrib_feature_name(col_s):
            src = _model_drift_feature_alias_source(col_s)
            sources.add(src or col_s)
            continue
        src = _model_drift_feature_alias_source(col_s)
        if src is not None:
            sources.add(src)
            if src == "feature_drift_psi_core":
                sources.add("feature_drift_psi_core_80")
            if src == "feature_drift_ks_core":
                sources.add("feature_drift_ks_bin_mean")
            if src == "row_drift_v1_psi_core":
                sources.add("row_drift_v1_psi_core_80")
            if src == "row_drift_v1_ks_core":
                sources.add("row_drift_v1_ks_bin_mean")
    return sources


def _selected_feature_owner(model: Any) -> Any:
    """Return the nested estimator that owns the real selected feature contract."""
    current = getattr(model, "best_model", model)
    seen: set[int] = set()
    for _ in range(8):
        if current is None:
            return model
        obj_id = id(current)
        if obj_id in seen:
            return current
        seen.add(obj_id)
        selected = getattr(current, "selected_features", None)
        if selected:
            return current
        for attr in ("estimator", "model", "clf", "classifier"):
            child = getattr(current, attr, None)
            if child is not None and child is not current:
                current = child
                break
        else:
            return current
    return current


def _lgbm_internal_metrics_frame(model: Any, X: Any) -> pd.DataFrame:
    """Return detailed LGBM internal metrics for direct or wrapped models."""
    candidates: list[Any] = []
    for candidate in (
        model,
        getattr(model, "best_model", None),
        _selected_feature_owner(model),
    ):
        if candidate is None:
            continue
        if any(id(candidate) == id(existing) for existing in candidates):
            continue
        candidates.append(candidate)
    for candidate in candidates:
        transform = getattr(candidate, "transform_internal_model_metrics", None)
        if not callable(transform):
            transform = getattr(candidate, "transform_meta_features", None)
        if not callable(transform):
            continue
        try:
            frame = transform(X)
        except Exception:
            continue
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            return frame
    return pd.DataFrame()


def _synthetic_ebm_raw_features(model: Any) -> list[str]:
    """Return an EBM f0/f1/... raw contract when the model uses one."""
    ebm_model = _extract_ebm_contract_model(model)
    if ebm_model is None:
        return []
    raw_features = [
        str(c) for c in (getattr(ebm_model, "raw_selected_features", []) or [])
    ]
    if raw_features and all(re.fullmatch(r"f\d+", name) for name in raw_features):
        return raw_features
    return []


def _alpha_prediction_frame_for_model(
    model: Any,
    aligned_features: pd.DataFrame,
    feat_cols: List[str],
    *,
    allow_native_missing: bool = False,
) -> pd.DataFrame:
    """Return the actual prediction frame expected by a persisted alpha model.

    Most alpha model bundles keep ``feat_cols`` as the real feature contract.
    Some older ModelRace/LGBM bundles persisted the inner stability model after
    feature selection with synthetic ``fN`` names, where ``N`` is the position
    in ``feat_cols``. Build that frame explicitly so inference, debug dumps, and
    replay all exercise the same model input contract.
    """
    if aligned_features.empty or not feat_cols:
        return aligned_features

    feat_cols = [str(c) for c in feat_cols]
    X = aligned_features.reindex(columns=feat_cols)
    try:
        if allow_native_missing:
            X = _strict_lgbm_native_missing_model_matrix(
                X,
                model_feature_cols=feat_cols,
                model_key="alpha_feature_contract",
            )
        else:
            X = validate_final_model_matrix(
                X,
                model_feature_cols=feat_cols,
                model_key="alpha_feature_contract",
                strict=True,
            )
    except FeatureParityError:
        raise
    inner = _selected_feature_owner(model)
    selected = [str(c) for c in (getattr(inner, "selected_features", []) or [])]
    input_features = [str(c) for c in (getattr(inner, "input_feature_names", []) or [])]
    has_named_aliases = (
        len(input_features) == len(selected)
        and bool(input_features)
        and input_features != selected
    )
    synthetic_selected = bool(selected) and all(
        re.fullmatch(r"f\d+", name) is not None for name in selected
    )
    if synthetic_selected and not has_named_aliases:
        mapped = pd.DataFrame(index=X.index)
        for name in selected:
            pos = int(name[1:])
            real_name = feat_cols[pos] if pos < len(feat_cols) else ""
            mapped[name] = X[real_name] if real_name in X.columns else np.nan
        if allow_native_missing:
            return _strict_lgbm_native_missing_model_matrix(
                mapped,
                model_feature_cols=selected,
                model_key="alpha_synthetic_feature_contract",
            )
        return validate_final_model_matrix(
            mapped,
            model_feature_cols=selected,
            model_key="alpha_synthetic_feature_contract",
            strict=True,
        )

    synthetic_raw = _synthetic_ebm_raw_features(model)
    if synthetic_raw and len(synthetic_raw) == len(feat_cols):
        X = X.copy()
        X.columns = synthetic_raw
    return X


def _effective_alpha_feature_contract(model_info: Dict[str, Any]) -> List[str]:
    """Return the real raw feature contract consumed by an alpha model.

    Older alpha bundles keep the broad pre-selection feature list in
    ``feat_cols`` while the persisted LGBM stability model stores the actual
    selected feature contract. Strict inference parity must validate the latter
    when it is expressed as real feature names. Synthetic ``fN`` contracts still
    need the broad feature list so they can be mapped by position.
    """
    if not isinstance(model_info, dict):
        return []
    feat_cols = [
        str(c)
        for c in (model_info.get("feat_cols", []) or [])
        if str(c) not in DELETED_MODEL_FEATURE_KEYS
    ]
    model = model_info.get("model")
    inner = _selected_feature_owner(model)
    selected = [str(c) for c in (getattr(inner, "selected_features", []) or [])]
    input_features = [
        str(c) for c in (getattr(inner, "input_feature_names", []) or [])
    ]
    if selected:
        if input_features and len(input_features) == len(selected):
            return [c for c in input_features if c not in DELETED_MODEL_FEATURE_KEYS]
        if all(re.fullmatch(r"f\d+", name) is not None for name in selected):
            return feat_cols
        return [c for c in selected if c not in DELETED_MODEL_FEATURE_KEYS]
    return feat_cols


def _strict_finite_model_matrix(
    X: pd.DataFrame,
    *,
    model_feature_cols: List[str],
    model_key: str,
) -> pd.DataFrame:
    """Return a strict final model matrix; never fill or drop bad model inputs."""
    cols = [str(c) for c in model_feature_cols]
    if X is None or not isinstance(X, pd.DataFrame) or X.empty:
        return validate_final_model_matrix(
            X,
            model_feature_cols=cols,
            model_key=model_key,
            strict=True,
        )
    X = X.reindex(columns=cols)
    try:
        X_float = X.astype(np.float32, copy=False)
    except Exception:
        return validate_final_model_matrix(
            X,
            model_feature_cols=cols,
            model_key=model_key,
            strict=True,
        )
    values = X_float.to_numpy(dtype=np.float32, copy=False)
    if np.isfinite(values).all():
        return validate_final_model_matrix(
            X_float,
            model_feature_cols=cols,
            model_key=model_key,
            strict=True,
        )
    return validate_final_model_matrix(
        X_float,
        model_feature_cols=cols,
        model_key=model_key,
        strict=True,
    )


def _cfg_flag_enabled(
    cfg: Optional[Mapping[str, Any]],
    key: str,
    env_key: str,
    default: bool = False,
) -> bool:
    if isinstance(cfg, Mapping) and key in cfg:
        value = cfg.get(key)
    else:
        value = os.environ.get(env_key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _allow_lgbm_native_missing_model_inputs(
    cfg: Optional[Mapping[str, Any]],
) -> bool:
    return _cfg_flag_enabled(
        cfg,
        "simple_policy_allow_lgbm_native_missing",
        "EPM_SIMPLE_POLICY_ALLOW_LGBM_NATIVE_MISSING",
        False,
    )


def _strict_lgbm_native_missing_model_matrix(
    X: pd.DataFrame,
    *,
    model_feature_cols: List[str],
    model_key: str,
) -> pd.DataFrame:
    """Return a final model matrix that preserves NaNs for LightGBM scoring."""
    cols = [str(c) for c in model_feature_cols]
    report = {
        "model_key": model_key,
        "global_errors": [],
        "native_missing_allowed": True,
    }
    if X is None or not isinstance(X, pd.DataFrame) or X.empty:
        report["global_errors"].append("empty_model_matrix")
    elif list(map(str, X.columns)) != cols:
        report["global_errors"].append("model_matrix_column_order_mismatch")
        report["expected_columns_sample"] = cols[:50]
        report["actual_columns_sample"] = list(map(str, X.columns))[:50]
    if not cols:
        report["global_errors"].append("empty_model_feature_contract")
    if report["global_errors"]:
        raise FeatureParityError("Final model matrix parity failed", report)

    try:
        X_float = X.astype(np.float32, copy=False)
    except Exception as exc:
        report["global_errors"].append(f"model_matrix_float32_cast_failed:{exc}")
        raise FeatureParityError("Final model matrix dtype parity failed", report) from exc

    values = X_float.to_numpy(dtype=np.float32, copy=False)
    if np.isinf(values).any():
        bad_cols = [
            str(col)
            for col in X_float.columns
            if np.isinf(
                X_float[col].to_numpy(dtype=np.float32, copy=False)
            ).any()
        ]
        report["global_errors"].append("model_matrix_nonfinite")
        report["infinite_features"] = bad_cols[:100]
        raise FeatureParityError(
            "Final model matrix contains infinite values", report
        )
    return X_float


def _model_matrix_nonfinite_summary(
    X: pd.DataFrame,
    *,
    limit: int = 20,
) -> Tuple[int, List[Dict[str, Any]]]:
    if X is None or not isinstance(X, pd.DataFrame) or X.empty:
        return 0, []
    try:
        X_float = X.astype(np.float32, copy=False)
        values = X_float.to_numpy(dtype=np.float32, copy=False)
    except Exception:
        return 0, []
    bad = ~np.isfinite(values)
    total = int(bad.sum())
    if total <= 0:
        return 0, []
    counts = bad.sum(axis=0)
    order = np.argsort(-counts)
    sample: List[Dict[str, Any]] = []
    cols = [str(c) for c in X_float.columns]
    for idx in order[: int(limit)]:
        count = int(counts[int(idx)])
        if count <= 0:
            continue
        sample.append({"feature": cols[int(idx)], "nonfinite": count})
    return total, sample


def _training_neutral_filled_model_matrix(
    X: pd.DataFrame,
    *,
    model_feature_cols: List[str],
) -> pd.DataFrame:
    """Match the LGBM training/scoring adapter: Inf and NaN model inputs become 0."""
    cols = [str(c) for c in model_feature_cols]
    X_float = X.reindex(columns=cols).astype(np.float32, copy=False)
    values = X_float.to_numpy(dtype=np.float32, copy=False)
    if np.isfinite(values).all():
        return X_float
    filled = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0, copy=True)
    return pd.DataFrame(filled, index=X_float.index, columns=cols, dtype=np.float32)


def _fill_optional_generated_model_features(
    X: pd.DataFrame,
    *,
    model_feature_cols: List[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Neutral-fill optional generated representation features only.

    AE/GMM/raw-state-SVD columns are downstream context features. Missing or
    non-finite values should not block a live row, but ordinary selected market
    and model-derived features must remain strict.
    """

    optional_cols = [
        str(c)
        for c in model_feature_cols
        if is_optional_generated_model_feature_key(c)
    ]
    if not optional_cols or not isinstance(X, pd.DataFrame):
        return X, [], []
    out = X.copy()
    added: list[str] = []
    repaired: list[str] = []
    for col in optional_cols:
        if col not in out.columns:
            out[col] = np.float32(0.0)
            added.append(col)
            continue
        series = pd.to_numeric(out[col], errors="coerce")
        values = series.to_numpy(dtype=np.float32, copy=False)
        if not np.isfinite(values).all():
            out[col] = np.nan_to_num(
                values,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32, copy=False)
            repaired.append(col)
        elif str(out[col].dtype) != "float32":
            out[col] = values.astype(np.float32, copy=False)
    return out, added, repaired


class ModelOrchestrator:
    """Orchestrates model inference pipeline with proper prediction order."""

    def __init__(
        self,
        model_bundle: Dict[str, Any],
        runtime_cfg: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the model orchestrator.

        Args:
            model_bundle: Loaded model bundle from model_loader
            runtime_cfg: Runtime configuration (optional, contains entry_policy_config, etc.)
        """
        self.cfg = runtime_cfg or {}
        self.full_state = (
            model_bundle
            if isinstance(model_bundle, dict) and "bundle" in model_bundle
            else {}
        )
        loaded_bundle = (
            model_bundle.get("bundle", {})
            if isinstance(model_bundle, dict) and "bundle" in model_bundle
            else (model_bundle or {})
        )
        runtime_bundle = (
            self.cfg.get("model_bundle", {}) if isinstance(self.cfg, dict) else {}
        )
        if (
            isinstance(model_bundle, dict)
            and "bundle" in model_bundle
            and isinstance(runtime_bundle, dict)
            and runtime_bundle
        ):
            self.bundle = dict(loaded_bundle)
            for key, value in runtime_bundle.items():
                if value:
                    self.bundle[key] = value
        else:
            self.bundle = loaded_bundle

        # Extract models from bundle
        self.alpha_models = self.bundle.get("alpha_models", {})
        self.alpha_by_strategy = self._build_alpha_strategy_index()
        self.meta_models = self.bundle.get("meta_models", {})
        self.spike_models = self.bundle.get("spike_models", {})  # GMM models
        self.ridge_weights = self.bundle.get("ridge_weights", {})
        self.bucket_params = (
            self.full_state.get("bucket_params", {})
            if isinstance(self.full_state, dict)
            and self.full_state.get("bucket_params")
            else self.bundle.get("bucket_params", {})
        )
        self.ridge_sizer = (
            self.full_state.get("ridge_sizer")
            if isinstance(self.full_state, dict)
            else None
        )
        self.booster_bundles = (
            self.full_state.get("booster_bundles", {})
            if isinstance(self.full_state, dict)
            else {}
        )
        self.regime_adaptors = (
            self.full_state.get("regime_adaptors", {})
            if isinstance(self.full_state, dict)
            else {}
        )
        self.ridge_params_per_bucket = {}
        if isinstance(self.ridge_weights, dict):
            self.ridge_params_per_bucket = (
                self.ridge_weights.get("params_per_bucket", {}) or {}
            )
            self.ridge_weight_map = self.ridge_weights.get("weights", {}) or {}
        else:
            self.ridge_weight_map = {}
        self._last_results: Dict[str, Any] = {}
        self._last_base_lgbm_diagnostics: Dict[str, float] = {}
        self._last_base_lgbm_diagnostics_by_key: Dict[str, Dict[str, float]] = {}
        self._last_meta_diagnostics: Dict[str, float] = {}
        self._last_meta_diagnostics_frame: pd.DataFrame = pd.DataFrame()
        self._last_mr_tf_route_frames_by_key: Dict[str, pd.DataFrame] = {}

        # Entry policy config from runtime_cfg or bucket_params
        self.entry_policy_config = self.cfg.get(
            "entry_policy_config"
        ) or self.bucket_params.get("entry_policy")

        # Extract feature columns from alpha models
        self.feature_columns = self._extract_feature_columns()

    def _build_alpha_strategy_index(self) -> Dict[str, Dict[str, Any]]:
        """Normalize flat and nested alpha bundle layouts."""
        out: Dict[str, Dict[str, Any]] = {}
        if not isinstance(self.alpha_models, dict):
            return out
        for key, value in self.alpha_models.items():
            if not isinstance(value, dict):
                continue
            if "model" in value or "feat_cols" in value:
                out[str(key)] = value
                continue
            for nested_key, model_info in value.items():
                if isinstance(model_info, dict):
                    out[f"{key}_{nested_key}"] = model_info
        return out

    def _alpha_model_info_for_kind(
        self,
        side: str,
        kind: str,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        key = str(kind)
        model_info = self.alpha_by_strategy.get(key)
        if model_info is not None:
            return key, model_info
        nested_key = f"{side}_{kind}"
        model_info = self.alpha_by_strategy.get(nested_key)
        if model_info is not None:
            return nested_key, model_info
        return key, None

    def available_strategies(
        self, side: str, allowed: Optional[set[str]] = None
    ) -> List[str]:
        """Return loaded strategies for a side after optional selection filtering."""
        side_l = str(side).lower()
        selected: List[str] = []
        for sid in sorted(self.alpha_by_strategy.keys()):
            inferred = strategy_side(sid)
            if inferred and inferred != side_l:
                continue
            if not strategy_id_matches(sid, allowed):
                continue
            selected.append(sid)
        return selected

    def _normalize_bucket_key(self, side: str, kind: str) -> str:
        return f"{str(side).lower()}_{str(kind).lower()}"

    def _policy_bucket_key(self, side: str, kind: str) -> str:
        core = strategy_core_id(str(kind or ""))
        if core and core not in {"mr", "tf", "none"}:
            return core
        return self._normalize_bucket_key(side, kind)

    def _align_alpha_feature_contract(
        self,
        features: pd.DataFrame,
        feat_cols: List[str],
    ) -> pd.DataFrame:
        """Return a contract-aligned feature frame for an alpha model.

        The alpha bundles were trained on a fixed feature contract. At
        inference time we first try to synthesize missing gated features from
        the shared market columns. In strict feature-parity mode we then fail
        closed if any trained column is still absent or non-finite; permissive
        legacy mode keeps the previous zero-fill behavior.
        """
        if features.empty or not feat_cols:
            return features

        aligned = features.copy()

        def _column_has_finite(name: str) -> bool:
            if name not in aligned.columns:
                return False
            values = pd.to_numeric(aligned[name], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            return bool(values.notna().any())

        if "G_VOL" in feat_cols and not _column_has_finite("G_VOL"):
            if {"mkt_rv", "mkt_rv_med"}.issubset(aligned.columns):
                aligned["G_VOL"] = (
                    aligned["mkt_rv"].astype(float)
                    > aligned["mkt_rv_med"].astype(float)
                ).astype(np.float32)

        if "G_TREND" in feat_cols and not _column_has_finite("G_TREND"):
            if {"mkt_ret24h", "mkt_rv"}.issubset(aligned.columns):
                daily_vol = aligned["mkt_rv"].astype(float) * np.sqrt(24.0)
                dyn_thr = np.maximum(daily_vol * 1.5, 0.005)
                aligned["G_TREND"] = (
                    aligned["mkt_ret24h"].astype(float).abs() > dyn_thr
                ).astype(np.float32)

        for gate_name in ("G_VOL", "G_TREND"):
            if gate_name not in aligned.columns:
                continue
            gate_series = aligned[gate_name].astype(np.float32)
            for feat_name in feat_cols:
                if f"_{gate_name}_" not in feat_name:
                    continue
                if feat_name in aligned.columns and _column_has_finite(feat_name):
                    continue
                base_part, state_part = feat_name.rsplit(f"_{gate_name}_", 1)
                if state_part not in {"0", "1"} or base_part not in aligned.columns:
                    continue
                base_vals = aligned[base_part].astype(np.float32)
                if state_part == "1":
                    aligned[feat_name] = (base_vals * gate_series).astype(np.float32)
                else:
                    aligned[feat_name] = (base_vals * (1.0 - gate_series)).astype(
                        np.float32
                    )

        strict = bool(self.cfg.get("strict_feature_parity", True))
        feat_cols_s = [
            str(c) for c in feat_cols if str(c) not in DELETED_MODEL_FEATURE_KEYS
        ]
        try:
            from extreme_price_movements.rule_mask_features import (
                append_rule_mask_features as _append_rule_mask_features,
                is_rule_mask_feature_name as _is_rule_mask_feature_name,
            )

            if any(_is_rule_mask_feature_name(c) for c in feat_cols_s):
                _rule_cfg = dict(self.cfg or {})
                _rule_cfg["lgbm_rule_mask_features_enabled"] = True
                aligned, _rule_diag = _append_rule_mask_features(
                    aligned,
                    _rule_cfg,
                    side=None,
                    context="inference:alpha",
                )
                if bool(self.cfg.get("model_inference_timing_enabled", False)):
                    tprint(
                        "Alpha inference: materialized rule-mask features "
                        f"n_rules={_rule_diag.get('n_rules')} "
                        f"source_available={_rule_diag.get('available_source_keys')}/"
                        f"{_rule_diag.get('n_source_keys')}"
                    )
        except Exception as exc:
            tprint(f"Alpha inference: rule-mask feature materialization failed: {exc}")
        if strict:
            missing = [c for c in feat_cols_s if c not in aligned.columns]
            if missing:
                tprint(
                    "Error aligning alpha feature contract: missing trained "
                    f"features ({len(missing)}): {missing[:20]}"
                )
                return pd.DataFrame(index=features.index)
            allow_native_missing = _allow_lgbm_native_missing_model_inputs(self.cfg)
            try:
                model_matrix = aligned.reindex(columns=feat_cols_s)
                if allow_native_missing:
                    X = _strict_lgbm_native_missing_model_matrix(
                        model_matrix,
                        model_feature_cols=feat_cols_s,
                        model_key="alpha",
                    )
                    if (
                        X.isna().to_numpy(dtype=bool, copy=False).any()
                        and not getattr(
                            self,
                            "_alpha_native_missing_warned",
                            False,
                        )
                    ):
                        tprint(
                            "Alpha inference: preserving NaN trained feature "
                            "values for LightGBM native missing-value handling."
                        )
                        self._alpha_native_missing_warned = True
                    return X
                return _strict_finite_model_matrix(
                    model_matrix,
                    model_feature_cols=feat_cols_s,
                    model_key="alpha",
                )
            except FeatureParityError as exc:
                report = getattr(exc, "report", {}) or {}
                errors = set(report.get("global_errors") or [])
                if "model_matrix_nonfinite" not in errors:
                    tprint(f"Error aligning alpha feature contract: {exc}")
                    return pd.DataFrame(index=features.index)
                if bool(
                    self.cfg.get(
                        "strict_feature_parity_neutral_fill_nonfinite", False
                    )
                ):
                    total_bad, sample = _model_matrix_nonfinite_summary(model_matrix)
                    try:
                        X = _training_neutral_filled_model_matrix(
                            model_matrix,
                            model_feature_cols=feat_cols_s,
                        )
                        if not getattr(
                            self,
                            "_alpha_neutral_fill_nonfinite_warned",
                            False,
                        ):
                            tprint(
                                "Alpha inference: neutral-filled non-finite "
                                "trained feature values with the LGBM "
                                "training/scoring adapter "
                                f"(values={total_bad}, sample={sample[:12]})."
                            )
                            self._alpha_neutral_fill_nonfinite_warned = True
                        return _strict_finite_model_matrix(
                            X,
                            model_feature_cols=feat_cols_s,
                            model_key="alpha",
                        )
                    except Exception as exc2:
                        tprint(
                            "Error aligning alpha feature contract after "
                            f"neutral fill: {exc2}"
                        )
                        return pd.DataFrame(index=features.index)
                matrix_float = model_matrix.astype(np.float32, copy=False)
                values = matrix_float.to_numpy(dtype=np.float32, copy=False)
                if allow_native_missing:
                    row_ok = ~np.isinf(values).any(axis=1)
                else:
                    row_ok = np.isfinite(values).all(axis=1)
                valid_rows = int(row_ok.sum())
                if valid_rows <= 0:
                    tprint(f"Error aligning alpha feature contract: {exc}")
                    return pd.DataFrame(index=features.index)
                dropped_rows = int(len(row_ok) - valid_rows)
                tprint(
                    "Alpha inference: dropped "
                    f"{dropped_rows}/{len(row_ok)} rows with non-finite trained "
                    f"features; predicting {valid_rows} strict rows."
                )
                try:
                    if allow_native_missing:
                        return _strict_lgbm_native_missing_model_matrix(
                            matrix_float.loc[row_ok],
                            model_feature_cols=feat_cols_s,
                            model_key="alpha",
                        )
                    return _strict_finite_model_matrix(
                        matrix_float.loc[row_ok],
                        model_feature_cols=feat_cols_s,
                        model_key="alpha",
                    )
                except FeatureParityError as exc2:
                    tprint(f"Error aligning alpha feature contract: {exc2}")
                    return pd.DataFrame(index=features.index)

        return aligned.reindex(columns=feat_cols_s, fill_value=0.0).fillna(0.0)

    def _get_bucket_policy(self, side: str, kind: str) -> Dict[str, Any]:
        bucket_key = self._policy_bucket_key(side, kind)
        bucket_cfg = {}
        if isinstance(self.ridge_params_per_bucket, dict):
            bucket_cfg = self.ridge_params_per_bucket.get(bucket_key, {}) or {}
        if not bucket_cfg and isinstance(self.bucket_params, dict):
            buckets = (
                self.bucket_params.get("buckets", {})
                if "buckets" in self.bucket_params
                else {}
            )
            bucket_cfg = (
                buckets.get(bucket_key.upper(), {})
                or buckets.get(bucket_key, {})
                or self.bucket_params.get(bucket_key, {})
                or {}
            )
        if isinstance(bucket_cfg, dict):
            return bucket_cfg
        return {}

    def _materialize_symbol_features(
        self,
        symbol: str,
        features: Any,
    ) -> pd.DataFrame:
        """Return a single-row feature frame for a symbol from either a DataFrame or feature dict."""
        if isinstance(features, pd.Series):
            features = features.to_frame().T
        if isinstance(features, pd.DataFrame):
            if symbol in features.index:
                return features.loc[[symbol]].copy()
            return features.copy()
        if isinstance(features, dict):
            df = get_features_for_candidates(features, [symbol])
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        return pd.DataFrame(index=[symbol])

    def _latest_panel_price(self, symbol: str, panel: Any) -> float:
        if not isinstance(panel, dict):
            return 1.0
        close = panel.get("close")
        if not isinstance(close, pd.DataFrame) or symbol not in close.columns:
            return 1.0
        series = close[symbol].dropna()
        if series.empty:
            return 1.0
        price = float(series.iloc[-1])
        return price if np.isfinite(price) and price > 0.0 else 1.0

    def _latest_atr_frac(self, features: pd.DataFrame) -> float:
        for col in ("atr_pct", "atr_pct_base", "realized_volatility_24h"):
            if col not in features.columns:
                continue
            val = float(features[col].iloc[0])
            if np.isfinite(val) and val > 0.0:
                return val
        return 0.01

    def _extract_feature_columns(self) -> Dict[str, List[str]]:
        """Extract feature column names from all loaded alpha models.

        Returns:
            Dictionary mapping ``"{side}_{kind}"`` to feature columns.
        """
        columns = {}

        if self.alpha_by_strategy:
            for sid, model_info in self.alpha_by_strategy.items():
                if isinstance(model_info, dict):
                    columns[sid] = _effective_alpha_feature_contract(model_info)
            return columns

        for side in ["long", "short"]:
            if side not in self.alpha_models:
                continue

            side_models = self.alpha_models.get(side, {})
            if not isinstance(side_models, dict):
                continue

            for kind, model_info in side_models.items():
                if not isinstance(model_info, dict):
                    continue
                feat_cols = _effective_alpha_feature_contract(model_info)
                columns[f"{side}_{kind}"] = feat_cols

        return columns

    # =========================================================================
    # STEP 1: GMM Spike Quality Filter
    # =========================================================================

    def predict_alpha(
        self,
        features: pd.DataFrame,
        side: str,
        kind: str = "mr",
    ) -> pd.Series:
        """Run alpha model prediction (Step 2: Base/Alpha model predictions).

        Args:
            features: Feature DataFrame (symbols as index)
            side: "long" or "short"
            kind: "mr" (mean reversion) or "tf" (trend following)

        Returns:
            Series of predictions indexed by symbol
        """
        self._last_results.pop("mr_tf_alpha_routing", None)
        key = str(kind)
        model_info = self.alpha_by_strategy.get(key)
        if model_info is None:
            nested_key = f"{side}_{kind}"
            model_info = self.alpha_by_strategy.get(nested_key)
            key = nested_key if model_info is not None else key
        self._last_mr_tf_route_frames_by_key.pop(str(key), None)
        if model_info is None:
            tprint(f"Warning: Alpha model not found for {key}")
            return pd.Series(dtype=float)

        model = model_info.get("model")
        feat_cols = _effective_alpha_feature_contract(model_info)

        if model is None:
            tprint(f"Warning: Model not loaded for {key}")
            return pd.Series(dtype=float)

        self._last_base_lgbm_diagnostics = {}
        self._last_base_lgbm_diagnostics_by_key.pop(str(key), None)
        timing_enabled = bool(self.cfg.get("inference_model_timing_enabled", True))
        t0 = time.perf_counter()
        aligned_features = self._align_alpha_feature_contract(features, feat_cols)
        if timing_enabled:
            tprint(
                "[Timing] model.alpha_align: "
                f"key={key} rows_in={len(features.index)} "
                f"rows_out={len(aligned_features.index)} features={len(feat_cols)} "
                f"stage={time.perf_counter() - t0:.3f}s {_process_rss_log_fields()}"
            )

        if aligned_features.empty:
            tprint(f"Warning: No matching features for {key}")
            return pd.Series(dtype=float)

        # Get feature matrix
        matrix_t0 = time.perf_counter()
        X = _alpha_prediction_frame_for_model(
            model,
            aligned_features,
            feat_cols,
            allow_native_missing=_allow_lgbm_native_missing_model_inputs(self.cfg),
        )
        if timing_enabled:
            tprint(
                "[Timing] model.alpha_matrix: "
                f"key={key} shape={getattr(X, 'shape', None)} "
                f"stage={time.perf_counter() - matrix_t0:.3f}s "
                f"{_process_rss_log_fields()}"
            )

        # Predict
        try:
            pred_t0 = time.perf_counter()
            preds = model.predict(X)
            if bool(self.cfg.get("inference_lgbm_internal_diagnostics_enabled", True)):
                diag_t0 = time.perf_counter()
                base_diag_frame = _lgbm_internal_metrics_frame(model, X)
                base_diag = _first_row_diagnostics(base_diag_frame)
                if base_diag:
                    self._last_base_lgbm_diagnostics = dict(base_diag)
                    self._last_base_lgbm_diagnostics_by_key[str(key)] = dict(base_diag)
                    if timing_enabled:
                        tprint(
                            "[Timing] model.alpha_diagnostics: "
                            f"key={key} fields={len(base_diag)} "
                            f"stage={time.perf_counter() - diag_t0:.3f}s "
                            f"{_process_rss_log_fields()}"
                        )
            if timing_enabled:
                tprint(
                    "[Timing] model.alpha_predict: "
                    f"key={key} rows={len(aligned_features.index)} "
                    f"stage={time.perf_counter() - pred_t0:.3f}s "
                    f"total={time.perf_counter() - t0:.3f}s "
                    f"{_process_rss_log_fields()}"
                )
            out = pd.Series(preds, index=aligned_features.index)
            specialists = model_info.get("mr_tf_specialists")
            if isinstance(specialists, dict):
                try:
                    from extreme_price_movements.mr_tf_masks import (
                        apply_mr_tf_masks as _apply_mr_tf_masks,
                        mr_tf_masks_enabled as _mr_tf_masks_enabled,
                    )

                    if bool(_mr_tf_masks_enabled(self.cfg)):
                        mask_diag = dict(
                            specialists.get("mask_diagnostics", {}) or {}
                        )
                        mask_params = mask_diag.get("params")
                        route_source = features
                        if isinstance(route_source, pd.DataFrame):
                            route_source = route_source.reindex(out.index)
                        route_frame, route_diag = _apply_mr_tf_masks(
                            route_source,
                            side=side,
                            cfg=self.cfg,
                            params=mask_params,
                        )
                        self._last_mr_tf_route_frames_by_key[str(key)] = route_frame[
                            [
                                c
                                for c in (
                                    "__mr_tf_route__",
                                    "__mr_mask__",
                                    "__tf_mask__",
                                    "__mixed_mask__",
                                    "__mr_tf_params_hash__",
                                )
                                if c in route_frame.columns
                            ]
                        ].copy()
                        route_counts = (
                            route_frame["__mr_tf_route__"].astype(str).value_counts()
                            if "__mr_tf_route__" in route_frame.columns
                            else pd.Series(dtype=int)
                        )
                        route_overrides = 0
                        for route_name in ("mr", "tf"):
                            route_info = (
                                (specialists.get("routes") or {}).get(route_name)
                                or {}
                            )
                            if not bool(
                                route_info.get(
                                    "enabled", route_info.get("promoted", False)
                                )
                            ):
                                continue
                            route_model = route_info.get("model")
                            if route_model is None:
                                continue
                            route_mask_col = f"__{route_name}_mask__"
                            if route_mask_col not in route_frame.columns:
                                continue
                            route_idx = route_frame.index[
                                np.asarray(route_frame[route_mask_col].values, dtype=bool)
                            ]
                            if len(route_idx) <= 0:
                                continue
                            route_feat_cols = [
                                str(c)
                                for c in (
                                    route_info.get("feat_cols")
                                    or route_info.get("selected_features")
                                    or _effective_alpha_feature_contract(route_info)
                                )
                            ]
                            if not route_feat_cols:
                                route_feat_cols = _effective_alpha_feature_contract(
                                    model_info
                                )
                            route_features = route_source.reindex(route_idx)
                            route_aligned = self._align_alpha_feature_contract(
                                route_features,
                                route_feat_cols,
                            )
                            if route_aligned.empty:
                                continue
                            route_X = _alpha_prediction_frame_for_model(
                                route_model,
                                route_aligned,
                                route_feat_cols,
                                allow_native_missing=(
                                    _allow_lgbm_native_missing_model_inputs(self.cfg)
                                ),
                            )
                            route_pred = route_model.predict(route_X)
                            out.loc[route_aligned.index] = route_pred
                            route_overrides += int(len(route_aligned))
                        self._last_results["mr_tf_alpha_routing"] = {
                            "key": key,
                            "params_hash": route_diag.get("params_hash", ""),
                            "counts": {
                                str(k): int(v) for k, v in route_counts.items()
                            },
                            "specialist_overrides": int(route_overrides),
                        }
                except Exception as _route_exc:
                    tprint(
                        f"Warning: MR/TF alpha specialist routing skipped for {key}: "
                        f"{_route_exc}"
                    )
            return out
        except Exception as e:
            tprint(f"Error predicting alpha for {key}: {e}")
            return pd.Series(dtype=float)

    def predict_alpha_all_horizons(
        self,
        features: pd.DataFrame,
        side: str,
    ) -> Dict[str, pd.Series]:
        """Run all loaded alpha model predictions for a side.

        Args:
            features: Feature DataFrame
            side: "long" or "short"

        Returns:
            Dictionary with predictions for each strategy kind (e.g. long_compression_ratio, ...).
        """
        results = {}

        for kind in self.available_strategies(side):
            preds = self.predict_alpha(features, side, kind)
            # Defensive check: ensure preds is a DataFrame/Series
            if isinstance(preds, pd.DataFrame):
                if not preds.empty:
                    results[kind] = preds
            elif isinstance(preds, pd.Series):
                if not preds.empty:
                    results[kind] = preds
            else:
                # Skip if preds is not a DataFrame or Series
                continue

        return results

    def predict_alpha_all_kinds(
        self,
        features: pd.DataFrame,
        side: str,
    ) -> Dict[str, pd.Series]:
        """Compatibility alias for code that expects a generic strategy fanout."""
        return self.predict_alpha_all_horizons(features, side)

    def get_last_results(self) -> Dict[str, Any]:
        return dict(self._last_results) if isinstance(self._last_results, dict) else {}

    # =========================================================================
    # STEP 3: Compute Disagreement Features
    # =========================================================================

    def compute_disagreement_features(
        self,
        meta_data: pd.DataFrame,
        mr_preds: pd.Series,
        tf_preds: pd.Series,
        kind_name: str = "mr",
    ) -> pd.Series:
        """Step 3: Compute disagreement features between MR and TF predictions.

        Replicates the _calculate_disagreement_features from engine.py.

        Args:
            meta_data: Meta features DataFrame
            mr_preds: Mean reversion predictions
            tf_preds: Trend following predictions
            kind_name: Name for logging

        Returns:
            Series of disagreement features
        """
        try:
            return _calculate_disagreement_features(
                meta_data=meta_data,
                h_preds={"mr": mr_preds, "tf": tf_preds},
                kind_name=kind_name,
            )
        except Exception as e:
            tprint(f"Error computing disagreement features: {e}")
            return pd.Series(0.0, index=meta_data.index)

    def _materialize_meta_model_derived_features(
        self,
        features: pd.DataFrame,
        meta_model: Any,
        *,
        side: str,
        kind: str,
    ) -> pd.DataFrame:
        """Build deterministic live values for causal model-derived meta keys.

        Raw market features must already be present in ``features``. This helper
        only materializes columns derived from the current base prediction itself.
        In strict parity mode, train-time performance diagnostics that require
        future labels or unavailable rank context are intentionally left missing
        so meta prediction fails closed instead of receiving neutral constants.
        """
        if not isinstance(features, pd.DataFrame) or features.empty:
            return features
        effective_cols = _effective_selected_feature_contract(meta_model)
        if effective_cols:
            feat_cols = effective_cols
        else:
            feat_cols = [str(c) for c in (getattr(meta_model, "feature_columns", []) or [])]
        if not feat_cols:
            return features

        out = features.copy()
        strict_parity = bool(self.cfg.get("strict_feature_parity", False))
        history_defaults = extract_model_effectiveness_history_defaults(meta_model)
        history_default_used: list[str] = []
        kind_s = str(kind)
        core = strategy_core_id(kind_s)
        core_no_head = re.sub(r"_(?:clf|reg|tbm_clf|early_inval)$", "", core)
        kind_no_head = re.sub(r"_(?:clf|reg|tbm_clf|early_inval)$", "", kind_s)
        base_series: pd.Series | None = None
        candidate_cols = [
            kind_s,
            kind_no_head,
            core,
            core_no_head,
            f"{side}_{core}",
            f"{side}_{core_no_head}",
            (
                getattr(meta_model, "meta_feature_contract_", {}).get(
                    "base_probability_column", ""
                )
                if isinstance(getattr(meta_model, "meta_feature_contract_", {}), dict)
                else ""
            ),
        ]
        candidate_cols.extend([c for c in feat_cols if re.match(r"^pred_.*_H\d+$", c)])
        candidate_cols.extend([c for c in feat_cols if re.match(r"^pred_H\d+$", c)])
        for col in candidate_cols:
            if col and col in out.columns:
                base_series = pd.to_numeric(out[col], errors="coerce").astype(float)
                break
        if base_series is None:
            return out

        base_prob = base_series.clip(1e-6, 1.0 - 1e-6).astype(float)
        base_logit = np.log(base_prob / (1.0 - base_prob))
        base_entropy = -(
            base_prob * np.log(base_prob)
            + (1.0 - base_prob) * np.log(1.0 - base_prob)
        )

        def _first_existing_col(names: list[str]) -> str | None:
            for name in names:
                if name in out.columns:
                    return name
            return None

        def _numeric_col(name: str) -> pd.Series:
            return pd.to_numeric(out[name], errors="coerce").astype(float)

        def _historical_default_series(name: str) -> pd.Series | None:
            if name not in history_defaults:
                return None
            value = float(history_defaults[name])
            if not np.isfinite(value):
                return None
            history_default_used.append(name)
            return pd.Series(value, index=out.index, dtype=np.float32)

        def _symbol_values() -> np.ndarray | None:
            for name in ("__symbol__", "symbol"):
                if name in out.columns:
                    return out[name].astype(str).to_numpy()
            if isinstance(out.index, pd.MultiIndex):
                for name in ("__symbol__", "symbol", "asset"):
                    if name in (out.index.names or []):
                        return out.index.get_level_values(name).astype(str).to_numpy()
            if out.index.name in {"__symbol__", "symbol", "asset"}:
                return out.index.astype(str).to_numpy()
            return None

        def _timestamp_values() -> pd.Series | None:
            for name in ("__ts__", "timestamp"):
                if name in out.columns:
                    ts = pd.to_datetime(out[name], errors="coerce", utc=True)
                    return pd.Series(ts, index=out.index)
            if isinstance(out.index, pd.MultiIndex):
                for name in ("__ts__", "timestamp"):
                    if name in (out.index.names or []):
                        ts = pd.to_datetime(
                            out.index.get_level_values(name), errors="coerce", utc=True
                        )
                        return pd.Series(ts, index=out.index)
            if out.index.name in {"__ts__", "timestamp"}:
                ts = pd.to_datetime(out.index, errors="coerce", utc=True)
                return pd.Series(ts, index=out.index)
            return None

        side_sign = 1.0 if str(side).lower() == "long" else -1.0
        trend24_col = _first_existing_col(["trend_slope_24h", "trend_t", "trend_pct"])
        trend72_col = _first_existing_col(["trend_slope_72h", "trend_slope_48h"])
        vol24_col = _first_existing_col(["vol_z24", "vol_z_4h", "vol_z"])
        vol96_col = _first_existing_col(["volatility_zscore", "vol_z"])
        trend_src = trend72_col or trend24_col
        vol_src = vol96_col or vol24_col
        eff_col = _first_existing_col(["efficiency_ratio_20", "path_efficiency_24"])
        comp_col = _first_existing_col(["compression_score"])

        added = 0
        for col in feat_cols:
            if col in out.columns:
                continue
            value: pd.Series | float | None = None
            if re.match(r"^pred_logit(?:_H\d+)?$", col):
                value = base_logit
            elif re.match(
                r"^pred(?:_.*)?_H\d+(?:_ebm_raw|_ebm_en|_ebm_uncertainty_weighted)?$",
                col,
            ):
                value = base_prob
            elif re.match(r"^base_H\d+_ebm_(?:raw|en|uncertainty_weighted)$", col):
                value = base_prob
            elif col in {"base_model_score", "base_med_pred"}:
                value = base_prob
            elif col == "base_model_margin":
                value = (base_prob - 0.5).abs()
            elif re.match(r"^(?:pred(?:_.*)?_H\d+|base_H\d+)_(?:vote_margin|vote_top_gap)$", col):
                value = (2.0 * (base_prob - 0.5).abs()).astype(np.float32)
            elif col == "base_model_score_pct":
                symbols = _symbol_values()
                timestamps = _timestamp_values()
                if symbols is not None and timestamps is not None:
                    window = int(self.cfg.get("meta_trade_rank_window", 240))
                    rank_pct = rolling_asset_percentile(
                        base_prob.to_numpy(dtype=np.float32),
                        symbols,
                        timestamps,
                        window=window,
                    )
                    value = pd.Series(rank_pct, index=out.index)
                elif strict_parity:
                    value = None
                else:
                    value = _historical_default_series(col)
                    if value is None:
                        value = pd.Series(0.5, index=out.index, dtype=np.float32)
            elif col in {
                "prob_error",
                "recent_prob_error_20",
                "base_model_abs_error_roll20",
            }:
                value = _historical_default_series(col)
                if value is None and not strict_parity:
                    value = pd.Series(0.5, index=out.index, dtype=np.float32)
            elif col.startswith("recent_hit_rate_"):
                value = _historical_default_series(col)
                if value is None and not strict_parity:
                    neutral = 0.5 if col == "recent_hit_rate_20" else 0.0
                    value = pd.Series(neutral, index=out.index, dtype=np.float32)
            elif (
                col.startswith("recent_global_")
                or col.startswith("recent_side_horizon_")
                or col.startswith("recent_bucket_")
                or col.startswith("recent_regime_")
                or col.startswith("recent_meta_")
                or col.startswith("recent_base_meta_disagreement_")
                or col.startswith("recent_base_internal_disagreement_")
            ):
                value = _historical_default_series(col)
                if value is None and not strict_parity:
                    value = pd.Series(0.0, index=out.index, dtype=np.float32)
            elif col == "rsi_z_x_regime_vol":
                if {"rsi_z", "regime_vol_score"}.issubset(out.columns):
                    value = _numeric_col("rsi_z") * _numeric_col("regime_vol_score")
            elif col == "base_med_x_side_aligned_trend":
                if trend_src is not None:
                    value = base_prob * side_sign * _numeric_col(trend_src)
            elif col == "base_med_x_vol_z":
                if vol_src is not None:
                    value = base_prob * _numeric_col(vol_src)
            elif col == "base_med_x_efficiency_ratio":
                if eff_col is not None:
                    value = base_prob * _numeric_col(eff_col)
            elif col == "base_med_x_compression_score":
                if comp_col is not None:
                    value = base_prob * _numeric_col(comp_col)
            elif col == "base_med_x_compression_x_vol_z":
                if comp_col is not None and vol_src is not None:
                    value = base_prob * _numeric_col(comp_col) * _numeric_col(vol_src)
            elif col == "base_med_x_side_trend_x_vol_z":
                if trend_src is not None and vol_src is not None:
                    value = (
                        base_prob
                        * side_sign
                        * _numeric_col(trend_src)
                        * _numeric_col(vol_src)
                    )
            elif col == "base_med_x_side_trend_x_efficiency":
                if trend_src is not None and eff_col is not None:
                    value = (
                        base_prob
                        * side_sign
                        * _numeric_col(trend_src)
                        * _numeric_col(eff_col)
                    )
            elif col == "base_med_x_trend_24h_x_trend_72h":
                if trend24_col is not None and trend72_col is not None:
                    value = base_prob * _numeric_col(trend24_col) * _numeric_col(
                        trend72_col
                    )
            elif col == "base_med_x_vol_z_24h_minus_96h":
                if vol24_col is not None and vol96_col is not None:
                    value = base_prob * (_numeric_col(vol24_col) - _numeric_col(vol96_col))
            elif col == "base_prob_x_vol_regime":
                src = _first_existing_col(["regime_vol_score", "asset_vol_level"])
                if src is not None:
                    value = base_prob * _numeric_col(src)
            elif col == "base_prob_x_entropy":
                src = _first_existing_col(["regime_transition_entropy_12h"])
                if src is not None:
                    value = base_prob * _numeric_col(src)
            elif re.match(r"^(?:pred(?:_.*)?_H\d+|base_H\d+)_vote_entropy$", col):
                horizon_match = re.search(r"_H(\d+)_vote_entropy$", col)
                h = horizon_match.group(1) if horizon_match else ""
                src = _first_existing_col(
                    [
                        f"pred_H{h}_vote_entropy",
                        f"base_H{h}_vote_entropy",
                        "oof_tree_vote_entropy",
                        "vote_entropy",
                    ]
                )
                if src is not None:
                    value = _numeric_col(src)
                else:
                    value = base_entropy
            elif col.startswith("base_prob_x_"):
                src = col.removeprefix("base_prob_x_")
                if src in out.columns:
                    value = base_prob * _numeric_col(src)
            elif col.startswith("base_med_x_"):
                src = col.removeprefix("base_med_x_")
                if src in out.columns:
                    value = base_prob * _numeric_col(src)

            if value is not None:
                out[col] = value
                added += 1
        out, alias_added = _materialize_model_drift_feature_aliases(out, set(feat_cols))
        added += alias_added
        if history_defaults:
            out, default_added, default_filled = apply_model_effectiveness_history_defaults(
                out,
                feat_cols,
                history_defaults,
            )
            history_default_used.extend(default_added)
            history_default_used.extend(default_filled)
            added += len(default_added) + len(default_filled)
        if history_default_used:
            self._last_results["meta_historical_effectiveness_defaults"] = {
                "kind": str(kind),
                "count": int(len(set(history_default_used))),
                "features": sorted(set(history_default_used))[:50],
                "source": "train_artifact_model_effectiveness_history_defaults",
            }
        if added and not getattr(self, "_meta_model_derived_warned", False):
            tprint(
                "Meta inference: materialized model-derived contract columns "
                f"from base prediction ({added} columns for {kind})."
            )
            self._meta_model_derived_warned = True
        return out

    def _materialize_alpha_model_meta_features(
        self,
        features: pd.DataFrame,
        meta_model: Any,
        *,
        side: str,
        kind: str,
    ) -> pd.DataFrame:
        """Attach alpha-model meta diagnostics needed by train/live parity."""
        if not isinstance(features, pd.DataFrame) or features.empty:
            return features

        feat_cols = [str(c) for c in (getattr(meta_model, "feature_columns", []) or [])]
        effective_cols = _effective_selected_feature_contract(meta_model)
        if effective_cols:
            feat_cols = effective_cols
        needed = set(feat_cols)
        needed_sources = _required_model_drift_sources(needed)
        if not needed_sources.intersection(ALPHA_MODEL_META_FEATURE_KEYS) and not any(
            is_raw_contrib_feature_name(str(src)) for src in needed_sources
        ):
            return features

        _, model_info = self._alpha_model_info_for_kind(side, kind)
        if not isinstance(model_info, dict):
            return features
        alpha_model = model_info.get("model")
        if alpha_model is None:
            return features

        candidate_owners: list[Any] = []
        seen_owner_ids: set[int] = set()

        def _collect_owner(candidate: Any) -> None:
            if candidate is None:
                return
            owner_id = id(candidate)
            if owner_id in seen_owner_ids:
                return
            seen_owner_ids.add(owner_id)
            candidate_owners.append(candidate)
            for attr in (
                "best_model",
                "estimator",
                "model",
                "base_model",
                "wrapped_model",
            ):
                try:
                    nested = getattr(candidate, attr, None)
                except Exception:
                    nested = None
                if nested is not candidate:
                    _collect_owner(nested)

        _collect_owner(alpha_model)
        transform_owners = [
            owner
            for owner in candidate_owners
            if callable(getattr(owner, "transform_meta_features", None))
        ]
        if not transform_owners:
            return features

        contexts: list[pd.DataFrame] = []

        def _append_context(frame: pd.DataFrame) -> None:
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                if len(frame) == len(features) and not frame.index.equals(features.index):
                    frame = frame.copy()
                    frame.index = features.index
                contexts.append(frame)

        for transform_owner in transform_owners:
            try:
                _append_context(transform_owner.transform_meta_features(features))
            except Exception:
                continue

        missing_sources_after_raw = {
            str(src)
            for src in needed_sources
            if (
                str(src) in ALPHA_MODEL_META_FEATURE_KEYS
                or is_raw_contrib_feature_name(str(src))
            )
            and not any(
                str(src) in map(str, ctx.columns)
                for ctx in contexts
                if isinstance(ctx, pd.DataFrame)
            )
        }
        if missing_sources_after_raw:
            try:
                alpha_feat_cols = _effective_alpha_feature_contract(model_info)
                aligned = self._align_alpha_feature_contract(features, alpha_feat_cols)
                if not aligned.empty:
                    alpha_frame = _alpha_prediction_frame_for_model(
                        alpha_model,
                        aligned,
                        alpha_feat_cols,
                        allow_native_missing=(
                            _allow_lgbm_native_missing_model_inputs(self.cfg)
                        ),
                    )
                    for transform_owner in transform_owners:
                        try:
                            _append_context(transform_owner.transform_meta_features(alpha_frame))
                        except Exception:
                            continue
            except Exception:
                pass

        if not contexts:
            last_exc: Exception | None = None
            try:
                alpha_feat_cols = _effective_alpha_feature_contract(model_info)
                aligned = self._align_alpha_feature_contract(features, alpha_feat_cols)
                if aligned.empty:
                    return features
                alpha_frame = _alpha_prediction_frame_for_model(
                    alpha_model,
                    aligned,
                    alpha_feat_cols,
                    allow_native_missing=(
                        _allow_lgbm_native_missing_model_inputs(self.cfg)
                    ),
                )
                for transform_owner in transform_owners:
                    try:
                        _append_context(transform_owner.transform_meta_features(alpha_frame))
                    except Exception as exc:
                        last_exc = exc
            except Exception as exc:
                last_exc = exc
            if not contexts:
                if not getattr(self, "_alpha_meta_context_warned", False):
                    tprint(
                        "Meta inference: failed to materialize alpha model meta "
                        f"context for {kind}: {last_exc}"
                    )
                    self._alpha_meta_context_warned = True
                return features

        out = features.copy()
        meta_context = pd.concat(
            [ctx.reindex(out.index) for ctx in contexts],
            axis=1,
            copy=False,
        )
        meta_context = meta_context.loc[
            :, ~meta_context.columns.astype(str).duplicated(keep="last")
        ]
        raw_state_sources = {
            str(src)
            for src in needed_sources
            if str(src) in set(RAW_STATE_SVD_FEATURE_NAMES)
            or str(src) in set(RAW_STATE_DIAGNOSTIC_FEATURE_NAMES)
        }
        meta_context_cols = set(map(str, meta_context.columns))
        if raw_state_sources and not raw_state_sources.issubset(meta_context_cols):
            raw_state = None
            for raw_state_owner in candidate_owners:
                raw_state = getattr(raw_state_owner, "raw_state_archetype_state", None)
                if raw_state is not None:
                    break
            if raw_state is not None:
                try:
                    raw_context = transform_raw_state_archetype_features(
                        features,
                        raw_state,
                        index=out.index,
                    )
                    if isinstance(raw_context, pd.DataFrame) and not raw_context.empty:
                        meta_context = pd.concat(
                            [meta_context, raw_context.reindex(out.index)],
                            axis=1,
                            copy=False,
                        )
                        meta_context = meta_context.loc[
                            :,
                            ~meta_context.columns.astype(str).duplicated(
                                keep="last"
                            ),
                        ]
                except Exception as exc:
                    if not getattr(self, "_alpha_raw_state_context_warned", False):
                        tprint(
                            "Meta inference: failed to materialize raw-state "
                        f"context for {kind}: {exc}"
                    )
                    self._alpha_raw_state_context_warned = True
        base_error_sources = {
            str(src)
            for src in needed_sources
            if str(src) in set(BASE_ERROR_ARCHETYPE_FEATURE_NAMES)
        }
        if base_error_sources:
            base_error_state = None
            for owner in (
                meta_model,
                getattr(meta_model, "best_model", None),
                getattr(meta_model, "estimator", None),
            ):
                if owner is None:
                    continue
                base_error_state = getattr(owner, "base_error_archetype_state_", None)
                if base_error_state is not None:
                    break
            if base_error_state is not None:
                try:
                    signature_context = pd.concat(
                        [out, meta_context.reindex(out.index)],
                        axis=1,
                        copy=False,
                    )
                    signature_context = signature_context.loc[
                        :,
                        ~signature_context.columns.astype(str).duplicated(
                            keep="last"
                        ),
                    ]
                    base_error_context = transform_residual_error_archetype_features(
                        signature_context,
                        base_error_state,
                        index=out.index,
                    )
                    if (
                        isinstance(base_error_context, pd.DataFrame)
                        and not base_error_context.empty
                    ):
                        meta_context = pd.concat(
                            [meta_context, base_error_context.reindex(out.index)],
                            axis=1,
                            copy=False,
                        )
                        meta_context = meta_context.loc[
                            :,
                            ~meta_context.columns.astype(str).duplicated(
                                keep="last"
                            ),
                        ]
                except Exception as exc:
                    if not getattr(self, "_alpha_base_error_context_warned", False):
                        tprint(
                            "Meta inference: failed to materialize base-error "
                            f"archetype context for {kind}: {exc}"
                        )
                        self._alpha_base_error_context_warned = True
            elif not getattr(self, "_alpha_base_error_context_missing_warned", False):
                tprint(
                    "Meta inference: base-error archetype features are required "
                    f"for {kind}, but no base_error_archetype_state_ is attached "
                    "to the meta model artifact."
                )
                self._alpha_base_error_context_missing_warned = True
        added = 0
        raw_contrib_sources = {
            str(src)
            for src in needed_sources
            if is_raw_contrib_feature_name(str(src))
        }
        for col in ALPHA_MODEL_META_FEATURE_KEYS:
            if col not in needed_sources or col not in meta_context.columns:
                continue
            if col in needed:
                out[col] = pd.to_numeric(meta_context[col], errors="coerce").astype(np.float32)
                added += 1
            for alias in sorted(needed):
                if _model_drift_feature_alias_source(alias) != col:
                    continue
                out[str(alias)] = pd.to_numeric(
                    meta_context[col], errors="coerce"
                ).astype(np.float32)
                added += 1
        for col in sorted(raw_contrib_sources):
            if col not in meta_context.columns:
                continue
            source = pd.to_numeric(meta_context[col], errors="coerce").astype(np.float32)
            if col in needed:
                out[col] = source
                added += 1
            for alias in sorted(needed):
                if _model_drift_feature_alias_source(alias) != col:
                    continue
                out[str(alias)] = source
                added += 1
        if (
            "feature_drift_psi_core" in needed_sources
            and "feature_drift_psi_core_80" in meta_context.columns
            and "feature_drift_psi_core" not in meta_context.columns
        ):
            source = pd.to_numeric(
                meta_context["feature_drift_psi_core_80"], errors="coerce"
            ).astype(np.float32)
            if "feature_drift_psi_core" in needed:
                out["feature_drift_psi_core"] = source
                added += 1
            for alias in sorted(needed):
                if _model_drift_feature_alias_source(alias) != "feature_drift_psi_core":
                    continue
                out[str(alias)] = source
                added += 1
        elif (
            "feature_drift_psi_core" in needed_sources
            and "feature_drift_psi_core" in meta_context.columns
        ):
            source = pd.to_numeric(
                meta_context["feature_drift_psi_core"], errors="coerce"
            ).astype(np.float32)
            for alias in sorted(needed):
                if (
                    alias in out.columns
                    or _model_drift_feature_alias_source(alias)
                    != "feature_drift_psi_core"
                ):
                    continue
                out[str(alias)] = source
                added += 1
        if (
            "feature_drift_psi_core" in needed
            and "feature_drift_psi_core" not in out.columns
            and "feature_drift_psi_core_80" in meta_context.columns
        ):
            out["feature_drift_psi_core"] = pd.to_numeric(
                meta_context["feature_drift_psi_core_80"], errors="coerce"
            ).astype(np.float32)
            added += 1
        if any(
            _model_drift_feature_alias_source(alias) == "feature_drift_ks_core"
            for alias in needed
        ):
            ks_source_col = None
            for candidate in (
                "feature_drift_ks_core",
                "feature_drift_ks_bin_mean",
                "feature_drift_ks_bin_max",
            ):
                if candidate in meta_context.columns:
                    ks_source_col = candidate
                    break
            if ks_source_col is not None:
                source = pd.to_numeric(
                    meta_context[ks_source_col],
                    errors="coerce",
                ).astype(np.float32)
                if "feature_drift_ks_core" in needed and "feature_drift_ks_core" not in out.columns:
                    out["feature_drift_ks_core"] = source
                    added += 1
                for alias in sorted(needed):
                    if (
                        alias in out.columns
                        or _model_drift_feature_alias_source(alias)
                        != "feature_drift_ks_core"
                    ):
                        continue
                    out[str(alias)] = source
                    added += 1
        out, alias_added = _materialize_model_drift_feature_aliases(
            out,
            needed,
            overwrite=False,
        )
        added += alias_added
        alias_source_frame = pd.concat(
            [out, meta_context.reindex(out.index)],
            axis=1,
            copy=False,
        )
        alias_source_frame = alias_source_frame.loc[
            :,
            ~alias_source_frame.columns.astype(str).duplicated(keep="first"),
        ]
        alias_source_frame, _ = _materialize_model_drift_feature_aliases(
            alias_source_frame,
            needed,
            overwrite=False,
        )
        alias_copy_added = 0
        for alias in sorted(needed):
            if alias in out.columns or alias not in alias_source_frame.columns:
                continue
            if _model_drift_feature_alias_source(alias) is None:
                continue
            out[str(alias)] = pd.to_numeric(
                alias_source_frame[alias],
                errors="coerce",
            ).astype(np.float32)
            alias_copy_added += 1
        added += alias_copy_added
        if added and not getattr(self, "_alpha_meta_context_warned", False):
            tprint(
                "Meta inference: materialized alpha drift/context columns "
                f"from base model ({added} columns for {kind})."
            )
            self._alpha_meta_context_warned = True
        return out

    def _materialize_meta_model_drift_features(
        self,
        features: pd.DataFrame,
        meta_model: Any,
        *,
        include_all: bool = False,
        prefix: str | None = None,
    ) -> pd.DataFrame:
        """Attach this meta model's own artifact-backed drift features.

        By default this only materializes columns required by the meta model's
        own feature contract.  ``include_all`` plus ``prefix`` is used by policy
        reporting/regime-adaptor paths to retain the full meta-layer drift and
        uncertainty diagnostics without colliding with unprefixed alpha context.
        """
        if not isinstance(features, pd.DataFrame) or features.empty:
            return features
        state = getattr(meta_model, "model_drift_state_", None)
        if not isinstance(state, dict) or not bool(state.get("enabled", False)):
            return features
        out = features.copy()
        drift = transform_model_drift_features(
            out,
            state,
            model=meta_model,
            index=out.index,
        )
        if drift.empty:
            return out
        effective_cols = _effective_selected_feature_contract(meta_model)
        needed = set(
            effective_cols
            or [str(c) for c in (getattr(meta_model, "feature_columns", []) or [])]
        )
        contract_sources = _required_model_drift_sources(needed)
        report_sources = set(MODEL_DRIFT_FEATURE_KEYS) if include_all else set()
        prefix_s = str(prefix).strip("_") if prefix else ""
        needed_sources = contract_sources | report_sources
        added = 0
        for col in drift.columns:
            if col not in needed_sources:
                continue
            values = pd.to_numeric(drift[col], errors="coerce").astype(np.float32)
            if col in contract_sources or (include_all and not prefix_s):
                out[col] = values
                added += 1
            if prefix_s and col in report_sources:
                out[f"{prefix_s}_{col}"] = values
                added += 1
        # Backward-compatible aggregate alias used by older feature contracts.
        if (
            "feature_drift_psi_core" in needed
            and "feature_drift_psi_core_80" in out.columns
        ):
            out["feature_drift_psi_core"] = out["feature_drift_psi_core_80"]
            added += 1
        if (
            "feature_drift_ks_core" in needed
            and "feature_drift_ks_core" not in out.columns
            and "feature_drift_ks_bin_mean" in out.columns
        ):
            out["feature_drift_ks_core"] = out["feature_drift_ks_bin_mean"]
            added += 1
        out, alias_added = _materialize_model_drift_feature_aliases(
            out,
            needed,
            overwrite=True,
        )
        added += alias_added
        if prefix_s and include_all:
            psi_col = f"{prefix_s}_feature_drift_psi_core_80"
            if psi_col in out.columns:
                out[f"{prefix_s}_feature_drift_psi_core"] = pd.to_numeric(
                    out[psi_col], errors="coerce"
                ).astype(np.float32)
                added += 1
            ks_col = f"{prefix_s}_feature_drift_ks_bin_mean"
            if ks_col in out.columns:
                out[f"{prefix_s}_feature_drift_ks_core"] = pd.to_numeric(
                    out[ks_col], errors="coerce"
                ).astype(np.float32)
                added += 1
        if added and not getattr(self, "_meta_drift_context_warned", False):
            tprint(
                "Meta inference: materialized meta-model drift columns "
                f"from artifact state ({added} columns)."
            )
            self._meta_drift_context_warned = True
        return out

    # =========================================================================
    # STEP 4: Meta Model Prediction
    # =========================================================================

    def predict_meta(
        self,
        features: pd.DataFrame,
        side: str,
        kind: str = "mr",
    ) -> pd.Series:
        """Step 4: Meta model prediction.

        Args:
            features: Feature DataFrame (with disagreement features and alpha preds)
            side: "long" or "short"
            kind: "mr" or "tf"

        Returns:
            Series of meta predictions
        """
        requested_kind = str(kind)
        key = str(kind)
        self._last_meta_model_key = None
        self._last_meta_model_input = None
        self._last_meta_model_features = []
        self._last_results.pop("mr_tf_meta_routing", None)
        self._last_mr_tf_route_frames_by_key.pop(f"meta:{requested_kind}", None)
        if key not in self.meta_models:
            core = strategy_core_id(str(kind))
            side_s = str(side or "").lower()
            candidates = [
                f"{side_s}_{kind}" if side_s else "",
                f"{key}_clf",
                f"{core}_clf" if core else "",
                f"{key}_tbm_clf",
                f"{core}_tbm_clf" if core else "",
                f"{side_s}_{kind}_clf" if side_s else "",
                f"{side_s}_{core}_clf" if side_s and core else "",
                f"{side_s}_{kind}_tbm_clf" if side_s else "",
                f"{side_s}_{core}_tbm_clf" if side_s and core else "",
            ]
            for candidate_key in candidates:
                if candidate_key and candidate_key in self.meta_models:
                    key = candidate_key
                    break

        if key not in self.meta_models:
            tprint(f"Warning: Meta model not found for {key}")
            return pd.Series(dtype=float)

        try:
            from extreme_price_movements.mr_tf_masks import (
                apply_mr_tf_masks as _apply_mr_tf_masks,
                mr_tf_masks_enabled as _mr_tf_masks_enabled,
            )

            if bool(_mr_tf_masks_enabled(self.cfg)) and not any(
                f"_{route}_tbm_clf" in str(key) for route in ("mr", "tf")
            ):
                mask_params = None
                try:
                    _, alpha_info_for_route = self._alpha_model_info_for_kind(
                        side,
                        requested_kind,
                    )
                    if isinstance(alpha_info_for_route, dict):
                        mask_params = (
                            (
                                alpha_info_for_route.get("mr_tf_specialists")
                                or {}
                            )
                            .get("mask_diagnostics", {})
                            .get("params")
                        )
                except Exception:
                    mask_params = None
                route_frame, route_diag = _apply_mr_tf_masks(
                    features,
                    side=side,
                    cfg=self.cfg,
                    params=mask_params,
                )
                self._last_mr_tf_route_frames_by_key[f"meta:{requested_kind}"] = route_frame[
                    [
                        c
                        for c in (
                            "__mr_tf_route__",
                            "__mr_mask__",
                            "__tf_mask__",
                            "__mixed_mask__",
                            "__mr_tf_params_hash__",
                        )
                        if c in route_frame.columns
                    ]
                ].copy()
                if "__mr_tf_route__" in route_frame.columns and len(route_frame):
                    route_values = set(route_frame["__mr_tf_route__"].astype(str))
                    route_values.discard("mixed")
                    if len(route_values) == 1:
                        route_name = next(iter(route_values))
                        side_s = str(side or "").lower()
                        core = strategy_core_id(str(requested_kind))
                        route_candidates = [
                            f"{side_s}_{requested_kind}_{route_name}_tbm_clf"
                            if side_s
                            else "",
                            f"{side_s}_{core}_{route_name}_tbm_clf"
                            if side_s and core
                            else "",
                            f"{requested_kind}_{route_name}_tbm_clf",
                            f"{core}_{route_name}_tbm_clf" if core else "",
                        ]
                        for route_key in route_candidates:
                            if route_key and route_key in self.meta_models:
                                self._last_results["mr_tf_meta_routing"] = {
                                    "requested_key": key,
                                    "selected_key": route_key,
                                    "route": route_name,
                                    "params_hash": route_diag.get(
                                        "params_hash", ""
                                    ),
                                }
                                key = route_key
                                break
        except Exception as _route_exc:
            tprint(
                f"Warning: MR/TF meta specialist routing skipped for {key}: "
                f"{_route_exc}"
            )

        meta_model = self.meta_models[key]

        if meta_model is None:
            return pd.Series(dtype=float)

        # Get feature columns from meta model
        try:
            timing_enabled = bool(self.cfg.get("inference_model_timing_enabled", True))
            t0 = time.perf_counter()
            effective_cols = _effective_selected_feature_contract(meta_model)
            if effective_cols:
                feat_cols = effective_cols
            elif hasattr(meta_model, "feature_columns"):
                feat_cols = meta_model.feature_columns
            else:
                feat_cols = list(features.columns)
            feat_cols = [
                str(c)
                for c in (feat_cols or [])
                if str(c) not in DELETED_MODEL_FEATURE_KEYS
            ]
            if not bool(
                self.cfg.get("preserve_logged_meta_model_derived_features", False)
            ):
                stale_model_derived_cols = []
                for col in feat_cols:
                    if col not in features.columns or not is_model_derived_feature_key(col):
                        continue
                    col_s = str(col)
                    # Bare base-score aliases are the causal source used to build
                    # downstream interactions. Drift/context diagnostics derived
                    # from model artifacts are recomputed below instead.
                    if re.match(
                        r"^(?:pred(?:_.*)?_H\d+|base_H\d+)"
                        r"(?:_ebm_raw|_ebm_en|_ebm_uncertainty_weighted)?$",
                        col_s,
                    ):
                        continue
                    stale_model_derived_cols.append(col_s)
                if stale_model_derived_cols:
                    features = features.drop(columns=stale_model_derived_cols).copy()
                    if timing_enabled and not getattr(
                        self, "_stale_meta_model_derived_warned", False
                    ):
                        tprint(
                            "Meta inference: dropped stale precomputed model-derived "
                            "feature columns before artifact-backed materialization "
                            f"(sample={stale_model_derived_cols[:12]})."
                        )
                        self._stale_meta_model_derived_warned = True

            if not bool(
                self.cfg.get(
                    "historical_inference_parity_skip_meta_materialization",
                    False,
                )
            ):
                features = self._materialize_alpha_model_meta_features(
                    features,
                    meta_model,
                    side=side,
                    kind=requested_kind,
                )
                features = self._materialize_meta_model_drift_features(
                    features,
                    meta_model,
                )
                features = self._materialize_meta_model_derived_features(
                    features,
                    meta_model,
                    side=side,
                    kind=requested_kind,
                )
            features, neutral_meta_fills = _fill_live_unavailable_meta_contract_features(
                features,
                feat_cols,
            )
            if neutral_meta_fills:
                self._last_results["meta_live_unavailable_neutral_fills"] = {
                    "key": key,
                    "features": neutral_meta_fills,
                }
                if timing_enabled:
                    tprint(
                        "Meta inference: filled selected live-unavailable "
                        "historical model-error/context features with neutral "
                        f"decision-time defaults for {key} "
                        f"(n={len(neutral_meta_fills)}, sample={neutral_meta_fills[:8]})."
                    )
            features, live_sparse_fills = _fill_live_sparse_meta_context_features(
                features,
                feat_cols,
            )
            if live_sparse_fills:
                self._last_results["meta_live_sparse_context_neutral_fills"] = {
                    "key": key,
                    "features": live_sparse_fills,
                }
                if timing_enabled:
                    tprint(
                        "Meta inference: neutral-filled selected live-sparse "
                        f"context features for {key} "
                        f"(n={len(live_sparse_fills)}, sample={live_sparse_fills[:8]})."
                    )
            raw_state_default_sources = set(RAW_STATE_SVD_FEATURE_NAMES) | set(
                RAW_STATE_DIAGNOSTIC_FEATURE_NAMES
            )
            features, artifact_context_fills = _fill_artifact_context_training_defaults(
                features,
                feat_cols,
                meta_model,
                allowed_sources=raw_state_default_sources,
            )
            if artifact_context_fills:
                self._last_results["meta_artifact_context_training_defaults"] = {
                    "key": key,
                    "features": artifact_context_fills,
                }
                if timing_enabled:
                    tprint(
                        "Meta inference: filled missing artifact-backed raw-state "
                        "context features from training feature stats "
                        f"for {key} (n={len(artifact_context_fills)}, "
                        f"sample={artifact_context_fills[:8]})."
                    )
            try:
                from extreme_price_movements.rule_mask_features import (
                    append_rule_mask_features as _append_rule_mask_features,
                    is_rule_mask_feature_name as _is_rule_mask_feature_name,
                )

                if any(_is_rule_mask_feature_name(c) for c in feat_cols):
                    _rule_cfg = dict(self.cfg or {})
                    _rule_cfg["lgbm_rule_mask_features_enabled"] = True
                    features, _rule_diag = _append_rule_mask_features(
                        features,
                        _rule_cfg,
                        side=side,
                        context=f"inference:meta:{key}",
                    )
                    if timing_enabled:
                        tprint(
                            "Meta inference: materialized rule-mask features "
                            f"key={key} n_rules={_rule_diag.get('n_rules')} "
                            f"source_available={_rule_diag.get('available_source_keys')}/"
                            f"{_rule_diag.get('n_source_keys')}"
                        )
            except Exception as exc:
                tprint(
                    f"Meta inference: rule-mask feature materialization failed "
                    f"for {key}: {exc}"
                )
            if timing_enabled:
                tprint(
                    "[Timing] model.meta_materialize: "
                    f"key={key} rows={len(features.index)} "
                    f"features_available={len(features.columns)} "
                    f"contract_features={len(feat_cols)} "
                    f"stage={time.perf_counter() - t0:.3f}s "
                    f"{_process_rss_log_fields()}"
                )

            missing_ebm_raw = _missing_ebm_raw_contract(meta_model, features)
            if missing_ebm_raw:
                reason = "missing_ebm_feature_contract"
                self._last_results["meta_contract_error"] = {
                    "key": key,
                    "reason": reason,
                    "missing_raw_features_count": len(missing_ebm_raw),
                    "missing_raw_features_sample": missing_ebm_raw[:20],
                }
                tprint(
                    f"Error predicting meta for {key}: {reason} "
                    f"({len(missing_ebm_raw)} missing raw EBM features)."
                )
                return pd.Series(dtype=float)

            strict = bool(self.cfg.get("strict_feature_parity", True))
            features, optional_added, optional_repaired = (
                _fill_optional_generated_model_features(
                    features,
                    model_feature_cols=feat_cols,
                )
            )
            if optional_added or optional_repaired:
                self._last_results["meta_optional_generated_features"] = {
                    "key": key,
                    "neutral_filled_missing_count": int(len(optional_added)),
                    "neutral_filled_missing_sample": optional_added[:20],
                    "repaired_nonfinite_count": int(len(optional_repaired)),
                    "repaired_nonfinite_sample": optional_repaired[:20],
                }
                if not getattr(
                    self,
                    "_meta_optional_generated_feature_warned",
                    False,
                ):
                    tprint(
                        "Meta inference: optional generated representation "
                        "features were neutral-filled; core feature parity remains "
                        "strict "
                        f"(missing={len(optional_added)}, "
                        f"nonfinite={len(optional_repaired)}, key={key})."
                    )
                    self._meta_optional_generated_feature_warned = True
            matrix_t0 = time.perf_counter()
            if strict:
                missing = [
                    c
                    for c in feat_cols
                    if c not in features.columns
                    and not is_optional_generated_model_feature_key(c)
                ]
                if missing:
                    reason = "missing_meta_feature_contract"
                    missing_sources = {
                        str(src)
                        for src in (
                            _model_drift_feature_alias_source(str(c)) for c in missing
                        )
                        if src is not None
                    }
                    related_columns: list[str] = []
                    if missing_sources:
                        for existing in map(str, features.columns):
                            existing_src = _model_drift_feature_alias_source(existing)
                            if existing_src in missing_sources or any(
                                existing == src or existing.endswith(f"_{src}")
                                for src in missing_sources
                            ):
                                related_columns.append(existing)
                        related_columns = sorted(dict.fromkeys(related_columns))
                    self._last_results["meta_contract_error"] = {
                        "key": key,
                        "reason": reason,
                        "missing_features_count": len(missing),
                        "missing_features_sample": missing[:20],
                        "missing_feature_sources_sample": sorted(missing_sources)[:20],
                        "available_related_features_sample": related_columns[:40],
                        "available_feature_count": int(len(features.columns)),
                    }
                    if missing_sources and not related_columns:
                        raw_state_like = sorted(
                            c
                            for c in map(str, features.columns)
                            if "raw_state" in c or "state_tod" in c
                        )
                        self._last_results["meta_contract_error"][
                            "available_raw_state_like_sample"
                        ] = raw_state_like[:40]
                        tprint(
                            "Meta feature contract debug: missing sources "
                            f"{sorted(missing_sources)[:20]} but no related "
                            f"columns found; raw_state_like_sample={raw_state_like[:20]}"
                        )
                    elif missing_sources:
                        tprint(
                            "Meta feature contract debug: missing sources "
                            f"{sorted(missing_sources)[:20]} related_columns_sample="
                            f"{related_columns[:20]}"
                        )
                    tprint(
                        f"Error predicting meta for {key}: {reason} "
                        f"({len(missing)} missing trained features): {missing[:20]}"
                    )
                    return pd.Series(dtype=float)
                allow_native_missing = _allow_lgbm_native_missing_model_inputs(
                    self.cfg
                )
                try:
                    model_matrix = features.reindex(columns=feat_cols)
                    model_matrix, _optional_added, _optional_repaired = (
                        _fill_optional_generated_model_features(
                            model_matrix,
                            model_feature_cols=feat_cols,
                        )
                    )
                    try:
                        if allow_native_missing:
                            X = _strict_lgbm_native_missing_model_matrix(
                                model_matrix,
                                model_feature_cols=feat_cols,
                                model_key=key,
                            )
                            if (
                                X.isna().to_numpy(dtype=bool, copy=False).any()
                                and not getattr(
                                    self,
                                    "_meta_native_missing_warned",
                                    False,
                                )
                            ):
                                tprint(
                                    f"Meta inference for {key}: preserving NaN "
                                    "trained feature values for LightGBM native "
                                    "missing-value handling."
                                )
                                self._meta_native_missing_warned = True
                        else:
                            X = _strict_finite_model_matrix(
                                model_matrix,
                                model_feature_cols=feat_cols,
                                model_key=key,
                            )
                    except FeatureParityError as exc:
                        report = getattr(exc, "report", {}) or {}
                        errors = set(report.get("global_errors") or [])
                        if "model_matrix_nonfinite" not in errors:
                            raise
                        if bool(
                            self.cfg.get(
                                "strict_feature_parity_neutral_fill_nonfinite",
                                False,
                            )
                        ):
                            total_bad, sample = _model_matrix_nonfinite_summary(
                                model_matrix
                            )
                            X = _training_neutral_filled_model_matrix(
                                model_matrix,
                                model_feature_cols=feat_cols,
                            )
                            self._last_results["meta_contract_error"] = {
                                "key": key,
                                "reason": "neutral_filled_nonfinite_meta_features",
                                "nonfinite_values": total_bad,
                                "nonfinite_features_sample": sample[:20],
                                "details": report,
                            }
                            if not getattr(
                                self,
                                "_meta_neutral_fill_nonfinite_warned",
                                False,
                            ):
                                tprint(
                                    f"Meta inference for {key}: neutral-filled "
                                    "non-finite trained feature values with the "
                                    "LGBM training/scoring adapter "
                                    f"(values={total_bad}, sample={sample[:12]})."
                                )
                                self._meta_neutral_fill_nonfinite_warned = True
                            X = _strict_finite_model_matrix(
                                X,
                                model_feature_cols=feat_cols,
                                model_key=key,
                            )
                        else:
                            matrix_float = model_matrix.astype(np.float32, copy=False)
                            values = matrix_float.to_numpy(
                                dtype=np.float32,
                                copy=False,
                            )
                            if allow_native_missing:
                                row_ok = ~np.isinf(values).any(axis=1)
                            else:
                                row_ok = np.isfinite(values).all(axis=1)
                            valid_rows = int(row_ok.sum())
                            if valid_rows <= 0:
                                raise
                            dropped_rows = int(len(row_ok) - valid_rows)
                            self._last_results["meta_contract_error"] = {
                                "key": key,
                                "reason": "dropped_nonfinite_meta_rows",
                                "dropped_rows": dropped_rows,
                                "valid_rows": valid_rows,
                                "details": report,
                            }
                            tprint(
                                f"Meta inference for {key}: dropped {dropped_rows}/"
                                f"{len(row_ok)} rows with non-finite trained "
                                f"features; predicting {valid_rows} strict rows."
                            )
                            if allow_native_missing:
                                X = _strict_lgbm_native_missing_model_matrix(
                                    matrix_float.loc[row_ok],
                                    model_feature_cols=feat_cols,
                                    model_key=key,
                                )
                            else:
                                X = _strict_finite_model_matrix(
                                    matrix_float.loc[row_ok],
                                    model_feature_cols=feat_cols,
                                    model_key=key,
                                )
                except FeatureParityError as exc:
                    self._last_results["meta_contract_error"] = {
                        "key": key,
                        "reason": "invalid_meta_feature_matrix",
                        "details": getattr(exc, "report", {}),
                    }
                    tprint(f"Error predicting meta for {key}: {exc}")
                    return pd.Series(dtype=float)
            else:
                available_cols = [c for c in feat_cols if c in features.columns]
                if not available_cols:
                    return pd.Series(dtype=float)
                ebm_contract_model = _extract_ebm_contract_model(meta_model)
                if ebm_contract_model is not None:
                    X = features.reindex(columns=feat_cols, fill_value=0.0).fillna(0)
                else:
                    X = features[available_cols].fillna(0)
            if timing_enabled:
                tprint(
                    "[Timing] model.meta_matrix: "
                    f"key={key} shape={getattr(X, 'shape', None)} "
                    f"strict={strict} stage={time.perf_counter() - matrix_t0:.3f}s "
                    f"{_process_rss_log_fields()}"
                )
            self._last_meta_model_key = key
            self._last_meta_model_features = [str(c) for c in list(X.columns)]
            self._last_meta_model_input = X.copy()
            self._last_meta_diagnostics = {}
            self._last_meta_diagnostics_frame = pd.DataFrame()
            if bool(self.cfg.get("inference_lgbm_internal_diagnostics_enabled", True)):
                try:
                    diag_t0 = time.perf_counter()
                    diag_frame = _lgbm_internal_metrics_frame(meta_model, X)
                    if isinstance(diag_frame, pd.DataFrame) and not diag_frame.empty:
                        self._last_meta_diagnostics_frame = diag_frame.copy()
                    self._last_meta_diagnostics = _first_row_diagnostics(diag_frame)
                    if timing_enabled and self._last_meta_diagnostics:
                        tprint(
                            "[Timing] model.meta_diagnostics: "
                            f"key={key} fields={len(self._last_meta_diagnostics)} "
                            f"stage={time.perf_counter() - diag_t0:.3f}s "
                            f"{_process_rss_log_fields()}"
                        )
                except Exception as exc:
                    self._last_meta_diagnostics_frame = pd.DataFrame()
                    self._last_meta_diagnostics = {
                        "lgbm_diagnostics_error": str(exc)[:240]
                    }
            pred_t0 = time.perf_counter()
            preds = meta_model.predict(X)
            if timing_enabled:
                tprint(
                    "[Timing] model.meta_predict: "
                    f"key={key} rows={len(X.index)} "
                    f"stage={time.perf_counter() - pred_t0:.3f}s "
                    f"total={time.perf_counter() - t0:.3f}s "
                    f"{_process_rss_log_fields()}"
                )

            out = pd.Series(preds, index=X.index)
            try:
                from extreme_price_movements.mr_tf_masks import (
                    mr_tf_masks_enabled as _mr_tf_masks_enabled,
                    route_series_from_frame as _route_series_from_frame,
                )

                is_route_specific_key = any(
                    f"_{route}_tbm_clf" in str(key) for route in ("mr", "tf")
                )
                if (
                    bool(_mr_tf_masks_enabled(self.cfg))
                    and not is_route_specific_key
                    and not bool(getattr(self, "_mr_tf_meta_override_active", False))
                ):
                    route_frame = (
                        getattr(self, "_last_mr_tf_route_frames_by_key", {}) or {}
                    ).get(f"meta:{requested_kind}")
                    if isinstance(route_frame, pd.DataFrame) and not route_frame.empty:
                        aligned_route = route_frame.reindex(out.index)
                        routes, known = _route_series_from_frame(aligned_route)
                        side_s = str(side or "").lower()
                        core = strategy_core_id(str(requested_kind))
                        route_overlay_report: Dict[str, Any] = {}
                        for route_name in ("mr", "tf"):
                            route_idx = out.index[
                                np.asarray(
                                    (routes == route_name) & known,
                                    dtype=bool,
                                )
                            ]
                            if len(route_idx) <= 0:
                                continue
                            route_candidates = [
                                f"{side_s}_{requested_kind}_{route_name}_tbm_clf"
                                if side_s
                                else "",
                                f"{side_s}_{core}_{route_name}_tbm_clf"
                                if side_s and core
                                else "",
                                f"{requested_kind}_{route_name}_tbm_clf",
                                f"{core}_{route_name}_tbm_clf" if core else "",
                            ]
                            route_key = next(
                                (
                                    candidate_key
                                    for candidate_key in route_candidates
                                    if candidate_key and candidate_key in self.meta_models
                                ),
                                "",
                            )
                            if not route_key:
                                continue
                            self._mr_tf_meta_override_active = True
                            try:
                                route_pred = self.predict_meta(
                                    features.reindex(route_idx),
                                    side,
                                    route_key,
                                )
                            finally:
                                self._mr_tf_meta_override_active = False
                            if isinstance(route_pred, pd.Series) and not route_pred.empty:
                                route_pred = route_pred.reindex(route_idx).dropna()
                                out.loc[route_pred.index] = route_pred.values
                                route_overlay_report[route_name] = {
                                    "selected_key": route_key,
                                    "rows": int(len(route_pred)),
                                }
                        if route_overlay_report:
                            self._last_results["mr_tf_meta_routing"] = {
                                "requested_key": str(key),
                                "mode": "per_row_overlay",
                                "routes": route_overlay_report,
                            }
            except Exception as route_exc:
                tprint(
                    f"Warning: MR/TF per-row meta specialist overlay skipped for {key}: "
                    f"{route_exc}"
                )
            return out
        except Exception as e:
            tprint(f"Error predicting meta for {key}: {e}")
            return pd.Series(dtype=float)

    def _find_strategy_id_for_bucket(self, bucket_key: str) -> str:
        buckets = (
            self.bucket_params.get("buckets", {})
            if isinstance(self.bucket_params, dict)
            else {}
        )
        for sid, cfg in buckets.items():
            if not isinstance(cfg, dict):
                continue
            side = str(cfg.get("side", "")).lower()
            if side and side not in bucket_key:
                continue
            return sid
        return ""

    def _predict_booster(
        self,
        bundle: Dict[str, Any],
        features: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        try:
            from extreme_price_movements.simple_position_sizer import (
                clean_and_standardize,
            )
        except ImportError:
            return np.array([]), np.array([])
        feature_keys = bundle.get("feature_keys", [])
        fold_models = bundle.get("fold_models", [])
        if not feature_keys or not fold_models:
            return np.array([]), np.array([])
        feat_cols = list(features.columns)
        feat_idx = [feat_cols.index(k) for k in feature_keys if k in feat_cols]
        if not feat_idx:
            return np.array([]), np.array([])
        X_raw = features.iloc[:, feat_idx].to_numpy(dtype=np.float64)
        winner = bundle.get("winner", "")
        n = X_raw.shape[0]
        pred_sum = np.zeros(n, dtype=np.float64)
        count = 0
        proba_sum = (
            np.zeros((n, 3), dtype=np.float64)
            if winner == "ridge_plus_lgbm_clf"
            else None
        )
        for fd in fold_models:
            if isinstance(fd, dict):
                model = fd["model"]
                medians = fd.get("medians")
                scaler = fd.get("scaler")
                c1d = fd.get("center_1d")
                s1d = fd.get("scale_1d")
            else:
                model = fd
                medians = scaler = c1d = s1d = None
            X_clean, _, _, _, _ = clean_and_standardize(
                X_raw, fit_medians=medians, scaler=scaler, center_1d=c1d, scale_1d=s1d
            )
            if winner == "ridge_plus_lgbm_clf":
                proba = np.asarray(model.predict_proba(X_clean), dtype=np.float32)
                score = proba[:, 0] - proba[:, 2]
                pred_sum += score
                if proba_sum is not None:
                    proba_sum += proba
            else:
                pred_sum += np.asarray(model.predict(X_clean), dtype=np.float64)
            count += 1
        if count == 0:
            return np.array([]), np.array([])
        booster_raw = pred_sum / count
        confidence = np.ones(n, dtype=np.float32)
        if proba_sum is not None and count > 1:
            p_mean = proba_sum / count
            p_mean = np.clip(p_mean, 1e-12, 1.0)
            entropy = -np.sum(p_mean * np.log(p_mean), axis=1)
            max_entropy = np.log(3.0)
            confidence = np.clip(1.0 - entropy / max_entropy, 0.7, 1.3).astype(
                np.float32
            )
        return booster_raw.astype(np.float32), confidence

    # =========================================================================
    # STEP 5: Ridge Position Sizing
    # =========================================================================

    def compute_ridge_position_size(
        self,
        features: pd.DataFrame,
        side: str,
        kind: str = "mr",
    ) -> Tuple[pd.Series, Dict[str, float]]:
        """Step 5: Ridge position sizer with optional booster mix.

        Args:
            features: Feature DataFrame (with meta predictions)
            side: "long" or "short"
            kind: "mr" or "tf"

        Returns:
            Tuple of (position_sizes Series, confidence dict)
        """
        bucket_key = self._policy_bucket_key(side, kind)
        ridge_preds = None
        skipped_unsafe_sizer = False

        if self.ridge_sizer is not None:
            try:
                model_names = getattr(self.ridge_sizer, "feature_names", None)
                if model_names is None:
                    model_names = getattr(self.ridge_sizer, "model_names_", [])
                model_names = list(model_names)
                unavailable = sorted(set(model_names) & LIVE_UNAVAILABLE_FEATURES)
                if unavailable:
                    skipped_unsafe_sizer = True
                    tprint(
                        "Ignoring legacy ridge position sizer for live inference; "
                        f"it requires target-derived fields: {unavailable}"
                    )
                    model_names = []

                if model_names:
                    for col in model_names:
                        if col not in features.columns:
                            features[col] = 0.0

                    if hasattr(self.ridge_sizer, "predict"):
                        ridge_preds = np.asarray(
                            self.ridge_sizer.predict(features), dtype=float
                        )
                    else:
                        model = getattr(self.ridge_sizer, "model", None)
                        if model is not None and hasattr(model, "predict"):
                            X = features[model_names].to_numpy(dtype=float)
                            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                            ridge_preds = np.asarray(model.predict(X), dtype=float)
            except Exception as e:
                if "live-unavailable" in str(e):
                    raise
                tprint(f"Warning: ridge_sizer.predict failed for {bucket_key}: {e}")

        if ridge_preds is None:
            if not isinstance(self.ridge_weight_map, dict) or not self.ridge_weight_map:
                tprint(f"Warning: Ridge weights not found for {bucket_key}")
                return pd.Series(0.0, index=features.index), {"confidence": 0.0}

            prefix = f"{bucket_key}_"
            bucket_weights = {
                k[len(prefix) :]: float(v)
                for k, v in self.ridge_weight_map.items()
                if isinstance(k, str) and k.startswith(prefix)
            }
            if not bucket_weights:
                fallback_size = self._policy_fallback_position_size(bucket_key)
                if fallback_size > 0.0:
                    if skipped_unsafe_sizer:
                        tprint(
                            f"Using policy fallback sizing for {bucket_key}: "
                            f"{fallback_size:.6g}"
                        )
                    return pd.Series(fallback_size, index=features.index), {
                        "confidence": min(1.0, fallback_size)
                    }
                tprint(f"Warning: No flattened ridge weights found for {bucket_key}")
                return pd.Series(0.0, index=features.index), {"confidence": 0.0}

            feature_names = list(bucket_weights.keys())
            unavailable = sorted(set(feature_names) & LIVE_UNAVAILABLE_FEATURES)
            if unavailable:
                fallback_size = self._policy_fallback_position_size(bucket_key)
                if fallback_size > 0.0:
                    tprint(
                        "Ignoring flattened ridge weights for live inference; "
                        f"they require target-derived fields: {unavailable}. "
                        f"Using policy fallback sizing for {bucket_key}: "
                        f"{fallback_size:.6g}"
                    )
                    return pd.Series(fallback_size, index=features.index), {
                        "confidence": min(1.0, fallback_size)
                    }
                tprint(
                    "Ignoring flattened ridge weights for live inference; "
                    f"they require target-derived fields: {unavailable}"
                )
                return pd.Series(0.0, index=features.index), {"confidence": 0.0}
            X = (
                features.reindex(columns=feature_names, fill_value=0.0)
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            coefs_array = np.asarray(
                [bucket_weights[c] for c in feature_names], dtype=float
            )
            ridge_preds = np.dot(X, coefs_array)

        # --- Apply booster mix if available ---
        final_preds = ridge_preds.copy()
        mix_meta: Dict[str, float] = {}
        if self.booster_bundles and isinstance(self.bucket_params, dict):
            strategy_id = bucket_key or self._find_strategy_id_for_bucket(bucket_key)
            booster_bundle = None
            if strategy_id:
                booster_bundle = self.booster_bundles.get(strategy_id)
            if not booster_bundle:
                for sid, bb in self.booster_bundles.items():
                    if isinstance(bb, dict) and bb.get("winner"):
                        booster_bundle = bb
                        strategy_id = sid
                        break
            if booster_bundle is not None:
                bucket_cfg = self.bucket_params.get("buckets", {}).get(strategy_id, {})
                if isinstance(bucket_cfg, dict):
                    mix_ridge_w = float(bucket_cfg.get("sizer_mix_ridge_w", 1.0))
                    mix_booster_w = float(bucket_cfg.get("sizer_mix_booster_w", 0.0))
                    mix_conf_mult = float(bucket_cfg.get("sizer_mix_conf_mult", 1.0))
                else:
                    mix_ridge_w = 1.0
                    mix_booster_w = 0.0
                    mix_conf_mult = 1.0
                if mix_booster_w > 0:
                    booster_raw, booster_conf = self._predict_booster(
                        booster_bundle, features
                    )
                    if len(booster_raw) == len(ridge_preds):
                        final_preds = (
                            mix_ridge_w * ridge_preds
                            + mix_booster_w
                            * booster_raw
                            * (booster_conf * mix_conf_mult)
                        )
                        mix_meta["booster_winner"] = 1.0
                        mix_meta["mix_ridge_w"] = mix_ridge_w
                        mix_meta["mix_booster_w"] = mix_booster_w
                        mix_meta["mix_conf_mult"] = mix_conf_mult

        strategy_id_for_regime = bucket_key or self._find_strategy_id_for_bucket(
            bucket_key
        )
        adaptor = None
        if isinstance(self.regime_adaptors, dict):
            adaptor = self.regime_adaptors.get(strategy_id_for_regime)
            if adaptor is None:
                for sid, candidate in self.regime_adaptors.items():
                    if strategy_id_matches(str(sid), {str(strategy_id_for_regime)}):
                        adaptor = candidate
                        break
        if isinstance(adaptor, dict) and regime_adaptor_inference_enabled(
            self.cfg, adaptor
        ):
            try:
                base_diagnostics_by_key = (
                    getattr(self, "_last_base_lgbm_diagnostics_by_key", {}) or {}
                )
                base_diagnostics = (
                    base_diagnostics_by_key.get(str(kind))
                    or base_diagnostics_by_key.get(f"{side}_{kind}")
                    or getattr(self, "_last_base_lgbm_diagnostics", {})
                    or {}
                )
                regime_source_features = features
                try:
                    route_frames = (
                        getattr(self, "_last_mr_tf_route_frames_by_key", {}) or {}
                    )
                    route_frame = None
                    for route_key in (str(kind), f"{side}_{kind}", f"meta:{kind}"):
                        candidate_route_frame = route_frames.get(route_key)
                        if isinstance(candidate_route_frame, pd.DataFrame):
                            route_frame = candidate_route_frame
                            break
                    if isinstance(route_frame, pd.DataFrame) and not route_frame.empty:
                        regime_source_features = features.copy()
                        aligned_route = route_frame.reindex(regime_source_features.index)
                        for route_col in aligned_route.columns:
                            if route_col not in regime_source_features.columns:
                                regime_source_features[route_col] = aligned_route[
                                    route_col
                                ].values
                except Exception:
                    regime_source_features = features
                regime_features = _feature_frame_with_lgbm_diagnostics(
                    regime_source_features,
                    base_diagnostics=base_diagnostics,
                    meta_diagnostics=getattr(self, "_last_meta_diagnostics", {}) or {},
                )
                try:
                    from extreme_price_movements.mr_tf_masks import (
                        append_mr_tf_route_features as _append_mr_tf_route_features,
                    )

                    regime_features = _append_mr_tf_route_features(regime_features)
                except Exception:
                    pass
                regime_features = _feature_frame_with_latest_drift_features(
                    regime_features,
                    self.cfg,
                )
                if "symbol" in regime_features.columns:
                    symbols = regime_features["symbol"].astype(str).to_numpy()
                else:
                    symbols = regime_features.index.astype(str).to_numpy()
                applied = apply_regime_adaptor(
                    regime_features,
                    final_preds,
                    adaptor,
                    timestamps=regime_features.index,
                    symbols=symbols,
                )
                regime_weight = np.asarray(
                    applied.get("regime_weight", np.ones(len(final_preds))), dtype=float
                )
                eligible = np.asarray(
                    applied.get("eligible", np.ones(len(final_preds), dtype=bool)),
                    dtype=bool,
                )
                final_preds = _regime_adaptor_score_from_applied(
                    applied, final_preds, regime_weight
                )
                final_preds = np.where(eligible, final_preds, 0.0)
                mix_meta["regime_adaptor_enabled"] = float(
                    bool(np.any(applied.get("regime_adjustment_enabled", [True])))
                )
                mix_meta["regime_eligible_share"] = float(np.mean(eligible))
                mix_meta["regime_weight_mean"] = float(np.mean(regime_weight))
                for key in (
                    "meta_correctness_probability",
                    "meta_correctness_probability_raw",
                    "meta_correctness_probability_calibrated",
                    "direct_label_regime_probability",
                    "direct_label_regime_probability_raw",
                    "meta_correctness_probability_calibrator_enabled",
                    "meta_correctness_zscore",
                    "meta_correctness_logit_offset",
                    "meta_correctness_feature_count",
                    "combined_score",
                    "deployment_score_pre_rank",
                    "deployment_score",
                    "deployment_score_reference_rank",
                    "deployment_score_rank",
                    "deployment_score_rank_reference_n",
                    "local_batch_rank",
                    "score_delta_from_regime_adjustment",
                ):
                    if key in applied:
                        arr = np.asarray(applied[key], dtype=float)
                        mix_meta[key] = float(np.nanmean(arr)) if len(arr) else 0.0
                for key in (
                    "selected_correctness_integration_params",
                    "rank_scope",
                    "regime_disabled_reason",
                    "meta_correctness_schema_missing_features",
                    "selected_regime_adaptor_head",
                ):
                    if key in applied:
                        vals = np.asarray(applied[key]).astype(str)
                        mix_meta[key] = vals[0] if len(vals) else ""
            except Exception as exc:
                tprint(f"Warning: regime adaptor failed for {bucket_key}: {exc}")

        conf = np.clip(np.abs(final_preds), 0.0, 1.0)
        return (
            pd.Series(final_preds, index=features.index),
            {"confidence": float(np.nanmean(conf)) if len(conf) else 0.0, **mix_meta},
        )

    def _policy_fallback_position_size(self, bucket_key: str) -> float:
        """Return a conservative live sizing fallback from persisted policy params."""
        if not isinstance(self.bucket_params, dict) or not bucket_key:
            return 0.0
        bucket_cfg = {}
        buckets = self.bucket_params.get("buckets", {})
        if isinstance(buckets, dict):
            bucket_cfg = buckets.get(bucket_key, {}) or buckets.get(
                f"{bucket_key}_tbm", {}
            )
        if not bucket_cfg:
            bucket_cfg = self.bucket_params.get(bucket_key, {}) or {}
        if not isinstance(bucket_cfg, dict):
            return 0.0

        raw = bucket_cfg.get("selection_frac")
        if raw is None:
            raw = bucket_cfg.get("position_size")
        if raw is None:
            return 0.0
        try:
            size = float(raw)
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(size) or size <= 0.0:
            return 0.0
        return float(np.clip(size, 0.0, 0.30))

    # =========================================================================
    # STEP 6: Entry Policy (Limit Offset Optimizer)
    # =========================================================================

    def compute_entry_policy(
        self,
        symbol: str,
        side: str,
        meta_pred: float,
        features: pd.DataFrame,
        position_result: Dict[str, Any],
        kind: str = "mr",
        entry_price: float = 1.0,
        atr_frac: float = 0.02,
    ) -> Dict[str, Any]:
        """Step 6: Entry policy with limit offset optimizer.

        Uses compute_entry_policy_decision from entry_policy.py.

        Args:
            symbol: Trading symbol
            side: "long" or "short"
            meta_pred: Meta model prediction (score)
            features: Feature DataFrame
            position_result: Result from compute_ridge_position_size
            entry_price: Entry price (default 1.0 for normalized)
            atr_frac: ATR fraction for offset calculation

        Returns:
            Full entry decision dict with place_order, entry_px, etc.
        """
        # Get feature dict for entry policy
        feat_dict = {}
        if isinstance(features, pd.DataFrame) and not features.empty:
            # Extract key features for entry policy
            for col in features.columns:
                if col in [
                    "u_hat_z",
                    "mae_hat_z",
                    "mfe_hat_z",
                    "dur_hat_z",
                    "u_hat",
                    "mae_hat",
                    "mfe_hat",
                ]:
                    feat_dict[col] = (
                        float(features[col].iloc[0]) if len(features) > 0 else 0.0
                    )

        # Use meta prediction as score
        score = float(meta_pred) if np.isfinite(meta_pred) else 0.0

        # Get entry policy config from bucket_params or runtime_cfg
        bucket_cfg = (
            self.entry_policy_config
            or self._get_bucket_policy(side, kind)
            or self.bucket_params
        )

        # Flatten if needed
        if bucket_cfg:
            bucket_cfg = flatten_bucket_policy(bucket_cfg)

        try:
            decision = compute_entry_policy_decision(
                entry_px=entry_price,
                score=score,
                bucket_cfg=bucket_cfg,
                features=feat_dict,
                **{"atr_frac": atr_frac},
            )

            # Add metadata to decision
            decision["symbol"] = symbol
            decision["side"] = side
            decision["meta_score"] = score
            decision["position_size"] = position_result.get("size", 0.0)

            return decision

        except Exception as e:
            tprint(f"Error computing entry policy for {symbol}: {e}")
            return {
                "place_order": True,  # Default to placing order on error
                "entry_px_fill": entry_price,
                "offset_bps": 0.0,
                "symbol": symbol,
                "side": side,
                "error": str(e),
            }

    # =========================================================================
    # Full Chain Orchestration
    # =========================================================================

    def run_full_chain(
        self,
        symbol: str,
        side: str,
        features: Any,
        panel: Any = None,
        kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run predictions in proper order:

        1. Alpha (Base) Model Predictions
        2. Disagreement Features
        3. Meta Model Prediction
        4. Ridge Position Sizing
        5. Entry Policy (Limit Offset Optimizer)

        Args:
            symbol: Trading symbol
            side: "long" or "short"
            features: Feature DataFrame for the candidate

        Returns:
            Dictionary with all results including final action decision
        """
        results = {"symbol": symbol, "side": side}

        # Ensure features is a DataFrame with proper index
        features = self._materialize_symbol_features(symbol, features)
        if features.empty:
            results["action"] = "no_features"
            return results
        if symbol not in features.index:
            features.index = pd.Index([symbol] * len(features))

        if kind is None:
            strategies = self.available_strategies(side)
            kind = strategies[0] if strategies else "mr"
        strategy_id = strategy_core_id(str(kind))
        results["strategy_id"] = strategy_id

        # =====================================================================
        # STEP 2: Alpha (Base) Model Predictions
        # =====================================================================
        alpha_pred = self.predict_alpha(features, side, str(kind))
        alpha_preds = (
            {str(kind): alpha_pred}
            if isinstance(alpha_pred, (pd.DataFrame, pd.Series))
            and not alpha_pred.empty
            else {}
        )
        results["alpha_preds"] = alpha_preds

        if not alpha_preds:
            results["action"] = "no_alpha_predictions"
            return results

        # =====================================================================
        # STEP 3: Compute Disagreement Features
        # =====================================================================
        # Build meta base features
        meta_base = features.copy()

        # Add alpha predictions to meta features
        for pred_key, pred_series in alpha_preds.items():
            if (
                isinstance(pred_series, (pd.DataFrame, pd.Series))
                and not pred_series.empty
            ):
                meta_base[pred_key] = pred_series

        # Compute disagreement features
        key_mr = f"{side}_mr"
        key_tf = f"{side}_tf"

        if key_mr in alpha_preds and key_tf in alpha_preds:
            mr_preds = alpha_preds[key_mr]
            tf_preds = alpha_preds[key_tf]

            disagreement = self.compute_disagreement_features(
                meta_base, mr_preds, tf_preds, side
            )

            if (
                isinstance(disagreement, (pd.DataFrame, pd.Series))
                and not disagreement.empty
            ):
                meta_base["disagreement"] = disagreement
                results["disagreement_features"] = disagreement.to_dict()

        # =====================================================================
        # STEP 4: Meta Model Prediction
        # =====================================================================
        meta_pred = self.predict_meta(meta_base, side, kind)
        if not isinstance(meta_pred, (pd.DataFrame, pd.Series)) or meta_pred.empty:
            results["action"] = "no_meta_prediction"
            results["reason"] = "meta_prediction_missing_no_base_fallback"
            return results

        meta_pred_val = float(meta_pred.iloc[0]) if len(meta_pred) > 0 else 0.0
        if not np.isfinite(meta_pred_val):
            results["action"] = "no_meta_prediction"
            results["reason"] = "meta_prediction_non_finite_no_base_fallback"
            return results
        results["meta_pred"] = meta_pred_val
        base_diagnostics_by_key = dict(
            getattr(self, "_last_base_lgbm_diagnostics_by_key", {}) or {}
        )
        base_diagnostics = base_diagnostics_by_key.get(str(kind))
        if not base_diagnostics and base_diagnostics_by_key:
            base_diagnostics = next(iter(base_diagnostics_by_key.values()))
        if not base_diagnostics:
            base_diagnostics = dict(
                getattr(self, "_last_base_lgbm_diagnostics", {}) or {}
            )
        if base_diagnostics:
            results["base_lgbm_diagnostics"] = dict(base_diagnostics)
            results["base_lgbm_diagnostics_by_key"] = base_diagnostics_by_key
            for diag_key, diag_value in base_diagnostics.items():
                results[f"base_lgbm_{diag_key}"] = diag_value
        meta_diagnostics = dict(getattr(self, "_last_meta_diagnostics", {}) or {})
        if meta_diagnostics:
            results["meta_lgbm_diagnostics"] = meta_diagnostics
            results["lgbm_diagnostics"] = meta_diagnostics
            for diag_key, diag_value in meta_diagnostics.items():
                results[diag_key] = diag_value

        # Merge Meta Model with Base Model Predictions
        # Final Prediction = Base Prediction + (Meta Prediction * Volatility Scale)
        base_key = str(kind)
        if base_key in alpha_preds:
            base_pred = alpha_preds[base_key]
            # Try to find vol scale, fallback to 1.0 if not found
            if "atr_pct" in meta_base.columns:
                vol_scale = meta_base["atr_pct"].astype(float).fillna(1.0)
            elif "realized_volatility_24h" in meta_base.columns:
                vol_scale = (
                    meta_base["realized_volatility_24h"].astype(float).fillna(1.0)
                )
            else:
                vol_scale = pd.Series(1.0, index=meta_base.index)

            # Reconstruct the calibrated regression prediction
            calibrated_reg_pred = base_pred + (meta_pred * vol_scale)
            results["calibrated_reg_pred"] = (
                float(calibrated_reg_pred.iloc[0])
                if len(calibrated_reg_pred) > 0
                else 0.0
            )

            meta_base["calibrated_reg_pred"] = calibrated_reg_pred

        # =====================================================================
        # STEP 5: Ridge Position Sizing
        # =====================================================================
        ridge_features = meta_base.copy()
        ridge_features["meta_pred"] = meta_pred
        if "calibrated_reg_pred" in meta_base.columns:
            ridge_features["calibrated_reg_pred"] = meta_base["calibrated_reg_pred"]

        position_size, confidence = self.compute_ridge_position_size(
            ridge_features, side, kind
        )

        position_val = float(position_size.iloc[0]) if len(position_size) > 0 else 0.0
        results["position_size"] = position_val
        results["ridge_confidence"] = confidence.get("confidence", 1.0)

        results["orchestrator_position_size"] = position_val
        if position_val <= 0:
            results["action"] = "no_entry"
            results["reason"] = "position_sizer_rejected"
            results["sizing_source"] = "position_sizer_rejected"
            return results
        else:
            results["sizing_source"] = "legacy_orchestrator_diagnostic"

        # =====================================================================
        # STEP 6: Entry Policy (Limit Offset Optimizer)
        # =====================================================================
        position_result = {
            "size": position_val,
            "confidence": confidence.get("confidence", 1.0),
        }

        entry_decision = self.compute_entry_policy(
            symbol=symbol,
            side=side,
            meta_pred=meta_pred_val,
            features=features,
            position_result=position_result,
            kind=kind,
            entry_price=self._latest_panel_price(symbol, panel),
            **{"atr_frac": self._latest_atr_frac(features)},
        )

        results["entry_policy"] = entry_decision

        # =====================================================================
        # Final Decision
        # =====================================================================
        if entry_decision.get("place_order", False):
            results["action"] = "enter"
            results["entry_px"] = entry_decision.get("entry_px_fill", 1.0)
            results["size"] = position_val
            results["stop_px"] = entry_decision.get("sl_distance_atr_eff")
            results["target_px"] = entry_decision.get("tp_distance_atr_eff")
            results["offset_bps"] = entry_decision.get("offset_bps", 0.0)
        else:
            results["action"] = "no_entry"
            results["reason"] = entry_decision.get("reason", "entry_policy_rejected")
        self._last_results = dict(results)
        return results

    def run_full_chain_batch(
        self,
        features_df: pd.DataFrame,
        side: str,
    ) -> List[Dict[str, Any]]:
        """Run full chain for multiple symbols.

        Args:
            features_df: DataFrame with features, indexed by symbol
            side: "long" or "short"

        Returns:
            List of results for each symbol
        """
        results = []

        for symbol in features_df.index:
            symbol_features = features_df.loc[symbol:symbol]
            result = self.run_full_chain(symbol, side, symbol_features)
            results.append(result)

        return results


def run_inference_chain(
    model_bundle: Dict[str, Any],
    runtime_cfg: Optional[Dict[str, Any]] = None,
    features: Optional[pd.DataFrame] = None,
    side: str = "long",
    symbol: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience function to run full inference chain.

    Args:
        model_bundle: Loaded model bundle
        runtime_cfg: Runtime configuration (optional)
        features: Input features DataFrame (optional)
        side: "long" or "short"
        symbol: Trading symbol (optional, used if features is a DataFrame)

    Returns:
        Dictionary with inference results
    """
    orchestrator = ModelOrchestrator(model_bundle, runtime_cfg)

    if features is not None and symbol is not None:
        return orchestrator.run_full_chain(symbol, side, features)
    elif features is not None:
        return orchestrator.run_full_chain_batch(features, side)
    else:
        raise ValueError(
            "Either (features and symbol) or just features must be provided"
        )
