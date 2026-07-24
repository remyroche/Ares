"""Live materializers for frozen S52 base/meta feature overlays.

The S52 frozen inference bundle uses a few train-time derived columns that are
not ordinary hourly feature-store keys:

* ``__regime_source_*`` observable source-regime scores.
* ``__meta_raw__*`` aliases copied from raw feature columns.
* ``rel_rankband_*`` / ``rel_marginband_*`` train-derived reliability priors.

This module keeps those overlays explicit and deterministic so the native live
path can satisfy the same selected-feature contract used by frozen replay.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from extreme_price_movements.feature_transform_contract import (
    file_sha256,
    ordered_names_hash,
)
from extreme_price_movements.features_gmm_ae import (
    AE_GMM_FEATURE_COLUMNS,
    ae_gmm_learned_transform_hash,
    load_ae_gmm_state_artifact,
    transform_ae_gmm_features,
)
from extreme_price_movements.inference.parity import strategy_core_id
from extreme_price_movements.utils import tprint

SOURCE_REGIME_PREFIX = "__regime_source_"
SOURCE_REGIME_SUFFIX = "__"
META_RAW_PREFIX = "__meta_raw__"

RELIABILITY_FEATURES = (
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

BASE_ANCHOR_FEATURES = (
    "score_base",
    "score",
    "base_score_rank_pct_train_prior",
    "base_margin_to_cutoff",
    "base_margin_to_cutoff_z",
    "base_signal_zscore_within_archetype",
)

AE_GMM_FEATURE_SET = {str(col) for col in AE_GMM_FEATURE_COLUMNS}


def _normalized_sha256(value: Any) -> str:
    """Compare repository digests independent of the optional scheme prefix."""
    digest = str(value or "").strip().lower()
    return digest.removeprefix("sha256:")


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _as_float32_series(
    values: Any, index: pd.Index, default: float = np.nan
) -> pd.Series:
    if isinstance(values, pd.Series):
        out = pd.to_numeric(values.reindex(index), errors="coerce")
    elif values is None:
        out = pd.Series(default, index=index, dtype=np.float32)
    else:
        out = pd.to_numeric(pd.Series(values, index=index), errors="coerce")
    return out.astype(np.float32, copy=False)


def _source_required_columns(required_columns: Any) -> set[str]:
    return {
        str(col)
        for col in (required_columns or [])
        if str(col).startswith(SOURCE_REGIME_PREFIX)
        and str(col).endswith(SOURCE_REGIME_SUFFIX)
    }


def _meta_raw_required_columns(required_columns: Any) -> set[str]:
    return {
        str(col)
        for col in (required_columns or [])
        if str(col).startswith(META_RAW_PREFIX)
    }


def _score_family(frame: pd.DataFrame, cols: tuple[str, ...]) -> pd.Series:
    present = [col for col in cols if col in frame.columns]
    if not present:
        return pd.Series(0.5, index=frame.index, dtype=np.float32)
    values = frame[present].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    return values.max(axis=1).fillna(0.5).clip(0.0, 1.0).astype(np.float32)


def materialize_live_source_regime_features(
    features: pd.DataFrame,
    *,
    side: str,
    signal_bar_ts: Any | None,
    required_columns: Any,
    overwrite_existing: bool = False,
) -> pd.DataFrame:
    """Add live-computable source-regime and raw-meta alias columns.

    The formulas are the same observable source-score formulas used when the
    S52 labels/features are materialized.  If the selected contract does not
    request these columns, the input frame is returned unchanged.
    """

    if not isinstance(features, pd.DataFrame) or features.empty:
        return features
    required_source = _source_required_columns(required_columns)
    required_raw = _meta_raw_required_columns(required_columns)
    if "side" in (required_columns or []) or required_source or required_raw:
        out = features.copy()
    else:
        return features

    side_value = np.float32(1.0 if str(side).lower().startswith("long") else -1.0)
    if "side" in (required_columns or []) or "side" not in out.columns:
        out["side"] = side_value
    if "side_name" not in out.columns:
        out["side_name"] = str(side).lower()

    for col in sorted(required_raw):
        raw_name = col[len(META_RAW_PREFIX) :]
        if (
            col in out.columns
            and pd.to_numeric(out[col], errors="coerce").notna().any()
        ):
            continue
        if raw_name in out.columns:
            out[col] = pd.to_numeric(out[raw_name], errors="coerce").astype(
                np.float32,
                copy=False,
            )

    if not required_source:
        return out

    try:
        from scripts.materialize_candidate_source_tags import (
            ARCHETYPE_COLS,
            COMPONENT_COLS,
            DEFAULT_CONFIG,
            build_archetype_scores,
            build_component_scores,
            build_feature_registry,
            load_config,
        )
    except Exception as exc:
        tprint(
            "Live source-regime feature materialization unavailable: "
            f"{exc}; strict model parity will decide row eligibility."
        )
        return out

    original_index = out.index
    original_symbols = (
        out["__symbol__"].astype(str).to_numpy(dtype=object, copy=False)
        if "__symbol__" in out.columns
        else out.index.astype(str).to_numpy(dtype=object, copy=False)
    )
    work = out.copy().reset_index(drop=True)
    work["__symbol__"] = original_symbols
    if signal_bar_ts is None:
        if "__ts__" not in work.columns:
            raise ValueError(
                "Batch source-regime materialization requires a __ts__ column"
            )
        timestamps = pd.to_datetime(work["__ts__"], utc=True, errors="coerce")
        if timestamps.isna().any():
            raise ValueError(
                "Batch source-regime materialization received invalid UTC timestamps"
            )
        work["__ts__"] = timestamps
    else:
        ts = pd.Timestamp(signal_bar_ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        work["__ts__"] = ts
    try:
        config = load_config(DEFAULT_CONFIG)
        config["timestamp_col"] = "__ts__"
        config["symbol_col"] = "__symbol__"
        registry = build_feature_registry(work, config)
        components, component_report = build_component_scores(work, registry, config)
        archetypes = build_archetype_scores(work, components, registry, config)
        archetypes["not_dirty_shock_score"] = (
            (
                1.0
                - pd.to_numeric(archetypes["dirty_shock_avoid_score"], errors="coerce")
            )
            .clip(0.0, 1.0)
            .astype(np.float32)
        )
        archetypes["loud_clean_source_score"] = (
            (
                pd.to_numeric(
                    archetypes["loud_breakout_impulse_score"], errors="coerce"
                )
                * archetypes["not_dirty_shock_score"]
            )
            .clip(0.0, 1.0)
            .astype(np.float32)
        )
        family_scores = pd.DataFrame(
            {
                "trend_following_score": _score_family(
                    archetypes,
                    (
                        "quiet_continuation_score",
                        "run_entry_score",
                        "late_run_continuation_score",
                        "clean_run_entry_score",
                    ),
                ),
                "mean_reversion_score": _score_family(
                    archetypes,
                    ("retest_reversal_score",),
                ),
                "vol_compression_score": _score_family(
                    archetypes,
                    (
                        "compression_release_score",
                        "compression_capture_candidate_score",
                        "risk_adjusted_capture_candidate_score",
                        "clean_economic_capture_candidate_score",
                    ),
                ),
                "breakout_impulse_score": _score_family(
                    archetypes,
                    (
                        "loud_breakout_impulse_score",
                        "loud_clean_execution_score",
                        "loud_clean_source_score",
                    ),
                ),
                "dirty_avoid_score": _score_family(
                    archetypes,
                    (
                        "dirty_shock_avoid_score",
                        "misleading_location_risk_score",
                    ),
                ),
            },
            index=work.index,
        ).astype(np.float32)
        source_values: dict[str, pd.Series] = {}
        for col in list(COMPONENT_COLS):
            if col in components.columns:
                source_values[f"{SOURCE_REGIME_PREFIX}{col}{SOURCE_REGIME_SUFFIX}"] = (
                    components[col]
                )
        for col in list(ARCHETYPE_COLS):
            if col in archetypes.columns:
                source_values[f"{SOURCE_REGIME_PREFIX}{col}{SOURCE_REGIME_SUFFIX}"] = (
                    archetypes[col]
                )
        for col in family_scores.columns:
            source_values[f"{SOURCE_REGIME_PREFIX}{col}{SOURCE_REGIME_SUFFIX}"] = (
                family_scores[col]
            )

        added = []
        for col in sorted(required_source):
            series = source_values.get(col)
            if series is None:
                continue
            candidate = pd.Series(
                pd.to_numeric(series, errors="coerce")
                .fillna(0.5)
                .clip(0.0, 1.0)
                .to_numpy(dtype=np.float32, copy=False),
                index=original_index,
            )
            if col in out.columns and not overwrite_existing:
                current = pd.to_numeric(out[col], errors="coerce").to_numpy(
                    dtype=np.float32, copy=True
                )
                replacement = candidate.to_numpy(dtype=np.float32, copy=False)
                missing = ~np.isfinite(current)
                current[missing] = replacement[missing]
                out[col] = current
            else:
                out[col] = candidate
            added.append(col)
        missing = sorted(required_source.difference(added))
        if added:
            tprint(
                "Materialized live source-regime selected features: "
                f"n={len(added)} source_cols_used={len(registry.get('source_columns') or [])} "
                f"sample={added[:8]}"
            )
        if missing:
            tprint(
                "Live source-regime selected features still unavailable after "
                f"materialization: n={len(missing)} sample={missing[:8]} "
                f"registry_group_counts="
                f"{ {str(k): int(len(v)) for k, v in (registry.get('available') or {}).items()} } "
                f"neutral_counts={component_report.get('component_neutral_counts', {})}"
            )
    except Exception as exc:
        tprint(
            "Live source-regime feature materialization failed; strict model "
            f"parity will decide row eligibility: {exc}"
        )
    return out


@lru_cache(maxsize=16)
def load_live_ae_gmm_state_payload(data_root: str, run_id: str) -> dict[str, Any]:
    """Load the frozen train-fitted AE/GMM state packaged for live inference."""

    artifact_root = Path(str(data_root or "data_perp")) / "artifacts" / str(run_id)
    route_manifest_path = artifact_root / "manifest.json"
    if route_manifest_path.exists():
        try:
            route_manifest = json.loads(route_manifest_path.read_text(encoding="utf-8"))
        except Exception:
            route_manifest = {}
        if str(route_manifest.get("schema") or "") == "side_routed_model_handoff_v1":
            states_by_side: dict[str, dict[str, Any]] = {}
            for side in ("long", "short"):
                route = (route_manifest.get("routes") or {}).get(side) or {}
                state_contract = route.get("ae_gmm") or {}
                raw_path = str(state_contract.get("path") or "").strip()
                if not raw_path:
                    raise ValueError(f"Side-routed AE/GMM contract has no {side} path")
                state_path = Path(raw_path)
                if not state_path.is_absolute() and not state_path.exists():
                    state_path = artifact_root / state_path
                state_path = state_path.resolve()
                expected_hash = str(state_contract.get("sha256") or "")
                if expected_hash and _normalized_sha256(file_sha256(state_path)) != _normalized_sha256(
                    expected_hash
                ):
                    raise ValueError(f"Side-routed AE/GMM {side} state SHA-256 mismatch")
                state = load_ae_gmm_state_artifact(state_path)
                input_features = [str(c) for c in state.get("feature_columns", []) or []]
                expected_order_hash = str(
                    state_contract.get("input_feature_order_hash") or ""
                )
                if expected_order_hash and _normalized_sha256(
                    ordered_names_hash(input_features)
                ) != _normalized_sha256(expected_order_hash):
                    raise ValueError(
                        f"Side-routed AE/GMM {side} input feature order mismatch"
                    )
                states_by_side[side] = {
                    "state": state,
                    "state_path": str(state_path),
                    "manifest_path": str(route_manifest_path),
                    "manifest": state_contract,
                    "input_feature_columns": input_features,
                    "generated_feature_columns": list(AE_GMM_FEATURE_COLUMNS),
                }
            return {
                "schema": "side_routed_ae_gmm_payload_v1",
                "states_by_side": states_by_side,
                "routing_contract": dict(route_manifest.get("routing_contract") or {}),
                "manifest_path": str(route_manifest_path),
            }
    candidates = [
        artifact_root / "ae_gmm_state" / "ae_gmm_state.pkl",
        artifact_root / "ae_gmm_state.pkl",
        artifact_root / "policy_params" / "ae_gmm_state.pkl",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            state = load_ae_gmm_state_artifact(path)
        except Exception as exc:
            tprint(f"Frozen AE/GMM state artifact failed to load: {path}: {exc}")
            continue
        manifest_path = path.with_name("ae_gmm_state_manifest.json")
        if not manifest_path.exists():
            manifest_path = path.with_suffix(".manifest.json")
        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                manifest = {}
        try:
            expected_state_hash = str(
                manifest.get("state_sha256") or manifest.get("sha256") or ""
            )
            if expected_state_hash and _normalized_sha256(
                file_sha256(path)
            ) != _normalized_sha256(expected_state_hash):
                raise ValueError("serialized state SHA-256 does not match manifest")
            input_features = [str(c) for c in state.get("feature_columns", []) or []]
            expected_order_hash = str(manifest.get("input_feature_order_hash") or "")
            if expected_order_hash and _normalized_sha256(
                ordered_names_hash(input_features)
            ) != _normalized_sha256(expected_order_hash):
                raise ValueError("AE/GMM input feature order does not match manifest")
            expected_transform_hash = str(
                manifest.get("learned_transform_hash")
                or manifest.get("cycle_state_hash")
                or ""
            )
            if expected_transform_hash and _normalized_sha256(
                ae_gmm_learned_transform_hash(state)
            ) != _normalized_sha256(expected_transform_hash):
                raise ValueError("AE/GMM learned transform does not match manifest")
            if str(manifest.get("contract") or "").startswith(
                "single_cycle_frozen_ae_gmm_bundle"
            ):
                required_manifest_fields = {
                    "input_feature_order_hash",
                    "learned_transform_hash",
                    "cycle_state_hash",
                    "materialized_transform_rules",
                }
                missing_manifest_fields = sorted(
                    key for key in required_manifest_fields if not manifest.get(key)
                )
                if missing_manifest_fields:
                    raise ValueError(
                        "single-cycle AE/GMM manifest is incomplete: "
                        + ", ".join(missing_manifest_fields)
                    )
        except Exception as exc:
            tprint(f"Frozen AE/GMM state contract failed: {path}: {exc}")
            continue
        return {
            "state": state,
            "state_path": str(path),
            "manifest_path": str(manifest_path) if manifest_path.exists() else "",
            "manifest": manifest,
            "input_feature_columns": input_features,
            "generated_feature_columns": list(AE_GMM_FEATURE_COLUMNS),
        }
    return {}


def _live_ae_gmm_payload_for_side(
    payload: Mapping[str, Any] | None,
    side: str | None,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    routed = payload.get("states_by_side")
    if not isinstance(routed, Mapping):
        return payload
    side_name = str(side or "").strip().lower()
    if side_name:
        selected = routed.get(side_name)
        return selected if isinstance(selected, Mapping) else {}
    return payload


def live_ae_gmm_input_feature_columns(
    payload: Mapping[str, Any] | None,
    side: str | None = None,
) -> list[str]:
    if isinstance(payload, Mapping) and isinstance(payload.get("states_by_side"), Mapping):
        if side is not None:
            return live_ae_gmm_input_feature_columns(
                _live_ae_gmm_payload_for_side(payload, side)
            )
        columns: list[str] = []
        for routed_side in ("long", "short"):
            columns.extend(
                live_ae_gmm_input_feature_columns(
                    _live_ae_gmm_payload_for_side(payload, routed_side)
                )
            )
        return list(dict.fromkeys(columns))
    state = (payload or {}).get("state") if isinstance(payload, Mapping) else None
    if not isinstance(state, Mapping):
        return []
    return [str(c) for c in state.get("feature_columns", []) or [] if str(c).strip()]


def materialize_live_ae_gmm_features(
    features: pd.DataFrame,
    *,
    side: str,
    signal_bar_ts: Any,
    required_columns: Any,
    state_payload: Mapping[str, Any] | None,
    overwrite_existing: bool = False,
) -> pd.DataFrame:
    """Append frozen AE/GMM features required by a selected model contract.

    The state is fitted on training rows and loaded from the deployment bundle.
    Live/OOS rows are transformed only; this function never refits AE/GMM.
    """

    if not isinstance(features, pd.DataFrame) or features.empty:
        return features
    required = {
        str(col) for col in (required_columns or []) if str(col) in AE_GMM_FEATURE_SET
    }
    if not required:
        return features
    missing_required = (
        set(required)
        if overwrite_existing
        else {
            col
            for col in required
            if col not in features.columns
            or not bool(
                np.isfinite(
                    pd.to_numeric(features[col], errors="coerce").to_numpy(
                        dtype=float,
                        copy=False,
                    )
                ).all()
            )
        }
    )
    if not missing_required:
        return features
    payload = dict(_live_ae_gmm_payload_for_side(state_payload, side))
    state = payload.get("state")
    if not isinstance(state, Mapping) or not bool(state.get("enabled", False)):
        tprint(
            "Frozen AE/GMM selected features requested but no enabled state "
            f"artifact is packaged; missing={sorted(required)[:12]}"
        )
        return features

    input_columns = live_ae_gmm_input_feature_columns(payload)
    if not input_columns:
        tprint(
            "Frozen AE/GMM selected features requested but state has no input "
            "feature columns; strict base parity will decide row eligibility."
        )
        return features

    out = materialize_live_source_regime_features(
        features,
        side=side,
        signal_bar_ts=signal_bar_ts,
        required_columns=input_columns,
        overwrite_existing=overwrite_existing,
    )
    # Source-regime materialization deliberately returns the input frame when
    # none of its own columns are requested.  AE/GMM outputs are side-routed,
    # so never append them to that shared caller-owned frame: doing so lets the
    # first side populate columns that make the second side skip its transform.
    if out is features:
        out = features.copy()
    x_base = out.reindex(columns=input_columns).apply(pd.to_numeric, errors="coerce")
    raw_values = x_base.to_numpy(dtype=np.float32, copy=False)
    complete_raw = np.isfinite(raw_values).all(axis=1)
    if not bool(complete_raw.any()):
        finite_counts = np.isfinite(raw_values).sum(axis=0)
        coverage_order = np.argsort(finite_counts, kind="stable")
        weakest = [
            {
                "feature": str(input_columns[int(pos)]),
                "finite": int(finite_counts[int(pos)]),
                "rows": int(len(x_base)),
            }
            for pos in coverage_order[: min(20, len(coverage_order))]
        ]
        tprint(
            "Frozen AE/GMM transform has no jointly finite input rows; "
            f"inputs={len(input_columns)} rows={len(x_base)} weakest={weakest}"
        )
    generated = pd.DataFrame(
        np.nan,
        index=out.index,
        columns=sorted(missing_required),
        dtype=np.float32,
    )
    try:
        if bool(complete_raw.any()):
            complete_index = out.index[complete_raw]
            generated.loc[complete_index] = (
                transform_ae_gmm_features(
                    x_base.loc[complete_index],
                    dict(state),
                    index=complete_index,
                )
                .reindex(columns=sorted(missing_required))
                .to_numpy(dtype=np.float32, copy=False)
            )
    except Exception as exc:
        tprint(
            "Frozen AE/GMM live transform failed; strict base parity will "
            f"decide row eligibility: {type(exc).__name__}: {exc}"
        )
        return out

    generated_block = generated.apply(pd.to_numeric, errors="coerce").astype(
        np.float32,
        copy=False,
    )
    if overwrite_existing:
        # Replace the block in one operation. Besides avoiding pandas frame
        # fragmentation, this guarantees stale cached outputs cannot survive
        # when the upstream context universe is recomputed.
        out = pd.concat(
            [
                out.drop(columns=list(generated_block.columns), errors="ignore"),
                generated_block.reindex(out.index),
            ],
            axis=1,
            copy=False,
        )
    else:
        for col in generated_block.columns:
            replacement = generated_block[col].reindex(out.index)
            if col not in out.columns:
                out[col] = replacement
                continue
            current = pd.to_numeric(out[col], errors="coerce").to_numpy(
                dtype=np.float32,
                copy=True,
            )
            candidate = replacement.to_numpy(dtype=np.float32, copy=False)
            fill = ~np.isfinite(current) & np.isfinite(candidate)
            current[fill] = candidate[fill]
            out[col] = current
    finite_counts = {
        str(col): int(pd.to_numeric(out[col], errors="coerce").notna().sum())
        for col in list(generated.columns)[:12]
    }
    tprint(
        "Materialized frozen AE/GMM selected features: "
        f"n={len(generated.columns)} rows={len(out)} "
        f"state={payload.get('state_path', '')} sample_finite={finite_counts}"
    )
    return out


def _quantile_label(edges: list[float], value: float, prefix: str) -> str:
    if not np.isfinite(value):
        return f"{prefix}__missing"
    clean = np.asarray(
        [float(v) for v in (edges or []) if np.isfinite(float(v))], dtype=np.float64
    )
    if clean.size == 0:
        return f"{prefix}__missing"
    return f"{prefix}__q{int(np.searchsorted(clean, float(value), side='right'))}"


def _derive_frontier_arch(side: str, score: float, payload: Mapping[str, Any]) -> str:
    side_s = "short" if str(side).lower().startswith("short") else "long"
    thresholds = (
        (payload.get("source_tag_score_thresholds") or {}).get(side_s, {})
        if isinstance(payload, Mapping)
        else {}
    )
    for bucket in ("top10", "top20", "top30"):
        min_score = _safe_float(thresholds.get(bucket), np.nan)
        if np.isfinite(min_score) and np.isfinite(score) and score >= min_score:
            return f"{side_s}__model_frontier_{bucket}"
    return f"{side_s}__model_candidate_background"


@lru_cache(maxsize=16)
def load_meta_reliability_prior_payload(data_root: str, run_id: str) -> dict[str, Any]:
    root = (
        Path(str(data_root or "data_perp"))
        / "artifacts"
        / str(run_id)
        / "policy_params"
    )
    candidates = [
        root / "meta_reliability_priors.json",
        Path(str(data_root or "data_perp"))
        / "artifacts"
        / str(run_id)
        / "meta_reliability_priors.json",
    ]
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


@lru_cache(maxsize=16)
def load_residual_reference_prior_payload(
    data_root: str, run_id: str
) -> dict[str, Any]:
    """Load priors fitted on the residual/V9 training reference.

    These priors are deliberately separate from the meta model's reliability
    payload: the frozen V9 predecessor was trained with the residual-reference
    fold features, whose support period and missing-group behavior differ.
    """

    root = Path(str(data_root or "data_perp")) / "artifacts" / str(run_id)
    candidates = [
        root / "policy_params" / "residual_reference_priors.json",
        root / "residual_reference_priors.json",
    ]
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def apply_live_meta_reliability_priors(
    features: pd.DataFrame,
    *,
    side: str,
    base_predictions: Mapping[str, Mapping[str, Any]],
    prior_payload: Mapping[str, Any] | None,
) -> pd.DataFrame:
    """Attach frozen train-derived reliability priors to meta input rows."""

    if not isinstance(features, pd.DataFrame) or features.empty:
        return features
    payload = dict(prior_payload or {})
    if not payload:
        return features
    required = set(str(c) for c in payload.get("feature_names", RELIABILITY_FEATURES))
    required.update(BASE_ANCHOR_FEATURES)
    if not required:
        return features

    out = features.copy()
    side_s = "short" if str(side).lower().startswith("short") else "long"
    rank_edges = [float(v) for v in payload.get("score_quantile_edges", [])]
    margin_edges = [float(v) for v in payload.get("margin_quantile_edges", [])]
    side_arch = payload.get("side_arch_priors", {}) or {}
    global_prior = payload.get("global_prior", {}) or {}
    groups = payload.get("groups", {}) or {}
    score_reference = np.asarray(
        payload.get("score_reference_quantiles") or [], dtype=np.float64
    )
    score_reference = np.sort(score_reference[np.isfinite(score_reference)])

    def _group_values(prefix: str, arch: str, band: str) -> dict[str, float]:
        exact_key = f"{prefix}|{side_s}|{arch}|{band}"
        if bool(payload.get("exact_groups_only", False)):
            values = groups.get(exact_key)
            if isinstance(values, Mapping):
                return {str(k): _safe_float(v, np.nan) for k, v in values.items()}
            return {}
        fallback_arch_key = f"{prefix}|{side_s}|*|{band}"
        fallback_band_key = f"{prefix}|*|*|{band}"
        values = (
            groups.get(exact_key)
            or groups.get(fallback_arch_key)
            or groups.get(fallback_band_key)
        )
        if isinstance(values, Mapping):
            return {str(k): _safe_float(v, np.nan) for k, v in values.items()}
        return {}

    for symbol in out.index:
        symbol_s = str(symbol)
        pred = dict(base_predictions.get(symbol_s) or {})
        score = _safe_float(
            pred.get(
                "base_pred",
                out.at[symbol, "score"] if "score" in out.columns else np.nan,
            ),
            np.nan,
        )
        arch = ""
        for arch_col in ("source_tag", "__source_tag__", "archetype_source_tag"):
            if arch_col in out.columns:
                raw_arch = str(out.at[symbol, arch_col]).strip()
                if raw_arch and raw_arch.lower() not in {"nan", "none", "null", "<na>"}:
                    arch = raw_arch
                    break
        if not arch:
            arch = _derive_frontier_arch(side_s, score, payload)
        rank_band = _quantile_label(rank_edges, score, "base_rank_band")
        cutoff = _safe_float(
            (side_arch.get(f"{side_s}|{arch}") or {}).get("cutoff")
            if isinstance(side_arch, Mapping)
            else np.nan,
            _safe_float(global_prior.get("cutoff"), np.nan),
        )
        local_prior = (
            side_arch.get(f"{side_s}|{arch}") or {}
            if isinstance(side_arch, Mapping)
            else {}
        )
        mean = _safe_float(
            local_prior.get("mean"), _safe_float(global_prior.get("score_mean"), 0.0)
        )
        std = _safe_float(
            local_prior.get("std"), _safe_float(global_prior.get("score_std"), 1.0)
        )
        if not np.isfinite(std) or std <= 1e-12:
            std = 1.0
        margin = (
            score - cutoff if np.isfinite(score) and np.isfinite(cutoff) else np.nan
        )
        margin_band = _quantile_label(margin_edges, margin, "base_margin_band")
        out.at[symbol, "base_rank_band"] = rank_band
        out.at[symbol, "base_margin_band"] = margin_band
        out.at[symbol, "source_tag"] = arch
        # Historical V9 training prepends ``score_base`` while retaining
        # ``score`` as a protected anchor. They are intentionally identical.
        out.at[symbol, "score_base"] = np.float32(score if np.isfinite(score) else 0.0)
        out.at[symbol, "score"] = np.float32(score if np.isfinite(score) else 0.0)
        if score_reference.size and np.isfinite(score):
            rank_pct = np.searchsorted(score_reference, score, side="right") / float(
                score_reference.size
            )
        else:
            rank_pct = 0.5
        out.at[symbol, "base_score_rank_pct_train_prior"] = np.float32(
            np.clip(rank_pct, 0.0, 1.0)
        )
        out.at[symbol, "base_margin_to_cutoff"] = np.float32(
            margin if np.isfinite(margin) else 0.0
        )
        out.at[symbol, "base_margin_to_cutoff_z"] = np.float32(
            margin / std if np.isfinite(margin) else 0.0
        )
        out.at[symbol, "base_signal_zscore_within_archetype"] = np.float32(
            (score - mean) / std if np.isfinite(score) else 0.0
        )

        for prefix, band in (
            ("rel_rankband", rank_band),
            ("rel_marginband", margin_band),
        ):
            vals = _group_values(prefix, arch, band)
            rows = _safe_float(vals.get("rows"), 0.0)
            clean = _safe_float(
                vals.get("clean_rate"), _safe_float(global_prior.get("clean_rate"), 0.0)
            )
            bad = _safe_float(
                vals.get("bad_mae_rate"),
                _safe_float(global_prior.get("bad_mae_rate"), 0.0),
            )
            timeout = _safe_float(
                vals.get("timeout_rate"),
                _safe_float(global_prior.get("timeout_rate"), 0.0),
            )
            dirty = _safe_float(
                vals.get("dirty_positive_rate"),
                _safe_float(global_prior.get("dirty_positive_rate"), 0.0),
            )
            exec_mean = _safe_float(
                vals.get("exec_margin_mean"),
                _safe_float(global_prior.get("exec_margin_mean"), 0.0),
            )
            edge = clean - bad - 0.50 * timeout + exec_mean
            out.at[symbol, f"{prefix}_rows_log1p"] = np.float32(
                np.log1p(max(rows, 0.0))
            )
            out.at[symbol, f"{prefix}_clean_rate"] = np.float32(clean)
            out.at[symbol, f"{prefix}_bad_mae_rate"] = np.float32(bad)
            out.at[symbol, f"{prefix}_timeout_rate"] = np.float32(timeout)
            out.at[symbol, f"{prefix}_dirty_positive_rate"] = np.float32(dirty)
            out.at[symbol, f"{prefix}_exec_margin_mean"] = np.float32(exec_mean)
            out.at[symbol, f"{prefix}_edge"] = np.float32(edge)

    for col in sorted(required):
        if col not in out.columns:
            out[col] = np.float32(0.0)
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(
            np.float32, copy=False
        )
    return out


def reliability_prior_payload_from_training_frame(
    frame: pd.DataFrame,
    *,
    selected_col: str = "selected_top30",
    shrinkage_k: float = 60.0,
) -> dict[str, Any]:
    """Build a JSON-serializable prior payload from training rows only."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Cannot build meta reliability priors from an empty frame")
    work = frame.copy()
    if "side_name" not in work.columns:
        raise ValueError("Training frame must contain side_name")
    score = pd.to_numeric(work.get("score"), errors="coerce").astype(np.float64)
    if selected_col not in work.columns:
        work[selected_col] = score >= float(score.quantile(0.70))
    selected = work[selected_col].astype(bool)
    side = work["side_name"].astype(str).str.lower()
    if "source_tag" in work.columns:
        arch = work["source_tag"].astype(str)
    elif "source_family" in work.columns:
        arch = side + "__" + work["source_family"].astype(str)
    else:
        arch = side + "__unknown"
    clean = (
        pd.to_numeric(
            work.get("clean_exec", work.get("clean_exec_label", 0.0)), errors="coerce"
        )
        .fillna(0.0)
        .clip(0.0, 1.0)
    )
    bad = (
        pd.to_numeric(work.get("full_path_bad_mae_1r", 0.0), errors="coerce")
        .fillna(0.0)
        .clip(0.0, 1.0)
    )
    timeout = (
        pd.to_numeric(work.get("timeout", 0.0), errors="coerce")
        .fillna(0.0)
        .clip(0.0, 1.0)
    )
    dirty = (
        pd.to_numeric(work.get("dirty_positive", 0.0), errors="coerce")
        .fillna(0.0)
        .clip(0.0, 1.0)
    )
    exec_margin = pd.to_numeric(work.get("exec_margin", 0.0), errors="coerce").fillna(
        0.0
    )

    global_cutoff = (
        float(score.loc[selected & score.notna()].min())
        if bool((selected & score.notna()).any())
        else float(score.quantile(0.70))
    )
    global_mean = float(score.mean()) if score.notna().any() else 0.0
    global_std = float(score.std()) if score.notna().sum() > 1 else 1.0
    if not np.isfinite(global_std) or global_std <= 1e-12:
        global_std = 1.0
    tmp = pd.DataFrame(
        {"side": side, "arch": arch, "score": score, "selected": selected}
    )
    side_arch_priors: dict[str, dict[str, float]] = {}
    for (side_value, arch_value), group in tmp.groupby(["side", "arch"], dropna=False):
        scores = pd.to_numeric(group["score"], errors="coerce").dropna()
        selected_scores = pd.to_numeric(
            group.loc[group["selected"].astype(bool), "score"], errors="coerce"
        ).dropna()
        cutoff = (
            float(selected_scores.min())
            if len(selected_scores)
            else (float(scores.quantile(0.70)) if len(scores) else global_cutoff)
        )
        mean = float(scores.mean()) if len(scores) else global_mean
        std = float(scores.std()) if len(scores) > 1 else global_std
        if not np.isfinite(std) or std <= 1e-12:
            std = global_std
        side_arch_priors[f"{side_value}|{arch_value}"] = {
            "cutoff": cutoff,
            "mean": mean,
            "std": std,
            "rows": int(len(group)),
        }

    score_edges = [
        float(v)
        for v in np.unique(
            score.dropna()
            .quantile(np.linspace(0.0, 1.0, 6)[1:-1])
            .to_numpy(dtype=np.float64)
        )
    ]
    cutoff_by_row = pd.Series(global_cutoff, index=work.index, dtype=np.float64)
    for idx, key in pd.Series(side + "|" + arch, index=work.index).items():
        prior = side_arch_priors.get(str(key), {})
        cutoff_by_row.loc[idx] = _safe_float(prior.get("cutoff"), global_cutoff)
    margin = score - cutoff_by_row
    margin_edges = [
        float(v)
        for v in np.unique(
            margin.dropna()
            .quantile(np.linspace(0.0, 1.0, 6)[1:-1])
            .to_numpy(dtype=np.float64)
        )
    ]
    rank_band = pd.Series(
        [_quantile_label(score_edges, v, "base_rank_band") for v in score],
        index=work.index,
        dtype=object,
    )
    margin_band = pd.Series(
        [_quantile_label(margin_edges, v, "base_margin_band") for v in margin],
        index=work.index,
        dtype=object,
    )
    stat_work = pd.DataFrame(
        {
            "side": side,
            "arch": arch,
            "rank_band": rank_band,
            "margin_band": margin_band,
            "clean": clean,
            "bad": bad,
            "timeout": timeout,
            "dirty": dirty,
            "exec": exec_margin,
        }
    )
    global_stats = {
        "cutoff": global_cutoff,
        "score_mean": global_mean,
        "score_std": global_std,
        "clean_rate": float(clean.mean()) if len(clean) else 0.0,
        "bad_mae_rate": float(bad.mean()) if len(bad) else 0.0,
        "timeout_rate": float(timeout.mean()) if len(timeout) else 0.0,
        "dirty_positive_rate": float(dirty.mean()) if len(dirty) else 0.0,
        "exec_margin_mean": float(exec_margin.mean()) if len(exec_margin) else 0.0,
    }

    def _aggregate(
        prefix: str,
        band_col: str,
        side_key: str | None = None,
        arch_key: str | None = None,
    ) -> dict[str, dict[str, float]]:
        groups: dict[str, dict[str, float]] = {}
        group_cols = ["side", "arch", band_col]
        for (side_value, arch_value, band_value), group in stat_work.groupby(
            group_cols, dropna=False
        ):
            rows = float(len(group))
            weight = rows / (rows + float(shrinkage_k))
            clean_rate = (
                float(group["clean"].mean()) if rows else global_stats["clean_rate"]
            )
            bad_rate = (
                float(group["bad"].mean()) if rows else global_stats["bad_mae_rate"]
            )
            timeout_rate = (
                float(group["timeout"].mean()) if rows else global_stats["timeout_rate"]
            )
            dirty_rate = (
                float(group["dirty"].mean())
                if rows
                else global_stats["dirty_positive_rate"]
            )
            exec_mean = (
                float(group["exec"].mean())
                if rows
                else global_stats["exec_margin_mean"]
            )
            vals = {
                "rows": rows,
                "clean_rate": weight * clean_rate
                + (1.0 - weight) * global_stats["clean_rate"],
                "bad_mae_rate": weight * bad_rate
                + (1.0 - weight) * global_stats["bad_mae_rate"],
                "timeout_rate": weight * timeout_rate
                + (1.0 - weight) * global_stats["timeout_rate"],
                "dirty_positive_rate": weight * dirty_rate
                + (1.0 - weight) * global_stats["dirty_positive_rate"],
                "exec_margin_mean": weight * exec_mean
                + (1.0 - weight) * global_stats["exec_margin_mean"],
            }
            groups[f"{prefix}|{side_value}|{arch_value}|{band_value}"] = vals
        # Side/global fallbacks by band keep live inference deterministic when a
        # rare derived source tag was not present in the fold training rows.
        for (side_value, band_value), group in stat_work.groupby(
            ["side", band_col], dropna=False
        ):
            rows = float(len(group))
            weight = rows / (rows + float(shrinkage_k))
            groups[f"{prefix}|{side_value}|*|{band_value}"] = {
                "rows": rows,
                "clean_rate": weight * float(group["clean"].mean())
                + (1.0 - weight) * global_stats["clean_rate"],
                "bad_mae_rate": weight * float(group["bad"].mean())
                + (1.0 - weight) * global_stats["bad_mae_rate"],
                "timeout_rate": weight * float(group["timeout"].mean())
                + (1.0 - weight) * global_stats["timeout_rate"],
                "dirty_positive_rate": weight * float(group["dirty"].mean())
                + (1.0 - weight) * global_stats["dirty_positive_rate"],
                "exec_margin_mean": weight * float(group["exec"].mean())
                + (1.0 - weight) * global_stats["exec_margin_mean"],
            }
        for band_value, group in stat_work.groupby(band_col, dropna=False):
            rows = float(len(group))
            weight = rows / (rows + float(shrinkage_k))
            groups[f"{prefix}|*|*|{band_value}"] = {
                "rows": rows,
                "clean_rate": weight * float(group["clean"].mean())
                + (1.0 - weight) * global_stats["clean_rate"],
                "bad_mae_rate": weight * float(group["bad"].mean())
                + (1.0 - weight) * global_stats["bad_mae_rate"],
                "timeout_rate": weight * float(group["timeout"].mean())
                + (1.0 - weight) * global_stats["timeout_rate"],
                "dirty_positive_rate": weight * float(group["dirty"].mean())
                + (1.0 - weight) * global_stats["dirty_positive_rate"],
                "exec_margin_mean": weight * float(group["exec"].mean())
                + (1.0 - weight) * global_stats["exec_margin_mean"],
            }
        return groups

    groups = {}
    groups.update(_aggregate("rel_rankband", "rank_band"))
    groups.update(_aggregate("rel_marginband", "margin_band"))

    source_thresholds: dict[str, dict[str, float]] = {}
    for side_value in sorted(side.dropna().unique()):
        side_thresholds: dict[str, float] = {}
        for bucket in ("top10", "top20", "top30"):
            tag = f"{side_value}__model_frontier_{bucket}"
            vals = score.loc[(side == side_value) & (arch == tag)].dropna()
            if len(vals):
                side_thresholds[bucket] = float(vals.min())
        if side_thresholds:
            source_thresholds[str(side_value)] = side_thresholds

    return {
        "schema": "s52_meta_reliability_priors_v1",
        "rows": int(len(work)),
        "selected_col": str(selected_col),
        "shrinkage_k": float(shrinkage_k),
        "feature_names": list(RELIABILITY_FEATURES),
        "score_quantile_edges": score_edges,
        "score_reference_quantiles": np.quantile(
            score.dropna().to_numpy(dtype=np.float64, copy=False),
            np.linspace(0.0, 1.0, 4097),
        )
        .astype(float)
        .tolist(),
        "margin_quantile_edges": margin_edges,
        "source_tag_score_thresholds": source_thresholds,
        "global_prior": global_stats,
        "side_arch_priors": side_arch_priors,
        "groups": groups,
        "leakage_contract": {
            "fit_scope": "training rows only for the scored OOS fold",
            "oos_usage": "live/OOS rows receive frozen side/archetype/band priors",
            "realized_outcomes": "used only to compute train-side priors, never read from live rows",
        },
    }
