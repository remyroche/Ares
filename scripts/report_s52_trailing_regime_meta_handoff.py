#!/usr/bin/env python3
"""Regime handoff audit for an S52 trailing-profit scored ledger.

The generic regime audit expects the older meta feature ledger contract.  This
script targets the S52 row-level scored ledger produced by
``run_s52_ranker_smoke.py`` and builds the first meta-handoff view for the new
trailing-profit label source:

* candidate regime summaries
* source concentration
* source x regime outcome/path geometry
* source x regime learnability
* fit-month action table with holdout validation

All continuous regime bins are fit on fit months only and then applied to the
holdout month.  Outcomes are used only for reporting and fit-month action
selection; the holdout metrics are validation-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

from extreme_price_movements.features_gmm_ae import (
    AE_GMM_FEATURE_COLUMNS,
    transform_ae_gmm_features,
)
from extreme_price_movements.data_store import _feature_schema_names
from extreme_price_movements.static_feature_store import (
    STATIC_FEATURE_ENDPOINT_VERSION,
    read_static_features,
)


DEFAULT_LEDGER = Path(
    "data_perp/reports/s52_trailing_profit_best_pointwise_scored_ledger_20260705_v1/"
    "s52_ranker_smoke_scored_ledger.parquet"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/s52_trailing_profit_best_pointwise_scored_ledger_20260705_v1/"
    "s52_trailing_regime_meta_handoff_v1"
)
DEFAULT_LABEL_CONTEXT_DIR = Path(
    "data_perp/artifacts/20260705_s52_trailing_tp075_sl050_tr035_fast12_bar30_cost100bps_labels/labels"
)
DEFAULT_FEATURE_DIR = Path("data_perp/features/20260617_090000")
DEFAULT_FIT_MONTHS = ("2026-04", "2026-05")
DEFAULT_HOLDOUT_MONTH = "2026-06"
DEFAULT_SELECTED_COL = "selected_top10"
HANDOFF_RANK_SCOPE = "timestamp_side"
HANDOFF_RANK_SCOPE_COLUMN = "candidate_handoff_rank_scope"
BASE_TARGET_CONTRACT_HASH_COLUMN = "base_target_contract_hash"
BASE_SAMPLE_WEIGHT_SPEC_HASH_COLUMN = "base_sample_weight_spec_hash"
HANDOFF_PROVENANCE_COLUMNS = (
    HANDOFF_RANK_SCOPE_COLUMN,
    BASE_TARGET_CONTRACT_HASH_COLUMN,
    BASE_SAMPLE_WEIGHT_SPEC_HASH_COLUMN,
)
FEATURE_STORE_SCOPES = (
    "cross_market",
    "config_meta_full",
    "all_safe",
    "aegmm_inputs",
)


def _frozen_ae_gmm_input_columns(state_path: Path | None) -> list[str]:
    if state_path is None:
        return []
    state = joblib.load(Path(state_path))
    return [str(column) for column in state.get("feature_columns", [])]


def _append_frozen_ae_gmm_context(
    frame: pd.DataFrame,
    state_path: Path | None,
    *,
    chunk_rows: int = 50_000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if state_path is None:
        return frame, {"status": "not_requested"}
    state_path = Path(state_path)
    state = joblib.load(state_path)
    feature_cols = [str(col) for col in state.get("feature_columns", [])]
    if not feature_cols:
        raise ValueError(f"frozen AE/GMM state has no feature contract: {state_path}")
    probe_values = state.get("cycle_input_fill_values", {})
    probe = pd.DataFrame(
        [[float(probe_values.get(col, 0.0)) for col in feature_cols]],
        columns=feature_cols,
        dtype=np.float32,
    )
    expected_generated = list(
        transform_ae_gmm_features(probe, state, index=probe.index).columns
    )
    available = set(frame.columns)
    present = [col for col in feature_cols if col in available]
    missing = [col for col in feature_cols if col not in available]
    existing_generated = [col for col in expected_generated if col in available]
    existing_complete = np.ones(len(frame), dtype=bool)
    if len(existing_generated) == len(expected_generated):
        for col in expected_generated:
            existing_complete &= pd.to_numeric(
                frame[col], errors="coerce"
            ).notna().to_numpy(dtype=bool, copy=False)
    else:
        existing_complete[:] = False
    existing_complete_rows = int(existing_complete.sum())
    if bool(existing_complete.all()):
        return frame, {
            "status": "existing_frozen_outputs_reused",
            "state_path": str(state_path),
            "cycle_state_hash": state.get("cycle_state_hash"),
            "state_input_features": int(len(feature_cols)),
            "state_input_features_present": int(len(present)),
            "state_input_coverage": float(len(present) / max(len(feature_cols), 1)),
            "missing_state_input_features_not_required_for_reuse": missing,
            "generated_features": int(len(expected_generated)),
            "generated_columns": expected_generated,
            "existing_complete_rows": existing_complete_rows,
            "recomputed_rows": 0,
            "transform_contract": "reuse exact base-emitted outputs from the same frozen cycle state; no handoff recomputation",
        }
    if missing:
        raise ValueError(
            "Cannot recompute frozen AE/GMM context: missing required inputs "
            f"{missing[:20]} (missing={len(missing)}), while only "
            f"{len(existing_generated)}/{len(expected_generated)} generated outputs are present "
            f"and {len(frame) - existing_complete_rows}/{len(frame)} rows have incomplete frozen outputs"
        )
    recompute_positions = np.flatnonzero(~existing_complete)
    cycle_fill = {
        str(col): float(value)
        for col, value in dict(state.get("cycle_input_fill_values", {}) or {}).items()
        if np.isfinite(value)
    }
    generated_parts: list[pd.DataFrame] = []
    for start in range(0, len(recompute_positions), max(int(chunk_rows), 1)):
        positions = recompute_positions[start : start + max(int(chunk_rows), 1)]
        x = frame.iloc[positions].loc[:, feature_cols]
        x = x.apply(pd.to_numeric, errors="coerce").astype(np.float32)
        if cycle_fill:
            x = x.fillna({col: cycle_fill[col] for col in feature_cols if col in cycle_fill})
        generated_parts.append(
            transform_ae_gmm_features(x, state, index=frame.index[positions])
        )
    generated = pd.concat(generated_parts, axis=0)
    out = frame
    for col in generated.columns:
        if col not in out.columns:
            out[col] = np.float32(np.nan)
        out.loc[generated.index, col] = generated[col].to_numpy(
            dtype=np.float32, copy=False
        )
    return out, {
        "status": (
            "existing_frozen_outputs_completed"
            if existing_generated
            else "frozen_state_transformed"
        ),
        "state_path": str(state_path),
        "state_input_features": int(len(feature_cols)),
        "state_input_features_present": int(len(present)),
        "state_input_coverage": float(len(present) / max(len(feature_cols), 1)),
        "missing_state_input_features": missing,
        "generated_features": int(len(generated.columns)),
        "generated_columns": list(generated.columns),
        "existing_complete_rows": existing_complete_rows,
        "recomputed_rows": int(len(recompute_positions)),
        "cycle_state_hash": state.get("cycle_state_hash"),
        "transform_contract": "exact frozen scaler/AE/GMM state with the complete ordered input contract and persisted cycle fill values",
    }

CROSS_MARKET_FEATURE_PREFIXES: tuple[str, ...] = (
    "q_tail_",
    "q_iqr_",
    "q_width_",
    "width_",
    "tail_",
    "asym_",
    "iqr_",
    "pct_assets_",
    "cs_",
    "cs_rank_",
    "btc_",
    "eth_",
    "eth_btc_",
    "xs_dispersion_",
    "trend_dispersion_",
    "spectral_",
    "state_spectral_",
    "xasset_",
    "mkt_",
    "eig_",
    "market_breadth_",
    "market_dispersion_",
    "xasset_mkt_",
    "market_index_",
    "cross_asset_",
    "median_asset_",
    "top_decile_asset_",
    "cross_asset_correlation_",
    "avg_pairwise_corr_",
)


REGIME_SPECS: tuple[tuple[str, str, str], ...] = (
    ("base_score_decile", "score", "quantile10"),
    ("aegmm_cluster", "gmm_cluster_id", "category"),
    ("aegmm_entropy_bin", "gmm_entropy", "quantile4"),
    ("aegmm_distance_bin", "mahalanobis_distance", "quantile4"),
    ("aegmm_expected_distance_bin", "expected_mahalanobis", "quantile4"),
    ("reconstruction_bin", "AE_reconstruction_error", "quantile4"),
    ("dae_reconstruction_bin", "dae_reconstruction_error", "quantile4"),
    ("cluster_speed_bin", "cluster_speed", "quantile4"),
    ("cluster_acceleration_bin", "cluster_acceleration", "quantile4"),
    ("latent_speed_bin", "latent_speed", "quantile4"),
    ("latent_acceleration_bin", "latent_acceleration", "quantile4"),
)

SUPERVISED_RISK_TARGETS: tuple[tuple[str, str], ...] = (
    ("regime_bad_mae_score_bin", "full_path_bad_mae_1r"),
    ("regime_first_touch_bad_mae_score_bin", "first_touch_bad_mae_1r"),
    ("regime_timeout_score_bin", "timeout"),
    ("regime_dirty_positive_score_bin", "dirty_positive"),
    ("regime_clean_exec_score_bin", "clean_exec"),
    ("regime_exec_margin_score_bin", "exec_margin"),
    ("regime_ev_score_bin", "ev_after_1pct"),
)

LGBM_LEAF_TARGETS: tuple[tuple[str, str], ...] = (
    ("regime_lgbm_leaf_clean_exec_k4", "clean_exec"),
    ("regime_lgbm_leaf_bad_mae_k4", "full_path_bad_mae_1r"),
    ("regime_lgbm_leaf_exec_margin_k4", "exec_margin"),
)

EXECUTION_POLICY_MENU: tuple[dict[str, Any], ...] = (
    {
        "policy": "P0_abstain",
        "kind": "abstain",
        "tp_r": 0.0,
        "sl_r": 0.0,
        "trail_start_r": 0.0,
        "trail_gap_r": 0.0,
    },
    {
        "policy": "P1_tight_scalp_tp075_sl050",
        "kind": "fixed_tp_sl",
        "tp_r": 0.75,
        "sl_r": 0.50,
        "trail_start_r": 0.0,
        "trail_gap_r": 0.0,
    },
    {
        "policy": "P2_fast_impulse_tp100_sl050",
        "kind": "fixed_tp_sl",
        "tp_r": 1.00,
        "sl_r": 0.50,
        "trail_start_r": 0.0,
        "trail_gap_r": 0.0,
    },
    {
        "policy": "P3_standard_tp125_sl075",
        "kind": "fixed_tp_sl",
        "tp_r": 1.25,
        "sl_r": 0.75,
        "trail_start_r": 0.0,
        "trail_gap_r": 0.0,
    },
    {
        "policy": "P4_trailing_runner_start100_gap050",
        "kind": "trailing_proxy",
        "tp_r": 0.0,
        "sl_r": 0.75,
        "trail_start_r": 1.00,
        "trail_gap_r": 0.50,
    },
    {
        "policy": "P5_wide_runner_tp200_sl100",
        "kind": "fixed_tp_sl",
        "tp_r": 2.00,
        "sl_r": 1.00,
        "trail_start_r": 0.0,
        "trail_gap_r": 0.0,
    },
)

SUPERVISED_RISK_FEATURES: tuple[str, ...] = (
    "score",
    "gmm_entropy",
    "cluster_entropy",
    "mahalanobis_distance",
    "expected_mahalanobis",
    "AE_reconstruction_error",
    "dae_reconstruction_error",
    "dae_reconstruction_error_zscore",
    "cluster_speed",
    "cluster_acceleration",
    "latent_mahalanobis_drift",
    "latent_speed",
    "latent_acceleration",
    "base_rank_pct_by_timestamp",
    "base_rank_pct_by_timestamp_side",
    "base_score_z_by_timestamp",
    "base_score_z_by_timestamp_side",
    "base_margin_to_cutoff",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _contract_hash(payload: Any) -> str:
    """Return a stable hash for a serializable model-contract payload."""

    encoded = json.dumps(
        _json_safe(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _contract_value_from_rows(
    frame: pd.DataFrame,
    candidates: Iterable[str],
) -> tuple[Any | None, str | None, list[str]]:
    """Resolve one stable JSON-like contract from compatible source columns."""

    values: list[Any] = []
    sources: list[str] = []
    for column in candidates:
        if column not in frame.columns:
            continue
        series = frame[column].dropna()
        for raw in series.astype(str):
            text = raw.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                value = text
            values.append(value)
            sources.append(str(column))
    if not values:
        return None, None, []
    by_hash: dict[str, Any] = {}
    for value in values:
        by_hash.setdefault(_contract_hash(value), value)
    if len(by_hash) != 1:
        return None, "mixed_row_contract_values", sorted(set(sources))
    return next(iter(by_hash.values())), None, sorted(set(sources))


def _inherited_base_contract(
    ledger: pd.DataFrame,
    *,
    strict: bool,
) -> dict[str, Any]:
    """Build explicit base target/weight provenance for the meta handoff.

    New base artifacts carry the two serialized contracts and hashes.  Legacy
    ledgers can still be materialized for diagnostics, but strict consumers
    must reject their derived fallback contract.
    """

    rank_scope_values = (
        ledger.get(HANDOFF_RANK_SCOPE_COLUMN, pd.Series(dtype="object"))
        .dropna()
        .astype(str)
        .str.strip()
    )
    unique_scopes = sorted(value for value in rank_scope_values.unique() if value)
    if unique_scopes and unique_scopes != [HANDOFF_RANK_SCOPE]:
        raise ValueError(
            "candidate_handoff_rank_scope must be timestamp_side; "
            f"found {unique_scopes}"
        )
    if strict and unique_scopes != [HANDOFF_RANK_SCOPE]:
        raise ValueError(
            "Strict meta handoff requires candidate_handoff_rank_scope="
            "timestamp_side on every base candidate row."
        )

    target_contract, target_error, target_sources = _contract_value_from_rows(
        ledger,
        (
            "base_target_contract_json",
            "base_target_contract",
            "target_contract_json",
        ),
    )
    weight_spec, weight_error, weight_sources = _contract_value_from_rows(
        ledger,
        (
            "base_sample_weight_spec_json",
            "base_sample_weight_spec",
            "sample_weight_spec_json",
        ),
    )
    target_mode = sorted(
        value
        for value in ledger.get("base_model_target_mode", pd.Series(dtype="object"))
        .dropna()
        .astype(str)
        .unique()
        if value
    )
    weight_arm = sorted(
        value
        for value in ledger.get("base_model_weight_arm", pd.Series(dtype="object"))
        .dropna()
        .astype(str)
        .unique()
        if value
    )
    explicit = target_contract is not None and weight_spec is not None
    if target_error or weight_error:
        if strict:
            raise ValueError(
                "Strict meta handoff cannot use mixed base provenance: "
                f"target={target_error}, weight={weight_error}."
            )
    if target_contract is None:
        target_contract = {
            "schema": "base_soft_label_contract_v1",
            "target_column": "__first_touch_target_soft__",
            "base_model_target_mode": target_mode,
            "provenance": "derived_legacy_default",
        }
    if weight_spec is None:
        weight_spec = {
            "schema": "target_strength_weight_v1",
            "spec": {},
            "base_model_weight_arm": weight_arm,
            "provenance": "derived_legacy_default",
        }
    if strict and not explicit:
        raise ValueError(
            "Strict meta handoff requires explicit base_target_contract and "
            "base_sample_weight_spec values from the base artifact."
        )
    target_hash = _contract_hash(target_contract)
    weight_hash = _contract_hash(weight_spec)
    source_target_hashes = sorted(
        value
        for value in ledger.get(
            BASE_TARGET_CONTRACT_HASH_COLUMN, pd.Series(dtype="object")
        )
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        if value
    )
    source_weight_hashes = sorted(
        value
        for value in ledger.get(
            BASE_SAMPLE_WEIGHT_SPEC_HASH_COLUMN, pd.Series(dtype="object")
        )
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        if value
    )
    if strict and (
        source_target_hashes != [target_hash]
        or source_weight_hashes != [weight_hash]
    ):
        raise ValueError(
            "Strict meta handoff requires one matching base contract hash on "
            "every candidate row; "
            f"target={source_target_hashes}, weight={source_weight_hashes}."
        )
    return {
        "schema": "base_to_meta_inherited_contract_v1",
        "candidate_handoff_rank_scope": HANDOFF_RANK_SCOPE,
        "rank_scope_status": "explicit" if unique_scopes else "derived_legacy_default",
        "base_target_contract": target_contract,
        "base_target_contract_hash": target_hash,
        "base_sample_weight_spec": weight_spec,
        "base_sample_weight_spec_hash": weight_hash,
        "explicit_base_contract": bool(explicit),
        "target_contract_source_columns": target_sources,
        "sample_weight_spec_source_columns": weight_sources,
        "target_contract_resolution_error": target_error,
        "sample_weight_spec_resolution_error": weight_error,
        "source_base_target_contract_hashes": source_target_hashes,
        "source_base_sample_weight_spec_hashes": source_weight_hashes,
        "strict": bool(strict),
    }


def _materialize_promoted_base_contract(ledger: pd.DataFrame) -> pd.DataFrame:
    """Upgrade completed base ledgers without rerunning model scoring.

    This is deterministic contract materialization from the uniform target mode
    and weight arm already recorded on every scored row.  It changes no score,
    target, rank, or candidate decision.
    """

    modes = set(
        ledger.get("base_model_target_mode", pd.Series(dtype="object"))
        .dropna()
        .astype(str)
    )
    weight_arms = set(
        ledger.get("base_model_weight_arm", pd.Series(dtype="object"))
        .dropna()
        .astype(str)
    )
    if len(modes) != 1 or len(weight_arms) != 1:
        return ledger
    required = {
        "base_target_contract_json",
        "base_sample_weight_spec_json",
        BASE_TARGET_CONTRACT_HASH_COLUMN,
        BASE_SAMPLE_WEIGHT_SPEC_HASH_COLUMN,
    }
    if required.issubset(ledger.columns):
        return ledger
    out = ledger.copy(deep=False)
    target_mode = next(iter(modes))
    weight_arm = next(iter(weight_arms))
    if target_mode == "side_continuous_geometry_v1":
        from extreme_price_movements.base_side_target_contract import (
            build_promoted_side_target,
            promoted_side_target_provenance,
        )

        promoted = build_promoted_side_target(out)
        out["__first_touch_target_soft__"] = promoted["target_soft"].to_numpy(
            dtype=np.float32, copy=False
        )
        provenance = promoted_side_target_provenance()
    else:
        target_contract = {
            "schema": "base_soft_label_contract_v1",
            "target_column": "__first_touch_target_soft__",
            "target_mode": target_mode,
            "source": "base_scoring_target_from_frame",
        }
        weight_spec = {
            "schema": "base_weight_arm_v1",
            "weight_arm": weight_arm,
            "source": "base_weight_series",
        }
        provenance = {
            "base_target_contract": target_contract,
            BASE_TARGET_CONTRACT_HASH_COLUMN: _contract_hash(target_contract),
            "base_sample_weight_spec": weight_spec,
            BASE_SAMPLE_WEIGHT_SPEC_HASH_COLUMN: _contract_hash(weight_spec),
        }
    out["base_target_contract_json"] = json.dumps(
        provenance["base_target_contract"], sort_keys=True, separators=(",", ":")
    )
    out["base_sample_weight_spec_json"] = json.dumps(
        provenance["base_sample_weight_spec"], sort_keys=True, separators=(",", ":")
    )
    out[BASE_TARGET_CONTRACT_HASH_COLUMN] = provenance[BASE_TARGET_CONTRACT_HASH_COLUMN]
    out[BASE_SAMPLE_WEIGHT_SPEC_HASH_COLUMN] = provenance[
        BASE_SAMPLE_WEIGHT_SPEC_HASH_COLUMN
    ]
    return out


def _parse_csv(value: str | None, default: Iterable[str]) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return list(default)
    return [part.strip() for part in text.split(",") if part.strip()]


def _num(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col in frame.columns:
        return pd.to_numeric(frame[col], errors="coerce")
    return pd.Series(float(default), index=frame.index, dtype=np.float64)


def _first_numeric(frame: pd.DataFrame, cols: Iterable[str], default: float = np.nan) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype=np.float64)
    for col in cols:
        if col not in frame.columns:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        out = out.where(out.notna(), values)
    if not pd.isna(default):
        out = out.fillna(float(default))
    return out


def _rate(values: Any) -> float:
    ser = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(ser.clip(0.0, 1.0).mean()) if len(ser) else float("nan")


def _mean(values: Any) -> float:
    ser = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(ser.mean()) if len(ser) else float("nan")


def _q(values: Any, q: float) -> float:
    ser = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(ser.quantile(float(q))) if len(ser) else float("nan")


def _spearman(x: Any, y: Any) -> float:
    xs = pd.to_numeric(pd.Series(x), errors="coerce")
    ys = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = xs.notna() & ys.notna()
    if int(mask.sum()) < 20:
        return float("nan")
    return float(xs[mask].rank().corr(ys[mask].rank()))


def _auc(y_true: Any, score: Any) -> float:
    y = pd.to_numeric(pd.Series(y_true), errors="coerce")
    s = pd.to_numeric(pd.Series(score), errors="coerce")
    mask = y.notna() & s.notna()
    y = y[mask].gt(0.5).astype(int)
    s = s[mask]
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = s.rank(method="average")
    auc = (float(ranks[y.eq(1)].sum()) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _logloss(y_true: Any, prob: Any) -> float:
    y = pd.to_numeric(pd.Series(y_true), errors="coerce")
    p = pd.to_numeric(pd.Series(prob), errors="coerce")
    mask = y.notna() & p.notna()
    if int(mask.sum()) == 0:
        return float("nan")
    yy = y[mask].clip(0.0, 1.0)
    pp = p[mask].clip(1e-6, 1.0 - 1e-6)
    return float(-(yy * np.log(pp) + (1.0 - yy) * np.log(1.0 - pp)).mean())


def _top_mask(score: pd.Series, frac: float) -> pd.Series:
    score = pd.to_numeric(score, errors="coerce")
    out = pd.Series(False, index=score.index)
    valid = score.dropna()
    if valid.empty:
        return out
    keep = max(1, int(math.ceil(len(valid) * float(frac))))
    out.loc[valid.sort_values(ascending=False).head(keep).index] = True
    return out


def _quantile_edges(values: pd.Series, q: int) -> np.ndarray:
    vals = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.nunique() < 2:
        return np.array([], dtype=np.float64)
    qs = np.linspace(0.0, 1.0, int(q) + 1)
    edges = np.unique(np.nanquantile(vals.to_numpy(dtype=np.float64), qs))
    if len(edges) < 3:
        return np.array([], dtype=np.float64)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges.astype(np.float64)


def _apply_edges(values: pd.Series, edges: np.ndarray, name: str) -> pd.Series:
    if edges.size < 3:
        return pd.Series(f"{name}__all", index=values.index, dtype="object")
    labels = [f"{name}__q{i}" for i in range(len(edges) - 1)]
    numeric = pd.to_numeric(values, errors="coerce").astype("float64")
    binned = pd.cut(numeric, bins=edges.astype("float64", copy=False), labels=labels, include_lowest=True)
    return binned.astype("object").fillna(f"{name}__missing")


def _supervised_risk_feature_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    cols = [col for col in SUPERVISED_RISK_FEATURES if col in frame.columns]
    posterior_cols = sorted(
        col for col in frame.columns if col.startswith("gmm_cluster_posterior_") or col.startswith("posterior_")
    )
    cols.extend(col for col in posterior_cols if col not in cols)
    data: dict[str, pd.Series] = {}
    for col in cols:
        data[col] = pd.to_numeric(frame[col], errors="coerce")
    if "side_name" in frame.columns:
        side = frame["side_name"].astype(str).str.lower()
        data["side_is_long"] = side.eq("long").astype(float)
        data["side_is_short"] = side.eq("short").astype(float)
    features = pd.DataFrame(data, index=frame.index)
    usable_cols = [col for col in features.columns if features[col].notna().any()]
    return features[usable_cols].astype(np.float32), usable_cols


def _clean_token(value: Any) -> str:
    text = str(value).strip().lower()
    out = []
    for ch in text:
        out.append(ch if ch.isalnum() else "_")
    token = "".join(out).strip("_")
    while "__" in token:
        token = token.replace("__", "_")
    return token or "missing"


def _join_label_context(ledger: pd.DataFrame, label_context_dir: Path | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join pre-entry label context columns that were not carried into the scored ledger."""

    if label_context_dir is None or not Path(label_context_dir).exists():
        return ledger, {
            "label_context_status": "missing",
            "label_context_dir": str(label_context_dir) if label_context_dir is not None else None,
        }
    parquet_files = sorted(Path(label_context_dir).glob("train_*.parquet"))
    if not parquet_files:
        return ledger, {"label_context_status": "missing_parquet", "label_context_dir": str(label_context_dir)}
    key_cols = ["__ts__", "__symbol__", "side_name"]
    context_cols = [
        "__regime_vol_12h__",
        "__regime_vol_48h__",
        "__regime_volume_12h__",
        "__regime_volume_48h__",
        "__regime_trend_12h__",
        "__regime_trend_48h__",
        "__meta_raw__volatility_zscore",
        "__meta_raw__asset_minus_mkt_oi_1d_peer_resid",
        "__meta_raw__return_autocorr_48",
        "G_VOL",
    ]
    parts: list[pd.DataFrame] = []
    loaded_cols: set[str] = set()
    for path in parquet_files:
        schema_cols = set(pd.read_parquet(path, columns=None).columns)
        cols = [col for col in key_cols + context_cols if col in schema_cols]
        if not set(key_cols).issubset(cols):
            continue
        part = pd.read_parquet(path, columns=cols)
        part["source_recipe_name"] = path.stem
        parts.append(part)
        loaded_cols.update(cols)
    if not parts:
        return ledger, {"label_context_status": "no_joinable_files", "label_context_dir": str(label_context_dir)}
    context = pd.concat(parts, ignore_index=True)
    # Keep exact point-in-time keys comparable regardless of whether a source
    # parquet persisted timezone metadata. No rounding or asof matching is used.
    ledger = ledger.copy()
    ledger["__ts__"] = pd.to_datetime(ledger["__ts__"], errors="coerce", utc=True)
    context["__ts__"] = pd.to_datetime(context["__ts__"], errors="coerce", utc=True)
    context = context.drop_duplicates(key_cols, keep="first")
    before = len(ledger)
    out = ledger.merge(context, on=key_cols, how="left", validate="one_to_one")
    matched = int(out["source_recipe_name"].notna().sum()) if "source_recipe_name" in out.columns else 0
    return out, {
        "label_context_status": "joined",
        "label_context_dir": str(label_context_dir),
        "label_context_files": [path.name for path in parquet_files],
        "loaded_columns": sorted(loaded_cols),
        "rows_before": int(before),
        "rows_after": int(len(out)),
        "matched_rows": matched,
        "match_rate": matched / max(before, 1),
    }


def _materialize_label_path_end(
    ledger: pd.DataFrame,
    label_context_dir: Path | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Persist the exact time at which every forward label is fully observable.

    The soft target uses full-path ordering diagnostics, so an early TP/SL touch
    does not make the label available early.  Resolution is therefore the close
    of the final configured path bar, not ``__first_touch_bar__``.
    """

    if label_context_dir is None:
        raise ValueError("Label-path resolution requires label_context_dir")
    summary_path = Path(label_context_dir) / "side_archetype_trailing_materialization_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing label-path materialization summary: {summary_path}"
        )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    datasets = payload.get("datasets", []) if isinstance(payload, dict) else []
    contracts = {
        (
            int(row.get("path_fetch", {}).get("path_len")),
            str(row.get("path_fetch", {}).get("path_timeframe")),
        )
        for row in datasets
        if isinstance(row, dict)
        and row.get("path_fetch", {}).get("path_len") is not None
        and row.get("path_fetch", {}).get("path_timeframe")
    }
    if len(contracts) != 1:
        raise ValueError(
            "Label datasets must share one path resolution contract; "
            f"found {sorted(contracts)}"
        )
    path_len, path_timeframe = next(iter(contracts))
    if path_len <= 0:
        raise ValueError(f"Invalid label path_len={path_len}")
    try:
        path_bar_delta = pd.Timedelta(path_timeframe)
    except ValueError as exc:
        raise ValueError(
            f"Invalid label path_timeframe={path_timeframe!r}"
        ) from exc
    if path_bar_delta <= pd.Timedelta(0):
        raise ValueError(f"Invalid label path delta={path_bar_delta}")
    if "__first_path_ts__" not in ledger.columns:
        raise ValueError("Scored ledger is missing __first_path_ts__")
    first_path_ts = pd.to_datetime(
        ledger["__first_path_ts__"], utc=True, errors="coerce"
    )
    if first_path_ts.isna().any():
        raise ValueError(
            "Scored ledger contains non-finite __first_path_ts__ rows: "
            f"{int(first_path_ts.isna().sum())}"
        )
    out = ledger.copy()
    out["__first_path_ts__"] = first_path_ts
    if "__decision_ts__" in out.columns:
        out["__decision_ts__"] = pd.to_datetime(
            out["__decision_ts__"], utc=True, errors="coerce"
        )
    label_horizon = path_bar_delta * int(path_len)
    out["__label_path_end_ts__"] = first_path_ts + label_horizon
    return out, {
        "schema": "forward_label_resolution_v1",
        "source": str(summary_path),
        "path_len": int(path_len),
        "path_timeframe": str(path_timeframe),
        "path_bar_seconds": float(path_bar_delta.total_seconds()),
        "label_horizon_seconds": float(label_horizon.total_seconds()),
        "resolution_column": "__label_path_end_ts__",
        "resolution_rule": "__first_path_ts__ + path_len * path_timeframe",
        "rows": int(len(out)),
        "missing_resolution_rows": 0,
    }


def _add_supervised_risk_regimes(
    out: pd.DataFrame,
    *,
    fit_mask: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    specs: dict[str, Any] = {}
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
    except Exception as exc:  # pragma: no cover - dependency fallback
        specs["supervised_risk_score_status"] = {
            "kind": "skipped",
            "reason": f"sklearn_unavailable:{type(exc).__name__}",
        }
        return out, specs

    features, feature_cols = _supervised_risk_feature_frame(out)
    if len(feature_cols) < 3:
        specs["supervised_risk_score_status"] = {
            "kind": "skipped",
            "reason": "insufficient_live_feature_columns",
            "feature_columns": feature_cols,
        }
        return out, specs

    fit_index = out.index[fit_mask]
    x_fit = features.loc[fit_index]
    for regime_name, target_col in SUPERVISED_RISK_TARGETS:
        if target_col not in out.columns:
            continue
        y_fit = pd.to_numeric(out.loc[fit_index, target_col], errors="coerce")
        valid = y_fit.notna()
        if int(valid.sum()) < 500 or y_fit.loc[valid].nunique() < 2:
            specs[regime_name] = {
                "source_column": target_col,
                "kind": "supervised_risk_score_quantile4",
                "status": "skipped",
                "reason": "insufficient_fit_target_support",
                "fit_non_null": int(valid.sum()),
            }
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                max_iter=64,
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                min_samples_leaf=80,
                random_state=17,
            ),
        )
        model.fit(x_fit.loc[valid], y_fit.loc[valid].astype(float).to_numpy())
        score_col = regime_name.replace("_bin", "")
        preds = pd.Series(model.predict(features), index=out.index, dtype=np.float64)
        if target_col not in ("exec_margin", "ev_after_1pct"):
            preds = preds.clip(0.0, 1.0)
        out[score_col] = preds.astype(np.float32)
        edges = _quantile_edges(out.loc[fit_mask, score_col], 4)
        out[regime_name] = _apply_edges(out[score_col], edges, regime_name)
        specs[regime_name] = {
            "source_column": score_col,
            "target_column": target_col,
            "kind": "supervised_risk_score_quantile4",
            "model": "HistGradientBoostingRegressor",
            "fit_rows": int(valid.sum()),
            "feature_columns": feature_cols,
            "edges": edges.tolist(),
        }
    return out, specs


def _add_lgbm_leaf_regimes(
    out: pd.DataFrame,
    *,
    fit_mask: pd.Series,
    max_fit_rows: int = 50_000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    specs: dict[str, Any] = {}
    try:
        from lightgbm import LGBMRegressor
        from sklearn.cluster import MiniBatchKMeans
    except Exception as exc:  # pragma: no cover - dependency fallback
        specs["lgbm_leaf_regime_status"] = {
            "kind": "skipped",
            "reason": f"dependency_unavailable:{type(exc).__name__}",
        }
        return out, specs

    features, feature_cols = _supervised_risk_feature_frame(out)
    if len(feature_cols) < 3:
        specs["lgbm_leaf_regime_status"] = {
            "kind": "skipped",
            "reason": "insufficient_live_feature_columns",
            "feature_columns": feature_cols,
        }
        return out, specs

    fit_index = out.index[fit_mask]
    if len(fit_index) > int(max_fit_rows):
        rng = np.random.default_rng(17)
        fit_sample_index = pd.Index(rng.choice(fit_index.to_numpy(), size=int(max_fit_rows), replace=False))
    else:
        fit_sample_index = fit_index
    x_fit = features.loc[fit_sample_index]

    for regime_name, target_col in LGBM_LEAF_TARGETS:
        if target_col not in out.columns:
            continue
        y_fit = pd.to_numeric(out.loc[fit_sample_index, target_col], errors="coerce")
        valid = y_fit.notna()
        if int(valid.sum()) < 500 or y_fit.loc[valid].nunique() < 2:
            specs[regime_name] = {
                "source_column": target_col,
                "kind": "lgbm_leaf_embedding_kmeans4",
                "status": "skipped",
                "reason": "insufficient_fit_target_support",
                "fit_non_null": int(valid.sum()),
            }
            continue
        model = LGBMRegressor(
            n_estimators=48,
            num_leaves=16,
            learning_rate=0.05,
            min_child_samples=100,
            subsample=0.80,
            colsample_bytree=0.80,
            reg_lambda=2.0,
            random_state=17,
            n_jobs=1,
            verbosity=-1,
        )
        model.fit(x_fit.loc[valid], y_fit.loc[valid].astype(float).to_numpy())
        leaf_all = model.predict(features, pred_leaf=True)
        leaf_fit = pd.DataFrame(leaf_all, index=out.index).loc[fit_index].to_numpy(dtype=np.float32)
        if len(leaf_fit) > int(max_fit_rows):
            leaf_fit_for_cluster = pd.DataFrame(leaf_all, index=out.index).loc[fit_sample_index].to_numpy(dtype=np.float32)
        else:
            leaf_fit_for_cluster = leaf_fit
        clusterer = MiniBatchKMeans(
            n_clusters=4,
            random_state=17,
            n_init=5,
            batch_size=4096,
            reassignment_ratio=0.01,
        )
        clusterer.fit(leaf_fit_for_cluster)
        labels = clusterer.predict(np.asarray(leaf_all, dtype=np.float32))
        out[regime_name] = pd.Series(labels, index=out.index).map(lambda value: f"{regime_name}__c{int(value)}")
        specs[regime_name] = {
            "source_column": "lgbm_leaf_indices",
            "target_column": target_col,
            "kind": "lgbm_leaf_embedding_kmeans4",
            "model": "LGBMRegressor",
            "clusterer": "MiniBatchKMeans",
            "fit_rows": int(valid.sum()),
            "cluster_fit_rows": int(len(leaf_fit_for_cluster)),
            "n_leaf_features": int(np.asarray(leaf_all).shape[1]) if np.asarray(leaf_all).ndim == 2 else 1,
            "feature_columns": feature_cols,
        }
    return out, specs


def _build_regime_columns(
    ledger: pd.DataFrame,
    *,
    fit_mask: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    # Regime construction overwrites derived columns which may already exist
    # in a saved handoff. Keep this isolated: frozen-model reconstruction must
    # not mutate the caller's source blocks through pandas copy-on-write
    # behavior, otherwise the supervised regime refit can drift from its
    # serialized training contract.
    out = ledger.copy()
    specs: dict[str, Any] = {}
    for regime_name, col, kind in REGIME_SPECS:
        if col not in out.columns:
            continue
        if kind == "category":
            out[regime_name] = out[col].where(out[col].notna(), "missing").astype(str).map(
                lambda value: f"{regime_name}__{value}"
            )
            specs[regime_name] = {"source_column": col, "kind": kind}
            continue
        q = int(kind.replace("quantile", ""))
        edges = _quantile_edges(out.loc[fit_mask, col], q)
        out[regime_name] = _apply_edges(out[col], edges, regime_name)
        specs[regime_name] = {
            "source_column": col,
            "kind": kind,
            "edges": edges.tolist(),
            "fit_non_null": int(pd.to_numeric(out.loc[fit_mask, col], errors="coerce").notna().sum()),
        }
    if "side_name" in out.columns and "gmm_cluster_id" in out.columns:
        out["side_aegmm_cluster"] = (
            out["side_name"].astype(str) + "__gmm_" + out["gmm_cluster_id"].fillna("missing").astype(str)
        )
        specs["side_aegmm_cluster"] = {"source_column": "side_name+gmm_cluster_id", "kind": "category"}
    if "__ts__" in out.columns:
        ts = pd.to_datetime(out["__ts__"], errors="coerce")
        month_value = ts.dt.strftime("%Y-%m").fillna("missing")
        week_start = (
            ts.dt.normalize() - pd.to_timedelta(ts.dt.dayofweek.fillna(0), unit="D")
        ).dt.strftime("%Y-%m-%d").fillna("missing")
        weekday_value = ts.dt.dayofweek.fillna(-1).astype(np.int8).astype(str)
        out["calendar_month_regime"] = pd.Categorical("calendar_month__" + month_value)
        out["calendar_week_regime"] = pd.Categorical("calendar_week__" + week_start)
        out["calendar_weekday_regime"] = pd.Categorical("weekday__" + weekday_value)
        hour = ts.dt.hour
        session = np.select(
            [hour.between(0, 5), hour.between(6, 11), hour.between(12, 17), hour.between(18, 23)],
            ["asia_late", "europe_open", "us_open", "us_late"],
            default="missing",
        )
        out["calendar_session_regime"] = pd.Categorical(
            "session__" + pd.Series(session, index=out.index, dtype="string").fillna("missing")
        )
        specs["calendar_month_regime"] = {"source_column": "__ts__", "kind": "calendar_month"}
        specs["calendar_week_regime"] = {"source_column": "__ts__", "kind": "calendar_week"}
        specs["calendar_weekday_regime"] = {"source_column": "__ts__", "kind": "calendar_weekday"}
        specs["calendar_session_regime"] = {"source_column": "__ts__", "kind": "calendar_session_utc"}
    structural_cols = [col for col in ("score", "gmm_entropy", "mahalanobis_distance", "AE_reconstruction_error") if col in out.columns]
    if "__ts__" in out.columns and structural_cols:
        ts = pd.to_datetime(out["__ts__"], errors="coerce")
        day = ts.dt.floor("D")
        daily = out.loc[fit_mask, structural_cols].apply(pd.to_numeric, errors="coerce").groupby(day[fit_mask]).median()
        if len(daily) >= 5:
            z = (daily - daily.median()) / daily.std(ddof=0).replace(0.0, np.nan)
            stress = z.abs().mean(axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            edges = _quantile_edges(stress, 3)
            fit_day_bucket = _apply_edges(stress, edges, "structural_change_point").to_dict()
            all_days = pd.Series(day, index=out.index)
            out["structural_change_point_regime"] = all_days.map(fit_day_bucket).fillna("structural_change_point__post_fit")
            specs["structural_change_point_regime"] = {
                "source_column": ",".join(structural_cols),
                "kind": "fit_month_daily_structural_stress_quantile3",
                "edges": edges.tolist(),
                "fit_days": int(len(daily)),
            }
    out, supervised_specs = _add_supervised_risk_regimes(out, fit_mask=fit_mask)
    specs.update(supervised_specs)
    out, leaf_specs = _add_lgbm_leaf_regimes(out, fit_mask=fit_mask)
    specs.update(leaf_specs)
    return out, specs


def _fit_quantile_threshold(values: pd.Series, q: float, default: float) -> float:
    vals = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(vals) < 20 or vals.nunique(dropna=True) < 2:
        return float(default)
    out = float(vals.quantile(float(q)))
    return out if math.isfinite(out) else float(default)


def _refine_long_source_families(
    out: pd.DataFrame,
    *,
    fit_mask: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Split broad long mixed/run-entry families using pre-entry AE/GMM/context state."""

    if "side_name" not in out.columns or "source_semantic_family" not in out.columns:
        return out, {"long_refinement_status": "skipped_missing_side_or_source_family"}
    refined = out["source_semantic_family"].astype(str).copy()
    out["source_semantic_family_base"] = refined
    side = out["side_name"].astype(str).str.lower()
    long_mask = side.eq("long")
    broad = refined.isin(["ambiguous_none", "run_entry", "mixed", "quiet_continuation", "late_run_continuation"])
    fit_long = fit_mask & long_mask
    if int(fit_long.sum()) < 100:
        out["long_source_regime_split"] = np.where(long_mask, refined, "not_long")
        return out, {
            "long_refinement_status": "skipped_insufficient_fit_long_rows",
            "fit_long_rows": int(fit_long.sum()),
        }

    vol = _num(out, "__meta_raw__volatility_zscore", 0.0)
    oi = _num(out, "__meta_raw__asset_minus_mkt_oi_1d_peer_resid", 0.0)
    ret_auto = _num(out, "__meta_raw__return_autocorr_48", 0.0)
    entropy = _num(out, "gmm_entropy", 0.0)
    mahal = _num(out, "mahalanobis_distance", 0.0)
    expected_mahal = _num(out, "expected_mahalanobis", 0.0)
    ae = _num(out, "AE_reconstruction_error", 0.0)
    latent_speed = _num(out, "latent_speed", 0.0).abs()
    cluster_speed = _num(out, "cluster_speed", 0.0).abs()
    accel = _num(out, "cluster_acceleration", 0.0).abs()

    thresholds = {
        "vol_low": _fit_quantile_threshold(vol.loc[fit_long], 0.35, 0.0),
        "vol_high": _fit_quantile_threshold(vol.loc[fit_long], 0.75, 0.75),
        "ret_low": _fit_quantile_threshold(ret_auto.loc[fit_long], 0.30, -0.25),
        "ret_high": _fit_quantile_threshold(ret_auto.loc[fit_long], 0.70, 0.25),
        "entropy_high": _fit_quantile_threshold(entropy.loc[fit_long], 0.70, 0.75),
        "mahal_high": _fit_quantile_threshold(mahal.loc[fit_long], 0.75, 2.0),
        "expected_mahal_high": _fit_quantile_threshold(expected_mahal.loc[fit_long], 0.75, 2.0),
        "ae_high": _fit_quantile_threshold(ae.loc[fit_long], 0.75, 1.0),
        "speed_high": _fit_quantile_threshold((latent_speed + cluster_speed).loc[fit_long], 0.70, 1.0),
        "accel_high": _fit_quantile_threshold(accel.loc[fit_long], 0.70, 1.0),
        "oi_abs_high": _fit_quantile_threshold(oi.loc[fit_long].abs(), 0.75, 1.0),
    }
    state_pressure = (mahal.ge(thresholds["mahal_high"]) | expected_mahal.ge(thresholds["expected_mahal_high"]))
    noisy = (
        vol.ge(thresholds["vol_high"])
        | entropy.ge(thresholds["entropy_high"])
        | ae.ge(thresholds["ae_high"])
        | state_pressure
        | accel.ge(thresholds["accel_high"])
    )
    quiet = (
        vol.le(thresholds["vol_low"])
        & ret_auto.abs().le(max(abs(thresholds["ret_high"]), abs(thresholds["ret_low"])))
        & ~state_pressure
        & entropy.lt(thresholds["entropy_high"])
    )
    trend_pullback = (
        ret_auto.ge(thresholds["ret_high"])
        & ~noisy
        & (latent_speed + cluster_speed).ge(thresholds["speed_high"] * 0.50)
    ) | (
        refined.eq("late_run_continuation")
        & ~noisy
        & ret_auto.ge(thresholds["ret_high"])
    )
    liquidity_stress = oi.abs().ge(thresholds["oi_abs_high"]) & (vol.ge(thresholds["vol_high"]) | state_pressure)
    noisy_breakout = noisy & (ret_auto.ge(thresholds["ret_high"]) | refined.eq("run_entry"))
    quiet_compression = quiet | (refined.eq("quiet_continuation") & ~noisy)

    target = long_mask & broad
    split = refined.copy()
    split.loc[target & quiet_compression] = "long_mixed_gmm_0_quiet_compression_low_momentum"
    split.loc[target & noisy_breakout] = "long_mixed_gmm_1_noisy_breakout_high_adverse_risk"
    split.loc[target & trend_pullback] = "long_mixed_gmm_2_trend_pullback_clean_continuation"
    split.loc[target & liquidity_stress] = "long_mixed_gmm_3_liquidity_stress_avoid"
    split.loc[target & split.eq(refined)] = "long_mixed_gmm_4_unresolved_context"
    out["long_source_regime_split"] = np.where(long_mask, split, "not_long")
    out["source_semantic_family"] = split
    out["source_family"] = split.astype(str)
    out["source_tag"] = side + "__" + split.astype(str)
    return out, {
        "long_refinement_status": "applied",
        "fit_long_rows": int(fit_long.sum()),
        "target_family_rows": int((long_mask & broad).sum()),
        "source_column": "side_name+source_semantic_family+AE/GMM/context_quantiles",
        "thresholds_fit_months": thresholds,
        "refined_families": sorted(str(value) for value in pd.Series(split.loc[long_mask & broad]).dropna().unique()),
    }


def _add_source_tags(ledger: pd.DataFrame, *, fit_mask: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add source tags that are live-computable from the S52 scored ledger.

    The current S52 ledger does not carry the original archetype/source-family
    columns.  Prefer them when present; otherwise fall back to side + model
    score intensity.  The fallback keeps source x regime matrices useful
    without using outcomes or strategy-id masks.
    """

    out = ledger.copy()
    side = out.get("side_name", pd.Series("unknown", index=out.index)).astype(str).str.lower()
    semantic_inputs = [
        "__meta_raw__volatility_zscore",
        "__meta_raw__asset_minus_mkt_oi_1d_peer_resid",
        "__meta_raw__return_autocorr_48",
    ]
    if all(col in out.columns for col in semantic_inputs):
        vol = pd.to_numeric(out["__meta_raw__volatility_zscore"], errors="coerce")
        oi = pd.to_numeric(out["__meta_raw__asset_minus_mkt_oi_1d_peer_resid"], errors="coerce")
        ret_auto = pd.to_numeric(out["__meta_raw__return_autocorr_48"], errors="coerce")
        fit_vol = vol.loc[fit_mask].replace([np.inf, -np.inf], np.nan).dropna()
        fit_oi_abs = oi.loc[fit_mask].abs().replace([np.inf, -np.inf], np.nan).dropna()
        fit_ret = ret_auto.loc[fit_mask].replace([np.inf, -np.inf], np.nan).dropna()
        vol_hi = max(float(fit_vol.quantile(0.75)) if len(fit_vol) else 0.5, 0.50)
        vol_lo = min(float(fit_vol.quantile(0.50)) if len(fit_vol) else 0.0, 0.10)
        oi_abs_hi = max(float(fit_oi_abs.quantile(0.75)) if len(fit_oi_abs) else 1.0, 1.00)
        ret_hi = float(fit_ret.quantile(0.70)) if len(fit_ret) else 0.35
        ret_lo = float(fit_ret.quantile(0.30)) if len(fit_ret) else -0.35
        ret_abs_mid = float(fit_ret.abs().quantile(0.50)) if len(fit_ret) else 0.35

        high_vol = vol.ge(vol_hi)
        low_vol = vol.le(vol_lo)
        high_oi = oi.abs().ge(oi_abs_hi)
        pos_ret = ret_auto.ge(ret_hi)
        neg_ret = ret_auto.le(ret_lo)
        quiet_ret = ret_auto.abs().le(ret_abs_mid)

        semantic = pd.Series("ambiguous_none", index=out.index, dtype="object")
        semantic.loc[low_vol & quiet_ret & ~high_oi] = "quiet_continuation"
        semantic.loc[low_vol & high_oi] = "compression_release"
        semantic.loc[pos_ret & ~high_vol] = "late_run_continuation"
        semantic.loc[pos_ret & high_vol] = "loud_breakout_impulse"
        semantic.loc[neg_ret & ~high_vol] = "retest_reversal"
        semantic.loc[neg_ret & high_vol] = "volatile_mean_reversion"
        semantic.loc[high_vol & high_oi] = "dirty_shock_avoid"
        semantic.loc[(~high_vol) & high_oi & ret_auto.gt(0.0)] = "run_entry"

        out["source_semantic_family"] = semantic
        out["source_volatility_state"] = np.select(
            [high_vol, low_vol],
            ["high_volatility", "low_volatility"],
            default="mid_volatility",
        )
        out["source_pressure_state"] = np.select(
            [high_oi & oi.gt(0.0), high_oi & oi.lt(0.0)],
            ["positive_oi_pressure", "negative_oi_pressure"],
            default="normal_oi_pressure",
        )
        out["source_trend_state"] = np.select(
            [pos_ret, neg_ret, quiet_ret],
            ["positive_autocorr", "negative_autocorr", "quiet_autocorr"],
            default="mixed_autocorr",
        )
        recipe = out.get("source_recipe_name", pd.Series("missing", index=out.index)).map(_clean_token)
        out["source_recipe_tag"] = side + "__" + recipe.astype(str)
        out["source_tag"] = side + "__" + semantic.astype(str)
        out["source_family"] = semantic.astype(str)
        if "score" in out.columns:
            edges = _quantile_edges(pd.to_numeric(out.loc[fit_mask, "score"], errors="coerce"), 10)
            score_bucket = _apply_edges(out["score"], edges, "source_score_decile")
            bucket_map = {
                "source_score_decile__q9": "model_frontier_top10",
                "source_score_decile__q8": "model_frontier_top20",
                "source_score_decile__q7": "model_frontier_top30",
            }
            out["source_score_intensity_tag"] = side + "__" + score_bucket.map(bucket_map).fillna(
                "model_candidate_background"
            ).astype(str)
        out, refine_contract = _refine_long_source_families(out, fit_mask=fit_mask)
        return out, {
            "source_tag_mode": "semantic_pre_entry_context",
            "source_column": "side_name+pre_entry_volatility_oi_autocorr_state",
            "semantic_inputs": semantic_inputs,
            "semantic_thresholds": {
                "vol_hi": vol_hi,
                "vol_lo": vol_lo,
                "oi_abs_hi": oi_abs_hi,
                "ret_hi": ret_hi,
                "ret_lo": ret_lo,
                "ret_abs_mid": ret_abs_mid,
            },
            "source_families": sorted(str(value) for value in pd.Series(semantic).dropna().unique()),
            "long_source_refinement": refine_contract,
        }
    source_cols = [
        col
        for col in (
            "source_tag",
            "label_archetype",
            "archetype",
            "regime_family",
            "source_family",
            "variant",
        )
        if col in out.columns and out[col].notna().any()
    ]
    if source_cols and source_cols[0] != "source_tag":
        base = out[source_cols[0]].astype(str).str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True)
        out["source_tag"] = side + "__" + base
        return out, {"source_tag_mode": "native_column", "source_column": source_cols[0]}

    if "score" in out.columns:
        edges = _quantile_edges(pd.to_numeric(out.loc[fit_mask, "score"], errors="coerce"), 10)
        score_bucket = _apply_edges(out["score"], edges, "source_score_decile")
        bucket_map = {
            "source_score_decile__q9": "model_frontier_top10",
            "source_score_decile__q8": "model_frontier_top20",
            "source_score_decile__q7": "model_frontier_top30",
        }
        intensity = score_bucket.map(bucket_map).fillna("model_candidate_background")
        out["source_tag"] = side + "__" + intensity.astype(str)
        out["source_family"] = side + "__s52_score_intensity"
        return out, {
            "source_tag_mode": "fallback_side_score_intensity",
            "source_column": "side_name+score_fit_decile",
            "edges": edges.tolist(),
        }

    out["source_tag"] = side + "__s52_candidate"
    out["source_family"] = side + "__s52_candidate"
    return out, {"source_tag_mode": "fallback_side_only", "source_column": "side_name"}


def _enrich_ledger(
    ledger: pd.DataFrame,
    *,
    embedded_round_trip_cost: float,
    executable_cost_floor: float,
) -> pd.DataFrame:
    out = ledger.copy(deep=False)
    out["month"] = out["month"].astype(str)
    out["source_tag"] = out["side_name"].astype(str) + "_trailing_tp075_sl050_tr035"
    out["source_family"] = out["side_name"].astype(str) + "_trailing_profit"

    first_touch_net = _first_numeric(
        out,
        (
            "first_touch_net",
            "__first_touch_capture_net__",
            "__first_touch_net__",
            "first_touch_capture_net",
        ),
        0.0,
    )
    first_touch_mae_norm = _first_numeric(
        out,
        (
            "first_touch_mae_norm",
            "__first_touch_mae_norm__",
            "first_touch_mae_to_sl",
            "__first_touch_mae_to_sl__",
        ),
    )
    first_touch_full_path_mae_norm = _first_numeric(
        out,
        (
            "first_touch_full_path_mae_norm",
            "__first_touch_full_path_mae_norm__",
            "first_touch_full_path_mae_to_sl",
            "__first_touch_full_path_mae_to_sl__",
        ),
    )
    timeout = _first_numeric(
        out,
        (
            "is_timeout",
            "__is_timeout__",
            "first_touch_timeout",
            "__first_touch_timeout__",
        ),
        0.0,
    )
    mfe_before_mae = _first_numeric(
        out,
        (
            "mfe_1r_before_mae_1r",
            "__mfe_1r_before_mae_1r__",
        ),
        0.0,
    )
    mae_before_mfe = _first_numeric(
        out,
        (
            "mae_1r_before_mfe_1r",
            "__mae_1r_before_mfe_1r__",
        ),
        0.0,
    )
    underwater_bars = _first_numeric(
        out,
        (
            "underwater_bars_before_mfe_1r",
            "__underwater_bars_before_mfe_1r__",
        ),
    )
    bars_to_mfe = _first_numeric(
        out,
        (
            "bars_to_mfe_1r",
            "__bars_to_mfe_1r__",
            "bars_to_mfe",
            "__bars_to_mfe__",
        ),
    )

    out["first_touch_gross"] = first_touch_net.fillna(0.0) + float(embedded_round_trip_cost)
    out["exec_margin"] = out["first_touch_gross"] - float(executable_cost_floor)
    # ``first_touch_net`` already contains ``embedded_round_trip_cost``. Rebuild
    # gross before applying the requested executable floor so the fee is
    # reconciled exactly once when embedded cost and floor are equal.
    out["ev_after_1pct"] = out["first_touch_gross"] - float(executable_cost_floor)
    out["first_touch_bad_mae_1r"] = first_touch_mae_norm.ge(1.0).astype(float)
    out["full_path_bad_mae_1r"] = first_touch_full_path_mae_norm.ge(1.0).astype(float)
    out["timeout"] = timeout.fillna(0.0).gt(0.5).astype(float)
    out["mfe_before_mae_1r"] = mfe_before_mae.fillna(0.0)
    out["mae_before_mfe_1r"] = mae_before_mfe.fillna(0.0)
    out["clean_exec"] = (
        out["exec_margin"].gt(0.0)
        & out["first_touch_bad_mae_1r"].lt(0.5)
        & out["timeout"].lt(0.5)
        & out["mfe_before_mae_1r"].gt(0.5)
    ).astype(float)
    out["dirty_positive"] = (
        out["exec_margin"].gt(0.0)
        & (
            out["first_touch_bad_mae_1r"].gt(0.5)
            | out["full_path_bad_mae_1r"].gt(0.5)
            | out["timeout"].gt(0.5)
            | out["mae_before_mfe_1r"].gt(0.5)
        )
    ).astype(float)
    is_long = out["side_name"].astype(str).str.lower().eq("long")
    first_mae = first_touch_mae_norm
    full_mae = first_touch_full_path_mae_norm
    time_to_profit = bars_to_mfe.where(bars_to_mfe.notna(), mfe_before_mae)
    post_mfe_drawdown = (full_mae - first_mae).clip(lower=0.0)
    slow_profit = underwater_bars.gt(16.0) | time_to_profit.gt(16.0)
    post_mfe_bad = post_mfe_drawdown.ge(0.50)
    long_clean = (
        out["exec_margin"].gt(0.0)
        & out["full_path_bad_mae_1r"].lt(0.5)
        & out["timeout"].lt(0.5)
        & out["mae_before_mfe_1r"].lt(0.5)
        & out["mfe_before_mae_1r"].gt(0.5)
        & slow_profit.fillna(False).eq(False)
        & post_mfe_bad.fillna(False).eq(False)
    )
    long_dirty = out["exec_margin"].gt(0.0) & (
        out["full_path_bad_mae_1r"].gt(0.5)
        | out["timeout"].gt(0.5)
        | out["mae_before_mfe_1r"].gt(0.5)
        | slow_profit.fillna(False)
        | post_mfe_bad.fillna(False)
    )
    out["long_path_full_bad_mae_1r"] = np.where(is_long, out["full_path_bad_mae_1r"], np.nan).astype(np.float32)
    out["long_path_time_to_profit_bars"] = np.where(is_long, time_to_profit, np.nan).astype(np.float32)
    out["long_path_slow_profit"] = np.where(is_long, slow_profit.fillna(False).astype(float), np.nan).astype(np.float32)
    out["long_path_post_mfe_drawdown_norm"] = np.where(is_long, post_mfe_drawdown, np.nan).astype(np.float32)
    out["long_path_post_mfe_bad_drawdown"] = np.where(
        is_long, post_mfe_bad.fillna(False).astype(float), np.nan
    ).astype(np.float32)
    out["long_path_clean_exec_label"] = np.where(is_long, long_clean.astype(float), np.nan).astype(np.float32)
    out["long_path_dirty_positive_label"] = np.where(is_long, long_dirty.astype(float), np.nan).astype(np.float32)
    out["long_path_quality_soft"] = np.where(
        is_long,
        (
            0.40 * long_clean.astype(float)
            + 0.20 * out["mfe_before_mae_1r"].clip(0.0, 1.0)
            + 0.20 * out["exec_margin"].gt(0.0).astype(float)
            - 0.20 * long_dirty.astype(float)
        ).clip(0.0, 1.0),
        np.nan,
    ).astype(np.float32)
    return out


def _add_base_score_context(
    ledger: pd.DataFrame,
    *,
    fit_mask: pd.Series,
    selected_col: str,
) -> pd.DataFrame:
    """Add pre-entry base-score context using fit-month priors for cutoffs."""

    out = ledger.copy(deep=False)
    score = _num(out, "score", np.nan)
    ts = pd.to_datetime(out.get("__ts__"), utc=True, errors="coerce")
    side = out.get("side_name", pd.Series("", index=out.index)).astype(str).str.lower()
    out["base_rank_pct_by_timestamp"] = score.groupby(ts).rank(pct=True).astype(np.float32)
    out["base_rank_pct_by_timestamp_side"] = score.groupby([ts, side]).rank(pct=True).astype(np.float32)
    mean_ts = score.groupby(ts).transform("mean")
    std_ts = score.groupby(ts).transform("std").replace(0.0, np.nan)
    out["base_score_z_by_timestamp"] = ((score - mean_ts) / std_ts).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    mean_tss = score.groupby([ts, side]).transform("mean")
    std_tss = score.groupby([ts, side]).transform("std").replace(0.0, np.nan)
    out["base_score_z_by_timestamp_side"] = ((score - mean_tss) / std_tss).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)

    archetype = (
        out["source_tag"].astype(str)
        if "source_tag" in out.columns
        else side + "__" + out.get("source_semantic_family", pd.Series("unknown", index=out.index)).astype(str)
    )
    keys = pd.DataFrame({"side": side, "arch": archetype, "score": score, "selected": out[selected_col].astype(bool)}, index=out.index)
    fit_keys = keys[fit_mask.reindex(out.index).fillna(False).astype(bool)].copy()
    fit_score = pd.to_numeric(fit_keys["score"], errors="coerce")
    fit_selected = fit_keys["selected"].astype(bool)
    global_cutoff = float(fit_score.loc[fit_selected & fit_score.notna()].min()) if bool((fit_selected & fit_score.notna()).any()) else float(fit_score.quantile(0.90))
    global_mean = float(fit_score.mean()) if fit_score.notna().any() else 0.0
    global_std = float(fit_score.std()) if fit_score.notna().sum() > 1 else 1.0
    if not np.isfinite(global_std) or global_std <= 1e-12:
        global_std = 1.0
    priors = []
    for (s, a), group in fit_keys.groupby(["side", "arch"], dropna=False):
        scores = pd.to_numeric(group["score"], errors="coerce").dropna()
        selected_scores = pd.to_numeric(group.loc[group["selected"].astype(bool), "score"], errors="coerce").dropna()
        cutoff = float(selected_scores.min()) if len(selected_scores) else (float(scores.quantile(0.90)) if len(scores) else global_cutoff)
        mean = float(scores.mean()) if len(scores) else global_mean
        std = float(scores.std()) if len(scores) > 1 else global_std
        if not np.isfinite(std) or std <= 1e-12:
            std = global_std
        priors.append({"side": s, "arch": a, "cutoff": cutoff, "mean": mean, "std": std})
    prior_df = pd.DataFrame(priors)
    joined = keys[["side", "arch"]].merge(prior_df, on=["side", "arch"], how="left")
    cutoff = pd.to_numeric(joined["cutoff"], errors="coerce").fillna(global_cutoff)
    mean = pd.to_numeric(joined["mean"], errors="coerce").fillna(global_mean)
    std = pd.to_numeric(joined["std"], errors="coerce").replace(0.0, np.nan).fillna(global_std)
    score_reset = score.reset_index(drop=True)
    out["base_margin_to_cutoff"] = (score_reset - cutoff).astype(np.float32).to_numpy()
    out["base_margin_to_cutoff_z"] = ((score_reset - cutoff) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32).to_numpy()
    out["base_signal_zscore_within_archetype"] = ((score_reset - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32).to_numpy()
    return out


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "rows": 0,
            "mean_exec_margin": float("nan"),
            "mean_ev_after_1pct": float("nan"),
            "positive_exec_margin_rate": float("nan"),
            "clean_exec_rate": float("nan"),
            "dirty_positive_rate": float("nan"),
            "first_touch_bad_mae_1r_rate": float("nan"),
            "full_path_bad_mae_1r_rate": float("nan"),
            "timeout_rate": float("nan"),
            "mfe_before_mae_1r_rate": float("nan"),
            "mae_before_mfe_1r_rate": float("nan"),
            "mean_underwater_bars": float("nan"),
            "p90_full_path_mae_norm": float("nan"),
            "mean_score": float("nan"),
        }
    return {
        "rows": int(len(frame)),
        "symbols": int(frame["__symbol__"].astype(str).nunique()) if "__symbol__" in frame.columns else 0,
        "months": int(frame["month"].astype(str).nunique()) if "month" in frame.columns else 0,
        "mean_exec_margin": _mean(frame["exec_margin"]),
        "mean_ev_after_1pct": _mean(frame["ev_after_1pct"]),
        "positive_exec_margin_rate": _rate(frame["exec_margin"].gt(0.0).astype(float)),
        "clean_exec_rate": _rate(frame["clean_exec"]),
        "dirty_positive_rate": _rate(frame["dirty_positive"]),
        "first_touch_bad_mae_1r_rate": _rate(frame["first_touch_bad_mae_1r"]),
        "full_path_bad_mae_1r_rate": _rate(frame["full_path_bad_mae_1r"]),
        "timeout_rate": _rate(frame["timeout"]),
        "mfe_before_mae_1r_rate": _rate(frame["mfe_before_mae_1r"]),
        "mae_before_mfe_1r_rate": _rate(frame["mae_before_mfe_1r"]),
        "mean_underwater_bars": _mean(frame.get("underwater_bars_before_mfe_1r", [])),
        "p90_full_path_mae_norm": _q(frame.get("first_touch_full_path_mae_norm", []), 0.90),
        "mean_score": _mean(frame.get("score", [])),
    }


def _source_concentration(rows: pd.DataFrame, regime_models: list[str]) -> pd.DataFrame:
    total = max(len(rows), 1)
    source_counts = rows["source_tag"].value_counts(dropna=False)
    out: list[dict[str, Any]] = []
    for regime in regime_models:
        for bucket, bucket_frame in rows.groupby(regime, observed=True, dropna=False):
            bucket_n = len(bucket_frame)
            for source, source_frame in bucket_frame.groupby("source_tag", observed=True, dropna=False):
                p_source_regime = len(source_frame) / max(bucket_n, 1)
                p_regime_source = len(source_frame) / max(int(source_counts.get(source, 0)), 1)
                p_source = int(source_counts.get(source, 0)) / total
                out.append(
                    {
                        "regime_model": regime,
                        "regime_bucket": str(bucket),
                        "source_tag": str(source),
                        "rows": int(len(source_frame)),
                        "regime_rows": int(bucket_n),
                        "source_rows": int(source_counts.get(source, 0)),
                        "p_source_given_regime": p_source_regime,
                        "p_regime_given_source": p_regime_source,
                        "lift_source_regime": p_source_regime / max(p_source, 1e-12),
                    }
                )
    return pd.DataFrame(out)


def _candidate_summary(rows: pd.DataFrame, regime_models: list[str]) -> pd.DataFrame:
    out: list[dict[str, Any]] = []
    for regime in regime_models:
        counts = rows[regime].astype(str).value_counts(dropna=False)
        month_counts = rows.groupby(regime, observed=True, dropna=False)["month"].nunique()
        side_counts = rows.groupby(regime, observed=True, dropna=False)["side_name"].nunique()
        out.append(
            {
                "regime_model": regime,
                "bucket_count": int(len(counts)),
                "rows": int(counts.sum()),
                "min_bucket_rows": int(counts.min()) if len(counts) else 0,
                "median_bucket_rows": float(counts.median()) if len(counts) else 0.0,
                "max_bucket_rows": int(counts.max()) if len(counts) else 0,
                "min_month_coverage": int(month_counts.min()) if len(month_counts) else 0,
                "min_side_coverage": int(side_counts.min()) if len(side_counts) else 0,
                "support_hhi": float(((counts / max(counts.sum(), 1)) ** 2).sum()) if len(counts) else float("nan"),
            }
        )
    return pd.DataFrame(out)


def _long_source_aegmm_month_metrics(rows: pd.DataFrame, *, selected_col: str) -> pd.DataFrame:
    if rows.empty or "side_name" not in rows.columns:
        return pd.DataFrame()
    regime_cols = [
        col
        for col in (
            "aegmm_cluster",
            "side_aegmm_cluster",
            "aegmm_expected_distance_bin",
            "reconstruction_bin",
            "latent_speed_bin",
            "regime_lgbm_leaf_bad_mae_k4",
        )
        if col in rows.columns
    ]
    if not regime_cols:
        return pd.DataFrame()
    long_rows = rows[rows["side_name"].astype(str).str.lower().eq("long")].copy()
    if long_rows.empty:
        return pd.DataFrame()
    source_col = "source_semantic_family" if "source_semantic_family" in long_rows.columns else "source_family"
    out: list[dict[str, Any]] = []
    for regime in regime_cols:
        group_cols = ["month", source_col, regime]
        for keys, group in long_rows.groupby(group_cols, observed=True, dropna=False):
            month, source, bucket = keys
            for scope, scoped in (
                ("all_rows", group),
                (selected_col, group[group[selected_col].astype(bool)] if selected_col in group.columns else group.iloc[:0]),
            ):
                record = {
                    "month": str(month),
                    "source_family": str(source),
                    "regime_model": regime,
                    "regime_bucket": str(bucket),
                    "selection_scope": scope,
                    **_metrics(scoped),
                }
                record["long_path_clean_exec_rate"] = _rate(scoped.get("long_path_clean_exec_label", []))
                record["long_path_dirty_positive_rate"] = _rate(scoped.get("long_path_dirty_positive_label", []))
                record["long_path_post_mfe_bad_drawdown_rate"] = _rate(
                    scoped.get("long_path_post_mfe_bad_drawdown", [])
                )
                record["long_path_slow_profit_rate"] = _rate(scoped.get("long_path_slow_profit", []))
                record["long_path_mean_post_mfe_drawdown_norm"] = _mean(
                    scoped.get("long_path_post_mfe_drawdown_norm", [])
                )
                record["long_path_mean_time_to_profit_bars"] = _mean(scoped.get("long_path_time_to_profit_bars", []))
                out.append(record)
    return pd.DataFrame(out).sort_values(["month", "selection_scope", "rows"], ascending=[True, True, False])


def _matrix_rows(
    rows: pd.DataFrame,
    regime_models: list[str],
    *,
    selected_col: str,
    scope_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcome_rows: list[dict[str, Any]] = []
    learn_rows: list[dict[str, Any]] = []
    for regime in regime_models:
        group_cols = ["source_tag", regime]
        for keys, frame in rows.groupby(group_cols, observed=True, dropna=False):
            source, bucket = keys
            frontier = frame[frame[selected_col].astype(bool)] if selected_col in frame.columns else frame.iloc[:0]
            top_in_bucket = frame[_top_mask(frame["score"], 0.10)] if "score" in frame.columns else frame.iloc[:0]
            base = {
                "scope": scope_name,
                "regime_model": regime,
                "regime_bucket": str(bucket),
                "source_tag": str(source),
            }
            outcome_rows.append({**base, "selection_scope": "all_rows", **_metrics(frame)})
            outcome_rows.append({**base, "selection_scope": selected_col, **_metrics(frontier)})
            learn_rows.append(
                {
                    **base,
                    "rows": int(len(frame)),
                    "frontier_rows": int(len(frontier)),
                    "bucket_top10_rows": int(len(top_in_bucket)),
                    "score_ic_exec_margin": _spearman(frame.get("score", []), frame["exec_margin"]),
                    "score_ic_clean_exec": _spearman(frame.get("score", []), frame["clean_exec"]),
                    **{f"frontier_{k}": v for k, v in _metrics(frontier).items()},
                    **{f"bucket_top10_{k}": v for k, v in _metrics(top_in_bucket).items()},
                }
            )
    return pd.DataFrame(outcome_rows), pd.DataFrame(learn_rows)


def _shrink_group_predictions(
    fit: pd.DataFrame,
    holdout: pd.DataFrame,
    *,
    target_col: str,
    key_cols: list[str],
    fit_fallback: pd.Series | float,
    holdout_fallback: pd.Series | float,
    shrinkage_k: float,
) -> tuple[pd.Series, pd.Series]:
    target = pd.to_numeric(fit[target_col], errors="coerce")
    valid = target.notna()
    if not bool(valid.any()):
        fit_base = (
            pd.Series(float(fit_fallback), index=fit.index, dtype=np.float64)
            if not isinstance(fit_fallback, pd.Series)
            else fit_fallback.reindex(fit.index).astype(float)
        )
        hold_base = (
            pd.Series(float(holdout_fallback), index=holdout.index, dtype=np.float64)
            if not isinstance(holdout_fallback, pd.Series)
            else holdout_fallback.reindex(holdout.index).astype(float)
        )
        return fit_base, hold_base
    tmp = fit.loc[valid, key_cols].copy()
    tmp["__target__"] = target.loc[valid].to_numpy(dtype=np.float64)
    stats = tmp.groupby(key_cols, observed=True, dropna=False)["__target__"].agg(["mean", "count"]).reset_index()
    fit_keys = fit[key_cols].merge(stats, on=key_cols, how="left")
    hold_keys = holdout[key_cols].merge(stats, on=key_cols, how="left")

    def blend(frame: pd.DataFrame, fallback: pd.Series | float, index: pd.Index) -> pd.Series:
        mean = pd.to_numeric(frame["mean"], errors="coerce")
        count = pd.to_numeric(frame["count"], errors="coerce").fillna(0.0)
        if isinstance(fallback, pd.Series):
            base = fallback.reindex(index).astype(float).reset_index(drop=True)
        else:
            base = pd.Series(float(fallback), index=range(len(frame)), dtype=np.float64)
        weight = count / (count + float(shrinkage_k))
        pred = weight * mean.fillna(base) + (1.0 - weight) * base
        pred.index = index
        return pred.astype(float)

    return (
        blend(fit_keys, fit_fallback, fit.index),
        blend(hold_keys, holdout_fallback, holdout.index),
    )


def _incremental_value_tests(
    fit_rows: pd.DataFrame,
    hold_rows: pd.DataFrame,
    regime_models: list[str],
    *,
    selected_col: str,
    shrinkage_k: float,
) -> pd.DataFrame:
    """Fit source/regime backoff models on fit months and score holdout.

    The comparison is:

    * source only
    * source + regime additive deviations
    * source x regime interaction, shrunk back to source

    This is intentionally a lightweight leakage-safe test of incremental regime
    value rather than another HPO surface.
    """

    targets: tuple[tuple[str, str], ...] = (
        ("exec_margin", "continuous"),
        ("ev_after_1pct", "continuous"),
        ("full_path_bad_mae_1r", "binary"),
        ("timeout", "binary"),
        ("mfe_before_mae_1r", "binary"),
        ("mae_before_mfe_1r", "binary"),
        ("clean_exec", "binary"),
        ("dirty_positive", "binary"),
    )
    scopes: tuple[tuple[str, pd.DataFrame, pd.DataFrame], ...] = (
        ("all_rows", fit_rows, hold_rows),
        (selected_col, fit_rows[fit_rows[selected_col].astype(bool)], hold_rows[hold_rows[selected_col].astype(bool)]),
    )
    out_rows: list[dict[str, Any]] = []
    for selection_scope, fit_scope, hold_scope in scopes:
        if fit_scope.empty or hold_scope.empty:
            continue
        for regime in regime_models:
            if regime not in fit_scope.columns or regime not in hold_scope.columns:
                continue
            for target_col, target_kind in targets:
                if target_col not in fit_scope.columns or target_col not in hold_scope.columns:
                    continue
                target = pd.to_numeric(fit_scope[target_col], errors="coerce")
                global_mean = float(target.mean()) if target.notna().any() else 0.0
                pred_a_fit, pred_a_hold = _shrink_group_predictions(
                    fit_scope,
                    hold_scope,
                    target_col=target_col,
                    key_cols=["source_tag"],
                    fit_fallback=global_mean,
                    holdout_fallback=global_mean,
                    shrinkage_k=shrinkage_k,
                )
                pred_reg_fit, pred_reg_hold = _shrink_group_predictions(
                    fit_scope,
                    hold_scope,
                    target_col=target_col,
                    key_cols=[regime],
                    fit_fallback=global_mean,
                    holdout_fallback=global_mean,
                    shrinkage_k=shrinkage_k,
                )
                pred_b_fit = pred_a_fit + pred_reg_fit - global_mean
                pred_b_hold = pred_a_hold + pred_reg_hold - global_mean
                pred_c_fit, pred_c_hold = _shrink_group_predictions(
                    fit_scope,
                    hold_scope,
                    target_col=target_col,
                    key_cols=["source_tag", regime],
                    fit_fallback=pred_a_fit,
                    holdout_fallback=pred_a_hold,
                    shrinkage_k=shrinkage_k,
                )
                model_preds = {
                    "source_only": pred_a_hold,
                    "source_plus_regime_additive": pred_b_hold,
                    "source_x_regime_interaction": pred_c_hold,
                }
                y = pd.to_numeric(hold_scope[target_col], errors="coerce")
                for model_name, pred in model_preds.items():
                    if target_kind == "binary":
                        eval_pred = pred.clip(1e-6, 1.0 - 1e-6)
                        primary_metric_name = "auc"
                        primary_metric = _auc(y, eval_pred)
                        secondary_metric_name = "logloss"
                        secondary_metric = _logloss(y, eval_pred)
                    else:
                        eval_pred = pred
                        primary_metric_name = "spearman_ic"
                        primary_metric = _spearman(eval_pred, y)
                        secondary_metric_name = "mse"
                        secondary_metric = _mean((eval_pred - y) ** 2)
                    top = hold_scope[_top_mask(eval_pred, 0.10)]
                    out_rows.append(
                        {
                            "selection_scope": selection_scope,
                            "regime_model": regime,
                            "target": target_col,
                            "target_kind": target_kind,
                            "incremental_model": model_name,
                            "fit_rows": int(len(fit_scope)),
                            "holdout_rows": int(len(hold_scope)),
                            "primary_metric_name": primary_metric_name,
                            "primary_metric": primary_metric,
                            "secondary_metric_name": secondary_metric_name,
                            "secondary_metric": secondary_metric,
                            "top10_rows": int(len(top)),
                            "top10_mean_exec_margin": _mean(top["exec_margin"]) if not top.empty else float("nan"),
                            "top10_clean_exec_rate": _rate(top["clean_exec"]) if not top.empty else float("nan"),
                            "top10_full_path_bad_mae_1r_rate": _rate(top["full_path_bad_mae_1r"])
                            if not top.empty
                            else float("nan"),
                            "top10_timeout_rate": _rate(top["timeout"]) if not top.empty else float("nan"),
                            "top10_mfe_before_mae_1r_rate": _rate(top["mfe_before_mae_1r"])
                            if not top.empty
                            else float("nan"),
                        }
                    )
    out = pd.DataFrame(out_rows)
    if out.empty:
        return out
    baseline = out[out["incremental_model"].eq("source_only")][
        [
            "selection_scope",
            "regime_model",
            "target",
            "primary_metric",
            "top10_mean_exec_margin",
            "top10_full_path_bad_mae_1r_rate",
        ]
    ].rename(
        columns={
            "primary_metric": "source_only_primary_metric",
            "top10_mean_exec_margin": "source_only_top10_mean_exec_margin",
            "top10_full_path_bad_mae_1r_rate": "source_only_top10_full_path_bad_mae_1r_rate",
        }
    )
    out = out.merge(baseline, on=["selection_scope", "regime_model", "target"], how="left")
    out["delta_primary_vs_source"] = out["primary_metric"] - out["source_only_primary_metric"]
    out["delta_top10_exec_margin_vs_source"] = out["top10_mean_exec_margin"] - out["source_only_top10_mean_exec_margin"]
    out["delta_top10_bad_mae_vs_source"] = (
        out["top10_full_path_bad_mae_1r_rate"] - out["source_only_top10_full_path_bad_mae_1r_rate"]
    )
    return out


def _regime_scores(
    fit_outcome: pd.DataFrame,
    fit_learnability: pd.DataFrame,
    concentration: pd.DataFrame,
    regime_models: list[str],
    *,
    selected_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frontier = fit_outcome[fit_outcome["selection_scope"].eq(str(selected_col))].copy()
    for regime in regime_models:
        f = frontier[frontier["regime_model"].eq(regime)]
        l = fit_learnability[fit_learnability["regime_model"].eq(regime)]
        c = concentration[concentration["regime_model"].eq(regime)]
        if f.empty:
            continue
        exec_std = float(pd.to_numeric(f["mean_exec_margin"], errors="coerce").std(skipna=True) or 0.0)
        bad_std = float(pd.to_numeric(f["full_path_bad_mae_1r_rate"], errors="coerce").std(skipna=True) or 0.0)
        support = pd.to_numeric(f["rows"], errors="coerce").fillna(0.0)
        source_score = min(float(np.nanmean(np.abs(np.log(np.clip(c.get("lift_source_regime", 1.0), 1e-6, 1e6))))) / 2.0, 1.0) if not c.empty else 0.0
        path_score = min(1.0, 80.0 * exec_std + 2.0 * bad_std)
        ic = pd.to_numeric(l.get("score_ic_exec_margin", pd.Series(dtype=float)), errors="coerce")
        top_exec = pd.to_numeric(l.get("bucket_top10_mean_exec_margin", pd.Series(dtype=float)), errors="coerce")
        frontier_bad = pd.to_numeric(l.get("frontier_full_path_bad_mae_1r_rate", pd.Series(dtype=float)), errors="coerce")
        learn_score = float(
            np.nanmean(
                [
                    np.nanmean(np.clip(ic, -0.25, 0.25)) * 2.0 + 0.5,
                    np.nanmean((top_exec > 0.0).astype(float)) if len(top_exec) else np.nan,
                    1.0 - np.nanmean(np.clip(frontier_bad, 0.0, 1.0)) if len(frontier_bad) else np.nan,
                ]
            )
        )
        stable_support = float((support.ge(30).mean() if len(support) else 0.0))
        regime_score = 0.10 * source_score + 0.35 * path_score + 0.35 * learn_score + 0.20 * stable_support
        rows.append(
            {
                "regime_model": regime,
                "regime_score": regime_score,
                "source_concentration_score": source_score,
                "path_outcome_interaction_score": path_score,
                "frontier_learnability_score": learn_score,
                "stability_support_score": stable_support,
                "fit_frontier_buckets": int(len(f)),
                "fit_frontier_rows": int(support.sum()),
                "fit_exec_margin_std": exec_std,
                "fit_full_path_bad_mae_std": bad_std,
            }
        )
    return pd.DataFrame(rows).sort_values("regime_score", ascending=False).reset_index(drop=True)


def _action_table(
    fit_outcome: pd.DataFrame,
    holdout_outcome: pd.DataFrame,
    *,
    selected_col: str,
    shrinkage_k: float,
) -> pd.DataFrame:
    fit = fit_outcome[fit_outcome["selection_scope"].eq(selected_col)].copy()
    hold = holdout_outcome[holdout_outcome["selection_scope"].eq(selected_col)].copy()
    parent_rows: list[dict[str, Any]] = []
    for source, group in fit.groupby("source_tag", observed=True, dropna=False):
        weights = pd.to_numeric(group["rows"], errors="coerce").fillna(0.0).clip(lower=0.0)
        if float(weights.sum()) <= 0.0:
            parent_rows.append(
                {
                    "source_tag": source,
                    "parent_mean_exec_margin": float("nan"),
                    "parent_full_path_bad_mae_1r_rate": float("nan"),
                }
            )
            continue
        parent_rows.append(
            {
                "source_tag": source,
                "parent_mean_exec_margin": float(
                    np.average(pd.to_numeric(group["mean_exec_margin"], errors="coerce").fillna(0.0), weights=weights)
                ),
                "parent_full_path_bad_mae_1r_rate": float(
                    np.average(
                        pd.to_numeric(group["full_path_bad_mae_1r_rate"], errors="coerce").fillna(0.0),
                        weights=weights,
                    )
                ),
            }
        )
    parent = pd.DataFrame(parent_rows)
    rows: list[dict[str, Any]] = []
    hold_keyed = hold.set_index(["regime_model", "regime_bucket", "source_tag"], drop=False)
    parent_keyed = parent.set_index("source_tag", drop=False)
    for _, row in fit.iterrows():
        key = (row["regime_model"], row["regime_bucket"], row["source_tag"])
        h = hold_keyed.loc[key].iloc[0] if key in hold_keyed.index and isinstance(hold_keyed.loc[key], pd.DataFrame) else (
            hold_keyed.loc[key] if key in hold_keyed.index else pd.Series(dtype=object)
        )
        p = parent_keyed.loc[row["source_tag"]] if row["source_tag"] in parent_keyed.index else pd.Series(dtype=object)
        n = float(row.get("rows", 0.0) or 0.0)
        shrink = n / (n + float(shrinkage_k)) if n >= 0 else 0.0
        parent_exec = float(p.get("parent_mean_exec_margin", np.nan))
        parent_bad = float(p.get("parent_full_path_bad_mae_1r_rate", np.nan))
        local_exec = float(row.get("mean_exec_margin", np.nan))
        local_bad = float(row.get("full_path_bad_mae_1r_rate", np.nan))
        local_timeout = float(row.get("timeout_rate", np.nan))
        delta_exec = local_exec - parent_exec if math.isfinite(parent_exec) and math.isfinite(local_exec) else float("nan")
        delta_bad = local_bad - parent_bad if math.isfinite(parent_bad) and math.isfinite(local_bad) else float("nan")
        if n < 30:
            action = "diagnostic_only"
            reason = "low_fit_frontier_support"
        elif local_exec > 0.0 and local_bad <= max(0.62, parent_bad - 0.03) and local_timeout <= 0.12:
            action = "upweight_or_lower_meta_threshold_candidate"
            reason = "fit_positive_margin_and_cleaner_than_parent"
        elif local_exec > 0.0 and local_bad > 0.65:
            action = "meta_filter_or_size_down_candidate"
            reason = "edge_exists_but_full_path_bad_mae_high"
        elif local_exec <= 0.0 and local_bad >= parent_bad:
            action = "downweight_or_require_meta_confirmation"
            reason = "weak_margin_and_not_cleaner_than_parent"
        else:
            action = "feature_only"
            reason = "mixed_fit_evidence"
        hold_rows = int(h.get("rows", 0) or 0) if not h.empty else 0
        hold_exec = float(h.get("mean_exec_margin", np.nan)) if not h.empty else float("nan")
        hold_bad = float(h.get("full_path_bad_mae_1r_rate", np.nan)) if not h.empty else float("nan")
        validation = "missing_holdout"
        if hold_rows >= 20:
            if action.startswith("upweight") and hold_exec > 0.0:
                validation = "holdout_confirms_margin"
            elif action.startswith("meta_filter") and hold_exec > 0.0 and hold_bad > 0.60:
                validation = "holdout_confirms_edge_path_ugly"
            elif action.startswith("downweight") and hold_exec <= 0.0:
                validation = "holdout_confirms_weak_margin"
            else:
                validation = "holdout_mixed"
        rows.append(
            {
                "regime_model": row["regime_model"],
                "regime_bucket": row["regime_bucket"],
                "source_tag": row["source_tag"],
                "recommended_action": action,
                "reason": reason,
                "fit_rows": int(n),
                "shrinkage_weight": shrink,
                "fit_mean_exec_margin": local_exec,
                "fit_full_path_bad_mae_1r_rate": local_bad,
                "fit_timeout_rate": local_timeout,
                "parent_mean_exec_margin": parent_exec,
                "parent_full_path_bad_mae_1r_rate": parent_bad,
                "expected_delta_exec_margin": delta_exec,
                "expected_delta_full_path_bad_mae": delta_bad,
                "holdout_rows": hold_rows,
                "holdout_mean_exec_margin": hold_exec,
                "holdout_full_path_bad_mae_1r_rate": hold_bad,
                "holdout_timeout_rate": float(h.get("timeout_rate", np.nan)) if not h.empty else float("nan"),
                "validation_status": validation,
                "promotion_status": "meta_context_candidate" if action != "diagnostic_only" else "diagnostic_only",
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    action_rank = {
        "upweight_or_lower_meta_threshold_candidate": 0,
        "meta_filter_or_size_down_candidate": 1,
        "downweight_or_require_meta_confirmation": 2,
        "feature_only": 3,
        "diagnostic_only": 4,
    }
    out["_action_rank"] = out["recommended_action"].map(action_rank).fillna(9)
    out = out.sort_values(
        ["_action_rank", "fit_rows", "expected_delta_full_path_bad_mae"],
        ascending=[True, False, True],
    ).drop(columns=["_action_rank"])
    return out


def _execution_policy_matrix(
    rows: pd.DataFrame,
    regime_models: list[str],
    *,
    selected_col: str,
    scope_name: str,
    shrinkage_k: float,
) -> pd.DataFrame:
    """Build a current-policy execution matrix.

    The S52 ledger only contains realized outcomes for the materialized trailing
    geometry.  This is therefore a fixed-policy matrix, not a policy-menu
    optimizer.  It still gives the meta layer a leakage-safe source x regime
    view of current execution quality.
    """

    out: list[dict[str, Any]] = []
    policy = "P_current_trailing_tp075_sl050_tr035"
    for regime in regime_models:
        for keys, frame in rows.groupby(["source_tag", regime], observed=True, dropna=False):
            source, bucket = keys
            for selection_scope, scoped in (
                ("all_rows", frame),
                (selected_col, frame[frame[selected_col].astype(bool)] if selected_col in frame.columns else frame.iloc[:0]),
            ):
                metrics = _metrics(scoped)
                support = float(metrics.get("rows", 0) or 0)
                shrinkage_weight = support / (support + float(shrinkage_k)) if support >= 0 else 0.0
                out.append(
                    {
                        "scope": scope_name,
                        "regime_model": regime,
                        "regime_bucket": str(bucket),
                        "source_tag": str(source),
                        "policy": policy,
                        "selection_scope": selection_scope,
                        "support": int(support),
                        "policy_EV": metrics["mean_exec_margin"],
                        "policy_bad_MAE": metrics["full_path_bad_mae_1r_rate"],
                        "policy_first_touch_bad_MAE": metrics["first_touch_bad_mae_1r_rate"],
                        "policy_timeout": metrics["timeout_rate"],
                        "policy_clean_exit_rate": metrics["clean_exec_rate"],
                        "policy_MFE_before_MAE": metrics["mfe_before_mae_1r_rate"],
                        "policy_MAE_before_MFE": metrics["mae_before_mfe_1r_rate"],
                        "policy_underwater_bars": metrics["mean_underwater_bars"],
                        "policy_p10_u": _q(scoped["exec_margin"], 0.10) if not scoped.empty else float("nan"),
                        "selected_policy": True,
                        "shrinkage_weight": shrinkage_weight,
                        "policy_search_status": "fixed_current_policy_only",
                    }
                )
    return pd.DataFrame(out)


def _proxy_policy_outcomes(rows: pd.DataFrame, policy: dict[str, Any]) -> pd.DataFrame:
    """Approximate a policy menu from stored path extrema/order diagnostics.

    This is not a replacement for barwise replay.  The S52 handoff ledger stores
    MFE/MAE extrema and 1R order flags, not full OHLC paths.  The proxy is useful
    for ranking source x regime execution geometry hypotheses for meta-layer
    context; exact promotion still requires replay on raw paths.
    """

    out = rows.copy()
    kind = str(policy["kind"])
    if kind == "abstain":
        out["policy_ev_proxy_r"] = 0.0
        out["policy_bad_mae_proxy"] = 0.0
        out["policy_timeout_proxy"] = 0.0
        out["policy_clean_exit_proxy"] = 1.0
        out["policy_mfe_before_mae_proxy"] = np.nan
        out["policy_mae_before_mfe_proxy"] = np.nan
        return out

    mfe = _num(out, "mfe_norm", 0.0).fillna(0.0)
    mae = _num(out, "mae_norm", 0.0).fillna(0.0)
    mfe_first = _num(out, "mfe_1r_before_mae_1r", 0.0).fillna(0.0).gt(0.5)
    mae_first = _num(out, "mae_1r_before_mfe_1r", 0.0).fillna(0.0).gt(0.5)
    timeout = _num(out, "timeout", 0.0).fillna(0.0).gt(0.5)
    if kind == "trailing_proxy":
        activation = float(policy["trail_start_r"])
        gap = float(policy["trail_gap_r"])
        sl_r = float(policy["sl_r"])
        activated = mfe.ge(activation)
        stop_hit = mae.ge(sl_r) & (~activated | mae_first)
        trail_exit_r = np.maximum(activation - gap, mfe - gap).clip(lower=0.0)
        ev_r = np.where(activated & ~stop_hit, trail_exit_r, np.where(stop_hit, -sl_r, np.minimum(mfe, activation) - mae))
        clean = activated & ~stop_hit & mfe_first
        bad_mae = mae.ge(max(1.0, sl_r))
    else:
        tp_r = float(policy["tp_r"])
        sl_r = float(policy["sl_r"])
        tp_hit = mfe.ge(tp_r)
        sl_hit = mae.ge(sl_r)
        win = tp_hit & (~sl_hit | mfe_first | (~mae_first & mfe.ge(mae)))
        stop = sl_hit & ~win
        timeout_proxy = ~(win | stop)
        ev_r = np.where(win, tp_r, np.where(stop, -sl_r, np.minimum(mfe, tp_r) - np.minimum(mae, sl_r)))
        ev_r = np.where(timeout_proxy & timeout, ev_r - 0.10, ev_r)
        clean = win & mae.lt(1.0) & mfe_first
        bad_mae = mae.ge(max(1.0, sl_r))

    out["policy_ev_proxy_r"] = pd.Series(ev_r, index=out.index).astype(float)
    out["policy_bad_mae_proxy"] = pd.Series(bad_mae, index=out.index).astype(float)
    out["policy_timeout_proxy"] = pd.Series(timeout, index=out.index).astype(float)
    out["policy_clean_exit_proxy"] = pd.Series(clean, index=out.index).astype(float)
    out["policy_mfe_before_mae_proxy"] = mfe_first.astype(float)
    out["policy_mae_before_mfe_proxy"] = mae_first.astype(float)
    return out


def _proxy_policy_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "support": 0,
            "policy_EV_proxy_R": float("nan"),
            "policy_bad_MAE_proxy": float("nan"),
            "policy_timeout_proxy": float("nan"),
            "policy_clean_exit_proxy": float("nan"),
            "policy_MFE_before_MAE_proxy": float("nan"),
            "policy_MAE_before_MFE_proxy": float("nan"),
            "policy_p10_proxy_R": float("nan"),
        }
    return {
        "support": int(len(frame)),
        "policy_EV_proxy_R": _mean(frame["policy_ev_proxy_r"]),
        "policy_bad_MAE_proxy": _rate(frame["policy_bad_mae_proxy"]),
        "policy_timeout_proxy": _rate(frame["policy_timeout_proxy"]),
        "policy_clean_exit_proxy": _rate(frame["policy_clean_exit_proxy"]),
        "policy_MFE_before_MAE_proxy": _rate(frame["policy_mfe_before_mae_proxy"]),
        "policy_MAE_before_MFE_proxy": _rate(frame["policy_mae_before_mfe_proxy"]),
        "policy_p10_proxy_R": _q(frame["policy_ev_proxy_r"], 0.10),
    }


def _execution_policy_menu_matrix(
    fit_rows: pd.DataFrame,
    hold_rows: pd.DataFrame,
    regime_models: list[str],
    *,
    selected_col: str,
    shrinkage_k: float,
) -> pd.DataFrame:
    out: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, str, str, str]] = set()
    fit_score_rows: list[dict[str, Any]] = []
    for policy in EXECUTION_POLICY_MENU:
        policy_fit = _proxy_policy_outcomes(fit_rows, policy)
        for regime in regime_models:
            for keys, frame in policy_fit.groupby(["source_tag", regime], observed=True, dropna=False):
                source, bucket = keys
                scoped = frame[frame[selected_col].astype(bool)] if selected_col in frame.columns else frame.iloc[:0]
                metrics = _proxy_policy_metrics(scoped)
                score = (
                    float(metrics["policy_EV_proxy_R"] or 0.0)
                    - 0.35 * float(metrics["policy_bad_MAE_proxy"] or 0.0)
                    - 0.15 * float(metrics["policy_timeout_proxy"] or 0.0)
                    + 0.10 * float(metrics["policy_clean_exit_proxy"] or 0.0)
                )
                if int(metrics["support"]) < 20:
                    score -= 1.0
                fit_score_rows.append(
                    {
                        "regime_model": regime,
                        "regime_bucket": str(bucket),
                        "source_tag": str(source),
                        "policy": str(policy["policy"]),
                        "fit_selection_score": score,
                        "fit_support": int(metrics["support"]),
                    }
                )
    score_frame = pd.DataFrame(fit_score_rows)
    if not score_frame.empty:
        score_frame = score_frame.sort_values(
            ["regime_model", "regime_bucket", "source_tag", "fit_selection_score", "fit_support"],
            ascending=[True, True, True, False, False],
        )
        best = score_frame.groupby(["regime_model", "regime_bucket", "source_tag"], observed=True, dropna=False).head(1)
        selected_keys = set(
            zip(
                best["regime_model"].astype(str),
                best["regime_bucket"].astype(str),
                best["source_tag"].astype(str),
                best["policy"].astype(str),
            )
        )

    for scope_name, source_rows in (("fit", fit_rows), ("holdout", hold_rows)):
        for policy in EXECUTION_POLICY_MENU:
            policy_rows = _proxy_policy_outcomes(source_rows, policy)
            for regime in regime_models:
                for keys, frame in policy_rows.groupby(["source_tag", regime], observed=True, dropna=False):
                    source, bucket = keys
                    for selection_scope, scoped in (
                        ("all_rows", frame),
                        (
                            selected_col,
                            frame[frame[selected_col].astype(bool)] if selected_col in frame.columns else frame.iloc[:0],
                        ),
                    ):
                        metrics = _proxy_policy_metrics(scoped)
                        support = float(metrics.get("support", 0) or 0)
                        out.append(
                            {
                                "scope": scope_name,
                                "regime_model": regime,
                                "regime_bucket": str(bucket),
                                "source_tag": str(source),
                                "policy": str(policy["policy"]),
                                "policy_kind": str(policy["kind"]),
                                "selection_scope": selection_scope,
                                "support": int(support),
                                "policy_EV": metrics["policy_EV_proxy_R"],
                                "policy_EV_basis": "normalized_R_proxy_from_extrema",
                                "policy_EV_exact": float("nan"),
                                "policy_EV_proxy_R": metrics["policy_EV_proxy_R"],
                                "policy_bad_MAE": metrics["policy_bad_MAE_proxy"],
                                "policy_first_touch_bad_MAE": float("nan"),
                                "policy_timeout": metrics["policy_timeout_proxy"],
                                "policy_clean_exit_rate": metrics["policy_clean_exit_proxy"],
                                "policy_MFE_before_MAE": metrics["policy_MFE_before_MAE_proxy"],
                                "policy_MAE_before_MFE": metrics["policy_MAE_before_MFE_proxy"],
                                "policy_underwater_bars": _mean(scoped.get("underwater_bars_before_mfe_1r", [])),
                                "policy_p10_u": metrics["policy_p10_proxy_R"],
                                "selected_policy": (
                                    str(regime),
                                    str(bucket),
                                    str(source),
                                    str(policy["policy"]),
                                )
                                in selected_keys,
                                "shrinkage_weight": support / (support + float(shrinkage_k)) if support >= 0 else 0.0,
                                "policy_search_status": "fit_month_proxy_menu_selected_applied_to_holdout",
                            }
                        )

    exact = pd.concat(
        [
            _execution_policy_matrix(
                fit_rows,
                regime_models,
                selected_col=selected_col,
                scope_name="fit",
                shrinkage_k=shrinkage_k,
            ),
            _execution_policy_matrix(
                hold_rows,
                regime_models,
                selected_col=selected_col,
                scope_name="holdout",
                shrinkage_k=shrinkage_k,
            ),
        ],
        ignore_index=True,
    )
    if not exact.empty:
        exact["policy_EV_basis"] = "exact_materialized_first_touch_return"
        exact["policy_EV_exact"] = exact["policy_EV"]
        exact["policy_EV_proxy_R"] = np.nan
        exact["policy_kind"] = "materialized_current_policy"
    return pd.concat([exact, pd.DataFrame(out)], ignore_index=True, sort=False)


def _bootstrap_metric_ci(values: np.ndarray, *, kind: str, n_boot: int, rng: np.random.Generator) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 20:
        return {"mean": float(np.nanmean(values)) if values.size else float("nan"), "lo": float("nan"), "hi": float("nan")}
    idx = rng.integers(0, values.size, size=(int(n_boot), values.size))
    samples = values[idx]
    if kind == "rate":
        boot = np.nanmean(np.clip(samples, 0.0, 1.0), axis=1)
    else:
        boot = np.nanmean(samples, axis=1)
    return {
        "mean": float(np.nanmean(values)),
        "lo": float(np.nanquantile(boot, 0.05)),
        "hi": float(np.nanquantile(boot, 0.95)),
    }


def _bootstrap_confidence_intervals(
    rows: pd.DataFrame,
    regime_models: list[str],
    *,
    selected_col: str,
    scope_name: str,
    n_boot: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(31 if scope_name == "fit" else 37)
    out: list[dict[str, Any]] = []
    metric_cols: tuple[tuple[str, str], ...] = (
        ("exec_margin", "continuous"),
        ("ev_after_1pct", "continuous"),
        ("full_path_bad_mae_1r", "rate"),
        ("timeout", "rate"),
        ("clean_exec", "rate"),
        ("dirty_positive", "rate"),
        ("mfe_before_mae_1r", "rate"),
        ("mae_before_mfe_1r", "rate"),
    )
    for regime in regime_models:
        if regime not in rows.columns:
            continue
        for keys, frame in rows.groupby(["source_tag", regime], observed=True, dropna=False):
            source, bucket = keys
            for selection_scope, scoped in (
                ("all_rows", frame),
                (selected_col, frame[frame[selected_col].astype(bool)] if selected_col in frame.columns else frame.iloc[:0]),
            ):
                base = {
                    "scope": scope_name,
                    "selection_scope": selection_scope,
                    "regime_model": regime,
                    "regime_bucket": str(bucket),
                    "source_tag": str(source),
                    "rows": int(len(scoped)),
                    "bootstrap_iterations": int(n_boot),
                }
                for metric, kind in metric_cols:
                    ci = _bootstrap_metric_ci(
                        pd.to_numeric(scoped.get(metric, pd.Series(dtype=float)), errors="coerce").to_numpy(),
                        kind=kind,
                        n_boot=int(n_boot),
                        rng=rng,
                    )
                    out.append(
                        {
                            **base,
                            "metric": metric,
                            "metric_kind": kind,
                            "mean": ci["mean"],
                            "ci05": ci["lo"],
                            "ci95": ci["hi"],
                            "ci_width": ci["hi"] - ci["lo"] if math.isfinite(ci["hi"]) and math.isfinite(ci["lo"]) else float("nan"),
                        }
                    )
    return pd.DataFrame(out)


def _train_meta_handoff(
    ledger: pd.DataFrame,
    actions: pd.DataFrame,
    regime_models: list[str],
    *,
    selected_col: str,
    extra_context_cols: Iterable[str] | None = None,
    strict_base_contract: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Materialize row-level regime/source context for train_meta consumption."""

    inherited_contract = _inherited_base_contract(
        ledger, strict=bool(strict_base_contract)
    )
    key_cols = [
        col
        for col in (
            "__ts__",
            "__symbol__",
            "side_name",
            "month",
            "__decision_ts__",
            "__first_path_ts__",
            "__label_path_end_ts__",
        )
        if col in ledger.columns
    ]
    archetype_cols = [
        col
        for col in (
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
            "archetype_label_family",
            "archetype_label_source",
            "archetype_policy_key",
            "archetype_policy_role",
            "archetype_policy_confidence",
            "archetype_policy_tp_r",
            "archetype_policy_sl_r",
            "archetype_policy_trail_r",
            "archetype_policy_max_bars_to_mfe",
            "archetype_policy_max_barrier",
            "policy_archetype",
            "local_side_archetype",
        )
        if col in ledger.columns
    ]
    base_cols = [
        col
        for col in (
            "score",
            selected_col,
            "source_tag",
            "source_family",
            "source_semantic_family",
            "source_semantic_family_base",
            "long_source_regime_split",
            "source_volatility_state",
            "source_pressure_state",
            "source_trend_state",
            "source_recipe_tag",
            "source_score_intensity_tag",
            "base_score_decile",
            "base_rank_pct_by_timestamp",
            "base_rank_pct_by_timestamp_side",
            "base_score_z_by_timestamp",
            "base_score_z_by_timestamp_side",
            "base_margin_to_cutoff",
            "base_margin_to_cutoff_z",
            "base_signal_zscore_within_archetype",
        )
        if col in ledger.columns
    ]
    continuous_cols = [
        col
        for col in (
            "gmm_cluster_id",
            "gmm_posterior_max",
            "gmm_posterior_margin",
            "gmm_entropy",
            "cluster_entropy",
            "mahalanobis_distance",
            "expected_mahalanobis",
            "min_mahalanobis",
            "cluster_speed",
            "cluster_acceleration",
            "time_since_cluster_change",
            "rolling_cluster_stability",
            "cluster_flip_count_20",
            "AE_reconstruction_error",
            "dae_reconstruction_error",
            "dae_reconstruction_error_zscore",
            "dae_reconstruction_error_delta_1",
            "dae_reconstruction_error_accel_1",
            "latent_mahalanobis_drift",
            "latent_speed",
            "latent_acceleration",
            "regime_bad_mae_score",
            "regime_first_touch_bad_mae_score",
            "regime_timeout_score",
            "regime_dirty_positive_score",
            "regime_clean_exec_score",
            "regime_exec_margin_score",
            "regime_ev_score",
        )
        if col in ledger.columns
    ]
    posterior_cols = sorted(
        col for col in ledger.columns if col.startswith("gmm_cluster_posterior_") or col.startswith("gmm_dist_center_")
    )
    aegmm_cols = [
        str(col) for col in AE_GMM_FEATURE_COLUMNS if str(col) in ledger.columns
    ]
    cross_market_cols = sorted(
        col
        for col in ledger.columns
        if col.startswith(CROSS_MARKET_FEATURE_PREFIXES)
        and pd.api.types.is_numeric_dtype(ledger[col])
    )
    extra_feature_store_cols = sorted(
        col
        for col in (extra_context_cols or [])
        if col in ledger.columns
        and col not in key_cols
        and _safe_feature_store_context_column(str(col))
        and pd.api.types.is_numeric_dtype(ledger[col])
    )
    cols = []
    for col in (
        key_cols
        + archetype_cols
        + base_cols
        + regime_models
        + aegmm_cols
        + continuous_cols
        + posterior_cols
        + cross_market_cols
        + extra_feature_store_cols
    ):
        if col in ledger.columns and col not in cols:
            cols.append(col)
    out = ledger[cols].copy()
    # Row-level hashes are auditable provenance only.  The full contracts live
    # in the sidecar JSON and neither belongs in the meta model matrix.
    out[HANDOFF_RANK_SCOPE_COLUMN] = HANDOFF_RANK_SCOPE
    out[BASE_TARGET_CONTRACT_HASH_COLUMN] = inherited_contract[
        BASE_TARGET_CONTRACT_HASH_COLUMN
    ]
    out[BASE_SAMPLE_WEIGHT_SPEC_HASH_COLUMN] = inherited_contract[
        BASE_SAMPLE_WEIGHT_SPEC_HASH_COLUMN
    ]
    for raw_col, alias_col in (
        ("__archetype_label_family__", "archetype_label_family"),
        ("__archetype_label_source__", "archetype_label_source"),
        ("__archetype_policy_key__", "archetype_policy_key"),
        ("__archetype_policy_role__", "archetype_policy_role"),
        ("__archetype_policy_confidence__", "archetype_policy_confidence"),
        ("__archetype_policy_tp_r__", "archetype_policy_tp_r"),
        ("__archetype_policy_sl_r__", "archetype_policy_sl_r"),
        ("__archetype_policy_trail_r__", "archetype_policy_trail_r"),
        ("__archetype_policy_max_bars_to_mfe__", "archetype_policy_max_bars_to_mfe"),
        ("__archetype_policy_max_barrier__", "archetype_policy_max_barrier"),
    ):
        if alias_col not in out.columns and raw_col in out.columns:
            out[alias_col] = out[raw_col]
    if "policy_archetype" not in out.columns:
        if "archetype_policy_key" in out.columns:
            out["policy_archetype"] = out["archetype_policy_key"].astype(str)
        elif "__archetype_policy_key__" in out.columns:
            out["policy_archetype"] = out["__archetype_policy_key__"].astype(str)
        elif "source_tag" in out.columns:
            out["policy_archetype"] = out["source_tag"].astype(str)
    if "local_side_archetype" not in out.columns and "policy_archetype" in out.columns:
        out["local_side_archetype"] = out["policy_archetype"].astype(str)

    action_names = [
        "upweight_or_lower_meta_threshold_candidate",
        "meta_filter_or_size_down_candidate",
        "downweight_or_require_meta_confirmation",
        "feature_only",
        "diagnostic_only",
    ]
    for name in action_names:
        out[f"meta_action_count__{name}"] = 0
    out["meta_action_max_shrinkage_weight"] = 0.0
    out["meta_action_mean_expected_delta_exec_margin"] = 0.0
    out["meta_action_min_expected_delta_full_path_bad_mae"] = np.nan
    out["meta_action_mean_holdout_exec_margin"] = np.nan
    out["meta_action_mean_holdout_bad_mae"] = np.nan
    out["meta_regime_action_matches"] = 0

    action_source = actions.copy()
    if not action_source.empty:
        for regime in regime_models:
            if regime not in ledger.columns:
                continue
            subset = action_source[action_source["regime_model"].eq(regime)].copy()
            if subset.empty:
                continue
            subset = subset.drop_duplicates(["source_tag", "regime_bucket"], keep="first")
            left = pd.DataFrame(
                {
                    "source_tag": ledger["source_tag"].astype(str).to_numpy(),
                    "regime_bucket": ledger[regime].astype(str).to_numpy(),
                },
                index=ledger.index,
            )
            joined = left.merge(subset, on=["source_tag", "regime_bucket"], how="left")
            matched = joined["recommended_action"].notna()
            out["meta_regime_action_matches"] += matched.astype(int).to_numpy()
            for name in action_names:
                out[f"meta_action_count__{name}"] += joined["recommended_action"].eq(name).astype(int).to_numpy()
            shrink = pd.to_numeric(joined.get("shrinkage_weight", np.nan), errors="coerce").fillna(0.0).to_numpy()
            out["meta_action_max_shrinkage_weight"] = np.maximum(
                out["meta_action_max_shrinkage_weight"].to_numpy(dtype=float),
                shrink,
            )
            delta_exec = pd.to_numeric(joined.get("expected_delta_exec_margin", np.nan), errors="coerce")
            delta_bad = pd.to_numeric(joined.get("expected_delta_full_path_bad_mae", np.nan), errors="coerce")
            hold_exec = pd.to_numeric(joined.get("holdout_mean_exec_margin", np.nan), errors="coerce")
            hold_bad = pd.to_numeric(joined.get("holdout_full_path_bad_mae_1r_rate", np.nan), errors="coerce")
            out["meta_action_mean_expected_delta_exec_margin"] += delta_exec.fillna(0.0).to_numpy()
            current_bad = pd.to_numeric(out["meta_action_min_expected_delta_full_path_bad_mae"], errors="coerce")
            out["meta_action_min_expected_delta_full_path_bad_mae"] = np.fmin(
                current_bad.fillna(np.inf).to_numpy(dtype=float),
                delta_bad.fillna(np.inf).to_numpy(dtype=float),
            )
            for target_col, values in (
                ("meta_action_mean_holdout_exec_margin", hold_exec),
                ("meta_action_mean_holdout_bad_mae", hold_bad),
            ):
                current = pd.to_numeric(out[target_col], errors="coerce")
                count = out["meta_regime_action_matches"].clip(lower=1).astype(float)
                # Running mean over matched regimes, using zero contribution for missing cells.
                out[target_col] = (
                    current.fillna(0.0).to_numpy(dtype=float) * (count.to_numpy(dtype=float) - 1.0)
                    + values.fillna(0.0).to_numpy(dtype=float)
                ) / count.to_numpy(dtype=float)
    matches = out["meta_regime_action_matches"].clip(lower=1).astype(float)
    out["meta_action_mean_expected_delta_exec_margin"] = (
        out["meta_action_mean_expected_delta_exec_margin"].astype(float) / matches
    )
    bad = pd.to_numeric(out["meta_action_min_expected_delta_full_path_bad_mae"], errors="coerce")
    out["meta_action_min_expected_delta_full_path_bad_mae"] = bad.replace(np.inf, np.nan)
    out["meta_context_weight_hint"] = (
        1.0
        + 0.10 * out["meta_action_count__upweight_or_lower_meta_threshold_candidate"].clip(upper=5)
        - 0.08 * out["meta_action_count__downweight_or_require_meta_confirmation"].clip(upper=5)
        - 0.06 * out["meta_action_count__meta_filter_or_size_down_candidate"].clip(upper=5)
    ).clip(0.25, 1.75)
    out["meta_threshold_adjustment_hint"] = (
        0.02 * out["meta_action_count__meta_filter_or_size_down_candidate"].clip(upper=5)
        + 0.015 * out["meta_action_count__downweight_or_require_meta_confirmation"].clip(upper=5)
        - 0.01 * out["meta_action_count__upweight_or_lower_meta_threshold_candidate"].clip(upper=5)
    )
    contract = {
        "kind": "train_meta_regime_feature_handoff",
        "row_count": int(len(out)),
        "selected_col": selected_col,
        "candidate_handoff_rank_scope": HANDOFF_RANK_SCOPE,
        "inherited_base_contract": inherited_contract,
        "provenance_columns": list(HANDOFF_PROVENANCE_COLUMNS),
        "key_columns": key_cols,
        "archetype_columns": [col for col in out.columns if "archetype" in str(col).lower()],
        "source_columns": [col for col in base_cols if col.startswith("source_") or col == "source_tag"],
        "regime_columns": regime_models,
        "ae_gmm_context_columns": aegmm_cols,
        "ae_gmm_context_column_count": int(len(aegmm_cols)),
        "continuous_context_columns": list(
            dict.fromkeys([*aegmm_cols, *continuous_cols, *posterior_cols])
        ),
        "cross_market_context_columns": cross_market_cols,
        "extra_feature_store_context_columns": extra_feature_store_cols,
        "extra_feature_store_context_column_count": int(len(extra_feature_store_cols)),
        "action_feature_columns": [col for col in out.columns if col.startswith("meta_action_")]
        + ["meta_context_weight_hint", "meta_threshold_adjustment_hint"],
        "leakage_contract": "row features are pre-entry/base OOF/context only; action aggregates are fit-month action-table joins; no holdout outcomes are used to choose actions",
    }
    return out, contract


def _feature_file_for_symbol(feature_dir: Path, symbol: str) -> Path:
    file_symbol = str(symbol).replace("/", "_")
    return Path(feature_dir) / f"symbol={file_symbol}.parquet"


def _safe_feature_store_context_column(name: str) -> bool:
    blocked = {
        "__symbol__",
        "symbol",
        "side",
        "side_name",
        "target",
        "target_soft",
        "target_hard",
        "label",
        "ret_fwd",
        "future_return",
    }
    lowered = str(name).lower()
    if name in blocked or lowered.startswith("__"):
        return False
    if any(token in lowered for token in ("future", "target", "label", "outcome", "forward")):
        return False
    return True


def _resolve_cfg_feature_keys(seed_keys: Iterable[str]) -> list[str]:
    """Resolve config.py feature-key indirections into concrete feature names."""

    try:
        from extreme_price_movements.config import CFG  # noqa: WPS433
    except Exception:
        return list(dict.fromkeys(str(key) for key in seed_keys if str(key).strip()))

    resolved: list[str] = []
    seen_refs: set[str] = set()

    def add_key(key: Any, depth: int = 0) -> None:
        text = str(key)
        if not text or depth > 8:
            return
        value = CFG.get(text)
        if isinstance(value, (list, tuple, set)) and text not in seen_refs:
            seen_refs.add(text)
            for item in value:
                add_key(item, depth + 1)
            return
        resolved.append(text)

    for key in seed_keys:
        add_key(key)
    return list(dict.fromkeys(resolved))


def _config_meta_full_feature_keys() -> list[str]:
    try:
        import extreme_price_movements.config as epm_config  # noqa: WPS433

        CFG = epm_config.CFG
    except Exception:
        return []
    seed_groups = [
        "base_shared_feature_keys",
        "base_long_feature_keys",
        "base_short_feature_keys",
        "meta_shared_feature_keys",
        "meta_product_feature_keys",
        "meta_reg_feature_keys",
        "meta_clf_feature_keys",
        "meta_mfe_feature_keys",
        "meta_mae_feature_keys",
        "meta_asym_feature_keys",
        "PERP_FEATURE_KEYS",
        "LGBM_PERP_FEATURE_KEYS",
        "PERP_META_PRIMARY_FEATURE_KEYS",
        "SPOT_FOR_PERPS_META_FEATURE_KEYS",
        "PERP_EVENT_RISK_FEATURE_KEYS",
        "OI_FEATURE_KEYS",
        "OI_TRADING_FEATURE_KEYS",
        "OI_NORMALIZED_FEATURE_KEYS",
        "LONG_HORIZON_PERP_META_FEATURE_KEYS",
        "VOLUME_FREE_PERP_META_FEATURE_KEYS",
        "RESIDUAL_FEATURE_KEYS",
        "RESIDUAL_META_FEATURE_KEYS",
        "ORDERBOOK_FEATURE_KEYS",
        "ORDERBOOK_RAW_META_FEATURE_KEYS",
        "ORDERBOOK_NORMALIZED_META_FEATURE_KEYS",
        "ORDERBOOK_META_FEATURE_KEYS",
        "META_ORDERBOOK_WALL_FEATURE_KEYS",
        "META_ORDERBOOK_BLOCKER_FEATURE_KEYS",
        "CROSS_ASSET_META_FEATURE_KEYS",
        "META_CROSS_SECTIONAL_REGIME_KEYS",
        "MODEL_REGIME_CONTEXT_META_FEATURE_KEYS",
        "MODEL_REGIME_XS_META_FEATURE_KEYS",
        "MODEL_REGIME_TAIL_META_FEATURE_KEYS",
        "MODEL_REGIME_EIGEN_META_FEATURE_KEYS",
        "MODEL_REGIME_COMPOSITE_META_FEATURE_KEYS",
        "MARKET_SPECTRAL_POSITION_META_FEATURE_KEYS",
        "META_MODEL_UNCERTAINTY_META_INPUT_FEATURE_KEYS",
        "BASE_LGBM_META_UNCERTAINTY_FEATURE_KEYS",
        "BASE_LGBM_RAW_STATE_DRIFT_FEATURE_KEYS",
        "BASE_LGBM_PREDICTIVE_ATLAS_FEATURE_KEYS",
        "BASE_LGBM_AE_GMM_FEATURE_KEYS",
        "LGBM_AE_GMM_FEATURE_KEYS",
        "spread_proxy_features",
    ]
    keys: list[str] = []
    for group in seed_groups:
        value = CFG.get(group, None)
        if value is None:
            value = getattr(epm_config, group, [])
        if isinstance(value, (list, tuple, set)):
            keys.extend(_resolve_cfg_feature_keys(value))
    # Include common ledger/scored-model context fields even when they do not
    # live in the feature store itself.
    keys.extend(
        [
            "score",
            "base_rank_pct_by_timestamp",
            "base_rank_pct_by_timestamp_side",
            "base_score_z_by_timestamp",
            "base_score_z_by_timestamp_side",
            "source_tag",
            "source_semantic_family",
            "source_semantic_family_base",
            "long_source_regime_split",
            "aegmm_cluster",
            "side_aegmm_cluster",
            "gmm_cluster_id",
            "gmm_entropy",
            "mahalanobis_distance",
            "AE_reconstruction_error",
            "cluster_speed",
            "cluster_acceleration",
            "regime_lgbm_leaf_bad_mae_k4",
            "regime_lgbm_leaf_exec_margin_k4",
        ]
    )
    return list(dict.fromkeys(str(key) for key in keys if _safe_feature_store_context_column(str(key))))


def _feature_store_context_columns(columns: Iterable[str], *, scope: str = "cross_market") -> list[str]:
    scope = str(scope or "cross_market").strip().lower()
    if scope not in FEATURE_STORE_SCOPES:
        raise ValueError(f"Unsupported feature store scope {scope!r}; expected one of {FEATURE_STORE_SCOPES}")
    # Storage keys are handled by the point-in-time join and must never become
    # model candidates. The shared reader materializes ``ts`` as each feature
    # panel's DatetimeIndex; timestamp fields are join keys, not model inputs.
    storage_keys = {"ts", "timestamp", "__ts__", "__symbol__", "symbol"}
    available = [str(col) for col in columns if str(col) not in storage_keys]
    available_set = set(available)
    if scope == "aegmm_inputs":
        return []
    if scope == "all_safe":
        return sorted(name for name in available if _safe_feature_store_context_column(name))
    if scope == "config_meta_full":
        config_keys = _config_meta_full_feature_keys()
        return sorted(name for name in config_keys if name in available_set and _safe_feature_store_context_column(name))
    out: list[str] = []
    for col in available:
        name = str(col)
        if not _safe_feature_store_context_column(name):
            continue
        if name.startswith(CROSS_MARKET_FEATURE_PREFIXES):
            out.append(name)
    return sorted(set(out))


def _read_feature_store_symbol_context(
    *,
    feature_store_ts: pd.Timestamp,
    data_root: Path,
    symbol: str,
    columns: list[str],
    start_ts: pd.Timestamp | None,
    end_ts: pd.Timestamp | None,
) -> pd.DataFrame:
    frame = read_static_features(
        feature_store_ts=feature_store_ts,
        data_root=data_root,
        feature_keys=columns,
        symbols=[symbol],
        start_ts=start_ts,
        end_ts=end_ts,
        output_layout="symbol_frame",
    )
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame(columns=["__ts__", "__symbol__", *columns])
    frame["__ts__"] = pd.to_datetime(
        frame.index, utc=True, errors="coerce"
    ).tz_convert(None)
    frame["__symbol__"] = str(symbol)
    keep = ["__ts__", "__symbol__"] + [col for col in columns if col in frame.columns]
    return (
        frame[keep]
        .dropna(subset=["__ts__", "__symbol__"])
        .drop_duplicates(["__ts__", "__symbol__"], keep="last")
    )


def _read_feature_store_symbol_context_batch(
    *,
    feature_store_ts: pd.Timestamp,
    data_root: Path,
    symbols: list[str],
    columns: list[str],
    start_ts: pd.Timestamp | None,
    end_ts: pd.Timestamp | None,
) -> dict[str, pd.DataFrame]:
    """Read a bounded symbol batch once, then materialize exact symbol frames."""

    requested_symbols = [str(symbol) for symbol in symbols]
    if not requested_symbols:
        return {}
    loaded = read_static_features(
        feature_store_ts=feature_store_ts,
        data_root=data_root,
        feature_keys=columns,
        symbols=requested_symbols,
        start_ts=start_ts,
        end_ts=end_ts,
        output_layout="panels",
    )
    if loaded is None or not hasattr(loaded, "symbol_frame"):
        return {}
    result: dict[str, pd.DataFrame] = {}
    for symbol in requested_symbols:
        frame = loaded.symbol_frame(symbol, keys=columns)
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        frame["__ts__"] = pd.to_datetime(
            frame.index, utc=True, errors="coerce"
        ).tz_convert(None)
        frame["__symbol__"] = symbol
        keep = [
            "__ts__",
            "__symbol__",
            *[column for column in columns if column in frame.columns],
        ]
        result[symbol] = (
            frame[keep]
            .dropna(subset=["__ts__", "__symbol__"])
            .drop_duplicates(["__ts__", "__symbol__"], keep="last")
        )
    return result


def _context_cache_paths(
    cache_root: Path,
    *,
    feature_dir: Path,
    scope: str,
    symbol: str,
    columns: list[str],
    source_signature: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Build a source-fingerprinted, per-symbol context-cache location."""

    contract = {
        "feature_dir": str(feature_dir.resolve()),
        "scope": str(scope),
        "symbol": str(symbol),
        "columns": list(columns),
        "source_signature": dict(source_signature),
    }
    digest = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    directory = cache_root / digest[:2] / digest
    return directory / "context.parquet", directory / "contract.json"


def _cached_symbol_context(
    *,
    cache_root: Path | None,
    feature_dir: Path,
    scope: str,
    symbol: str,
    columns: list[str],
    requested_timestamps: pd.Series,
    source_signature: Mapping[str, Any],
) -> pd.DataFrame | None:
    """Return a cache hit only when every requested timestamp is present."""

    if cache_root is None or requested_timestamps.empty:
        return None
    data_path, contract_path = _context_cache_paths(
        cache_root,
        feature_dir=feature_dir,
        scope=scope,
        symbol=symbol,
        columns=columns,
        source_signature=source_signature,
    )
    if not data_path.exists() or not contract_path.exists():
        return None
    try:
        stored = json.loads(contract_path.read_text(encoding="utf-8"))
        expected = {
            "feature_dir": str(feature_dir.resolve()),
            "scope": str(scope),
            "symbol": str(symbol),
            "columns": list(columns),
            "source_signature": dict(source_signature),
        }
        if stored != expected:
            return None
        available = pd.Index(
            pd.to_datetime(
                pd.read_parquet(data_path, columns=["__ts__"])["__ts__"], errors="coerce"
            ).dropna().unique()
        )
        requested = pd.Index(pd.to_datetime(requested_timestamps, errors="coerce").dropna().unique())
        if requested.empty or not requested.isin(available).all():
            return None
        return pd.read_parquet(data_path, columns=["__ts__", "__symbol__", *columns])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _cache_symbol_context(
    frame: pd.DataFrame,
    *,
    cache_root: Path | None,
    feature_dir: Path,
    scope: str,
    symbol: str,
    columns: list[str],
    source_signature: Mapping[str, Any],
) -> None:
    if cache_root is None or frame.empty:
        return
    data_path, contract_path = _context_cache_paths(
        cache_root,
        feature_dir=feature_dir,
        scope=scope,
        symbol=symbol,
        columns=columns,
        source_signature=source_signature,
    )
    contract = {
        "feature_dir": str(feature_dir.resolve()),
        "scope": str(scope),
        "symbol": str(symbol),
        "columns": list(columns),
        "source_signature": dict(source_signature),
    }
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        values = frame.loc[:, ["__ts__", "__symbol__", *columns]].copy(deep=False)
        if data_path.exists() and contract_path.exists():
            if json.loads(contract_path.read_text(encoding="utf-8")) == contract:
                values = pd.concat([pd.read_parquet(data_path), values], ignore_index=True, copy=False)
        values.drop_duplicates(["__ts__", "__symbol__"], keep="last").to_parquet(
            data_path, index=False, compression="zstd"
        )
        contract_path.write_text(json.dumps(contract, sort_keys=True), encoding="utf-8")
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        # This cache is optional. On any issue the caller remains exact by
        # reading the published static store directly.
        return


def _feature_store_signature(paths: Iterable[Path]) -> dict[str, Any]:
    """Fingerprint the logical files read by a published handoff artifact."""

    entries: list[tuple[str, int, int]] = []
    for parquet_path in sorted({Path(path) for path in paths}, key=str):
        related = [
            parquet_path,
            Path(str(parquet_path) + ".deltas.duckdb"),
            Path(str(parquet_path).removesuffix(".parquet") + ".meta.json"),
        ]
        delta_dir = Path(str(parquet_path) + ".deltas")
        if delta_dir.exists():
            related.extend(sorted(delta_dir.glob("*.parquet")))
        for path in related:
            if not path.exists() or not path.is_file():
                continue
            stat = path.stat()
            entries.append((str(path), int(stat.st_size), int(stat.st_mtime_ns)))
    digest = hashlib.sha256()
    for path, size, mtime_ns in entries:
        digest.update(f"{path}\0{size}\0{mtime_ns}\n".encode("utf-8"))
    return {
        "algorithm": "sha256_path_size_mtime_ns_v1",
        "digest": digest.hexdigest(),
        "file_count": int(len(entries)),
        "total_size": int(sum(size for _, size, _ in entries)),
        "max_mtime_ns": int(max((mtime for _, _, mtime in entries), default=0)),
    }


def _join_feature_store_context(
    ledger: pd.DataFrame,
    feature_dir: Path | None,
    *,
    feature_store_scope: str = "cross_market",
    required_context_columns: Iterable[str] = (),
    context_allowlist: Iterable[str] = (),
    context_cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join live feature-store columns by timestamp and symbol."""

    if feature_dir is None or not Path(feature_dir).exists():
        return ledger, {
            "feature_store_context_status": "missing",
            "feature_dir": str(feature_dir) if feature_dir is not None else None,
        }
    if "__ts__" not in ledger.columns or "__symbol__" not in ledger.columns:
        return ledger, {
            "feature_store_context_status": "missing_keys",
            "feature_dir": str(feature_dir),
            "required_keys": ["__ts__", "__symbol__"],
        }
    feature_dir = Path(feature_dir)
    try:
        feature_store_ts = pd.to_datetime(
            feature_dir.name, format="%Y%m%d_%H%M%S", utc=True
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Feature directory must be a timestamped shared static store, e.g. "
            "data_perp/features/20260711_070000"
        ) from exc
    if feature_dir.parent.name != "features":
        raise ValueError(
            f"Feature directory is outside the shared static-store layout: {feature_dir}"
        )
    data_root = feature_dir.parent.parent
    symbols = sorted(ledger["__symbol__"].dropna().astype(str).unique().tolist())
    schema_columns: set[str] = set()
    available_files: dict[str, Path] = {}
    missing_symbols: list[str] = []
    for symbol in symbols:
        path = _feature_file_for_symbol(feature_dir, symbol)
        if not path.exists():
            missing_symbols.append(symbol)
            continue
        available_files[symbol] = path
        schema_columns.update(_feature_schema_names(str(path)))
    requested_context_cols = list(
        dict.fromkeys(
            [
                *_feature_store_context_columns(
                    schema_columns, scope=feature_store_scope
                ),
                *[
                    str(column)
                    for column in required_context_columns
                    if str(column) in schema_columns
                    and _safe_feature_store_context_column(str(column))
                ],
            ]
        )
    )
    allowlist = list(dict.fromkeys(str(column) for column in context_allowlist))
    if allowlist:
        allowed = set(allowlist)
        requested_context_cols = [
            column for column in requested_context_cols if column in allowed
        ]
    # Ledger-native columns include the exact OOS base outputs and may also
    # include raw inputs used by the frozen state. Never let a store join suffix
    # or replace them.
    existing_ledger_cols = set(ledger.columns)
    context_cols = [
        col for col in requested_context_cols if col not in existing_ledger_cols
    ]
    if not context_cols or not available_files:
        return ledger, {
            "feature_store_context_status": "no_joinable_context_columns",
            "feature_dir": str(feature_dir),
            "feature_store_scope": str(feature_store_scope),
            "available_symbol_files": int(len(available_files)),
            "missing_symbol_files": int(len(missing_symbols)),
            "context_columns": context_cols,
            "context_allowlist": allowlist,
            "skipped_existing_columns": sorted(
                set(requested_context_cols) & existing_ledger_cols
            ),
        }
    # Enrich one symbol slice at a time. Building a complete context frame and
    # then merging it with the complete candidate ledger retained multiple
    # copies of a very wide matrix. Exact-key semantics are unchanged here,
    # while peak memory is bounded by one symbol slice plus the final output.
    left = ledger.copy(deep=False)
    left["__ts__"] = pd.to_datetime(
        left["__ts__"], utc=True, errors="coerce"
    ).dt.tz_convert(None)
    left["__symbol__"] = left["__symbol__"].astype(str)
    row_order_col = "__handoff_row_order__"
    if row_order_col in left.columns:
        raise ValueError(f"Reserved handoff column already exists: {row_order_col}")
    left[row_order_col] = np.arange(len(left), dtype=np.int64)
    before_cols = set(left.columns)
    parts: list[pd.DataFrame] = []
    matched_rows = 0
    batch_size = max(
        1,
        int(os.environ.get("EPM_META_HANDOFF_FEATURE_SYMBOL_BATCH_SIZE", "4")),
    )
    cache_root = (
        Path(context_cache_dir)
        if context_cache_dir is not None
        else (
            Path(os.environ["EPM_META_HANDOFF_CONTEXT_CACHE_DIR"])
            if os.environ.get("EPM_META_HANDOFF_CONTEXT_CACHE_DIR")
            else None
        )
    )
    cache_hits = 0
    cache_misses = 0
    for batch_start in range(0, len(symbols), batch_size):
        batch_symbols = symbols[batch_start : batch_start + batch_size]
        batch_mask = left["__symbol__"].isin(batch_symbols)
        batch_rows = left.loc[batch_mask]
        batch_available = [
            symbol for symbol in batch_symbols if symbol in available_files
        ]
        batch_context: dict[str, pd.DataFrame] = {}
        symbols_to_load: list[str] = []
        for symbol in batch_available:
            symbol_rows = batch_rows.loc[batch_rows["__symbol__"].eq(symbol)]
            signature = _feature_store_signature([available_files[symbol]])
            cached = _cached_symbol_context(
                cache_root=cache_root,
                feature_dir=feature_dir,
                scope=feature_store_scope,
                symbol=symbol,
                columns=context_cols,
                requested_timestamps=symbol_rows["__ts__"],
                source_signature=signature,
            )
            if cached is None:
                symbols_to_load.append(symbol)
                cache_misses += 1
            else:
                batch_context[symbol] = cached
                cache_hits += 1
        if symbols_to_load:
            loaded_context = _read_feature_store_symbol_context_batch(
                feature_store_ts=feature_store_ts,
                data_root=data_root,
                symbols=symbols_to_load,
                columns=context_cols,
                start_ts=(
                    batch_rows["__ts__"].min() if not batch_rows.empty else None
                ),
                end_ts=(
                    batch_rows["__ts__"].max() if not batch_rows.empty else None
                ),
            )
            for symbol, feature_part in loaded_context.items():
                batch_context[symbol] = feature_part
                _cache_symbol_context(
                    feature_part,
                    cache_root=cache_root,
                    feature_dir=feature_dir,
                    scope=feature_store_scope,
                    symbol=symbol,
                    columns=context_cols,
                    source_signature=_feature_store_signature([available_files[symbol]]),
                )
        for symbol in batch_symbols:
            symbol_rows = batch_rows.loc[
                batch_rows["__symbol__"].eq(symbol)
            ].copy(deep=False)
            if symbol_rows.empty:
                continue
            feature_part = batch_context.get(symbol)
            if feature_part is None or feature_part.empty:
                parts.append(symbol_rows)
                continue
            symbol_keys = (
                symbol_rows.loc[:, ["__ts__", "__symbol__"]]
                .dropna(subset=["__ts__", "__symbol__"])
                .drop_duplicates()
            )
            if not symbol_keys.empty:
                feature_part = feature_part.merge(
                    symbol_keys,
                    on=["__ts__", "__symbol__"],
                    how="inner",
                    validate="one_to_one",
                )
            if feature_part.empty:
                parts.append(symbol_rows)
                continue
            feature_part["__feature_store_match__"] = np.uint8(1)
            joined_symbol = symbol_rows.merge(
                feature_part,
                on=["__ts__", "__symbol__"],
                how="left",
                validate="many_to_one",
                sort=False,
            )
            matched_rows += int(
                joined_symbol["__feature_store_match__"].fillna(0).sum()
            )
            parts.append(joined_symbol)
    if not parts:
        return ledger, {
            "feature_store_context_status": "no_matching_rows",
            "feature_dir": str(feature_dir),
            "feature_store_scope": str(feature_store_scope),
            "available_symbol_files": int(len(available_files)),
            "missing_symbol_files": int(len(missing_symbols)),
            "context_columns": context_cols,
        }
    out = pd.concat(parts, ignore_index=True, copy=False)
    out = out.sort_values(row_order_col, kind="stable").reset_index(drop=True)
    out.drop(columns=[row_order_col], inplace=True)
    loaded_cols = [col for col in context_cols if col in out.columns and col not in before_cols]
    if "__feature_store_match__" in out.columns:
        matched_any = out.pop("__feature_store_match__").fillna(0).astype(bool)
    else:
        matched_any = pd.Series(False, index=out.index)
    coverage = {
        col: float(out[col].notna().mean())
        for col in loaded_cols
    }
    coverage_values = np.asarray(list(coverage.values()), dtype=np.float64)
    worst_coverage = sorted(coverage.items(), key=lambda item: (item[1], item[0]))[:50]
    contract = {
        "feature_store_context_status": "joined",
        "feature_dir": str(feature_dir),
        "feature_store_scope": str(feature_store_scope),
        "required_context_columns": [
            str(column) for column in required_context_columns
        ],
        "context_allowlist": allowlist,
        "context_allowlist_count": int(len(allowlist)),
        "available_symbol_files": int(len(available_files)),
        "missing_symbol_files": int(len(missing_symbols)),
        "loaded_columns": loaded_cols,
        "loaded_column_count": int(len(loaded_cols)),
        "skipped_existing_columns": sorted(
            set(requested_context_cols) & existing_ledger_cols
        ),
        "logical_store_reader": "static_feature_store.read_static_features",
        "static_feature_endpoint_version": STATIC_FEATURE_ENDPOINT_VERSION,
        "store_access": "read_only",
        "context_cache_dir": str(cache_root) if cache_root is not None else None,
        "context_cache_hits": int(cache_hits),
        "context_cache_misses": int(cache_misses),
        "source_signature": _feature_store_signature(available_files.values()),
        "matched_rows": int(matched_rows),
        "row_count": int(len(out)),
        "match_rate": float(matched_any.mean()) if len(out) else float("nan"),
        "feature_coverage_summary": {
            "min": float(np.nanmin(coverage_values)) if coverage_values.size else float("nan"),
            "p10": float(np.nanquantile(coverage_values, 0.10)) if coverage_values.size else float("nan"),
            "median": float(np.nanmedian(coverage_values)) if coverage_values.size else float("nan"),
            "p90": float(np.nanquantile(coverage_values, 0.90)) if coverage_values.size else float("nan"),
            "max": float(np.nanmax(coverage_values)) if coverage_values.size else float("nan"),
            "fully_covered_columns": int(sum(value >= 0.999 for value in coverage.values())),
            "columns": int(len(coverage)),
        },
        "worst_feature_coverage": [
            {"feature": feature, "finite_rate": rate}
            for feature, rate in worst_coverage
        ],
        "context_key_filter": "exact_candidate_timestamp_symbol",
        "leakage_contract": "logical feature-store context (Parquet plus append-only delta sidecars) joined exactly on row timestamp and symbol; columns are live/pre-entry fields and exclude target/label/future names",
    }
    return out, contract


def _write_report(
    path: Path,
    manifest: dict[str, Any],
    regime_scores: pd.DataFrame,
    actions: pd.DataFrame,
    execution_policy: pd.DataFrame,
    incremental_value: pd.DataFrame,
    bootstrap_ci: pd.DataFrame,
) -> None:
    incremental_preview = incremental_value[
        incremental_value["incremental_model"].eq("source_x_regime_interaction")
        & incremental_value["selection_scope"].eq(manifest["selected_col"])
    ].copy()
    if not incremental_preview.empty:
        incremental_preview["_rank"] = incremental_preview["delta_top10_exec_margin_vs_source"].fillna(-999.0)
        incremental_preview = incremental_preview.sort_values("_rank", ascending=False).drop(columns=["_rank"])
    lines = [
        "# S52 Trailing Regime Meta Handoff",
        "",
        f"Ledger: `{manifest['ledger_path']}`",
        f"Fit months: `{', '.join(manifest['fit_months'])}`",
        f"Holdout month: `{manifest['holdout_month']}`",
        f"Selected frontier: `{manifest['selected_col']}`",
        "",
        "## Top Regime Scores",
        "",
        regime_scores.head(12).to_markdown(index=False) if not regime_scores.empty else "No regime scores.",
        "",
        "## Top Action Candidates",
        "",
        actions.head(20).to_markdown(index=False) if not actions.empty else "No action candidates.",
        "",
        "## Execution Policy Matrix",
        "",
        execution_policy[
            execution_policy["selection_scope"].eq(manifest["selected_col"])
            & execution_policy["scope"].eq("fit")
            & execution_policy["selected_policy"].astype(bool)
        ]
        .sort_values(["policy_EV", "support"], ascending=[False, False])
        .head(20)
        .to_markdown(index=False)
        if not execution_policy.empty
        else "No execution policy rows.",
        "",
        "## Bootstrap CI Preview",
        "",
        bootstrap_ci[
            bootstrap_ci["selection_scope"].eq(manifest["selected_col"])
            & bootstrap_ci["scope"].eq("holdout")
            & bootstrap_ci["metric"].isin(["exec_margin", "full_path_bad_mae_1r", "timeout"])
        ]
        .sort_values(["rows", "ci_width"], ascending=[False, True])
        .head(20)
        .to_markdown(index=False)
        if not bootstrap_ci.empty
        else "No bootstrap CI rows.",
        "",
        "## Incremental Regime Value Tests",
        "",
        incremental_preview.head(20).to_markdown(index=False)
        if not incremental_preview.empty
        else "No incremental value rows.",
        "",
        "## Read",
        "",
        "- Actions are fit-month choices; holdout columns are validation only.",
        "- Incremental value tests fit source/regime backoff models on fit months and evaluate holdout only.",
        "- Buckets with positive margin but high full-path bad-MAE should feed meta filtering/sizing, not base hard gates.",
        "- The execution policy matrix includes exact current-policy outcomes plus a normalized-R proxy menu selected on fit months only.",
        "- Proxy policy-menu rows are not frozen replay evidence; exact promotion still requires barwise policy replay.",
        "- This report is a handoff artifact for the meta/regime layer, not a frozen policy replay.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_report(
    *,
    ledger_path: Path,
    output_dir: Path,
    label_context_dir: Path | None,
    feature_dir: Path | None,
    feature_store_scope: str,
    fixed_ae_gmm_state_pkl: Path | None,
    fit_months: list[str],
    holdout_month: str,
    selected_col: str,
    embedded_round_trip_cost: float,
    executable_cost_floor: float,
    shrinkage_k: float,
    bootstrap_iterations: int,
    strict_base_contract: bool = False,
    feature_context_allowlist: Iterable[str] = (),
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = _materialize_promoted_base_contract(pd.read_parquet(ledger_path))
    ledger, label_context_contract = _join_label_context(ledger, label_context_dir)
    ledger, label_resolution_contract = _materialize_label_path_end(
        ledger, label_context_dir
    )
    state_input_columns = _frozen_ae_gmm_input_columns(fixed_ae_gmm_state_pkl)
    ledger, feature_store_context_contract = _join_feature_store_context(
        ledger,
        feature_dir,
        feature_store_scope=feature_store_scope,
        required_context_columns=state_input_columns,
        context_allowlist=feature_context_allowlist,
    )
    ledger, frozen_ae_gmm_contract = _append_frozen_ae_gmm_context(
        ledger, fixed_ae_gmm_state_pkl
    )
    if selected_col not in ledger.columns:
        raise ValueError(f"selected column not found in ledger: {selected_col}")
    ledger = _enrich_ledger(
        ledger,
        embedded_round_trip_cost=float(embedded_round_trip_cost),
        executable_cost_floor=float(executable_cost_floor),
    )
    fit_mask = ledger["month"].isin(fit_months)
    holdout_mask = ledger["month"].eq(str(holdout_month))
    ledger, source_contract = _add_source_tags(ledger, fit_mask=fit_mask)
    ledger = _add_base_score_context(ledger, fit_mask=fit_mask, selected_col=selected_col)
    ledger, regime_specs = _build_regime_columns(ledger, fit_mask=fit_mask)
    regime_models = [name for name in regime_specs if name in ledger.columns]
    fit_rows = ledger[fit_mask].copy()
    hold_rows = ledger[holdout_mask].copy()
    concentration = _source_concentration(ledger, regime_models)
    candidate_summary = _candidate_summary(ledger, regime_models)
    long_source_aegmm_month = _long_source_aegmm_month_metrics(ledger, selected_col=selected_col)
    fit_outcome, fit_learnability = _matrix_rows(fit_rows, regime_models, selected_col=selected_col, scope_name="fit")
    holdout_outcome, holdout_learnability = _matrix_rows(
        hold_rows,
        regime_models,
        selected_col=selected_col,
        scope_name="holdout",
    )
    outcome = pd.concat([fit_outcome, holdout_outcome], ignore_index=True)
    learnability = pd.concat([fit_learnability, holdout_learnability], ignore_index=True)
    regime_scores = _regime_scores(
        fit_outcome,
        fit_learnability,
        concentration,
        regime_models,
        selected_col=selected_col,
    )
    actions = _action_table(
        fit_outcome,
        holdout_outcome,
        selected_col=selected_col,
        shrinkage_k=float(shrinkage_k),
    )
    execution_policy = _execution_policy_menu_matrix(
        fit_rows,
        hold_rows,
        regime_models,
        selected_col=selected_col,
        shrinkage_k=float(shrinkage_k),
    )
    incremental_value = _incremental_value_tests(
        fit_rows,
        hold_rows,
        regime_models,
        selected_col=selected_col,
        shrinkage_k=float(shrinkage_k),
    )
    train_meta_handoff, train_meta_contract = _train_meta_handoff(
        ledger,
        actions,
        regime_models,
        selected_col=selected_col,
        extra_context_cols=feature_store_context_contract.get("loaded_columns", []),
        strict_base_contract=bool(strict_base_contract),
    )
    train_meta_contract["label_resolution_contract"] = label_resolution_contract
    bootstrap_ci = pd.concat(
        [
            _bootstrap_confidence_intervals(
                fit_rows,
                regime_models,
                selected_col=selected_col,
                scope_name="fit",
                n_boot=int(bootstrap_iterations),
            ),
            _bootstrap_confidence_intervals(
                hold_rows,
                regime_models,
                selected_col=selected_col,
                scope_name="holdout",
                n_boot=int(bootstrap_iterations),
            ),
        ],
        ignore_index=True,
    )
    outputs = {
        "scored_ledger": output_dir / "s52_trailing_regime_scored_ledger.parquet",
        "candidate_summary": output_dir / "regime_candidate_summary.csv",
        "source_concentration": output_dir / "source_concentration_matrix.csv",
        "long_source_aegmm_month_metrics": output_dir / "long_source_aegmm_month_metrics.csv",
        "source_regime_outcome": output_dir / "source_regime_outcome_matrix.csv",
        "source_regime_learnability": output_dir / "source_regime_learnability_matrix.csv",
        "execution_policy_matrix": output_dir / "execution_policy_matrix.csv",
        "incremental_value_tests": output_dir / "incremental_value_tests.csv",
        "bootstrap_confidence_intervals": output_dir / "bootstrap_confidence_intervals.csv",
        "regime_scores": output_dir / "regime_scores.csv",
        "policy_recommendations": output_dir / "policy_recommendation_table.csv",
        "train_meta_regime_handoff": output_dir / "train_meta_regime_handoff.parquet",
        "train_meta_regime_handoff_contract": output_dir / "train_meta_regime_handoff_contract.json",
        "manifest": output_dir / "manifest.json",
        "report": output_dir / "s52_trailing_regime_meta_handoff.md",
    }
    ledger.to_parquet(outputs["scored_ledger"], index=False)
    candidate_summary.to_csv(outputs["candidate_summary"], index=False)
    concentration.to_csv(outputs["source_concentration"], index=False)
    long_source_aegmm_month.to_csv(outputs["long_source_aegmm_month_metrics"], index=False)
    outcome.to_csv(outputs["source_regime_outcome"], index=False)
    learnability.to_csv(outputs["source_regime_learnability"], index=False)
    execution_policy.to_csv(outputs["execution_policy_matrix"], index=False)
    incremental_value.to_csv(outputs["incremental_value_tests"], index=False)
    bootstrap_ci.to_csv(outputs["bootstrap_confidence_intervals"], index=False)
    regime_scores.to_csv(outputs["regime_scores"], index=False)
    actions.to_csv(outputs["policy_recommendations"], index=False)
    train_meta_handoff.to_parquet(outputs["train_meta_regime_handoff"], index=False)
    outputs["train_meta_regime_handoff_contract"].write_text(
        json.dumps(_json_safe(train_meta_contract), indent=2),
        encoding="utf-8",
    )
    manifest = {
        "scope": "s52_trailing_regime_meta_handoff",
        "ledger_path": str(ledger_path),
        "output_dir": str(output_dir),
        "label_context_dir": str(label_context_dir) if label_context_dir is not None else None,
        "feature_dir": str(feature_dir) if feature_dir is not None else None,
        "feature_store_scope": str(feature_store_scope),
        "fit_months": fit_months,
        "holdout_month": str(holdout_month),
        "selected_col": str(selected_col),
        "strict_base_contract": bool(strict_base_contract),
        "embedded_round_trip_cost": float(embedded_round_trip_cost),
        "executable_cost_floor": float(executable_cost_floor),
        "rows": int(len(ledger)),
        "fit_rows": int(len(fit_rows)),
        "holdout_rows": int(len(hold_rows)),
        "regime_specs": regime_specs,
        "label_context_contract": label_context_contract,
        "label_resolution_contract": label_resolution_contract,
        "feature_store_context_contract": feature_store_context_contract,
        "frozen_ae_gmm_context_contract": frozen_ae_gmm_contract,
        "source_contract": source_contract,
        "execution_policy_status": "exact_current_policy_plus_fit_month_proxy_policy_menu",
        "incremental_value_test_status": "source_only_vs_source_plus_regime_vs_source_x_regime_fit_months_to_holdout",
        "lgbm_leaf_embedding_status": "fit_month_lgbm_leaf_models_clustered_with_fit_month_kmeans_applied_to_holdout",
        "bootstrap_ci_status": "source_x_regime_metric_ci_fit_and_holdout",
        "train_meta_handoff_status": "row_level_regime_source_action_features_materialized",
        "outputs": {key: str(value) for key, value in outputs.items()},
        "leakage_contract": {
            "continuous_bins": "fit_month_quantiles_applied_to_holdout",
            "label_context_join": "pre-entry label context joined by timestamp, symbol, and side; joined columns exclude future outcomes from source construction",
            "feature_store_context_join": "pre-entry feature-store context joined exactly by timestamp and symbol; joined columns are safe live/pre-entry feature-store fields and exclude target/label/future names",
            "source_tags": "semantic pre-entry source tags from volatility, OI pressure, and autocorrelation state when available; otherwise native pre-entry source/archetype column or side plus fit-month score decile fallback, no outcomes",
            "supervised_risk_score_regimes": "HistGradientBoostingRegressor models fit on fit months only using base score and AE/GMM state descriptors, then applied to holdout before quantile binning",
            "lgbm_leaf_embedding_regimes": "LGBMRegressor leaf models and MiniBatchKMeans leaf clusterers fit on fit months only, then frozen and applied to holdout",
            "actions": "selected_from_fit_month_metrics_only",
            "execution_policy_matrix": "current materialized policy summarized exactly; alternate proxy policy menu selected from fit months and reported on holdout without holdout policy selection",
            "incremental_value_tests": "target-encoding backoff models fitted on fit months and evaluated on holdout only",
            "bootstrap_confidence_intervals": "resampling is performed within each reported scope and does not feed policy selection",
            "train_meta_regime_handoff": "row-level pre-entry/source/regime context plus fit-month action aggregates; intended as train_meta feature/weight/threshold candidate input, not a final trading policy",
            "holdout": "reported_as_validation_only",
            "features": "OOF/base score and AE/GMM state descriptors from scored ledger",
        },
    }
    outputs["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    _write_report(outputs["report"], manifest, regime_scores, actions, execution_policy, incremental_value, bootstrap_ci)
    return manifest


def run_handoff_only(
    *,
    ledger_path: Path,
    output_dir: Path,
    label_context_dir: Path | None,
    feature_dir: Path | None,
    feature_store_scope: str,
    fixed_ae_gmm_state_pkl: Path | None,
    fit_months: list[str],
    holdout_month: str,
    selected_col: str,
    embedded_round_trip_cost: float,
    executable_cost_floor: float,
    context_cache_dir: Path | None = None,
    strict_base_contract: bool = False,
    feature_context_allowlist: Iterable[str] = (),
) -> dict[str, Any]:
    """Materialize feature-enriched scored ledger and train-meta handoff only."""

    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = _materialize_promoted_base_contract(pd.read_parquet(ledger_path))
    rows_before_selected_filter = int(len(ledger))
    if selected_col not in ledger.columns:
        match = re.fullmatch(r"selected_top(\d+)", str(selected_col))
        required_rank = {"__ts__", "side_name", "base_rank_within_timestamp_side"}
        if match is None or not required_rank.issubset(ledger.columns):
            raise ValueError(f"selected column not found in ledger: {selected_col}")
        fraction = float(match.group(1)) / 100.0
        if not 0.0 < fraction < 1.0:
            raise ValueError(f"invalid selected frontier: {selected_col}")
        group_rows = ledger.groupby(
            ["__ts__", "side_name"], sort=False, observed=True
        )["base_rank_within_timestamp_side"].transform("size")
        cutoff = np.ceil(group_rows.to_numpy(dtype=float) * fraction).astype(np.int32)
        ledger[selected_col] = (
            pd.to_numeric(ledger["base_rank_within_timestamp_side"], errors="coerce")
            .fillna(np.iinfo(np.int32).max)
            .to_numpy(dtype=np.int64)
            <= cutoff
        )
    selected_mask = ledger[selected_col].fillna(False).astype(bool)
    ledger = ledger.loc[selected_mask].copy()
    rows_after_selected_filter = int(len(ledger))
    ledger, label_context_contract = _join_label_context(ledger, label_context_dir)
    ledger, label_resolution_contract = _materialize_label_path_end(
        ledger, label_context_dir
    )
    # Base scored ledgers carry the complete frozen AE/GMM output registry. Do
    # not rejoin hundreds of raw state inputs speculatively; the exact frozen
    # output check below reuses complete outputs and fails closed if any are
    # absent rather than widening the handoff silently.
    state_input_columns: list[str] = []
    ledger, feature_store_context_contract = _join_feature_store_context(
        ledger,
        feature_dir,
        feature_store_scope=feature_store_scope,
        required_context_columns=state_input_columns,
        context_allowlist=feature_context_allowlist,
        context_cache_dir=context_cache_dir,
    )
    ledger, frozen_ae_gmm_contract = _append_frozen_ae_gmm_context(
        ledger, fixed_ae_gmm_state_pkl
    )
    ledger = _enrich_ledger(
        ledger,
        embedded_round_trip_cost=float(embedded_round_trip_cost),
        executable_cost_floor=float(executable_cost_floor),
    )
    fit_mask = ledger["month"].isin(fit_months)
    ledger, source_contract = _add_source_tags(ledger, fit_mask=fit_mask)
    ledger = _add_base_score_context(ledger, fit_mask=fit_mask, selected_col=selected_col)
    ledger, regime_specs = _build_regime_columns(ledger, fit_mask=fit_mask)
    regime_models = [name for name in regime_specs if name in ledger.columns]
    train_meta_handoff, train_meta_contract = _train_meta_handoff(
        ledger,
        pd.DataFrame(),
        regime_models,
        selected_col=selected_col,
        extra_context_cols=feature_store_context_contract.get("loaded_columns", []),
        strict_base_contract=bool(strict_base_contract),
    )
    train_meta_contract["label_resolution_contract"] = label_resolution_contract
    outputs = {
        "scored_ledger": output_dir / "s52_trailing_regime_scored_ledger.parquet",
        "train_meta_regime_handoff": output_dir / "train_meta_regime_handoff.parquet",
        "train_meta_regime_handoff_contract": output_dir / "train_meta_regime_handoff_contract.json",
        "manifest": output_dir / "manifest.json",
    }
    ledger.to_parquet(outputs["scored_ledger"], index=False)
    train_meta_handoff.to_parquet(outputs["train_meta_regime_handoff"], index=False)
    outputs["train_meta_regime_handoff_contract"].write_text(
        json.dumps(_json_safe(train_meta_contract), indent=2),
        encoding="utf-8",
    )
    manifest = {
        "scope": "s52_trailing_regime_meta_handoff",
        "mode": "handoff_only",
        "ledger_path": str(ledger_path),
        "output_dir": str(output_dir),
        "label_context_dir": str(label_context_dir) if label_context_dir is not None else None,
        "feature_dir": str(feature_dir) if feature_dir is not None else None,
        "feature_store_scope": str(feature_store_scope),
        "feature_context_allowlist": list(feature_context_allowlist),
        "fit_months": fit_months,
        "holdout_month": str(holdout_month),
        "selected_col": str(selected_col),
        "strict_base_contract": bool(strict_base_contract),
        "embedded_round_trip_cost": float(embedded_round_trip_cost),
        "executable_cost_floor": float(executable_cost_floor),
        "context_cache_dir": str(context_cache_dir) if context_cache_dir is not None else None,
        "rows": int(len(ledger)),
        "rows_before_selected_filter": rows_before_selected_filter,
        "rows_after_selected_filter": rows_after_selected_filter,
        "selected_filter_applied_before_context_joins": True,
        "fit_rows": int(fit_mask.sum()),
        "holdout_rows": int(ledger["month"].eq(str(holdout_month)).sum()) if "month" in ledger.columns else 0,
        "regime_specs": regime_specs,
        "label_context_contract": label_context_contract,
        "label_resolution_contract": label_resolution_contract,
        "feature_store_context_contract": feature_store_context_contract,
        "frozen_ae_gmm_context_contract": frozen_ae_gmm_contract,
        "source_contract": source_contract,
        "train_meta_handoff_status": "row_level_regime_source_action_features_materialized_handoff_only",
        "outputs": {key: str(value) for key, value in outputs.items()},
        "leakage_contract": {
            "continuous_bins": "fit_month_quantiles_applied_to_holdout",
            "label_context_join": "pre-entry label context joined by timestamp, symbol, and side; joined columns exclude future outcomes from source construction",
            "feature_store_context_join": "pre-entry feature-store context joined exactly by timestamp and symbol; joined columns are safe live/pre-entry feature-store fields and exclude target/label/future names",
            "source_tags": "semantic pre-entry source tags from volatility, OI pressure, and autocorrelation state when available; otherwise native pre-entry source/archetype column or side plus fit-month score decile fallback, no outcomes",
            "train_meta_regime_handoff": "row-level pre-entry/source/regime/context features; handoff-only mode uses zero action aggregates and skips validation/report diagnostics",
            "holdout": "not_evaluated_in_handoff_only_mode",
        },
    }
    outputs["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-path", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label-context-dir", type=Path, default=DEFAULT_LABEL_CONTEXT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-store-scope", choices=FEATURE_STORE_SCOPES, default="cross_market")
    parser.add_argument(
        "--feature-context-allowlist",
        type=Path,
        default=None,
        help=(
            "Optional CSV/JSON feature candidate list applied before the wide "
            "static-store join. Ledger-native base and AE/GMM outputs remain available."
        ),
    )
    parser.add_argument("--fixed-ae-gmm-state-pkl", type=Path, default=None)
    parser.add_argument("--fit-months", default=",".join(DEFAULT_FIT_MONTHS))
    parser.add_argument("--holdout-month", default=DEFAULT_HOLDOUT_MONTH)
    parser.add_argument("--selected-col", default=DEFAULT_SELECTED_COL)
    parser.add_argument("--embedded-round-trip-cost", type=float, default=0.003)
    parser.add_argument("--executable-cost-floor", type=float, default=0.010)
    parser.add_argument("--shrinkage-k", type=float, default=100.0)
    parser.add_argument("--bootstrap-iterations", type=int, default=200)
    parser.add_argument("--handoff-only", action="store_true")
    parser.add_argument(
        "--strict-base-contract",
        action="store_true",
        help=(
            "Require explicit uniform base target/weight provenance hashes and "
            "timestamp_side candidate ranking before writing the meta handoff."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    feature_context_allowlist: list[str] = []
    if args.feature_context_allowlist is not None:
        path = Path(args.feature_context_allowlist)
        if path.suffix.lower() == ".json":
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                raw = raw.get("features", raw.get("selected_features", []))
            if not isinstance(raw, list):
                raise ValueError(f"Invalid feature context allowlist JSON: {path}")
            feature_context_allowlist = [str(value) for value in raw]
        else:
            table = pd.read_csv(path)
            column = "feature" if "feature" in table.columns else str(table.columns[0])
            feature_context_allowlist = (
                table[column].dropna().astype(str).drop_duplicates().tolist()
            )
    common = {
        "ledger_path": args.ledger_path,
        "output_dir": args.output_dir,
        "label_context_dir": args.label_context_dir,
        "feature_dir": args.feature_dir,
        "feature_store_scope": str(args.feature_store_scope),
        "fixed_ae_gmm_state_pkl": args.fixed_ae_gmm_state_pkl,
        "fit_months": _parse_csv(args.fit_months, DEFAULT_FIT_MONTHS),
        "holdout_month": str(args.holdout_month),
        "selected_col": str(args.selected_col),
        "embedded_round_trip_cost": float(args.embedded_round_trip_cost),
        "executable_cost_floor": float(args.executable_cost_floor),
        "strict_base_contract": bool(args.strict_base_contract),
        "feature_context_allowlist": feature_context_allowlist,
    }
    if args.handoff_only:
        manifest = run_handoff_only(**common)
    else:
        manifest = run_report(
            **common,
            shrinkage_k=float(args.shrinkage_k),
            bootstrap_iterations=int(args.bootstrap_iterations),
        )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
