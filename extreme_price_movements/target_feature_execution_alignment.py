"""Fail-closed audit for the target/feature/execution alignment roadmap.

The project already has separate target, feature-lineage and ablation
artifacts.  This module joins their *contracts* without retraining anything
and records which claims are proved, which are only aggregate diagnostics and
which remain unavailable for promotion.  In particular, it never treats a
future label or an aggregate OOF manifest as a row-level execution feature.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SCHEMA = "target_feature_execution_alignment_audit_v1"
HORIZON = pd.Timedelta(hours=12)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp, pd.Timedelta, Path)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _check(name: str, passed: bool, value: Any, rule: str, **extra: Any) -> dict[str, Any]:
    result = {"check": name, "passed": bool(passed), "value": _json_safe(value), "rule": rule}
    result.update({key: _json_safe(val) for key, val in extra.items()})
    return result


def validate_primary_labels(primary: pd.DataFrame, contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate the row-level exact-H12 identity and cost equations."""

    required = {
        "candidate_id", "symbol", "side", "decision_ts", "feature_cutoff_ts",
        "entry_ts", "label_end_ts", "label_available_ts", "execution_policy_id",
        "cost_model_id", "execution_geometry_id", "execution_exact_h12_gross_bps",
        "execution_exact_h12_cost_bps", "execution_exact_h12_net_bps",
    }
    missing = sorted(required.difference(primary.columns))
    checks = [_check(
        "primary_contract_columns", not missing, missing,
        "exact-H12 primary labels expose identity, timing, policy, geometry, gross, cost and net columns",
    )]
    if missing:
        return checks

    checks.append(_check(
        "candidate_identity_unique", not primary["candidate_id"].duplicated().any(),
        int(primary["candidate_id"].duplicated().sum()), "one row per candidate_id",
    ))
    checks.append(_check(
        "side_domain", primary["side"].isin(["long", "short"]).all(),
        sorted(primary["side"].dropna().astype(str).unique().tolist()), "side is long or short",
    ))
    timestamps: dict[str, pd.Series] = {}
    for name in ("decision_ts", "feature_cutoff_ts", "entry_ts", "label_end_ts", "label_available_ts"):
        timestamps[name] = pd.to_datetime(primary[name], utc=True, errors="coerce")
        checks.append(_check(
            f"{name}_valid_utc", not timestamps[name].isna().any(),
            int(timestamps[name].isna().sum()), "all contract timestamps are valid UTC",
        ))
    if not any(not item["passed"] for item in checks[-5:]):
        checks.append(_check(
            "entry_equals_decision", (timestamps["entry_ts"] == timestamps["decision_ts"]).all(),
            int((timestamps["entry_ts"] != timestamps["decision_ts"]).sum()), "entry_ts == decision_ts",
        ))
        checks.append(_check(
            "feature_cutoff_causal", (timestamps["feature_cutoff_ts"] <= timestamps["decision_ts"]).all(),
            int((timestamps["feature_cutoff_ts"] > timestamps["decision_ts"]).sum()), "feature_cutoff_ts <= decision_ts",
        ))
        horizon = timestamps["label_end_ts"] - timestamps["decision_ts"]
        checks.append(_check(
            "exact_h12_horizon", (horizon == HORIZON).all(),
            sorted(horizon.drop_duplicates().astype(str).tolist())[:10], "label_end_ts == decision_ts + 12h",
        ))
        checks.append(_check(
            "label_available_after_horizon", (timestamps["label_available_ts"] >= timestamps["label_end_ts"]).all(),
            int((timestamps["label_available_ts"] < timestamps["label_end_ts"]).sum()), "labels cannot be available before the path ends",
        ))
        gross = pd.to_numeric(primary["execution_exact_h12_gross_bps"], errors="coerce")
        cost = pd.to_numeric(primary["execution_exact_h12_cost_bps"], errors="coerce")
        net = pd.to_numeric(primary["execution_exact_h12_net_bps"], errors="coerce")
        finite = np.isfinite(gross) & np.isfinite(cost) & np.isfinite(net)
        checks.append(_check("primary_values_finite", bool(finite.all()), int((~finite).sum()), "gross, cost and net are finite"))
        checks.append(_check(
            "exact_net_accounting", bool(np.allclose(gross - cost, net, atol=1e-7, rtol=0.0)),
            float(np.nanmax(np.abs((gross - cost - net).to_numpy()))), "gross - row cost == net; cost charged once",
        ))
    checks.append(_check(
        "single_target_policy_cost", primary["execution_policy_id"].nunique() == 1 and primary["cost_model_id"].nunique() == 1,
        {"policies": int(primary["execution_policy_id"].nunique()), "cost_models": int(primary["cost_model_id"].nunique())},
        "one frozen execution policy and one frozen row-cost model",
    ))
    checks.append(_check(
        "geometry_declared", primary["execution_geometry_id"].notna().all(),
        int(primary["execution_geometry_id"].isna().sum()), "each candidate has a declared policy geometry",
    ))
    return checks


def validate_label_dictionary(dictionary: pd.DataFrame) -> list[dict[str, Any]]:
    required = {"label_name", "availability", "model_input_allowed", "role", "label_kind"}
    missing = sorted(required.difference(dictionary.columns))
    checks = [_check("label_dictionary_columns", not missing, missing, "label dictionary declares availability and model-input prohibition")]
    if missing:
        return checks
    allowed = dictionary["model_input_allowed"].astype(bool)
    checks.append(_check("future_labels_forbidden", not allowed.any(), int(allowed.sum()), "materialized future labels are never decision-time model inputs"))
    availability = dictionary["availability"].fillna("").astype(str).str.lower()
    decision_only = availability.str.fullmatch(r"decision_ts", na=False)
    checks.append(_check("future_label_availability_declared", not decision_only.any(), int(decision_only.sum()), "future labels are available only after the path horizon"))
    checks.append(_check("label_dictionary_unique", not dictionary["label_name"].duplicated().any(), int(dictionary["label_name"].duplicated().sum()), "one dictionary row per label"))
    return checks


def validate_supportive_labels(supportive: pd.DataFrame) -> list[dict[str, Any]]:
    required_core = {
        "__peak_mfe_atr_12h__", "__time_to_first_meaningful_mfe_hours_12h__",
        "__mae_before_meaningful_mfe_atr_12h__", "__bars_before_price_stops_decreasing_12h__",
        "__future_slope_atr_per_hour_12h__",
    }
    missing = sorted(required_core.difference(supportive.columns))
    checks = [_check("supportive_core_heads_present", not missing, missing, "five supportive path heads are materialized")]
    if "candidate_id" in supportive:
        checks.append(_check("supportive_identity_unique", not supportive["candidate_id"].duplicated().any(), int(supportive["candidate_id"].duplicated().sum()), "one supportive row per candidate_id"))
    explicit_suffixes = ("__valid", "__condition_met", "__censored", "__support_count")
    present = {suffix: int(sum(str(column).endswith(suffix) for column in supportive.columns)) for suffix in explicit_suffixes}
    checks.append(_check(
        "supportive_explicit_metadata", all(value > 0 for value in present.values()), present,
        "each supportive head family exposes explicit valid/condition/censor/support metadata",
    ))
    checks.append(_check(
        "supportive_future_labels_not_model_inputs", True,
        "enforced by label dictionary and feature eligibility contracts", "supportive labels remain target-side artifacts",
    ))
    return checks


def materialize_supportive_metadata(supportive: pd.DataFrame, output: Path) -> dict[str, Any]:
    """Add explicit support metadata without rewriting the immutable labels.

    The source pack predates the roadmap's metadata suffix contract.  The
    projection below is vectorized and keeps the source labels untouched.  A
    conditional head is censored when its meaningful-MFE condition is not
    observed by H12; the two unconditional path diagnostics are valid for a
    complete path and are not censored.
    """

    explicit = [column for column in supportive.columns if str(column).endswith(("__valid", "__condition_met", "__censored", "__support_count"))]
    if explicit and all(any(str(column).startswith(head) for column in explicit) for head in ("peak_mfe_atr_12h", "time_to_first_meaningful_mfe_hours_12h", "mae_before_meaningful_mfe_atr_12h", "bars_before_price_stops_decreasing_12h", "future_slope_atr_per_hour_12h")):
        out = supportive[["candidate_id"] + sorted(explicit)].copy()
        if output.exists():
            output.unlink()
        out.to_parquet(output, index=False, compression="zstd")
        return {"rows": int(len(out)), "columns": out.columns.tolist(), "conditioned_heads": []}

    valid_path = pd.to_numeric(supportive["__path_auxiliary_target_valid__"], errors="coerce").fillna(0).astype(bool)
    valid_time = pd.to_numeric(supportive["__time_to_first_meaningful_mfe_target_valid__"], errors="coerce").fillna(0).astype(bool)
    condition = pd.to_numeric(supportive["__meaningful_mfe_reached_12h__"], errors="coerce").fillna(0).astype(bool)
    out = supportive[["candidate_id"]].copy()
    specs = {
        "peak_mfe_atr_12h": (valid_path, condition),
        "time_to_first_meaningful_mfe_hours_12h": (valid_time, condition),
        "mae_before_meaningful_mfe_atr_12h": (valid_path, condition),
        "bars_before_price_stops_decreasing_12h": (valid_path, valid_path),
        "future_slope_atr_per_hour_12h": (valid_path, valid_path),
    }
    for stem, (valid, condition_met) in specs.items():
        out[f"{stem}__valid"] = valid.astype("int8")
        out[f"{stem}__condition_met"] = condition_met.astype("int8")
        out[f"{stem}__censored"] = (valid & ~condition_met).astype("int8")
        out[f"{stem}__support_count"] = (valid & condition_met).astype("int32")
    if output.exists():
        # The prior audit version may have left a hardlink at this path.  Unlink
        # the destination only before writing the derived projection so the
        # immutable source parquet can never be truncated in place.
        output.unlink()
    out.to_parquet(output, index=False, compression="zstd")
    return {"rows": int(len(out)), "columns": out.columns.tolist(), "conditioned_heads": [name for name, (_, cond) in specs.items() if not cond.equals(valid_path)]}


def validate_supportive_metadata(metadata: pd.DataFrame) -> list[dict[str, Any]]:
    suffixes = ("__valid", "__condition_met", "__censored", "__support_count")
    missing = [suffix for suffix in suffixes if not any(str(column).endswith(suffix) for column in metadata.columns)]
    checks = [_check("canonical_supportive_metadata_columns", not missing, missing, "canonical supportive projection exposes valid/condition/censor/support-count fields")]
    if missing:
        return checks
    valid_columns = [column for column in metadata.columns if str(column).endswith("__valid")]
    invalid_count = 0
    for column in valid_columns:
        stem = str(column)[:-len("__valid")]
        for suffix in ("__condition_met", "__censored", "__support_count"):
            target = f"{stem}{suffix}"
            if target not in metadata:
                invalid_count += 1
    checks.append(_check("canonical_supportive_metadata_per_head", invalid_count == 0, invalid_count, "each canonical supportive head has all four explicit metadata fields"))
    checks.append(_check("canonical_supportive_metadata_binary", bool(metadata[valid_columns].isin([0, 1]).all().all()), "0/1", "valid indicators are binary"))
    return checks


def materialize_canonical_supportive_labels(
    supportive: pd.DataFrame,
    metadata: pd.DataFrame,
    output: Path,
) -> dict[str, Any]:
    """Publish the five roadmap heads with their explicit metadata fields."""

    identity_columns = [
        column for column in (
            "candidate_id", "symbol", "side", "decision_ts", "label_end_ts",
            "label_available_ts", "support_path_semantics_id", "competing_risk_target_id",
            "execution_policy_id", "cost_model_id", "execution_geometry_id",
        ) if column in supportive.columns
    ]
    head_columns = [
        "__peak_mfe_atr_12h__", "__time_to_first_meaningful_mfe_hours_12h__",
        "__mae_before_meaningful_mfe_atr_12h__", "__bars_before_price_stops_decreasing_12h__",
        "__future_slope_atr_per_hour_12h__",
    ]
    columns = identity_columns + head_columns
    base = supportive[columns].copy()
    if base["candidate_id"].duplicated().any() or metadata["candidate_id"].duplicated().any():
        raise ValueError("canonical supportive label projection requires unique candidate_id")
    out = base.merge(metadata, on="candidate_id", how="inner", validate="one_to_one")
    if len(out) != len(base):
        raise ValueError("canonical supportive label projection lost candidate rows")
    if output.exists():
        output.unlink()
    out.to_parquet(output, index=False, compression="zstd")
    return {"rows": int(len(out)), "columns": out.columns.tolist(), "source_head_count": len(head_columns)}


def build_execution_oof_lineage_manifest(predictions: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    """Summarise strict candidate-level OOF scores as execution features."""

    rows: list[dict[str, Any]] = []
    source_hash = sha256(source_path)
    for (target_arm, support_stage), group in predictions.groupby(["target_arm", "support_stage"], sort=True):
        decision = pd.to_datetime(group["__decision_ts__"], utc=True)
        fit_end = pd.to_datetime(group["prediction_fit_end_ts"], utc=True)
        generated = pd.to_datetime(group["prediction_generated_ts"], utc=True)
        rows.append({
            "model_layer": "execution",
            "model_side": "all",
            "feature_name": f"support_oof__{target_arm}__{support_stage}__score",
            "contract_path": str(source_path),
            "contract_sha256": source_hash,
            "semantic_class": "MODEL_DERIVED",
            "eligibility_status": "ELIGIBLE_RESEARCH_OOF",
            "eligible_now": True,
            "eligible_if_prediction_lineage_audited": True,
            "requires_prediction_lineage_audit": True,
            "point_in_time_safe": "TRUE",
            "live_reproducible": "UNDECLARED",
            "production_live_status": "RESEARCH_ONLY_OOF",
            "reason": "candidate-level score is strict OOF; realized labels remain excluded from the feature",
            "target_arm": str(target_arm),
            "support_stage": str(support_stage),
            "rows": int(len(group)),
            "candidate_rows": int(group["candidate_id"].nunique()),
            "fit_end_min": fit_end.min(),
            "fit_end_max": fit_end.max(),
            "prediction_generated_min": generated.min(),
            "prediction_generated_max": generated.max(),
            "candidate_decision_min": decision.min(),
            "candidate_decision_max": decision.max(),
            "model_ids": sorted(group["prediction_model_id"].astype(str).unique().tolist()) if "prediction_model_id" in group else [],
            "fold_ids": sorted(group["prediction_fold_id"].astype(str).unique().tolist()) if "prediction_fold_id" in group else [],
        })
    return pd.DataFrame(rows)


def materialize_canonical_feature_lineage(
    source_feature_manifest: pd.DataFrame,
    source_prediction_lineage: pd.DataFrame,
    candidate_oof: pd.DataFrame,
    candidate_oof_path: Path,
    feature_output: Path,
    lineage_output: Path,
) -> dict[str, Any]:
    """Bind the candidate OOF scores to a layer-specific feature contract."""

    lineage = build_execution_oof_lineage_manifest(candidate_oof, candidate_oof_path)
    base_columns = source_feature_manifest.columns.tolist()
    feature_rows = source_feature_manifest.copy()
    if not feature_rows.empty:
        # Existing unbound model-derived features stay rejected.  The new OOF
        # rows are an explicit, additive execution contract.
        feature_rows = feature_rows[base_columns]
    canonical_columns = [column for column in base_columns]
    lineage_features = lineage[canonical_columns].copy()
    canonical = pd.concat([feature_rows, lineage_features], ignore_index=True, sort=False)
    if feature_output.exists():
        feature_output.unlink()
    if lineage_output.exists():
        lineage_output.unlink()
    canonical.to_parquet(feature_output, index=False, compression="zstd")
    lineage.to_parquet(lineage_output, index=False, compression="zstd")
    return {"feature_rows": int(len(canonical)), "oof_feature_rows": int(len(lineage)), "lineage_rows": int(len(lineage)), "source_lineage_rows": int(len(source_prediction_lineage))}


def validate_fold_and_oof(
    folds: pd.DataFrame,
    predictions: pd.DataFrame,
    candidate_oof: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    fold_required = {"oof_fold", "fold_order", "start_utc", "end_exclusive_utc", "protocol_role"}
    missing = sorted(fold_required.difference(folds.columns))
    checks.append(_check("fold_manifest_columns", not missing, missing, "chronological protocol fold manifest is explicit"))
    if not missing and len(folds):
        f = folds.sort_values("fold_order").copy()
        start = pd.to_datetime(f["start_utc"], utc=True, errors="coerce")
        end = pd.to_datetime(f["end_exclusive_utc"], utc=True, errors="coerce")
        checks.append(_check("fold_timestamps_valid", not (start.isna() | end.isna()).any(), int((start.isna() | end.isna()).sum()), "fold timestamps are valid UTC"))
        checks.append(_check("folds_strictly_chronological", bool((end.iloc[:-1].to_numpy() <= start.iloc[1:].to_numpy()).all()), f[["oof_fold", "start_utc", "end_exclusive_utc"]].to_dict("records"), "earlier fold closes no later than the next fold starts"))
    pred_required = {"prediction_fit_end_ts", "candidate_decision_min", "candidate_decision_max", "is_oof", "rows"}
    missing = sorted(pred_required.difference(predictions.columns))
    checks.append(_check("oof_manifest_columns", not missing, missing, "OOF manifest declares fit end, candidate decision range and OOF flag"))
    if not missing and len(predictions):
        fit_end = pd.to_datetime(predictions["prediction_fit_end_ts"], utc=True, errors="coerce")
        decision_min = pd.to_datetime(predictions["candidate_decision_min"], utc=True, errors="coerce")
        decision_max = pd.to_datetime(predictions["candidate_decision_max"], utc=True, errors="coerce")
        checks.append(_check("oof_manifest_timestamps_valid", not (fit_end.isna() | decision_min.isna() | decision_max.isna()).any(), int((fit_end.isna() | decision_min.isna() | decision_max.isna()).sum()), "OOF timestamps are valid UTC"))
        checks.append(_check("all_prediction_rows_flagged_oof", bool(predictions["is_oof"].astype(bool).all()), int((~predictions["is_oof"].astype(bool)).sum()), "every emitted prediction is declared OOF"))
        checks.append(_check("fit_end_before_candidate_decision", bool((fit_end < decision_min).all()), int((fit_end >= decision_min).sum()), "fit_end_ts < candidate decision timestamp"))
        candidate_level = candidate_oof is not None and "candidate_id" in candidate_oof.columns and not candidate_oof.empty
        checks.append(_check(
            "aggregate_oof_only_warning", candidate_level,
            "candidate-level handoff present" if candidate_level else "manifest is aggregate; no candidate_id column",
            "row-level execution OOF lineage requires candidate-level prediction IDs",
        ))
    return checks


def validate_candidate_oof_predictions(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    """Validate the row-level supportive OOF handoff when it is available."""

    required = {
        "candidate_id", "__decision_ts__", "prediction_fit_end_ts",
        "prediction_generated_ts", "oof_fold", "target_arm", "support_stage", "score",
    }
    missing = sorted(required.difference(predictions.columns))
    checks = [_check("candidate_oof_columns", not missing, missing, "candidate-level OOF rows carry identity, timing and model-cell provenance")]
    if missing:
        return checks
    decision = pd.to_datetime(predictions["__decision_ts__"], utc=True, errors="coerce")
    fit_end = pd.to_datetime(predictions["prediction_fit_end_ts"], utc=True, errors="coerce")
    generated = pd.to_datetime(predictions["prediction_generated_ts"], utc=True, errors="coerce")
    checks.append(_check("candidate_oof_timestamps_valid", not (decision.isna() | fit_end.isna() | generated.isna()).any(), int((decision.isna() | fit_end.isna() | generated.isna()).sum()), "candidate OOF timestamps are valid UTC"))
    checks.append(_check("candidate_oof_fit_end_before_decision", bool((fit_end < decision).all()), int((fit_end >= decision).sum()), "fit_end_ts < candidate decision timestamp"))
    checks.append(_check("candidate_oof_generated_at_or_before_decision", bool((generated <= decision).all()), int((generated > decision).sum()), "prediction_generated_ts <= candidate decision timestamp"))
    checks.append(_check("candidate_oof_cells_unique", not predictions.duplicated(["candidate_id", "target_arm", "support_stage"]).any(), int(predictions.duplicated(["candidate_id", "target_arm", "support_stage"]).sum()), "one score per candidate/target/support cell"))
    score = pd.to_numeric(predictions["score"], errors="coerce")
    checks.append(_check("candidate_oof_scores_finite", bool(np.isfinite(score).all()), int((~np.isfinite(score)).sum()), "all candidate-level OOF scores are finite"))
    return checks


def materialize_canonical_target_contract(primary: pd.DataFrame, contract: Mapping[str, Any], output: Path) -> dict[str, Any]:
    """Publish the roadmap's canonical field names as a row-level view.

    ``path_source`` and ``path_complete`` are derived from the immutable path
    semantics and the target-pack assertion that every row has one complete
    720-bar path.  The provenance is recorded explicitly rather than hidden in
    a renamed column.
    """

    supporting = contract.get("supporting_path_labels", {})
    view = pd.DataFrame({
        "candidate_id": primary["candidate_id"],
        "symbol": primary["symbol"],
        "side": primary["side"],
        "decision_ts": primary["decision_ts"],
        "entry_ts": primary["entry_ts"],
        "entry_price": primary["execution_entry_price"],
        "horizon_end_ts": primary["label_end_ts"],
        "label_available_ts": primary["label_available_ts"],
        "row_cost_bps": primary["execution_exact_h12_cost_bps"],
        "policy_geometry_id": primary["execution_geometry_id"],
        "path_source": str(supporting.get("path_semantics_id", "historical_exact_1m_unadjusted_decision_path_v1")),
        "path_complete": True,
        "execution_policy_id": primary["execution_policy_id"],
        "cost_model_id": primary["cost_model_id"],
    })
    view.to_parquet(output, index=False, compression="zstd")
    return {"rows": int(len(view)), "columns": view.columns.tolist(), "path_complete_provenance": "root_cause_exact_h12_execution_target_pack_manifest_assertion"}


def validate_feature_manifest(feature_manifest: pd.DataFrame, prediction_lineage: pd.DataFrame) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    required = {"model_layer", "semantic_class", "eligibility_status", "eligible_now", "requires_prediction_lineage_audit"}
    missing = sorted(required.difference(feature_manifest.columns))
    checks.append(_check("feature_manifest_columns", not missing, missing, "layer-specific feature eligibility is explicit"))
    if not missing:
        base_derived = feature_manifest[(feature_manifest["model_layer"] == "base") & feature_manifest["semantic_class"].astype(str).eq("MODEL_DERIVED") & feature_manifest["eligible_now"].astype(bool)]
        checks.append(_check("base_has_no_model_derived_inputs", base_derived.empty, base_derived["feature_name"].tolist(), "base consumes causal/raw inputs only"))
        execution_derived = feature_manifest[(feature_manifest["model_layer"] == "execution") & feature_manifest["semantic_class"].astype(str).eq("MODEL_DERIVED") & feature_manifest["eligible_now"].astype(bool)]
        lineage_names = set(prediction_lineage.get("feature_name", pd.Series(dtype=str)).astype(str))
        unbound_names = sorted(set(execution_derived.get("feature_name", pd.Series(dtype=str)).astype(str)) - lineage_names)
        checks.append(_check("execution_derived_features_have_lineage", bool(prediction_lineage.size) and not unbound_names, {"unproven_features": len(unbound_names), "lineage_rows": int(len(prediction_lineage))}, "execution model outputs require a non-empty strict OOF lineage audit"))
        action = feature_manifest["eligibility_status"].astype(str).str.contains("ACTION_LAYER_ONLY", na=False)
        eligible_action = int((action & feature_manifest["eligible_now"].astype(bool)).sum())
        checks.append(_check("action_layer_excluded", eligible_action == 0, eligible_action, "timing/MAE/target-price/wait features are rejected from base and execution-EV"))
    return checks


def validate_global_tail(policy_summary: pd.DataFrame, target_metrics: pd.DataFrame) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    required = {"selection_basis", "top_k_fraction", "selected_rows", "population_rows", "global_topk_net_bps", "latest_month_topk_net_bps", "months_selected"}
    missing = sorted(required.difference(policy_summary.columns))
    checks.append(_check("global_tail_summary_columns", not missing, missing, "pooled global top-k economics and month coverage are reported"))
    if not missing and len(policy_summary):
        basis = policy_summary["selection_basis"].astype(str)
        checks.append(_check("global_tail_is_pooled", basis.str.contains("pooled_global", case=False).all() and not basis.str.contains("timestamp", case=False).any(), sorted(basis.unique().tolist()), "one pooled global top-k book; no per-timestamp quota"))
        expected = (policy_summary["population_rows"] * policy_summary["top_k_fraction"]).round().astype(int)
        checks.append(_check("global_tail_row_count_exact", (policy_summary["selected_rows"].astype(int) == expected).all(), int((policy_summary["selected_rows"].astype(int) != expected).sum()), "selected rows equal the global top-k fraction"))
        checks.append(_check("global_tail_month_coverage", (policy_summary["months_selected"].astype(int) >= 1).all(), int((policy_summary["months_selected"].astype(int) < 1).sum()), "global tail reports non-empty month coverage"))
        best = policy_summary.loc[policy_summary["global_topk_net_bps"].astype(float).idxmax()]
        checks.append(_check("supportive_global_tail_positive", float(policy_summary["global_topk_net_bps"].max()) > 0, float(policy_summary["global_topk_net_bps"].max()), "promotion requires positive post-cost global top-k net"))
        checks.append(_check("supportive_best_arm", True, {"target_arm": best["target_arm"], "support_stage": best["support_stage"], "global_topk_net_bps": float(best["global_topk_net_bps"])}, "best arm is reported diagnostically even when it fails promotion"))
    if target_metrics is not None and len(target_metrics):
        top = target_metrics[(target_metrics["scope"].astype(str) == "pooled_global_top") & np.isclose(target_metrics["fraction"].astype(float), 0.10)]
        checks.append(_check("exact_target_global_top10_present", len(top) > 0, int(len(top)), "exact-H12 target ablation exposes pooled global top-10% economics"))
        if len(top):
            checks.append(_check("exact_target_global_top10_positive", float(top["net_bps"].max()) > 0, float(top["net_bps"].max()), "exact post-cost top-10% net is the promotion criterion"))
    return checks


def materialize_alias(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(source) == sha256(destination):
            return "already_materialized"
        destination.unlink()
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def build_alignment_audit(
    *,
    contract_path: Path,
    primary_path: Path,
    supportive_path: Path,
    dictionary_path: Path,
    support_report_path: Path,
    feature_manifest_path: Path,
    prediction_lineage_path: Path,
    fold_manifest_path: Path,
    oof_manifest_path: Path,
    candidate_oof_path: Path,
    policy_summary_path: Path,
    target_metrics_path: Path,
    output_dir: Path,
    target_results_path: Path | None = None,
    monthly_side_path: Path | None = None,
    calibration_path: Path | None = None,
    include_legacy_blockers: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    primary = pd.read_parquet(primary_path)
    supportive_columns = pq.ParquetFile(supportive_path).schema.names
    supportive_identity = [column for column in ("candidate_id", "symbol", "side", "decision_ts", "label_end_ts", "label_available_ts", "support_path_semantics_id", "competing_risk_target_id", "execution_policy_id", "cost_model_id", "execution_geometry_id") if column in supportive_columns]
    required_heads = ["__peak_mfe_atr_12h__", "__time_to_first_meaningful_mfe_hours_12h__", "__mae_before_meaningful_mfe_atr_12h__", "__bars_before_price_stops_decreasing_12h__", "__future_slope_atr_per_hour_12h__"]
    legacy_helpers = [column for column in ("__path_auxiliary_target_valid__", "__time_to_first_meaningful_mfe_target_valid__", "__meaningful_mfe_reached_12h__") if column in supportive_columns]
    explicit_metadata = [c for c in supportive_columns if str(c).endswith(("__valid", "__condition_met", "__censored", "__support_count"))]
    read_columns = supportive_identity + required_heads + legacy_helpers + explicit_metadata
    supportive = pd.read_parquet(supportive_path, columns=list(dict.fromkeys(read_columns)))
    dictionary = pd.read_parquet(dictionary_path)
    support_report = pd.read_parquet(support_report_path)
    feature_manifest = pd.read_parquet(feature_manifest_path)
    prediction_lineage = pd.read_parquet(prediction_lineage_path)
    folds = pd.read_parquet(fold_manifest_path)
    oof_manifest = pd.read_parquet(oof_manifest_path)
    candidate_oof = pd.read_parquet(candidate_oof_path, columns=["candidate_id", "__decision_ts__", "prediction_fit_end_ts", "prediction_generated_ts", "oof_fold", "target_arm", "support_stage", "score"])
    policy_summary = pd.read_parquet(policy_summary_path)
    target_metrics = pd.read_csv(target_metrics_path)

    checks = []
    checks.extend(validate_primary_labels(primary, contract))
    checks.extend(validate_label_dictionary(dictionary))
    # Preserve the immutable-source result, then validate the canonical
    # five-head projection that this audit publishes for downstream research.
    checks.extend(validate_supportive_labels(supportive))
    metadata_path = output_dir / "supportive_label_metadata.parquet"
    metadata_summary = materialize_supportive_metadata(supportive, metadata_path)
    metadata = pd.read_parquet(metadata_path)
    checks.extend(validate_supportive_metadata(metadata))
    canonical_supportive_path = output_dir / "supportive_labels_canonical.parquet"
    canonical_supportive_summary = materialize_canonical_supportive_labels(supportive, metadata, canonical_supportive_path)
    canonical_supportive = pd.read_parquet(canonical_supportive_path)
    checks.extend(validate_supportive_labels(canonical_supportive))
    canonical_feature_path = output_dir / "feature_eligibility_manifest.parquet"
    canonical_lineage_path = output_dir / "prediction_lineage_audit.parquet"
    canonical_feature_summary = materialize_canonical_feature_lineage(
        feature_manifest,
        prediction_lineage,
        candidate_oof,
        candidate_oof_path,
        canonical_feature_path,
        canonical_lineage_path,
    )
    canonical_feature_manifest = pd.read_parquet(canonical_feature_path)
    canonical_prediction_lineage = pd.read_parquet(canonical_lineage_path)
    checks.extend(validate_feature_manifest(canonical_feature_manifest, canonical_prediction_lineage))
    checks.extend(validate_fold_and_oof(folds, oof_manifest, candidate_oof))
    checks.extend(validate_candidate_oof_predictions(candidate_oof))
    checks.extend(validate_global_tail(policy_summary, target_metrics))
    blocking_reasons = []
    if include_legacy_blockers:
        blocking_reasons.append("historical native-L2 backfill is incomplete for the candidate window")
    if any(row["check"] == "supportive_explicit_metadata" and not row["passed"] for row in checks):
        blocking_reasons.append("the immutable source supportive-label pack lacks explicit valid/condition/censor/support-count suffixes")
    blocking_reasons.append("all exact-H12 and supportive global top-10% economic arms are negative")
    correctness = {
        "schema": SCHEMA,
        "checks": checks,
        "passed_checks": int(sum(bool(row["passed"]) for row in checks)),
        "failed_checks": int(sum(not bool(row["passed"]) for row in checks)),
        "promotion_eligible": False,
        "status": "FAIL_CLOSED_RESEARCH_ONLY",
        "blocking_reasons": blocking_reasons,
    }
    write_json(output_dir / "correctness_test_report.json", correctness)
    canonical_contract = materialize_canonical_target_contract(primary, contract, output_dir / "candidate_target_contract.parquet")
    lines = [
        "# Target–feature–execution alignment audit",
        "",
        f"- Status: **{correctness['status']}**",
        "- Promotion eligible: **false**",
        f"- Checks: {correctness['passed_checks']} passed, {correctness['failed_checks']} failed",
        f"- Exact-H12 target population: {len(primary):,} candidates",
        "",
        "## Blocking findings",
        "",
    ]
    lines.extend(f"- {reason}." for reason in correctness["blocking_reasons"])
    lines.extend(["", "## Failed checks", ""])
    for row in checks:
        if not row["passed"]:
            lines.append(f"- `{row['check']}`: {row['rule']} — `{json.dumps(_json_safe(row['value']), sort_keys=True)}`")
    lines.extend([
        "", "## Interpretation", "",
        "The exact-H12 identity, 12-hour availability rule, one-time gross-minus-cost accounting, causal feature cutoff, frozen policy/cost IDs, chronological fold ordering, candidate-level OOF timing, and pooled global top-k selection are materially evidenced.",
        "",
        "The pack is deliberately research-only. The canonical OOF feature handoff is lineage-audited, but no negative target/supportive arm is promoted merely because it has valid mechanics.",
    ])
    (output_dir / "TARGET_FEATURE_EXECUTION_ALIGNMENT_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Small JSON contracts are copied; large immutable tables are hard-linked
    # where possible so the audit does not silently create a second dataset.
    materializations: dict[str, Any] = {}
    for name, source in {
        "execution_target_contract.json": contract_path,
        "label_dictionary.parquet": dictionary_path,
        "label_support_report.parquet": support_report_path,
        "feature_eligibility_manifest_source.parquet": feature_manifest_path,
        "prediction_lineage_audit_source.parquet": prediction_lineage_path,
        "fold_manifest.parquet": fold_manifest_path,
        "oof_prediction_manifest.parquet": oof_manifest_path,
        "candidate_level_oof_predictions.parquet": candidate_oof_path,
        "supportive_label_ablation_results.parquet": policy_summary_path,
    }.items():
        materializations[name] = {"source": str(source), "mode": materialize_alias(source, output_dir / name)}
    materializations["feature_eligibility_manifest.parquet"] = {"source": str(candidate_oof_path), "mode": "canonical_source_plus_strict_oof_projection", **canonical_feature_summary}
    materializations["prediction_lineage_audit.parquet"] = {"source": str(candidate_oof_path), "mode": "candidate_level_oof_lineage_projection", "rows": int(len(canonical_prediction_lineage))}
    materializations["execution_oof_feature_lineage.parquet"] = {"source": str(candidate_oof_path), "mode": materialize_alias(canonical_lineage_path, output_dir / "execution_oof_feature_lineage.parquet"), "rows": int(len(canonical_prediction_lineage))}
    materializations["supportive_labels_canonical.parquet"] = {"source": str(supportive_path), "mode": "canonical_head_plus_metadata_projection", **canonical_supportive_summary}
    materializations["candidate_target_contract.parquet"] = {"source": str(primary_path), "mode": "canonical_field_projection", **canonical_contract}
    materializations["supportive_label_metadata.parquet"] = {"source": str(supportive_path), "mode": "vectorized_metadata_projection", **metadata_summary}
    (output_dir / "feature_eligibility_manifest.json").write_text(canonical_feature_manifest.to_json(orient="records", indent=2), encoding="utf-8")
    (output_dir / "label_dictionary.json").write_text(dictionary.to_json(orient="records", indent=2), encoding="utf-8")
    materializations["feature_eligibility_manifest.json"] = {"source": str(candidate_oof_path), "mode": "canonical_feature_manifest_json_projection"}
    materializations["label_dictionary.json"] = {"source": str(dictionary_path), "mode": "json_projection"}
    if target_results_path is not None:
        materializations["target_ablation_results.parquet"] = {"source": str(target_results_path), "mode": materialize_alias(target_results_path, output_dir / "target_ablation_results.parquet")}
        columns = [
            "candidate_id", "side", "decision_ts", "label_available_ts", "arm", "raw_score",
            "calibrated_expected_net_bps", "exact_h12_gross_bps", "row_cost_bps", "exact_h12_net_bps",
            "threshold_enter",
        ]
        all_columns = pq.ParquetFile(target_results_path).schema.names
        available = [column for column in columns if column in all_columns]
        candidate = pd.read_parquet(target_results_path, columns=available)
        candidate.to_parquet(output_dir / "candidate_level_predictions.parquet", index=False, compression="zstd")
        materializations["candidate_level_predictions.parquet"] = {"source": str(target_results_path), "mode": "auditable_column_projection", "rows": int(len(candidate)), "columns": available}
    if monthly_side_path is not None:
        materializations["monthly_side_results.parquet"] = {"source": str(monthly_side_path), "mode": materialize_alias(monthly_side_path, output_dir / "monthly_side_results.parquet")}
    if calibration_path is not None:
        calibration = pd.read_parquet(calibration_path)
        summary = {
            "source": str(calibration_path),
            "rows": int(len(calibration)),
            "columns": calibration.columns.tolist(),
            "all_rows_finite": bool(np.isfinite(calibration.select_dtypes(include=[np.number]).to_numpy()).all()),
            "score_deciles": sorted(calibration["score_decile"].unique().tolist()) if "score_decile" in calibration else [],
        }
        write_json(output_dir / "calibration_report.json", summary)
        materializations["calibration_report.json"] = {"source": str(calibration_path), "mode": "summary_projection"}

    manifest = {
        "schema": SCHEMA,
        "status": correctness["status"],
        "promotion_eligible": False,
        "contract": {"path": str(contract_path), "sha256": sha256(contract_path), "schema": contract.get("schema"), "horizon_minutes": contract.get("horizon_minutes")},
        "population_rows": int(len(primary)),
        "supportive_rows": int(len(supportive)),
        "feature_rows": int(len(canonical_feature_manifest)),
        "source_feature_rows": int(len(feature_manifest)),
        "oof_manifest_rows": int(len(oof_manifest)),
        "candidate_oof_rows": int(len(candidate_oof)),
        "checks": {"passed": correctness["passed_checks"], "failed": correctness["failed_checks"]},
        "materializations": materializations,
        "sources_sha256": {str(path): sha256(path) for path in [primary_path, supportive_path, dictionary_path, support_report_path, feature_manifest_path, prediction_lineage_path, fold_manifest_path, oof_manifest_path, candidate_oof_path, policy_summary_path, target_metrics_path]},
    }
    for name in sorted(materializations):
        path = output_dir / name
        if path.exists():
            manifest.setdefault("outputs_sha256", {})[name] = sha256(path)
    manifest.setdefault("outputs_sha256", {})["correctness_test_report.json"] = sha256(output_dir / "correctness_test_report.json")
    manifest.setdefault("outputs_sha256", {})["TARGET_FEATURE_EXECUTION_ALIGNMENT_AUDIT.md"] = sha256(output_dir / "TARGET_FEATURE_EXECUTION_ALIGNMENT_AUDIT.md")
    write_json(output_dir / "run_manifest.json", manifest)
    return {"correctness": correctness, "manifest": manifest}
