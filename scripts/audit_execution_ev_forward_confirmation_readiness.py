#!/usr/bin/env python3
"""Audit and freeze a genuinely-forward execution-EV confirmation block.

The audit has two deliberately separate stages:

``source_lock``
    Before the first eligible decision, verify final-refit model lineage, the
    causal calibrator seed, global admission semantics, and immutable
    policy/cost inputs.  Only this stage may write the source lock.

``preoutcome``
    After decisions have accumulated but before their outcomes are opened for
    evaluation, re-verify the source lock and the live-equivalent scored
    population, point-in-time availability, and coverage gates.

``confirmation``
    Re-verify the source lock and require exact 12-hour execution outcomes on
    the identical population.  Outcomes are never needed to create the lock.

Missing inputs produce a machine-readable ``not_ready`` report.  They never
silently reduce the cohort, substitute OOF predictions for final models, or
relax the frozen global-ranking contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SPEC_SCHEMA = "execution_ev_forward_confirmation_candidate_v1"
REPORT_SCHEMA = "execution_ev_forward_confirmation_readiness_v1"
LOCK_SCHEMA = "execution_ev_forward_confirmation_source_lock_v1"
REQUIRED_MODEL_ROLES = (
    "base_long",
    "base_short",
    "residual_long",
    "residual_short",
    "frozen_representation_long",
    "frozen_representation_short",
    "peak_mfe_long",
    "peak_mfe_short",
    "path_catboost_long",
    "path_catboost_short",
    "clean_favorable_event_long",
    "clean_favorable_event_short",
    "direct_exact_net_ev_long",
    "direct_exact_net_ev_short",
    "capture_probability_long",
    "capture_probability_short",
)
IDENTITY_DEFAULT = ("candidate_id", "__ts__", "__symbol__", "side_name")
FORBIDDEN_MODEL_PROVENANCE = ("oof_prediction", "checkpoint_prediction", "fold_prediction")
REQUIRED_SOURCE_CODE = (
    "forward_preoutcome_orchestrator",
    "packb_final_refit_forward_scorer",
    "execution_ev_forward_preentry_materializer",
    "forward_final_head_fitter",
    "forward_calibrator_seed_builder",
    "forward_population_scorer",
    "readiness_auditor",
    "integrated_oof_reference_runner",
    "policy_label_materializer",
    "canonical_join_builder",
    "capture_label_materializer",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _stable_payload_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_safe(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _identity_hash(frame: pd.DataFrame, identity: Sequence[str]) -> str:
    ordered = frame.loc[:, list(identity)].astype(str).sort_values(
        list(identity), kind="mergesort"
    )
    payload = "\n".join(
        "\x1f".join(row)
        for row in ordered.itertuples(index=False, name=None)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _utc(value: object, *, field: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware UTC")
    return timestamp.tz_convert("UTC")


def _utc_series(series: pd.Series, *, field: str) -> pd.Series:
    if not isinstance(series.dtype, pd.DatetimeTZDtype):
        raise ValueError(f"{field} must be stored as timezone-aware UTC")
    converted = series.dt.tz_convert("UTC")
    if converted.isna().any():
        raise ValueError(f"{field} contains null timestamps")
    return converted


def _strict_boolean(values: pd.Series, *, field: str) -> pd.Series:
    """Parse booleans without treating arbitrary non-empty strings as true."""

    if pd.api.types.is_bool_dtype(values.dtype):
        if values.isna().any():
            raise ValueError(f"{field} contains null booleans")
        return values.astype(bool)
    normalized = values.astype("string").str.strip().str.lower()
    parsed = normalized.map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if parsed.isna().any():
        raise ValueError(f"{field} must contain only true/false or 1/0")
    return parsed.astype(bool)


def _resolve(path_value: object, *, root: Path) -> Path:
    path = Path(str(path_value))
    return path if path.is_absolute() else root / path


def _read_table(path: Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, usecols=list(columns) if columns else None)
    return pd.read_parquet(path, columns=list(columns) if columns else None)


def _file_check(
    record: Mapping[str, Any],
    *,
    root: Path,
    name: str,
    blockers: list[str],
) -> dict[str, Any]:
    path = _resolve(record.get("path", ""), root=root)
    result: dict[str, Any] = {"name": name, "path": path, "exists": path.is_file()}
    if not path.is_file():
        blockers.append(f"missing_file:{name}")
        return result
    actual = _sha256(path)
    expected = record.get("sha256")
    result.update({"sha256": actual, "expected_sha256": expected})
    if not expected:
        blockers.append(f"missing_expected_hash:{name}")
    elif actual != expected:
        blockers.append(f"hash_mismatch:{name}")
    return result


def audit_models(
    records: Iterable[Mapping[str, Any]],
    *,
    root: Path,
    first_decision_exclusive: pd.Timestamp,
    default_training_lineage: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Verify final-model presence and that every training label resolves first."""

    blockers: list[str] = []
    by_role = {str(record.get("role")): record for record in records}
    unknown = sorted(set(by_role).difference(REQUIRED_MODEL_ROLES))
    if unknown:
        blockers.extend(f"unknown_model_role:{role}" for role in unknown)
    results: list[dict[str, Any]] = []
    for role in REQUIRED_MODEL_ROLES:
        record = by_role.get(role)
        if record is None:
            blockers.append(f"missing_model_role:{role}")
            results.append({"role": role, "ready": False})
            continue
        file_result = _file_check(record, root=root, name=f"model:{role}", blockers=blockers)
        provenance = str(record.get("provenance", "")).lower()
        serialized = bool(record.get("serialized_final_refit", False))
        if not serialized:
            blockers.append(f"not_serialized_final_refit:{role}")
        if any(token in provenance for token in FORBIDDEN_MODEL_PROVENANCE):
            blockers.append(f"oof_or_checkpoint_model_provenance:{role}")
        outcome_free = bool(record.get("outcome_free", False))
        cutoff_field = (
            "reference_data_max_utc" if outcome_free else "training_label_end_max_utc"
        )
        cutoff_raw = record.get(cutoff_field)
        cutoff: pd.Timestamp | None = None
        if cutoff_raw is None:
            blockers.append(
                f"missing_{'reference_data' if outcome_free else 'training_label'}_cutoff:{role}"
            )
        else:
            try:
                cutoff = _utc(cutoff_raw, field=f"{role}.{cutoff_field}")
            except ValueError:
                blockers.append(
                    f"invalid_{'reference_data' if outcome_free else 'training_label'}_cutoff:{role}"
                )
            else:
                if cutoff >= first_decision_exclusive:
                    blockers.append(
                        f"{'reference_data' if outcome_free else 'training_label'}"
                        f"_cutoff_not_before_forward_block:{role}"
                    )
        feature_contract = record.get("feature_contract")
        if not isinstance(feature_contract, Mapping):
            blockers.append(f"missing_feature_contract:{role}")
            feature_result: dict[str, Any] | None = None
        else:
            feature_result = _file_check(
                feature_contract,
                root=root,
                name=f"feature_contract:{role}",
                blockers=blockers,
            )
        lineage_record = record.get("training_lineage", default_training_lineage)
        lineage_result: dict[str, Any] | None = None
        if not isinstance(lineage_record, Mapping):
            blockers.append(f"missing_training_lineage:{role}")
        else:
            lineage_result = _file_check(
                lineage_record,
                root=root,
                name=f"training_lineage:{role}",
                blockers=blockers,
            )
            if lineage_result["exists"]:
                try:
                    lineage = json.loads(Path(lineage_result["path"]).read_text())
                    entry = lineage["models"][role]
                except Exception as exc:
                    blockers.append(f"invalid_training_lineage:{role}")
                    lineage_result["read_error"] = str(exc)
                else:
                    if lineage.get("schema") != "execution_ev_forward_model_lineage_v1":
                        blockers.append(f"invalid_training_lineage_schema:{role}")
                    if entry.get("model_sha256") != file_result.get("sha256"):
                        blockers.append(f"training_lineage_model_hash_mismatch:{role}")
                    lineage_cutoff = entry.get(cutoff_field)
                    try:
                        lineage_cutoff_utc = _utc(
                            lineage_cutoff,
                            field=f"lineage.{role}.{cutoff_field}",
                        )
                    except (TypeError, ValueError):
                        blockers.append(f"training_lineage_cutoff_invalid:{role}")
                    else:
                        if cutoff is None or lineage_cutoff_utc != cutoff:
                            blockers.append(f"training_lineage_cutoff_mismatch:{role}")
        results.append(
            {
                "role": role,
                "serialized_final_refit": serialized,
                "provenance": provenance,
                "outcome_free": outcome_free,
                cutoff_field: cutoff,
                "model_file": file_result,
                "feature_contract": feature_result,
                "training_lineage": lineage_result,
                "ready": not any(item.endswith(f":{role}") for item in blockers),
            }
        )
    return results, blockers


def audit_scored_population(
    record: Mapping[str, Any],
    *,
    root: Path,
    first_decision_exclusive: pd.Timestamp,
    requested_last_decision: pd.Timestamp,
    minimum_scored_rows: int,
    minimum_global_topk_rows: int,
    minimum_complete_days: int,
    top_k_fraction: float,
) -> tuple[dict[str, Any], list[str], pd.DataFrame | None]:
    blockers: list[str] = []
    checked = _file_check(record, root=root, name="scored_population", blockers=blockers)
    if not checked["exists"]:
        return checked, blockers, None
    path = Path(checked["path"])
    full_columns = list(pd.read_parquet(path).columns)
    forbidden_exact = {
        "execution_net_ev_12h",
        "execution_gross_ev_12h",
        "execution_mfe_return_12h",
        "execution_mae_return_12h",
        "execution_label_end_utc",
        "exact_1m_path_complete",
    }
    forbidden_columns = sorted(
        column
        for column in full_columns
        if column in forbidden_exact
        or str(column).startswith(("target_", "label_"))
    )
    if forbidden_columns:
        blockers.append("scored_preoutcome_contains_outcome_or_label_columns")
        checked["forbidden_columns"] = forbidden_columns
    identity = tuple(record.get("identity_columns", IDENTITY_DEFAULT))
    decision_column = str(record.get("decision_column", "execution_decision_utc"))
    admitted_column = str(record.get("admitted_column", "globally_admitted"))
    coverage_member_column = str(
        record.get("coverage_member_column", admitted_column)
    )
    availability_columns = tuple(record.get("availability_columns", ()))
    score_column = str(record.get("mapped_score_column", "mapped_execution_ev"))
    required = list(
        dict.fromkeys(
            [
                *identity,
                decision_column,
                admitted_column,
                coverage_member_column,
                score_column,
                *availability_columns,
            ]
        )
    )
    try:
        frame = _read_table(path, required)
    except Exception as exc:
        blockers.append("scored_population_schema_or_read_failure")
        checked["read_error"] = str(exc)
        return checked, blockers, None
    missing = [column for column in required if column not in frame]
    if missing:
        blockers.append("scored_population_missing_required_columns")
        checked["missing_columns"] = missing
        return checked, blockers, None
    try:
        decisions = _utc_series(frame[decision_column], field=decision_column)
    except ValueError as exc:
        blockers.append("scored_population_decision_not_utc")
        checked["timestamp_error"] = str(exc)
        return checked, blockers, None
    frame = frame.copy()
    frame[decision_column] = decisions
    if frame.duplicated(list(identity)).any():
        blockers.append("scored_population_duplicate_identity")
    if len(frame) == 0:
        blockers.append("scored_population_empty")
    elif decisions.min() <= first_decision_exclusive:
        blockers.append("scored_population_not_strictly_after_frozen_cutoff")
    if len(frame) and decisions.max() > requested_last_decision:
        blockers.append("scored_population_exceeds_requested_endpoint")
    if len(frame) < minimum_scored_rows:
        blockers.append("minimum_scored_rows_not_met")
    score_values = pd.to_numeric(
        frame[score_column], errors="coerce"
    ).to_numpy(float)
    if not np.isfinite(score_values).all():
        blockers.append("scored_population_nonfinite_mapped_score")
    for column in availability_columns:
        try:
            available = _utc_series(frame[column], field=column)
        except ValueError:
            blockers.append(f"availability_not_utc:{column}")
            continue
        if (available > decisions).any():
            blockers.append(f"feature_available_after_decision:{column}")
    try:
        admitted = _strict_boolean(
            frame[admitted_column], field=admitted_column
        )
    except ValueError:
        blockers.append("globally_admitted_not_boolean")
        admitted = pd.Series(False, index=frame.index)
    admitted_rows = int(admitted.sum())
    try:
        coverage_member = _strict_boolean(
            frame[coverage_member_column], field=coverage_member_column
        )
    except ValueError:
        blockers.append("global_topk_coverage_member_not_boolean")
        coverage_member = pd.Series(False, index=frame.index)
    expected_capacity = np.zeros(len(frame), dtype=bool)
    if len(frame) and np.isfinite(score_values).all():
        if not 0.0 < top_k_fraction <= 1.0:
            raise ValueError("ranking top_k_fraction must be in (0, 1]")
        candidate_id = frame["candidate_id"].astype(str).to_numpy()
        order = np.lexsort((candidate_id, -score_values))
        count = max(1, int(math.ceil(top_k_fraction * len(frame))))
        expected_capacity[order[:count]] = True
        if not np.array_equal(
            coverage_member.to_numpy(dtype=bool), expected_capacity
        ):
            blockers.append("global_topk_capacity_membership_mismatch")
        expected_admitted = expected_capacity & (score_values > 0.0)
        if not np.array_equal(
            admitted.to_numpy(dtype=bool), expected_admitted
        ):
            blockers.append("economic_admission_membership_mismatch")
    global_topk_rows = int(expected_capacity.sum())
    if global_topk_rows < minimum_global_topk_rows:
        blockers.append("minimum_global_topk_coverage_rows_not_met")
    side_values = set(frame["side_name"].astype(str).str.lower()) if "side_name" in frame else set()
    if not {"long", "short"}.issubset(side_values):
        blockers.append("both_sides_not_present")
    coverage_record = record.get("daily_coverage")
    complete_days = 0
    coverage_result: dict[str, Any] | None = None
    if not isinstance(coverage_record, Mapping):
        blockers.append("missing_complete_day_coverage_source")
    else:
        coverage_result = _file_check(
            coverage_record,
            root=root,
            name="daily_coverage",
            blockers=blockers,
        )
        if coverage_result["exists"]:
            try:
                coverage = _read_table(
                    Path(coverage_result["path"]),
                    [
                        str(coverage_record.get("date_column", "utc_date")),
                        str(coverage_record.get("complete_column", "complete")),
                        str(
                            coverage_record.get(
                                "both_sides_column", "both_sides"
                            )
                        ),
                    ],
                )
                date_column = str(coverage_record.get("date_column", "utc_date"))
                complete_column = str(coverage_record.get("complete_column", "complete"))
                both_sides_column = str(
                    coverage_record.get("both_sides_column", "both_sides")
                )
                dates = pd.to_datetime(coverage[date_column], utc=True, errors="raise")
                complete = _strict_boolean(
                    coverage[complete_column], field=complete_column
                )
                both_sides = _strict_boolean(
                    coverage[both_sides_column], field=both_sides_column
                )
                normalized_dates = dates.dt.normalize()
                if normalized_dates.duplicated().any():
                    blockers.append("daily_coverage_duplicate_utc_date")
                if (complete & ~both_sides).any():
                    blockers.append("complete_day_missing_both_sides")
                complete_dates = set(normalized_dates.loc[complete & both_sides])
                if len(frame):
                    decision_dates = decisions.dt.normalize()
                    outside = {
                        date
                        for date in complete_dates
                        if date < decision_dates.min()
                        or date > decision_dates.max()
                    }
                    if outside:
                        blockers.append(
                            "complete_day_outside_scored_decision_range"
                        )
                    observed_both = (
                        frame.assign(__utc_date__=decision_dates)
                        .groupby("__utc_date__")["side_name"]
                        .agg(lambda values: {"long", "short"}.issubset(
                            set(values.astype(str).str.lower())
                        ))
                    )
                    if any(
                        not bool(observed_both.get(date, False))
                        for date in complete_dates
                    ):
                        blockers.append(
                            "complete_day_not_observed_with_both_sides"
                        )
                complete_days = len(complete_dates)
            except Exception as exc:
                blockers.append("daily_coverage_schema_or_read_failure")
                coverage_result["read_error"] = str(exc)
    if complete_days < minimum_complete_days:
        blockers.append("minimum_complete_utc_days_not_met")
    checked.update(
        {
            "rows": int(len(frame)),
            "admitted_rows": admitted_rows,
            "global_topk_coverage_rows": global_topk_rows,
            "coverage_member_column": coverage_member_column,
            "complete_utc_days": complete_days,
            "decision_min_utc": decisions.min() if len(frame) else None,
            "decision_max_utc": decisions.max() if len(frame) else None,
            "identity_columns": identity,
            "daily_coverage": coverage_result,
        }
    )
    return checked, blockers, frame


def audit_calibrator(
    record: Mapping[str, Any],
    *,
    root: Path,
    first_decision_exclusive: pd.Timestamp,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    result = _file_check(record, root=root, name="causal_calibrator_state", blockers=blockers)
    if str(record.get("mapping")) != "causal_recent_side_isotonic_ev_21d":
        blockers.append("calibrator_mapping_contract_mismatch")
    if int(record.get("lookback_days", -1)) != 21:
        blockers.append("calibrator_lookback_mismatch")
    cutoff_raw = record.get("resolved_label_max_utc")
    if cutoff_raw is None:
        blockers.append("calibrator_missing_resolved_label_cutoff")
    else:
        try:
            cutoff = _utc(cutoff_raw, field="calibrator.resolved_label_max_utc")
        except ValueError:
            blockers.append("calibrator_invalid_resolved_label_cutoff")
        else:
            result["resolved_label_max_utc"] = cutoff
            if cutoff >= first_decision_exclusive:
                blockers.append("calibrator_uses_unresolved_or_forward_labels")
    if not bool(record.get("sequential_updates_only_after_resolution", False)):
        blockers.append("calibrator_sequential_resolution_contract_missing")
    return result, blockers


def audit_preoutcome_seal(
    record: Mapping[str, Any] | None,
    *,
    root: Path,
    scored: pd.DataFrame | None,
    scored_record: Mapping[str, Any],
    source_lock: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(record, Mapping):
        return None, ["missing_preoutcome_seal_contract"]
    blockers: list[str] = []
    result = _file_check(
        record, root=root, name="preoutcome_seal", blockers=blockers
    )
    if not result["exists"]:
        return result, blockers
    payload = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    if payload.get("schema") != "execution_ev_forward_preoutcome_seal_v1":
        blockers.append("preoutcome_seal_schema_mismatch")
    if payload.get("status") != (
        "sealed_preoutcome_population_not_performance_evidence"
    ):
        blockers.append("preoutcome_seal_status_mismatch")
    declared_seal_fingerprint = payload.get("seal_fingerprint")
    seal_core = dict(payload)
    seal_core.pop("seal_fingerprint", None)
    if declared_seal_fingerprint != _stable_payload_hash(seal_core):
        blockers.append("preoutcome_seal_fingerprint_mismatch")
    expected_fingerprint = (
        source_lock.get("lock_fingerprint")
        if isinstance(source_lock, Mapping)
        else None
    )
    if (
        payload.get("source_lock", {}).get("fingerprint")
        != expected_fingerprint
    ):
        blockers.append("preoutcome_seal_source_lock_mismatch")
    output = payload.get("outputs", {})
    scored_path = _resolve(scored_record.get("path"), root=root)
    coverage_path = _resolve(
        scored_record.get("daily_coverage", {}).get("path"), root=root
    )
    if (
        output.get("scored_population", {}).get("sha256")
        != _sha256(scored_path)
    ):
        blockers.append("preoutcome_seal_scored_population_hash_mismatch")
    if (
        output.get("daily_coverage", {}).get("sha256")
        != _sha256(coverage_path)
    ):
        blockers.append("preoutcome_seal_daily_coverage_hash_mismatch")
    if scored is not None:
        identity = tuple(
            scored_record.get("identity_columns", IDENTITY_DEFAULT)
        )
        if payload.get("candidate_identity_sha256") != _identity_hash(
            scored, identity
        ):
            blockers.append("preoutcome_seal_identity_hash_mismatch")
    result["seal_fingerprint"] = payload.get("seal_fingerprint")
    return result, blockers


def audit_confirmation_outcomes(
    record: Mapping[str, Any],
    *,
    root: Path,
    scored: pd.DataFrame | None,
    scored_record: Mapping[str, Any],
    horizon_hours: float,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    result = _file_check(record, root=root, name="confirmation_outcomes", blockers=blockers)
    if not result["exists"] or scored is None:
        return result, blockers
    identity = tuple(scored_record.get("identity_columns", IDENTITY_DEFAULT))
    decision_column = str(scored_record.get("decision_column", "execution_decision_utc"))
    columns = [
        *identity,
        decision_column,
        "execution_label_end_utc",
        "exact_1m_path_complete",
        "execution_gross_ev_12h",
        "execution_cost_return",
        "execution_net_ev_12h",
    ]
    try:
        outcomes = _read_table(Path(result["path"]), columns)
    except Exception as exc:
        blockers.append("confirmation_outcome_schema_or_read_failure")
        result["read_error"] = str(exc)
        return result, blockers
    if outcomes.duplicated(list(identity)).any():
        blockers.append("confirmation_outcome_duplicate_identity")
    left = scored.loc[:, list(identity)].astype(str)
    right = outcomes.loc[:, list(identity)].astype(str)
    left_keys = pd.MultiIndex.from_frame(left)
    right_keys = pd.MultiIndex.from_frame(right)
    if len(left_keys) != len(right_keys) or not left_keys.equals(right_keys):
        if set(left_keys) != set(right_keys):
            blockers.append("confirmation_outcome_identity_mismatch")
        else:
            # Row order is allowed to differ; exact identity coverage is not.
            outcomes = outcomes.set_index(list(identity)).loc[
                pd.MultiIndex.from_frame(scored.loc[:, list(identity)])
            ].reset_index()
    try:
        path_complete = _strict_boolean(
            outcomes["exact_1m_path_complete"],
            field="exact_1m_path_complete",
        )
    except ValueError:
        blockers.append("confirmation_exact_1m_path_complete_not_boolean")
    else:
        if not path_complete.all():
            blockers.append("confirmation_exact_1m_path_incomplete")
    try:
        decision = _utc_series(outcomes[decision_column], field=decision_column)
        label_end = _utc_series(outcomes["execution_label_end_utc"], field="execution_label_end_utc")
    except ValueError:
        blockers.append("confirmation_outcome_timestamps_not_utc")
    else:
        expected_end = decision + pd.to_timedelta(horizon_hours, unit="h")
        if not label_end.eq(expected_end).all():
            blockers.append("confirmation_label_horizon_mismatch")
    gross = pd.to_numeric(outcomes["execution_gross_ev_12h"], errors="coerce").to_numpy(float)
    cost = pd.to_numeric(outcomes["execution_cost_return"], errors="coerce").to_numpy(float)
    net = pd.to_numeric(outcomes["execution_net_ev_12h"], errors="coerce").to_numpy(float)
    if not np.isfinite(np.column_stack([gross, cost, net])).all():
        blockers.append("confirmation_nonfinite_economics")
    elif not np.allclose(gross - cost, net, rtol=0.0, atol=1e-10):
        blockers.append("confirmation_gross_cost_net_reconciliation_failed")
    result["rows"] = int(len(outcomes))
    return result, blockers


def _audit_schema_parity(
    record: Mapping[str, Any] | None,
    *,
    root: Path,
) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(record, Mapping):
        return None, ["missing_schema_parity_contract"]
    blockers: list[str] = []
    reference = _file_check(record.get("reference", {}), root=root, name="schema_reference", blockers=blockers)
    candidate = _file_check(record.get("candidate", {}), root=root, name="schema_candidate", blockers=blockers)
    result = {"reference": reference, "candidate": candidate, "equal": False}
    if not reference["exists"] or not candidate["exists"]:
        return result, blockers
    reference_frame = pd.read_parquet(Path(reference["path"]))
    candidate_frame = pd.read_parquet(Path(candidate["path"]))
    reference_schema = [(str(column), str(dtype)) for column, dtype in reference_frame.dtypes.items()]
    candidate_schema = [(str(column), str(dtype)) for column, dtype in candidate_frame.dtypes.items()]
    result.update(
        {
            "reference_schema": reference_schema,
            "candidate_schema": candidate_schema,
            "equal": reference_schema == candidate_schema,
        }
    )
    if not result["equal"]:
        blockers.append("canonical_schema_order_or_dtype_mismatch")
    return result, blockers


def build_readiness(
    spec: Mapping[str, Any],
    *,
    root: Path,
    stage: str,
    source_lock: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if spec.get("schema") != SPEC_SCHEMA:
        raise ValueError(f"expected spec schema {SPEC_SCHEMA!r}")
    if stage not in {"source_lock", "preoutcome", "confirmation"}:
        raise ValueError("stage must be source_lock, preoutcome, or confirmation")
    first_decision = _utc(spec["first_decision_exclusive_utc"], field="first_decision_exclusive_utc")
    requested_last = _utc(spec["requested_last_decision_utc"], field="requested_last_decision_utc")
    if requested_last <= first_decision:
        raise ValueError("requested_last_decision_utc must be after the frozen cutoff")
    horizon_hours = float(spec.get("label_horizon_hours", 12.0))
    if horizon_hours != 12.0:
        raise ValueError("the frozen confirmation contract requires a 12-hour horizon")
    gate = dict(spec.get("coverage_gate", {}))
    window_mode = str(
        gate.get("window_mode", "fixed_window_all_gates_required")
    )
    if window_mode != "fixed_window_all_gates_required":
        raise ValueError(
            "forward confirmation requires fixed_window_all_gates_required"
        )
    minimum_scored_rows = int(gate.get("minimum_scored_rows", 5_000))
    minimum_rows = int(
        gate.get(
            "minimum_global_topk_rows",
            gate.get("minimum_globally_admitted_rows", 500),
        )
    )
    minimum_days = int(gate.get("minimum_complete_utc_days", 14))
    ranking_contract = dict(spec.get("ranking_contract", {}))
    top_k_fraction = float(ranking_contract.get("top_k_fraction", 0.10))
    blockers: list[str] = []
    evidence: dict[str, Any] = {}

    frozen_manifest_record = dict(spec.get("frozen_confirmation_manifest", {}))
    evidence["frozen_confirmation_manifest"] = _file_check(
        frozen_manifest_record,
        root=root,
        name="frozen_confirmation_manifest",
        blockers=blockers,
    )
    model_results, model_blockers = audit_models(
        spec.get("models", []),
        root=root,
        first_decision_exclusive=first_decision,
        default_training_lineage=spec.get("training_lineage"),
    )
    blockers.extend(model_blockers)
    scored_record = dict(spec.get("scored_population", {}))
    scored_result: dict[str, Any] | None = None
    scored: pd.DataFrame | None = None
    if stage != "source_lock":
        scored_result, scored_blockers, scored = audit_scored_population(
            scored_record,
            root=root,
            first_decision_exclusive=first_decision,
            requested_last_decision=requested_last,
            minimum_scored_rows=minimum_scored_rows,
            minimum_global_topk_rows=minimum_rows,
            minimum_complete_days=minimum_days,
            top_k_fraction=top_k_fraction,
        )
        blockers.extend(scored_blockers)
    calibrator_result, calibrator_blockers = audit_calibrator(
        dict(spec.get("calibrator", {})),
        root=root,
        first_decision_exclusive=first_decision,
    )
    blockers.extend(calibrator_blockers)
    immutable_inputs: dict[str, Any] = {}
    for name in ("policy", "spread_baseline"):
        immutable_inputs[name] = _file_check(
            dict(spec.get(name, {})),
            root=root,
            name=name,
            blockers=blockers,
        )
    code_by_name = {
        str(record.get("name")): record for record in spec.get("source_code", [])
    }
    source_code: dict[str, Any] = {}
    for name in REQUIRED_SOURCE_CODE:
        record = code_by_name.get(name)
        if record is None:
            blockers.append(f"missing_source_code:{name}")
            continue
        source_code[name] = _file_check(
            record,
            root=root,
            name=f"source_code:{name}",
            blockers=blockers,
        )
    ranking = dict(spec.get("ranking_contract", {}))
    expected_ranking = {
        "mapping": "causal_recent_side_isotonic_ev_21d",
        "scope": "one_pooled_global_top_k_across_timestamps_and_sides",
        "top_k_fraction": 0.1,
        "per_timestamp_quota": False,
        "side_quota": False,
        "asset_quota": False,
        "allow_zero_trades": True,
    }
    for key, expected in expected_ranking.items():
        if ranking.get(key) != expected:
            blockers.append(f"ranking_contract_mismatch:{key}")

    lock_fingerprint_payload = {
        "first_decision_exclusive_utc": first_decision,
        "requested_last_decision_utc": requested_last,
        "label_horizon_hours": horizon_hours,
        "coverage_gate": {
            "window_mode": window_mode,
            "minimum_scored_rows": minimum_scored_rows,
            "minimum_global_topk_rows": minimum_rows,
            "minimum_complete_utc_days": minimum_days,
        },
        "scored_population_contract": {
            key: scored_record.get(key)
            for key in (
                "identity_columns",
                "decision_column",
                "admitted_column",
                "coverage_member_column",
                "mapped_score_column",
                "availability_columns",
            )
        },
        "daily_coverage_contract": {
            key: scored_record.get("daily_coverage", {}).get(key)
            for key in (
                "date_column",
                "complete_column",
                "both_sides_column",
            )
        },
        "models": [
            {
                "role": item["role"],
                "model_sha256": item.get("model_file", {}).get("sha256"),
                "training_label_end_max_utc": item.get(
                    "training_label_end_max_utc"
                ),
                "reference_data_max_utc": item.get("reference_data_max_utc"),
                "feature_contract_sha256": (
                    item.get("feature_contract") or {}
                ).get("sha256"),
                "training_lineage_sha256": (
                    item.get("training_lineage") or {}
                ).get("sha256"),
            }
            for item in model_results
        ],
        "calibrator_sha256": calibrator_result.get("sha256"),
        "policy_sha256": immutable_inputs["policy"].get("sha256"),
        "spread_baseline_sha256": immutable_inputs["spread_baseline"].get("sha256"),
        "source_code_sha256": {
            name: record.get("sha256") for name, record in source_code.items()
        },
        "ranking_contract": ranking,
    }
    lock_fingerprint = _stable_payload_hash(lock_fingerprint_payload)
    source_lock_result: dict[str, Any] | None = None
    if stage != "source_lock":
        if source_lock is None:
            blockers.append("missing_frozen_source_lock")
        else:
            source_lock_result = {
                "schema": source_lock.get("schema"),
                "status": source_lock.get("status"),
                "lock_fingerprint": source_lock.get("lock_fingerprint"),
            }
            if source_lock.get("schema") != LOCK_SCHEMA:
                blockers.append("frozen_source_lock_schema_mismatch")
            if source_lock.get("status") != "frozen_before_forward_outcomes":
                blockers.append("frozen_source_lock_status_mismatch")
            if source_lock.get("lock_fingerprint") != lock_fingerprint:
                blockers.append("frozen_source_lock_fingerprint_mismatch")

    preoutcome_seal_result: dict[str, Any] | None = None
    if stage != "source_lock":
        preoutcome_seal_result, seal_blockers = audit_preoutcome_seal(
            scored_record.get("preoutcome_seal"),
            root=root,
            scored=scored,
            scored_record=scored_record,
            source_lock=source_lock,
        )
        blockers.extend(seal_blockers)

    schema_result: dict[str, Any] | None = None
    outcome_result: dict[str, Any] | None = None
    if stage == "confirmation":
        outcome_result, outcome_blockers = audit_confirmation_outcomes(
            dict(spec.get("confirmation_outcomes", {})),
            root=root,
            scored=scored,
            scored_record=scored_record,
            horizon_hours=horizon_hours,
        )
        blockers.extend(outcome_blockers)
        schema_result, schema_blockers = _audit_schema_parity(
            spec.get("canonical_schema_parity"),
            root=root,
        )
        blockers.extend(schema_blockers)
    blockers = sorted(set(blockers))
    return {
        "schema": REPORT_SCHEMA,
        "stage": stage,
        "status": (
            "ready_to_freeze_source_lock"
            if not blockers and stage == "source_lock"
            else (
                "ready_for_outcome_sealed_population"
                if not blockers and stage == "preoutcome"
                else (
                    "ready_for_one_shot_evaluation"
                    if not blockers and stage == "confirmation"
                    else "not_ready"
                )
            )
        ),
        "ready": not blockers,
        "first_decision_exclusive_utc": first_decision,
        "requested_last_decision_utc": requested_last,
        "label_endpoint_required_utc": requested_last + pd.Timedelta(hours=horizon_hours),
        "coverage_gate": {
            "window_mode": window_mode,
            "minimum_scored_rows": minimum_scored_rows,
            "minimum_global_topk_rows": minimum_rows,
            "minimum_complete_utc_days": minimum_days,
            "completion_rule": "fixed_window_all_gates_required",
        },
        "blockers": blockers,
        "models": model_results,
        "scored_population": scored_result,
        "preoutcome_seal": preoutcome_seal_result,
        "calibrator": calibrator_result,
        "immutable_inputs": immutable_inputs,
        "source_code": source_code,
        "ranking_contract": ranking,
        "lock_fingerprint": lock_fingerprint,
        "lock_fingerprint_payload": lock_fingerprint_payload,
        "frozen_source_lock": source_lock_result,
        "frozen_evidence": evidence,
        "confirmation_outcomes": outcome_result,
        "canonical_schema_parity": schema_result,
    }


def freeze_source_lock(
    spec: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    spec_path: Path,
    output: Path,
) -> dict[str, Any]:
    if report.get("stage") != "source_lock" or not report.get("ready"):
        raise ValueError("source lock may be frozen only from a passing source_lock audit")
    payload = {
        "schema": LOCK_SCHEMA,
        "status": "frozen_before_forward_outcomes",
        "spec": {"path": spec_path, "sha256": _sha256(spec_path)},
        "first_decision_exclusive_utc": report["first_decision_exclusive_utc"],
        "requested_last_decision_utc": report["requested_last_decision_utc"],
        "label_endpoint_required_utc": report["label_endpoint_required_utc"],
        "coverage_gate": report["coverage_gate"],
        "ranking_contract": report["ranking_contract"],
        "lock_fingerprint": report["lock_fingerprint"],
        "lock_fingerprint_payload": report["lock_fingerprint_payload"],
        "models": report["models"],
        "scored_population": report["scored_population"],
        "calibrator": report["calibrator"],
        "immutable_inputs": report["immutable_inputs"],
        "source_code": report["source_code"],
        "frozen_evidence": report["frozen_evidence"],
        "decision": (
            "Do not refit, remap, reselect, retune, reduce identities, or alter "
            "the interaction after this lock is written."
        ),
    }
    _write_json(output, payload)
    return payload


def _parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("source_lock", "preoutcome", "confirmation"),
        default="source_lock",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--freeze-source-lock", type=Path)
    parser.add_argument(
        "--source-lock",
        type=Path,
        help="required for preoutcome/confirmation re-verification",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    spec_path = args.spec.resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    root = ROOT
    source_lock = (
        json.loads(args.source_lock.read_text(encoding="utf-8"))
        if args.source_lock is not None
        else None
    )
    report = build_readiness(
        spec,
        root=root,
        stage=args.stage,
        source_lock=source_lock,
    )
    _write_json(args.report, report)
    if args.freeze_source_lock is not None:
        freeze_source_lock(
            spec,
            report,
            spec_path=spec_path,
            output=args.freeze_source_lock,
        )
    return report


if __name__ == "__main__":
    result = run(_parser())
    print(
        json.dumps(
            {
                "status": result["status"],
                "ready": result["ready"],
                "blockers": result["blockers"],
            },
            indent=2,
        )
    )
