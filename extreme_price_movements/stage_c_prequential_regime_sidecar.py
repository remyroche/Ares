"""Fail-closed candidate-level F7 adapter for Stage-C.

This is intentionally an adapter, not a model trainer.  It joins only the
sealed hourly *blocked-OOF* regime/transition sidecars to the frozen Stage-C
candidate identity on the exact completed feature-cutoff hour.  Any row whose
fit or label-resolution provenance is not strictly prequential is rejected
before a feature value can be written.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCHEMA = "stage_c_candidate_prequential_regime_sidecar_v1"
OOF_PARTITION = "blocked_oof_2022_2025"
HORIZON = pd.Timedelta(hours=12)
MIN_STAGE1_AVAILABILITY = 0.50

# The authoritative sidecar excludes diagonal/sticky/DAE state identities and
# morphology IDs.  This compact list retains only causal soft context and
# uncertainty, never a post-labelled destination or a trading decision.
F7_FIELDS = {
    "f7_lgbm_transition_probability": "lgbm_transition_probability",
    "f7_lgbm_transition_entropy": "lgbm_entropy",
    "f7_lgbm_transition_margin": "lgbm_margin",
    "f7_bocpd_change_probability_mean": "bocpd__change_probability_mean",
    "f7_bocpd_change_probability_max": "bocpd__change_probability_max",
    "f7_bocpd_run_length_mean": "bocpd__run_length_mean",
    "f7_bocpd_run_length_entropy": "bocpd__run_length_entropy",
    "f7_bocpd_state_age_hours": "bocpd__state_age_hours",
    "f7_bocpd_onset_h1_probability": "bocpd_onset_h1_probability",
    "f7_bocpd_onset_h3_probability": "bocpd_onset_h3_probability",
    "f7_bocpd_onset_h6_probability": "bocpd_onset_h6_probability",
    "f7_bocpd_onset_h12_probability": "bocpd_onset_h12_probability",
    "f7_bocpd_stable_vs_transition_probability": "bocpd_stable_vs_transition_probability",
    "f7_bocpd_stable_vs_transition_entropy": "bocpd_stable_vs_transition_entropy",
    "f7_bocpd_stable_vs_transition_margin": "bocpd_stable_vs_transition_margin",
}
REGIME_SOURCE_FIELDS = {
    "bocpd__change_probability_mean",
    "bocpd__change_probability_max",
    "bocpd__run_length_mean",
    "bocpd__run_length_entropy",
    "bocpd__state_age_hours",
}


class PrequentialSidecarError(RuntimeError):
    """Raised when a row might otherwise admit an in-sample regime output."""


@dataclass(frozen=True)
class Inputs:
    candidates: Path
    soft_regime_hourly: Path
    soft_transition_hourly: Path
    authoritative_manifest: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _utc(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    result = frame.copy()
    for column in ("source_utc",):
        result[column] = pd.to_datetime(result[column], utc=True, errors="coerce")
    if result.source_utc.isna().any() or result.source_utc.duplicated().any():
        raise PrequentialSidecarError(f"{name} must have one valid source_utc row per hour")
    return result


def _load(inputs: Inputs) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    missing = [str(path) for path in vars(inputs).values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required Stage-C F7 input absent: {missing}")
    manifest = json.loads(inputs.authoritative_manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != "authoritative_soft_regime_transition_sidecars_v1" or manifest.get("status") != "SEALED_CAUSAL_SOFT_REGIME_TRANSITION_SIDECARS":
        raise PrequentialSidecarError("authoritative regime sidecar is not the sealed causal v1 contract")
    candidates = pd.read_parquet(inputs.candidates)
    required_candidates = {"candidate_id", "side", "source_symbol", "decision_ts", "feature_cutoff_ts", "retain_h0_given_clear__valid"}
    if missing := sorted(required_candidates.difference(candidates.columns)):
        raise PrequentialSidecarError(f"Stage-C candidate input lacks fields: {missing}")
    if candidates.candidate_id.duplicated().any():
        raise PrequentialSidecarError("Stage-C candidate identity is not unique")
    for column in ("decision_ts", "feature_cutoff_ts"):
        candidates[column] = pd.to_datetime(candidates[column], utc=True, errors="coerce")
    if candidates[["decision_ts", "feature_cutoff_ts"]].isna().any().any() or not candidates.feature_cutoff_ts.le(candidates.decision_ts).all():
        raise PrequentialSidecarError("candidate decision/cutoff time contract failed")
    return candidates, _utc(pd.read_parquet(inputs.soft_regime_hourly), "soft regime"), _utc(pd.read_parquet(inputs.soft_transition_hourly), "soft transition"), manifest


def _prequential_candidate_sidecar(*, inputs: Inputs) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Exact-hourly candidate join, with source provenance checked before use."""
    candidates, regime, transition, manifest = _load(inputs)
    transition_required = {
        "source_utc", "lgbm_transition_available", "provenance_partition_lgbm",
        "train_end_exclusive_utc_lgbm", "fit_label_resolution_max_utc_lgbm",
        *(set(F7_FIELDS.values()).difference(REGIME_SOURCE_FIELDS)),
    }
    regime_required = {
        "source_utc", "bocpd_regime_available", "provenance_partition_bocpd",
        "train_end_exclusive_utc_bocpd", "fit_label_resolution_max_utc_bocpd", *REGIME_SOURCE_FIELDS,
    }
    if missing := sorted(transition_required.difference(transition.columns)):
        raise PrequentialSidecarError(f"transition source lacks required causal fields: {missing}")
    if missing := sorted(regime_required.difference(regime.columns)):
        raise PrequentialSidecarError(f"regime source lacks required causal fields: {missing}")
    transition = transition.loc[:, sorted(transition_required)].copy()
    regime = regime.loc[:, sorted(regime_required)].copy()
    merged_source = transition.merge(regime, on="source_utc", how="inner", validate="one_to_one")
    if len(merged_source) != len(transition) or len(merged_source) != len(regime):
        raise PrequentialSidecarError("regime and transition timelines do not share an exact hourly identity")
    # Many symbols/sides can share one completed hourly market-context row;
    # the authoritative timeline itself remains unique by source hour.
    frame = candidates.merge(merged_source, left_on="feature_cutoff_ts", right_on="source_utc", how="left", validate="many_to_one")
    for column in (
        "train_end_exclusive_utc_lgbm", "fit_label_resolution_max_utc_lgbm",
        "train_end_exclusive_utc_bocpd", "fit_label_resolution_max_utc_bocpd",
    ):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    source_matched = frame.source_utc.notna()
    claimed_available = source_matched & frame.lgbm_transition_available.fillna(False).astype(bool) & frame.bocpd_regime_available.fillna(False).astype(bool)
    valid_partition = frame.provenance_partition_lgbm.eq(OOF_PARTITION) & frame.provenance_partition_bocpd.eq(OOF_PARTITION)
    prequential = (
        frame.train_end_exclusive_utc_lgbm.le(frame.source_utc)
        & frame.fit_label_resolution_max_utc_lgbm.le(frame.source_utc)
        & frame.train_end_exclusive_utc_bocpd.le(frame.source_utc)
        & frame.fit_label_resolution_max_utc_bocpd.le(frame.source_utc)
    )
    time_safe = frame.source_utc.le(frame.feature_cutoff_ts) & frame.source_utc.le(frame.decision_ts)
    fields_present = frame.loc[:, list(F7_FIELDS.values())].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    invalid_claim = claimed_available & ~(valid_partition & prequential & time_safe & fields_present)
    if invalid_claim.any():
        bad = frame.loc[invalid_claim, ["candidate_id", "source_utc", "provenance_partition_lgbm", "provenance_partition_bocpd"]].head(5).to_dict(orient="records")
        raise PrequentialSidecarError(f"an available F7 source row lacks strict prequential provenance: {bad}")
    valid = claimed_available & valid_partition & prequential & time_safe & fields_present
    output_columns = ["candidate_id", "source_symbol", "side", "decision_ts", "feature_cutoff_ts"]
    output = frame.loc[:, output_columns].copy()
    output["f7_source_utc"] = frame.source_utc
    output["f7_available_ts"] = frame.source_utc.where(valid)
    output["f7_prequential_valid"] = valid
    output["f7_provenance_partition"] = frame.provenance_partition_lgbm.where(valid)
    output["f7_train_end_exclusive_utc"] = frame.train_end_exclusive_utc_lgbm.where(valid)
    output["f7_fit_label_resolution_max_utc"] = frame.fit_label_resolution_max_utc_lgbm.where(valid)
    for output_name, source_name in F7_FIELDS.items():
        output[output_name] = pd.to_numeric(frame[source_name], errors="coerce").where(valid)
    if output.candidate_id.duplicated().any() or len(output) != len(candidates):
        raise PrequentialSidecarError("candidate F7 output identity changed")
    if output.loc[~output.f7_prequential_valid, list(F7_FIELDS)].notna().any(axis=None):
        raise PrequentialSidecarError("unavailable F7 candidate received a raw or in-sample value")
    if not output.loc[output.f7_prequential_valid, "f7_available_ts"].le(output.loc[output.f7_prequential_valid, "decision_ts"]).all():
        raise PrequentialSidecarError("F7 availability is later than a candidate decision")

    coverage_source = output.assign(
        month=output.decision_ts.dt.strftime("%Y-%m"),
        clear_first=candidates.retain_h0_given_clear__valid.astype(bool).to_numpy(),
    )
    coverage = coverage_source.groupby(["month", "side", "source_symbol"], as_index=False).agg(
        candidate_rows=("candidate_id", "size"),
        f7_prequential_rows=("f7_prequential_valid", "sum"),
        clear_first_rows=("clear_first", "sum"),
    )
    coverage["f7_prequential_coverage"] = coverage.f7_prequential_rows / coverage.candidate_rows
    folds: list[dict[str, Any]] = []
    for month in ("2024-04", "2024-05", "2024-06", "2024-07", "2024-08"):
        start = pd.Timestamp(f"{month}-01", tz="UTC")
        train = coverage_source.loc[coverage_source.decision_ts.lt(start - HORIZON)].copy()
        clear = train.loc[train.clear_first]
        availability = float(clear.f7_prequential_valid.mean()) if len(clear) else 0.0
        folds.append({"fold_start": start, "clear_first_training_rows": len(clear), "f7_prequential_training_rows": int(clear.f7_prequential_valid.sum()), "f7_prequential_training_coverage": availability, "meets_stage1_minimum_availability": availability >= MIN_STAGE1_AVAILABILITY})
    stage1_admissible = bool(all(row["meets_stage1_minimum_availability"] for row in folds))
    readiness = {
        "schema": SCHEMA,
        "status": "MATERIALIZED_STRICT_PREQUENTIAL_BUT_NOT_ADMISSIBLE_TO_SEALED_V4" if not stage1_admissible else "MATERIALIZED_STRICT_PREQUENTIAL_STAGE1_READY",
        "candidate_rows": len(output),
        "prequential_candidate_rows": int(valid.sum()),
        "candidate_coverage": float(valid.mean()),
        "exact_hour_cutoff_join": True,
        "stage1_v4_admissible": stage1_admissible,
        "reason": "The sealed Stage-C v4 protocol retains 2023-04..2024-03 history; F7 begins in 2024-01, so clear-first training coverage is below the frozen 50% feature-admission threshold in early folds." if not stage1_admissible else "all frozen Stage-C v4 development/final training folds meet the predeclared availability floor",
        "fold_training_coverage": folds,
        "contract": {
            "permitted_partition": OOF_PARTITION,
            "forbidden": ["raw regime fields", "in-sample predictions", "forward rows", "GMM/DAE/sticky identities", "morphology/destination labels"],
            "source_join": "candidate feature_cutoff_ts == authoritative source_utc",
            "prequential_rule": "both train_end_exclusive_utc and fit_label_resolution_max_utc are <= source_utc",
            "feature_available_rule": "f7_available_ts <= candidate decision_ts",
        },
        "source_manifest_schema": manifest.get("schema"),
    }
    return output, coverage, readiness


def run(*, inputs: Inputs, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    sidecar, coverage, readiness = _prequential_candidate_sidecar(inputs=inputs)
    stage = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.staging-"))
    try:
        files: list[str] = []
        for name, frame in (("stage_c_candidate_prequential_f7.parquet", sidecar), ("stage_c_f7_coverage_by_month_side_symbol.parquet", coverage)):
            frame.to_parquet(stage / name, index=False, compression="zstd"); files.append(name)
        _write_json(stage / "f7_readiness.json", readiness); files.append("f7_readiness.json")
        report = "# Stage-C F7 strict prequential sidecar\n\n"
        report += f"Status: **{readiness['status']}**.\n\n"
        report += f"Candidate coverage: {readiness['prequential_candidate_rows']:,}/{readiness['candidate_rows']:,} ({readiness['candidate_coverage']:.2%}).\n\n"
        report += readiness["reason"] + "\n"
        (stage / "STAGE_C_F7_READINESS.md").write_text(report, encoding="utf-8"); files.append("STAGE_C_F7_READINESS.md")
        correctness = {
            "schema": SCHEMA,
            "passed": True,
            "checks": {
                "candidate_identity_preserved": bool(sidecar.candidate_id.is_unique),
                "invalid_rows_have_no_f7_values": not sidecar.loc[~sidecar.f7_prequential_valid, list(F7_FIELDS)].notna().any(axis=None),
                "valid_rows_have_available_by_decision": bool(sidecar.loc[sidecar.f7_prequential_valid, "f7_available_ts"].le(sidecar.loc[sidecar.f7_prequential_valid, "decision_ts"]).all()),
                "only_blocked_oof_partition_admitted": bool(sidecar.loc[sidecar.f7_prequential_valid, "f7_provenance_partition"].eq(OOF_PARTITION).all()),
                "no_model_fit_started": True,
            },
            "stage1_v4_admissible": readiness["stage1_v4_admissible"],
        }
        _write_json(stage / "correctness_test_report.json", correctness); files.append("correctness_test_report.json")
        manifest = {"schema": SCHEMA, "status": readiness["status"], "inputs": {str(path): _sha256(path) for path in vars(inputs).values()}, "readiness": readiness, "outputs": {name: _sha256(stage / name) for name in files}}
        _write_json(stage / "run_manifest.json", manifest)
        os.replace(stage, output)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
