"""Checksummed causal Stage-I-to-II direct-FQ3 handoff materializer.

Stage-I OOS predictions carry native base states and joint meta scores, while
their immutable input panel carries the causal regime/context fields.  This
adapter joins only those two same-identity sources; it never fits, maps, or
uses realised path fields as inference inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd


SCHEMA = "stage_i_to_stage_ii_direct_fq3_handoff_v1"
IDENTITY = ("candidate_id", "__ts__", "__symbol__")


class StageIToStageIIHandoffError(ValueError):
    """Raised when an immutable Stage-I source cannot prove a causal handoff."""


def file_sha256(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class StageIToStageIIHandoffSpec:
    stage_i_oos_dir: Path
    stage_i_inputs_dir: Path
    output_dir: Path


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise StageIToStageIIHandoffError(f"expected JSON object: {path}")
    return value


def _required_hash(manifest: Mapping[str, Any], path: Path, *, label: str) -> None:
    expected = str((manifest.get("files") or manifest.get("artifact_sha256") or {}).get(path.name, ""))
    if expected != file_sha256(path):
        raise StageIToStageIIHandoffError(f"{label}: artifact checksum drift")


def _state_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    names = tuple(sorted(
        (str(column) for column in frame.columns if str(column).startswith("base_state_p")),
        key=lambda name: int(name.rsplit("p", 1)[1]),
    ))
    if len(names) not in (2, 3, 5):
        raise StageIToStageIIHandoffError(f"unsupported base state width: {len(names)}")
    values = frame.loc[:, list(names)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
        raise StageIToStageIIHandoffError("base state handoff is not a finite simplex")
    return names


def materialize_stage_i_to_stage_ii_handoff(spec: StageIToStageIIHandoffSpec) -> Mapping[str, Any]:
    if spec.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite Stage-I-to-II handoff: {spec.output_dir}")
    oos_manifest_path = spec.stage_i_oos_dir / "manifest.json"
    prediction_path = spec.stage_i_oos_dir / "strict_oof_predictions.parquet"
    oos_manifest = _read(oos_manifest_path)
    if oos_manifest.get("status") != "complete":
        raise StageIToStageIIHandoffError("Stage-I OOS artifact is incomplete")
    _required_hash(oos_manifest, prediction_path, label="Stage-I OOS")
    prediction = pd.read_parquet(prediction_path)
    required_prediction = {
        "candidate_id", "side_name", "decision_ts", "label_available_ts", "exact_gross_bps",
        "exact_net_bps", "base_direct_score", "base_strict_oof_available",
    }
    if missing := required_prediction.difference(prediction.columns):
        raise StageIToStageIIHandoffError(f"Stage-I OOS prediction lacks {sorted(missing)}")
    # Stage II requires a direct base prediction for every row it receives.
    # Keep the initial prequential burn-in visible in the Stage-I artifact, but
    # never turn an unavailable base output into a Stage-II training example.
    strict_flag = pd.to_numeric(prediction["base_strict_oof_available"], errors="coerce")
    if strict_flag.isna().any() or not strict_flag.isin((0.0, 1.0)).all():
        raise StageIToStageIIHandoffError("base_strict_oof_available must be explicit 0/1")
    input_rows = int(len(prediction))
    prediction = prediction.loc[strict_flag.eq(1.0)].copy()
    if prediction.empty:
        raise StageIToStageIIHandoffError("Stage-I OOS contains no strict base-OOF rows")
    prediction["base_strict_oof_available"] = True
    state_columns = _state_columns(prediction)
    prediction["candidate_id"] = prediction.candidate_id.astype(str)
    prediction["side_name"] = prediction.side_name.astype(str).str.lower()
    prediction["decision_ts"] = pd.to_datetime(prediction.decision_ts, utc=True, errors="coerce")
    prediction["label_available_ts"] = pd.to_datetime(prediction.label_available_ts, utc=True, errors="coerce")
    if prediction.decision_ts.isna().any() or prediction.label_available_ts.isna().any() or prediction.duplicated(["candidate_id", "side_name", "decision_ts"]).any():
        raise StageIToStageIIHandoffError("Stage-I OOS identity/timing contract is invalid")
    parts: list[pd.DataFrame] = []
    input_hashes: dict[str, str] = {}
    source_fields: set[str] = set()
    for side in ("long", "short"):
        root = spec.stage_i_inputs_dir / side
        input_manifest = _read(root / "manifest.json")
        if input_manifest.get("status") != "complete" or str(input_manifest.get("side", "")) != side:
            raise StageIToStageIIHandoffError(f"{side}: Stage-I input panel is incomplete/cross-side")
        feature_path, contract_path = root / "features.parquet", root / "contract.parquet"
        for path, label in ((feature_path, "features"), (contract_path, "contract")):
            expected = str((input_manifest.get("artifact_sha256") or {}).get(path.name, ""))
            if expected != file_sha256(path):
                raise StageIToStageIIHandoffError(f"{side}: {label} checksum drift")
        features, contract = pd.read_parquet(feature_path), pd.read_parquet(contract_path)
        if not features.loc[:, list(IDENTITY)].astype(str).equals(contract.loc[:, list(IDENTITY)].astype(str)):
            raise StageIToStageIIHandoffError(f"{side}: input feature/contract identity drift")
        feature_fields = [name for name in features.columns if name not in IDENTITY]
        source_fields.update(feature_fields)
        contract = contract.loc[:, [*IDENTITY, "side_name", "decision_ts", "label_available_ts"]].copy()
        contract["candidate_id"] = contract.candidate_id.astype(str)
        contract["side_name"] = contract.side_name.astype(str).str.lower()
        contract["decision_ts"] = pd.to_datetime(contract.decision_ts, utc=True, errors="coerce")
        contract["label_available_ts"] = pd.to_datetime(contract.label_available_ts, utc=True, errors="coerce")
        context = contract.merge(
            features.loc[:, [*IDENTITY, *feature_fields]], on=list(IDENTITY), how="inner", validate="one_to_one",
        )
        scored = prediction.loc[prediction.side_name.eq(side)].copy()
        joined = scored.merge(
            context, on=["candidate_id", "side_name", "decision_ts", "label_available_ts"],
            how="inner", validate="one_to_one", suffixes=("", "_input"),
        )
        if len(joined) != len(scored):
            raise StageIToStageIIHandoffError(f"{side}: frozen input panel does not cover every Stage-I OOS prediction")
        joined["symbol"] = joined["__symbol__"].astype(str)
        joined["signal_close_ts"] = pd.to_datetime(joined["__ts__"], utc=True, errors="coerce")
        if joined.signal_close_ts.isna().any() or not joined.decision_ts.eq(joined.signal_close_ts + pd.Timedelta(hours=1)).all():
            raise StageIToStageIIHandoffError(f"{side}: entry timing drift")
        states = joined.loc[:, list(state_columns)].to_numpy(float)
        order = np.sort(states, axis=1)
        joined["base_output_entropy"] = (-np.where(states > 0, states * np.log(states), 0.0).sum(axis=1)).astype(np.float32)
        joined["base_output_top2_margin"] = (order[:, -1] - order[:, -2]).astype(np.float32)
        joined["base_output_max_probability"] = order[:, -1].astype(np.float32)
        parts.append(joined)
        input_hashes[side] = file_sha256(root / "manifest.json")
    output = pd.concat(parts, ignore_index=True)
    if output.duplicated(["candidate_id", "side_name", "decision_ts"]).any():
        raise StageIToStageIIHandoffError("pooled Stage-I-to-II handoff is duplicated")
    output = output.sort_values(["decision_ts", "side_name", "candidate_id"], kind="stable").reset_index(drop=True)
    spec.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{spec.output_dir.name}.", dir=spec.output_dir.parent))
    try:
        ledger_path = temporary / "direct_stage_i_ledger.parquet"
        output.to_parquet(ledger_path, index=False, compression="zstd")
        manifest = {
            "schema": SCHEMA, "status": "complete", "rows": int(len(output)),
            "source_rows": input_rows,
            "excluded_non_strict_base_oof_rows": input_rows - int(len(output)),
            "base_state_columns": list(state_columns), "base_state_width": len(state_columns),
            "causal_source_feature_columns": sorted(source_fields),
            "derived_trust_columns": ["base_output_entropy", "base_output_top2_margin", "base_output_max_probability"],
            "shared_population_contract_sha256": oos_manifest.get("shared_population_contract_sha256"),
            "source_lineage": {
                "stage_i_oos_manifest": file_sha256(oos_manifest_path),
                "stage_i_oos_predictions": file_sha256(prediction_path),
                "stage_i_input_manifests": input_hashes,
            },
            "semantics": "native same-side base output/state plus causal input-panel context; no pre-meta bps input",
            "files": {ledger_path.name: file_sha256(ledger_path)},
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, spec.output_dir)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


__all__ = ["SCHEMA", "StageIToStageIIHandoffError", "StageIToStageIIHandoffSpec", "materialize_stage_i_to_stage_ii_handoff"]
