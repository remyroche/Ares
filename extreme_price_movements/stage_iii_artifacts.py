"""Compact, immutable publication contract for Stage-III development evidence.

The sequential funnel can be large.  Publication therefore stores one winner
OOF ledger plus compact metrics/audits for every arm, rather than duplicating
every arm's row-level predictions.  The caller must explicitly opt into writing
and supply the full reproducibility manifest; this module never launches a run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import pandas as pd


SCHEMA = "stage_iii_compact_artifact_bundle_v1"


class StageIIIArtifactError(ValueError):
    """Raised when evidence cannot be published immutably and reproducibly."""


@dataclass(frozen=True)
class StageIIIReproducibilityManifest:
    run_id: str
    dataset_id: str
    dataset_sha256: str
    label_manifest_id: str
    label_manifest_sha256: str
    feature_contract_sha256: str
    input_lineage_contract_sha256: str
    code_revision: str
    split_definition: Mapping[str, Any]
    model_configuration: Mapping[str, Any]
    random_seeds: Sequence[int]
    evaluation_status: str = "DEVELOPMENT_STRICT_OOF_NOT_FINAL_TEST"
    schema: str = SCHEMA

    def validate(self) -> None:
        for name in ("run_id", "dataset_id", "label_manifest_id", "code_revision"):
            if not str(getattr(self, name)).strip():
                raise StageIIIArtifactError(f"{name} must be non-empty")
        for name in (
            "dataset_sha256", "label_manifest_sha256", "feature_contract_sha256",
            "input_lineage_contract_sha256",
        ):
            value = str(getattr(self, name))
            if not re.fullmatch(r"[0-9a-f]{64}", value) or len(set(value)) == 1:
                raise StageIIIArtifactError(f"{name} must be a non-placeholder SHA-256")
        if self.schema != SCHEMA:
            raise StageIIIArtifactError("unsupported Stage-III artifact schema")
        if "FINAL" in self.evaluation_status and self.evaluation_status != "DEVELOPMENT_STRICT_OOF_NOT_FINAL_TEST":
            raise StageIIIArtifactError("development funnel publication cannot claim a final test")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and (pd.isna(value) or value in (float("inf"), float("-inf"))):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(value), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _require_frame(value: Any, name: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise StageIIIArtifactError(f"{name} must be a DataFrame")
    return value


def _normalize_feature_lists(
    value: Mapping[str, Sequence[str]] | None,
    *,
    winner: Any,
) -> dict[str, list[str]]:
    """Return the exact ordered model lists that made up this publication.

    A Stage-III winner always has a residual/meta list.  A base list belongs to
    its upstream immutable feature contract and is therefore caller-supplied
    when the compact bundle spans both layers.  We do not infer either list
    from a dataframe's incidental columns.
    """
    source: dict[str, Sequence[str]] = {}
    if value is not None:
        if not isinstance(value, Mapping):
            raise StageIIIArtifactError("feature_lists must map layer names to ordered feature lists")
        source.update({str(layer): names for layer, names in value.items()})
    winner_features = getattr(winner, "model_feature_names", ())
    if "meta_residual" not in source and "meta" not in source and winner_features:
        source["meta_residual"] = winner_features
    normalized: dict[str, list[str]] = {}
    for layer, names in source.items():
        if not layer.strip():
            raise StageIIIArtifactError("feature-list layer name must be non-empty")
        if isinstance(names, (str, bytes)):
            raise StageIIIArtifactError(f"feature list {layer!r} must be a sequence of names")
        try:
            ordered = [str(name) for name in names]
        except TypeError as exc:
            raise StageIIIArtifactError(
                f"feature list {layer!r} must be a sequence of names"
            ) from exc
        if not ordered or any(not name.strip() for name in ordered):
            raise StageIIIArtifactError(f"feature list {layer!r} must contain non-empty names")
        if len(set(ordered)) != len(ordered):
            raise StageIIIArtifactError(f"feature list {layer!r} contains duplicates")
        normalized[layer] = ordered
    if not normalized:
        raise StageIIIArtifactError(
            "feature lists are required; supply base/meta lists or a winner with model_feature_names"
        )
    return dict(sorted(normalized.items()))


def _arm_feature_contracts(arms: Sequence[Any]) -> pd.DataFrame:
    """Persist one exact ordered feature contract per tested arm, not just winner."""
    records: list[dict[str, Any]] = []
    for arm in arms:
        names = getattr(arm, "model_feature_names", ())
        if isinstance(names, (str, bytes)):
            raise StageIIIArtifactError("arm model_feature_names must be an ordered sequence")
        try:
            ordered = [str(name) for name in names]
        except TypeError as exc:
            raise StageIIIArtifactError("arm model_feature_names must be an ordered sequence") from exc
        if not ordered or any(not name.strip() for name in ordered) or len(set(ordered)) != len(ordered):
            raise StageIIIArtifactError("every compact arm must expose a unique non-empty feature list")
        calculated_hash = sha256(
            json.dumps(ordered, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        recorded_hash = str(getattr(arm, "model_feature_contract_sha256", ""))
        if recorded_hash and recorded_hash != calculated_hash:
            raise StageIIIArtifactError("arm feature-contract hash does not match its exact ordered list")
        records.append({
            "round": str(getattr(arm, "round_name", "")),
            "arm": str(getattr(arm, "arm", "")),
            "feature_count": len(ordered),
            "feature_names_json": json.dumps(ordered, separators=(",", ":")),
            "feature_contract_sha256": calculated_hash,
            "source_feature_contract_sha256": str(
                getattr(arm, "source_feature_contract_sha256", "")
            ),
        })
    return pd.DataFrame(records)


def publish_stage_iii_compact_bundle(
    result: Any,
    output_directory: str | Path,
    *,
    reproducibility: StageIIIReproducibilityManifest,
    winner_prediction_columns: Sequence[str],
    report_tables: Any | None = None,
    feature_lists: Mapping[str, Sequence[str]] | None = None,
) -> Path:
    """Atomically publish compact Stage-III evidence without legacy buildup."""
    reproducibility.validate()
    output = Path(output_directory).resolve()
    if output.exists():
        raise StageIIIArtifactError(f"immutable output already exists: {output}")
    if not output.parent.exists():
        raise StageIIIArtifactError(f"output parent does not exist: {output.parent}")
    winner = getattr(result, "winner", None)
    arms = tuple(getattr(result, "arms", ()))
    if winner is None or not arms:
        raise StageIIIArtifactError("result must contain a winner and all sequential arms")
    normalized_feature_lists = _normalize_feature_lists(feature_lists, winner=winner)
    predictions = _require_frame(getattr(winner, "oof_predictions", None), "winner OOF predictions")
    columns = tuple(dict.fromkeys(str(name) for name in winner_prediction_columns))
    missing = [name for name in columns if name not in predictions]
    if not columns or missing:
        raise StageIIIArtifactError(f"winner prediction publication lacks columns: {missing}")
    required_identity = {"candidate_id", "symbol", "decision_ts", "side_name"}
    if not required_identity.issubset(columns):
        raise StageIIIArtifactError("winner publication must preserve candidate/symbol/time/side identity")

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        _require_frame(getattr(result, "arm_summary", None), "arm summary").to_parquet(
            temporary / "arm_summary.parquet", index=False, compression="zstd"
        )
        _require_frame(getattr(result, "transport_matrix", None), "transport matrix").to_parquet(
            temporary / "transport_matrix.parquet", index=False, compression="zstd"
        )
        predictions.loc[:, columns].to_parquet(
            temporary / "winner_oof_predictions.parquet", index=False, compression="zstd"
        )
        _require_frame(getattr(winner, "metrics", None), "winner metrics").to_parquet(
            temporary / "winner_metrics.parquet", index=False, compression="zstd"
        )
        _require_frame(getattr(winner, "fold_audit", None), "winner fold audit").to_parquet(
            temporary / "winner_fold_audit.parquet", index=False, compression="zstd"
        )
        _require_frame(getattr(winner, "calibration_audit", None), "winner calibration audit").to_parquet(
            temporary / "winner_calibration_audit.parquet", index=False, compression="zstd"
        )
        arm_metrics = []
        for arm in arms:
            metrics = _require_frame(getattr(arm, "metrics", None), "arm metrics").copy()
            metrics.insert(0, "arm", str(getattr(arm, "arm", "")))
            metrics.insert(0, "round", str(getattr(arm, "round_name", "")))
            arm_metrics.append(metrics)
        pd.concat(arm_metrics, ignore_index=True).to_parquet(
            temporary / "all_arm_metrics.parquet", index=False, compression="zstd"
        )
        _arm_feature_contracts(arms).to_parquet(
            temporary / "all_arm_feature_contracts.parquet", index=False, compression="zstd"
        )
        report_names: list[str] = []
        if report_tables is not None:
            if str(getattr(report_tables, "schema", "")) != "stage_iii_pooled_global_reporting_v1":
                raise StageIIIArtifactError("report_tables has an unsupported schema")
            for name in (
                "tail_summary", "selected_attribution", "residual_diagnostics",
                "time_concentration", "hit_surprise",
            ):
                table = _require_frame(getattr(report_tables, name, None), f"report {name}")
                table.to_parquet(temporary / f"{name}.parquet", index=False, compression="zstd")
                report_names.append(name)
        _write_json(temporary / "advancement_gates.json", getattr(result, "advancement_gates", {}))
        _write_json(temporary / "feature_lists.json", normalized_feature_lists)
        manifest = {
            **asdict(reproducibility),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "runner_schema": str(getattr(result, "schema", "")),
            "round_winners": dict(getattr(result, "round_winners", {})),
            "winner_arm": str(getattr(winner, "arm", "")),
            "search_breadth_arms": len(arms),
            "winner_oof_rows": len(predictions),
            "winner_prediction_columns": list(columns),
            "feature_list_layers": list(normalized_feature_lists),
            "feature_list_sha256": sha256(
                json.dumps(normalized_feature_lists, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "storage_policy": "winner_rows_plus_all_arm_compact_metrics",
            "all_arm_feature_contracts": "all_arm_feature_contracts.parquet",
            "report_tables": report_names,
        }
        _write_json(temporary / "run_manifest.json", manifest)
        files = sorted(path for path in temporary.iterdir() if path.is_file())
        _write_json(
            temporary / "checksums.json",
            {path.name: _digest(path) for path in files},
        )
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


__all__ = [
    "SCHEMA", "StageIIIArtifactError", "StageIIIReproducibilityManifest",
    "publish_stage_iii_compact_bundle",
]
