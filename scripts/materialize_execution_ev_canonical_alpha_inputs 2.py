#!/usr/bin/env python3
"""Bridge canonical Pack-B artifacts into the strict execution-EV alpha adapter.

This materializer does not alter any upstream artifact.  It emits:

* a compact candidate handoff carrying observable base-archetype context; and
* supplemental candidate/residual manifests with exact per-side lineage.

The resulting files are inputs to ``materialize_execution_ev_alpha_oof.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

JOIN_KEYS = ("__ts__", "__symbol__", "side_name", "candidate_id")
OOF_LINEAGE_COLUMNS = (
    *JOIN_KEYS,
    "oof_fold",
    "validation_start",
    "train_decision_cutoff",
    "label_resolution_available_at",
)
SIDES = ("long", "short")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path, *, source: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{source}: cannot read valid JSON from {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{source}: expected a JSON object")
    return payload


def _declared_output_sha(payload: Mapping[str, Any], *, source: str) -> str:
    output = payload.get("output")
    if not isinstance(output, Mapping) or not isinstance(output.get("sha256"), str):
        raise ValueError(f"{source}: output.sha256 is required")
    return str(output["sha256"]).lower()


def _require_bound_artifact(
    path: Path, payload: Mapping[str, Any], *, source: str
) -> None:
    if _declared_output_sha(payload, source=source) != _sha256(path):
        raise ValueError(f"{source}: output.sha256 does not bind {path}")


def _normalise_identity(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    missing = sorted(set(JOIN_KEYS).difference(frame.columns))
    if missing:
        raise ValueError(f"{source}: missing identity columns {missing!r}")
    result = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(frame["__ts__"], utc=True, errors="coerce"),
            "__symbol__": frame["__symbol__"].astype("string").str.strip(),
            "side_name": frame["side_name"].astype("string").str.strip().str.lower(),
            "candidate_id": frame["candidate_id"].astype("string").str.strip(),
        }
    )
    if result.isna().any().any():
        raise ValueError(f"{source}: identity contains missing or invalid values")
    if not result["side_name"].isin(SIDES).all():
        raise ValueError(f"{source}: side_name must be canonical long/short")
    if result.duplicated(list(JOIN_KEYS)).any():
        raise ValueError(f"{source}: duplicate candidate identity")
    return result


def _lineage_hash(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    ordered = frame.loc[:, list(columns)].sort_values(list(columns), kind="stable")
    digest = hashlib.sha256()
    for row in ordered.itertuples(index=False, name=None):
        values = [
            value.isoformat() if isinstance(value, pd.Timestamp) else str(value)
            for value in row
        ]
        digest.update(
            json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _side_fold_records(
    candidate_manifest: Mapping[str, Any], side: str
) -> list[Mapping[str, Any]]:
    provenance = candidate_manifest.get("fold_provenance")
    folds = provenance.get("folds") if isinstance(provenance, Mapping) else None
    if not isinstance(folds, Mapping):
        raise ValueError("candidate manifest: fold_provenance.folds is required")
    records = [
        record
        for key, record in sorted(folds.items())
        if str(key).endswith(f"/{side}") and isinstance(record, Mapping)
    ]
    if not records:
        raise ValueError(f"candidate manifest: no fold records for {side}")
    return records


def _candidate_fitted_hashes(
    candidate_manifest: Mapping[str, Any], side: str
) -> dict[str, str]:
    records = _side_fold_records(candidate_manifest, side)
    fields = {
        "model_hash": "model_sha256",
        "feature_contract_hash": "feature_contract_sha256",
        "parameter_hash": "hpo_parameters_sha256",
    }
    result: dict[str, str] = {}
    for output, source in fields.items():
        values = [str(record.get(source, "")).lower() for record in records]
        if any(len(value) != 64 for value in values):
            raise ValueError(f"candidate manifest: invalid {source} for {side}")
        result[output] = _json_hash({"side": side, source: values})
    return result


def _residual_fitted_hashes(
    residual_manifest: Mapping[str, Any], side: str
) -> dict[str, str]:
    sides = residual_manifest.get("sides")
    record = sides.get(side) if isinstance(sides, Mapping) else None
    if not isinstance(record, Mapping):
        raise ValueError(f"residual manifest: missing {side} record")
    folds = record.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError(f"residual manifest: missing {side} folds")
    model_hashes = []
    for fold in folds:
        hashes = fold.get("hashes") if isinstance(fold, Mapping) else None
        value = hashes.get("model_sha256") if isinstance(hashes, Mapping) else None
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"residual manifest: invalid {side} fold model hash")
        model_hashes.append(value.lower())
    feature_hash = record.get("feature_contract_sha256")
    parameter_hash = record.get("hpo_contract_sha256")
    if not isinstance(feature_hash, str) or len(feature_hash) != 64:
        raise ValueError(f"residual manifest: invalid {side} feature contract")
    if not isinstance(parameter_hash, str) or len(parameter_hash) != 64:
        raise ValueError(f"residual manifest: invalid {side} HPO contract")
    return {
        "model_hash": _json_hash({"side": side, "models": model_hashes}),
        "feature_contract_hash": feature_hash.lower(),
        "parameter_hash": parameter_hash.lower(),
    }


def _residual_folds(residual_manifest: Mapping[str, Any]) -> list[dict[str, str]]:
    calendar = residual_manifest.get("calendar")
    rows = calendar.get("oof_folds") if isinstance(calendar, Mapping) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("residual manifest: calendar.oof_folds is required")
    result = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("residual manifest: invalid OOF fold record")
        result.append(
            {
                "fold_id": str(row["fold"]),
                "test_start": str(row["start"]),
                "test_end_exclusive": str(row["end"]),
            }
        )
    return result


def run(args: argparse.Namespace) -> dict[str, Path]:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    candidate_output = args.output_dir / "candidate_handoff.parquet"
    residual_output = args.output_dir / "residual_oof.parquet"
    candidate_manifest_output = args.output_dir / "candidate_handoff.manifest.json"
    residual_manifest_output = args.output_dir / "residual_oof.manifest.json"

    candidate_manifest = _load_json(
        args.candidate_manifest, source="candidate manifest"
    )
    context_manifest = _load_json(args.context_manifest, source="context manifest")
    residual_manifest = _load_json(args.residual_manifest, source="residual manifest")
    _require_bound_artifact(
        args.candidate_population,
        candidate_manifest,
        source="candidate manifest",
    )
    _require_bound_artifact(args.context, context_manifest, source="context manifest")
    declared_residual_sha = residual_manifest.get("oof_predictions_sha256")
    if declared_residual_sha != _sha256(args.residual_oof):
        raise ValueError("residual manifest: OOF hash does not bind residual parquet")

    candidate_raw = pd.read_parquet(args.candidate_population)
    context_raw = pd.read_parquet(args.context)
    candidates = _normalise_identity(candidate_raw, source="candidate population")
    context_identity = _normalise_identity(context_raw, source="context")
    context_columns = (
        "archetype_label_family",
        "archetype_policy_key",
    )
    missing_context = sorted(set(context_columns).difference(context_raw.columns))
    if missing_context:
        raise ValueError(f"context: missing columns {missing_context!r}")
    context = context_identity.copy()
    for column in context_columns:
        values = context_raw[column].astype("string").str.strip()
        if values.isna().any() or values.eq("").any():
            raise ValueError(f"context: {column} contains blank values")
        context[column] = values.to_numpy()
    handoff = candidates.merge(
        context, on=list(JOIN_KEYS), how="left", validate="one_to_one", indicator=True
    )
    if not handoff["_merge"].eq("both").all():
        raise ValueError("context does not cover every candidate identity")
    handoff = handoff.drop(columns="_merge")
    handoff["support_leaf_archetype_family"] = handoff["archetype_label_family"]
    handoff["support_leaf_policy_key"] = handoff["archetype_policy_key"]
    handoff["available_at"] = handoff["__ts__"]
    handoff = handoff.sort_values(list(JOIN_KEYS), kind="stable").reset_index(drop=True)
    handoff.to_parquet(candidate_output, index=False)

    oof_raw = pd.read_parquet(args.residual_oof)
    oof = _normalise_identity(oof_raw, source="residual OOF")
    oof["oof_fold"] = oof_raw["residual_oof_fold"].astype(str).to_numpy()
    for source, output in (
        ("residual_validation_start", "validation_start"),
        ("residual_train_decision_cutoff", "train_decision_cutoff"),
    ):
        oof[output] = pd.to_datetime(oof_raw[source], utc=True, errors="coerce")
    # The residual runner's ``residual_train_decision_cutoff`` is the maximum
    # resolution time among labels admitted to that fold's training ledger.
    oof["label_resolution_available_at"] = oof["train_decision_cutoff"]
    if oof[list(OOF_LINEAGE_COLUMNS)].isna().any().any():
        raise ValueError("residual OOF: invalid fold/cutoff lineage")
    residual_normalized = oof.copy()
    residual_normalized["available_at"] = pd.to_datetime(
        oof_raw["residual_prediction_available_at"], utc=True, errors="coerce"
    )
    for column in (
        "__label_resolution_ts__",
        "base_expected_ev",
        "residual_delta_ev",
        "residual_expected_ev",
    ):
        residual_normalized[column] = oof_raw[column].to_numpy()
    if residual_normalized["available_at"].isna().any():
        raise ValueError("residual OOF: invalid prediction availability")
    residual_normalized = residual_normalized.sort_values(
        list(JOIN_KEYS), kind="stable"
    ).reset_index(drop=True)
    residual_normalized.to_parquet(residual_output, index=False)

    candidate_sides: dict[str, dict[str, str]] = {}
    residual_sides: dict[str, dict[str, Any]] = {}
    for side in SIDES:
        candidate_rows = handoff.loc[handoff["side_name"].eq(side)]
        oof_rows = oof.loc[oof["side_name"].eq(side)]
        fitted = _candidate_fitted_hashes(candidate_manifest, side)
        packb = {
            "side": side,
            "source_hash": _json_hash(
                {
                    "side": side,
                    "candidate_population": _sha256(args.candidate_population),
                    "context": _sha256(args.context),
                }
            ),
            **fitted,
            "candidate_row_identity_hash": _lineage_hash(candidate_rows, JOIN_KEYS),
            "oof_row_identity_hash": _lineage_hash(oof_rows, JOIN_KEYS),
            "oof_fold_cutoff_hash": _lineage_hash(oof_rows, OOF_LINEAGE_COLUMNS),
        }
        candidate_sides[side] = packb
        residual_fitted = _residual_fitted_hashes(residual_manifest, side)
        residual_sides[side] = {
            "side": side,
            "source_hash": _json_hash(
                {
                    "side": side,
                    "residual_oof": _sha256(residual_output),
                    "source_hashes": residual_manifest.get("source_hashes"),
                }
            ),
            **residual_fitted,
            "oof_row_identity_hash": packb["oof_row_identity_hash"],
            "oof_fold_cutoff_hash": packb["oof_fold_cutoff_hash"],
            "upstream_packb": dict(packb),
        }

    candidate_supplement = {
        "schema": "packb_candidate_handoff_lineage_v1",
        "source_artifacts": {
            "candidate_handoff": {"sha256": _sha256(candidate_output)},
            "candidate_population": {
                "path": str(args.candidate_population.resolve()),
                "sha256": _sha256(args.candidate_population),
            },
            "canonical_context": {
                "path": str(args.context.resolve()),
                "sha256": _sha256(args.context),
            },
        },
        "packb_per_side_lineage": {
            "model_side_scope": "per_side",
            "sides": candidate_sides,
        },
    }
    residual_supplement = dict(residual_manifest)
    residual_supplement.update(
        {
            "source_artifacts": {
                "residual_oof": {"sha256": _sha256(residual_output)},
                "upstream_residual_oof": {
                    "path": str(args.residual_oof.resolve()),
                    "sha256": _sha256(args.residual_oof),
                },
            },
            "folds": _residual_folds(residual_manifest),
            "target_mode": "residual_net_ev_after_1pct",
            "residual_expert_target": (
                "ev_after_1pct minus train_only_hierarchical_expected_ev"
            ),
            "residual_per_side_lineage": {
                "model_side_scope": "per_side",
                "sides": residual_sides,
            },
        }
    )
    candidate_manifest_output.write_text(
        json.dumps(candidate_supplement, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    residual_manifest_output.write_text(
        json.dumps(residual_supplement, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "candidate_handoff": candidate_output,
        "candidate_manifest": candidate_manifest_output,
        "residual_oof": residual_output,
        "residual_manifest": residual_manifest_output,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-population", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--context-manifest", type=Path, required=True)
    parser.add_argument("--residual-oof", type=Path, required=True)
    parser.add_argument("--residual-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    outputs = run(_parser().parse_args())
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
