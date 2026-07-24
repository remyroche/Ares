#!/usr/bin/env python3
"""Materialize one canonical timestamp x side OOF base candidate population."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.base_candidate_population import (  # noqa: E402
    BaseCandidatePopulationContract,
    candidate_identity_sha256,
    deterministic_candidate_ids,
    select_base_candidate_population,
)

STRICT_PROVENANCE_COLUMNS = (
    "oos_fold",
    "validation_start",
    "train_decision_cutoff",
    "label_resolution_available_at",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: object) -> object:
    if value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _utc(value: object, *, source: str, field: str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"{source}: invalid required UTC {field!r}")
    return pd.Timestamp(parsed)


def _fold_id(value: object) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError("base candidate source has blank oos_fold")
    return result


def _manifest_timestamp(payload: dict[str, object], *names: str) -> object:
    for name in names:
        value = payload.get(name)
        if value is not None:
            return _utc(value, source="base fold manifest", field=name)
    return pd.NaT


def _load_fold_provenance(manifest_root: Path) -> dict[str, dict[str, object]]:
    manifests = sorted(manifest_root.glob("*/manifest.json"))
    if not manifests:
        raise ValueError(
            f"no base fold manifests found under {manifest_root}; expected <fold>/manifest.json"
        )
    result: dict[str, dict[str, object]] = {}
    for path in manifests:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read base fold manifest {path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"base fold manifest {path} must be a JSON object")
        fold = _fold_id(payload.get("fold", path.parent.name))
        if fold in result:
            raise ValueError(f"duplicate base fold manifest for oos_fold={fold!r}")
        validation_start = _utc(payload.get("valid_start"), source=str(path), field="valid_start")
        validation_end = _manifest_timestamp(payload, "valid_end", "valid_end_exclusive")
        if not pd.isna(validation_end) and validation_end < validation_start:
            raise ValueError(f"{path}: valid_end precedes valid_start")
        leakage_contract = payload.get("leakage_contract")
        fit_scope = (
            leakage_contract.get("fit_scope")
            if isinstance(leakage_contract, dict)
            else leakage_contract
        )
        train_cutoff = _manifest_timestamp(
            payload, "train_decision_cutoff", "train_decision_cutoff_utc"
        )
        label_resolution = _manifest_timestamp(
            payload,
            "train_label_resolution_available_at",
            "label_resolution_available_at",
        )
        observed = not pd.isna(train_cutoff) and not pd.isna(label_resolution)
        if observed and not (label_resolution <= train_cutoff < validation_start):
            raise ValueError(
                f"{path}: observed train-label resolution/cutoff evidence violates temporal order"
            )
        result[fold] = {
            "validation_start": validation_start,
            "validation_end": validation_end,
            "fit_scope": fit_scope,
            "train_decision_cutoff": train_cutoff,
            "label_resolution_available_at": label_resolution,
            "validation_boundary_evidence": "observed_model_manifest_valid_start",
            "train_decision_cutoff_evidence": (
                "observed_model_manifest" if observed else "not_observed_in_model_manifest"
            ),
            "label_resolution_evidence": (
                "observed_model_manifest" if observed else "not_observed_in_model_manifest"
            ),
            "manifest_path": str(path.resolve()),
            "manifest_sha256": _sha256(path),
        }
    return result


def _attach_fold_provenance(
    selected: pd.DataFrame,
    *,
    manifests: dict[str, dict[str, object]],
) -> pd.DataFrame:
    if "oos_fold" not in selected:
        raise ValueError(
            "base candidate source lacks oos_fold; regenerate the base OOF population with fold identity"
        )
    result = selected.copy()
    folds = result["oos_fold"].map(_fold_id)
    missing = sorted(set(folds).difference(manifests))
    if missing:
        raise ValueError(f"base fold manifests are missing oos_fold values: {missing[:5]}")
    result["oos_fold"] = folds.astype("string")
    for column in (
        "validation_start",
        "validation_end",
        "train_decision_cutoff",
        "label_resolution_available_at",
        "validation_boundary_evidence",
        "train_decision_cutoff_evidence",
        "label_resolution_evidence",
        "base_fold_fit_scope",
        "base_fold_manifest_path",
        "base_fold_manifest_sha256",
    ):
        result[column] = [
            manifests[fold][
                {
                    "base_fold_fit_scope": "fit_scope",
                    "base_fold_manifest_path": "manifest_path",
                    "base_fold_manifest_sha256": "manifest_sha256",
                }.get(column, column)
            ]
            for fold in folds
        ]
    return result


def run(
    source: Path,
    output_dir: Path,
    contract: BaseCandidatePopulationContract,
    *,
    model_manifest_root: Path,
    include_all_columns: bool = False,
) -> dict[str, object]:
    source_columns = pq.read_schema(source).names
    required = [
        contract.timestamp_col,
        contract.symbol_col,
        contract.side_col,
        contract.score_col,
        "oos_fold",
    ]
    optional = [
        "base_rank_within_timestamp_side",
        "base_rank_pct_timestamp_side",
        "base_cutoff_score_timestamp_side",
    ]
    read_columns = source_columns if include_all_columns else [
        column for column in [*required, *optional] if column in source_columns
    ]
    frame = pd.read_parquet(source, columns=read_columns)
    selected = select_base_candidate_population(frame, contract)
    selected["timeframe"] = str(contract.timeframe).strip()
    selected["candidate_id"] = deterministic_candidate_ids(
        selected,
        timestamp_col=contract.timestamp_col,
        symbol_col=contract.symbol_col,
        side_col=contract.side_col,
        timeframe=contract.timeframe,
    )
    fold_provenance = _load_fold_provenance(model_manifest_root)
    selected = _attach_fold_provenance(selected, manifests=fold_provenance)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "base_candidate_population.parquet"
    selected.to_parquet(output, index=False, compression="zstd", compression_level=5)
    selected_col = f"selected_top{int(round(100.0 * contract.top_fraction))}"
    manifest = {
        "schema": "base_candidate_population_v2",
        "source": {"path": str(source), "sha256": _sha256(source)},
        "output": {"path": str(output), "sha256": _sha256(output)},
        "source_rows": int(len(frame)),
        "selected_rows": int(len(selected)),
        "contract": asdict(contract),
        "selected_column": selected_col,
        "included_all_source_columns": bool(include_all_columns),
        "selection_authority": "persisted integer rank within UTC timestamp x side",
        "candidate_identity_sha256": candidate_identity_sha256(
            selected,
            columns=(contract.timestamp_col, contract.symbol_col, contract.side_col, "candidate_id"),
        ),
        "candidate_id_contract": {
            "column": "candidate_id",
            "rule": "symbol | UTC ISO-Z decision timestamp | explicit timeframe | canonical side",
            "identity": [
                contract.timestamp_col,
                contract.symbol_col,
                "timeframe",
                contract.side_col,
            ],
            "timeframe": contract.timeframe,
            "shared_helper": "extreme_price_movements.side_aware.candidate_id_series",
        },
        "fold_provenance": {
            "manifest_root": str(model_manifest_root.resolve()),
            "folds": fold_provenance,
            "strict_execution_ev_handoff": {
                "status": (
                    "ready"
                    if selected["train_decision_cutoff_evidence"].eq("observed_model_manifest").all()
                    else "blocked_regenerate_upstream_evidence"
                ),
                "required_columns": list(STRICT_PROVENANCE_COLUMNS),
                "reason": (
                    "Fold manifests record valid_start/fit scope but not the observed maximum "
                    "train decision whose label had resolved. validation_start is retained as a "
                    "validation boundary only and is never substituted for train_decision_cutoff."
                ),
                "regeneration_action": (
                    "Regenerate each upstream OOF prediction artifact with candidate_id, "
                    "oos_fold, validation_start, train_decision_cutoff, and "
                    "label_resolution_available_at recorded from the actual resolved training rows."
                ),
            },
        },
        "downstream_required_consumers": [
            "alpha_residual_model",
            "catboost_path_archetype",
            "path_auxiliary_heads",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-fraction", type=float, default=0.40)
    parser.add_argument("--timestamp-col", default="__ts__")
    parser.add_argument("--symbol-col", default="__symbol__")
    parser.add_argument("--side-col", default="side_name")
    parser.add_argument("--score-col", default="score")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--include-all-columns", action="store_true")
    parser.add_argument(
        "--model-manifest-root",
        type=Path,
        required=True,
        help="Directory containing <oos_fold>/manifest.json base OOF manifests.",
    )
    args = parser.parse_args()
    contract = BaseCandidatePopulationContract(
        top_fraction=args.top_fraction,
        timestamp_col=args.timestamp_col,
        symbol_col=args.symbol_col,
        side_col=args.side_col,
        score_col=args.score_col,
        timeframe=args.timeframe,
    )
    print(
        json.dumps(
            run(
                args.source,
                args.output_dir,
                contract,
                model_manifest_root=args.model_manifest_root,
                include_all_columns=args.include_all_columns,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
