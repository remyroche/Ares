#!/usr/bin/env python3
"""Run the bounded Stage-II meta-archetype development funnel.

This command consumes a *completed* Stage-I production artifact and a separate
identity-aligned enriched ledger.  The latter must contain already materialised
causal meta fields and realised path descriptors.  It intentionally refuses to
derive path descriptors from Stage-I scores or to use the original Stage-I
residual as a Stage-II control.

The candidate JSON is bounded and explicit, for example::

  {
    "meta_feature_cols": ["..."],
    "meta_params": {"objective": "huber", "n_estimators": 300, "learning_rate": 0.03},
    "min_train_rows": 500,
    "n_validation_folds": 4,
    "candidates": [{"candidate_id": "path_k3", "causal_feature_cols": ["..."],
      "config": {"path_descriptor_cols": ["..."], "components": 3}}]
  }
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extreme_price_movements.stage_ii_execution import (
    StageIIExecutionError,
    build_stage_ii_ledger,
    file_sha256,
    make_side_local_strict_meta_predictor,
    validate_enriched_ledger_manifest,
    write_development_checkpoint,
)
from extreme_price_movements.stage_ii_meta_archetype_funnel import (
    StageIIDiscoveryCandidate,
    StageIIFunnelSpec,
    run_stage_ii_meta_archetype_funnel,
)
from extreme_price_movements.stage_ii_meta_archetypes import (
    StageIIMetaArchetypeConfig,
    membership_feature_names,
    stage_ii_feature_names,
)
from extreme_price_movements.stage_ii_production_oos import (
    StageIIWindowContract,
    StageIIWinnerManifest,
    _identity_digest,
    publish_stage_ii_winner_bundle,
)


def _load_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise StageIIExecutionError(f"JSON object expected: {path}")
    return raw


def _candidate_spec(raw: Mapping[str, Any]) -> tuple[tuple[StageIIDiscoveryCandidate, ...], tuple[str, ...], dict[str, Any], int, int]:
    meta = tuple(dict.fromkeys(map(str, raw.get("meta_feature_cols", ()))))
    params = raw.get("meta_params")
    entries = raw.get("candidates")
    if not meta or not isinstance(params, Mapping) or not isinstance(entries, list):
        raise StageIIExecutionError("candidate JSON requires non-empty meta_feature_cols, meta_params, and candidates")
    if not 1 <= len(entries) <= 8:
        raise StageIIExecutionError("candidate JSON must contain one to eight predeclared candidates")
    candidates: list[StageIIDiscoveryCandidate] = []
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("config"), Mapping):
            raise StageIIExecutionError("every Stage-II candidate needs candidate_id, causal_feature_cols and config")
        config = StageIIMetaArchetypeConfig(**dict(entry["config"]))
        candidates.append(StageIIDiscoveryCandidate(
            candidate_id=str(entry.get("candidate_id", "")), config=config,
            causal_feature_cols=tuple(map(str, entry.get("causal_feature_cols", ()))),
        ))
    return tuple(candidates), meta, dict(params), int(raw.get("n_validation_folds", 4)), int(raw.get("min_train_rows", 500))


def _funnel_spec(raw: Mapping[str, Any], *, meta_features: tuple[str, ...], fold_catalogue: tuple[dict[str, Any], ...]) -> StageIIFunnelSpec:
    values = dict(raw.get("funnel_spec", {}))
    forbidden = {
        "meta_feature_cols", "base_r3_oof_fold_catalog", "base_expected_net_column",
        "base_r3_probability_columns", "base_r3_oof_flag_column", "base_r3_source_side_column",
        "base_r3_fit_end_column", "base_r3_semantics_column", "base_r3_fold_id_column",
        "base_map_prequential_flag_column", "base_map_source_side_column", "base_map_max_label_available_column",
    }
    overlap = sorted(forbidden.intersection(values))
    if overlap:
        raise StageIIExecutionError(f"candidate JSON may not override frozen Stage-I handoff fields: {overlap}")
    return StageIIFunnelSpec(meta_feature_cols=meta_features, base_r3_oof_fold_catalog=fold_catalogue, **values)


def _release_winner(
    *, root: Path, release: Mapping[str, Any], result: Any, ledger: pd.DataFrame,
    candidates: tuple[StageIIDiscoveryCandidate, ...], meta_features: tuple[str, ...],
    params: Mapping[str, Any], stage_i_oos_dir: Path,
) -> Path | None:
    # A control which keeps no archetype feature is a negative Stage-II result,
    # not a license to call a Stage-I rerun a Stage-II winner.
    if result.selected_candidate_id is None or result.selected_control_arm not in {"soft_memberships", "prior", "both"}:
        return None
    if result.oof_features is None or len(result.oof_features) != len(ledger):
        raise StageIIExecutionError("selected Stage-II archetype OOF features do not align with development ledger")
    selected = next(item for item in candidates if item.candidate_id == result.selected_candidate_id)
    available_col = "meta_conversion_arch_available"
    available = pd.to_numeric(result.oof_features[available_col], errors="coerce").eq(1.0).to_numpy()
    identity = ledger.loc[available, ["candidate_id", "symbol", "signal_close_ts", "decision_ts", "label_available_ts", "side_name"]].copy()
    if identity.empty:
        raise StageIIExecutionError("selected Stage-II winner has no immutable development identities")
    required_release = {
        "run_id", "dataset_id", "dataset_sha256", "label_manifest_id", "label_manifest_sha256",
        "universe_id", "universe_sha256", "code_revision", "stage_i_base_winner_artifact_id", "window",
    }
    missing = sorted(required_release.difference(release))
    if missing:
        raise StageIIExecutionError(f"release metadata lacks required immutable fields: {missing}")
    winner_file = stage_i_oos_dir / "winner_bundle.json"
    if not winner_file.is_file():
        raise StageIIExecutionError("completed Stage-I artifact lacks winner_bundle.json")
    arm = str(result.selected_control_arm)
    # The release contract publishes the complete frozen transform vector,
    # including soft memberships when the winner consumes only the prior.  The
    # selected arm in ``selected_config`` is still the exact meta input subset;
    # publishing the vector lets locked OOS validate its soft-simplex output.
    archetype = tuple(stage_ii_feature_names(selected.config.components))
    ordered_meta = tuple(dict.fromkeys(("prequential_base_expected_net_bps", "r3_p_adverse", "r3_p_weak", "r3_p_clear", *meta_features)))
    manifest = StageIIWinnerManifest(
        run_id=str(release["run_id"]), dataset_id=str(release["dataset_id"]), dataset_sha256=str(release["dataset_sha256"]),
        label_manifest_id=str(release["label_manifest_id"]), label_manifest_sha256=str(release["label_manifest_sha256"]),
        universe_id=str(release["universe_id"]), universe_sha256=str(release["universe_sha256"]), code_revision=str(release["code_revision"]),
        stage_i_base_winner_artifact_id=str(release["stage_i_base_winner_artifact_id"]),
        stage_i_base_winner_artifact_sha256=file_sha256(winner_file),
        stage_i_base_oof_ledger_sha256=file_sha256(stage_i_oos_dir / "full_history_raw_oof_predictions.parquet"),
        selected_discovery_candidate_id=selected.candidate_id, selected_control_arm=arm,
        selected_config={"candidate": {"candidate_id": selected.candidate_id, "config": asdict(selected.config), "causal_feature_cols": selected.causal_feature_cols}, "meta_params": dict(params), "selection_manifest": dict(result.manifest)},
        ordered_meta_features=ordered_meta, ordered_archetype_features=tuple(archetype),
        development_identity_sha256=_identity_digest(identity, columns=("candidate_id", "symbol", "signal_close_ts", "decision_ts", "label_available_ts", "side_name")),
        window=StageIIWindowContract(**dict(release["window"])),
    )
    return publish_stage_ii_winner_bundle(
        root / "winner_bundle", manifest=manifest, development_identity=identity,
        development_metrics=result.control_metrics, candidate_audit=result.candidate_audit,
        control_metrics=result.control_metrics,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-i-oos-dir", type=Path, required=True)
    parser.add_argument("--enriched-ledger", type=Path, required=True)
    parser.add_argument("--enriched-manifest", type=Path, required=True)
    parser.add_argument("--candidate-spec-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--release-metadata-json", type=Path)
    args = parser.parse_args()
    stage_i_manifest = _load_json(args.stage_i_oos_dir / "manifest.json")
    if stage_i_manifest.get("status") != "complete" or stage_i_manifest.get("schema") != "stage_i_production_winner_oos_v1":
        raise StageIIExecutionError("--stage-i-oos-dir must be a completed Stage-I production OOS artifact")
    required_stage_i = ("full_history_raw_oof_predictions.parquet", "fold_provenance.parquet", "winner_bundle.json")
    if any(not (args.stage_i_oos_dir / name).is_file() for name in required_stage_i):
        raise StageIIExecutionError("completed Stage-I artifact lacks a required full-history OOF/provenance/winner file")
    raw = _load_json(args.candidate_spec_json)
    candidates, meta_features, params, n_folds, min_train = _candidate_spec(raw)
    required = tuple(dict.fromkeys([*meta_features, *(name for candidate in candidates for name in (*candidate.causal_feature_cols, *candidate.config.path_descriptor_cols))]))
    required_causal = tuple(dict.fromkeys([
        *meta_features,
        *(name for candidate in candidates for name in candidate.causal_feature_cols),
    ]))
    required_path = tuple(dict.fromkeys([
        name for candidate in candidates for name in candidate.config.path_descriptor_cols
    ]))
    validate_enriched_ledger_manifest(
        _load_json(args.enriched_manifest), ledger_path=args.enriched_ledger,
        required_causal_columns=required_causal, required_path_columns=required_path,
    )
    ledger, catalogue = build_stage_ii_ledger(
        stage_i_predictions=pd.read_parquet(args.stage_i_oos_dir / "full_history_raw_oof_predictions.parquet"),
        stage_i_fold_provenance=pd.read_parquet(args.stage_i_oos_dir / "fold_provenance.parquet"),
        enriched_ledger=pd.read_parquet(args.enriched_ledger), required_enriched_columns=required,
    )
    if ledger.duplicated("candidate_id").any():
        raise StageIIExecutionError("Stage-II execution requires candidate_id to be globally unique across sides")
    spec = _funnel_spec(raw, meta_features=meta_features, fold_catalogue=catalogue)
    predictor = make_side_local_strict_meta_predictor(
        side_by_candidate_id=dict(zip(ledger.candidate_id.astype(str), ledger.side_name.astype(str), strict=True)),
        params=params, n_validation_folds=n_folds, min_train_rows=min_train,
    )
    result = run_stage_ii_meta_archetype_funnel(ledger, spec=spec, candidates=candidates, meta_oof_predictor=predictor)
    root = write_development_checkpoint(args.output_dir, stage_i_oos_dir=args.stage_i_oos_dir, enriched_ledger=args.enriched_ledger, result=result, base_fold_catalogue=catalogue, candidate_spec=raw)
    winner = _release_winner(root=root, release=_load_json(args.release_metadata_json) if args.release_metadata_json else {}, result=result, ledger=ledger, candidates=candidates, meta_features=meta_features, params=params, stage_i_oos_dir=args.stage_i_oos_dir) if args.release_metadata_json else None
    print(json.dumps({"status": "complete", "output_dir": str(root), "decision": result.manifest.get("decision"), "selected_candidate_id": result.selected_candidate_id, "selected_control_arm": result.selected_control_arm, "winner_bundle": str(winner) if winner else None}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
