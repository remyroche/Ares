#!/usr/bin/env python3
"""One-shot frozen Stage-II locked-OOS scorer and publisher.

No discovery, control comparison, HPO, feature selection, or post-hoc mapping
is exposed here.  It fits the already-selected causal recogniser and
side-local residual once on the winner's history+development windows, scores
the locked evaluation identity once, and delegates immutable publication and
pooled-global raw/admitted reporting to ``stage_ii_production_oos``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extreme_price_movements.stage_ii_execution import (
    StageIIExecutionError,
    build_stage_ii_ledger,
    file_sha256,
    make_locked_stage_ii_scorer,
    validate_enriched_ledger_manifest,
)
from extreme_price_movements.stage_ii_meta_archetypes import StageIIMetaArchetypeConfig
from extreme_price_movements.stage_ii_production_oos import (
    StageIIFoldLineage,
    StageIILockedOOSScoringRequest,
    StageIILockedOOSScoringResult,
    _digest_bytes,
    _feature_contract_hash,
    _fold_lineage_hash,
    load_stage_ii_winner_bundle,
    run_and_publish_stage_ii_locked_oos_scoring,
)


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StageIIExecutionError(f"JSON object expected: {path}")
    return value


def _window_mask(frame: pd.DataFrame, window: object, interval: str) -> pd.Series:
    decision = pd.to_datetime(frame.decision_ts, utc=True, errors="raise")
    return pd.Series(window.contains(decision, interval=interval), index=frame.index)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-i-oos-dir", type=Path, required=True)
    parser.add_argument("--stage-ii-winner-bundle", type=Path, required=True)
    parser.add_argument("--enriched-ledger", type=Path, required=True)
    parser.add_argument("--enriched-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    stage_i_manifest = _json(args.stage_i_oos_dir / "manifest.json")
    if stage_i_manifest.get("status") != "complete" or stage_i_manifest.get("schema") != "stage_i_production_winner_oos_v1":
        raise StageIIExecutionError("Stage-II locked OOS requires a completed Stage-I production artifact")
    winner = load_stage_ii_winner_bundle(args.stage_ii_winner_bundle)
    winner_file = args.stage_i_oos_dir / "winner_bundle.json"
    source_file = args.stage_i_oos_dir / "full_history_raw_oof_predictions.parquet"
    if not winner_file.is_file() or not source_file.is_file():
        raise StageIIExecutionError("completed Stage-I artifact lacks its frozen winner/full-history OOF files")
    if file_sha256(winner_file) != winner.stage_i_base_winner_artifact_sha256 or file_sha256(source_file) != winner.stage_i_base_oof_ledger_sha256:
        raise StageIIExecutionError("Stage-II winner is not bound to this exact Stage-I winner/OOF ledger")
    selected = winner.selected_config.get("candidate")
    params = winner.selected_config.get("meta_params")
    if not isinstance(selected, dict) or not isinstance(selected.get("config"), dict) or not isinstance(params, dict):
        raise StageIIExecutionError("frozen Stage-II winner lacks candidate config or Huber parameter contract")
    config = StageIIMetaArchetypeConfig(**dict(selected["config"]))
    causal = tuple(map(str, selected.get("causal_feature_cols", ())))
    if not causal:
        raise StageIIExecutionError("frozen Stage-II winner lacks causal recogniser features")
    base_meta = {"prequential_base_expected_net_bps", "r3_p_adverse", "r3_p_weak", "r3_p_clear"}
    meta = tuple(name for name in winner.ordered_meta_features if name not in base_meta)
    enriched_manifest = _json(args.enriched_manifest)
    validate_enriched_ledger_manifest(
        enriched_manifest, ledger_path=args.enriched_ledger,
        required_causal_columns=tuple(dict.fromkeys((*meta, *causal))),
        required_path_columns=config.path_descriptor_cols,
    )
    full, base_catalog = build_stage_ii_ledger(
        stage_i_predictions=pd.read_parquet(source_file),
        stage_i_fold_provenance=pd.read_parquet(args.stage_i_oos_dir / "fold_provenance.parquet"),
        enriched_ledger=pd.read_parquet(args.enriched_ledger),
        required_enriched_columns=tuple(dict.fromkeys((*meta, *causal, *config.path_descriptor_cols))),
    )
    history = full.loc[_window_mask(full, winner.window, "history")].copy()
    development = full.loc[_window_mask(full, winner.window, "development")].copy()
    evaluation = full.loc[_window_mask(full, winner.window, "locked_evaluation")].copy()
    if history.empty or development.empty or evaluation.empty:
        raise StageIIExecutionError("frozen Stage-II windows have no complete direct-base/enriched candidate population")
    base_folds = tuple(StageIIFoldLineage(**dict(item)) for item in base_catalog)
    eval_start = pd.to_datetime(evaluation.decision_ts, utc=True).min()
    eval_end = pd.to_datetime(evaluation.decision_ts, utc=True).max() + pd.Timedelta(nanoseconds=1)
    meta_folds = (StageIIFoldLineage(0, pd.to_datetime(pd.concat([history.label_available_ts, development.label_available_ts]), utc=True).max(), eval_start, eval_end),)
    request = StageIILockedOOSScoringRequest(
        winner_bundle=args.stage_ii_winner_bundle, history=history, development=development,
        evaluation_identity=evaluation.loc[:, ["candidate_id", "symbol", "signal_close_ts", "decision_ts", "side_name"]].copy(),
        base_folds=base_folds, meta_folds=meta_folds,
        stage_i_base_winner_artifact_sha256=winner.stage_i_base_winner_artifact_sha256,
        stage_i_base_oof_ledger_sha256=winner.stage_i_base_oof_ledger_sha256,
    )
    raw_scorer = make_locked_stage_ii_scorer(
        full_ledger=full, candidate_config=config, causal_feature_cols=causal,
        meta_feature_cols=meta, selected_control_arm=winner.selected_control_arm,
        meta_params=params,
    )

    def scorer(context: dict) -> StageIILockedOOSScoringResult:
        result = raw_scorer(context)
        provenance = {
            **dict(result.provenance),
            "winner_manifest_sha256": _digest_bytes(args.stage_ii_winner_bundle / "winner_manifest.json"),
            "feature_contract_sha256": _feature_contract_hash(winner),
            "label_manifest_sha256": winner.label_manifest_sha256,
            "stage_i_base_winner_artifact_sha256": winner.stage_i_base_winner_artifact_sha256,
            "stage_i_base_oof_ledger_sha256": winner.stage_i_base_oof_ledger_sha256,
            "base_fold_lineage_sha256": _fold_lineage_hash(base_folds),
            "meta_fold_lineage_sha256": _fold_lineage_hash(meta_folds),
        }
        return StageIILockedOOSScoringResult(result.ledger, provenance)

    output = run_and_publish_stage_ii_locked_oos_scoring(args.output_dir, request, scorer=scorer)
    print(json.dumps({"status": "complete", "output_dir": str(output), "selected_candidate_id": winner.selected_discovery_candidate_id, "selected_control_arm": winner.selected_control_arm, "selection_forbidden": True}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
