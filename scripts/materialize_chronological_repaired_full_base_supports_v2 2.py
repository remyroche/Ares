#!/usr/bin/env python3
"""Decision-time-corrected chronological repaired-base support sidecars.

This is the v2 successor to ``materialize_chronological_repaired_full_base_supports``.
V1 assigned folds from the signal timestamp (``__ts__``), whereas every target
and downstream decision is made at ``execution_decision_utc == __ts__ + 1h``.
V2 uses the decision timestamp for both validation membership and train
cutoffs.  The earlier v1 implementation and artifact remain preserved and are
explicitly invalidated; this module never writes to them.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import materialize_chronological_repaired_full_base_supports as v1

ROOT = v1.ROOT
CONVERSION = v1.CONVERSION
FULL_PANEL = v1.FULL_PANEL
REPAIR = v1.REPAIR
REPAIR_SOURCE = v1.REPAIR_SOURCE
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/canonical_repaired_full_base_chronological_supports_20260730_v2"

IDENTITY = v1.IDENTITY
SIDES = v1.SIDES
DECISION = "execution_decision_utc"
MARCH_START = pd.Timestamp("2025-03-13T00:00:00Z")
APRIL_START = pd.Timestamp("2025-04-01T00:00:00Z")
FOLDS = (
    ("march_13_15_history", MARCH_START, pd.Timestamp("2025-03-16T00:00:00Z")),
    ("march_16_18_history", pd.Timestamp("2025-03-16T00:00:00Z"), pd.Timestamp("2025-03-19T00:00:00Z")),
    ("march_19_precalibration", pd.Timestamp("2025-03-19T00:00:00Z"), pd.Timestamp("2025-03-20T00:00:00Z")),
    ("march_20_22_mapping_calibration", pd.Timestamp("2025-03-20T00:00:00Z"), pd.Timestamp("2025-03-23T00:00:00Z")),
    ("march_23_25_selection", pd.Timestamp("2025-03-23T00:00:00Z"), pd.Timestamp("2025-03-26T00:00:00Z")),
    ("march_26_28_selection", pd.Timestamp("2025-03-26T00:00:00Z"), pd.Timestamp("2025-03-29T00:00:00Z")),
    ("march_29_31_selection", pd.Timestamp("2025-03-29T00:00:00Z"), APRIL_START),
)


class SupportMaterializationError(v1.SupportMaterializationError):
    pass


def march_folds() -> tuple[tuple[str, pd.Timestamp, pd.Timestamp], ...]:
    previous_end: pd.Timestamp | None = None
    for name, start, end in FOLDS:
        if not start.tzinfo or not end.tzinfo or not start < end:
            raise SupportMaterializationError(f"invalid decision-time fold: {name}")
        if previous_end is not None and start != previous_end:
            raise SupportMaterializationError("decision-time folds must be contiguous")
        previous_end = end
    return FOLDS


def candidate_population(conversion: pd.DataFrame) -> pd.DataFrame:
    required = {
        *IDENTITY,
        DECISION,
        "model_development_eligible",
        "candidate_score_is_oof",
        "upstream_scores_are_outer_oof",
        "residual_is_oof",
    }
    missing = sorted(required.difference(conversion.columns))
    if missing:
        raise SupportMaterializationError(f"conversion input misses required provenance: {missing}")
    result = conversion.copy()
    result["__ts__"] = pd.to_datetime(result["__ts__"], utc=True, errors="raise")
    result[DECISION] = pd.to_datetime(result[DECISION], utc=True, errors="raise")
    if not result[DECISION].eq(result["__ts__"] + pd.Timedelta(hours=1)).all():
        raise SupportMaterializationError("conversion decision timestamp is not signal+1h")
    eligible = (
        result["model_development_eligible"].astype(bool)
        & result[DECISION].ge(MARCH_START)
        & result[DECISION].lt(APRIL_START)
    )
    if not (
        result.loc[eligible, "candidate_score_is_oof"].astype(bool).all()
        and result.loc[eligible, "upstream_scores_are_outer_oof"].astype(bool).all()
        and result.loc[eligible, "residual_is_oof"].astype(bool).all()
    ):
        raise SupportMaterializationError("March support candidates are not strict upstream OOF")
    result = result.loc[eligible, [*IDENTITY, DECISION]].copy()
    if len(result) != 41_472 or result.duplicated(list(IDENTITY)).any():
        raise SupportMaterializationError(
            f"full-March support identity contract failed: {len(result)} rows"
        )
    return result.sort_values(list(IDENTITY), kind="stable").reset_index(drop=True)


def strict_train_mask(frame: pd.DataFrame, validation_start: pd.Timestamp) -> np.ndarray:
    start = pd.Timestamp(validation_start)
    decision = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    resolution = pd.to_datetime(frame["execution_label_end_utc"], utc=True, errors="raise")
    mask = decision.lt(start).to_numpy() & resolution.lt(start).to_numpy()
    if mask.any() and not (
        decision.loc[mask].lt(start).all() and resolution.loc[mask].lt(start).all()
    ):
        raise AssertionError("decision-time support train includes unresolved/future rows")
    return mask


def fold_mask(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    decision = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    return decision.ge(start).to_numpy() & decision.lt(end).to_numpy()


def score_fold(
    source: pd.DataFrame,
    evaluation: pd.DataFrame,
    configs: Sequence[v1.FrozenConfig],
    *,
    fold_name: str,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    threads: int,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    train_mask = strict_train_mask(source, validation_start)
    result = evaluation.loc[:, [*IDENTITY, DECISION]].copy()
    provenance: list[dict[str, Any]] = []
    geometry_by_name = {geometry.name: geometry for geometry in v1.full_base.GEOMETRIES}
    for side_index, side in enumerate(SIDES):
        train = source.loc[train_mask & source.side_name.eq(side)].copy()
        valid = evaluation.loc[evaluation.side_name.eq(side)].copy()
        if train.empty or valid.empty:
            raise SupportMaterializationError(f"empty {side} support split for {fold_name}")
        train_end = pd.to_datetime(train.execution_label_end_utc, utc=True, errors="raise").max()
        train_decision = pd.to_datetime(train.__decision_ts__, utc=True, errors="raise").max()
        if not train_end < validation_start or not train_decision < validation_start:
            raise SupportMaterializationError(f"decision-time support cutoff failed for {fold_name}/{side}")
        valid_index = valid.index.to_numpy()
        result.loc[valid_index, "support_fold"] = fold_name
        result.loc[valid_index, "support_validation_start_utc"] = validation_start
        result.loc[valid_index, "support_validation_end_utc"] = validation_end
        result.loc[valid_index, "support_train_decision_max_utc"] = train_decision
        result.loc[valid_index, "support_train_label_end_max_utc"] = train_end
        result.loc[valid_index, "support_train_rows"] = len(train)
        for config_index, config in enumerate(configs):
            features = tuple(v1.full_base.arm_features(config.arm, side))
            prediction, _ = v1.full_base.fit_predict_model(
                v1.full_base.numeric_features(train, features),
                v1.full_base.target_values(train, config.target),
                v1.full_base.numeric_features(valid, features),
                target=config.target,
                geometry=geometry_by_name[config.geometry],
                seed=int(seed + 10_000 * config_index + 100 * side_index),
                threads=int(threads),
            )
            result.loc[valid_index, f"support__{config.name}"] = prediction
            provenance.append(
                {
                    "fold": fold_name,
                    "validation_start_utc": validation_start,
                    "validation_end_utc": validation_end,
                    "side_name": side,
                    "config": config.name,
                    "target": config.target,
                    "arm": config.arm,
                    "geometry": config.geometry,
                    "features_json": json.dumps(features),
                    "feature_count": len(features),
                    "train_rows": len(train),
                    "train_decision_max_utc": train_decision,
                    "train_label_end_max_utc": train_end,
                    "validation_rows": len(valid),
                }
            )
    support_columns = [f"support__{config.name}" for config in configs]
    if result[support_columns].isna().any().any() or not np.isfinite(result[support_columns].to_numpy(float)).all():
        raise SupportMaterializationError(f"support scoring produced incomplete values for {fold_name}")
    return result, provenance


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    conversion_manifest = v1.verify_seal(args.conversion, "v5_conversion_residual_input_v3")
    full_manifest = v1.verify_seal(args.full_panel, "canonical_opportunity_payoff_trust_panel_v2")
    repair_manifest = v1.verify_seal(args.repair, "canonical_full_base_opportunity_ablation_raw_oof_repair_v2")
    source_manifest = v1.verify_seal(args.repair_source, "canonical_full_base_opportunity_ablation_v1")
    v1.verify_output_hash(args.conversion, conversion_manifest, "panel.parquet")
    configs = v1.frozen_configs(repair_manifest, source_manifest)

    conversion = pd.read_parquet(
        args.conversion / "panel.parquet",
        columns=[*IDENTITY, DECISION, "model_development_eligible", "candidate_score_is_oof", "upstream_scores_are_outer_oof", "residual_is_oof"],
        filters=[("model_development_eligible", "==", True)],
    )
    candidates = candidate_population(conversion)
    source = v1.load_pre_april_full_source(args.full_panel, full_manifest)
    evaluation = source.merge(candidates, on=list(IDENTITY), how="inner", validate="one_to_one")
    if len(evaluation) != len(candidates) or not evaluation["__decision_ts__"].eq(evaluation[DECISION]).all():
        raise SupportMaterializationError("full source/candidate decision-time parity failed")

    pieces: list[pd.DataFrame] = []
    provenance: list[dict[str, Any]] = []
    for fold_number, (fold_name, start, end) in enumerate(march_folds()):
        valid = evaluation.loc[fold_mask(evaluation, start, end)].copy()
        if valid.empty:
            raise SupportMaterializationError(f"empty candidate support fold: {fold_name}")
        scored, audit = score_fold(
            source, valid, configs, fold_name=fold_name, validation_start=start,
            validation_end=end, threads=args.threads, seed=args.seed + fold_number * 1_000_000,
        )
        pieces.append(scored)
        provenance.extend(audit)
    sidecars = pd.concat(pieces, ignore_index=True).sort_values(list(IDENTITY), kind="stable").reset_index(drop=True)
    if len(sidecars) != 41_472 or sidecars.duplicated(list(IDENTITY)).any():
        raise SupportMaterializationError("decision-time chronological support output identity drift")
    if not (
        sidecars.support_train_decision_max_utc.lt(sidecars.support_validation_start_utc).all()
        and sidecars.support_train_label_end_max_utc.lt(sidecars.support_validation_start_utc).all()
    ):
        raise SupportMaterializationError("persisted decision-time support provenance violates cutoff")

    stage = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent))
    try:
        sidecars.to_parquet(stage / "support_sidecars.parquet", index=False, compression="zstd")
        pd.DataFrame(provenance).to_csv(stage / "fold_provenance.csv", index=False)
        feature_contract = {
            config.name: {side: list(v1.full_base.arm_features(config.arm, side)) for side in SIDES}
            for config in configs
        }
        v1.write_json(stage / "feature_contract.json", feature_contract)
        outputs = {path.name: v1.sha256(path) for path in stage.iterdir() if path.is_file()}
        manifest = {
            "schema": "canonical_repaired_full_base_chronological_supports_v2",
            "run_id": args.output_dir.name,
            "status": "SEALED_RESEARCH_SUPPORT_SIDECARS_DECISION_TIME_NO_HPO_NO_MAPPING_NO_PROMOTION",
            "promotion_eligible": False,
            "portfolio_replay": "NOT_RUN",
            "rows": len(sidecars),
            "side_rows": sidecars.groupby("side_name").size().to_dict(),
            "folds": [{"name": name, "validation_start_utc": start, "validation_end_utc": end} for name, start, end in FOLDS],
            "frozen_configs": [{"name": config.name, "target": config.target, "arm": config.arm, "geometry": config.geometry} for config in configs],
            "training_contract": {
                "side_local_models": True,
                "decision_time_fold_membership": DECISION,
                "chronological_rule": "__decision_ts__ < validation_start and execution_label_end_utc < validation_start",
                "purge": "exact deployed 12h label resolution cutoff",
                "selection_hpo": "none; reuse only repair-v2 frozen target/arm/geometry/features",
                "no_april_candidates_or_labels": True,
            },
            "input_sha256": {
                "conversion_manifest": v1.sha256(args.conversion / "manifest.json"),
                "conversion_panel": v1.sha256(args.conversion / "panel.parquet"),
                "full_panel_manifest": v1.sha256(args.full_panel / "manifest.json"),
                "full_panel": full_manifest["outputs_sha256"]["panel.parquet"],
                "repair_manifest": v1.sha256(args.repair / "manifest.json"),
                "repair_source_manifest": v1.sha256(args.repair_source / "manifest.json"),
            },
            "outputs_sha256": outputs,
            "runner": {"path": str(Path(__file__).resolve()), "sha256": v1.sha256(Path(__file__).resolve())},
            "limitations": [
                "No causal EV mapping or policy evaluation is produced.",
                "No April candidate or label row enters a side-local fit or output row.",
                "The frozen full-base configuration selection remains historical static-OOF research evidence; this runner repairs only prediction provenance.",
            ],
        }
        v1.write_json(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(v1.sha256(stage / "manifest.json") + "  manifest.json\n")
        os.replace(stage, args.output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--conversion", type=Path, default=CONVERSION)
    command.add_argument("--full-panel", type=Path, default=FULL_PANEL)
    command.add_argument("--repair", type=Path, default=REPAIR)
    command.add_argument("--repair-source", type=Path, default=REPAIR_SOURCE)
    command.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    command.add_argument("--threads", type=int, default=4)
    command.add_argument("--seed", type=int, default=v1.full_base.SEED)
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(v1.safe(run(args)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
