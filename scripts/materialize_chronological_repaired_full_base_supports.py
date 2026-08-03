#!/usr/bin/env python3
"""Materialize strict chronological support scores for the repaired base heads.

The eight full-base configurations were selected previously and are *frozen*.
This runner deliberately does not revisit feature selection, target selection,
or geometry selection.  It re-fits those fixed side-local CatBoost heads only
on labels resolved before each March validation block, so a later reliability
head can consume decision-time-valid support scores rather than the earlier
static blocked-CV sidecars.

Output is research-only.  April is not read, no mapper is fit, and no policy
or promotion decision is made here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_canonical_full_base_opportunity_ablation as full_base

CONVERSION = ROOT / "data_perp/artifacts/v5_conversion_residual_input_20260730_v3"
FULL_PANEL = ROOT / "data_perp/artifacts/canonical_opportunity_payoff_trust_panel_20260729_v2"
REPAIR = ROOT / "data_perp/artifacts/canonical_full_base_opportunity_ablation_20260730_v2"
REPAIR_SOURCE = ROOT / "data_perp/artifacts/canonical_full_base_opportunity_ablation_20260729_v1"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/canonical_repaired_full_base_chronological_supports_20260730_v1"

IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
SIDES = ("long", "short")
MARCH_START = pd.Timestamp("2025-03-20T00:00:00Z")
MARCH_END = pd.Timestamp("2025-03-31T00:00:00Z")
FOLDS = (
    ("march_20_22_mapping_calibration", MARCH_START, pd.Timestamp("2025-03-23T00:00:00Z")),
    ("march_23_25_selection", pd.Timestamp("2025-03-23T00:00:00Z"), pd.Timestamp("2025-03-26T00:00:00Z")),
    ("march_26_28_selection", pd.Timestamp("2025-03-26T00:00:00Z"), pd.Timestamp("2025-03-29T00:00:00Z")),
    ("march_29_30_selection", pd.Timestamp("2025-03-29T00:00:00Z"), MARCH_END),
)


class SupportMaterializationError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrozenConfig:
    target: str
    arm: str
    geometry: str

    @property
    def name(self) -> str:
        return "__".join((self.target, self.arm, self.geometry))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(safe(dict(payload)), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def verify_seal(root: Path, schema: str) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    seal_path = root / "manifest.sha256"
    if not manifest_path.is_file() or not seal_path.is_file():
        raise SupportMaterializationError(f"sealed source is missing: {root}")
    if sha256(manifest_path) != seal_path.read_text().split()[0]:
        raise SupportMaterializationError(f"manifest seal mismatch: {root}")
    payload = json.loads(manifest_path.read_text())
    if payload.get("schema") != schema:
        raise SupportMaterializationError(f"unexpected schema for {root}: {payload.get('schema')}")
    return payload


def verify_output_hash(root: Path, manifest: Mapping[str, Any], output: str) -> None:
    path = root / output
    expected = str(manifest.get("outputs_sha256", {}).get(output, ""))
    if not expected or not path.is_file() or sha256(path) != expected:
        raise SupportMaterializationError(f"sealed output mismatch: {root / output}")


def frozen_configs(
    repair_manifest: Mapping[str, Any], source_manifest: Mapping[str, Any]
) -> tuple[FrozenConfig, ...]:
    rows = repair_manifest.get("repair", {}).get("selected_configs", [])
    configs = tuple(
        FrozenConfig(str(row["target"]), str(row["arm"]), str(row["geometry"]))
        for row in rows
    )
    if len(configs) != 8 or len({config.name for config in configs}) != 8:
        raise SupportMaterializationError("repair does not contain exactly eight frozen configs")
    allowed_targets = set(full_base.TARGETS)
    geometries = {item.name for item in full_base.GEOMETRIES}
    source_arms = source_manifest.get("features", {}).get("primary_arms", {})
    for config in configs:
        if config.target not in allowed_targets or config.arm not in source_arms:
            raise SupportMaterializationError(f"frozen config escaped original source: {config.name}")
        if config.geometry not in geometries:
            raise SupportMaterializationError(f"unknown frozen geometry: {config.name}")
        for side in SIDES:
            declared = tuple(map(str, source_arms[config.arm][side]))
            actual = tuple(full_base.arm_features(config.arm, side))
            if declared != actual:
                raise SupportMaterializationError(
                    f"source feature drift for {config.name}/{side}: {declared} != {actual}"
                )
    return configs


def strict_train_mask(frame: pd.DataFrame, validation_start: pd.Timestamp) -> np.ndarray:
    start = pd.Timestamp(validation_start)
    decision = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    resolution = pd.to_datetime(frame["execution_label_end_utc"], utc=True, errors="raise")
    mask = decision.lt(start).to_numpy() & resolution.lt(start).to_numpy()
    if mask.any() and not resolution.loc[mask].lt(start).all():
        raise AssertionError("chronological support train includes unresolved labels")
    return mask


def march_folds() -> tuple[tuple[str, pd.Timestamp, pd.Timestamp], ...]:
    previous_end: pd.Timestamp | None = None
    for name, start, end in FOLDS:
        if not start.tzinfo or not end.tzinfo or not start < end:
            raise SupportMaterializationError(f"invalid support fold: {name}")
        if previous_end is not None and start != previous_end:
            raise SupportMaterializationError("support folds must be contiguous")
        previous_end = end
    return FOLDS


def candidate_population(conversion: pd.DataFrame) -> pd.DataFrame:
    required = {*IDENTITY, "model_development_eligible", "candidate_score_is_oof", "upstream_scores_are_outer_oof", "residual_is_oof"}
    missing = sorted(required.difference(conversion.columns))
    if missing:
        raise SupportMaterializationError(f"conversion input misses required provenance: {missing}")
    result = conversion.copy()
    result["__ts__"] = pd.to_datetime(result["__ts__"], utc=True, errors="raise")
    eligible = (
        result["model_development_eligible"].astype(bool)
        & result["__ts__"].ge(MARCH_START)
        & result["__ts__"].lt(MARCH_END)
    )
    if not (
        result.loc[eligible, "candidate_score_is_oof"].astype(bool).all()
        and result.loc[eligible, "upstream_scores_are_outer_oof"].astype(bool).all()
        and result.loc[eligible, "residual_is_oof"].astype(bool).all()
    ):
        raise SupportMaterializationError("March support candidates are not strict upstream OOF")
    result = result.loc[eligible, list(IDENTITY)].copy()
    if result.empty or result.duplicated(list(IDENTITY)).any():
        raise SupportMaterializationError("March support candidate identity contract failed")
    return result.sort_values(list(IDENTITY), kind="stable").reset_index(drop=True)


def load_pre_april_full_source(root: Path, manifest: Mapping[str, Any]) -> pd.DataFrame:
    """Load only the February/March rows needed for chronological support fits."""

    verify_output_hash(root, manifest, "panel.parquet")
    required = list(full_base.required_columns())
    source = pd.read_parquet(
        root / "panel.parquet",
        columns=required,
        filters=[("__ts__", "<", MARCH_END)],
    )
    for column in ("__ts__", "__decision_ts__", "execution_label_end_utc", "effective_label_resolution_utc"):
        source[column] = pd.to_datetime(source[column], utc=True, errors="raise")
    if source.empty or source["__ts__"].ge(MARCH_END).any():
        raise SupportMaterializationError("pre-April source filter failed")
    if source.duplicated(list(IDENTITY)).any() or source.candidate_id.astype(str).duplicated().any():
        raise SupportMaterializationError("pre-April full source identity contract failed")
    if not source.side_name.isin(SIDES).all():
        raise SupportMaterializationError("pre-April full source has unknown side")
    if not source["__decision_ts__"].eq(source["__ts__"] + pd.Timedelta(hours=1)).all():
        raise SupportMaterializationError("pre-April full source decision timestamp mismatch")
    if not source["execution_label_end_utc"].eq(source["__decision_ts__"] + pd.Timedelta(hours=12)).all():
        raise SupportMaterializationError("pre-April full source label horizon mismatch")
    if not np.allclose(
        source["execution_gross_ev_12h"] - source["execution_cost_return"],
        source["execution_net_ev_12h"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise SupportMaterializationError("pre-April full source gross-cost-net mismatch")
    return source


def score_fold(
    source: pd.DataFrame,
    evaluation: pd.DataFrame,
    configs: Sequence[FrozenConfig],
    *,
    fold_name: str,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    threads: int,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    train_mask = strict_train_mask(source, validation_start)
    result = evaluation.loc[:, list(IDENTITY)].copy()
    provenance: list[dict[str, Any]] = []
    geometry_by_name = {geometry.name: geometry for geometry in full_base.GEOMETRIES}
    for side_index, side in enumerate(SIDES):
        train = source.loc[train_mask & source.side_name.eq(side)].copy()
        valid = evaluation.loc[evaluation.side_name.eq(side)].copy()
        if train.empty or valid.empty:
            raise SupportMaterializationError(f"empty {side} support split for {fold_name}")
        train_end = pd.to_datetime(train.execution_label_end_utc, utc=True, errors="raise").max()
        if not train_end < validation_start:
            raise SupportMaterializationError(f"support train cutoff failed for {fold_name}/{side}")
        valid_index = valid.index.to_numpy()
        result.loc[valid_index, "support_fold"] = fold_name
        result.loc[valid_index, "support_validation_start_utc"] = validation_start
        result.loc[valid_index, "support_validation_end_utc"] = validation_end
        result.loc[valid_index, "support_train_label_end_max_utc"] = train_end
        result.loc[valid_index, "support_train_rows"] = len(train)
        for config_index, config in enumerate(configs):
            features = tuple(full_base.arm_features(config.arm, side))
            prediction, _ = full_base.fit_predict_model(
                full_base.numeric_features(train, features),
                full_base.target_values(train, config.target),
                full_base.numeric_features(valid, features),
                target=config.target,
                geometry=geometry_by_name[config.geometry],
                seed=int(seed + 10_000 * config_index + 100 * side_index),
                threads=int(threads),
            )
            column = f"support__{config.name}"
            result.loc[valid_index, column] = prediction
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
    conversion_manifest = verify_seal(args.conversion, "v5_conversion_residual_input_v3")
    full_manifest = verify_seal(args.full_panel, "canonical_opportunity_payoff_trust_panel_v2")
    repair_manifest = verify_seal(args.repair, "canonical_full_base_opportunity_ablation_raw_oof_repair_v2")
    source_manifest = verify_seal(args.repair_source, "canonical_full_base_opportunity_ablation_v1")
    verify_output_hash(args.conversion, conversion_manifest, "panel.parquet")
    configs = frozen_configs(repair_manifest, source_manifest)

    conversion = pd.read_parquet(
        args.conversion / "panel.parquet",
        columns=[*IDENTITY, "model_development_eligible", "candidate_score_is_oof", "upstream_scores_are_outer_oof", "residual_is_oof"],
        filters=[("model_development_eligible", "==", True), ("__ts__", ">=", MARCH_START), ("__ts__", "<", MARCH_END)],
    )
    candidates = candidate_population(conversion)
    source = load_pre_april_full_source(args.full_panel, full_manifest)
    evaluation = source.merge(candidates, on=list(IDENTITY), how="inner", validate="one_to_one")
    if len(evaluation) != len(candidates):
        raise SupportMaterializationError("full panel does not cover all chronological support candidates")
    if evaluation["__ts__"].ge(MARCH_END).any() or evaluation["__ts__"].lt(MARCH_START).any():
        raise SupportMaterializationError("support evaluation escaped March 20-30")

    pieces: list[pd.DataFrame] = []
    provenance: list[dict[str, Any]] = []
    for fold_number, (fold_name, start, end) in enumerate(march_folds()):
        valid = evaluation.loc[evaluation["__ts__"].ge(start) & evaluation["__ts__"].lt(end)].copy()
        if valid.empty:
            raise SupportMaterializationError(f"empty candidate support fold: {fold_name}")
        scored, audit = score_fold(
            source,
            valid,
            configs,
            fold_name=fold_name,
            validation_start=start,
            validation_end=end,
            threads=args.threads,
            seed=args.seed + fold_number * 1_000_000,
        )
        pieces.append(scored)
        provenance.extend(audit)
    sidecars = pd.concat(pieces, ignore_index=True).sort_values(list(IDENTITY), kind="stable").reset_index(drop=True)
    if len(sidecars) != len(candidates) or sidecars.duplicated(list(IDENTITY)).any():
        raise SupportMaterializationError("chronological support output identity drift")
    if not sidecars.support_train_label_end_max_utc.lt(sidecars.support_validation_start_utc).all():
        raise SupportMaterializationError("persisted support provenance violates chronological cutoff")

    stage = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent))
    try:
        sidecars.to_parquet(stage / "support_sidecars.parquet", index=False, compression="zstd")
        pd.DataFrame(provenance).to_csv(stage / "fold_provenance.csv", index=False)
        feature_contract = {
            config.name: {side: list(full_base.arm_features(config.arm, side)) for side in SIDES}
            for config in configs
        }
        write_json(stage / "feature_contract.json", feature_contract)
        outputs = {path.name: sha256(path) for path in stage.iterdir() if path.is_file()}
        manifest = {
            "schema": "canonical_repaired_full_base_chronological_supports_v1",
            "run_id": args.output_dir.name,
            "status": "SEALED_RESEARCH_SUPPORT_SIDECARS_NO_HPO_NO_MAPPING_NO_PROMOTION",
            "promotion_eligible": False,
            "portfolio_replay": "NOT_RUN",
            "rows": len(sidecars),
            "side_rows": sidecars.groupby("side_name").size().to_dict(),
            "folds": [
                {"name": name, "validation_start_utc": start, "validation_end_utc": end}
                for name, start, end in FOLDS
            ],
            "frozen_configs": [
                {"name": config.name, "target": config.target, "arm": config.arm, "geometry": config.geometry}
                for config in configs
            ],
            "training_contract": {
                "side_local_models": True,
                "chronological_rule": "decision_ts < validation_start and execution_label_end_utc < validation_start",
                "purge": "exact deployed 12h label resolution cutoff",
                "selection_hpo": "none; reuse only repair-v2 frozen target/arm/geometry/features",
                "no_april_read": True,
            },
            "input_sha256": {
                "conversion_manifest": sha256(args.conversion / "manifest.json"),
                "conversion_panel": sha256(args.conversion / "panel.parquet"),
                "full_panel_manifest": sha256(args.full_panel / "manifest.json"),
                "full_panel": full_manifest["outputs_sha256"]["panel.parquet"],
                "repair_manifest": sha256(args.repair / "manifest.json"),
                "repair_source_manifest": sha256(args.repair_source / "manifest.json"),
            },
            "outputs_sha256": outputs,
            "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
            "limitations": [
                "These are research support features only; no causal EV mapper or policy evaluation is produced.",
                "April is not read or scored by this materialization.",
                "The eight configurations remain frozen from the repaired static-OOF selection; this runner does not make that historical selection chronological.",
            ],
        }
        write_json(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(sha256(stage / "manifest.json") + "  manifest.json\n")
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
    command.add_argument("--seed", type=int, default=full_base.SEED)
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(safe(run(args)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
