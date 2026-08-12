#!/usr/bin/env python3
"""Run the frozen, sequential Stage-III shared-regime residual funnel.

This command deliberately accepts one *enriched* Stage-II locked-OOS ledger:
the supplied ledger must retain the exact Stage-II OOS identity and economic
columns, while adding the separately frozen causal soft-regime, relative, and
validity fields required by Stage III.  It refuses an unbound parquet file,
hard routing, absent transport evaluation, or a mutable output directory.

The A -> T -> B -> C -> D -> E -> F sequence is fixed in code.  Intermediate
checkpoints are compact audit receipts only; they are not model checkpoints and
there is intentionally no unsafe partial-run resume mode.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.shared_regime_residual_expert import SoftRegimeResidualConfig
from extreme_price_movements.stage_ii_production_oos import load_stage_ii_winner_bundle
from extreme_price_movements.stage_iii_artifacts import (
    StageIIIReproducibilityManifest,
    publish_stage_iii_compact_bundle,
)
from extreme_price_movements.stage_iii_reporting import (
    StageIIIReportingConfig,
    build_stage_iii_report_tables,
)
from extreme_price_movements.stage_iii_shared_expert_runner import (
    StageIIIInputLineageContract,
    StageIIIRunnerConfig,
    build_expanding_environment_folds,
    run_stage_iii_sequential_funnel,
)


SCHEMA = "stage_iii_shared_expert_cli_v1"
_IDENTITY = ("candidate_id", "symbol", "decision_ts", "side_name")
_STAGE_II_ECONOMICS = (
    "exact_gross_bps", "exact_net_bps", "total_cost_bps",
    "prequential_base_expected_net_bps", "r3_p_adverse", "r3_p_weak", "r3_p_clear",
)
_STAGE_II_ALIASES = {
    "r3_is_strict_oof": "base_is_strict_oof",
    "r3_source_side": "base_source_side",
    "r3_fit_end_ts": "base_train_max_label_available_ts",
    "r3_score_semantics": "base_score_semantics",
    "causal_21d_admitted": "meta_causal_21d_side_admitted_ge_50bps",
    "causal_21d_admission_is_prequential": "meta_causal_21d_admission_is_prequential",
    "causal_21d_admission_source_side": "meta_causal_21d_admission_source_side",
    "causal_21d_admission_max_label_available_ts": "meta_causal_21d_admission_max_label_available_ts",
    "causal_21d_admission_window_days": "meta_causal_21d_admission_window_days",
}


class StageIIICommandError(ValueError):
    """Raised when a command-line run lacks immutable predecessor evidence."""


def _read_json(path: Path, *, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StageIIICommandError(f"{name} must be readable JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise StageIIICommandError(f"{name} must be a JSON object")
    return value


def _digest(path: Path) -> str:
    if not path.is_file():
        raise StageIIICommandError(f"required frozen artifact is missing: {path}")
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_identity_digest(frame: pd.DataFrame, *, columns: tuple[str, ...] = _IDENTITY) -> str:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise StageIIICommandError(f"ledger lacks immutable identity columns: {missing}")
    work = frame.loc[:, list(columns)].copy()
    work["decision_ts"] = pd.to_datetime(work["decision_ts"], utc=True, errors="coerce")
    if work.isna().any().any() or work.duplicated(list(columns)).any():
        raise StageIIICommandError("ledger identity is invalid or duplicated")
    ordered = work.astype("string").sort_values(list(columns), kind="stable")
    digest = sha256()
    for row in ordered.to_numpy(str):
        digest.update("|".join(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _stable_source_digest(frame: pd.DataFrame) -> str:
    economics = list(_STAGE_II_ECONOMICS)
    if "prequential_joint_expected_net_bps" in frame:
        economics.append("prequential_joint_expected_net_bps")
    missing = [column for column in (*_IDENTITY, *economics) if column not in frame]
    if missing:
        raise StageIIICommandError(f"ledger loses frozen Stage-II economic fields: {missing}")
    work = frame.loc[:, [*_IDENTITY, *economics]].copy()
    work["decision_ts"] = pd.to_datetime(work["decision_ts"], utc=True, errors="coerce")
    for column in economics:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    if work.isna().any().any() or work.duplicated(list(_IDENTITY)).any():
        raise StageIIICommandError("frozen Stage-II economic source fields are invalid")
    work = work.sort_values(list(_IDENTITY), kind="stable").reset_index(drop=True)
    values = pd.util.hash_pandas_object(work, index=False, categorize=True).to_numpy(np.uint64)
    return sha256(values.tobytes()).hexdigest()


def _validate_stage_ii_sources(winner_directory: Path, oos_directory: Path) -> tuple[Any, dict[str, Any]]:
    """Load a complete Stage-II winner and a checksummed locked-OOS bundle."""
    winner = load_stage_ii_winner_bundle(winner_directory)
    required = ("locked_oos_ledger.parquet", "run_manifest.json", "checksums.json")
    if not oos_directory.is_dir() or any(not (oos_directory / name).is_file() for name in required):
        raise StageIIICommandError("Stage-II locked OOS bundle is incomplete")
    checksums = _read_json(oos_directory / "checksums.json", name="Stage-II OOS checksums")
    if not checksums:
        raise StageIIICommandError("Stage-II OOS checksums are empty")
    for name, expected in checksums.items():
        if not isinstance(expected, str) or _digest(oos_directory / str(name)) != expected:
            raise StageIIICommandError("Stage-II locked OOS checksum mismatch")
    manifest = dict(_read_json(oos_directory / "run_manifest.json", name="Stage-II OOS manifest"))
    expected_winner_hash = _digest(winner_directory / "winner_manifest.json")
    if str(manifest.get("winner_manifest_sha256", "")) != expected_winner_hash:
        raise StageIIICommandError("Stage-II OOS bundle is not bound to the supplied frozen winner")
    for field in ("stage_i_base_winner_artifact_sha256", "stage_i_base_oof_ledger_sha256"):
        if str(manifest.get(field, "")) != str(getattr(winner, field)):
            raise StageIIICommandError("Stage-II OOS bundle is not bound to the winner's frozen Stage-I base contract")
    if manifest.get("selection_forbidden") is not True or manifest.get("reselection_forbidden") is not True:
        raise StageIIICommandError("Stage-II OOS source did not forbid selection/reselection")
    return winner, manifest


def _apply_stage_ii_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    """Expose Stage-II names under the Stage-III direct-R3/admission contract."""
    work = frame.copy()
    for target, source in _STAGE_II_ALIASES.items():
        if target in work and source in work and not work[target].equals(work[source]):
            raise StageIIICommandError(f"conflicting Stage-II/III lineage aliases: {target} != {source}")
        if target not in work and source in work:
            work[target] = work[source]
    return work


def _parse_feature_groups(path: Path) -> dict[str, Any]:
    value = dict(_read_json(path, name="feature groups"))
    required = (
        "soft_regime_columns", "invariant_features", "regime_relative_features",
        "restricted_interaction_features", "validity_feature_groups",
    )
    missing = [name for name in required if name not in value]
    if missing:
        raise StageIIICommandError(f"feature groups lacks fields: {missing}")
    for name in required[:-1]:
        if isinstance(value[name], (str, bytes)) or not isinstance(value[name], list):
            raise StageIIICommandError(f"feature groups {name} must be a JSON list")
        value[name] = tuple(map(str, value[name]))
    if not isinstance(value["validity_feature_groups"], Mapping):
        raise StageIIICommandError("validity_feature_groups must be a JSON object")
    value["validity_feature_groups"] = {
        str(key): tuple(map(str, fields))
        for key, fields in value["validity_feature_groups"].items()
        if isinstance(fields, list)
    }
    if len(value["validity_feature_groups"]) != len(_read_json(path, name="feature groups")["validity_feature_groups"]):
        raise StageIIICommandError("every validity feature group must be a list")
    if len(value["soft_regime_columns"]) < 2 or not value["invariant_features"]:
        raise StageIIICommandError("Stage III requires >=2 soft regime fields and an invariant feature core")
    return value


def _parse_runner_config(path: Path) -> StageIIIRunnerConfig:
    value = dict(_read_json(path, name="Stage-III runner config"))
    if "baseline_config" in value:
        if not isinstance(value["baseline_config"], Mapping):
            raise StageIIICommandError("baseline_config must be an object")
        value["baseline_config"] = SoftRegimeResidualConfig(**dict(value["baseline_config"]))
    if "top_fractions" in value:
        value["top_fractions"] = tuple(float(item) for item in value["top_fractions"])
    try:
        config = StageIIIRunnerConfig(**value)
    except TypeError as exc:
        raise StageIIICommandError("Stage-III runner config has unsupported fields") from exc
    config.validate()
    if not config.run_transport_matrix:
        raise StageIIICommandError("Stage III requires the train→test transport matrix; run_transport_matrix=false is forbidden")
    if not config.hard_regime_column:
        raise StageIIICommandError("A2 requires an explicit causal hard_regime_column for its diagnostic baseline")
    return config


def _parse_reproducibility(path: Path, *, lineage: StageIIIInputLineageContract) -> StageIIIReproducibilityManifest:
    value = dict(_read_json(path, name="reproducibility manifest"))
    try:
        reproducibility = StageIIIReproducibilityManifest(**value)
    except TypeError as exc:
        raise StageIIICommandError("reproducibility manifest has unsupported/missing fields") from exc
    reproducibility.validate()
    if reproducibility.feature_contract_sha256 != lineage.feature_contract_sha256:
        raise StageIIICommandError("reproducibility feature contract does not equal frozen lineage contract")
    if reproducibility.input_lineage_contract_sha256 != lineage.contract_sha256:
        raise StageIIICommandError("reproducibility lineage hash does not equal the supplied input lineage")
    return reproducibility


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, default=str)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _checkpoint_callback(checkpoint_directory: Path, *, source_digest: str):
    def callback(round_name: str, arms: tuple[Any, ...], winner: Any) -> None:
        payload = {
            "schema": SCHEMA,
            "status": "round_complete_not_resumable",
            "round": round_name,
            "source_stage_ii_economic_digest": source_digest,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "winner": winner.arm,
            "winner_selection": dict(winner.selection_summary),
            "winner_feature_contract_sha256": winner.model_feature_contract_sha256,
            "arms": [
                {"arm": arm.arm, "selection": dict(arm.selection_summary),
                 "feature_contract_sha256": arm.model_feature_contract_sha256}
                for arm in arms
            ],
        }
        _atomic_json(checkpoint_directory / f"{round_name}.json", payload)
    return callback


def _preflight(
    *, ledger: pd.DataFrame, source_ledger: pd.DataFrame,
    config: StageIIIRunnerConfig, lineage: StageIIIInputLineageContract,
    groups: Mapping[str, Any],
) -> str:
    if _canonical_identity_digest(ledger) != _canonical_identity_digest(source_ledger):
        raise StageIIICommandError("enriched ledger identity differs from frozen Stage-II OOS ledger")
    if _stable_source_digest(ledger) != _stable_source_digest(source_ledger):
        raise StageIIICommandError("enriched ledger changes frozen Stage-II economics/base outputs")
    if lineage.require_direct_fq3_meta_lineage:
        required = (
            lineage.direct_joint_expected_net_column,
            lineage.direct_joint_mapping_semantics_column,
        )
        missing = [name for name in required if name not in source_ledger or name not in ledger]
        if missing:
            raise StageIIICommandError(f"direct FQ3 predecessor lacks canonical joint mapping evidence: {missing}")
        if not np.allclose(
            pd.to_numeric(ledger[lineage.direct_joint_expected_net_column], errors="coerce").to_numpy(float),
            pd.to_numeric(source_ledger[lineage.direct_joint_expected_net_column], errors="coerce").to_numpy(float),
            equal_nan=False,
        ) or not ledger[lineage.direct_joint_mapping_semantics_column].astype(str).equals(
            source_ledger[lineage.direct_joint_mapping_semantics_column].astype(str)
        ):
            raise StageIIICommandError("enriched ledger changes the frozen direct FQ3 joint mapping")
    lineage.validate(
        ledger, config=config, soft_regime_columns=groups["soft_regime_columns"],
        invariant_features=groups["invariant_features"],
        regime_relative_features=groups["regime_relative_features"],
        restricted_interaction_features=groups["restricted_interaction_features"],
        validity_feature_groups=groups["validity_feature_groups"],
    )
    folds = build_expanding_environment_folds(ledger, config=config)
    if not folds:
        raise StageIIICommandError("no strict expanding environment folds are available")
    return _stable_source_digest(source_ledger)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-ii-winner-bundle", type=Path, required=True)
    parser.add_argument("--stage-ii-oos-bundle", type=Path, required=True)
    parser.add_argument("--enriched-ledger", type=Path, required=True)
    parser.add_argument("--input-lineage", type=Path, required=True)
    parser.add_argument("--feature-groups", type=Path, required=True)
    parser.add_argument("--runner-config", type=Path, required=True)
    parser.add_argument("--reproducibility", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight:
        if args.output_dir is not None or args.checkpoint_dir is not None:
            parser.error("--preflight does not accept output/checkpoint directories")
    elif args.output_dir is None or args.checkpoint_dir is None:
        parser.error("--output-dir and --checkpoint-dir are required for a run")

    winner, stage_ii_manifest = _validate_stage_ii_sources(
        args.stage_ii_winner_bundle.resolve(), args.stage_ii_oos_bundle.resolve()
    )
    source_ledger = pd.read_parquet(args.stage_ii_oos_bundle / "locked_oos_ledger.parquet")
    ledger = _apply_stage_ii_aliases(pd.read_parquet(args.enriched_ledger))
    lineage = StageIIIInputLineageContract.from_dict(_read_json(args.input_lineage, name="input lineage"))
    groups = _parse_feature_groups(args.feature_groups)
    config = _parse_runner_config(args.runner_config)
    reproducibility = _parse_reproducibility(args.reproducibility, lineage=lineage)
    source_digest = _preflight(
        ledger=ledger, source_ledger=source_ledger, config=config,
        lineage=lineage, groups=groups,
    )
    if args.preflight:
        print(json.dumps({
            "schema": SCHEMA, "status": "preflight_complete",
            "stage_ii_run_id": winner.run_id,
            "stage_ii_oos_content_sha256": stage_ii_manifest.get("oos_content_sha256"),
            "source_stage_ii_economic_digest": source_digest,
            "enriched_rows": len(ledger), "expanding_folds": len(build_expanding_environment_folds(ledger, config=config)),
        }, indent=2))
        return 0

    output = args.output_dir.resolve()
    checkpoints = args.checkpoint_dir.resolve()
    if output.exists():
        raise StageIIICommandError("Stage-III output must be a new immutable directory")
    if checkpoints.exists() and any(checkpoints.iterdir()):
        raise StageIIICommandError("checkpoint directory is non-empty; unsafe resume is forbidden")
    if not output.parent.is_dir() or not checkpoints.parent.is_dir():
        raise StageIIICommandError("output and checkpoint parents must already exist")
    checkpoints.mkdir()
    _atomic_json(checkpoints / "00_validated.json", {
        "schema": SCHEMA, "status": "validated_not_resumable",
        "source_stage_ii_economic_digest": source_digest,
        "stage_ii_winner_run_id": winner.run_id,
        "stage_ii_oos_content_sha256": stage_ii_manifest.get("oos_content_sha256"),
        "declared_rounds": ["A_target_normalization", "T_residual_target", "B_training_robustness", "C_conditioning", "D_model_validity", "E_calibration", "F_pairwise_ranking"],
    })
    try:
        result = run_stage_iii_sequential_funnel(
            ledger, config=config, input_lineage=lineage,
            soft_regime_columns=groups["soft_regime_columns"],
            invariant_features=groups["invariant_features"],
            regime_relative_features=groups["regime_relative_features"],
            restricted_interaction_features=groups["restricted_interaction_features"],
            validity_feature_groups=groups["validity_feature_groups"],
            round_checkpoint_callback=_checkpoint_callback(checkpoints, source_digest=source_digest),
        )
        reports = build_stage_iii_report_tables(
            result.winner.oof_predictions,
            score_columns={
                "base_expected_net_bps": config.base_expected_net_column,
                "shared_residual_expected_net_bps": "score_bps",
            },
            config=StageIIIReportingConfig(
                top_fractions=config.top_fractions, require_hit_surprise=False,
            ),
        )
        winner_columns = tuple(dict.fromkeys([
            "candidate_id", "symbol", "decision_ts", "side_name", "signal_close_ts",
            config.environment_column, config.exact_gross_column or "exact_gross_bps",
            config.exact_net_column, config.base_expected_net_column,
            "raw_shared_common_bps", "score_bps", "predicted_candidate_residual_bps",
            config.admission_column,
        ]))
        published = publish_stage_iii_compact_bundle(
            result, output, reproducibility=reproducibility,
            winner_prediction_columns=winner_columns, report_tables=reports,
            feature_lists={
                "stage_ii_meta": tuple(winner.ordered_meta_features),
                "stage_ii_archetype": tuple(winner.ordered_archetype_features),
                "stage_iii_shared_residual": tuple(result.winner.model_feature_names),
            },
        )
        _atomic_json(checkpoints / "99_completed.json", {
            "schema": SCHEMA, "status": "completed",
            "output_dir": str(published), "winner_arm": result.winner.arm,
            "round_winners": dict(result.round_winners),
            "advancement_gates": dict(result.advancement_gates),
        })
    except Exception as exc:
        _atomic_json(checkpoints / "99_failed.json", {
            "schema": SCHEMA, "status": "failed_not_resumable",
            "error_type": type(exc).__name__, "error": str(exc),
        })
        raise
    print(json.dumps({
        "schema": SCHEMA, "status": "complete", "output_dir": str(published),
        "winner_arm": result.winner.arm, "round_winners": dict(result.round_winners),
        "advances": bool(result.advancement_gates.get("advances")),
        "terminal_decision_code": result.advancement_gates.get("terminal_decision_code"),
    }, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
