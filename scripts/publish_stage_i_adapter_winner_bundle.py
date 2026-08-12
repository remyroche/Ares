#!/usr/bin/env python3
"""Publish one immutable, authorized Stage-I direct-FQ3 winner bundle.

The command is deliberately a no-fit handoff step.  It accepts only one of
the R3/scalar/ordinal finalists named by ``target_finalist_contracts.json``;
it binds that exact family and arm to the signed per-side common evaluation
universe, then writes ``winner_bundle.json`` for the target-specific
materializer and OOS runner.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.stage_i_adapter_winner_bundle import (
    build_stage_i_adapter_winner_bundle,
    freeze_stage_i_adapter_winner_bundle,
)
from extreme_price_movements.stage_i_shared_population import (
    SharedPopulationError,
    file_sha256,
    validate_shared_population,
)


FAMILY_TO_TARGET = {
    "R3_control": "legacy_R3_multiclass3_control",
    "scalar_S": "soft_scalar_S",
    "ordinal_O": "cumulative_ordinal5_O",
}


def _canonical_sha(value: Mapping[str, Any]) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _load_contract(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("target-finalist contract must be one JSON object")
    if raw.get("schema") != "stage_i_base_target_joint_finalists_v2" or raw.get("status") != "complete":
        raise ValueError("target-finalist contract is incomplete or unsupported")
    expected = _canonical_sha({key: value for key, value in raw.items() if key != "contract_sha256"})
    if str(raw.get("contract_sha256", "")) != expected:
        raise ValueError("target-finalist contract checksum drift")
    return raw


def _authorized_finalist(raw: Mapping[str, Any], *, family: str, arm: str) -> Mapping[str, Any]:
    finalist = next(
        (item for item in raw.get("finalists", ()) if item.get("family") == family and item.get("arm") == arm),
        None,
    )
    if not isinstance(finalist, Mapping) or finalist.get("must_advance_to_joint_base_meta_evaluation") is not True:
        raise ValueError(f"{family}/{arm} is not an authorized joint finalist")
    return finalist


def _validate_target_winner_source(
    finalist: Mapping[str, Any], *, family: str, arm: str, target_winner_dir: Path | None,
) -> None:
    source = finalist.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("authorized finalist has no source binding")
    if family == "R3_control":
        if source.get("kind") != "existing_completed_r3_base_selection":
            raise ValueError("R3 finalist source binding drift")
        if target_winner_dir is not None:
            raise ValueError("R3 finalist must not receive a scalar/ordinal target-winner directory")
        return
    if source.get("kind") != "target_specific_winner_bundle_requires_new_base_mda":
        raise ValueError(f"{family} finalist source binding drift")
    if target_winner_dir is None:
        raise ValueError(f"{family} finalist requires --target-winner-dir")
    manifest_path = target_winner_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping) or manifest.get("status") != "complete":
        raise ValueError("target winner bundle is incomplete")
    if manifest.get("family") != family or manifest.get("target_name") != arm:
        raise ValueError("target winner family/arm does not match the authorized finalist")
    if str(source.get("bundle_manifest_sha256", "")) != file_sha256(manifest_path):
        raise ValueError("target winner manifest checksum does not match the finalist contract")


def build_authorization(
    *, target_finalists_contract: Path, shared_population_dir: Path,
    family: str, arm: str, target_winner_dir: Path | None,
) -> dict[str, Any]:
    """Return the exact signed handoff payload; exposed for focused tests."""
    contract = _load_contract(target_finalists_contract)
    finalist = _authorized_finalist(contract, family=family, arm=arm)
    _validate_target_winner_source(
        finalist, family=family, arm=arm, target_winner_dir=target_winner_dir,
    )
    contract_shared = str(contract.get("shared_population_contract_sha256", ""))
    if len(contract_shared) != 64:
        raise ValueError("target-finalist contract does not bind a signed shared population")
    try:
        _frame, shared = validate_shared_population(shared_population_dir)
    except SharedPopulationError as exc:
        raise ValueError(str(exc)) from exc
    if contract_shared != str(shared.get("contract_sha256", "")):
        raise ValueError("target-finalist contract and shared population checksum disagree")
    population_path, manifest_path = (
        shared_population_dir / "shared_population.parquet",
        shared_population_dir / "manifest.json",
    )
    embedded = contract.get("shared_population")
    if embedded is not None:
        if not isinstance(embedded, Mapping):
            raise ValueError("target-finalist shared population reference is malformed")
        if (
            str(embedded.get("contract_sha256", "")) != contract_shared
            or str(embedded.get("manifest_sha256", "")) != file_sha256(manifest_path)
            or str(embedded.get("population_file_sha256", "")) != file_sha256(population_path)
            or dict(embedded.get("per_side", {})) != dict(shared.get("per_side", {}))
        ):
            raise ValueError("target-finalist embedded shared population reference drift")
    return {
        "schema": "stage_i_adapter_joint_finalist_authorization_v1",
        "target_finalist_contract_sha256": str(contract["contract_sha256"]),
        "target_finalist_contract_file_sha256": file_sha256(target_finalists_contract),
        "family": family,
        "arm": arm,
        "shared_population_contract_sha256": contract_shared,
        "shared_population_manifest_sha256": file_sha256(manifest_path),
        "shared_population_file_sha256": file_sha256(population_path),
        "shared_population_per_side": dict(shared["per_side"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-finalists-contract", type=Path, required=True)
    parser.add_argument("--shared-population-dir", type=Path, required=True)
    parser.add_argument("--family", choices=tuple(FAMILY_TO_TARGET), required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--base-selection-dir", type=Path, required=True)
    parser.add_argument("--meta-selection-dir", type=Path, required=True)
    parser.add_argument("--target-winner-dir", type=Path)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    authorization = build_authorization(
        target_finalists_contract=args.target_finalists_contract,
        shared_population_dir=args.shared_population_dir,
        family=str(args.family), arm=str(args.arm), target_winner_dir=args.target_winner_dir,
    )
    bundle = build_stage_i_adapter_winner_bundle(
        base_selection_dir=args.base_selection_dir,
        meta_selection_dir=args.meta_selection_dir,
        code_revision=args.code_revision,
        run_id=args.run_id,
        joint_finalist_authorization=authorization,
    )
    observed = {cell.base_target_contract.family for cell in bundle.cells}
    if observed != {FAMILY_TO_TARGET[str(args.family)]}:
        raise ValueError(f"winner bundle base family drift: expected {args.family}, got {sorted(observed)}")
    destination = args.output_dir / "winner_bundle.json"
    status = freeze_stage_i_adapter_winner_bundle(bundle, destination)
    print(json.dumps({
        "status": status, "winner_bundle": str(destination.resolve()),
        "winner_bundle_sha256": bundle.sha256, "family": args.family, "arm": args.arm,
        "shared_population_contract_sha256": authorization["shared_population_contract_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
