#!/usr/bin/env python3
"""Run the fail-closed F4 portable feature-contract selector."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.feature_portability_selection import (
    FeaturePortabilitySelectionError,
    FeaturePortabilitySelectionPolicy,
    _read_json,
    _read_table,
    completed_f0_f3_evidence,
    select_feature_portability_contract,
    validate_lineage_and_audit,
    write_feature_portability_selection_artifacts,
)


LINK_TRANSPORTS = (
    "transport_a_2023q4_to_2024h1",
    "transport_b_2024h1_to_2024h2_to_date",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--evidence", type=Path, help="Complete F4 development evidence parquet/csv.")
    mode.add_argument("--completed-root", type=Path, help="Completed F0--F3 merged artifact root.")
    parser.add_argument("--lineage", type=Path, help="Lineage JSON; required with --evidence.")
    parser.add_argument("--audit", type=Path, help="Reference-ready Stage-A coverage audit parquet; required with --evidence.")
    parser.add_argument(
        "--compact-contracts", type=Path,
        help="f4_compact_contracts.json from the exact evidence materialisation; required with --evidence.",
    )
    parser.add_argument("--stage-a-root", type=Path, help="Stage-A artifact root; required with --completed-root.")
    parser.add_argument("--mda", type=Path, help="Optional completed chronological grouped-MDA evidence parquet/csv.")
    parser.add_argument("--output", required=True, type=Path, help="New immutable F4 output directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # The linked roadmap names these two development transports explicitly.
    # Generic evidence remains reusable for isolated unit tests, but the
    # concrete project entrypoint cannot silently accept another two-period
    # pair and call it a cross-era F4 decision.
    policy = FeaturePortabilitySelectionPolicy(
        required_transports=LINK_TRANSPORTS,
        required_representation_prefix="F4_compact_top",
        require_nonnegative_f3_control_lift=True,
    )
    inputs: dict[str, Path] = {}
    compact_contracts: dict[str, object] | None = None
    try:
        if args.evidence:
            if not args.lineage or not args.audit or not args.compact_contracts:
                raise FeaturePortabilitySelectionError("--evidence requires --lineage, --audit, and --compact-contracts")
            raw = _read_table(args.evidence)
            lineage = _read_json(args.lineage)
            audit = _read_table(args.audit)
            candidate_contracts = _read_json(args.compact_contracts)
            if not isinstance(candidate_contracts, dict):
                raise FeaturePortabilitySelectionError("--compact-contracts must be an F4 compact-contract JSON object")
            compact_contracts = candidate_contracts
            evidence = validate_lineage_and_audit(raw, lineage, audit, policy=policy)
            inputs = {
                "evidence": args.evidence, "lineage": args.lineage, "audit": args.audit,
                "compact_contracts": args.compact_contracts,
            }
        else:
            if not args.stage_a_root:
                raise FeaturePortabilitySelectionError("--completed-root requires --stage-a-root")
            root = args.completed_root
            stage_a = args.stage_a_root
            result_path = root / "base_feature_ablation_results.parquet"
            lineage_path = root / "base_feature_arm_lineage.json"
            manifest_path = root / "run_manifest.json"
            audit_path = stage_a / "feature_portability_era_audit.parquet"
            mda = _read_table(args.mda) if args.mda else None
            evidence = completed_f0_f3_evidence(
                _read_table(result_path), _read_json(lineage_path), _read_table(audit_path),
                result_manifest=_read_json(manifest_path), mda=mda, policy=policy,
            )
            inputs = {"results": result_path, "lineage": lineage_path, "audit": audit_path, "manifest": manifest_path}
            if args.mda:
                inputs["mda"] = args.mda
        result = select_feature_portability_contract(evidence, policy=policy)
        paths = write_feature_portability_selection_artifacts(
            result, args.output, input_paths=inputs, compact_contracts=compact_contracts,
        )
    except (FeaturePortabilitySelectionError, FileNotFoundError, ValueError) as exc:
        print(f"F4 selector failed closed: {exc}", file=sys.stderr)
        return 2
    emitted = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    print(json.dumps({"status": emitted["status"], "output": str(args.output), "manifest": str(paths["manifest"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
