#!/usr/bin/env python3
"""Seal a byte-preserving v4 wrapper that admits score-decile context only.

This is deliberately a contract-only materialization.  It never reads and
rewrites the v3 panel: the source ``panel.parquet`` is copied byte-for-byte.
The only changed output is the feature-role declaration that permits the
already-causal, score-only ``frozen_base_score_decile`` in a candidate-context
ablation.  In particular it does not make targets, realised exits, or the
duplicate score-decile group-size alias available to a model.
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

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import materialize_canonical_execution_reliability_input as v2
from scripts import materialize_canonical_execution_reliability_input_v3 as v3


SOURCE = ROOT / "data_perp/artifacts/canonical_execution_reliability_input_20260730_v3"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/canonical_execution_reliability_input_20260730_v4"
DECILE = "frozen_base_score_decile"
DECILE_ALIAS = "frozen_base_score_decile_group_rows"
SIDE_GROUP_SIZE = "base_group_rows_timestamp_side"


class ReliabilityV4Error(RuntimeError):
    """Raised when the frozen v3-to-v4 wrapper contract is broken."""


def verify(root: Path, schema: str) -> dict[str, Any]:
    """Verify a sealed artifact and all outputs named by its manifest."""

    manifest_path = root / "manifest.json"
    seal_path = root / "manifest.sha256"
    if not manifest_path.is_file() or not seal_path.is_file():
        raise ReliabilityV4Error(f"sealed artifact missing: {root}")
    if v2.sha256(manifest_path) != seal_path.read_text().split()[0]:
        raise ReliabilityV4Error(f"manifest seal mismatch: {root}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != schema:
        raise ReliabilityV4Error(f"schema mismatch for {root}: {manifest.get('schema')}")
    for name, expected in dict(manifest.get("outputs_sha256", {})).items():
        path = root / name
        if not path.is_file() or v2.sha256(path) != expected:
            raise ReliabilityV4Error(f"output hash mismatch: {path}")
    return manifest


def updated_roles(old: Mapping[str, Any]) -> dict[str, Any]:
    """Add exactly one causal score-decile candidate-context input."""

    roles = dict(old)
    default_inputs = list(roles["default_ev_inputs"])
    if DECILE_ALIAS in default_inputs:
        raise ReliabilityV4Error("duplicate score-decile group-size alias is already a feature")
    if DECILE not in default_inputs:
        default_inputs.append(DECILE)
    roles["default_ev_inputs"] = default_inputs

    existing_context = list(roles.get("candidate_context_inputs", []))
    # The v3 contract did not name this collection independently.  Populate it
    # from its causal context fields and retain the new field exactly once.
    for column in (
        "base_rank_pct_timestamp_side",
        "base_score_z_timestamp_side",
        SIDE_GROUP_SIZE,
        "base_margin_to_top40_cutoff",
        "base_margin_to_top40_cutoff_z",
        "base_rank_pct_timestamp_global",
        "base_score_z_timestamp_global",
        "base_group_rows_timestamp_global",
        DECILE,
    ):
        if column not in existing_context:
            existing_context.append(column)
    if DECILE_ALIAS in existing_context:
        raise ReliabilityV4Error("duplicate score-decile group-size alias escaped into context")
    roles["candidate_context_inputs"] = existing_context
    roles["candidate_context_contract"] = {
        "approved_rank_decile": DECILE,
        "availability": "deterministic contemporaneous side-local rank of frozen base OOF score; score/symbol/candidate-ID tie order only",
        "ablation_scope": "candidate-context only; not a target, exit, mapping coordinate, timing action, or policy feature",
        "duplicate_alias": {
            "column": DECILE_ALIAS,
            "equals": SIDE_GROUP_SIZE,
            "treatment": "documented only; excluded from default_ev_inputs and candidate_context_inputs to avoid a duplicate feature",
        },
    }
    targets = set(roles.get("target_only_never_features", []))
    prohibited = targets.intersection({DECILE, DECILE_ALIAS})
    if prohibited:
        raise ReliabilityV4Error(f"score context incorrectly declared target-only: {sorted(prohibited)}")
    if DECILE_ALIAS in roles["default_ev_inputs"]:
        raise ReliabilityV4Error("alias cannot be a default EV input")
    return roles


def verify_panel_contract(panel_path: Path) -> dict[str, int]:
    """Independently prove the v4-eligible context is already causal and unique."""

    panel = pd.read_parquet(panel_path, columns=[DECILE, DECILE_ALIAS, SIDE_GROUP_SIZE])
    if panel[DECILE].isna().any() or panel[DECILE_ALIAS].isna().any():
        raise ReliabilityV4Error("score-decile context is incomplete")
    if not panel[DECILE].between(0, 9).all():
        raise ReliabilityV4Error("score decile is outside its deterministic 0..9 contract")
    if not panel[DECILE_ALIAS].equals(panel[SIDE_GROUP_SIZE]):
        raise ReliabilityV4Error("documented score-decile group-size alias no longer equals side group size")
    return {
        "rows": int(len(panel)),
        "rank_decile_min": int(panel[DECILE].min()),
        "rank_decile_max": int(panel[DECILE].max()),
        "alias_equal_rows": int(len(panel)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    source_manifest = verify(args.source, "canonical_execution_reliability_input_v3")
    source_panel = args.source / "panel.parquet"
    source_roles = args.source / "feature_roles.json"
    source_capture_support = args.source / "capture_support.csv"
    for path in (source_panel, source_roles, source_capture_support):
        if not path.is_file():
            raise ReliabilityV4Error(f"required v3 output missing: {path}")
    panel_contract = verify_panel_contract(source_panel)
    roles = updated_roles(json.loads(source_roles.read_text()))

    stage = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent))
    try:
        # copy2 deliberately preserves the exact panel bytes and source metadata.
        shutil.copy2(source_panel, stage / "panel.parquet")
        shutil.copy2(source_capture_support, stage / "capture_support.csv")
        if v2.sha256(stage / "panel.parquet") != source_manifest["outputs_sha256"]["panel.parquet"]:
            raise ReliabilityV4Error("v4 panel is not byte-identical to v3")
        if v2.sha256(stage / "capture_support.csv") != source_manifest["outputs_sha256"]["capture_support.csv"]:
            raise ReliabilityV4Error("v4 capture support is not byte-identical to v3")
        v2.write_json(stage / "feature_roles.json", roles)
        outputs = {path.name: v2.sha256(path) for path in stage.iterdir() if path.is_file()}
        manifest = {
            "schema": "canonical_execution_reliability_input_v4",
            "run_id": args.output_dir.name,
            "status": "SEALED_RESEARCH_INPUT_BYTE_IDENTICAL_V3_PANEL_RANK_DECILE_CONTEXT_ONLY_NO_PROMOTION",
            "promotion_eligible": False,
            "rows": panel_contract["rows"],
            "input_sha256": {
                "source_manifest": v2.sha256(args.source / "manifest.json"),
                "source_panel": source_manifest["outputs_sha256"]["panel.parquet"],
                "source_feature_roles": source_manifest["outputs_sha256"]["feature_roles.json"],
                "source_capture_support": source_manifest["outputs_sha256"]["capture_support.csv"],
            },
            "byte_identity": {
                "panel.parquet": True,
                "capture_support.csv": True,
            },
            "rank_decile_context": panel_contract,
            "feature_contract": roles,
            "outputs_sha256": outputs,
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": v2.sha256(Path(__file__).resolve()),
            },
            "limitations": [
                "This wrapper changes only feature-role authorization; it does not create new predictions or promotion evidence.",
                "frozen_base_score_decile_group_rows is exactly base_group_rows_timestamp_side and remains excluded as a duplicate feature.",
                "Realized targets, exits, mapping coordinates, timing and wait actions remain excluded from model inputs.",
            ],
        }
        v2.write_json(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(v2.sha256(stage / "manifest.json") + "  manifest.json\n")
        os.replace(stage, args.output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--source", type=Path, default=SOURCE)
    command.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(v2.safe(run(args)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
