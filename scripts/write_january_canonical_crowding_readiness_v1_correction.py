#!/usr/bin/env python3
"""Seal a terminology correction for the January canonical-extension audit.

The v1 audit correctly failed closed on canonical score/economics lineage, but
its ``candidate_group_rows`` language overstated eligible-universe cardinality
as signal crowding.  This immutable sidecar withdraws that interpretation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "data_perp/artifacts/january_canonical_crowding_readiness_20260730_v1"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/january_canonical_crowding_readiness_20260730_v1_correction"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def correction_rows() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "v1_term": "candidate_group_rows_timestamp_side / candidate_group_size_bin_january_local",
            "corrected_term": "eligible_universe_cardinality_timestamp_side / January-local cardinality bucket",
            "authoritative_interpretation": "Number of eligible assets/candidates at a timestamp and side; not a measure of signal density, candidate overlap, or trading crowding.",
            "withdrawn_claim": "No high-crowding q2 or high-crowding/high-score support was quantified in v1.",
        },
        {
            "v1_term": "pre_outcome_crowding_support.csv",
            "corrected_term": "pre_outcome_universe_cardinality_only",
            "authoritative_interpretation": "Outcome-free identity/universe availability only. It cannot establish the February-to-March crowding estimand.",
            "withdrawn_claim": "January is not required merely to match a q2 asset-count/cardinality segment.",
        },
    ])


def run(output_dir: Path = DEFAULT_OUTPUT, v1: Path = V1) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output_dir}")
    manifest_path = v1 / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("sealed v1 readiness artifact is absent")
    prior = json.loads(manifest_path.read_text(encoding="utf-8"))
    if prior["status"] != "NOT_READY_FAIL_CLOSED_NO_CANONICAL_JANUARY_SCORE_BRIDGE":
        raise ValueError("correction may only attach to the expected fail-closed v1 artifact")
    stage = Path(tempfile.mkdtemp(dir=output_dir.parent, prefix=f".{output_dir.name}."))
    try:
        table = correction_rows()
        table.to_csv(stage / "terminology_correction.csv", index=False)
        manifest = {
            "schema": "january_canonical_crowding_readiness_v1_terminology_correction",
            "status": "CORRECTION_TO_V1_UNIVERSE_CARDINALITY_NOT_SIGNAL_CROWDING",
            "promotion_eligible": False,
            "supersedes_interpretation_only": True,
            "v1_readiness_findings_retained": {
                "canonical_score_stream_absent": True,
                "canonical_current_spread_exact_policy_h12_economics_absent": True,
                "no_historical_base_soft_oof_bridge": True,
            },
            "authoritative_scope": {
                "signal_density_or_crowding_support": "NOT_QUANTIFIED",
                "high_score_support": "NOT_QUANTIFIED because canonical January base_oof_score is absent",
                "january_requirement_for_q2_asset_count": "NOT_REQUIRED: universe cardinality is not the crowding estimand",
                "allowed_use_of_v1_cardinality_table": "identity/universe completeness only",
            },
            "source_v1": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
            "outputs_sha256": {"terminology_correction.csv": sha256(stage / "terminology_correction.csv")},
            "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        }
        write_json(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(f"{sha256(stage / 'manifest.json')}  manifest.json\n", encoding="utf-8")
        os.replace(stage, output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--v1", type=Path, default=V1)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.v1), sort_keys=True))
