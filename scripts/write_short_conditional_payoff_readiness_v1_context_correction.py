#!/usr/bin/env python3
"""Correct the aggregate compact-context coverage wording in the sealed v1 audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "data_perp/artifacts/short_conditional_payoff_readiness_20260730_v1"
OUTPUT = ROOT / "data_perp/artifacts/short_conditional_payoff_readiness_20260730_v1_context_coverage_correction"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(output_dir: Path = OUTPUT, v1: Path = V1) -> dict:
    if output_dir.exists(): raise FileExistsError(f"refusing to overwrite immutable output {output_dir}")
    inventory_path = v1 / "feature_inventory.csv"; manifest_path = v1 / "manifest.json"
    if not inventory_path.is_file() or not manifest_path.is_file(): raise FileNotFoundError("v1 audit inputs absent")
    inventory = pd.read_csv(inventory_path)
    context = inventory.loc[inventory.family.eq("causal_context")].groupby("field", as_index=False).agg(min_finite_fraction=("finite_fraction", "min"), max_missing_rows=("finite_rows", lambda x: 0))
    # Missing rows are derived from the original per-slice counts, rather than
    # treating a fully observed later slice as proof of global completeness.
    missing = inventory.loc[inventory.family.eq("causal_context")].assign(missing=lambda x: x.rows-x.finite_rows).groupby("field", as_index=False).agg(min_finite_fraction=("finite_fraction", "min"), total_missing_rows=("missing", "sum"))
    missing["coverage_status"] = missing.min_finite_fraction.eq(1.0).map({True: "FULL_ALL_MONTH_SIDE_SLICES", False: "PARTIAL_EXCLUDE_UNLESS_MISSINGNESS_RULE_PREDECLARED"})
    stage = Path(tempfile.mkdtemp(dir=output_dir.parent, prefix=f".{output_dir.name}."))
    try:
        missing.to_csv(stage / "corrected_context_coverage.csv", index=False)
        full = sorted(missing.loc[missing.coverage_status.eq("FULL_ALL_MONTH_SIDE_SLICES"), "field"])
        partial = sorted(missing.loc[missing.coverage_status.ne("FULL_ALL_MONTH_SIDE_SLICES"), "field"])
        payload = {
            "schema": "short_conditional_payoff_readiness_v1_context_coverage_correction",
            "status": "CORRECTION_OPTIONAL_CONTEXT_COVERAGE_ONLY",
            "promotion_eligible": False,
            "source_v1": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
            "correction": {"error": "v1 derived full-context membership from any complete slice rather than all month/side slices", "full_coverage_joinable": full, "partial_coverage_exclude_without_predeclared_missingness_rule": partial, "smallest_recommended_sets_unchanged": True, "reason": "all optional compact contexts were excluded from those sets"},
            "outputs_sha256": {"corrected_context_coverage.csv": sha256(stage / "corrected_context_coverage.csv")},
            "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        }
        target = stage / "manifest.json"; target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (stage / "manifest.sha256").write_text(f"{sha256(target)}  manifest.json\n", encoding="utf-8")
        os.replace(stage, output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True); raise
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-dir", type=Path, default=OUTPUT); parser.add_argument("--v1", type=Path, default=V1)
    print(json.dumps(run(parser.parse_args().output_dir, parser.parse_args().v1), sort_keys=True))
