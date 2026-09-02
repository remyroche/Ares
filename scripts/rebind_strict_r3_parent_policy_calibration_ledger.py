#!/usr/bin/env python3
"""Rebind a verified resolved score ledger to its recoverable parent policy.

The historical source ledger is byte-preserved.  Its original policy JSON was
not retained, but its three economic policy values are recorded in the
source-aligned outcome manifest that produced every label.  This creates an
explicit, versioned semantic reconstruction for runtime calibration only; it
does not refit a model, recompute an outcome, or alter any ledger row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_LEDGER = ROOT / (
    "data_perp/artifacts/strict_r3_exact15m_entry_resolved_prefix_"
    "homogeneous28_exact170_long_aug1_13_20260813_v1/"
    "walkforward_scored_label_ledger.parquet"
)
SOURCE_OUTCOMES_MANIFEST = ROOT / (
    "data_perp/artifacts/strict_r3_source_aligned_optimized_policy_outcomes_"
    "long_2024jan_jul2026_20260812_v1/run_manifest.json"
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    if not SOURCE_LEDGER.is_file() or not SOURCE_OUTCOMES_MANIFEST.is_file():
        raise FileNotFoundError("source ledger or source-aligned outcome manifest is absent")

    source = json.loads(SOURCE_OUTCOMES_MANIFEST.read_text())
    policy = dict(source.get("policy") or {})
    required = ("sl_mult", "trailing_activation_mult", "fixed_trailing_gap_mult")
    if any(key not in policy for key in required):
        raise ValueError("source outcome manifest lacks frozen parent-policy values")
    if float(source.get("cost_bps_once", float("nan"))) != 100.0:
        raise ValueError("source outcome manifest does not prove the one-time 100-bps cost")

    # This is a semantic reconstruction, deliberately not a claim that the
    # missing original JSON bytes were recovered.  The ledger's original
    # provenance hash is preserved separately in the manifest.
    parent_policy = {
        "schema": "strict_r3_source_aligned_parent_policy_semantic_rebind_v1",
        "side": "long",
        "winner": {key: float(policy[key]) for key in required},
        "timeout_hours": 12,
        "cost_bps_once": 100.0,
        "entry": "first available bar open at signal close + one hour",
        "source_outcome_manifest": str(SOURCE_OUTCOMES_MANIFEST.relative_to(ROOT)),
        "source_outcome_policy_json_sha256": source.get("policy_json_sha256"),
        "semantics": "exact three-value parent-policy reconstruction; original JSON bytes unavailable",
    }
    out.mkdir(parents=True)
    policy_path = out / "parent_policy_semantic_rebind.json"
    policy_path.write_text(json.dumps(parent_policy, indent=2, sort_keys=True) + "\n")
    ledger_path = out / SOURCE_LEDGER.name
    shutil.copyfile(SOURCE_LEDGER, ledger_path)
    if _sha(ledger_path) != _sha(SOURCE_LEDGER):
        raise AssertionError("rebound ledger is not byte-identical to its source")
    rows = int(len(pd.read_parquet(ledger_path, columns=["candidate_id"])))
    manifest = {
        "schema": "strict_r3_parent_policy_calibration_ledger_rebind_v1",
        "status": "complete",
        "operation": "byte_preserving_ledger_rebind_no_model_or_label_recompute",
        "source_ledger": str(SOURCE_LEDGER.relative_to(ROOT)),
        "source_ledger_sha256": _sha(SOURCE_LEDGER),
        "rebound_ledger": ledger_path.name,
        "rebound_ledger_sha256": _sha(ledger_path),
        "rows": rows,
        "source_outcomes_manifest": str(SOURCE_OUTCOMES_MANIFEST.relative_to(ROOT)),
        "source_outcomes_manifest_sha256": _sha(SOURCE_OUTCOMES_MANIFEST),
        "original_policy_json_sha256_recorded_by_source": source.get("policy_json_sha256"),
        "parent_policy_semantic_rebind": policy_path.name,
        "parent_policy_semantic_rebind_sha256": _sha(policy_path),
        "semantic_policy": parent_policy["winner"],
        "cost_bps_once": 100.0,
        "prohibitions": ["no_model_refit", "no_label_recompute", "no_exchange_io", "no_order_submission"],
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
