#!/usr/bin/env python3
"""H2 readiness v7: final-12h raw inputs exist but frozen context append fails closed."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data_perp/artifacts"
V6 = ART / "h2_2025_identical_row_oof_bridge_audit_20260730_v6"
EXT = ART / "dec2025_final12h_frozen_predec_regime_transition_context_extension_20260730_v1"
OUT = ART / "h2_2025_identical_row_oof_bridge_audit_20260730_v7"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _dump(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def _sealed(root: Path, schema: str, status: str) -> dict:
    manifest = root / "manifest.json"
    marker = root / "manifest.sha256"
    if not manifest.is_file() or not marker.is_file() or marker.read_text().split()[0] != _sha(manifest):
        raise RuntimeError(f"unsealed: {root}")
    value = json.loads(manifest.read_text())
    if value.get("schema") != schema or value.get("status") != status:
        raise RuntimeError(f"wrong contract: {root}")
    return value


def run(output: Path = OUT) -> Path:
    output = Path(output)
    if output.exists():
        raise RuntimeError(output)
    v6 = _sealed(V6, "h2_2025_identical_row_oof_bridge_audit_v6", "SEALED_H2_COMMON30_READINESS_AUDIT_NON_PROMOTION")
    extension = _sealed(EXT, "dec2025_final12h_frozen_predec_regime_transition_context_extension_v1", "SEALED_FAIL_CLOSED_FROZEN_CONTEXT_REPRODUCTION_MISMATCH")
    parent = json.loads((V6 / "readiness_report.json").read_text())
    mismatch = json.loads((EXT / "readiness_report.json").read_text())
    transition = mismatch["overlap_validation"]["transition"]
    if mismatch["overlap_validation"]["regime"]["max_abs"] != 0.0 or transition["max_abs"] <= 1e-8:
        raise RuntimeError("unexpected final12 mismatch proof")
    stage = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    try:
        report = {
            "status": "H2_COMMON30_BRIDGES_COMPLETE_CONTEXT_FINAL12_FAIL_CLOSED",
            "promotion_eligible": False, "decision_cadence": "1h", "exact_replay_bar_cadence": "1m_labels_only",
            "supersedes": "h2_2025_identical_row_oof_bridge_audit_20260730_v6",
            "unchanged_valid_context_sensitivity": parent["december_context"],
            "final12_investigation": {
                "raw_input_coverage": mismatch["raw_input_availability"],
                "regime_overlap_max_abs": mismatch["overlap_validation"]["regime"]["max_abs"],
                "transition_overlap_max_abs": transition["max_abs"],
                "largest_transition_divergence": "BOCPD stable-vs-transition margin 0.0282629058; probability 0.0141314529",
                "verdict": "withheld; no context row, imputation, forward fill, or reroute is permitted",
                "required_remediation": mismatch["safe_remediation"],
            },
            "remaining_blockers": [*parent["remaining_blockers"], "final twelve December context hours cannot be appended until exact frozen LGBM/BOCPD model state is recovered or exactly reproduced"],
            "authorized_use": "the sealed 43,920-row common-context sensitivity remains diagnostic only; no 44,640-row context-arm, mapping, promotion, or replay conclusion is authorized",
        }
        _dump(stage / "readiness_report.json", report)
        (stage / "source_compatibility_ledger.csv").write_text((V6 / "source_compatibility_ledger.csv").read_text())
        files = [path for path in stage.iterdir() if path.is_file()]
        manifest = {"schema": "h2_2025_identical_row_oof_bridge_audit_v7", "status": "SEALED_H2_COMMON30_READINESS_AUDIT_NON_PROMOTION",
                    "promotion_eligible": False, "decision_cadence": "1h", "exact_replay_bar_cadence": "1m_labels_only",
                    "inputs": {str(path.resolve()): _sha(path) for path in (V6 / "manifest.json", EXT / "manifest.json", EXT / "readiness_report.json")},
                    "outputs_sha256": {path.name: _sha(path) for path in files}}
        _dump(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(f"{_sha(stage / 'manifest.json')}  manifest.json\n")
        os.replace(stage, output)
        return output
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == "__main__":
    print(run())
