#!/usr/bin/env python3
"""H2 readiness v6: December common30 bridge is complete; context is partial."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data_perp/artifacts"
JULY = ART / "july2025_common30_final_base_residual_oof_bridge_20260730_v1"
AUGNOV = ART / "augnov2025_common30_frozen_july_base_residual_oos_bridge_20260730_v1"
DEC = ART / "dec2025_common30_frozen_august_base_residual_oos_bridge_20260730_v1"
CONTEXT = ART / "dec2025_common30_fixed_preaug_context_oos_extension_20260730_v1"
OUT = ART / "h2_2025_identical_row_oof_bridge_audit_20260730_v6"


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
        raise RuntimeError(f"unsealed {root}")
    value = json.loads(manifest.read_text())
    if value.get("schema") != schema or value.get("status") != status:
        raise RuntimeError(f"wrong contract {root}")
    return value


def run(output: Path = OUT) -> Path:
    output = Path(output)
    if output.exists():
        raise RuntimeError(output)
    july_manifest = _sealed(JULY, "july2025_common30_final_base_residual_oof_bridge_v1", "SEALED_COMMON30_BLOCKED_OOF_BRIDGE_NON_PROMOTION")
    augnov_manifest = _sealed(AUGNOV, "augnov2025_common30_frozen_july_base_residual_oos_bridge_v1", "SEALED_COMMON30_FROZEN_JULY_OOS_SCORE_BRIDGE_NON_PROMOTION")
    dec_manifest = _sealed(DEC, "dec2025_common30_frozen_august_base_residual_oos_bridge_v1", "SEALED_COMMON30_FROZEN_PRE_DECEMBER_BASE_RESIDUAL_OOS_SCORE_BRIDGE_NON_PROMOTION")
    context_manifest = _sealed(CONTEXT, "dec2025_common30_fixed_preaug_context_oos_extension_v1", "SEALED_FIXED_PREDEC_CONTEXT_OOS_EXTENSION_NON_PROMOTION")
    july_path, augnov_path, dec_path = JULY / "oof_predictions.parquet", AUGNOV / "oos_predictions.parquet", DEC / "oos_predictions.parquet"
    july_contract = json.loads((JULY / "bridge_contract.json").read_text())
    if july_contract.get("outputs", {}).get(july_path.name) != _sha(july_path):
        raise RuntimeError("July companion contract does not bind current OOF file")
    if augnov_manifest["outputs_sha256"].get(augnov_path.name) != _sha(augnov_path) or dec_manifest["outputs_sha256"].get(dec_path.name) != _sha(dec_path):
        raise RuntimeError("OOS bridge checksum mismatch")
    july = pd.read_parquet(july_path, columns=["candidate_id", "__ts__", "execution_label_end_utc", "residual_is_oof"])
    augnov = pd.read_parquet(augnov_path, columns=["candidate_id", "__ts__", "execution_label_end_utc", "residual_is_oos"])
    dec = pd.read_parquet(dec_path, columns=["candidate_id", "__ts__", "execution_label_end_utc", "execution_label_available_at", "residual_is_oos"])
    for frame in (july, augnov, dec):
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
        frame["execution_label_end_utc"] = pd.to_datetime(frame["execution_label_end_utc"], utc=True, errors="raise")
    if (len(july) != 44_640 or len(augnov) != 175_680 or len(dec) != 44_640 or any(frame.candidate_id.duplicated().any() for frame in (july, augnov, dec))
            or not july.residual_is_oof.all() or not augnov.residual_is_oos.all() or not dec.residual_is_oos.all()):
        raise RuntimeError("H2 bridge coverage or score lineage is invalid")
    unavailable = pd.read_parquet(CONTEXT / "context_unavailable_candidates.parquet")
    summary = pd.read_csv(CONTEXT / "metrics_summary.csv")
    coverage = context_manifest["contract"]["context_coverage"]
    if len(unavailable) != 720 or coverage != {"input_rows": 44_640, "context_scored_rows": 43_920, "excluded_rows": 720, "excluded_hourly_timestamps": 12, "missingness_policy": "no fill/no forward fill/no reroute; excluded candidate IDs are emitted explicitly"}:
        raise RuntimeError("December context boundary changed")
    source = pd.DataFrame([
        {"period": "2025-07", "rows": len(july), "availability": "sealed_common30_blocked_oof", "score_lineage": "strict blocked OOF", "candidate_scope": "common30", "context_rows": None},
        {"period": "2025-08_to_2025-11", "rows": len(augnov), "availability": "sealed_common30_frozen_preaug_oos", "score_lineage": "frozen pre-August OOS", "candidate_scope": "common30", "context_rows": None},
        {"period": "2025-12", "rows": len(dec), "availability": "sealed_common30_frozen_predec_oos", "score_lineage": "immutable pre-August OOS", "candidate_scope": "common30", "context_rows": 43_920},
    ])
    stage = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    try:
        source.to_csv(stage / "source_compatibility_ledger.csv", index=False)
        summary.to_csv(stage / "december_context_economics_summary.csv", index=False)
        report = {
            "status": "JULY_AUGNOV_AND_DECEMBER_COMMON30_BRIDGES_AVAILABLE_CONTEXT_PARTIAL",
            "promotion_eligible": False, "decision_cadence": "1h", "exact_replay_bar_cadence": "1m_labels_only",
            "supersedes": "h2_2025_identical_row_oof_bridge_audit_20260730_v5",
            "available": {"july_rows": int(len(july)), "augnov_rows": int(len(augnov)), "december_rows": int(len(dec)),
                          "label_end_max_utc": max(augnov.execution_label_end_utc.max(), dec.execution_label_end_utc.max()),
                          "scope": "all bridges are common30, not population-identical v3 extensions"},
            "december_context": {**coverage, "unavailable_candidate_ledger": "context_unavailable_candidates.parquet",
                                  "verdict": "context arms are valid only on the explicit common subset; no full-December context conclusion"},
            "lineage_note": "July parent manifest has a stale oof_predictions output hash; its matching sealed companion bridge_contract.json is the binding used by the established Aug-Nov extension and this audit. No parent artifact was modified.",
            "remaining_blockers": ["common30 scope prevents treating H2 as a full v3 identical-row replacement or promotion evidence", "December regime/transition context lacks its final 12 hourly timestamps; no fill, forward fill, or reroute is authorized", "no causal EV map may be refreshed from these population-mismatched H2 bridges for promotion"],
            "authorized_use": "non-promotional common30 H2 sensitivity, regime/transition diagnostics, and explicitly common-subset raw-score economics only; models/maps must retain population-mismatch and partial-context provenance",
        }
        _dump(stage / "readiness_report.json", report)
        files = [path for path in stage.iterdir() if path.is_file()]
        manifest = {"schema": "h2_2025_identical_row_oof_bridge_audit_v6", "status": "SEALED_H2_COMMON30_READINESS_AUDIT_NON_PROMOTION",
                    "promotion_eligible": False, "decision_cadence": "1h", "exact_replay_bar_cadence": "1m_labels_only",
                    "inputs": {str(path.resolve()): _sha(path) for path in (JULY / "manifest.json", JULY / "bridge_contract.json", AUGNOV / "manifest.json", DEC / "manifest.json", CONTEXT / "manifest.json")},
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
