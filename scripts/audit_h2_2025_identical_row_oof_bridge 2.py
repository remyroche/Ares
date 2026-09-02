#!/usr/bin/env python3
"""Fail-closed audit for a pre-2026 H2-2025 identical-row OOF bridge."""
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
OUT = ART / "h2_2025_identical_row_oof_bridge_audit_20260730_v3"
LEDGER = ART / "frozen_contextual_score_arms_2023apr_2025jun_20260730_v1/blocked_oof_training_panel.parquet"
DIRECT = ART / "febjul2025_execution_ev_common30_two_layer_oof_20260727_v3/two_layer_direct_ev_strict_oof.parquet"
LABELS = {
    "2025-07": ART / "mayjul2025_execution_ev_common30_labels_20260727_v2/labels.parquet",
    "2025-08_to_10": ART / "augoct2025_execution_ev_common30_labels_20260727_v1/labels.parquet",
    "2025-11": ART / "nov2025_execution_ev_common30_labels_20260727_v1/labels.parquet",
}
REQUIRED = {"candidate_id", "__ts__", "__symbol__", "side_name", "execution_label_end_utc", "execution_net_ev_12h", "execution_gross_ev_12h", "execution_cost_return", "__first_touch_target_soft__", "score_base_alpha", "score_residual_expected_ev"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(output: Path = OUT) -> Path:
    if output.exists():
        raise FileExistsError(output)
    ledger = pd.read_parquet(LEDGER)
    ledger["__ts__"] = pd.to_datetime(ledger["__ts__"], utc=True)
    ledger["execution_label_end_utc"] = pd.to_datetime(ledger["execution_label_end_utc"], utc=True)
    ledger["execution_label_available_at"] = pd.to_datetime(ledger["execution_label_available_at"], utc=True)
    resolved_at = ledger["execution_label_end_utc"].combine_first(ledger["execution_label_available_at"])
    alias_pairs = (("score_base_alpha", "base_oof_score"), ("score_residual_expected_ev", "residual_expected_ev"))
    alias_audit = []
    for canonical, alias in alias_pairs:
        left, right = pd.to_numeric(ledger[canonical], errors="coerce"), pd.to_numeric(ledger[alias], errors="coerce")
        overlap = left.notna() & right.notna()
        conflicts = overlap & ~left.eq(right)
        alias_audit.append({"canonical": canonical, "alias": alias, "canonical_present": int(left.notna().sum()), "alias_present": int(right.notna().sum()), "coalesced_present": int(left.combine_first(right).notna().sum()), "overlap_rows": int(overlap.sum()), "conflict_rows": int(conflicts.sum())})
    ledger_checks = {
        "hourly_decision_rows": bool((ledger.__ts__.dt.minute.eq(0) & ledger.__ts__.dt.second.eq(0)).all()),
        "execution_label_end_present_rows": int(ledger.execution_label_end_utc.notna().sum()),
        "execution_label_available_at_present_rows": int(ledger.execution_label_available_at.notna().sum()),
        "resolved_label_timestamp_rows_after_coalesce": int(resolved_at.notna().sum()),
        "resolved_label_timestamp_unresolved_rows": int(resolved_at.isna().sum()),
        "resolved_label_timestamp_before_or_at_decision_rows": int((resolved_at <= ledger.__ts__).sum()),
        "missing_final_runner_fields": sorted(REQUIRED - set(ledger.columns)),
        "end_utc": str(ledger.__ts__.max()),
        "h2_rows": int(ledger.__ts__.ge(pd.Timestamp("2025-07-01", tz="UTC")).sum()),
    }
    sources: list[dict[str, object]] = []
    for name, path in {"final_identical_row_ledger": LEDGER, "h2_direct_oof": DIRECT, **{f"exact_labels_{key}": value for key, value in LABELS.items()}}.items():
        frame = pd.read_parquet(path)
        timestamp = "__ts__"
        frame[timestamp] = pd.to_datetime(frame[timestamp], utc=True)
        cols = set(frame.columns)
        row = {"source": name, "path": str(path), "sha256": sha(path), "rows": len(frame), "start_utc": str(frame[timestamp].min()), "end_utc": str(frame[timestamp].max()), "hourly_decision_rows": bool((frame[timestamp].dt.minute.eq(0) & frame[timestamp].dt.second.eq(0)).all()), "has_exact_economics": {"execution_net_ev_12h", "execution_gross_ev_12h", "execution_cost_return"}.issubset(cols), "has_final_score_pair": {"score_base_alpha", "score_residual_expected_ev"}.issubset(cols), "has_label_resolution": "execution_label_end_utc" in cols, "missing_final_runner_fields": ";".join(sorted(REQUIRED - cols))}
        sources.append(row)
    direct = pd.read_parquet(DIRECT)
    direct["__ts__"] = pd.to_datetime(direct["__ts__"], utc=True)
    july = direct.loc[direct.__ts__.dt.strftime("%Y-%m").eq("2025-07")]
    direct_check = {"july_rows": len(july), "h2_direct_oof_end_utc": str(direct.__ts__.max()), "is_common30_not_identical_row": True, "missing_base_residual_pair": sorted({"score_base_alpha", "score_residual_expected_ev"} - set(direct.columns)), "missing_first_touch_target": "__first_touch_target_soft__" not in direct}
    blockers = [
        "authoritative final identical-row OOF ledger ends 2025-06-30; it contains zero H2 rows",
        "the only July direct-OOF surface is a 30-symbol common-universe direct-EV score, not the final base+residual score pair",
        "August-November have exact 1m-derived economics labels but no compatible candidate OOF base/residual scores or strict score-fit provenance",
        "therefore no H2 rows can be appended, substituted or used to refresh the frozen map without breaking identical-row lineage",
    ]
    stage = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    try:
        pd.DataFrame(sources).to_csv(stage / "source_compatibility_ledger.csv", index=False)
        pd.DataFrame(alias_audit).to_csv(stage / "historical_score_alias_audit.csv", index=False)
        (stage / "missing_data_report.json").write_text(json.dumps({"status": "FAIL_CLOSED_INCOMPATIBLE_H2_OOF_LINEAGE", "model_sample_cadence": "1h", "exact_replay_bar_cadence": "1m_labels_only", "ledger_checks": ledger_checks, "historical_score_alias_audit": alias_audit, "direct_oof_check": direct_check, "blockers": blockers, "required_materialization": "side-local strict blocked-OOF score_base_alpha and score_residual_expected_ev on the final candidate identity, with exact gross/cost/net and resolved label timestamps, for July-November 2025; December remains incomplete", "promotion_eligible": False}, indent=2, sort_keys=True) + "\n")
        files = [path for path in stage.iterdir() if path.is_file()]
        manifest = {"schema": "h2_2025_identical_row_oof_bridge_audit_v3", "status": "SEALED_FAIL_CLOSED_MISSING_DATA_REPORT", "model_sample_cadence": "1h", "assessment_sample_cadence": "1h", "exact_replay_bar_cadence": "1m_labels_only", "no_2026_fit_or_map_labels_used": True, "promotion_eligible": False, "inputs": {str(path): sha(path) for path in [LEDGER, DIRECT, *LABELS.values()]}, "outputs_sha256": {path.name: sha(path) for path in files}}
        manifest_path = stage / "manifest.json"; manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n"); (stage / "manifest.sha256").write_text(f"{sha(manifest_path)}  manifest.json\n"); os.replace(stage, output)
        return output
    except Exception:
        shutil.rmtree(stage, ignore_errors=True); raise


if __name__ == "__main__":
    print(audit())
