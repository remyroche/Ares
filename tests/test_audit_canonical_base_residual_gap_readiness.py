import json
from pathlib import Path

import pandas as pd

from scripts.audit_canonical_base_residual_gap_readiness import (
    build_gap_readiness,
    materialize_gap_readiness,
)


def _fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    labels = root / "labels"
    labels.mkdir()
    entries = []
    for month in ("2025-01", "2025-02"):
        for side in ("long", "short"):
            filename = f"train_global_{side}_5_{month.replace('-', '_')}.parquet"
            timestamp = pd.Timestamp(f"{month}-01T00:00:00Z")
            data = pd.DataFrame({
                "candidate_id": [f"{side}-{month}"], "__ts__": [timestamp], "__decision_ts__": [timestamp + pd.Timedelta(hours=1)],
                "side_name": [side], "__first_touch_target_soft__": [0.5], "__first_touch_capture_net__": [0.01], "__first_touch_valid_path__": [1],
                **{f"feature_{number}": [float(number)] for number in range(20)},
            })
            data.to_parquet(labels / filename, index=False)
            entries.append({"file": filename, "month": month, "side_from_filename": side, "rows": 1, "expected_current_rows": 1})
    inventory = root / "inventory.json"
    inventory.write_text(json.dumps({"status": "COMPLETE", "per_file": entries}), encoding="utf-8")
    oof = root / "feb.parquet"
    pd.DataFrame({
        "candidate_id": ["long-2025-02", "short-2025-02"],
        "__ts__": [pd.Timestamp("2025-02-01T00:00:00Z")] * 2,
        "__decision_ts__": [pd.Timestamp("2025-02-01T01:00:00Z")] * 2,
        "side_name": ["long", "short"], "base_oof_score": [0.4, 0.6], "residual_is_oof": [False, False],
    }).to_parquet(oof, index=False)
    gate = root / "gate.json"
    gate.write_text(json.dumps({"status": "FEBRUARY_WARMUP"}), encoding="utf-8")
    return inventory, labels, oof, gate


def test_marks_base_only_february_without_promoting_it_to_residual(tmp_path: Path) -> None:
    inventory, labels, oof, _ = _fixture(tmp_path)
    ledger, feb = build_gap_readiness(inventory_path=inventory, label_dir=labels, febapr_oof_path=oof)
    feb_base = ledger.loc[(ledger.month == "2025-02") & ledger.stage.eq("canonical_base_oof_score")].iloc[0]
    feb_residual = ledger.loc[(ledger.month == "2025-02") & ledger.stage.eq("canonical_residual_oof_score")].iloc[0]
    assert feb_base.status == "READY" and len(feb) == 2
    assert feb_residual.status == "BLOCKED"
    assert "warmup" in feb_residual.reason


def test_missing_gap_months_are_explicit_blockers_not_silently_skipped(tmp_path: Path) -> None:
    inventory, labels, oof, _ = _fixture(tmp_path)
    ledger, _ = build_gap_readiness(inventory_path=inventory, label_dir=labels, febapr_oof_path=oof)
    jan = ledger.loc[ledger.month.eq("2025-01")]
    assert len(jan) == 5
    assert jan.loc[jan.stage.eq("canonical_base_oof_score"), "status"].iloc[0] == "BLOCKED"
    assert jan.loc[jan.stage.eq("candidate_feature_base_label"), "status"].eq("READY").all()


def test_materialized_ledger_has_detached_manifest_checksum(tmp_path: Path) -> None:
    inventory, labels, oof, gate = _fixture(tmp_path)
    output = tmp_path / "out"
    manifest = materialize_gap_readiness(inventory_path=inventory, label_dir=labels, febapr_oof_path=oof, febapr_gate_path=gate, output_dir=output)
    ledger = pd.read_csv(output / "canonical_gap_stage_ledger.csv")
    assert manifest["promotion_eligible"] is False
    assert (output / "manifest.sha256").exists()
    assert (output / "february_2025_base_oof_warmup.parquet").exists()
    assert ledger["status"].eq("BLOCKED").any()
