from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_label_side_coverage_audit import build_audit, _registry_reports


def _rows(*, sides: list[str]) -> pd.DataFrame:
    base_ts = pd.Timestamp("2026-04-01 12:00:00", tz="UTC")
    rows = []
    for idx, side in enumerate(sides):
        side_value = -1 if side == "short" else 1
        ts = base_ts + pd.Timedelta(hours=idx)
        rows.append(
            {
                "__ts__": ts,
                "__symbol__": "BTC",
                "side": side_value,
                "side_name": side,
                "__side__": side_value,
                "timeframe": "1h",
                "candidate_id": f"BTC|{ts.strftime('%Y-%m-%dT%H:%M:%SZ')}|1h|{side}",
                "candidate": f"candidate_{side}",
            }
        )
    return pd.DataFrame(rows)


def _write_bundle(root: Path, labels_dir: Path, artifact_root: Path, *, sides: list[str]) -> None:
    rows = _rows(sides=sides)
    labels_dir.mkdir(parents=True)
    rows.to_parquet(labels_dir / "labels.parquet", index=False)
    paths = [
        root / "utility_risk_gate_candidate_weekly/candidate_selected_rows.csv",
        root / "utility_path_risk_dual_head/source_utility_path_risk_dual_head_selected_rows.csv",
        root / "utility_path_timeout_joint_risk/source_utility_path_timeout_risk_selected_rows.csv",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(path, index=False)
    archetype = root / "source_archetypes_v2/candidate_source_archetypes_v2.parquet"
    archetype.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(archetype, index=False)
    registry = artifact_root / "run_a/strategy_registry/selected_single_head_strategy_registry.csv"
    registry.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "strategy_id": [f"{side}_strategy" for side in sides],
            "trade_side": sides,
            "side": sides,
        }
    ).to_csv(registry, index=False)


def test_side_coverage_audit_blocks_long_only_evidence(tmp_path: Path) -> None:
    root = tmp_path / "reports"
    labels = tmp_path / "labels"
    artifacts = tmp_path / "artifacts"
    out = tmp_path / "out"
    _write_bundle(root, labels, artifacts, sides=["long", "long"])

    result = build_audit(input_dir=root, labels_path=labels, artifact_root=artifacts, output_dir=out)

    assert result["decision"] == "long_only_or_missing_short_evidence"
    assert result["bidirectional_evidence_ready"] is False
    assert result["registry_summary"]["registries_with_short"] == 0
    assert any(row["role"] == "label_ledger" for row in result["blocking_artifacts"])


def test_side_coverage_audit_passes_bidirectional_evidence(tmp_path: Path) -> None:
    root = tmp_path / "reports"
    labels = tmp_path / "labels"
    artifacts = tmp_path / "artifacts"
    out = tmp_path / "out"
    _write_bundle(root, labels, artifacts, sides=["long", "short"])

    result = build_audit(input_dir=root, labels_path=labels, artifact_root=artifacts, output_dir=out)

    assert result["decision"] == "ready_bidirectional"
    assert result["bidirectional_evidence_ready"] is True
    assert result["blocking_artifacts"] == []
    assert result["registry_summary"]["registries_with_short"] == 1
    assert (out / "label_side_coverage_audit.md").exists()


def test_registry_reports_are_newest_first(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    old_path = artifact_root / "old_run/strategy_registry/selected_single_head_strategy_registry.csv"
    new_path = artifact_root / "new_run/strategy_registry/selected_single_head_strategy_registry.csv"
    for path, side in ((old_path, "long"), (new_path, "short")):
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"strategy_id": [f"{side}_strategy"], "trade_side": [side]}).to_csv(
            path,
            index=False,
        )
    old_time = 1_700_000_000
    new_time = old_time + 100
    old_path.touch()
    new_path.touch()
    import os

    os.utime(old_path, (old_time, old_time))
    os.utime(new_path, (new_time, new_time))

    reports = _registry_reports(artifact_root, max_reports=2)

    assert [row["role"] for row in reports] == [
        "strategy_registry:new_run",
        "strategy_registry:old_run",
    ]
