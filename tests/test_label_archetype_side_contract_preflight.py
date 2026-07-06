from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_label_archetype_side_contract_preflight import build_preflight


def _side_rows() -> pd.DataFrame:
    ts = pd.Timestamp("2026-04-01 12:00:00", tz="UTC")
    return pd.DataFrame(
        {
            "__ts__": [ts, ts],
            "__symbol__": ["BTC", "BTC"],
            "side": [1, -1],
            "side_name": ["long", "short"],
            "timeframe": ["1h", "1h"],
            "candidate_id": [
                "BTC|2026-04-01T12:00:00Z|1h|long",
                "BTC|2026-04-01T12:00:00Z|1h|short",
            ],
            "candidate": ["c_long", "c_short"],
            "primary_source_tag": ["risk", "compression"],
            "u_policy_net": [0.01, -0.01],
        }
    )


def _write_side_aware_root(root: Path) -> None:
    rows = _side_rows()
    candidate_dir = root / "utility_risk_gate_candidate_weekly"
    path_dir = root / "utility_path_risk_dual_head"
    joint_dir = root / "utility_path_timeout_joint_risk"
    archetype_dir = root / "source_archetypes_v2"
    for directory in (candidate_dir, path_dir, joint_dir, archetype_dir):
        directory.mkdir(parents=True)
    rows.to_csv(candidate_dir / "candidate_selected_rows.csv", index=False)
    rows.to_csv(path_dir / "source_utility_path_risk_dual_head_selected_rows.csv", index=False)
    rows.to_csv(joint_dir / "source_utility_path_timeout_risk_selected_rows.csv", index=False)
    rows.to_parquet(archetype_dir / "candidate_source_archetypes_v2.parquet", index=False)
    pd.DataFrame(
        {
            "archetype": ["path_geometry_archetype"],
            "top_side_share": [0.5],
            "top_symbol_share": [1.0],
        }
    ).to_csv(archetype_dir / "source_archetypes_v2_scorecard.csv", index=False)
    (archetype_dir / "manifest.json").write_text(
        json.dumps(
            {
                "join_report": {"join_mode": "candidate_id"},
                "side_counts_full": {"long": 1, "short": 1},
                "side_counts_joined": {"long": 1, "short": 1},
            }
        ),
        encoding="utf-8",
    )


def test_side_contract_preflight_ready_for_refreshed_side_aware_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "reports"
    out = tmp_path / "out"
    _write_side_aware_root(root)

    result = build_preflight(input_dir=root, output_dir=out)

    assert result["ready_for_side_aware_stage0"] is True
    assert result["decision"] == "ready"
    assert result["blocking_artifacts"] == []
    assert (out / "label_archetype_side_contract_preflight.json").exists()
    assert (out / "label_archetype_side_contract_preflight.md").exists()


def test_side_contract_preflight_blocks_side_blind_selected_rows(tmp_path: Path) -> None:
    root = tmp_path / "reports"
    out = tmp_path / "out"
    _write_side_aware_root(root)
    stale = pd.DataFrame(
        {
            "__ts__": ["2026-04-01 12:00:00+00:00", "2026-04-01 12:00:00+00:00"],
            "__symbol__": ["BTC", "BTC"],
            "candidate": ["c1", "c2"],
        }
    )
    stale.to_csv(root / "utility_risk_gate_candidate_weekly" / "candidate_selected_rows.csv", index=False)

    result = build_preflight(input_dir=root, output_dir=out)

    assert result["ready_for_side_aware_stage0"] is False
    assert result["decision"] == "refresh_required"
    block = [row for row in result["blocking_artifacts"] if row["role"] == "weekly_candidate_selected_rows"]
    assert block
    assert "missing_side" in block[0]["failures"]
    assert "missing_candidate_id" in block[0]["failures"]


def test_side_contract_preflight_allows_same_candidate_id_across_replay_candidates(tmp_path: Path) -> None:
    root = tmp_path / "reports"
    out = tmp_path / "out"
    _write_side_aware_root(root)
    rows = _side_rows()
    duplicate = rows.iloc[[0, 0]].copy()
    duplicate["candidate"] = ["replay_a", "replay_b"]
    duplicate.to_csv(root / "utility_risk_gate_candidate_weekly" / "candidate_selected_rows.csv", index=False)

    result = build_preflight(input_dir=root, output_dir=out)

    assert result["ready_for_side_aware_stage0"] is True


def test_side_contract_preflight_blocks_duplicate_materialized_candidate_id(tmp_path: Path) -> None:
    root = tmp_path / "reports"
    out = tmp_path / "out"
    _write_side_aware_root(root)
    rows = _side_rows()
    duplicate = pd.concat([rows.iloc[[0]], rows.iloc[[0]]], ignore_index=True)
    duplicate.to_parquet(root / "source_archetypes_v2" / "candidate_source_archetypes_v2.parquet", index=False)

    result = build_preflight(input_dir=root, output_dir=out)

    assert result["ready_for_side_aware_stage0"] is False
    block = [row for row in result["blocking_artifacts"] if row["role"] == "archetype_materialized_rows"]
    assert block
    assert "duplicate_candidate_id" in block[0]["failures"]


def test_side_contract_preflight_blocks_missing_required_artifact(tmp_path: Path) -> None:
    root = tmp_path / "reports"
    out = tmp_path / "out"
    _write_side_aware_root(root)
    (root / "source_archetypes_v2" / "source_archetypes_v2_scorecard.csv").unlink()

    result = build_preflight(input_dir=root, output_dir=out)

    assert result["ready_for_side_aware_stage0"] is False
    block = [row for row in result["blocking_artifacts"] if row["role"] == "archetype_scorecard"]
    assert block
    assert block[0]["status"] == "missing_required_artifact"
