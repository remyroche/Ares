from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.audit_market_state_priority_window_readiness import audit_window_readiness


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_candidate(
    root: Path,
    *,
    timestamps: list[str],
    rank_scope: str = "global_over_time",
    rank_contract: str = "anchor_global_policy_rank_reference",
    rank_reference_run_id: str = "prejune_rankref",
    qfail_active: bool = False,
    head_health_active: bool | None = None,
) -> Path:
    candidate_dir = root / "simple_policy_optimiser"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for ts in timestamps:
        for head in ["short_asset", "short_boll"]:
            rows.append(
                {
                    "timestamp": pd.Timestamp(ts),
                    "head": head,
                    "symbol": f"{head}_BTC",
                    "strategy_id": f"{head}_strategy",
                    "side": "short",
                }
            )
    candidate = candidate_dir / "simple_policy_candidates.parquet"
    pd.DataFrame(rows).to_parquet(candidate)
    _write_json(
        root / "t1_repaired_static_baseline_manifest.json",
        {
            "generated_by": "materialize_t1_repaired_static_baseline",
            "active_stack": {
                "rank_contract": rank_contract,
                "rank_scope": rank_scope,
                "rank_reference_run_id": rank_reference_run_id,
                "promotion_status": "rank_contract_challenger",
                "policy_variant": "refit_bar4_strategy_bar2",
                "auction": "global_auction",
                "enabled_heads": ["short_asset", "short_boll"],
                "disabled_heads": ["long_bars", "long_dist"],
                "qfail_active": qfail_active,
                "head_health_active": head_health_active,
                "market_state_threshold_controller_active": False,
            },
        },
    )
    return candidate


def _write_existing_manifest(path: Path) -> None:
    _write_json(
        path,
        {
            "contract": {
                "candidate_rank_contracts": [
                    {
                        "rank_contract": "anchor_global_policy_rank_reference",
                        "rank_scope": "global_over_time",
                        "rank_reference_run_id": "prejune_rankref",
                    }
                ],
                "candidate_rank_scopes": ["global_over_time"],
                "candidate_rank_contract_names": ["anchor_global_policy_rank_reference"],
            },
            "windows": [
                {
                    "label": "existing",
                    "candidate": {
                        "path": "existing.parquet",
                        "sha256": "oldsha",
                        "rows": 10,
                        "timestamp_count": 2,
                        "start": "2026-06-15T00:00:00+00:00",
                        "end": "2026-06-15T01:00:00+00:00",
                        "heads": ["short_asset", "short_boll"],
                        "contract": {
                            "rank_contract": "anchor_global_policy_rank_reference",
                            "rank_scope": "global_over_time",
                            "rank_reference_run_id": "prejune_rankref",
                            "policy_variant": "refit_bar4_strategy_bar2",
                        },
                    },
                }
            ],
        },
    )


def test_window_readiness_accepts_fresh_matching_global_rank_candidate(tmp_path: Path) -> None:
    existing = tmp_path / "existing" / "manifest.json"
    _write_existing_manifest(existing)
    candidate = _write_candidate(
        tmp_path / "candidate",
        timestamps=[
            "2026-06-24T09:00:00Z",
            "2026-06-24T10:00:00Z",
            "2026-06-24T11:00:00Z",
        ],
    )

    summary = audit_window_readiness(
        candidates=[candidate],
        existing_manifest=existing,
        output_dir=tmp_path / "out",
        min_timestamp_count=3,
    )

    assert summary["passed"] is True
    assert summary["passing_candidate_count"] == 1
    row = summary["candidate_rows"][0]
    assert row["status"] == "pass"
    assert row["rank_scope"] == "global_over_time"
    assert (tmp_path / "out" / "market_state_priority_window_readiness.csv").exists()


def test_window_readiness_rejects_overlapping_window(tmp_path: Path) -> None:
    existing = tmp_path / "existing" / "manifest.json"
    _write_existing_manifest(existing)
    candidate = _write_candidate(
        tmp_path / "candidate",
        timestamps=[
            "2026-06-15T00:00:00Z",
            "2026-06-15T01:00:00Z",
            "2026-06-15T02:00:00Z",
        ],
    )

    summary = audit_window_readiness(
        candidates=[candidate],
        existing_manifest=existing,
        output_dir=tmp_path / "out",
        min_timestamp_count=3,
    )

    assert summary["passed"] is False
    assert "candidate_window_overlaps_existing_shadow_window" in summary["candidate_rows"][0]["reasons"]


def test_window_readiness_rejects_mixed_rank_contract_candidate(tmp_path: Path) -> None:
    existing = tmp_path / "existing" / "manifest.json"
    _write_existing_manifest(existing)
    candidate = _write_candidate(
        tmp_path / "candidate",
        timestamps=[
            "2026-06-24T09:00:00Z",
            "2026-06-24T10:00:00Z",
            "2026-06-24T11:00:00Z",
        ],
        rank_scope="within_timestamp",
        rank_contract="short_boll_timestamp_rank",
    )

    summary = audit_window_readiness(
        candidates=[candidate],
        existing_manifest=existing,
        output_dir=tmp_path / "out",
        min_timestamp_count=3,
    )

    assert summary["passed"] is False
    reasons = summary["candidate_rows"][0]["reasons"]
    assert "rank_scope_mismatch" in reasons
    assert "rank_contract_mismatch" in reasons


def test_window_readiness_rejects_qfail_active_candidate(tmp_path: Path) -> None:
    existing = tmp_path / "existing" / "manifest.json"
    _write_existing_manifest(existing)
    candidate = _write_candidate(
        tmp_path / "candidate",
        timestamps=[
            "2026-06-24T09:00:00Z",
            "2026-06-24T10:00:00Z",
            "2026-06-24T11:00:00Z",
        ],
        qfail_active=True,
    )

    summary = audit_window_readiness(
        candidates=[candidate],
        existing_manifest=existing,
        output_dir=tmp_path / "out",
        min_timestamp_count=3,
    )

    assert summary["passed"] is False
    assert "qfail_active" in summary["candidate_rows"][0]["reasons"]
