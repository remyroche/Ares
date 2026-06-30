from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.discover_market_state_priority_append_windows import (
    discover_append_windows,
    discover_candidates,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_candidate(
    root: Path,
    *,
    timestamps: list[str],
    t1: bool = True,
    rank_scope: str = "global_over_time",
    rank_contract: str = "anchor_global_policy_rank_reference",
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
    path = candidate_dir / "simple_policy_candidates.parquet"
    pd.DataFrame(rows).to_parquet(path)
    if t1:
        _write_json(
            root / "t1_repaired_static_baseline_manifest.json",
            {
                "generated_by": "materialize_t1_repaired_static_baseline",
                "active_stack": {
                    "rank_contract": rank_contract,
                    "rank_scope": rank_scope,
                    "rank_reference_run_id": "prejune_rankref",
                    "promotion_status": "rank_contract_challenger",
                    "policy_variant": "refit_bar4_strategy_bar2",
                    "auction": "global_auction",
                    "enabled_heads": ["short_asset", "short_boll"],
                    "disabled_heads": ["long_bars", "long_dist"],
                    "qfail_active": False,
                    "head_health_active": None,
                    "market_state_threshold_controller_active": False,
                },
            },
        )
    else:
        _write_json(root / "other_manifest.json", {"generated_by": "other"})
    return path


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
                        "end": "2026-06-15T02:00:00+00:00",
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


def test_discover_candidates_filters_to_t1_like_ledgers(tmp_path: Path) -> None:
    good = _write_candidate(
        tmp_path / "T1_good",
        timestamps=[
            "2026-06-24T09:00:00Z",
            "2026-06-24T10:00:00Z",
            "2026-06-24T11:00:00Z",
        ],
    )
    _write_candidate(
        tmp_path / "unrelated",
        timestamps=[
            "2026-06-24T09:00:00Z",
            "2026-06-24T10:00:00Z",
            "2026-06-24T11:00:00Z",
        ],
        t1=False,
    )

    found = discover_candidates([tmp_path], include_regex="T1|unrelated")

    assert found == [good]


def test_discover_append_windows_reports_only_readiness_passing_candidates(tmp_path: Path) -> None:
    existing = tmp_path / "existing" / "manifest.json"
    _write_existing_manifest(existing)
    passing = _write_candidate(
        tmp_path / "T1_future",
        timestamps=[
            "2026-06-24T09:00:00Z",
            "2026-06-24T10:00:00Z",
            "2026-06-24T11:00:00Z",
        ],
    )
    _write_candidate(
        tmp_path / "T1_overlap",
        timestamps=[
            "2026-06-15T00:00:00Z",
            "2026-06-15T01:00:00Z",
            "2026-06-15T02:00:00Z",
        ],
    )
    _write_candidate(
        tmp_path / "T1_timestamp_contract",
        timestamps=[
            "2026-06-24T09:00:00Z",
            "2026-06-24T10:00:00Z",
            "2026-06-24T11:00:00Z",
        ],
        rank_scope="within_timestamp",
        rank_contract="short_boll_timestamp_rank",
    )

    summary = discover_append_windows(
        roots=[tmp_path],
        existing_manifest=existing,
        output_dir=tmp_path / "out",
        include_regex="T1_",
        min_timestamp_count=3,
    )

    assert summary["discovered_candidate_count"] == 3
    assert summary["appendable_candidate_count"] == 1
    appendable = pd.read_csv(summary["appendable_csv"])
    assert appendable["path"].tolist() == [str(passing)]
    assert (tmp_path / "out" / "market_state_priority_append_window_discovery_report.md").exists()
