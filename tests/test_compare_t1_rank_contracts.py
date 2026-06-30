from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts import compare_t1_rank_contracts as compare


def _write_candidates(root: Path, rows: pd.DataFrame) -> None:
    policy_dir = root / "simple_policy_optimiser"
    policy_dir.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(policy_dir / "simple_policy_candidates_broad.parquet", index=False)


def test_candidate_universe_reports_duplicate_and_key_counts(tmp_path: Path) -> None:
    root = tmp_path / "arm"
    rows = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-06-23T00:00:00Z", "2026-06-23T00:00:00Z"],
                utc=True,
            ),
            "symbol": ["BTC", "BTC"],
            "strategy_id": ["short_boll_s1", "short_boll_s1"],
            "side": ["short", "short"],
            "head": ["short_boll", "short_boll"],
        }
    )
    _write_candidates(root, rows)

    frame, keys = compare._candidate_universe(root, "timestamp")

    assert len(keys) == 1
    assert int(frame.iloc[0]["rows"]) == 2
    assert int(frame.iloc[0]["unique_decision_keys"]) == 1
    assert int(frame.iloc[0]["duplicate_decision_keys"]) == 1
    assert frame.iloc[0]["heads"] == "short_boll"
    assert len(str(frame.iloc[0]["sha256"])) == 64


def test_comparison_contract_fails_when_candidate_universe_differs(tmp_path: Path) -> None:
    manifest = {
        "active_stack": {
            "rank_contract": "short_boll_timestamp_rank",
            "rank_scope": "within_timestamp",
            "score_path": "anchor_meta_calibrated_score",
            "active_score_column": "calibrated_score",
            "static_base_thresholds": True,
            "policy_variant": "refit_bar4_strategy_bar2",
            "enabled_heads": ["short_asset", "short_boll"],
            "disabled_heads": ["long_bars", "long_dist"],
            "auction": "global_auction",
            "ev_mapping": "hierarchical_ev_curves_fitted_on_train_deployable",
            "qfail_active": False,
            "native_reliability_blend_active": False,
            "market_state_threshold_controller_active": False,
        }
    }
    challenger = json.loads(json.dumps(manifest))
    challenger["active_stack"]["rank_contract"] = "anchor_global_policy_rank_reference"
    challenger["active_stack"]["rank_scope"] = "global_over_time"
    challenger["validation"] = {
        "rank_reference_contract": {"required": True, "passed": True, "failures": []}
    }
    challenger["rank_reference_diagnostics"] = {
        "eval": {
            "rank_source": "policy_rank_reference_percentile",
            "missing_rank_rows": 0,
            "missing_auction_rank_rows": 0,
            "window_rank_debug_used": False,
        },
        "train_deployable": {
            "rank_source": "policy_rank_reference_percentile",
            "missing_rank_rows": 0,
            "missing_auction_rank_rows": 0,
            "window_rank_debug_used": False,
        },
    }
    candidate_universe = pd.DataFrame(
        {
            "contract_name": ["timestamp", "global"],
            "rows": [2, 3],
            "unique_decision_keys": [2, 3],
            "duplicate_decision_keys": [0, 0],
        }
    )
    base_dir = tmp_path / "base"
    challenger_dir = tmp_path / "challenger"
    base_dir.mkdir()
    challenger_dir.mkdir()
    (base_dir / "t1_repaired_static_baseline_manifest.json").write_text("{}", encoding="utf-8")
    (challenger_dir / "t1_repaired_static_baseline_manifest.json").write_text("{}", encoding="utf-8")

    payload = compare._comparison_contract(
        base_dir=base_dir,
        challenger_dir=challenger_dir,
        base_name="timestamp",
        challenger_name="global",
        base_manifest=manifest,
        challenger_manifest=challenger,
        candidate_universe=candidate_universe,
        candidate_universe_overlap={
            "identical": False,
            "base_duplicate_decision_keys": 0,
            "challenger_duplicate_decision_keys": 0,
        },
    )

    assert payload["validation"]["passed"] is False
    assert payload["validation"]["candidate_universe_identical"] is False
    assert "candidate_universe_not_identical" in payload["validation"]["failures"]
