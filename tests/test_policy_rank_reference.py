import json

import numpy as np
import pandas as pd

from extreme_price_movements.inference.policy_rank_reference import (
    PolicyRankReferenceStore,
    apply_policy_rank_percentile_gate,
    persist_policy_rank_reference,
    policy_rank_pct_from_sorted_scores,
)


def test_persist_policy_rank_reference_manifest(tmp_path):
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, tz="UTC"),
            "symbol": ["AAA/USDC", "BBB/USDC", "CCC/USDC"],
            "strategy_id": ["long_demo"] * 3,
            "calibrated_score": [0.1, 0.4, 0.8],
            "rank_pct": [1 / 3, 2 / 3, 1.0],
        }
    )

    out = persist_policy_rank_reference(
        df,
        data_root=tmp_path,
        run_id="run_a",
        strategy_id="long_demo",
        market_mode="spot",
    )

    manifest_path = (
        tmp_path
        / "artifacts"
        / "run_a"
        / "simple_policy_optimiser"
        / "rank_reference"
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    assert out.exists()
    assert manifest["schema_version"] == "policy_rank_reference_v1"
    assert manifest["generated_by"] == "simple_policy_optimiser"
    row = manifest["strategies"]["long_demo"]
    assert row["n_rows"] == 3
    assert row["score_col"] == "calibrated_score"
    assert row["rank_col"] == "rank_pct"
    assert row["min_score"] == 0.1
    assert row["max_score"] == 0.8


def test_policy_rank_pct_searchsorted_right():
    scores = np.asarray([0.1, 0.2, 0.4, 0.8])

    assert policy_rank_pct_from_sorted_scores(scores, 0.4) == 0.75
    assert policy_rank_pct_from_sorted_scores(scores, 0.9) == 1.0
    assert policy_rank_pct_from_sorted_scores(scores, 0.05) == 0.0


def test_live_gate_uses_policy_rank_pct_not_meta_train_rank(tmp_path):
    df = pd.DataFrame(
        {
            "strategy_id": ["long_demo"] * 5,
            "calibrated_score": [0.10, 0.20, 0.30, 0.40, 0.50],
            "rank_pct": [0.2, 0.4, 0.6, 0.8, 1.0],
        }
    )
    persist_policy_rank_reference(
        df,
        data_root=tmp_path,
        run_id="run_a",
        strategy_id="long_demo",
        market_mode="spot",
    )
    store = PolicyRankReferenceStore(data_root=tmp_path, run_id="run_a")

    rejected = {
        "strategy_id": "long_demo",
        "side": "long",
        "threshold_space": "rank_percentile",
        "calibrated_score": 0.25,
        "meta_train_rank_pct": 0.95,
        "effective_threshold": 0.58,
        "chain_results": {"meta_train_rank_pct": 0.95},
    }
    assert apply_policy_rank_percentile_gate(rejected, store=store) == (
        False,
        "rank_below_dynamic_threshold",
    )
    assert rejected["policy_rank_pct"] == 0.4
    assert rejected["rank_score_source"] == "policy_rank_reference_percentile"

    accepted = {
        "strategy_id": "long_demo",
        "side": "long",
        "threshold_space": "rank_percentile",
        "calibrated_score": 0.30,
        "meta_train_rank_pct": 0.40,
        "effective_threshold": 0.58,
        "chain_results": {"meta_train_rank_pct": 0.40},
    }
    assert apply_policy_rank_percentile_gate(accepted, store=store) == (True, None)
    assert accepted["policy_rank_pct"] == 0.6
    assert accepted["rank_score_source"] == "policy_rank_reference_percentile"


def test_missing_policy_rank_reference_fails_closed(tmp_path):
    store = PolicyRankReferenceStore(data_root=tmp_path, run_id="missing_run")
    decision = {
        "strategy_id": "long_demo",
        "side": "long",
        "threshold_space": "rank_percentile",
        "calibrated_score": 0.95,
        "sizer_rank_percentile": 1.0,
        "effective_threshold": 0.50,
        "chain_results": {},
    }

    assert apply_policy_rank_percentile_gate(decision, store=store) == (
        False,
        "missing_policy_rank_reference_percentile",
    )
    assert decision["rank_score_source"] == "missing_policy_rank_reference_percentile"

    debug_decision = dict(decision)
    debug_decision["chain_results"] = {}
    assert apply_policy_rank_percentile_gate(
        debug_decision,
        store=store,
        allow_live_batch_rank_fallback_for_debug=True,
    ) == (True, None)
    assert debug_decision["rank_score_source"] == "live_batch_percentile_fallback_debug"


def test_calibrated_score_threshold_space_unchanged(tmp_path):
    store = PolicyRankReferenceStore(data_root=tmp_path, run_id="missing_run")
    decision = {
        "strategy_id": "long_demo",
        "side": "long",
        "threshold_space": "calibrated_score",
        "calibrated_score": 0.1,
        "effective_threshold": 0.9,
        "chain_results": {},
    }

    assert apply_policy_rank_percentile_gate(decision, store=store) == (True, None)
    assert "policy_rank_pct" not in decision
