import json

import numpy as np
import pandas as pd

from extreme_price_movements.inference.policy_rank_reference import (
    PolicyRankReferenceStore,
    apply_policy_rank_percentile_gate,
    invalidate_auction_rank_reference,
    persist_auction_rank_reference,
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
            "policy_oos_generation_source": ["generated_from_train_meta_state:labels"] * 3,
            "policy_oos_source_model_fit_end": ["2026-01-19T06:00:00+00:00"] * 3,
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
    assert row["policy_oos_contract"]["policy_oos_generation_source"] == (
        "generated_from_train_meta_state:labels"
    )
    assert row["policy_oos_contract"]["rank_normalization"] == (
        "policy_rank_reference_percentile_from_policy_oos_clf"
    )
    assert manifest["policy_oos_contract"]["policy_oos_generation_source"] == (
        "generated_from_train_meta_state:labels"
    )
    assert manifest["policy_oos_contract"]["rank_normalization"] == (
        "policy_rank_reference_percentile_from_policy_oos_clf"
    )


def test_policy_rank_reference_bad_policy_oos_contract_fails_closed(tmp_path):
    df = pd.DataFrame(
        {
            "strategy_id": ["long_demo"] * 3,
            "calibrated_score": [0.1, 0.4, 0.8],
            "rank_pct": [1 / 3, 2 / 3, 1.0],
            "policy_oos_generation_source": ["generated_from_train_meta_state:labels"] * 3,
            "policy_oos_source_model_fit_end": ["2026-01-19T06:00:00+00:00"] * 3,
        }
    )
    persist_policy_rank_reference(
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
    manifest["strategies"]["long_demo"]["policy_oos_contract"][
        "policy_oos_generation_source"
    ] = "generated_from_inference_models:labels"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    store = PolicyRankReferenceStore(data_root=tmp_path, run_id="run_a")
    result = store.lookup(
        strategy_id="long_demo",
        calibrated_score=0.4,
        side="long",
    )

    assert np.isnan(result.policy_rank_pct)
    assert result.n_rows == 0


def test_policy_rank_reference_bad_manifest_policy_oos_contract_fails_closed(tmp_path):
    df = pd.DataFrame(
        {
            "strategy_id": ["long_demo"] * 3,
            "calibrated_score": [0.1, 0.4, 0.8],
            "rank_pct": [1 / 3, 2 / 3, 1.0],
            "policy_oos_generation_source": ["generated_from_train_meta_state:labels"] * 3,
            "policy_oos_source_model_fit_end": ["2026-01-19T06:00:00+00:00"] * 3,
        }
    )
    persist_policy_rank_reference(
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
    manifest["policy_oos_contract"]["policy_oos_generation_source"] = (
        "generated_from_inference_models:labels"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    store = PolicyRankReferenceStore(data_root=tmp_path, run_id="run_a")
    result = store.lookup(
        strategy_id="long_demo",
        calibrated_score=0.4,
        side="long",
    )

    assert np.isnan(result.policy_rank_pct)
    assert result.n_rows == 0


def test_policy_rank_reference_loader_survives_promoted_external_manifest_path(tmp_path):
    df = pd.DataFrame(
        {
            "strategy_id": ["long_demo"] * 3,
            "calibrated_score": [0.1, 0.4, 0.8],
            "rank_pct": [1 / 3, 2 / 3, 1.0],
            "policy_oos_generation_source": ["generated_from_train_meta_state:labels"] * 3,
            "policy_oos_source_model_fit_end": ["2026-01-19T06:00:00+00:00"] * 3,
        }
    )
    persist_policy_rank_reference(
        df,
        data_root=tmp_path,
        run_id="run_a",
        strategy_id="long_demo",
        market_mode="perp",
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
    manifest["strategies"]["long_demo"]["path"] = (
        "extreme_price_movements/reports/promoted_copy/simple_policy_optimiser/"
        "rank_reference/long_demo.parquet"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    store = PolicyRankReferenceStore(data_root=tmp_path, run_id="run_a")
    result = store.lookup(strategy_id="long_demo", calibrated_score=0.4, side="long")

    assert result.n_rows == 3
    assert result.policy_rank_pct == 2 / 3
    assert result.source.endswith("rank_reference/long_demo.parquet")


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
    persist_auction_rank_reference(
        pd.DataFrame(
            {
                "calibrated_score": [0.05, 0.15, 0.25, 0.35, 0.45],
                "strategy_id": ["s"] * 5,
            }
        ),
        data_root=tmp_path,
        run_id="run_a",
    )

    rejected = {
        "strategy_id": "long_demo",
        "side": "long",
        "threshold_space": "rank_percentile",
        "calibrated_score": 0.25,
        "meta_train_rank_pct": 0.95,
        "effective_threshold": 0.70,
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
    assert accepted["normalized_rank_score"] == 0.6
    assert accepted["auction_rank_score_source"] == "cross_strategy_auction_reference"
    assert accepted["rank_score_source"] == "policy_rank_reference_percentile"


def test_live_gate_can_require_cross_strategy_auction_rank(tmp_path):
    df = pd.DataFrame(
        {
            "strategy_id": ["short_demo"] * 5,
            "calibrated_score": [0.10, 0.20, 0.30, 0.40, 0.90],
            "rank_pct": [0.2, 0.4, 0.6, 0.8, 1.0],
        }
    )
    persist_policy_rank_reference(
        df,
        data_root=tmp_path,
        run_id="run_a",
        strategy_id="short_demo",
        market_mode="perp",
    )
    persist_auction_rank_reference(
        pd.DataFrame(
            {
                "calibrated_score": [0.60, 0.70, 0.80, 0.90],
                "strategy_id": ["long_a", "long_b", "short_demo", "long_c"],
            }
        ),
        data_root=tmp_path,
        run_id="run_a",
    )
    store = PolicyRankReferenceStore(data_root=tmp_path, run_id="run_a")
    decision = {
        "strategy_id": "short_demo",
        "side": "short",
        "threshold_space": "rank_percentile",
        "calibrated_score": 0.40,
        "effective_threshold": 0.50,
        "chain_results": {},
    }

    assert apply_policy_rank_percentile_gate(
        decision,
        store=store,
        require_cross_strategy_auction_rank=True,
    ) == (False, "rank_below_dynamic_threshold")
    assert decision["policy_rank_pct"] == 0.8
    assert decision["normalized_rank_score"] == 0.0
    assert decision["threshold_rank_score"] == 0.0
    assert decision["threshold_rank_score_source"] == "cross_strategy_auction_reference"


def test_deployment_threshold_uses_auction_rank_not_strategy_rank(tmp_path):
    persist_policy_rank_reference(
        pd.DataFrame(
            {
                "strategy_id": ["long_demo"] * 5,
                "calibrated_score": [0.10, 0.20, 0.30, 0.40, 0.90],
                "rank_pct": [0.2, 0.4, 0.6, 0.8, 1.0],
            }
        ),
        data_root=tmp_path,
        run_id="run_a",
        strategy_id="long_demo",
        market_mode="perps",
    )
    persist_auction_rank_reference(
        pd.DataFrame(
            {
                "calibrated_score": [0.50, 0.60, 0.70, 0.80],
                "strategy_id": ["short_a", "long_b", "short_c", "long_d"],
            }
        ),
        data_root=tmp_path,
        run_id="run_a",
    )
    store = PolicyRankReferenceStore(data_root=tmp_path, run_id="run_a")
    decision = {
        "strategy_id": "long_demo",
        "side": "long",
        "threshold_space": "rank_percentile",
        "calibrated_score": 0.40,
        "effective_threshold": 0.70,
        "chain_results": {},
    }

    assert apply_policy_rank_percentile_gate(
        decision,
        store=store,
        require_cross_strategy_auction_rank=True,
        use_auction_rank_for_threshold=True,
    ) == (False, "rank_below_dynamic_threshold")
    assert decision["policy_rank_pct"] == 0.8
    assert decision["auction_rank_pct"] == 0.0
    assert decision["threshold_rank_score"] == 0.0
    assert decision["threshold_rank_score_source"] == "cross_strategy_auction_reference"


def test_invalidate_auction_rank_reference_clears_manifest_entry(tmp_path):
    persist_policy_rank_reference(
        pd.DataFrame(
            {
                "strategy_id": ["long_a"] * 2,
                "calibrated_score": [0.60, 0.70],
                "rank_pct": [0.5, 1.0],
            }
        ),
        data_root=tmp_path,
        run_id="run_a",
        strategy_id="long_a",
    )
    persist_auction_rank_reference(
        pd.DataFrame(
            {
                "calibrated_score": [0.60, 0.70],
                "strategy_id": ["long_a", "short_b"],
            }
        ),
        data_root=tmp_path,
        run_id="run_a",
    )
    manifest_path = (
        tmp_path
        / "artifacts"
        / "run_a"
        / "simple_policy_optimiser"
        / "rank_reference"
        / "manifest.json"
    )
    assert "auction" in json.loads(manifest_path.read_text())

    invalidate_auction_rank_reference(
        data_root=tmp_path,
        run_id="run_a",
        market_mode="perps",
        reason="test_new_export",
    )

    manifest = json.loads(manifest_path.read_text())
    assert "auction" not in manifest
    assert "strategies" not in manifest
    assert manifest["previous_auction"]["n_rows"] == 2
    assert manifest["previous_strategy_count"] == 1
    assert manifest["auction_invalidated_reason"] == "test_new_export"

    store = PolicyRankReferenceStore(data_root=tmp_path, run_id="run_a")
    assert store.lookup_auction(calibrated_score=0.7).n_rows == 0


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
