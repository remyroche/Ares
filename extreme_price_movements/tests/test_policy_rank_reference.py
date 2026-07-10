import pandas as pd

from extreme_price_movements.inference.policy_rank_reference import (
    PolicyRankReferenceStore,
    apply_policy_rank_percentile_gate,
    persist_auction_rank_reference,
    persist_policy_rank_reference,
)


def test_policy_rank_reference_persists_archetype_columns(tmp_path):
    frame = pd.DataFrame(
        {
            "strategy_id": ["long_breakout"] * 2,
            "side": ["long", "long"],
            "policy_archetype": ["long__compression_release"] * 2,
            "calibrated_score": [0.1, 0.9],
            "rank_pct": [0.5, 1.0],
        }
    )

    path = persist_policy_rank_reference(
        frame,
        data_root=tmp_path,
        run_id="run",
        strategy_id="long_breakout",
    )
    auction_path = persist_auction_rank_reference(
        frame,
        data_root=tmp_path,
        run_id="run",
    )

    stored = pd.read_parquet(path)
    auction = pd.read_parquet(auction_path)
    assert stored["policy_archetype"].tolist() == [
        "long__compression_release",
        "long__compression_release",
    ]
    assert auction["policy_archetype"].tolist() == [
        "long__compression_release",
        "long__compression_release",
    ]


def test_strategy_ev_lookup_prefers_side_archetype_candidate_curve(tmp_path):
    run_id = "run"
    candidates_dir = tmp_path / "artifacts" / run_id / "simple_policy_optimiser"
    candidates_dir.mkdir(parents=True)
    candidates = pd.DataFrame(
        {
            "strategy_id": ["long_breakout"] * 6,
            "side": ["long"] * 6,
            "policy_archetype": [
                "long__clean",
                "long__clean",
                "long__clean",
                "long__dirty",
                "long__dirty",
                "long__dirty",
            ],
            "strategy_rank_pct": [0.1, 0.8, 0.9, 0.1, 0.8, 0.9],
            "net_return": [0.001, 0.03, 0.04, -0.02, -0.01, -0.005],
        }
    )
    candidates.to_parquet(candidates_dir / "simple_policy_candidates.parquet", index=False)
    store = PolicyRankReferenceStore(data_root=tmp_path, run_id=run_id)

    clean = store.strategy_threshold_for_ev(
        strategy_id="long_breakout",
        side="long",
        policy_archetype="long__clean",
        target_mean_net_return=0.03,
        min_hit_rate=1.0,
        fallback_threshold=1.0,
    )
    dirty = store.strategy_threshold_for_ev(
        strategy_id="long_breakout",
        side="long",
        policy_archetype="long__dirty",
        target_mean_net_return=0.02,
        min_hit_rate=1.0,
        fallback_threshold=1.0,
    )

    assert clean.enabled is True
    assert clean.reason == "strategy_side_archetype_ev_threshold"
    assert clean.threshold == 0.8
    assert dirty.enabled is False
    assert dirty.reason == "no_strategy_threshold_meets_ev_and_hit_rate_constraints"


def test_strategy_ev_gate_uses_side_archetype_candidate_curve(tmp_path):
    run_id = "run"
    candidates_dir = tmp_path / "artifacts" / run_id / "simple_policy_optimiser"
    candidates_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "strategy_id": ["short_mr", "short_mr"],
            "side": ["short", "short"],
            "policy_archetype": ["short__continuation", "short__continuation"],
            "strategy_rank_pct": [0.8, 0.9],
            "net_return": [0.02, 0.03],
        }
    ).to_parquet(candidates_dir / "simple_policy_candidates.parquet", index=False)
    store = PolicyRankReferenceStore(data_root=tmp_path, run_id=run_id)

    gate = store.strategy_ev_gate(
        strategy_id="short_mr",
        side="short",
        policy_archetype="short__continuation",
        target_mean_net_return=0.015,
        min_hit_rate=1.0,
    )

    assert gate.allowed is True
    assert gate.reason == "strategy_side_archetype_ev_gate_pass"
    assert gate.mean_net_return >= 0.02


def test_policy_rank_gate_protects_regime_ev_top10_admission(tmp_path):
    run_id = "run"
    frame = pd.DataFrame(
        {
            "strategy_id": ["long_breakout"] * 10,
            "side": ["long"] * 10,
            "policy_archetype": ["long__clean"] * 10,
            "calibrated_score": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
            "rank_pct": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
        }
    )
    persist_policy_rank_reference(
        frame,
        data_root=tmp_path,
        run_id=run_id,
        strategy_id="long_breakout",
    )
    persist_auction_rank_reference(frame, data_root=tmp_path, run_id=run_id)
    store = PolicyRankReferenceStore(data_root=tmp_path, run_id=run_id)
    decision = {
        "strategy_id": "long_breakout",
        "side": "long",
        "calibrated_score": 0.80,
        "raw_calibrated_score": 1.00,
        "effective_threshold": 0.90,
        "regime_ev_protect_admission_rank": True,
        "regime_ev_protected_admission_floor": 0.90,
        "regime_ev_retained_surplus_frac": 0.50,
        "chain_results": {
            "portfolio_rank_adjustment": 0.0,
        },
    }

    allowed, reason = apply_policy_rank_percentile_gate(
        decision,
        store=store,
    )

    assert allowed is True
    assert reason is None
    assert decision["policy_rank_pct"] == 0.95
    assert decision["threshold_rank_score"] == 0.95
    assert decision["policy_rank_pct_raw_calibrated_score"] == 1.0
    assert decision["threshold_rank_score_source"].endswith(
        "_protected_regime_ev_floor"
    )
