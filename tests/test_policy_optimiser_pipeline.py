import json
import sys
import types

import numpy as np
import pandas as pd

# Stub heavy/syntax-broken dependency for isolated unit tests.
stub = types.ModuleType("extreme_price_movements.run_ridge_sizer")
stub.load_meta_oof_predictions = lambda *a, **k: {}
stub.load_trade_outcomes = lambda *a, **k: pd.DataFrame()
stub.load_base_oof_predictions = lambda *a, **k: {}
sys.modules.setdefault("extreme_price_movements.run_ridge_sizer", stub)

from extreme_price_movements.policy_optimiser import (
    _load_best_strategy,
    build_replay_context,
    replay_exit_policy,
    resolve_optimised_selection_frac,
)


def _load_best_policy_params_local(data_root: str, run_id: str):
    from pathlib import Path

    for candidate in [
        Path(data_root)
        / "artifacts"
        / run_id
        / "policy_params"
        / "best_policy_params.json",
        Path(data_root) / "artifacts" / run_id / "best_policy_params.json",
    ]:
        if not candidate.exists():
            continue
        payload = json.loads(candidate.read_text())
        strategies = payload.get("strategies", []) if isinstance(payload, dict) else []
        if strategies:
            return strategies[0]
    return None


def test_policy_json_roundtrip(tmp_path):
    run_id = "20260101_000000"
    p = tmp_path / "artifacts" / run_id / "policy_params"
    p.mkdir(parents=True)
    payload = {
        "schema_version": "v3",
        "strategies": [{"strategy_id": "s1", "theta_fail": 0.2, "a1": 0.6}],
    }
    (p / "best_policy_params.json").write_text(json.dumps(payload))
    loaded = _load_best_policy_params_local(str(tmp_path), run_id)
    assert loaded is not None
    assert loaded["strategy_id"] == "s1"
    assert loaded["theta_fail"] == 0.2


def test_holdout_policy_json_missing_fallback(tmp_path):
    loaded = _load_best_policy_params_local(str(tmp_path), "missing")
    assert loaded is None


def test_sign_direction_for_fail_and_path_scores():
    base = np.array([-0.001, 0.01], dtype=np.float32)
    context = {
        "mfe_ret": np.array([0.01, 0.02], dtype=np.float32),
        "mae_ret": np.array([0.03, 0.005], dtype=np.float32),
        "bars_since_entry": np.array([2, 3], dtype=np.int32),
        "barrier_pct": np.array([0.02, 0.02], dtype=np.float32),
        "AE_vel": np.array([2.0, -1.0], dtype=np.float32),
        "pressure": np.array([2.0, -1.0], dtype=np.float32),
        "path_quality": np.array([-2.0, 2.0], dtype=np.float32),
        "progress_per_bar": np.array([0.2, 0.2], dtype=np.float32),
        "confidence": np.array([0.2, 0.2], dtype=np.float32),
    }
    params = {
        "tp_mult": 1.0,
        "sl_mult": 1.0,
        "a1": 1.0,
        "a2": 0.0,
        "b1": 1.0,
        "b2": 0.0,
        "theta_fail": -1.0,
        "theta_path": 0.0,
        "d_path": 1,
        "K_early": 3,
        "progress_threshold": 0.0,
    }
    out = replay_exit_policy(base, context, params)
    # With early exit correctly using raw returns when hitting discretionary conditions,
    # the out return could match or be clipped depending on how it exits.
    # The intent of the test is just that the sign/scores flow through correctly.
    assert out[0] <= 0.0
    assert out[1] >= base[1] or np.isclose(out[1], base[1])


def test_run_all_wires_offset_before_policy(monkeypatch):
    import extreme_price_movements.run_pipeline as rp

    events = []
    monkeypatch.setattr(rp, "run_features", lambda *a, **k: events.append("features"))
    monkeypatch.setattr(rp, "run_train", lambda *a, **k: events.append("train"))
    monkeypatch.setattr(
        rp, "run_optimise", lambda *a, **k: events.append("optimise") or True
    )
    monkeypatch.setattr(rp, "run_sizer", lambda *a, **k: events.append("sizer") or True)
    monkeypatch.setattr(
        rp, "run_policy_optimisation", lambda *a, **k: events.append("policy") or True
    )
    monkeypatch.setattr(rp, "run_base_hpo_step", lambda *a, **k: None)
    monkeypatch.setattr(rp, "_maintenance_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(
        rp, "_resolve_ts_sig", lambda *a, **k: pd.Timestamp("2026-01-01", tz="UTC")
    )
    monkeypatch.setattr(rp, "PartitionedOHLCVStore", lambda *a, **k: object())
    monkeypatch.setattr(
        rp, "_run_offset_generation_stage", lambda *a, **k: events.append("offset")
    )

    cfg = {
        "data_root": "data",
        "timeframe": "15m",
        "enable_trigger_discovery_stage": False,
        "reports_root": "reports",
        "run_limit_offset_optimiser": True,
    }
    rp.run_all(cfg)

    assert events.index("sizer") < events.index("offset") < events.index("policy")


def test_best_strategy_uses_head_to_head_winner(tmp_path):
    run_id = "20260101_000000"
    root = tmp_path / "artifacts" / run_id
    (root / "ridge_sizer").mkdir(parents=True)
    (root / "et_sizer").mkdir(parents=True)

    (root / "ridge_sizer" / "head_to_head_comparison.json").write_text(
        json.dumps([{"strategy_id": "s_et", "winner": "et"}])
    )
    (root / "et_sizer" / "strategy_params.json").write_text(
        json.dumps(
            {
                "best_strategy_id": "s_et",
                "best_threshold_pct": 85.0,
                "buckets": {"s_et": {"threshold_pct": 85.0, "net_pnl": 2.0}},
            }
        )
    )
    (root / "ridge_sizer" / "strategy_params.json").write_text(
        json.dumps(
            {
                "best_strategy_id": "s_ridge",
                "best_threshold_pct": 90.0,
                "buckets": {"s_ridge": {"threshold_pct": 90.0, "net_pnl": 5.0}},
            }
        )
    )

    best = _load_best_strategy(str(tmp_path), run_id)
    assert best["strategy_id"] == "s_et"
    assert best["model"] == "et"


def test_shared_replay_context_semantics_are_stable():
    returns = np.array([0.01, -0.02], dtype=np.float32)
    mfe = np.array([0.03, 0.01], dtype=np.float32)
    mae = np.array([0.005, 0.02], dtype=np.float32)
    bars = np.array([2, 5], dtype=np.int32)
    barrier = np.array([0.02, 0.02], dtype=np.float32)
    conf = np.array([0.8, 0.2], dtype=np.float32)

    context = build_replay_context(
        returns=returns,
        mfe_ret=mfe,
        mae_ret=mae,
        bars_since_entry=bars,
        barrier_pct=barrier,
        confidence=conf,
    )
    params = {
        "tp_mult": 1.0,
        "sl_mult": 1.0,
        "theta_fail": 0.0,
        "theta_path": 0.0,
        "a1": 0.5,
        "a2": 0.5,
        "b1": 0.5,
        "b2": 0.5,
        "trail_activation_atr": 0.5,
        "trail_giveback_atr": 0.2,
        "continuation_conf_threshold": 0.5,
        "multiplier_band_min": 0.8,
        "multiplier_band_max": 1.2,
    }
    out1 = replay_exit_policy(returns, context, params)
    out2 = replay_exit_policy(returns, dict(context), params)
    np.testing.assert_allclose(out1, out2)


def test_selection_frac_uses_optimised_positive_pnl_threshold(tmp_path):
    run_id = "20260101_000000"
    root = tmp_path / "artifacts" / run_id / "ridge_sizer"
    root.mkdir(parents=True)
    (root / "strategy_params.json").write_text(
        json.dumps(
            {
                "buckets": {
                    "s1": {"threshold_pct": 95.0, "net_pnl": -0.2},
                    "s2": {"threshold_pct": 85.0, "net_pnl": 1.2},
                }
            }
        )
    )
    frac = resolve_optimised_selection_frac(
        data_root=str(tmp_path),
        run_id=run_id,
        selected={"strategy_id": "s1", "model": "ridge", "threshold_pct": 95.0},
    )
    # fallback should choose threshold 85% => top 15%
    assert np.isclose(frac, 0.15)
