import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.inference.policy_rank_reference import (
    persist_fullscope_score_distribution_reference,
)
from extreme_price_movements.portfolio_policy_replay import (
    fit_hierarchical_ev_curves,
    normalise_candidate_table,
)
from scripts.generate_fixed_tpsl_reliability_blend_metrics import (
    BarrierConfig,
    OUT_TP,
    _label_symbol_rows,
)
from scripts.materialize_live_reliability_blend_scores import (
    OOF_TO_LIVE_FEATURES,
    _timestamp_block_split,
    _timestamp_features,
)
from scripts.build_native_reliability_component_feature_ledger import build_feature_ledger
from scripts.materialize_native_reliability_blend_scores import (
    _component_ablation_scores,
    _ensure_head_column,
    _select_component_model,
)
from scripts import materialize_live_ledger_blend_native_candidates as live_ledger_materializer
from scripts.reliability_blend_rank_reference import apply_frozen_policy_rank_reference


def _candidate_row(
    ts: str,
    symbol: str,
    strategy_id: str,
    rank: float,
    net_return: float,
) -> dict:
    timestamp = pd.Timestamp(ts, tz="UTC")
    return {
        "timestamp": timestamp,
        "symbol": symbol,
        "side": "short" if strategy_id.startswith("short") else "long",
        "strategy_id": strategy_id,
        "normalized_rank_score": rank,
        "base_strategy_threshold": 0.50,
        "calibrated_score": rank,
        "entry_price": 100.0,
        "exit_timestamp": timestamp + pd.Timedelta(hours=1),
        "exit_price": 101.0,
        "net_return": net_return,
        "gross_return": net_return + 0.001,
        "fees_bps": 10.0,
        "slippage_bps": 0.0,
        "holding_bars": 4,
        "simple_policy_exit_reason": "tp" if net_return > 0 else "full_sl",
    }


def test_timestamp_features_are_scoped_by_head_and_timestamp():
    frame = pd.DataFrame(
        {
            "head": ["a", "a", "b", "b"],
            "timestamp": [pd.Timestamp("2026-01-01", tz="UTC")] * 4,
            "score": [0.10, 0.90, 0.20, 0.30],
        }
    )

    feats = _timestamp_features(frame, "score")

    assert feats["score_minus_ts_mean"].tolist() == pytest.approx([-0.40, 0.40, -0.05, 0.05])
    assert feats["score_ts_rank"].tolist() == pytest.approx([0.5, 1.0, 0.5, 1.0])
    assert np.expm1(feats["timestamp_row_count_log"].to_numpy()) == pytest.approx([2, 2, 2, 2])


def test_timestamp_split_uses_complete_timestamps_only():
    timestamps = pd.date_range("2026-01-01", periods=6, tz="UTC", freq="h")
    frame = pd.DataFrame(
        {
            "timestamp": np.repeat(timestamps, 3),
            "x": np.arange(18),
        }
    )

    train, valid, diag = _timestamp_block_split(
        frame,
        train_fraction=0.5,
        embargo_hours=1.0,
        min_rows=1,
    )

    train_ts = set(pd.to_datetime(train["timestamp"], utc=True))
    valid_ts = set(pd.to_datetime(valid["timestamp"], utc=True))
    assert train_ts.isdisjoint(valid_ts)
    assert diag["split_type"] == "complete_timestamp_embargo"


def test_leaf_depth_feature_mapping_uses_depth_not_count():
    mapping = {name: live_col for name, _oof_col, live_col in OOF_TO_LIVE_FEATURES}

    assert mapping["oof_leaf_depth_mean"] == "meta_lgbm_leaf_depth_mean"


def test_native_component_scorer_maps_live_strategy_id_to_head():
    strategy_id = (
        "long_bars_in_high_vol_state_log_norm_-0_49417102_loc_range_pos_48_0_22034115"
        "_loc_swing_range_pos_24_1_0002919_atr_percentile_-1_477338_range_24h_pct_0_13988039"
        "_variance_ratio_10_48_0_92117828"
    )
    frame = pd.DataFrame({"strategy_id": [strategy_id], "timestamp": [pd.Timestamp("2026-01-01", tz="UTC")], "symbol": ["BTC/USD"]})

    mapped = _ensure_head_column(frame)

    assert mapped["head"].tolist() == ["long_bars"]


def test_native_component_scorer_fails_closed_for_unknown_strategy_id():
    frame = pd.DataFrame({"strategy_id": ["unknown_strategy"], "timestamp": [pd.Timestamp("2026-01-01", tz="UTC")], "symbol": ["BTC/USD"]})

    with pytest.raises(RuntimeError, match="cannot be mapped"):
        _ensure_head_column(frame)


def test_native_component_ledger_resolves_oof_aliases_and_timestamp_aggregates():
    frame = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-01-01", tz="UTC")] * 4,
            "symbol": ["A", "B", "C", "D"],
            "strategy_id": ["long_bars_in_high_vol_state_log_norm_-0_49417102_loc_range_pos_48_0_22034115_loc_swing_range_pos_24_1_0002919_atr_percentile_-1_477338_range_24h_pct_0_13988039_variance_ratio_10_48_0_92117828"] * 2
            + ["short_asset_ret_4h_-0_44735157_funding_abs_z_-1_117185_asset_minus_mkt_oi_1d_-0_5707467_H10"] * 2,
            "head": ["long_bars", "long_bars", "short_asset", "short_asset"],
            "calibrated_score": [0.20, 0.60, 0.30, 0.90],
            "policy_rank_pct": [0.25, 0.75, 0.40, 0.95],
        }
    )
    required = {
        "long_bars": [
            "control__export__oof_pred",
            "mean__control__export__oof_pred",
            "std__control__export__oof_pred",
        ],
        "short_asset": [
            "control__export__oof_pred",
            "mean__control__export__oof_pred",
            "std__control__export__oof_pred",
        ],
    }

    features, diagnostics = build_feature_ledger(frame, required)

    assert diagnostics["resolved"].all()
    assert features["control__export__oof_pred"].tolist() == pytest.approx([0.20, 0.60, 0.30, 0.90])
    assert features["mean__control__export__oof_pred"].tolist() == pytest.approx([0.40, 0.40, 0.60, 0.60])
    assert features["std__control__export__oof_pred"].tolist() == pytest.approx(
        [np.sqrt(0.08), np.sqrt(0.08), np.sqrt(0.18), np.sqrt(0.18)]
    )


def test_native_component_scorer_prefers_full_fit_artifact():
    fold_artifact = {"component": "qfail_soft", "fold": 2, "backend": "constant", "fill_value": 0.1}
    full_fit_artifact = {
        "component": "qfail_soft",
        "fold": "full_fit",
        "model_scope": "full_fit",
        "backend": "constant",
        "fill_value": 0.2,
    }

    selected = _select_component_model(
        [fold_artifact, full_fit_artifact],
        component="qfail_soft",
        allow_oof_fold_models=False,
    )

    assert selected is full_fit_artifact


def test_native_component_scorer_emits_anchor_qfail_period_and_full_blend_arms():
    anchor = np.array([0.20, 0.80], dtype=np.float32)
    period = np.array([0.10, 0.90], dtype=np.float32)
    qfail = np.array([0.70, 0.30], dtype=np.float32)
    config = {
        "alpha": 0.25,
        "beta": -0.40,
        "period_power": 1.0,
        "period_side": "high",
        "qfail_power": 1.0,
        "qfail_side": "high",
    }

    arms = _component_ablation_scores(anchor, period, qfail, config)

    assert sorted(arms) == [
        "reliability_anchor_only_score",
        "reliability_anchor_period_score",
        "reliability_anchor_qfail_score",
        "reliability_blend_score",
    ]
    assert arms["reliability_anchor_only_score"].tolist() == pytest.approx([0.20, 0.80])
    assert arms["reliability_anchor_qfail_score"].tolist() == pytest.approx([0.20 - 0.40 * 0.70, 0.80 - 0.40 * 0.30])
    assert arms["reliability_anchor_period_score"].tolist() == pytest.approx([0.20 + 0.25 * 0.10, 0.80 + 0.25 * 0.90])
    assert arms["reliability_blend_score"].tolist() == pytest.approx(
        [0.20 + 0.25 * 0.10 - 0.40 * 0.70, 0.80 + 0.25 * 0.90 - 0.40 * 0.30]
    )


def test_rank_reference_helper_fails_closed_without_reference():
    frame = pd.DataFrame(
        {
            "strategy_id": ["long_demo"],
            "side": ["long"],
            "calibrated_score": [0.8],
        }
    )

    with pytest.raises(RuntimeError, match="Missing --rank-reference-run-id"):
        apply_frozen_policy_rank_reference(frame, data_root=".", run_id=None)


def test_rank_reference_helper_uses_frozen_distribution(tmp_path):
    run_id = "rank-ref"
    persist_fullscope_score_distribution_reference(
        {
            "long_demo": pd.DataFrame(
                {
                    "calibrated_score": [0.10, 0.20, 0.50, 0.90],
                    "timestamp": pd.date_range("2026-01-01", periods=4, tz="UTC", freq="h"),
                    "symbol": ["A", "B", "C", "D"],
                }
            )
        },
        data_root=tmp_path,
        run_id=run_id,
        market_mode="perps",
    )
    frame = pd.DataFrame(
        {
            "strategy_id": ["long_demo", "long_demo"],
            "side": ["long", "long"],
            "calibrated_score": [0.19, 0.95],
        }
    )

    ranked, diag = apply_frozen_policy_rank_reference(frame, data_root=tmp_path, run_id=run_id)

    assert diag["rank_source"] == "policy_rank_reference_percentile"
    assert ranked["policy_rank_pct"].tolist() == pytest.approx([0.25, 1.0])
    assert ranked["auction_rank_score"].tolist() == pytest.approx([0.25, 1.0])


def test_live_ledger_market_state_overlay_accepts_score_bundle_contract(tmp_path, monkeypatch):
    ts = pd.date_range("2026-06-15", periods=2, freq="h", tz="UTC")
    candidates = pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": ["AAA/USD:USD", "BBB/USD:USD"],
            "side": ["short", "short"],
            "strategy_id": ["short_asset_demo", "short_asset_demo"],
            "head": ["short_asset", "short_asset"],
            "calibrated_score": [0.80, 0.82],
            "normalized_rank_score": [0.80, 0.82],
            "strategy_rank_pct": [0.80, 0.82],
            "policy_rank_pct": [0.80, 0.82],
            "rank_pct": [0.80, 0.82],
            "base_strategy_threshold": [0.70, 0.70],
            "deployment_rank_threshold": [0.70, 0.70],
            "entry_price": [100.0, 100.0],
            "exit_timestamp": ts + pd.Timedelta(hours=1),
            "exit_price": [99.0, 98.0],
            "net_return": [0.01, 0.02],
            "gross_return": [0.011, 0.021],
            "holding_bars": [1, 1],
            "simple_policy_exit_reason": ["tp", "tp"],
        }
    )
    deployable = candidates.copy()
    feature_store_dir = tmp_path / "features"
    feature_store_dir.mkdir()

    bundle = {
        "rank_contract": "strict",
        "disabled_heads": [],
        "selected_arm": "S0_rejected_controller_noop",
        "controller_execution_enabled": False,
        "state_spec": {"state_level": "observed"},
        "controller_enabled_heads": [],
    }

    def _fake_load_bundle(path):
        assert path == tmp_path / "noop_bundle.joblib"
        return bundle

    def _fake_score_candidates(
        *,
        bundle,
        candidates,
        feature_store_dir,
        feature_store_symbol_cap,
        allow_candidate_state_fallback,
    ):
        scored = candidates.copy()
        predictions = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(candidates["timestamp"], utc=True).drop_duplicates(),
                "state_low_input_coverage": [0.0, 0.0],
            }
        )
        state = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(candidates["timestamp"], utc=True).drop_duplicates(),
                "state_input_coverage": [1.0, 1.0],
            }
        )
        schedule = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(candidates["timestamp"], utc=True),
                "strategy_id": candidates["strategy_id"].astype(str),
                "head": candidates["head"].astype(str),
                "base_threshold": [0.70, 0.70],
                "state_threshold": [0.70, 0.70],
            }
        )
        report = {
            "controller_execution_enabled": False,
            "state_rows": int(len(state)),
            "schedule_rows": int(len(schedule)),
        }
        return scored, predictions, schedule, state, report

    monkeypatch.setattr(live_ledger_materializer, "_load_market_state_controller_bundle", _fake_load_bundle)
    monkeypatch.setattr(
        live_ledger_materializer,
        "_score_market_state_controller_candidates",
        _fake_score_candidates,
    )

    overlay = live_ledger_materializer._materialize_market_state_controller_overlay(
        candidates=candidates,
        deployable=deployable,
        controller_bundle_path=tmp_path / "noop_bundle.joblib",
        feature_store_dir=feature_store_dir,
        feature_store_symbol_cap=10,
        allow_candidate_state_fallback=False,
        portfolio_policy_manifest=None,
        portfolio_policy_variant=None,
        ev_reference_candidates=None,
        market_mode="perps",
        policy_dir=tmp_path,
    )

    assert overlay["enabled"] is True
    assert overlay["selected_arm"] == "S0_rejected_controller_noop"
    assert overlay["deployable_added"] == 0
    assert overlay["deployable_removed"] == 0
    assert overlay["deployable_jaccard"] == pytest.approx(1.0)
    assert overlay["score_report"]["controller_execution_enabled"] is False
    assert Path(overlay["outputs"]["state"]).exists()
    assert Path(overlay["outputs"]["market_state_timestamp_panel"]).exists()
    assert Path(overlay["outputs"]["market_state_feature_coverage"]).exists()
    assert Path(overlay["outputs"]["strategy_threshold_schedule"]).exists()
    assert Path(overlay["outputs"]["strategy_threshold_action_audit"]).exists()
    assert Path(overlay["outputs"]["strategy_threshold_controller_config"]).exists()
    assert set(overlay["output_sha256"]) == set(overlay["outputs"])
    assert all(isinstance(value, str) and len(value) == 64 for value in overlay["output_sha256"].values())
    timestamp_panel = pd.read_parquet(overlay["outputs"]["market_state_timestamp_panel"])
    feature_coverage = pd.read_csv(overlay["outputs"]["market_state_feature_coverage"])
    action_audit = pd.read_csv(overlay["outputs"]["strategy_threshold_action_audit"])
    controller_config = json.loads(Path(overlay["outputs"]["strategy_threshold_controller_config"]).read_text())
    assert timestamp_panel.columns[:3].tolist() == ["split", "state_level", "timestamp"]
    assert timestamp_panel["split"].unique().tolist() == ["live_overlay"]
    assert timestamp_panel["state_level"].unique().tolist() == ["observed"]
    assert feature_coverage["feature"].tolist() == ["state_input_coverage"]
    assert feature_coverage["finite_share"].tolist() == pytest.approx([1.0])
    assert float(action_audit.loc[action_audit["scope"].eq("all"), "max_threshold_delta"].iloc[0]) == 0.0
    assert controller_config["controller_execution_enabled"] is False
    assert controller_config["invariants"]["controller_changes_scores_or_ranks"] is False
    assert controller_config["invariants"]["controller_changes_auction_ordering"] is False


def test_hierarchical_ev_curve_persists_strategy_side_curves_with_shrinkage():
    rows = []
    for i in range(6):
        rows.append(_candidate_row(f"2026-01-01 0{i}:00", f"A{i}", "long_a", 0.10 + i * 0.15, -0.01 + i * 0.005))
        rows.append(_candidate_row(f"2026-01-02 0{i}:00", f"B{i}", "short_b", 0.10 + i * 0.15, 0.01 - i * 0.004))

    curve = fit_hierarchical_ev_curves(pd.DataFrame(rows), min_group_rows=2, shrink_rows=2)

    assert curve["schema"] == "hierarchical_ev_curve_v1"
    assert "long_a|long" in curve["strategy_side"]
    assert "short_b|short" in curve["strategy_side"]
    assert 0.0 < curve["strategy_side"]["long_a|long"]["shrink_weight"] < 1.0


def test_candidate_key_duplicates_fail_closed():
    rows = pd.DataFrame(
        [
            _candidate_row("2026-01-01 00:00", "BTC/USD", "long_a", 0.9, 0.01),
            _candidate_row("2026-01-01 00:00", "BTC/USD", "long_a", 0.8, -0.01),
        ]
    )

    with pytest.raises(ValueError, match="duplicate decision keys"):
        normalise_candidate_table(rows)


def test_vol_normalized_tpsl_records_effective_barriers():
    ohlcv = pd.DataFrame(
        {
            "ts": pd.date_range("2026-01-01", periods=8, tz="UTC", freq="h"),
            "open": [100.0, 101.0, 100.5, 101.5, 102.0, 102.5, 103.0, 103.5],
            "high": [100.5, 101.5, 101.0, 102.0, 104.5, 103.0, 103.5, 104.0],
            "low": [99.5, 100.5, 100.0, 101.0, 101.5, 102.0, 102.5, 103.0],
            "close": [100.0, 101.0, 100.5, 101.5, 102.0, 102.5, 103.0, 103.5],
        }
    )
    rows = pd.DataFrame(
        {
            "head": ["long_bars"],
            "row_id": [1],
            "timestamp": [pd.Timestamp("2026-01-01 03:00", tz="UTC")],
            "symbol": ["BTC/USD"],
        }
    )
    cfg = BarrierConfig(
        barrier_mode="vol_norm",
        tp_mult=1.5,
        sl_mult=1.0,
        fixed_tp=0.03,
        fixed_sl=0.02,
        horizon_hours=3.0,
        vol_lookback_hours=4,
        vol_min_periods=2,
        min_barrier=0.005,
        max_barrier=0.05,
    )

    labeled = _label_symbol_rows(ohlcv, rows, side="long", cfg=cfg)

    assert labeled["fixed_barrier_mode"].iloc[0] == "vol_norm"
    assert labeled["fixed_barrier_pct"].iloc[0] >= 0.005
    assert labeled["fixed_effective_tp"].iloc[0] == pytest.approx(1.5 * labeled["fixed_barrier_pct"].iloc[0])
    assert labeled["fixed_effective_sl"].iloc[0] == pytest.approx(labeled["fixed_barrier_pct"].iloc[0])
    assert labeled["fixed_outcome"].iloc[0] in {OUT_TP, 0, 1}
