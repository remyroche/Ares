import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from extreme_price_movements.inference import run_inference as ri
from extreme_price_movements.inference.model_orchestrator import ModelOrchestrator
from extreme_price_movements.inference.policy_rank_reference import (
    AuctionEvThresholdResult,
    StrategyEvGateResult,
    persist_auction_rank_reference,
    persist_fullscope_score_distribution_reference,
    persist_policy_rank_reference,
)
from extreme_price_movements.inference.safety_switches import StrategyKillSwitch
from extreme_price_movements.portfolio_manager import PortfolioManager


@pytest.fixture(autouse=True)
def _synthetic_strategy_ev_contract(monkeypatch):
    """Step-parity fixtures do not mount policy-OOS EV tables."""

    def _threshold(
        self,
        *,
        strategy_id,
        side,
        policy_archetype=None,
        target_mean_net_return,
        min_hit_rate,
        fallback_threshold,
    ):
        return AuctionEvThresholdResult(
            threshold=0.0,
            target_mean_net_return=float(target_mean_net_return),
            target_hit_rate=float(min_hit_rate),
            mean_net_return=0.01,
            hit_rate=1.0,
            n_trades=100,
            source="synthetic_test_contract",
            enabled=True,
            reason="synthetic_test_contract",
        )

    def _gate(
        self,
        *,
        strategy_id,
        side,
        policy_archetype=None,
        target_mean_net_return,
        min_hit_rate,
    ):
        return StrategyEvGateResult(
            allowed=True,
            target_mean_net_return=float(target_mean_net_return),
            min_hit_rate=float(min_hit_rate),
            mean_net_return=0.01,
            hit_rate=1.0,
            source="synthetic_test_contract",
            reason="synthetic_test_contract",
        )

    monkeypatch.setattr(
        ri.PolicyRankReferenceStore,
        "strategy_threshold_for_ev",
        _threshold,
    )
    monkeypatch.setattr(
        ri.PolicyRankReferenceStore,
        "strategy_ev_gate",
        _gate,
    )


def test_load_normalized_threshold_map_prefers_policy_deployment_rank(tmp_path):
    run_id = "run"
    sizer_dir = tmp_path / "artifacts" / run_id / "simple_position_sizer"
    sizer_dir.mkdir(parents=True)
    (sizer_dir / "normalized_strategy_thresholds.json").write_text(
        json.dumps(
            {
                "threshold_space": "rank_percentile",
                "strategies": {
                    "mr": {
                        "normalized_threshold": 0.59,
                        "viability_margin": 0.01,
                    }
                },
            }
        )
    )
    policy_dir = tmp_path / "artifacts" / run_id / "policy_params"
    policy_dir.mkdir(parents=True)
    (policy_dir / "strategy_for_inference.json").write_text(
        json.dumps(
            {
                "strategies": [
                    {
                        "strategy_id": "long_mr",
                        "canonical_strategy_id": "mr",
                        "strategy_for_inference": "long_mr",
                        "side": "long",
                        "selected": True,
                        "deployment_rank_threshold": 0.73,
                        "avg_trades_per_day_at_top_1pct": 9.0,
                        "avg_holding_time_hours": 8.0,
                    }
                ]
            }
        )
    )

    thresholds = ri._load_normalized_threshold_map(str(tmp_path), run_id)

    assert thresholds["long_mr"]["threshold_space"] == "rank_percentile"
    assert thresholds["long_mr"]["normalized_threshold"] == 0.73
    assert (
        thresholds["long_mr"]["threshold_scope"] == "per_strategy_prediction_rank_only"
    )
    assert thresholds["mr"]["normalized_threshold"] == 0.73


def test_candidate_priority_uses_live_friction_ev_rank_and_hr_adjustment():
    base_decision = {
        "normalized_rank_score": 0.99,
        "effective_threshold": 0.90,
        "expected_friction_bps": 0.0,
        "chain_results": {
            "portfolio_rank_adjustment": -0.01,
            "portfolio_priority_multiplier": 1.0,
            "portfolio_priority_adjustment": 0.0,
        },
    }
    live_ev_decision = {
        **base_decision,
        "chain_results": {
            **base_decision["chain_results"],
            "threshold_rank_score_after_friction_ev": 0.93,
        },
    }

    assert ri._candidate_rank_score(base_decision) == pytest.approx(0.98)
    assert ri._candidate_rank_score(live_ev_decision) == pytest.approx(0.92)
    assert ri._candidate_threshold_rank_score(live_ev_decision) == pytest.approx(0.93)
    assert ri._candidate_portfolio_priority(
        live_ev_decision
    ) < ri._candidate_portfolio_priority(base_decision)


def test_corrected_ev_rank_is_canonical_for_position_sizing():
    decision = {
        "normalized_rank_score": 0.93,
        "chain_results": {
            "threshold_basis_corrected_expected_ev_rank": 0.97,
        },
    }

    assert ri._candidate_corrected_ev_rank_for_sizing(decision) == pytest.approx(0.97)


def test_policy_size_power_prefers_deployed_field_with_legacy_fallback():
    assert ri._resolve_policy_size_power(
        {"size_power": 1.4, "best_size_power": 2.0}
    ) == pytest.approx(1.4)
    assert ri._resolve_policy_size_power({"best_size_power": 2.0}) == pytest.approx(
        2.0
    )


def test_prediction_ledger_row_persists_replay_strategy_auction_and_portfolio_state():
    now = pd.Timestamp("2026-07-10T09:00:00Z")
    pm = PortfolioManager(max_positions=4, portfolio_value=1000.0)
    pm.record_position_open(
        symbol="BTC/USD:USD",
        side="long",
        strategy_id="long_s52_meta_threshold_handoff",
        position_size=25.0,
        entry_price=100.0,
        entry_time=now - pd.Timedelta(hours=1),
    )
    decision = {
        "symbol": "ETH/USD:USD",
        "side": "short",
        "strategy_id": "s52_meta_threshold_handoff",
        "raw_score": 0.88,
        "calibrated_score": 0.91,
        "normalized_rank_score": 0.96,
        "effective_threshold": 0.90,
        "model_artifact_run_id": "model_run",
        "policy_artifact_run_id": "policy_run",
        "chain_results": {
            "meta_pred": 0.88,
            "calibrated_score": 0.91,
            "threshold_rank_score": 0.96,
            "effective_threshold": 0.90,
        },
    }

    before_capacity = pm.get_portfolio_capacity(
        side="short",
        strategy_id="short_s52_meta_threshold_handoff",
    )
    ri._attach_portfolio_replay_state_for_ledger(
        decision,
        portfolio_mgr=pm,
        capacity=before_capacity,
        now_utc=now,
    )
    ri._attach_global_auction_metadata(
        [decision],
        entry_cap=2,
        max_new_entries_per_bar=1,
        sorted_at=now,
    )
    pm.record_position_open(
        symbol="ETH/USD:USD",
        side="short",
        strategy_id="short_s52_meta_threshold_handoff",
        position_size=30.0,
        entry_price=90.0,
        entry_time=now,
    )
    after_capacity = pm.get_portfolio_capacity(
        side="short",
        strategy_id="short_s52_meta_threshold_handoff",
    )
    ri._attach_portfolio_replay_state_after_for_ledger(
        decision,
        portfolio_mgr=pm,
        capacity=after_capacity,
        now_utc=now,
    )

    row = ri._prediction_ledger_row(
        decision,
        timestamp=now.isoformat(),
        side="short",
        portfolio_decision="traded",
        was_traded=True,
    )

    assert row["strategy_id"] == "short_s52_meta_threshold_handoff"
    assert row["decision_strategy_id"] == "short_s52_meta_threshold_handoff"
    assert row["source_strategy_id"] == "s52_meta_threshold_handoff"
    assert row["canonical_strategy_id"] == "s52_meta_threshold_handoff"
    assert row["auction_policy_version"] == "global_auction_v1"
    assert row["auction_candidate_count"] == 1
    assert row["auction_rank_number"] == 1
    assert row["auction_selected_before_capacity"] is True
    assert row["open_positions_before_count"] == 1
    assert row["open_positions_after_count"] == 2
    assert json.loads(row["portfolio_state_snapshot_json"])["schema"] == (
        "portfolio_replay_state_v4_pre_leverage_wallet"
    )
    assert json.loads(row["portfolio_state_after_snapshot_json"])["schema"] == (
        "portfolio_replay_state_v4_pre_leverage_wallet"
    )


def test_policy_rank_threshold_source_assertion_accepts_strategy_rank_gate():
    decision = {
        "threshold_space": "rank_percentile",
        "policy_rank_pct": 0.82,
        "auction_rank_pct": 0.41,
        "threshold_rank_score": 0.82,
        "threshold_rank_score_source": "policy_rank_reference_percentile",
        "threshold_rank_score_source_preference": "policy_rank_pct",
        "chain_results": {
            "policy_rank_pct": 0.82,
            "auction_rank_pct": 0.41,
            "threshold_rank_score": 0.82,
            "threshold_rank_score_source": "policy_rank_reference_percentile",
        },
    }

    ri._assert_policy_rank_threshold_source(decision)


def test_perp_rank_context_prefers_model_artifact_run_for_shadow(tmp_path):
    strategy_id = "short_demo"
    model_run_id = "model_run"
    shadow_run_id = "shadow_run"
    core = ri.strategy_core_id(strategy_id)
    meta_dir = tmp_path / "artifacts" / model_run_id / "meta_oof"
    meta_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "oof_pred": [0.95, 0.80, 0.40, 0.20],
            "target": [1, 1, 0, 0],
        }
    ).to_parquet(meta_dir / f"meta_oof_short_{core}_tbm_clf.parquet", index=False)
    (tmp_path / "artifacts" / shadow_run_id).mkdir(parents=True)

    ri._PERP_RANK_CONTEXT_CACHE.clear()
    resolved_run_id = ri._perp_rank_context_run_id(
        {
            "run_id": shadow_run_id,
            "model_artifact_run_id": model_run_id,
        }
    )
    context = ri._perp_rank_context(
        data_root=str(tmp_path),
        run_id=resolved_run_id,
        side="short",
        strategy_id=strategy_id,
        score=0.90,
    )

    assert resolved_run_id == model_run_id
    assert context["rank_number"] == 2
    assert f"{tmp_path}|{model_run_id}|short|{strategy_id}" in ri._PERP_RANK_CONTEXT_CACHE
    assert (
        f"{tmp_path}|{shadow_run_id}|short|{strategy_id}"
        not in ri._PERP_RANK_CONTEXT_CACHE
    )


def test_perp_rank_context_falls_back_to_policy_rank_reference(tmp_path):
    run_id = "frozen_run"
    strategy_id = "long_demo"
    ref_dir = (
        tmp_path
        / "artifacts"
        / run_id
        / "simple_policy_optimiser"
        / "rank_reference"
    )
    ref_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "strategy_id": [strategy_id] * 5,
            "calibrated_score": [0.10, 0.20, 0.50, 0.80, 0.90],
            "rank_pct": [0.20, 0.40, 0.60, 0.80, 1.00],
        }
    ).to_parquet(ref_dir / f"{strategy_id}.parquet", index=False)

    ri._PERP_RANK_CONTEXT_CACHE.clear()
    context = ri._perp_rank_context(
        data_root=str(tmp_path),
        run_id=run_id,
        side="long",
        strategy_id=strategy_id,
        score=0.75,
    )

    assert context["rank_number"] == 3
    assert context["rank_x"] == 1
    assert context["profitable_rank_count"] == 1
    assert context["rank_context_source"].endswith(f"{strategy_id}.parquet")


def test_meta_hit_rate_calibration_resolves_side_tbm_alias(tmp_path):
    run_id = "run"
    report_dir = tmp_path / "artifacts" / run_id / "meta_oof"
    report_dir.mkdir(parents=True)
    (report_dir / "meta_calibration_report.json").write_text(
        json.dumps(
            {
                "short_mr_tbm_clf": {
                    "move_calibration": {
                        "reliability_curve": [
                            {"mean_pred": 0.25, "mean_true": 0.40, "count": 10},
                            {"mean_pred": 0.75, "mean_true": 0.80, "count": 30},
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    ri._META_HIT_RATE_CALIBRATION_CACHE.clear()

    calibration = ri._load_meta_hit_rate_calibration(str(tmp_path), run_id)
    resolved = ri._estimated_hit_rate_from_meta_prediction(
        0.75,
        "mr",
        calibration,
    )

    assert "mr" in calibration
    assert "short_mr_tbm_clf" in calibration
    assert resolved["estimated_hit_rate"] == pytest.approx(0.80)
    assert resolved["estimated_hit_rate_calibration_n"] == 40
    assert resolved["estimated_hit_rate_source"].endswith(":mr")


def test_live_regime_calibration_raw_contract_excludes_generated_and_injected():
    artifact = {
        "effects": [
            {
                "feature_cols": [
                    "raw_market_feature",
                    "resid_event_aegmm_gmm_entropy",
                    "hit_probability",
                    "calibrated_score",
                    "__regime_source_shock_impulse_score__",
                ]
            }
        ]
    }
    residual_payload = {
        "generated_feature_columns": ["resid_event_aegmm_gmm_entropy"]
    }

    assert ri._live_regime_calibration_raw_feature_columns(
        artifact, residual_payload
    ) == ["raw_market_feature"]


def test_live_postprocessor_hydration_only_fails_missing_residual_inputs():
    frame = pd.DataFrame({"strict_observable": [1.0, 2.0]})

    hydrated, strict_missing, optional_missing = (
        ri._hydrate_optional_frozen_features(
            frame,
            attempted_columns=["strict_observable", "optional_calibration_bin"],
            strict_columns=["strict_observable"],
        )
    )

    assert strict_missing == []
    assert optional_missing == ["optional_calibration_bin"]
    assert "optional_calibration_bin" in hydrated
    assert hydrated["optional_calibration_bin"].isna().all()


def test_live_postprocessor_hydration_keeps_residual_contract_fail_closed():
    frame = pd.DataFrame(index=[0, 1])

    _, strict_missing, optional_missing = ri._hydrate_optional_frozen_features(
        frame,
        attempted_columns=["strict_observable", "optional_calibration_bin"],
        strict_columns=["strict_observable"],
    )

    assert strict_missing == ["strict_observable"]
    assert optional_missing == ["optional_calibration_bin"]


def test_effective_live_entry_cap_never_exceeds_policy_cap():
    assert ri._effective_live_entry_cap(4, 2, entries_allowed=True) == 2
    assert ri._effective_live_entry_cap(1, 2, entries_allowed=True) == 1
    assert ri._effective_live_entry_cap(4, 2, entries_allowed=False) == 0


def test_fullscope_score_distribution_reference_is_mapping_only(tmp_path, monkeypatch):
    data_root = tmp_path
    run_id = "run"
    policy = pd.DataFrame(
        {
            "strategy_id": ["short_mr"] * 3,
            "calibrated_score": [0.1, 0.2, 0.3],
            "rank_pct": [1 / 3, 2 / 3, 1.0],
        }
    )
    persist_policy_rank_reference(
        policy,
        data_root=data_root,
        run_id=run_id,
        strategy_id="short_mr",
        market_mode="perps",
    )
    persist_auction_rank_reference(
        policy,
        data_root=data_root,
        run_id=run_id,
        market_mode="perps",
    )
    manifest = persist_fullscope_score_distribution_reference(
        {
            "short_mr": pd.DataFrame(
                {
                    "calibrated_score": [0.1, 0.2, 0.9, 0.95],
                    "timestamp": pd.date_range(
                        "2026-01-01", periods=4, tz="UTC", freq="h"
                    ),
                    "symbol": ["A", "B", "C", "D"],
                }
            )
        },
        data_root=data_root,
        run_id=run_id,
        market_mode="perps",
    )

    store = ri.PolicyRankReferenceStore(data_root=data_root, run_id=run_id)
    fullscope_rank = store.lookup(
        strategy_id="mr",
        side="short",
        calibrated_score=0.9,
    )
    auction_rank = store.lookup_auction(calibrated_score=0.9)

    assert fullscope_rank.policy_rank_pct == pytest.approx(0.75)
    assert auction_rank.policy_rank_pct == pytest.approx(0.75)
    assert "fullscope_score_distribution" in fullscope_rank.source
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["reference_is_in_sample"] is True
    assert payload["performance_claim"] == "none"
    assert payload["ev_claim"] == "none"

    monkeypatch.setenv("EPM_POLICY_RANK_USE_FULLSCOPE_SCORE_DISTRIBUTION", "0")
    fallback = ri.PolicyRankReferenceStore(data_root=data_root, run_id=run_id).lookup(
        strategy_id="mr",
        side="short",
        calibrated_score=0.9,
    )

    assert fallback.policy_rank_pct == pytest.approx(1.0)
    assert "fullscope_score_distribution" not in fallback.source


def test_load_lgbm_strategy_masks_prefers_embedded_strategy_contract(tmp_path):
    run_id = "run"
    policy_dir = tmp_path / "artifacts" / run_id / "policy_params"
    policy_dir.mkdir(parents=True)
    (policy_dir / "strategy_for_inference.json").write_text(
        json.dumps(
            {
                "strategies": [
                    {
                        "strategy_id": "long_mr",
                        "canonical_strategy_id": "mr",
                        "strategy_for_inference": "long_mr",
                        "side": "long",
                        "selected": True,
                        "lgbm_regime_mask": {
                            "strategy_id": "registry_mr",
                            "trade_side": "long",
                            "base_event_trigger": "price_up|dist_ema_fast|bucket_hi",
                            "mask_params": {
                                "canonical_key": "price_up|dist_ema_fast|bucket_hi",
                                "threshold": 0.2,
                            },
                            "source_target": "price_up",
                            "source_horizon": 12,
                        },
                    }
                ]
            }
        )
    )

    masks = ri._load_lgbm_strategy_mask_rows(str(tmp_path), run_id)

    assert set(masks) >= {"long_mr", "mr"}
    assert masks["long_mr"]["strategy_id"] == "long_mr"
    assert masks["long_mr"]["base_event_trigger"] == "price_up|dist_ema_fast|bucket_hi"
    assert (
        masks["long_mr"]["mask_params"]["canonical_key"]
        == "price_up|dist_ema_fast|bucket_hi"
    )
    assert masks["mr"]["strategy_id"] == "long_mr"


def test_load_lgbm_strategy_masks_fallback_filters_to_selected_strategies(
    tmp_path, monkeypatch
):
    run_id = "run"
    policy_dir = tmp_path / "artifacts" / run_id / "policy_params"
    policy_dir.mkdir(parents=True)
    (policy_dir / "strategy_for_inference.json").write_text(
        json.dumps(
            {
                "strategies": [
                    {
                        "strategy_id": "long_selected",
                        "canonical_strategy_id": "selected",
                        "strategy_for_inference": "long_selected",
                        "side": "long",
                        "selected": True,
                        "lgbm_regime_mask": {},
                    }
                ]
            }
        )
    )

    from extreme_price_movements.offline_optimisers import params_store

    monkeypatch.setattr(
        params_store,
        "load_inference_candidate_mask_params_per_bucket",
        lambda **_: [
            {
                "strategy_id": "selected",
                "trade_side": "long",
                "base_event_trigger": "selected_rule",
                "mask_params": {"canonical_key": "selected_rule"},
            },
            {
                "strategy_id": "unselected",
                "trade_side": "short",
                "base_event_trigger": "unselected_rule",
                "mask_params": {"canonical_key": "unselected_rule"},
            },
        ],
    )

    masks = ri._load_lgbm_strategy_mask_rows(str(tmp_path), run_id)

    assert {row["strategy_id"] for row in masks.values()} == {"selected"}
    assert set(masks) >= {"selected", "long_selected"}
    assert "unselected" not in masks


def test_lgbm_strategy_mask_coverage_fails_closed_for_missing_selected_strategy():
    with pytest.raises(RuntimeError, match="regime masks missing"):
        ri._validate_lgbm_strategy_mask_coverage(
            {
                "long_present": {
                    "strategy_id": "present",
                    "base_event_trigger": "(*)|(x>0)|(*)",
                }
            },
            {"long_missing"},
        )


def test_lgbm_strategy_mask_coverage_accepts_side_alias_for_selected_strategy():
    ri._validate_lgbm_strategy_mask_coverage(
        {
            "long_selected": {
                "strategy_id": "selected",
                "base_event_trigger": "(*)|(x>0)|(*)",
            }
        },
        {"long_selected"},
    )


def test_lgbm_strategy_mask_coverage_fails_closed_for_missing_trigger():
    with pytest.raises(RuntimeError, match="missing base_event_trigger"):
        ri._validate_lgbm_strategy_mask_coverage(
            {"long_selected": {"strategy_id": "selected"}},
            {"long_selected"},
        )


def test_strategy_asset_exclusion_matches_across_usdt_usdc_quote():
    assert ri._is_symbol_blocked_for_strategy(
        "BTC/USDC",
        "long_mr",
        {"long_mr": {"BTC/USDT"}},
    )


class _DummyOrchestrator:
    bucket_params = {}
    alpha_by_strategy = {
        "long_mr": {"feat_cols": ["dummy"]},
        "short_mr": {"feat_cols": ["dummy"]},
    }

    def _align_alpha_feature_contract(self, features, feat_cols):
        return features.reindex(columns=feat_cols)

    def predict_alpha(self, features, side, kind):
        return pd.Series(0.9, index=features.index, dtype=float)

    def predict_meta(self, features, side, kind):
        return pd.Series(0.9, index=features.index, dtype=float)

    def run_full_chain(self, symbol, side, features, panel=None):
        return {
            "symbol": symbol,
            "side": side,
            "action": "enter",
            "position_size": 9000.0,
            "meta_pred": 0.9,
            "strategy_id": f"{side}_mr",
        }


class _NoEntryOrchestrator:
    bucket_params = {}
    alpha_by_strategy = {"long_mr": {"feat_cols": ["dummy"]}}

    def _align_alpha_feature_contract(self, features, feat_cols):
        return features.reindex(columns=feat_cols)

    def predict_alpha(self, features, side, kind):
        return pd.Series(0.9, index=features.index, dtype=float)

    def predict_meta(self, features, side, kind):
        return pd.Series(0.9, index=features.index, dtype=float)

    def run_full_chain(self, symbol, side, features, panel=None):
        return {
            "symbol": symbol,
            "side": side,
            "action": "no_entry",
            "reason": "entry_policy_rejected",
            "position_size": 0.3,
            "meta_pred": 0.9,
            "strategy_id": f"{side}_mr",
        }


class _DummyExecutor:
    mode = "shadow"

    def __init__(self):
        self.calls = []
        self._entry_budget_tmpdir = tempfile.TemporaryDirectory()
        self.config = {
            "allow_live_batch_rank_fallback_for_debug": True,
            "entry_budget_guard_path": str(
                Path(self._entry_budget_tmpdir.name) / "entry_budget.sqlite"
            ),
        }

    def get_cooldown_hours(self, bucket_key):
        return 0.0

    def get_active_positions(self):
        return {}

    def execute_trade(self, symbol, side, size, price=None, bucket_key=None):
        self.calls.append(
            {"symbol": symbol, "side": side, "size": size, "bucket_key": bucket_key}
        )
        return {"status": "recorded", "success": True}


class _DummyLogger:
    def __init__(self):
        self.entries = []

    def get_last_trade_timestamp(self, symbol):
        return None

    def log_entry(self, **kwargs):
        self.entries.append(kwargs)


def test_model_orchestrator_uses_runtime_model_bundle_when_full_state_is_partial():
    full_state = {
        "bundle": {
            "alpha_models": {"short_old_strategy": {"model": object(), "feat_cols": []}}
        }
    }
    runtime_cfg = {
        "model_bundle": {
            "alpha_models": {
                "short_selected_strategy": {"model": object(), "feat_cols": []}
            }
        }
    }

    orchestrator = ModelOrchestrator(full_state, runtime_cfg)

    assert "short_selected_strategy" in orchestrator.alpha_by_strategy
    assert "short_old_strategy" not in orchestrator.alpha_by_strategy


def test_model_orchestrator_calls_ridge_sizer_with_named_dataframe():
    class _Sizer:
        model_names_ = ["meta_pred", "calibrated_reg_pred"]

        def __init__(self):
            self.seen_columns = None

        def predict(self, frame):
            self.seen_columns = list(frame.columns)
            return [0.12]

    sizer = _Sizer()
    orchestrator = ModelOrchestrator({"bundle": {}, "ridge_sizer": sizer}, {})
    features = pd.DataFrame({"meta_pred": [0.8]}, index=["BTC/USDT"])

    position_size, _ = orchestrator.compute_ridge_position_size(
        features, side="long", kind="long_mr"
    )

    assert position_size.iloc[0] == 0.12
    assert "calibrated_reg_pred" in sizer.seen_columns


def test_run_inference_step_applies_strategy_rank_and_portfolio_caps(monkeypatch):
    idx = pd.date_range("2026-03-01", periods=10, freq="1h", tz="UTC")
    close = pd.DataFrame({"BTC/USDT": [100.0] * len(idx)}, index=idx)
    panel = {
        "close": close,
        "high": close,
        "low": close,
        "open": close,
        "volume": close,
    }
    feats = {
        "ret12h": close.pct_change().fillna(0.0),
        "ret24h": close.pct_change().fillna(0.0),
        "range_12h_pct": close * 0.0,
        "volatility_zscore": close * 0.0,
    }

    monkeypatch.setattr(ri, "select_candidates", lambda **kwargs: (["BTC/USDT"], []))
    monkeypatch.setattr(
        ri,
        "get_features_for_candidates",
        lambda feats, candidates: pd.DataFrame(
            {"dummy": [1.0] * len(candidates)}, index=candidates
        ),
    )

    orchestrator = _DummyOrchestrator()
    executor = _DummyExecutor()
    logger = _DummyLogger()
    portfolio_mgr = PortfolioManager(portfolio_value=10000.0)

    calibration_data = {
        "long_mr": {
            "p75_threshold": 0.6,
            "calibration_curve": [(0.0, 0.0), (1.0, 1.0)],
        }
    }

    results = ri.run_inference_step(
        orchestrator=orchestrator,
        panel=panel,
        feats=feats,
        thresholds={
            "extreme_pct": None,
            "min_move_12h_pct": None,
            "min_range_pct": None,
            "min_vol_zscore": None,
            "metric": "ret12h",
        },
        executor=executor,
        logger=logger,
        accepted_strategies={"long_mr"},
        calibration_data=calibration_data,
        portfolio_mgr=portfolio_mgr,
        initial_rank_threshold=0.5,
    )

    assert len(results["trades"]) == 1
    assert executor.calls, "expected trade execution call"
    # Rank sizing reserves capacity per slot so early positions cannot consume
    # the whole safe book.
    assert executor.calls[0]["size"] <= 1500.0 + 1e-9


def test_run_inference_step_does_not_replace_policy_threshold_with_ev_table_gate(
    monkeypatch,
):
    idx = pd.date_range("2026-03-01", periods=10, freq="1h", tz="UTC")
    close = pd.DataFrame({"BTC/USDT": [100.0] * len(idx)}, index=idx)
    panel = {
        "close": close,
        "high": close,
        "low": close,
        "open": close,
        "volume": close,
    }
    feats = {
        "ret12h": close.pct_change().fillna(0.0),
        "ret24h": close.pct_change().fillna(0.0),
        "range_12h_pct": close * 0.0,
        "volatility_zscore": close * 0.0,
    }

    monkeypatch.setattr(ri, "select_candidates", lambda **kwargs: (["BTC/USDT"], []))
    monkeypatch.setattr(
        ri,
        "get_features_for_candidates",
        lambda feats, candidates: pd.DataFrame(
            {"dummy": [1.0] * len(candidates)}, index=candidates
        ),
    )

    def _disabled_ev_table_threshold(
        self,
        *,
        strategy_id,
        side,
        policy_archetype=None,
        target_mean_net_return,
        min_hit_rate,
        fallback_threshold,
    ):
        return AuctionEvThresholdResult(
            threshold=float(fallback_threshold),
            target_mean_net_return=float(target_mean_net_return),
            target_hit_rate=float(min_hit_rate),
            mean_net_return=0.011,
            hit_rate=0.53,
            n_trades=269,
            source="synthetic_disabled_ev_table",
            enabled=False,
            reason="no_strategy_threshold_meets_ev_and_hit_rate_constraints",
        )

    def _diagnostic_failed_ev_gate(
        self,
        *,
        strategy_id,
        side,
        policy_archetype=None,
        target_mean_net_return,
        min_hit_rate,
    ):
        return StrategyEvGateResult(
            allowed=False,
            target_mean_net_return=float(target_mean_net_return),
            min_hit_rate=float(min_hit_rate),
            mean_net_return=0.011,
            hit_rate=0.53,
            source="synthetic_policy_artifact",
            reason="strategy_ev_gate_failed",
        )

    monkeypatch.setattr(
        ri.PolicyRankReferenceStore,
        "strategy_threshold_for_ev",
        _disabled_ev_table_threshold,
    )
    monkeypatch.setattr(
        ri.PolicyRankReferenceStore,
        "strategy_ev_gate",
        _diagnostic_failed_ev_gate,
    )

    executor = _DummyExecutor()
    results = ri.run_inference_step(
        orchestrator=_DummyOrchestrator(),
        panel=panel,
        feats=feats,
        thresholds={
            "extreme_pct": None,
            "min_move_12h_pct": None,
            "min_range_pct": None,
            "min_vol_zscore": None,
            "metric": "ret12h",
        },
        executor=executor,
        logger=_DummyLogger(),
        accepted_strategies={"long_mr"},
        calibration_data={
            "long_mr": {
                "p75_threshold": 0.7,
                "calibration_curve": [(0.0, 0.0), (1.0, 1.0)],
            }
        },
        portfolio_mgr=PortfolioManager(portfolio_value=10000.0),
        initial_rank_threshold=0.7,
    )

    assert len(results["trades"]) == 1
    assert executor.calls, "policy artifact threshold should remain deployable"
    assert results["side_metrics"]["long"]["threshold_pass"] == 1


def test_run_inference_step_global_auction_executes_best_cross_side_first(monkeypatch):
    idx = pd.date_range("2026-03-01", periods=3, freq="1h", tz="UTC")
    close = pd.DataFrame(
        {"LONG/USDT": [100.0] * len(idx), "SHORT/USDT": [100.0] * len(idx)},
        index=idx,
    )
    panel = {
        "close": close,
        "high": close,
        "low": close,
        "open": close,
        "volume": close,
    }
    feats = {"ret12h": close.pct_change().fillna(0.0)}

    monkeypatch.setattr(
        ri, "select_candidates", lambda **kwargs: (["LONG/USDT"], ["SHORT/USDT"])
    )
    monkeypatch.setattr(
        ri,
        "get_features_for_candidates",
        lambda feats, candidates: pd.DataFrame(
            {"dummy": [1.0] * len(candidates)}, index=candidates
        ),
    )

    def _gate(decision, **kwargs):
        score = 0.99 if decision["side"] == "short" else 0.80
        decision["policy_rank_pct"] = score
        decision["sizer_rank_percentile"] = score
        decision["threshold_score"] = score
        decision["normalized_rank_score"] = score
        chain = dict(decision.get("chain_results") or {})
        chain["policy_rank_pct"] = score
        chain["sizer_rank_percentile"] = score
        chain["normalized_rank_score"] = score
        decision["chain_results"] = chain
        return True, None

    monkeypatch.setattr(ri, "apply_policy_rank_percentile_gate", _gate)

    executor = _DummyExecutor()
    results = ri.run_inference_step(
        orchestrator=_DummyOrchestrator(),
        panel=panel,
        feats=feats,
        thresholds={"metric": "ret12h"},
        executor=executor,
        logger=_DummyLogger(),
        accepted_strategies={"long_mr", "short_mr"},
        calibration_data={
            "long_mr": {
                "p75_threshold": 0.5,
                "calibration_curve": [(0.0, 0.0), (1.0, 1.0)],
            },
            "short_mr": {
                "p75_threshold": 0.5,
                "calibration_curve": [(0.0, 0.0), (1.0, 1.0)],
            },
        },
        portfolio_mgr=PortfolioManager(portfolio_value=10000.0, max_positions=1),
        initial_rank_threshold=0.5,
        max_entries_total=1,
    )

    assert [call["symbol"] for call in executor.calls] == ["SHORT/USDT"]
    assert results["trades"][0]["side"] == "short"


def test_run_inference_step_global_auction_fails_closed_when_one_side_errors(
    monkeypatch, tmp_path
):
    idx = pd.date_range("2026-03-01", periods=3, freq="1h", tz="UTC")
    close = pd.DataFrame(
        {"LONG/USDT": [100.0] * len(idx), "SHORT/USDT": [100.0] * len(idx)},
        index=idx,
    )
    panel = {name: close for name in ("close", "high", "low", "open", "volume")}
    feats = {"ret12h": close.pct_change().fillna(0.0)}
    monkeypatch.setattr(
        ri, "select_candidates", lambda **kwargs: (["LONG/USDT"], ["SHORT/USDT"])
    )
    monkeypatch.setattr(
        ri,
        "get_features_for_candidates",
        lambda feats, candidates: pd.DataFrame(
            {"dummy": [1.0] * len(candidates)}, index=candidates
        ),
    )
    monkeypatch.setattr(
        ri,
        "apply_policy_rank_percentile_gate",
        lambda decision, **kwargs: (True, None),
    )

    class _OneSideFailureOrchestrator(_DummyOrchestrator):
        def predict_alpha(self, features, side, kind):
            if side == "short":
                raise RuntimeError("synthetic short-side failure")
            return super().predict_alpha(features, side, kind)

    executor = _DummyExecutor()
    executor.config["entry_budget_guard_path"] = str(tmp_path / "budget.sqlite")
    results = ri.run_inference_step(
        orchestrator=_OneSideFailureOrchestrator(),
        panel=panel,
        feats=feats,
        thresholds={"metric": "ret12h"},
        executor=executor,
        logger=_DummyLogger(),
        accepted_strategies={"long_mr", "short_mr"},
        calibration_data={
            side: {
                "p75_threshold": 0.5,
                "calibration_curve": [(0.0, 0.0), (1.0, 1.0)],
            }
            for side in ("long_mr", "short_mr")
        },
        portfolio_mgr=PortfolioManager(portfolio_value=10000.0),
        initial_rank_threshold=0.5,
        max_entries_total=1,
    )

    assert executor.calls == []
    assert results["global_auction_completeness"]["complete"] is False
    assert results["global_auction_completeness"]["completed_sides"] == ["long"]
    assert "short" in results["global_auction_completeness"]["failed_sides"]


def test_run_inference_step_rejects_portfolio_strategy_contract_mismatch(monkeypatch):
    idx = pd.date_range("2026-03-01", periods=3, freq="1h", tz="UTC")
    close = pd.DataFrame({"LONG/USDT": [100.0] * len(idx)}, index=idx)
    panel = {
        "close": close,
        "high": close,
        "low": close,
        "open": close,
        "volume": close,
    }
    feats = {"ret12h": close.pct_change().fillna(0.0)}

    monkeypatch.setattr(ri, "select_candidates", lambda **kwargs: (["LONG/USDT"], []))

    with pytest.raises(ValueError, match="Portfolio strategy contract mismatch"):
        ri.run_inference_step(
            orchestrator=_DummyOrchestrator(),
            panel=panel,
            feats=feats,
            thresholds={"metric": "ret12h"},
            executor=_DummyExecutor(),
            logger=_DummyLogger(),
            accepted_strategies={"long_mr"},
            portfolio_policy=ri.PortfolioPolicyConfig(strategy_ids=("short_mr",)),
        )


def test_portfolio_contract_strategy_filter_overrides_manifest_filter():
    policy = ri.PortfolioPolicyConfig(strategy_ids=("long_contract", "short_contract"))

    selected = ri._resolve_portfolio_contract_strategy_filter(
        policy,
        {"long_manifest"},
    )

    assert selected == {"long_contract", "short_contract"}


def test_portfolio_contract_strategy_filter_can_use_cores_without_manifest():
    policy = ri.PortfolioPolicyConfig(strategy_cores=("core_a", "core_b"))

    selected = ri._resolve_portfolio_contract_strategy_filter(policy, set())

    assert selected == {"core_a", "core_b"}


def test_run_inference_step_global_auction_keeps_ticker_liquidity_gate(monkeypatch):
    idx = pd.date_range("2026-03-01", periods=3, freq="1h", tz="UTC")
    close = pd.DataFrame({"WIDE/USDT": [100.0] * len(idx)}, index=idx)
    panel = {
        "close": close,
        "high": close,
        "low": close,
        "open": close,
        "volume": close,
    }
    feats = {"ret12h": close.pct_change().fillna(0.0)}

    monkeypatch.setattr(ri, "select_candidates", lambda **kwargs: (["WIDE/USDT"], []))
    monkeypatch.setattr(
        ri,
        "get_features_for_candidates",
        lambda feats, candidates: pd.DataFrame(
            {"dummy": [1.0] * len(candidates)}, index=candidates
        ),
    )

    class _WideSpreadExchange:
        markets = {"WIDE/USDT": {"limits": {"cost": {"min": 1.0}}}}

        def fetch_ticker(self, symbol):
            return {"bid": 100.0, "ask": 103.0, "last": 101.5}

        def fetch_order_book(self, symbol):
            return {"asks": [[103.0, 100.0]], "bids": [[100.0, 100.0]]}

        def market(self, symbol):
            return self.markets[symbol]

    executor = _DummyExecutor()
    executor.exchange = _WideSpreadExchange()
    executor.config["allow_live_batch_rank_fallback_for_debug"] = True

    def _gate(decision, **kwargs):
        score = float(decision.get("calibrated_score") or 0.9)
        decision["threshold_rank_score"] = score
        decision["policy_rank_pct"] = score
        decision["normalized_rank_score"] = score
        chain = dict(decision.get("chain_results") or {})
        chain["policy_rank_pct"] = score
        chain["sizer_rank_percentile"] = score
        chain["normalized_rank_score"] = score
        decision["chain_results"] = chain
        return True, None

    monkeypatch.setattr(ri, "apply_policy_rank_percentile_gate", _gate)

    results = ri.run_inference_step(
        orchestrator=_DummyOrchestrator(),
        panel=panel,
        feats=feats,
        thresholds={"metric": "ret12h"},
        executor=executor,
        logger=_DummyLogger(),
        accepted_strategies={"long_mr"},
        calibration_data={
            "mr": {
                "p75_threshold": 0.5,
                "calibration_curve": [(0.0, 0.0), (1.0, 1.0)],
            }
        },
        portfolio_mgr=PortfolioManager(portfolio_value=10000.0),
        initial_rank_threshold=0.5,
    )

    assert executor.calls == []
    assert results["trades"] == []
    assert results["side_metrics"]["long"]["non_fatal_issues"] >= 1


def test_run_inference_step_blocks_strategy_kill_switch(monkeypatch, tmp_path):
    idx = pd.date_range("2026-03-01", periods=3, freq="1h", tz="UTC")
    close = pd.DataFrame({"BTC/USDT": [100.0, 100.0, 100.0]}, index=idx)
    panel = {
        "close": close,
        "high": close,
        "low": close,
        "open": close,
        "volume": close,
    }
    feats = {"ret12h": close.pct_change().fillna(0.0)}

    monkeypatch.setattr(ri, "select_candidates", lambda **kwargs: (["BTC/USDT"], []))
    monkeypatch.setattr(
        ri,
        "get_features_for_candidates",
        lambda feats, candidates: pd.DataFrame(
            {"dummy": [1.0] * len(candidates)}, index=candidates
        ),
    )
    switch = StrategyKillSwitch(
        tmp_path / "strategy_kill_switches.json",
        observe_only=False,
    )
    switch.set_state("long_mr", active=True, reason="weak_hit_rate")
    executor = _DummyExecutor()

    results = ri.run_inference_step(
        orchestrator=_DummyOrchestrator(),
        panel=panel,
        feats=feats,
        thresholds={"metric": "ret12h"},
        executor=executor,
        logger=_DummyLogger(),
        accepted_strategies={"long_mr"},
        calibration_data={
            "long_mr": {
                "p75_threshold": 0.6,
                "calibration_curve": [(0.0, 0.0), (1.0, 1.0)],
            }
        },
        portfolio_mgr=PortfolioManager(portfolio_value=10000.0),
        initial_rank_threshold=0.5,
        strategy_kill_switch=switch,
    )

    assert results["trades"] == []
    assert executor.calls == []


def test_run_inference_step_allows_unblocked_strategy_kill_switch(
    monkeypatch, tmp_path
):
    idx = pd.date_range("2026-03-01", periods=3, freq="1h", tz="UTC")
    close = pd.DataFrame({"BTC/USDT": [100.0, 100.0, 100.0]}, index=idx)
    panel = {
        "close": close,
        "high": close,
        "low": close,
        "open": close,
        "volume": close,
    }
    feats = {"ret12h": close.pct_change().fillna(0.0)}

    monkeypatch.setattr(ri, "select_candidates", lambda **kwargs: (["BTC/USDT"], []))
    monkeypatch.setattr(
        ri,
        "get_features_for_candidates",
        lambda feats, candidates: pd.DataFrame(
            {"dummy": [1.0] * len(candidates)}, index=candidates
        ),
    )
    switch = StrategyKillSwitch(
        tmp_path / "strategy_kill_switches.json",
        observe_only=False,
    )
    executor = _DummyExecutor()

    results = ri.run_inference_step(
        orchestrator=_DummyOrchestrator(),
        panel=panel,
        feats=feats,
        thresholds={"metric": "ret12h"},
        executor=executor,
        logger=_DummyLogger(),
        accepted_strategies={"long_mr"},
        calibration_data={
            "long_mr": {
                "p75_threshold": 0.6,
                "calibration_curve": [(0.0, 0.0), (1.0, 1.0)],
            }
        },
        portfolio_mgr=PortfolioManager(portfolio_value=10000.0),
        initial_rank_threshold=0.5,
        strategy_kill_switch=switch,
    )

    assert len(results["trades"]) == 1
    assert executor.calls, "expected unblocked strategy to continue to execution"


def test_run_inference_step_sizes_from_calibrated_meta_policy_power(monkeypatch):
    idx = pd.date_range("2026-03-01", periods=3, freq="1h", tz="UTC")
    close = pd.DataFrame({"BTC/USDT": [100.0, 100.0, 100.0]}, index=idx)
    panel = {
        "close": close,
        "high": close,
        "low": close,
        "open": close,
        "volume": close,
    }
    feats = {"ret12h": close.pct_change().fillna(0.0)}

    monkeypatch.setattr(ri, "select_candidates", lambda **kwargs: (["BTC/USDT"], []))
    monkeypatch.setattr(
        ri,
        "get_features_for_candidates",
        lambda feats, candidates: pd.DataFrame(
            {"dummy": [1.0] * len(candidates)}, index=candidates
        ),
    )

    class _PolicySizingOrchestrator(_DummyOrchestrator):
        bucket_params = {
            "long_mr": {
                "best_size_power": 2.0,
                "asset_metrics": [
                    {
                        "symbol": "BTC/USDT",
                        "asset_decision": "down_weight",
                        "asset_weight_multiplier": 0.5,
                    }
                ],
            }
        }

        def run_full_chain(self, symbol, side, features, panel=None, kind=None):
            out = super().run_full_chain(symbol, side, features, panel=panel)
            out["position_size"] = 9999.0
            out["orchestrator_position_size"] = 9999.0
            return out

    executor = _DummyExecutor()
    portfolio_mgr = PortfolioManager(portfolio_value=10000.0)
    portfolio_policy = ri.PortfolioPolicyConfig()

    results = ri.run_inference_step(
        orchestrator=_PolicySizingOrchestrator(),
        panel=panel,
        feats=feats,
        thresholds={"metric": "ret12h"},
        executor=executor,
        logger=_DummyLogger(),
        accepted_strategies={"long_mr"},
        calibration_data={
            "long_mr": {
                "p75_threshold": 0.5,
                "calibration_curve": [(0.0, 0.0), (1.0, 1.0)],
            }
        },
        portfolio_mgr=portfolio_mgr,
        portfolio_policy=portfolio_policy,
    )

    assert len(results["trades"]) == 1
    # With PortfolioManager active, live rank-sizing supersedes legacy
    # calibrated policy sizing and symbol-underperformance downweights. A
    # single-candidate auction rank is 1.0, so the default global-auction policy
    # fills one reserved slot, capped by max_position_wallet_pct.
    expected_size = min(
        10000.0 * portfolio_policy.max_position_wallet_pct,
        portfolio_policy.max_position_quote_notional,
    )
    assert abs(executor.calls[0]["size"] - expected_size) < 1e-9


def test_portfolio_manager_hard_gates_require_manual_reset():
    portfolio_mgr = PortfolioManager(
        portfolio_value=10000.0,
        max_consecutive_losing_trades=10,
        max_consecutive_losing_trades_per_archetype=0,
    )
    now = pd.Timestamp("2026-03-01 00:00", tz="UTC")
    for i in range(10):
        symbol = f"LOSS{i}/USDT"
        portfolio_mgr.record_position_open(
            symbol=symbol,
            side="long",
            strategy_id="long_mr",
            position_size=100.0,
            entry_price=100.0,
            entry_time=now + pd.Timedelta(minutes=i),
        )
        portfolio_mgr.record_position_close(
            symbol=symbol,
            exit_price=99.0,
            exit_time=now + pd.Timedelta(minutes=i, seconds=1),
            exit_reason="test_loss",
        )

    allowed, info = portfolio_mgr.can_enter_position(
        symbol="NEXT/USDT",
        side="long",
        strategy_id="long_mr",
        confidence_score=1.0,
        initial_threshold=0.5,
        current_time=now + pd.Timedelta(minutes=10),
        requested_position_size=100.0,
    )
    assert not allowed
    assert info["hard_limits"]["manual_reset_required"]
    assert "consecutive_losing_trades" in info["hard_limits"]["hard_limit_reason"]

    portfolio_mgr.manual_reset_hard_limits()
    allowed_after_reset, _ = portfolio_mgr.can_enter_position(
        symbol="NEXT/USDT",
        side="long",
        strategy_id="long_mr",
        confidence_score=1.0,
        initial_threshold=0.5,
        current_time=now + pd.Timedelta(minutes=11),
        requested_position_size=100.0,
    )
    assert allowed_after_reset


def test_run_inference_step_blocks_non_accepted_strategy(monkeypatch):
    idx = pd.date_range("2026-03-01", periods=3, freq="1h", tz="UTC")
    close = pd.DataFrame({"BTC/USDT": [100.0, 100.0, 100.0]}, index=idx)
    panel = {
        "close": close,
        "high": close,
        "low": close,
        "open": close,
        "volume": close,
    }
    feats = {"ret12h": close.pct_change().fillna(0.0)}

    monkeypatch.setattr(ri, "select_candidates", lambda **kwargs: (["BTC/USDT"], []))
    monkeypatch.setattr(
        ri,
        "get_features_for_candidates",
        lambda feats, candidates: pd.DataFrame(
            {"dummy": [1.0] * len(candidates)}, index=candidates
        ),
    )

    orchestrator = _DummyOrchestrator()
    executor = _DummyExecutor()
    logger = _DummyLogger()

    results = ri.run_inference_step(
        orchestrator=orchestrator,
        panel=panel,
        feats=feats,
        thresholds={
            "extreme_pct": None,
            "min_move_12h_pct": None,
            "min_range_pct": None,
            "min_vol_zscore": None,
            "metric": "ret12h",
        },
        executor=executor,
        logger=logger,
        accepted_strategies={"short_mr"},
        calibration_data={},
    )

    assert not executor.calls
    assert results["trades"] == []


def test_run_inference_step_can_force_shadow_entry_for_integration(monkeypatch):
    idx = pd.date_range("2026-03-01", periods=3, freq="1h", tz="UTC")
    close = pd.DataFrame({"BTC/USDT": [100.0, 100.0, 100.0]}, index=idx)
    panel = {
        "close": close,
        "high": close,
        "low": close,
        "open": close,
        "volume": close,
    }
    feats = {"ret12h": close.pct_change().fillna(0.0)}

    monkeypatch.setattr(ri, "select_candidates", lambda **kwargs: (["BTC/USDT"], []))
    monkeypatch.setattr(
        ri,
        "get_features_for_candidates",
        lambda feats, candidates: pd.DataFrame(
            {"dummy": [1.0] * len(candidates)}, index=candidates
        ),
    )

    executor = _DummyExecutor()
    executor.config["force_shadow_entry_for_integration"] = True
    logger = _DummyLogger()

    results = ri.run_inference_step(
        orchestrator=_NoEntryOrchestrator(),
        panel=panel,
        feats=feats,
        thresholds={"metric": "ret12h"},
        executor=executor,
        logger=logger,
        accepted_strategies={"long_mr"},
        calibration_data={
            "long_mr": {
                "p75_threshold": 0.5,
                "calibration_curve": [(0.0, 0.0), (1.0, 1.0)],
            }
        },
    )

    assert len(results["trades"]) == 1
    assert executor.calls


def test_run_inference_step_blocks_policy_excluded_asset(monkeypatch):
    idx = pd.date_range("2026-03-01", periods=3, freq="1h", tz="UTC")
    close = pd.DataFrame({"BTC/USDT": [100.0, 100.0, 100.0]}, index=idx)
    panel = {
        "close": close,
        "high": close,
        "low": close,
        "open": close,
        "volume": close,
    }
    feats = {"ret12h": close.pct_change().fillna(0.0)}

    monkeypatch.setattr(ri, "select_candidates", lambda **kwargs: (["BTC/USDT"], []))
    monkeypatch.setattr(
        ri,
        "get_features_for_candidates",
        lambda feats, candidates: pd.DataFrame(
            {"dummy": [1.0] * len(candidates)}, index=candidates
        ),
    )

    executor = _DummyExecutor()
    results = ri.run_inference_step(
        orchestrator=_DummyOrchestrator(),
        panel=panel,
        feats=feats,
        thresholds={"metric": "ret12h"},
        executor=executor,
        logger=_DummyLogger(),
        accepted_strategies={"long_mr"},
        strategy_asset_exclusions={"long_mr": {"BTC/USDT"}},
    )

    assert not executor.calls
    assert results["trades"] == []


def test_run_inference_step_ranks_after_policy_asset_exclusions(monkeypatch):
    idx = pd.date_range("2026-03-01", periods=3, freq="1h", tz="UTC")
    symbols = ["BLOCKED/USDT", "OK/USDT"]
    close = pd.DataFrame(
        {symbol: [100.0, 100.0, 100.0] for symbol in symbols},
        index=idx,
    )
    panel = {
        "close": close,
        "high": close,
        "low": close,
        "open": close,
        "volume": close,
    }
    feats = {"ret12h": close.pct_change().fillna(0.0)}

    monkeypatch.setattr(ri, "select_candidates", lambda **kwargs: (symbols, []))
    monkeypatch.setattr(
        ri,
        "get_features_for_candidates",
        lambda feats, candidates: pd.DataFrame(
            {"dummy": list(range(len(candidates)))}, index=candidates
        ),
    )

    class _ExclusionAwareOrchestrator:
        def __init__(self):
            self.full_chain_symbols = []

        def available_strategies(self, side, accepted=None):
            return ["long_mr"]

        def predict_alpha(self, features, side, kind):
            return pd.Series(
                [0.99 if idx == "BLOCKED/USDT" else 0.5 for idx in features.index],
                index=features.index,
            )

        def run_full_chain(self, symbol, side, features, panel=None, kind=None):
            self.full_chain_symbols.append(symbol)
            return {
                "symbol": symbol,
                "side": side,
                "action": "enter",
                "position_size": 100.0,
                "meta_pred": 0.9,
                "strategy_id": "long_mr",
            }

    orchestrator = _ExclusionAwareOrchestrator()
    executor = _DummyExecutor()
    results = ri.run_inference_step(
        orchestrator=orchestrator,
        panel=panel,
        feats=feats,
        thresholds={"metric": "ret12h"},
        executor=executor,
        logger=_DummyLogger(),
        accepted_strategies={"long_mr"},
        strategy_asset_exclusions={"long_mr": {"BLOCKED/USDT"}},
    )

    assert orchestrator.full_chain_symbols == ["OK/USDT"]
    assert [call["symbol"] for call in executor.calls] == ["OK/USDT"]
    assert results["trades"][0]["symbol"] == "OK/USDT"


def test_run_inference_step_gates_meta_to_top_quartile_base_preds(monkeypatch):
    idx = pd.date_range("2026-03-01", periods=3, freq="1h", tz="UTC")
    symbols = ["A/USDT", "B/USDT", "C/USDT", "D/USDT"]
    close = pd.DataFrame(
        {symbol: [100.0, 100.0, 100.0] for symbol in symbols},
        index=idx,
    )
    panel = {
        "close": close,
        "high": close,
        "low": close,
        "open": close,
        "volume": close,
    }
    feats = {"ret12h": close.pct_change().fillna(0.0)}

    monkeypatch.setattr(ri, "select_candidates", lambda **kwargs: (symbols, []))
    monkeypatch.setattr(
        ri,
        "get_features_for_candidates",
        lambda feats, candidates: pd.DataFrame(
            {"dummy": list(range(len(candidates)))}, index=candidates
        ),
    )

    class _GatedOrchestrator:
        def __init__(self):
            self.full_chain_symbols = []

        def available_strategies(self, side, accepted=None):
            return ["long_mr"]

        def predict_alpha(self, features, side, kind):
            return pd.Series(
                [0.1, 0.9, 0.2, 0.3],
                index=["A/USDT", "B/USDT", "C/USDT", "D/USDT"],
            )

        def run_full_chain(self, symbol, side, features, panel=None, kind=None):
            self.full_chain_symbols.append(symbol)
            return {
                "symbol": symbol,
                "side": side,
                "action": "enter",
                "position_size": 100.0,
                "meta_pred": 0.9,
                "strategy_id": "long_mr",
            }

    orchestrator = _GatedOrchestrator()
    executor = _DummyExecutor()
    logger = _DummyLogger()

    results = ri.run_inference_step(
        orchestrator=orchestrator,
        panel=panel,
        feats=feats,
        thresholds={"metric": "ret12h"},
        executor=executor,
        logger=logger,
        accepted_strategies={"long_mr"},
        calibration_data={},
    )

    assert orchestrator.full_chain_symbols == ["B/USDT", "D/USDT"]
    assert [call["symbol"] for call in executor.calls] == ["B/USDT", "D/USDT"]
    assert results["trades"][0]["symbol"] == "B/USDT"


def test_run_inference_step_reuses_batch_meta_without_full_chain(monkeypatch):
    idx = pd.date_range("2026-03-01", periods=3, freq="1h", tz="UTC")
    symbols = ["A/USDT", "B/USDT"]
    close = pd.DataFrame(
        {symbol: [100.0, 100.0, 100.0] for symbol in symbols},
        index=idx,
    )
    panel = {
        "close": close,
        "high": close,
        "low": close,
        "open": close,
        "volume": close,
    }
    feats = {"ret12h": close.pct_change().fillna(0.0)}

    monkeypatch.setattr(ri, "select_candidates", lambda **kwargs: (symbols, []))
    monkeypatch.setattr(
        ri,
        "get_features_for_candidates",
        lambda feats, candidates: pd.DataFrame(
            {"dummy": [1.0] * len(candidates)}, index=candidates
        ),
    )

    class _BatchOnlyOrchestrator:
        bucket_params = {}
        alpha_by_strategy = {"long_mr": {"feat_cols": ["dummy"]}}

        def __init__(self):
            self.run_full_chain_calls = 0
            self._last_meta_model_input = pd.DataFrame()
            self._last_meta_diagnostics_frame = pd.DataFrame()

        def available_strategies(self, side, accepted=None):
            return ["long_mr"]

        def _align_alpha_feature_contract(self, features, feat_cols):
            return features.reindex(columns=feat_cols)

        def predict_alpha(self, features, side, kind):
            return pd.Series(0.9, index=features.index, dtype=float)

        def predict_meta(self, features, side, kind):
            self._last_meta_model_input = features.copy()
            self._last_meta_diagnostics_frame = pd.DataFrame(
                {"prob_uncertainty": [0.10] * len(features.index)},
                index=features.index,
            )
            return pd.Series(0.95, index=features.index, dtype=float)

        def run_full_chain(self, symbol, side, features, panel=None, kind=None):
            self.run_full_chain_calls += 1
            raise AssertionError("run_full_chain should not be called")

    orchestrator = _BatchOnlyOrchestrator()
    executor = _DummyExecutor()

    results = ri.run_inference_step(
        orchestrator=orchestrator,
        panel=panel,
        feats=feats,
        thresholds={"metric": "ret12h"},
        executor=executor,
        logger=_DummyLogger(),
        accepted_strategies={"long_mr"},
        calibration_data={
            "long_mr": {
                "p75_threshold": 0.5,
                "calibration_curve": [(0.0, 0.0), (1.0, 1.0)],
            }
        },
    )

    assert orchestrator.run_full_chain_calls == 0
    assert results["side_metrics"]["long"]["base_gate_pass"] == 1


def test_trade_execution_health_records_rejections_and_api_failures():
    portfolio_mgr = PortfolioManager(max_consecutive_order_rejections=5)

    ri._record_trade_execution_health(
        portfolio_mgr,
        {
            "success": False,
            "error_category": "duplicate_client_order_id",
            "error": "Duplicate clientOrderId was sent",
        },
    )
    assert portfolio_mgr.consecutive_order_rejections == 1
    assert portfolio_mgr.order_rejection_backoff_until is not None

    ri._record_trade_execution_health(
        portfolio_mgr,
        {
            "success": False,
            "error_category": "network_timeout",
            "error": "network timeout while sending order",
        },
    )
    assert len(portfolio_mgr.failed_api_events) == 1


def test_live_policy_archetype_classifier_fallback_normalizes_side_prefix():
    class _Classifier:
        def predict(self, frame):
            assert "__live_side_is_long" in frame.columns
            assert "__live_side_is_short" in frame.columns
            assert float(frame["feature_x"].iloc[0]) == 1.25
            return ["long_mixed_wideslow_tentative"]

    payload = {
        "model": _Classifier(),
        "feature_columns": ["feature_x", "__live_side_is_long", "__live_side_is_short"],
        "feature_medians": {"feature_x": 0.0},
        "side_defaults": {"long": "long__long_mixed_wideslow_tentative"},
    }
    predicted = ri.predict_live_policy_archetype(
        side="long",
        payload=payload,
        candidate_feature_row=pd.DataFrame({"feature_x": [1.25]}, index=["BTC/USD:USD"]),
        meta_model_input_row=None,
    )
    assert predicted == "long__long_mixed_wideslow_tentative"


def test_observable_policy_archetype_matches_label_time_family_contract():
    scores = {
        "__regime_source_trend_following_score__": 0.31,
        "__regime_source_mean_reversion_score__": 0.44,
        "__regime_source_vol_compression_score__": 0.62,
        "__regime_source_breakout_impulse_score__": 0.41,
        "__regime_source_dirty_avoid_score__": 0.27,
    }
    predicted = ri.predict_observable_policy_archetype(
        side="long",
        candidate_feature_row=pd.DataFrame([scores]),
    )
    assert predicted == "long__long_volcompression_wideslow_candidate"


def test_observable_policy_archetype_uses_mixed_for_small_score_gap():
    scores = {
        "__regime_source_trend_following_score__": 0.61,
        "__regime_source_mean_reversion_score__": 0.60,
        "__regime_source_vol_compression_score__": 0.20,
        "__regime_source_breakout_impulse_score__": 0.19,
        "__regime_source_dirty_avoid_score__": 0.18,
    }
    assert (
        ri.predict_observable_policy_archetype(
            side="short",
            candidate_feature_row=pd.DataFrame([scores]),
        )
        == "short__short_mixed_clean_path"
    )


def test_live_policy_archetype_prefers_existing_row_value():
    row = pd.DataFrame(
        {"policy_archetype": ["short__short_breakout_precision"]},
        index=["ETH/USD:USD"],
    )
    assert (
        ri._infer_live_policy_archetype(
            side="short",
            chain_results={},
            candidate_feature_row=row,
            meta_model_input_row=None,
        )
        == "short__short_breakout_precision"
    )
