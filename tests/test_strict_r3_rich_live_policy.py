"""Focused parity tests for the promoted strict-R3 rich exit branch."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.inference.strict_r3_live_execution import (
    StrictR3ExecutionContract,
    _advance_rich_policy_position,
    _conservative_headroom_fallback_snapshot,
    _confirmed_exchange_absent_exit,
    _entry_leverage_from_policy_stop,
    _best_directional_quote_from_book,
    _expected_sell_vwap,
    _fetch_exchange_positions,
    _frozen_policy_vwap_sentinel_evidence,
    _protective_stop_trigger_for_exit_vwap,
    _rich_geometry,
    _rich_policy_params,
)
from extreme_price_movements.strict_r3_rich_policy import RichPolicyParams


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / (
    "data_perp/artifacts/strict_r3_rich_policy_hpo_long_20260817_v1/"
    "frozen_challenger.json"
)


def _payload() -> dict:
    return json.loads(FROZEN.read_text())


def _position(*, maximum_favourable: float = 0.0) -> dict:
    return {
        "entry_price": 100.0,
        "atr": 1.7935832787339956,
        "entry_ts": "2026-08-17T00:00:00Z",
        "timeout_ts": "2026-08-17T12:00:00Z",
        "next_bar_ts": "2026-08-17T00:00:00Z",
        "maximum_favourable": maximum_favourable,
        "maximum_adverse": 0.0,
        "trailing_armed": False,
        "capital_protect_armed": False,
        "rich_adaptive_activation_multiplier": 1.0,
    }


def _bars(rows: list[tuple[str, float, float, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["timestamp", "high", "low", "close"])
    return frame.set_index(pd.to_datetime(frame.pop("timestamp"), utc=True))


def test_rich_geometry_uses_distinct_transforms_and_absolute_stop_cap() -> None:
    params, median = _rich_policy_params(_payload())
    geometry = _rich_geometry(
        entry_price=100.0, atr=median * 100.0,
        params=params, median_atr_fraction=median,
    )
    assert np.isclose(geometry["sl_barrier_frac"], 1.25 * median)
    assert np.isclose(geometry["tp_barrier_frac"], 1.25 * median)
    # The selected 4.38 ATR stop would exceed 5%; its frozen absolute cap is
    # therefore active at the entry boundary.
    assert np.isclose(geometry["sl_distance"], 5.0)


def test_entry_stop_vwap_uses_the_same_book_snapshot_as_exit_vwap() -> None:
    """A ticker that moves after the book must not abort a valid entry.

    The previous implementation compared an exit VWAP from the order-book
    response with a separately fetched ticker bid.  A fast rebound can make
    that stale ticker bid lower than the still-valid book VWAP.  The native
    stop trigger must instead derive both values from the same book snapshot.
    """
    bids = [[101.0, 10.0], [100.5, 10.0]]
    exit_vwap = _expected_sell_vwap(bids, required_contracts=5.0)
    book_best_bid = _best_directional_quote_from_book(bids, side="long")
    stop = _protective_stop_trigger_for_exit_vwap(
        policy_exit_price=95.0,
        best_directional_quote=book_best_bid,
        expected_exit_vwap=exit_vwap,
        enabled=True,
        side="long",
    )
    assert exit_vwap == pytest.approx(101.0)
    assert book_best_bid == pytest.approx(101.0)
    assert stop["exchange_trigger_stop_price"] == pytest.approx(95.0)


def test_kraken_reconciliation_uses_direct_openpositions_not_stale_ccxt_rows() -> None:
    """A stale adapter row must never reserve inventory or block a candle."""
    class _Exchange:
        id = "krakenfutures"
        markets_by_id = {"PF_ACEUSD": [{"symbol": "ACE/USD:USD"}]}

        @staticmethod
        def fetch_positions() -> list[dict]:
            return [{"symbol": "GHOST/USD:USD", "contracts": 10, "side": "long"}]

        @staticmethod
        def privateGetOpenpositions(_: dict) -> dict:
            return {
                "result": "success",
                "openPositions": [{"symbol": "PF_ACEUSD", "size": "3", "side": "long"}],
            }

    positions = _fetch_exchange_positions(_Exchange(), expected_side="long")
    assert list(positions) == ["ACE/USD:USD"]
    assert float(positions["ACE/USD:USD"]["contracts"]) == pytest.approx(3.0)


def test_inverse_stop_leverage_uses_the_declared_absolute_percent_formula(
    tmp_path: Path,
) -> None:
    policy = tmp_path / "simple_policy.json"
    policy.write_text(json.dumps({"winner": {"sl_mult": 3.0}}))
    contract = SimpleNamespace(
        leverage_sizing_mode="inverse_policy_stop_absolute_pct",
        leverage_risk_budget_pct=66.0,
        leverage_maximum=10.0,
        leverage=10.0,
        exit_policy=policy,
    )
    leverage, details = _entry_leverage_from_policy_stop(
        contract=contract,
        shadow_position={"atr": 10.0},
        reference_price=100.0,
    )
    # Stop is 30%; min(10, 66 / 30) = 2.2x.
    assert leverage == pytest.approx(2.2)
    assert details["policy_stop_absolute_pct"] == pytest.approx(30.0)
    assert details["unclipped_leverage"] == pytest.approx(2.2)


def test_inverse_stop_liquidation_headroom_caps_before_maintenance_breach(
    tmp_path: Path,
) -> None:
    policy = tmp_path / "simple_policy.json"
    policy.write_text(json.dumps({"winner": {"sl_mult": 5.0}}))
    contract = SimpleNamespace(
        leverage_sizing_mode="inverse_policy_stop_absolute_pct",
        leverage_risk_budget_pct=66.0,
        leverage_maximum=10.0,
        leverage=10.0,
        leverage_liquidation_headroom_enabled=True,
        leverage_minimum_margin_level_after_stress=1.5,
        leverage_stressed_exit_slippage_bps=50.0,
        leverage_margin_schedule_platform="europa",
        leverage_margin_schedule_account_class="retail",
        exit_policy=policy,
    )
    # The generic public field intentionally advertises the wrong, looser
    # 5%/2.5% tier.  Production sizing must instead use the account's actual
    # Europa retail isolated schedule (10%/5%).
    leverage, details = _entry_leverage_from_policy_stop(
        contract=contract,
        shadow_position={"atr": 1.0},
        reference_price=100.0,
        slot_margin=100.0,
        preflight={
            "amount": 100.0,
            "market": {
                "info": {
                    "retailMarginLevels": [{
                        "numNonContractUnits": "0",
                        "initialMargin": "0.05",
                        "maintenanceMargin": "0.025",
                    }],
                    "marginSchedules": {"europa": {"retail": [{
                        "numNonContractUnits": "0",
                        "initialMargin": "0.1",
                        "maintenanceMargin": "0.05",
                    }]}},
                }
            },
        },
    )
    assert leverage == pytest.approx(1.0 / (0.055 + 1.5 * 0.05))
    assert details["inverse_stop_requested_leverage"] == pytest.approx(10.0)
    assert details["liquidation_headroom_capped"] is True
    assert details["margin_tier_source"] == "marginSchedules.europa.retail"
    assert details["projected_isolated_margin_level_at_stressed_stop"] == pytest.approx(1.5)


def test_liquidation_headroom_fails_closed_without_fresh_margin_inputs(
    tmp_path: Path,
) -> None:
    policy = tmp_path / "simple_policy.json"
    policy.write_text(json.dumps({"winner": {"sl_mult": 3.0}}))
    contract = SimpleNamespace(
        leverage_sizing_mode="inverse_policy_stop_absolute_pct",
        leverage_risk_budget_pct=66.0,
        leverage_maximum=10.0,
        leverage=10.0,
        leverage_liquidation_headroom_enabled=True,
        leverage_minimum_margin_level_after_stress=1.5,
        leverage_stressed_exit_slippage_bps=50.0,
        exit_policy=policy,
    )
    with pytest.raises(ValueError, match="requires fresh preflight and a positive isolated slot margin"):
        _entry_leverage_from_policy_stop(
            contract=contract,
            shadow_position={"atr": 1.0},
            reference_price=100.0,
        )


def test_vwap_sentinel_uses_the_undiscounted_prior_policy_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Entry costs are PnL telemetry, never extra downside risk authority."""
    import extreme_price_movements.inference.strict_r3_live_execution as live

    monkeypatch.setattr(
        live,
        "_full_vwap_hard_stop_evidence",
        lambda *_args, **_kwargs: {
            "available": True,
            "directional_executable_vwap": 99.5,
            "book": {},
        },
    )
    evidence = _frozen_policy_vwap_sentinel_evidence(
        object(),
        position={
            "side": "long",
            "entry_slippage_bps": 80.0,
            "entry_preflight_market_snapshot": {
                "ticker": {"payload": {"bid": 99.0, "ask": 101.0}}
            },
        },
        frozen_policy_threshold_price=100.0,
    )
    assert evidence["entry_execution_discount_bps"] > 80.0
    assert evidence["hard_risk_trigger_discount_bps"] == 0.0
    assert evidence["effective_executable_threshold_price"] == pytest.approx(100.0)
    assert evidence["would_exit"] is True


def test_liquidation_headroom_fallback_caps_at_five_x_with_missing_retail_tier(
    tmp_path: Path,
) -> None:
    policy = tmp_path / "simple_policy.json"
    policy.write_text(json.dumps({"winner": {"sl_mult": 5.0}}))
    contract = StrictR3ExecutionContract(
        inference_bundle=Path("bundle.json"), inference_bundle_sha256="bundle",
        exit_policy=policy, exit_policy_sha256="policy",
        exchange_id="krakenfutures", side="long", leverage=10.0,
        maximum_decision_age_seconds=900, order_submission_authorized=True,
        activation_authorization=None, activation_authorization_sha256=None,
        authorized_after=None, maximum_entry_slippage_bps=100.0,
        maximum_exit_slippage_bps=100.0,
        leverage_sizing_mode="inverse_policy_stop_absolute_pct",
        leverage_risk_budget_pct=66.0, leverage_maximum=10.0,
        leverage_liquidation_headroom_enabled=True,
        leverage_minimum_margin_level_after_stress=1.5,
        leverage_stressed_exit_slippage_bps=50.0,
        leverage_headroom_fallback_enabled=True,
        leverage_headroom_fallback_maximum=5.0,
        leverage_headroom_fallback_max_snapshot_age_seconds=300.0,
        leverage_headroom_fallback_equity_haircut=0.75,
        leverage_headroom_fallback_maintenance_uplift=1.25,
        leverage_headroom_fallback_initial_margin_rate=0.20,
        leverage_headroom_fallback_maintenance_margin_rate=0.10,
    )
    leverage, details = _entry_leverage_from_policy_stop(
        contract=contract,
        shadow_position=_position(), reference_price=100.0, slot_margin=10.0,
        preflight={"market": {"info": {}}, "amount": 1.0},
        flex_margin={
            "margin_equity": 100.0, "maintenance_margin": 0.0,
            "headroom_fallback_active": True,
            "headroom_fallback_reasons": ["fresh_flex_margin_unavailable"],
        },
    )
    assert leverage == pytest.approx(5.0)
    assert details["liquidation_headroom_fallback_active"] is True
    assert details["margin_tier_source"] == "configured_conservative_fallback_tier"
    assert details["mode"] == "fixed_5x_missing_margin_or_tier_fallback"
    assert details["liquidation_safe_leverage"] is None


def test_liquidation_headroom_missing_maintenance_uses_fixed_five_x(
    tmp_path: Path,
) -> None:
    policy = tmp_path / "simple_policy.json"
    policy.write_text(json.dumps({"winner": {"sl_mult": 5.0}}))
    contract = StrictR3ExecutionContract(
        inference_bundle=Path("bundle.json"), inference_bundle_sha256="bundle",
        exit_policy=policy, exit_policy_sha256="policy",
        exchange_id="krakenfutures", side="long", leverage=10.0,
        maximum_decision_age_seconds=900, order_submission_authorized=True,
        activation_authorization=None, activation_authorization_sha256=None,
        authorized_after=None, maximum_entry_slippage_bps=100.0,
        maximum_exit_slippage_bps=100.0,
        leverage_sizing_mode="inverse_policy_stop_absolute_pct",
        leverage_risk_budget_pct=66.0, leverage_maximum=10.0,
        leverage_liquidation_headroom_enabled=True,
        leverage_minimum_margin_level_after_stress=1.5,
        leverage_stressed_exit_slippage_bps=50.0,
        leverage_headroom_fallback_enabled=True,
        leverage_headroom_fallback_maximum=5.0,
    )
    leverage, details = _entry_leverage_from_policy_stop(
        contract=contract,
        shadow_position=_position(), reference_price=100.0, slot_margin=10.0,
        preflight={
            "market": {"info": {"retailMarginLevels": [{
                "numNonContractUnits": "0", "initialMargin": "0.1",
                "maintenanceMargin": "0.05",
            }]}},
            "amount": 1.0,
        },
        flex_margin={
            "headroom_fallback_active": True,
            "headroom_fallback_reasons": ["fresh_maintenance_margin_unavailable"],
        },
    )
    assert leverage == pytest.approx(5.0)
    assert details["mode"] == "fixed_5x_missing_margin_or_tier_fallback"


def test_cached_headroom_fallback_is_haircut_and_time_bounded(tmp_path: Path) -> None:
    policy = tmp_path / "simple_policy.json"
    policy.write_text(json.dumps({"winner": {"sl_mult": 5.0}}))
    contract = StrictR3ExecutionContract(
        inference_bundle=Path("bundle.json"), inference_bundle_sha256="bundle",
        exit_policy=policy, exit_policy_sha256="policy", exchange_id="krakenfutures",
        side="long", leverage=10.0, maximum_decision_age_seconds=900,
        order_submission_authorized=True, activation_authorization=None,
        activation_authorization_sha256=None, authorized_after=None,
        maximum_entry_slippage_bps=100.0, maximum_exit_slippage_bps=100.0,
        leverage_sizing_mode="inverse_policy_stop_absolute_pct",
        leverage_risk_budget_pct=66.0, leverage_maximum=10.0,
        leverage_liquidation_headroom_enabled=True,
        leverage_minimum_margin_level_after_stress=1.5,
        leverage_stressed_exit_slippage_bps=50.0,
        leverage_headroom_fallback_enabled=True,
        leverage_headroom_fallback_maximum=5.0,
        leverage_headroom_fallback_max_snapshot_age_seconds=300.0,
        leverage_headroom_fallback_equity_haircut=0.75,
        leverage_headroom_fallback_maintenance_uplift=1.25,
        leverage_headroom_fallback_initial_margin_rate=0.20,
        leverage_headroom_fallback_maintenance_margin_rate=0.10,
    )
    snapshot = {
        "captured_at": "2026-08-21T00:00:00Z",
        "margin_equity": 100.0,
        "maintenance_margin": 8.0,
    }
    fallback = _conservative_headroom_fallback_snapshot(
        snapshot=snapshot, contract=contract,
        observed_at="2026-08-21T00:04:59Z", reason="unit_test",
    )
    assert fallback["margin_equity"] == pytest.approx(75.0)
    assert fallback["maintenance_margin"] == pytest.approx(10.0)
    assert fallback["headroom_fallback_active"] is True
    with pytest.raises(ValueError, match="stale"):
        _conservative_headroom_fallback_snapshot(
            snapshot=snapshot, contract=contract,
            observed_at="2026-08-21T00:05:01Z", reason="unit_test",
        )


def test_rich_policy_arms_trailing_from_prior_peak_then_uses_fixed_gap() -> None:
    params, median = _rich_policy_params(_payload())
    first = _advance_rich_policy_position(
        position=_position(maximum_favourable=3.1),
        bars=_bars([("2026-08-17T00:00:00Z", 100.2, 99.9, 100.0)]),
        params=params, median_atr_fraction=median,
    )
    assert first["trailing_armed"] is True
    second_position = {**_position(), **first}
    second = _advance_rich_policy_position(
        position=second_position,
        bars=_bars([("2026-08-17T00:01:00Z", 103.2, 102.5, 102.7)]),
        params=params, median_atr_fraction=median,
    )
    assert second["exit"]["exit_reason"] == "trailing"
    assert second["exit"]["exit_price"] > 100.0


def test_rich_fast_adverse_uses_frozen_theta_before_full_stop() -> None:
    params, median = _rich_policy_params(_payload())
    outcome = _advance_rich_policy_position(
        position=_position(),
        bars=_bars([("2026-08-17T00:00:00Z", 100.0, 97.0, 97.2)]),
        params=params, median_atr_fraction=median,
    )
    assert outcome["exit"]["exit_reason"] == "fast_adverse"
    assert np.isclose(outcome["exit"]["exit_price"], 97.2)


def test_absent_position_reconciles_only_an_exact_external_full_close() -> None:
    class _Exchange:
        def fetch_my_trades(self, symbol: str, *, since: int, limit: int) -> list[dict]:
            assert symbol == "PORTAL/USD:USD"
            return [{
                "side": "sell",
                "amount": 1430.0,
                "price": 0.01734,
                "timestamp": int(pd.Timestamp("2026-08-17T13:29:53Z").timestamp() * 1000),
                "order": "external-close",
                "fillType": "taker",
            }]

    confirmation, kind = _confirmed_exchange_absent_exit(
        _Exchange(),
        position={
            "exchange_symbol": "PORTAL/USD:USD",
            "side": "long",
            "amount": 1430.0,
            "entry_ts": "2026-08-17T10:03:27Z",
            "stop_price": 0.014,
        },
    )
    assert kind == "external_full_exit"
    assert confirmation["resolved_via"] == "fetch_my_trades_full_external_exit"
    assert np.isclose(confirmation["fill_price"], 0.01734)


def test_smooth_protection_arms_on_one_bar_and_can_only_exit_on_a_later_bar() -> None:
    params = RichPolicyParams(
        sl_mult=10.0,
        smooth_capital_protection_enabled=True,
        protection_unit="raw_decision_time_atr",
        protection_activation_atr=1.5,
        protection_strength=0.5,
        protection_power=1.5,
        adverse_exit_enabled=False,
    )
    first = _advance_rich_policy_position(
        position=_position(),
        # Reaches 2 ATR then gives back below its newly-computed lock.  The
        # lock must be persisted but cannot be triggered by this same bar.
        bars=_bars([("2026-08-17T00:00:00Z", 104.0, 100.4, 101.0)]),
        params=params,
        median_atr_fraction=0.01,
    )
    assert first.get("exit") is None
    assert first["smooth_armed"] is True
    assert first["smooth_armed_now"] is True
    assert first["smooth_lock_price"] > 100.0
    second = _advance_rich_policy_position(
        position={**_position(), **first},
        bars=_bars([("2026-08-17T00:01:00Z", 101.1, 100.1, 100.4)]),
        params=params,
        median_atr_fraction=0.01,
    )
    assert second["exit"]["exit_reason"] == "smooth_capital_protect"
    assert np.isclose(second["exit"]["exit_price"], first["smooth_lock_price"])
