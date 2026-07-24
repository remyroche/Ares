import pandas as pd
from types import SimpleNamespace

import extreme_price_movements.scripts.live_closed_trade_exit_replay as replay_module

from extreme_price_movements.scripts.live_closed_trade_exit_replay import (
    ParsedRecap,
    _logged_live_exchange_stop_fill,
    _logged_live_software_handoff_exit,
    _parse_recap_observations,
    _combined_cached_bars,
    _recover_barrier_frac,
    _select_closed_trade_rows,
    _summarise,
    replay_one_anchor,
)


def test_recover_barrier_uses_promoted_trailing_policy_anchor() -> None:
    assert _recover_barrier_frac(
        {},
        {
            "policy_pathway_id": "joint_trailing_total_mfe_raw_bayesian_v1",
            "median_barrier_frac": 0.0075,
        },
    ) == 0.0075


def test_recover_barrier_prefers_causal_entry_stop_over_policy_median() -> None:
    observed = _recover_barrier_frac(
        {"shadow_initial_stop_price": 102.0},
        {"sl_mult": 4.0, "median_barrier_frac": 0.01},
        entry_price=100.0,
        side="short",
    )
    assert observed == 0.005


def test_parse_recap_keeps_15m_bars_spread_and_stop_replacements() -> None:
    parsed = _parse_recap_observations(
        "\n".join(
            [
                "2026-07-17T20:30:00+00:00 price_bar_15m "
                "open=1 high=1.1 low=0.9 close=1.05",
                "2026-07-17T20:31:00+00:00 lightweight_stop_sentinel_sample "
                "side=long price=1.06 spread_bps=42.5",
                "2026-07-17T20:31:01+00:00 stop_replaced "
                "stop_reason=trailing_profit previous_stop=0.95 new_stop=1.01",
                "2026-07-17T20:32:00+00:00 software_policy_stop_close "
                "reason=policy_stop_crossed_no_valid_exchange_stop "
                "current_price=1.02",
            ]
        )
    )

    assert parsed.bars["observation_source"].tolist() == [
        "live_trade_recap_price_bar_15m",
        "live_trade_recap_stop_sentinel_sample",
        "live_trade_recap_stop_replaced",
        "live_trade_recap_software_policy_stop_sample",
    ]
    assert parsed.bars.iloc[1]["spread_bps"] == 42.5
    assert parsed.bars.iloc[2]["logged_stop_price"] == 1.01
    assert parsed.bars.iloc[2]["logged_stop_reason"] == "trailing_profit"


def test_parse_recap_keeps_exchange_stop_adopted_on_restart() -> None:
    parsed = _parse_recap_observations(
        "2026-07-19T02:01:11.753921+00:00 "
        "existing_stop_adopted_on_reattach "
        "previous_stop_order_id=old stop_order_id=current "
        "stop_price=0.000916 stop_trigger_signal=last"
    )

    assert parsed.bars["observation_source"].tolist() == [
        "live_trade_recap_exchange_stop_adopted"
    ]
    assert parsed.bars.iloc[0]["logged_stop_price"] == 0.000916
    assert parsed.bars.iloc[0]["logged_stop_reason"] == ""


def test_strict_minute_replay_keeps_only_exchange_stop_state(
    monkeypatch, tmp_path
) -> None:
    minute = pd.DataFrame(
        [
            {
                "ts": pd.Timestamp("2026-07-19T03:00:00Z"),
                "open": 0.000924,
                "high": 0.000924,
                "low": 0.000915,
                "close": 0.000915,
                "volume": 1.0,
                "observation_source": "execution_1m_cache",
            }
        ]
    )
    monkeypatch.setattr(replay_module, "_read_cached_execution_1m", lambda **_: minute)
    recap = "\n".join(
        [
            "2026-07-19T02:59:00+00:00 price_bar_15m "
            "open=1 high=1 low=0.5 close=0.5",
            "2026-07-19T02:01:11+00:00 existing_stop_adopted_on_reattach "
            "stop_price=0.000916 stop_trigger_signal=last",
        ]
    )

    bars, source, _ = _combined_cached_bars(
        data_root=tmp_path,
        row={"symbol": "CKB/USD:USD", "trade_recap": recap},
        workspace=tmp_path,
        start=pd.Timestamp("2026-07-19T02:00:00Z"),
        end=pd.Timestamp("2026-07-19T03:05:00Z"),
        strict_execution_1m_only=True,
    )

    assert source == "execution_1m_cache+live_trade_recap_exchange_stop_state"
    assert bars["observation_source"].tolist() == [
        "live_trade_recap_exchange_stop_adopted",
        "execution_1m_cache",
    ]


def test_select_closed_trade_rows_keeps_latest_rows_after_symbol_filter() -> None:
    closed = pd.DataFrame(
        [
            {
                "symbol": "AAVE/USD:USD",
                "entry_time": "2026-07-10T10:00:00Z",
                "exit_time": "2026-07-10T10:15:00Z",
            },
            {
                "symbol": "FIL/USD:USD",
                "entry_time": "2026-07-10T11:00:00Z",
                "exit_time": "2026-07-10T11:15:00Z",
            },
            {
                "symbol": "HOODX/USD:USD",
                "entry_time": "2026-07-10T12:00:00Z",
                "exit_time": "2026-07-10T12:15:00Z",
            },
            {
                "symbol": "AAVE/USD:USD",
                "entry_time": "2026-07-10T13:00:00Z",
                "exit_time": "2026-07-10T13:15:00Z",
            },
        ]
    )

    selected = _select_closed_trade_rows(
        closed,
        symbols="AAVE/USD:USD,FIL/USD:USD,HOODX/USD:USD",
        limit=2,
    )

    assert selected["symbol"].tolist() == ["HOODX/USD:USD", "AAVE/USD:USD"]
    assert selected["exit_time"].tolist() == [
        "2026-07-10T12:15:00Z",
        "2026-07-10T13:15:00Z",
    ]


def test_select_closed_trade_rows_prefers_valid_close_over_naive_display_entry() -> None:
    closed = pd.DataFrame(
        [
            {
                "symbol": "ONDO/USD:USD",
                "timestamp": "2026-07-17 18:00:00",
                "entry_time": "2026-07-17T15:00:00Z",
                "exit_time": None,
            },
            {
                "symbol": "ONDO/USD:USD",
                "timestamp": "2026-07-17 17:20:00",
                "entry_time": "2026-07-17T15:00:00Z",
                "exit_time": "2026-07-17T15:20:00Z",
            },
        ]
    )

    selected = _select_closed_trade_rows(
        closed,
        symbols="ONDO/USD:USD",
        limit=1,
    )

    assert len(selected) == 1
    assert selected.iloc[0]["exit_time"] == "2026-07-17T15:20:00Z"


def test_select_closed_trade_rows_applies_since_to_utc_exit_time() -> None:
    closed = pd.DataFrame(
        [
            {
                "symbol": "OLD/USD:USD",
                "entry_time": "2026-07-17T18:00:00Z",
                "exit_time": "2026-07-17T18:30:00Z",
            },
            {
                "symbol": "NEW/USD:USD",
                "entry_time": "2026-07-17T20:00:00Z",
                "exit_time": "2026-07-17T20:30:00Z",
            },
        ]
    )

    selected = _select_closed_trade_rows(
        closed,
        since="2026-07-17T19:00:00Z",
    )

    assert selected["symbol"].tolist() == ["NEW/USD:USD"]


def test_summarise_handles_invalid_rows_without_entry_anchor() -> None:
    summary = _summarise(
        pd.DataFrame(
            [
                {
                    "symbol": "TEST/USD:USD",
                    "coverage_status": "invalid_time_window",
                    "replay_hit": False,
                }
            ]
        )
    )

    assert summary["rows"] == 1
    assert summary["coverage_by_anchor"] == [
        {"coverage_status": "invalid_time_window", "rows": 1}
    ]
    assert summary["exit_parity_status"] == "fail"


def test_summarise_exit_parity_accepts_one_matching_anchor_per_trade() -> None:
    summary = _summarise(
        pd.DataFrame(
            [
                {
                    "symbol": "ETHFI/USD:USD",
                    "entry_time": "2026-07-17T19:08:40Z",
                    "exit_time": "2026-07-17T20:19:44Z",
                    "entry_anchor": "policy_entry",
                    "coverage_status": "ok",
                    "replay_hit": True,
                    "live_exit_reason_detail": "original_stop_loss: sl_mult=2.7",
                    "replay_exit_reason": "original_stop_loss",
                    "replay_exit_price_vs_live_bps": -65.0,
                },
                {
                    "symbol": "ETHFI/USD:USD",
                    "entry_time": "2026-07-17T19:08:40Z",
                    "exit_time": "2026-07-17T20:19:44Z",
                    "entry_anchor": "realized_entry",
                    "coverage_status": "ok",
                    "replay_hit": True,
                    "live_exit_reason_detail": "original_stop_loss: sl_mult=2.7",
                    "replay_exit_reason": "original_stop_loss",
                    "replay_exit_price_vs_live_bps": -4.28,
                },
            ]
        ),
        exit_tolerance_bps=50.0,
    )

    assert summary["exit_parity_status"] == "pass"
    assert summary["exit_parity_mismatch_trades"] == 0


def test_summarise_exit_parity_enforces_timing_when_requested() -> None:
    rows = pd.DataFrame(
        [
            {
                "symbol": "ETHFI/USD:USD",
                "entry_time": "2026-07-17T19:08:40Z",
                "exit_time": "2026-07-17T20:19:44Z",
                "replay_exit_ts": "2026-07-17T20:22:00Z",
                "entry_anchor": "realized_entry",
                "coverage_status": "ok",
                "replay_hit": True,
                "live_exit_reason_detail": "original_stop_loss: sl_mult=2.7",
                "replay_exit_reason": "original_stop_loss",
                "replay_exit_price_vs_live_bps": -4.28,
            }
        ]
    )

    summary = _summarise(
        rows,
        exit_tolerance_bps=10.0,
        exit_time_tolerance_seconds=90.0,
    )

    assert summary["exit_parity_status"] == "fail"
    assert summary["max_abs_exit_time_gap_seconds"] == 136.0


def test_logged_live_software_handoff_exit_is_replayable() -> None:
    row = {
        "reason": (
            "software_executable_stop_breach_pretrigger:"
            "exchange_valid_giveback_fallback_handoff"
        ),
        "close_trigger_type": "software_bid_ask_sentinel",
        "close_execution_method": "ask_bid_software_close",
        "exit_time": "2026-07-10T16:16:17Z",
        "exit_price": "0.804",
    }

    event = _logged_live_software_handoff_exit(row)

    assert event is not None
    assert event["status"] == "logged_live_software_handoff"
    assert event["price"] == 0.804
    assert event["ts"].isoformat() == "2026-07-10T16:16:17+00:00"


def test_logged_live_exchange_stop_fill_is_replayable_from_closed_trade() -> None:
    row = {
        "reason": "stop_loss_filled:trailing_risk_reduction",
        "close_trigger_type": "exchange_stop_order",
        "close_price_source": "exchange_stop_order_fill",
        "exit_time": "2026-07-10T11:49:20Z",
        "exit_price": "68.983",
    }

    event = _logged_live_exchange_stop_fill(row)

    assert event is not None
    assert event["status"] == "logged_live_exchange_stop_fill_from_closed_trade"
    assert event["reason"] == "stop_loss_filled:trailing_risk_reduction"
    assert event["price"] == 68.983


def test_strict_replay_does_not_short_circuit_to_logged_software_exit(
    monkeypatch,
) -> None:
    row = {
        "symbol": "TEST/USD:USD",
        "side": "long",
        "strategy_id": "long_test",
        "entry_price": 100.0,
        "realized_exit_price": 101.0,
        "exit_time": "2026-07-10T11:15:00Z",
        "exit_reason": (
            "software_executable_stop_breach_pretrigger:"
            "exchange_valid_giveback_fallback_handoff"
        ),
        "close_trigger_type": "software_bid_ask_sentinel",
        "close_execution_method": "ask_bid_software_close",
    }
    policy = {"barrier_frac": 0.02, "sl_mult": 1.0}
    monkeypatch.setattr(
        replay_module,
        "compute_initial_simple_policy_stop_decision",
        lambda **_: SimpleNamespace(
            stop_price=98.0,
            barrier_frac=0.02,
            reason="original_stop_loss",
        ),
    )
    monkeypatch.setattr(
        replay_module,
        "compute_simple_policy_stop_decision",
        lambda **_: SimpleNamespace(
            peak_price=100.2,
            mfe=0.002,
            mae=0.001,
            should_exit=False,
            should_replace=False,
            stop_price=None,
            reason="original_stop_loss",
            reason_detail="unchanged",
            exit_reason="",
        ),
    )
    bars = pd.DataFrame(
        [{"ts": "2026-07-10T11:10:00Z", "open": 100.0, "high": 100.2, "low": 99.9, "close": 100.1}]
    )
    recap = ParsedRecap(
        bars=pd.DataFrame(),
        stop_fill_ts=None,
        stop_fill_price=float("nan"),
        stop_reason="",
        source="",
    )

    observed = replay_one_anchor(
        row=row,
        policy_params=policy,
        entry_price=100.0,
        entry_anchor="realized_entry",
        bars=bars,
        recap=recap,
    )
    strict = replay_one_anchor(
        row=row,
        policy_params=policy,
        entry_price=100.0,
        entry_anchor="realized_entry",
        bars=bars,
        recap=recap,
        ignore_logged_exit_events=True,
    )

    assert observed["replay_vs_live_exit_status"] == "logged_live_software_handoff"
    assert strict["replay_hit"] is False
    assert strict["replay_exit_reason"] == "not_hit_in_cached_bars"


def test_sentinel_does_not_apply_non_ticker_policy_stop_replacement(
    monkeypatch,
) -> None:
    row = {
        "symbol": "TEST/USD:USD",
        "side": "long",
        "strategy_id": "long_test",
        "entry_price": 100.0,
        "realized_exit_price": 99.0,
        "shadow_initial_stop_price": 98.0,
    }
    monkeypatch.setattr(
        replay_module,
        "compute_initial_simple_policy_stop_decision",
        lambda **_: SimpleNamespace(
            stop_price=98.0,
            barrier_frac=0.02,
            reason="original_stop_loss",
        ),
    )
    monkeypatch.setattr(
        replay_module,
        "compute_simple_policy_stop_decision",
        lambda **_: SimpleNamespace(
            peak_price=100.0,
            mfe=0.0,
            mae=0.01,
            capital_protect_armed=False,
            capital_protect_armed_now=False,
            should_exit=False,
            should_replace=True,
            stop_price=99.5,
            reason="policy_stop_loss",
            reason_detail="bar-only pressure update",
            exit_reason="",
        ),
    )
    bars = pd.DataFrame(
        [
            {
                "ts": "2026-07-10T11:10:00Z",
                "open": 99.0,
                "high": 99.0,
                "low": 99.0,
                "close": 99.0,
                "observation_source": "live_trade_recap_stop_sentinel_sample",
            }
        ]
    )
    recap = ParsedRecap(
        bars=pd.DataFrame(),
        stop_fill_ts=None,
        stop_fill_price=float("nan"),
        stop_reason="",
        source="",
    )

    strict = replay_one_anchor(
        row=row,
        policy_params={"barrier_frac": 0.02, "sl_mult": 1.0},
        entry_price=100.0,
        entry_anchor="realized_entry",
        bars=bars,
        recap=recap,
        ignore_logged_exit_events=True,
    )

    assert strict["replay_hit"] is False
    assert strict["events_json"] == "[]"


def test_replay_persists_delayed_capital_protection_observation_state(
    monkeypatch,
) -> None:
    row = {
        "symbol": "TEST/USD:USD",
        "side": "long",
        "strategy_id": "long_test",
        "entry_price": 100.0,
        "realized_exit_price": 101.0,
        "shadow_initial_stop_price": 98.0,
    }
    monkeypatch.setattr(
        replay_module,
        "compute_initial_simple_policy_stop_decision",
        lambda **_: SimpleNamespace(
            stop_price=98.0,
            barrier_frac=0.02,
            reason="original_stop_loss",
        ),
    )
    observed_states = []

    def _decision(**kwargs):
        state = dict(kwargs["state"])
        observed_states.append(state)
        crossed_ts = state.get("capital_protect_crossed_ts")
        if crossed_ts is None:
            crossed_ts = pd.Timestamp(
                state["capital_protect_observation_ts"]
            ).isoformat()
        return SimpleNamespace(
            peak_price=101.5,
            mfe=0.015,
            mae=0.0,
            capital_protect_armed=False,
            capital_protect_armed_now=False,
            capital_protect_crossed_ts=crossed_ts,
            capital_protect_pending=True,
            should_exit=False,
            should_replace=False,
            stop_price=None,
            reason="original_stop_loss",
            reason_detail="capital protection pending",
            exit_reason="",
        )

    monkeypatch.setattr(
        replay_module,
        "compute_simple_policy_stop_decision",
        _decision,
    )
    bars = pd.DataFrame(
        [
            {
                "ts": "2026-07-17T10:00:00Z",
                "open": 101.0,
                "high": 101.5,
                "low": 100.9,
                "close": 101.2,
            },
            {
                "ts": "2026-07-17T10:05:00Z",
                "open": 101.2,
                "high": 101.6,
                "low": 101.0,
                "close": 101.3,
            },
        ]
    )
    recap = ParsedRecap(
        bars=pd.DataFrame(),
        stop_fill_ts=None,
        stop_fill_price=float("nan"),
        stop_reason="",
        source="",
    )

    strict = replay_one_anchor(
        row=row,
        policy_params={"barrier_frac": 0.02, "sl_mult": 1.0},
        entry_price=100.0,
        entry_anchor="realized_entry",
        bars=bars,
        recap=recap,
        ignore_logged_exit_events=True,
    )

    assert strict["replay_hit"] is False
    assert observed_states[0]["capital_protect_observation_ts"] == pd.Timestamp(
        "2026-07-17T10:00:00Z"
    )
    assert observed_states[0]["capital_protect_current_price"] == 101.2
    assert observed_states[1]["capital_protect_observation_ts"] == pd.Timestamp(
        "2026-07-17T10:05:00Z"
    )
    assert observed_states[1]["capital_protect_current_price"] == 101.3
    assert observed_states[1]["capital_protect_crossed_ts"] == (
        "2026-07-17T10:00:00+00:00"
    )
    assert observed_states[1]["capital_protect_pending"] is True


def test_execution_1m_replay_advances_policy_bar_clock(monkeypatch) -> None:
    row = {
        "symbol": "TEST/USD:USD",
        "side": "long",
        "strategy_id": "long_test",
        "entry_price": 100.0,
        "realized_exit_price": 100.0,
        "shadow_initial_stop_price": 98.0,
    }
    monkeypatch.setattr(
        replay_module,
        "compute_initial_simple_policy_stop_decision",
        lambda **_: SimpleNamespace(
            stop_price=98.0,
            barrier_frac=0.02,
            reason="original_stop_loss",
        ),
    )
    observed_bars_in_trade = []

    def _decision(**kwargs):
        observed_bars_in_trade.append(int(kwargs["state"]["bars_in_trade"]))
        return SimpleNamespace(
            peak_price=100.5,
            mfe=0.005,
            mae=0.0,
            capital_protect_armed=False,
            capital_protect_armed_now=False,
            capital_protect_crossed_ts=None,
            capital_protect_pending=False,
            should_exit=False,
            should_replace=False,
            stop_price=None,
            reason="original_stop_loss",
            reason_detail="unchanged",
            exit_reason="",
        )

    monkeypatch.setattr(replay_module, "compute_simple_policy_stop_decision", _decision)
    bars = pd.DataFrame(
        [
            {
                "ts": "2026-07-17T10:00:00Z",
                "open": 100.0,
                "high": 100.5,
                "low": 99.9,
                "close": 100.2,
                "observation_source": "execution_1m_cache",
            },
            {
                "ts": "2026-07-17T10:01:00Z",
                "open": 100.2,
                "high": 100.5,
                "low": 100.1,
                "close": 100.3,
                "observation_source": "execution_1m_cache",
            },
        ]
    )
    recap = ParsedRecap(
        bars=pd.DataFrame(),
        stop_fill_ts=None,
        stop_fill_price=float("nan"),
        stop_reason="",
        source="",
    )

    strict = replay_one_anchor(
        row=row,
        policy_params={
            "barrier_frac": 0.02,
            "sl_mult": 1.0,
            "replay_timeframe": "1m",
        },
        entry_price=100.0,
        entry_anchor="realized_entry",
        bars=bars,
        recap=recap,
        ignore_logged_exit_events=True,
    )

    assert strict["replay_hit"] is False
    assert observed_bars_in_trade == [0, 1]
