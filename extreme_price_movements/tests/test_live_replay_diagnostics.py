import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.inference.live_feature_parity import (
    build_feature_parity_report,
    summarize_feature_parity,
)
from extreme_price_movements.inference.live_feature_parity_job import (
    run_offline_feature_parity_job,
)
from extreme_price_movements.inference.live_gap_report import (
    build_live_gap_report,
    classify_gap_rows,
    render_live_gap_report_markdown,
)
from extreme_price_movements.inference.live_gap_diagnostics import (
    attach_strategy_oos_expectations,
    load_strategy_oos_expectations,
)
from extreme_price_movements.inference.live_replay import (
    attach_forward_outcomes,
    build_live_candidate_replay_table,
    build_live_replay_table,
    collapse_trade_lifecycle,
    summarize_gap_decomposition,
)


def test_build_feature_parity_report_detects_exact_mismatch_and_missing():
    idx = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    live = {
        "ret1h": pd.DataFrame({"BTC/USDT": [0.01, 0.02]}, index=idx),
        "vol_z": pd.DataFrame({"BTC/USDT": [1.0, 2.0]}, index=idx),
    }
    oos = {"ret1h": pd.DataFrame({"BTC/USDT": [0.01, 0.025]}, index=idx)}
    decisions = pd.DataFrame({"timestamp": [idx[1]], "symbol": ["BTC/USDT"]})

    report = build_feature_parity_report(live, oos, decisions=decisions, include_extra_features=True)

    ret_row = report.loc[report["feature"] == "ret1h"].iloc[0]
    assert ret_row["parity_status"] == "mismatch"
    assert np.isclose(ret_row["abs_diff"], 0.005)
    vol_row = report.loc[report["feature"] == "vol_z"].iloc[0]
    assert vol_row["parity_status"] == "missing_feature"

    summary = summarize_feature_parity(report)
    assert set(summary["feature"]) == {"ret1h", "vol_z"}


def test_offline_feature_parity_job_writes_reports(tmp_path):
    idx = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    decisions = pd.DataFrame(
        {
            "timestamp": [idx[1]],
            "signal_bar_ts": [idx[1]],
            "symbol": ["BTC/USDT"],
        }
    )
    live_dir = tmp_path / "live"
    oos_dir = tmp_path / "oos"
    live_dir.mkdir()
    oos_dir.mkdir()
    decisions_path = tmp_path / "decisions.parquet"
    decisions.to_parquet(decisions_path, index=False)
    pd.DataFrame({"BTC/USDT": [1.0, 2.0]}, index=idx).to_parquet(
        live_dir / "ret1h.parquet"
    )
    pd.DataFrame({"BTC/USDT": [1.0, 2.0]}, index=idx).to_parquet(
        oos_dir / "ret1h.parquet"
    )

    report, summary = run_offline_feature_parity_job(
        decisions_path=decisions_path,
        live_features_path=live_dir,
        oos_features_path=oos_dir,
        output_dir=tmp_path / "out",
    )

    assert report.loc[0, "parity_status"] == "match"
    assert summary.loc[0, "matches"] == 1
    assert (tmp_path / "out" / "feature_parity_report.csv").exists()
    assert (tmp_path / "out" / "feature_parity_summary.csv").exists()


def test_feature_parity_asof_never_looks_after_decision():
    idx = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    live = {"ret1h": pd.DataFrame({"ETH/USDT": [0.03, 0.04]}, index=idx)}
    oos = {"ret1h": pd.DataFrame({"ETH/USDT": [0.03, 0.04]}, index=idx)}

    report = build_feature_parity_report(
        live,
        oos,
        timestamps=[pd.Timestamp("2026-01-01 00:30", tz="UTC")],
        symbols=["ETH/USDT"],
        allow_asof=True,
    )

    row = report.iloc[0]
    assert row["parity_status"] == "match_asof"
    assert row["live_feature_bar_ts"] == idx[0]
    assert row["oos_feature_bar_ts"] == idx[0]
    assert not row["lookahead_violation"]


def test_feature_parity_unsorted_asof_future_row_and_signal_bar_ts():
    idx = pd.to_datetime(
        ["2026-01-01 00:00", "2026-01-01 02:00", "2026-01-01 01:00"], utc=True
    )
    live = {"ret1h": pd.DataFrame({"BTC/USDT": [1.0, 99.0, 2.0]}, index=idx)}
    oos = {"ret1h": pd.DataFrame({"BTC/USDT": [1.0, 99.0, 2.0]}, index=idx)}
    decisions = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-01-01 01:30", tz="UTC")],
            "signal_bar_ts": [pd.Timestamp("2026-01-01 01:15", tz="UTC")],
            "symbol": ["BTC/USDT"],
            "side": ["long"],
            "strategy_id": ["s1"],
        }
    )

    report = build_feature_parity_report(live, oos, decisions=decisions, allow_asof=True)
    row = report.iloc[0]
    assert row["live_value"] == 2.0
    assert row["live_feature_bar_ts"] == pd.Timestamp("2026-01-01 01:00", tz="UTC")
    assert row["signal_bar_ts"] != row["decision_ts"]
    assert not row["lookahead_violation"]


def test_feature_parity_nat_and_missing_required_feature():
    idx = pd.date_range("2026-01-01", periods=1, freq="h", tz="UTC")
    live = {"ret1h": pd.DataFrame({"BTC/USDT": [0.1]}, index=idx)}
    oos = {"ret1h": pd.DataFrame({"BTC/USDT": [0.1]}, index=idx)}
    decisions = pd.DataFrame({"timestamp": [idx[0]], "signal_bar_ts": [pd.NaT], "symbol": ["BTC/USDT"]})
    report = build_feature_parity_report(live, oos, decisions=decisions, feature_keys=["ret1h", "missing"])
    assert set(report["parity_status"]) == {"match", "missing_feature"}


def test_build_live_replay_table_outputs_requested_schema_and_decomposition():
    live = pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z"],
            "symbol": ["BTC/USDT"],
            "side": ["long"],
            "strategy_id": ["long_rule"],
            "adjusted_rank_score": [0.8],
            "rank_percentile": [0.95],
            "final_threshold": [0.7],
            "signal_price": [100.0],
            "decision_mid": [100.0],
            "expected_fill_price": [100.1],
            "realized_entry_price": [100.2],
            "realized_exit_price": [101.0],
            "gross_to_net_cost_pct": [0.001],
            "ticker_spread_bps": [5.0],
            "expected_total_entry_friction_bps": [7.0],
            "exit_reason": ["take_profit"],
            "holding_bars": [4],
            "net_pnl_pct": [0.006],
        }
    )
    oos = pd.DataFrame(
        {
            "signal_bar_ts": ["2026-01-01T00:00:00Z"],
            "symbol": ["BTC/USDT"],
            "side": ["long"],
            "strategy_id": ["long_rule"],
            "expected_net": [0.012],
            "selected": [True],
        }
    )

    replay = build_live_replay_table(live, oos_policy=oos, default_expected_fee_bps=4.0)

    assert replay.loc[0, "rank_score"] == 0.8
    assert np.isclose(replay.loc[0, "entry_drag_bps"], (100.2 / 100.1 - 1.0) * 10000.0)
    assert np.isclose(replay.loc[0, "gap_oos_vs_realized_bps"], 60.0)
    assert replay.loc[0, "fees_bps"] == 10.0
    assert "residual_bps" in replay.columns
    assert "gap_oos_vs_realized_bps" in set(summarize_gap_decomposition(replay)["component"])


def test_collapse_trade_lifecycle_entry_exit_and_open_position():
    log = pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "position_id": "p1",
                "trade_id": "t1",
                "action": "enter",
                "lifecycle_event": "entry_filled",
                "symbol": "BTC/USDT",
                "side": "long",
                "strategy_id": "s1",
                "signal_price": 100.0,
                "realized_entry_price": 100.2,
            },
            {
                "timestamp": "2026-01-01T01:00:00Z",
                "position_id": "p1",
                "action": "exit",
                "lifecycle_event": "exit_filled",
                "symbol": "BTC/USDT",
                "side": "long",
                "strategy_id": "s1",
                "realized_exit_price": 101.0,
                "exit_reason": "take_profit",
                "net_pnl_pct": 0.008,
            },
            {
                "timestamp": "2026-01-01T02:00:00Z",
                "position_id": "p2",
                "action": "enter",
                "lifecycle_event": "entry_filled",
                "symbol": "ETH/USDT",
                "side": "short",
                "strategy_id": "s2",
                "signal_price": 50.0,
                "realized_entry_price": 49.8,
            },
        ]
    )
    collapsed = collapse_trade_lifecycle(log).set_index("position_id")
    assert collapsed.loc["p1", "signal_price"] == 100.0
    assert collapsed.loc["p1", "realized_exit_price"] == 101.0
    assert collapsed.loc["p1", "exit_reason"] == "take_profit"
    assert pd.isna(collapsed.loc["p2", "realized_exit_price"])
    assert collapsed.loc["p2", "was_traded"] == True


def test_collapse_trade_lifecycle_marks_failed_entry_not_traded():
    log = pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "position_id": "p1",
                "action": "enter",
                "lifecycle_event": "enter_failed",
                "status": "failed",
                "symbol": "BTC/USDT",
                "side": "long",
                "strategy_id": "long_s1",
                "error": "missing barrier",
            }
        ]
    )
    collapsed = collapse_trade_lifecycle(log)
    assert collapsed.loc[0, "was_traded"] == False
    assert collapsed.loc[0, "strategy_id"] == "s1"


def test_candidate_replay_links_closed_exit_by_stop_order_id():
    ledger = pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z"],
            "signal_bar_ts": ["2026-01-01T00:00:00Z"],
            "symbol": ["BTC/USDT"],
            "side": ["long"],
            "strategy_id": ["long_s1"],
            "was_traded": [True],
            "order_id": ["100"],
            "signal_price": [100.0],
            "decision_mid": [100.0],
            "realized_entry_price": [100.0],
        }
    )
    trade_log = pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T00:00:01Z",
                "position_id": "run:100",
                "action": "enter",
                "lifecycle_event": "entry_placed",
                "exchange_order_id": "100",
                "stop_order_id": "200",
                "symbol": "BTC/USDT",
                "side": "long",
                "strategy_id": "long_s1",
                "actual_entry_price": 100.0,
            },
            {
                "timestamp": "2026-01-01T01:00:00Z",
                "position_id": "run:200",
                "action": "exit",
                "lifecycle_event": "exit_filled",
                "exchange_order_id": "200",
                "symbol": "BTC/USDT",
                "side": "long",
                "strategy_id": "long_s1",
                "realized_exit_price": 101.0,
                "net_pnl_pct": 0.01,
            },
        ]
    )
    replay = build_live_candidate_replay_table(ledger, trade_log=trade_log)
    assert replay.loc[0, "realized_exit_price"] == pytest.approx(101.0)
    assert replay.loc[0, "realized_trade_net_bps"] == pytest.approx(100.0)
    assert replay.loc[0, "diagnostic_complete"] == False


def test_strategy_level_oos_expectations_match_live_ids_without_side_prefix(tmp_path):
    artifact = tmp_path / "strategy_for_inference.json"
    artifact.write_text(
        __import__("json").dumps(
            {
                "strategies": [
                    {
                        "strategy_id": "long_s1",
                        "side": "long",
                        "selected": True,
                        "avg_net_pnl_per_trade": 0.0012,
                    }
                ]
            }
        )
    )
    expectations = load_strategy_oos_expectations(artifact)
    replay = pd.DataFrame(
        {
            "strategy_id": ["s1"],
            "side": ["long"],
            "oos_expected_net_bps": [np.nan],
        }
    )
    out = attach_strategy_oos_expectations(replay, expectations)
    assert out.loc[0, "oos_expected_net_bps"] == pytest.approx(12.0)
    assert out.loc[0, "oos_expectation_source"] == "strategy_level_policy_artifact"


def test_forward_outcomes_long_short_and_no_realized_exit_for_prediction():
    idx = pd.date_range("2026-01-01", periods=6, freq="15min", tz="UTC")
    close = pd.DataFrame({"BTC/USDT": [100, 101, 102, 103, 104, 105], "ETH/USDT": [100, 99, 98, 97, 96, 95]}, index=idx)
    replay = pd.DataFrame(
        {
            "timestamp": [idx[0], idx[0]],
            "signal_bar_ts": [idx[0], idx[0]],
            "symbol": ["BTC/USDT", "ETH/USDT"],
            "side": ["long", "short"],
            "signal_price": [100.0, 100.0],
            "decision_mid": [100.0, 100.0],
            "realized_entry_price": [102.0, 98.0],
            "realized_exit_price": [1.0, 1000.0],
            "fees_bps": [0.0, 0.0],
            "oos_assumed_cost_bps": [0.0, 0.0],
        }
    )
    out = attach_forward_outcomes(replay, close=close, horizons=(1, 4), primary_horizon=4)
    assert np.isclose(out.loc[0, "signal_forward_return_4bar"], 0.04)
    assert np.isclose(out.loc[1, "signal_forward_return_4bar"], 100 / 96 - 1)
    assert out.loc[0, "fill_forward_return_4bar"] < out.loc[0, "signal_forward_return_4bar"]
    assert np.isclose(out.loc[0, "signal_forward_net_bps"], 400.0)
    assert out.loc[0, "primary_horizon_bars"] == 4
    assert out.loc[0, "bar_minutes"] == 15


def test_oos_join_signal_bar_exact_asof_and_outside_tolerance():
    trade = pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:30Z", "2026-01-01T01:00:20Z", "2026-01-01T02:10:00Z"],
            "signal_bar_ts": ["2026-01-01T00:00:00Z", "2026-01-01T01:00:30Z", "2026-01-01T02:10:00Z"],
            "symbol": ["BTC/USDT"] * 3,
            "side": ["long"] * 3,
            "strategy_id": ["s1"] * 3,
            "realized_entry_price": [100, 100, 100],
            "realized_exit_price": [100, 100, 100],
        }
    )
    oos = pd.DataFrame(
        {
            "signal_bar_ts": ["2026-01-01T00:00:00Z", pd.NaT],
            "timestamp": ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"],
            "symbol": ["BTC/USDT", "BTC/USDT"],
            "side": ["long", "long"],
            "strategy_id": ["s1", "s1"],
            "expected_net_bps": [10.0, 20.0],
        }
    )
    replay = build_live_replay_table(trade, oos_policy=oos, oos_join_tolerance=pd.Timedelta("1min"))
    assert replay.loc[0, "oos_expected_net_bps"] == 10.0
    assert replay.loc[1, "oos_expected_net_bps"] == 20.0
    assert pd.isna(replay.loc[2, "oos_expected_net_bps"])


def test_candidate_replay_includes_traded_and_rejected():
    idx = pd.date_range("2026-01-01", periods=6, freq="15min", tz="UTC")
    ledger = pd.DataFrame(
        {
            "timestamp": [idx[0], idx[0]],
            "signal_bar_ts": [idx[0], idx[0]],
            "symbol": ["BTC/USDT", "ETH/USDT"],
            "side": ["long", "long"],
            "strategy_id": ["s1", "s1"],
            "was_traded": [True, False],
            "portfolio_decision": ["traded", "rejected"],
            "portfolio_reject_reason": ["", "gross_cap"],
            "signal_price": [100.0, 50.0],
            "decision_mid": [100.0, 50.0],
            "realized_entry_price": [100.0, np.nan],
        }
    )
    close = pd.DataFrame({"BTC/USDT": [100, 101, 102, 103, 104, 105], "ETH/USDT": [50, 51, 52, 53, 54, 55]}, index=idx)
    replay = build_live_candidate_replay_table(ledger, forward_close=close)
    assert len(replay) == 2
    assert set(replay["was_traded"].astype(bool)) == {True, False}
    assert replay.loc[1, "portfolio_reject_reason"] == "gross_cap"


def test_gap_classification_all_required_buckets():
    replay = pd.DataFrame(
        {
            "signal_forward_net_bps": [10.0, -1.0, 10.0, 10.0],
            "fill_forward_net_bps": [-1.0, -2.0, 10.0, 10.0],
            "realized_trade_net_bps": [5.0, -2.0, np.nan, -5.0],
            "was_traded": [True, True, False, True],
        }
    )
    out = classify_gap_rows(replay)
    assert list(out["gap_classification"]) == [
        "execution_timing_gap",
        "prediction_or_live_feature_drift",
        "selection_or_gating_gap",
        "exit_stop_slippage_cost_gap",
    ]


def test_live_gap_report_dict_and_markdown_interpretations():
    replay = pd.DataFrame(
        {
            "strategy_id": ["s1"],
            "symbol": ["BTC/USDT"],
            "exit_reason": ["take_profit"],
            "portfolio_reject_reason": [""],
            "signal_forward_net_bps": [10.0],
            "fill_forward_net_bps": [-1.0],
            "realized_trade_net_bps": [5.0],
            "oos_expected_net_bps": [12.0],
            "was_traded": [True],
        }
    )
    report = build_live_gap_report(replay)
    assert {
        "summary",
        "classification_counts",
        "by_strategy",
        "four_element_diagnosis",
    }.issubset(report)
    assert (
        report["four_element_diagnosis"]["signal_forward_good_fill_forward_bad"][
            "rows"
        ]
        == 1
    )
    md = render_live_gap_report_markdown(report)
    for text in [
        "execution/timing gap",
        "model/rank/live-feature drift",
        "selection/gating/portfolio constraints issue",
        "exit/stop/slippage/cost issue",
        "Four-element diagnosis",
    ]:
        assert text in md


def test_report_includes_coverage_metadata_and_unit_warnings():
    replay = pd.DataFrame(
        {
            "strategy_id": ["s1"],
            "symbol": ["BTC/USDT"],
            "was_traded": [True],
            "oos_expected_net_bps": [5.0],
            "signal_forward_net_bps": [10.0],
            "fill_forward_net_bps": [9.0],
            "realized_trade_net_bps": [8.0],
            "realized_exit_price": [101.0],
            "diagnostic_complete": [True],
            "primary_horizon_bars": [4],
            "bar_minutes": [15],
            "decision_ts": [pd.Timestamp("2026-01-01", tz="UTC")],
            "signal_bar_ts": [pd.Timestamp("2026-01-01", tz="UTC")],
            "feature_source_max_ts": [pd.Timestamp("2026-01-01", tz="UTC")],
            "feature_available_ts": [pd.Timestamp("2026-01-01 00:01", tz="UTC")],
            "unit_warning": ["net_pnl_pct_abs_gt_1_check_units"],
        }
    )
    report = build_live_gap_report(replay)
    assert report["diagnostic_coverage"]["rows_with_oos_join"] == 1
    assert report["diagnostic_coverage"]["diagnostic_complete_rows"] == 1
    assert report["metadata"]["primary_horizon_bars"] == [4]
    assert report["unit_warnings"] == {"net_pnl_pct_abs_gt_1_check_units": 1}
    md = render_live_gap_report_markdown(report)
    assert "Diagnostic coverage" in md
    assert "primary_horizon_bars" in md


def test_suspicious_fraction_style_legacy_units_are_warned():
    live = pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z"],
            "symbol": ["BTC/USDT"],
            "side": ["long"],
            "strategy_id": ["s1"],
            "net_pnl_pct": [2.5],
            "expected_net": [1.5],
        }
    )
    replay = build_live_replay_table(live)
    assert "net_pnl_pct_abs_gt_1_check_units" in replay.loc[0, "unit_warning"]
    assert "expected_net_abs_gt_1_check_units" in replay.loc[0, "unit_warning"]


def test_bps_columns_are_not_fraction_scaled():
    live = pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z"],
            "symbol": ["BTC/USDT"],
            "side": ["long"],
            "strategy_id": ["s1"],
            "realized_trade_net_bps": [1.0],
            "oos_expected_net_bps": [2.0],
        }
    )
    replay = build_live_replay_table(live)
    assert replay.loc[0, "realized_trade_net_bps"] == 1.0
    assert replay.loc[0, "oos_expected_net_bps"] == 2.0


def test_string_false_was_traded_is_rejected_in_decomposition_and_classification():
    replay = pd.DataFrame(
        {
            "signal_forward_net_bps": [10.0],
            "fill_forward_net_bps": [np.nan],
            "realized_trade_net_bps": [np.nan],
            "was_traded": ["False"],
        }
    )
    decomposed = attach_forward_outcomes(
        pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-01", tz="UTC")],
                "signal_bar_ts": [pd.Timestamp("2026-01-01", tz="UTC")],
                "symbol": ["BTCUSDT"],
                "side": ["long"],
                "signal_price": [100.0],
                "decision_mid": [100.0],
                "realized_entry_price": [100.0],
                "was_traded": ["False"],
            }
        ),
        close=pd.DataFrame(
            {"BTC/USDT:USDT": [100.0, 101.0, 102.0, 103.0, 104.0]},
            index=pd.date_range("2026-01-01", periods=5, freq="15min", tz="UTC"),
        ),
        horizons=(4,),
    )
    assert decomposed.loc[0, "selection_gap_bps"] > 0
    out = classify_gap_rows(replay)
    assert out.loc[0, "gap_classification"] == "selection_or_gating_gap"


def test_gap_report_includes_ic_metrics_across_symbols_weeks_months():
    ts = pd.date_range("2026-01-01", periods=8, freq="7D", tz="UTC")
    replay = pd.DataFrame(
        {
            "timestamp": ts,
            "signal_bar_ts": ts,
            "symbol": ["BTC/USDT", "BTC/USDT", "ETH/USDT", "ETH/USDT"] * 2,
            "rank_score": [1, 2, 1, 2, 2, 3, 2, 3],
            "signal_forward_net_bps": [10, 20, 10, 20, 20, 30, 20, 30],
            "was_traded": [True] * 8,
        }
    )
    report = build_live_gap_report(replay)
    assert report["ic_metrics"]["overall_ic"] > 0.9
    assert report["ic_metrics"]["ic_n_symbols"] >= 2
    assert "ic_std_across_weeks" in report["ic_metrics"]
    assert "ic_std_across_months" in report["ic_metrics"]
    assert "IC metrics" in render_live_gap_report_markdown(report)
