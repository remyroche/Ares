import numpy as np
import pandas as pd

from scripts import materialize_t1_repaired_static_baseline as mat
from scripts import run_t1_rank_contract_walkforward as wf


def _candidate_frame(**values) -> pd.DataFrame:
    frame = pd.DataFrame(values)
    n = len(frame)
    frame["base_strategy_threshold"] = 0.70
    frame["entry_price"] = 100.0
    frame["exit_price"] = 101.0
    frame["exit_timestamp"] = pd.date_range("2026-01-03", periods=n, freq="h", tz="UTC")
    frame["gross_return"] = 0.01
    frame["net_return"] = 0.009
    frame["holding_bars"] = 1
    frame["simple_policy_exit_reason"] = "tp"
    return frame


def test_make_time_folds_uses_embargo_and_complete_timestamp_blocks() -> None:
    timestamps = pd.Series(pd.date_range("2026-01-01", periods=96, freq="h", tz="UTC"))

    folds = wf._make_time_folds(
        timestamps,
        train_min_days=1,
        valid_days=1,
        step_days=1,
        embargo_hours=2,
    )

    assert folds
    first = folds[0]
    assert first.train_end_exclusive == pd.Timestamp("2026-01-01 22:00:00", tz="UTC")
    assert first.valid_start == pd.Timestamp("2026-01-02 00:00:00", tz="UTC")
    assert first.valid_end_exclusive == pd.Timestamp("2026-01-03 00:00:00", tz="UTC")
    assert first.train_end_exclusive <= first.valid_start - pd.Timedelta(hours=2)


def test_fold_global_rank_reference_uses_training_distribution_only() -> None:
    train = _candidate_frame(
        timestamp=pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
        symbol=["A", "B", "C", "D"],
        side=["short", "short", "short", "short"],
        strategy_id=["short_boll_s1"] * 4,
        head=["short_boll"] * 4,
        calibrated_score=[0.10, 0.20, 0.50, 0.90],
        normalized_rank_score=[0.0] * 4,
        strategy_rank_pct=[0.0] * 4,
        policy_rank_pct=[0.0] * 4,
        rank_pct=[0.0] * 4,
    )
    valid = _candidate_frame(
        timestamp=pd.date_range("2026-01-02", periods=3, freq="h", tz="UTC"),
        symbol=["E", "F", "G"],
        side=["short", "short", "short"],
        strategy_id=["short_boll_s1"] * 3,
        head=["short_boll"] * 3,
        calibrated_score=[0.05, 0.20, 0.95],
        normalized_rank_score=[0.0] * 3,
        strategy_rank_pct=[0.0] * 3,
        policy_rank_pct=[0.0] * 3,
        rank_pct=[0.0] * 3,
    )

    out, diag = wf._apply_fold_global_rank_reference(valid, reference_train=train)

    np.testing.assert_allclose(out["policy_rank_pct"].to_numpy(), [0.0, 0.5, 1.0])
    np.testing.assert_allclose(out["auction_rank_score"].to_numpy(), [0.0, 0.5, 1.0])
    assert diag["missing_policy_rank_rows"] == 0
    assert diag["missing_auction_rank_rows"] == 0
    assert diag["ranked_rows"] == 3
    assert set(out["rank_contract_source"]) == {"fold_causal_global_score_distribution"}


def test_materialized_global_rank_contract_requires_complete_frozen_rank_diagnostics() -> None:
    eval_diag = {
        "rank_reference_run_id": "prejune_ref",
        "rank_source": "policy_rank_reference_percentile",
        "missing_rank_rows": 0,
        "missing_auction_rank_rows": 0,
        "ranked_rows": 3,
        "auction_ranked_rows": 3,
        "policy_rank_reference_n_min": 10,
        "auction_rank_reference_n_min": 30,
        "window_rank_debug_used": False,
    }
    train_diag = {
        **eval_diag,
        "ranked_rows": 5,
        "auction_ranked_rows": 5,
    }

    report = mat._rank_reference_contract_report(
        rank_contract="anchor_global_policy_rank_reference",
        eval_diag=eval_diag,
        train_diag=train_diag,
        eval_rows=3,
        train_rows=5,
    )
    assert report["passed"] is True

    bad = mat._rank_reference_contract_report(
        rank_contract="anchor_global_policy_rank_reference",
        eval_diag={**eval_diag, "missing_auction_rank_rows": 1},
        train_diag=train_diag,
        eval_rows=3,
        train_rows=5,
    )
    assert bad["passed"] is False
    assert "eval.missing_auction_rank_rows_nonzero" in bad["failures"]


def test_timestamp_accepted_summary_records_zero_and_head_utility() -> None:
    timestamps = pd.Series(pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"))
    accepted = pd.DataFrame(
        {
            "timestamp": [timestamps.iloc[0], timestamps.iloc[0], timestamps.iloc[2]],
            "head": ["short_asset", "short_boll", "short_boll"],
            "net_pnl": [1.0, -0.5, 2.0],
            "simple_policy_exit_reason": ["tp", "sl", "timeout"],
        }
    )

    out = wf._timestamp_accepted_summary(accepted, timestamps, prefix="timestamp_rank")

    assert out["timestamp_rank_trade_count"].tolist() == [2.0, 0.0, 1.0]
    assert out["timestamp_rank_net_pnl"].tolist() == [0.5, 0.0, 2.0]
    assert out["timestamp_rank_short_asset_net_pnl"].tolist() == [1.0, 0.0, 0.0]
    assert out["timestamp_rank_short_boll_net_pnl"].tolist() == [-0.5, 0.0, 2.0]
    assert out["timestamp_rank_full_sl_rate"].tolist() == [0.5, 0.0, 0.0]
    assert out["timestamp_rank_timeout_rate"].tolist() == [0.0, 0.0, 1.0]
