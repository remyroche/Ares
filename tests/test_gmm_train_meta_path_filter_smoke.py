import pandas as pd

from scripts.run_gmm_train_meta_path_filter_smoke import (
    DEFAULT_THRESHOLDS,
    _simple_policy_handoff_from_selected,
    run_meta_filter_from_ledger,
)


def test_meta_filter_reduces_dirty_positive_candidate_stream() -> None:
    rows = []
    for period in ("2026-04", "2026-05"):
        for idx in range(12):
            clean = idx < 6
            rows.append(
                {
                    "period": period,
                    "selector_variant": "s18_demo",
                    "side": 1 if idx % 2 == 0 else -1,
                    "selector_score": 1.0 - idx / 20.0,
                    "selector_rank_pct": 1.0 - idx / 20.0,
                    "selector_ts_rank_pct": 1.0 - idx / 20.0,
                    "selector_ts_side_rank_pct": 1.0 - idx / 20.0,
                    "clean_path_pred": 0.9 if clean else 0.1,
                    "lgbm_clean_path_pred": 0.85 if clean else 0.15,
                    "lgbm_bad_mae_pred": 0.1 if clean else 0.9,
                    "lgbm_timeout_pred": 0.05,
                    "u_policy_net": 0.01 if clean else 0.006,
                    "mae_norm": 0.3 if clean else 1.4,
                    "is_timeout": 0,
                    "bad_mae_1r": not clean,
                    "clean_positive": clean,
                    "dirty_positive": not clean,
                    "oracle_top": clean and idx < 4,
                    "clean_oracle_top": clean and idx < 4,
                    "oracle_rows_total": 4,
                    "clean_oracle_rows_total": 4,
                }
            )
    ledger = pd.DataFrame(rows)

    monthly, aggregate = run_meta_filter_from_ledger(
        ledger,
        keep_fracs=[0.50],
        max_side_share=0.70,
        min_train_rows=8,
        seed=17,
        thresholds=dict(DEFAULT_THRESHOLDS),
    )

    assert not monthly.empty
    assert not aggregate.empty
    row = aggregate.iloc[0]
    assert row["bad_mae_1r_rate"] <= 0.50
    assert row["timeout_rate"] <= 0.12
    assert row["final_oracle_recall"] >= 0.02


def test_simple_policy_handoff_applies_barrier_multiplier() -> None:
    selected = pd.DataFrame(
        {
            "timestamp": ["2026-05-01T00:00:00Z", "2026-05-01T01:00:00Z"],
            "symbol": ["BTC/USD:USD", "ETH/USD:USD"],
            "side": [1.0, -1.0],
            "meta_selected_score": [0.2, 0.1],
            "meta_score_rank_pct": [0.9, 0.8],
            "meta_score_rank_pct_selected": [1.0, 0.0],
            "barrier": [0.005, 0.010],
            "joint_bad_risk": [0.1, 0.2],
            "selector_variant": ["s8", "s8"],
            "meta_variant": ["meta", "meta"],
            "keep_frac": [0.61, 0.61],
            "meta_bad_risk": [0.1, 0.2],
            "meta_timeout_risk": [0.05, 0.06],
            "joint_timeout_risk": [0.04, 0.05],
        }
    )

    handoff = _simple_policy_handoff_from_selected(
        selected,
        barrier_multiplier=4.0,
    )

    assert [round(float(v), 6) for v in handoff["barrier_pct"].tolist()] == [0.02, 0.04]
    assert handoff["simple_policy_barrier_multiplier"].tolist() == [4.0, 4.0]


def test_include_first_period_marks_insufficient_train_rows() -> None:
    rows = []
    for period in ("2026-04", "2026-05"):
        for idx in range(6):
            clean = idx % 2 == 0
            rows.append(
                {
                    "period": period,
                    "selector_variant": "s22_demo",
                    "side": 1 if idx % 2 == 0 else -1,
                    "selector_score": 1.0 - idx / 10.0,
                    "selector_rank_pct": 1.0 - idx / 10.0,
                    "selector_ts_rank_pct": 1.0 - idx / 10.0,
                    "selector_ts_side_rank_pct": 1.0 - idx / 10.0,
                    "u_policy_net": 0.002 if clean else -0.001,
                    "bad_mae_1r": 0 if clean else 1,
                    "is_timeout": 0,
                    "clean_positive": clean,
                    "dirty_positive": not clean,
                    "oracle_top": clean and idx < 2,
                    "clean_oracle_top": clean and idx < 2,
                    "oracle_rows_total": 2 if period == "2026-05" else 2,
                    "clean_oracle_rows_total": 2 if period == "2026-05" else 2,
                }
            )
    ledger = pd.DataFrame(rows)
    monthly, aggregate = run_meta_filter_from_ledger(
        ledger,
        keep_fracs=[0.50],
        max_side_share=0.70,
        min_train_rows=5,
        seed=7,
        thresholds=dict(DEFAULT_THRESHOLDS),
        include_first_period=True,
    )

    assert set(monthly["period"]) == {"2026-04", "2026-05"}
    first = monthly[monthly["period"] == "2026-04"].iloc[0]
    second = monthly[monthly["period"] == "2026-05"].iloc[0]
    assert first["meta_eval_status"] == "insufficient_train_rows"
    assert int(first["meta_train_rows"]) == 0
    assert float(first["selected_rows"]) == 0.0
    assert second["meta_eval_status"] == "ok"
    assert int(second["meta_train_rows"]) == 6
    assert int(aggregate["meta_oos_months"].iloc[0]) == 1
    assert int(aggregate["meta_skipped_months_due_to_insufficient_train"].iloc[0]) == 1
