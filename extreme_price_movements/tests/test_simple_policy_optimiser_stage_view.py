import json

import numpy as np
import pandas as pd

from extreme_price_movements.simple_policy_optimiser import (
    _load_policy_stage_view,
    _load_slice_plan_source_validation,
)
from extreme_price_movements.slice_plan_store import (
    _build_final_tail_policy_plan,
    _build_pre_policy_training_plan,
    slice_plan_is_stale,
)


def test_simple_policy_stage_view_refuses_utility_policy_fallback(tmp_path):
    path = tmp_path / "slice_plan.json"
    path.write_text(
        json.dumps(
            {
                "materialized_views": {
                    "utility_policy_optimisation": {
                        "stage_name": "utility_policy_optimisation",
                        "n_plans": 1,
                        "allowed_periods": [
                            {
                                "start_ts": "2026-01-01T00:00:00+00:00",
                                "end_ts": "2026-01-02T00:00:00+00:00",
                            }
                        ],
                    }
                },
                "consumer_plans": {},
            }
        )
    )

    view, stage_name = _load_policy_stage_view(path)

    assert view == {}
    assert stage_name == "missing_policy_optimiser_stage_view"


def test_simple_policy_stage_view_requires_policy_optimiser_consumer_plan(tmp_path):
    path = tmp_path / "slice_plan.json"
    path.write_text(
        json.dumps(
            {
                "materialized_views": {
                    "utility_policy_optimisation": {
                        "stage_name": "utility_policy_optimisation",
                        "n_plans": 1,
                    }
                },
                "consumer_plans": {
                    "policy_optimiser": [
                        {
                            "symbols_predict": ["BTC/USDC"],
                            "metadata": {
                                "predict_start": "2026-01-03T00:00:00+00:00",
                                "predict_end": "2026-01-04T00:00:00+00:00",
                            },
                        }
                    ]
                },
            }
        )
    )

    view, stage_name = _load_policy_stage_view(path)

    assert stage_name == "policy_optimiser"
    assert view["stage_name"] == "policy_optimiser"
    assert view["source_roles"] == ["policy_optimiser"]


def test_simple_policy_source_validation_requires_final_tail_policy_plan(tmp_path):
    path = tmp_path / "slice_plan.json"
    path.write_text(
        json.dumps(
            {
                "version": 3,
                "exchange_context": {
                    "market_mode": "perps",
                    "exchange": "krakenfutures",
                },
                "materialized_views": {
                    "train_base": {"stage_name": "train_base", "n_plans": 1},
                    "utility_policy_optimisation": {
                        "stage_name": "utility_policy_optimisation",
                        "n_plans": 1,
                    },
                    "policy_optimiser": {
                        "stage_name": "policy_optimiser",
                        "n_plans": 1,
                    },
                },
                "consumer_plans": {
                    "policy_optimiser": [
                        {
                            "fit_idx": [0, 1, 2],
                            "predict_idx": [3, 4],
                            "symbols_predict": ["BTC/USD", "ETH/USD"],
                            "metadata": {
                                "predict_role": "policy_holdout_tail",
                                "policy_optimiser_tail_months": 3,
                                "policy_optimiser_all_symbols": True,
                                "fit_end": "2026-01-01T00:00:00+00:00",
                                "predict_actual_start": "2026-01-02T00:00:00+00:00",
                                "predict_actual_end": "2026-04-01T00:00:00+00:00",
                            },
                        }
                    ]
                },
            }
        )
    )

    validation = _load_slice_plan_source_validation(path)

    assert validation["oos_policy_slice_verified"] is True
    assert validation["policy_holdout_predict_roles"] == ["policy_holdout_tail"]
    assert validation["policy_optimiser_tail_months"] == [3]
    assert validation["policy_optimiser_all_symbols"] is True
    assert validation["exchange_context"] == {
        "market_mode": "perps",
        "exchange": "krakenfutures",
    }


def test_slice_plan_stale_on_exchange_context_change():
    existing = {
        "version": 3,
        "event_fingerprint": {"rows": 10},
        "planner": {"preset": "fast", "symbol_policy_mode": "all_symbols"},
        "allocation_targets": {"train_base": 0.7},
        "exchange_context": {"market_mode": "perps", "exchange": "binanceusdm"},
    }

    assert slice_plan_is_stale(
        existing,
        {"rows": 10},
        {"preset": "fast", "symbol_policy_mode": "all_symbols"},
        {"train_base": 0.7},
        exchange_context={"market_mode": "perps", "exchange": "krakenfutures"},
    )


def test_final_tail_policy_plan_caps_last_four_months_to_30pct():
    events = pd.DataFrame(
        {
            "t0": pd.date_range(
                "2026-01-01", periods=100, freq="D", tz="UTC"
            ),
            "symbol": np.where(np.arange(100) % 2 == 0, "BTC/USD", "ETH/USD"),
        }
    )

    plans = _build_final_tail_policy_plan(
        events, tail_months=4, max_sample_fraction=0.30
    )

    assert len(plans) == 1
    plan = plans[0]
    assert len(plan.predict_idx) == 30
    assert len(plan.fit_idx) == 70
    assert plan.metadata["policy_optimiser_tail_months"] == 4
    assert plan.metadata["policy_optimiser_sample_fraction_cap_applied"] is True
    assert plan.metadata["predict_fraction"] == 0.30

    base_plan = _build_pre_policy_training_plan(
        events,
        policy_plan=plan,
        consumer_role="base_model_fit",
        tag="train_base_pre_policy_tail",
    )[0]
    assert np.array_equal(base_plan.fit_idx, plan.fit_idx)
    assert base_plan.metadata["policy_optimiser_tail_excluded"] is True
    assert max(base_plan.fit_idx) < min(plan.predict_idx)


def test_slice_plan_stale_on_policy_tail_config_change():
    existing = {
        "version": 3,
        "event_fingerprint": {"rows": 10},
        "planner": {
            "preset": "fast",
            "symbol_policy_mode": "all_symbols",
            "policy_optimiser_tail_months": 3,
            "policy_optimiser_max_sample_fraction": 0.30,
        },
        "allocation_targets": {"train_base": 0.7},
    }

    assert slice_plan_is_stale(
        existing,
        {"rows": 10},
        {
            "preset": "fast",
            "symbol_policy_mode": "all_symbols",
            "policy_optimiser_tail_months": 4,
            "policy_optimiser_max_sample_fraction": 0.30,
        },
        {"train_base": 0.7},
    )
    assert slice_plan_is_stale(
        existing,
        {"rows": 10},
        {
            "preset": "fast",
            "symbol_policy_mode": "all_symbols",
            "policy_optimiser_tail_months": 3,
            "policy_optimiser_max_sample_fraction": 0.25,
        },
        {"train_base": 0.7},
    )
