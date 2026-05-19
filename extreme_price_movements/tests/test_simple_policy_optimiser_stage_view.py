import json

from extreme_price_movements.simple_policy_optimiser import _load_policy_stage_view


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
