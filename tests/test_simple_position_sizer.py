import pytest
import numpy as np
import pandas as pd
from extreme_price_movements.simple_position_sizer import (
    detect_meta_head_keys,
    run_simple_position_sizer,
    run_bucketed_simple_position_sizer
)

@pytest.fixture
def dummy_data():
    n_samples = 100
    np.random.seed(42)

    feature_dict = {
        "model_edge": np.random.randn(n_samples),
        "model_mae": np.random.rand(n_samples) * 0.05,
        "model_mfe": np.random.rand(n_samples) * 0.1,
        "model_asym": np.random.rand(n_samples),
        "meta_clf_prob": np.random.rand(n_samples),
        "random_noise": np.random.randn(n_samples)  # Should be ignored
    }

    trade_outcomes = pd.DataFrame({"id": range(n_samples)})
    y_raw_net_return = feature_dict["model_edge"] * 0.01 + np.random.randn(n_samples) * 0.005
    y_downside = feature_dict["model_mae"] + np.random.rand(n_samples) * 0.01
    timestamps = np.arange(n_samples) * 3600

    return feature_dict, trade_outcomes, y_raw_net_return, y_downside, timestamps

def test_detect_meta_head_keys(dummy_data):
    feature_dict, _, _, _, _ = dummy_data
    heads = detect_meta_head_keys(feature_dict)

    assert "model_edge" in heads
    assert heads["model_edge"] == "return-like"
    assert "model_mae" in heads
    assert heads["model_mae"] == "risk-like"
    assert "random_noise" not in heads

def test_run_simple_position_sizer(dummy_data):
    feature_dict, trade_outcomes, y_raw_net_return, y_downside, timestamps = dummy_data

    res = run_simple_position_sizer(
        feature_dict=feature_dict,
        trade_outcomes=trade_outcomes,
        y_raw_net_return=y_raw_net_return,
        y_downside=y_downside,
        timestamps=timestamps,
        use_ridge_head_sizer=True
    )

    assert "feature_coverage_report_" in res
    assert "model_edge" in res["feature_coverage_report_"]["used_heads"]
    assert "random_noise" not in res["feature_coverage_report_"]["used_heads"]

    assert "head_diagnostics_table_" in res
    assert not res["head_diagnostics_table_"].empty

    assert "combo_race_table_" in res
    assert not res["combo_race_table_"].empty

    assert "ridge_sizer_eval_" in res
    assert res["best_simple_score_"] is not None
    assert len(res["best_simple_score_"]) == 100

    assert "profit_proxy_table_" in res
    assert not res["profit_proxy_table_"].empty

def test_run_bucketed_simple_position_sizer(dummy_data):
    feature_dict, trade_outcomes, y_raw_net_return, y_downside, timestamps = dummy_data
    bucket_labels = np.random.choice([0, 1], size=len(y_raw_net_return))

    res = run_bucketed_simple_position_sizer(
        feature_dict=feature_dict,
        trade_outcomes=trade_outcomes,
        y_raw_net_return=y_raw_net_return,
        y_downside=y_downside,
        timestamps=timestamps,
        bucket_labels=bucket_labels,
        min_bucket_samples=10  # Low threshold for test
    )

    assert "bucket_summary_table_" in res
    assert "bucket_results" in res
    assert len(res["bucket_summary_table_"]) > 0
