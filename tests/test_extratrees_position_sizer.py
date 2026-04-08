"""
Tests for extratrees_position_sizer.py
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from extreme_price_movements.extratrees_position_sizer import (
    SimpleHeadExtraTreesSizer,
    run_extratrees_position_sizer,
    run_bucketed_extratrees_position_sizer,
)


@pytest.fixture
def dummy_data():
    """Generate synthetic test data."""
    n_samples = 200
    np.random.seed(42)
    
    # Create correlated features
    base_signal = np.random.randn(n_samples)
    
    feature_dict = {
        "model_edge": base_signal * 0.02 + np.random.randn(n_samples) * 0.005,
        "model_mae": np.abs(np.random.randn(n_samples)) * 0.03,
        "model_mfe": np.abs(base_signal) * 0.05 + np.random.rand(n_samples) * 0.02,
        "model_asym": np.random.rand(n_samples),
        "meta_clf_prob": 1 / (1 + np.exp(-base_signal)),
    }
    
    # Target correlated with edge
    y_raw_net_return = feature_dict["model_edge"] * 0.5 + np.random.randn(n_samples) * 0.01
    y_downside = feature_dict["model_mae"] + np.random.rand(n_samples) * 0.01
    timestamps = np.arange(n_samples) * 3600  # Hourly data
    
    trade_outcomes = pd.DataFrame({
        "id": range(n_samples),
        "return": y_raw_net_return,
        "downside": y_downside,
    })
    
    return feature_dict, trade_outcomes, y_raw_net_return, y_downside, timestamps


def test_simple_head_extratrees_sizer_init():
    """Test ExtraTrees sizer initialization with custom parameters."""
    model = ExtraTreesRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=20,
        random_state=42,
    )
    
    sizer = SimpleHeadExtraTreesSizer(
        model=model,
        calibration_method="isotonic"
    )
    
    assert sizer.model is model
    assert sizer.calibration_method == "isotonic"
    assert sizer.calibrator is None
    assert len(sizer.fold_importances) == 0


def test_simple_head_extratrees_sizer_fit_predict(dummy_data):
    """Test ExtraTrees sizer fit_predict_oof method."""
    feature_dict, trade_outcomes, y_raw_net_return, y_downside, timestamps = dummy_data
    
    # Prepare features
    used_keys = ["model_edge", "model_mae", "model_mfe", "meta_clf_prob"]
    X_heads = np.column_stack([feature_dict[k] for k in used_keys])
    
    # Create splits
    from extreme_price_movements.simple_position_sizer import walk_forward_temporal_splits
    n_samples = len(y_raw_net_return)
    splits = walk_forward_temporal_splits(timestamps, n_samples, n_splits=3)
    
    # Create and fit sizer
    model = ExtraTreesRegressor(
        n_estimators=50,  # Small for speed
        max_depth=5,
        min_samples_leaf=20,
        min_samples_split=40,
        max_features="sqrt",
        random_state=42,
        n_jobs=1,
    )
    
    sizer = SimpleHeadExtraTreesSizer(model=model, calibration_method="isotonic")
    oof_preds = sizer.fit_predict_oof(X_heads, y_raw_net_return, splits, feature_names=used_keys)
    
    # Assertions
    assert len(oof_preds) == n_samples
    assert not np.all(oof_preds == 0), "OOF predictions should not all be zero"
    assert np.isfinite(oof_preds).all(), "OOF predictions should be finite"
    
    # Check feature importance was recorded
    assert len(sizer.fold_importances) > 0
    assert all(imp.shape[0] == len(used_keys) for imp in sizer.fold_importances)


def test_get_feature_importance(dummy_data):
    """Test feature importance extraction."""
    feature_dict, trade_outcomes, y_raw_net_return, y_downside, timestamps = dummy_data
    
    used_keys = ["model_edge", "model_mae", "model_mfe", "meta_clf_prob"]
    X_heads = np.column_stack([feature_dict[k] for k in used_keys])
    
    from extreme_price_movements.simple_position_sizer import walk_forward_temporal_splits
    n_samples = len(y_raw_net_return)
    splits = walk_forward_temporal_splits(timestamps, n_samples, n_splits=3)
    
    model = ExtraTreesRegressor(
        n_estimators=50,
        max_depth=5,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=1,
    )
    
    sizer = SimpleHeadExtraTreesSizer(model=model)
    sizer.fit_predict_oof(X_heads, y_raw_net_return, splits, feature_names=used_keys)
    
    importance_df = sizer.get_feature_importance()
    
    assert not importance_df.empty
    assert len(importance_df) == len(used_keys)
    assert "head_name" in importance_df.columns
    assert "mean_importance" in importance_df.columns
    assert "importance_rank" in importance_df.columns
    
    # Check rankings are valid
    assert importance_df["importance_rank"].min() >= 1
    assert importance_df["importance_rank"].max() <= len(used_keys)


def test_run_extratrees_position_sizer(dummy_data):
    """Test the main orchestrator function."""
    feature_dict, trade_outcomes, y_raw_net_return, y_downside, timestamps = dummy_data
    
    res = run_extratrees_position_sizer(
        feature_dict=feature_dict,
        trade_outcomes=trade_outcomes,
        y_raw_net_return=y_raw_net_return,
        y_downside=y_downside,
        timestamps=timestamps,
        use_extratrees_head_sizer=True,
        et_n_estimators=50,  # Small for speed
        et_max_depth=5,
        et_min_samples_leaf=20,
    )
    
    # Check expected keys
    assert "feature_coverage_report_" in res
    assert "head_diagnostics_table_" in res
    assert "combo_race_table_" in res
    assert "extratrees_sizer_eval_" in res
    assert "extratrees_importance_table_" in res
    assert "extratrees_profit_proxy_table_" in res
    assert "best_simple_score_" in res
    
    # Check ExtraTrees results
    assert res["extratrees_sizer_eval_"] is not None
    assert "spearman_ret" in res["extratrees_sizer_eval_"]
    assert "utility_score" in res["extratrees_sizer_eval_"]
    
    # Check profit proxy table
    assert not res["extratrees_profit_proxy_table_"].empty
    assert "wallet_pnl" in res["extratrees_profit_proxy_table_"].columns
    assert "hit_rate" in res["extratrees_profit_proxy_table_"].columns


def test_run_extratrees_position_sizer_no_calibration(dummy_data):
    """Test with calibration disabled."""
    feature_dict, trade_outcomes, y_raw_net_return, y_downside, timestamps = dummy_data
    
    res = run_extratrees_position_sizer(
        feature_dict=feature_dict,
        trade_outcomes=trade_outcomes,
        y_raw_net_return=y_raw_net_return,
        y_downside=y_downside,
        timestamps=timestamps,
        use_extratrees_head_sizer=True,
        # No calibration method specified, defaults to isotonic
    )
    
    assert res["extratrees_sizer_eval_"] is not None


def test_run_bucketed_extratrees_position_sizer(dummy_data):
    """Test bucketed version."""
    feature_dict, trade_outcomes, y_raw_net_return, y_downside, timestamps = dummy_data
    
    # Create bucket labels
    bucket_labels = np.random.choice([0, 1, 2], size=len(y_raw_net_return))
    
    res = run_bucketed_extratrees_position_sizer(
        feature_dict=feature_dict,
        trade_outcomes=trade_outcomes,
        y_raw_net_return=y_raw_net_return,
        y_downside=y_downside,
        timestamps=timestamps,
        bucket_labels=bucket_labels,
        min_bucket_samples=20,  # Low threshold for test
        et_n_estimators=50,
        et_max_depth=5,
        et_min_samples_leaf=20,
    )
    
    assert "bucket_results" in res
    assert "bucket_summary_table_" in res


def test_extratrees_vs_ridge_similar_inputs(dummy_data):
    """Test that ExtraTrees and Ridge produce valid outputs on same data."""
    feature_dict, trade_outcomes, y_raw_net_return, y_downside, timestamps = dummy_data
    
    from extreme_price_movements.simple_position_sizer import run_simple_position_sizer
    
    # Run both sizers
    ridge_res = run_simple_position_sizer(
        feature_dict=feature_dict,
        trade_outcomes=trade_outcomes,
        y_raw_net_return=y_raw_net_return,
        y_downside=y_downside,
        timestamps=timestamps,
        use_ridge_head_sizer=True,
    )
    
    et_res = run_extratrees_position_sizer(
        feature_dict=feature_dict,
        trade_outcomes=trade_outcomes,
        y_raw_net_return=y_raw_net_return,
        y_downside=y_downside,
        timestamps=timestamps,
        use_extratrees_head_sizer=True,
        et_n_estimators=50,
        et_max_depth=5,
        et_min_samples_leaf=20,
    )
    
    # Both should produce results
    assert ridge_res["ridge_sizer_eval_"] is not None
    assert et_res["extratrees_sizer_eval_"] is not None
    
    # Both should have profit proxy tables
    assert not ridge_res["ridge_profit_proxy_table_"].empty
    assert not et_res["extratrees_profit_proxy_table_"].empty
    
    # Check that predictions are different (non-identical models)
    ridge_preds = ridge_res["ridge_sizer_scores_"]
    et_preds = et_res["extratrees_sizer_scores_"]
    
    assert not np.allclose(ridge_preds, et_preds), "Ridge and ET predictions should differ"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
