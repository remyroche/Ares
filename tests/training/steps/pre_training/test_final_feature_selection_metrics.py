from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.training.steps.pre_training.final_feature_selection_pipeline import (
    FeatureSelectionConfig,
    MultiStageFeatureSelector,
)


def _build_selector(**kwargs) -> MultiStageFeatureSelector:
    config = FeatureSelectionConfig(
        rf_n_estimators=kwargs.get("rf_n_estimators", 25),
        cv_folds=kwargs.get("cv_folds", 3),
        trading_cost=kwargs.get("trading_cost", 0.0),
        trading_horizon=kwargs.get("trading_horizon", 52),
        turnover_penalty=kwargs.get("turnover_penalty", 0.0),
        ic_method=kwargs.get("ic_method", "spearman"),
        save_analysis=False,
        verbose=False,
    )
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "src.training.steps.pre_training.final_feature_selection_pipeline.get_unified_matrix_operations",
                return_value=None,
            )
        )
        stack.enter_context(
            patch(
                "src.training.steps.pre_training.final_feature_selection_pipeline.get_batch_matrix_processor",
                return_value=None,
            )
        )
        stack.enter_context(
            patch(
                "src.training.steps.pre_training.final_feature_selection_pipeline.get_unified_hardware_manager",
                return_value=None,
            )
        )
        stack.enter_context(
            patch(
                "src.training.steps.pre_training.final_feature_selection_pipeline.get_adaptive_optimization_engine",
                return_value=None,
            )
        )
        stack.enter_context(
            patch(
                "src.training.steps.pre_training.final_feature_selection_pipeline.get_advanced_memory_optimizer",
                return_value=None,
            )
        )
        return MultiStageFeatureSelector(config)


def test_trading_metrics_reward_directional_skill():
    selector = _build_selector(trading_cost=0.0, turnover_penalty=0.0)
    n_samples = 120
    index = pd.RangeIndex(n_samples)
    returns = pd.Series(np.linspace(0.0005, 0.002, n_samples), index=index)
    predictions = returns.copy()

    metrics = selector._evaluate_trading_metrics(returns, predictions)

    assert metrics["information_coefficient"] > 0.99
    assert metrics["long_short_sharpe"] > 0
    assert metrics["turnover"] < 0.05  # near buy-and-hold turnover


def test_trading_metrics_penalize_turnover():
    selector = _build_selector(trading_cost=0.0, turnover_penalty=0.5)
    n_samples = 200
    index = pd.RangeIndex(n_samples)
    base_returns = pd.Series(
        0.001 + 0.0006 * np.sin(np.linspace(0, 4 * np.pi, n_samples)), index=index
    )
    low_turnover_signal = base_returns.copy()
    high_freq_component = 0.0005 * np.sign(np.sin(np.linspace(0, 40 * np.pi, n_samples)))
    high_turnover_signal = base_returns + high_freq_component

    metrics_low = selector._evaluate_trading_metrics(base_returns, low_turnover_signal)
    metrics_high = selector._evaluate_trading_metrics(base_returns, high_turnover_signal)

    assert metrics_high["turnover"] > metrics_low["turnover"]
    assert metrics_high["long_short_sharpe"] < metrics_low["long_short_sharpe"]


def test_compile_results_populates_trading_metrics():
    selector = _build_selector(trading_cost=0.0001, turnover_penalty=0.1, rf_n_estimators=10)

    n_samples = 90
    rng = np.random.default_rng(42)
    index = pd.RangeIndex(n_samples)
    base_signal = np.sin(np.linspace(0, 3 * np.pi, n_samples))

    X = pd.DataFrame(
        {
            "feature_a": base_signal + 0.05 * rng.standard_normal(n_samples),
            "feature_b": np.roll(base_signal, 1),
            "feature_c": np.roll(base_signal, 2),
        },
        index=index,
    )
    y = pd.Series(base_signal + 0.02 * rng.standard_normal(n_samples), index=index)

    features = list(X.columns)
    scores = {feature: 1.0 for feature in features}

    selector._compile_results(
        X,
        y,
        features,
        features,
        features,
        scores,
        scores,
        scores,
    )

    final_scores = selector.results.final_scores
    assert "information_coefficient" in final_scores
    assert "long_short_sharpe" in final_scores
    assert "turnover" in final_scores
    assert final_scores.get("cv_folds") == selector.config.cv_folds
    assert selector.results.model_performance["fold_metrics"]
    assert selector.results.model_performance["validation_predictions"]
