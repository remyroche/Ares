import numpy as np
import pandas as pd

from src.training.steps.pre_training.final_feature_selection_pipeline import (
    FeatureSelectionConfig,
    MultiStageFeatureSelector,
)


def _make_classification_dataset(num_samples: int = 150, num_features: int = 12):
    rng = np.random.default_rng(42)
    data = rng.normal(size=(num_samples, num_features))
    columns = [f"feature_{idx}" for idx in range(num_features)]
    X = pd.DataFrame(data, columns=columns)

    signal = 0.6 * X[columns[0]] - 0.4 * X[columns[1]] + 0.2 * rng.normal(size=num_samples)
    threshold = np.median(signal)
    y = pd.Series((signal > threshold).astype(int), name="target")

    return X, y


def test_stage_two_and_three_execute_without_unbound_target():
    X, y = _make_classification_dataset()

    config = FeatureSelectionConfig(
        initial_features=12,
        stage_1_target=8,
        stage_2_target=5,
        stage_3_target=3,
        min_variance_threshold=0.0,
        min_correlation_threshold=0.999,
        enable_mutual_information=False,
        enable_chunked_processing=False,
        enable_rfe=False,
        enable_early_termination=False,
        save_models=False,
        save_analysis=False,
        separate_directional_features=False,
        cv_scoring="accuracy",
        rf_n_estimators=25,
        rf_max_depth=6,
        rf_min_samples_split=2,
    )

    selector = MultiStageFeatureSelector(config)

    prepared = selector._prepare_initial_features(X, y)
    stage_1_features, _ = selector._stage_1_selection(prepared, y, target_count=config.stage_1_target)

    stage_2_features, stage_2_scores = selector._stage_2_selection(
        prepared[stage_1_features],
        y,
        target_count=config.stage_2_target,
    )

    assert len(stage_2_features) == config.stage_2_target
    assert stage_2_scores

    stage_3_result = selector._stage_3_selection(
        prepared[stage_1_features][stage_2_features],
        y,
        target_count=config.stage_3_target,
    )

    assert stage_3_result is None or (
        isinstance(stage_3_result, tuple) and len(stage_3_result) == 2
    )
