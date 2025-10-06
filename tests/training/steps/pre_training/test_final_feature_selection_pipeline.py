import numpy as np
import pandas as pd

from src.training.steps.pre_training.final_feature_selection_pipeline import (
    FeatureSelectionConfig,
    MultiStageFeatureSelector,
)


def test_multistage_selector_default_stage_targets_progress():
    rng = np.random.default_rng(42)
    samples = 60
    features = 6

    X = pd.DataFrame(
        rng.normal(size=(samples, features)),
        columns=[f"feature_{idx}" for idx in range(features)],
    )
    y = pd.Series(rng.normal(size=samples))

    config = FeatureSelectionConfig(
        initial_features=features,
        stage_1_target=4,
        stage_2_target=3,
        stage_3_target=2,
        rf_n_estimators=5,
        rf_max_depth=3,
        enable_chunked_processing=False,
        enable_early_termination=False,
        enable_rfe=False,
        enable_mutual_information=False,
        save_models=False,
        save_analysis=False,
        separate_directional_features=False,
        cv_folds=2,
    )

    selector = MultiStageFeatureSelector(config=config)

    result = selector.select_features(X, y)

    assert len(result.stage_1_features) == config.stage_1_target
    assert len(result.stage_2_features) == config.stage_2_target
    assert len(result.stage_3_features) == config.stage_3_target
    assert result.final_features == result.stage_3_features
    assert result.feature_counts["stage_1"] == config.stage_1_target
    assert result.feature_counts["stage_2"] == config.stage_2_target
    assert result.feature_counts["stage_3"] == config.stage_3_target
