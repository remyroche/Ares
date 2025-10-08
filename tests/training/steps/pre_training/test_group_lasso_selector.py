import sys
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd


_STUBBED_MODULES = {
    "src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation.pipeline": {
        "CrossTimeframePipeline": object,
    },
    "src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation.phase1_probe": {
        "Phase1HTFProbe": object,
    },
    "src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation.phase2_optimization": {
        "Phase2Optimization": object,
    },
    "src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation.regime_segmentation": {
        "RegimeSegmentation": object,
    },
    "src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation.scoring_system": {
        "AdaptiveScoringSystem": object,
    },
    "src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation.ehu_rih_assignment": {
        "EHU_RIH_Assignment": object,
    },
    "src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation.knapsack_selection": {
        "KnapsackSelection": object,
        "CrossTimeframeKnapsackSelectionResult": object,
    },
    "src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation.htf_materialization": {
        "HTFMaterialization": object,
    },
    "src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation.interaction_templates": {
        "HTFInteractionTemplates": object,
    },
    "src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation.evaluation": {
        "WalkForwardEvaluation": object,
    },
    "src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation.monitoring": {
        "MonitoringSystem": object,
    },
    "src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation.staleness_curve": {
        "StalenessCurveCalculator": object,
        "StalenessCurve": object,
        "StalenessSummary": object,
    },
}


for module_name, attributes in _STUBBED_MODULES.items():
    if module_name not in sys.modules:
        module = types.ModuleType(module_name)
        for attr, value in attributes.items():
            setattr(module, attr, value)
        sys.modules[module_name] = module


import src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation.statistical_selection as statistical_selection_module


def test_group_lasso_scaler_uses_training_slice_only():
    selector = statistical_selection_module.GroupLASSOSelector(SimpleNamespace())

    index = pd.date_range("2023-01-01", periods=10, freq="D")
    feature_values = np.arange(30).reshape(10, 3).astype(float)
    features = pd.DataFrame(feature_values, index=index, columns=["f1", "f2", "f3"])
    targets = pd.Series(np.linspace(0, 1, 10), index=index)

    train_features = features.iloc[:6]
    val_features = features.iloc[6:]
    train_targets = targets.iloc[:6]
    val_targets = targets.iloc[6:]

    feature_groups = {
        "group_a": ["f1", "f2"],
        "group_b": ["f3"],
    }

    selector.select_features(
        features,
        targets,
        feature_groups,
        split=(train_features, train_targets, val_features, val_targets),
    )

    assert selector._last_scaler is not None

    expected_mean = train_features.mean().values
    expected_var = train_features.var(ddof=0).values

    np.testing.assert_allclose(selector._last_scaler.mean_, expected_mean)
    np.testing.assert_allclose(selector._last_scaler.var_, expected_var)
