import sys
import types

import numpy as np
import pandas as pd


def _install_feature_selection_stubs() -> None:
    """Install lightweight stubs to avoid heavy optional imports during testing."""

    selection_module_name = 'src.training.utils.feature_selection.selection_methods'
    if selection_module_name not in sys.modules:
        selection_module = types.ModuleType(selection_module_name)

        def _selection_stub(*args, **kwargs):  # pragma: no cover - trivial stub
            return []

        selection_module.mrmr_selection = _selection_stub
        selection_module.lasso_selection = _selection_stub
        selection_module.correlation_filtering = _selection_stub
        selection_module.recursive_feature_elimination = _selection_stub
        selection_module.variance_filtering = _selection_stub
        sys.modules[selection_module_name] = selection_module

    analyzer_module_name = 'src.utils.feature_selection.feature_importance_analyzer'
    if analyzer_module_name not in sys.modules:
        analyzer_module = types.ModuleType(analyzer_module_name)

        class FeatureImportanceAnalyzer:  # pragma: no cover - placeholder
            def __init__(self, *args, **kwargs) -> None:
                pass

        class ImportanceMethod:  # pragma: no cover - placeholder
            pass

        class FeatureImportanceConfig:  # pragma: no cover - placeholder
            def __init__(self, *args, **kwargs) -> None:
                pass

        analyzer_module.FeatureImportanceAnalyzer = FeatureImportanceAnalyzer
        analyzer_module.ImportanceMethod = ImportanceMethod
        analyzer_module.FeatureImportanceConfig = FeatureImportanceConfig
        sys.modules[analyzer_module_name] = analyzer_module

    quality_module_name = 'src.training.utils.feature_selection.quality_metrics'
    if quality_module_name not in sys.modules:
        quality_module = types.ModuleType(quality_module_name)

        def calculate_feature_quality_metrics(*args, **kwargs):  # pragma: no cover - stub
            return {}

        class FeatureQualityMetrics:  # pragma: no cover - placeholder
            pass

        quality_module.calculate_feature_quality_metrics = calculate_feature_quality_metrics
        quality_module.FeatureQualityMetrics = FeatureQualityMetrics
        sys.modules[quality_module_name] = quality_module

    matrix_module_name = 'src.utils.matrix_operations'
    if matrix_module_name not in sys.modules:
        matrix_module = types.ModuleType(matrix_module_name)

        class _DummyMatrixOps:  # pragma: no cover - placeholder
            pass

        def get_unified_matrix_operations(*args, **kwargs):  # pragma: no cover - stub
            return _DummyMatrixOps()

        def _matrix_stub(*args, **kwargs):  # pragma: no cover - stub
            return None

        matrix_module.get_unified_matrix_operations = get_unified_matrix_operations
        matrix_module.correlation_matrix_gpu = _matrix_stub
        matrix_module.matrix_correlation_analysis = _matrix_stub
        matrix_module.batch_correlation_analysis = _matrix_stub
        matrix_module.optimize_dataframe = lambda df, *args, **kwargs: df
        matrix_module.get_batch_matrix_processor = _matrix_stub
        sys.modules[matrix_module_name] = matrix_module

    hardware_module_name = 'src.utils.hardware'
    if hardware_module_name not in sys.modules:
        hardware_module = types.ModuleType(hardware_module_name)

        def _hardware_stub(*args, **kwargs):  # pragma: no cover - stub
            return object()

        class WorkloadType:  # pragma: no cover - placeholder
            GENERAL = 'general'

        hardware_module.get_unified_hardware_manager = _hardware_stub
        hardware_module.get_adaptive_optimization_engine = _hardware_stub
        hardware_module.get_advanced_memory_optimizer = _hardware_stub
        hardware_module.WorkloadType = WorkloadType
        sys.modules[hardware_module_name] = hardware_module


_install_feature_selection_stubs()

from src.training.steps.pre_training.final_feature_selection_pipeline import (
    FeatureSelectionConfig,
    MultiStageFeatureSelector,
)


def test_feature_selector_reports_pr_auc_for_skewed_labels():
    """The feature selector should report PR-AUC for imbalanced labels."""

    rng = np.random.default_rng(42)
    n_samples = 120
    n_features = 8

    feature_columns = [f"feature_{i}" for i in range(n_features)]
    X = pd.DataFrame(rng.normal(size=(n_samples, n_features)), columns=feature_columns)

    # Highly skewed binary labels with a small positive class
    y = pd.Series(np.zeros(n_samples, dtype=int))
    positive_indices = rng.choice(n_samples, size=max(3, n_samples // 20), replace=False)
    y.iloc[positive_indices] = 1

    config = FeatureSelectionConfig(
        initial_features=n_features,
        stage_1_target=6,
        stage_2_target=5,
        stage_3_target=4,
        target_features=4,
        min_features=4,
        max_features=n_features,
        model_type='FinancialResNet',
        separate_directional_features=False,
        enable_mutual_information=False,
        enable_rfe=False,
        enable_chunked_processing=False,
        enable_entropy_balancing=False,
        enable_early_termination=False,
        use_existing_framework=False,
        selection_methods=[],
        existing_methods=[],
        shap_max_features=16,
        shap_sample_size=64,
        rf_n_estimators=25,
        cv_folds=3,
    )

    selector = MultiStageFeatureSelector(config=config)
    result = selector.select_features(X, y)

    assert 'average_precision' in result.final_scores
    pr_auc = result.final_scores['average_precision']
    assert isinstance(pr_auc, float)
    assert 0.0 <= pr_auc <= 1.0
