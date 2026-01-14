import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_module(module_name: str, relative_path: str):
    import sys
    import types

    if "h5py" not in sys.modules:
        sys.modules["h5py"] = types.ModuleType("h5py")

    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_onc_clusters_transpose_guard():
    rng = np.random.default_rng(42)
    data = pd.DataFrame(rng.normal(size=(10, 3)), columns=["f1", "f2", "f3"])
    module = _load_module(
        "de_prado_feature_engine",
        "src/training/steps/labeling/de_prado_feature_engine.py",
    )
    engine = module.DePradoFeatureEngine(max_clusters=3)

    clusters = engine._get_onc_clusters(data.T)

    assert len(clusters) == 3


def test_afml_align_probabilities_preserves_neutral_fillers():
    module = _load_module(
        "afml_specialist_mixin",
        "src/training/steps/market_analysis/afml_specialist_mixin.py",
    )

    class Dummy(module.AFMLSpecialistMixin):
        pass

    mixin = Dummy()
    target_index = pd.date_range("2024-01-01", periods=6, freq="15T")
    oof_probs = pd.Series(
        [0.45, 0.62, 0.49],
        index=target_index[-3:],
    )

    aligned_probs, aligned_preds = mixin._align_probabilities_to_index(
        oof_probs=oof_probs,
        target_index=target_index,
        neutral_value=0.5,
        threshold=0.5,
    )

    # First three rows were neutral fillers; ensure they stay neutral and binary=0
    pd.testing.assert_index_equal(aligned_probs.index, target_index)
    assert (aligned_probs.iloc[:3] == 0.5).all()
    assert (aligned_preds.iloc[:3] == 0).all()

    # Real probabilities thresholded on >=0.5 for confident rows only
    assert aligned_preds.iloc[3] == 0  # 0.45 -> 0
    assert aligned_preds.iloc[4] == 1  # 0.62 -> 1
    assert aligned_preds.iloc[5] == 0  # 0.49 -> 0


def test_catboost_baseline_no_shrinkage_warning():
    """Regression test: CatBoost with baseline should not produce shrinkage warning."""
    import io
    import sys
    import contextlib
    from catboost import CatBoostRegressor, Pool

    # Generate synthetic data
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(200, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(rng.normal(size=200))
    baseline = pd.Series(rng.normal(size=200))

    # CatBoost params matching the fixed params in train_specialists_with_gmm_step.py
    cb_params = {
        'subsample': 0.6,
        'colsample_bylevel': 0.5,
        'leaf_estimation_iterations': 10,
        'l2_leaf_reg': 20,
        'random_strength': 5,
        'bootstrap_type': 'MVS',
        'verbose': False,
        'allow_writing_files': False,
        'model_shrink_rate': 0,  # This should prevent the warning
        'iterations': 10,  # Small for test
        'depth': 4,
    }

    # Capture stdout/stderr
    captured_output = io.StringIO()
    with contextlib.redirect_stderr(captured_output), contextlib.redirect_stdout(captured_output):
        # Train with baseline (warm start)
        train_pool = Pool(X, y, baseline=baseline)
        model = CatBoostRegressor(**cb_params)
        model.fit(train_pool)

    # Check that the shrinkage warning is NOT present
    output = captured_output.getvalue()
    assert "Model shrinkage in combination with baseline column is not implemented yet" not in output
    assert "Reset model_shrink_rate to 0" not in output
