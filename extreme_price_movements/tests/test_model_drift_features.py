import numpy as np
import pandas as pd

from extreme_price_movements.model_drift_features import (
    MODEL_DRIFT_FEATURE_KEYS,
    fit_model_drift_state,
    transform_model_drift_features,
)


def test_model_drift_features_are_artifact_backed_and_stable():
    rng = np.random.default_rng(42)
    x_train = pd.DataFrame(
        rng.normal(size=(120, 6)).astype(np.float32),
        columns=[f"f{i}" for i in range(6)],
    )
    state = fit_model_drift_state(x_train, feature_columns=list(x_train.columns), window=20)
    assert state["enabled"] is True

    drift = transform_model_drift_features(x_train.iloc[:10], state)

    assert list(drift.columns) == list(MODEL_DRIFT_FEATURE_KEYS)
    assert np.isfinite(drift.to_numpy(dtype=np.float32)).all()
    assert drift["regime_centroid_similarity_train"].between(0.0, 1.0).all()
    assert drift["feature_drift_psi_core_50"].between(0.0, 1.0).all()
    assert drift["feature_drift_psi_core_80"].between(0.0, 1.0).all()
    assert drift["inference_drift_score"].between(0.0, 1.0).all()


def test_model_drift_score_increases_for_shifted_rows():
    rng = np.random.default_rng(7)
    x_train = pd.DataFrame(
        rng.normal(size=(200, 5)).astype(np.float32),
        columns=[f"f{i}" for i in range(5)],
    )
    state = fit_model_drift_state(x_train, feature_columns=list(x_train.columns))
    in_domain = transform_model_drift_features(x_train.iloc[:20], state)
    shifted = transform_model_drift_features(x_train.iloc[:20] + 4.0, state)

    assert float(shifted["inference_drift_score"].mean()) > float(
        in_domain["inference_drift_score"].mean()
    )
    assert float(shifted["regime_centroid_similarity_train"].mean()) < float(
        in_domain["regime_centroid_similarity_train"].mean()
    )


def test_model_drift_features_are_row_stable_across_batch_shapes():
    rng = np.random.default_rng(11)
    x_train = pd.DataFrame(
        rng.normal(size=(180, 5)).astype(np.float32),
        columns=[f"f{i}" for i in range(5)],
    )
    state = fit_model_drift_state(
        x_train,
        feature_columns=list(x_train.columns),
        window=20,
    )
    batch = x_train.iloc[10:18].copy()
    batch_drift = transform_model_drift_features(batch, state, index=batch.index)

    for idx in batch.index:
        single = transform_model_drift_features(
            batch.loc[[idx]],
            state,
            index=pd.Index([idx]),
        )
        assert np.allclose(
            batch_drift.loc[idx].to_numpy(dtype=np.float32),
            single.iloc[0].to_numpy(dtype=np.float32),
            rtol=0.0,
            atol=1e-7,
        )
