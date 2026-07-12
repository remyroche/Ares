from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

from extreme_price_movements.alternative_meta_residual_bundle import (
    _apply_ood_state,
    _apply_residual_representation,
)


def test_apply_ood_state_uses_frozen_train_statistics() -> None:
    frame = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, 8.0]})
    state = {
        "columns": ["a", "b"],
        "mean": np.asarray([1.0, 2.0], dtype=np.float32),
        "std": np.asarray([1.0, 2.0], dtype=np.float32),
        "q25": np.asarray([0.5, 1.0], dtype=np.float32),
        "q75": np.asarray([1.5, 3.0], dtype=np.float32),
    }
    out = _apply_ood_state(frame, state)
    assert out.loc[0, "meta_sel_ood_abs_z_max"] == 0.0
    assert out.loc[1, "meta_sel_ood_missing_frac"] == 0.5
    assert out.loc[1, "meta_sel_ood_abs_z_max"] == 3.0


def test_apply_residual_representation_reuses_frozen_robust_pca() -> None:
    train = np.asarray(
        [[0.0, 1.0], [1.0, 2.0], [2.0, 4.0], [3.0, 8.0]], dtype=np.float32
    )
    scaler = RobustScaler(quantile_range=(10.0, 90.0)).fit(train)
    scaled = np.clip(scaler.transform(train), -2.0, 2.0)
    pca = PCA(n_components=2, random_state=7).fit(scaled)
    state = {
        "kind": "robust_pca",
        "columns": ["a", "b"],
        "medians": np.asarray([1.5, 3.0], dtype=np.float32),
        "low": np.asarray([0.0, 1.0], dtype=np.float32),
        "high": np.asarray([3.0, 8.0], dtype=np.float32),
        "scaler": scaler,
        "pca": pca,
        "scaled_clip": 2.0,
        "output_columns": ["meta_resid_pca_00", "meta_resid_pca_01"],
    }
    frame = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, 20.0]})
    out = _apply_residual_representation(frame, state, batch_rows=1)
    expected_values = np.asarray([[1.0, 2.0], [1.5, 8.0]], dtype=np.float32)
    expected = pca.transform(np.clip(scaler.transform(expected_values), -2.0, 2.0))
    np.testing.assert_allclose(
        out[["meta_resid_pca_00", "meta_resid_pca_01"]].to_numpy(),
        expected,
        atol=1e-7,
    )
