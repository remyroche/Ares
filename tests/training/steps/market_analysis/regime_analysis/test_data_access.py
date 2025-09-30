import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.training.steps.market_analysis.regime_analysis import data_access


def _write_sample_regime_file(path: Path) -> pd.DataFrame:
    try:
        import pyarrow  # noqa: F401  # ensure parquet engine is available
    except ImportError:  # pragma: no cover - environment dependent
        pytest.skip("pyarrow is required to generate parquet fixtures")

    timestamps = pd.date_range("2024-01-01", periods=5, freq="h")
    regime_frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "regime_id": [0, 1, 1, 2, 0],
            "regime_prob": [0.8, 0.7, 0.65, 0.9, 0.6],
            "nas_feature_0": [10.0, 12.0, 9.5, 11.2, 8.7],
            "nas_feature_1": [0.5, 0.7, 0.6, 0.2, 0.1],
            "tas_feature_0": [100.0, 120.0, 110.0, 130.0, 95.0],
            "tas_feature_1": [1.1, 0.9, 1.4, 1.2, 0.7],
        }
    ).set_index("timestamp")

    path.parent.mkdir(parents=True, exist_ok=True)
    regime_frame.to_parquet(path, engine="pyarrow")
    return regime_frame


def test_get_clustering_directory(tmp_path):
    cluster_dir = tmp_path / "nas_tas_clustering" / "ETHUSDT"
    cluster_dir.mkdir(parents=True)

    result = data_access.get_clustering_directory(tmp_path, "ETHUSDT")
    assert result == cluster_dir


def test_find_latest_regime_file(tmp_path):
    cluster_dir = tmp_path / "nas_tas_clustering" / "ETHUSDT"
    cluster_dir.mkdir(parents=True)
    older = cluster_dir / "nas_tas_regime_assignments_old.parquet"
    newer = cluster_dir / "nas_tas_regime_assignments_new.parquet"
    older.touch()
    newer.touch()
    base_ts = time.time()
    os.utime(older, (base_ts, base_ts))
    os.utime(newer, (base_ts + 5, base_ts + 5))

    latest = data_access.find_latest_regime_file(cluster_dir)
    assert latest == newer


def test_load_regime_assignments(tmp_path):
    regime_file = tmp_path / "nas_tas_regime_assignments_sample.parquet"
    expected_frame = _write_sample_regime_file(regime_file)

    frame = data_access.load_regime_assignments(regime_file)
    pdt.assert_frame_equal(frame, expected_frame)

    assert set(frame.columns) >= {
        "regime_id",
        "nas_feature_0",
        "nas_feature_1",
        "tas_feature_0",
        "tas_feature_1",
    }


def test_load_regime_datasets(tmp_path):
    cluster_dir = tmp_path / "nas_tas_clustering" / "ETHUSDT"
    regime_file = cluster_dir / "nas_tas_regime_assignments_20240101.parquet"
    expected_frame = _write_sample_regime_file(regime_file)

    datasets = data_access.load_regime_datasets(tmp_path, "ETHUSDT")
    nas_features, nas_labels, tas_features, tas_labels = datasets

    expected_nas = expected_frame[["nas_feature_0", "nas_feature_1"]].to_numpy()
    expected_tas = expected_frame[["tas_feature_0", "tas_feature_1"]].to_numpy()

    nas_mean = expected_nas.mean(axis=0)
    nas_std = expected_nas.std(axis=0, ddof=0)
    nas_std[nas_std == 0.0] = 1.0
    tas_mean = expected_tas.mean(axis=0)
    tas_std = expected_tas.std(axis=0, ddof=0)
    tas_std[tas_std == 0.0] = 1.0

    np.testing.assert_allclose(nas_features, (expected_nas - nas_mean) / nas_std)
    np.testing.assert_allclose(tas_features, (expected_tas - tas_mean) / tas_std)
    np.testing.assert_array_equal(nas_labels, expected_frame["regime_id"].to_numpy())
    np.testing.assert_array_equal(tas_labels, expected_frame["regime_id"].to_numpy())

    np.testing.assert_allclose(nas_features.mean(axis=0), np.zeros(nas_features.shape[1]), atol=1e-9)
    np.testing.assert_allclose(tas_features.mean(axis=0), np.zeros(tas_features.shape[1]), atol=1e-9)
