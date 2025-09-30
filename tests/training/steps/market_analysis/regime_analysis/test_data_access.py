import os
import time

import pandas as pd
import numpy as np

from src.training.steps.market_analysis.regime_analysis import data_access


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


def test_load_regime_assignments(monkeypatch, tmp_path):
    regime_file = tmp_path / "nas_tas_regime_assignments_sample.parquet"
    regime_file.touch()

    called = {}

    def fake_read_parquet(path):
        called["path"] = path
        return pd.DataFrame({"regime_id": [0, 1, 1]})

    monkeypatch.setattr(data_access.pd, "read_parquet", fake_read_parquet)
    frame = data_access.load_regime_assignments(regime_file)
    assert called["path"] == regime_file
    assert list(frame["regime_id"]) == [0, 1, 1]


def test_load_regime_datasets(monkeypatch, tmp_path):
    cluster_dir = tmp_path / "nas_tas_clustering" / "ETHUSDT"
    cluster_dir.mkdir(parents=True)
    regime_file = cluster_dir / "nas_tas_regime_assignments_sample.parquet"
    regime_file.touch()

    frame = pd.DataFrame({"regime_id": [0, 1, 1, 2]})
    monkeypatch.setattr(data_access, "load_regime_assignments", lambda _: frame)

    datasets = data_access.load_regime_datasets(tmp_path, "ETHUSDT")
    nas_features, nas_labels, tas_features, tas_labels = datasets

    assert nas_features.shape == (4, data_access.DEFAULT_FEATURE_COUNT)
    assert tas_features.shape == (4, data_access.DEFAULT_FEATURE_COUNT)
    np.testing.assert_array_equal(nas_labels, frame["regime_id"].values)
    np.testing.assert_array_equal(tas_labels, frame["regime_id"].values)
