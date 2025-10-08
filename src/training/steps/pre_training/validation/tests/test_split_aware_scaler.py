import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from ..schemas import SplitAwareScaler


def test_split_aware_scaler_uses_train_statistics():
    data = pd.DataFrame({"value": [0.0, 1.0, 2.0, 10.0, 11.0, 12.0]})
    splits = {
        "train": np.array([0, 1, 2]),
        "val": np.array([3, 4]),
        "test": np.array([5]),
    }

    scaler = SplitAwareScaler(StandardScaler(), split_indices=splits)

    with pytest.raises(RuntimeError):
        scaler.transform(data, split="train")

    scaler.fit(data, split="train")

    train_values = data.iloc[splits["train"]].to_numpy()
    train_mean = train_values.mean()
    train_std = train_values.std(ddof=0)

    assert pytest.approx(scaler.base_scaler.mean_[0]) == train_mean
    assert pytest.approx(scaler.base_scaler.scale_[0]) == train_std

    val_transformed = scaler.transform(data, split="val")
    expected_val = (data.iloc[splits["val"]].to_numpy() - train_mean) / train_std
    assert np.allclose(val_transformed, expected_val)

    np.testing.assert_array_equal(scaler.get_split_indices("test"), splits["test"])

    scaler_two = SplitAwareScaler(StandardScaler(), split_indices=splits)
    with pytest.raises(ValueError):
        scaler_two.fit(data, split="val")

    with pytest.raises(ValueError):
        scaler.fit_transform(data, split="val")
