"""Small model adapters used by side-routed inference bundles."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class BaseScorePassthroughRegressor:
    """Expose the base score while a downstream residual expert owns ranking."""

    def __init__(self, feature_name: str = "base_score_raw") -> None:
        self.feature_name = str(feature_name)
        self.selected_feature_names_ = [self.feature_name]
        self.feature_columns = [self.feature_name]

    def predict(self, matrix: Any) -> np.ndarray:
        if isinstance(matrix, pd.DataFrame):
            values = pd.to_numeric(matrix[self.feature_name], errors="coerce").to_numpy(
                dtype=np.float32,
                copy=False,
            )
        else:
            values = np.asarray(matrix, dtype=np.float32)
            if values.ndim == 2:
                values = values[:, 0]
        return np.asarray(values, dtype=np.float32).reshape(-1)
