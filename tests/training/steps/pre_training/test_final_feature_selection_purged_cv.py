import sys
import types
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd


class StubPurgedKFoldTime:
    def __init__(self, n_splits: int = 5, purge: pd.Timedelta | int = 0, embargo: pd.Timedelta | int = 0):
        self.n_splits = n_splits
        self.purge = purge
        self.embargo = embargo

    def _to_timedelta(self, value: pd.Timedelta | int) -> pd.Timedelta:
        if isinstance(value, pd.Timedelta):
            return value
        return pd.Timedelta(minutes=int(value))

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[List[object]] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        index = X.index
        n_samples = len(X)
        if self.n_splits < 2 or self.n_splits > n_samples:
            raise ValueError("n_splits must be at least 2 and at most n_samples")

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1

        current = 0
        folds: List[Tuple[int, int]] = []
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            folds.append((start, stop))
            current = stop

        is_time = isinstance(index, pd.DatetimeIndex)

        for start, stop in folds:
            val_idx = np.arange(start, stop)
            train_mask = np.ones(n_samples, dtype=bool)

            if is_time:
                purge_delta = self._to_timedelta(self.purge)
                embargo_delta = self._to_timedelta(self.embargo)
                left_bound = index[start] - purge_delta
                right_bound = index[stop - 1] + embargo_delta
                in_window = (index >= left_bound) & (index <= right_bound)
                train_mask[in_window] = False
            else:
                purge_n = int(self.purge) if isinstance(self.purge, (int, float)) else 0
                embargo_n = int(self.embargo) if isinstance(self.embargo, (int, float)) else 0
                left = max(0, start - purge_n)
                right = min(n_samples, stop + embargo_n)
                train_mask[left:right] = False

            train_mask[val_idx] = False
            train_idx = np.nonzero(train_mask)[0]
            yield train_idx, val_idx

    def get_n_splits(self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None, groups: Optional[List[object]] = None) -> int:
        return self.n_splits


stub_labeler_module = types.ModuleType(
    "src.training.steps.pre_training.multi_horizon_profit_labeler"
)
stub_labeler_module.MultiHorizonProfitLabeler = object
stub_labeler_module.MultiHorizonConfig = object
stub_labeler_module.create_multi_horizon_labeler = lambda *args, **kwargs: None
stub_labeler_module.apply_multi_horizon_labeling = lambda *args, **kwargs: None
sys.modules.setdefault(
    "src.training.steps.pre_training.multi_horizon_profit_labeler",
    stub_labeler_module,
)

stub_cv_module = types.ModuleType("src.utils.ml_common.validation.cv")
stub_cv_module.PurgedKFoldTime = StubPurgedKFoldTime
sys.modules.setdefault("src.utils.ml_common.validation.cv", stub_cv_module)

from src.training.steps.pre_training.final_feature_selection_pipeline import (  # noqa: E402
    FeatureSelectionConfig,
    MultiStageFeatureSelector,
)


def test_purged_cv_respects_temporal_embargo():
    config = FeatureSelectionConfig(
        cv_folds=3,
        rf_n_estimators=10,
        rf_max_depth=3,
        label_horizon_minutes=5,
        purge_minutes=10,
        embargo_minutes=12,
    )
    selector = MultiStageFeatureSelector(config=config)

    rng = np.random.default_rng(42)
    n_samples = 90
    base_times = pd.date_range("2021-01-01", periods=n_samples, freq="T")
    shuffle_order = rng.permutation(n_samples)

    X = pd.DataFrame(
        rng.normal(size=(n_samples, 4)),
        columns=[f"feature_{i}" for i in range(4)],
    ).iloc[shuffle_order].reset_index(drop=True)
    y = pd.Series(rng.normal(size=n_samples), index=X.index)
    event_times = pd.Series(base_times[shuffle_order], index=X.index)

    X_cv, y_cv, splitter = selector._prepare_cv_splitter(X, y, event_times)

    assert isinstance(splitter, StubPurgedKFoldTime)

    assert isinstance(X_cv.index, pd.DatetimeIndex)
    assert X_cv.index.is_monotonic_increasing

    splits = list(splitter.split(X_cv, y_cv))
    assert len(splits) == config.cv_folds

    purge_delta = pd.Timedelta(minutes=max(config.label_horizon_minutes, config.purge_minutes or 0))
    embargo_delta = pd.Timedelta(minutes=max(config.label_horizon_minutes, config.embargo_minutes or 0))

    for train_idx, val_idx in splits:
        val_times = X_cv.index[val_idx]
        train_times = X_cv.index[train_idx]

        assert val_times.is_monotonic_increasing

        before_mask = train_times < val_times.min()
        if before_mask.any():
            assert train_times[before_mask].max() <= val_times.min() - purge_delta

        after_mask = train_times > val_times.max()
        if after_mask.any():
            assert train_times[after_mask].min() >= val_times.max() + embargo_delta

        assert not ((train_times >= val_times.min()) & (train_times <= val_times.max())).any()

    scores = selector._cross_validate_feature_importance(X, y, event_times)
    assert set(scores.keys()) == set(X.columns)
