import pandas as pd

from src.training.steps.pre_training.validation.cv import (
    purged_walk_forward_cv,
    validate_cv_no_leakage,
)


def test_purged_walk_forward_cv_respects_purge_and_embargo():
    data = pd.DataFrame(
        {"value": range(240)},
        index=pd.date_range("2023-01-01", periods=240, freq="h"),
    )

    folds = list(
        purged_walk_forward_cv(
            data,
            n_splits=4,
            train_ratio=0.5,
            validation_ratio=0.3,
            test_ratio=0.2,
            purge_window_hours=6,
            embargo_window_hours=3,
        )
    )

    assert len(folds) >= 2

    validate_cv_no_leakage(
        [fold.to_mapping() for fold in folds],
        purge_window_hours=6,
        embargo_window_hours=3,
    )

    purge_delta = pd.Timedelta(hours=6)
    embargo_delta = pd.Timedelta(hours=3)
    last_evaluation_end = None

    for fold in folds:
        mapping = fold.to_mapping()
        train_index = mapping['train'].index
        val_index = mapping['validation'].index
        test_index = mapping['test'].index

        assert train_index.is_monotonic_increasing
        assert val_index.is_monotonic_increasing
        assert test_index.is_monotonic_increasing

        if not train_index.empty and not val_index.empty:
            assert train_index.max() <= val_index.min() - purge_delta

        evaluation_end = test_index.max() if not test_index.empty else val_index.max()

        if last_evaluation_end is not None and not val_index.empty:
            assert val_index.min() >= last_evaluation_end + embargo_delta

        last_evaluation_end = evaluation_end
