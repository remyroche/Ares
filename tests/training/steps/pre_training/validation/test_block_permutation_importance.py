"""Tests for the block permutation importance helper."""

from __future__ import annotations

from collections import Counter
import numpy as np
import pandas as pd

from src.training.steps.pre_training.validation.schemas import block_permutation_importance


class _ConstantModel:
    def predict(self, X: pd.DataFrame) -> np.ndarray:  # pragma: no cover - simple stub
        return np.zeros(len(X), dtype=float)


class _SignalModel:
    def predict(self, X: pd.DataFrame) -> np.ndarray:  # pragma: no cover - simple stub
        return X["signal"].to_numpy(copy=False)


def test_block_permutation_preserves_order_within_blocks() -> None:
    index = pd.date_range("2021-01-01", periods=9, freq="D")
    original_values = np.arange(len(index), dtype=float)
    features = pd.DataFrame({"signal": original_values}, index=index)
    labels = pd.Series(np.zeros(len(index)), index=index)

    recorded_columns: list[np.ndarray] = []

    def recording_scorer(model: _ConstantModel, X: pd.DataFrame, y: pd.Series) -> float:
        recorded_columns.append(X["signal"].to_numpy(copy=True))
        return 0.0

    block_size = 3

    block_permutation_importance(
        _ConstantModel(),
        features,
        labels,
        block_size=block_size,
        n_repeats=2,
        scoring_func=recording_scorer,
        random_state=np.random.default_rng(42),
    )

    assert recorded_columns, "expected scorer to be invoked"
    baseline = recorded_columns[0]
    assert np.array_equal(baseline, original_values)

    original_blocks = Counter(
        tuple(original_values[start : min(start + block_size, len(original_values))])
        for start in range(0, len(original_values), block_size)
    )

    for permuted_column in recorded_columns[1:]:
        permuted_blocks = Counter(
            tuple(permuted_column[start : min(start + block_size, len(permuted_column))])
            for start in range(0, len(permuted_column), block_size)
        )
        assert permuted_blocks == original_blocks


def test_block_permutation_importance_flags_predictive_feature() -> None:
    rng = np.random.default_rng(7)
    index = pd.date_range("2021-01-01", periods=60, freq="h")
    signal = np.linspace(-1.0, 1.0, len(index))
    noise = rng.normal(scale=0.5, size=len(index))
    features = pd.DataFrame({"signal": signal, "noise": noise}, index=index)
    labels = pd.Series(signal + rng.normal(scale=0.01, size=len(index)), index=index)

    importance = block_permutation_importance(
        _SignalModel(),
        features,
        labels,
        block_size=5,
        n_repeats=5,
        random_state=np.random.default_rng(13),
    )

    assert importance.loc["signal"] > 0
    assert importance.loc["signal"] > importance.loc["noise"]
