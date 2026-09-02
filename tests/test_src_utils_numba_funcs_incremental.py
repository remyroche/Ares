from __future__ import annotations

import numpy as np

from extreme_price_movements.src_utils_numba_funcs import (
    _numba_rolling_kurt,
    _numba_rolling_skew,
)


def test_rolling_skew_returns_nan_for_undersized_incremental_append() -> None:
    """A one-row state append has no full rolling skew window yet.

    This is an inference safety boundary: the compiled function must return
    the canonical all-NaN warm-up result instead of reading beyond the one
    available source row.
    """

    result = _numba_rolling_skew(np.asarray([1.0], dtype=np.float32), 4)

    assert result.dtype == np.float32
    assert result.shape == (1,)
    assert np.isnan(result).all()


def test_rolling_kurt_returns_nan_for_undersized_incremental_append() -> None:
    """A one-row state append has no full rolling kurtosis window yet."""

    result = _numba_rolling_kurt(np.asarray([1.0], dtype=np.float32), 4)

    assert result.dtype == np.float32
    assert result.shape == (1,)
    assert np.isnan(result).all()
