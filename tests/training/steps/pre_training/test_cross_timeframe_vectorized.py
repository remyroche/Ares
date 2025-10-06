import time

import numpy as np
import pytest

from src.training.steps.pre_training.pid_based_feature_generation.cross_timeframe_feature_generator import (
    CrossTimeframeConfig,
    CrossTimeframeFeatureGenerator,
)


def _manual_rolling_mean(data: np.ndarray, window: int) -> np.ndarray:
    result = np.zeros_like(data, dtype=float)
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result[i] = float(np.mean(data[start : i + 1]))
    return result


def _manual_rolling_stat(data: np.ndarray, window: int, stat_func) -> np.ndarray:
    result = np.zeros_like(data, dtype=float)
    for i in range(window - 1, len(data)):
        start = i - window + 1
        result[i] = float(stat_func(data[start : i + 1]))
    return result


def _manual_rolling_corr(x1: np.ndarray, x2: np.ndarray, window: int) -> np.ndarray:
    result = np.zeros_like(x1, dtype=float)
    buf1 = np.zeros(window, dtype=float)
    buf2 = np.zeros(window, dtype=float)
    for i in range(window - 1, len(x1)):
        start = i - window + 1
        buf1[:] = x1[start : i + 1]
        buf2[:] = x2[start : i + 1]
        corr = np.corrcoef(buf1, buf2)[0, 1]
        result[i] = corr if np.isfinite(corr) else 0.0
    return result


def _manual_trend_alignment(x1: np.ndarray, x2: np.ndarray, window: int) -> np.ndarray:
    idx = np.arange(window, dtype=float)
    idx_mean = idx.mean()
    denom = ((idx - idx_mean) ** 2).sum()
    trend = np.zeros_like(x1, dtype=float)
    for t in range(window - 1, len(x1)):
        s1 = x1[t - window + 1 : t + 1]
        s2 = x2[t - window + 1 : t + 1]
        s1_mean = s1.mean()
        s2_mean = s2.mean()
        slope1 = ((idx - idx_mean) * (s1 - s1_mean)).sum() / denom
        slope2 = ((idx - idx_mean) * (s2 - s2_mean)).sum() / denom
        trend[t] = slope1 * slope2
    return trend


@pytest.fixture
def generator() -> CrossTimeframeFeatureGenerator:
    config = CrossTimeframeConfig()
    return CrossTimeframeFeatureGenerator(config)


def test_vectorized_rolling_mean_matches_manual_and_caches(generator):
    rng = np.random.default_rng(42)
    data = rng.normal(size=256)
    window = 17

    result = generator._rolling_aggregation(data, window, cache_params=('mean', 'feature', window))
    expected = _manual_rolling_mean(data, window)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)

    hits_before = generator._rolling_cache_hits
    _ = generator._rolling_aggregation(data, window, cache_params=('mean', 'feature', window))
    assert generator._rolling_cache_hits > hits_before


def test_vectorized_statistic_matches_manual_and_caches(generator):
    rng = np.random.default_rng(7)
    data = rng.normal(size=512)
    window = 31

    result = generator._compute_rolling_statistic_efficient(
        data,
        window,
        np.std,
        cache_params=('std', 'feature', window),
    )
    expected = _manual_rolling_stat(data, window, np.std)
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-8)

    hits_before = generator._rolling_cache_hits
    _ = generator._compute_rolling_statistic_efficient(
        data,
        window,
        np.std,
        cache_params=('std', 'feature', window),
    )
    assert generator._rolling_cache_hits > hits_before


def test_vectorized_correlation_and_trend_alignment_match_manual(generator):
    rng = np.random.default_rng(99)
    x1 = rng.normal(size=640)
    x2 = rng.normal(size=640)
    corr_window = 41
    trend_window = 29

    corr_result = generator._compute_rolling_correlation_efficient(
        x1,
        x2,
        corr_window,
        cache_params=('corr', 'x1', 'x2', corr_window),
    )
    corr_expected = _manual_rolling_corr(x1, x2, corr_window)
    np.testing.assert_allclose(corr_result, corr_expected, rtol=1e-6, atol=1e-8)

    trend_result = generator._compute_trend_alignment(
        x1,
        x2,
        trend_window,
        cache_params=('trend', 'x1', 'x2', trend_window),
    )
    trend_expected = _manual_trend_alignment(x1, x2, trend_window)
    np.testing.assert_allclose(trend_result, trend_expected, rtol=1e-6, atol=1e-8)


def test_vectorized_correlation_and_trend_are_faster_than_manual(generator):
    rng = np.random.default_rng(123)
    x1 = rng.normal(size=2048)
    x2 = rng.normal(size=2048)
    corr_window = 64
    trend_window = 48

    start = time.perf_counter()
    _manual_rolling_corr(x1, x2, corr_window)
    manual_corr_time = time.perf_counter() - start

    start = time.perf_counter()
    generator._compute_rolling_correlation_efficient(x1, x2, corr_window)
    vector_corr_time = time.perf_counter() - start

    assert vector_corr_time <= manual_corr_time * 0.6

    start = time.perf_counter()
    _manual_trend_alignment(x1, x2, trend_window)
    manual_trend_time = time.perf_counter() - start

    start = time.perf_counter()
    generator._compute_trend_alignment(x1, x2, trend_window)
    vector_trend_time = time.perf_counter() - start

    assert vector_trend_time <= manual_trend_time * 0.6
