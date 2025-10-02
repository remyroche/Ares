import numpy as np
import pandas as pd

from src.training.steps.market_analysis.shared_utils.feature_filters import (
    apply_quality_thresholds,
    filter_low_variance,
    prune_correlated_features,
    winsorize_frame,
)


def test_winsorize_frame_caps_extremes():
    frame = pd.DataFrame({
        'a': [0.0, 1.0, 100.0, 2.0, 3.0],
        'b': [-100.0, -1.0, 0.0, 1.0, 2.0],
    })

    capped, metadata = winsorize_frame(frame, 0.1, 0.9)

    expected_a_upper = frame['a'].quantile(0.9)
    expected_b_lower = frame['b'].quantile(0.1)

    assert capped['a'].max() <= expected_a_upper + 1e-9
    assert capped['b'].min() >= expected_b_lower - 1e-9
    assert metadata['a']['upper'] == expected_a_upper
    assert metadata['b']['lower'] == expected_b_lower


def test_filter_low_variance_removes_constant_features():
    frame = pd.DataFrame({
        'constant': [1.0] * 10,
        'varying': np.linspace(0, 1, 10),
    })

    result = filter_low_variance(frame, min_variance=1e-4)

    assert result.frame.shape[1] == 1
    assert list(result.frame.columns) == ['varying']
    assert 'constant' in result.dropped_columns
    assert result.column_metadata['constant']['variance'] == 0.0


def test_prune_correlated_features_drops_highly_correlated_column():
    frame = pd.DataFrame({
        'x': np.arange(10, dtype=float),
        'y': np.arange(10, dtype=float),
        'z': np.sin(np.linspace(0, np.pi, 10)),
    })

    result = prune_correlated_features(frame, threshold=0.95)

    assert 'y' in result.dropped_columns
    assert 'x' not in result.dropped_columns
    assert result.frame.shape[1] == 2
    assert set(result.frame.columns) == {'x', 'z'}


def test_apply_quality_thresholds_flags_expected_columns():
    frame = pd.DataFrame({
        'persistent': [1.0, 1.0, 1.0, 1.0, 1.0],
        'noisy': [1.0, -1.0, 1.0, -1.0, 1.0],
        'unstable': [1.0, 2.0, 1.0, 2.0, 1.0],
    })

    filtered, metrics, dropped = apply_quality_thresholds(
        frame,
        min_persistence=0.6,
        max_noise_ratio=1.0,
        min_stability=0.2,
    )

    assert 'persistent' in filtered.columns
    assert 'noisy' not in filtered.columns
    assert 'unstable' not in filtered.columns

    assert metrics['persistent']['persistence'] >= 0.6
    assert metrics['noisy']['persistence'] < 0.6
    assert metrics['unstable']['stability'] < 0.2

    assert 'noisy' in dropped
    assert 'unstable' in dropped
    assert 'persistence' in dropped['noisy']
    assert 'stability' in dropped['unstable']
