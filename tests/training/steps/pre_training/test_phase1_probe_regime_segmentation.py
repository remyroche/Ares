from dataclasses import dataclass
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_module(module_name: str, relative_path: str):
    root = Path(__file__).resolve().parents[4]
    module_path = root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_phase1_probe = _load_module(
    'phase1_probe_module',
    'src/training/steps/pre_training/interaction_feature_generator/cross_timeframe_generation/phase1_probe.py',
)
Phase1HTFProbe = _phase1_probe.Phase1HTFProbe

_regime_segmentation = _load_module(
    'regime_segmentation_module',
    'src/training/steps/pre_training/interaction_feature_generator/cross_timeframe_generation/regime_segmentation.py',
)
RegimeSegment = _regime_segmentation.RegimeSegment


@dataclass
class _MinimalConfig:
    coarse_grid_min: int = 15
    coarse_grid_max: int = 298
    lambda_unc: float = 0.10
    lambda_cost: float = 0.05
    lambda_stale: float = 0.05
    base_timeframe_minutes: int = 5


def _build_segments(index: pd.DatetimeIndex) -> dict:
    midpoint = len(index) // 2
    first = RegimeSegment(
        start_idx=0,
        end_idx=midpoint,
        start_time=index[0],
        end_time=index[midpoint - 1],
        regime_type='low_vol',
        volatility_level=0.1,
        mean_return=0.01,
        volatility_proxy=0.1,
        metadata={},
    )
    second = RegimeSegment(
        start_idx=midpoint,
        end_idx=len(index),
        start_time=index[midpoint],
        end_time=index[-1],
        regime_type='high_vol',
        volatility_level=0.5,
        mean_return=-0.02,
        volatility_proxy=0.5,
        metadata={},
    )
    return {'segments': [first, second]}


def test_score_candidate_creates_regime_specific_variants():
    config = _MinimalConfig()
    probe = Phase1HTFProbe(config)

    index = pd.date_range('2021-01-01', periods=240, freq='5min')
    feature_values = np.concatenate([np.linspace(0, 1, 120), np.linspace(0, 1, 120)])
    # Positive correlation in first regime, negative in second
    target_values = np.concatenate([np.linspace(0, 1, 120), -np.linspace(0, 1, 120)])

    htf_feature = pd.Series(feature_values, index=index)
    targets = pd.Series(target_values, index=index)

    regime_segments = _build_segments(index)

    candidates = probe._score_candidate(
        htf_feature,
        base_feature='p/price_ema10_pct',
        lookback=60,
        family='trend_level_vol',
        regime_segments=regime_segments,
        targets=targets,
    )

    assert len(candidates) == 2

    indexed = {
        cand.metadata['regime_segment']['segment_index']: cand for cand in candidates
    }
    assert set(indexed.keys()) == {0, 1}

    assert indexed[0].regime == 'low_vol'
    assert indexed[0].ic_oos > 0

    assert indexed[1].regime == 'high_vol'
    assert indexed[1].ic_oos < 0

    for cand in candidates:
        segment_meta = cand.metadata['regime_segment']
        assert 'segment_length' in segment_meta and segment_meta['segment_length'] > 0
        assert 'performance' not in segment_meta  # ensure regime metadata is isolated
        performance_meta = cand.metadata['performance']
        assert performance_meta['ic_oos'] == cand.ic_oos


def test_score_candidate_falls_back_to_mixed_when_no_segments():
    config = _MinimalConfig()
    probe = Phase1HTFProbe(config)

    index = pd.date_range('2021-01-01', periods=200, freq='5min')
    feature_values = np.linspace(0, 1, len(index))
    target_values = np.linspace(0, 1, len(index))

    htf_feature = pd.Series(feature_values, index=index)
    targets = pd.Series(target_values, index=index)

    candidates = probe._score_candidate(
        htf_feature,
        base_feature='p/price_ema10_pct',
        lookback=45,
        family='trend_level_vol',
        regime_segments={},
        targets=targets,
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.regime == 'mixed'
    assert candidate.metadata['regime_segment']['segment_index'] is None
    assert candidate.metadata['regime_segment']['segment_length'] == len(index)
