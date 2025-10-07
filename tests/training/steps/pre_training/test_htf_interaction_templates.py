from datetime import datetime

import numpy as np
import pandas as pd

from src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation.interaction_templates import (
    HTFInteractionTemplates,
)
from src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation.htf_materialization import (
    HTFFeatureState,
    MaterializedHTF,
    UpdateStyle,
)


def _make_state(feature_name: str, lookback: int, metadata=None) -> HTFFeatureState:
    return HTFFeatureState(
        feature_name=feature_name,
        lookback=lookback,
        update_style=UpdateStyle.EHU,
        last_update=datetime(2023, 1, 1),
        current_value=0.0,
        state_data={},
        metadata=metadata or {},
    )


def _make_htf(
    feature_name: str,
    family: str,
    base_feature: str,
    index: pd.DatetimeIndex,
    extra_metadata=None,
    state_metadata=None,
) -> MaterializedHTF:
    metadata = {
        'base_feature': base_feature,
        'lookback_minutes': 60,
        'created_at': datetime(2023, 1, 1),
        'data_length': len(index),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    feature_series = pd.Series(np.linspace(0.0, 1.0, len(index)), index=index, name=feature_name)
    state = _make_state(feature_name, 60, metadata=state_metadata)

    return MaterializedHTF(
        feature_name=feature_name,
        family=family,
        lookback=60,
        update_style=UpdateStyle.EHU,
        feature_series=feature_series,
        transform_applied='ewz12',
        state=state,
        metadata=metadata,
    )


def test_htf_aware_interactions_are_generated_for_required_templates():
    index = pd.date_range('2023-01-01', periods=120, freq='min')

    base_features = {
        'p/liquidity_depth': pd.Series(np.linspace(0.1, 0.5, len(index)), index=index),
        'p/momentum_signal': pd.Series(np.linspace(-0.2, 0.3, len(index)), index=index),
        'p/vwap_deviation': pd.Series(np.linspace(0.0, 0.2, len(index)), index=index),
    }

    htf_features = {
        htf.feature_name: htf
        for htf in [
            _make_htf(
                't/p/price_ema10_pct_htf60/ewz12',
                'trend_level_vol',
                'p/price_ema10_pct',
                index,
            ),
            _make_htf(
                't/p/sigma_ew_htf60/ewz12',
                'trend_level_vol',
                'p/sigma_ew',
                index,
            ),
            _make_htf(
                't/p/rsi14_htf60/ewz12',
                'oscillators',
                'p/rsi14',
                index,
            ),
            _make_htf(
                't/p/vwap_session_dist_htf60/ewz12',
                'anchors',
                'p/vwap_session_dist',
                index,
            ),
            _make_htf(
                't/regime_indicator_htf60/ewz12',
                'regime',
                'regime_indicator',
                index,
                extra_metadata={'regime_type': 'low_vol'},
                state_metadata={'regime_type': 'low_vol'},
            ),
        ]
    }

    templates = HTFInteractionTemplates(config={'enable_cross_asset': False})
    interactions = templates.generate_interactions(htf_features, base_features)

    htf_interactions = [i for i in interactions if getattr(i, 'interaction_type', None) == 'htf_aware']
    template_names = {i.metadata.get('template_name') for i in htf_interactions}

    expected_templates = {
        'htf_trend_liquidity_interaction',
        'htf_vol_signal_interaction',
        'htf_momentum_conflict_interaction',
        'htf_regime_base_interaction',
        'htf_anchor_deviation_interaction',
    }

    assert expected_templates.issubset(template_names)
