
import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.training.steps.market_analysis.afml_specialist_mixin import AFMLSpecialistMixin
from src.training.steps.market_analysis.specialist_data_standard import SpecialistType

class MockSpecialist(AFMLSpecialistMixin):
    def __init__(self):
        self.step_name = "MockStep"
        self.specialist_type = SpecialistType.VOLUME_FORCE # Default anchor=volume

def test_afml_mixin_volume_anchor_real_bars():
    # 1. Setup Mock Data (1m)
    dates = pd.date_range(start='2024-01-01', periods=2000, freq='1T')
    data = {
        'open': np.linspace(100, 110, 2000) + np.random.normal(0, 0.1, 2000),
        'high': np.linspace(101, 111, 2000) + np.random.normal(0, 0.1, 2000),
        'low': np.linspace(99, 109, 2000) + np.random.normal(0, 0.1, 2000),
        'close': np.linspace(100, 110, 2000) + np.random.normal(0, 0.1, 2000),
        'volume': np.abs(np.random.normal(1000, 500, 2000)) + 100
    }
    df_1m = pd.DataFrame(data, index=dates)

    specialist = MockSpecialist()

    # Mock Data Loader to return our DF
    mock_manager = MagicMock()
    mock_manager.read_data.return_value = df_1m

    # Mock AFML Sampling utils (keep these mocked to isolate bar gen logic)
    def mock_sampling(df, config, filter_type='price'):
        # Return every 5th row of the actual anchor bars
        subset = df.iloc[::5]
        return subset, subset.index

    specialist.apply_afml_sampling = MagicMock(side_effect=mock_sampling)

    # We need generate_tbm_labels to return data aligned with the sampled events
    def mock_tbm(df, t_events, config, pt_sl):
        return pd.DataFrame({
            'bin': 1, 't1': t_events, 'ret': 0.01, 'mfe': 0.02, 'mae': -0.01
        }, index=t_events)

    specialist.generate_tbm_labels = MagicMock(side_effect=mock_tbm)

    specialist.get_concurrent_weights = MagicMock(return_value=pd.Series(1.0, index=df_1m.index)) # Index doesn't strictly matter for this mock return structure in tests usually, but let's be loose

    # Mock Utils
    with patch('src.training.steps.market_analysis.afml_specialist_mixin.get_daily_vol') as mock_vol, \
         patch('src.training.steps.market_analysis.afml_specialist_mixin.compute_master_weight') as mock_weight, \
         patch('src.training.steps.market_analysis.afml_specialist_mixin.get_sample_uniqueness') as mock_uniq, \
         patch('src.training.steps.market_analysis.afml_specialist_mixin.get_wavelet_features') as mock_wave, \
         patch('src.training.steps.market_analysis.afml_specialist_mixin.get_klines_manager', return_value=mock_manager):

        mock_vol.return_value = pd.Series(0.01, index=df_1m.index[::50])
        mock_weight.return_value = np.ones(len(df_1m.index[::50]))
        mock_uniq.return_value = pd.Series(1.0, index=df_1m.index[::50])
        mock_wave.return_value = {'hf_lf_ratio': 0.5, 'energy_lvl_0': 1.0, 'entropy_lvl_0': 0.5}

        # 3. Run Pipeline
        feature_df = pd.DataFrame({'feat1': np.random.rand(len(dates))}, index=dates)
        config = {'specialist_type': SpecialistType.VOLUME_FORCE, 'symbol': 'BTCUSDT'}

        # Use Volume Anchor
        X, y, weights = specialist.prepare_specialist_data(df_1m, feature_df, config)

        print("Final Columns:", X.columns)

        assert 'latency' in X.columns
        assert 'filling_ratio' in X.columns
        assert 'relative_wavelet_entropy' in X.columns
        assert 'close_volume_bar' in X.columns
        assert 'close_range_bar' in X.columns

def test_pit_bars_logic():
    # Test PiT generation specifically
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='1T')
    data = {
        'open': np.ones(1000) * 100,
        'high': np.ones(1000) * 105,
        'low': np.ones(1000) * 95,
        'close': np.linspace(100, 200, 1000), # Big price move
        'volume': np.ones(1000) * 1000 # Constant volume
    }
    df_1m = pd.DataFrame(data, index=dates)

    specialist = MockSpecialist()
    mock_manager = MagicMock()
    mock_manager.read_data.return_value = df_1m

    with patch('src.training.steps.market_analysis.afml_specialist_mixin.get_klines_manager', return_value=mock_manager):
        pit_df = specialist.generate_pit_bars(df_1m, {'symbol': 'BTCUSDT'})
        assert pit_df is not None
        assert 'reason' in pit_df.columns
        print("PiT Bars generated:", len(pit_df))
        print(pit_df.head())

if __name__ == "__main__":
    print("Testing Volume Anchor...")
    test_afml_mixin_volume_anchor_real_bars()
    print("Testing PiT Bars...")
    test_pit_bars_logic()
    print("All Verifications Passed!")
