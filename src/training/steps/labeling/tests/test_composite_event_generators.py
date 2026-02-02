import unittest
import numpy as np
import pandas as pd
from src.training.steps.labeling.composite_event_generators import (
    compute_microstructure_signals,
    TradeIntensityEvents,
    OrderFlowImbalanceEvents,
    BarPressureEvents,
    CompositeEventGenerator,
    CrossAssetEventGenerator
)

class TestCompositeEventGenerators(unittest.TestCase):
    def setUp(self):
        # Create synthetic data for single asset
        dates = pd.date_range(start="2024-01-01", periods=500, freq="15min")
        self.df = pd.DataFrame(index=dates)

        # Random walk price
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 500)
        price = 100 * np.cumprod(1 + returns)

        self.df['close'] = price
        self.df['open'] = price * (1 + np.random.normal(0, 0.002, 500))
        self.df['high'] = np.maximum(self.df['open'], self.df['close']) * 1.005
        self.df['low'] = np.minimum(self.df['open'], self.df['close']) * 0.995
        self.df['volume'] = np.random.randint(100, 1000, 500).astype(float)

    def test_compute_signals_single_asset(self):
        signals = compute_microstructure_signals(self.df)

        # Check dimensions
        self.assertEqual(len(signals), len(self.df))

        # Check expected columns
        expected_cols = [
            'trade_intensity', 'order_flow_imbalance', 'bar_pressure',
            'volume_spike', 'return_shock', 'volatility_regime',
            'parkinson_vol', 'vwap_displacement', 'rsi'
        ]
        for col in expected_cols:
            self.assertIn(col, signals.columns)

        # Check values are not all NaN (after warmup)
        self.assertFalse(signals['rsi'].iloc[50:].isna().all())
        self.assertFalse(signals['vwap_displacement'].iloc[50:].isna().all())

    def test_multi_asset_isolation(self):
        # Create two assets with a massive price gap
        df1 = self.df.copy()
        df1['asset_id'] = 'asset_1'

        df2 = self.df.copy()
        df2['asset_id'] = 'asset_2'
        # Shift price significantly to detect bleeding
        df2['close'] = df2['close'] * 1000
        df2['open'] = df2['open'] * 1000
        df2['high'] = df2['high'] * 1000
        df2['low'] = df2['low'] * 1000

        # Concatenate
        df_multi = pd.concat([df1, df2])

        # Compute signals
        signals = compute_microstructure_signals(df_multi)

        # Check return shock
        # If bleeding occurs, the first return of asset_2 would be massive (100 -> 100000)
        # Using groupby, the first return of asset_2 should be 0 or NaN

        # Get start index of asset_2
        start_idx_2 = len(df1)

        # Check return_shock at start of asset 2
        # return_shock is abs(ret) / std.
        # If grouped, ret[0] of asset 2 is NaN or 0.

        # We can inspect 'return_shock' directly
        shock_at_boundary = signals['return_shock'].iloc[start_idx_2]

        # It should be 0 or NaN (or small if fillna(0) and std is small)
        # Definitely not huge
        self.assertLess(shock_at_boundary, 10.0)

        # Verify alignment
        self.assertEqual(len(signals), len(df_multi))

        # Verify RSI calculation per asset
        # RSI should reset for asset 2
        rsi_at_boundary = signals['rsi'].iloc[start_idx_2]
        # Should be 50 (default fillna) or NaN, not influenced by previous asset
        self.assertEqual(rsi_at_boundary, 50.0)

    def test_generators_integration(self):
        # Test standalone generator classes
        gen = TradeIntensityEvents(threshold=1.5)
        events = gen.generate(self.df)
        self.assertIsInstance(events, pd.DatetimeIndex)

        gen2 = OrderFlowImbalanceEvents(threshold=1.5)
        events2 = gen2.generate(self.df)
        self.assertIsInstance(events2, pd.DatetimeIndex)

        gen3 = BarPressureEvents(threshold=1.5)
        events3 = gen3.generate(self.df)
        self.assertIsInstance(events3, pd.DatetimeIndex)

    def test_cross_asset_generator(self):
        # Mock cross asset feature
        self.df['ca__lead_lag_w48'] = np.random.normal(0, 1, len(self.df))
        # Add a spike at index 400 (after 300 warmup)
        self.df.iloc[400, self.df.columns.get_loc('ca__lead_lag_w48')] = 10.0

        gen = CrossAssetEventGenerator(quantile_threshold=2.0)
        events = gen.generate(self.df)

        self.assertTrue(len(events) > 0)
        self.assertIn(self.df.index[400], events)

    def test_precomputed_signal_usage(self):
        # Create a mock signal column with a specific pattern
        # 'trade_intensity' usually computed from vol/range.
        # We manually set it to 10.0 at index 50, and 0 everywhere else.
        self.df['trade_intensity'] = 0.0
        self.df.iloc[50, self.df.columns.get_loc('trade_intensity')] = 10.0

        # If the generator uses the pre-computed column, it should find index 50.
        # If it re-computes, it won't find it (as random data doesn't have such intensity at 50).

        gen = TradeIntensityEvents(threshold=5.0)
        events = gen.generate(self.df)

        self.assertIn(self.df.index[50], events)
        self.assertEqual(len(events), 1)

if __name__ == '__main__':
    unittest.main()
