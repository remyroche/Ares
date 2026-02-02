import unittest
import numpy as np
import pandas as pd
from src.training.steps.labeling.composite_event_generators import CompositeEventGenerator, TradeIntensityEvents
from src.utils.numba_funcs import _numba_shift

class TestCompositeEventGenerators(unittest.TestCase):
    def setUp(self):
        # Create dummy OHLCV data
        N = 1000
        self.dates = pd.date_range(start='2023-01-01', periods=N, freq='15min')
        self.df = pd.DataFrame({
            'open': np.random.randn(N) + 100,
            'high': np.random.randn(N) + 105,
            'low': np.random.randn(N) + 95,
            'close': np.random.randn(N) + 100,
            'volume': np.abs(np.random.randn(N) * 1000)
        }, index=self.dates)

        # Ensure high >= low
        self.df['high'] = np.maximum(self.df['high'], self.df['close'])
        self.df['low'] = np.minimum(self.df['low'], self.df['close'])
        self.df['high'] = np.maximum(self.df['high'], self.df['open'])
        self.df['low'] = np.minimum(self.df['low'], self.df['open'])

    def test_numba_shift(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

        # Shift 1
        shifted = _numba_shift(arr, 1)
        expected = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        np.testing.assert_array_equal(shifted[1:], expected[1:])
        self.assertTrue(np.isnan(shifted[0]))

        # Shift -1
        shifted_neg = _numba_shift(arr, -1)
        expected_neg = np.array([2.0, 3.0, 4.0, 5.0, np.nan], dtype=np.float32)
        np.testing.assert_array_equal(shifted_neg[:-1], expected_neg[:-1])
        self.assertTrue(np.isnan(shifted_neg[-1]))

    def test_composite_generator_signals(self):
        gen = CompositeEventGenerator(verbose=False)
        signals = gen.compute_base_signals(self.df)

        self.assertFalse(signals.empty)
        expected_cols = [
            'return_shock', 'volume_spike', 'trade_intensity',
            'flow_imbalance', 'order_flow_imbalance',
            'volatility_regime', 'volatility_spike',
            'trend_strength', 'trend_direction'
        ]
        for col in expected_cols:
            self.assertIn(col, signals.columns)

        # Check for NaNs
        self.assertFalse(signals.isnull().any().any())

    def test_trade_intensity_events(self):
        # Create a specific scenario where TR depends on prev close
        # Day 1: Close = 100
        # Day 2: Open=110, High=112, Low=108, Close=110. Volume=1000.
        # Gap up. TR should be max(112-108, |112-100|=12, |108-100|=8) = 12.

        dates = pd.date_range(start='2023-01-01', periods=2, freq='D')
        df = pd.DataFrame({
            'open': [100.0, 110.0],
            'high': [102.0, 112.0],
            'low': [98.0, 108.0],
            'close': [100.0, 110.0],
            'volume': [500.0, 1000.0]
        }, index=dates)

        # We can't easily introspect internal vars of TradeIntensityEvents,
        # but we can check if it runs without error.

        gen = TradeIntensityEvents(threshold=0.1, window=2)
        events = gen.generate(df)
        # Should not crash
        self.assertIsInstance(events, pd.DatetimeIndex)

    def test_nan_handling(self):
        # Insert NaNs in volume
        self.df.iloc[10:20, self.df.columns.get_loc('volume')] = np.nan

        gen = CompositeEventGenerator(verbose=False)
        signals = gen.compute_base_signals(self.df)

        self.assertFalse(signals['volume_spike'].isnull().any())
        self.assertFalse(signals['trade_intensity'].isnull().any())

if __name__ == '__main__':
    unittest.main()
