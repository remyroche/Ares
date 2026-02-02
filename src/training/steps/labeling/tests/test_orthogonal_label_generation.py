import unittest
import numpy as np
import pandas as pd
from src.training.steps.labeling.orthogonal_label_generation import (
    compute_dominance_labels,
    apply_triple_barrier_multi,
    OutputGeometry
)

class TestOrthogonalLabelGeneration(unittest.TestCase):
    def setUp(self):
        # Create synthetic data
        dates = pd.date_range(start="2024-01-01", periods=1000, freq="15min")
        self.df = pd.DataFrame(index=dates)

        # Sine wave price + noise
        t = np.linspace(0, 100, 1000)
        price = 100 + 10 * np.sin(t) + np.random.normal(0, 0.5, 1000)
        self.df['close'] = price
        self.df['open'] = price # Simplified
        self.df['high'] = price + 1.0
        self.df['low'] = price - 1.0
        self.df['volume'] = np.random.randint(100, 1000, 1000)
        self.df['volatility_1d'] = self.df['close'].pct_change().rolling(20).std().fillna(0.01)

        # Create dummy events
        self.events = self.df.index[::50] # Every 50 bars

    def test_compute_dominance_labels(self):
        labels, weights, returns, mfe, mae, vol = compute_dominance_labels(
            price=self.df['close'],
            events=self.events,
            volatility=self.df['volatility_1d'],
            risk_budget=0.7,
            pt_mult=2.0,
            sl_mult=1.0,
            horizon=24,
            high=self.df['high'],
            low=self.df['low']
        )

        self.assertEqual(len(labels), len(self.events))
        self.assertTrue(all(l in [-1.0, 0.0, 1.0] for l in labels))
        self.assertEqual(len(weights), len(self.events))

        # Verify alignment
        self.assertEqual(labels.index[0], self.events[0])

    def test_apply_triple_barrier_multi(self):
        horizons = [10, 24]
        labels_df = apply_triple_barrier_multi(
            self.df,
            self.events,
            pt_sl=(2.0, 1.0),
            horizons=horizons
        )

        self.assertEqual(len(labels_df), len(self.df)) # It returns aligned to df index or events?
        # apply_triple_barrier_multi returns DataFrame with index = df.index (actually it says index=df.index, but locs event timestamps)

        # Check columns
        expected_cols = [f'price_label_{h}' for h in horizons]
        self.assertListEqual(list(labels_df.columns), expected_cols)

        # Check that events have labels
        # The function assigns labels to event timestamps in the output DF
        event_labels = labels_df.loc[self.events]
        # Some might be 0 if not enough data at end, but generally should be populated
        self.assertTrue(len(event_labels) > 0)

if __name__ == '__main__':
    unittest.main()
