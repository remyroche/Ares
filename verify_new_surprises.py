
import unittest
import pandas as pd
import numpy as np
from src.training.steps.labeling.predictor_geometry_generators import ContinuousPredictorGenerator

class TestNewSurprises(unittest.TestCase):
    def test_specialist_surprises(self):
        # Create dummy OHLCV data
        n = 300
        idx = pd.date_range('2023-01-01', periods=n, freq='15min')
        df = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(n)),
            'volume': np.abs(1000 + np.random.randn(n)*100),
            'high': 105 + np.cumsum(np.random.randn(n)),
            'low': 95 + np.cumsum(np.random.randn(n))
        }, index=idx)

        # Ensure high > low
        df['high'] = df[['high', 'close']].max(axis=1) + 1
        df['low'] = df[['low', 'close']].min(axis=1) - 1

        gen = ContinuousPredictorGenerator(verbose=True)
        predictors = gen._generate_specialist_surprises(df)

        # We expect 5 new predictors
        self.assertEqual(len(predictors), 5)

        names = [p.name for p in predictors]
        print(f"Generated predictors: {names}")

        expected_types = [
            "drift_surprise_10",
            "vol_of_vol_surprise_10_24",
            "trend_persistence_surprise_24",
            "range_surprise_10",
            "volume_mean_surprise_10"
        ]

        for name in expected_types:
            self.assertIn(name, names)

        for p in predictors:
            # Check length and finiteness
            self.assertEqual(len(p.values), n)
            self.assertTrue(np.all(np.isfinite(p.values)), f"{p.name} contains non-finite values")

            # Check if values are within clipped range +/- 5
            self.assertTrue(p.values.abs().max() <= 5.00001, f"{p.name} values exceed clip range")

        print("✅ New specialist surprises verified.")

if __name__ == '__main__':
    unittest.main()
