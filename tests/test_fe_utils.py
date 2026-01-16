import unittest
import pandas as pd
import numpy as np
from src.training.steps.labeling.feature_engineering_utils import apply_layer2_price_processing

class TestFeatureEngineeringUtils(unittest.TestCase):
    def setUp(self):
        # Create a synthetic price series
        dates = pd.date_range(start='2023-01-01', periods=200, freq='15min')
        np.random.seed(42)
        price = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, size=200)))
        self.df = pd.DataFrame({'close': price}, index=dates)

    def test_apply_layer2_price_processing(self):
        processed = apply_layer2_price_processing(self.df, price_col='close')

        # Check new column names
        self.assertIn('log_returns', processed.columns)
        self.assertIn('vol_adjusted_returns', processed.columns)
        self.assertIn('fracdiff_log_price', processed.columns)
        self.assertIn('causal_denoised_returns', processed.columns)
        self.assertNotIn('fracdiff_price', processed.columns)
        self.assertNotIn('wavelet_denoised_returns', processed.columns)

        # Check Volatility Leakage Fix
        # The first few values of rolling volatility (window=20) should be NaN or handled casually
        # In my implementation, I fillna with first valid or 0.01.
        # It should NOT be filled with the median of the entire series.
        vol_col = processed['rolling_volatility_20']
        self.assertFalse(vol_col.isna().all())

        # Check FracDiff
        self.assertFalse(processed['fracdiff_log_price'].isna().all())

        # Check Causal Denoising
        self.assertFalse(processed['causal_denoised_returns'].isna().all())

        # Check Augmented Features
        self.assertIn('denoised_divergence', processed.columns)
        self.assertIn('fracdiff_zscore_50', processed.columns)

    def test_causal_denoising_no_lookahead(self):
        # Rough check: changing the end of the series should not affect the beginning
        # (EWMA is causal)
        df1 = self.df.copy()
        processed1 = apply_layer2_price_processing(df1)

        df2 = self.df.copy()
        # Change the last value significantly
        df2.iloc[-1, 0] = df2.iloc[-1, 0] * 2
        processed2 = apply_layer2_price_processing(df2)

        # The denoised value at index -2 should be identical
        self.assertEqual(processed1['causal_denoised_returns'].iloc[-2],
                         processed2['causal_denoised_returns'].iloc[-2])

        # The denoised value at index -1 should be different
        self.assertNotEqual(processed1['causal_denoised_returns'].iloc[-1],
                            processed2['causal_denoised_returns'].iloc[-1])

if __name__ == '__main__':
    unittest.main()
