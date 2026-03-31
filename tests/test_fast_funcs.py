import unittest
import numpy as np
import pandas as pd
import extreme_price_movements.fast_funcs as ff
from extreme_price_movements.feature_transforms import CausalFeatureTransformer

class TestFastFuncs(unittest.TestCase):
    def test_rolling_zscore_nan_safe_preserves_late_start_series(self):
        x = np.array(
            [np.nan, np.nan, np.nan, 10.0, 11.0, 12.0, 13.0, 14.0],
            dtype=np.float32,
        )
        z = ff._numba_rolling_zscore_nan_safe_1d(x, window=4)

        self.assertTrue(np.isnan(z[:3]).all())
        self.assertTrue(np.isfinite(z[3:]).all())
        self.assertGreater(pd.Series(z[3:]).nunique(dropna=True), 1)

    def test_rolling_max(self):
        data = np.array([1, 2, 5, 3, 4, 1], dtype=np.float32)
        df = pd.DataFrame({'a': data})
        res = ff.numba_rolling_max(df, 3)
        res_arr = res['a'].to_numpy()
        self.assertEqual(res_arr[0], 1.0)
        self.assertEqual(res_arr[2], 5.0)
        self.assertEqual(res_arr[5], 4.0)

    def test_rolling_quantile(self):
        data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        df = pd.DataFrame({'a': data})
        # Window 3. Median (q=0.5).
        res = ff.numba_rolling_quantile(df, 3, 0.5)
        res_arr = res['a'].to_numpy()
        self.assertEqual(res_arr[0], 1.0)
        self.assertAlmostEqual(res_arr[1], 1.5)
        self.assertEqual(res_arr[2], 2.0)
        self.assertEqual(res_arr[4], 4.0)

    def test_rolling_corr(self):
        s1 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        s2 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        df1 = pd.DataFrame({'a': s1})
        df2 = pd.DataFrame({'a': s2})

        res = ff.numba_rolling_corr(df1, df2, 3)
        res_arr = res['a'].to_numpy()
        # Initial points might be 0 due to 0 variance logic in numba correlation
        self.assertEqual(res_arr[2], 1.0)
        self.assertEqual(res_arr[4], 1.0)

    def test_grouped_rolling_mean(self):
        v = pd.DataFrame({'a': [1, 10, 2, 20, 3, 30]}, dtype=np.float32)
        g = pd.Series([0, 1, 0, 1, 0, 1])

        res = ff.numba_grouped_rolling_mean(v, g, 2)
        res_arr = res['a'].to_numpy()

        self.assertEqual(res_arr[0], 1.0)
        self.assertEqual(res_arr[1], 10.0)
        self.assertEqual(res_arr[2], 1.5)
        self.assertEqual(res_arr[3], 15.0)
        self.assertEqual(res_arr[4], 2.5)

    def test_transformer(self):
        data = np.random.randn(100, 2).astype(np.float32)
        df = pd.DataFrame(data, columns=['a', 'b'])
        tf = CausalFeatureTransformer(winsor_qt=0.05, roll_window=10)
        res = tf.transform(df)
        self.assertEqual(res.shape, df.shape)
        self.assertFalse(res.isna().all().all())

    def test_series_input(self):
        s = pd.Series([1, 2, 3, 4, 5], name='a', dtype=np.float32)
        # Test basic rolling mean wrapper
        res = ff.numba_rolling_mean(s, 3)
        self.assertIsInstance(res, pd.Series)
        res_arr = res.to_numpy()
        # i=0: [1] -> 1
        # i=1: [1,2] -> 1.5
        # i=2: [1,2,3] -> 2
        self.assertEqual(res_arr[0], 1.0)
        self.assertEqual(res_arr[2], 2.0)

        # Test grouped mean with series
        g = pd.Series([0, 0, 0, 0, 0])
        res_g = ff.numba_grouped_rolling_mean(s, g, 3)
        self.assertIsInstance(res_g, pd.Series)
        self.assertEqual(res_g.to_numpy()[2], 2.0)

    def test_frac_diff(self):
        # Test Frac Diff with d=1 (should differencing)
        # Note: Fixed Window Frac Diff implementation
        # w_0 = 1, w_1 = -1, w_2 = 0... for d=1?
        # Let's check coefficients:
        # w_0 = 1
        # w_1 = -1 * (1 - 1 + 1)/1 = -1
        # w_2 = -(-1) * (1 - 2 + 1)/2 = 1 * 0 = 0.
        # So d=1 implies w=[1, -1, 0, 0...].
        # So y[t] = x[t] - x[t-1].

        data = np.array([1, 2, 4, 7, 11], dtype=np.float32)
        df = pd.DataFrame({'a': data})

        # Window needs to be at least 2 for d=1 to have effect
        res = ff.numba_frac_diff(df, d=1.0, window=3)
        res_arr = res['a'].to_numpy()

        # Valid from index = window - 1 = 2
        # i=2: x[2] - x[1] = 4 - 2 = 2.
        self.assertAlmostEqual(res_arr[2], 2.0)
        # i=3: x[3] - x[2] = 7 - 4 = 3.
        self.assertAlmostEqual(res_arr[3], 3.0)

    def test_atr_no_norm(self):
        h = np.array([10, 12, 11], dtype=np.float32)
        l = np.array([8, 9, 10], dtype=np.float32)
        c = np.array([9, 11, 10], dtype=np.float32)
        # TR[0] = 10-8 = 2.
        # TR[1] = max(12-9, |12-9|, |9-9|) = 3.
        # TR[2] = max(11-10, |11-11|, |10-11|) = 1.

        # ATR EWM(n=2). alpha=1/2 = 0.5. adjust=False.
        # ATR[0] = TR[0] = 2.
        # ATR[1] = (1-0.5)*2 + 0.5*3 = 1 + 1.5 = 2.5.
        # ATR[2] = (1-0.5)*2.5 + 0.5*1 = 1.25 + 0.5 = 1.75.

        hf = pd.DataFrame({'a': h})
        lf = pd.DataFrame({'a': l})
        cf = pd.DataFrame({'a': c})

        res = ff.numba_atr_no_norm(hf, lf, cf, n=2)
        res_arr = res['a'].to_numpy()

        self.assertAlmostEqual(res_arr[0], 2.0)
        self.assertAlmostEqual(res_arr[1], 2.5)
        self.assertAlmostEqual(res_arr[2], 1.75)

    def test_adx_wide_matrix_matches_series_path(self):
        rng = np.random.default_rng(42)
        n = 400
        noise = rng.normal(0.0, 0.7, n).astype(np.float32)
        drift = np.linspace(0.0, 12.0, n, dtype=np.float32)
        wave = 3.0 * np.sin(np.linspace(0.0, 18.0, n, dtype=np.float32))
        close = (100.0 + drift + wave + noise).astype(np.float32)
        spread = (1.0 + np.abs(rng.normal(0.0, 0.4, n))).astype(np.float32)
        high = close + spread
        low = close - spread

        cols = [f"s{i}" for i in range(64)]
        h_df = pd.DataFrame({c: high for c in cols}, dtype=np.float32)
        l_df = pd.DataFrame({c: low for c in cols}, dtype=np.float32)
        c_df = pd.DataFrame({c: close for c in cols}, dtype=np.float32)

        adx_wide, _, _ = ff.numba_adx(h_df, l_df, c_df, 14)
        adx_single, _, _ = ff.numba_adx(
            pd.Series(high, dtype=np.float32),
            pd.Series(low, dtype=np.float32),
            pd.Series(close, dtype=np.float32),
            14,
        )

        wide_col = adx_wide["s0"].to_numpy(dtype=np.float32)
        single_col = adx_single.to_numpy(dtype=np.float32)

        self.assertGreater(np.unique(single_col[~np.isnan(single_col)]).size, 10)
        self.assertGreater(np.unique(wide_col[~np.isnan(wide_col)]).size, 10)
        self.assertTrue(np.allclose(wide_col, single_col, equal_nan=True, atol=1e-6))

    def test_adx_wide_matrix_with_trailing_nan_padding_preserves_valid_segment(self):
        rng = np.random.default_rng(7)
        n = 320
        base = np.cumsum(rng.normal(0.0, 0.5, n).astype(np.float32)) + 100.0
        spread = (1.0 + np.abs(rng.normal(0.0, 0.2, n))).astype(np.float32)
        high = base + spread
        low = base - spread
        close = base.astype(np.float32)

        high_padded = high.copy()
        low_padded = low.copy()
        close_padded = close.copy()
        high_padded[-24:] = np.nan
        low_padded[-24:] = np.nan
        close_padded[-24:] = np.nan

        h_df = pd.DataFrame(
            {"dense": high, "padded": high_padded}, dtype=np.float32
        )
        l_df = pd.DataFrame(
            {"dense": low, "padded": low_padded}, dtype=np.float32
        )
        c_df = pd.DataFrame(
            {"dense": close, "padded": close_padded}, dtype=np.float32
        )

        adx_wide, _, _ = ff.numba_adx(h_df, l_df, c_df, 14)

        dense = adx_wide["dense"].to_numpy(dtype=np.float32)
        padded = adx_wide["padded"].to_numpy(dtype=np.float32)

        self.assertGreater(np.unique(dense[~np.isnan(dense)]).size, 10)
        self.assertGreater(np.unique(padded[:-24][~np.isnan(padded[:-24])]).size, 10)
        self.assertTrue(np.isnan(padded[-24:]).all())

if __name__ == '__main__':
    unittest.main()
