import unittest
import numpy as np
import pandas as pd
import extreme_price_movements.fast_funcs as ff
from extreme_price_movements.feature_transforms import CausalFeatureTransformer

class TestFastFuncs(unittest.TestCase):
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
        # Use datetime index to fix test failure
        idx = pd.date_range("2021-01-01", periods=100, freq="1h")
        df = pd.DataFrame(data, columns=['a', 'b'], index=idx)
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
        data = np.array([1, 2, 4, 7, 11], dtype=np.float32)
        df = pd.DataFrame({'a': data})

        res = ff.numba_frac_diff(df, d=1.0, window=3)
        res_arr = res['a'].to_numpy()

        self.assertAlmostEqual(res_arr[2], 2.0)
        self.assertAlmostEqual(res_arr[3], 3.0)

    def test_atr_no_norm(self):
        h = np.array([10, 12, 11], dtype=np.float32)
        l = np.array([8, 9, 10], dtype=np.float32)
        c = np.array([9, 11, 10], dtype=np.float32)

        hf = pd.DataFrame({'a': h})
        lf = pd.DataFrame({'a': l})
        cf = pd.DataFrame({'a': c})

        res = ff.numba_atr_no_norm(hf, lf, cf, n=2)
        res_arr = res['a'].to_numpy()

        self.assertAlmostEqual(res_arr[0], 2.0)
        self.assertAlmostEqual(res_arr[1], 2.5)
        self.assertAlmostEqual(res_arr[2], 1.75)

    def test_rolling_entropy_proxy(self):
        # Generate random data
        np.random.seed(42)
        df = pd.DataFrame(np.random.randn(100, 5), columns=['a', 'b', 'c', 'd', 'e'])
        window = 20

        # Run function
        res = ff.numba_rolling_entropy_proxy(df, window, min_periods=window)

        # Check output shape
        self.assertEqual(res.shape, df.shape)

        # Check values are in valid range [0.0, 1.0] (after warmup)
        valid_res = res.iloc[window:].dropna()
        self.assertTrue((valid_res >= 0.0).all().all())
        self.assertTrue((valid_res <= 1.0).all().all())

        # Basic check against manual calculation for one window
        # Last window of col 'a'
        w_data = df['a'].iloc[-window:].values.astype(np.float32)

        # Sort
        w_sorted = np.sort(w_data)

        def get_q(data, q):
            idx = q * (len(data) - 1)
            lower = int(np.floor(idx))
            fraction = idx - lower
            if lower >= len(data) - 1: return data[-1]
            return data[lower] + (data[lower+1] - data[lower]) * fraction

        q25 = get_q(w_sorted, 0.25)
        q75 = get_q(w_sorted, 0.75)
        iqr = abs(q75 - q25)

        q01 = get_q(w_sorted, 0.01)
        q99 = get_q(w_sorted, 0.99)
        full_range = abs(q99 - q01)
        if full_range < 1e-12: full_range = 1e-12

        spread_ratio = np.clip(iqr / full_range, 0, 1)

        mean = np.mean(w_data)
        std = np.std(w_data) # ddof=0
        if std < 1e-12: std = 1e-12
        median = np.median(w_data)

        skew_proxy = min(abs(mean - median) / std, 3.0)

        term2 = np.clip(1.0 - skew_proxy / 6.0, 0.5, 1.0)
        entropy = spread_ratio * term2

        numba_val = res['a'].iloc[-1]

        # Allow small float diff
        self.assertAlmostEqual(numba_val, entropy, places=5)

if __name__ == '__main__':
    unittest.main()
