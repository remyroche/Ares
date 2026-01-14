
import pytest
import numpy as np
from src.utils.entropy_optimized import lempel_ziv_complexity_numba, shannon_entropy_numba, rolling_entropy_numba

class TestEntropyOptimized:

    def test_shannon_entropy(self):
        # Uniform distribution
        data = np.array([1, 2, 3, 4, 5])
        # With n_bins=5, each value in own bin -> max entropy log2(5)
        # However, min=1, max=5. width=0.8.
        # 1 -> 0, 2 -> 1.25(1), 3 -> 2.5(2), 4 -> 3.75(3), 5 -> 5(4?)
        # 5 is edge case. (5-1)/0.8 = 5.0 -> bin 5. clipped to 4.
        # So bins: 0, 1, 2, 3, 4. Uniform.
        entropy = shannon_entropy_numba(data, n_bins=5)
        expected = np.log2(5)
        assert np.isclose(entropy, expected, atol=0.1)

        # Constant
        data_const = np.ones(10)
        entropy_c = shannon_entropy_numba(data_const, n_bins=5)
        assert entropy_c == 0.0

    def test_lz_complexity_basics(self):
        # Constant sequence
        data = np.zeros(100)
        lz = lempel_ziv_complexity_numba(data, normalize=False)
        # Should be 2.0 eventually
        assert lz[-1] == 2.0

        # Random sequence
        np.random.seed(42)
        data_rand = np.random.randn(100)
        lz_rand = lempel_ziv_complexity_numba(data_rand, normalize=False)
        assert lz_rand[-1] > 2.0

        # Normalization
        lz_norm = lempel_ziv_complexity_numba(data_rand, normalize=True)
        assert np.allclose(lz_norm, lz_rand / (np.arange(100) + 1))

    def test_rolling_entropy(self):
        data = np.random.randn(100)
        window = 10
        rolling = rolling_entropy_numba(data, window, n_bins=5)
        assert len(rolling) == 100
        assert np.isnan(rolling[0])
        assert not np.isnan(rolling[window])
