
import unittest
import numpy as np
from src.utils.entropy_optimized import lempel_ziv_complexity_numba

class TestEntropyOptimized(unittest.TestCase):
    def test_lz_complexity_basic(self):
        # S = 1 0 1 0 1 0 ...
        # Based on my manual trace and the code logic:
        # i=0 (1) -> c=1. LZ=1.
        # i=1 (10) -> c=2. LZ=2.
        # i=2 (101) -> c=3. LZ=3.
        # i=3 (1010) -> found 10. c=3. LZ=3.
        # i=4 (10101) -> found 101. c=3. LZ=3.

        values = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float64)
        # normalize=False to check raw counts
        complexity = lempel_ziv_complexity_numba(values, normalize=False)

        # We expect [1, 2, 3, 3, 3, 3, 3, 3] or similar?
        # My previous test output [2, 3, 3...] because of off-by-one in the test code logic vs loop.
        # Let's verify with the CURRENT implementation.

        # Current implementation:
        # c=1.
        # i=0, l=1. Target S[0:1]=1. Search S[0:0]. Not found.
        # complexity[0]=1. c=2. i=1.
        # i=1, l=1. Target S[1:2]=0. Search S[0:1]=1. Not found.
        # complexity[1]=2. c=3. i=2.
        # i=2, l=1. Target S[2:3]=1. Search S[0:2]=10. Found at 0.
        # complexity[2]=3. l=2.
        # i=2, l=2. Target S[2:4]=10. Search S[0:3]=101. Found at 0.
        # complexity[3]=3. l=3.
        # i=2, l=3. Target S[2:5]=101. Search S[0:4]=1010. Found at 0.
        # complexity[4]=3. l=4.
        # i=2, l=4. Target S[2:6]=1010. Search S[0:5]=10101. Found at 0.
        # complexity[5]=3. l=5.
        # i=2, l=5. Target S[2:7]=10101. Search S[0:6]=101010. Found at 0.
        # complexity[6]=3. l=6.
        # i=2, l=6. Target S[2:8]=101010. Search S[0:7]=1010101. Found at 0.
        # complexity[7]=3. l=7.

        expected = np.array([1, 2, 3, 3, 3, 3, 3, 3], dtype=np.float64)
        np.testing.assert_array_almost_equal(complexity, expected)

    def test_lz_complexity_constant(self):
        # S = 1 1 1 1 ...
        values = np.ones(10, dtype=np.float64)
        complexity = lempel_ziv_complexity_numba(values, normalize=False)

        # i=0 (1) -> Not found. c=1 -> 2. Comp[0]=1.
        # i=1 (1) -> Found. l=2. Comp[1]=2.
        # i=1 (11) -> Found. l=3. Comp[2]=2.
        # ...
        # Complexity should be 2 for the rest (Phrase 1 is '1', Phrase 2 is '1' found in history?)
        # Wait, S[1:2]=1. Found in S[0:1]=1.
        # S[1:3]=11. Found in S[0:2]=11? No. Only if overlapping allowed?
        # LZ76 allows overlapping in the source (history + current prefix).
        # My implementation: history = S[0 : i+l-1].
        # Target = S[i : i+l].
        # If target starts at p < i, it can overlap into the current phrase.
        # Example S=1111. i=1. l=2. Target=11. History=S[0:2]=11.
        # Search range p in 0..0.
        # Check S[0:2] vs Target. S[0]=1, S[1]=1. Target[0]=1, Target[1]=1.
        # Match!

        # So for constant string, we can match infinitely long strings from the first char?
        # Yes, LZ76 complexity of constant string is very low (constant 2).

        expected = np.array([1] + [2]*9, dtype=np.float64)
        np.testing.assert_array_almost_equal(complexity, expected)

    def test_lz_complexity_normalization(self):
        values = np.array([1, 0, 1, 0], dtype=np.float64)
        complexity = lempel_ziv_complexity_numba(values, normalize=True)

        # Raw: 1, 2, 3, 3
        # Norm: 1/1, 2/2, 3/3, 3/4
        expected = np.array([1.0, 1.0, 1.0, 0.75])
        np.testing.assert_array_almost_equal(complexity, expected)

if __name__ == '__main__':
    unittest.main()
