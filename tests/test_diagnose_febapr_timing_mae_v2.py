import numpy as np

from scripts.diagnose_febapr_timing_mae_v2 import pava_non_decreasing


def test_pava_pools_adjacent_violations_with_correct_block_weights():
    # The last three points form one equally weighted block: (0.8+0.3+0.4)/3.
    raw = np.array([0.2, 0.8, 0.3, 0.4])
    projected = pava_non_decreasing(raw)
    np.testing.assert_allclose(projected, [0.2, 0.5, 0.5, 0.5])
    assert np.all(np.diff(projected) >= 0.0)


def test_pava_leaves_a_coherent_cdf_unchanged():
    raw = np.array([0.1, 0.3, 0.7, 0.9])
    np.testing.assert_array_equal(pava_non_decreasing(raw), raw)


def test_interval_masses_from_projected_cdf_are_nonnegative_and_sum_to_one():
    projected = pava_non_decreasing(np.array([0.5, 0.2, 0.7, 0.6]))
    masses = np.array([
        projected[0], projected[1] - projected[0], projected[2] - projected[1],
        projected[3] - projected[2], 1.0 - projected[3],
    ])
    assert np.all(masses >= 0.0)
    np.testing.assert_allclose(masses.sum(), 1.0)
