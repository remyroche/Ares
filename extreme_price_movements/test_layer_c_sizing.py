import numpy as np
from extreme_price_movements.position_sizer_v2 import (
    fit_sizing_normalizer,
    transform_scores_to_sizing_input,
    apply_sizing_curve,
    LayerCExecutionOptimizer
)

def test_legacy_mode_invariance():
    # Legacy mode relies on passed batch
    threshold = 0.5
    normalizer = fit_sizing_normalizer(np.array([]), threshold, mode="legacy_batch_minmax")

    # Batch 1: [0.6, 1.0] -> norm [0.0, 1.0]
    batch1 = np.array([0.6, 1.0])
    norm1 = transform_scores_to_sizing_input(batch1, normalizer, threshold)
    assert np.allclose(norm1, [0.0, 1.0]), "Legacy batch1 failed"

    # Batch 2: [0.6, 2.0] -> norm [0.0, 1.0]
    batch2 = np.array([0.6, 2.0])
    norm2 = transform_scores_to_sizing_input(batch2, normalizer, threshold)
    assert np.allclose(norm2, [0.0, 1.0]), "Legacy batch2 failed"

    # Score 0.6 produces different normalized input based on context
    assert norm1[0] == norm2[0] == 0.0

def test_absolute_mode_invariance():
    # Absolute mode anchors
    threshold = 0.5
    train_scores = np.array([0.5, 0.75, 1.0, 1.5, 2.0])
    normalizer = fit_sizing_normalizer(train_scores, threshold, mode="train_distribution_absolute")
    # p95 of [0.5, 0.75, 1.0, 1.5, 2.0] is 1.9

    # Test identical score mapping across contexts
    batch1 = np.array([0.6, 1.0])
    batch2 = np.array([0.6, 2.0])

    norm1 = transform_scores_to_sizing_input(batch1, normalizer, threshold)
    norm2 = transform_scores_to_sizing_input(batch2, normalizer, threshold)

    assert norm1[0] == norm2[0], "Absolute mode is not invariant"
    assert norm1[0] > 0, "Absolute mode shouldn't map valid active to 0 like legacy"

def test_degenerate_constant_scores():
    threshold = 0.5
    train_scores = np.array([0.5, 0.5, 0.5]) # Constant
    normalizer = fit_sizing_normalizer(train_scores, threshold, mode="train_distribution_absolute")

    # upper anchor should fallback safely to lower_anchor
    assert normalizer["upper_anchor"] <= normalizer["lower_anchor"]

    test_scores = np.array([0.5, 0.6, 1.0])
    norm = transform_scores_to_sizing_input(test_scores, normalizer, threshold)
    # Should safely fallback to zeros
    assert np.allclose(norm, 0.0), f"Degenerate fallback failed: {norm}"

def test_single_sample_live():
    threshold = 0.5
    train_scores = np.array([0.5, 1.0, 1.5])
    normalizer = fit_sizing_normalizer(train_scores, threshold, mode="train_distribution_absolute")

    # Legacy would fail here
    single_score = np.array([1.0])
    norm = transform_scores_to_sizing_input(single_score, normalizer, threshold)
    size = apply_sizing_curve(norm, 0.05, "linear")

    assert norm[0] > 0.0, "Single sample should not zero out"
    assert size[0] > 0.05, "Single sample should get size boost"

if __name__ == "__main__":
    print("Running Layer C Sizing tests...")
    test_legacy_mode_invariance()
    print("Legacy mode tests passed.")

    test_absolute_mode_invariance()
    print("Absolute mode tests passed.")

    test_degenerate_constant_scores()
    print("Degenerate score handling passed.")

    test_single_sample_live()
    print("Single sample live tests passed.")

    print("All tests passed successfully!")
