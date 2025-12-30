
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

try:
    from src.training.steps.labeling.lgbm_feature_selection import lgbm_feature_selection_pipeline
except ImportError:
    print("Could not import lgbm_feature_selection_pipeline")
    sys.exit(1)

def test_pipeline_adaptation():
    print("Testing Titan RFE adaptation...")

    # Create small dataset (e.g. 150 samples)
    # Limit should be 1 feature (150 // 100 = 1).
    # Or 200 samples -> 2 features.

    n_samples = 200
    n_features = 20

    X = pd.DataFrame(np.random.randn(n_samples, n_features), columns=[f"feat_{i}" for i in range(n_features)])
    y = pd.Series(np.random.randint(0, 2, n_samples))

    # Make some features predictive
    X['feat_0'] = y * 2 + np.random.normal(0, 0.5, n_samples)
    X['feat_1'] = y * -1.5 + np.random.normal(0, 0.5, n_samples)

    # Run pipeline
    # Default target sets are [80, 70, 60, 50] which are all > 2.
    # We expect it to NOT fail, but currently it might return 50 features or fail/warn.

    try:
        feature_sets, log = lgbm_feature_selection_pipeline(
            X, y,
            target_feature_sets=[10, 5],
            log_dir=None
        )
        print("Feature sets keys:", feature_sets.keys())
        for k, v in feature_sets.items():
            print(f"Set {k}: {len(v)} features")

        # Check if it respected the sample size constraint (which is not yet implemented)
        # 200 samples => max 2 features ideally.
        if any(len(v) > n_samples // 100 for v in feature_sets.values()):
            print("FAIL: Did not adapt to sample size (expected)")
        else:
            print("SUCCESS: Adapted to sample size")

    except Exception as e:
        print(f"Pipeline failed: {e}")

if __name__ == "__main__":
    test_pipeline_adaptation()
