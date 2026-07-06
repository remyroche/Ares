import numpy as np
import pandas as pd

from scripts.diagnose_gmm_train_base_feature_gap import (
    _feature_gap_table,
    _selected_indices_for_variant,
)


def test_feature_gap_table_ranks_feature_separating_rejected_clean_from_selected_risky() -> None:
    good_values = ([3.0, 2.8, 2.6, -1.0, -1.2, -1.4] * 40)
    noise_values = ([0.1, -0.2, 0.2, 0.1, -0.1, 0.0] * 40)
    clean_u = ([0.01, 0.02, 0.015, -0.01, -0.02, -0.005] * 40)
    mae = ([0.2, 0.3, 0.4, 1.2, 1.3, 1.5] * 40)
    timeout = ([False, False, False, False, True, False] * 40)
    selected_values = ([False, False, False, True, True, True] * 40)
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-04-01"] * 120 + ["2026-05-01"] * 120),
            "good_feature": good_values,
            "noise_feature": noise_values,
        }
    )
    metrics = pd.DataFrame(
        {
            "u_policy_net": clean_u,
            "mae_norm": mae,
            "is_timeout": timeout,
        }
    )
    selected = pd.Series(selected_values)

    gap = _feature_gap_table(
        frame=frame,
        metrics=metrics,
        features=["good_feature", "noise_feature"],
        selected_mask=selected,
    )

    assert gap.iloc[0]["feature"] == "good_feature"
    assert gap.iloc[0]["auc_rejected_clean_vs_selected_risky"] == 1.0
    assert gap.iloc[0]["clean_sign_stability"] == 1.0


def test_selected_indices_for_bad_mae_cap_variant_uses_hard_caps_and_side_limit() -> None:
    utility = pd.Series([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], dtype=float)
    bad = pd.Series([0.30, 0.40, 0.70, 0.20, 0.45, 0.35], dtype=float)
    timeout = pd.Series([0.02, 0.03, 0.01, 0.20, 0.02, 0.02], dtype=float)
    clean = pd.Series(np.zeros(6), dtype=float)
    side = pd.Series([1, 1, 1, -1, -1, -1])

    _score, selected, diag = _selected_indices_for_variant(
        selector_variant="strong_bad_mae_timeout_penalty_pred_bad_mae_cap_50_side_cap_70",
        utility_score=utility,
        bad_mae_pred=bad,
        timeout_pred=timeout,
        clean_path_pred=clean,
        side=side,
        top_frac=1.0,
        max_side_share=0.70,
        pred_timeout_cap=0.12,
    )

    assert set(selected.tolist()) <= {0, 1, 4, 5}
    assert 2 not in selected
    assert 3 not in selected
    assert diag["pred_bad_mae_cap"] == 0.50
