from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import run_s52_two_head_selector_smoke as two_head


def test_path_clean_target_requires_positive_clean_ordered_path():
    label = pd.DataFrame(
        {
            "first_pass_good": [1, 1, 1, 0],
        }
    )
    metrics = pd.DataFrame(
        {
            "first_touch_net": [0.01, 0.01, -0.01, 0.02],
            "first_touch_full_path_mae_norm": [0.8, 1.2, 0.5, 0.4],
            "mfe_1r_before_mae_1r": [1, 1, 1, 1],
            "mae_1r_before_mfe_1r": [0, 0, 0, 0],
            "is_timeout": [0, 0, 0, 0],
        }
    )

    clean = two_head._path_clean_target(label, metrics, full_path_mae_cap=1.0)

    assert clean.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_sample_weight_emphasizes_clean_and_dirty_positive_rows():
    label = pd.DataFrame({"first_pass_bad": [0, 1, 0]})
    metrics = pd.DataFrame(
        {
            "first_touch_net": [0.01, 0.01, -0.01],
            "first_touch_full_path_mae_norm": [0.5, 2.0, 0.5],
        }
    )
    clean = pd.Series([1.0, 0.0, 0.0])

    weights = two_head._sample_weight(label, metrics, clean)

    assert weights.iloc[0] > weights.iloc[2]
    assert weights.iloc[1] > weights.iloc[2]


def test_combine_scores_adds_side_normalized_clean_head():
    opp = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    clean = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    metrics = pd.DataFrame({"side": [1, 1, -1, -1]})

    score_opp_only = two_head._combine_scores(opp, clean, metrics, clean_weight=0.0)
    score_with_clean = two_head._combine_scores(opp, clean, metrics, clean_weight=1.0)

    assert score_with_clean.iloc[0] > score_opp_only.iloc[0]
    assert score_with_clean.iloc[1] < score_opp_only.iloc[1]
    assert score_with_clean.iloc[2] > score_opp_only.iloc[2]
    assert score_with_clean.iloc[3] < score_opp_only.iloc[3]
