from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_bad_mae_recovery_feature_gap import (  # noqa: E402
    _add_bad_mae_flags,
    _contrast_group,
)


def test_add_bad_mae_flags_splits_negative_recovered_fast_and_late() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-04-01", periods=4, freq="h", tz="UTC"),
            "__symbol__": ["BTC"] * 4,
            "u_policy_net": [-0.01, 0.02, -0.03, 0.01],
            "mae_norm": [1.2, 1.3, 1.4, 0.4],
            "bars_policy": [3.0, 20.0, 24.0, 2.0],
            "barrier": [0.01, 0.01, 0.03, 0.01],
            "is_timeout": [False, False, True, False],
        }
    )

    out = _add_bad_mae_flags(frame)

    assert out["row_bad_mae_1r"].tolist() == [True, True, True, False]
    assert out["row_bad_mae_negative"].tolist() == [True, False, True, False]
    assert out["row_bad_mae_recovered"].tolist() == [False, True, False, False]
    assert out["row_fast_bad_mae"].tolist() == [True, False, False, False]
    assert out["row_late_bad_mae"].tolist() == [False, True, True, False]


def test_contrast_group_orients_feature_auc_toward_negative_or_recovered() -> None:
    group = pd.DataFrame(
        {
            "row_bad_mae_negative": [True, True, False, False],
            "row_bad_mae_recovered": [False, False, True, True],
            "row_fast_bad_mae": [True, True, True, True],
            "feature_high_negative": [4.0, 5.0, 1.0, 2.0],
            "feature_high_recovered": [1.0, 2.0, 4.0, 5.0],
        }
    )

    contrasts = _contrast_group(
        group,
        feature_cols=["feature_high_negative", "feature_high_recovered"],
        negative_mask=group["row_bad_mae_negative"],
        recovered_mask=group["row_bad_mae_recovered"],
        prefix={
            "scope": "test",
            "contrast": "bad_mae_negative_vs_recovered",
            "selection": "s",
            "feature_set": "f",
            "source_bucket": "b",
            "causal_gate": "g",
        },
        min_rows=2,
    )

    by_feature = contrasts.set_index("feature")
    assert by_feature.loc["feature_high_negative", "best_direction"] == "higher_in_negative"
    assert by_feature.loc["feature_high_recovered", "best_direction"] == "higher_in_recovered"
    assert by_feature.loc["feature_high_negative", "best_auc"] == 1.0
    assert by_feature.loc["feature_high_recovered", "best_auc"] == 1.0
