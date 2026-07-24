from __future__ import annotations

import pandas as pd

from scripts.audit_breakout_path_quality_labels import _economic_rows, _redundancy_rows, _stability


def test_breakout_audit_reports_redundancy_and_economic_ordering() -> None:
    labels = pd.DataFrame(
        {
            "breakout_quality_label_valid": [1, 1, 1, 1],
            "breakout_retention_failure": [0, 1, 0, 1],
            "breakout_low_efficiency": [0, 1, 0, 1],
            "breakout_participation_failure": [0, 0, 1, 1],
            "breakout_rapid_reversal": [0, 1, 1, 0],
        }
    )
    values = pd.DataFrame(
        {
            "u_policy_net": [0.02, -0.01, 0.01, -0.02],
            "clean_exec": [1, 0, 1, 0],
            "full_path_bad_mae": [0, 1, 0, 1],
            "breakout_retention_outcome": [1, 0, 1, 0],
            "breakout_reversal_magnitude_outcome": [0.1, 2.0, 1.5, 0.2],
        }
    )
    base = {"fold_start": "2026-01-01", "fold_end": "2026-04-01", "side_name": "short", "archetype_policy_key": "short_breakout_precision", "train_rows": 100, "eval_rows": 4}
    redundancy = _redundancy_rows(labels, base=base)
    economic = _economic_rows(values, labels, base=base)

    identical = next(row for row in redundancy if row["left_label"] == "breakout_retention_failure" and row["right_label"] == "breakout_low_efficiency")
    assert identical["phi"] == 1.0
    assert identical["jaccard"] == 1.0
    retention = next(row for row in economic if row["label"] == "breakout_retention_failure")
    assert retention["positive_rate"] == 0.5
    assert retention["delta_positive_minus_negative_mean_ev"] < 0.0
    assert _stability(pd.DataFrame(economic))["folds"].eq(1).all()
