from __future__ import annotations

import pandas as pd

from scripts.audit_febapr_canonical_top40_residual_readiness_v3 import month_start_residual_support


def test_monthly_residual_support_excludes_first_month_without_prior_resolved_rows() -> None:
    rows = pd.DataFrame(
        {
            "__decision_ts__": pd.to_datetime(["2025-02-01T01:00:00Z", "2025-02-28T23:00:00Z", "2025-03-01T01:00:00Z"]),
            "native_label_resolution_utc": pd.to_datetime(["2025-02-02T01:00:00Z", "2025-03-01T23:00:00Z", "2025-03-02T01:00:00Z"]),
            "side_name": ["long", "short", "long"],
        }
    )
    result = month_start_residual_support(rows)
    feb = result.loc[result["candidate_month"].eq("2025-02")].iloc[0]
    march = result.loc[result["candidate_month"].eq("2025-03")].iloc[0]
    assert not feb["monthly_residual_oof_supported"]
    assert march["monthly_residual_oof_supported"]
    assert march["prior_resolved_top40_rows"] == 1
