from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.report_exit_opportunity_regret_decomposition import decompose


def test_decomposition_reconciles_path_capture_cost_and_regret() -> None:
    rows = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
            "__symbol__": ["A", "B"],
            "side_name": ["long", "short"],
            "candidate_id": ["a", "b"],
            "mfe__parent": [0.04, 0.03],
            "gross__parent": [0.01, 0.00],
            "cost__parent": [0.01, 0.01],
            "net__parent": [0.00, -0.01],
            "gross__challenger": [0.02, 0.005],
            "cost__challenger": [0.01, 0.01],
            "net__challenger": [0.01, -0.005],
        }
    )
    result = decompose(rows, ["parent", "challenger"])
    np.testing.assert_allclose(result["family_oracle_net_return"], [0.01, -0.005])
    np.testing.assert_allclose(result["small_family_policy_regret"], [0.01, 0.005])
    np.testing.assert_allclose(result["path_to_family_gross_gap"], [0.02, 0.025])
    np.testing.assert_allclose(
        result["path_opportunity_net_of_parent_cost"], [0.03, 0.02]
    )
