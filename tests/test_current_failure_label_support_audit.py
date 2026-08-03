from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

STAGING = Path(__file__).resolve().parents[1] / "scripts"
if str(STAGING) not in sys.path:
    sys.path.insert(0, str(STAGING))

from audit_current_failure_label_support import (  # noqa: E402
    role_score_summary,
    selection_cutoff_diagnostics,
)


def test_cutoff_audit_reports_plateau_support() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c", "d", "e"],
            "causal_recent_side_isotonic_ev": [3.0, 2.0, 2.0, 2.0, 1.0],
        }
    )
    result = selection_cutoff_diagnostics(frame, fraction=0.4)
    assert result["selected_rows"] == 2
    assert result["cutoff"] == 2.0
    assert result["rows_strictly_above_cutoff"] == 1
    assert result["rows_equal_cutoff"] == 3
    assert result["selected_rows_equal_cutoff"] == 1


def test_role_summary_keeps_lineages_separate() -> None:
    frame = pd.DataFrame(
        {
            "failure_first_history_role": ["oof", "oof", "forward"],
            "__ts__": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-02-01"], utc=True
            ),
            "causal_recent_side_isotonic_ev": [0.1, 0.2, -0.1],
            "execution_net_ev_12h": [0.2, -0.1, -0.3],
        }
    )
    result = role_score_summary(frame).set_index("role")
    assert result.loc["oof", "rows"] == 2
    assert result.loc["forward", "rows"] == 1
    assert np.isclose(result.loc["oof", "mapped_mean"], 0.15)
    assert result.loc["forward", "positive_net_rate"] == 0.0
