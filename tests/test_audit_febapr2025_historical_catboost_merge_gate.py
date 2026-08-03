from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_febapr2025_historical_catboost_merge_gate import _economic_merge_gate


def test_fast_merge_requires_positive_supported_source_classes_per_side() -> None:
    rows = []
    for side in ("long", "short"):
        for label, value in (("fast_clean_winner", 0.02), ("fast_winner_early_drawdown", 0.01)):
            rows.extend({"side_name": side, "path_shape_archetype": label, "execution_net_ev_12h": value} for _ in range(100))
    report, passed = _economic_merge_gate(pd.DataFrame(rows))
    assert passed is True
    assert report["sides"]["long"]["fast_realization_winner"]["rows"] == 200

    failed = pd.DataFrame(rows)
    failed.loc[failed.index[:100], "execution_net_ev_12h"] = -0.01
    _, passed = _economic_merge_gate(failed)
    assert passed is False
