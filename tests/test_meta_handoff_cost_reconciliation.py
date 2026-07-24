from __future__ import annotations

import pandas as pd
import pytest

from scripts.report_s52_trailing_regime_meta_handoff import _enrich_ledger


def test_enrich_ledger_reconciles_round_trip_cost_once() -> None:
    ledger = pd.DataFrame(
        {
            "month": ["2026-05"],
            "side_name": ["long"],
            "first_touch_net": [0.0125],
            "first_touch_mae_norm": [0.2],
            "first_touch_full_path_mae_norm": [0.3],
            "is_timeout": [0.0],
        }
    )
    enriched = _enrich_ledger(
        ledger,
        embedded_round_trip_cost=0.01,
        executable_cost_floor=0.01,
    )
    assert enriched.loc[0, "first_touch_gross"] == pytest.approx(0.0225)
    assert enriched.loc[0, "exec_margin"] == pytest.approx(0.0125)
    assert enriched.loc[0, "ev_after_1pct"] == pytest.approx(0.0125)
