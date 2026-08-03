from __future__ import annotations

import pandas as pd
import pytest

from scripts.materialize_canonical_janfeb2025_gap_closure import verify_february_compatibility


def _frames():
    keys = pd.DataFrame({"candidate_id": ["a", "b"], "side_name": ["long", "short"], "__symbol__": ["BTC", "ETH"], "__ts__": pd.to_datetime(["2025-02-01", "2025-02-01"], utc=True)})
    base = keys.assign(base_oof_score=[.4, .3])
    top = keys.assign(base_oof_score=[.4, .3])
    warm = keys.assign(__decision_ts__=pd.to_datetime(["2025-02-01 01:00", "2025-02-01 01:00"], utc=True), base_oof_score=[.4, .3], residual_is_oof=False)
    population = keys.assign(execution_decision_utc=pd.to_datetime(["2025-02-01 01:00", "2025-02-01 01:00"], utc=True), execution_label_end_utc=pd.to_datetime(["2025-02-01 13:00", "2025-02-01 13:00"], utc=True), execution_label_available_at_utc=pd.to_datetime(["2025-02-01 13:00", "2025-02-01 13:00"], utc=True), execution_gross_ev_12h=[.02, .01], execution_cost_return=[.01, .01], execution_net_ev_12h=[.01, 0.], execution_exit_reason=["timeout", "trailing"], execution_exit_minute=[719, 12])
    return base, top, warm, population


def test_verified_february_sidecar_is_base_only_and_reconciled() -> None:
    sealed, proof = verify_february_compatibility(**dict(zip(("base", "top40", "warmup", "population"), _frames())))
    assert len(sealed) == 2
    assert proof["residual_score_materialized"] is False
    assert sealed["residual_stage"].str.contains("NOT_MATERIALIZED").all()
    assert (sealed["execution_gross_ev_12h"] - sealed["execution_cost_return"] == sealed["execution_net_ev_12h"]).all()


def test_warmup_must_not_claim_residual_oof() -> None:
    base, top, warm, population = _frames()
    warm["residual_is_oof"] = True
    with pytest.raises(ValueError, match="residual OOF"):
        verify_february_compatibility(base=base, top40=top, warmup=warm, population=population)
