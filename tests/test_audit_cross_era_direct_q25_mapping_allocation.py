import numpy as np
import pandas as pd

from scripts.audit_cross_era_direct_q25_mapping_allocation import (
    SCORE,
    _set_primary,
    economics,
    historical_ledger,
    plateau_diagnostics,
    prepare,
)
from scripts.diagnose_cross_era_tail_payoff_mapping_flip import causal_map


def _frame():
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    for day in range(3):
        for side in ("long", "short"):
            for number in range(10):
                rows.append({
                    "candidate_id": f"{day}-{side}-{number}", "__ts__": start + pd.Timedelta(days=day),
                    "side_name": side, SCORE: float(number), "mapped_q25_bps": 1.0,
                    "execution_net_ev_12h": float(number - 5) / 1e4,
                    "label_resolution_utc": start + pd.Timedelta(days=day, hours=13),
                })
    return pd.DataFrame(rows)


def test_prepare_current_does_not_need_or_consult_a_label_resolution_column():
    frame = _frame().drop(columns=["label_resolution_utc"])
    prepared = prepare(frame, current=True)
    assert prepared["label_resolution_utc"].equals(prepared["__ts__"])


def test_plateau_order_preserves_primary_but_uses_continuous_q25_secondary():
    frame = prepare(_frame(), current=False)
    scored = _set_primary(frame, np.ones(len(frame)), "raw_percentile")
    assert (scored["mapped_bps"] == 1.0).all()
    selected = economics(scored, arm="test", split="historical_oof")
    aggregate = next(row for row in selected if row["level"] == "aggregate")
    assert aggregate["rows"] == 6
    plateau = plateau_diagnostics(scored, arm="test", split="historical_oof")
    assert plateau["cutoff_plateau_rows"] == len(scored)
    assert plateau["selected_from_cutoff_plateau"] == 6


def test_historical_selection_and_current_mapping_need_no_current_outcomes():
    frame = prepare(_frame(), current=False)
    raw = _set_primary(frame, frame[SCORE], "candidate_id")
    rows = economics(raw, arm="raw", split="historical_oof")
    ledger = historical_ledger(pd.DataFrame(rows))
    assert list(ledger["arm"]) == ["raw"]
    assert ledger.iloc[0]["months"] == "2026-01"
    prelabel_current = prepare(_frame().drop(columns=["label_resolution_utc", "execution_net_ev_12h"]), current=True)
    mapped, _ = causal_map(frame, prelabel_current, variant="pooled", score=SCORE, min_pooled_rows=10)
    assert np.isfinite(mapped["mapped_bps"]).all()
    assert mapped["execution_net_ev_12h"].isna().all()
