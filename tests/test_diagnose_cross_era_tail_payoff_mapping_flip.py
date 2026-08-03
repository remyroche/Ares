import numpy as np
import pandas as pd

from scripts.diagnose_cross_era_tail_payoff_mapping_flip import add_secondary_order, causal_map, select_top


def _rows():
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    records = []
    for day in range(3):
        for side, sign in (("long", 1.0), ("short", -1.0)):
            for n in range(120):
                records.append({"candidate_id": f"{day}-{side}-{n}", "__ts__": start + pd.Timedelta(days=day),
                                "label_resolution_utc": start + pd.Timedelta(days=day, hours=12), "side_name": side,
                                "tail_ev_bps": sign * n, "execution_net_ev_12h": sign * n / 1e4})
    return pd.DataFrame(records)


def test_causal_map_excludes_same_day_and_unresolved_rows():
    frame = _rows()
    scored, audit = causal_map(frame, frame, variant="pooled", min_pooled_rows=10, min_side_rows=10)
    first = audit.loc[audit["day"].eq(pd.Timestamp("2026-01-01T00:00:00Z"))]
    second = audit.loc[audit["day"].eq(pd.Timestamp("2026-01-02T00:00:00Z"))]
    assert (first["reference_rows"] == 0).all()
    assert (second["reference_rows"] == 240).all()
    assert np.isfinite(scored["mapped_bps"]).all()


def test_side_shrink_is_convex_and_exposed():
    frame = _rows()
    scored, _ = causal_map(frame, frame, variant="side_shrunk", shrink_rows=120, min_pooled_rows=10, min_side_rows=10)
    mapped = scored.loc[scored["__ts__"].eq(pd.Timestamp("2026-01-03T00:00:00Z"))]
    assert (mapped["side_shrink_weight"] == 2 / 3).all()
    expected = mapped["pooled_mapped_bps"] + mapped["side_shrink_weight"] * (mapped["side_mapped_bps"] - mapped["pooled_mapped_bps"])
    assert np.allclose(mapped["mapped_bps"], expected)


def test_secondary_breaks_isotonic_ties_without_changing_primary():
    frame = pd.DataFrame({"candidate_id": ["b", "a", "c"], "side_name": ["long", "long", "short"],
                          "tail_ev_bps": [3.0, 1.0, 2.0], "mapped_bps": [1.0, 1.0, 0.0]})
    ordered = add_secondary_order(frame, "raw_percentile")
    chosen = select_top(ordered)
    assert chosen.iloc[0]["candidate_id"] == "b"
    assert ordered["mapped_bps"].equals(frame["mapped_bps"])
