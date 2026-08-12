import json
import pandas as pd

from extreme_price_movements.query_candidate_definitions import materialize_query_membership
from extreme_price_movements.query_funnel import (
    aggregate_portability, load_frozen_query_shortlist, portability_metrics,
    select_pareto_shortlist, validity_audit,
)


def test_query_funnel_produces_inference_valid_shortlist():
    x=pd.DataFrame({
        "candidate_id":["a","b","c","d"],
        "__ts__":pd.to_datetime(["2024-01-01T00:00Z","2024-01-01T00:00Z","2024-02-01T00:00Z","2024-02-01T00:00Z"]),
        "side_name":["long","long","long","long"],
        "net_bps":[-50.,100.,-25.,125.], "grade":[0,3,0,4], "fold":["a","a","b","b"],
    })
    membership=materialize_query_membership(x)
    audit=validity_audit(x,membership)
    assert (audit.future_membership_violation_count==0).all()
    era=portability_metrics(x,membership,grade_column="grade")
    summary=aggregate_portability(era)
    shortlist=select_pareto_shortlist(summary)
    assert "q0_exact_timestamp_side" in set(shortlist.query_candidate)
    assert shortlist.shortlisted.any()


def test_frozen_shortlist_is_loaded_without_rescoring(tmp_path):
    path = tmp_path / "shortlist.json"
    path.write_text(json.dumps({"shortlist": ["q0_exact_timestamp_side", "q1_cycle_4h_side"]}))
    assert load_frozen_query_shortlist(path) == (
        "q0_exact_timestamp_side", "q1_cycle_4h_side",
    )
