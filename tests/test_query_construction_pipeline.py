import pandas as pd

from extreme_price_movements.query_candidate_definitions import materialize_query_membership
from extreme_price_movements.query_construction_pipeline import query_common_shock_metrics, query_geometry, query_oracle_metrics, query_pair_metrics


def test_query_proxy_pipeline_produces_rankable_metrics():
    frame=pd.DataFrame({"candidate_id":["a","b","c","d"],"__ts__":pd.to_datetime(["2026-01-01T00:00Z"]*4),"side_name":["long"]*4,"fold":["x"]*4,"grade":[0,1,2,2],"net_bps":[-100.,0.,100.,200.],"atr_bps":[100.]*4})
    membership=materialize_query_membership(frame)
    assert query_geometry(frame,membership,grade_column="grade").rankable_row_fraction.min()==1.0
    assert query_pair_metrics(frame,membership,grade_column="grade").economic_pair_count.min()>0
    assert query_oracle_metrics(frame,membership).oracle_top1_uplift.min()>0
    assert query_common_shock_metrics(frame,membership).query_fixed_effect_r2.notna().all()


def test_pair_screen_caps_large_query_without_quadratic_materialisation():
    n = 300
    frame = pd.DataFrame({
        "candidate_id": [f"id_{i}" for i in range(n)],
        "__ts__": pd.to_datetime(["2026-01-01T00:00Z"] * n),
        "side_name": ["long"] * n,
        "fold": ["x"] * n,
        "grade": [i % 5 for i in range(n)],
        "net_bps": [float(i - 150) for i in range(n)],
        "atr_bps": [100.0] * n,
    })
    metrics = query_pair_metrics(
        frame, materialize_query_membership(frame), grade_column="grade",
        pair_cap_per_query=128,
    )
    # Exact and 15-minute grammars split this fixture into two timestamp
    # groups, so the aggregate remains bounded by the cap per query.
    assert (metrics.sampled_pair_count <= 128 * 2).all()
    assert (metrics.pair_sample_rate < 1.0).all()
