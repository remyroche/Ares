import pandas as pd

from extreme_price_movements.query_candidate_definitions import (
    QueryDefinition, assign_query_ids, query_definitions_by_name,
    recommended_query_definitions,
)
from extreme_price_movements.query_rank_patterns import GradePattern, construct_first_touch_grades, construct_grades


def test_query_ids_are_side_local_and_cycle_stable():
    x = pd.DataFrame({"__ts__": pd.to_datetime(["2026-01-01T01:00Z", "2026-01-01T02:00Z"]), "side_name": ["long", "short"]})
    ids = assign_query_ids(x, QueryDefinition("cycle", "cycle", cycle_hours=2))
    assert ids.nunique() == 2


def test_exact_timestamp_is_not_silently_coarsened_to_an_hour():
    x = pd.DataFrame({
        "__ts__": pd.to_datetime(["2026-01-01T00:05Z", "2026-01-01T00:55Z"]),
        "side_name": ["long", "long"],
    })
    exact, = query_definitions_by_name(["q0_exact_timestamp_side"])
    hourly, = query_definitions_by_name(["q1_cycle_1h_side"])
    assert assign_query_ids(x, exact).nunique() == 2
    assert assign_query_ids(x, hourly).nunique() == 1


def test_recommended_grammar_contains_the_longer_query_candidates():
    names = {definition.name for definition in recommended_query_definitions()}
    assert {"q1_cycle_6h_side", "q1_cycle_8h_side", "q1_cycle_12h_side"}.issubset(names)


def test_grade_zero_and_one_retain_economic_distinction():
    x = pd.DataFrame({"gross_bps": [-20., 40., 120.], "net_bps": [-120., -60., 20.], "path_timeout": [False, True, False], "favorable_first": [False, False, True], "adverse_first": [True, False, False]})
    result = construct_grades(x, GradePattern("test", "triple_barrier", 1., lower_atr=2., upper_atr=2.))
    assert result.tolist() == [0, 1, 2]


def test_first_touch_grade_uses_adverse_tie_and_absolute_guardrails():
    result=construct_first_touch_grades(gross_bps=[-20.,40.,110.,160.,210.],net_bps=[-120.,-60.,10.,60.,110.],favorable_minutes=[[2,2],[1,1],[1,1],[1,1],[1,1]],adverse_minutes=[[1,1],[-1,-1],[2,2],[2,2],[2,2]],thresholds=[2.,4.],lower_atr=2.,upper_atr=2.)
    assert result.tolist()==[0,1,2,3,4]
