import pandas as pd
from scripts.run_causal_opportunity_hurdle_mapping_ablations import top10,quantile_rank,resolved_before
def test_top10_is_global_and_deterministic():
 d=pd.DataFrame({'candidate_id':['b','a','c'],'score':[1.,1.,0.]});out=top10(d,'score');assert out.sum()==1 and out.iloc[1]
def test_rank_uses_training_reference_only():
 value=pd.Series([10.,20.]);ref=pd.Series([0.,10.]);assert quantile_rank(value,ref).tolist()==[1.,1.]
def test_only_resolved_exact_12h_outcomes_enter_fit_or_hpo():
 d=pd.DataFrame({'__ts__':pd.to_datetime(['2026-05-31 11:59Z','2026-05-31 12:00Z'],utc=True)})
 out=resolved_before(d,pd.Timestamp('2026-06-01',tz='UTC'))
 assert out.index.tolist()==[0]
