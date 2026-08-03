import pandas as pd
from scripts.diagnose_alpha_execution_ev_gap import global_topk

def test_global_topk_is_cross_side_and_deterministic() -> None:
 d=pd.DataFrame({'lineage':['x']*4,'month':['2026-01']*4,'candidate_id':['d','b','a','c'],'score':[.9,.9,.8,.7],'side_name':['long','short','long','short']})
 out=global_topk(d,score='score',fraction=.25)
 assert out.sum()==1 and out.loc[1]  # candidate b beats d on ascending deterministic tie break

def test_global_topk_keeps_one_for_small_complete_group() -> None:
 d=pd.DataFrame({'lineage':['x'],'month':['2026-01'],'candidate_id':['a'],'score':[.1]})
 assert global_topk(d,score='score').item()
