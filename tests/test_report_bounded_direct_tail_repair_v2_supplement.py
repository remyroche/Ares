import pandas as pd
from scripts.report_bounded_direct_tail_repair_v2_supplement import tie
def test_tie_bounds_span_random_allocation():
 x=pd.DataFrame({'candidate_id':['a','b','c'],'side_name':['long']*3,'__symbol__':['X']*3,'__ts__':[1,2,3],'s':[2.,1.,1.],'execution_net_ev_12h':[.1,.2,-.2],'opp':[True,True,False]})
 r=tie(x,'s',.67);assert r['worst_net_bps']<=r['expected_net_bps']<=r['best_net_bps']
