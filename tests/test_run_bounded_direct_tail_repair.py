from __future__ import annotations
import pandas as pd
from scripts.run_bounded_direct_tail_repair import order, confirmation_eligible

def test_global_order_uses_candidate_id_ties():
 x=pd.DataFrame({'candidate_id':['z','a'],'side_name':['long','short'],'__symbol__':['BTC','BTC'],'__ts__':[2,1],'s':[1.,1.]})
 assert order(x,'s',.5).candidate_id.tolist()==['a']

def test_confirmation_training_excludes_unresolved_label_paths():
 x=pd.DataFrame({'execution_label_end_utc':['2025-03-31T23:00:00Z','2025-04-01T00:00:00Z']})
 z=confirmation_eligible(x,pd.Timestamp('2025-04-01T00:00:00Z'))
 assert len(z)==1
