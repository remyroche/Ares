import pandas as pd
from scripts.run_strict_forward_transition_challenger_v2 import family_features,platt
def test_family_selection_does_not_need_targets_or_state():
 d=pd.DataFrame({'source_utc':pd.date_range('2025-01-01',periods=30,tz='UTC',freq='h'),'x':[float(i) for i in range(30)],'transition_new__x':[float(i) for i in range(30)],'target__phase':[0]*30,'state_context__current_state':[0]*30})
 assert family_features(d,d,'dynamics')==['transition_new__x']
def test_platt_uses_prior_data_only_and_returns_probability():
 train=pd.DataFrame({'raw':[.1,.2,.8,.9],'y':[0,0,1,1]});test=pd.DataFrame({'raw':[.3,.7]});out=platt(train,test);assert len(out)==2 and ((out>0)&(out<1)).all()
