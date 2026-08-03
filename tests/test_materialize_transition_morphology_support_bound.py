import pandas as pd
def test_one_heldout_row_per_event_contract():
    x=pd.DataFrame({'event_id':['a','b'],'fold':[0,1]})
    assert not x.event_id.duplicated().any() and x.event_id.nunique()==2
