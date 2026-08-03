import pandas as pd
from scripts.attribute_native12_execution_ev_failures import paired,phase,select

def test_phase_and_global_selection_are_deterministic():
    x=pd.DataFrame({'expost_transition_active':[0,1,0],'transition_window_member':[0,1,1],'score':[.2,.9,.5]})
    assert phase(x).tolist()==['outside','active','window_nonactive']
    assert select(x,'score').tolist()==[False,True,False]

def test_paired_selection_partitions_the_population():
    x=pd.DataFrame({'old_selected':[True,True,False,False],'new_selected':[True,False,True,False]})
    assert paired(x).tolist()==['both','old_only','new_only','neither']
