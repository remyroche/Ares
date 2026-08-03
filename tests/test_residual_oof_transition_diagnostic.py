import pandas as pd
from extreme_price_movements.residual_oof_transition_diagnostic import build_residual_transition_diagnostic

def test_strict_residual_event_diagnostic() -> None:
    rows=[]
    for ts in pd.date_range('2025-03-01',periods=74,freq='h',tz='UTC'):
        for side in ('long','short'):
            for n in range(3): rows.append({'candidate_id':f'{ts}-{side}-{n}','side_name':side,'__symbol__':str(n),'__ts__':ts,'base_expected_ev':.1*n,'residual_expected_ev':.2*n,'residual_is_oof':True,'selected_top40':n==0,'__first_touch_capture_net__':.01*n,'execution_net_ev_12h':.001*n})
    windows=pd.DataFrame({'transition_event_id':[f'e{i}' for i in range(11)],'transition_window_start_utc':[pd.Timestamp('2025-03-02T00:00Z')]*11,'transition_window_end_utc':[pd.Timestamp('2025-03-02T23:00Z')]*11,'transition_active_hours':[1]*11})
    # Distinct IDs require non-overlapping source hours; testing one event is enough for the mechanics, and duplicate events are rejected by 11 requirement.
    windows['transition_window_start_utc']=pd.date_range('2025-03-02',periods=11,freq='D',tz='UTC');windows['transition_window_end_utc']=windows.transition_window_start_utc+pd.Timedelta(hours=23)
    active=pd.DataFrame({'source_utc':pd.date_range('2025-03-01',periods=24*14,freq='h',tz='UTC'),'target__event_id':[None]*(24*14),'target__transition_active':[0]*(24*14)})
    for i,start in enumerate(windows.transition_window_start_utc): active.loc[active.source_utc.eq(start),['target__event_id','target__transition_active']]=[f'e{i}',1]
    coverage,metrics,summary=build_residual_transition_diagnostic(pd.DataFrame(rows),windows,active)
    assert len(coverage)==11 and coverage.window_complete.all() and coverage.active_complete.all()
    assert len(metrics)==132 and summary['readiness']['sufficient_for_descriptive_health']
