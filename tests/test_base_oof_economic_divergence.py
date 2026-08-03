import pandas as pd
from extreme_price_movements.base_oof_economic_divergence import build_divergence_diagnostic
def test_identical_row_divergence_contract_and_horizons():
 rows=[]
 for ts in pd.date_range('2025-02-01',periods=4,freq='h',tz='UTC'):
  for side in ('long','short'):
   for n in range(2):rows.append({'candidate_id':f'{ts}-{side}-{n}','side_name':side,'__symbol__':str(n),'__ts__':ts,'base_oof_score':n,'__first_touch_target_soft__':n,'__first_touch_capture_net__':.01*n,'__decision_ts__':ts+pd.Timedelta(hours=1),'base_label_resolution_utc':ts+pd.Timedelta(hours=25),'execution_gross_ev_12h':.02*n,'execution_cost_return':.01,'execution_net_ev_12h':.02*n-.01,'execution_exit_reason':'timeout','execution_exit_hour':12,'execution_mfe_return_12h':.03,'execution_mae_return_12h':.01,'execution_expected_spread_bps':50,'execution_label_end_utc':ts+pd.Timedelta(hours=13),'transition_window_member':n==0,'expost_transition_active':False})
 x=pd.DataFrame(rows);base=x[['candidate_id','side_name','__symbol__','__ts__','base_oof_score','__first_touch_target_soft__','__first_touch_capture_net__','__decision_ts__','base_label_resolution_utc']];execution=x[['candidate_id','side_name','__symbol__','__ts__','execution_gross_ev_12h','execution_cost_return','execution_net_ev_12h','execution_exit_reason','execution_exit_hour','execution_mfe_return_12h','execution_mae_return_12h','execution_expected_spread_bps','execution_label_end_utc']];population=x[['candidate_id','side_name','__ts__','transition_window_member','expost_transition_active']]
 tables,summary=build_divergence_diagnostic(base,execution,population)
 assert len(tables['month_side_metrics'])==2 and tables['horizon_audit'].loc[0,'native_label_horizon_hours']==24
 assert summary['row_join_exact'] and not tables['score_decile_monotonicity'].empty
