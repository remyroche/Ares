from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
top=pd.read_parquet(ROOT/'data_perp/artifacts/febapr2025_canonical_residual_top40_20260727_v1/population.parquet')
path=pd.read_parquet(ROOT/'data_perp/artifacts/febapr2025_execution_ev_deployed_policy_inputs_20260727_v1/path_targets.parquet')
lab=pd.read_parquet(ROOT/'data_perp/artifacts/febapr2025_execution_ev_current_spread_12h_labels_20260727_v1/labels.parquet',columns=['candidate_id','execution_entry_half_spread_bps','execution_exit_half_spread_bps','execution_cost_return'])
out=top[['candidate_id','side_name','__ts__']].merge(path[['candidate_id','__barrier_pct__','__path_auxiliary_atr_fraction__']],on='candidate_id',how='left',validate='one_to_one').merge(lab,on='candidate_id',how='left',validate='one_to_one')
if out.isna().any().any() or len(out)!=len(top):raise RuntimeError('frozen input join failed')
out['__symbol__']=out.candidate_id.str.split('|',n=1).str[0];out['__decision_ts__']=pd.to_datetime(out.__ts__,utc=True)+pd.Timedelta(hours=1);out['atr_fraction']=out['__path_auxiliary_atr_fraction__'];out['fee']=0.;out['entry_spread']=out.execution_entry_half_spread_bps;out['exit_spread']=out.execution_exit_half_spread_bps
dest=ROOT/'data_perp/artifacts/febapr2025_top40_exact1m_path_inputs_20260727_v1';dest.mkdir();out.to_parquet(dest/'candidates.parquet',index=False)
