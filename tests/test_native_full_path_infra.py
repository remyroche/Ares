import pandas as pd
from pathlib import Path
from scripts.materialize_febapr_native_12h_full_paths import prepare
def test_prepare_creates_deterministic_native_only_shards(tmp_path):
 base=tmp_path/'base.parquet';labels=tmp_path/'labels';labels.mkdir()
 rows=[]
 for side in ('long','short'):
  rows.append({'candidate_id':side,'side_name':side,'__symbol__':'A/USD:USD','__ts__':pd.Timestamp('2025-02-01T00:00Z'),'__decision_ts__':pd.Timestamp('2025-02-01T01:00Z')})
  pd.DataFrame([{**rows[-1],'__barrier_pct__':.01,'__tp__':.02,'__sl__':.01,'__first_touch_round_trip_cost__':.01,'__first_touch_target_soft__':.5,'__first_touch_capture_net__':.01,'__first_touch_effective_tp_abs__':.02,'__first_touch_effective_sl_abs__':.01,'__first_touch_effective_trail_abs__':.005}]).to_parquet(labels/f'train_global_{side}_5_2025_02.parquet',index=False)
  for m in (3,4): pd.DataFrame(columns=pd.DataFrame([{**rows[-1],'__barrier_pct__':.01,'__tp__':.02,'__sl__':.01,'__first_touch_round_trip_cost__':.01,'__first_touch_target_soft__':.5,'__first_touch_capture_net__':.01,'__first_touch_effective_tp_abs__':.02,'__first_touch_effective_sl_abs__':.01,'__first_touch_effective_trail_abs__':.005}]).columns).to_parquet(labels/f'train_global_{side}_5_2025_{m:02d}.parquet',index=False)
 pd.DataFrame(rows).to_parquet(base,index=False);out=tmp_path/'out';prepare(out,base,labels,1)
 # Both sides for a symbol/month are deliberately kept together so a store
 # read is shared; the one two-row unit therefore remains one shard.
 assert (out/'candidate_inputs.parquet').exists() and len(pd.read_parquet(out/'shard_index.parquet'))==1
