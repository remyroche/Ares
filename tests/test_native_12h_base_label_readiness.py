import json
import pandas as pd
from extreme_price_movements.native_12h_base_label_readiness import build_readiness_gate
def test_rejects_top40_only_path_source_without_recipe(tmp_path):
 base=tmp_path/'base.parquet';native=tmp_path/'native.parquet';paths=tmp_path/'paths.parquet';manifest=tmp_path/'manifest.json'
 pd.DataFrame({'candidate_id':['a','b']}).to_parquet(base,index=False)
 pd.DataFrame({'candidate_id':['a'],'__decision_ts__':[pd.Timestamp('2025-01-01T00:00Z')],'__barrier_pct__':[.01],'__tp__':[.02],'__sl__':[.01],'__first_touch_target_soft__':[.5],'__first_touch_capture_net__':[.01]}).to_parquet(native,index=False)
 pd.DataFrame({'candidate_id':['a'],'execution_future_path':['[]']}).to_parquet(paths,index=False)
 manifest.write_text(json.dumps({'path':{'fixed_length':720},'timing':{'cadence_minutes':1,'path_minutes':720}}))
 gate=build_readiness_gate(base_oof=base,native_label_example=native,exact_12h_paths=paths,paths_manifest=manifest)
 assert gate['status'].startswith('BLOCKED')
 assert 'MISSING_FULL_BASE_UNIVERSE_EXACT_12H_PATHS' in gate['blockers']
 assert 'MISSING_FROZEN_NATIVE_FIRST_TOUCH_RECIPE_AND_24H_REPLAY_PARITY' in gate['blockers']
