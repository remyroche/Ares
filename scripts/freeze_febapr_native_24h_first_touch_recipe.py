#!/usr/bin/env python3
"""Freeze located native first-touch recipe evidence and replay-parity contract."""
from __future__ import annotations
import hashlib,json,inspect
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import run_label_first_touch_capture_proxy as recipe
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 out=ROOT/'data_perp/artifacts/febapr2025_native_first_touch_recipe_20260729_v1';out.mkdir(exist_ok=False)
 summary=ROOT/'data_perp/artifacts/20260720_s59_h5_signalclose_causal_trailing_cost100bps_labels_v2/labels/side_archetype_trailing_materialization_summary.json'
 source=ROOT/'scripts/run_label_first_touch_capture_proxy.py'
 payload={'schema':'native_first_touch_recipe_freeze_v1','status':'LOCATED_REPLAY_PARITY_PENDING','native_only':True,'evidence':{'archived_summary':{'path':str(summary),'sha256':sha(summary)},'implementation':{'path':str(source),'sha256':sha(source),'first_touch_capture_outcome_source_sha256':hashlib.sha256(inspect.getsource(recipe._first_touch_capture_outcome).encode()).hexdigest(),'same_bar_first_touch_source_sha256':hashlib.sha256(inspect.getsource(recipe._same_bar_first_touch).encode()).hexdigest()}},'located_contract':{'outcome_mode':'trailing_profit','historical_path':'96 x 15m = 24h','signal_to_decision':'1h','round_trip_cost':0.01,'path_order':'ordered OHLC with explicit same-bar tie-break','native_target':'__first_touch_target_soft__ / __first_touch_capture_net__','geometry':'row barrier and archived effective TP/SL/trail fields'},'replay_parity_contract':{'required_before_12h_challenger':'Run frozen implementation against canonical 24h paths on fixed candidate-id audit sample and require equality/declared tolerance for target_soft, capture_net, hit/stop/timeout and first-touch bar. Record source store parts and candidate hash.','then_12h':'truncate the identical ordered path at 720x1m, retain recipe/constants, set resolution to decision+12h, and never ingest execution EV/policy exit fields.'},'limitation':'Located current implementation and archival summary are frozen evidence; 24h exact paths are still required to prove historical output parity.'}
 (out/'recipe_freeze.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
