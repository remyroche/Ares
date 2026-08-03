#!/usr/bin/env python3
"""Materialise the predeclared valid-neighbourhood certainty surface for T2."""
from __future__ import annotations
import argparse, json, os, sys, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from extreme_price_movements.t2_atr_funnel import BarrierGeometry, materialize_contract_events_bulk
from scripts.run_t2_atr_sequential_funnel import _read_paths

CONTRACTS=(
 ("H12_TP1.6_SL0.8",BarrierGeometry(1.6,.8),720),
 ("H12_TP1.8_SL0.9",BarrierGeometry(1.8,.9),720),
 ("H12_TP2.0_SL1.0",BarrierGeometry(2.,1.),720),
 ("H12_TP2.2_SL1.1",BarrierGeometry(2.2,1.1),720),
 ("H12_TP2.4_SL1.2",BarrierGeometry(2.4,1.2),720),
 ("H8_TP1.8_SL0.9",BarrierGeometry(1.8,.9),480),
 ("H8_TP2.0_SL1.0",BarrierGeometry(2.,1.),480),
 ("H8_TP2.2_SL1.1",BarrierGeometry(2.2,1.1),480),
)
def main():
 p=argparse.ArgumentParser();p.add_argument('--ledger',type=Path,required=True);p.add_argument('--paths',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists(): raise FileExistsError(a.output)
 ledger=pd.read_parquet(a.ledger,columns=['candidate_id'])
 paths=_read_paths(set(ledger.candidate_id.astype(str)),list(a.paths))
 stage=Path(tempfile.mkdtemp(prefix='.'+a.output.name+'.',dir=a.output.parent))
 try:
  events=materialize_contract_events_bulk(paths,CONTRACTS)
  canonical=events['H12_TP2.0_SL1.0'].set_index('candidate_id')
  state=np.stack([e.set_index('candidate_id').loc[canonical.index,['upper_first','lower_first','timeout']].to_numpy() for e in events.values()])
  canonical_state=state[2].argmax(1)
  agree=(state.argmax(2)==canonical_state).mean(0)
  upper_margin=np.maximum(2.-canonical.terminal_atr.to_numpy(),0); lower_margin=np.maximum(canonical.terminal_atr.to_numpy()+1.,0)
  distance=np.minimum(upper_margin,lower_margin)
  certainty=np.clip(.55*agree+.25*np.tanh(distance)+.20*(1-canonical.same_minute_conflict.to_numpy(float)),0,1)
  out=pd.DataFrame({'candidate_id':canonical.index,'event_agreement_rate':agree,'target_sign_agreement_rate':agree,'top_bottom_state_agreement':agree,'target_value_dispersion':state.std(0).max(1),'distance_nearest_barrier_atr':distance,'same_bar_conflict_flag':canonical.same_minute_conflict.to_numpy(float),'path_completeness':canonical.path_completeness.to_numpy(float),'sensitivity_to_entry_delay':np.nan,'sensitivity_to_atr_definition':np.nan,'label_certainty':certainty})
  out.to_parquet(stage/'label_certainty_diagnostics.parquet',index=False,compression='zstd')
  for name,e in events.items(): e.to_parquet(stage/(name+'.parquet'),index=False,compression='zstd')
  manifest={'canonical_contract':'H12_TP2.0_SL1.0','valid_contracts':[{'name':n,'tp_atr':g.tp_atr,'sl_atr':g.sl_atr,'horizon_minutes':h} for n,g,h in CONTRACTS],'unavailable_contracts':['H16: frozen paths stop at H12','delayed entry: requires post-delay H12 path','short/long ATR variants: no exact pre-entry ATR definitions materialised'],'certainty_formula':'0.55 event agreement + 0.25 tanh(distance to nearest barrier) + 0.20 no same-bar conflict','future_label_training_only':True}
  (stage/'label_stability_contracts.json').write_text(json.dumps(manifest,indent=2)+'\n');os.replace(stage,a.output)
 except Exception:
  import shutil;shutil.rmtree(stage,ignore_errors=True);raise
if __name__=='__main__':main()
