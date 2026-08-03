#!/usr/bin/env python3
"""Seal actionable availability limits for category-stability evidence; no gate."""
from pathlib import Path
import hashlib,json,os,tempfile,shutil
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'data_perp/artifacts';SRC=ART/'heldout_regime_category_economics_stability_20260730_v1';OUT=ART/'heldout_regime_category_economics_stability_20260730_v2_availability'
def h(p):
 d=hashlib.sha256();d.update(p.read_bytes());return d.hexdigest()
def main():
 if OUT.exists():raise FileExistsError(OUT)
 q=pd.read_csv(SRC/'category_stability_qualification.csv');c=pd.read_csv(SRC/'context_coverage.csv');m=json.load(open(SRC/'manifest.json'))
 exact=q[q.economics_cohort.eq('exact_usd_linear_policy')]
 req=pd.DataFrame([{'requirement':'independent_eras','current_exact_max':int(exact.observed_eras.max()),'required':3,'missing':max(0,3-int(exact.observed_eras.max()))},{'requirement':'both_side_category_support','current_exact_categories_with_2_sides':int((exact.observed_sides>=2).sum()),'required':'each candidate category across >=3 eras','missing':'category-specific compatible rows'},{'requirement':'causal_context_rows','current_context_attributed_rows':m['counts']['context_attributed_rows'],'required':'same-lineage 2022-25 OOF plus 2026 forward','missing':'historical exact-context coverage outside current 50,220 rows'},{'requirement':'score_economics_lineage','current':'incompatible cohorts kept separate','required':'one candidate identity + score + exact economics contract','missing':'compatible cross-era ledger'}])
 st=Path(tempfile.mkdtemp(dir=OUT.parent,prefix='.'+OUT.name+'.'))
 try:
  q.to_csv(st/'source_qualification.csv',index=False);c.to_csv(st/'source_context_coverage.csv',index=False);req.to_csv(st/'availability_requirements.csv',index=False)
  rep={'schema':'heldout_regime_category_economics_stability_availability_v2','status':'SEALED_NO_GATE_INSUFFICIENT_COMPATIBLE_SUPPORT','promotion_eligible':False,'model_sample_cadence':'1h','assessment_sample_cadence':'1h','regime_transition_separate':True,'no_ex_post_phase_gate':True,'findings':['exact 2025-26 cohort has at most two independent eras; minimum is three','only 50,220 context-attributed rows exist under fixed-global-top10','incompatible economics/score lineages must not be pooled','existing stable poor categories are research-only and cannot become a gate'],'required_next':'materialize a common candidate-identity hourly OOF score/economics/context ledger across three pre-2026 eras plus untouched 2026, with both-side category support and decision-time soft regime/transition fields only'}
  (st/'report.json').write_text(json.dumps(rep,indent=2)+'\n');files=list(st.iterdir());man={**rep,'inputs':{str(SRC/'manifest.json'):h(SRC/'manifest.json')},'outputs_sha256':{p.name:h(p) for p in files}};mp=st/'manifest.json';mp.write_text(json.dumps(man,indent=2,sort_keys=True)+'\n');(st/'manifest.sha256').write_text(h(mp)+'  manifest.json\n');os.replace(st,OUT)
 except Exception:shutil.rmtree(st,ignore_errors=True);raise
if __name__=='__main__':main()
