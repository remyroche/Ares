#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,os,uuid,pandas as pd
R=Path(__file__).resolve().parents[1];A=R/'data_perp/artifacts';O=A/'nested_morphology_increment_readiness_20260730_v1'
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run():
 if O.exists():raise FileExistsError(O)
 a=pd.read_parquet(A/'leave_one_era_out_transition_morphology_20260730_v1/oof_assignments.parquet');b=pd.read_parquet(A/'transition_event_outcome_binding_20260730_v1/event_outcomes.parquet')
 rows=[{'requirement':'train_only_prototype_matching','pass':False,'evidence':'OOF assignments retain fold_component/posterior only; no train-only prototype descriptor vectors or pairwise matching matrix are sealed','reason':'matching fold-local numeric IDs would be invalid'}, {'requirement':'held_era_predictive_increment','pass':False,'evidence':'outcome binding lacks current regime and transition-probability baseline; no matched B_2026 rows','reason':'no nested held-era incremental model can control the required baseline'}, {'requirement':'outcome_support','pass':False,'evidence':f'{len(b)} event-source slices: '+str(b.source_grade.value_counts().to_dict()),'reason':'sources/economics are separate and B_2025 has only 13 slices'}]
 d=O.parent/f'.{O.name}.{uuid.uuid4().hex}';d.mkdir();pd.DataFrame(rows).to_csv(d/'readiness.csv',index=False);m={'schema':'nested_morphology_increment_readiness_v1','status':'STATISTICALLY_INSUFFICIENT_NO_MATCHING_OR_INCREMENT','promotion_eligible':False,'assignment_rows':len(a),'outcome_rows':len(b),'outputs':{'readiness.csv':h(d/'readiness.csv')},'inputs':{'assignments':h(A/'leave_one_era_out_transition_morphology_20260730_v1/oof_assignments.parquet'),'outcomes':h(A/'transition_event_outcome_binding_20260730_v1/event_outcomes.parquet')}};(d/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');(d/'manifest.sha256').write_text(h(d/'manifest.json')+'  manifest.json\n');os.replace(d,O);return m
if __name__=='__main__':print(json.dumps(run()))
