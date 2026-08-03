import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('alignment',ROOT/'scripts/run_leave_one_era_out_transition_morphology_alignment.py')
MOD=importlib.util.module_from_spec(SPEC);assert SPEC and SPEC.loader;SPEC.loader.exec_module(MOD)

def test_hungarian_alignment_recovers_permuted_prototypes():
 reference=np.array([[0.,0.],[2.,2.],[4.,4.]])
 candidate=reference[[2,0,1]]
 mapping,cost,confidence=MOD.hungarian_alignment(reference,candidate)
 assert mapping.tolist()==[1,2,0]
 assert np.allclose(cost,0.) and np.allclose(confidence,1.)

def test_outcomes_remain_grade_separated_and_not_increment_claims():
 assignments=pd.DataFrame({'event_id':['a','b'],'heldout_era':['2024','2024'],'semantic_component_id':['semantic_m00','semantic_m00']})
 outcomes=pd.DataFrame({'event_id':['a','b'],'source_grade':['A','B'],'candidate_rows':[1,1],'execution_net_ev_12h':[.01,-.01],'execution_gross_ev_12h':[.02,0.]})
 result=MOD.outcome_increment(assignments,outcomes)
 assert set(result.source_grade)=={'A','B'}
 assert result.outcome_increment_status.eq('NOT_IDENTIFIABLE_NO_MATCHED_CAUSAL_BASELINE').all()
