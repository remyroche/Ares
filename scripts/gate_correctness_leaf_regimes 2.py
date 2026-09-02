#!/usr/bin/env python3
"""Apply local causal-admission gates to leaf-regime representations.

Rule identity is intentionally *not* matched across outer folds.  Each outer
fold rebuilds its dictionary from prior-resolved history, which is the intended
adaptive representation.  Portability is assessed later on the meta-model's
chronological OOF economics, not by requiring trees to reuse identical leaves.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'data_perp/artifacts/correctness_leaf_regime_oof_20260803_v3'

def run(source:Path=DEFAULT):
 lineage=pd.read_parquet(source/'representation_lineage.parquet')
 diag=pd.read_parquet(source/'standalone_feature_diagnostics.parquet')
 diag=diag[diag.outcome.eq('residual_bps')][['target','fold','side_name','feature','rank_ic','rows']]
 result=lineage.merge(diag,on=['target','fold','side_name','feature'],how='left',validate='one_to_one').copy()
 # Leaf clusters were already screened pairwise for logical consistency,
 # membership agreement and economic sign during average-linkage formation.
 # These remaining gates only remove unusable sparse activations.
 result['support_ok']=result.active_share.ge(.01)
 result['episode_ok']=result.episodes.ge(5)
 result['accepted_pre_meta']=result.support_ok&result.episode_ok
 result['rejection_reason']='accepted'
 result.loc[~result.support_ok,'rejection_reason']='active_share_below_1pct'
 result.loc[result.support_ok&~result.episode_ok,'rejection_reason']='fewer_than_5_independent_episodes'
 result.to_parquet(source/'representation_gate_audit.parquet',index=False)
 accepted=result[result.accepted_pre_meta].copy()
 accepted.to_parquet(source/'accepted_leaf_regime_candidates.parquet',index=False)
 summary={'candidate_representations':int(len(result)),'accepted_pre_meta':int(len(accepted)),'criteria':{'minimum_active_share':.01,'minimum_independent_episodes':5,'rule_cluster_similarity':'average linkage >=0.70','pairwise_hard_gates':'conflict/disjoint interval/shared mechanism/membership correlation/economic direction'},'transport_assessment':'performed on the downstream chronological OOF meta comparison; no leaf identity recurrence gate'}
 (source/'representation_gate_summary.json').write_text(pd.Series(summary).to_json(indent=2)+'\n')
 print(pd.Series(summary).to_json(indent=2));return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=DEFAULT);a=p.parse_args();run(a.source)
