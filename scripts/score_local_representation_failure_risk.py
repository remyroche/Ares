#!/usr/bin/env python3
"""Create causal representation x side x archetype latent-risk OOS scores.

Each retained taxonomy winner is independently refit on prior daily observable
state. Cluster risk is a smoothed prior negative-EV frequency from the train
period only. This turns descriptive PCA/GMM, PCA/Student-t, and DAE/GMM
discoveries into a testable pre-entry hard-block score without using a global
state ID or same-day outcomes.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPRegressor

from extreme_price_movements.unsupervised_regime_learning.failure_detector import (
    ProspectiveFailureDetectorConfig, add_causal_state_dynamics,
    attach_failure_mode_targets, nonlinear_feature_screen, is_batch_layout_dependent_ae_gmm_feature,
)
from extreme_price_movements.unsupervised_regime_learning.failure_taxonomy_models import DiagonalStudentTMixture

ROOT = Path('data_perp/reports/failure_episode_taxonomy_20260719_v17_three_year_taxonomy')
OUT = Path('data_perp/reports/local_representation_failure_risk_20260719_v1/oos_scores.parquet')


def _scale(train: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med=np.nanmedian(train,axis=0); scale=np.maximum(np.nanquantile(train,.75,axis=0)-np.nanquantile(train,.25,axis=0),1e-4)
    return (np.clip(np.nan_to_num((train-med)/scale),-8,8),np.clip(np.nan_to_num((score-med)/scale),-8,8))


def _dae(train: np.ndarray, score: np.ndarray, dim: int, seed: int) -> tuple[np.ndarray,np.ndarray]:
    # Strongly regularized denoising reconstruction. Extract the bottleneck
    # deterministically from the trained encoder layers.
    hidden=max(8, min(24, 3*dim))
    noisy=train + np.random.default_rng(seed).normal(0,.08,train.shape)
    model=MLPRegressor(hidden_layer_sizes=(hidden,dim,hidden),alpha=.05,learning_rate_init=5e-4,max_iter=220,early_stopping=True,validation_fraction=.15,random_state=seed,batch_size=min(128,len(train)))
    model.fit(noisy,train)
    def enc(x):
        h=np.maximum(0,x@model.coefs_[0]+model.intercepts_[0])
        return np.maximum(0,h@model.coefs_[1]+model.intercepts_[1])
    return enc(train),enc(score)


def run(args: argparse.Namespace) -> None:
    root=Path(args.taxonomy)
    state=pd.read_parquet(root/'daily_observable_state.parquet')
    state=add_causal_state_dynamics(state,lookback_days=35,add_market_geometry=True)
    cal=pd.read_parquet(root/'local_adverse_calendar.parquet')
    ass=pd.read_parquet(root/'local_frozen_failure_mode_assignments.parquet')
    frame=attach_failure_mode_targets(state,cal,ass)
    diag=pd.read_csv(root/'local_failure_mixture_diagnostics.csv')
    winners=diag.loc[diag['is_winner'].fillna(False).astype(bool) & diag['method'].isin(['pca_gmm','pca_student_t','small_dae_gmm'])].copy()
    keys={'day','side_name','archetype_policy_key'}
    features=[c for c in state if c not in keys and not is_batch_layout_dependent_ae_gmm_feature(c)]
    outputs=[]
    for winner in winners.itertuples(index=False):
        local=frame.loc[(frame.side_name==winner.side_name)&(frame.archetype_policy_key==winner.archetype_policy_key)].sort_values('day').reset_index(drop=True)
        target='target__negative_ev_day'
        days=pd.DatetimeIndex(local.day.drop_duplicates().sort_values())
        for start in range(120,len(days),int(args.eval_days)):
            eval_days=days[start:start+int(args.eval_days)]
            if not len(eval_days): continue
            boundary=eval_days.min()
            train=local.loc[local.day<boundary].copy(); score=local.loc[(local.day>=boundary)&(local.day<=eval_days.max())].copy()
            if train[target].sum()<5 or score.empty: continue
            selected=nonlinear_feature_screen(train,features,target,maximum=int(args.max_features),bins=8).feature.tolist()
            if len(selected)<2: continue
            xt,xs=_scale(train[selected].apply(pd.to_numeric,errors='coerce').to_numpy(float),score[selected].apply(pd.to_numeric,errors='coerce').to_numpy(float))
            dim=min(int(winner.latent_dim),xt.shape[1],max(1,len(xt)-1))
            if winner.method=='small_dae_gmm': zt,zs=_dae(xt,xs,dim,20260719+start)
            else:
                pca=PCA(n_components=dim,random_state=20260719+start); zt=pca.fit_transform(xt); zs=pca.transform(xs)
            if winner.method=='pca_student_t': mix=DiagonalStudentTMixture(int(winner.clusters),random_state=20260719+start).fit(zt)
            else: mix=GaussianMixture(n_components=int(winner.clusters),covariance_type='diag',reg_covar=.001,n_init=2,random_state=20260719+start).fit(zt)
            labels=mix.predict(zt); prob=mix.predict_proba(zs); prior=float(train[target].mean()); smooth=8.0
            risk=np.array([(train[target].to_numpy()[labels==k].sum()+smooth*prior)/((labels==k).sum()+smooth) for k in range(int(winner.clusters))])
            out=score[['day','side_name','archetype_policy_key']].copy(); out['risk']=prob@risk; out['representation']=winner.method; out['latent_dim']=int(winner.latent_dim); out['clusters']=int(winner.clusters); out['train_end']=boundary; out['selected_features']='|'.join(selected); outputs.append(out)
    result=pd.concat(outputs,ignore_index=True).sort_values(['representation','side_name','archetype_policy_key','day'])
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); result.to_parquet(args.output,index=False)
    print({'rows':len(result),'representations':result.representation.value_counts().to_dict(),'cells':result[['representation','side_name','archetype_policy_key']].drop_duplicates().shape[0]},flush=True)

def parse_args():
 p=argparse.ArgumentParser(); p.add_argument('--taxonomy',type=Path,default=ROOT); p.add_argument('--output',type=Path,default=OUT); p.add_argument('--eval-days',type=int,default=30); p.add_argument('--max-features',type=int,default=32); return p.parse_args()
if __name__=='__main__': run(parse_args())
