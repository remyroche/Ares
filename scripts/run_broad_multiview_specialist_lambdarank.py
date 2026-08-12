#!/usr/bin/env python3
"""Broad data-discovered specialists feeding the native LambdaRank residual meta.

Views are discovered once, side-locally, on a designated pre-test development
population from opportunity-conditioned activation co-clusters.  The exact
ordered memberships and imputers are then frozen and refit per fold.  This
prevents a positional input such as ``mv__view_05`` from silently changing
economic meaning across folds.  Specialists use one frozen target only: exact
H12 net outcome > +50 bps; their outputs feed the same per-row ordinal-bps
LambdaRank residual meta contract.
"""
from __future__ import annotations
import argparse, hashlib, json
import gc
from pathlib import Path
import sys
import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from extreme_price_movements.multiview_specialists import discover_broad_opportunity_views, opportunity_conditioned_synergy, apply_synergy_features, permitted_causal_features
from extreme_price_movements.specialist_head_selection import select_complementary_heads
from extreme_price_movements.packb_static_point_feature_loader import _provenance_backed_raw_allowlist
from scripts.run_market_spine_covariance_meta import LONG_HISTORY_FOLDS, _utc, fit_side_ranker, fit_residual_calibration

LEDGER=ROOT/'data_perp/artifacts/tp6_m6_shared_regime_residual_features_20260809_v3/shared_regime_residual_ledger.parquet'
STORE=ROOT/'data_perp/artifacts/full_universe_t2_t4_panel_20260801_v3/parts/*.parquet'
OUT=ROOT/'data_perp/artifacts/broad_multiview_specialist_lambdarank_20260810_v1'
DELAY=pd.Timedelta(hours=13); SEED=20260810; TAILS=(.01,.05,.10)
MAX_PROXY_ROWS=20_000; MAX_TRAIN_ROWS=150_000
SPECIALIST_COUNT=7
VIEW_DISCOVERY_END=pd.Timestamp('2024-06-01',tz='UTC')

def _schema():
    con=duckdb.connect(); rows=con.execute('DESCRIBE SELECT * FROM read_parquet(?)',[str(STORE)]).fetchall(); con.close()
    excluded={'candidate_id','__ts__','__symbol__','__decision_ts__','side_name'}
    registry_allowlist, _, _, _ = _provenance_backed_raw_allowlist()
    schema_columns=[str(r[0]) for r in rows if str(r[0]) not in excluded]
    selected=permitted_causal_features(schema_columns,causal_allowlist=registry_allowlist)
    if not selected: raise ValueError('no provenance-backed causal feature columns in store schema')
    return sorted(selected)

def _write_feature_contract(out, available):
    payload={'schema':'broad_multiview_raw_causal_contract_v1','source':'current_generator_registry_allowlist_intersected_with_store_schema','feature_columns':list(available)}
    payload['feature_contract_sha256']=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    (out/'feature_contract.json').write_text(json.dumps(payload,indent=2)+'\n')

def _frozen_view_contract(base, available, out, *, specialist_count):
    """Discover once before every evaluated fold and persist stable view IDs.

    Tree models do not need feature scaling, but their missing-value treatment
    must remain fixed when a specialist output is interpreted by a later meta
    model.  The medians below therefore belong to the view contract too.
    """
    path=out/'frozen_specialist_views.json'
    if path.exists():
        payload=json.loads(path.read_text())
        if int(payload['specialist_count']) != int(specialist_count):
            raise ValueError('existing frozen view contract has a different specialist count')
        return payload
    dev=base[base.__ts__.lt(VIEW_DISCOVERY_END)&base.label_available_ts.lt(VIEW_DISCOVERY_END)].copy()
    if dev.empty: raise ValueError('no pre-test rows for frozen view discovery')
    payload={'schema':'frozen_side_specialist_views_v1','discovery_end_utc':VIEW_DISCOVERY_END.isoformat(),'specialist_count':int(specialist_count),'target':'exact_h12_net_bps_gt_50','method':'opportunity_conditioned_activation_synergy_cross_fold_frozen','sides':{}}
    for side in ('long','short'):
        proxy=_sample(dev[dev.side_name.eq(side)].copy(),MAX_PROXY_ROWS)
        proxy=proxy.merge(_store_rows(proxy,available),on='candidate_id',validate='one_to_one')
        eligible=[]
        for field in available:
            value=pd.to_numeric(proxy[field],errors='coerce').to_numpy(float)
            if np.isfinite(value).mean()>=.90 and np.nanmedian(np.abs(value-np.nanmedian(value)))*1.4826>1e-8:
                eligible.append(field)
        if len(eligible)<24*specialist_count:
            raise ValueError(f'{side}: only {len(eligible)} development-stable fields for {specialist_count} views')
        min_per_view=max(24,min(40,len(eligible)//specialist_count))
        max_per_view=max(min_per_view,min(80,len(eligible)//specialist_count))
        proxy['binary_h12_net50']=(proxy.net_bps>50.).astype(np.int8)
        views,field_audit,edges=discover_broad_opportunity_views(proxy,eligible,base_score_column='base_score',label_column='binary_h12_net50',specialist_count=specialist_count,min_features_per_view=min_per_view,max_features_per_view=max_per_view,max_proxy_features=min(480,len(eligible)),min_joint_rows=80)
        named={f'{side}_view_{int(name.rsplit("_",1)[1]):02d}': list(fields) for name,fields in views.items()}
        medians={field:float(pd.to_numeric(proxy[field],errors='coerce').median()) for fields in named.values() for field in fields}
        if not np.isfinite(list(medians.values())).all(): raise ValueError(f'{side}: frozen imputer is non-finite')
        payload['sides'][side]={'views':named,'imputation_medians':medians,'eligible_field_count':len(eligible),'field_audit':field_audit.to_dict(orient='records'),'synergy_edges':edges.to_dict(orient='records')}
    payload['contract_sha256']=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    path.write_text(json.dumps(payload,indent=2)+'\n')
    return payload

def _view_stability_audit(predictions, contract):
    """Audit fixed memberships plus score correlation for every test fold."""
    rows=[]
    for side, side_contract in contract['sides'].items():
        views=side_contract['views']
        for name, fields in views.items():
            rows.append({'audit':'membership','side':side,'view':name,'other_view':name,'value':1.,'detail':'fixed_cross_fold_membership','field_count':len(fields)})
        columns=['mv__'+name for name in views]
        for fold, frame in predictions[predictions.side_name.eq(side)].groupby('fold',observed=True):
            present=[c for c in columns if c in frame]
            corr=frame[present].corr(method='spearman')
            for i,left in enumerate(present):
                for right in present[i:]:
                    rows.append({'audit':'test_output_spearman','fold':fold,'side':side,'view':left.removeprefix('mv__'),'other_view':right.removeprefix('mv__'),'value':float(corr.loc[left,right]),'detail':'same frozen view IDs; fold-specific refits','field_count':pd.NA})
    return pd.DataFrame(rows)

def _base():
    cols=['candidate_id','__ts__','side_name','event','net_bps','gross_bps','p_clear','p_adverse','p_weak','prequential_base_expected_net_bps','shared_regime_contract_complete']
    d=pd.read_parquet(LEDGER,columns=cols); d=d[d.shared_regime_contract_complete.fillna(False)].copy(); d['__ts__']=pd.to_datetime(d.__ts__,utc=True); d['label_available_ts']=d.__ts__+DELAY
    for c in cols[3:-1]: d[c]=pd.to_numeric(d[c],errors='coerce').astype(np.float32)
    d['base_score']=(d.p_clear-.5*d.p_adverse).astype(np.float32); d=d[np.isfinite(d[['event','net_bps','p_clear','p_adverse','p_weak','prequential_base_expected_net_bps']]).all(axis=1)].copy()
    if not np.allclose(d.gross_bps-d.net_bps,100.,atol=.02): raise ValueError('cost contract failed')
    return d.sort_values(['__ts__','candidate_id'],kind='stable').reset_index(drop=True)

def _store_rows(ids, fields):
    con=duckdb.connect(); con.register('ids',ids[['candidate_id']])
    q='SELECT i.candidate_id,'+','.join('s."'+f.replace('"','""')+'"' for f in fields)+' FROM ids i INNER JOIN read_parquet(?) s USING(candidate_id)'
    out=con.execute(q,[str(STORE)]).fetchdf(); con.close()
    if len(out)!=len(ids) or out.candidate_id.duplicated().any(): raise ValueError('feature-store candidate join failed')
    return out.set_index('candidate_id').reindex(ids.candidate_id).reset_index()

def _sample(d,n): return d if len(d)<=n else d.sample(n,random_state=SEED,replace=False).sort_values(['__ts__','candidate_id'],kind='stable')

def _rank_target(residual):
    return np.select((residual<=-150,residual<=-50,residual<=50,residual<=150),(0,1,2,3),default=4).astype(np.int32)

def _ranker(features,target, *, query_id=None, params=None):
    x=features.copy(); x['q']=query_id.to_numpy() if query_id is not None else x['__ts__'].dt.floor('h'); x['row']=np.arange(len(x)); x=x[x.groupby('q').q.transform('size').ge(2)].sort_values(['q','row'],kind='stable'); order=x.row.to_numpy(int); groups=x.groupby('q',sort=False).size().to_numpy(np.int32); fields=[c for c in features if c!='__ts__']
    base=dict(objective='lambdarank',metric='ndcg',label_gain=[0,1,2,3,4],n_estimators=180,learning_rate=.04,num_leaves=20,min_child_samples=400,colsample_bytree=.8,reg_lambda=20.,random_state=SEED,n_jobs=1,verbosity=-1)
    if params: base.update(params)
    m=lgb.LGBMRanker(**base)
    m.fit(x[fields].replace([np.inf,-np.inf],np.nan),target[order],group=groups); return m,fields

def _metric(d,score,fold,arm,target,level,specialist=''):
    z=d[['candidate_id','net_bps','gross_bps']].copy(); z['score']=score; rows=[]
    for side,q in [('pooled',z),*[(s,x) for s,x in z.join(d.side_name).groupby('side_name',observed=True)]]:
        for tail in TAILS:
            n=max(1,int(np.ceil(len(q)*tail))); a=q.sort_values(['score','candidate_id'],ascending=[False,True],kind='stable').head(n)
            rows.append(dict(fold=fold,side=side,arm=arm,target=target,level=level,specialist=specialist,tail=tail,rows=len(q),tail_rows=n,net_bps=float(a.net_bps.mean()),gross_bps=float(a.gross_bps.mean()),rank_ic=float(q.score.rank().corr(q.net_bps.rank()))))
    return rows

def _fold(base,available,fold, frozen_contract, *, specialist_count=SPECIALIST_COUNT, max_meta_heads=6):
    a,b,c,e=map(_utc,(fold.train_start,fold.calibration_start,fold.test_start,fold.test_end)); tr=base[base.__ts__.between(a,b,inclusive='left')&base.label_available_ts.lt(b)]; ca=base[base.__ts__.between(b,c,inclusive='left')&base.label_available_ts.lt(c)]; te=base[base.__ts__.between(c,e,inclusive='left')]
    metrics=_metric(te,te.base_score.to_numpy(float),fold.name,'base_control','base','meta_impact'); pred=[]; selections=[]; routing=[]; global_meta={}
    for side in ('long','short'):
        train,cal,test=(x[x.side_name.eq(side)].copy() for x in (tr,ca,te))
        side_contract=frozen_contract['sides'][side]; views=side_contract['views']; medians=pd.Series(side_contract['imputation_medians'],dtype=float)
        if len(views)!=specialist_count: raise ValueError(f'{side}: frozen view count mismatch')
        for view,fields in views.items():
            selections.append(pd.DataFrame({'fold':fold.name,'side':side,'audit':'frozen_view_membership','view':view,'feature':fields,'selected':True}))
        fit=_sample(train,MAX_TRAIN_ROWS)
        scores={'binary_h12_net50':{'cal':{},'test':{}}}
        for view,fields in views.items():
            # Retrieve and release one 40--80-field view at a time: this
            # prevents retaining the 320--640-field union for every row.
            # Avoid a single 0.5m+ row joined frame per view.  The old
            # concatenation duplicated the complete OOS population and was
            # the main memory spike in 6--12 specialist sweeps.
            fitx=fit.merge(_store_rows(fit,fields),on='candidate_id',validate='one_to_one')
            calx=cal.merge(_store_rows(cal,fields),on='candidate_id',validate='one_to_one')
            testx=test.merge(_store_rows(test,fields),on='candidate_id',validate='one_to_one')
            med=medians.reindex(fields); X=fitx[fields].fillna(med).astype(np.float32); C=calx[fields].fillna(med).astype(np.float32); T=testx[fields].fillna(med).astype(np.float32)
            # Do not reuse the legacy R3 `event`: target is exact H12 net.
            target=(fitx.net_bps.to_numpy(float)>50.).astype(np.int8)
            clf=lgb.LGBMClassifier(objective='binary',n_estimators=180,learning_rate=.04,num_leaves=20,min_child_samples=400,colsample_bytree=.8,reg_lambda=20.,random_state=SEED,n_jobs=1,verbosity=-1).fit(X,target); scores['binary_h12_net50']['cal'][view]=clf.predict_proba(C)[:,1]; scores['binary_h12_net50']['test'][view]=clf.predict_proba(T)[:,1]
            del fitx,calx,testx,X,C,T
            gc.collect()
        for target,pack in scores.items():
            cal_s=cal.copy(); test_s=test.copy(); mapping={v:'mv__'+v for v in views}
            for v,col in mapping.items():
                cal_s[col]=pack['cal'][v]; test_s[col]=pack['test'][v]
                # Specialist standalone diagnostics remain side-local; global
                # selection is assessed only after common-bps meta mapping.
                metrics.extend(row for row in _metric(test,pack['test'][v],fold.name,'specialist',target,'standalone',v) if row['side']==side)
            cal_s['binary_h12_net50']=(cal_s.net_bps>50.).astype(np.int8)
            diag,pairs=opportunity_conditioned_synergy(cal_s,mapping,base_score_column='base_score',label_column='binary_h12_net50'); diag['fold']=fold.name;diag['side']=side;diag['target']=target;routing.append(diag)
            applied=apply_synergy_features(test_s,mapping,diag,base_score_column='base_score')
            for col in pairs: cal_s[col]=pairs[col]; test_s[col]=applied[col] if col in applied else 0.
            # Select heads on the earlier half of the temporally prior meta
            # calibration period.  The later half trains the residual model;
            # test labels never influence either selection or routing.
            cal_s=cal_s.sort_values(['__ts__','candidate_id'],kind='stable').reset_index(drop=True)
            select_rows=max(1,len(cal_s)//2); select_frame=cal_s.iloc[:select_rows].copy()
            select_frame['residual_grade']=_rank_target(select_frame.net_bps.to_numpy(float)-select_frame.prequential_base_expected_net_bps.to_numpy(float))
            selected,head_audit=select_complementary_heads(select_frame,list(mapping.values()),target_column='residual_grade',base_score_column='base_score',max_heads=max(1,min(int(max_meta_heads),len(mapping))),minimum_cmi=.001)
            head_audit['fold']=fold.name; head_audit['side']=side; head_audit['target']=target; head_audit['audit']='conditional_mi_head_selection'; selections.append(head_audit)
            pair_columns=[col for col in pairs.columns if any(f'__{name.removeprefix("mv__")}__' in col for name in selected)]
            meta_fields=['p_clear','p_adverse','p_weak','base_score','prequential_base_expected_net_bps',*selected,*pair_columns]
            cal_s=cal_s.iloc[select_rows:].copy()
            test_s=test_s.copy()
            residual=cal_s.net_bps.to_numpy(float)-cal_s.prequential_base_expected_net_bps.to_numpy(float); meta,fields=_ranker(pd.concat([cal_s[['__ts__']],cal_s[meta_fields]],axis=1),_rank_target(residual)); raw=meta.predict(test_s[fields].fillna(0.)); rawcal=meta.predict(cal_s[fields].fillna(0.)); iso=fit_residual_calibration(rawcal,residual); score=test_s.prequential_base_expected_net_bps.to_numpy(float)+np.clip(iso.predict(raw),-50.,50.)
            z=test[['candidate_id','__ts__','side_name','net_bps','gross_bps','prequential_base_expected_net_bps']].copy();z['fold']=fold.name;z['target']=target;z['score']=score
            for col in mapping.values(): z[col]=test_s[col].to_numpy(np.float32)
            pred.append(z);global_meta.setdefault(target,[]).append(z)
    for target,pieces in global_meta.items():
        combined=pd.concat(pieces,ignore_index=True)
        metrics.extend(_metric(combined,combined.score.to_numpy(float),fold.name,'meta_lambdarank',target,'meta_impact'))
    return metrics,pred,selections,routing

def run(out=OUT,folds=LONG_HISTORY_FOLDS[3:], *, specialist_count=SPECIALIST_COUNT, max_meta_heads=6):
    base=_base(); available=_schema(); out.mkdir(parents=True,exist_ok=True); _write_feature_contract(out,available); frozen_contract=_frozen_view_contract(base,available,out,specialist_count=specialist_count); m=[];p=[];s=[];r=[]; completed=[]
    for fold in folds:
        a,b,c,d=_fold(base,available,fold,frozen_contract,specialist_count=specialist_count,max_meta_heads=max_meta_heads);m+=a;p+=b;s+=c;r+=d
        # A multi-fold sweep can be interrupted by resource management.  Each
        # completed chronological fold is independently useful and must not be
        # lost merely because a later fold fails or is cancelled.
        pd.DataFrame(m).to_parquet(out/'metrics.checkpoint.parquet',index=False)
        pd.concat(p,ignore_index=True).to_parquet(out/'predictions.checkpoint.parquet',index=False)
        pd.concat(s,ignore_index=True).to_parquet(out/'view_discovery.checkpoint.parquet',index=False)
        pd.concat(r,ignore_index=True).to_parquet(out/'routing.checkpoint.parquet',index=False)
        completed.append(fold.name)
        (out/'progress.json').write_text(json.dumps({'completed_folds':completed,'last_completed_fold':fold.name,'status':'running'},indent=2)+'\n')
    predictions=pd.concat(p,ignore_index=True)
    pd.DataFrame(m).to_parquet(out/'metrics.parquet',index=False);predictions.to_parquet(out/'predictions.parquet',index=False);pd.concat(s,ignore_index=True).to_parquet(out/'view_discovery.parquet',index=False);pd.concat(r,ignore_index=True).to_parquet(out/'routing.parquet',index=False);_view_stability_audit(predictions,frozen_contract).to_parquet(out/'view_stability.parquet',index=False)
    (out/'manifest.json').write_text(json.dumps({'schema':'broad_multiview_specialist_lambdarank_v3','views':f'{specialist_count} side-specific, cross-fold-frozen opportunity coactivation clusters','view_contract':'frozen_specialist_views.json','max_meta_heads':int(max_meta_heads),'raw_feature_contract':'provenance-backed generator-registry allowlist','specialist_target':'exact_h12_net_bps_gt_50','meta':'per-row bps ordinal LambdaRank residual; CMI-selected specialist scores and selected synergy fields'},indent=2)+'\n')
    (out/'progress.json').write_text(json.dumps({'completed_folds':completed,'last_completed_fold':completed[-1] if completed else None,'status':'complete'},indent=2)+'\n')
    return out

def run_residual_only_control(out,folds=LONG_HISTORY_FOLDS[3:]):
    """Matched LambdaRank residual control with no specialist inputs."""
    base=_base(); metrics=[]; predictions=[]
    for fold in folds:
        a,b,c,e=map(_utc,(fold.train_start,fold.calibration_start,fold.test_start,fold.test_end))
        cal=base[base.__ts__.between(b,c,inclusive='left')&base.label_available_ts.lt(c)]
        test=base[base.__ts__.between(c,e,inclusive='left')]
        pieces=[]
        for side in ('long','short'):
            ca=cal[cal.side_name.eq(side)].copy(); te=test[test.side_name.eq(side)].copy()
            fields=['p_clear','p_adverse','p_weak','base_score','prequential_base_expected_net_bps']
            residual=ca.net_bps.to_numpy(float)-ca.prequential_base_expected_net_bps.to_numpy(float)
            model,usable=_ranker(pd.concat([ca[['__ts__']],ca[fields]],axis=1),_rank_target(residual))
            raw_cal=model.predict(ca[usable]); raw_test=model.predict(te[usable]); iso=fit_residual_calibration(raw_cal,residual)
            z=te[['candidate_id','__ts__','side_name','net_bps','gross_bps','prequential_base_expected_net_bps']].copy();z['score']=te.prequential_base_expected_net_bps.to_numpy(float)+np.clip(iso.predict(raw_test),-50.,50.); pieces.append(z)
        combined=pd.concat(pieces,ignore_index=True);metrics.extend(_metric(combined,combined.score.to_numpy(float),fold.name,'meta_lambdarank','no_specialist','meta_impact'));combined['fold']=fold.name;predictions.append(combined)
    out.mkdir(parents=True,exist_ok=True);pd.DataFrame(metrics).to_parquet(out/'metrics.parquet',index=False);pd.concat(predictions,ignore_index=True).to_parquet(out/'predictions.parquet',index=False);(out/'manifest.json').write_text(json.dumps({'schema':'matched_no_specialist_lambdarank_control_v1','meta':'same side-local ordinal-bps native LambdaRank and isotonic residual reconstruction; specialist outputs and routing features omitted'},indent=2)+'\n');return out
if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--out',type=Path,default=OUT);q.add_argument('--all-folds',action='store_true');q.add_argument('--residual-only-control',action='store_true');q.add_argument('--specialist-count',type=int,default=SPECIALIST_COUNT,choices=range(6,13));q.add_argument('--max-meta-heads',type=int,default=6);q.add_argument('--fold-index',type=int,action='append',default=[]);a=q.parse_args();folds=LONG_HISTORY_FOLDS if a.all_folds else LONG_HISTORY_FOLDS[3:];folds=[folds[index] for index in a.fold_index] if a.fold_index else folds;print(run_residual_only_control(a.out,folds) if a.residual_only_control else run(a.out,folds,specialist_count=a.specialist_count,max_meta_heads=a.max_meta_heads))
