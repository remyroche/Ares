#!/usr/bin/env python3
"""Memory-bounded all-meta-feature, transport-supervised rule discovery."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import duckdb, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.tree import DecisionTreeRegressor, _tree
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from extreme_price_movements.config import CFG
from extreme_price_movements.transport_supervised_archetypes import configured_available_meta_features, training_univariate_screen

LEDGER=ROOT/'data_perp/artifacts/tp6_m6_shared_regime_residual_features_20260809_v3/shared_regime_residual_ledger.parquet'
PANEL=ROOT/'data_perp/artifacts/full_universe_t2_t4_panel_20260801_v3/parts/*.parquet'
OUT=ROOT/'data_perp/artifacts/transport_supervised_archetypes_20260803_v1'

def extract(tree, names, fold, side, head, seed, depth, leaf_share):
    out=[]; t=tree.tree_
    def walk(node, path):
        if t.feature[node] != _tree.TREE_UNDEFINED:
            f=names[t.feature[node]]; th=float(t.threshold[node]); walk(t.children_left[node],path+[(f,-1,th)]);walk(t.children_right[node],path+[(f,1,th)])
        elif path: out.append({'fold':fold,'side_name':side,'head':head,'seed':seed,'depth':depth,'leaf_share':leaf_share,'conditions':json.dumps(path),'n_conditions':len(path),'leaf_value':float(t.value[node][0,0]),'leaf_rows':int(t.n_node_samples[node])})
    walk(0,[]);return out


SETUP_FIELDS = (
    "p_adverse", "p_weak", "p_clear", "prequential_base_expected_net_bps",
    "cost_to_atr",
)


def setup_oof_residual(train: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Strict expanding-OOF setup expectation for one side/event head.

    This deliberately has no searched market-context inputs.  At every OOF
    block it knows only the same frozen base output, its probability simplex
    and executable cost/ATR geometry.  The first chronological block is not
    scoreable and is retained only as fitting history, never silently given a
    zero residual.
    """
    ordered = train.sort_values("__ts__", kind="stable").copy()
    # Split *timestamp groups*, not rows.  A rank(method="first") split can
    # otherwise place two candidates from the same decision bar in different
    # inner folds.  The additional availability embargo below then excludes
    # the trailing unresolved H12 labels from fitting history.
    timestamps = pd.Index(ordered["__ts__"].drop_duplicates().sort_values())
    block_by_timestamp = {
        timestamp: min(3, int(4 * position / max(len(timestamps), 1)))
        for position, timestamp in enumerate(timestamps)
    }
    ordered["_inner_block"] = ordered["__ts__"].map(block_by_timestamp).astype(int)
    prediction = pd.Series(np.nan, index=ordered.index, dtype=float)
    audit: list[dict[str, object]] = []
    for current in range(1, 4):
        held = ordered.loc[ordered._inner_block.eq(current)]
        held_start = held.__ts__.min()
        history = ordered.loc[
            ordered._inner_block.lt(current)
            & ordered.__ts__.lt(held_start - pd.Timedelta(hours=13))
        ]
        # Decision-time labels resolve after the decision; the expanding block
        # boundary is wider than H12+entry latency and therefore all history is
        # resolved before this held block begins.
        if len(history) < 2_000 or len(held) == 0:
            continue
        median = history.loc[:, SETUP_FIELDS].replace([np.inf, -np.inf], np.nan).median().fillna(0.)
        x_train = history.loc[:, SETUP_FIELDS].replace([np.inf, -np.inf], np.nan).fillna(median).to_numpy(np.float32)
        x_held = held.loc[:, SETUP_FIELDS].replace([np.inf, -np.inf], np.nan).fillna(median).to_numpy(np.float32)
        model = lgb.LGBMRegressor(
            objective="huber", alpha=.9, n_estimators=120, learning_rate=.04,
            num_leaves=16, min_child_samples=750, colsample_bytree=1.,
            reg_lambda=30., random_state=20260803 + current, n_jobs=1,
            verbosity=-1,
        ).fit(x_train, history.net_bps.to_numpy(float))
        predicted = model.predict(x_held)
        prediction.loc[held.index] = predicted
        audit.append({
            "inner_block": current, "history_rows": len(history), "held_rows": len(held),
            "max_history_decision_ts": str(history.__ts__.max()),
            "min_held_decision_ts": str(held.__ts__.min()),
            "label_availability_embargo_hours": 13,
            "setup_fields": list(SETUP_FIELDS),
            "oof_setup_mae_bps": float(np.mean(np.abs(held.net_bps.to_numpy(float) - predicted))),
        })
    return prediction, pd.DataFrame(audit)

def run():
    OUT.mkdir(parents=True,exist_ok=True); con=duckdb.connect(config={'threads':'2','memory_limit':'512MB','temp_directory':'/tmp'})
    panel_cols=con.execute(f"DESCRIBE SELECT * FROM read_parquet('{PANEL.as_posix()}') LIMIT 1").fetchdf().column_name.tolist(); feats=configured_available_meta_features(CFG,panel_cols)
    # Deterministic 20% proxy: every eligible feature is screened; rows—not
    # features—are subsampled solely for constrained rule generation.
    cols=', '.join('p."'+x+'"' for x in feats)
    q=f'''SELECT l.candidate_id,l.__ts__,l.side_name,l.era,l.net_bps,l.gross_bps,l.event,l.p_adverse,l.p_weak,l.p_clear,l.prequential_base_expected_net_bps,
    p."atr_1h", p."decision_price", p."assumed_round_trip_cost_bps", {cols}
    FROM read_parquet('{LEDGER.as_posix()}') l JOIN read_parquet('{PANEL.as_posix()}') p USING(candidate_id)
    WHERE l.shared_regime_contract_complete AND l.prequential_base_expected_net_bps IS NOT NULL AND abs(hash(l.candidate_id)) % 5 = 0'''
    d=con.execute(q).fetchdf();con.close();d['__ts__']=pd.to_datetime(d.__ts__,utc=True);d=d.sort_values('__ts__').reset_index(drop=True)
    atr_bps = np.abs(d.atr_1h.to_numpy(float)) / np.maximum(np.abs(d.decision_price.to_numpy(float)), 1e-12) * 1e4
    d["cost_to_atr"] = np.clip(d.assumed_round_trip_cost_bps.to_numpy(float) / np.maximum(atr_bps, 1e-6), 0., 100.)
    coverage=1-d[feats].isna().mean(); usable=coverage[coverage>=.90].index.tolist();pd.DataFrame({'feature':feats,'coverage':coverage.reindex(feats),'usable':pd.Index(feats).isin(usable)}).to_parquet(OUT/'meta_feature_coverage.parquet',index=False)
    # Five chronological environments.  Clear/adverse are the two realised
    # event populations available in TP6; weak is explicitly recorded absent.
    # Environment folds likewise respect whole decision bars.
    timestamps = pd.Index(d["__ts__"].drop_duplicates().sort_values())
    fold_lookup = {timestamp: min(4, int(5 * position / max(len(timestamps), 1))) for position, timestamp in enumerate(timestamps)}
    d['fold']=d["__ts__"].map(fold_lookup).astype(int);records=[];screens=[];scalers=[]
    setup_audits=[]
    for fold in range(2,5):
      train=d[d.fold.lt(fold)];
      for side in ('long','short'):
       for head,mask in [('clear',lambda x:x.event.eq(1)),('adverse',lambda x:x.event.eq(0))]:
        tr=train[train.side_name.eq(side)&mask(train)].copy()
        if len(tr)<2000:continue
        setup_prediction, setup_audit = setup_oof_residual(tr)
        setup_audit["fold"], setup_audit["side_name"], setup_audit["head"] = fold, side, head
        setup_audits.append(setup_audit)
        tr = tr.loc[setup_prediction.notna()].copy()
        tr["setup_oof_expected_net_bps"] = setup_prediction.loc[tr.index].to_numpy(float)
        if len(tr) < 2_000:
            continue
        # Strict OOF conditional-payoff residual.  Context enters only below,
        # after this setup expectation has been generated from earlier labels.
        y=tr.net_bps.to_numpy(float)-tr.setup_oof_expected_net_bps.to_numpy(float); selected=training_univariate_screen(tr,usable,y,maximum=64);screens += [{'fold':fold,'side_name':side,'head':head,'feature':f,'rank':i} for i,f in enumerate(selected)]
        med=tr[selected].median().fillna(0);scale=(tr[selected].quantile(.75)-tr[selected].quantile(.25)).replace(0,1).fillna(1)
        scalers += [{"fold": fold, "side_name": side, "head": head, "feature": field, "center": float(med[field]), "scale": float(scale[field]), "scaler_source": "earlier_only_strict_oof_setup_residual_training_rows"} for field in selected]
        x=((tr[selected].fillna(med)-med)/scale).clip(-8,8).to_numpy(np.float32)
        weights=np.sqrt(len(tr)/tr.groupby(tr.era).era.transform('size').to_numpy(float));weights/=weights.mean()
        for seed in (11,29,47):
         for depth in (2,3,4):
          for leaf_share in (.01,.02,.05):
           m=DecisionTreeRegressor(max_depth=depth,min_samples_leaf=max(20,int(len(tr)*leaf_share)),max_features=min(4,len(selected)),random_state=seed,ccp_alpha=.001)
           m.fit(x,y,sample_weight=weights);records += extract(m,selected,fold,side,head,seed,depth,leaf_share)
    pd.DataFrame(records).to_parquet(OUT/'archetype_rule_candidates.parquet',index=False);pd.DataFrame(screens).to_parquet(OUT/'archetype_fold_feature_screens.parquet',index=False)
    pd.DataFrame(scalers).to_parquet(OUT/'archetype_feature_scalers.parquet', index=False)
    pd.concat(setup_audits, ignore_index=True).to_parquet(OUT/'archetype_setup_baseline_oof.parquet', index=False)
    (OUT/'archetype_discovery_manifest.json').write_text(json.dumps({'all_configured_meta_features':len(feats),'usable_meta_features':len(usable),'row_proxy':'deterministic 20% candidate sample; every usable feature screened in every fold/head/side','folds':5,'seeds':[11,29,47],'depths':[2,3,4],'leaf_shares':[.01,.02,.05],'heads':['clear','adverse'],'weak_head':'UNAVAILABLE_IN_TP6_BINARY_EVENT_LEDGER','setup_baseline':'strict expanding OOF Huber expectation: base simplex + prequential expected-net + decision-time cost_to_atr only','setup_fields':list(SETUP_FIELDS),'feature_scaler_lineage':'fold × side × event-head medians/IQR scales from earlier-only strict-OOF setup-residual rows','strictness':'rules generated only from earlier chronological folds; context target is setup OOF residual'},indent=2)+'\n')
    print(OUT)
if __name__=='__main__':run()
