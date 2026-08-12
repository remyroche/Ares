#!/usr/bin/env python3
"""Nested D2-archetype residual-meta ablation: train fold 3, test fold 4.

The same frozen D2 rule catalogue is represented on both folds.  This is the
first causal test of whether transport-supervised soft archetypes add to the
base/setup context, rather than merely describing their own discovery period.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from extreme_price_movements.base_error_tercile_meta import (  # noqa: E402
    expected_base_error_bps, fit_base_error_tercile_map, labels_from_base_error,
)
from extreme_price_movements.transport_supervised_archetypes import training_univariate_screen  # noqa: E402

ARTIFACT = ROOT / "data_perp/artifacts/transport_supervised_archetypes_20260803_v1"


def _matrix(train: pd.DataFrame, test: pd.DataFrame, fields: list[str]) -> tuple[np.ndarray, np.ndarray]:
    median = train.loc[:, fields].replace([np.inf, -np.inf], np.nan).median().fillna(0.)
    return (
        train.loc[:, fields].replace([np.inf, -np.inf], np.nan).fillna(median).to_numpy(np.float32),
        test.loc[:, fields].replace([np.inf, -np.inf], np.nan).fillna(median).to_numpy(np.float32),
    )


def _top(frame: pd.DataFrame, score: str, arm: str) -> list[dict[str, object]]:
    order = frame.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable")
    records = []
    for fraction in (.01, .05, .10):
        chosen = order.head(max(1, int(np.ceil(len(order) * fraction))))
        for scope, part in (("global", chosen), ("long", chosen[chosen.side_name.eq("long")]), ("short", chosen[chosen.side_name.eq("short")])):
            if len(part): records.append({"arm": arm, "scope": scope, "top_fraction": fraction, "rows": len(part), "net_bps": float(part.net_bps.mean()), "gross_bps": float(part.gross_bps.mean()), "total_net_bps": float(part.net_bps.sum()), "long_share": float(chosen.side_name.eq("long").mean())})
    return records


def run() -> None:
    frame = pd.read_parquet(ARTIFACT / "archetype_soft_memberships_oof.parquet")
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
    memberships = [field for field in frame if field.startswith("frozen_d2__")]
    train = frame.loc[frame.fold.eq(3)].copy()
    test = frame.loc[frame.fold.eq(4)].copy()
    # Membership output is OOF but labels still observe the normal H12+entry
    # delay.  Explicitly embargo the tail of the meta training fold.
    train = train.loc[train.__ts__.lt(test.__ts__.min() - pd.Timedelta(hours=13))].copy()
    mandatory = ["p_adverse", "p_weak", "p_clear", "prequential_base_expected_net_bps", "cost_to_atr"]
    output = test[["candidate_id", "__ts__", "side_name", "net_bps", "gross_bps", "prequential_base_expected_net_bps"]].copy()
    metric_rows, economic_rows, selection_rows = [], [], []
    for arm in ("A0_setup_only", "A1_setup_plus_transport_archetypes"):
        score = np.empty(len(test), dtype=float)
        probability = np.empty((len(test), 3), dtype=float)
        selected_by_side = {}
        for side in ("long", "short"):
            tr = train.loc[train.side_name.eq(side)].copy(); te = test.loc[test.side_name.eq(side)].copy()
            mapping = fit_base_error_tercile_map(train, shrinkage_support=1_000.)
            label = labels_from_base_error(tr, mapping)
            selected = [] if arm == "A0_setup_only" else training_univariate_screen(tr, memberships, label.astype(float), maximum=12)
            fields = [*mandatory, *selected]
            x_train, x_test = _matrix(tr, te, fields)
            count = np.bincount(label, minlength=3).astype(float)
            weight = np.sqrt(len(label) / np.maximum(3. * count[label], 1.)); weight = np.clip(weight / weight.mean(), .5, 2.)
            model = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=140, learning_rate=.04, num_leaves=16, min_child_samples=600, colsample_bytree=.9, reg_lambda=30., random_state=20260803, n_jobs=1, verbosity=-1).fit(x_train, label, sample_weight=weight)
            p = np.clip(model.predict_proba(x_test), 1e-6, 1.); p /= p.sum(axis=1, keepdims=True)
            correction = expected_base_error_bps(p, te.side_name, mapping)
            position = test.index.get_indexer(te.index); probability[position] = p; score[position] = te.prequential_base_expected_net_bps.to_numpy(float) + correction
            observed = labels_from_base_error(te, mapping)
            metric_rows.append({"arm": arm, "side_name": side, "train_rows": len(tr), "test_rows": len(te), "selected_archetype_memberships": selected, "test_log_loss": float(log_loss(observed, p, labels=[0, 1, 2]))})
            selected_by_side[side] = selected
        output[f"{arm}_score_bps"] = score
        output[f"{arm}_p_overestimate"] = probability[:, 0]
        output[f"{arm}_p_correct"] = probability[:, 1]
        output[f"{arm}_p_underestimate"] = probability[:, 2]
        economic_rows.extend(_top(output.assign(net_bps=test.net_bps.to_numpy(float), gross_bps=test.gross_bps.to_numpy(float)), f"{arm}_score_bps", arm))
        if arm == "A1_setup_plus_transport_archetypes":
            # Permutation MDA is strictly an evaluation diagnostic.  It never
            # feeds feature selection or model fitting.
            for side, selected in selected_by_side.items():
                if not selected: continue
                tr = train.loc[train.side_name.eq(side)].copy(); te = test.loc[test.side_name.eq(side)].copy(); position = test.index.get_indexer(te.index)
                mapping = fit_base_error_tercile_map(train, shrinkage_support=1_000.); label = labels_from_base_error(tr, mapping); fields=[*mandatory,*selected]; x_train,_ = _matrix(tr,te,fields)
                model=lgb.LGBMClassifier(objective="multiclass",num_class=3,n_estimators=140,learning_rate=.04,num_leaves=16,min_child_samples=600,colsample_bytree=.9,reg_lambda=30.,random_state=20260803,n_jobs=1,verbosity=-1).fit(x_train,label)
                observed=labels_from_base_error(te,mapping); base_p=probability[position]
                rng=np.random.default_rng(20260803)
                for field in selected:
                    perturbed=te.loc[:,fields].copy(); perturbed[field]=rng.permutation(perturbed[field].to_numpy())
                    median=tr.loc[:,fields].replace([np.inf,-np.inf],np.nan).median().fillna(0.); p=np.clip(model.predict_proba(perturbed.replace([np.inf,-np.inf],np.nan).fillna(median).to_numpy(np.float32)),1e-6,1.);p/=p.sum(axis=1,keepdims=True)
                    selection_rows.append({"side_name":side,"membership_column":field,"log_loss_increase":float(log_loss(observed,p,labels=[0,1,2])-log_loss(observed,base_p,labels=[0,1,2]))})
    # Nested archetype-count ablation.  Ranking is based only on the fold-3
    # training target; fold 4 is opened once for each predeclared count.
    count_rows = []
    for count_limit in (0, 1, 2, 4, 8, 12):
        score = np.empty(len(test), dtype=float)
        loss_by_side = []
        for side in ("long", "short"):
            tr = train.loc[train.side_name.eq(side)].copy(); te = test.loc[test.side_name.eq(side)].copy()
            mapping = fit_base_error_tercile_map(train, shrinkage_support=1_000.); label = labels_from_base_error(tr, mapping)
            selected = training_univariate_screen(tr, memberships, label.astype(float), maximum=count_limit) if count_limit else []
            fields = [*mandatory, *selected]; x_train, x_test = _matrix(tr, te, fields)
            class_count=np.bincount(label,minlength=3).astype(float); weight=np.sqrt(len(label)/np.maximum(3.*class_count[label],1.));weight=np.clip(weight/weight.mean(),.5,2.)
            model=lgb.LGBMClassifier(objective="multiclass",num_class=3,n_estimators=140,learning_rate=.04,num_leaves=16,min_child_samples=600,colsample_bytree=.9,reg_lambda=30.,random_state=20260803,n_jobs=1,verbosity=-1).fit(x_train,label,sample_weight=weight)
            p=np.clip(model.predict_proba(x_test),1e-6,1.);p/=p.sum(axis=1,keepdims=True); correction=expected_base_error_bps(p,te.side_name,mapping)
            score[test.index.get_indexer(te.index)] = te.prequential_base_expected_net_bps.to_numpy(float)+correction
            loss_by_side.append(float(log_loss(labels_from_base_error(te,mapping),p,labels=[0,1,2])))
        ranked=test.assign(score_bps=score).sort_values(["score_bps","candidate_id"],ascending=[False,True],kind="stable")
        for fraction in (.01,.05,.10):
            chosen=ranked.head(max(1,int(np.ceil(len(ranked)*fraction))))
            count_rows.append({"archetype_count":count_limit,"top_fraction":fraction,"rows":len(chosen),"net_bps":float(chosen.net_bps.mean()),"gross_bps":float(chosen.gross_bps.mean()),"mean_side_log_loss":float(np.mean(loss_by_side)),"long_share":float(chosen.side_name.eq("long").mean())})
    output.to_parquet(ARTIFACT / "archetype_nested_meta_oof_predictions.parquet", index=False)
    pd.DataFrame(metric_rows).to_parquet(ARTIFACT / "archetype_nested_meta_metrics.parquet", index=False)
    pd.DataFrame(economic_rows).to_parquet(ARTIFACT / "archetype_nested_meta_economics.parquet", index=False)
    pd.DataFrame(selection_rows).to_parquet(ARTIFACT / "archetype_transport_mda.parquet", index=False)
    pd.DataFrame(count_rows).to_parquet(ARTIFACT / "archetype_nested_count_ablation.parquet", index=False)
    (ARTIFACT / "archetype_nested_meta_manifest.json").write_text(json.dumps({"schema":"transport_archetype_nested_meta_v1","train_fold":3,"test_fold":4,"train_embargo_hours":13,"catalogue":"frozen D2 discovery-only catalogue; built before fold 3 and held fixed through test fold 4","arms":["A0_setup_only","A1_setup_plus_transport_archetypes"],"global_ranking":"after common-bps base + expected residual correction","promotion_status":"DIAGNOSTIC_PENDING_CONDITIONAL_SUPPORT_EFFECT_TRANSPORT_AND_WORST_PERIOD_GATES"},indent=2)+"\n")


if __name__ == "__main__": run()
