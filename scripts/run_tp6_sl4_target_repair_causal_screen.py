#!/usr/bin/env python3
"""Chronological causal screen: B3 control vs TP6/SL4 robust-clear repairs."""
from __future__ import annotations

import argparse, gc, json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SIDES = ("long", "short")
PARAMS = dict(n_estimators=140, learning_rate=.05, num_leaves=31, min_child_samples=350,
              subsample=.8, colsample_bytree=.8, reg_lambda=8., random_state=20260802, n_jobs=1, verbosity=-1)


def _assert_winner_contract(root: Path) -> None:
    """Fail closed if a stale sidecar is accidentally supplied.

    Target validity depends on the selected execution geometry.  In
    particular, this research is *not* an evaluation of the earlier 3/2 ATR
    contract, even though the frozen feature-manifest keys retain that legacy
    experiment name.
    """
    manifest = json.loads((root / "manifest.json").read_text())
    exit_contract = str(manifest.get("exit", ""))
    if "TP=+6 ATR" not in exit_contract or "SL=-4 ATR" not in exit_contract or "H12" not in exit_contract:
        raise ValueError(f"expected selected TP6/SL4/H12 contract, got: {exit_contract!r}")
    if float(manifest.get("cost", {}).get("round_trip_bps", float("nan"))) != 100.0:
        raise ValueError("expected selected 100bps fixed-cost contract")


def _features(root: Path) -> dict[str, list[str]]:
    out = {}
    for side in SIDES:
        x = json.loads((root / side / "target_family_manifest.json").read_text())
        cols = x["feature_contract"][f"T2_soft_barrier|tp3_sl2|{side}"]
        if not 30 <= len(cols) <= 40: raise ValueError(f"bad frozen {side} contract")
        out[side] = cols
    return out


def _read_side(panel: Path, winner: Path, robust: Path, side: str, cols: list[str]) -> pd.DataFrame:
    pieces = []
    identity = ["candidate_id", "__ts__", "side_name", *cols]
    winner_cols = ["candidate_id", "t2_tp6_sl4_event", "t2_tp6_sl4_exit_minute", "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps", "__label_available_at__"]
    robust_cols = ["candidate_id", "label_valid", "lower_touch_minute", "atr_bps", "robust_clear_event_b0", "robust_clear_event_b25", "robust_clear_event_b50", "robust_clear_soft_b0_t50", "robust_clear_soft_b25_t50", "robust_clear_soft_b50_t50"]
    for p in sorted((panel / "parts").glob("*.parquet")):
        base = pd.read_parquet(p, columns=identity)
        base = base.loc[base.side_name.eq(side)]
        if base.empty: continue
        w = pd.read_parquet(winner / "parts" / p.name, columns=winner_cols)
        r = pd.read_parquet(robust / "parts" / p.name, columns=robust_cols)
        x = base.merge(w, on="candidate_id", how="inner", validate="one_to_one").merge(r, on="candidate_id", how="left", validate="one_to_one")
        x = x.loc[x.label_valid.eq(True)]
        pieces.append(x)
    x = pd.concat(pieces, ignore_index=True)
    x["__ts__"] = pd.to_datetime(x["__ts__"], utc=True); x["__label_available_at__"] = pd.to_datetime(x["__label_available_at__"], utc=True)
    if not np.allclose(x.t4_tp6_sl4_gross_bps - 100., x.t4_tp6_sl4_net_bps, atol=2e-3): raise ValueError("cost mismatch")
    event, minute = x.t2_tp6_sl4_event.to_numpy(int), x.t2_tp6_sl4_exit_minute.to_numpy(float)
    confidence = .75 + .25*np.exp(-np.minimum(minute,720.)/60./8.)
    x["R0_B3"] = np.where(event == 0, confidence, (1-confidence)/2.)
    x["R1_direct_net"] = 1/(1+np.exp(-np.clip(x.t4_tp6_sl4_net_bps.to_numpy(float)/50.,-35,35)))
    x["R4_ordinal"] = np.select([x.t4_tp6_sl4_net_bps.le(-200),x.t4_tp6_sl4_net_bps.le(0),x.t4_tp6_sl4_net_bps.le(50)],[0.,1.,2.],default=3.)/3.
    # R3 uses mutually exclusive, path-ordered memberships.  A robust clear
    # has precedence: it genuinely cleared cost+25bps before a later adverse
    # touch.  Otherwise an adverse touch is a failure; the remainder are
    # complete, executable weak/unresolved timeouts.  This removes the old
    # single middle-value timeout collision without treating an invalid path
    # as an economic failure (those rows were filtered above).
    x["R3_economic_simplex_b25"] = np.select(
        [x.robust_clear_event_b25.eq(1.), x.lower_touch_minute.ge(0)],
        [2, 0], default=1,
    ).astype(np.int8)
    # A strict economic refinement: some paths clear early but surrender the
    # opportunity by the selected contract's realised exit.  Require a +50bp
    # retained net result for the clear class; unresolved or surrendered paths
    # remain weak, while adverse-first remains explicitly adverse.
    x["R3_retained_clear_b25_n50"] = np.select(
        [x.robust_clear_event_b25.eq(1.) & x.t4_tp6_sl4_net_bps.gt(50.), x.lower_touch_minute.ge(0)],
        [2, 0], default=1,
    ).astype(np.int8)
    return x


def _matrix(x: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return x[cols].replace([np.inf,-np.inf],np.nan).fillna(0.).to_numpy(np.float32)


def _map_bps(pred_cal: np.ndarray, net_cal: np.ndarray, pred_test: np.ndarray) -> np.ndarray:
    # A calibration-only ten-bin conditional mean map gives both sides a common
    # bps scale. No evaluation outcome participates in the conversion.
    edges = np.unique(np.quantile(pred_cal, np.linspace(0,1,11)))
    if len(edges) < 3: return np.full(len(pred_test), float(np.mean(net_cal)))
    bins = np.clip(np.digitize(pred_cal, edges[1:-1], right=True),0,9)
    means = np.array([net_cal[bins==i].mean() if (bins==i).any() else net_cal.mean() for i in range(10)])
    test_bins = np.clip(np.digitize(pred_test, edges[1:-1], right=True),0,9)
    return means[test_bins]


def _r3_weights(train: pd.DataFrame, target: str, mode: str) -> np.ndarray | None:
    """Fit all weight quantities on resolved training rows only.

    The volatility groups are decision-time ATR-bps quartiles, not a future
    outcome regime.  Contract certainty is derived from three predeclared
    cost buffers and is training-only; neither is supplied as a model field.
    """
    if mode == "uniform":
        return None
    agreement = train[["robust_clear_event_b0", "robust_clear_event_b25", "robust_clear_event_b50"]].nunique(axis=1).eq(1).to_numpy(float)
    certainty = .5 + .5 * agreement
    if mode == "certainty":
        return certainty
    q = np.unique(np.quantile(train.atr_bps.to_numpy(float), [.25, .5, .75]))
    bins_train = np.digitize(train.atr_bps.to_numpy(float), q, right=True)
    counts = np.bincount(bins_train, minlength=4).astype(float)
    regime = np.sqrt(len(train) / np.maximum(counts, 1.))[bins_train]
    regime /= regime.mean()
    if mode == "regime":
        return regime
    if mode == "composite":
        cls = train[target].to_numpy(int)
        class_counts = np.bincount(cls, minlength=3).astype(float)
        class_weight = np.sqrt(len(train) / np.maximum(class_counts, 1.))[cls]
        class_weight /= class_weight.mean()
        result = certainty * regime * class_weight
        result = np.clip(result, .25, 4.)
        return result / result.mean()
    raise ValueError(f"unknown R3 weight mode: {mode}")


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel",type=Path,default=ROOT/"data_perp/artifacts/full_universe_t2_t4_panel_20260801_v3")
    p.add_argument("--winner",type=Path,default=ROOT/"data_perp/artifacts/full_universe_tp6_sl4_h12_sidecar_20260802_v1")
    p.add_argument("--robust",type=Path,default=ROOT/"data_perp/artifacts/tp6_sl4_robust_clear_labels_20260802_v1")
    p.add_argument("--features",type=Path,default=ROOT/"data_perp/artifacts/full_universe_base_hpo_20260802_v1")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--side", choices=SIDES, default=None, help="run one side as a resumable causal-screen shard")
    p.add_argument("--arms", default="", help="comma-separated target arms; empty runs the declared full screen")
    p.add_argument("--r3-weight", choices=("uniform", "certainty", "regime", "composite"), default="uniform",
                   help="predeclared training-only weighting arm for R3")
    p.add_argument("--train-end",default="2024-03-01");p.add_argument("--calibration-end",default="2024-05-01");p.add_argument("--eval-end",default="2024-12-01")
    a=p.parse_args()
    if a.out.exists(): raise FileExistsError(a.out)
    _assert_winner_contract(a.winner)
    train_end,cal_end,eval_end=(pd.Timestamp(v,tz="UTC") for v in (a.train_end,a.calibration_end,a.eval_end))
    feats=_features(a.features); arms=("R0_B3","R1_direct_net","robust_clear_soft_b0_t50","robust_clear_soft_b25_t50","robust_clear_soft_b50_t50","R2_event_b25","R3_economic_simplex_b25","R3_retained_clear_b25_n50","R4_ordinal","R4_ordinal_cumulative")
    if a.arms:
        arms=tuple(item.strip() for item in a.arms.split(",") if item.strip())
        unknown=set(arms)-{"R0_B3","R1_direct_net","robust_clear_soft_b0_t50","robust_clear_soft_b25_t50","robust_clear_soft_b50_t50","R2_event_b25","R3_economic_simplex_b25","R3_retained_clear_b25_n50","R4_ordinal","R4_ordinal_cumulative"}
        if unknown: raise ValueError(f"unknown arms: {sorted(unknown)}")
    outputs=[];lineage=[]
    for side in ((a.side,) if a.side else SIDES):
        x=_read_side(a.panel,a.winner,a.robust,side,feats[side])
        train=x[x.__label_available_at__.lt(train_end)]; cal=x[(x.__ts__.ge(train_end))&(x.__ts__.lt(cal_end))]; test=x[(x.__ts__.ge(cal_end))&(x.__ts__.lt(eval_end))]
        if min(map(len,(train,cal,test)))<10000: raise ValueError(f"insufficient {side} support")
        for n,arm in enumerate(arms):
            extras: dict[str, np.ndarray] = {}
            if arm == "R2_event_b25":
                # This is a deliberately narrow ablation of the central R2
                # question.  It differs from the soft regressor only in loss:
                # predict a cost+25bps robust clear event directly.
                m=lgb.LGBMClassifier(objective="binary",**{**PARAMS,"random_state":20260802+n}).fit(_matrix(train,feats[side]),train.robust_clear_event_b25.to_numpy(int))
                pcal=m.predict_proba(_matrix(cal,feats[side]))[:,1]; ptest=m.predict_proba(_matrix(test,feats[side]))[:,1]
                extras = {"prob_clear": ptest, "target_event": test.robust_clear_event_b25.to_numpy(int)}
            elif arm.startswith("R3_"):
                # P(clear) - P(adverse) is deliberately only a raw economic
                # ordering.  The subsequent calibration-period map supplies
                # a common bps scale before global cross-side ranking.
                sample_weight = _r3_weights(train, arm, a.r3_weight)
                m=lgb.LGBMClassifier(objective="multiclass",num_class=3,**{**PARAMS,"random_state":20260802+n}).fit(_matrix(train,feats[side]),train[arm].to_numpy(int), sample_weight=sample_weight)
                pcal_raw=m.predict_proba(_matrix(cal,feats[side])); ptest_raw=m.predict_proba(_matrix(test,feats[side]))
                pcal=pcal_raw[:,2]-pcal_raw[:,0]; ptest=ptest_raw[:,2]-ptest_raw[:,0]
                extras = {"prob_adverse": ptest_raw[:, 0], "prob_weak": ptest_raw[:, 1],
                          "prob_clear": ptest_raw[:, 2], "target_class": test[arm].to_numpy(int)}
            elif arm == "R4_ordinal_cumulative":
                # Three threshold models implement a cumulative ordinal loss:
                # severe loss / loss / marginal clear / robust clear.  This
                # preserves ordering without asking Huber regression to place
                # arbitrary distances between the four economic states.
                train_net = train.t4_tp6_sl4_net_bps.to_numpy(float)
                cal_scores = []; test_scores = []
                for threshold in (-200., 0., 50.):
                    model = lgb.LGBMClassifier(objective="binary", **{**PARAMS, "random_state": 20260802 + n + int(threshold + 200)}).fit(
                        _matrix(train, feats[side]), (train_net > threshold).astype(int)
                    )
                    cal_scores.append(model.predict_proba(_matrix(cal, feats[side]))[:, 1])
                    test_scores.append(model.predict_proba(_matrix(test, feats[side]))[:, 1])
                pcal = np.mean(cal_scores, axis=0); ptest = np.mean(test_scores, axis=0)
                extras = {"ordinal_expected_rank": ptest}
            else:
                m=lgb.LGBMRegressor(objective="huber",alpha=.9,**{**PARAMS,"random_state":20260802+n}).fit(_matrix(train,feats[side]),train[arm].to_numpy(float))
                pcal=m.predict(_matrix(cal,feats[side])); ptest=m.predict(_matrix(test,feats[side]))
            score=_map_bps(pcal,cal.t4_tp6_sl4_net_bps.to_numpy(float),ptest)
            out=test[["candidate_id","__ts__","side_name","t4_tp6_sl4_gross_bps","t4_tp6_sl4_net_bps"]].copy();out.columns=["candidate_id","__ts__","side_name","gross_bps","net_bps"]
            target_name = f"{arm}_w{a.r3_weight}" if arm.startswith("R3_") else arm
            out["target"]=target_name;out["raw_prediction"]=ptest;out["score_bps"]=score
            for key, values in extras.items():
                out[key] = values
            outputs.append(out)
            lineage.append({"side":side,"target":target_name,"features":feats[side],"train_rows":len(train),"calibration_rows":len(cal),"evaluation_rows":len(test),"train_labels_available_before":str(train_end),"calibration_before_evaluation":True,"r3_weight":a.r3_weight if arm.startswith("R3_") else None,"weights_fit_on_training_rows_only":arm.startswith("R3_")})
        # The long and short frames each contain hundreds of thousands of
        # 36-field rows.  Release one side before reading the next instead of
        # allowing pandas/LightGBM buffers to overlap at peak memory.
        del x, train, cal, test
        gc.collect()
    pred=pd.concat(outputs,ignore_index=True);rows=[]
    for target,g in pred.groupby("target",observed=True):
        for frac in (.01,.05,.10,.20):
            z=g.sort_values(["score_bps","candidate_id"],ascending=[False,True],kind="mergesort").head(int(np.ceil(len(g)*frac)))
            rows.append({"target":target,"top_fraction":frac,"n":len(z),"gross_bps":z.gross_bps.mean(),"net_bps":z.net_bps.mean(),"long_n":int(z.side_name.eq("long").sum()),"short_n":int(z.side_name.eq("short").sum())})
    a.out.mkdir();pred.to_parquet(a.out/"target_repair_oof_predictions.parquet",index=False);pd.DataFrame(rows).to_parquet(a.out/"target_repair_results.parquet",index=False)
    manifest={"schema":"tp6_sl4_target_repair_causal_screen_v1","status":"COMPLETED","contract":{"features":"frozen side-local 36-feature base subsets","target_rows":"complete exact H12 only","mapping":"side-local calibration-period score->net-bps bins, then global ranking","top_k":"global across sides/assets/timestamps"},"windows":{"train_end":str(train_end),"calibration_end":str(cal_end),"evaluation_end":str(eval_end)},"lineage":lineage,"metrics":rows}
    (a.out/"run_manifest.json").write_text(json.dumps(manifest,indent=2,default=lambda value: value.item() if hasattr(value,"item") else str(value)));print(pd.DataFrame(rows).to_string(index=False))
if __name__=="__main__": main()
