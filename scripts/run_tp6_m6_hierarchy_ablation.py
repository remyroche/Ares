#!/usr/bin/env python3
"""Narrow H0--H3 conversion-head hierarchy ablation on compatible TP6 history.

This is deliberately *not* a mixture-of-experts search.  It uses the frozen
TP6/SL4/H12, 100-bps population and pre-existing same-side chronological R3
base OOF predictions.  Each evaluation era is scored by models fitted only on
earlier eras; rankings are global after a common M6 probability/map.

H0  side-local compact M6: four base outputs only.
H1  one shared M6: side flag + four base outputs + fixed 14-field causal pack.
H2  H1 plus only three predeclared base-clear interactions: trend, breadth,
    and data-validity risk.
H3  H1 raw score plus a causal broad-regime (trend x breadth) probability map
    fit on the train eras, with side-regime-bin estimates shrunk to side-bin
    then global-bin priors.  The map is a ranking ablation, not a promotion.

The shadow validity controller is an audit-only causal diagnostic: it reports
data/context completeness and an entropy-derived risk flag but never removes
or re-ranks a candidate.  It must not be interpreted as an admission policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data_perp/artifacts/full_universe_t2_t4_panel_20260801_v3"
OUT = ROOT / "data_perp/artifacts/tp6_m6_hierarchy_20260809_v1"

CONTEXT = [
    "mkt_ret_eq_24h", "regime_liquidity_score", "mkt_rv_ratio_1h_24h", "mkt_oi_chg_z_24h",
    "mkt_funding_dispersion", "cross_asset_corr_4h", "mkt_systemic_deleveraging_score",
    "mkt_flush_exhaustion_score", "post_liquidation_rebound_score", "negative_breadth_pct",
    "btc_resilience_alt_weakness", "short_covering_score_market", "deleveraging_without_followthrough",
    "short_signal_recovery_conflict",
]
BASE = ["p_adverse", "p_weak", "p_clear", "base_raw"]
ERAS = (
    ("2023-07_08", "2023-07-01", "2023-09-01", "oof23_f0"),
    ("2023-09_10", "2023-09-01", "2023-11-01", "oof23_f1"),
    ("2023-11_12", "2023-11-01", "2024-01-01", "oof23_f2"),
    ("2024-01_02", "2024-01-01", "2024-03-01", "oof23_f3"),
    ("2024-05_06", "2024-05-01", "2024-07-01", "ledger24"),
    ("2024-07_08", "2024-07-01", "2024-09-01", "ledger24"),
    ("2024-09_10", "2024-09-01", "2024-11-01", "ledger24"),
    ("2024-11", "2024-11-01", "2024-12-01", "ledger24"),
)
TOPS = (.005, .01, .02, .05, .10)
SEED = 20260809


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _model() -> lgb.LGBMClassifier:
    # Fixed before observing this ablation's test results.  The same model is
    # used in H0--H3 so the comparison isolates hierarchy/context geometry.
    return lgb.LGBMClassifier(objective="binary", n_estimators=180, learning_rate=.04,
        num_leaves=24, min_child_samples=400, colsample_bytree=.8, subsample=.8,
        reg_lambda=12., random_state=SEED, n_jobs=1, verbosity=-1)


def _read_ledger() -> pd.DataFrame:
    p = ROOT / "data_perp/artifacts/tp6_sl4_b10_bw4_base_oof_20260802_v1/base_oof_ledger.parquet"
    x = pd.read_parquet(p).rename(columns={"t4_tp6_sl4_net_bps": "net_bps", "base_expected_net_bps": "base_raw",
        "base_p_lower": "p_adverse", "base_p_timeout": "p_weak", "base_p_upper": "p_clear"})
    x["gross_bps"] = x.net_bps + 100.
    x["source"] = "ledger24"
    if not (pd.to_datetime(x.base_fit_resolved_before, utc=True) <= pd.to_datetime(x.__ts__, utc=True)).all():
        raise ValueError("non-chronological 2024 base OOF lineage")
    return x[["candidate_id", "__ts__", "side_name", "gross_bps", "net_bps", *BASE, "source"]]


def _load_era(start: str, end: str, source: str) -> pd.DataFrame:
    if source.startswith("oof23_f"):
        fold = int(source.rsplit("f", 1)[1]); pieces = []
        for side in ("long", "short"):
            p = ROOT / f"data_perp/artifacts/tp6_r3_r5_{side}_baseoof_fold{fold}_20260802_v1/base_oof_predictions.parquet"
            x = pd.read_parquet(p).rename(columns={"prob_adverse": "p_adverse", "prob_weak": "p_weak", "prob_clear": "p_clear"})
            pieces.append(x[["candidate_id", "__ts__", "side_name", "gross_bps", "net_bps", *BASE]])
        x = pd.concat(pieces, ignore_index=True)
    elif source == "ledger24":
        x = _read_ledger()
    else:
        raise ValueError(source)
    x["__ts__"] = pd.to_datetime(x.__ts__, utc=True)
    x = x[(x.__ts__ >= pd.Timestamp(start, tz="UTC")) & (x.__ts__ < pd.Timestamp(end, tz="UTC"))].copy()
    if not np.allclose(x.gross_bps - x.net_bps, 100., atol=.02):
        raise ValueError("cost must be exactly 100 bps once")
    return x


def _context(ids: set[str]) -> pd.DataFrame:
    cols = ["candidate_id", *CONTEXT]; got = []
    for p in sorted((PANEL / "parts").glob("*.parquet")):
        x = pd.read_parquet(p, columns=cols); x = x[x.candidate_id.isin(ids)]
        if not x.empty: got.append(x)
    out = pd.concat(got, ignore_index=True)
    if out.candidate_id.duplicated().any(): raise ValueError("nonunique context identity")
    return out


def _arr(x: pd.DataFrame, fields: list[str]) -> np.ndarray:
    return x[fields].replace([np.inf, -np.inf], np.nan).fillna(0.).to_numpy(np.float32)


def _make_states(train: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    """Causal broad states; cutpoints exclusively fit on earlier train rows."""
    out = x.copy()
    trend_cut = float(train.mkt_ret_eq_24h.median())
    breadth_cut = float(train.negative_breadth_pct.median())
    out["trend_up"] = (out.mkt_ret_eq_24h >= trend_cut).astype(np.float32)
    out["breadth_stressed"] = (out.negative_breadth_pct >= breadth_cut).astype(np.float32)
    # Availability is known at decision time.  It never introduces labels and
    # the controller below never gates/ranks.  The interaction is predeclared.
    full = np.isfinite(out[CONTEXT].replace([np.inf, -np.inf], np.nan).to_numpy(float)).mean(axis=1)
    probs = out[["p_adverse", "p_weak", "p_clear"]].to_numpy(float)
    entropy = -np.sum(np.clip(probs, 1e-6, 1) * np.log(np.clip(probs, 1e-6, 1)), axis=1) / np.log(3.)
    out["validity_risk"] = (1. - full) + .25 * entropy
    out["broad_regime"] = np.where(
        out.trend_up.eq(1),
        np.where(out.breadth_stressed.eq(1), "up_stressed", "up_broad"),
        np.where(out.breadth_stressed.eq(1), "down_stressed", "down_broad"),
    )
    return out


def _features(arm: str) -> list[str]:
    if arm == "H0_compact_side_local": return BASE
    base = ["side_is_long", *BASE, *CONTEXT]
    if arm == "H1_shared_context": return base
    if arm == "H2_predeclared_interactions": return [*base, "clear_x_trend", "clear_x_breadth", "clear_x_validity_risk"]
    if arm == "H3_hierarchical_calibration": return base
    raise ValueError(arm)


def _prepare_features(x: pd.DataFrame, arm: str) -> pd.DataFrame:
    z = x.copy(); z["side_is_long"] = z.side_name.eq("long").astype(np.float32)
    z["clear_x_trend"] = z.p_clear * z.trend_up
    z["clear_x_breadth"] = z.p_clear * z.breadth_stressed
    z["clear_x_validity_risk"] = z.p_clear * z.validity_risk
    return z


def _map_h3(train: pd.DataFrame, test: pd.DataFrame, raw_train: np.ndarray, raw_test: np.ndarray) -> np.ndarray:
    """Side×causal-regime bin map, shrunk group→side→global (training only)."""
    fit = train[["side_name", "broad_regime", "event"]].copy(); fit["raw"] = raw_train
    # fixed deciles avoids choosing bin count on test; duplicates intentionally
    # collapse to a single bin rather than manufacturing resolution.
    edges = np.unique(np.quantile(raw_train, np.linspace(0, 1, 11)))
    if len(edges) < 3: return raw_test
    edges[0], edges[-1] = -np.inf, np.inf
    fit["bin"] = np.clip(np.digitize(fit.raw, edges[1:-1], right=True), 0, len(edges) - 2)
    q = np.clip(np.digitize(raw_test, edges[1:-1], right=True), 0, len(edges) - 2)
    global_map = fit.groupby("bin", as_index=False).event.mean().rename(columns={"event": "global_mean"})
    side_map = fit.groupby(["side_name", "bin"], as_index=False).event.agg(["mean", "count"]).reset_index()
    side_map = side_map.merge(global_map, on="bin", how="left")
    side_map["side_prior"] = (side_map["count"] * side_map["mean"] + 240. * side_map.global_mean) / (side_map["count"] + 240.)
    grp_map = fit.groupby(["side_name", "broad_regime", "bin"], as_index=False).event.agg(["mean", "count"]).reset_index()
    grp_map = grp_map.merge(side_map[["side_name", "bin", "side_prior"]], on=["side_name", "bin"], how="left")
    grp_map["group_score"] = (grp_map["count"] * grp_map["mean"] + 120. * grp_map.side_prior) / (grp_map["count"] + 120.)
    lookup = test[["side_name", "broad_regime"]].copy(); lookup["bin"] = q
    lookup = lookup.merge(grp_map[["side_name", "broad_regime", "bin", "group_score"]], on=["side_name", "broad_regime", "bin"], how="left")
    lookup = lookup.merge(side_map[["side_name", "bin", "side_prior"]], on=["side_name", "bin"], how="left")
    lookup = lookup.merge(global_map, on="bin", how="left")
    return lookup.group_score.fillna(lookup.side_prior).fillna(lookup.global_mean).to_numpy(float)


def _rows(x: pd.DataFrame, score: str, common: dict) -> list[dict]:
    ans=[]
    for view, z in [("global", x), ("long", x[x.side_name.eq("long")]), ("short", x[x.side_name.eq("short")])]:
        if not len(z): continue
        y=z.event.to_numpy(int); s=z[score].to_numpy(float)
        r={**common,"view":view,"metric":"all","top_fraction":np.nan,"n":len(z),
           "net_bps":np.nan,"gross_bps":np.nan,"event_prevalence":float(y.mean()),
           "roc_auc":float(roc_auc_score(y,s)) if y.min()!=y.max() else np.nan,
           "pr_auc":float(average_precision_score(y,s)) if y.min()!=y.max() else np.nan,
           "brier":float(brier_score_loss(y,np.clip(s,1e-6,1-1e-6))),"net_ic":float(spearmanr(s,z.net_bps).statistic)}
        ans.append(r)
        for top in TOPS:
            take=z.sort_values([score,"candidate_id"],ascending=[False,True],kind="mergesort").head(max(1,int(np.ceil(len(z)*top))))
            ans.append({**r,"metric":"top","top_fraction":top,"n":len(take),"net_bps":float(take.net_bps.mean()),"gross_bps":float(take.gross_bps.mean())})
    return ans


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--out",type=Path,default=OUT)
    ap.add_argument("--resume-stage", action="store_true", help="reuse a fully materialised, exact stage after an interrupted scoring run")
    ap.add_argument("--only-test-era", choices=[x[0] for x in ERAS], help="score one bounded held-out era checkpoint")
    ap.add_argument("--finalize", action="store_true", help="assemble completed bounded checkpoints into the final artifact")
    a=ap.parse_args()
    if a.out.exists() and not a.finalize: raise FileExistsError(a.out)
    stage=a.out.with_name(a.out.name+"_stage")
    if stage.exists() and not (a.resume_stage or a.finalize): raise FileExistsError(stage)
    if not stage.exists(): stage.mkdir(parents=True)
    paths={}; coverage_n=np.zeros(len([*BASE,*CONTEXT])); total=0
    for name,start,end,source in ERAS:
        staged = stage / f"{name}.parquet"
        if a.resume_stage:
            if not staged.exists(): raise FileNotFoundError(f"cannot resume: missing {staged}")
            x=pd.read_parquet(staged)
        else:
            x=_load_era(start,end,source); c=_context(set(x.candidate_id)); x=x.merge(c,on="candidate_id",how="inner",validate="one_to_one")
            if len(x)!=len(c): raise ValueError(f"context join lost OOF rows {name}")
            x["event"]=x.net_bps.gt(50).astype(int); x.to_parquet(staged,index=False)
        paths[name]=staged
        coverage_n += np.isfinite(x[[*BASE,*CONTEXT]].replace([np.inf,-np.inf],np.nan).to_numpy(float)).sum(0); total += len(x)
        print(f"joined {name}: {len(x):,}",flush=True)
    coverage=pd.Series(coverage_n/total,index=[*BASE,*CONTEXT])
    if (coverage<.90).any(): raise ValueError(f"coverage<90%: {coverage[coverage<.90].to_dict()}")
    if a.finalize:
        checkpoint = stage / "checkpoints"
        metric_paths=sorted(checkpoint.glob("metrics_*.parquet")); pred_paths=sorted(checkpoint.glob("predictions_*.parquet")); shadow_paths=sorted(checkpoint.glob("shadow_*.parquet"))
        expected={x[0] for x in ERAS[1:]}
        got={p.stem.removeprefix("metrics_") for p in metric_paths}
        if got != expected: raise ValueError(f"cannot finalise missing/extra era checkpoints: expected={expected} got={got}")
        a.out.mkdir(parents=True); mm=pd.concat([pd.read_parquet(p) for p in metric_paths],ignore_index=True); pp=pd.concat([pd.read_parquet(p) for p in pred_paths],ignore_index=True); ss=pd.concat([pd.read_parquet(p) for p in shadow_paths],ignore_index=True)
        mm.to_parquet(a.out/"metrics.parquet",index=False); pp.to_parquet(a.out/"predictions.parquet",index=False); ss.to_parquet(a.out/"shadow_validity_controller_audit.parquet",index=False)
        top=mm[(mm.view.eq("global"))&(mm.metric.eq("top"))&(mm.top_fraction.eq(.01))&(mm.month.isna())]
        summary=top.groupby("arm",as_index=False).agg(eras=("test_era","nunique"),mean_top1_net_bps=("net_bps","mean"),worst_top1_net_bps=("net_bps","min"),best_top1_net_bps=("net_bps","max"),mean_auc=("roc_auc","mean")); summary.to_parquet(a.out/"summary.parquet",index=False)
        lines=["# TP6/SL4 M6 hierarchy H0--H3", "", "All values are strict chronological held-out-era metrics. Ranking is global, not timestamp- or side-local.", "", "| Arm | Eras | Mean top-1% net | Worst era top-1% net | Best era | Mean AUC |", "|---|---:|---:|---:|---:|---:|"]
        for _,r in summary.iterrows(): lines.append(f"| {r.arm} | {r.eras} | {r.mean_top1_net_bps:+.2f} | {r.worst_top1_net_bps:+.2f} | {r.best_top1_net_bps:+.2f} | {r.mean_auc:.3f} |")
        lines += ["", "## Boundary", "", "H3's map is fit only on earlier realised outcomes and uses a side×causal-broad-regime bin map with fixed group→side→global shrinkage. The shadow validity controller is audit-only and never changes ranking, selection, or policy. No arm is promoted from this report; an arm must pass worst-era and causal-admission validation separately."]
        (a.out/"REPORT.md").write_text("\n".join(lines)+"\n")
        manifest={"schema":"tp6_m6_hierarchy_ablation_v1","status":"COMPLETED_DIAGNOSTIC_NO_PROMOTION","geometry":"TP6/SL4/H12","cost_bps":100,"m6_target":"exact net > +50 bps","base_lineage":"pre-existing strict same-side chronological OOF","ranking":"global common-probability / common-H3-map ranking","arms":{"H0":"side-local compact base outputs only","H1":"shared M6 with 14 fixed causal regime/context fields","H2":"H1 plus predeclared p_clear×trend/breadth/validity-risk interactions","H3":"H1 + prior-only hierarchical side×broad-regime calibration"},"coverage":coverage.to_dict(),"shadow_validity_controller":"audit only; not applied to selection","input_sha256":{"script":_sha(Path(__file__)),"ledger":_sha(ROOT/"data_perp/artifacts/tp6_sl4_b10_bw4_base_oof_20260802_v1/base_oof_ledger.parquet")}}
        (a.out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        print(json.dumps({"out":str(a.out),"summary":summary.to_dict("records")},indent=2)); return
    arms=("H0_compact_side_local","H1_shared_context","H2_predeclared_interactions","H3_hierarchical_calibration")
    metrics=[]; preds=[]; shadow=[]; names=list(paths)
    test_indices=[names.index(a.only_test_era)] if a.only_test_era else list(range(1,len(names)))
    for i in test_indices:
        test_name=names[i]
        train=pd.concat([pd.read_parquet(paths[n]) for n in names[:i]],ignore_index=True).sort_values("__ts__",kind="mergesort")
        test=pd.read_parquet(paths[test_name]).copy()
        if not pd.to_datetime(train.__ts__,utc=True).max() < pd.to_datetime(test.__ts__,utc=True).min(): raise ValueError("chronology breach")
        train=_make_states(train,train); test=_make_states(train,test)
        for z in (train,test):
            z["side_is_long"]=z.side_name.eq("long").astype(np.float32)
        common={"train_through":names[i-1],"test_era":test_name,"train_rows":len(train),"test_rows":len(test)}
        # Audit-only, not applied: predeclared data completeness + entropy risk.
        shadow.extend([{**common,"side_name":s,"n":len(z),"full_context_rate":float((z.validity_risk<.25).mean()),"mean_validity_risk":float(z.validity_risk.mean())} for s,z in test.groupby("side_name")])
        for arm in arms:
            tr=_prepare_features(train,arm); te=_prepare_features(test,arm); fields=_features(arm)
            if arm=="H0_compact_side_local":
                raw_train=np.empty(len(tr)); raw_test=np.empty(len(te))
                for side in ("long","short"):
                    im=tr.side_name.eq(side).to_numpy(); jm=te.side_name.eq(side).to_numpy(); m=_model().fit(_arr(tr.loc[im],fields),tr.loc[im,"event"])
                    raw_train[im]=m.predict_proba(_arr(tr.loc[im],fields))[:,1]; raw_test[jm]=m.predict_proba(_arr(te.loc[jm],fields))[:,1]
            else:
                m=_model().fit(_arr(tr,fields),tr.event); raw_train=m.predict_proba(_arr(tr,fields))[:,1]; raw_test=m.predict_proba(_arr(te,fields))[:,1]
            te["score_raw"]=raw_test
            te["score"]= _map_h3(tr,te,raw_train,raw_test) if arm=="H3_hierarchical_calibration" else raw_test
            metrics += _rows(te,"score",{**common,"arm":arm})
            for month,z in te.assign(month=pd.to_datetime(te.__ts__,utc=True).dt.to_period("M").astype(str)).groupby("month",sort=True):
                metrics += _rows(z,"score",{**common,"arm":arm,"month":month})
            preds.append(te[["candidate_id","__ts__","side_name","net_bps","gross_bps","event","broad_regime","validity_risk","score_raw","score"]].assign(**common,arm=arm))
        print(f"scored {test_name}",flush=True)
    if a.only_test_era:
        checkpoint=stage/"checkpoints"; checkpoint.mkdir(exist_ok=True)
        pd.concat(preds,ignore_index=True).to_parquet(checkpoint/f"predictions_{a.only_test_era}.parquet",index=False)
        pd.DataFrame(metrics).to_parquet(checkpoint/f"metrics_{a.only_test_era}.parquet",index=False)
        pd.DataFrame(shadow).to_parquet(checkpoint/f"shadow_{a.only_test_era}.parquet",index=False)
        print(json.dumps({"checkpoint":a.only_test_era,"rows":len(preds[0]),"out":str(checkpoint)},indent=2)); return
    a.out.mkdir(parents=True); pd.concat(preds,ignore_index=True).to_parquet(a.out/"predictions.parquet",index=False)
    mm=pd.DataFrame(metrics); mm.to_parquet(a.out/"metrics.parquet",index=False); pd.DataFrame(shadow).to_parquet(a.out/"shadow_validity_controller_audit.parquet",index=False)
    # summary determines no winner; it exposes average/worst strictly held-out era only.
    top=mm[(mm.view.eq("global"))&(mm.metric.eq("top"))&(mm.top_fraction.eq(.01))&(mm.month.isna())]
    summary=top.groupby("arm",as_index=False).agg(eras=("test_era","nunique"),mean_top1_net_bps=("net_bps","mean"),worst_top1_net_bps=("net_bps","min"),best_top1_net_bps=("net_bps","max"),mean_auc=("roc_auc","mean"))
    summary.to_parquet(a.out/"summary.parquet",index=False)
    lines=["# TP6/SL4 M6 hierarchy H0--H3", "", "All values are strict chronological held-out-era metrics. Ranking is global, not timestamp- or side-local.", "", "| Arm | Eras | Mean top-1% net | Worst era top-1% net | Best era | Mean AUC |", "|---|---:|---:|---:|---:|---:|"]
    for _,r in summary.iterrows(): lines.append(f"| {r.arm} | {r.eras} | {r.mean_top1_net_bps:+.2f} | {r.worst_top1_net_bps:+.2f} | {r.best_top1_net_bps:+.2f} | {r.mean_auc:.3f} |")
    lines += ["", "## Boundary", "", "H3's map is fit only on earlier realised outcomes and uses a side×causal-broad-regime bin map with fixed group→side→global shrinkage. The shadow validity controller is audit-only and never changes ranking, selection, or policy. No arm is promoted from this report; an arm must pass worst-era and causal-admission validation separately."]
    (a.out/"REPORT.md").write_text("\n".join(lines)+"\n")
    manifest={"schema":"tp6_m6_hierarchy_ablation_v1","status":"COMPLETED_DIAGNOSTIC_NO_PROMOTION","geometry":"TP6/SL4/H12","cost_bps":100,"m6_target":"exact net > +50 bps","base_lineage":"pre-existing strict same-side chronological OOF","ranking":"global common-probability / common-H3-map ranking","arms":{"H0":"side-local compact base outputs only","H1":"shared M6 with 14 fixed causal regime/context fields","H2":"H1 plus predeclared p_clear×trend/breadth/validity-risk interactions","H3":"H1 + prior-only hierarchical side×broad-regime calibration"},"coverage":coverage.to_dict(),"shadow_validity_controller":"audit only; not applied to selection","input_sha256":{"script":_sha(Path(__file__)),"ledger":_sha(ROOT/"data_perp/artifacts/tp6_sl4_b10_bw4_base_oof_20260802_v1/base_oof_ledger.parquet")}}
    (a.out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"out":str(a.out),"eras":len(names)-1,"summary":summary.to_dict("records")},indent=2))


if __name__=="__main__": main()
