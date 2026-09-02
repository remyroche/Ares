#!/usr/bin/env python3
"""Legacy aggregate-only control for Strict-R3 MC1 admissions research.

This runner never reads live state and never changes the frozen MC1_d2 bundle.
It uses prior-resolved policy outcomes only, retains the frozen 12-hour policy
label, and evaluates a simple, matched 2-entry/8-concurrent-position auction.
2025 is the development/HPO period; 2026 is reported separately as opened
validation/model-selection evidence, never as untouched evidence.

This script intentionally predates the regenerated individual-ten-head ledger
and remains only a reproducible aggregate-feature control.  It must not be
used to select the current per-head agreement/independence architecture.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "data_perp/artifacts/strict_r3_lockstep_history_long_2024apr_jul2026_strictfull_prior28_optimizedpolicy_20260813_v2/walkforward_scored_label_ledger.parquet"
BASE = ("final_score", "base_rank42", "conditional_consensus_rank", "upstream", "ordinary_shadow_consensus_rank", "correctness_rank")
RANKS = ("base_rank42", "conditional_consensus_rank", "ordinary_shadow_consensus_rank", "correctness_rank")
TARGET_EDGES = (-np.inf, -200.0, -50.0, 50.0, 150.0, 250.0, np.inf)


def utc(value: str | pd.Timestamp) -> pd.Timestamp:
    x = pd.Timestamp(value)
    return x.tz_localize("UTC") if x.tzinfo is None else x.tz_convert("UTC")


def rank_fraction(frame: pd.DataFrame, field: str, fraction: float) -> pd.Series:
    ordered = frame.loc[:, ["__decision_ts__", "candidate_id", field]].sort_values(
        ["__decision_ts__", field, "candidate_id"], ascending=[True, False, True], kind="stable",
    )
    position = ordered.groupby("__decision_ts__", sort=False).cumcount()
    count = ordered.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size")
    keep = position.lt(np.maximum(1, np.ceil(count * fraction).astype(int)))
    return pd.Series(keep.to_numpy(bool), index=ordered.index).reindex(frame.index, fill_value=False)


def add_static_geometry(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    out = frame.copy()
    x = out.loc[:, RANKS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    med = np.nanmedian(x, axis=1)
    out["agr_mean"] = np.nanmean(x, axis=1)
    out["agr_median"] = med
    out["agr_std"] = np.nanstd(x, axis=1)
    out["agr_mad"] = np.nanmedian(np.abs(x - med[:, None]), axis=1)
    out["agr_iqr"] = np.nanquantile(x, .75, axis=1) - np.nanquantile(x, .25, axis=1)
    out["agr_range"] = np.nanmax(x, axis=1) - np.nanmin(x, axis=1)
    out["agr_max_minus_median"] = np.nanmax(x, axis=1) - med
    for threshold in (.90, .95, .97, .98, .99):
        tag = str(threshold).replace(".", "")
        out[f"agr_frac_ge_{tag}"] = np.nanmean(x >= threshold, axis=1)
        out[f"agr_excess_ge_{tag}"] = np.nanmean(np.maximum(x - threshold, 0.0), axis=1)
    for threshold in (.99, .98, .95, .90):
        tag = str(threshold).replace(".", "")
        out[f"agr_tail_{tag}"] = np.nanmean(x >= threshold, axis=1)
    out["agr_second_best"] = np.sort(x, axis=1)[:, -2]
    out["agr_third_best"] = np.sort(x, axis=1)[:, -3]
    out["agr_polarisation"] = np.nanmean(np.abs(x - .5), axis=1)
    out["agr_vote_imbalance"] = np.abs(np.nansum(np.sign(x - .5), axis=1)) / len(RANKS)
    out["agr_upper_mass"] = np.nanmean(x >= .75, axis=1)
    out["agr_lower_mass"] = np.nanmean(x <= .25, axis=1)
    out["agr_near_consensus"] = np.nanmean(np.abs(x - med[:, None]) <= .05, axis=1)
    bins = np.clip((x * 5).astype(int), 0, 4)
    entropy = np.zeros(len(out), float)
    for j in range(5):
        p = np.mean(bins == j, axis=1)
        entropy -= np.where(p > 0, p * np.log(p), 0.0)
    out["agr_rank_entropy"] = entropy
    out["gap_base_conditional"] = out.base_rank42 - out.conditional_consensus_rank
    out["gap_base_ordinary"] = out.base_rank42 - out.ordinary_shadow_consensus_rank
    out["gap_conditional_correctness"] = out.conditional_consensus_rank - out.correctness_rank
    out["gap_best_to_base"] = np.nanmax(x[:, 1:], axis=1) - out.base_rank42.to_numpy(float)
    groups = {
        "base": list(BASE),
        "agreement_level": ["agr_mean", "agr_median", *[f"agr_frac_ge_{str(v).replace('.', '')}" for v in (.90, .95, .97, .98, .99)], *[f"agr_excess_ge_{str(v).replace('.', '')}" for v in (.90, .95, .97, .98, .99)]],
        "agreement_dispersion": ["agr_std", "agr_mad", "agr_iqr", "agr_range", "agr_max_minus_median"],
        "tail_agreement": [*[f"agr_tail_{str(v).replace('.', '')}" for v in (.99, .98, .95, .90)], "agr_second_best", "agr_third_best"],
        "agreement_shape": ["agr_polarisation", "agr_vote_imbalance", "agr_upper_mass", "agr_lower_mass", "agr_near_consensus", "agr_rank_entropy"],
        "base_vs_consensus": ["gap_base_conditional", "gap_base_ordinary", "gap_conditional_correctness", "gap_best_to_base"],
    }
    return out, groups


def add_causal_state(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Timestamp summaries first, then strictly prior 6h rolling summaries."""
    out = frame.copy()
    out["__bucket6h__"] = out.__decision_ts__.dt.floor("6h")
    ts = out.groupby("__bucket6h__", sort=True).agg(
        state_base_mean=("base_score", "mean"), state_consensus_mean=("conditional_consensus_rank", "mean"),
        state_disagreement=("agr_std", "mean"), state_base_gap=("gap_base_conditional", "mean"),
        state_entropy=("agr_rank_entropy", "mean"), state_final_mean=("final_score", "mean"),
    )
    idx = pd.date_range(ts.index.min(), ts.index.max(), freq="6h", tz="UTC")
    ts = ts.reindex(idx)
    cols: list[str] = []
    for source in list(ts.columns):
        prior = ts[source].shift(1)
        for window in ("12h", "1D", "2D", "3D", "7D", "14D", "21D", "28D"):
            name = f"{source}_{window.lower()}_mean"
            ts[name] = prior.rolling(window, min_periods=max(2, int(pd.Timedelta(window) / pd.Timedelta("6h") / 4))).mean()
            cols.append(name)
        for window in ("3D", "7D"):
            name = f"{source}_{window.lower()}_median"
            ts[name] = prior.rolling(window, min_periods=4).median()
            cols.append(name)
        for window in ("7D", "21D"):
            name = f"{source}_{window.lower()}_mad"
            ts[name] = prior.rolling(window, min_periods=8).apply(lambda a: np.median(np.abs(a - np.median(a))), raw=True)
            cols.append(name)
    resolved = out.loc[out.policy_path_valid.fillna(False) & out.policy_net_bps.notna()].copy()
    resolved["__available_bucket__"] = pd.to_datetime(resolved.policy_label_available_ts, utc=True).dt.floor("6h")
    resolved = resolved.groupby("__available_bucket__", sort=True).agg(
        state_realised_ev=("policy_net_bps", "mean"), state_hit_rate=("policy_net_bps", lambda s: float((s > 0).mean())),
    ).reindex(idx)
    for source in resolved.columns:
        prior = resolved[source].shift(1)
        for window in ("3D", "7D", "14D", "28D"):
            name = f"{source}_{window.lower()}_mean"
            resolved[name] = prior.rolling(window, min_periods=4).mean()
            cols.append(name)
    state = pd.concat([ts[cols], resolved[[c for c in cols if c in resolved]]], axis=1)
    state = state.loc[:, ~state.columns.duplicated()]
    out = out.join(state, on="__bucket6h__")
    return out, state.columns.tolist()


def target_values(train: pd.DataFrame, kind: str) -> tuple[np.ndarray, dict[str, float]]:
    y = pd.to_numeric(train.policy_net_bps, errors="coerce").to_numpy(float)
    lo, hi = np.nanquantile(y, [.02, .98])
    if kind.endswith("asin"):
        scale = max(abs(lo), abs(hi), 250.0)
        return np.arcsin(np.clip(np.clip(y, lo, hi) / scale, -.999, .999)), {"lo": lo, "hi": hi, "scale": scale}
    if kind == "ordinal":
        return np.digitize(y, TARGET_EDGES[1:-1], right=True), {"lo": lo, "hi": hi}
    return np.clip(y, lo, hi), {"lo": lo, "hi": hi}


def fit_predict(train: pd.DataFrame, valid: pd.DataFrame, features: list[str], kind: str, params: dict[str, float | int], seed: int) -> np.ndarray:
    med = train.loc[:, features].median(numeric_only=True)
    x = train.loc[:, features].apply(pd.to_numeric, errors="coerce").fillna(med)
    z = valid.loc[:, features].apply(pd.to_numeric, errors="coerce").fillna(med)
    y, transform = target_values(train, kind)
    common = dict(n_estimators=500, learning_rate=float(params["learning_rate"]), num_leaves=int(params["num_leaves"]), max_depth=int(params["max_depth"]), min_child_samples=int(params["min_child_samples"]), feature_fraction=float(params["feature_fraction"]), reg_lambda=float(params["reg_lambda"]), random_state=seed, n_jobs=6, verbosity=-1)
    if kind == "ordinal":
        model = lgb.LGBMClassifier(objective="multiclass", num_class=6, **common).fit(x, y)
        values = np.array([-300., -125., 0., 100., 200., 350.])
        return model.predict_proba(z).dot(values)
    objective = "regression_l1" if kind.startswith("l1") else "huber"
    model = lgb.LGBMRegressor(objective=objective, **common).fit(x, y, callbacks=[lgb.early_stopping(40, verbose=False)])
    pred = model.predict(z)
    return np.sin(pred) * transform["scale"] if kind.endswith("asin") else pred


def portfolio(frame: pd.DataFrame, score: str, admission: str) -> pd.DataFrame:
    """Matched simple 2-entry / 8-concurrent canonical-auction proxy."""
    work = frame.loc[frame[admission].ge(50) & frame.policy_path_valid.fillna(False)].copy()
    work = work.sort_values(["__decision_ts__", score, "candidate_id"], ascending=[True, False, True], kind="stable")
    active: list[pd.Timestamp] = []; accepted: list[bool] = []
    for ts, group in work.groupby("__decision_ts__", sort=True):
        active = [x for x in active if x > ts]
        taken = 0
        for row in group.itertuples():
            exit_ts = ts + pd.Timedelta(minutes=15 * max(1.0, float(getattr(row, "policy_exit_bar_15m", 48))))
            yes = taken < 2 and len(active) < 8
            accepted.append(yes)
            if yes:
                active.append(exit_ts); taken += 1
    work["accepted"] = accepted
    return work.loc[work.accepted].copy()


def metrics(frame: pd.DataFrame, prediction: str, auction: str) -> dict[str, float]:
    selected = portfolio(frame, auction, prediction)
    valid = frame.loc[frame.policy_path_valid.fillna(False) & frame.policy_net_bps.notna()].copy()
    admitted = valid.loc[valid[prediction].ge(50)].copy()
    ic = admitted.groupby("__decision_ts__", sort=False).apply(lambda g: g[prediction].corr(g.policy_net_bps, method="spearman"), include_groups=False).dropna()
    monthly = selected.groupby(selected.__decision_ts__.dt.strftime("%Y-%m")).policy_net_bps.mean()
    weekly = selected.groupby(selected.__decision_ts__.dt.strftime("%G-W%V")).policy_net_bps.mean()
    contested = admitted.groupby("__decision_ts__").filter(lambda g: len(g) > 2)
    picked = portfolio(contested, auction, prediction)
    rejected = contested.loc[~contested.candidate_id.isin(picked.candidate_id)]
    return {"rows": len(frame), "admitted": len(admitted), "accepted": len(selected), "net_ev_bps": float(selected.policy_net_bps.mean()) if len(selected) else np.nan, "net_sum_bps": float(selected.policy_net_bps.sum()), "within_admission_ic": float(ic.mean()) if len(ic) else np.nan, "worst_month_bps": float(monthly.min()) if len(monthly) else np.nan, "worst_week_bps": float(weekly.min()) if len(weekly) else np.nan, "contested_selected_bps": float(picked.policy_net_bps.mean()) if len(picked) else np.nan, "contested_rejected_bps": float(rejected.policy_net_bps.mean()) if len(rejected) else np.nan}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--hpo-trials", type=int, default=8)
    p.add_argument("--max-train-rows", type=int, default=160_000)
    args = p.parse_args()
    if args.out_dir.exists(): raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    columns = ["candidate_id", "__decision_ts__", "__symbol__", "policy_label_available_ts", "policy_path_valid", "policy_net_bps", "policy_exit_bar_15m", *BASE]
    raw = pd.read_parquet(args.ledger, columns=columns)
    raw.__decision_ts__ = pd.to_datetime(raw.__decision_ts__, utc=True); raw.policy_label_available_ts = pd.to_datetime(raw.policy_label_available_ts, utc=True)
    raw, families = add_static_geometry(raw)
    raw, temporal = add_causal_state(raw)
    families["recent_state"] = temporal
    raw["pool_base30"] = rank_fraction(raw, "base_score", .30).to_numpy()
    raw["pool_consensus30"] = rank_fraction(raw, "conditional_consensus_rank", .30).to_numpy()
    raw["pool_union30"] = raw.pool_base30 | raw.pool_consensus30
    candidate_features = [c for cols in families.values() for c in cols]
    # CMI against a strict prequential control residual: score-bin is the condition.
    valid = raw.policy_path_valid.fillna(False) & raw.policy_net_bps.notna()
    research = raw.loc[valid & raw.__decision_ts__.between(utc("2025-01-01"), utc("2026-08-01"), inclusive="left")].copy()
    research["score_bin"] = np.minimum(9, np.floor(research.final_score * 10)).astype(int)
    residual = research.policy_net_bps - research.groupby("score_bin").policy_net_bps.transform("mean")
    mi_rows=[]
    for name in candidate_features:
        x = pd.to_numeric(research[name], errors="coerce"); ok=x.notna() & residual.notna()
        if ok.sum() < 1000: continue
        # Deterministic binned MI; monthly median/MAD form the portability filter.
        bins = pd.qcut(x[ok], min(16, x[ok].nunique()), labels=False, duplicates="drop")
        values=[]
        for _, g in pd.DataFrame({"x":bins, "y":residual[ok], "m":research.loc[ok,"__decision_ts__"].dt.to_period("M")}).groupby("m"):
            if len(g)>100: values.append(mutual_info_regression(g[["x"]], g.y, discrete_features=True, random_state=1729)[0])
        if values: mi_rows.append({"feature":name,"family":next(k for k,v in families.items() if name in v),"mi_median":float(np.median(values)),"mi_mad":float(np.median(np.abs(values-np.median(values)))),"mi_months":len(values)})
    mi=pd.DataFrame(mi_rows).sort_values(["mi_median","mi_mad"],ascending=[False,True]); mi.to_parquet(args.out_dir/"feature_binned_mi.parquet",index=False)
    selected=[]
    for family, group in mi.groupby("family", sort=False): selected += group.sort_values(["mi_median","mi_mad"],ascending=[False,True]).head(3).feature.tolist()
    features=list(dict.fromkeys(list(BASE)+selected))
    folds=[utc(x) for x in ("2025-05-01","2025-08-01","2025-11-01")]
    kinds=("huber_clip","l1_clip","huber_asin","l1_asin","ordinal")
    results=[]; best={}
    for kind in kinds:
        def objective(trial: optuna.Trial) -> float:
            params={"learning_rate":trial.suggest_float("learning_rate",.02,.08),"num_leaves":trial.suggest_int("num_leaves",7,31),"max_depth":trial.suggest_int("max_depth",2,4),"min_child_samples":trial.suggest_int("min_child_samples",100,900),"feature_fraction":trial.suggest_float("feature_fraction",.65,.95),"reg_lambda":trial.suggest_float("reg_lambda",.1,30,log=True)}
            values=[]
            for start in folds:
                train=research.loc[research.policy_label_available_ts.lt(start)].copy(); test=research.loc[research.__decision_ts__.between(start,start+pd.offsets.MonthBegin(1),inclusive="left")].copy()
                if len(train)>args.max_train_rows: train=train.sample(args.max_train_rows,random_state=1729)
                test["pred"]=fit_predict(train,test,features,kind,params,1729); m=metrics(test,"pred","final_score"); values.append(m["net_ev_bps"] - max(0.,-m["worst_week_bps"])*.25)
            return float(np.nanmedian(values))
        study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=1729)); study.optimize(objective,n_trials=args.hpo_trials)
        best[kind]=study.best_params; results += [{"stage":"hpo","kind":kind,"trial":t.number,"value":t.value,**t.params} for t in study.trials]
    hpo=pd.DataFrame(results); hpo.to_parquet(args.out_dir/"hpo_trials.parquet",index=False)
    # Monthly prequential predictions, frozen after 2025 HPO; all arms share these rows.
    prediction_rows=[]
    for start in pd.date_range("2025-01-01","2026-07-01",freq="MS",tz="UTC"):
        stop=start+pd.offsets.MonthBegin(1); train=research.loc[research.policy_label_available_ts.lt(start)].copy(); test=research.loc[research.__decision_ts__.between(start,stop,inclusive="left")].copy()
        if len(train)<5000 or test.empty: continue
        if len(train)>args.max_train_rows: train=train.sample(args.max_train_rows,random_state=1729)
        for kind, params in best.items():
            test[f"pred_{kind}"]=fit_predict(train,test,features,kind,params,1729)
        prediction_rows.append(test)
    pred=pd.concat(prediction_rows,ignore_index=True)
    # Pool, blend and auction ablations; LambdaRank extension intentionally uses the best regression later.
    arms=[]
    for kind in kinds:
        col=f"pred_{kind}"
        for pool in ("pool_base30","pool_consensus30","pool_union30"):
            for auction in ("final_score",col):
                for year in (2025,2026):
                    part=pred.loc[pred.__decision_ts__.dt.year.eq(year)&pred[pool]].copy(); m=metrics(part,col,auction); arms.append({"arm":f"{kind}|{pool}|auction={auction}","year":year,"kind":kind,"pool":pool,"auction":auction,**m})
    pd.DataFrame(arms).to_parquet(args.out_dir/"arm_metrics.parquet",index=False)
    pred.to_parquet(args.out_dir/"prequential_predictions.parquet",index=False,compression="zstd")
    (args.out_dir/"run_manifest.json").write_text(json.dumps({"schema":"strict_r3_mc1_admission_funnel_v1_legacy_aggregate_control","purpose":"offline aggregate-only MC1 admissions control; superseded for per-head selection by strict_r3_ten_head_history_long_2024apr_today_20260816_v2","ledger":str(args.ledger),"features":features,"feature_families":families,"per_head_limitation":"This legacy control deliberately does not consume the individual ten-head ledger and is not eligible to select per-head, independence-weighted, or conditional-usefulness feature families.","train_outcome_rule":"policy_label_available_ts < fold_start","development":"2025 HPO","opened_validation":"2026 monthly prequential replay","policy":"frozen H12 SimplePolicyOptimiser label; costs embedded once","portfolio":"matched 2-entry/8-concurrent proxy","hpo_trials":args.hpo_trials},indent=2)+"\n")

if __name__ == "__main__": main()
