#!/usr/bin/env python3
"""Sealed pre-2026 OOF study of explicit residual-model failure/value targets."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data_perp/artifacts"
OUT = ART / "pre2026_oof_model_failure_incremental_value_20260730_v3"
TOP = 0.10
SOURCES = {
    "blocked_oof_panel": ART / "frozen_contextual_score_arms_2023apr_2025jun_20260730_v1/blocked_oof_training_panel.parquet",
    "july_bridge": ART / "july2025_common30_final_base_residual_oof_bridge_20260730_v1/oof_predictions.parquet",
    "augnov_bridge": ART / "augnov2025_common30_frozen_july_base_residual_oos_bridge_20260730_v1/oos_predictions.parquet",
    "dec_bridge": ART / "dec2025_common30_frozen_august_base_residual_oos_bridge_20260730_v1/oos_predictions.parquet",
}
V2 = ART / "pre2026_oof_model_failure_incremental_value_20260730_v2"
REGIME = [
    "regime_change_probability_mean", "regime_change_probability_max", "regime_run_length_mean",
    "regime_run_length_q05", "regime_run_length_entropy", "regime_signal_count",
    "regime_state_age_hours", "regime_is_persistent_24h", "regime_is_persistent_72h",
]
TRANSITION = [
    "transition_lgbm_probability", "transition_lgbm_entropy", "transition_lgbm_margin",
    "transition_bocpd_stable_probability", "transition_bocpd_onset_h1_probability",
    "transition_bocpd_onset_h3_probability", "transition_bocpd_onset_h6_probability",
    "transition_bocpd_onset_h12_probability",
]
TRAJECTORY = ["trajectory_available", "trajectory_transition_probability", "trajectory_probability_entropy", "trajectory_top2_margin"]
CORE = ["base_score", "residual_score", "residual_minus_base"]
ARMS = {
    "regime": CORE + REGIME,
    "transition": CORE + TRANSITION,
    "trajectory": CORE + TRAJECTORY,
    "combined": CORE + REGIME + TRANSITION + TRAJECTORY,
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: object) -> None:
    temp = path.with_name("." + path.name + ".partial")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temp, path)


def era_name(ts: pd.Series) -> pd.Series:
    m = ts.dt.strftime("%Y-%m")
    out = np.select(
        [m.le("2023-12"), m.le("2024-06"), m.le("2024-12"), m.isin(["2025-03", "2025-04"]),
         m.isin(["2025-05", "2025-06"]), m.eq("2025-07"), m.isin(["2025-08", "2025-09"]),
         m.isin(["2025-10", "2025-11"]), m.eq("2025-12")],
        ["2023_apr_dec", "2024_h1", "2024_h2", "2025_marapr", "2025_mayjun", "2025_jul", "2025_augsep", "2025_octnov", "2025_dec"],
        default="incompatible",
    )
    return pd.Series(out, index=ts.index)


def basic(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    x = frame.copy()
    x["__ts__"] = pd.to_datetime(x["__ts__"], utc=True)
    x["execution_label_end_utc"] = pd.to_datetime(x["execution_label_end_utc"], utc=True)
    if "execution_label_available_at" in x:
        x["execution_label_end_utc"] = x["execution_label_end_utc"].combine_first(pd.to_datetime(x["execution_label_available_at"], utc=True))
    x["source"] = source
    return x


def load_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    p = pd.read_parquet(SOURCES["blocked_oof_panel"], columns=["candidate_id", "__ts__", "__symbol__", "side_name", "execution_label_end_utc", "execution_label_available_at", "execution_net_ev_12h", "base_expected_ev", "score_base_expected_ev", "residual_expected_ev", "score_residual_expected_ev"])
    p = basic(p, "blocked_oof_panel")
    p["base_score"] = p["base_expected_ev"].combine_first(p["score_base_expected_ev"])
    p["residual_score"] = p["residual_expected_ev"].combine_first(p["score_residual_expected_ev"])
    p = p[["candidate_id", "__ts__", "__symbol__", "side_name", "execution_label_end_utc", "execution_net_ev_12h", "base_score", "residual_score", "source"]]

    j = basic(pd.read_parquet(SOURCES["july_bridge"], columns=["candidate_id", "__ts__", "__symbol__", "side_name", "execution_label_end_utc", "execution_label_available_at", "execution_net_ev_12h", "base_expected_ev", "residual_expected_ev"]), "july_bridge")
    j = j.rename(columns={"base_expected_ev": "base_score", "residual_expected_ev": "residual_score"})
    j = j[["candidate_id", "__ts__", "__symbol__", "side_name", "execution_label_end_utc", "execution_net_ev_12h", "base_score", "residual_score", "source"]]

    a = basic(pd.read_parquet(SOURCES["augnov_bridge"], columns=["candidate_id", "__ts__", "__symbol__", "side_name", "execution_label_end_utc", "execution_label_available_at", "execution_net_ev_12h", "base_expected_ev", "residual_expected_ev"]), "augnov_bridge")
    a = a.rename(columns={"base_expected_ev": "base_score", "residual_expected_ev": "residual_score"})
    a = a[["candidate_id", "__ts__", "__symbol__", "side_name", "execution_label_end_utc", "execution_net_ev_12h", "base_score", "residual_score", "source"]]

    d = basic(pd.read_parquet(SOURCES["dec_bridge"], columns=["candidate_id", "__ts__", "__symbol__", "side_name", "execution_label_end_utc", "execution_label_available_at", "execution_net_ev_12h", "base_expected_ev", "residual_expected_ev"]), "dec_bridge")
    d = d.rename(columns={"base_expected_ev": "base_score", "residual_expected_ev": "residual_score"})
    d = d[["candidate_id", "__ts__", "__symbol__", "side_name", "execution_label_end_utc", "execution_net_ev_12h", "base_score", "residual_score", "source"]]
    raw = pd.concat([p, j, a, d], ignore_index=True)
    raw = raw[raw.execution_label_end_utc.lt(pd.Timestamp("2026-01-01", tz="UTC"))].copy()
    raw["era"] = era_name(raw.__ts__)
    raw["residual_minus_base"] = raw.residual_score - raw.base_score
    raw = raw[raw.era.ne("incompatible") & raw[["execution_net_ev_12h", "base_score", "residual_score"]].notna().all(axis=1)].copy()
    # Source windows are date-disjoint by contract; avoid a memory-heavy full-string identity sort here.
    windows = raw.groupby("source", as_index=False)["__ts__"].agg(start="min", end="max").sort_values("start", kind="stable").reset_index(drop=True)
    if (windows.end.iloc[:-1].to_numpy() >= windows.start.iloc[1:].to_numpy()).any():
        raise ValueError("overlapping source date windows")
    audit = raw.groupby(["source", "era"], as_index=False).agg(rows=("candidate_id", "size"), start=("__ts__", "min"), end=("__ts__", "max"), label_end_max=("execution_label_end_utc", "max"))
    return raw, audit


def attach_context(x: pd.DataFrame) -> pd.DataFrame:
    s = ART / "authoritative_soft_regime_transition_sidecars_20260730_v1"
    r = pd.read_parquet(s / "soft_regime_hourly.parquet")
    t = pd.read_parquet(s / "soft_transition_hourly.parquet")
    q = pd.read_parquet(ART / "hourly_trajectory_transition_soft_sidecar_20260730_v1/hourly_trajectory_transition_soft_sidecar.parquet")
    for z in (r, t, q):
        z["source_utc"] = pd.to_datetime(z.source_utc, utc=True)
    r = r.rename(columns={
        "bocpd__change_probability_mean": "regime_change_probability_mean", "bocpd__change_probability_max": "regime_change_probability_max",
        "bocpd__run_length_mean": "regime_run_length_mean", "bocpd__run_length_q05": "regime_run_length_q05",
        "bocpd__run_length_entropy": "regime_run_length_entropy", "bocpd__signal_count": "regime_signal_count",
        "bocpd__state_age_hours": "regime_state_age_hours", "bocpd__is_persistent_24h": "regime_is_persistent_24h",
        "bocpd__is_persistent_72h": "regime_is_persistent_72h",
    })
    t = t.rename(columns={
        "lgbm_transition_probability": "transition_lgbm_probability", "lgbm_entropy": "transition_lgbm_entropy", "lgbm_margin": "transition_lgbm_margin",
        "bocpd_stable_vs_transition_probability": "transition_bocpd_stable_probability", "bocpd_onset_h1_probability": "transition_bocpd_onset_h1_probability",
        "bocpd_onset_h3_probability": "transition_bocpd_onset_h3_probability", "bocpd_onset_h6_probability": "transition_bocpd_onset_h6_probability",
        "bocpd_onset_h12_probability": "transition_bocpd_onset_h12_probability",
    })
    q = q.rename(columns={"probability_entropy": "trajectory_probability_entropy", "top2_margin": "trajectory_top2_margin"})
    keep_r = ["source_utc", "bocpd_regime_available"] + REGIME
    keep_t = ["source_utc", "lgbm_transition_available"] + TRANSITION
    keep_q = ["source_utc"] + TRAJECTORY
    c = r[keep_r].merge(t[keep_t], on="source_utc", validate="one_to_one").merge(q[keep_q], on="source_utc", validate="one_to_one")
    z = x.merge(c, left_on="__ts__", right_on="source_utc", how="left", validate="many_to_one").drop(columns="source_utc")
    z["context_complete"] = z[REGIME + TRANSITION + TRAJECTORY].notna().all(axis=1) & z.bocpd_regime_available.fillna(False) & z.lgbm_transition_available.fillna(False) & z.trajectory_available.fillna(False)
    z["features_complete_regime"] = z[CORE + REGIME].notna().all(axis=1) & z.bocpd_regime_available.fillna(False)
    z["features_complete_transition"] = z[CORE + TRANSITION].notna().all(axis=1) & z.lgbm_transition_available.fillna(False)
    z["features_complete_trajectory"] = z[CORE + TRAJECTORY].notna().all(axis=1) & z.trajectory_available.fillna(False)
    z["features_complete_combined"] = z[CORE + REGIME + TRANSITION + TRAJECTORY].notna().all(axis=1) & z.bocpd_regime_available.fillna(False) & z.lgbm_transition_available.fillna(False) & z.trajectory_available.fillna(False)
    return z


def selected_flags(x: pd.DataFrame) -> pd.DataFrame:
    z = x.copy()
    z["base_selected_global_top10"] = False
    z["residual_selected_global_top10"] = False
    for era, g in z.groupby("era", sort=True):
        n = math.ceil(len(g) * TOP)
        b = g.sort_values(["base_score", "candidate_id"], ascending=[False, True], kind="stable").index[:n]
        r = g.sort_values(["residual_score", "candidate_id"], ascending=[False, True], kind="stable").index[:n]
        z.loc[b, "base_selected_global_top10"] = True
        z.loc[r, "residual_selected_global_top10"] = True
    z["incremental_selected_book_utility"] = z.execution_net_ev_12h * (z.residual_selected_global_top10.astype(int) - z.base_selected_global_top10.astype(int))
    z["residual_selected_net_failure"] = (z.residual_selected_global_top10 & z.execution_net_ev_12h.le(0)).astype(int)
    z["residual_selected_net_clearing"] = (z.residual_selected_global_top10 & z.execution_net_ev_12h.gt(0)).astype(int)
    z["residual_selected_false_positive_severity"] = np.where(z.residual_selected_global_top10, (-z.execution_net_ev_12h).clip(lower=0), np.nan)
    return z


def model_predictions(train: pd.DataFrame, test: pd.DataFrame, fs: list[str], target: str, classification: bool) -> tuple[np.ndarray, np.ndarray]:
    if len(train) > 150_000:
        order = pd.util.hash_pandas_object(train.candidate_id, index=False).to_numpy().argsort(kind="stable")[:150_000]
        train = train.iloc[order].copy()
    x, xx = train[fs].astype(np.float32), test[fs].astype(np.float32)
    if classification:
        def make(): return Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=0.02, max_iter=300, class_weight="balanced", random_state=17))])
    else:
        def make(): return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=30.0))])
    pred = []
    for seed in range(3):
        take = (pd.util.hash_pandas_object(train.candidate_id, index=False).to_numpy() % 3) != seed
        use = train.loc[take]
        if classification and use[target].nunique() < 2:
            use = train
        m = make().fit(use[fs], use[target])
        pred.append(m.predict_proba(xx)[:, 1] if classification else m.predict(xx))
    return np.mean(pred, axis=0), np.std(pred, axis=0)


def metric_rows(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, arm, era), g in pred.groupby(["target", "arm", "era"], sort=True):
        for scope, h in [("pooled", g), ("long", g[g.side_name.eq("long")]), ("short", g[g.side_name.eq("short")])]:
            if len(h) < 20:
                continue
            row = {"target": target, "arm": arm, "era": era, "scope": scope, "rows": len(h), "prediction_std_mean": h.prediction_std.mean()}
            if target == "selected_net_failure":
                row["rank_metric"] = roc_auc_score(h.actual_target, h.prediction) if h.actual_target.nunique() == 2 else np.nan
                row["average_precision"] = average_precision_score(h.actual_target, h.prediction) if h.actual_target.nunique() == 2 else np.nan
                row["high_minus_low_net_ev"] = h.loc[h.prediction.ge(h.prediction.quantile(.9)), "execution_net_ev_12h"].mean() - h.loc[h.prediction.le(h.prediction.quantile(.1)), "execution_net_ev_12h"].mean()
            else:
                row["rank_metric"] = h.prediction.corr(h.actual_target, method="spearman")
                row["high_minus_low_target"] = h.loc[h.prediction.ge(h.prediction.quantile(.9)), "actual_target"].mean() - h.loc[h.prediction.le(h.prediction.quantile(.1)), "actual_target"].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def run(output: Path = OUT) -> Path:
    output = Path(output)
    if output.exists(): raise RuntimeError(f"immutable output exists: {output}")
    if not V2.exists() or sha(V2 / "manifest.json") != (V2 / "manifest.sha256").read_text().split()[0]: raise RuntimeError("fail closed: corrected v2 ledger unavailable")
    common = ["candidate_id", "__ts__", "__symbol__", "side_name", "source", "era", "execution_label_end_utc", "execution_net_ev_12h", "base_score", "residual_score", "residual_minus_base", "base_selected_global_top10", "residual_selected_global_top10", "incremental_selected_book_utility", "residual_selected_net_failure", "residual_selected_false_positive_severity", "bocpd_regime_available", "lgbm_transition_available", "trajectory_available"]
    summary = pd.read_parquet(V2 / "materialized_targets.parquet", columns=common)
    summary["__ts__"] = pd.to_datetime(summary.__ts__, utc=True); summary["execution_label_end_utc"] = pd.to_datetime(summary.execution_label_end_utc, utc=True)
    hourly_id = summary.candidate_id.astype(str).str.contains("|1h|", regex=False)
    if summary.execution_label_end_utc.ge(pd.Timestamp("2026-01-01", tz="UTC")).any() or summary.__ts__.dt.minute.ne(0).any() or summary.__ts__.dt.second.ne(0).any() or not hourly_id.all(): raise RuntimeError("fail closed: cadence/label boundary")
    cadence = summary.groupby(["source", "era"], as_index=False).agg(candidate_rows=("candidate_id", "size"), all_candidate_ids_1h=("candidate_id", lambda s: s.astype(str).str.contains("|1h|", regex=False).all()), all_timestamps_hour_aligned=("__ts__", lambda s: s.dt.minute.eq(0).all() and s.dt.second.eq(0).all()), label_end_min=("execution_label_end_utc", "min"), label_end_max=("execution_label_end_utc", "max"))
    coverage = summary.groupby(["source", "era"], as_index=False).agg(rows=("candidate_id", "size"), base_top10_rows=("base_selected_global_top10", "sum"), residual_top10_rows=("residual_selected_global_top10", "sum"), label_end_max=("execution_label_end_utc", "max"))
    for arm in ARMS:
        ok = summary[["base_score", "residual_score", "residual_minus_base"]].notna().all(axis=1)
        if arm in ("regime", "combined"): ok &= summary.bocpd_regime_available.fillna(False)
        if arm in ("transition", "combined"): ok &= summary.lgbm_transition_available.fillna(False)
        if arm in ("trajectory", "combined"): ok &= summary.trajectory_available.fillna(False)
        coverage[f"{arm}_feature_complete_rows"] = summary.assign(_ok=ok).groupby(["source", "era"])._ok.sum().to_numpy()
    econ = [{"source": source, "era": era, "all_candidate_rows": len(g), "base_top10_rows": int(g.base_selected_global_top10.sum()), "residual_top10_rows": int(g.residual_selected_global_top10.sum()), "base_top10_net_ev": g.loc[g.base_selected_global_top10, "execution_net_ev_12h"].mean(), "residual_top10_net_ev": g.loc[g.residual_selected_global_top10, "execution_net_ev_12h"].mean(), "incremental_book_ev": g.incremental_selected_book_utility.sum() / max(1, int(g.residual_selected_global_top10.sum()))} for (source, era), g in summary.groupby(["source", "era"], sort=True)]
    raw_rows = len(summary); del summary
    targets = {"incremental_selected_book_utility": ("incremental_selected_book_utility", False), "selected_net_failure": ("residual_selected_net_failure", True), "top_tail_false_positive_severity": ("residual_selected_false_positive_severity", False)}
    stage = Path(tempfile.mkdtemp(dir=output.parent, prefix="." + output.name + ".")); metric_parts=[]; prediction_rows=0; fold_audit=[]; arm_counts={}
    try:
        for arm, fs in ARMS.items():
            frame = pd.read_parquet(V2 / "materialized_targets.parquet", columns=list(dict.fromkeys(common + fs)))
            avail = pd.Series(True, index=frame.index)
            if arm in ("regime", "combined"): avail &= frame.bocpd_regime_available.fillna(False)
            if arm in ("transition", "combined"): avail &= frame.lgbm_transition_available.fillna(False)
            if arm in ("trajectory", "combined"): avail &= frame.trajectory_available.fillna(False)
            frame = frame.loc[avail & frame[fs].notna().all(axis=1)].copy(); arm_counts[arm] = {"rows": len(frame), "eras": sorted(frame.era.unique().tolist())}
            if frame.era.nunique() < 6: raise RuntimeError(f"fail closed: eras for {arm}")
            for name, (target, classification) in targets.items():
                use = frame.dropna(subset=[target]); use = use if name == "incremental_selected_book_utility" else use[use.residual_selected_global_top10]
                out=[]
                for era, test in use.groupby("era", sort=True):
                    train=use[use.era.ne(era)]
                    for side, te in test.groupby("side_name", sort=True):
                        tr=train[train.side_name.eq(side)]
                        if len(tr)<500 or len(te)<50 or (classification and tr[target].nunique()<2): raise RuntimeError(f"fail closed fold {name}/{arm}/{era}/{side}")
                        prediction,std=model_predictions(tr,te,fs,target,classification)
                        out.append(te[["candidate_id","__ts__","__symbol__","side_name","source","era","execution_net_ev_12h","base_selected_global_top10","residual_selected_global_top10",target]].rename(columns={target:"actual_target"}).assign(target=name,arm=arm,prediction=prediction,prediction_std=std))
                        fold_audit.append({"target":name,"arm":arm,"held_era":era,"side_name":side,"train_rows":len(tr),"test_rows":len(te),"feature_count":len(fs),"features":"|".join(fs),"classification":classification})
                pred=pd.concat(out,ignore_index=True); pred.to_parquet(stage/f"leave_era_oof_{name}_{arm}.parquet",index=False); prediction_rows+=len(pred); metric_parts.append(metric_rows(pred)); del pred,out,use
            del frame
        metrics=pd.concat(metric_parts,ignore_index=True); stability=[]
        for (target,arm),g in metrics[metrics.scope.eq("pooled")].groupby(["target","arm"],sort=True):
            threshold=.52 if target=="selected_net_failure" else .02; valid=g.rank_metric.dropna(); stable=len(valid)>=6 and valid.median()>=threshold and (valid>(.5 if target=="selected_net_failure" else 0)).mean()>=.75 and valid.min()>=(.48 if target=="selected_net_failure" else -.02)
            stability.append({"target":target,"arm":arm,"held_eras":len(valid),"median_rank_metric":valid.median(),"min_rank_metric":valid.min(),"positive_fold_fraction":(valid>(.5 if target=="selected_net_failure" else 0)).mean(),"predeclared_stable_transfer":bool(stable)})
        lineage = pd.read_csv(V2 / "source_lineage_audit.csv")
        contract = {
            "schema": "pre2026_oof_model_failure_incremental_value_v3", "status": "SEALED_PRE2026_OOF_FAILURE_INCREMENTAL_VALUE_NON_PROMOTION", "promotion_eligible": False,
            "decision_cadence": "1h", "exact_replay_bar_cadence": "1m_labels_only", "cadence_rule": "training, leave-era OOF, mappings and candidate decision rows are exactly 1h (candidate ID and UTC timestamp audited); any 1m data is nested label/economics evidence only", "scope": "pre-2026 leave-era-out diagnostic only; no frozen-2026 score or economics application",
            "selection": "one pooled global top10 per fixed pre-2026 era, across timestamps and sides; labels retain this membership before context-completeness filtering",
            "targets": {"incremental_selected_book_utility": "net EV times residual-top10 minus base-top10 membership", "selected_net_failure": "residual global-top10 candidate has net EV <= 0", "top_tail_false_positive_severity": "residual global-top10 max(0, -net EV)"},
            "feature_arms": {k: v for k, v in ARMS.items()}, "learner": "side-local leave-era-out fixed StandardScaler+Ridge(alpha=30) or LogisticRegression(C=.02); deterministic candidate-id hash cap of 150,000 training rows per side/fold when needed, then three deterministic 2/3 subsample fits for prediction uncertainty; no HPO",
            "lineage_rule": "Panel execution_label_available_at is the established complete 13h label-resolution alias when execution_label_end_utc is absent; alternate Mar-Apr causal bridge is excluded because outcomes/residual match but its base map is non-identical, preventing duplicate/mixed base lineage. Feature completeness is arm-local: trajectory may use its complete 2023 context, while regime/transition/combined may not.",
            "stability_gate": "at least six held eras, median metric >= .02 (utility/severity) or .52 (failure AUC), >=75% positive folds, and min >=-.02 or .48; otherwise fail closed",
            "prohibited": ["2026 labels or application", "1m model rows", "state/type/component IDs", "availability imputation", "ex-post target/feature selection"],
        }
        dump(stage / "materialized_targets_reference.json", {"source": str((V2 / "materialized_targets.parquet").resolve()), "sha256": sha(V2 / "materialized_targets.parquet"), "note": "v3 reuses the immutable v2 all-row target materialization and corrects only arm-local feature coverage"})
        lineage.to_csv(stage / "source_lineage_audit.csv", index=False)
        pd.DataFrame(econ).to_csv(stage / "global_top10_economics.csv", index=False)
        coverage.to_csv(stage / "context_coverage.csv", index=False)
        cadence.to_csv(stage / "cadence_provenance_audit.csv", index=False)
        pd.DataFrame(fold_audit).to_csv(stage / "fold_audit.csv", index=False)
        metrics.to_csv(stage / "fold_metrics.csv", index=False)
        pd.DataFrame(stability).to_csv(stage / "stability_summary.csv", index=False)
        dump(stage / "contract.json", contract)
        files = [p for p in stage.iterdir() if p.is_file()]
        manifest = {"schema": contract["schema"], "status": contract["status"], "promotion_eligible": False, "contract": contract, "counts": {"raw_rows": raw_rows, "arm_local_feature_coverage": arm_counts, "prediction_rows": prediction_rows}, "inputs_sha256": {str((V2 / "manifest.json").resolve()): sha(V2 / "manifest.json"), str((V2 / "materialized_targets.parquet").resolve()): sha(V2 / "materialized_targets.parquet")}, "outputs_sha256": {p.name: sha(p) for p in files}}
        dump(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(f"{sha(stage / 'manifest.json')}  manifest.json\n")
        os.replace(stage, output)
        return output
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == "__main__":
    print(run())
