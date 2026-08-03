#!/usr/bin/env python3
"""Train-only recurring transition prototype study with untouched-2026 transfer.

This replaces neither the blocked-OOF transition heads nor their soft sidecars.
It asks a narrower, outcome-free question: do causal *pre-onset* trajectories
form recurring transition types which are distinguishable from matched stable
controls, retain identity under leave-era-out discovery, and transfer to 2026?

Every observation is an hourly anchor.  One-minute execution paths are not
opened.  Destination state is reported as transition topology after an event
is labelled, but is never used in the causal transition-vs-stable classifier.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, average_precision_score, brier_score_loss, roc_auc_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CAT = ROOT / "data_perp/artifacts/transition_pattern_catalogue_20260730_v6"
SIDECAR = ROOT / "data_perp/artifacts/authoritative_soft_regime_transition_sidecars_20260730_v1"
OUT = ROOT / "data_perp/artifacts/trainonly_recurring_transition_prototype_study_20260730_v3"
SPLIT = pd.Timestamp("2026-01-01", tz="UTC")
RANDOM = 729

# Fixed outcome-free observables, grouped by the causal phase their lookback
# summarizes.  These names are not selected against 2026 labels or economics.
BASE_SIGNALS = [
    "breadth_dispersion", "broad_washout_recovery", "btc_resilience_alt_weakness",
    "correlation_breakdown_dispersion", "deleveraged_range_climax_reversal",
    "deleveraging_without_followthrough", "downside_breadth_intensity",
    "funding_confirmed_long_flush", "funding_confirmed_short_covering",
    "funding_deleveraging_divergence", "peer_volatility_decoupling",
    "short_breakout_exhaustion",
]
PHASES = {
    "precondition_168h": (168, ("mean", "delta")),
    "approach_24h": (24, ("mean", "delta")),
    "acceleration_6h": (6, ("slope_per_hour", "delta")),
    "trigger_3h": (3, ("slope_per_hour", "delta")),
}
SIDECAR_FIELDS = [
    "bocpd__change_probability_mean", "bocpd__run_length_mean",
    "bocpd__run_length_entropy", "bocpd__state_age_hours",
    "bocpd__is_persistent_24h", "lgbm_transition_probability",
    "lgbm_entropy", "bocpd_onset_h1_probability", "bocpd_onset_h3_probability",
    "bocpd_onset_h6_probability", "bocpd_onset_h12_probability",
    "bocpd_stable_vs_transition_probability",
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path: Path, value: object) -> None:
    partial = path.with_name("." + path.name + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(partial, path)


def _hourly(x: pd.Series, name: str) -> None:
    if x.duplicated().any() or (x.astype("int64") % pd.Timedelta(hours=1).value != 0).any():
        raise ValueError(f"{name} must be unique hourly timestamps")


def phase_columns(frame: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    rows = []
    fields: list[str] = []
    for phase, (horizon, stats) in PHASES.items():
        for signal in BASE_SIGNALS:
            for stat in stats:
                column = f"sequence__{signal}__{stat}_{horizon}h"
                if column not in frame:
                    raise KeyError(f"required causal sequence field missing: {column}")
                fields.append(column)
                rows.append({"phase": phase, "horizon_hours": horizon, "statistic": stat,
                             "source_signal": signal, "field": column,
                             "causal_at_anchor": True})
    return fields, pd.DataFrame(rows)


def load() -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    seq = pd.read_parquet(CAT / "stable_transition_sequence_inputs.parquet")
    seq["anchor_source_utc"] = pd.to_datetime(seq.anchor_source_utc, utc=True, errors="raise")
    seq["sequence_available_utc"] = pd.to_datetime(seq.sequence_available_utc, utc=True, errors="raise")
    _hourly(seq.anchor_source_utc, "sequence anchor")
    if not seq.sequence_available_utc.le(seq.anchor_source_utc).all():
        raise ValueError("non-causal sequence availability")
    if seq.event_id.duplicated().any() or set(seq.target__stable_vs_transition.unique()) != {0, 1}:
        raise ValueError("invalid event/control identity")
    fields, contract = phase_columns(seq)
    reg = pd.read_parquet(SIDECAR / "soft_regime_hourly.parquet").rename(columns={"source_utc": "anchor_source_utc"})
    trn = pd.read_parquet(SIDECAR / "soft_transition_hourly.parquet").rename(columns={"source_utc": "anchor_source_utc"})
    for item in (reg, trn):
        item["anchor_source_utc"] = pd.to_datetime(item.anchor_source_utc, utc=True, errors="raise")
        _hourly(item.anchor_source_utc, "sidecar")
    reg = reg.loc[:, ["anchor_source_utc", *[f for f in SIDECAR_FIELDS if f in reg]]]
    trn = trn.loc[:, ["anchor_source_utc", *[f for f in SIDECAR_FIELDS if f in trn and f not in reg]]]
    x = seq.merge(reg, on="anchor_source_utc", how="left", validate="one_to_one")
    x = x.merge(trn, on="anchor_source_utc", how="left", validate="one_to_one")
    missing = [f for f in SIDECAR_FIELDS if f not in x]
    if missing:
        raise KeyError(f"authoritative sidecar fields missing: {missing}")
    x["era"] = x.anchor_source_utc.dt.year.astype(int)
    x["topology"] = np.where(x.target__stable_vs_transition.eq(0), "stable",
                               "state_" + x.source_state.astype(str) + "_to_state_" + x.destination_state.astype(str))
    return x, fields, contract


def _transformer(n_components: int = 10) -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     # Event trajectories contain rare but legitimate shock
                     # values.  Median/IQR scaling is fit on training events
                     # only and avoids declaring an arbitrary singleton shock
                     # a reusable type merely because it dominates variance.
                     ("scale", RobustScaler(quantile_range=(10, 90))),
                     ("pca", PCA(n_components=n_components, random_state=RANDOM))])


def fit_cluster(x: pd.DataFrame, fields: list[str], k: int) -> tuple[Pipeline, KMeans, np.ndarray]:
    n_components = min(10, len(fields), max(2, len(x) - 1))
    pipe = _transformer(n_components)
    z = pipe.fit_transform(x[fields].apply(pd.to_numeric, errors="coerce"))
    model = KMeans(n_clusters=k, n_init=50, random_state=RANDOM).fit(z)
    return pipe, model, z


def bootstrap_stability(x: pd.DataFrame, fields: list[str], reference_labels: np.ndarray, k: int, draws: int = 100) -> tuple[float, float, float]:
    rng = np.random.default_rng(RANDOM + k)
    aris = []
    for _ in range(draws):
        ix = rng.integers(0, len(x), len(x))
        sample = x.iloc[ix]
        try:
            pipe, model, _ = fit_cluster(sample, fields, k)
            pred = model.predict(pipe.transform(x[fields].apply(pd.to_numeric, errors="coerce")))
            aris.append(adjusted_rand_score(reference_labels, pred))
        except ValueError:
            continue
    return float(np.mean(aris)), float(np.quantile(aris, .05)), float(np.quantile(aris, .95))


def candidate_clusters(train_events: pd.DataFrame, fields: list[str]) -> tuple[pd.DataFrame, dict[int, tuple[Pipeline, KMeans, np.ndarray]]]:
    rows, fits = [], {}
    for k in (2, 3, 4, 5):
        pipe, model, z = fit_cluster(train_events, fields, k)
        labels = model.labels_
        mean_ari, q05, q95 = bootstrap_stability(train_events, fields, labels, k)
        support = train_events.assign(cluster=labels).groupby("cluster", observed=True).agg(events=("event_id", "size"), eras=("era", "nunique")).reset_index()
        rows.append({"k": k, "train_silhouette": float(silhouette_score(z, labels)),
                     "bootstrap_ari_mean": mean_ari, "bootstrap_ari_q05": q05,
                     "bootstrap_ari_q95": q95, "min_component_events": int(support.events.min()),
                     "min_component_eras": int(support.eras.min()),
                     "stable_candidate": bool(mean_ari >= .60 and q05 >= .35 and support.events.min() >= 12 and support.eras.min() >= 3)})
        fits[k] = (pipe, model, labels)
    return pd.DataFrame(rows), fits


def centroid_alignment(left: np.ndarray, right: np.ndarray) -> tuple[float, list[tuple[int, int, float]]]:
    # Centers live in their own standardized-PCA spaces.  Match their rank
    # descriptors after L2 normalization; this tests coarse identity, not IDs.
    a = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
    b = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
    sim = a @ b.T
    rid, cid = linear_sum_assignment(-sim)
    matches = [(int(i), int(j), float(sim[i, j])) for i, j in zip(rid, cid)]
    return float(np.mean([m[2] for m in matches])), matches


def leave_era_alignment(train_events: pd.DataFrame, fields: list[str], k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = {}
    assignment = []
    for era in sorted(train_events.era.unique()):
        fit = train_events.loc[train_events.era.ne(era)].copy()
        held = train_events.loc[train_events.era.eq(era)].copy()
        pipe, model, _ = fit_cluster(fit, fields, k)
        label = model.predict(pipe.transform(held[fields].apply(pd.to_numeric, errors="coerce")))
        distances = ((pipe.transform(held[fields].apply(pd.to_numeric, errors="coerce")))[:, None, :] - model.cluster_centers_[None, :, :]) ** 2
        # A scalar nearest-center confidence per held event, not one value per
        # component.  Component IDs remain fold-local diagnostic geometry.
        confidence = np.exp(-np.min(distances.sum(axis=2), axis=1))
        assignment.extend({"held_era": int(era), "event_id": e, "assigned_cluster": int(c), "assignment_confidence": float(v)} for e,c,v in zip(held.event_id, label, confidence))
        folds[int(era)] = model.cluster_centers_
    rows=[]
    for left,right in combinations(sorted(folds),2):
        score,matches=centroid_alignment(folds[left],folds[right])
        rows.append({"fold_a":left,"fold_b":right,"mean_matched_centroid_cosine":score,
                     "min_matched_centroid_cosine":min(m[2] for m in matches),"matches":json.dumps(matches)})
    return pd.DataFrame(rows),pd.DataFrame(assignment)


def soft_membership(z: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d2 = ((z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    weights = np.exp(-0.5 * (d2 - d2.min(axis=1, keepdims=True)))
    probs = weights / weights.sum(axis=1, keepdims=True)
    return probs.argmax(axis=1), probs.max(axis=1), np.sqrt(d2.min(axis=1))


def make_classifier(numeric: list[str]) -> Pipeline:
    prep=ColumnTransformer([("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),numeric),
                            ("source",OneHotEncoder(handle_unknown="ignore"),["source_state"])])
    return Pipeline([("prep",prep),("logit",LogisticRegression(C=.10,class_weight="balanced",max_iter=3000,random_state=RANDOM))])


def soft_rows(frame: pd.DataFrame, probability: np.ndarray, arm: str, partition: str, held_era: int | None) -> pd.DataFrame:
    out=frame[["event_id","anchor_source_utc","sequence_available_utc","era","source_state","destination_state","topology","target__stable_vs_transition"]].copy()
    out["arm"]=arm;out["score_partition"]=partition;out["oof_held_era"]=held_era;out["transition_probability"]=probability
    out["probability_entropy"] = -(probability*np.log(np.clip(probability,1e-12,1))+(1-probability)*np.log(np.clip(1-probability,1e-12,1)))
    out["top2_margin"] = np.abs(2*probability-1)
    # Retain a causal current-context availability flag; destination topology
    # remains descriptive and must not be supplied to a causal classifier.
    out["current_context_complete"] = frame[SIDECAR_FIELDS].notna().all(axis=1).to_numpy()
    out["destination_is_descriptive_only"] = True
    return out


def classifier_transfer(train: pd.DataFrame, assess: pd.DataFrame, fields: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # destination_state is deliberately absent: it occurs after the observed
    # state change. source_state/current-sidecar values are permissible context.
    rows=[]; coeff=[]
    arms={"trajectory_only": fields, "trajectory_plus_current_regime_transition": [*fields,*SIDECAR_FIELDS]}
    oof_rows=[]; forward_rows=[]
    for arm, numeric in arms.items():
        model=make_classifier(numeric)
        model.fit(train[numeric+["source_state"]],train.target__stable_vs_transition)
        for part, frame in [("train_2022_2025",train),("assessment_2026",assess)]:
            p=model.predict_proba(frame[numeric+["source_state"]])[:,1]; y=frame.target__stable_vs_transition.to_numpy(int)
            rows.append({"arm":arm,"partition":part,"rows":len(frame),"transition_events":int(y.sum()),"stable_controls":int((1-y).sum()),
                         "roc_auc":float(roc_auc_score(y,p)),"average_precision":float(average_precision_score(y,p)),"brier":float(brier_score_loss(y,p))})
        names=model.named_steps["prep"].get_feature_names_out(); values=model.named_steps["logit"].coef_.ravel()
        coeff.extend({"arm":arm,"feature":str(n),"coefficient":float(c)} for n,c in zip(names,values))
        forward_rows.append(soft_rows(assess,model.predict_proba(assess[numeric+["source_state"]])[:,1],arm,"untouched_2026",None))
        # Each pre-2026 probability is generated by a classifier that excludes
        # its entire calendar era.  This is a candidate-level OOF soft context
        # source, distinct from unstable prototype component numbers.
        for held in sorted(train.era.unique()):
            fit=train.loc[train.era.ne(held)]; test=train.loc[train.era.eq(held)]
            fold=make_classifier(numeric);fold.fit(fit[numeric+["source_state"]],fit.target__stable_vs_transition)
            oof_rows.append(soft_rows(test,fold.predict_proba(test[numeric+["source_state"]])[:,1],arm,"leave_era_out_2022_2025",int(held)))
    return pd.DataFrame(rows),pd.DataFrame(coeff),pd.concat(oof_rows,ignore_index=True),pd.concat(forward_rows,ignore_index=True)


def run(output: Path = OUT) -> Path:
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    x,sequence_fields,phase_contract=load()
    train=x.loc[x.anchor_source_utc.lt(SPLIT)].copy(); assess=x.loc[x.anchor_source_utc.ge(SPLIT)].copy()
    if train.empty or assess.empty or train.era.max()!=2025 or assess.era.min()!=2026: raise ValueError("strict 2022-2025/2026 split failed")
    transition_train=train.loc[train.target__stable_vs_transition.eq(1)].copy()
    transition_assess=assess.loc[assess.target__stable_vs_transition.eq(1)].copy()
    # Cluster features include causal trajectories plus observable current
    # sidecar context.  The latter is imputed only inside the train fit.
    prototype_fields=[*sequence_fields,*SIDECAR_FIELDS]
    candidates,fits=candidate_clusters(transition_train,prototype_fields)
    approved=candidates.loc[candidates.stable_candidate]
    selected_k=int(approved.sort_values(["bootstrap_ari_mean","train_silhouette"],ascending=False).iloc[0].k) if len(approved) else None
    # Even when no k is stable enough to name a type, evaluate the best
    # train-only geometry for *failure diagnosis*.  These local component
    # numbers are never types, features, or policy inputs; they make the
    # leave-era and 2026 non-transfer evidence auditable rather than absent.
    diagnostic_k=int(candidates.sort_values(["bootstrap_ari_mean","train_silhouette"],ascending=False).iloc[0].k)
    stage=Path(tempfile.mkdtemp(dir=output.parent,prefix="."+output.name+"."))
    try:
        candidates.to_csv(stage/"prototype_candidate_stability.csv",index=False)
        phase_contract.to_csv(stage/"phase_horizon_contract.csv",index=False)
        top_train=train.groupby(["era","target__stable_vs_transition","topology"],observed=True).size().rename("anchors").reset_index()
        top_assess=assess.groupby(["era","target__stable_vs_transition","topology"],observed=True).size().rename("anchors").reset_index()
        pd.concat([top_train.assign(partition="train_2022_2025"),top_assess.assign(partition="assessment_2026")]).to_csv(stage/"current_destination_topology_support.csv",index=False)
        transfer,coeff,soft_oof,soft_forward=classifier_transfer(train,assess,sequence_fields)
        transfer.to_csv(stage/"transition_vs_stable_transfer.csv",index=False); coeff.to_csv(stage/"transition_vs_stable_coefficients.csv",index=False)
        soft_oof.to_parquet(stage/"pre2026_oof_transition_vs_stable_soft_probabilities.parquet",index=False)
        soft_forward.to_parquet(stage/"assessment_2026_transition_vs_stable_soft_probabilities.parquet",index=False)
        effective_k = selected_k if selected_k is not None else diagnostic_k
        pipe,model,labels=fits[effective_k]
        ztrain=pipe.transform(transition_train[prototype_fields].apply(pd.to_numeric,errors="coerce")); ztest=pipe.transform(transition_assess[prototype_fields].apply(pd.to_numeric,errors="coerce"))
        tr_id,tr_conf,tr_dist=soft_membership(ztrain,model.cluster_centers_); te_id,te_conf,te_dist=soft_membership(ztest,model.cluster_centers_)
        train_membership=transition_train[["event_id","anchor_source_utc","era","source_state","destination_state","topology"]].copy(); train_membership["diagnostic_component"]=tr_id;train_membership["membership_confidence"]=tr_conf;train_membership["prototype_distance"]=tr_dist;train_membership["component_is_promotable_type"]=False
        assess_membership=transition_assess[["event_id","anchor_source_utc","era","source_state","destination_state","topology"]].copy(); assess_membership["diagnostic_component"]=te_id;assess_membership["membership_confidence"]=te_conf;assess_membership["prototype_distance"]=te_dist;assess_membership["component_is_promotable_type"]=False
        train_membership.to_parquet(stage/"train_prototype_membership.parquet",index=False);assess_membership.to_parquet(stage/"assessment_2026_prototype_membership.parquet",index=False)
        align, held=leave_era_alignment(transition_train,prototype_fields,effective_k);align.to_csv(stage/"leave_era_identity_alignment.csv",index=False);held.to_csv(stage/"leave_era_held_assignments.csv",index=False)
        support=train_membership.groupby("diagnostic_component",observed=True).agg(events=("event_id","size"),eras=("era","nunique"),mean_confidence=("membership_confidence","mean")).reset_index();support.to_csv(stage/"prototype_train_support.csv",index=False)
        assess_support=assess_membership.groupby("diagnostic_component",observed=True).agg(events=("event_id","size"),mean_confidence=("membership_confidence","mean"),mean_distance=("prototype_distance","mean")).reset_index();assess_support.to_csv(stage/"prototype_2026_transfer_support.csv",index=False)
        if selected_k is None:
            status="SEALED_NO_STABLE_RECURRING_PROTOTYPES"
            decision=f"No k met the pre-registered training-only recurrence/stability gate; no recurring transition type is named or emitted. k={diagnostic_k} is retained only to expose its leave-era/2026 failure geometry."
        else:
            status="SEALED_TRAINONLY_RECURRING_PROTOTYPES_TRANSFERRED"; decision=f"k={selected_k} satisfies the explicit unsupervised support/stability gate; still diagnostic-only until economic category stability is separately proved."
        cadence=pd.DataFrame([{"table":"sequence_transition_and_stable_anchors","rows":len(x),"non_hourly_rows":0,"cadence":"1h"},{"table":"train_transition_events","rows":len(transition_train),"non_hourly_rows":0,"cadence":"1h"},{"table":"assessment_2026_transition_events","rows":len(transition_assess),"non_hourly_rows":0,"cadence":"1h"}]);cadence.to_csv(stage/"cadence_audit.csv",index=False)
        contract={"split":"all prototype selection, feature contract, clustering and transition/stable classifier fitting use 2022-2025 only; 2026 only transforms frozen fits","soft_probabilities":"pre2026_oof_transition_vs_stable_soft_probabilities.parquet contains one leave-calendar-era-out hourly probability per anchor; assessment_2026_transition_vs_stable_soft_probabilities.parquet uses the frozen all-2022-2025 classifier. Both carry event/anchor identity, source/destination topology, probability entropy, top2 margin, held era and current-context availability. Prototype component values stay in separate diagnostic tables.","outcomes":"no execution economics, alpha, residual, policy, PnL or 1m path outcome read","cadence":"all anchors/model rows are 1h; 1m remains nested labels elsewhere only","causality":"sequence availability is verified <= anchor; destination state is reported only as post-labelled topology and excluded from causal classifier","separation":"current-regime/source-state and transition-onset sidecar fields are jointly available to prototype discovery, while topology and classifier reports separate causal current-regime from non-causal destination","uncertainty":"prototype membership confidence/distance plus bootstrap and leave-era identity alignment"}
        dump(stage/"contract.json",contract)
        files=[p for p in stage.iterdir() if p.is_file()]
        manifest={"schema":"trainonly_recurring_transition_prototype_study_v3","status":status,"promotion_eligible":False,"decision":decision,"selected_k":selected_k,"diagnostic_k":diagnostic_k,"counts":{"all_hourly_anchors":len(x),"train_2022_2025":len(train),"train_transition_events":len(transition_train),"assessment_2026":len(assess),"assessment_transition_events":len(transition_assess)},"contract":contract,"inputs_sha256":{str((CAT/'stable_transition_sequence_inputs.parquet').resolve()):sha(CAT/'stable_transition_sequence_inputs.parquet'),str((SIDECAR/'manifest.json').resolve()):sha(SIDECAR/'manifest.json')},"outputs_sha256":{p.name:sha(p) for p in files}}
        dump(stage/"manifest.json",manifest);(stage/"manifest.sha256").write_text(f"{sha(stage/'manifest.json')}  manifest.json\n");os.replace(stage,output);return output
    except Exception:
        shutil.rmtree(stage,ignore_errors=True);raise

if __name__=="__main__": print(run())
