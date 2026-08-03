#!/usr/bin/env python3
"""Round D0--D4 shadow model-validity funnel for the fixed shared C2 expert.

This deliberately runs a *single* residual expert.  The D inputs are soft
decision-time/prequential context only: no target values are model features,
there is no rejection gate, and no per-regime/local model is fitted.

D0: fixed C2 control.
D1: relationship-break residuals: how abnormal is the current relation
    between the base simplex and the causal market context, relative to the
    preceding 28 calendar days?
D2: score/contribution-distribution OOD: robust novelty of base probabilities,
    causal base bps and the compact context distribution relative to the same
    preceding 28 calendar days.
D3: an active failure probability.  This is a weekly logistic model trained
    only on *previously resolved* weekly blocks.  Its outcome is whether the
    week-global top decile under the causal base expected-bps map failed to
    clear zero net bps.  Its input is the preceding four complete weeks'
    decision-time score/context distribution; candidate-week outcomes are
    never inputs.
D4: D1 + D2 + D3.

All arms retain C2's invariant core, sealed soft-state surface and the 16
predeclared base-value x soft-state interactions.  Strict outer eras and
pooled-global rankings are identical to C0--C3.  D is SHADOW ONLY: it tests
whether a shared expert can use validity context; it does not define a policy
or hard admission rule.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data_perp/artifacts/tp6_m6_shared_regime_residual_features_20260809_v3/shared_regime_residual_ledger.parquet"
DEFAULT_OUT = ROOT / "data_perp/artifacts/tp6_shared_residual_d0_d4_20260809_v1"
ERAS = ("2023-07_08", "2023-09_10", "2023-11_12", "2024-01_02", "2024-05_06", "2024-07_08", "2024-09_10", "2024-11")
BASE = ("p_adverse", "p_weak", "p_clear")
REGIME_PRIOR = "soft_regime_prior_residual_bps"
REGIME_CENTERED_TARGET = "target__soft_regime_centered_residual_bps"
CONTEXT = (
    "mkt_ret_eq_24h", "regime_liquidity_score", "mkt_rv_ratio_1h_24h",
    "mkt_oi_chg_z_24h", "mkt_funding_dispersion", "cross_asset_corr_4h",
    "mkt_systemic_deleveraging_score", "mkt_flush_exhaustion_score",
    "post_liquidation_rebound_score", "negative_breadth_pct",
    "btc_resilience_alt_weakness", "short_covering_score_market",
    "deleveraging_without_followthrough", "short_signal_recovery_conflict",
)
INVARIANT = ("side_is_long", *BASE, "prequential_base_expected_net_bps", *CONTEXT)
SOFT_STATE = (
    "regime_p_calm", "regime_p_trend", "regime_p_stress", "regime_p_transition",
    "regime_entropy", "regime_transition_onset_proxy", "regime_state_duration_hours",
)
INTERACTION_SOURCE = ("p_adverse", "p_weak", "p_clear", "prequential_base_expected_net_bps")
INTERACTION_STATE = ("regime_p_calm", "regime_p_trend", "regime_p_stress", "regime_p_transition")
C2 = (*INVARIANT, *SOFT_STATE, *(f"interaction__{a}__x__{b}" for a in INTERACTION_SOURCE for b in INTERACTION_STATE))
C2_RAW = (*INVARIANT, *SOFT_STATE)

# A deliberately compact, predeclared causal diagnostics contract.  The
# relationship probe predicts base probabilities from the same side/context
# surface.  D2 evaluates raw distribution novelty in a distinct, compact
# vector.  These are inputs, not selected from outcome performance.
REL_CONTEXT = ("side_is_long", *CONTEXT, "regime_entropy", "regime_transition_onset_proxy")
REL_TARGETS = BASE
OOD_VECTOR = (*BASE, "prequential_base_expected_net_bps", "regime_entropy", *CONTEXT)
D1 = ("trust_relationship_break_mean_abs", "trust_relationship_break_max_abs")
D2 = ("trust_score_ood_mean_abs_z", "trust_score_ood_max_abs_z")
D3 = ("trust_active_failure_probability", "trust_active_failure_support_weeks")
ARMS = {
    "D0_c2_control": (),
    "D1_relationship_break": D1,
    "D2_score_distribution_ood": D2,
    "D3_active_failure_probability": D3,
    "D4_compact_combination": (*D1, *D2, *D3),
}
TOPS = (.01, .05, .10)
SEED = 20260809
PARAMS: dict[str, Any] = dict(n_estimators=180, learning_rate=.035, num_leaves=24, min_child_samples=400, colsample_bytree=.80, subsample=.80, reg_lambda=12.)
LOOKBACK_DAYS = 28
HEALTH_LOOKBACK_WEEKS = 4
MIN_HEALTH_TRAIN_WEEKS = 12
LABEL_AVAILABILITY_DELAY = pd.Timedelta(hours=13)  # signal close +1h entry + H12


def fit_model(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(objective="huber", alpha=.9, random_state=SEED, n_jobs=1, verbosity=-1, **PARAMS).fit(x, y, sample_weight=weight)


def ensure_label_available_ts(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "label_available_ts" in out:
        available = pd.to_datetime(out["label_available_ts"], utc=True, errors="raise")
    elif "__label_available_at__" in out:
        available = pd.to_datetime(out["__label_available_at__"], utc=True, errors="raise")
    else:
        available = pd.to_datetime(out["__ts__"], utc=True) + LABEL_AVAILABILITY_DELAY
    if available.isna().any() or (available < pd.to_datetime(out["__ts__"], utc=True) + LABEL_AVAILABILITY_DELAY).any():
        raise ValueError("H12 label availability must be signal-close +13h or later")
    out["label_available_ts"] = available
    return out


def assert_outer_train_resolved(train: pd.DataFrame, test: pd.DataFrame) -> None:
    cutoff = pd.to_datetime(test["__ts__"], utc=True).min()
    latest = pd.to_datetime(train["label_available_ts"], utc=True).max()
    if latest >= cutoff:
        raise ValueError(f"outer training contains unresolved labels: {latest} >= {cutoff}")


def weights(train: pd.DataFrame) -> np.ndarray:
    count = train.groupby("era").size()
    value = train.era.map(np.sqrt(len(train) / (len(count) * count))).to_numpy(float)
    return np.clip(value / value.mean(), .25, 4.)


def add_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for a in INTERACTION_SOURCE:
        for b in INTERACTION_STATE:
            out[f"interaction__{a}__x__{b}"] = out[a].to_numpy(float) * out[b].to_numpy(float)
    return out


def robust_location_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(values, axis=0)
    mad = np.nanmedian(np.abs(values - med), axis=0) * 1.4826
    # A constant feature has no novelty contribution.  This is a defined
    # reference-scale fallback, never zero-imputation of a missing input.
    return med, np.where(np.isfinite(mad) & (mad > 1e-8), mad, np.inf)


def week_start(ts: pd.Series) -> pd.Series:
    return ts.dt.floor("D") - pd.to_timedelta(ts.dt.weekday, unit="D")


def weekly_prequential_trust(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return candidate features and a causality/support audit.

    Every snapshot's reference rows are strictly earlier than its Monday UTC
    start.  The health classifier additionally filters blocks whose H12
    labels have resolved before that start.  Hence current-week economics are
    neither direct nor indirect feature inputs.
    """
    x = frame.sort_values(["__ts__", "candidate_id"], kind="stable").copy()
    x["trust_week_start_utc"] = week_start(x.__ts__)
    out = pd.DataFrame(index=x.index, columns=[*D1, *D2, *D3], dtype=float)
    audits: list[dict[str, Any]] = []
    all_weeks = sorted(x.trust_week_start_utc.drop_duplicates().tolist())
    # Produce all raw relationship / distribution references first.  This
    # sees only earlier raw decision-time values and no outcomes.
    for wk in all_weeks:
        mask = x.trust_week_start_utc.eq(wk)
        hist = x[(x.__ts__ < wk) & (x.__ts__ >= wk - pd.Timedelta(days=LOOKBACK_DAYS))]
        eval_ = x.loc[mask]
        if len(hist) < 500 or len(eval_) == 0:
            out.loc[mask, list(D1)] = 0.
            out.loc[mask, list(D2)] = 0.
            audits.append({"week_start_utc": wk, "rows": int(mask.sum()), "reference_rows": len(hist), "relationship_available": False, "ood_available": False})
            continue
        # Relationship-break: side/context -> simplex relationship is frozen
        # at snapshot; each output is scaled against the prior residual MAD.
        hx, ex = hist.loc[:, REL_CONTEXT].to_numpy(float), eval_.loc[:, REL_CONTEXT].to_numpy(float)
        rel_errors: list[np.ndarray] = []
        for target in REL_TARGETS:
            scaler = RobustScaler(quantile_range=(25, 75)).fit(hx)
            model = Ridge(alpha=4.).fit(scaler.transform(hx), hist[target].to_numpy(float))
            hist_err = hist[target].to_numpy(float) - model.predict(scaler.transform(hx))
            _, scale = robust_location_scale(hist_err.reshape(-1, 1))
            eval_err = np.abs(eval_[target].to_numpy(float) - model.predict(scaler.transform(ex))) / scale[0]
            rel_errors.append(np.clip(eval_err, 0., 25.))
        rel = np.vstack(rel_errors)
        out.loc[mask, "trust_relationship_break_mean_abs"] = rel.mean(axis=0)
        out.loc[mask, "trust_relationship_break_max_abs"] = rel.max(axis=0)
        # Score-distribution OOD: independent robust univariate distance.
        med, scale = robust_location_scale(hist.loc[:, OOD_VECTOR].to_numpy(float))
        z = np.abs((eval_.loc[:, OOD_VECTOR].to_numpy(float) - med) / scale)
        z = np.nan_to_num(z, nan=0., posinf=0., neginf=0.)
        out.loc[mask, "trust_score_ood_mean_abs_z"] = np.clip(z, 0., 25.).mean(axis=1)
        out.loc[mask, "trust_score_ood_max_abs_z"] = np.clip(z, 0., 25.).max(axis=1)
        audits.append({"week_start_utc": wk, "rows": int(mask.sum()), "reference_rows": len(hist), "relationship_available": True, "ood_available": True})

    # Week-level active failure label.  It deliberately measures current
    # reliability of *the causal base value map*, not the candidate's own
    # outcome.  Top 10% is global-within-week, matching the global ranking
    # premise while retaining enough observations for weekly health labels.
    blocks: list[dict[str, Any]] = []
    for wk, q in x.groupby("trust_week_start_utc", sort=True):
        order = q.sort_values(["prequential_base_expected_net_bps", "candidate_id"], ascending=[False, True], kind="stable")
        tail = order.head(max(1, int(np.ceil(.10 * len(order)))))
        blocks.append({"week_start_utc": wk, "block_end_resolved_utc": pd.to_datetime(q.label_available_ts, utc=True).max(),
                       "failure": float(tail.net_bps.mean() <= 0.), "tail_net_bps": float(tail.net_bps.mean()), "rows": len(q)})
    block = pd.DataFrame(blocks).sort_values("week_start_utc", kind="stable")
    # Inputs at block start: the immediately prior *raw*, outcome-free
    # distribution.  These are exactly the D2 components and context state.
    health_cols = [*D2, "regime_entropy", "regime_p_calm", "regime_p_trend", "regime_p_stress", "regime_p_transition"]
    # D2 values were materialised above in ``out``; join them only for the
    # weekly raw-feature aggregation.  They remain derived exclusively from
    # preceding decision-time rows.
    weekly_source = x.join(out.loc[:, D2])
    weekly = weekly_source.groupby("trust_week_start_utc", sort=True)[health_cols].mean().reset_index().rename(columns={"trust_week_start_utc": "week_start_utc"})
    block = block.merge(weekly, on="week_start_utc", how="left", validate="one_to_one")
    # Replace a week's own aggregates with preceding-four-week aggregates:
    # no same-week candidate (or end-of-week score distribution) leaks in.
    prior = block.loc[:, health_cols].rolling(HEALTH_LOOKBACK_WEEKS, min_periods=HEALTH_LOOKBACK_WEEKS).mean().shift(1)
    block.loc[:, health_cols] = prior
    out.loc[:, "trust_active_failure_probability"] = .5
    out.loc[:, "trust_active_failure_support_weeks"] = 0.
    for row in block.itertuples(index=False):
        wk = row.week_start_utc
        usable = block[(block.block_end_resolved_utc < wk) & block[health_cols].notna().all(axis=1)]
        if len(usable) and not pd.to_datetime(usable["block_end_resolved_utc"], utc=True).max() < wk:
            raise ValueError("D3 prior fit includes a label not available before the weekly cutoff")
        mask = x.trust_week_start_utc.eq(wk)
        if len(usable) < MIN_HEALTH_TRAIN_WEEKS or not np.isfinite(np.asarray([getattr(row, c) for c in health_cols], float)).all() or usable.failure.nunique() < 2:
            continue
        # Mildly regularised weekly logistic health model.  The classifier is
        # re-fit for every snapshot from prior-resolved blocks only.
        scaler = RobustScaler(quantile_range=(20, 80)).fit(usable.loc[:, health_cols])
        health = LogisticRegression(C=.25, class_weight="balanced", max_iter=500, random_state=SEED)
        health.fit(scaler.transform(usable.loc[:, health_cols]), usable.failure.to_numpy(int))
        v = np.asarray([[getattr(row, c) for c in health_cols]], dtype=float)
        out.loc[mask, "trust_active_failure_probability"] = float(health.predict_proba(scaler.transform(v))[0, 1])
        out.loc[mask, "trust_active_failure_support_weeks"] = float(len(usable))
    audit = pd.DataFrame(audits).merge(block[["week_start_utc", "failure", "tail_net_bps", "rows", "block_end_resolved_utc"]], on="week_start_utc", how="left", validate="one_to_one")
    audit["health_feature_source"] = "preceding_four_complete_weeks_raw_decision_time_features"
    audit["health_training_rule"] = "previously_resolved_weekly_global_top10_base_map_failure"
    audit["label_availability_contract"] = "exact label_available_ts; signal-close +1h entry + H12 (=13h fallback)"
    return out, audit


def matrix(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    value = frame.loc[:, columns].to_numpy(np.float32)
    if not np.isfinite(value).all():
        bad = [c for c in columns if not np.isfinite(frame[c].to_numpy(float)).all()]
        raise ValueError(f"nonfinite D feature(s): {bad}")
    return value


def metric_rows(frame: pd.DataFrame, score: np.ndarray, common: dict[str, Any], period: str, scope: str) -> list[dict[str, Any]]:
    z = frame.copy(); z["score_bps"] = score
    rows: list[dict[str, Any]] = []
    for view, q in (("global", z), ("long", z[z.side_name.eq("long")]), ("short", z[z.side_name.eq("short")])):
        ic = spearmanr(q.score_bps, q.net_bps).statistic if len(q) > 1 else np.nan
        for top in TOPS:
            take = q.sort_values(["score_bps", "candidate_id"], ascending=[False, True], kind="stable").head(max(1, int(np.ceil(len(q) * top))))
            rows.append({**common, "scope": scope, "period": period, "view": view, "top_fraction": top, "n": len(take), "net_bps": float(take.net_bps.mean()), "gross_bps": float(take.gross_bps.mean()), "all_rows_net_bps": float(q.net_bps.mean()), "score_net_spearman": float(ic), "selected_long_fraction": float(take.side_name.eq("long").mean()), "positive_net_fraction": float(take.net_bps.gt(0).mean()), "clear_50_fraction": float(take.net_bps.gt(50).mean())})
    return rows


def robust(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {"mean_top1_net_bps": float(a.mean()), "median_top1_net_bps": float(np.median(a)), "worst_top1_net_bps": float(a.min()), "std_top1_net_bps": float(a.std(ddof=0)), "positive_eras": int((a > 0).sum()), "catastrophic_eras": int((a < -100.).sum())}


def validate(x: pd.DataFrame) -> None:
    # Interaction columns are deliberately generated locally immediately after
    # the raw frozen contract has been validated.
    required = {"candidate_id", "__ts__", "label_available_ts", "side_name", "era", "net_bps", "gross_bps", "shared_regime_contract_complete", REGIME_PRIOR, REGIME_CENTERED_TARGET, *C2_RAW}
    if missing := sorted(required - set(x.columns)):
        raise ValueError(f"missing shared C2 contract: {missing}")
    if x.candidate_id.duplicated().any(): raise ValueError("duplicate candidates")
    if not np.allclose(x.gross_bps.to_numpy(float) - x.net_bps.to_numpy(float), 100., atol=.02): raise ValueError("fixed 100-bps cost contract failed")
    if not set(x.era.unique()).issubset(ERAS): raise ValueError("unexpected eras")
    p = x[["regime_p_calm", "regime_p_trend", "regime_p_stress", "regime_p_transition"]]
    if not np.allclose(p.sum(axis=1), 1., atol=1e-5): raise ValueError("soft-regime simplex failed")


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    x = pd.read_parquet(INPUT)
    x = x[x.shared_regime_contract_complete.astype(bool)].copy()
    x["side_is_long"] = x.side_name.eq("long").astype(float)
    x["__ts__"] = pd.to_datetime(x["__ts__"], utc=True)
    x = ensure_label_available_ts(x)
    x = x[np.isfinite(x.loc[:, [*C2_RAW, REGIME_PRIOR, REGIME_CENTERED_TARGET]].to_numpy(float)).all(axis=1)].copy()
    validate(x)
    x = add_interactions(x.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True))
    trust, audit = weekly_prequential_trust(x)
    x = x.join(trust)
    if x.loc[:, [*D1, *D2, *D3]].isna().any().any(): raise ValueError("trust materialisation left missing values")
    return x, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--only-arm", choices=tuple(ARMS))
    parser.add_argument("--only-era", choices=ERAS[1:])
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args(); stage = args.out.with_name(args.out.name + "_stage")
    if args.finalize:
        ck = sorted((stage / "checkpoints").glob("*.parquet"))
        if not ck: raise FileNotFoundError("no D checkpoints")
        metrics = pd.concat([pd.read_parquet(p) for p in ck], ignore_index=True).drop_duplicates(["arm", "test_era", "scope", "period", "view", "top_fraction"], keep="last")
        expected = {(arm, era) for arm in ARMS for era in ERAS[1:]}
        got = set(metrics.loc[metrics.scope.eq("era"), ["arm", "test_era"]].itertuples(index=False, name=None))
        if missing := sorted(expected - got): raise ValueError(f"missing strict D cells: {missing}")
        outer = metrics[(metrics.scope.eq("era")) & (metrics.view.eq("global")) & (metrics.top_fraction.eq(.01))]
        summary = pd.DataFrame([{"arm": arm, **robust(q.net_bps.tolist())} for arm,q in outer.groupby("arm", sort=True)]).sort_values(["worst_top1_net_bps", "mean_top1_net_bps", "std_top1_net_bps"], ascending=[False,False,True], kind="stable").reset_index(drop=True)
        pooled: list[dict[str, Any]] = []
        for arm in ARMS:
            ps = sorted((stage / "predictions").glob(f"*_{arm}.parquet")); pred = pd.concat([pd.read_parquet(p) for p in ps], ignore_index=True)
            pooled.extend(metric_rows(pred, pred.score_bps.to_numpy(float), {"arm": arm, "test_era": "ALL_OUTER", "train_through": "rolling", "model_name": "fixed_C2_shared_huber", "weight_profile": "sqrt_era"}, "ALL_OUTER", "pooled_global"))
        audits = sorted((stage / "trust_audit").glob("*.parquet"))
        args.out.mkdir(parents=True, exist_ok=True)
        metrics.to_parquet(args.out / "metrics.parquet", index=False); summary.to_parquet(args.out / "summary.parquet", index=False); pd.DataFrame(pooled).to_parquet(args.out / "pooled_global_metrics.parquet", index=False)
        if audits: pd.concat([pd.read_parquet(p) for p in audits], ignore_index=True).drop_duplicates("week_start_utc", keep="last").to_parquet(args.out / "trust_prequential_audit.parquet", index=False)
        report = ["# D0–D4 shared-expert validity/OOD shadow funnel", "", "Fixed C2 shared Huber residual expert, square-root era weights, TP6/SL4/H12 and 100-bps cost. Soft validity context only: no gate, no policy change, no local/per-regime expert.", "", "## Lexicographic selection: worst held-out-era global top-1%, then mean", "", "| rank | arm | worst net bps | mean net bps | median | std | positive eras | catastrophic eras |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
        for i,r in enumerate(summary.itertuples(index=False),1): report.append(f"| {i} | {r.arm} | {r.worst_top1_net_bps:.3f} | {r.mean_top1_net_bps:.3f} | {r.median_top1_net_bps:.3f} | {r.std_top1_net_bps:.3f} | {r.positive_eras} | {r.catastrophic_eras} |")
        report += ["", "D1 relationship-break references use raw decision-time rows strictly before each Monday UTC snapshot. D2 uses the same prior raw reference distribution. D3 weekly-health outcomes resolve no earlier than week-end + H12; its predictors use only the preceding four complete weeks and its logistic fit uses prior-resolved blocks. Warm-up uses the neutral 0.5 probability and a support count, never a fabricated health signal."]
        (args.out / "REPORT.md").write_text("\n".join(report)+"\n")
        manifest = {"schema":"tp6_shared_residual_d0_d4_v1","status":"COMPLETED_SHADOW_DIAGNOSTIC_NO_PROMOTION","input":str(INPUT),"contract":{"base_recipe":"fixed shared regime-centered Huber residual","target":REGIME_CENTERED_TARGET,"reconstruction":"prequential_base_expected_net_bps + soft_regime_prior_residual_bps + predicted_candidate_residual_bps","geometry":"TP6/SL4/H12","cost_bps":100.,"ranking":"global top-k inside held-out era and pooled outer ledger","no_hard_gate":True,"no_local_or_regime_experts":True,"trust_lineage":"weekly decision-time/prequential"},"arms":{k:list(v) for k,v in ARMS.items()},"features":{"C2":list(C2),"D1":list(D1),"D2":list(D2),"D3":list(D3)},"selection":"lexicographic worst era, then mean, then lower dispersion","winner":summary.iloc[0].to_dict()}
        (args.out / "manifest.json").write_text(json.dumps(manifest,indent=2,default=str)+"\n"); print(json.dumps(summary.to_dict(orient="records"),indent=2)); return
    if not args.only_era: raise ValueError("use --only-era for resumable strict outer cell, then --finalize")
    x,audit = load(); index=ERAS.index(args.only_era); train=x[x.era.isin(ERAS[:index])].copy(); test=x[x.era.eq(args.only_era)].copy()
    if train.empty or test.empty: raise ValueError("empty strict split")
    assert_outer_train_resolved(train, test)
    use=(args.only_arm,) if args.only_arm else tuple(ARMS); rows=[]
    for arm in use:
        features=(*C2,*ARMS[arm]); target=train[REGIME_CENTERED_TARGET].to_numpy(float)
        model=fit_model(matrix(train,features),target,weights(train)); candidate_correction=model.predict(matrix(test,features))
        score=test.prequential_base_expected_net_bps.to_numpy(float)+test[REGIME_PRIOR].to_numpy(float)+candidate_correction
        common={"arm":arm,"test_era":args.only_era,"train_through":ERAS[index-1],"train_rows":len(train),"test_rows":len(test),"model_name":"fixed_C2_shared_huber","weight_profile":"sqrt_era","feature_count":len(features)}
        rows.extend(metric_rows(test,score,common,args.only_era,"era"))
        for month,q in test.assign(__month__=test.__ts__.dt.strftime("%Y-%m")).groupby("__month__",sort=True): rows.extend(metric_rows(q,score[q.index.to_numpy()-test.index.min()],common,month,"month"))
        pred=test[["candidate_id","__ts__","label_available_ts","side_name","net_bps","gross_bps","era","prequential_base_expected_net_bps",REGIME_PRIOR,*D1,*D2,*D3]].copy();pred["predicted_candidate_residual_bps"]=candidate_correction;pred["score_bps"]=score;pred["arm"]=arm;pred["feature_count"]=len(features)
        d=stage/"predictions";d.mkdir(parents=True,exist_ok=True);pred.to_parquet(d/f"{args.only_era}_{arm}.parquet",index=False)
    d=stage/"checkpoints";d.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_parquet(d/f"{args.only_era}{'_'+args.only_arm if args.only_arm else ''}.parquet",index=False)
    d=stage/"trust_audit";d.mkdir(parents=True,exist_ok=True);audit.to_parquet(d/f"{args.only_era}.parquet",index=False)
    print(json.dumps({"test_era":args.only_era,"arms":list(use),"train_rows":len(train),"test_rows":len(test),"trust_weeks":len(audit)},indent=2))


if __name__ == "__main__": main()
