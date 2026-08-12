#!/usr/bin/env python3
"""TP6 frozen-base portability, mapping and admission decomposition.

This is deliberately *not* another learned residual head.  It starts from the
same-side frozen base expected-net score, keeps within-side ranks fixed where a
mapping is tested, and separates deployable prior-only maps from test-label
oracle diagnostics.  Final executable comparisons always rank globally only
after a common-bps reconstruction and allow a no-trade outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data_perp/artifacts/tp6_m6_shared_regime_residual_features_20260809_v3/shared_regime_residual_ledger.parquet"
OUT = ROOT / "data_perp/artifacts/tp6_score_portability_admission_20260803_v1"
ERAS = ("2023-07_08", "2023-09_10", "2023-11_12", "2024-01_02", "2024-05_06", "2024-07_08", "2024-09_10", "2024-11")
TRANSPORTS = (
    ("transport_2023q4_to_2024h1", ERAS[1:3], ERAS[3:5]),
    ("transport_2024h1_to_h2", ERAS[3:5], ERAS[5:]),
)
TOPS = (.01, .05, .10)
THRESHOLDS = (-50., 0., 25., 50., 100.)
LABEL_DELAY = pd.Timedelta(hours=13)
META_FEATURES = (
    "base_raw", "prequential_base_expected_net_bps", "p_adverse", "p_weak", "p_clear",
    "mkt_ret_eq_24h", "regime_liquidity_score", "mkt_rv_ratio_1h_24h",
    "mkt_oi_chg_z_24h", "mkt_funding_dispersion", "cross_asset_corr_4h",
    "mkt_systemic_deleveraging_score", "mkt_flush_exhaustion_score",
    "post_liquidation_rebound_score", "negative_breadth_pct",
    "btc_resilience_alt_weakness", "short_covering_score_market",
    "deleveraging_without_followthrough", "short_signal_recovery_conflict",
)


@dataclass(frozen=True)
class BinMap:
    edges: np.ndarray
    side_values: dict[str, np.ndarray]
    pooled_values: np.ndarray
    side_support: dict[str, np.ndarray]
    pooled_support: np.ndarray
    uncertainty: dict[str, np.ndarray]
    name: str


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read() -> pd.DataFrame:
    columns = [
        "candidate_id", "__ts__", "side_name", "era", "gross_bps", "net_bps",
        "base_raw", "p_adverse", "p_weak", "p_clear",
        "prequential_base_expected_net_bps", "shared_regime_contract_complete",
        "mkt_ret_eq_24h", "regime_liquidity_score", "mkt_rv_ratio_1h_24h",
        "mkt_oi_chg_z_24h", "mkt_funding_dispersion", "cross_asset_corr_4h",
        "mkt_systemic_deleveraging_score", "mkt_flush_exhaustion_score",
        "post_liquidation_rebound_score", "negative_breadth_pct",
        "btc_resilience_alt_weakness", "short_covering_score_market",
        "deleveraging_without_followthrough", "short_signal_recovery_conflict",
    ]
    frame = pd.read_parquet(INPUT, columns=columns)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    numeric = ["gross_bps", "net_bps", "base_raw", "prequential_base_expected_net_bps", "p_adverse", "p_weak", "p_clear"]
    valid = frame["shared_regime_contract_complete"].astype(bool) & np.isfinite(frame[numeric]).all(axis=1)
    frame["base_contract_valid"] = valid
    frame = frame.loc[valid].copy()
    if frame["candidate_id"].duplicated().any():
        raise ValueError("frozen ledger must contain one row per candidate")
    if not np.allclose(frame["gross_bps"] - frame["net_bps"], 100., atol=.02):
        raise ValueError("frozen TP6 ledger must apply the 100-bps cost exactly once")
    if not set(frame["side_name"].unique()).issubset({"long", "short"}):
        raise ValueError("side contract is invalid")
    frame["label_available_ts"] = frame["__ts__"] + LABEL_DELAY
    frame["month"] = frame["__ts__"].dt.to_period("M").astype(str)
    frame["candidate_key"] = pd.util.hash_pandas_object(frame["candidate_id"], index=False).astype("uint64")
    return frame.sort_values(["__ts__", "candidate_key"], kind="stable").reset_index(drop=True)


def _rank_within_side(frame: pd.DataFrame, score: str) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for (_ts, _side), part in frame.groupby(["__ts__", "side_name"], observed=True, sort=False):
        order = part.sort_values([score, "candidate_key"], ascending=[True, True], kind="stable")
        result.loc[order.index] = np.arange(len(order), dtype=float) / max(len(order) - 1, 1)
    return result


def _fit_bin_map(train: pd.DataFrame, *, score: str, bins: int, shrink_rows: float, window_name: str) -> BinMap:
    """Prior-resolved side score bins, shrunk to pooled score-bin economics."""
    source = train[[score, "net_bps", "side_name"]].dropna().copy()
    quantiles = np.linspace(0., 1., bins + 1)
    edges = np.unique(source[score].quantile(quantiles).to_numpy(float))
    if len(edges) < 3:
        lo, hi = source[score].min(), source[score].max()
        edges = np.linspace(lo - 1e-6, hi + 1e-6, bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    index = np.clip(np.searchsorted(edges, source[score].to_numpy(float), side="right") - 1, 0, len(edges) - 2)
    n_bins = len(edges) - 1
    pooled_sum = np.bincount(index, weights=source["net_bps"], minlength=n_bins)
    pooled_n = np.bincount(index, minlength=n_bins).astype(float)
    pooled_global = float(source["net_bps"].mean())
    pooled = (pooled_sum + shrink_rows * pooled_global) / (pooled_n + shrink_rows)
    side_values: dict[str, np.ndarray] = {}
    side_support: dict[str, np.ndarray] = {}
    uncertainty: dict[str, np.ndarray] = {}
    for side in ("long", "short"):
        part = source[source["side_name"].eq(side)]
        ix = np.clip(np.searchsorted(edges, part[score].to_numpy(float), side="right") - 1, 0, n_bins - 1)
        sums = np.bincount(ix, weights=part["net_bps"], minlength=n_bins)
        count = np.bincount(ix, minlength=n_bins).astype(float)
        side_values[side] = (sums + shrink_rows * pooled) / (count + shrink_rows)
        side_support[side] = count
        # conservative uncertainty: shrunk bin standard error plus a 25-bps
        # minimum to avoid confidence from one unusually quiet historical bin.
        std = np.zeros(n_bins, dtype=float)
        for b in range(n_bins):
            values = part.loc[ix == b, "net_bps"].to_numpy(float)
            std[b] = values.std(ddof=1) if len(values) > 1 else 250.
        uncertainty[side] = np.maximum(25., std / np.sqrt(np.maximum(count, 1.)))
    return BinMap(edges, side_values, pooled, side_support, pooled_n, uncertainty, window_name)


def _apply_bin_map(frame: pd.DataFrame, mapping: BinMap, *, score: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_bins = len(mapping.edges) - 1
    ix = np.clip(np.searchsorted(mapping.edges, frame[score].to_numpy(float), side="right") - 1, 0, n_bins - 1)
    value = np.empty(len(frame), dtype=float); uncertainty = np.empty(len(frame), dtype=float); support = np.empty(len(frame), dtype=float)
    for side in ("long", "short"):
        mask = frame["side_name"].eq(side).to_numpy()
        value[mask] = mapping.side_values[side][ix[mask]]
        uncertainty[mask] = mapping.uncertainty[side][ix[mask]]
        support[mask] = mapping.side_support[side][ix[mask]]
    return value, uncertainty, support


def _fit_affine(train: pd.DataFrame, score: str) -> tuple[float, float, dict[str, tuple[float, float]]]:
    """Global affine map plus heavily shrunk side deviations, all prior-only."""
    x = train[score].to_numpy(float); y = train["net_bps"].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    slope = max(float(slope), 0.0)
    result: dict[str, tuple[float, float]] = {}
    for side in ("long", "short"):
        p = train[train["side_name"].eq(side)]
        sx = p[score].to_numpy(float); sy = p["net_bps"].to_numpy(float)
        local_slope, local_intercept = np.polyfit(sx, sy, 1)
        local_slope = max(float(local_slope), 0.0)
        shrink = len(p) / (len(p) + 25_000.)
        result[side] = (intercept + shrink * (local_intercept - intercept), slope + shrink * (local_slope - slope))
    return intercept, slope, result


def _apply_affine(frame: pd.DataFrame, score: str, model: tuple[float, float, dict[str, tuple[float, float]]]) -> np.ndarray:
    _intercept, _slope, sides = model
    output = np.empty(len(frame), dtype=float)
    for side in ("long", "short"):
        mask = frame["side_name"].eq(side).to_numpy(); intercept, slope = sides[side]
        output[mask] = intercept + slope * frame.loc[mask, score].to_numpy(float)
    return output


def _oracle_scores(test: pd.DataFrame, frozen: np.ndarray) -> dict[str, np.ndarray]:
    """Test-label diagnostics only.  These maps never enter deployable rows."""
    y = test["net_bps"].to_numpy(float); x = np.asarray(frozen, dtype=float)
    iso = IsotonicRegression(out_of_bounds="clip").fit(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    return {
        "C0_frozen": x,
        "C1_oracle_monotonic": iso.predict(x),
        "C2_oracle_intercept": x + float((y - x).mean()),
        "C3_oracle_affine": intercept + slope * x,
    }


def _oracle_side_scores(test: pd.DataFrame, frozen: np.ndarray) -> dict[str, np.ndarray]:
    """S1/S2 diagnostic side calibration; test labels are never deployable."""
    output_i = np.asarray(frozen, dtype=float).copy()
    output_a = np.asarray(frozen, dtype=float).copy()
    for side in ("long", "short"):
        mask = test["side_name"].eq(side).to_numpy(); x = output_i[mask]; y = test.loc[mask, "net_bps"].to_numpy(float)
        output_i[mask] = x + float((y - x).mean())
        slope, intercept = np.polyfit(x, y, 1)
        output_a[mask] = intercept + slope * x
    return {"S1_oracle_side_intercept": output_i, "S2_oracle_side_affine": output_a}


def _tercile_meta_score(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Side-specific residual tercile classifier, fitted only on prior labels.

    C0 is base overestimation, C1 is approximately correct, and C2 is base
    underestimation.  Class edges and bps class means are training-only.
    """
    output=np.empty(len(test),dtype=float); uncertainty=np.empty(len(test),dtype=float); support=np.full(len(test),np.inf); audit=[]
    for side in ("long","short"):
        tr=train[train.side_name.eq(side)].copy(); te=test[test.side_name.eq(side)].copy()
        residual=(tr.net_bps-tr.prequential_base_expected_net_bps).to_numpy(float)
        edges=np.quantile(residual,[1/3,2/3]); label=np.digitize(residual,edges,right=True).astype(int)
        median=tr.loc[:,META_FEATURES].replace([np.inf,-np.inf],np.nan).median().fillna(0.)
        xtr=tr.loc[:,META_FEATURES].replace([np.inf,-np.inf],np.nan).fillna(median).to_numpy(np.float32)
        xte=te.loc[:,META_FEATURES].replace([np.inf,-np.inf],np.nan).fillna(median).to_numpy(np.float32)
        counts=np.bincount(label,minlength=3).astype(float); weight=np.sqrt(len(label)/np.maximum(3*counts[label],1.)); weight=np.clip(weight/weight.mean(),.5,2.)
        model=lgb.LGBMClassifier(objective="multiclass",num_class=3,n_estimators=120,learning_rate=.04,num_leaves=20,min_child_samples=750,colsample_bytree=.8,reg_lambda=20.,random_state=20260803,n_jobs=1,verbosity=-1)
        model.fit(xtr,label,sample_weight=weight); p=np.clip(model.predict_proba(xte),1e-5,1.); p/=p.sum(axis=1,keepdims=True)
        class_mean=np.array([residual[label==k].mean() for k in range(3)],dtype=float)
        expected_residual=p@class_mean
        local_output=te.prequential_base_expected_net_bps.to_numpy(float)+expected_residual
        position=test.index.get_indexer(te.index)
        output[position]=local_output; uncertainty[position]=np.sqrt(np.maximum((p * (class_mean[None, :] - expected_residual[:, None]) ** 2).sum(axis=1),0.))
        observed=np.digitize((te.net_bps-te.prequential_base_expected_net_bps).to_numpy(float),edges,right=True).astype(int)
        audit.append({"arm":"B3_tercile_meta_classifier","side_name":side,"lower_edge_bps":float(edges[0]),"upper_edge_bps":float(edges[1]),"class_means_bps":class_mean.tolist(),"train_rows":len(tr),"test_rows":len(te),"test_log_loss":float(log_loss(observed,p,labels=[0,1,2])),"test_accuracy":float((p.argmax(axis=1)==observed).mean())})
    return output, uncertainty, support, audit


def _top_metrics(frame: pd.DataFrame, score: np.ndarray, *, arm: str, transport: str, basis: str, top: float) -> list[dict[str, object]]:
    ranked = frame.assign(score_bps=np.asarray(score, dtype=float)).sort_values(["score_bps", "candidate_key"], ascending=[False, True], kind="stable")
    n = max(1, int(np.ceil(len(ranked) * top))); selected = ranked.head(n)
    result: list[dict[str, object]] = []
    for scope, part in [("global", selected), ("long", selected[selected.side_name.eq("long")]), ("short", selected[selected.side_name.eq("short")])]:
        if part.empty: continue
        result.append({"family":"topk", "transport":transport, "arm":arm, "basis":basis, "scope":scope, "top_fraction":top, "n":len(part), "net_bps":float(part.net_bps.mean()), "gross_bps":float(part.gross_bps.mean()), "total_net_bps":float(part.net_bps.sum()), "net_ic":float(spearmanr(ranked.score_bps, ranked.net_bps).statistic), "adverse_first_rate":float(part.p_adverse.mean()), "robust_clear_rate":float(part.p_clear.mean()), "long_share":float(selected.side_name.eq("long").mean()), "short_share":float(selected.side_name.eq("short").mean())})
    return result


def _side_ranking(frame: pd.DataFrame, score: np.ndarray, transport: str, arm: str) -> list[dict[str, object]]:
    work = frame.assign(score_bps=score)
    records: list[dict[str, object]] = []
    for (side, month), part in work.groupby(["side_name", "month"], observed=True):
        decile = pd.qcut(part.score_bps.rank(method="first"), 10, labels=False, duplicates="drop")
        decile_means = part.assign(decile=decile).groupby("decile", observed=True).net_bps.mean().to_numpy(float)
        monotonic = float(spearmanr(np.arange(len(decile_means)), decile_means).statistic) if len(decile_means) > 1 else np.nan
        for top in TOPS:
            n=max(1,int(np.ceil(len(part)*top))); take=part.sort_values(["score_bps","candidate_key"],ascending=[False,True],kind="stable").head(n)
            records.append({"transport":transport,"arm":arm,"side_name":side,"month":month,"top_fraction":top,"n":len(take),"raw_score_ic":float(spearmanr(part.score_bps,part.net_bps).statistic),"net_bps":float(take.net_bps.mean()),"gross_bps":float(take.gross_bps.mean()),"adverse_first_rate":float(take.p_adverse.mean()),"robust_clear_rate":float(take.p_clear.mean()),"decile_monotonicity":monotonic})
    return records


def _admission_rows(frame: pd.DataFrame, score: np.ndarray, uncertainty: np.ndarray, support: np.ndarray, *, transport: str, arm: str) -> list[dict[str, object]]:
    work = frame.assign(score_bps=score, uncertainty_bps=uncertainty, score_bin_support=support)
    result: list[dict[str, object]] = []
    variants = {
        "P1_absolute": work.score_bps,
        "P2_lower_calibration_bound": work.score_bps - work.uncertainty_bps,
        "P3_lcb_side_cap_75pct": work.score_bps - work.uncertainty_bps,
        "P4_lcb_support_100": work.score_bps - work.uncertainty_bps,
    }
    for variant, policy_score in variants.items():
        for threshold in THRESHOLDS:
            keep = policy_score.gt(threshold)
            if variant == "P4_lcb_support_100": keep &= work.score_bin_support.ge(100.)
            chosen = work.loc[keep].copy()
            if variant == "P3_lcb_side_cap_75pct" and not chosen.empty:
                limit = int(np.floor(.75 * len(chosen))); counts = chosen.side_name.value_counts()
                for side, count in counts.items():
                    if count > limit:
                        drop = chosen[chosen.side_name.eq(side)].sort_values(["score_bps","candidate_key"],ascending=[True,True],kind="stable").head(count-limit).index
                        chosen = chosen.drop(index=drop)
            # Report pooled outcome and its calendar attribution; empty policy
            # outcomes are intentional no-trade records, not missing data.
            positive_months = int(chosen.groupby("month").net_bps.mean().gt(0).sum()) if len(chosen) else 0
            worst_month = float(chosen.groupby("month").net_bps.mean().min()) if len(chosen) else np.nan
            result.append({"transport":transport,"arm":arm,"variant":variant,"threshold_bps":threshold,"rows":len(chosen),"coverage":len(chosen)/len(work),"trades_per_day":len(chosen)/max(work.__ts__.dt.normalize().nunique(),1),"net_bps":float(chosen.net_bps.mean()) if len(chosen) else np.nan,"total_net_bps":float(chosen.net_bps.sum()) if len(chosen) else 0.,"long_share":float(chosen.side_name.eq("long").mean()) if len(chosen) else np.nan,"positive_month_count":positive_months,"worst_month_net_bps":worst_month,"no_trade":bool(chosen.empty)})
            for month, part in chosen.groupby("month", observed=True):
                result.append({"transport":transport,"arm":arm,"variant":variant,"threshold_bps":threshold,"month":month,"rows":len(part),"coverage":len(part)/len(work),"trades_per_day":len(part)/max(work[work.month.eq(month)].__ts__.dt.normalize().nunique(),1),"net_bps":float(part.net_bps.mean()),"total_net_bps":float(part.net_bps.sum()),"long_share":float(part.side_name.eq("long").mean()),"positive_month_count":np.nan,"worst_month_net_bps":np.nan,"no_trade":False})
    return result


def _oracle_side_allocation(test: pd.DataFrame, frozen: np.ndarray, transport: str) -> list[dict[str, object]]:
    """Frozen within-side ranks; test labels decide only allocation, diagnostic."""
    work=test.assign(score_bps=frozen); long=work[work.side_name.eq("long")].sort_values(["score_bps","candidate_key"],ascending=[False,True],kind="stable"); short=work[work.side_name.eq("short")].sort_values(["score_bps","candidate_key"],ascending=[False,True],kind="stable")
    rows=[]
    for top in TOPS:
        n=max(1,int(np.ceil(len(work)*top))); options=[]
        for n_long in np.linspace(0,n,101,dtype=int):
            pick=pd.concat([long.head(n_long),short.head(n-n_long)])
            if len(pick)==n: options.append((float(pick.net_bps.mean()),n_long,pick))
        value,n_long,pick=max(options,key=lambda item:item[0])
        rows.append({"transport":transport,"arm":"S3_oracle_side_allocation","top_fraction":top,"n":n,"oracle_net_bps":value,"oracle_long_share":n_long/n,"oracle_short_share":1-n_long/n})
    return rows


def _matched_breaks(frame: pd.DataFrame, transport: str) -> list[dict[str, object]]:
    """A compact matched conditional diagnostic for the portable break proxies in ledger."""
    fields=("mkt_oi_chg_z_24h","mkt_funding_dispersion","cross_asset_corr_4h","mkt_flush_exhaustion_score","deleveraging_without_followthrough")
    work=frame.copy(); work["base_bucket"]=pd.qcut(work.prequential_base_expected_net_bps.rank(method="first"),10,labels=False,duplicates="drop"); work["p_clear_bucket"]=pd.qcut(work.p_clear.rank(method="first"),5,labels=False,duplicates="drop"); work["p_adverse_bucket"]=pd.qcut(work.p_adverse.rank(method="first"),5,labels=False,duplicates="drop")
    output=[]
    for field in fields:
        for side, part in work.groupby("side_name",observed=True):
            local=part.dropna(subset=[field]).copy(); local["feature_half"]=local.groupby(["month","base_bucket","p_clear_bucket","p_adverse_bucket"],observed=True)[field].transform(lambda s:s.ge(s.median()).astype(int))
            group=local.groupby("feature_half",observed=True).net_bps.agg(["mean","count"])
            if set(group.index)=={0,1}:
                output.append({"transport":transport,"side_name":side,"feature":field,"matched_high_minus_low_net_bps":float(group.loc[1,"mean"]-group.loc[0,"mean"]),"matched_rows":int(group["count"].sum()),"role":"candidate_for_constrained_feature" if abs(group.loc[1,"mean"]-group.loc[0,"mean"])>=10 else "no_material_conditional_effect"})
    return output


def run(out: Path = OUT) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    frame=_read(); metrics=[]; ranking=[]; calibration=[]; admissions=[]; allocation=[]; matched=[]; map_audit=[]; meta_audit=[]
    for name, train_eras, test_eras in TRANSPORTS:
        test=frame[frame.era.isin(test_eras)].copy(); start=test.__ts__.min(); train=frame[frame.era.isin(train_eras)&frame.label_available_ts.lt(start)].copy()
        if train.empty or test.empty: raise ValueError(f"empty causal transport {name}")
        frozen=test.prequential_base_expected_net_bps.to_numpy(float)
        # Deployable mappings, all fitted only on labels resolved before test start.
        mappings={"M0_frozen":(frozen,np.zeros(len(test)),np.full(len(test),np.inf))}
        mapping_score="prequential_base_expected_net_bps"
        m1=_fit_bin_map(train,score=mapping_score,bins=10,shrink_rows=1_000.,window_name="M1_side_shrunk")
        mappings["M1_side_shrunk"]=_apply_bin_map(test,m1,score=mapping_score)
        recent=[]
        for days in (90,180,365):
            segment=train[train.__ts__.ge(start-pd.Timedelta(days=days))]
            if len(segment)>10_000: recent.append(_apply_bin_map(test,_fit_bin_map(segment,score=mapping_score,bins=10,shrink_rows=1_000.,window_name=f"{days}d"),score=mapping_score))
        recent.append(_apply_bin_map(test,_fit_bin_map(train,score=mapping_score,bins=10,shrink_rows=1_000.,window_name="expanding"),score=mapping_score))
        mappings["M2_window_ensemble"]=(np.mean([x[0] for x in recent],axis=0),np.max([x[1] for x in recent],axis=0),np.min([x[2] for x in recent],axis=0))
        mappings["M3_hierarchical_affine"] = (_apply_affine(test,mapping_score,_fit_affine(train,mapping_score)),np.full(len(test),50.),np.full(len(test),np.inf))
        m4=_fit_bin_map(train,score=mapping_score,bins=5,shrink_rows=5_000.,window_name="M4_monotonic_strong_shrink")
        mappings["M4_monotonic_bins"]=_apply_bin_map(test,m4,score=mapping_score)
        meta_score,meta_uncertainty,meta_support,meta_detail=_tercile_meta_score(train,test)
        mappings["B3_tercile_meta_classifier"]=(meta_score,meta_uncertainty,meta_support); meta_audit.extend([{**row,"transport":name} for row in meta_detail])
        for arm,(score,uncertainty,support) in mappings.items():
            for top in TOPS: metrics.extend(_top_metrics(test,score,arm=arm,transport=name,basis="deployable_prior_resolved",top=top))
            ranking.extend(_side_ranking(test,score,name,arm)); admissions.extend(_admission_rows(test,score,uncertainty,support,transport=name,arm=arm))
        for arm,score in _oracle_scores(test,frozen).items():
            for top in TOPS: metrics.extend(_top_metrics(test,score,arm=arm,transport=name,basis="ORACLE_TEST_LABEL_DIAGNOSTIC",top=top))
            # Explicitly quarantined diagnostic: asks whether an otherwise
            # perfect score-to-bps level map would make absolute admission
            # viable.  It is never a deployable policy input.
            admissions.extend(_admission_rows(test,score,np.zeros(len(test)),np.full(len(test),np.inf),transport=name,arm=arm))
            calibration.append({"transport":name,"arm":arm,"predicted_minus_realised_bps":float((score-test.net_bps.to_numpy(float)).mean()),"score_slope":float(np.polyfit(frozen,test.net_bps.to_numpy(float),1)[0]),"score_intercept":float(np.polyfit(frozen,test.net_bps.to_numpy(float),1)[1])})
        for arm, score in _oracle_side_scores(test, frozen).items():
            for top in TOPS: metrics.extend(_top_metrics(test,score,arm=arm,transport=name,basis="ORACLE_TEST_LABEL_DIAGNOSTIC",top=top))
        allocation.extend(_oracle_side_allocation(test,frozen,name)); matched.extend(_matched_breaks(test,name))
        for arm,(score,uncertainty,support) in mappings.items(): map_audit.append({"transport":name,"arm":arm,"train_rows":len(train),"test_rows":len(test),"uses_test_labels":False,"within_side_rank_changed":arm!="M0_frozen","mean_score_bps":float(np.mean(score)),"mean_uncertainty_bps":float(np.mean(uncertainty[np.isfinite(uncertainty)])) if np.isfinite(uncertainty).any() else np.nan})
    pd.DataFrame(metrics).to_parquet(out/"score_portability_metrics.parquet",index=False); pd.DataFrame(ranking).to_parquet(out/"side_ranking_diagnostics.parquet",index=False); pd.DataFrame(calibration).to_parquet(out/"calibration_oracle_diagnostics.parquet",index=False); pd.DataFrame(admissions).to_parquet(out/"admission_policy_ablation.parquet",index=False); pd.DataFrame(allocation).to_parquet(out/"oracle_side_allocation.parquet",index=False); pd.DataFrame(matched).to_parquet(out/"matched_feature_portability.parquet",index=False); pd.DataFrame(map_audit).to_parquet(out/"mapping_audit.parquet",index=False); pd.DataFrame(meta_audit).to_parquet(out/"tercile_meta_classifier_audit.parquet",index=False)
    manifest={"schema":"tp6_score_portability_admission_v1","status":"COMPLETED_DIAGNOSTIC","input":str(INPUT),"input_sha256":_sha(INPUT),"population_rows":len(frame),"geometry":"TP6/SL4/H12","cost_bps":100.,"strict_label_availability":"decision timestamp +13h < transport test start","frozen_base":"prequential_base_expected_net_bps","global_selection":"one pooled global book after common-bps mapping","oracle_arms":["C1_oracle_monotonic","C2_oracle_intercept","C3_oracle_affine","S1_oracle_side_intercept","S2_oracle_side_affine","S3_oracle_side_allocation"],"deployable_maps":["M0_frozen","M1_side_shrunk","M2_window_ensemble","M3_hierarchical_affine","M4_monotonic_bins","B3_tercile_meta_classifier"],"tercile_meta":{"target":"side-specific train-residual terciles: base overestimate / approximately correct / base underestimate","features":list(META_FEATURES),"promotion_eligible":False},"admission_arms":["P1_absolute","P2_lower_calibration_bound","P3_lcb_side_cap_75pct","P4_lcb_support_100"]}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return out


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--out",type=Path,default=OUT); args=parser.parse_args(); print(run(args.out))
