"""Sequential, label-only screening for inference-valid LambdaRank queries.

The helpers here deliberately stop before a model fit.  They make it possible
to reject bad query geometry without spending an HPO budget or, worse, letting
final OOS outcomes choose the training grouping.
"""
from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _era(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame:
        return frame[column].astype("string").fillna("unknown")
    return pd.to_datetime(frame["__ts__"], utc=True).dt.to_period("M").astype("string")


def validity_audit(frame: pd.DataFrame, membership: pd.DataFrame, *,
                   fold_column: str = "fold", decision_column: str = "__ts__",
                   feature_available_column: str | None = None,
                   executable_column: str | None = None) -> pd.DataFrame:
    """Check query membership only uses candidates available at decision time."""
    required = ["candidate_id", decision_column]
    if fold_column in frame:
        required.append(fold_column)
    x = membership.merge(frame[required + ([feature_available_column] if feature_available_column in frame else []) + ([executable_column] if executable_column in frame else [])], on="candidate_id", validate="many_to_one")
    decision = pd.to_datetime(x[decision_column], utc=True, errors="coerce")
    future = pd.Series(False, index=x.index)
    if feature_available_column and feature_available_column in x:
        future |= pd.to_datetime(x[feature_available_column], utc=True, errors="coerce").gt(decision).fillna(True)
    executable = pd.Series(True, index=x.index)
    if executable_column and executable_column in x:
        executable = x[executable_column].fillna(False).astype(bool)
    records=[]
    for name, group in x.groupby("query_candidate", observed=True):
        qfold = group.groupby("query_id", observed=True)[fold_column].nunique() if fold_column in group else pd.Series(1, index=group.query_id.unique())
        records.append({"query_candidate":name, "rows":len(group),
                        "simultaneous_availability_rate":float((~future.loc[group.index]).mean()),
                        "future_membership_violation_count":int(future.loc[group.index].sum()),
                        "candidate_duplicate_membership_rate":float(group.duplicated(["candidate_id"]).mean()),
                        "query_boundary_violation_count":int((qfold > 1).sum()),
                        "entry_executable_rate":float(executable.loc[group.index].mean())})
    return pd.DataFrame(records)


def portability_metrics(frame: pd.DataFrame, membership: pd.DataFrame, *,
                        grade_column: str, utility_column: str = "net_bps",
                        era_column: str = "era") -> pd.DataFrame:
    """Per-era query coverage, grade geometry, and economic headroom."""
    cols=["candidate_id", grade_column, utility_column, "__ts__"]
    if era_column in frame: cols.append(era_column)
    x=membership.merge(frame[cols],on="candidate_id",validate="many_to_one")
    x["_era"]=_era(x,era_column)
    records=[]
    for (name, era), group in x.groupby(["query_candidate","_era"],observed=True):
        queries=list(group.groupby("query_id",observed=True))
        sizes=np.asarray([len(q) for _,q in queries],float)
        distinct=np.asarray([q[grade_column].nunique() for _,q in queries],int)
        rankable=(sizes>=2)&(distinct>=2)
        utility=pd.to_numeric(group[utility_column],errors="coerce")
        uplifts=np.asarray([pd.to_numeric(q[utility_column],errors="coerce").max()-pd.to_numeric(q[utility_column],errors="coerce").mean() for _,q in queries],float)
        records.append({"query_candidate":name,"era":era,"rows":len(group),"query_count":len(queries),
                        "rankable_row_coverage":float(sizes[rankable].sum()/max(len(group),1)),
                        "singleton_query_rate":float((sizes==1).mean()),
                        "distinct_grade_query_fraction":float((distinct>=2).mean()),
                        "median_group_size":float(np.median(sizes)),
                        "group_size_mad":float(np.median(np.abs(sizes-np.median(sizes)))),
                        "oracle_top1_uplift":float(np.nanmean(uplifts)),
                        "utility_iqr":float(utility.quantile(.75)-utility.quantile(.25))})
    return pd.DataFrame(records)


def aggregate_portability(era_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarise median, worst and MAD without allowing a single era to hide."""
    rows=[]
    metrics=[c for c in era_metrics if c not in {"query_candidate","era","rows","query_count"}]
    for name,g in era_metrics.groupby("query_candidate",observed=True):
        row={"query_candidate":name,"era_count":g.era.nunique()}
        for c in metrics:
            v=pd.to_numeric(g[c],errors="coerce").dropna().to_numpy(float)
            if len(v):
                med=float(np.median(v)); row[f"{c}_median"]=med; row[f"{c}_worst"]=float(np.min(v)); row[f"{c}_mad"]=float(np.median(np.abs(v-med)))
        rows.append(row)
    return pd.DataFrame(rows)


def select_pareto_shortlist(summary: pd.DataFrame, *, baseline: str = "q0_exact_timestamp_side", limit: int = 6) -> pd.DataFrame:
    """Small deterministic shortlist; always retain exact-time control.

    Dominance uses rankable coverage, portable-pair density, oracle headroom,
    common-shock fit and worst-era coverage.  Missing metrics are never used
    to manufacture a win.
    """
    x=summary.copy()
    candidates=[c for c in ("rankable_row_coverage_median","portable_pair_density_median","oracle_top1_uplift_median","query_fixed_effect_r2_median","rankable_row_coverage_worst") if c in x]
    if not candidates: raise ValueError("shortlist requires at least one screened proxy")
    values=x[candidates].fillna(-np.inf).to_numpy(float); dominated=np.zeros(len(x),bool)
    for i in range(len(x)):
        for j in range(len(x)):
            if i!=j and np.all(values[j]>=values[i]) and np.any(values[j]>values[i]): dominated[i]=True; break
    x["pareto_frontier"]=~dominated
    # z-score only inside the development table provided to this function.
    z=np.zeros(len(x))
    weights={"rankable_row_coverage_median":.20,"portable_pair_density_median":.20,"oracle_top1_uplift_median":.20,"query_fixed_effect_r2_median":.15,"rankable_row_coverage_worst":.15}
    for c,w in weights.items():
        if c in x:
            v=pd.to_numeric(x[c],errors="coerce"); sd=float(v.std(ddof=0)); z += w*((v-v.mean())/sd if sd>0 else 0.)
    if "singleton_query_rate_median" in x: z-=.10*pd.to_numeric(x.singleton_query_rate_median,errors="coerce").fillna(0.)
    x["query_score"]=z
    chosen=x[x.pareto_frontier].sort_values(["query_score","query_candidate"],ascending=[False,True],kind="stable").head(limit)
    if baseline in set(x.query_candidate) and baseline not in set(chosen.query_candidate):
        chosen=pd.concat([x[x.query_candidate.eq(baseline)],chosen]).drop_duplicates("query_candidate").head(limit)
    return x.merge(chosen[["query_candidate"]].assign(shortlisted=True),on="query_candidate",how="left").assign(shortlisted=lambda q:q.shortlisted.fillna(False))


def load_frozen_query_shortlist(path: str | Path) -> tuple[str, ...]:
    """Load the output of the no-model query screen without re-scoring it.

    A runner may consume either the compact JSON emitted by
    ``run_query_construction_screen.py`` or its parquet frontier.  This helper
    performs no ranking and no outcome access; it only validates the immutable
    query names selected by that prior development-stage artifact.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"frozen query shortlist does not exist: {source}")
    if source.suffix == ".json":
        names = json.loads(source.read_text()).get("shortlist", [])
    elif source.suffix == ".parquet":
        frame = pd.read_parquet(source)
        if "query_candidate" not in frame:
            raise KeyError("query shortlist parquet needs query_candidate")
        if "shortlisted" in frame:
            frame = frame.loc[frame["shortlisted"].fillna(False).astype(bool)]
        names = frame["query_candidate"].dropna().astype(str).tolist()
    else:
        raise ValueError("frozen query shortlist must be JSON or parquet")
    names = tuple(dict.fromkeys(map(str, names)))
    if not names:
        raise ValueError("frozen query shortlist is empty")
    from extreme_price_movements.query_candidate_definitions import query_definitions_by_name

    query_definitions_by_name(names)
    return names
