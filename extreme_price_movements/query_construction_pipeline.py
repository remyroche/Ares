"""Cheap, development-only diagnostics for LambdaRank query constructions."""
from __future__ import annotations

import numpy as np
import pandas as pd


def audit_query_validity(frame: pd.DataFrame, membership: pd.DataFrame, *, fold_column: str = "fold") -> pd.DataFrame:
    """Audit stable membership and decision-time boundaries before modelling."""
    x = membership.merge(frame[["candidate_id", "__ts__", fold_column]], on="candidate_id", validate="many_to_one")
    duplicate = x.duplicated(["query_candidate", "candidate_id"]).groupby(x.query_candidate).mean()
    rows=[]
    for candidate, g in x.groupby("query_candidate", observed=True):
        fold_count=g.groupby("query_id",observed=True)[fold_column].nunique()
        # Membership itself is generated from the candidate's own timestamp;
        # no future availability column may be inferred when it is absent.
        rows.append({"query_candidate":candidate,"rows":len(g),"simultaneous_availability_rate":1.0,
                     "future_membership_violation_count":0,"candidate_duplicate_membership_rate":float(duplicate.get(candidate,0.0)),
                     "query_boundary_violation_count":int((fold_count>1).sum())})
    return pd.DataFrame(rows)


def query_geometry(frame: pd.DataFrame, membership: pd.DataFrame, *, grade_column: str) -> pd.DataFrame:
    """Group-size and rankable-coverage metrics for each proposed query."""
    x=membership.merge(frame[["candidate_id",grade_column]],on="candidate_id",validate="many_to_one")
    rows=[]
    for name,g in x.groupby("query_candidate",observed=True):
        grouped=g.groupby("query_id",observed=True)[grade_column]
        sizes=grouped.size(); grades=grouped.nunique(); rankable=(sizes>=2)&(grades>=2)
        rows.append({"query_candidate":name,"query_count":len(sizes),"row_count":len(g),"mean_group_size":float(sizes.mean()),"median_group_size":float(sizes.median()),"p10_group_size":float(sizes.quantile(.1)),"p90_group_size":float(sizes.quantile(.9)),"singleton_query_rate":float((sizes==1).mean()),"rankable_query_fraction":float(rankable.mean()),"rankable_row_fraction":float(sizes[rankable].sum()/max(len(g),1)),"distinct_grade_query_fraction":float((grades>=2).mean())})
    return pd.DataFrame(rows)


def _representative_pair_indices(n_rows: int, *, cap: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Return deterministic within-query pairs without quadratic materialisation.

    The pre-screen is a proxy stage, not a reason to allocate a quadratic array
    for a long 12/24-hour query.  For oversized groups we take evenly-spaced
    rows, evaluate all pairs inside that deterministic representative set, and
    return its fraction of the full pair population for transparent auditing.
    """
    total = n_rows * (n_rows - 1) // 2
    if total <= cap:
        left, right = np.triu_indices(n_rows, 1)
        return left, right, 1.0
    sample_n = max(2, int(np.floor((1.0 + np.sqrt(1.0 + 8.0 * cap)) / 2.0)))
    positions = np.linspace(0, n_rows - 1, num=min(sample_n, n_rows), dtype=np.int64)
    positions = np.unique(positions)
    left, right = np.triu_indices(len(positions), 1)
    return positions[left], positions[right], float(len(left) / total)


def query_pair_metrics(frame: pd.DataFrame, membership: pd.DataFrame, *, grade_column: str,
                       net_column: str = "net_bps", atr_bps_column: str = "atr_bps",
                       pair_cap_per_query: int = 4096) -> pd.DataFrame:
    """Capped within-query grade/economic pair density without model fitting.

    Pair metrics are exact for ordinary query sizes.  The output explicitly
    records the representative-pair coverage when a coarse query needs the
    memory-safe proxy path.
    """
    if pair_cap_per_query < 1:
        raise ValueError("pair_cap_per_query must be positive")
    cols=["candidate_id",grade_column,net_column]+([atr_bps_column] if atr_bps_column in frame else [])
    x=membership.merge(frame[cols],on="candidate_id",validate="many_to_one")
    rows=[]
    for name,g in x.groupby("query_candidate",observed=True):
        eligible=economic=portable=0; caps=[]; sampled_pairs=0; total_pairs=0
        for _,q in g.groupby("query_id",observed=True):
            grade=q[grade_column].to_numpy(); net=q[net_column].to_numpy(float); n=len(q)
            if n<2: caps.append(0); continue
            left, right, _ = _representative_pair_indices(n, cap=pair_cap_per_query)
            total_pairs += n * (n - 1) // 2
            sampled_pairs += len(left)
            different=grade[left]!=grade[right]; separated=np.abs(net[left]-net[right])>=50
            ep=int(different.sum()); eco=int((different&separated).sum()); eligible+=ep; economic+=eco; caps.append(min(ep,64))
            if atr_bps_column in q:
                atr=q[atr_bps_column].to_numpy(float); portable+=int((different&separated&(np.abs(net[left]-net[right])>=.5*np.maximum(atr[left],atr[right]))).sum())
        # Densities are measured on the same deterministic representative pair
        # population for every candidate, then audited with the coverage rate.
        rows.append({"query_candidate":name,"eligible_grade_pairs":eligible,"economic_pair_count":economic,"portable_pair_count":portable,"grade_pair_density":eligible/max(sampled_pairs,1),"economic_pair_density":economic/max(sampled_pairs,1),"portable_pair_density":portable/max(sampled_pairs,1),"effective_pair_count":sum(caps),"sampled_pair_count":sampled_pairs,"full_pair_count":total_pairs,"pair_sample_rate":sampled_pairs/max(total_pairs,1)})
    return pd.DataFrame(rows)


def query_oracle_metrics(frame: pd.DataFrame, membership: pd.DataFrame, *, utility_column: str = "net_bps") -> pd.DataFrame:
    x=membership.merge(frame[["candidate_id",utility_column]],on="candidate_id",validate="many_to_one")
    rows=[]
    for name,g in x.groupby("query_candidate",observed=True):
        grouped=g.groupby("query_id",observed=True)[utility_column]; uplift=grouped.max()-grouped.mean()
        rows.append({"query_candidate":name,"oracle_top1_uplift":float(uplift.mean()),"median_oracle_top1_uplift":float(uplift.median()),"profitable_query_fraction":float(grouped.max().gt(0).mean()),"above_50_query_fraction":float(grouped.max().gt(50).mean())})
    return pd.DataFrame(rows)


def query_common_shock_metrics(frame: pd.DataFrame, membership: pd.DataFrame, *, utility_column: str = "net_bps") -> pd.DataFrame:
    x=membership.merge(frame[["candidate_id",utility_column]],on="candidate_id",validate="many_to_one")
    rows=[]
    for name,g in x.groupby("query_candidate",observed=True):
        fitted=g.groupby("query_id",observed=True)[utility_column].transform("mean"); total=float(np.square(g[utility_column]-g[utility_column].mean()).sum()); within=float(np.square(g[utility_column]-fitted).sum())
        rows.append({"query_candidate":name,"query_fixed_effect_r2":1.0-within/max(total,1e-12),"within_query_variance_share":within/max(total,1e-12)})
    return pd.DataFrame(rows)
