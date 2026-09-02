#!/usr/bin/env python3
"""Matched offline comparison for the P8u and optimized U50 router contracts.

The score inputs are target-free router, Base, and consensus receipts.  Policy
outcomes are joined only after deterministic timestamp-local ranking.  This is
a report-only tool: it never refits a model, alters an artifact, or touches a
live/exchange path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


SCHEMA = "strict_r3_router_hpo_comparison_v1"
UTILITY_FLOOR_BPS = 50.0
UTILITY_CAP_BPS = 300.0
UTILITY_POWER = 0.5
ROUTE_FRACTION = 0.50
TOP_K = (1, 2, 5, 10)
THRESHOLDS = (30, 40, 50)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _exclusive_json(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _months(root: Path) -> tuple[str, ...]:
    values = []
    for path in root.glob("target_free_scores/month=*.parquet"):
        values.append(path.stem.split("month=", 1)[1])
    if not values:
        raise AssertionError(f"{root}: no target-free monthly score files")
    return tuple(sorted(values))


def _month_labels(path: Path, month: str) -> pd.DataFrame:
    """Load only one UTC label month so the report never holds a full label panel."""
    start = pd.Timestamp(f"{month}-01", tz="UTC")
    end = start + pd.offsets.MonthBegin(1)
    frame = pd.read_parquet(
        path,
        columns=["candidate_id", "__decision_ts__", "policy_path_valid", "policy_net_bps"],
        filters=[("__decision_ts__", ">=", start), ("__decision_ts__", "<", end)],
    )
    if frame["candidate_id"].duplicated().any():
        raise AssertionError(f"{month}: policy labels are not unique by candidate ID")
    return frame.drop(columns="__decision_ts__")


def _require_target_free(frame: pd.DataFrame, score: str, origin: Path) -> None:
    required = {"candidate_id", "__decision_ts__", score}
    if missing := required - set(frame.columns):
        raise AssertionError(f"{origin}: missing columns {sorted(missing)}")
    forbidden = [
        column for column in frame.columns
        if any(token in column.lower() for token in ("policy_", "label", "outcome", "net_bps", "gross_bps", "path_valid"))
    ]
    if forbidden:
        raise AssertionError(f"{origin}: outcome-like target-free columns {forbidden}")
    if frame["candidate_id"].duplicated().any():
        raise AssertionError(f"{origin}: duplicate candidate IDs")


def _rank(frame: pd.DataFrame, score: str) -> pd.DataFrame:
    work = frame.loc[:, ["candidate_id", "__decision_ts__", score]].copy()
    work["__score__"] = pd.to_numeric(work[score], errors="coerce").fillna(-np.inf)
    work = work.sort_values(["__decision_ts__", "__score__", "candidate_id"], ascending=[True, False, True], kind="stable")
    work["rank_local"] = work.groupby("__decision_ts__", sort=False).cumcount() + 1
    work["candidate_count"] = work.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size")
    work["rank_pct"] = (work["rank_local"] - 1) / work["candidate_count"].clip(lower=1)
    work["selected_top50"] = (work["rank_local"] <= np.ceil(ROUTE_FRACTION * work["candidate_count"])) & np.isfinite(work["__score__"])
    return work.drop(columns="__score__")


def _utility(net: np.ndarray, valid: np.ndarray) -> np.ndarray:
    excess = np.where(valid, np.maximum(net - UTILITY_FLOOR_BPS, 0.0), 0.0)
    return np.power(np.minimum(excess, UTILITY_CAP_BPS) / UTILITY_CAP_BPS, UTILITY_POWER)


def _timestamp_rows(joined: pd.DataFrame, selection: str) -> pd.DataFrame:
    work = joined.copy()
    valid = work["policy_path_valid"].fillna(False).to_numpy(bool) & np.isfinite(work["policy_net_bps"].to_numpy(float))
    net = pd.to_numeric(work["policy_net_bps"], errors="coerce").fillna(0.0).to_numpy(float)
    select = work[selection].fillna(False).to_numpy(bool)
    utility = _utility(net, valid)
    work["__utility__"] = utility
    work["__valid__"] = valid
    work["__selected__"] = select
    work["__selected_utility__"] = np.where(select, utility, 0.0)
    work["__selected_net__"] = np.where(select & valid, net, 0.0)
    work["__winner50__"] = valid & (net > 50.0)
    work["__winner100__"] = valid & (net > 100.0)
    grouped = work.groupby("__decision_ts__", sort=False).agg(
        candidate_rows=("candidate_id", "size"),
        selected_rows=("__selected__", "sum"),
        valid_rows=("__valid__", "sum"),
        selected_valid_rows=("__selected__", lambda x: int((x & work.loc[x.index, "__valid__"]).sum())),
        utility_sum=("__utility__", "sum"),
        selected_utility_sum=("__selected_utility__", "sum"),
        selected_net_bps=("__selected_net__", "sum"),
        winners50=("__winner50__", "sum"),
        winners100=("__winner100__", "sum"),
        selected_winners50=("__winner50__", lambda x: int((x & work.loc[x.index, "__selected__"]).sum())),
        selected_winners100=("__winner100__", lambda x: int((x & work.loc[x.index, "__selected__"]).sum())),
    ).reset_index()
    grouped["utility_recall"] = grouped["selected_utility_sum"] / grouped["utility_sum"].replace(0.0, np.nan)
    grouped["recall50"] = grouped["selected_winners50"] / grouped["winners50"].replace(0.0, np.nan)
    grouped["recall100"] = grouped["selected_winners100"] / grouped["winners100"].replace(0.0, np.nan)
    grouped["selected_net_ev_bps"] = grouped["selected_net_bps"] / grouped["selected_valid_rows"].replace(0.0, np.nan)
    positive = grouped["utility_sum"] > 0
    median = float(grouped.loc[positive, "utility_sum"].median()) if positive.any() else np.nan
    grouped["utility_weight"] = 0.0
    if np.isfinite(median) and median > 0:
        grouped.loc[positive, "utility_weight"] = np.minimum(np.sqrt(grouped.loc[positive, "utility_sum"] / median), 2.0)
    grouped["month"] = pd.to_datetime(grouped["__decision_ts__"], utc=True).dt.strftime("%Y-%m")
    grouped["week"] = pd.to_datetime(grouped["__decision_ts__"], utc=True).dt.to_period("W-SUN").dt.start_time.astype(str)
    return grouped


def _summary(rows: pd.DataFrame) -> dict[str, float | int]:
    relevant = rows["utility_recall"].notna() & (rows["utility_weight"] > 0)
    weights = rows.loc[relevant, "utility_weight"]
    utility = float(np.average(rows.loc[relevant, "utility_recall"], weights=weights)) if float(weights.sum()) else np.nan
    recall50 = float(rows.loc[rows["recall50"].notna(), "recall50"].mean())
    recall100 = float(rows.loc[rows["recall100"].notna(), "recall100"].mean())
    return {
        "timestamps": int(len(rows)),
        "r50_utility": utility,
        "r50_count": recall50,
        "r100_count": recall100,
        "s_router": float(.7 * utility + .15 * recall50 + .15 * recall100),
        "selected_rows": int(rows["selected_rows"].sum()),
        "selected_valid_rows": int(rows["selected_valid_rows"].sum()),
        "selected_net_bps": float(rows["selected_net_bps"].sum()),
        "selected_net_ev_bps": float(rows["selected_net_ev_bps"].mean()),
    }


def _stability(rows: pd.DataFrame) -> dict[str, float]:
    weekly = pd.DataFrame([{"week": week, **_summary(group)} for week, group in rows.groupby("week", sort=True)])
    values = weekly["s_router"].to_numpy(float)
    q20, q80 = np.quantile(values, [.20, .80])
    robust = float(values[(values >= q20) & (values <= q80)].mean())
    lower = float(np.mean(np.quantile(values, [.15, .10, .05])))
    return {"weekly_s_router_robust": robust, "weekly_s_router_lower": lower, "weekly_s_stable": robust + .5 * lower,
            "weekly_s_router_min": float(values.min()), "weekly_s_router_q25": float(np.quantile(values, .25))}


def _ranked_topk(joined: pd.DataFrame, score: str, layer: str) -> pd.DataFrame:
    ranked = _rank(joined, score)
    ranked = ranked.merge(joined.loc[:, ["candidate_id", "policy_path_valid", "policy_net_bps"]], on="candidate_id", how="left", validate="one_to_one")
    ranked["month"] = pd.to_datetime(ranked["__decision_ts__"], utc=True).dt.strftime("%Y-%m")
    rows = []
    for k in TOP_K:
        selected = ranked.loc[(ranked["rank_local"] <= k) & ranked["policy_path_valid"].fillna(False)].copy()
        means = selected.groupby("__decision_ts__", sort=False)["policy_net_bps"].mean()
        for period, series in [("global", means), *[(m, means.loc[means.index.to_series().dt.strftime("%Y-%m").eq(m)]) for m in sorted(ranked["month"].unique())]]:
            rows.append({"layer": layer, "period": period, "top_k": k, "timestamps": int(series.size), "mean_timestamp_net_bps": float(series.mean()), "median_timestamp_net_bps": float(series.median()), "trade_rows": int(selected.shape[0] if period == "global" else selected.loc[selected["month"].eq(period)].shape[0])})
    return pd.DataFrame(rows)


def _base_and_meta(root: Path, labels_path: Path, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_parts, meta_parts = [], []
    for month in ("2026-04", "2026-05", "2026-06", "2026-07"):
        labels = _month_labels(labels_path, month)
        base = pd.read_parquet(
            root / f"target_free_monthly/month={month}/scores_features.parquet",
            columns=["candidate_id", "__decision_ts__", "enhanced_base_bps"],
        )
        _require_target_free(base, "enhanced_base_bps", root / f"target_free_monthly/month={month}/scores_features.parquet")
        base_parts.append(base.merge(labels, on="candidate_id", how="left", validate="one_to_one"))
        for family in ("current", "bcf"):
            p = root / f"target_free_scores/{family}/month={month}.parquet"
            score = f"{family}_final_score"
            frame = pd.read_parquet(p, columns=["candidate_id", "__decision_ts__", "final_score"]).rename(columns={"final_score": score})
            _require_target_free(frame, score, p)
            meta_parts.append(frame.merge(labels, on="candidate_id", how="left", validate="one_to_one").assign(family=family))
    base = pd.concat(base_parts, ignore_index=True)
    meta_long = pd.concat(meta_parts, ignore_index=True)
    meta_rows = []
    for family, group in meta_long.groupby("family", sort=True):
        score = f"{family}_final_score"
        joined = group.drop(columns="family")
        meta_rows.append(_ranked_topk(joined, score, f"meta_{family}_{label}"))
    return _ranked_topk(base, "enhanced_base_bps", f"base_{label}"), pd.concat(meta_rows, ignore_index=True)


def _mc1(root: Path, label: str) -> pd.DataFrame:
    frame = pd.read_parquet(root / "dual_mc1_predictions.parquet")
    valid = frame["policy_path_valid"].fillna(False) & np.isfinite(frame["policy_net_bps"])
    rows = []
    for threshold in THRESHOLDS:
        admitted = valid & (frame["current_mc1_expected_bps"] >= threshold) & (frame["bcf_mc1_expected_bps"] >= threshold)
        part = frame.loc[admitted].copy()
        rows.append({"model": label, "threshold_bps": threshold, "admitted_rows": int(len(part)),
                     "realised_net_ev_bps": float(part["policy_net_bps"].mean()),
                     "realised_net_sum_bps": float(part["policy_net_bps"].sum()),
                     "months": int(part["__decision_ts__"].dt.strftime("%Y-%m").nunique())})
    return pd.DataFrame(rows)


def _portfolio(root: Path, label: str) -> pd.DataFrame:
    frame = pd.read_parquet(root / "portfolio_metrics.parquet").copy()
    frame["model"] = label
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-router", type=Path, required=True)
    parser.add_argument("--optimized-router", type=Path, required=True)
    parser.add_argument("--baseline-downstream", type=Path, required=True)
    parser.add_argument("--optimized-downstream", type=Path, required=True)
    parser.add_argument("--policy-labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"immutable output already exists: {args.out}")
    args.out.mkdir(parents=True)
    pmonths, umonths = _months(args.baseline_router), _months(args.optimized_router)
    months = tuple(sorted(set(pmonths) & set(umonths)))
    if not months:
        raise AssertionError("no common router score months")
    per_model, substitutions = [], []
    for month in months:
        ppath = args.baseline_router / f"target_free_scores/month={month}.parquet"
        upath = args.optimized_router / f"target_free_scores/month={month}.parquet"
        pframe, uframe = pd.read_parquet(ppath), pd.read_parquet(upath)
        _require_target_free(pframe, "router_primary_only_rank", ppath)
        _require_target_free(uframe, "router_primary_only_rank", upath)
        if set(pframe.candidate_id) != set(uframe.candidate_id):
            raise AssertionError(f"{month}: candidate identities differ between router receipts")
        labels = _month_labels(args.policy_labels, month)
        p = _rank(pframe, "router_primary_only_rank").merge(labels, on="candidate_id", how="left", validate="one_to_one")
        u = _rank(uframe, "router_primary_only_rank").merge(labels, on="candidate_id", how="left", validate="one_to_one")
        for name, frame in (("P8u", p), ("U50_HPO1732", u)):
            timestamp = _timestamp_rows(frame, "selected_top50")
            per_model.append(timestamp.assign(model=name, held_month=month))
        both = p.loc[:, ["candidate_id", "selected_top50", "policy_path_valid", "policy_net_bps"]].merge(
            u.loc[:, ["candidate_id", "selected_top50"]], on="candidate_id", suffixes=("_p8", "_u"), validate="one_to_one")
        both["cohort"] = np.select(
            [both.selected_top50_p8 & both.selected_top50_u, both.selected_top50_p8, both.selected_top50_u],
            ["both", "p8u_only", "u50_only"], default="neither")
        for cohort, group in both.groupby("cohort", sort=True):
            valid = group.policy_path_valid.fillna(False) & np.isfinite(group.policy_net_bps)
            utility = _utility(group.policy_net_bps.fillna(0.0).to_numpy(float), valid.to_numpy(bool))
            substitutions.append({"month": month, "cohort": cohort, "rows": int(len(group)), "valid_rows": int(valid.sum()),
                                  "net_ev_bps": float(group.loc[valid, "policy_net_bps"].mean()), "net_sum_bps": float(group.loc[valid, "policy_net_bps"].sum()),
                                  "utility_sum": float(utility.sum()), "winner50_rate": float((group.loc[valid, "policy_net_bps"] > 50).mean())})
    timestamps = pd.concat(per_model, ignore_index=True)
    router_rows = []
    for model, group in timestamps.groupby("model", sort=True):
        router_rows.append({"model": model, "period": "global", **_summary(group), **_stability(group)})
        for month, part in group.groupby("month", sort=True):
            router_rows.append({"model": model, "period": month, **_summary(part)})
    router_metrics = pd.DataFrame(router_rows)
    sub = pd.DataFrame(substitutions)
    sub_summary = sub.groupby("cohort", as_index=False).agg(rows=("rows", "sum"), valid_rows=("valid_rows", "sum"), net_sum_bps=("net_sum_bps", "sum"), utility_sum=("utility_sum", "sum"))
    sub_summary["net_ev_bps"] = sub_summary["net_sum_bps"] / sub_summary["valid_rows"].replace(0, np.nan)
    base_rows, meta_rows = [], []
    for label, root in (("P8u", args.baseline_downstream), ("U50_HPO1732", args.optimized_downstream)):
        b, m = _base_and_meta(root, args.policy_labels, label)
        base_rows.append(b); meta_rows.append(m)
    base_metrics, meta_metrics = pd.concat(base_rows, ignore_index=True), pd.concat(meta_rows, ignore_index=True)
    mc1 = pd.concat([_mc1(args.baseline_downstream, "P8u"), _mc1(args.optimized_downstream, "U50_HPO1732")], ignore_index=True)
    portfolio = pd.concat([_portfolio(args.baseline_downstream, "P8u"), _portfolio(args.optimized_downstream, "U50_HPO1732")], ignore_index=True)
    router_metrics.to_parquet(args.out / "router_metrics.parquet", index=False, compression="zstd")
    timestamps.to_parquet(args.out / "router_timestamp_metrics.parquet", index=False, compression="zstd")
    sub.to_parquet(args.out / "router_substitution_by_month.parquet", index=False, compression="zstd")
    sub_summary.to_parquet(args.out / "router_substitution_summary.parquet", index=False, compression="zstd")
    base_metrics.to_parquet(args.out / "base_timestamp_topk_metrics.parquet", index=False, compression="zstd")
    meta_metrics.to_parquet(args.out / "meta_timestamp_topk_metrics.parquet", index=False, compression="zstd")
    mc1.to_parquet(args.out / "dual_mc1_admission_metrics.parquet", index=False, compression="zstd")
    portfolio.to_parquet(args.out / "portfolio_metrics.parquet", index=False, compression="zstd")
    _exclusive_json(args.out / "run_manifest.json", {
        "schema": SCHEMA, "scope": "offline-report-only; target-free rank then outcome join", "router_score": "router_primary_only_rank",
        "route": "timestamp-local Top-50%; candidate_id tie break", "utility": {"floor_bps": 50.0, "cap_bps": 300.0, "power": .5},
        "hpo_stability": "weekly robust mean Q20-Q80 S_router + 0.5*mean(Q15,Q10,Q5)",
        "months": months, "downstream_period": "2026-04 through 2026-07", "policy_labels": str(args.policy_labels),
        "hashes": {"baseline_router": _sha256(args.baseline_router / "run_manifest.json") if (args.baseline_router / "run_manifest.json").exists() else None,
                   "optimized_router": _sha256(args.optimized_router / "run_manifest.json"),
                   "policy_labels": _sha256(args.policy_labels)},
        "contracts": {"base_meta_mc1": "routed-only; no numeric router score input; only target-free selected identities differ", "portfolio": "existing frozen dual-MC1 portfolios; no refit"},
    })


if __name__ == "__main__":
    main()
