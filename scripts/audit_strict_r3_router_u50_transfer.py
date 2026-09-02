#!/usr/bin/env python3
"""Audit why an optimized Router50 does or does not transfer downstream.

This is a report-only, offline utility.  It first persists a target-free trace
of router/Base/Meta/MC1 outputs for the fixed P8u and U50 contracts.  Canonical
rich-policy outcomes are joined only after that receipt is sealed.  It never
fits models, adjusts thresholds, or touches an inference/live artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


SCHEMA = "strict_r3_router_u50_transfer_audit_v1"
MONTHS = ("2026-04", "2026-05", "2026-06", "2026-07")
ROUTE_FRACTION = 0.50
MC1_THRESHOLD_BPS = 50.0
BASE_BUCKETS = ((0, 5), (5, 10), (10, 20), (20, 30), (30, 50), (50, 100))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _receipt_hash(root: Path) -> str:
    for name in ("run_manifest.json", "run_contract.json"):
        candidate = root / name
        if candidate.exists():
            return _sha256(candidate)
    raise FileNotFoundError(f"{root}: no run_manifest.json or run_contract.json")


def _write_json_exclusive(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _top_fraction(frame: pd.DataFrame, field: str) -> pd.Series:
    """Exact canonical Top-50 gate: descending score, candidate-id tie break."""
    work = frame.loc[:, ["candidate_id", "__decision_ts__", field]].copy()
    work["__pos__"] = np.arange(len(work), dtype=np.int64)
    work["__score__"] = pd.to_numeric(work[field], errors="coerce").fillna(-np.inf)
    work = work.sort_values(
        ["__decision_ts__", "__score__", "candidate_id"],
        ascending=[True, False, True], kind="stable",
    )
    work["__ord__"] = work.groupby("__decision_ts__", sort=False).cumcount() + 1
    count = work.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size")
    work["__selected__"] = np.isfinite(work["__score__"]) & work["__ord__"].le(
        np.ceil(ROUTE_FRACTION * count).astype(np.int64)
    )
    return work.sort_values("__pos__", kind="stable")["__selected__"].reset_index(drop=True)


def _read_router(root: Path, month: str, prefix: str) -> pd.DataFrame:
    path = root / f"target_free_scores/month={month}.parquet"
    frame = pd.read_parquet(
        path, columns=["candidate_id", "__decision_ts__", "side_name", "router_primary_rank"]
    )
    if frame["candidate_id"].duplicated().any():
        raise AssertionError(f"{path}: duplicate candidate identity")
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    frame[f"{prefix}_route"] = _top_fraction(frame, "router_primary_rank").to_numpy(bool)
    return frame.rename(columns={"router_primary_rank": f"{prefix}_router_rank"}).drop(columns="side_name")


def _read_score(root: Path, month: str, prefix: str) -> pd.DataFrame:
    base_path = root / f"target_free_monthly/month={month}/scores_features.parquet"
    base = pd.read_parquet(
        base_path,
        columns=["candidate_id", "__decision_ts__", "enhanced_base_bps", "base_rank_ts"],
    ).rename(columns={
        "enhanced_base_bps": f"{prefix}_base_bps",
        "base_rank_ts": f"{prefix}_base_rank_ts",
    })
    if base["candidate_id"].duplicated().any():
        raise AssertionError(f"{base_path}: duplicate target-free base identity")
    base["__decision_ts__"] = pd.to_datetime(base["__decision_ts__"], utc=True, errors="raise")
    rows = [base]
    wanted = [
        "candidate_id", "__decision_ts__", "final_score", "upstream",
        "corrected_current_bps", "corrected_bcf_bps", "conditional_consensus_rank",
        "ordinary_shadow_consensus_rank", "correctness_rank", "head_agreement_std",
    ]
    for family in ("current", "bcf"):
        path = root / f"target_free_scores/{family}/month={month}.parquet"
        frame = pd.read_parquet(path, columns=wanted)
        if frame["candidate_id"].duplicated().any():
            raise AssertionError(f"{path}: duplicate target-free Meta identity")
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
        renames = {"final_score": f"{prefix}_{family}_final_score"}
        for col in wanted:
            if col not in {"candidate_id", "__decision_ts__", "final_score"}:
                renames[col] = f"{prefix}_{family}_{col}"
        rows.append(frame.rename(columns=renames))
    output = rows[0]
    for part in rows[1:]:
        output = output.merge(part, on=["candidate_id", "__decision_ts__"], how="outer", validate="one_to_one")
    if output.isna().all(axis=1).any():
        raise AssertionError(f"{root}/{month}: empty score merge")
    return output


def _read_mc1(root: Path, prefix: str) -> pd.DataFrame:
    # Read score columns only.  This replay panel also stores outcomes, but
    # those columns are deliberately not read until the second audit stage.
    frame = pd.read_parquet(
        root / "dual_mc1_predictions.parquet",
        columns=["candidate_id", "__decision_ts__", "current_mc1_expected_bps", "bcf_mc1_expected_bps"],
    ).rename(columns={
        "current_mc1_expected_bps": f"{prefix}_current_mc1_expected_bps",
        "bcf_mc1_expected_bps": f"{prefix}_bcf_mc1_expected_bps",
    })
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    frame = frame.loc[frame["__decision_ts__"].dt.strftime("%Y-%m").isin(MONTHS)].copy()
    if frame["candidate_id"].duplicated().any():
        raise AssertionError(f"{root}: duplicate MC1 identity")
    return frame


def _rank_top_pct(frame: pd.DataFrame, field: str) -> pd.Series:
    work = frame.loc[:, ["candidate_id", "__decision_ts__", field]].copy()
    work["__index__"] = work.index
    work["__score__"] = pd.to_numeric(work[field], errors="coerce").fillna(-np.inf)
    work = work.sort_values(["__decision_ts__", "__score__", "candidate_id"], ascending=[True, False, True], kind="stable")
    ordinal = work.groupby("__decision_ts__", sort=False).cumcount().to_numpy(float)
    size = work.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy(float)
    work["__top_pct__"] = 100.0 * ordinal / np.maximum(size, 1.0)
    return work.set_index("__index__")["__top_pct__"]


def _top_k(frame: pd.DataFrame, field: str, k: int) -> pd.Series:
    work = frame.loc[:, ["candidate_id", "__decision_ts__", field]].copy()
    work["__index__"] = work.index
    work["__score__"] = pd.to_numeric(work[field], errors="coerce").fillna(-np.inf)
    work = work.sort_values(["__decision_ts__", "__score__", "candidate_id"], ascending=[True, False, True], kind="stable")
    work["__selected__"] = work.groupby("__decision_ts__", sort=False).cumcount().lt(int(k)) & np.isfinite(work["__score__"])
    return work.set_index("__index__")["__selected__"]


def _utility(net: pd.Series, valid: pd.Series) -> np.ndarray:
    values = pd.to_numeric(net, errors="coerce").fillna(0.0).to_numpy(float)
    eligible = valid.fillna(False).to_numpy(bool) & np.isfinite(values)
    return np.sqrt(np.minimum(np.maximum(values - 50.0, 0.0), 300.0) / 300.0) * eligible


def _cohort_stats(frame: pd.DataFrame, score_prefix: str, cohort: str) -> dict[str, object]:
    part = frame.loc[frame["cohort"].eq(cohort)].copy()
    valid = part["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(part["policy_net_bps"])
    current = pd.to_numeric(part[f"{score_prefix}_current_mc1_expected_bps"], errors="coerce")
    bcf = pd.to_numeric(part[f"{score_prefix}_bcf_mc1_expected_bps"], errors="coerce")
    dual = current.ge(MC1_THRESHOLD_BPS) & bcf.ge(MC1_THRESHOLD_BPS)
    return {
        "cohort": cohort,
        "score_arm": score_prefix,
        "rows": int(len(part)),
        "valid_rows": int(valid.sum()),
        "realised_ev_bps": float(part.loc[valid, "policy_net_bps"].mean()),
        "realised_sum_bps": float(part.loc[valid, "policy_net_bps"].sum()),
        "utility_sum": float(_utility(part["policy_net_bps"], valid).sum()),
        "win_gt50_rate": float(part.loc[valid, "policy_net_bps"].gt(50.0).mean()),
        "win_gt100_rate": float(part.loc[valid, "policy_net_bps"].gt(100.0).mean()),
        "win_gt200_rate": float(part.loc[valid, "policy_net_bps"].gt(200.0).mean()),
        "dual_mc1_admitted": int((dual & valid).sum()),
        "dual_mc1_admitted_ev_bps": float(part.loc[dual & valid, "policy_net_bps"].mean()),
        "dual_mc1_rejected": int((~dual & valid).sum()),
        "dual_mc1_rejected_ev_bps": float(part.loc[(~dual) & valid, "policy_net_bps"].mean()),
    }


def _bucket_stats(frame: pd.DataFrame, arm: str) -> pd.DataFrame:
    score = f"{arm}_base_top_pct"
    rows = []
    valid = frame["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(frame["policy_net_bps"])
    for lo, hi in BASE_BUCKETS:
        bucket = frame[score].ge(lo) & frame[score].lt(hi)
        for cohort in ("common", f"{arm}_only"):
            part = frame.loc[bucket & frame["cohort"].eq(cohort)].copy()
            mask = valid.loc[part.index]
            rows.append({
                "arm": arm, "cohort": cohort, "base_top_pct_bucket": f"{lo}-{hi}",
                "rows": int(len(part)), "valid_rows": int(mask.sum()),
                "realised_ev_bps": float(part.loc[mask, "policy_net_bps"].mean()),
                "winner_gt50_rate": float(part.loc[mask, "policy_net_bps"].gt(50.0).mean()),
            })
    return pd.DataFrame(rows)


def _mc1_failure_stats(frame: pd.DataFrame, arm: str) -> pd.DataFrame:
    valid = frame["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(frame["policy_net_bps"])
    current = pd.to_numeric(frame[f"{arm}_current_mc1_expected_bps"], errors="coerce")
    bcf = pd.to_numeric(frame[f"{arm}_bcf_mc1_expected_bps"], errors="coerce")
    outcome = np.select(
        [current.ge(50) & bcf.ge(50), current.lt(50) & bcf.ge(50), current.ge(50) & bcf.lt(50)],
        ["dual_pass", "current_fail", "bcf_fail"], default="both_fail_or_missing",
    )
    work = frame.loc[:, ["cohort", "policy_net_bps"]].copy()
    work["failure_state"] = outcome
    work["min_map_bps"] = np.minimum(current, bcf)
    work["distance_below_50_bps"] = np.maximum(50.0 - work["min_map_bps"], 0.0)
    work["valid"] = valid
    work = work.loc[work["cohort"].isin(["common", f"{arm}_only"])]
    rows = []
    for (cohort, state), part in work.groupby(["cohort", "failure_state"], sort=True):
        mask = part["valid"]
        rows.append({
            "arm": arm, "cohort": cohort, "failure_state": state, "rows": int(len(part)),
            "valid_rows": int(mask.sum()), "realised_ev_bps": float(part.loc[mask, "policy_net_bps"].mean()),
            "mean_min_map_bps": float(part["min_map_bps"].mean()),
            "mean_distance_below_50_bps": float(part["distance_below_50_bps"].mean()),
        })
    return pd.DataFrame(rows)


def _base_transfer(frame: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Utility retained by each exclusive Router cohort at Base Top-20%."""
    valid = frame["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(frame["policy_net_bps"])
    exclusive = frame["cohort"].eq(f"{arm}_only")
    utility = _utility(frame["policy_net_bps"], valid)
    top20 = frame[f"{arm}_base_top_pct"].lt(20.0)
    rows = []
    for threshold in (5.0, 10.0, 20.0, 30.0, 50.0):
        mask = exclusive & frame[f"{arm}_base_top_pct"].lt(threshold)
        denom = float(utility[exclusive].sum())
        rows.append({
            "arm": arm, "cohort": f"{arm}_only", "base_top_pct": threshold,
            "rows": int(mask.sum()), "valid_rows": int((mask & valid).sum()),
            "utility_retained": float(utility[mask].sum()),
            "utility_transfer_rate": float(utility[mask].sum() / denom) if denom > 0 else np.nan,
            "realised_ev_bps": float(frame.loc[mask & valid, "policy_net_bps"].mean()),
        })
    return pd.DataFrame(rows)


def _meta_top1(frame: pd.DataFrame, family: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare candidate-specific Current/BCF Meta Top-1 substitutions."""
    work = frame.copy()
    for arm in ("p8u", "u50"):
        route = work[f"{arm}_route"].fillna(False).astype(bool)
        field = f"{arm}_{family}_final_score"
        work[f"{arm}_{family}_meta_top1"] = False
        # `_top_k` intentionally sorts to form the timestamp ranking, so
        # assign its flags back by the original candidate index rather than
        # positional array order.
        flags = _top_k(work.loc[route], field, 1)
        work.loc[route, f"{arm}_{family}_meta_top1"] = flags.reindex(work.loc[route].index).to_numpy(bool)
    p = work[f"p8u_{family}_meta_top1"]
    u = work[f"u50_{family}_meta_top1"]
    work["meta_top1_state"] = np.select(
        [p & u, p, u], ["both_top1", "p8u_top1_only", "u50_top1_only"], default="neither"
    )
    valid = work["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(work["policy_net_bps"])
    rows = []
    for state, part in work.loc[work["meta_top1_state"].ne("neither")].groupby("meta_top1_state", sort=True):
        mask = valid.loc[part.index]
        arm = "u50" if state == "u50_top1_only" else "p8u"
        current = pd.to_numeric(part[f"{arm}_current_mc1_expected_bps"], errors="coerce")
        bcf = pd.to_numeric(part[f"{arm}_bcf_mc1_expected_bps"], errors="coerce")
        rows.append({
            "family": family, "meta_top1_state": state, "rows": int(len(part)), "valid_rows": int(mask.sum()),
            "realised_ev_bps": float(part.loc[mask, "policy_net_bps"].mean()),
            "mean_base_top_pct": float(part[f"{arm}_base_top_pct"].mean()),
            # This receipt's historical `corrected_*_bps` coordinate is
            # intentionally blank; the executable target-free correction is
            # the final CDF rank relative to upstream rank.
            "mean_meta_final_rank_lift": float((part[f"{arm}_{family}_final_score"] - part[f"{arm}_{family}_upstream"]).mean()),
            "dual_mc1_pass_rate": float((current.ge(50) & bcf.ge(50)).mean()),
        })
    return pd.DataFrame(rows), work.loc[work["meta_top1_state"].eq("u50_top1_only")].copy()


def _u50_good_map_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(frame["policy_net_bps"])
    current = pd.to_numeric(frame["u50_current_mc1_expected_bps"], errors="coerce")
    bcf = pd.to_numeric(frame["u50_bcf_mc1_expected_bps"], errors="coerce")
    part = frame.loc[frame["cohort"].eq("u50_only") & valid & frame["policy_net_bps"].gt(50.0)].copy()
    part["min_map_bps"] = np.minimum(current.loc[part.index], bcf.loc[part.index])
    part["map_bucket"] = pd.cut(part["min_map_bps"], bins=[-np.inf, 30.0, 40.0, 50.0, np.inf], labels=["<30", "30-40", "40-50", "50+"])
    return part.groupby("map_bucket", observed=False).agg(
        rows=("candidate_id", "size"), realised_ev_bps=("policy_net_bps", "mean"),
        mean_current_map_bps=("u50_current_mc1_expected_bps", "mean"),
        mean_bcf_map_bps=("u50_bcf_mc1_expected_bps", "mean"),
        mean_base_top_pct=("u50_base_top_pct", "mean"),
        mean_bcf_meta_top_pct=("u50_bcf_meta_top_pct", "mean"),
    ).reset_index()


def _markdown(table: pd.DataFrame) -> str:
    visible = table.copy()
    for column in visible.columns:
        if column.endswith("_bps"):
            visible[column] = visible[column].map(lambda x: "—" if not np.isfinite(x) else f"{x:+.2f}")
        elif column.endswith("_rate"):
            visible[column] = visible[column].map(lambda x: "—" if not np.isfinite(x) else f"{100*x:.1f}%")
    headers = list(visible.columns)
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in visible.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(x).replace("|", "\\|") for x in row) + " |")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p8u-router", type=Path, required=True)
    parser.add_argument("--u50-router", type=Path, required=True)
    parser.add_argument("--p8u-stack", type=Path, required=True)
    parser.add_argument("--u50-stack", type=Path, required=True)
    parser.add_argument("--policy-labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"immutable output already exists: {args.out}")
    args.out.mkdir(parents=True)

    trace_parts = []
    for month in MONTHS:
        p_router = _read_router(args.p8u_router, month, "p8u")
        u_router = _read_router(args.u50_router, month, "u50")
        if set(p_router["candidate_id"]) != set(u_router["candidate_id"]):
            raise AssertionError(f"{month}: routers score different candidate identities")
        trace = p_router.merge(u_router, on=["candidate_id", "__decision_ts__"], how="inner", validate="one_to_one")
        p_score = _read_score(args.p8u_stack, month, "p8u")
        u_score = _read_score(args.u50_stack, month, "u50")
        trace = trace.merge(p_score, on=["candidate_id", "__decision_ts__"], how="left", validate="one_to_one")
        trace = trace.merge(u_score, on=["candidate_id", "__decision_ts__"], how="left", validate="one_to_one")
        trace_parts.append(trace)
    trace = pd.concat(trace_parts, ignore_index=True)
    trace["cohort"] = np.select(
        [trace["p8u_route"] & trace["u50_route"], trace["p8u_route"], trace["u50_route"]],
        ["common", "p8u_only", "u50_only"], default="neither",
    )
    p_mc1 = _read_mc1(args.p8u_stack, "p8u")
    u_mc1 = _read_mc1(args.u50_stack, "u50")
    trace = trace.merge(p_mc1, on=["candidate_id", "__decision_ts__"], how="left", validate="one_to_one")
    trace = trace.merge(u_mc1, on=["candidate_id", "__decision_ts__"], how="left", validate="one_to_one")
    for arm in ("p8u", "u50"):
        route = trace[f"{arm}_route"].fillna(False).astype(bool)
        absent = trace.loc[route, [f"{arm}_base_bps", f"{arm}_current_final_score", f"{arm}_bcf_final_score"]].isna().any(axis=1)
        if absent.any():
            raise AssertionError(f"{arm}: routed target-free scores missing for {int(absent.sum())} rows")
        meta = trace.loc[route, [f"{arm}_current_mc1_expected_bps", f"{arm}_bcf_mc1_expected_bps"]].isna().any(axis=1)
        if meta.any():
            raise AssertionError(f"{arm}: routed MC1 scores missing for {int(meta.sum())} rows")
        trace[f"{arm}_base_top_pct"] = _rank_top_pct(trace.loc[route], f"{arm}_base_bps").reindex(trace.index)
        for family in ("current", "bcf"):
            trace[f"{arm}_{family}_meta_top_pct"] = _rank_top_pct(
                trace.loc[route], f"{arm}_{family}_final_score"
            ).reindex(trace.index)

    # This receipt intentionally has no policy outcome fields.
    outcome_tokens = ("policy_", "outcome", "label", "net_bps", "gross_bps", "path_valid")
    if [c for c in trace.columns if any(t in c.lower() for t in outcome_tokens)]:
        raise AssertionError("target-free transfer trace unexpectedly contains outcome-like fields")
    trace.to_parquet(args.out / "transfer_trace_target_free.parquet", index=False, compression="zstd")

    labels = pd.read_parquet(
        args.policy_labels,
        columns=["candidate_id", "policy_path_valid", "policy_net_bps", "policy_label_available_ts"],
    )
    if labels["candidate_id"].duplicated().any():
        raise AssertionError("canonical policy labels duplicate candidate identities")
    joined = trace.merge(labels, on="candidate_id", how="left", validate="one_to_one")
    joined.to_parquet(args.out / "transfer_outcome_joined.parquet", index=False, compression="zstd")

    cohorts = pd.DataFrame([
        _cohort_stats(joined, arm, cohort)
        for arm in ("p8u", "u50")
        for cohort in ("common", f"{arm}_only")
    ])
    buckets = pd.concat([_bucket_stats(joined, arm) for arm in ("p8u", "u50")], ignore_index=True)
    failures = pd.concat([_mc1_failure_stats(joined, arm) for arm in ("p8u", "u50")], ignore_index=True)
    transfer = pd.concat([_base_transfer(joined, arm) for arm in ("p8u", "u50")], ignore_index=True)
    meta_top1, u50_current_top1 = _meta_top1(joined, "current")
    bcf_top1, _ = _meta_top1(joined, "bcf")
    meta_top1 = pd.concat([meta_top1, bcf_top1], ignore_index=True)
    u50_good_buckets = _u50_good_map_buckets(joined)
    good_rows = []
    valid = joined["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(joined["policy_net_bps"])
    uonly = joined["cohort"].eq("u50_only")
    dual = pd.to_numeric(joined["u50_current_mc1_expected_bps"], errors="coerce").ge(50) & pd.to_numeric(joined["u50_bcf_mc1_expected_bps"], errors="coerce").ge(50)
    for threshold in (50.0, 100.0):
        good = uonly & valid & joined["policy_net_bps"].gt(threshold)
        for name, mask in (("all", good), ("mc1_rejected", good & ~dual), ("mc1_admitted", good & dual)):
            part = joined.loc[mask]
            good_rows.append({"definition": f"u50_only_policy_net_gt_{int(threshold)}", "mc1_state": name, "rows": int(len(part)),
                              "realised_ev_bps": float(part["policy_net_bps"].mean()),
                              "mean_current_map_bps": float(part["u50_current_mc1_expected_bps"].mean()),
                              "mean_bcf_map_bps": float(part["u50_bcf_mc1_expected_bps"].mean()),
                              "mean_base_top_pct": float(part["u50_base_top_pct"].mean()),
                              "mean_bcf_meta_top_pct": float(part["u50_bcf_meta_top_pct"].mean())})
    good = pd.DataFrame(good_rows)
    cohorts.to_parquet(args.out / "cohort_transfer_metrics.parquet", index=False, compression="zstd")
    buckets.to_parquet(args.out / "base_rank_bucket_metrics.parquet", index=False, compression="zstd")
    failures.to_parquet(args.out / "mc1_failure_metrics.parquet", index=False, compression="zstd")
    transfer.to_parquet(args.out / "base_utility_transfer_metrics.parquet", index=False, compression="zstd")
    meta_top1.to_parquet(args.out / "meta_top1_substitution_metrics.parquet", index=False, compression="zstd")
    u50_current_top1.to_parquet(args.out / "u50_current_meta_top1_substitutions.parquet", index=False, compression="zstd")
    u50_good_buckets.to_parquet(args.out / "u50_only_good_mc1_map_buckets.parquet", index=False, compression="zstd")
    good.to_parquet(args.out / "u50_only_good_transfer_metrics.parquet", index=False, compression="zstd")

    report = "# U50 Router Transfer Audit\n\n"
    report += "Offline diagnosis only. Scores were sealed in `transfer_trace_target_free.parquet`; canonical rich-policy outcomes were joined only afterwards. Period: April–July 2026.\n\n"
    report += "## Router cohort transfer\n\n" + _markdown(cohorts) + "\n\n"
    report += "## Base-rank buckets (percentile from best within the routed population)\n\n" + _markdown(buckets) + "\n\n"
    report += "## Utility retained from router-exclusive cohorts by Base cutoff\n\n" + _markdown(transfer) + "\n\n"
    report += "## Meta Top-1 substitutions\n\n" + _markdown(meta_top1) + "\n\n"
    report += "## MC1 failure states\n\n" + _markdown(failures) + "\n\n"
    report += "## U50-only realised >+50 winners by minimum dual-MC1 map\n\n" + _markdown(u50_good_buckets) + "\n\n"
    report += "## U50-only realised winners and dual-MC1 treatment\n\n" + _markdown(good) + "\n"
    (args.out / "TRANSFER_AUDIT_REPORT.md").write_text(report, encoding="utf-8")
    _write_json_exclusive(args.out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline report-only; no refit, live, or exchange effect",
        "period": list(MONTHS), "route": "exact timestamp-local Top-50%; descending Router score; candidate_id tie break",
        "mc1_gate": "separate Current and BCF maps, both >= +50 bps",
        "trace_then_label_join": True,
        "hashes": {
            "p8u_router": _receipt_hash(args.p8u_router),
            "u50_router": _receipt_hash(args.u50_router),
            "p8u_stack": _receipt_hash(args.p8u_stack),
            "u50_stack": _receipt_hash(args.u50_stack),
            "policy_labels": _sha256(args.policy_labels),
        },
    })


if __name__ == "__main__":
    main()
