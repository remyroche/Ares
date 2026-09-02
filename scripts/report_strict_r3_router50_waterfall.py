#!/usr/bin/env python3
"""Produce a target-free router-to-portfolio waterfall for the selected stack.

The report deliberately separates *ranking diagnostics* from causal admission
and the chronological constrained portfolio.  Oracle rows are labelled
post-hoc and are never used to choose a model, threshold, or live policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


START = pd.Timestamp("2026-04-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
ROUTE_FRACTION = 0.50


def _months() -> list[str]:
    return [f"{value:%Y-%m}" for value in pd.date_range(START, END - pd.Timedelta(nanoseconds=1), freq="MS", tz="UTC")]


def _topn(frame: pd.DataFrame, score: str, n: int = 2) -> pd.DataFrame:
    ordered = frame.sort_values(["__decision_ts__", score, "candidate_id"], ascending=[True, False, True], kind="stable")
    return ordered.groupby("__decision_ts__", sort=False, group_keys=False).head(n).copy()


def _ranking_metrics(frame: pd.DataFrame, score: str, label: str) -> dict[str, object]:
    chosen = _topn(frame, score)
    by_timestamp = chosen.groupby("__decision_ts__", sort=False)["policy_net_bps"].mean()
    return {
        "stage": label,
        "kind": "ranking diagnostic" if score != "policy_net_bps" else "post-hoc oracle diagnostic",
        "candidate_rows": int(len(frame)),
        "timestamps": int(frame["__decision_ts__"].nunique()),
        "top2_rows": int(len(chosen)),
        "top2_timestamp_net_ev_bps": float(by_timestamp.mean()),
        "top2_total_net_bps": float(chosen["policy_net_bps"].sum()),
        "top2_precision_gt50": float(chosen["policy_net_bps"].gt(50.0).mean()),
    }


def _admission_oracle_metrics(frame: pd.DataFrame, label: str) -> dict[str, object]:
    """Post-hoc ceiling within an already causal-admitted population.

    This is deliberately separate from ``_ranking_metrics``: it ranks the
    available candidates by their eventual policy outcome only to quantify
    remaining selection opportunity after an admission gate.  It has no score
    authority and must never be used for model or threshold selection.
    """
    valid = frame.loc[
        frame["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
    ].copy()
    chosen = _topn(valid, "policy_net_bps")
    by_timestamp = chosen.groupby("__decision_ts__", sort=False)["policy_net_bps"].mean()
    by_month = chosen.groupby(chosen["__decision_ts__"].dt.to_period("M"), sort=False)["policy_net_bps"].mean()
    return {
        "stage": label,
        "kind": "post-hoc oracle after causal admission",
        "candidate_rows": int(len(valid)),
        "timestamps": int(valid["__decision_ts__"].nunique()),
        "top2_rows": int(len(chosen)),
        "top2_timestamp_net_ev_bps": float(by_timestamp.mean()),
        "top2_total_net_bps": float(chosen["policy_net_bps"].sum()),
        "top2_precision_gt50": float(chosen["policy_net_bps"].gt(50.0).mean()),
        "worst_month_bps": float(by_month.min()),
    }


def _valid_policy(path: Path) -> pd.DataFrame:
    wanted = ["candidate_id", "policy_path_valid", "policy_net_bps"]
    work = pd.read_parquet(path, columns=wanted)
    work["policy_net_bps"] = pd.to_numeric(work["policy_net_bps"], errors="coerce")
    return work.loc[work["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(work["policy_net_bps"])].copy()


def _read_base(root: Path, router_root: Path, policy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_parts, route_parts = [], []
    for token in _months():
        base_parts.append(pd.read_parquet(
            root / "target_free_monthly" / f"month={token}" / "scores_features.parquet",
            columns=["candidate_id", "__decision_ts__", "side_name", "enhanced_base_bps"],
        ))
        route_parts.append(pd.read_parquet(
            router_root / "target_free_scores" / f"month={token}.parquet",
            columns=["candidate_id", "__decision_ts__", "side_name", "router_primary_rank"],
        ))
    base = pd.concat(base_parts, ignore_index=True)
    route = pd.concat(route_parts, ignore_index=True)
    for work in (base, route):
        work["__decision_ts__"] = pd.to_datetime(work["__decision_ts__"], utc=True, errors="raise")
        if work["candidate_id"].duplicated().any():
            raise AssertionError("duplicate target-free candidate identity")
    route = route.sort_values(["__decision_ts__", "router_primary_rank", "candidate_id"], ascending=[True, False, True], kind="stable")
    route["routed"] = route.groupby("__decision_ts__", sort=False).cumcount().lt(
        route.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").mul(ROUTE_FRACTION).pipe(np.ceil).astype(int).clip(lower=1)
    )
    common = base.merge(route[["candidate_id", "routed"]], on="candidate_id", how="inner", validate="one_to_one")
    common = common.merge(policy, on="candidate_id", how="inner", validate="one_to_one")
    return common.copy(), common.loc[common["routed"].astype(bool)].copy()


def _read_final(root: Path, policy: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for token in _months():
        path = root / "target_free_scores" / "bcf" / f"month={token}.parquet"
        parts.append(pd.read_parquet(path, columns=["candidate_id", "__decision_ts__", "final_score"]))
    out = pd.concat(parts, ignore_index=True)
    out["__decision_ts__"] = pd.to_datetime(out["__decision_ts__"], utc=True, errors="raise")
    return out.merge(policy, on="candidate_id", how="inner", validate="one_to_one")


def _read_mc1(root: Path, policy: pd.DataFrame) -> pd.DataFrame:
    out = pd.read_parquet(root / "dual_mc1_predictions.parquet")
    out["__decision_ts__"] = pd.to_datetime(out["__decision_ts__"], utc=True, errors="raise")
    out = out.loc[out["__decision_ts__"].ge(START) & out["__decision_ts__"].lt(END)].copy()
    # The final dual-MC1 panel is intentionally outcome-joined for replay,
    # whereas the current/BCF score panels above are target-free.  Do not
    # merge the ledger a second time: that would create policy_net_bps_x/y and
    # silently make the waterfall depend on an ambiguous column.  Instead
    # prove the existing replay outcome exactly matches the canonical ledger.
    if {"policy_path_valid", "policy_net_bps"}.issubset(out.columns):
        reference = policy.rename(columns={"policy_path_valid": "__policy_valid__", "policy_net_bps": "__policy_net__"})
        check = out.loc[:, ["candidate_id", "policy_path_valid", "policy_net_bps"]].merge(
            reference, on="candidate_id", how="left", validate="one_to_one"
        )
        valid = check["__policy_valid__"].fillna(False).astype(bool)
        same = np.isclose(
            pd.to_numeric(check.loc[valid, "policy_net_bps"], errors="coerce"),
            pd.to_numeric(check.loc[valid, "__policy_net__"], errors="coerce"),
            equal_nan=True,
        )
        if not bool(np.all(same)):
            raise AssertionError("dual-MC1 replay outcomes differ from canonical policy ledger")
        return out
    return out.merge(policy, on="candidate_id", how="inner", validate="one_to_one")


def _portfolio(root: Path, threshold: int) -> dict[str, object]:
    table = pd.read_parquet(root / "portfolio_metrics.parquet")
    row = table.loc[pd.to_numeric(table["threshold_bps"], errors="coerce").eq(float(threshold))].iloc[0]
    return {
        "stage": f"dual MC1 >= {threshold} bps + portfolio",
        "kind": "causal admission plus chronological constrained portfolio",
        "candidate_rows": int(row["candidate_admitted_rows"]),
        "timestamps": np.nan,
        "top2_rows": int(row["accepted_rows"]),
        "top2_timestamp_net_ev_bps": float(row["net_ev_bps_per_realised_trade"]),
        "top2_total_net_bps": float(row["net_sum_bps_realised"]),
        "top2_precision_gt50": np.nan,
        "worst_month_bps": float(row["worst_month_bps"]),
        "worst_week_bps": float(row["worst_week_bps"]),
        "max_drawdown": float(row["max_drawdown"]),
    }


def _markdown(table: pd.DataFrame) -> str:
    visible = table.copy()
    for column in ("top2_timestamp_net_ev_bps", "top2_total_net_bps", "top2_precision_gt50", "worst_month_bps", "worst_week_bps", "max_drawdown"):
        if column in visible:
            visible[column] = visible[column].map(lambda value: "—" if not np.isfinite(value) else (f"{value:.2%}" if column == "max_drawdown" or column == "top2_precision_gt50" else f"{value:,.2f}"))
    # Keep this report dependency-free: the workspace Python does not bundle
    # pandas' optional ``tabulate`` package.
    columns = list(visible.columns)
    def cell(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for values in visible.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(cell(value) for value in values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-base-root", type=Path, required=True)
    parser.add_argument("--routed-base-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--mc1-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    policy = _valid_policy(args.policy_path)
    full_base, full_base_routed = _read_base(args.full_base_root, args.router_root, policy)
    _, routed_base = _read_base(args.routed_base_root, args.router_root, policy)
    final_score = _read_final(args.score_root, policy)
    mc1 = _read_mc1(args.mc1_root, policy)
    if not set(final_score["candidate_id"]).issubset(set(routed_base["candidate_id"])):
        raise AssertionError("T6/T9 score source contains candidates outside router top-50%")
    rows = [
        _ranking_metrics(full_base, "enhanced_base_bps", "enhanced base, full universe"),
        _ranking_metrics(full_base_routed, "enhanced_base_bps", "full-trained enhanced base after router top-50%"),
        _ranking_metrics(routed_base, "enhanced_base_bps", "selected routed-trained enhanced base"),
        _ranking_metrics(routed_base, "policy_net_bps", "oracle after router top-50%"),
        _ranking_metrics(final_score, "final_score", "selected T6/T9 BCF score"),
    ]
    for threshold in (30, 40, 50):
        dual = mc1.loc[
            pd.to_numeric(mc1["current_mc1_expected_bps"], errors="coerce").ge(threshold)
            & pd.to_numeric(mc1["bcf_mc1_expected_bps"], errors="coerce").ge(threshold)
        ].copy()
        rows.append(_ranking_metrics(dual, "bcf_mc1_expected_bps", f"dual MC1 >= {threshold} bps; no cross-time constraints"))
        rows.append(_admission_oracle_metrics(dual, f"oracle after dual MC1 >= {threshold} bps"))
        rows.append(_portfolio(args.mc1_root, threshold))
    table = pd.DataFrame(rows)
    table.to_parquet(args.out / "waterfall_metrics.parquet", index=False, compression="zstd")
    text = "# Router-50 Waterfall (research only)\n\n"
    text += "Period: April–July 2026. Rich-policy outcomes are joined after target-free scores. Oracle rows are diagnostics only.\n\n"
    text += _markdown(table) + "\n"
    (args.out / "pipeline_metrics_with_router.md").write_text(text)
    (args.out / "run_manifest.json").write_text(json.dumps({
        "scope": "offline research only", "period": [str(START), str(END)],
        "route_fraction": ROUTE_FRACTION, "full_base_root": str(args.full_base_root),
        "routed_base_root": str(args.routed_base_root), "router_root": str(args.router_root),
        "score_root": str(args.score_root), "mc1_root": str(args.mc1_root),
        "policy": str(args.policy_path),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
