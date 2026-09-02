#!/usr/bin/env python3
"""Evaluate router route fractions from an immutable target-free score ledger.

No model is fitted here.  Scores are read first from an existing completed
strict-OOF router artifact; policy outcomes are joined only to evaluate the
predeclared 30--50% routing frontier.  Symbol concentration is an evaluation
diagnostic and never a score-time input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_strict_r3_economic_recall_router as rr  # noqa: E402


def _fractions(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not result or tuple(sorted(set(result))) != result or any(not .0 < item <= 1.0 for item in result):
        raise argparse.ArgumentTypeError("fractions must be unique sorted values in (0, 1]")
    return result


def _write(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    rr._write_json_exclusive(path, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, default=rr.DEFAULT_FEATURE_ROOT)
    parser.add_argument("--policy-path", type=Path, default=rr.DEFAULT_POLICY)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fractions", type=_fractions, default=(.30, .35, .40, .45, .50))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    contract = json.loads((args.source / "run_contract.json").read_text())
    manifest = json.loads((args.source / "run_manifest.json").read_text())
    if manifest.get("status") != "complete" or contract.get("schema") != rr.SCHEMA:
        raise AssertionError("source must be a completed strict-router artifact")
    months = tuple(rr._utc(token + "-01") for token in contract["months"])
    policy = rr._policy_window(args.policy_path, months[0] - pd.DateOffset(months=contract["train_months"] + 2), rr._month_end(months[-1]))
    args.out.mkdir(parents=True)
    all_metrics: list[pd.DataFrame] = []
    concentration: list[dict[str, object]] = []
    for month in months:
        score = pd.read_parquet(args.source / "target_free_scores" / f"month={month:%Y-%m}.parquet")
        allowed = {"candidate_id", "__decision_ts__", "side_name"}
        if any(column not in allowed and not column.startswith("router_") for column in score):
            raise AssertionError("source score panel includes an undeclared non-score field")
        metric_rows, timestamp = rr._metric_rows(score, policy, month, args.fractions)
        # A primary-only score ledger exposes ``router_full_ae_rank`` as an
        # intentional alias for schema compatibility.  It is not a second
        # route.  Keep only the declared primary coordinate in this audit.
        metric_frame = pd.DataFrame(metric_rows)
        metric_frame = metric_frame.loc[metric_frame["score"].eq("router_primary_only_rank")].copy()
        timestamp = timestamp.loc[timestamp["score"].eq("router_primary_only_rank")].copy()
        all_metrics.append(metric_frame)
        timestamp.to_parquet(args.out / f"timestamp_metrics_month={month:%Y-%m}.parquet", index=False, compression="zstd")
        # Candidate identity is the target-free canonical
        # ``SYMBOL|SIDE|SIGNAL_TIMESTAMP`` key.  The source feature panel
        # intentionally does not carry a redundant symbol field, so recover
        # the symbol from that identity rather than joining a different
        # candidate universe (and therefore changing concentration support).
        parsed = score["candidate_id"].astype(str).str.rsplit("|", n=2, expand=True)
        if parsed.shape[1] != 3 or parsed.isna().any().any() or not parsed.iloc[:, 1].eq(score["side_name"].astype(str)).all():
            raise AssertionError("target-free candidate identity does not satisfy SYMBOL|SIDE|SIGNAL_TIMESTAMP")
        score["__symbol__"] = parsed.iloc[:, 0].astype(str)
        joined = score.merge(policy, on="candidate_id", how="left", validate="one_to_one")
        valid = joined["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(pd.to_numeric(joined["policy_net_bps"], errors="coerce"))
        for fraction in args.fractions:
            selected = rr._route_rank(joined, "router_primary_only_rank", fraction)
            counts = joined.loc[selected & valid].groupby("__symbol__", sort=False).size()
            total = int(counts.sum())
            shares = counts.to_numpy(float) / total if total else np.asarray([], dtype=float)
            concentration.append({
                "held_month": month.strftime("%Y-%m"), "route_fraction": fraction,
                "selected_valid_rows": total, "symbols": int(len(counts)),
                "largest_symbol_share": float(np.max(shares)) if len(shares) else np.nan,
                "symbol_hhi": float(np.square(shares).sum()) if len(shares) else np.nan,
                "effective_symbols": float(1.0 / np.square(shares).sum()) if len(shares) and np.square(shares).sum() else np.nan,
            })
    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_parquet(args.out / "route_pareto_metrics.parquet", index=False, compression="zstd")
    rr._aggregate_metrics(metrics).to_parquet(args.out / "route_pareto_summary.parquet", index=False, compression="zstd")
    pd.DataFrame(concentration).to_parquet(args.out / "route_symbol_concentration.parquet", index=False, compression="zstd")
    _write(args.out / "run_manifest.json", {
        "schema": "strict_r3_router_route_pareto_v1", "status": "complete", "source": str(args.source),
        "source_contract_sha256": rr._sha256_file(args.source / "run_contract.json"), "months": [month.strftime("%Y-%m") for month in months],
        "fractions": list(args.fractions), "causality": "target-free score identities fixed before policy outcome join; symbol concentration diagnostic only",
        "selection": "Pareto evidence only; no route is promoted until constrained downstream replay", "outputs": ["route_pareto_metrics.parquet", "route_pareto_summary.parquet", "route_symbol_concentration.parquet"],
    })


if __name__ == "__main__":
    main()
