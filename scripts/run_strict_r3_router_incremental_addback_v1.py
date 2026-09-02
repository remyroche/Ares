#!/usr/bin/env python3
"""Strict-OOF incremental add-back test for the full-universe Router pool.

The replacement compression ladder may fail even when new causal fields have
information conditional on the frozen 30-field Router.  This bounded stage
therefore retains the exact frozen control and adds 10/20/40/80/120 novel
fields from each of the predeclared stability, prescreen, and blended orders.
It uses the same strict labels, query weighting, folds, cheap RankXENDCG
configuration, and target-free-before-outcome scoring protocol as the ladder.

It selects no production contract.  Only candidates that beat the exact
frozen control while preserving both R50 and R100 count within one percentage
point can advance to HPO/downstream work.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import run_strict_r3_economic_recall_router as router  # noqa: E402
import run_strict_r3_router_full_universe_stability_v1 as stability  # noqa: E402
import run_strict_r3_router_subset_ladder_v1 as ladder  # noqa: E402


SCHEMA = "strict_r3_router_incremental_addback_v1"
SEED = 1729
ADDBACKS = (10, 20, 40, 80, 120)


def _write_once(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _hash_lines(values: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def run(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(f"immutable output exists: {args.out}")
    roots = ladder._roots(args.feature_roots)
    stable = ladder._load_list(args.stability_contract.resolve(), minimum=300, maximum=500)
    control = ladder._load_list(args.control_contract.resolve(), minimum=30, maximum=30)
    held_months = ladder._parse_months(args.held_months)
    orders = ladder._orders(stable=stable, stability_evidence=args.stability_evidence.resolve(), prescreen=args.prescreen.resolve())
    additions = {
        name: tuple(field for field in ordering if field not in set(control))
        for name, ordering in orders.items()
    }
    if any(len(values) < max(ADDBACKS) for values in additions.values()):
        raise AssertionError("stability shortlist has inadequate novel fields for add-back stage")
    policy = router._policy_window(
        args.policy.resolve(), held_months[0] - pd.DateOffset(months=args.train_months + 2),
        held_months[-1] + pd.offsets.MonthBegin(1),
    ).loc[:, ["candidate_id", "policy_path_valid", "policy_net_bps", "policy_gross_bps", "policy_label_available_ts"]].copy()
    all_fields = tuple(dict.fromkeys((*control, *stable)))
    args.out.mkdir(parents=True)
    _write_once(args.out / "run_contract.json", {
        "schema": SCHEMA,
        "scope": "offline strict-OOF Router incremental add-back; no live, exchange, base, consensus, MC1, or portfolio mutation",
        "feature_roots": [str(root) for root in roots], "control_contract": str(args.control_contract.resolve()),
        "stability_contract": str(args.stability_contract.resolve()), "control_feature_count": len(control),
        "addbacks": list(ADDBACKS), "orders": list(orders), "held_months": [f"{month:%Y-%m}" for month in held_months],
        "strict_train": {"train_months": args.train_months, "reserve_days": args.reserve_days, "label_available_before_reserve": True},
        "target": args.primary_target, "row_weight_scheme": args.row_weight_scheme,
        "advance_guards": {"r50_count_max_drop": .01, "r100_count_max_drop": .01, "requires_positive_sstable_delta": True},
        "target_free_held_scores_persist_before_metric_join": True,
    })
    candidates: list[tuple[str, tuple[str, ...]]] = [("frozen30_control", control)]
    for name, ordered in additions.items():
        for count in ADDBACKS:
            subset = tuple((*control, *ordered[:count]))
            candidates.append((f"{name}_plus{count:03d}", subset))
    metrics: list[dict[str, object]] = []
    membership: dict[str, tuple[str, ...]] = {name: subset for name, subset in candidates}
    score_root = args.out / "target_free_scores"
    for fold_index, held_month in enumerate(held_months):
        train, held, train_matrix, held_matrix, _medians = stability._prepare_fold(
            roots=roots, fields=all_fields, policy=policy, held_month=held_month,
            train_months=args.train_months, reserve_days=args.reserve_days,
            train_cap=args.train_cap, held_cap=args.held_cap,
        )
        fold_dir = score_root / f"fold={held_month:%Y-%m}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        for index, (name, subset) in enumerate(candidates):
            rank, weight_summary = ladder._fit_score(
                train=train, held=held, train_matrix=train_matrix, held_matrix=held_matrix,
                matrix_fields=all_fields, subset=subset, primary=args.primary_target,
                scheme=args.row_weight_scheme, seed=SEED + 10_000 * fold_index + index,
                n_jobs=args.n_jobs,
            )
            score = held.loc[:, list(ladder.IDENTITY)].copy()
            score["router_primary_rank"] = rank
            score.to_parquet(fold_dir / f"{name}.parquet", index=False, compression="zstd")
            summary = stability._metric(held, rank, policy)
            metrics.append({"held_month": f"{held_month:%Y-%m}", "candidate": name, "feature_count": len(subset), **summary, **weight_summary})
        with (args.out / "progress.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": "fold_complete", "held_month": f"{held_month:%Y-%m}", "candidates": len(candidates)}) + "\n")
        del train, held, train_matrix, held_matrix
        gc.collect()
    frame = pd.DataFrame(metrics)
    rows: list[dict[str, object]] = []
    for name, data in frame.groupby("candidate", sort=False):
        values = data.s_router.to_numpy(float)
        rows.append({
            "candidate": name, "feature_count": int(data.feature_count.iloc[0]), "folds": len(data),
            "s_router_mean": float(values.mean()), "s_router_q25": float(data.s_router.quantile(.25)),
            "s_router_worst": float(values.min()),
            "s_stable": float(.65 * values.mean() + .25 * data.s_router.quantile(.25) + .10 * values.min()),
            "r50_count_mean": float(data.r50_count.mean()), "r100_count_mean": float(data.r100_count.mean()),
            "r50_utility_mean": float(data.r50_utility.mean()), "r100_count_q25": float(data.r100_count.quantile(.25)),
        })
    summary = pd.DataFrame(rows)
    control_row = summary.loc[summary.candidate.eq("frozen30_control")].iloc[0]
    summary["delta_s_stable_vs_control"] = summary.s_stable - float(control_row.s_stable)
    summary["delta_r50_count_vs_control"] = summary.r50_count_mean - float(control_row.r50_count_mean)
    summary["delta_r100_count_vs_control"] = summary.r100_count_mean - float(control_row.r100_count_mean)
    summary["passes_advance_guard"] = (
        summary.candidate.ne("frozen30_control")
        & summary.delta_s_stable_vs_control.gt(0.0)
        & summary.delta_r50_count_vs_control.ge(-.01)
        & summary.delta_r100_count_vs_control.ge(-.01)
    )
    summary = summary.sort_values(["passes_advance_guard", "s_stable", "candidate"], ascending=[False, False, True], kind="stable")
    finalists = summary.loc[summary.passes_advance_guard].head(3)
    final_contracts = [
        {
            "rank": position + 1, "candidate": item.candidate, "feature_contract": list(membership[item.candidate]),
            "feature_contract_sha256": _hash_lines(membership[item.candidate]), "s_stable": float(item.s_stable),
            "delta_s_stable_vs_control": float(item.delta_s_stable_vs_control),
        }
        for position, item in enumerate(finalists.itertuples(index=False))
    ]
    frame.to_parquet(args.out / "fold_metrics.parquet", index=False, compression="zstd")
    summary.to_parquet(args.out / "addback_summary.parquet", index=False, compression="zstd")
    pd.DataFrame(
        {"candidate": name, "position": position, "feature": field}
        for name, subset in membership.items() for position, field in enumerate(subset, 1)
    ).to_parquet(args.out / "candidate_membership.parquet", index=False, compression="zstd")
    _write_once(args.out / "finalists.json", {
        "schema": SCHEMA, "scope": "research-only Router incremental add-back finalists; requires HPO then downstream test",
        "control": {"feature_contract": list(control), "feature_contract_sha256": _hash_lines(control), "s_stable": float(control_row.s_stable)},
        "finalists": final_contracts, "advance_guards": {"r50_count_max_drop": .01, "r100_count_max_drop": .01, "requires_positive_sstable_delta": True},
    })
    _write_once(args.out / "run_manifest.json", {
        "schema": SCHEMA, "status": "complete", "candidates": len(candidates), "finalists": len(final_contracts),
        "held_months": [f"{month:%Y-%m}" for month in held_months],
        "scope": "offline Router incremental add-back complete; no live/exchange mutation",
    })
    print(json.dumps({"event": "complete", "finalists": len(final_contracts), "best_s_stable": float(summary.iloc[0].s_stable)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-roots", required=True)
    parser.add_argument("--stability-contract", type=Path, required=True)
    parser.add_argument("--stability-evidence", type=Path, required=True)
    parser.add_argument("--prescreen", type=Path, required=True)
    parser.add_argument("--control-contract", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=router.DEFAULT_POLICY)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--held-months", default="2025-10,2026-02,2026-06")
    parser.add_argument("--primary-target", default="U50_p050_c300", choices=router.ALL_PRIMARY_TARGETS)
    parser.add_argument("--row-weight-scheme", default="positive_125", choices=router._ROW_WEIGHT_SCHEMES)
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--train-cap", type=int, default=30_000)
    parser.add_argument("--held-cap", type=int, default=12_000)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    if args.train_months < 3 or args.reserve_days < 28 or args.train_cap < 20_000 or args.held_cap < 5_000:
        raise ValueError("strict Router add-back support below predeclared minimum")
    run(args)


if __name__ == "__main__":
    main()
