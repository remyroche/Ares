#!/usr/bin/env python3
"""Cross-fold subset-compression ladder for the strict-R3 Router shortlist.

The preceding full-universe stages deliver a 300-field stability shortlist.
This script performs the predeclared 130/110/100/80/60/50/40/30 compression
ladder.  At every width it evaluates three deterministic, predeclared order
families (stability, prescreen, and their rank blend) on the same strict OOF
folds, carrying the top three configurations as the bounded beam.  It also
fits the frozen 30-field contract under the identical cheap model as a
control.  It persists target-free held score receipts before the rich-policy
join used exclusively for OOF measurement.

It is a research-only selector: no live, exchange, base, consensus, MC1, or
portfolio artifact is modified.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import run_strict_r3_economic_recall_router as router  # noqa: E402
import screen_strict_r3_router_full_universe_v1 as screen  # noqa: E402
import run_strict_r3_router_full_universe_stability_v1 as stability  # noqa: E402


SCHEMA = "strict_r3_router_subset_ladder_v1"
SEED = 1729
IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
WIDTHS = (130, 110, 100, 80, 60, 50, 40, 30)


def _write_once(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _hash_lines(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _load_list(path: Path, *, field: str = "feature_contract", minimum: int, maximum: int) -> tuple[str, ...]:
    payload = json.loads(path.read_text())
    values = tuple(map(str, payload.get(field, ())))
    if not minimum <= len(values) <= maximum or len(values) != len(set(values)):
        raise AssertionError(f"invalid feature contract {path}")
    hash_field = f"{field}_sha256"
    if hash_field in payload and payload[hash_field] != _hash_lines(values):
        raise AssertionError(f"feature hash mismatch for {path}")
    return values


def _parse_months(value: str) -> tuple[pd.Timestamp, ...]:
    values = tuple(pd.Timestamp(f"{token.strip()}-01", tz="UTC") for token in value.split(",") if token.strip())
    if len(values) != 3 or tuple(sorted(values)) != values or len(set(values)) != 3:
        raise ValueError("need exactly three chronological held months")
    return values


def _roots(value: str) -> tuple[Path, ...]:
    roots = tuple(Path(token.strip()).resolve() for token in value.split(",") if token.strip())
    if not roots or len(roots) != len(set(roots)):
        raise ValueError("feature roots must be a non-empty unique list")
    return roots


def _orders(*, stable: tuple[str, ...], stability_evidence: Path, prescreen: Path) -> dict[str, tuple[str, ...]]:
    stability_frame = pd.read_parquet(stability_evidence)
    stability_order = [field for field in stability_frame.sort_values(
        ["stability_score", "mean_r50_utility_delta", "feature"], ascending=[False, False, True], kind="stable",
    ).feature.astype(str) if field in set(stable)]
    prescreen_frame = pd.read_parquet(prescreen)
    prescreen_order = [field for field in prescreen_frame.sort_values(
        ["screen_score", "feature"], ascending=[False, True], kind="stable",
    ).feature.astype(str) if field in set(stable)]
    if set(stability_order) != set(stable) or set(prescreen_order) != set(stable):
        raise AssertionError("evidence/order does not cover the exact stability shortlist")
    stable_rank = {field: index for index, field in enumerate(stability_order)}
    screen_rank = {field: index for index, field in enumerate(prescreen_order)}
    blended = tuple(sorted(stable, key=lambda field: (.75 * stable_rank[field] + .25 * screen_rank[field], field)))
    return {"stability": tuple(stability_order), "prescreen": tuple(prescreen_order), "blend75_25": blended}


def _fit_score(
    *, train: pd.DataFrame, held: pd.DataFrame, train_matrix: np.ndarray, held_matrix: np.ndarray,
    matrix_fields: tuple[str, ...], subset: tuple[str, ...], primary: str, scheme: str,
    seed: int, n_jobs: int,
) -> tuple[np.ndarray, dict[str, float]]:
    target = router._primary_target(train, primary).astype(np.int32)
    work = train.copy()
    work["__target__"] = target
    utility = router._primary_weight_utility(work, primary, target)
    ordered, groups, weights, weight_summary = router._query_weights(work, scheme=scheme, primary_utility=utility)
    indices = np.asarray([matrix_fields.index(field) for field in subset], dtype=np.int64)
    rows = ordered["__row__"].to_numpy(np.int64)
    model = LGBMRanker(
        objective="rank_xendcg", metric="ndcg", label_gain=[0, 1, 2, 4, 7, 11],
        n_estimators=300, learning_rate=.045, max_depth=4, num_leaves=15,
        min_child_samples=max(300, int(.012 * len(rows))), min_split_gain=.002,
        subsample=.78, subsample_freq=1, colsample_bytree=.78,
        reg_alpha=.02, reg_lambda=1.5, max_bin=127,
        lambdarank_truncation_level=12, random_state=seed, n_jobs=n_jobs,
        deterministic=True, force_col_wise=True, verbosity=-1,
    )
    model.fit(train_matrix[rows][:, indices], work.iloc[rows]["__target__"].to_numpy(np.int32), group=groups, sample_weight=weights)
    train_raw = model.predict(train_matrix[rows][:, indices]).astype(np.float32)
    held_raw = model.predict(held_matrix[:, indices]).astype(np.float32)
    rank = screen._rank_reference(train_raw, held_raw)
    del model, train_raw, held_raw
    gc.collect()
    return rank, weight_summary


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (width, order), data in frame.groupby(["width", "order"], sort=False):
        values = data.s_router.to_numpy(float)
        rows.append({
            "width": int(width), "order": str(order), "folds": len(data),
            "s_router_mean": float(values.mean()), "s_router_q25": float(np.quantile(values, .25)),
            "s_router_worst": float(values.min()),
            "s_stable": float(.65 * values.mean() + .25 * np.quantile(values, .25) + .10 * values.min()),
            "r50_utility_mean": float(data.r50_utility.mean()), "r50_utility_q25": float(data.r50_utility.quantile(.25)),
            "r50_count_mean": float(data.r50_count.mean()), "r100_count_mean": float(data.r100_count.mean()),
            "r200_count_mean": float(data.r200_count.mean()),
        })
    return pd.DataFrame(rows).sort_values(["s_stable", "width", "order"], ascending=[False, True, True], kind="stable")


def run(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(f"immutable output already exists: {args.out}")
    roots = _roots(args.feature_roots)
    stable = _load_list(args.stability_contract.resolve(), minimum=300, maximum=500)
    control = _load_list(args.control_contract.resolve(), minimum=30, maximum=30)
    hygienic = set(map(str, json.loads(args.hygiene_contract.read_text())["feature_contract"]))
    # The frozen control predates the explicit 95%-coverage gate.  Four of its
    # fields are causal but only 93--95% observed in the new broad source;
    # retain that exact control with its original train-median imputation so
    # the comparison is honest, while separately reporting the 26-field
    # hygiene intersection.  Candidate Router subsets below never include
    # these legacy-exception fields.
    legacy_control_exceptions = tuple(field for field in control if field not in hygienic)
    hygiene_control = tuple(field for field in control if field in hygienic)
    if len(hygiene_control) < 20:
        raise AssertionError("frozen control hygiene intersection unexpectedly small")
    held_months = _parse_months(args.held_months)
    orders = _orders(stable=stable, stability_evidence=args.stability_evidence.resolve(), prescreen=args.prescreen.resolve())
    policy = router._policy_window(
        args.policy.resolve(), held_months[0] - pd.DateOffset(months=args.train_months + 2),
        held_months[-1] + pd.offsets.MonthBegin(1),
    ).loc[:, ["candidate_id", "policy_path_valid", "policy_net_bps", "policy_gross_bps", "policy_label_available_ts"]].copy()
    all_fields = tuple(dict.fromkeys((*stable, *control)))
    args.out.mkdir(parents=True)
    _write_once(args.out / "run_contract.json", {
        "schema": SCHEMA,
        "scope": "offline strict-OOF Router subset ladder; no live, exchange, base, consensus, MC1, or portfolio mutation",
        "feature_roots": [str(root) for root in roots], "stability_contract": str(args.stability_contract.resolve()),
        "control_contract": str(args.control_contract.resolve()), "stability_feature_count": len(stable),
        "control_feature_count": len(control), "hygiene_control_feature_count": len(hygiene_control),
        "legacy_control_coverage_exceptions": list(legacy_control_exceptions),
        "held_months": [f"{value:%Y-%m}" for value in held_months],
        "widths": list(WIDTHS), "beam_width": 3, "order_families": list(orders),
        "strict_train": {"train_months": args.train_months, "reserve_days": args.reserve_days, "label_available_before_reserve": True},
        "target": args.primary_target, "row_weight_scheme": args.row_weight_scheme,
        "model": "cheap rank_xendcg d4/l15/300 trees; HPO deferred until feature-width finalist",
        "target_free_held_scores_persist_before_metric_join": True,
    })
    metrics_rows: list[dict[str, object]] = []
    # Membership is a contract-level definition, not a fold-level statistic.
    # Keep one ordered record per candidate set, then assert that every fold
    # consumed that same exact definition below.
    subset_definitions: dict[tuple[int, str], tuple[str, ...]] = {}
    score_root = args.out / "target_free_scores"
    for fold_index, held_month in enumerate(held_months):
        train, held, train_matrix, held_matrix, _medians = stability._prepare_fold(
            roots=roots, fields=all_fields, policy=policy, held_month=held_month,
            train_months=args.train_months, reserve_days=args.reserve_days,
            train_cap=args.train_cap, held_cap=args.held_cap,
        )
        candidates: list[tuple[int, str, tuple[str, ...]]] = [
            (30, "frozen30_control", control),
            (len(hygiene_control), "hygiene26_control", hygiene_control),
        ]
        for width in WIDTHS:
            for order_name, ordering in orders.items():
                subset = tuple(ordering[:width])
                candidates.append((width, order_name, subset))
        for candidate_index, (width, order_name, subset) in enumerate(candidates):
            previous = subset_definitions.setdefault((width, order_name), subset)
            if previous != subset:
                raise AssertionError(f"fold-specific subset definition drift for {width}/{order_name}")
            rank, weight_summary = _fit_score(
                train=train, held=held, train_matrix=train_matrix, held_matrix=held_matrix,
                matrix_fields=all_fields, subset=subset, primary=args.primary_target,
                scheme=args.row_weight_scheme, seed=SEED + 1_000 * fold_index + candidate_index,
                n_jobs=args.n_jobs,
            )
            score = held.loc[:, list(IDENTITY)].copy()
            score["router_primary_rank"] = rank
            fold_score_root = score_root / f"fold={held_month:%Y-%m}"
            fold_score_root.mkdir(parents=True, exist_ok=True)
            score.to_parquet(fold_score_root / f"width={width:03d}__{order_name}.parquet", index=False, compression="zstd")
            summary = stability._metric(held, rank, policy)
            metrics_rows.append({"held_month": f"{held_month:%Y-%m}", "width": width, "order": order_name, "feature_count": len(subset), **summary, **weight_summary})
        with (args.out / "progress.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": "fold_complete", "held_month": f"{held_month:%Y-%m}", "candidates": len(candidates)}) + "\n")
        del train, held, train_matrix, held_matrix
        gc.collect()
    metric_frame = pd.DataFrame(metrics_rows)
    summary = _summary(metric_frame)
    control_row = summary.loc[summary.order.eq("frozen30_control")].iloc[0]
    summary["delta_s_stable_vs_control"] = summary.s_stable - float(control_row.s_stable)
    summary["delta_r50_utility_vs_control"] = summary.r50_utility_mean - float(control_row.r50_utility_mean)
    # The beam is predeclared as the best three width/order combinations by
    # cross-fold stable utility.  Later HPO starts only from these exact sets.
    finalists = summary.loc[~summary.order.isin(["frozen30_control", "hygiene26_control"])].head(3).copy()
    finalist_contracts: list[dict[str, object]] = []
    for rank_index, item in finalists.reset_index(drop=True).iterrows():
        subset = list(subset_definitions[(int(item.width), str(item.order))])
        finalist_contracts.append({
            "rank": int(rank_index + 1), "width": int(item.width), "order": str(item.order),
            "feature_contract": subset, "feature_contract_sha256": _hash_lines(subset),
            "s_stable": float(item.s_stable), "delta_s_stable_vs_control": float(item.delta_s_stable_vs_control),
        })
    metric_frame.to_parquet(args.out / "fold_metrics.parquet", index=False, compression="zstd")
    summary.to_parquet(args.out / "ladder_summary.parquet", index=False, compression="zstd")
    subset_membership = pd.DataFrame(
        {"width": width, "order": order, "feature": feature, "position": position}
        for (width, order), subset in subset_definitions.items()
        for position, feature in enumerate(subset, 1)
    )
    subset_membership.to_parquet(args.out / "subset_membership.parquet", index=False, compression="zstd")
    _write_once(args.out / "beam_finalists.json", {
        "schema": SCHEMA, "scope": "research-only Router subset-ladder finalists; needs HPO and downstream test",
        "control": {
            "feature_contract": list(control), "feature_contract_sha256": _hash_lines(control),
            "s_stable": float(control_row.s_stable), "legacy_coverage_exceptions": list(legacy_control_exceptions),
            "hygiene_intersection_contract": list(hygiene_control), "hygiene_intersection_sha256": _hash_lines(hygiene_control),
        },
        "finalists": finalist_contracts,
        "selection": "top three cross-fold Sstable width/order configurations from predeclared 130..30 ladder",
    })
    _write_once(args.out / "run_manifest.json", {
        "schema": SCHEMA, "status": "complete", "held_months": [f"{value:%Y-%m}" for value in held_months],
        "candidates_per_fold": 2 + len(WIDTHS) * 3, "finalists": len(finalist_contracts),
        "scope": "offline Router subset ladder complete; no live/exchange mutation",
    })
    print(json.dumps({"event": "complete", "finalists": len(finalist_contracts), "best_s_stable": float(finalists.iloc[0].s_stable)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-roots", required=True)
    parser.add_argument("--hygiene-contract", type=Path, required=True)
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
        raise ValueError("strict Router ladder support below predeclared minimum")
    run(args)


if __name__ == "__main__":
    main()
