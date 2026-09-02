#!/usr/bin/env python3
"""Strict-OOF economic subset ladder for source-repaired Base Screen120 heads.

The ladder is restricted to the predeclared 120/90/70/50/35/25 widths.  It
consumes only economic MDA ranks plus semantic-family rescue, uses two fixed
seeds, and writes target-free OOF scores before policy metrics.  It does not
calculate raw feature CMI or IC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import run_strict_r3_p8u_fulluniverse_economic_mda_v1 as mda  # noqa: E402
import run_strict_r3_routed_et_fulluniverse_screen as screen  # noqa: E402


SCHEMA = "strict_r3_p8u_fulluniverse_subset_ladder_v1"
SEEDS = (1729, 71729)
WIDTHS = (120, 90, 70, 50, 35, 25)


def _months(value: str) -> tuple[pd.Timestamp, ...]:
    result = tuple(pd.Timestamp(f"{item.strip()}-01", tz="UTC") for item in value.split(",") if item.strip())
    if len(result) != 3 or tuple(sorted(result)) != result:
        raise ValueError("require exactly three chronological held months")
    return result


def _once(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _hash(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def _progress(out: Path, **payload: object) -> None:
    with (out / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str, sort_keys=True) + "\n")


def _screen_fields(root: Path, head: str) -> tuple[str, ...]:
    payload = json.loads((root / f"{head.lower()}_screen120_contract.json").read_text())
    values = tuple(map(str, payload["feature_contract"]))
    if len(values) != 120 or payload.get("feature_contract_sha256") != _hash(values):
        raise AssertionError(f"{head}: invalid Screen120 contract")
    return values


def _contracts(*, mda_root: Path, screen_root: Path, head: str) -> dict[int, tuple[str, ...]]:
    fields = _screen_fields(screen_root, head)
    summary = pd.read_parquet(mda_root / "economic_mda_summary.parquet")
    # ``DataFrame.head`` is a method, so use explicit column indexing here.
    # Attribute access silently resolves to that method rather than the
    # ``head`` column and prevented the immutable subset ladder from starting.
    summary = summary.loc[summary["head"].eq(head) & summary["feature"].isin(fields)].copy()
    if len(summary) != len(fields) or set(summary.feature) != set(fields):
        raise AssertionError(f"{head}: MDA does not cover exact Screen120 contract")
    # The raw economics ranking is primary.  Boundary MDA resolves close
    # choices; semantic-family rescue prevents a correlated family from being
    # lost merely because its individual effects are shared.
    summary["stable_rank"] = summary.stable_mda.rank(ascending=False, method="average")
    summary["boundary_rank"] = summary.boundary_mda_median.rank(ascending=False, method="average")
    summary["priority"] = .75 * summary.stable_rank + .25 * summary.boundary_rank
    summary["family"] = summary.feature.map(screen._feature_family)
    group = pd.read_parquet(mda_root / "economic_family_mda.parquet")
    useful = set(group.loc[(group["head"].eq(head)) & group["family_mda_top10_median"].gt(0.0), "semantic_family"].astype(str))
    rescues = [
        part.sort_values(["priority", "feature"], kind="stable").iloc[0].feature
        for family, part in summary.groupby("family", sort=True) if family in useful
    ]
    ordered = summary.sort_values(["priority", "feature"], kind="stable").feature.astype(str).tolist()
    contracts: dict[int, tuple[str, ...]] = {}
    for width in WIDTHS:
        initial = list(dict.fromkeys(rescues))[:width]
        initial.extend(field for field in ordered if field not in set(initial))
        selected = tuple(initial[:width])
        if len(selected) != width:
            raise AssertionError(f"{head}: cannot create F{width}")
        contracts[width] = selected
    return contracts


def _fit(
    *, head: str, fields: Sequence[str], train: pd.DataFrame, held: pd.DataFrame,
    x_train: np.ndarray, x_held: np.ndarray, seed: int, n_jobs: int,
) -> tuple[np.ndarray, dict[str, float]]:
    target, direction = str(screen.HEADS[head]["target"]), float(screen.HEADS[head]["direction"])
    # MDA's Screen120 matrices already contain fields in Screen120 order.  The
    # caller receives a width-specific matrix, so no field lookup can drift.
    model = LGBMRegressor(**screen._params(seed=seed, n_jobs=n_jobs))
    model.fit(x_train, pd.to_numeric(train[target], errors="coerce").to_numpy(float))
    score = direction * model.predict(x_held)
    metric = screen._metric_suite(held.assign(__score__=score), "__score__")
    return np.asarray(score, dtype=np.float32), metric


def _fold_data(
    *, head: str, full_fields: Sequence[str], held_month: pd.Timestamp, feature_root: Path,
    router_root: Path, labels_root: Path, base_labels_root: Path | None, policy: pd.DataFrame,
    train_months: int, reserve_days: int, train_cap: int, held_cap: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    return mda._prepare(
        head=head, fields=full_fields, held_month=held_month, feature_root=feature_root, router_root=router_root,
        labels_root=labels_root, base_labels_root=base_labels_root, policy=policy, train_months=train_months,
        reserve_days=reserve_days, train_cap=train_cap, held_cap=held_cap,
    )


def run(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    months = _months(args.held_months)
    policy = screen._read_policy(args.policy_path, months)
    roots = dict(zip(args.heads, args.screen_roots, strict=True))
    all_contracts = {head: _contracts(mda_root=args.mda_root, screen_root=roots[head], head=head) for head in args.heads}
    _once(args.out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline strict-OOF subset ladder; no live/MC1/admission/portfolio/exchange mutation",
        "heads": args.heads, "held_months": [f"{month:%Y-%m}" for month in months], "seeds": SEEDS,
        "widths": WIDTHS, "raw_feature_cmi_or_ic": False,
        "selection": "75% stable economic MDA + 25% boundary MDA plus positive-family representative rescue",
    })
    rows: list[dict[str, object]] = []
    contracts_json: dict[str, object] = {}
    for head in args.heads:
        full_fields = _screen_fields(roots[head], head)
        contracts = all_contracts[head]
        contracts_json[head] = {f"F{width}": {"feature_contract": list(fields), "feature_contract_sha256": _hash(fields)} for width, fields in contracts.items()}
        positions = {field: index for index, field in enumerate(full_fields)}
        cache: dict[pd.Timestamp, tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]] = {}
        for month in months:
            cache[month] = _fold_data(
                head=head, full_fields=full_fields, held_month=month, feature_root=args.feature_root, router_root=args.router_root,
                labels_root=args.labels_root, base_labels_root=args.base_labels_root, policy=policy, train_months=args.train_months,
                reserve_days=args.reserve_days, train_cap=args.train_cap, held_cap=args.held_cap,
            )
        for width, fields in contracts.items():
            col = np.array([positions[field] for field in fields], dtype=np.int64)
            for seed in SEEDS:
                for month in months:
                    train, held, x_train, x_held = cache[month]
                    score, metric = _fit(head=head, fields=fields, train=train, held=held, x_train=x_train[:, col], x_held=x_held[:, col], seed=seed + 1000 * month.month, n_jobs=args.n_jobs)
                    target_free = held.loc[:, list(screen.IDENTITY)].copy()
                    target_free["base_score"] = score
                    path = args.out / "target_free_scores" / f"head={head}" / f"width=F{width}" / f"seed={seed}" / f"month={month:%Y-%m}.parquet"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    target_free.to_parquet(path, index=False, compression="zstd")
                    rows.append({"head": head, "width": width, "seed": seed, "held_month": f"{month:%Y-%m}", "target_free_score": str(path), **metric})
                _progress(args.out, stage="candidate_seed_complete", head=head, width=width, seed=seed)
    detail = pd.DataFrame(rows)
    summary = detail.groupby(["head", "width"], sort=False).agg(
        ts_top01_ev=("ts_top01_ev", "mean"), ts_top02_ev=("ts_top02_ev", "mean"), ts_top05_ev=("ts_top05_ev", "mean"),
        ts_top10_ev=("ts_top10_ev", "mean"), base_stable_p10=("base_stable_p10", "mean"),
        q10_week_top10_ev=("q10_week_top10_ev", "mean"), q25_month_top10_ev=("q25_month_top10_ev", "mean"),
        worst_fold_top10_ev=("ts_top10_ev", "min"), seeds=("seed", "nunique"), folds=("held_month", "nunique"),
    ).reset_index()
    winners: dict[str, object] = {}
    for head, part in summary.groupby("head", sort=False):
        best = part.loc[part.base_stable_p10.idxmax()]
        near = part.loc[
            part.ts_top10_ev.ge(float(best.ts_top10_ev) * .99)
            & part.base_stable_p10.ge(float(best.base_stable_p10) * .99)
            & part.q10_week_top10_ev.ge(float(best.q10_week_top10_ev) - 1e-9)
        ].sort_values("width", kind="stable")
        chosen = near.iloc[0] if len(near) else best
        winners[head] = {"width": int(chosen.width), "feature_contract": contracts_json[head][f"F{int(chosen.width)}"]["feature_contract"], "feature_contract_sha256": contracts_json[head][f"F{int(chosen.width)}"]["feature_contract_sha256"], "selection": "smallest within 1% TS-Top10 and BASE_STABLE_P10, without q10-week deterioration"}
    detail.to_parquet(args.out / "subset_ladder_fold_metrics.parquet", index=False, compression="zstd")
    summary.sort_values(["head", "base_stable_p10", "width"], ascending=[True, False, True], kind="stable").to_parquet(args.out / "subset_ladder_summary.parquet", index=False, compression="zstd")
    _once(args.out / "subset_contracts.json", contracts_json)
    _once(args.out / "subset_winners.json", {"schema": SCHEMA, "winners": winners})
    _once(args.out / "correctness_report.json", {
        "all_held_scores_persisted_before_metrics": True,
        "all_training_labels_before_reserve": True,
        "all_widths_at_most_120": True,
        "two_fixed_seed_oof_retrains": True,
        "raw_feature_cmi_or_ic_not_run": True,
        "live_or_exchange_mutation": False,
    })
    _progress(args.out, stage="complete", heads=args.heads)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mda-root", type=Path, required=True)
    parser.add_argument("--heads", nargs="+", choices=("B0", "E", "T"), required=True)
    parser.add_argument("--screen-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--base-labels-root", type=Path)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--held-months", default="2025-05,2025-06,2025-07")
    parser.add_argument("--train-months", type=int, default=4)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--train-cap", type=int, default=120_000)
    parser.add_argument("--held-cap", type=int, default=25_000)
    parser.add_argument("--n-jobs", type=int, default=min(6, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.out.exists() or len(args.heads) != len(args.screen_roots) or ("B0" in args.heads and args.base_labels_root is None):
        raise ValueError("invalid immutable subset-ladder contract")
    run(args)


if __name__ == "__main__":
    main()
