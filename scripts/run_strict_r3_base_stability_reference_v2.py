#!/usr/bin/env python3
"""Evaluate the large and compact contracts emitted by the stability selector.

The selector's screen is deliberately cheap.  This producer is the first
stronger validation stage: it fits a candidate physical head on five strict
blocked OOF folds, scores the complete frozen B/E/T blend, and compares it to
the frozen incumbent head on exactly the same target-free candidates.

It is research-only.  It neither imports nor writes any inference, exchange,
consensus, MC1, admission, portfolio, or execution object.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from run_strict_r3_base_stability_selector_v2 import (
    HEADS, IDENTITY, SCORE_FIELDS, SEED, _enhanced_score, _held_rows,
    _impute, _materialize, _model, _next_month, _read_policy, _sample_whole_queries,
    _timestamp_metrics, _train_rows, _utc, _window,
)


def _exclusive(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _progress(root: Path, **payload: object) -> None:
    with (root / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _read_contract(path: Path) -> tuple[str, list[str]]:
    data = json.loads(path.read_text())
    features = list(data["features"])
    if not features or len(features) > 420:
        raise AssertionError(f"{path}: invalid compact contract size {len(features)}")
    return str(data["head"]), features


def _apply_base_override(result: pd.DataFrame, override: pd.DataFrame | None) -> pd.DataFrame:
    """Replace only the B coordinate with a strict OOF B0 score ledger."""
    if override is None:
        return result
    joined = result.merge(override, on=list(IDENTITY), how="left", validate="one_to_one")
    if joined.b0_f72_score.isna().any():
        raise AssertionError("enhanced-B0 OOF ledger does not cover every held counterpart row")
    joined["base_bps"] = joined.b0_f72_score.to_numpy(float)
    return joined.drop(columns=["b0_f72_score"])


def _evaluate(
    *, args: argparse.Namespace, head: str, features: Sequence[str], policy: pd.DataFrame,
    held_months: Sequence[pd.Timestamp], seeds: Sequence[int], name: str,
    base_override: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full strict-OOF evaluation of one physical-head contract."""
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for fold, held_month in enumerate(held_months):
        reserve = held_month - pd.Timedelta(days=args.reserve_days)
        start = reserve - pd.DateOffset(months=args.train_months)
        window = _window(
            head=head, feature_root=args.feature_root, router_root=args.router_root,
            score_root=args.score_root, label_root=args.label_root, policy=policy,
            start=start, end=_next_month(held_month), route_fraction=args.route_fraction,
        )
        train = _train_rows(window.loc[window.__decision_ts__.lt(reserve)].copy(), head, reserve, args.train_cap)
        held = _sample_whole_queries(_held_rows(window.loc[window.__decision_ts__.ge(held_month)].copy()), args.held_cap, seed=SEED + 77 * fold)
        if len(train) < args.min_train_rows or len(held) < args.min_held_rows:
            raise AssertionError(f"{name}/{held_month:%Y-%m}: insufficient support train={len(train)}, held={len(held)}")
        selected = pd.concat([train, held], ignore_index=True)
        values = _impute(_materialize(args.feature_root, selected, features), len(train))
        target = pd.to_numeric(train[str(HEADS[head]["target"])], errors="coerce")
        for seed in seeds:
            model = _model(head, seed=seed + fold * 101, n_jobs=args.n_jobs, cheap=False)
            if head == "B":
                group = train.groupby("__decision_ts__", sort=False).size().to_numpy(np.int32)
                model.fit(values[:len(train)], target.to_numpy(np.int32), group=group)
            else:
                model.fit(values[:len(train)], target.to_numpy(float))
            candidate = float(HEADS[head]["direction"]) * model.predict(values[len(train):])
            result = held.loc[:, [*IDENTITY, "policy_net_bps", *SCORE_FIELDS.values()]].copy().reset_index(drop=True)
            result = _apply_base_override(result, base_override)
            result["candidate_score"] = candidate
            result["enhanced_score"] = _enhanced_score(result, head, candidate)
            metrics = _timestamp_metrics(result, "enhanced_score")
            metric_rows.append({
                "contract": name, "head": head, "held_month": f"{held_month:%Y-%m}", "seed": seed,
                "features": len(features), "train_rows": len(train), "held_rows": len(held), **metrics,
            })
            prediction_rows.append(result.assign(contract=name, head=head, held_month=f"{held_month:%Y-%m}", seed=seed))
            del model, result
        _progress(args.out, stage="fold_complete", contract=name, held_month=f"{held_month:%Y-%m}", seeds=list(seeds), train_rows=len(train), held_rows=len(held))
        del values, selected, train, held, window
        gc.collect()
    return pd.DataFrame(metric_rows), pd.concat(prediction_rows, ignore_index=True)


def _incumbent(
    *, args: argparse.Namespace, head: str, policy: pd.DataFrame, held_months: Sequence[pd.Timestamp],
    base_override: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    for fold, held_month in enumerate(held_months):
        reserve = held_month - pd.Timedelta(days=args.reserve_days)
        window = _window(
            head=head, feature_root=args.feature_root, router_root=args.router_root,
            score_root=args.score_root, label_root=args.label_root, policy=policy,
            start=reserve - pd.DateOffset(months=args.train_months), end=_next_month(held_month), route_fraction=args.route_fraction,
        )
        held = _sample_whole_queries(_held_rows(window.loc[window.__decision_ts__.ge(held_month)].copy()), args.held_cap, seed=SEED + 77 * fold)
        result = held.loc[:, [*IDENTITY, "policy_net_bps", *SCORE_FIELDS.values()]].copy().reset_index(drop=True)
        result = _apply_base_override(result, base_override)
        candidate = pd.to_numeric(result[SCORE_FIELDS[head]], errors="coerce").to_numpy(float)
        result["candidate_score"] = candidate
        result["enhanced_score"] = _enhanced_score(result, head, candidate)
        metrics.append({
            "contract": "incumbent", "head": head, "held_month": f"{held_month:%Y-%m}", "seed": -1,
            "features": np.nan, "train_rows": 0, "held_rows": len(held), **_timestamp_metrics(result, "enhanced_score"),
        })
        predictions.append(result.assign(contract="incumbent", head=head, held_month=f"{held_month:%Y-%m}", seed=-1))
        _progress(args.out, stage="incumbent_fold_complete", held_month=f"{held_month:%Y-%m}", held_rows=len(held))
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True)


def _summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    value_columns = [
        "stable_top10_5_2", "ts_top01_ev", "ts_top02_ev", "ts_top05_ev", "ts_top10_ev",
        "fixed_k1_ev", "fixed_k2_ev", "fixed_k3_ev", "fixed_k5_ev", "fixed_k10_ev",
        "weekly_q10_top10", "weekly_q10_top02", "monthly_q25_top05", "worst_month_top10", "positive_month_fraction_top10",
    ]
    result = metrics.groupby(["contract", "head", "features"], dropna=False, sort=False)[value_columns].mean().reset_index()
    for column in ("stable_top10_5_2", "ts_top10_ev", "ts_top05_ev", "ts_top02_ev"):
        result[f"{column}_std"] = metrics.groupby(["contract", "head", "features"], dropna=False, sort=False)[column].std().to_numpy(float)
    return result.sort_values("stable_top10_5_2", ascending=False, kind="stable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--selector-root", type=Path, required=True)
    parser.add_argument("--base-score-oof", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--held-months", nargs="+", default=("2025-11-01", "2026-01-01", "2026-03-01", "2026-05-01", "2026-07-01"))
    parser.add_argument("--top-n", type=int, help="Evaluate the top-N stable-inclusion features from the selector evidence")
    parser.add_argument("--seeds", nargs="+", type=int, default=(1729, 71729))
    parser.add_argument("--route-fraction", type=float, default=.50)
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--train-cap", type=int, default=60_000)
    parser.add_argument("--held-cap", type=int, default=36_000)
    parser.add_argument("--min-train-rows", type=int, default=8_000)
    parser.add_argument("--min-held-rows", type=int, default=2_000)
    parser.add_argument("--n-jobs", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    head, features = _read_contract(args.selector_root / "prescreen_contract.json")
    if args.top_n is not None:
        if not 1 <= args.top_n < len(features):
            raise ValueError(f"--top-n must be between 1 and {len(features) - 1}")
        evidence = pd.read_parquet(args.selector_root / "feature_inclusion_evidence.parquet")
        ranked = evidence.loc[evidence["head"].eq(head)].sort_values(
            ["prescreen_score", "stable_inclusion", "feature"], ascending=[False, False, True], kind="stable"
        )
        features = ranked.feature.head(args.top_n).tolist()
        if len(features) != args.top_n:
            raise AssertionError("selector evidence does not support requested compact contract")
    held_months = tuple(_utc(value) for value in args.held_months)
    if len(held_months) < 5 or tuple(sorted(held_months)) != held_months:
        raise ValueError("reference stage requires at least five chronological blocked folds")
    # A compact contract is portable only if it survives multiple market
    # episodes.  Enforce a cross-year, eight-month validation span rather than
    # allowing a dense run of adjacent folds to be mislabelled as portability.
    month_span = (held_months[-1].year - held_months[0].year) * 12 + held_months[-1].month - held_months[0].month
    if len({month.year for month in held_months}) < 2 or month_span < 8:
        raise ValueError(
            "reference folds must span at least eight calendar months and two calendar years"
        )
    args.out.mkdir(parents=True)
    policy = _read_policy(args.policy_path)
    base_override = None
    if args.base_score_oof is not None:
        base_override = pd.read_parquet(args.base_score_oof, columns=[*IDENTITY, "b0_f72_score"])
        base_override["__decision_ts__"] = pd.to_datetime(base_override["__decision_ts__"], utc=True, errors="raise")
        if base_override.duplicated(list(IDENTITY)).any():
            raise AssertionError("enhanced-B0 OOF ledger has duplicate candidate identities")
    _exclusive(args.out / "run_manifest.json", {
        "schema": "strict_r3_base_stability_reference_v2",
        "scope": "offline research only; no live/inference/exchange/consensus/MC1/admission/portfolio/execution changes",
        "head": head, "selector_root": str(args.selector_root),
        "contract_features": len(features), "contract_sha256": hashlib.sha256("\n".join(features).encode()).hexdigest(),
        "contract_source": "stable_inclusion_top_n" if args.top_n is not None else "prescreen_420",
        "held_months": [f"{item:%Y-%m}" for item in held_months], "seeds": list(args.seeds),
        "strict_train": {"pre_reserve_labels": True, "reserve_days": args.reserve_days, "train_months": args.train_months},
        "selection_metric": "timestamp-local STABLE_TOP10_5_2 on frozen-counterpart enhanced B/E/T rank blend",
        "base_counterpart": "strict_oof_b0_f72" if base_override is not None else "incumbent_base_bps",
        "target_fields_in_feature_matrix": False,
    })
    incumbent_metrics, incumbent_predictions = _incumbent(args=args, head=head, policy=policy, held_months=held_months, base_override=base_override)
    candidate_metrics, candidate_predictions = _evaluate(args=args, head=head, features=features, policy=policy, held_months=held_months, seeds=tuple(args.seeds), name="prescreen_large", base_override=base_override)
    metrics = pd.concat([incumbent_metrics, candidate_metrics], ignore_index=True)
    predictions = pd.concat([incumbent_predictions, candidate_predictions], ignore_index=True)
    metrics.to_parquet(args.out / "fold_metrics.parquet", index=False, compression="zstd")
    predictions.to_parquet(args.out / "oof_predictions.parquet", index=False, compression="zstd")
    _summarize(metrics).to_parquet(args.out / "summary.parquet", index=False, compression="zstd")
    _progress(args.out, stage="reference_complete", head=head, features=len(features), folds=len(held_months), seeds=list(args.seeds))


if __name__ == "__main__":
    main()
