#!/usr/bin/env python3
"""Strict-OOF economic and boundary MDA for one Router50 single-Base arm.

This is a research-only selector.  It operates strictly inside the Router50
candidate population, trains on labels resolved before the fold reserve, and
permutates held causal fields *within timestamp*.  It never writes a held
outcome into a score receipt and never alters inference or live trading.

The resulting compact contracts are only development candidates.  They must
subsequently be rebuilt as strict OOF Base -> R/U -> dual-MC1 ledgers and
compared against the frozen F72 contract on an untouched downstream period.
"""
from __future__ import annotations

import argparse
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
import run_strict_r3_router_single_base_prescreen_v1 as base  # noqa: E402


SCHEMA = "strict_r3_router_single_base_mda_v1"
SEEDS = (1729, 71729)
SUBSET_SIZES = (120, 90, 70, 50, 35, 25)


def _write_once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _progress(path: Path, **payload: object) -> None:
    with (path / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _months(value: str) -> tuple[pd.Timestamp, ...]:
    result = tuple(
        pd.Timestamp(f"{token.strip()}-01", tz="UTC")
        for token in value.split(",") if token.strip()
    )
    if not result or tuple(sorted(result)) != result:
        raise ValueError("held months must be a non-empty chronological list")
    return result


def _family(name: str) -> str:
    lower = name.lower()
    if any(token in lower for token in ("fund", "carry")):
        return "funding"
    if any(token in lower for token in ("oi", "open_interest", "leverage")):
        return "oi_leverage"
    if any(token in lower for token in ("liq", "book", "impact", "amihud", "volume", "vp_")):
        return "liquidity"
    if any(token in lower for token in ("rv", "vol", "atr", "range", "vov", "semivariance")):
        return "volatility"
    if any(token in lower for token in ("trend", "ret", "price", "donchian", "adx", "ker", "bollinger", "wick", "body")):
        return "price_trend"
    if any(token in lower for token in ("mkt", "beta", "corr", "peer", "bench", "universe", "xs_")):
        return "cross_asset"
    if any(token in lower for token in ("regime", "state", "entropy", "transition", "tail", "climax", "exhaust")):
        return "state"
    return "other"


def _permutation(frame: pd.DataFrame, seed: int) -> np.ndarray:
    work = frame.loc[:, ["candidate_id", "__decision_ts__"]].reset_index(drop=True).copy()
    work["__row__"] = np.arange(len(work), dtype=np.int64)
    work["__hash__"] = pd.util.hash_pandas_object(
        work.candidate_id.astype(str) + f"|{seed}", index=False,
    ).to_numpy(np.uint64)
    output = np.arange(len(work), dtype=np.int64)
    for _, group in work.sort_values(["__decision_ts__", "__hash__", "candidate_id"], kind="stable").groupby("__decision_ts__", sort=False):
        rows = group["__row__"].to_numpy(np.int64)
        if len(rows) > 1:
            output[rows] = np.roll(rows, 1)
    return output


def _metrics(held: pd.DataFrame, score: np.ndarray) -> dict[str, float]:
    scored = held.loc[:, list(base.IDENTITY)].copy()
    scored["base_score"] = np.asarray(score, dtype=np.float32)
    labels = held.loc[:, ["candidate_id", "policy_ordinal_valid", "policy_net_bps"]].copy()
    return base._metrics(scored, labels)


def _boundary(held: pd.DataFrame, baseline: np.ndarray, altered: np.ndarray) -> tuple[float, float]:
    work = held.loc[:, ["candidate_id", "__decision_ts__", "policy_ordinal_valid", "policy_net_bps"]].copy()
    work["__row__"] = np.arange(len(work), dtype=np.int64)
    work["base"] = np.asarray(baseline, dtype=float)
    work["altered"] = np.asarray(altered, dtype=float)
    sets: list[set[tuple[pd.Timestamp, str]]] = []
    ranks: list[np.ndarray] = []
    for field in ("base", "altered"):
        ranked = work.loc[:, ["__row__", "candidate_id", "__decision_ts__", field]].sort_values(
            ["__decision_ts__", field, "candidate_id"], ascending=[True, False, True], kind="stable",
        )
        ordinal = ranked.groupby("__decision_ts__", sort=False).cumcount().to_numpy(float) + 1.0
        count = ranked.groupby("__decision_ts__", sort=False).candidate_id.transform("size").to_numpy(float)
        selected = ordinal <= np.ceil(count * .10)
        pairs = set(zip(ranked.loc[selected, "__decision_ts__"], ranked.loc[selected, "candidate_id"], strict=True))
        rank = np.empty(len(work), dtype=np.float32)
        rank[ranked.__row__.to_numpy(np.int64)] = 1.0 - (ordinal - .5) / count
        sets.append(pairs)
        ranks.append(rank)
    base_set, altered_set = sets
    pair = list(zip(work.__decision_ts__, work.candidate_id, strict=True))
    work["base_selected"] = [item in base_set for item in pair]
    work["altered_selected"] = [item in altered_set for item in pair]
    work["near"] = (ranks[0] >= .75) | (ranks[1] >= .75)
    valid = work.policy_ordinal_valid.fillna(False).astype(bool) & np.isfinite(pd.to_numeric(work.policy_net_bps, errors="coerce"))
    deltas: list[float] = []
    for _, group in work.loc[work.near & valid].groupby("__decision_ts__", sort=False):
        retained = group.loc[group.base_selected & ~group.altered_selected, "policy_net_bps"]
        replacement = group.loc[group.altered_selected & ~group.base_selected, "policy_net_bps"]
        if len(retained) and len(replacement):
            deltas.append(float(retained.mean() - replacement.mean()))
    return float(np.mean(deltas)) if deltas else 0.0, len(base_set & altered_set) / max(1, len(base_set | altered_set))


def _target_train(frame: pd.DataFrame, spec: base.TargetSpec, reserve: pd.Timestamp, cap: int) -> pd.DataFrame:
    available = pd.to_datetime(frame[spec.available_column], utc=True, errors="coerce")
    valid = frame[spec.valid_column].fillna(False).astype(bool)
    value = pd.to_numeric(frame[spec.value_column], errors="coerce")
    train = frame.loc[valid & available.lt(reserve) & np.isfinite(value)].copy()
    return base._sample_complete_queries(train, cap).sort_values(["__decision_ts__", "candidate_id"], kind="stable")


def _fit_score(*, train: pd.DataFrame, held: pd.DataFrame, fields: Sequence[str], spec: base.TargetSpec,
               objective: str, gain: str, truncation: int, sigmoid: float, seed: int, n_jobs: int) -> np.ndarray:
    y, _ = base._target_labels(train, held, spec)
    x_train, medians = base._numeric_matrix(train, fields)
    x_held, _ = base._numeric_matrix(held, fields, medians)
    params: dict[str, object] = dict(
        objective=objective, metric="ndcg", n_estimators=180, learning_rate=.045,
        max_depth=4, num_leaves=15, min_child_samples=260,
        subsample=.8, subsample_freq=1, colsample_bytree=.8,
        reg_alpha=.05, reg_lambda=8., min_split_gain=.001,
        lambdarank_truncation_level=truncation, label_gain=base.GAIN_SCHEDULES[gain],
        lambdarank_norm=True, random_state=seed, n_jobs=n_jobs,
        deterministic=True, force_col_wise=True, verbosity=-1,
    )
    if objective == "lambdarank":
        params["sigmoid"] = sigmoid
    model = LGBMRanker(**params)
    model.fit(x_train, y, group=base._query_groups(train))
    return model.predict(x_held).astype(np.float32)


def _summary(observations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature, group in observations.groupby("feature", sort=False):
        row: dict[str, object] = {"feature": feature, "observations": len(group)}
        for metric in ("mda_tip_bps", "mda_dtp1_bps", "mda_dtp2_bps", "mda_dtp5_bps", "mda_dtp10_bps", "mda_breadth", "boundary_mda_bps"):
            data = pd.to_numeric(group[metric], errors="coerce")
            row[f"median_{metric}"] = float(data.median())
            row[f"iqr_{metric}"] = float(data.quantile(.75) - data.quantile(.25))
            row[f"worst_{metric}"] = float(data.min())
            row[f"positive_{metric}_fraction"] = float(data.gt(0).mean())
        row["mda_score"] = float(
            .45 * (row["median_mda_tip_bps"] - .5 * row["iqr_mda_tip_bps"])
            + .20 * (row["median_mda_dtp10_bps"] - .5 * row["iqr_mda_dtp10_bps"])
            + .20 * (row["median_boundary_mda_bps"] - .5 * row["iqr_boundary_mda_bps"])
            + .15 * row["median_mda_breadth"]
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["mda_score", "feature"], ascending=[False, True], kind="stable")


def _aggregate_subset(rows: list[dict[str, object]], size: int) -> dict[str, object]:
    data = pd.DataFrame(rows)
    means = data.mean(numeric_only=True).to_dict()
    tip = .25 * means["dtp1_bps"] + .25 * means["dtp2_bps"] + .50 * means["dtp5_bps"]
    breadth = .30 * means["er50_at20"] + .25 * means["recall50_at20"] + .25 * means["recall100_at20"] + .20 * means["er100_at20"]
    stability = .5 * means["q10_week_dtp5_bps"] + .5 * means["q25_month_dtp5_bps"]
    return {"subset_size": size, **means, "tip_bps": tip, "breadth": breadth, "stability_bps": stability}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-roots", required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--selection-receipt", type=Path, required=True)
    parser.add_argument("--target", choices=base.TARGETS, required=True)
    parser.add_argument("--gain", choices=base.GAIN_SCHEDULES, required=True)
    parser.add_argument("--objective", choices=("lambdarank", "rank_xendcg"), required=True)
    parser.add_argument("--truncation", type=int, required=True)
    parser.add_argument("--sigmoid", type=float, default=1.0)
    parser.add_argument("--held-months", default="2025-11,2026-01,2026-03")
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--train-cap", type=int, default=60000)
    parser.add_argument("--held-cap", type=int, default=15000)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.feature_roots = tuple(Path(item).resolve() for item in args.feature_roots.split(",") if item.strip())
    held_months = _months(args.held_months)
    fields = list(base._load_f72_fields(args.selection_receipt))
    if any(any(token in field.lower() for token in base.PROHIBITED_SCORE_TOKENS) for field in fields):
        raise AssertionError("selection contract contains prohibited outcome-like feature name")
    spec = base.TARGETS[args.target]
    args.out.mkdir(parents=True)
    _write_once(args.out / "run_manifest.json", {
        "schema": SCHEMA, "scope": "offline Router50 Base MDA only; no R/U, MC1, live, or exchange mutation",
        "selection_receipt": str(args.selection_receipt.resolve()), "target": args.target,
        "objective": args.objective, "gain": args.gain, "truncation": args.truncation,
        "sigmoid": args.sigmoid, "held_months": [f"{month:%Y-%m}" for month in held_months],
        "strict_oof": {"train_months": args.train_months, "reserve_days": args.reserve_days,
                       "train_cap_complete_queries": args.train_cap, "held_cap_complete_queries": args.held_cap},
        "seeds": list(SEEDS), "permutation": "deterministic within decision timestamp",
        "feature_count": len(fields), "router": "exact target-free top50 identity; router numeric never a Base input",
    })
    observations: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    fold_cache: list[tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]] = []
    for fold, held_month in enumerate(held_months):
        reserve = held_month - pd.Timedelta(days=args.reserve_days)
        start = reserve - pd.DateOffset(months=args.train_months)
        window, _ = base._load_window(candidate_root=None, feature_root=args.feature_roots,
            label_root=args.label_root, router_root=args.router_root, start=start,
            end=held_month + pd.offsets.MonthBegin(1), fields=fields)
        train = _target_train(window.loc[window.__decision_ts__.lt(reserve)].copy(), spec, reserve, args.train_cap)
        held = base._sample_complete_queries(window.loc[window.__decision_ts__.ge(held_month)].copy(), args.held_cap)
        held = held.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
        if len(train) < 20_000 or len(held) < 2_000:
            raise AssertionError(f"{held_month:%Y-%m}: insufficient strict support {len(train)} / {len(held)}")
        fold_cache.append((train, held, held_month))
        for seed in SEEDS:
            # Fit once per strict fold/seed.  MDA changes only the held
            # inference matrix, so refitting per feature would be both wrong
            # in spirit and needlessly expensive.
            y, _ = base._target_labels(train, held, spec)
            x_train, medians = base._numeric_matrix(train, fields)
            x_held, _ = base._numeric_matrix(held, fields, medians)
            params: dict[str, object] = dict(
                objective=args.objective, metric="ndcg", n_estimators=180, learning_rate=.045,
                max_depth=4, num_leaves=15, min_child_samples=260,
                subsample=.8, subsample_freq=1, colsample_bytree=.8,
                reg_alpha=.05, reg_lambda=8., min_split_gain=.001,
                lambdarank_truncation_level=args.truncation, label_gain=base.GAIN_SCHEDULES[args.gain],
                lambdarank_norm=True, random_state=seed + fold, n_jobs=args.n_jobs,
                deterministic=True, force_col_wise=True, verbosity=-1,
            )
            if args.objective == "lambdarank":
                params["sigmoid"] = args.sigmoid
            model = LGBMRanker(**params)
            model.fit(x_train, y, group=base._query_groups(train))
            baseline = model.predict(x_held).astype(np.float32)
            base_metrics = _metrics(held, baseline)
            baseline_rows.append({"held_month": f"{held_month:%Y-%m}", "seed": seed, **base_metrics})
            source = _permutation(held, seed + 10000 * fold)
            for index, field in enumerate(fields):
                # MDA is inference-side only: fit is untouched; only the
                # held causal column is permuted inside each timestamp query.
                values = x_held.copy()
                values[:, index] = values[source, index]
                altered = model.predict(values).astype(np.float32)
                changed = _metrics(held, altered)
                boundary, jaccard = _boundary(held, baseline, altered)
                tip = .25 * (base_metrics["dtp1_bps"] - changed["dtp1_bps"]) + .25 * (base_metrics["dtp2_bps"] - changed["dtp2_bps"]) + .5 * (base_metrics["dtp5_bps"] - changed["dtp5_bps"])
                breadth = (.30 * (base_metrics["er50_at20"] - changed["er50_at20"]) + .25 * (base_metrics["recall50_at20"] - changed["recall50_at20"]) + .25 * (base_metrics["recall100_at20"] - changed["recall100_at20"]) + .20 * (base_metrics["er100_at20"] - changed["er100_at20"]))
                observations.append({"feature": field, "family": _family(field), "held_month": f"{held_month:%Y-%m}", "seed": seed,
                    "mda_tip_bps": tip, "mda_dtp1_bps": base_metrics["dtp1_bps"] - changed["dtp1_bps"],
                    "mda_dtp2_bps": base_metrics["dtp2_bps"] - changed["dtp2_bps"], "mda_dtp5_bps": base_metrics["dtp5_bps"] - changed["dtp5_bps"],
                    "mda_dtp10_bps": base_metrics["dtp10_bps"] - changed["dtp10_bps"], "mda_breadth": breadth,
                    "boundary_mda_bps": boundary, "top10_jaccard": jaccard})
            _progress(args.out, stage="mda_fold_seed_complete", held_month=f"{held_month:%Y-%m}", seed=seed, fields=len(fields))
            del model, x_train, x_held
    observations_frame = pd.DataFrame(observations)
    summary = _summary(observations_frame)
    summary["family"] = summary.feature.map(_family)
    family = summary.groupby("family", sort=False).agg(
        fields=("feature", "size"), mda_score=("mda_score", "median"),
        positive_tip=("positive_mda_tip_bps_fraction", "mean"),
    ).reset_index().sort_values("mda_score", ascending=False, kind="stable")
    rescue = []
    for name, group in summary.groupby("family", sort=False):
        if float(group.mda_score.max()) > 0.0:
            rescue.append(str(group.sort_values(["mda_score", "feature"], ascending=[False, True], kind="stable").iloc[0].feature))
    ranked = rescue + [str(item) for item in summary.loc[
        summary.positive_mda_tip_bps_fraction.ge(.50), "feature"
    ] if item not in rescue]
    ranked += [field for field in summary.feature.astype(str) if field not in ranked]
    contracts: dict[int, list[str]] = {size: ranked[:min(size, len(ranked))] for size in SUBSET_SIZES}
    subset_rows: list[dict[str, object]] = []
    for size, selected in contracts.items():
        metrics: list[dict[str, object]] = []
        for fold, (train, held, held_month) in enumerate(fold_cache):
            for seed in SEEDS:
                score = _fit_score(train=train, held=held, fields=selected, spec=spec, objective=args.objective,
                    gain=args.gain, truncation=args.truncation, sigmoid=args.sigmoid, seed=seed + fold, n_jobs=args.n_jobs)
                metrics.append({"held_month": f"{held_month:%Y-%m}", "seed": seed, **_metrics(held, score)})
        subset_rows.append(_aggregate_subset(metrics, size))
    subsets = pd.DataFrame(subset_rows).sort_values("subset_size", ascending=False, kind="stable")
    reference = subsets.loc[subsets.subset_size.eq(max(subsets.subset_size))].iloc[0]
    subsets["passes_tip_guard"] = (subsets.dtp1_bps.ge(.97 * reference.dtp1_bps) & subsets.dtp2_bps.ge(.98 * reference.dtp2_bps) & subsets.dtp5_bps.ge(.98 * reference.dtp5_bps))
    subsets["passes_economic_gap"] = subsets.tip_bps.ge(.99 * reference.tip_bps) & subsets.stability_bps.ge(.98 * reference.stability_bps)
    eligible = subsets.loc[subsets.passes_tip_guard & subsets.passes_economic_gap].sort_values("subset_size", kind="stable")
    selected_size = int(eligible.iloc[0].subset_size) if len(eligible) else int(reference.subset_size)
    selected = contracts[selected_size]
    observations_frame.to_parquet(args.out / "economic_boundary_mda_observations.parquet", index=False, compression="zstd")
    summary.to_parquet(args.out / "economic_boundary_mda_summary.parquet", index=False, compression="zstd")
    family.to_parquet(args.out / "semantic_family_mda_summary.parquet", index=False, compression="zstd")
    pd.DataFrame(baseline_rows).to_parquet(args.out / "mda_baseline_metrics.parquet", index=False, compression="zstd")
    subsets.to_parquet(args.out / "subset_ladder_metrics.parquet", index=False, compression="zstd")
    for size, features in contracts.items():
        _write_once(args.out / f"subset{size}_contract.json", {"schema": SCHEMA, "selected_features": features,
            "feature_count": len(features), "target": args.target, "selection": "strict OOF within-timestamp economic and Top10-boundary MDA with family rescue"})
    _write_once(args.out / "selected_contract.json", {"schema": SCHEMA, "selected_features": selected,
        "feature_count": len(selected), "target": args.target, "objective": args.objective,
        "gain": args.gain, "truncation": args.truncation, "sigmoid": args.sigmoid,
        "selection": "smallest MDA subset within 1% tip and 2% stability of Screen120 on development folds",
        "selection_sha256": hashlib.sha256("\n".join(selected).encode()).hexdigest()})
    _write_once(args.out / "correctness_report.json", {"router_top50_only": True, "router_numeric_absent": True,
        "held_feature_permutation_within_timestamp": True, "train_target_resolved_before_reserve": True,
        "held_outcomes_metric_only": True, "no_live_or_exchange_mutation": True})
    _progress(args.out, stage="complete", selected_size=selected_size, fields=len(fields))


if __name__ == "__main__":
    main()
