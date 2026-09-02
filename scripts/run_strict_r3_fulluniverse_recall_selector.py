#!/usr/bin/env python3
"""Strict temporal feature selection for the economic-recall router.

This is purpose-built for the router, whose job is recall of economically
useful opportunities at a *predeclared timestamp-local route width*.  It is
not MDA and it never searches the live/meta contract.  It starts from the
complete causal feature universe, joins policy outcomes only after features
are fixed, and retains a frozen ordered contract for the downstream ranker.

Selection is confined to supplied development months.  A later scoring/HPO
stage must use a separate, untouched block.  Each fold is chronological:

    train labels resolved before reserve -> held target-free score -> metrics

The score used for feature selection is mean timestamp-local recall of rows
with policy net >= the declared hurdle after keeping the top 50% per exact
decision timestamp.  Top-10 precision and top-two policy EV are tie-breaks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker


SCHEMA = "strict_r3_fulluniverse_recall_feature_selector_v1"
SEED = 1729
IDENTITIES = ("candidate_id", "__decision_ts__", "side_name")
EXCLUDE = frozenset({"candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name"})
FORBIDDEN_SCORE_INPUTS = frozenset({
    "policy_path_valid", "policy_gross_bps", "policy_net_bps", "policy_exit_bar_15m",
    "policy_entry_price", "policy_exit_price", "policy_exit_reason", "policy_label_available_ts",
    "policy_cost_bps", "policy_outcome_source", "label_source_complete_1m_path",
})


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _months(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[str]:
    periods = pd.period_range(start.to_period("M"), (end - pd.Timedelta(nanoseconds=1)).to_period("M"), freq="M")
    for value in periods:
        yield value.strftime("%Y-%m")


def _month_end(month: pd.Timestamp) -> pd.Timestamp:
    return month + pd.offsets.MonthBegin(1)


def _parse_months(value: str) -> tuple[pd.Timestamp, ...]:
    result = tuple(_utc(f"{part.strip()}-01") for part in value.split(",") if part.strip())
    if not result or tuple(sorted(result)) != result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("held months must be unique chronological YYYY-MM values")
    return result


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _feature_path(root: Path, month: str) -> Path:
    path = root / f"month={month}" / "causal_feature_universe.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing full-universe feature panel: {path}")
    return path


def _all_fields(root: Path, month: pd.Timestamp) -> tuple[str, ...]:
    frame = pd.read_parquet(_feature_path(root, f"{month:%Y-%m}"), columns=None)
    fields = [column for column in frame.columns if column not in EXCLUDE and pd.api.types.is_numeric_dtype(frame[column])]
    leaked = sorted(set(fields).intersection(FORBIDDEN_SCORE_INPUTS))
    if leaked:
        raise AssertionError(f"full-universe feature panel illegally includes outcome fields: {leaked}")
    if len(fields) < 300:
        raise AssertionError(f"full-universe panel exposes only {len(fields)} numeric feature fields")
    return tuple(fields)


def _load_features(root: Path, start: pd.Timestamp, end: pd.Timestamp, fields: Sequence[str]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    columns = [*IDENTITIES, *fields]
    for token in _months(start, end):
        part = pd.read_parquet(_feature_path(root, token), columns=columns)
        part["__decision_ts__"] = pd.to_datetime(part["__decision_ts__"], utc=True, errors="raise")
        part = part.loc[part["__decision_ts__"].ge(start) & part["__decision_ts__"].lt(end)]
        if not part.empty:
            pieces.append(part)
    if not pieces:
        return pd.DataFrame(columns=columns)
    result = pd.concat(pieces, ignore_index=True)
    if result.candidate_id.duplicated().any():
        raise AssertionError("full-universe feature source duplicates a target-free candidate identity")
    return result


def _load_policy(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    required = ["candidate_id", "policy_path_valid", "policy_net_bps", "policy_label_available_ts"]
    result = pd.read_parquet(path, columns=required)
    result["policy_label_available_ts"] = pd.to_datetime(result["policy_label_available_ts"], utc=True, errors="raise")
    # The candidate identity itself contains signal time; the caller will use
    # an already target-free feature frame as the actual temporal filter.
    if result.candidate_id.duplicated().any():
        raise AssertionError("canonical policy label source duplicates candidate identity")
    return result


def _query_cap(frame: pd.DataFrame, cap: int) -> pd.DataFrame:
    counts = frame.groupby("__decision_ts__", sort=False).size().rename("n").reset_index()
    if int(counts.n.sum()) <= cap:
        return frame
    token = counts["__decision_ts__"].astype(str).map(lambda item: int(hashlib.sha256(item.encode()).hexdigest()[:16], 16))
    counts = counts.assign(_hash=token).sort_values("_hash", kind="stable")
    total = counts.n.cumsum()
    keep = counts.loc[total.le(cap), "__decision_ts__"]
    if len(keep) < 20:
        keep = counts.head(20)["__decision_ts__"]
    return frame.loc[frame["__decision_ts__"].isin(set(keep))].copy()


def _prepare_train(features: pd.DataFrame, policy: pd.DataFrame, reserve_start: pd.Timestamp, hurdle: float) -> pd.DataFrame:
    result = features.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    valid = (
        result["policy_path_valid"].fillna(False).astype(bool)
        & result["policy_label_available_ts"].lt(reserve_start)
        & np.isfinite(pd.to_numeric(result["policy_net_bps"], errors="coerce"))
    )
    result = result.loc[valid].copy()
    if result.empty:
        raise AssertionError("no resolved policy labels before the training reserve")
    result["_target"] = (pd.to_numeric(result["policy_net_bps"], errors="coerce") >= hurdle).astype(np.int8)
    return result


def _matrix(frame: pd.DataFrame, fields: Sequence[str], medians: pd.Series | None = None) -> tuple[np.ndarray, pd.Series]:
    raw = frame.loc[:, fields].apply(pd.to_numeric, errors="coerce")
    if medians is None:
        medians = raw.median(axis=0, numeric_only=True).fillna(0.0)
    work = raw.fillna(medians).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return work.to_numpy(dtype=np.float32, copy=False), medians


def _groups(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("__decision_ts__", sort=False).size().to_numpy(dtype=np.int32)
    if len(counts) < 20 or int(counts.min()) < 2:
        raise AssertionError("ranker requires at least twenty non-degenerate timestamp queries")
    return counts


def _ranker(fields: Sequence[str], depth: int, leaves: int, n_estimators: int, n_jobs: int) -> LGBMRanker:
    return LGBMRanker(
        objective="lambdarank", metric="None", label_gain=[0, 1],
        n_estimators=n_estimators, learning_rate=.045, max_depth=depth,
        num_leaves=leaves, min_child_samples=450, min_split_gain=.001,
        subsample=.82, colsample_bytree=.82, reg_lambda=6.0, reg_alpha=.05,
        max_bin=127, random_state=SEED, n_jobs=n_jobs, verbosity=-1,
    )


def _timestamp_metrics(frame: pd.DataFrame, score: np.ndarray, hurdle: float) -> tuple[dict[str, float], pd.DataFrame]:
    work = frame.loc[:, ["candidate_id", "__decision_ts__", "policy_net_bps", "_target"]].copy()
    work["score"] = score
    rows: list[dict[str, float]] = []
    for stamp, group in work.groupby("__decision_ts__", sort=False):
        group = group.sort_values(["score", "candidate_id"], ascending=[False, True], kind="stable")
        n = len(group)
        top50 = group.head(max(1, int(math.ceil(n * .50))))
        top10 = group.head(max(1, int(math.ceil(n * .10))))
        top2 = group.head(min(2, n))
        positives = int(group._target.sum())
        rows.append({
            "__decision_ts__": stamp,
            "recall50": float(top50._target.sum() / positives) if positives else np.nan,
            "precision10": float(top10._target.mean()),
            "top2_ev_bps": float(pd.to_numeric(top2.policy_net_bps, errors="coerce").mean()),
            "selected50_ev_bps": float(pd.to_numeric(top50.policy_net_bps, errors="coerce").mean()),
            "positive_rows": positives,
        })
    timestamp = pd.DataFrame(rows)
    summary = {
        "timestamp_recall50": float(timestamp.recall50.mean(skipna=True)),
        "timestamp_precision10": float(timestamp.precision10.mean()),
        "timestamp_top2_ev_bps": float(timestamp.top2_ev_bps.mean()),
        "timestamp_selected50_ev_bps": float(timestamp.selected50_ev_bps.mean()),
        "timestamps": int(len(timestamp)),
    }
    return summary, timestamp


def _coverage(root: Path, fields: Sequence[str], months: Sequence[pd.Timestamp]) -> pd.DataFrame:
    count = pd.Series(0, index=fields, dtype=np.int64)
    finite = pd.Series(0, index=fields, dtype=np.int64)
    distinct: dict[str, set[float]] = {field: set() for field in fields}
    for month in months:
        frame = pd.read_parquet(_feature_path(root, f"{month:%Y-%m}"), columns=list(fields))
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        count += len(numeric)
        finite += np.isfinite(numeric.to_numpy(dtype=float)).sum(axis=0)
        # A small deterministic sample is sufficient only for the variance
        # gate; model selection never uses this sample's outcomes.
        for field in fields:
            values = numeric[field].dropna().iloc[::max(1, len(numeric) // 64)].to_numpy(float)
            distinct[field].update(np.round(values[np.isfinite(values)], 8).tolist())
    result = pd.DataFrame({
        "feature": fields,
        "finite_fraction": [float(finite[field] / max(count[field], 1)) for field in fields],
        "sampled_unique": [len(distinct[field]) for field in fields],
    })
    result["coverage_pass"] = result.finite_fraction.ge(.90) & result.sampled_unique.ge(2)
    return result


def _univariate_hold_metrics(train: pd.DataFrame, held: pd.DataFrame, fields: Sequence[str], hurdle: float) -> pd.DataFrame:
    """Timestamp-local one-field recall rescue, direction learnt strictly in train."""
    train_y = train["_target"].to_numpy(float)
    raw = train.loc[:, fields].apply(pd.to_numeric, errors="coerce")
    corr = raw.corrwith(pd.Series(train_y, index=train.index), method="pearson").fillna(0.0)
    # Evaluate only a bounded training-chosen rescue set; this keeps the full
    # universe screen economical without using held outcomes to choose fields.
    candidates = corr.abs().sort_values(ascending=False, kind="stable").head(min(320, len(fields))).index.tolist()
    # Do exactly the old stable per-timestamp sort, but in column batches.
    # The previous nested ``field -> timestamp -> sort`` implementation made
    # the selector effectively serial and, worse, repeated this diagnostic in
    # every subset-ladder fit. ``rank(method='first')`` preserves the incoming
    # row order for ties, which is the same stable-order convention as the old
    # sort.  This is a performance-only rewrite: the training-derived sign and
    # held-month metrics are unchanged.
    query = held["__decision_ts__"]
    target = held["_target"].astype(float)
    query_size = query.groupby(query, sort=False).transform("size").astype(float)
    cut50 = np.maximum(1.0, np.ceil(query_size * .50))
    cut10 = np.maximum(1.0, np.ceil(query_size * .10))
    query_positive = target.groupby(query, sort=False).sum()
    rows: list[dict[str, float]] = []
    batch_size = 32
    for start in range(0, len(candidates), batch_size):
        batch_fields = candidates[start:start + batch_size]
        values = held.loc[:, batch_fields].apply(pd.to_numeric, errors="coerce")
        values = values.fillna(values.median()).fillna(0.0)
        direction = pd.Series({field: 1.0 if corr[field] >= 0 else -1.0 for field in batch_fields})
        values = values.mul(direction, axis="columns")
        rank = values.groupby(query, sort=False).rank(method="first", ascending=False)
        selected50 = rank.le(cut50, axis="index")
        selected10 = rank.le(cut10, axis="index")
        recall = selected50.mul(target, axis="index").groupby(query, sort=False).sum().div(query_positive, axis="index")
        precision = selected10.mul(target, axis="index").groupby(query, sort=False).sum().div(
            cut10.groupby(query, sort=False).first(), axis="index"
        )
        for field in batch_fields:
            rows.append({
                "feature": field,
                "univariate_recall50": float(recall.loc[query_positive.gt(0), field].mean()),
                "univariate_precision10": float(precision[field].mean()),
                "train_corr": float(corr[field]),
            })
    return pd.DataFrame(rows)


def _fold(
    root: Path, policy: pd.DataFrame, fields: Sequence[str], held_month: pd.Timestamp,
    args: argparse.Namespace, *, compute_univariate: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    held_end = _month_end(held_month)
    reserve_start = held_month - pd.Timedelta(days=args.reserve_days)
    train_start = reserve_start - pd.DateOffset(months=args.train_months)
    train_feature = _load_features(root, train_start, reserve_start, fields)
    held_feature = _load_features(root, held_month, held_end, fields)
    train_feature = _query_cap(train_feature, args.train_cap)
    train = _prepare_train(train_feature, policy, reserve_start, args.hurdle_bps)
    held = _prepare_train(held_feature, policy, held_end, args.hurdle_bps)
    if len(train) < 20_000 or len(held) < 5_000:
        raise AssertionError(f"{held_month:%Y-%m}: inadequate support train={len(train)} held={len(held)}")
    model = _ranker(fields, args.depth, args.num_leaves, args.n_estimators, args.n_jobs)
    x_train, medians = _matrix(train, fields)
    model.fit(x_train, train._target.to_numpy(np.int32), group=_groups(train))
    x_held, _ = _matrix(held, fields, medians)
    score = model.predict(x_held)
    metrics, timestamp = _timestamp_metrics(held, score, args.hurdle_bps)
    metrics.update({
        "held_month": held_month.strftime("%Y-%m"), "train_start": str(train_start), "reserve_start": str(reserve_start),
        "train_rows": int(len(train)), "held_rows": int(len(held)),
    })
    imp = pd.DataFrame({
        "feature": fields,
        "gain": model.booster_.feature_importance(importance_type="gain"),
        "split": model.booster_.feature_importance(importance_type="split"),
        "held_month": held_month.strftime("%Y-%m"),
    })
    if compute_univariate:
        uni = _univariate_hold_metrics(train, held, fields, args.hurdle_bps)
        uni["held_month"] = held_month.strftime("%Y-%m")
        imp = imp.merge(uni, on=["feature", "held_month"], how="outer")
    timestamp["held_month"] = held_month.strftime("%Y-%m")
    return pd.DataFrame([metrics]), imp, timestamp


def _union_find_cluster(corr: pd.DataFrame, ranked: pd.DataFrame, threshold: float) -> list[str]:
    parent = {field: field for field in corr.columns}
    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value
    def union(a: str, b: str) -> None:
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a
    values = corr.to_numpy(float)
    names = list(corr.columns)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if np.isfinite(values[i, j]) and abs(values[i, j]) >= threshold:
                union(names[i], names[j])
    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(find(name), []).append(name)
    order = ranked.set_index("feature")["stability_score"].to_dict()
    chosen = [max(group, key=lambda field: (order.get(field, -np.inf), field)) for group in groups.values()]
    return sorted(chosen, key=lambda field: (-order.get(field, -np.inf), field))


def _subset_ladder(root: Path, policy: pd.DataFrame, ordered: Sequence[str], args: argparse.Namespace) -> tuple[pd.DataFrame, dict[int, list[str]]]:
    ladder = [size for size in (350, 250, 200, 125, 80, 50, 30) if size <= len(ordered)]
    if not ladder:
        ladder = [len(ordered)]
    rows: list[dict[str, object]] = []
    contracts: dict[int, list[str]] = {}
    for size in ladder:
        fields = tuple(ordered[:size])
        fold_rows: list[pd.DataFrame] = []
        for held in args.held_months:
            # Univariate rescue is selected once from the complete development
            # screen.  It is not a subset-ladder criterion, so recomputing it
            # here only burns CPU and cannot change the selected subset.
            metric, _, _ = _fold(root, policy, fields, held, args, compute_univariate=False)
            fold_rows.append(metric)
        metrics = pd.concat(fold_rows, ignore_index=True)
        row = {
            "feature_count": size,
            "timestamp_recall50": float(metrics.timestamp_recall50.mean()),
            "worst_month_recall50": float(metrics.timestamp_recall50.min()),
            "timestamp_precision10": float(metrics.timestamp_precision10.mean()),
            "timestamp_top2_ev_bps": float(metrics.timestamp_top2_ev_bps.mean()),
            "worst_month_top2_ev_bps": float(metrics.timestamp_top2_ev_bps.min()),
        }
        rows.append(row)
        contracts[size] = list(fields)
    result = pd.DataFrame(rows).sort_values("feature_count", ascending=False, kind="stable")
    best = float(result.timestamp_recall50.max())
    eligible = result.loc[result.timestamp_recall50.ge(best - args.recall_tolerance)]
    # Prefer the smallest equally-recalling contract, then precision/top-two
    # economic quality and finally its worst monthly recall.
    winner = eligible.sort_values(
        ["feature_count", "timestamp_precision10", "timestamp_top2_ev_bps", "worst_month_recall50"],
        ascending=[True, False, False, False], kind="stable",
    ).iloc[0]
    result["selected"] = result.feature_count.eq(int(winner.feature_count))
    return result, contracts


def run(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(args.out)
    all_fields = _all_fields(args.feature_root, args.held_months[0])
    coverage_months = tuple(sorted({
        *(args.held_months),
        *(_utc(args.held_months[0] - pd.DateOffset(months=4) + pd.DateOffset(months=i)) for i in range(4)),
    }))
    coverage = _coverage(args.feature_root, all_fields, coverage_months)
    eligible = coverage.loc[coverage.coverage_pass, "feature"].tolist()
    if len(eligible) < 150:
        raise AssertionError(f"only {len(eligible)} full-universe fields passed 90% coverage/variance")
    policy = _load_policy(args.policy_path, pd.Timestamp.min.tz_localize("UTC"), pd.Timestamp.max.tz_localize("UTC"))
    args.out.mkdir(parents=True)
    coverage.to_parquet(args.out / "feature_coverage_audit.parquet", index=False, compression="zstd")

    fold_metrics: list[pd.DataFrame] = []
    importance: list[pd.DataFrame] = []
    timestamps: list[pd.DataFrame] = []
    for month in args.held_months:
        metric, imp, timestamp = _fold(args.feature_root, policy, tuple(eligible), month, args)
        fold_metrics.append(metric)
        importance.append(imp)
        timestamps.append(timestamp)
        print(json.dumps({"event": "screened_fold", **metric.iloc[0].to_dict()}, default=str), flush=True)
    folds = pd.concat(fold_metrics, ignore_index=True)
    imp = pd.concat(importance, ignore_index=True).fillna(0.0)
    timestamp = pd.concat(timestamps, ignore_index=True)
    imp["gain_norm"] = imp.groupby("held_month")["gain"].transform(lambda value: value / max(float(value.sum()), 1e-12))
    imp["split_norm"] = imp.groupby("held_month")["split"].transform(lambda value: value / max(float(value.sum()), 1e-12))
    summary = imp.groupby("feature", sort=False).agg(
        gain_median=("gain_norm", "median"), gain_iqr=("gain_norm", lambda value: value.quantile(.75) - value.quantile(.25)),
        split_frequency=("split_norm", lambda value: float((value > 0).mean())),
        univariate_recall50=("univariate_recall50", "mean"),
        univariate_precision10=("univariate_precision10", "mean"),
    ).reset_index()
    baseline_uni = float(summary.univariate_recall50.median())
    summary["univariate_rescue"] = summary.univariate_recall50 - baseline_uni
    summary["stability_score"] = summary.gain_median - .5 * summary.gain_iqr + .05 * summary.split_frequency + .15 * summary.univariate_rescue
    # Union of three development-only screens: stable gain, recurrent split,
    # and timestamp-local one-feature recall.  No held block outside the
    # declared development months participates in this selection.
    chosen = set(summary.nlargest(min(260, len(summary)), "stability_score").feature)
    chosen.update(summary.nlargest(min(220, len(summary)), "split_frequency").feature)
    chosen.update(summary.nlargest(min(180, len(summary)), "univariate_recall50").feature)
    ranked = summary.loc[summary.feature.isin(chosen)].sort_values("stability_score", ascending=False, kind="stable").reset_index(drop=True)
    # Spearman is a redundancy veto only.  Its sample is target-free and does
    # not route, score, or otherwise influence a live decision.
    correlation_start = args.held_months[0] - pd.DateOffset(months=args.train_months + 1)
    corr_frame = _load_features(args.feature_root, correlation_start, args.held_months[0], ranked.feature.tolist())
    if len(corr_frame) > args.correlation_sample_rows:
        corr_frame = corr_frame.iloc[::max(1, len(corr_frame) // args.correlation_sample_rows)].head(args.correlation_sample_rows)
    corr = corr_frame.loc[:, ranked.feature].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
    survivors = _union_find_cluster(corr, ranked, args.redundancy_threshold)
    ranked["survives_redundancy_veto"] = ranked.feature.isin(survivors)
    ordered = [field for field in survivors if field in set(ranked.feature)]
    if len(ordered) < 30:
        raise AssertionError("redundancy veto left fewer than thirty full-universe candidates")

    ladder, contracts = _subset_ladder(args.feature_root, policy, ordered, args)
    winning_size = int(ladder.loc[ladder.selected, "feature_count"].iloc[0])
    fields = contracts[winning_size]
    feature_contract = {"feature_contract": fields, "feature_contract_sha256": hashlib.sha256("\n".join(fields).encode()).hexdigest()}
    (args.out / "selected_full_feature_contract.json").write_text(json.dumps(feature_contract, indent=2, sort_keys=True))
    folds.to_parquet(args.out / "screen_fold_metrics.parquet", index=False, compression="zstd")
    timestamp.to_parquet(args.out / "screen_timestamp_metrics.parquet", index=False, compression="zstd")
    imp.to_parquet(args.out / "feature_fold_importance.parquet", index=False, compression="zstd")
    ranked.to_parquet(args.out / "feature_selection_ranked.parquet", index=False, compression="zstd")
    ladder.to_parquet(args.out / "subset_ladder_metrics.parquet", index=False, compression="zstd")
    _write_exclusive(args.out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "research-only full-causal-universe router selection; no live or execution mutation",
        "feature_root": str(args.feature_root), "policy_path": str(args.policy_path), "policy_sha256": _sha(args.policy_path),
        "development_held_months": [f"{value:%Y-%m}" for value in args.held_months],
        "strict_fold": {"train_months": args.train_months, "reserve_days": args.reserve_days, "label_requirement": "policy label available before reserve start"},
        "target": f"policy_net_bps >= {args.hurdle_bps:g}", "selection_unit": "equal-weight exact decision timestamp",
        "predeclared_primary_metric": "mean timestamp-local top-50% recall of policy_net >= hurdle",
        "tie_breaks": ["timestamp-local top-10% precision", "timestamp-local top-two policy net EV", "worst development month recall"],
        "coverage_gate": "finite coverage >=90% and at least two sampled finite values across train/development periods",
        "selection_pipeline": ["gain/split/stability", "timestamp-local univariate recall rescue", "Spearman redundancy veto", "subset ladder"],
        "mda": "not run by explicit instruction", "full_universe_fields": len(all_fields), "eligible_fields": len(eligible),
        "post_veto_fields": len(ordered), "selected_fields": len(fields), "feature_contract_sha256": feature_contract["feature_contract_sha256"],
        "selected_feature_contract": "selected_full_feature_contract.json", "selection_status": "development selection only; separate HPO/untouched block required",
    })
    print(json.dumps({"event": "complete", "full_fields": len(all_fields), "eligible": len(eligible), "post_veto": len(ordered), "selected": len(fields)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--held-months", type=_parse_months, default=_parse_months("2026-02,2026-03,2026-04"))
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--train-cap", type=int, default=120_000)
    parser.add_argument("--hurdle-bps", type=float, default=50.0)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--n-estimators", type=int, default=140)
    parser.add_argument("--n-jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--recall-tolerance", type=float, default=.005)
    parser.add_argument("--redundancy-threshold", type=float, default=.95)
    parser.add_argument("--correlation-sample-rows", type=int, default=25_000)
    args = parser.parse_args()
    if args.train_months < 2 or args.reserve_days < 12 or args.train_cap < 20_000:
        raise ValueError("invalid strict temporal training specification")
    if not 0.0 <= args.recall_tolerance <= .02 or not .85 <= args.redundancy_threshold <= .99:
        raise ValueError("invalid recall tolerance or redundancy threshold")
    run(args)


if __name__ == "__main__":
    main()
