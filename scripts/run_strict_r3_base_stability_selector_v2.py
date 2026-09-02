#!/usr/bin/env python3
"""Stability-first, conditional feature selection for the routed B/E/T base heads.

This is the offline research implementation of the "Full-Universe Base Feature
Selection v2" contract.  It deliberately replaces the previous one-shot
gain/SHAP/MDA selector with:

* strict causal hygiene and an aggressive point-in-time redundancy veto;
* 40--60 randomized feature/query subspaces per physical head;
* selection on the *complete* timestamp-local B/E/T blend rather than a
  head's native target metric; and
* cross-year blocked OOF folds with whole-query sampling.

The program has no live imports and does not write model or execution bundles.
It emits immutable research receipts for a later compression/HPO stage.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMRanker, LGBMRegressor


ROOT = Path(__file__).resolve().parents[1]
SEED = 1729
IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
SCORE_FIELDS = {"B": "base_bps", "E": "efficiency_bps", "T": "timing_bps"}
HEADS: dict[str, dict[str, object]] = {
    "B": {
        "target": "policy_ordinal_base_grade",
        "valid": "policy_ordinal_base_valid",
        "available": "label_available_ts",
        "objective": "lambdarank",
        "direction": 1.0,
    },
    "E": {
        "target": "supportive_path_efficiency_h12",
        "valid": "supportive_path_valid",
        "available": "supportive_label_available_ts",
        "objective": "huber",
        "direction": 1.0,
    },
    "T": {
        "target": "supportive_time_to_meaningful_mfe_h12",
        "valid": "supportive_path_valid",
        "available": "supportive_label_available_ts",
        "objective": "huber",
        "direction": -1.0,
    },
}
GAIN_G3 = [0.0, 0.5, 2.0, 3.0, 6.0, 8.0]

warnings.filterwarnings("ignore", message="X does not have valid feature names")


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _months(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Timestamp, ...]:
    first = _utc(start).normalize().replace(day=1)
    final = (_utc(end) - pd.Timedelta(nanoseconds=1)).normalize().replace(day=1)
    return tuple(pd.date_range(first, final, freq="MS", tz="UTC"))


def _next_month(stamp: pd.Timestamp) -> pd.Timestamp:
    return _utc(stamp) + pd.offsets.MonthBegin(1)


def _exclusive_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _append_progress(out: Path, **payload: object) -> None:
    with (out / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    for item in paths:
        digest.update(str(item).encode())
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _numeric_fields(feature_root: Path, month: pd.Timestamp) -> list[str]:
    path = feature_root / f"month={month:%Y-%m}" / "causal_feature_universe.parquet"
    schema = pq.ParquetFile(path).schema_arrow
    banned = set(IDENTITY) | {"__ts__", "__symbol__"}
    fields = [
        field.name for field in schema
        if field.name not in banned and pd.api.types.is_numeric_dtype(field.type.to_pandas_dtype())
    ]
    if len(fields) < 1_000:
        raise AssertionError(f"{path}: expected full causal universe, found {len(fields)} numeric fields")
    return fields


def _coverage(feature_root: Path, fields: Sequence[str], months: Sequence[pd.Timestamp]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for month in months:
        path = feature_root / f"month={month:%Y-%m}" / "feature_coverage.parquet"
        data = pd.read_parquet(path, columns=["feature", "rows", "finite_rows", "finite_fraction", "n_unique"])
        data = data.loc[data.feature.isin(fields)].copy()
        data["month"] = f"{month:%Y-%m}"
        parts.append(data)
    frame = pd.concat(parts, ignore_index=True)
    result = frame.groupby("feature", sort=False).agg(
        rows=("rows", "sum"),
        finite_rows=("finite_rows", "sum"),
        min_fold_coverage=("finite_fraction", "min"),
        min_unique=("n_unique", "min"),
        observed_months=("month", "nunique"),
    ).reset_index()
    result["global_coverage"] = result.finite_rows / result.rows.clip(lower=1)
    result["hygiene_pass"] = (
        result.global_coverage.ge(.95)
        & result.min_fold_coverage.ge(.90)
        & result.min_unique.ge(3)
        & result.observed_months.eq(len(months))
    )
    return result.sort_values(["hygiene_pass", "global_coverage", "feature"], ascending=[False, False, True], kind="stable")


def _read_router(router_root: Path, month: pd.Timestamp) -> pd.DataFrame:
    path = router_root / "target_free_scores" / f"month={month:%Y-%m}.parquet"
    data = pd.read_parquet(path, columns=[*IDENTITY, "router_primary_rank"])
    data["__decision_ts__"] = pd.to_datetime(data["__decision_ts__"], utc=True, errors="raise")
    if data.candidate_id.duplicated().any():
        raise AssertionError(f"{path}: duplicate candidate IDs")
    return data


def _route_top_fraction(frame: pd.DataFrame, fraction: float) -> pd.Series:
    ranked = frame.loc[:, ["candidate_id", "__decision_ts__", "router_primary_rank"]].copy()
    ranked["__row__"] = np.arange(len(ranked), dtype=np.int64)
    ranked = ranked.sort_values(
        ["__decision_ts__", "router_primary_rank", "candidate_id"],
        ascending=[True, False, True], kind="stable",
    )
    ranked["__ordinal__"] = ranked.groupby("__decision_ts__", sort=False).cumcount() + 1
    size = ranked.groupby("__decision_ts__", sort=False).candidate_id.transform("size").to_numpy(float)
    selected = ranked.__ordinal__.to_numpy() <= np.ceil(size * fraction)
    output = pd.Series(False, index=np.arange(len(frame)))
    output.iloc[ranked.__row__.to_numpy(np.int64)] = selected
    return output


def _read_scores(score_root: Path, month: pd.Timestamp) -> pd.DataFrame:
    path = score_root / "target_free_monthly" / f"month={month:%Y-%m}" / "scores_features.parquet"
    data = pd.read_parquet(path, columns=[*IDENTITY, *SCORE_FIELDS.values()])
    data["__decision_ts__"] = pd.to_datetime(data["__decision_ts__"], utc=True, errors="raise")
    if data.candidate_id.duplicated().any():
        raise AssertionError(f"{path}: duplicate target-free scores")
    return data


def _read_policy(policy_path: Path) -> pd.DataFrame:
    columns = ["candidate_id", "policy_path_valid", "policy_net_bps", "policy_label_available_ts"]
    data = pd.read_parquet(policy_path, columns=columns)
    data.policy_path_valid = data.policy_path_valid.fillna(False).astype(bool)
    data.policy_net_bps = pd.to_numeric(data.policy_net_bps, errors="coerce")
    data.policy_label_available_ts = pd.to_datetime(data.policy_label_available_ts, utc=True, errors="coerce")
    if data.candidate_id.duplicated().any():
        raise AssertionError("canonical policy labels contain duplicate candidate IDs")
    return data


def _read_head_labels(head: str, label_root: Path, months: Sequence[pd.Timestamp]) -> pd.DataFrame:
    spec = HEADS[head]
    parts: list[pd.DataFrame] = []
    for month in months:
        if head == "B":
            path = label_root / f"month={month:%Y-%m}" / "b0_replacement_targets.parquet"
            columns = ["candidate_id", "label_available_ts", str(spec["valid"]), str(spec["target"])]
        else:
            path = label_root / "parts" / f"month={month:%Y-%m}" / "side=long.parquet"
            columns = [
                "candidate_id", "supportive_label_available_ts", "supportive_path_valid",
                "supportive_target_invalid", str(spec["target"]),
            ]
        data = pd.read_parquet(path, columns=columns)
        available = str(spec["available"])
        if head == "B":
            data = data.rename(columns={"label_available_ts": available})
        data[available] = pd.to_datetime(data[available], utc=True, errors="coerce")
        if head != "B":
            data["supportive_target_invalid"] = data.supportive_target_invalid.fillna(True).astype(bool)
        if data.candidate_id.duplicated().any():
            raise AssertionError(f"{path}: duplicate labels")
        parts.append(data)
    return pd.concat(parts, ignore_index=True)


def _window(
    *, head: str, feature_root: Path, router_root: Path, score_root: Path,
    label_root: Path, policy: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
    route_fraction: float,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for month in _months(start, end):
        feature_path = feature_root / f"month={month:%Y-%m}" / "causal_feature_universe.parquet"
        identities = pd.read_parquet(feature_path, columns=list(IDENTITY))
        identities["__decision_ts__"] = pd.to_datetime(identities["__decision_ts__"], utc=True, errors="raise")
        router = _read_router(router_root, month)
        scores = _read_scores(score_root, month)
        labels = _read_head_labels(head, label_root, (month,))
        data = identities.merge(router, on=list(IDENTITY), how="inner", validate="one_to_one")
        data = data.merge(scores, on=list(IDENTITY), how="inner", validate="one_to_one")
        if len(data) != len(identities):
            raise AssertionError(f"{month:%Y-%m}: target-free feature/router/score identity mismatch")
        data = data.merge(labels, on="candidate_id", how="left", validate="one_to_one")
        data = data.merge(policy, on="candidate_id", how="left", validate="one_to_one")
        data["router_selected"] = _route_top_fraction(data, route_fraction).to_numpy(bool)
        pieces.append(data)
    data = pd.concat(pieces, ignore_index=True)
    data = data.loc[data.__decision_ts__.ge(start) & data.__decision_ts__.lt(end)].copy()
    spec = HEADS[head]
    valid = str(spec["valid"])
    data[valid] = data[valid].fillna(False).astype(bool)
    data.policy_path_valid = data.policy_path_valid.fillna(False).astype(bool)
    data["label_joined"] = data[str(spec["target"])].notna()
    return data


def _hash_order(values: pd.Series, seed: int) -> np.ndarray:
    return pd.util.hash_pandas_object(values.astype(str) + f"|{seed}", index=False).to_numpy(np.uint64)


def _sample_whole_queries(frame: pd.DataFrame, cap: int, *, seed: int) -> pd.DataFrame:
    """Subsample whole decision timestamps, balanced across calendar months."""
    if len(frame) <= cap:
        return frame.copy()
    work = frame.copy()
    work["__month__"] = work.__decision_ts__.dt.strftime("%Y-%m")
    queries = work.loc[:, ["__decision_ts__", "__month__"]].drop_duplicates().copy()
    queries["__hash__"] = _hash_order(queries.__decision_ts__.astype(str), seed)
    counts = work.groupby("__decision_ts__", sort=False).size().rename("__rows__")
    queries = queries.merge(counts, on="__decision_ts__", how="left", validate="one_to_one")
    quota = max(1, cap // max(1, queries.__month__.nunique()))
    keep: list[pd.Timestamp] = []
    for _, month_queries in queries.sort_values(["__month__", "__hash__", "__decision_ts__"], kind="stable").groupby("__month__", sort=False):
        used = 0
        for decision_ts, _month, _hash, row_count in month_queries.itertuples(index=False, name=None):
            if used and used + int(row_count) > quota:
                continue
            keep.append(decision_ts)
            used += int(row_count)
    result = work.loc[work.__decision_ts__.isin(keep)].drop(columns="__month__", errors="ignore")
    if len(result) == 0:
        raise AssertionError("query-safe subsampling produced no rows")
    return result.copy()


def _train_rows(frame: pd.DataFrame, head: str, reserve: pd.Timestamp, cap: int) -> pd.DataFrame:
    spec = HEADS[head]
    target, valid, available = (str(spec["target"]), str(spec["valid"]), str(spec["available"]))
    mask = (
        frame.router_selected
        & frame[valid]
        & frame[available].lt(reserve)
        & frame.policy_label_available_ts.lt(reserve)
        & frame.policy_path_valid
        & np.isfinite(pd.to_numeric(frame[target], errors="coerce"))
    )
    if head != "B":
        mask &= ~frame.supportive_target_invalid.fillna(True).astype(bool)
    return _sample_whole_queries(frame.loc[mask].copy(), cap, seed=SEED + 71)


def _held_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Target-free candidate evaluation: only the policy outcome is joined downstream."""
    mask = frame.router_selected & frame.policy_path_valid & np.isfinite(pd.to_numeric(frame.policy_net_bps, errors="coerce"))
    return frame.loc[mask].copy()


def _materialize(feature_root: Path, selected: pd.DataFrame, fields: Sequence[str]) -> np.ndarray:
    selected = selected.reset_index(drop=True)
    result = np.full((len(selected), len(fields)), np.nan, dtype=np.float32)
    for token, indexes in selected.__decision_ts__.dt.strftime("%Y-%m").groupby(selected.__decision_ts__.dt.strftime("%Y-%m"), sort=True).groups.items():
        path = feature_root / f"month={token}" / "causal_feature_universe.parquet"
        source_ids = pd.read_parquet(path, columns=["candidate_id"])
        lookup = pd.Series(np.arange(len(source_ids), dtype=np.int64), index=source_ids.candidate_id)
        positions = np.asarray(list(indexes), dtype=np.int64)
        source_rows = lookup.reindex(selected.iloc[positions].candidate_id).to_numpy()
        if pd.isna(source_rows).any():
            raise AssertionError(f"{token}: feature source is missing selected candidate IDs")
        for begin in range(0, len(fields), 48):
            end = min(len(fields), begin + 48)
            values = pd.read_parquet(path, columns=list(fields[begin:end])).iloc[source_rows.astype(np.int64)]
            result[np.ix_(positions, np.arange(begin, end))] = values.apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    return result


def _impute(values: np.ndarray, train_count: int) -> np.ndarray:
    medians = np.nanmedian(values[:train_count], axis=0).astype(np.float32)
    medians[~np.isfinite(medians)] = 0.0
    for begin in range(0, values.shape[1], 48):
        end = min(values.shape[1], begin + 48)
        block = values[:, begin:end]
        missing = ~np.isfinite(block)
        if missing.any():
            block[missing] = np.broadcast_to(medians[begin:end], block.shape)[missing]
    return values


def _rank01(frame: pd.DataFrame, score: str) -> np.ndarray:
    ranked = frame.loc[:, ["__decision_ts__", "candidate_id", score]].copy()
    ranked["__row__"] = np.arange(len(ranked), dtype=np.int64)
    ranked["__score__"] = pd.to_numeric(ranked[score], errors="coerce").fillna(-np.inf)
    ranked = ranked.sort_values(["__decision_ts__", "__score__", "candidate_id"], ascending=[True, False, True], kind="stable")
    rank = ranked.groupby("__decision_ts__", sort=False).cumcount().to_numpy(float) + 1.0
    size = ranked.groupby("__decision_ts__", sort=False).candidate_id.transform("size").to_numpy(float)
    output = np.empty(len(frame), dtype=np.float64)
    output[ranked.__row__.to_numpy(np.int64)] = 1.0 - (rank - .5) / size
    return output


def _timestamp_metrics(frame: pd.DataFrame, score: str) -> dict[str, float]:
    data = frame.loc[:, ["__decision_ts__", "candidate_id", "policy_net_bps", score]].copy()
    data["__score__"] = pd.to_numeric(data[score], errors="coerce").fillna(-np.inf)
    data = data.sort_values(["__decision_ts__", "__score__", "candidate_id"], ascending=[True, False, True], kind="stable")
    data["__rank__"] = data.groupby("__decision_ts__", sort=False).cumcount() + 1
    data["__n__"] = data.groupby("__decision_ts__", sort=False).candidate_id.transform("size")
    data["__weight__"] = 1.0 / np.log2(data.__rank__.to_numpy(float) + 1.0)
    summaries: dict[str, pd.Series] = {}
    output: dict[str, float] = {}
    for name, count in (("top10", None), ("top05", 5), ("top02", 2), ("top01", 1)):
        if count is None:
            take = data.__rank__.to_numpy() <= np.ceil(data.__n__.to_numpy(float) * .10)
        else:
            take = data.__rank__.to_numpy() <= count
        selected = data.loc[take]
        weighted = selected.assign(__weighted__=selected.policy_net_bps * selected.__weight__).groupby("__decision_ts__", sort=False)
        values = weighted.__weighted__.sum() / weighted.__weight__.sum()
        summaries[name] = values
        output[f"ts_{name}_ev"] = float(values.mean())
    top10 = summaries["top10"]
    top05 = summaries["top05"]
    top02 = summaries["top02"]
    week = top10.index.isocalendar().year.astype(str) + "-" + top10.index.isocalendar().week.astype(str)
    month = top10.index.tz_localize(None).to_period("M")
    output["weekly_q10_top10"] = float(top10.groupby(week).mean().quantile(.10))
    output["monthly_median_top10"] = float(top10.groupby(month).mean().median())
    output["monthly_q25_top05"] = float(top05.groupby(top05.index.tz_localize(None).to_period("M")).mean().quantile(.25))
    output["weekly_q10_top02"] = float(top02.groupby(top02.index.isocalendar().year.astype(str) + "-" + top02.index.isocalendar().week.astype(str)).mean().quantile(.10))
    output["worst_month_top10"] = float(top10.groupby(month).mean().min())
    output["positive_month_fraction_top10"] = float(top10.groupby(month).mean().gt(0.0).mean())
    output["stable_top10_5_2"] = float(
        .30 * output["ts_top10_ev"]
        + .25 * output["ts_top05_ev"]
        + .20 * output["ts_top02_ev"]
        + .15 * output["monthly_median_top10"]
        + .05 * float(top02.groupby(top02.index.tz_localize(None).to_period("M")).mean().median())
        + .05 * output["monthly_q25_top05"]
        + .05 * output["weekly_q10_top02"]
    )
    for k in (1, 2, 3, 5, 10):
        selected = data.loc[data.__rank__.le(k)]
        output[f"fixed_k{k}_ev"] = float(selected.groupby("__decision_ts__", sort=False).policy_net_bps.mean().mean())
    return output


def _enhanced_score(frame: pd.DataFrame, head: str, candidate: np.ndarray) -> np.ndarray:
    work = frame.loc[:, ["__decision_ts__", "candidate_id", *SCORE_FIELDS.values()]].copy()
    work["__candidate__"] = candidate
    ranks: dict[str, np.ndarray] = {}
    for key, field in SCORE_FIELDS.items():
        ranks[key] = _rank01(work, "__candidate__" if key == head else field)
    return (ranks["B"] + ranks["E"] + ranks["T"]) / 3.0


def _apply_base_override(frame: pd.DataFrame, override: pd.DataFrame | None) -> pd.DataFrame:
    """Use a strict-OOF enhanced-B0 coordinate for conditional E/T tests."""
    if override is None:
        return frame
    joined = frame.merge(override, on=list(IDENTITY), how="left", validate="one_to_one")
    if joined.b0_f72_score.isna().any():
        raise AssertionError("enhanced-B0 OOF ledger does not cover every held screen row")
    joined["base_bps"] = joined.b0_f72_score.to_numpy(float)
    return joined.drop(columns=["b0_f72_score"])


def _groups(frame: pd.DataFrame) -> np.ndarray:
    return frame.groupby("__decision_ts__", sort=False).size().to_numpy(np.int32)


def _model(head: str, *, seed: int, n_jobs: int, cheap: bool) -> LGBMRanker | LGBMRegressor:
    if head == "B":
        return LGBMRanker(
            objective="lambdarank", metric="ndcg", n_estimators=100 if cheap else 240,
            learning_rate=.055 if cheap else .035, max_depth=3 if cheap else 4, num_leaves=15 if cheap else 31,
            min_child_samples=280 if cheap else 190, subsample=.78, subsample_freq=1,
            colsample_bytree=.75, reg_alpha=.05, reg_lambda=6.0, min_split_gain=.001,
            lambdarank_truncation_level=12, label_gain=GAIN_G3, lambdarank_norm=True,
            random_state=seed, deterministic=True, force_col_wise=True, verbosity=-1, n_jobs=n_jobs,
        )
    return LGBMRegressor(
        objective="huber", alpha=.90, n_estimators=100 if cheap else 240,
        learning_rate=.055 if cheap else .035, max_depth=3 if cheap else 4, num_leaves=15 if cheap else 31,
        min_child_samples=280 if cheap else 190, subsample=.78, subsample_freq=1,
        colsample_bytree=.75, reg_alpha=.05, reg_lambda=6.0,
        random_state=seed, deterministic=True, force_col_wise=True, verbosity=-1, n_jobs=n_jobs,
    )


def _correlation_sample(feature_root: Path, fields: Sequence[str], months: Sequence[pd.Timestamp], rows_per_month: int) -> pd.DataFrame:
    samples: list[pd.DataFrame] = []
    for month in months:
        path = feature_root / f"month={month:%Y-%m}" / "causal_feature_universe.parquet"
        identifiers = pd.read_parquet(path, columns=["candidate_id"])
        chosen = np.argsort(_hash_order(identifiers.candidate_id, SEED + int(month.month)), kind="stable")[:rows_per_month]
        blocks: list[pd.DataFrame] = []
        for begin in range(0, len(fields), 48):
            block = pd.read_parquet(path, columns=list(fields[begin:begin + 48])).iloc[chosen]
            blocks.append(block.apply(pd.to_numeric, errors="coerce"))
        samples.append(pd.concat(blocks, axis=1))
    return pd.concat(samples, ignore_index=True)


def _redundancy(fields: Sequence[str], coverage: pd.DataFrame, sample: pd.DataFrame, threshold: float) -> tuple[list[str], pd.DataFrame]:
    parent = list(range(len(fields)))
    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node
    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left
    ranked = np.empty((len(sample), len(fields)), dtype=np.float32)
    for column, field in enumerate(fields):
        values = pd.to_numeric(sample[field], errors="coerce").replace([np.inf, -np.inf], np.nan)
        value = values.rank(method="average", na_option="keep").to_numpy(np.float64)
        fill = float(np.nanmedian(value)) if np.isfinite(value).any() else 0.0
        value = np.nan_to_num(value, nan=fill)
        scale = max(float(value.std()), 1e-12)
        ranked[:, column] = ((value - float(value.mean())) / scale).astype(np.float32)
    approximate_floor = threshold - .01
    pairs: list[tuple[int, int]] = []
    for begin in range(0, len(fields), 40):
        end = min(len(fields), begin + 40)
        corr = np.abs((ranked[:, begin:end].T @ ranked) / max(1, len(sample) - 1))
        for local, row in enumerate(corr):
            index = begin + local
            pairs.extend((index, int(other)) for other in np.flatnonzero(row[index + 1:] >= approximate_floor) + index + 1)
    for left, right in pairs:
        x = pd.to_numeric(sample[fields[left]], errors="coerce")
        y = pd.to_numeric(sample[fields[right]], errors="coerce")
        valid = x.notna() & y.notna()
        if valid.sum() >= 64:
            value = x.loc[valid].corr(y.loc[valid], method="spearman")
            if np.isfinite(value) and abs(float(value)) >= threshold:
                union(left, right)
    groups: dict[int, list[str]] = defaultdict(list)
    for index, field in enumerate(fields):
        groups[find(index)].append(field)
    coverage_map = coverage.set_index("feature").global_coverage.to_dict()
    rows: list[dict[str, object]] = []
    kept: list[str] = []
    for cluster in groups.values():
        # Fold-stable univariate DTP is supplied after the randomized screen;
        # before that evidence exists, coverage is the explicit deterministic
        # tiebreaker and every discarded alternative remains auditable.
        representative = max(cluster, key=lambda value: (coverage_map.get(value, 0.0), -len(value), value))
        kept.append(representative)
        for feature in cluster:
            rows.append({"feature": feature, "cluster_size": len(cluster), "representative": representative, "retained": feature == representative})
    return sorted(kept), pd.DataFrame(rows)


def _subspace_fields(fields: Sequence[str], fraction: float, seed: int) -> list[str]:
    count = max(16, int(math.ceil(len(fields) * fraction)))
    order = np.argsort(_hash_order(pd.Series(fields), seed), kind="stable")
    return [fields[index] for index in order[:count]]


def _screen_fold(
    *, args: argparse.Namespace, head: str, fields: Sequence[str], policy: pd.DataFrame,
    held_month: pd.Timestamp, model_index: int, base_override: pd.DataFrame | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    reserve = held_month - pd.Timedelta(days=args.reserve_days)
    start = reserve - pd.DateOffset(months=args.train_months)
    window = _window(
        head=head, feature_root=args.feature_root, router_root=args.router_root, score_root=args.score_root,
        label_root=args.label_root, policy=policy, start=start, end=_next_month(held_month), route_fraction=args.route_fraction,
    )
    train = _train_rows(window.loc[window.__decision_ts__.lt(reserve)].copy(), head, reserve, args.train_cap)
    held = _sample_whole_queries(_held_rows(window.loc[window.__decision_ts__.ge(held_month)].copy()), args.held_cap, seed=SEED + 900 + model_index)
    if len(train) < args.min_train_rows or len(held) < args.min_held_rows:
        raise AssertionError(f"{head}/{held_month:%Y-%m}: insufficient strict support train={len(train)}, held={len(held)}")
    selected = pd.concat([train, held], ignore_index=True)
    values = _materialize(args.feature_root, selected, fields)
    values = _impute(values, len(train))
    target = pd.to_numeric(train[str(HEADS[head]["target"])], errors="coerce").to_numpy(float)
    metric_rows: list[dict[str, object]] = []
    membership_rows: list[dict[str, object]] = []
    for trial in range(args.random_models):
        seed = SEED + 10_000 * (1 + model_index) + 97 * trial + (0 if head == "B" else 1_000 if head == "E" else 2_000)
        feature_fraction = (.35, .45, .55)[trial % 3]
        subset = _subspace_fields(fields, feature_fraction, seed)
        indices = np.asarray([fields.index(field) for field in subset], dtype=np.int64)
        train_subset = _sample_whole_queries(train, int(len(train) * (.65 + .05 * (trial % 4))), seed=seed)
        lookup = pd.Series(np.arange(len(train), dtype=np.int64), index=train.index)
        positions = lookup.reindex(train_subset.index).to_numpy(np.int64)
        model = _model(head, seed=seed, n_jobs=args.n_jobs, cheap=True)
        if head == "B":
            model.fit(values[positions][:, indices], pd.to_numeric(train_subset[str(HEADS[head]["target"])], errors="coerce").to_numpy(np.int32), group=_groups(train_subset))
        else:
            model.fit(values[positions][:, indices], pd.to_numeric(train_subset[str(HEADS[head]["target"])], errors="coerce").to_numpy(float))
        predicted = float(HEADS[head]["direction"]) * model.predict(values[len(train):][:, indices])
        held_work = held.loc[:, [*IDENTITY, "policy_net_bps", *SCORE_FIELDS.values()]].copy().reset_index(drop=True)
        held_work = _apply_base_override(held_work, base_override)
        held_work["candidate_score"] = predicted
        held_work["enhanced_score"] = _enhanced_score(held_work, head, predicted)
        metrics = _timestamp_metrics(held_work, "enhanced_score")
        record = {
            "head": head, "held_month": f"{held_month:%Y-%m}", "trial": trial, "seed": seed,
            "features": len(subset), "feature_fraction": feature_fraction, "train_rows": len(train_subset),
            "held_rows": len(held_work), **metrics,
        }
        metric_rows.append(record)
        membership_rows.extend({"head": head, "held_month": f"{held_month:%Y-%m}", "trial": trial, "feature": field, "included": True} for field in subset)
        del model, held_work, train_subset
    del values, selected, train, held, window
    gc.collect()
    return metric_rows, membership_rows


def _inclusion(metrics: pd.DataFrame, memberships: pd.DataFrame, fields: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = metrics.groupby(["head", "trial"], sort=False).agg(
        stable_top10_5_2=("stable_top10_5_2", "mean"),
        mean_top10=("ts_top10_ev", "mean"), mean_top05=("ts_top05_ev", "mean"), mean_top02=("ts_top02_ev", "mean"),
        q10_week=("weekly_q10_top10", "mean"), q25_month=("monthly_q25_top05", "mean"),
        positive_month_fraction=("positive_month_fraction_top10", "mean"),
    ).reset_index()
    all_trials = summary.loc[:, ["head", "trial", "stable_top10_5_2", "mean_top10", "q10_week", "q25_month", "positive_month_fraction"]]
    included = memberships.merge(all_trials, on=["head", "trial"], how="inner", validate="many_to_one")
    rows: list[dict[str, object]] = []
    for head in included["head"].unique():
        head_trials = all_trials.loc[all_trials["head"].eq(head)]
        selected = included.loc[included["head"].eq(head)]
        selected_set = selected.groupby("feature", sort=False).trial.agg(set).to_dict()
        for feature in fields:
            in_trials = selected_set.get(feature, set())
            yes = head_trials.loc[head_trials.trial.isin(in_trials)]
            no = head_trials.loc[~head_trials.trial.isin(in_trials)]
            if len(yes) < 3 or len(no) < 3:
                continue
            delta = yes.stable_top10_5_2.to_numpy(float) - float(no.stable_top10_5_2.mean())
            rows.append({
                "head": head, "feature": feature, "included_models": len(yes), "excluded_models": len(no),
                "inclusion_mean": float(delta.mean()), "inclusion_median": float(np.median(delta)),
                "inclusion_iqr": float(np.quantile(delta, .75) - np.quantile(delta, .25)),
                "stable_inclusion": float(np.median(delta) - .5 * (np.quantile(delta, .75) - np.quantile(delta, .25))),
                "positive_model_fraction": float((yes.stable_top10_5_2 > no.stable_top10_5_2.mean()).mean()),
                "positive_month_fraction": float(yes.positive_month_fraction.mean()),
                "worst_trial_uplift": float(delta.min()),
            })
    evidence = pd.DataFrame(rows)
    if evidence.empty:
        raise AssertionError("random-subspace inclusion evidence is empty")
    evidence["selection_rank"] = evidence.groupby("head", sort=False).stable_inclusion.rank(ascending=False, method="first")
    return evidence.sort_values(["head", "selection_rank"], kind="stable"), summary


def _pair_synergy(metrics: pd.DataFrame, memberships: pd.DataFrame, evidence: pd.DataFrame, top_n: int) -> pd.DataFrame:
    score = metrics.groupby(["head", "trial"], sort=False).stable_top10_5_2.mean().rename("score").reset_index()
    rows: list[dict[str, object]] = []
    for head in evidence["head"].unique():
        leaders = evidence.loc[evidence["head"].eq(head)].nsmallest(top_n, "selection_rank").feature.tolist()
        trial_map = memberships.loc[memberships["head"].eq(head)].groupby("trial", sort=False).feature.agg(set).to_dict()
        head_scores = score.loc[score["head"].eq(head)].set_index("trial").score
        for left_index, left in enumerate(leaders):
            for right in leaders[left_index + 1:]:
                groups: dict[tuple[bool, bool], list[float]] = defaultdict(list)
                for trial, value in head_scores.items():
                    chosen = trial_map.get(trial, set())
                    groups[(left in chosen, right in chosen)].append(float(value))
                if min(len(groups[key]) for key in ((True, True), (True, False), (False, True), (False, False))) < 2:
                    continue
                synergy = (
                    np.mean(groups[(True, True)]) - np.mean(groups[(True, False)])
                    - np.mean(groups[(False, True)]) + np.mean(groups[(False, False)])
                )
                rows.append({"head": head, "feature_left": left, "feature_right": right, "pair_synergy": float(synergy), "support": len(groups[(True, True)])})
    return pd.DataFrame(rows).sort_values(["head", "pair_synergy"], ascending=[True, False], kind="stable") if rows else pd.DataFrame(columns=["head", "feature_left", "feature_right", "pair_synergy", "support"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True, help="B labels for B; supportive labels for E/T")
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--base-score-oof", type=Path)
    parser.add_argument("--head", choices=tuple(HEADS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--held-months", nargs="+", default=("2025-11-01", "2026-03-01", "2026-06-01"))
    parser.add_argument("--route-fraction", type=float, default=.50)
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--train-cap", type=int, default=42_000)
    parser.add_argument("--held-cap", type=int, default=28_000)
    parser.add_argument("--min-train-rows", type=int, default=8_000)
    parser.add_argument("--min-held-rows", type=int, default=2_000)
    parser.add_argument("--random-models", type=int, default=48)
    parser.add_argument("--n-jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--correlation-rows-per-month", type=int, default=1_024)
    parser.add_argument("--redundancy-threshold", type=float, default=.97)
    parser.add_argument("--survivor-count", type=int, default=420)
    parser.add_argument("--pair-top-n", type=int, default=200)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    folds = tuple(_utc(value) for value in args.held_months)
    if len(folds) != 3 or tuple(sorted(folds)) != folds:
        raise ValueError("screen stage requires exactly three chronological held months")
    # Portability is a hard contract, not a reporting convenience.  A screen
    # cannot qualify on nearby observations from one market episode: it must
    # cross a calendar-year boundary and cover at least six calendar months.
    if len({fold.year for fold in folds}) < 2 or (folds[-1].year - folds[0].year) * 12 + folds[-1].month - folds[0].month < 6:
        raise ValueError(
            "screen folds must span at least six calendar months and two calendar years"
        )
    if not .30 <= args.route_fraction <= .60 or not .95 <= args.redundancy_threshold <= .99:
        raise ValueError("invalid route fraction or redundancy threshold")
    if args.random_models < 12:
        raise ValueError("random-subspace inclusion requires at least 12 models; the research contract uses 40--60")
    args.out.mkdir(parents=True)
    # The coverage pool includes every feature-selection training and held
    # month, hence spans calendar years rather than reflecting one regime.
    coverage_months = tuple(sorted({month for fold in folds for month in _months(fold - pd.Timedelta(days=args.reserve_days) - pd.DateOffset(months=args.train_months), _next_month(fold))}))
    fields = _numeric_fields(args.feature_root, coverage_months[-1])
    coverage = _coverage(args.feature_root, fields, coverage_months)
    hygienic = coverage.loc[coverage.hygiene_pass, "feature"].tolist()
    if len(hygienic) < 700:
        raise AssertionError(f"hygiene retained only {len(hygienic)} fields")
    correlation = _correlation_sample(args.feature_root, hygienic, coverage_months, args.correlation_rows_per_month)
    survivors, clusters = _redundancy(hygienic, coverage, correlation, args.redundancy_threshold)
    if len(survivors) < 500:
        raise AssertionError(f"rho {args.redundancy_threshold} pruning left too little diversity: {len(survivors)}")
    policy = _read_policy(args.policy_path)
    base_override = None
    if args.base_score_oof is not None:
        base_override = pd.read_parquet(args.base_score_oof, columns=[*IDENTITY, "b0_f72_score"])
        base_override["__decision_ts__"] = pd.to_datetime(base_override["__decision_ts__"], utc=True, errors="raise")
        if base_override.duplicated(list(IDENTITY)).any():
            raise AssertionError("enhanced-B0 OOF ledger has duplicate candidate identities")
    manifest = {
        "schema": "strict_r3_base_stability_selector_v2_screen",
        "scope": "offline research only; never reads/writes live model, inference, exchange, or execution state",
        "head": args.head, "head_contract": HEADS[args.head],
        "feature_root": str(args.feature_root), "router_root": str(args.router_root), "score_root": str(args.score_root),
        "label_root": str(args.label_root), "policy_path": str(args.policy_path),
        "feature_root_sha256": _sha(args.feature_root), "router_root_sha256": _sha(args.router_root),
        "route": {"rank": "router_primary_rank", "fraction": args.route_fraction, "timestamp_local": True},
        "folds": [f"{fold:%Y-%m}" for fold in folds], "coverage_months": [f"{month:%Y-%m}" for month in coverage_months],
        "strict_train": {"label_available_before_reserve": True, "policy_label_available_before_reserve": True, "reserve_days": args.reserve_days, "train_months": args.train_months},
        "base_counterpart": "strict_oof_b0_f72" if base_override is not None else "incumbent_base_bps",
        "hygiene": {"global_coverage": .95, "per_fold_coverage": .90, "redundancy_abs_spearman": args.redundancy_threshold},
        "random_subspaces": {"models": args.random_models, "feature_fraction": [.35, .45, .55], "query_fraction": [.65, .70, .75, .80], "whole_query_sampling": True},
        "selection_metric": "STABLE_TOP10_5_2, computed on enhanced timestamp-rank blend with frozen counterpart heads",
        "target_fields_in_feature_matrix": False,
    }
    _exclusive_json(args.out / "run_manifest.json", manifest)
    coverage.to_parquet(args.out / "feature_hygiene.parquet", index=False, compression="zstd")
    clusters.to_parquet(args.out / "correlation_clusters.parquet", index=False, compression="zstd")
    _append_progress(args.out, stage="hygiene_complete", full_fields=len(fields), hygienic=len(hygienic), post_correlation=len(survivors), coverage_months=manifest["coverage_months"])
    all_metrics: list[dict[str, object]] = []
    all_memberships: list[dict[str, object]] = []
    for fold_index, held_month in enumerate(folds):
        metrics, memberships = _screen_fold(args=args, head=args.head, fields=survivors, policy=policy, held_month=held_month, model_index=fold_index, base_override=base_override)
        all_metrics.extend(metrics)
        all_memberships.extend(memberships)
        _append_progress(args.out, stage="screen_fold_complete", held_month=f"{held_month:%Y-%m}", trials=args.random_models)
    metric_frame = pd.DataFrame(all_metrics)
    membership_frame = pd.DataFrame(all_memberships)
    inclusion, trial_summary = _inclusion(metric_frame, membership_frame, survivors)
    pairs = _pair_synergy(metric_frame, membership_frame, inclusion, args.pair_top_n)
    evidence = inclusion.copy()
    pair_bonus = pd.concat([
        pairs.loc[:, ["head", "feature_left", "pair_synergy"]].rename(columns={"feature_left": "feature"}),
        pairs.loc[:, ["head", "feature_right", "pair_synergy"]].rename(columns={"feature_right": "feature"}),
    ], ignore_index=True).groupby(["head", "feature"], sort=False).pair_synergy.max()
    evidence = evidence.merge(pair_bonus.rename("max_pair_synergy"), on=["head", "feature"], how="left")
    evidence.max_pair_synergy = evidence.max_pair_synergy.fillna(0.0)
    evidence["prescreen_score"] = evidence.stable_inclusion + .15 * evidence.max_pair_synergy + .05 * evidence.positive_model_fraction
    selected = evidence.nlargest(args.survivor_count, "prescreen_score").feature.tolist()
    if len(selected) < min(args.survivor_count, len(survivors)):
        raise AssertionError("pre-screen survivor construction unexpectedly underfilled")
    _exclusive_json(args.out / "prescreen_contract.json", {
        "head": args.head, "features": selected, "feature_count": len(selected),
        "sha256": hashlib.sha256("\n".join(selected).encode()).hexdigest(),
        "selection": "random-subspace stable inclusion, positive-model/month evidence, supported pair synergy; no semantic-family rescue",
    })
    metric_frame.to_parquet(args.out / "random_subspace_fold_metrics.parquet", index=False, compression="zstd")
    trial_summary.to_parquet(args.out / "random_subspace_trial_metrics.parquet", index=False, compression="zstd")
    membership_frame.to_parquet(args.out / "random_subspace_membership.parquet", index=False, compression="zstd")
    evidence.sort_values("prescreen_score", ascending=False, kind="stable").to_parquet(args.out / "feature_inclusion_evidence.parquet", index=False, compression="zstd")
    pairs.to_parquet(args.out / "pair_synergy.parquet", index=False, compression="zstd")
    _append_progress(args.out, stage="screen_complete", prescreen_features=len(selected), sha256=hashlib.sha256("\n".join(selected).encode()).hexdigest())


if __name__ == "__main__":
    main()
