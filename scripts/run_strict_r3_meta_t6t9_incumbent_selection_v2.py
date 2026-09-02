#!/usr/bin/env python3
"""Incumbent-anchored, interaction-aware T6/T9 correction-head selection.

This runner intentionally supersedes the invalid F72-style *replacement*
experiment.  It never changes a physical head's target, query, cap, sampling,
weighting, LightGBM geometry, seed, or HPO during feature selection.  It asks
one question only: do incumbent-preserving additions/removals improve the
actual S11 decision at timestamp ranks 1--5?

The runner is research-only.  It does not write model bundles, MC1, admission,
portfolio, execution, or live state.  Every held score is persisted target-free
before canonical policy outcomes are joined for metrics.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata, spearmanr
from lightgbm import LGBMRegressor

import run_strict_r3_enhanced_base_live_stack_challenger as parent
import run_strict_r3_o3v2_target_funnel as target_contract
import run_strict_r3_meta_t6t9_f72_selection_v1 as legacy


SCHEMA = "strict_r3_meta_t6t9_incumbent_selection_v3"
SEED = 1729
TRAIN_MONTHS = 6
RESERVE_DAYS = 28
LABEL_HORIZON_HOURS = 12
IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
S11_WEIGHTS = {"base": 0.75, "T6": 0.20, "T9": 0.05}
TOP_K = (1, 2, 3, 4, 5)
# The source begins in April 2025, so October is the first complete
# strict-prequential T6/T9 score month.  May--July therefore provide the
# longest available incumbent-preserving selection panel; it is research
# evidence only, not enough folds for promotion.
DEFAULT_HELD_MONTHS = "2026-05,2026-06,2026-07"
PROBE_FIELD_FRACTIONS = (0.15, 0.25, 0.40)
PROBE_MODELS = 60
PROBE_INNER_FOLDS = 3
MAX_SCREEN_ROWS = 30_000
MAX_CORR_ROWS = 18_000
PROBE_TRAIN_CAP = 15_000
PROBE_VALID_CAP = 8_000
PROBE_CANDIDATE_FIELDS = 120
MAX_ADDITIONS = 150
BEAM_WIDTH = 3
MAX_ADDITION_ROUNDS = 5
MAX_FINAL_ADDITIONS = 15
EQUIVALENCE_RELATIVE = 0.005
MIN_STABLE_UPLIFT = 1e-9
PROHIBITED = set(target_contract.PROHIBITED_SCORE_COLUMNS)


@dataclass(frozen=True)
class Fold:
    held_month: pd.Timestamp
    train: pd.DataFrame
    held: pd.DataFrame
    held_policy: pd.DataFrame


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    for item in paths:
        digest.update(str(item).encode())
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _append_progress(root: Path, **payload: object) -> None:
    with (root / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _parse_months(value: str) -> tuple[pd.Timestamp, ...]:
    months = tuple(pd.Timestamp(f"{item.strip()}-01", tz="UTC") for item in value.split(",") if item.strip())
    if not months:
        raise ValueError("--held-months must contain YYYY-MM values")
    return months


def _month_end(month: pd.Timestamp) -> pd.Timestamp:
    return month + pd.offsets.MonthBegin(1)


def _months_between(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Timestamp, ...]:
    first = start.to_period("M").to_timestamp().tz_localize("UTC")
    last = (end - pd.Timedelta(nanoseconds=1)).to_period("M").to_timestamp().tz_localize("UTC")
    return tuple(pd.date_range(first, last, freq="MS"))


def _head_arm(head: str) -> str:
    return "T6_rank_error_ordinal" if head == "T6" else "T9_exit5_ordinal"


def _physical_spec(base_fields: Sequence[str], head: str) -> parent.ConsensusHeadSpec:
    expected = "cap80_ordinary" if head == "T6" else "cap120_equal_month"
    specs = {spec.name: spec for spec in parent.load_conditional_consensus_contract(tuple(base_fields), side="long")}
    if expected not in specs:
        raise AssertionError(f"frozen physical head {expected} is absent")
    return specs[expected]


def _full_base(base_root: Path, start: pd.Timestamp, end: pd.Timestamp, base_fields: Sequence[str]) -> pd.DataFrame:
    columns = list(dict.fromkeys((
        *IDENTITY, "enhanced_base_bps", "base_rank_ts", "base_bps", "efficiency_bps", "timing_bps",
        "e_minus_t", "e_minus_b0", "t_minus_b0", "base_component_std", *base_fields,
    )))
    parts: list[pd.DataFrame] = []
    for month in _months_between(start, end):
        source = base_root / f"month={month:%Y-%m}" / "scores_features.parquet"
        schema = set(pq.ParquetFile(source).schema_arrow.names)
        missing = sorted(set(columns) - schema)
        if missing:
            raise AssertionError(f"{source}: frozen feature contract missing {missing[:5]}")
        data = pd.read_parquet(source, columns=columns)
        leaked = sorted(PROHIBITED.intersection(data.columns))
        if leaked:
            raise AssertionError(f"{source}: target-derived field leaked into base source {leaked}")
        data["__decision_ts__"] = pd.to_datetime(data["__decision_ts__"], utc=True, errors="raise")
        parts.append(data.loc[data.__decision_ts__.ge(start) & data.__decision_ts__.lt(end)].copy())
    out = pd.concat(parts, ignore_index=True)
    if out.duplicated(IDENTITY).any():
        raise AssertionError("base feature source has duplicate row identities")
    return out


def _read_semantics(root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for month in _months_between(start, end):
        source = root / "parts" / f"month={month:%Y-%m}" / "semantics.parquet"
        data = pd.read_parquet(source)
        data["__decision_ts__"] = pd.to_datetime(data["__decision_ts__"], utc=True, errors="raise")
        parts.append(data.loc[data.__decision_ts__.ge(start) & data.__decision_ts__.lt(end)].copy())
    result = pd.concat(parts, ignore_index=True)
    if result.duplicated(IDENTITY).any():
        raise AssertionError("semantic ledger has duplicate identities")
    return result


def _read_policy(path: Path) -> pd.DataFrame:
    policy = pd.read_parquet(path, columns=["candidate_id", "policy_path_valid", "policy_net_bps", "policy_label_available_ts"])
    policy["policy_path_valid"] = policy["policy_path_valid"].fillna(False).astype(bool)
    policy["policy_net_bps"] = pd.to_numeric(policy["policy_net_bps"], errors="coerce")
    policy["policy_label_available_ts"] = pd.to_datetime(policy["policy_label_available_ts"], utc=True, errors="coerce")
    if policy.candidate_id.duplicated().any():
        raise AssertionError("policy ledger has duplicate candidate identities")
    return policy


def _counterpart(score_root: Path, month: pd.Timestamp, arm: str) -> pd.DataFrame:
    source = score_root / "target_free_scores" / arm / f"month={month:%Y-%m}.parquet"
    data = pd.read_parquet(source)
    leaked = sorted(PROHIBITED.intersection(data.columns))
    if leaked:
        raise AssertionError(f"{source}: target columns in supposedly target-free score receipt: {leaked}")
    ranks = [field for field in data.columns if field.startswith("head__") and field.endswith("__rank")]
    if len(ranks) != 1:
        raise AssertionError(f"{source}: expected exactly one head rank")
    data = data.loc[:, [*IDENTITY, ranks[0]]].rename(columns={ranks[0]: arm})
    data["__decision_ts__"] = pd.to_datetime(data["__decision_ts__"], utc=True, errors="raise")
    return data


def _counterpart_window(score_root: Path, start: pd.Timestamp, end: pd.Timestamp, arm: str) -> pd.DataFrame:
    """Load only strict-OOF head ranks whose decision timestamps are in-window.

    The incumbent-selection geometry is allowed to use historical T6/T9
    predictions only when those predictions were made by a model fit before
    the corresponding held month.  A missing monthly receipt is a contract
    failure, rather than an invitation to manufacture an in-sample score.
    """
    pieces: list[pd.DataFrame] = []
    for month in _months_between(start, end):
        source = score_root / "target_free_scores" / arm / f"month={month:%Y-%m}.parquet"
        if not source.exists():
            raise AssertionError(
                f"{arm}: missing strict-OOF incumbent geometry receipt for {month:%Y-%m}"
            )
        data = _counterpart(score_root, month, arm)
        pieces.append(data.loc[data.__decision_ts__.ge(start) & data.__decision_ts__.lt(end)].copy())
    result = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=[*IDENTITY, arm])
    if result.duplicated(IDENTITY).any():
        raise AssertionError(f"{arm}: duplicate strict-OOF incumbent geometry identities")
    return result


def _top30(frame: pd.DataFrame) -> np.ndarray:
    work = frame.loc[:, ["candidate_id", "__decision_ts__", "enhanced_base_bps"]].copy()
    work["__row__"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(["__decision_ts__", "enhanced_base_bps", "candidate_id"], ascending=[True, False, True], kind="stable")
    ordinal = work.groupby("__decision_ts__", sort=False).cumcount().to_numpy() + 1
    total = work.groupby("__decision_ts__", sort=False).candidate_id.transform("size").to_numpy(float)
    result = np.zeros(len(frame), dtype=bool)
    result[work.__row__.to_numpy(np.int64)] = ordinal <= np.ceil(.30 * total)
    return result


def _prepare_folds(
    *, base_root: Path, semantic_root: Path, score_root: Path, policy: pd.DataFrame,
    base_fields: Sequence[str], held_months: Sequence[pd.Timestamp],
) -> tuple[Fold, ...]:
    earliest = pd.Timestamp("2025-04-01", tz="UTC")
    folds: list[Fold] = []
    for month in held_months:
        reserve_start = month - pd.Timedelta(days=RESERVE_DAYS)
        train_start = reserve_start - pd.DateOffset(months=TRAIN_MONTHS)
        if train_start < earliest:
            raise AssertionError(f"{month:%Y-%m}: six-month new-base training support starts {train_start:%Y-%m-%d}, before available source")
        end = _month_end(month)
        base = _full_base(base_root, train_start, end, base_fields)
        semantic = _read_semantics(semantic_root, train_start, reserve_start)
        full = base.merge(semantic, on=IDENTITY, how="left", validate="one_to_one")
        full["enhanced_base_routed"] = _top30(full)
        full = target_contract._base_geometry(full)
        train = full.loc[full.__decision_ts__.lt(reserve_start)].copy()
        valid = (
            train.enhanced_base_routed.fillna(False).astype(bool)
            & train.semantic_path_valid.fillna(False).astype(bool)
            & train.semantic_label_available_ts.lt(reserve_start)
            & np.isfinite(pd.to_numeric(train.semantic_policy_net_bps, errors="coerce"))
        )
        train = train.loc[valid].copy()
        held = full.loc[full.__decision_ts__.ge(month) & full.enhanced_base_routed.fillna(False).astype(bool)].copy()
        # T6/T9 are interaction coordinates during selection.  Training rows
        # must receive their own strict-prequential historical scores, never a
        # same-fold or score-family-mismatched substitute.
        for arm in ("T6_rank_error_ordinal", "T9_exit5_ordinal"):
            geometry = _counterpart_window(score_root, train_start, reserve_start, arm)
            train = train.merge(geometry, on=IDENTITY, how="left", validate="one_to_one")
            if train[arm].isna().any():
                raise AssertionError(f"{month:%Y-%m}: {arm} missing on a strict-OOF training row")
            held_before = len(held)
            held = held.merge(_counterpart(score_root, month, arm), on=IDENTITY, how="inner", validate="one_to_one")
            if len(held) != held_before:
                raise AssertionError(f"{month:%Y-%m}: {arm} strict-OOF held receipt changed routed identities")
        held_policy = held.loc[:, ["candidate_id"]].merge(policy, on="candidate_id", how="left", validate="one_to_one")
        if len(held_policy) != len(held):
            raise AssertionError(f"{month:%Y-%m}: policy metric join changed held target-free identities")
        if len(train) < 20_000 or len(held) < 5_000:
            raise AssertionError(f"{month:%Y-%m}: insufficient support train={len(train)} held={len(held)}")
        folds.append(Fold(month, train.reset_index(drop=True), held.reset_index(drop=True), held_policy.reset_index(drop=True)))
    return tuple(folds)


def _raw_fields(raw_root: Path, month: pd.Timestamp) -> list[str]:
    source = raw_root / f"month={month:%Y-%m}" / "causal_feature_universe.parquet"
    schema = pq.ParquetFile(source).schema_arrow
    blocked = set(IDENTITY) | {"__ts__", "__symbol__"}
    fields = [item.name for item in schema if item.name not in blocked and pd.api.types.is_numeric_dtype(item.type.to_pandas_dtype())]
    if len(fields) < 1_000:
        raise AssertionError(f"{source}: expected full causal universe, found {len(fields)} fields")
    return fields


def _hygiene(raw_root: Path, fields: Sequence[str], months: Sequence[pd.Timestamp]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for month in months:
        source = raw_root / f"month={month:%Y-%m}" / "feature_coverage.parquet"
        data = pd.read_parquet(source, columns=["feature", "rows", "finite_rows", "finite_fraction", "n_unique"])
        data = data.loc[data.feature.isin(fields)].copy()
        data["month"] = f"{month:%Y-%m}"
        parts.append(data)
    data = pd.concat(parts, ignore_index=True)
    result = data.groupby("feature", sort=True).agg(
        rows=("rows", "sum"), finite_rows=("finite_rows", "sum"),
        min_coverage=("finite_fraction", "min"), min_unique=("n_unique", "min"),
        observed_months=("month", "nunique"),
    ).reset_index()
    result["coverage"] = result.finite_rows / result.rows.clip(lower=1)
    result["pass"] = result.coverage.ge(.95) & result.min_coverage.ge(.90) & result.min_unique.ge(3) & result.observed_months.eq(len(months))
    return result.sort_values(["pass", "coverage", "feature"], ascending=[False, False, True], kind="stable")


def _raw_matrix(raw_root: Path, frame: pd.DataFrame, fields: Sequence[str]) -> np.ndarray:
    if not fields:
        return np.empty((len(frame), 0), dtype=np.float32)
    result = np.full((len(frame), len(fields)), np.nan, dtype=np.float32)
    work = frame.reset_index(drop=True)
    for token, positions in work.groupby(work.__decision_ts__.dt.strftime("%Y-%m"), sort=True).groups.items():
        source = raw_root / f"month={token}" / "causal_feature_universe.parquet"
        ids = pd.read_parquet(source, columns=["candidate_id"])
        lookup = pd.Series(np.arange(len(ids), dtype=np.int64), index=ids.candidate_id.astype(str))
        index = np.asarray(list(positions), dtype=np.int64)
        source_rows = lookup.reindex(work.iloc[index].candidate_id.astype(str)).to_numpy()
        if pd.isna(source_rows).any():
            raise AssertionError(f"{token}: raw source misses target-free routed candidate identities")
        for start in range(0, len(fields), 48):
            stop = min(len(fields), start + 48)
            values = pd.read_parquet(source, columns=list(fields[start:stop])).iloc[source_rows.astype(np.int64)]
            result[np.ix_(index, np.arange(start, stop))] = values.apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    return result


def _attach_raw(frame: pd.DataFrame, raw_root: Path, fields: Sequence[str]) -> pd.DataFrame:
    fields = tuple(field for field in fields if field not in frame.columns)
    if not fields:
        return frame.copy()
    values = _raw_matrix(raw_root, frame, fields)
    out = frame.copy()
    for index, field in enumerate(fields):
        out[field] = values[:, index]
    return out


def _rank(frame: pd.DataFrame, values: Sequence[float]) -> np.ndarray:
    work = frame.loc[:, ["candidate_id", "__decision_ts__"]].copy()
    work["value"] = np.asarray(values, dtype=float)
    work["__row__"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(["__decision_ts__", "value", "candidate_id"], ascending=[True, False, True], kind="stable")
    ordinal = work.groupby("__decision_ts__", sort=False).cumcount().to_numpy(float) + 1
    count = work.groupby("__decision_ts__", sort=False).candidate_id.transform("size").to_numpy(float)
    output = np.empty(len(work), dtype=np.float32)
    output[work.__row__.to_numpy(np.int64)] = 1.0 - (ordinal - .5) / count
    return output


def _candidate_spec(spec: parent.ConsensusHeadSpec, additions: Sequence[str]) -> parent.ConsensusHeadSpec:
    fields = tuple(dict.fromkeys((*spec.fields, *additions)))
    return parent.ConsensusHeadSpec(spec.name, spec.cap, spec.weight_mode, spec.query, fields, spec.target_edges_bps, dict(spec.params))


def _fit_score(
    *, train: pd.DataFrame, held: pd.DataFrame, head: str, spec: parent.ConsensusHeadSpec,
    seed: int, n_jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    target, grade, objective, _mode = target_contract._anchor_and_targets(train, _head_arm(head))
    heads, _pair_audit = parent._fit_heads(train, target, (spec,), objective=objective, grade=grade, n_jobs=n_jobs)
    scored = parent._score_heads(held, heads)
    raw = pd.to_numeric(scored[f"head__{spec.name}__raw"], errors="raise").to_numpy(np.float32)
    rank = pd.to_numeric(scored[f"head__{spec.name}__rank"], errors="raise").to_numpy(np.float32)
    return raw, rank


def _combined(held: pd.DataFrame, rank: np.ndarray, head: str) -> np.ndarray:
    base = pd.to_numeric(held.base_rank_ts, errors="raise").to_numpy(float)
    t6 = pd.to_numeric(held.T6_rank_error_ordinal, errors="raise").to_numpy(float)
    t9 = pd.to_numeric(held.T9_exit5_ordinal, errors="raise").to_numpy(float)
    if head == "T6":
        t6 = rank
    else:
        t9 = rank
    return S11_WEIGHTS["base"] * base + S11_WEIGHTS["T6"] * t6 + S11_WEIGHTS["T9"] * t9


def _weights(k: int) -> np.ndarray:
    return 1.0 / np.log2(np.arange(1, k + 1, dtype=float) + 1.0)


def _ordered_metrics(score: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    valid = score.loc[score.policy_path_valid.fillna(False).astype(bool) & np.isfinite(pd.to_numeric(score.policy_net_bps, errors="coerce"))].copy()
    valid = valid.sort_values(["__decision_ts__", "combined_score", "candidate_id"], ascending=[True, False, True], kind="stable")
    valid["rank"] = valid.groupby("__decision_ts__", sort=False).cumcount() + 1
    rows: dict[str, float] = {"metric_timestamps": float(valid.__decision_ts__.nunique()), "metric_rows": float(len(valid))}
    by_timestamp: list[dict[str, object]] = []
    for timestamp, group in valid.groupby("__decision_ts__", sort=False):
        value: dict[str, object] = {"__decision_ts__": timestamp}
        for k in TOP_K:
            selected = group.head(k)
            if selected.empty:
                value[f"top{k}"] = np.nan
                value[f"dtp{k}"] = np.nan
                continue
            y = selected.policy_net_bps.to_numpy(float)
            value[f"top{k}"] = float(np.mean(y))
            w = _weights(len(y))
            value[f"dtp{k}"] = float(np.average(y, weights=w))
        by_timestamp.append(value)
    per_ts = pd.DataFrame(by_timestamp)
    for k in TOP_K:
        rows[f"top{k}"] = float(per_ts[f"top{k}"].mean())
        rows[f"dtp{k}"] = float(per_ts[f"dtp{k}"].mean())
    per_ts["week"] = per_ts.__decision_ts__.dt.to_period("W-SUN").astype(str)
    per_ts["month"] = per_ts.__decision_ts__.dt.strftime("%Y-%m")
    weekly = per_ts.groupby("week", sort=False).agg({"dtp2": "mean", "dtp3": "mean", "dtp5": "mean", "top2": "mean", "top5": "mean"}).reset_index()
    monthly = per_ts.groupby("month", sort=False).agg({"dtp2": "mean", "dtp3": "mean", "dtp5": "mean", "top2": "mean", "top5": "mean"}).reset_index()
    rows["median_month_dtp2"] = float(monthly.dtp2.median())
    rows["q25_month_dtp3"] = float(monthly.dtp3.quantile(.25))
    rows["q25_month_dtp2"] = float(monthly.dtp2.quantile(.25))
    rows["q25_month_dtp5"] = float(monthly.dtp5.quantile(.25))
    rows["q10_week_dtp2"] = float(weekly.dtp2.quantile(.10))
    rows["q25_week_dtp2"] = float(weekly.dtp2.quantile(.25))
    rows["q10_week_dtp5"] = float(weekly.dtp5.quantile(.10))
    rows["worst_week_dtp2"] = float(weekly.dtp2.min())
    rows["worst_month_dtp2"] = float(monthly.dtp2.min())
    rows["positive_week_fraction"] = float((weekly.dtp2.gt(0)).mean())
    rows["positive_month_fraction"] = float((monthly.dtp2.gt(0)).mean())
    tail_mean = .30 * rows["dtp1"] + .25 * rows["dtp2"] + .18 * rows["dtp3"] + .15 * rows["dtp4"] + .12 * rows["dtp5"]
    rows["tail_mean"] = float(tail_mean)
    rows["residual_selection_score"] = float(.65 * tail_mean + .15 * rows["median_month_dtp2"] + .10 * rows["q25_month_dtp3"] + .10 * rows["q10_week_dtp2"])
    return rows, per_ts, valid


def _substitution(candidate: pd.DataFrame, incumbent: pd.DataFrame) -> pd.DataFrame:
    left = candidate.loc[:, ["candidate_id", "__decision_ts__", "rank", "policy_net_bps"]].rename(columns={"rank": "candidate_rank", "policy_net_bps": "candidate_policy"})
    right = incumbent.loc[:, ["candidate_id", "__decision_ts__", "rank", "policy_net_bps"]].rename(columns={"rank": "incumbent_rank", "policy_net_bps": "incumbent_policy"})
    detail = left.merge(right, on=["candidate_id", "__decision_ts__"], how="outer", validate="one_to_one")
    rows: list[dict[str, object]] = []
    for k in TOP_K:
        c = set(map(tuple, left.loc[left.candidate_rank.le(k), ["candidate_id", "__decision_ts__"]].to_numpy()))
        i = set(map(tuple, right.loc[right.incumbent_rank.le(k), ["candidate_id", "__decision_ts__"]].to_numpy()))
        candidate_only = c - i
        incumbent_only = i - c
        cand_y = [float(left.loc[(left.candidate_id.eq(key[0])) & (left.__decision_ts__.eq(key[1])), "candidate_policy"].iloc[0]) for key in candidate_only]
        inc_y = [float(right.loc[(right.candidate_id.eq(key[0])) & (right.__decision_ts__.eq(key[1])), "incumbent_policy"].iloc[0]) for key in incumbent_only]
        delta = float(np.mean(cand_y) - np.mean(inc_y)) if cand_y and inc_y else np.nan
        rows.append({
            "k": k, "candidate_only_rows": len(cand_y), "incumbent_only_rows": len(inc_y),
            "overlap_rows": len(c & i), "overlap_rate": len(c & i) / max(1, len(c | i)),
            "candidate_only_ev": float(np.mean(cand_y)) if cand_y else np.nan,
            "incumbent_only_ev": float(np.mean(inc_y)) if inc_y else np.nan,
            "paired_substitution_delta": delta,
            "positive_substitution_fraction": float(np.mean(np.asarray(cand_y) > 0.0)) if cand_y else np.nan,
        })
    return pd.DataFrame(rows)


def _evaluation_summary(folds: Sequence[dict[str, object]]) -> dict[str, float]:
    data = pd.DataFrame([row["metrics"] for row in folds])
    summary: dict[str, float] = {"folds": float(len(data))}
    for name in data.columns:
        if pd.api.types.is_numeric_dtype(data[name]):
            summary[f"mean_{name}"] = float(data[name].mean())
            summary[f"worst_{name}"] = float(data[name].min())
    return summary


def _materialise_fold(frame: pd.DataFrame, raw_root: Path, additions: Sequence[str]) -> pd.DataFrame:
    return _attach_raw(frame, raw_root, additions)


def _evaluate_contract(
    *, folds: Sequence[Fold], raw_root: Path, head: str, base_spec: parent.ConsensusHeadSpec,
    additions: Sequence[str], n_jobs: int, root: Path | None = None, label: str = "candidate",
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_folds: list[dict[str, object]] = []
    score_parts: list[pd.DataFrame] = []
    per_ts_parts: list[pd.DataFrame] = []
    ordered_parts: list[pd.DataFrame] = []
    spec = _candidate_spec(base_spec, additions)
    for index, fold in enumerate(folds):
        train = _materialise_fold(fold.train, raw_root, additions)
        held = _materialise_fold(fold.held, raw_root, additions)
        raw, rank = _fit_score(train=train, held=held, head=head, spec=spec, seed=SEED + 10_000 * (1 + index), n_jobs=n_jobs)
        target_free = held.loc[:, list(IDENTITY)].copy()
        target_free["head_raw"] = raw
        target_free["head_rank"] = rank
        target_free["held_month"] = f"{fold.held_month:%Y-%m}"
        # Persist target-free before policy is permitted to enter the metric frame.
        if root is not None:
            path = root / "target_free_scores" / label / f"month={fold.held_month:%Y-%m}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                target_free.to_parquet(path, index=False, compression="zstd")
        score = held.loc[:, ["candidate_id", "__decision_ts__", "base_rank_ts", "T6_rank_error_ordinal", "T9_exit5_ordinal"]].copy()
        score["candidate_head_rank"] = rank
        score["combined_score"] = _combined(held, rank, head)
        score = score.merge(fold.held_policy, on="candidate_id", how="left", validate="one_to_one")
        metric, per_ts, ordered = _ordered_metrics(score)
        metric["held_month"] = f"{fold.held_month:%Y-%m}"
        all_folds.append({"metrics": metric})
        score_parts.append(score)
        per_ts["held_month"] = f"{fold.held_month:%Y-%m}"
        per_ts_parts.append(per_ts)
        ordered["held_month"] = f"{fold.held_month:%Y-%m}"
        ordered_parts.append(ordered)
        del train, held, raw, rank
        gc.collect()
    metrics = pd.DataFrame([item["metrics"] for item in all_folds])
    summary = _evaluation_summary(all_folds)
    return summary, metrics, pd.concat(per_ts_parts, ignore_index=True), pd.concat(ordered_parts, ignore_index=True)


def _field_corr_prune(
    *, raw_root: Path, sample: pd.DataFrame, fields: Sequence[str], max_rows: int,
) -> tuple[list[str], pd.DataFrame]:
    sampled = legacy._sample_queries(sample, max_rows, salt=SEED + 101)
    values = _raw_matrix(raw_root, sampled, fields)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    ranks = np.empty_like(values, dtype=np.float32)
    for column in range(values.shape[1]):
        ranks[:, column] = rankdata(values[:, column], method="average").astype(np.float32)
    ranks -= ranks.mean(axis=0, keepdims=True)
    denom = np.sqrt(np.maximum((ranks * ranks).sum(axis=0), 1e-12))
    normalized = ranks / denom
    corr = np.abs(normalized.T @ normalized)
    keep: list[int] = []
    dropped_by: list[str | None] = [None] * len(fields)
    for index in range(len(fields)):
        conflict = next((candidate for candidate in keep if corr[index, candidate] >= .97), None)
        if conflict is None:
            keep.append(index)
        else:
            dropped_by[index] = str(fields[conflict])
    audit = pd.DataFrame({"feature": list(fields), "kept": [index in set(keep) for index in range(len(fields))], "correlation_representative": dropped_by})
    return [str(fields[index]) for index in keep], audit


def _conditional_screen(
    *, folds: Sequence[Fold], raw_root: Path, fields: Sequence[str], head: str, max_rows: int,
) -> pd.DataFrame:
    """Cheap conditional screens before expensive physical-head tests.

    The response is residual target value after removing the contemporaneous
    base-coordinate rank.  This is not an advancement metric: exact S11
    retraining remains the only advancement authority.
    """
    rows: list[dict[str, object]] = []
    for index, fold in enumerate(folds):
        train = legacy._sample_queries(fold.train, max_rows, salt=SEED + 200 + index)
        values = _raw_matrix(raw_root, train, fields)
        target, _grade, _objective, _mode = target_contract._anchor_and_targets(train, _head_arm(head))
        base = pd.to_numeric(train.base_rank_ts, errors="coerce").to_numpy(float)
        residual = rankdata(target) - rankdata(base)
        # Frozen incumbent error cohorts are calculated against a target-free
        # score coordinate supplied by the matched control only on held data;
        # training-screen values are a cheap proxy, never a promotion claim.
        for column, field in enumerate(fields):
            x = values[:, column]
            finite = np.isfinite(x) & np.isfinite(residual)
            if finite.sum() < 200:
                rho = 0.0
            else:
                rho = float(spearmanr(x[finite], residual[finite]).statistic)
                if not np.isfinite(rho):
                    rho = 0.0
            # A simple high/low residual discrimination proxy catches fields
            # relevant to economically wrong upgrades/downgrades.
            threshold = np.nanmedian(x[finite]) if finite.any() else np.nan
            if np.isfinite(threshold) and finite.any():
                high = residual[finite & (x >= threshold)]
                low = residual[finite & (x < threshold)]
                discrimination = float(abs(np.nanmean(high) - np.nanmean(low))) if len(high) and len(low) else 0.0
            else:
                discrimination = 0.0
            rows.append({"fold": f"{fold.held_month:%Y-%m}", "feature": field, "partial_rank_corr": rho, "bad_substitution_proxy": discrimination})
        del values
        gc.collect()
    result = pd.DataFrame(rows)
    summary = result.groupby("feature", sort=False).agg(
        corr_median=("partial_rank_corr", "median"), corr_q25=("partial_rank_corr", lambda x: np.quantile(np.abs(x), .25)),
        discrimination_median=("bad_substitution_proxy", "median"), positive_fold_fraction=("partial_rank_corr", lambda x: float(np.mean(np.asarray(x) > 0))),
    ).reset_index()
    summary["screen_score"] = .55 * summary.corr_median.abs().rank(pct=True) + .30 * summary.discrimination_median.rank(pct=True) + .15 * summary.positive_fold_fraction
    return summary.sort_values(["screen_score", "feature"], ascending=[False, True], kind="stable")


def _inner_slices(frame: pd.DataFrame, count: int) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    timestamps = np.asarray(sorted(frame.__decision_ts__.drop_duplicates().to_list()))
    boundaries = np.linspace(0, len(timestamps), count + 2, dtype=int)
    for index in range(1, count + 1):
        start, end = int(boundaries[index]), int(boundaries[index + 1])
        validation_ts = timestamps[start:end]
        if not len(validation_ts):
            continue
        cutoff = pd.Timestamp(validation_ts[0]) - pd.Timedelta(hours=LABEL_HORIZON_HOURS)
        train = frame.loc[frame.__decision_ts__.lt(cutoff)].copy()
        valid = frame.loc[frame.__decision_ts__.isin(validation_ts)].copy()
        if len(train) >= 5_000 and len(valid) >= 1_000:
            yield train, valid


def _probe_metrics(
    *, train: pd.DataFrame, valid: pd.DataFrame, head: str,
    additions: Sequence[str], seed: int, n_jobs: int,
) -> float:
    """Return a shallow blocked-OOF score for a residual interaction probe.

    This is deliberately *not* an exact physical-head fit.  The v2 protocol
    calls for 50--100 cheap random-subspace probes to discover interacting
    fields before the costly, exact cap/query/weight contract is tested.  The
    probe receives the mandatory incumbent geometry and asks whether a random
    raw-field subset predicts what remains unexplained by the frozen S11
    coordinate.  Exact S11 substitution economics remains the sole authority
    for feature advancement.
    """
    mandatory = (
        "base_rank_ts", "enhanced_base_bps", "base_bps", "efficiency_bps",
        "timing_bps", "T6_rank_error_ordinal", "T9_exit5_ordinal",
    )
    fields = tuple(dict.fromkeys((*mandatory, *additions)))
    # The caller already query-safely sampled before materialising the raw
    # feature subset.  Re-sampling here would make the same probe depend on a
    # second arbitrary population and needlessly duplicate memory.
    sampled_train = train
    sampled_valid = valid
    median = sampled_train.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce").median()
    x_train = sampled_train.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce").fillna(median)
    x_valid = sampled_valid.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce").fillna(median)
    target_train, _grade, _objective, _mode = target_contract._anchor_and_targets(sampled_train, _head_arm(head))
    target_valid, _grade, _objective, _mode = target_contract._anchor_and_targets(sampled_valid, _head_arm(head))
    incumbent_train = (
        S11_WEIGHTS["base"] * pd.to_numeric(sampled_train["base_rank_ts"], errors="coerce").to_numpy(float)
        + S11_WEIGHTS["T6"] * pd.to_numeric(sampled_train["T6_rank_error_ordinal"], errors="coerce").to_numpy(float)
        + S11_WEIGHTS["T9"] * pd.to_numeric(sampled_train["T9_exit5_ordinal"], errors="coerce").to_numpy(float)
    )
    incumbent_valid = (
        S11_WEIGHTS["base"] * pd.to_numeric(sampled_valid["base_rank_ts"], errors="coerce").to_numpy(float)
        + S11_WEIGHTS["T6"] * pd.to_numeric(sampled_valid["T6_rank_error_ordinal"], errors="coerce").to_numpy(float)
        + S11_WEIGHTS["T9"] * pd.to_numeric(sampled_valid["T9_exit5_ordinal"], errors="coerce").to_numpy(float)
    )
    residual_train = rankdata(target_train, method="average") - rankdata(incumbent_train, method="average")
    residual_valid = rankdata(target_valid, method="average") - rankdata(incumbent_valid, method="average")
    model = LGBMRegressor(
        objective="regression_l2", n_estimators=90, learning_rate=.05,
        max_depth=3, num_leaves=7,
        min_child_samples=max(160, int(.03 * len(sampled_train))),
        colsample_bytree=.80, subsample=.80, subsample_freq=1,
        reg_alpha=.05, reg_lambda=5.0, min_split_gain=.002, max_bin=63,
        random_state=seed, n_jobs=n_jobs, deterministic=True,
        force_col_wise=True, verbosity=-1,
    ).fit(x_train, residual_train)
    predicted = model.predict(x_valid)
    value = float(spearmanr(predicted, residual_valid).statistic)
    return value if np.isfinite(value) else 0.0


def _random_probes(
    *, folds: Sequence[Fold], raw_root: Path, head: str, base_spec: parent.ConsensusHeadSpec,
    fields: Sequence[str], probes: int, inner_folds: int, n_jobs: int, root: Path,
) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + (61 if head == "T6" else 79))
    records: list[dict[str, object]] = []
    subsets: list[tuple[float, tuple[str, ...]]] = []
    for number in range(probes):
        fraction = PROBE_FIELD_FRACTIONS[number % len(PROBE_FIELD_FRACTIONS)]
        size = max(1, int(round(fraction * len(fields))))
        subset = tuple(sorted(rng.choice(np.asarray(fields, dtype=object), size=size, replace=False).tolist()))
        subsets.append((fraction, subset))
    # Materialise raw fields only after each inner block has been reduced to
    # its capped, query-safe sample.  This bounds peak memory by roughly
    # (15k + 8k) x 120 fields rather than whole six-month training folds.
    values_by_probe: dict[int, list[float]] = {number: [] for number in range(probes)}
    for fold_index, fold in enumerate(folds):
        for inner_index, (train, valid) in enumerate(_inner_slices(fold.train, inner_folds)):
            train = legacy._sample_queries(train, PROBE_TRAIN_CAP, salt=SEED + 20_000 + 100 * fold_index + inner_index)
            valid = legacy._sample_queries(valid, PROBE_VALID_CAP, salt=SEED + 30_000 + 100 * fold_index + inner_index)
            train = _attach_raw(train, raw_root, fields)
            valid = _attach_raw(valid, raw_root, fields)
            for number, (fraction, subset) in enumerate(subsets):
                value = _probe_metrics(train=train, valid=valid, head=head, additions=subset, seed=SEED + 100_000 + 1000 * number + 100 * fold_index + inner_index, n_jobs=n_jobs)
                values_by_probe[number].append(value)
                records.append({"probe": number, "head": head, "outer_fold": f"{fold.held_month:%Y-%m}", "inner_fold": inner_index, "field_fraction": fraction, "fields": list(subset), "probe_score": value})
            del train, valid
            gc.collect()
    for number, (fraction, subset) in enumerate(subsets):
        _append_progress(root, stage="probe_complete", head=head, probe=number, fields=len(subset), mean_score=float(np.mean(values_by_probe[number])) if values_by_probe[number] else np.nan)
    return pd.DataFrame(records)


def _probe_evidence(probes: pd.DataFrame, fields: Sequence[str]) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    compact = probes.groupby("probe", sort=False).agg(score=("probe_score", "mean"), fields=("fields", "first")).reset_index()
    records: list[dict[str, object]] = []
    for field in fields:
        included = compact.loc[compact.fields.map(lambda values: field in values), "score"].to_numpy(float)
        excluded = compact.loc[~compact.fields.map(lambda values: field in values), "score"].to_numpy(float)
        uplift = float(np.mean(included) - np.mean(excluded)) if len(included) and len(excluded) else -np.inf
        iqr = float(np.subtract(*np.percentile(included, [75, 25]))) if len(included) >= 4 else np.inf
        records.append({"feature": field, "inclusion_uplift": uplift, "inclusion_iqr": iqr, "stable_inclusion": uplift - .5 * iqr, "positive_model_fraction": float(np.mean(included > 0.0)) if len(included) else 0.0, "models_included": int(len(included))})
    evidence = pd.DataFrame(records).sort_values(["stable_inclusion", "feature"], ascending=[False, True], kind="stable")
    top = evidence.head(min(200, len(evidence))).feature.astype(str).tolist()
    pairs: list[tuple[str, str]] = []
    for left_index, left in enumerate(top):
        for right in top[left_index + 1:]:
            with_both = compact.loc[compact.fields.map(lambda values: left in values and right in values), "score"].to_numpy(float)
            only_left = compact.loc[compact.fields.map(lambda values: left in values and right not in values), "score"].to_numpy(float)
            only_right = compact.loc[compact.fields.map(lambda values: right in values and left not in values), "score"].to_numpy(float)
            neither = compact.loc[compact.fields.map(lambda values: left not in values and right not in values), "score"].to_numpy(float)
            if min(len(with_both), len(only_left), len(only_right), len(neither)) < 2:
                continue
            synergy = float(with_both.mean() - only_left.mean() - only_right.mean() + neither.mean())
            if synergy > 0.0:
                pairs.append((left, right))
    pairs.sort()
    return evidence, pairs[:50]


def _advance(
    *, candidate: dict[str, float], incumbent: dict[str, float], substitutions: pd.DataFrame,
) -> bool:
    top12 = substitutions.loc[substitutions.k.le(2), "paired_substitution_delta"].to_numpy(float)
    return bool(
        candidate["mean_residual_selection_score"] > incumbent["mean_residual_selection_score"] + MIN_STABLE_UPLIFT
        and candidate["mean_top2"] >= incumbent["mean_top2"] - 1e-9
        and candidate["mean_top5"] >= incumbent["mean_top5"] - 1e-9
        and candidate["mean_q10_week_dtp2"] >= incumbent["mean_q10_week_dtp2"] - 1e-9
        and candidate["mean_q25_month_dtp3"] >= incumbent["mean_q25_month_dtp3"] - 1e-9
        and np.isfinite(top12).all() and bool((top12 > 0.0).all())
    )


def _candidate_summary(
    *, summary: dict[str, float], additions: Sequence[str], contract_fields: int,
    substitutions: pd.DataFrame, label: str,
) -> dict[str, object]:
    return {"label": label, "additions": list(additions), "addition_count": len(additions), "feature_count": contract_fields + len(additions), **summary,
            "top1_substitution_delta": float(substitutions.loc[substitutions.k.eq(1), "paired_substitution_delta"].iloc[0]),
            "top2_substitution_delta": float(substitutions.loc[substitutions.k.eq(2), "paired_substitution_delta"].iloc[0])}


def _run_exact_additions(
    *, folds: Sequence[Fold], raw_root: Path, head: str, base_spec: parent.ConsensusHeadSpec,
    incumbent_summary: dict[str, float], incumbent_ordered: pd.DataFrame, candidates: Sequence[Sequence[str]],
    n_jobs: int, root: Path, limit: int,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for index, additions in enumerate(candidates[:limit]):
        label = "plus__" + "__".join(additions)
        summary, fold_metrics, per_ts, ordered = _evaluate_contract(folds=folds, raw_root=raw_root, head=head, base_spec=base_spec, additions=additions, n_jobs=n_jobs, root=root, label=label)
        substitutions = _substitution(ordered, incumbent_ordered)
        substitutions["label"] = label
        if not (root / "substitution" / f"{label}.parquet").exists():
            (root / "substitution").mkdir(exist_ok=True)
            substitutions.to_parquet(root / "substitution" / f"{label}.parquet", index=False, compression="zstd")
        record = _candidate_summary(summary=summary, additions=additions, contract_fields=len(base_spec.fields), substitutions=substitutions, label=label)
        record["advance"] = _advance(candidate=summary, incumbent=incumbent_summary, substitutions=substitutions)
        records.append(record)
        _append_progress(root, stage="exact_addition_complete", head=head, index=index, label=label, advance=record["advance"])
        del fold_metrics, per_ts, ordered, substitutions
        gc.collect()
    return pd.DataFrame(records).sort_values(["advance", "mean_residual_selection_score", "label"], ascending=[False, False, True], kind="stable")


def _beam_search(
    *, folds: Sequence[Fold], raw_root: Path, head: str, base_spec: parent.ConsensusHeadSpec,
    incumbent_summary: dict[str, float], incumbent_ordered: pd.DataFrame, seed_rows: pd.DataFrame,
    interaction_pairs: Sequence[tuple[str, str]], n_jobs: int, root: Path,
) -> tuple[list[str], pd.DataFrame]:
    beam: list[tuple[str, ...]] = [tuple()]
    all_rows: list[dict[str, object]] = []
    no_advance = 0
    pool = [tuple(row) if isinstance(row, list) else tuple(str(row).split("__")) for row in seed_rows.head(BEAM_WIDTH * 4).additions]
    pool.extend(tuple(pair) for pair in interaction_pairs[:20])
    for round_index in range(MAX_ADDITION_ROUNDS):
        candidates: set[tuple[str, ...]] = set()
        for base in beam:
            for unit in pool:
                merged = tuple(dict.fromkeys((*base, *unit)))
                if len(merged) <= MAX_FINAL_ADDITIONS and merged != base:
                    candidates.add(merged)
        evaluated: list[tuple[tuple[str, ...], dict[str, object]]] = []
        for additions in sorted(candidates):
            label = f"beam{round_index}__" + "__".join(additions)
            summary, _fold, _per_ts, ordered = _evaluate_contract(folds=folds, raw_root=raw_root, head=head, base_spec=base_spec, additions=additions, n_jobs=n_jobs, root=root, label=label)
            substitutions = _substitution(ordered, incumbent_ordered)
            record = _candidate_summary(summary=summary, additions=additions, contract_fields=len(base_spec.fields), substitutions=substitutions, label=label)
            record["round"] = round_index
            record["advance"] = _advance(candidate=summary, incumbent=incumbent_summary, substitutions=substitutions)
            evaluated.append((additions, record)); all_rows.append(record)
            del _fold, _per_ts, ordered, substitutions
        advanced = [(items, record) for items, record in evaluated if bool(record["advance"])]
        if not advanced:
            no_advance += 1
        else:
            no_advance = 0
        advanced.sort(key=lambda item: (-float(item[1]["mean_residual_selection_score"]), len(item[0]), item[0]))
        beam = [items for items, _record in advanced[:BEAM_WIDTH]] or beam
        _append_progress(root, stage="beam_round_complete", head=head, round=round_index, candidates=len(candidates), advancing=len(advanced))
        if no_advance >= 2:
            break
    records = pd.DataFrame(all_rows)
    if not records.empty:
        records = records.sort_values(["advance", "mean_residual_selection_score", "feature_count"], ascending=[False, False, True], kind="stable")
    winner = list(beam[0]) if beam else []
    return winner, records


def _prune_final(
    *, folds: Sequence[Fold], raw_root: Path, head: str, base_spec: parent.ConsensusHeadSpec,
    incumbent_summary: dict[str, float], incumbent_ordered: pd.DataFrame, additions: Sequence[str],
    n_jobs: int, root: Path,
) -> tuple[list[str], pd.DataFrame]:
    current = list(additions)
    rows: list[dict[str, object]] = []
    changed = True
    while changed and current:
        changed = False
        trials: list[tuple[str, dict[str, object]]] = []
        for field in current:
            candidate = [value for value in current if value != field]
            label = "drop__" + field
            summary, _fold, _per_ts, ordered = _evaluate_contract(folds=folds, raw_root=raw_root, head=head, base_spec=base_spec, additions=candidate, n_jobs=n_jobs, root=root, label=label)
            substitutions = _substitution(ordered, incumbent_ordered)
            record = _candidate_summary(summary=summary, additions=candidate, contract_fields=len(base_spec.fields), substitutions=substitutions, label=label)
            record["removed"] = field
            record["advance"] = _advance(candidate=summary, incumbent=incumbent_summary, substitutions=substitutions)
            trials.append((field, record)); rows.append(record)
            del _fold, _per_ts, ordered, substitutions
        if trials:
            trials.sort(key=lambda item: (-float(item[1]["mean_residual_selection_score"]), len(item[1]["additions"]), item[0]))
            field, best = trials[0]
            # Retain a smaller equivalent contract even when only economically
            # indistinguishable from the current challenger.
            if best["advance"]:
                current.remove(field); changed = True
                _append_progress(root, stage="backward_prune_remove", head=head, feature=field)
    return current, pd.DataFrame(rows)


def _run_head(
    *, head: str, folds: Sequence[Fold], raw_root: Path, base_spec: parent.ConsensusHeadSpec,
    raw_fields: Sequence[str], n_jobs: int, root: Path, probes: int, max_additions: int,
) -> dict[str, object]:
    head_root = root / head.lower(); head_root.mkdir(exist_ok=True)
    # Exact control: no raw additions and exact frozen physical contract.
    control_summary, control_fold, control_ts, control_ordered = _evaluate_contract(folds=folds, raw_root=raw_root, head=head, base_spec=base_spec, additions=(), n_jobs=n_jobs, root=head_root, label="frozen_control")
    if not (head_root / "frozen_control_fold_metrics.parquet").exists():
        control_fold.to_parquet(head_root / "frozen_control_fold_metrics.parquet", index=False, compression="zstd")
        control_ts.to_parquet(head_root / "frozen_control_timestamp_metrics.parquet", index=False, compression="zstd")
        control_ordered.to_parquet(head_root / "frozen_control_ordered_metrics.parquet", index=False, compression="zstd")
        _write_json_exclusive(head_root / "frozen_control_summary.json", {"head": head, "physical_spec": {"name": base_spec.name, "cap": base_spec.cap, "weight_mode": base_spec.weight_mode, "query": base_spec.query, "params": base_spec.params, "fields": list(base_spec.fields)}, "metrics": control_summary})
    # The interaction screen is only valid when its mandatory frozen T6/T9
    # coordinates are themselves strict-OOF on every training row.  The
    # currently available baseline receipt starts in the held months, so using
    # it in training would either be in-sample or a mismatched score family.
    # Fail closed with a concrete materialisation requirement; the exact
    # frozen control above remains a valid diagnostic but cannot authorize a
    # feature candidate on its own.
    required_geometry = {"T6_rank_error_ordinal", "T9_exit5_ordinal"}
    missing_geometry = sorted(required_geometry - set(folds[0].train.columns))
    if missing_geometry:
        support = {
            "head": head,
            "status": "SUPPORT_INSUFFICIENT_STRICT_OOF_INCUMBENT_GEOMETRY",
            "missing_training_coordinates": missing_geometry,
            "available_held_months": [f"{fold.held_month:%Y-%m}" for fold in folds],
            "required": "target-free, strict-prequential T6/T9 prediction ledger over every six-month training window; no in-sample or mismatched-base substitute is allowed",
            "frozen_control_metrics": control_summary,
            "decision": "RETAIN_FROZEN_CONTROL_RESEARCH_ONLY",
        }
        if not (head_root / "support_insufficient.json").exists():
            _write_json_exclusive(head_root / "support_insufficient.json", support)
        return support
    # Correlation pruning and conditional residual screen are exploration-only;
    # their labels never enter held prediction frames.
    sample = pd.concat([fold.train for fold in folds], ignore_index=True)
    prune_fields, corr = _field_corr_prune(raw_root=raw_root, sample=sample, fields=raw_fields, max_rows=MAX_CORR_ROWS)
    if not (head_root / "correlation_prune.parquet").exists():
        corr.to_parquet(head_root / "correlation_prune.parquet", index=False, compression="zstd")
    screen = _conditional_screen(folds=folds, raw_root=raw_root, fields=prune_fields, head=head, max_rows=MAX_SCREEN_ROWS)
    if not (head_root / "conditional_screen.parquet").exists():
        screen.to_parquet(head_root / "conditional_screen.parquet", index=False, compression="zstd")
    screened = screen.head(min(PROBE_CANDIDATE_FIELDS, len(screen))).feature.astype(str).tolist()
    probes_data = _random_probes(folds=folds, raw_root=raw_root, head=head, base_spec=base_spec, fields=screened, probes=probes, inner_folds=PROBE_INNER_FOLDS, n_jobs=n_jobs, root=head_root)
    if not (head_root / "random_subspace_probes.parquet").exists():
        probes_data.to_parquet(head_root / "random_subspace_probes.parquet", index=False, compression="zstd")
    evidence, pairs = _probe_evidence(probes_data, screened)
    if not (head_root / "interaction_evidence.parquet").exists():
        evidence.to_parquet(head_root / "interaction_evidence.parquet", index=False, compression="zstd")
        _write_json_exclusive(head_root / "interaction_pairs.json", {"head": head, "pairs": [list(pair) for pair in pairs], "count": len(pairs)})
    candidates = [(field,) for field in evidence.head(max_additions).feature.astype(str).tolist()]
    # Include proven interaction blocks in exact testing without allowing a
    # standalone replacement contract.
    candidates.extend(tuple(pair) for pair in pairs[:min(50, max_additions // 3)])
    exact = _run_exact_additions(folds=folds, raw_root=raw_root, head=head, base_spec=base_spec, incumbent_summary=control_summary, incumbent_ordered=control_ordered, candidates=candidates, n_jobs=n_jobs, root=head_root, limit=max_additions)
    if not (head_root / "exact_additions.parquet").exists():
        exact.to_parquet(head_root / "exact_additions.parquet", index=False, compression="zstd")
    winner_additions, beam = _beam_search(folds=folds, raw_root=raw_root, head=head, base_spec=base_spec, incumbent_summary=control_summary, incumbent_ordered=control_ordered, seed_rows=exact.loc[exact.advance].copy(), interaction_pairs=pairs, n_jobs=n_jobs, root=head_root)
    if not (head_root / "beam_search.parquet").exists():
        beam.to_parquet(head_root / "beam_search.parquet", index=False, compression="zstd")
    final_additions, prune = _prune_final(folds=folds, raw_root=raw_root, head=head, base_spec=base_spec, incumbent_summary=control_summary, incumbent_ordered=control_ordered, additions=winner_additions, n_jobs=n_jobs, root=head_root)
    if not (head_root / "backward_pruning.parquet").exists():
        prune.to_parquet(head_root / "backward_pruning.parquet", index=False, compression="zstd")
    final_summary, final_fold, final_ts, final_ordered = _evaluate_contract(folds=folds, raw_root=raw_root, head=head, base_spec=base_spec, additions=final_additions, n_jobs=n_jobs, root=head_root, label="final")
    substitution = _substitution(final_ordered, control_ordered)
    promotion_support = len(folds) >= 4
    promoted = promotion_support and _advance(candidate=final_summary, incumbent=control_summary, substitutions=substitution)
    decision = {"head": head, "control_metrics": control_summary, "candidate_metrics": final_summary, "additions": final_additions, "promotion_support_sufficient": promotion_support, "advance_economic_gate": _advance(candidate=final_summary, incumbent=control_summary, substitutions=substitution), "decision": "ADVANCE_TO_MATCHED_MC1_PORTFOLIO" if promoted else "RETAIN_FROZEN_CONTROL_RESEARCH_ONLY"}
    if not (head_root / "decision.json").exists():
        final_fold.to_parquet(head_root / "final_fold_metrics.parquet", index=False, compression="zstd")
        final_ts.to_parquet(head_root / "final_timestamp_metrics.parquet", index=False, compression="zstd")
        substitution.to_parquet(head_root / "final_substitution.parquet", index=False, compression="zstd")
        _write_json_exclusive(head_root / "decision.json", decision)
    return decision


def run(args: argparse.Namespace) -> None:
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output root already exists: {out}")
    held_months = _parse_months(args.held_months)
    paths = parent.Paths(*(Path("unused") for _ in range(5)), args.bundle_root)
    base_fields = parent._base_fields(paths)
    specs = {"T6": _physical_spec(base_fields, "T6"), "T9": _physical_spec(base_fields, "T9")}
    policy = _read_policy(args.policy_path)
    folds = _prepare_folds(base_root=args.base_root, semantic_root=args.semantic_root, score_root=args.baseline_score_root, policy=policy, base_fields=base_fields, held_months=held_months)
    all_months = tuple(pd.Timestamp(value, tz="UTC") for value in pd.date_range("2025-11-01", "2026-07-01", freq="MS").strftime("%Y-%m-01"))
    raw = _raw_fields(args.raw_feature_root, all_months[0])
    incumbent = set(specs["T6"].fields) | set(specs["T9"].fields)
    hygiene = _hygiene(args.raw_feature_root, raw, all_months)
    raw = hygiene.loc[
        hygiene["pass"] & ~hygiene["feature"].isin(incumbent),
        "feature",
    ].astype(str).tolist()
    manifest = {
        "schema": SCHEMA, "scope": "offline research only; does not mutate live/MC1/admission/portfolio/execution", "held_months": [f"{month:%Y-%m}" for month in held_months],
        "base_root": str(args.base_root), "raw_feature_root": str(args.raw_feature_root), "semantic_root": str(args.semantic_root), "baseline_score_root": str(args.baseline_score_root), "policy_path": str(args.policy_path), "bundle_root": str(args.bundle_root),
        "source_hashes": {"base_root": _sha(args.base_root), "raw_feature_root": _sha(args.raw_feature_root), "semantic_root": _sha(args.semantic_root), "baseline_score_root": _sha(args.baseline_score_root), "policy_path": _sha(args.policy_path)},
        "contracts": {"S11_weights": S11_WEIGHTS, "top_k": TOP_K, "physical": {head: {"name": spec.name, "cap": spec.cap, "weight_mode": spec.weight_mode, "query": spec.query, "params": spec.params, "fields": list(spec.fields)} for head, spec in specs.items()}, "selection": "incumbent-additive, interaction-aware, target-free held scores before policy metrics", "promotion_min_held_months": 4},
        "support": {"folds": len(folds), "raw_candidate_fields_after_hygiene_and_incumbent_dedup": len(raw), "training_rows": {f"{fold.held_month:%Y-%m}": len(fold.train) for fold in folds}, "held_rows": {f"{fold.held_month:%Y-%m}": len(fold.held) for fold in folds}},
    }
    out.mkdir(parents=True)
    _write_json_exclusive(out / "run_manifest.json", manifest)
    hygiene.to_parquet(out / "hygiene.parquet", index=False, compression="zstd")
    decisions: dict[str, object] = {}
    # T6 must complete and freeze before T9's final challenger is assessed.
    decisions["T6"] = _run_head(head="T6", folds=folds, raw_root=args.raw_feature_root, base_spec=specs["T6"], raw_fields=raw, n_jobs=args.n_jobs, root=out, probes=args.probes, max_additions=args.max_additions)
    # Until T6 has >=4 held-month support, T9 deliberately retains the frozen
    # T6 control; the future valid extension can re-run both in sequence.
    decisions["T9"] = _run_head(head="T9", folds=folds, raw_root=args.raw_feature_root, base_spec=specs["T9"], raw_fields=raw, n_jobs=args.n_jobs, root=out, probes=args.probes, max_additions=args.max_additions)
    _write_json_exclusive(out / "decision.json", {"schema": SCHEMA, "heads": decisions, "combined_status": "NO_PROMOTION_WITHOUT_FOUR_HELD_MONTHS_AND_MATCHED_MC1_PORTFOLIO"})
    _write_json_exclusive(out / "correctness_report.json", {"target_free_before_policy_metric_join": True, "exact_physical_contracts_preserved": True, "base_labels_rebuilt_from_new_base": True, "held_month_support": len(folds), "promotion_blocked_by_support": len(folds) < 4, "no_live_mutation": True})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--raw-feature-root", type=Path, required=True)
    parser.add_argument("--semantic-root", type=Path, required=True)
    parser.add_argument("--baseline-score-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--held-months", default=DEFAULT_HELD_MONTHS)
    parser.add_argument("--probes", type=int, default=PROBE_MODELS)
    parser.add_argument("--max-additions", type=int, default=MAX_ADDITIONS)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    if args.probes < 8 or args.max_additions < 8 or args.n_jobs < 1:
        parser.error("probes/max-additions must be >=8 and n-jobs >=1")
    run(args)


if __name__ == "__main__":
    main()
