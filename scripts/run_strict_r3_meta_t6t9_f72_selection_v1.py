#!/usr/bin/env python3
"""Target-aware F72-style selection and HPO for T6/T9 on the new B/E/T base.

The runner deliberately reuses the *sequence* that selected F72 for B0 while
changing the supervised target and economic selector for correction heads:

``hygiene -> full-model gain/tail-SHAP screen -> univariate rescue ->
randomised stability -> Screen120 -> OOF economic/boundary MDA -> semantic
family MDA -> 120/90/70/50/35/25 ladder -> family add-back -> HPO -> freeze``.

T6 is trained on its train-only rank-error ordinal target and T9 on its
train-only exit-quality ordinal target.  CMI is intentionally not a ranking
term: target-specific relevance and incremental timestamp-local blend
economics are what matter here.  All held score vectors are constructed before
canonical policy outcomes are joined for diagnostics.

This is an offline research runner.  It cannot modify base models, MC1,
admission, portfolio, live inference, or exchange state.
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
from typing import Any, Iterable, Sequence

import numpy as np
import optuna
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMRegressor, early_stopping
from scipy.stats import rankdata

import run_strict_r3_o3v2_target_funnel as target_contract


SCHEMA = "strict_r3_meta_t6t9_f72_selection_v1"
SEED = 1729
TRAIN_MONTHS = 6
RESERVE_DAYS = 28
MAX_TRAIN_ROWS = 60_000
MAX_HELD_ROWS = 18_000
SCREEN_KEEP = 120
SIZES = (120, 90, 70, 50, 35, 25)
IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
CORE = (
    "base_rank_ts", "base_bps", "efficiency_bps", "timing_bps", "enhanced_base_bps",
    "e_minus_t", "e_minus_b0", "t_minus_b0", "base_component_std",
    "o3v2_b0_rank", "o3v2_e_rank", "o3v2_t_rank", "o3v2_coord_min",
    "o3v2_coord_max", "o3v2_coord_median", "o3v2_coord_std", "o3v2_coord_range",
    "o3v2_query_count", "o3v2_query_std", "o3v2_query_range",
    "o3v2_query_top_gap", "o3v2_query_top2_gap",
)
PROHIBITED = set(target_contract.PROHIBITED_SCORE_COLUMNS)


@dataclass(frozen=True)
class Fold:
    held_month: pd.Timestamp
    train: pd.DataFrame
    held: pd.DataFrame
    held_policy: pd.DataFrame


def _exclusive_json(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _progress(root: Path, **payload: object) -> None:
    with (root / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    sources = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    for source in sources:
        digest.update(str(source).encode())
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _month_end(month: pd.Timestamp) -> pd.Timestamp:
    return month + pd.offsets.MonthBegin(1)


def _months_between(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Timestamp, ...]:
    return tuple(pd.date_range(start.to_period("M").to_timestamp().tz_localize("UTC"), (end - pd.Timedelta(nanoseconds=1)).to_period("M").to_timestamp().tz_localize("UTC"), freq="MS"))


def _parse_months(raw: str) -> tuple[pd.Timestamp, ...]:
    values = tuple(pd.Timestamp(f"{item.strip()}-01", tz="UTC") for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("--held-months must contain one or more YYYY-MM values")
    return values


def _family(field: str) -> str:
    name = str(field).lower()
    if any(token in name for token in ("fund", "oi_", "open_interest", "leverage")):
        return "funding_open_interest_flow"
    if any(token in name for token in ("spread", "impact", "depth", "amihud", "liquidity", "ob_")):
        return "liquidity_microstructure"
    if any(token in name for token in ("vol", "rv", "atr", "tail", "cvar", "variance")):
        return "volatility_tails"
    if any(token in name for token in ("trend", "ret", "adx", "ker", "chop", "momentum", "dir_")):
        return "returns_trend"
    if any(token in name for token in ("support", "resistance", "donchian", "pivot", "range", "loc_", "dist_")):
        return "structure_location"
    if any(token in name for token in ("btc", "eth", "mkt", "beta", "corr", "peer", "cross")):
        return "cross_asset_market"
    if any(token in name for token in ("regime", "state", "entropy", "transition", "gmm", "cluster")):
        return "regime_state"
    return "other_causal"


def _numeric_fields(feature_root: Path, month: pd.Timestamp) -> list[str]:
    path = feature_root / f"month={month:%Y-%m}" / "causal_feature_universe.parquet"
    schema = pq.ParquetFile(path).schema_arrow
    forbidden = set(IDENTITY) | {"__ts__", "__symbol__"}
    fields = [
        value.name for value in schema
        if value.name not in forbidden and pd.api.types.is_numeric_dtype(value.type.to_pandas_dtype())
    ]
    if len(fields) < 1_000:
        raise AssertionError(f"{path}: expected full causal universe, found only {len(fields)} numeric fields")
    return fields


def _hygiene(feature_root: Path, fields: Sequence[str], months: Sequence[pd.Timestamp]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for month in months:
        source = feature_root / f"month={month:%Y-%m}" / "feature_coverage.parquet"
        data = pd.read_parquet(source, columns=["feature", "rows", "finite_rows", "finite_fraction", "n_unique"])
        data = data.loc[data.feature.isin(fields)].copy()
        data["month"] = f"{month:%Y-%m}"
        parts.append(data)
    coverage = pd.concat(parts, ignore_index=True)
    summary = coverage.groupby("feature", sort=True).agg(
        rows=("rows", "sum"), finite_rows=("finite_rows", "sum"),
        min_coverage=("finite_fraction", "min"), min_unique=("n_unique", "min"),
        observed_months=("month", "nunique"),
    ).reset_index()
    summary["coverage"] = summary.finite_rows / summary.rows.clip(lower=1)
    summary["hygiene_pass"] = (
        summary.coverage.ge(.95) & summary.min_coverage.ge(.90)
        & summary.min_unique.ge(3) & summary.observed_months.eq(len(months))
    )
    summary["family"] = summary.feature.map(_family)
    return summary.sort_values(["hygiene_pass", "coverage", "feature"], ascending=[False, False, True], kind="stable")


def _read_base(feature_root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for month in _months_between(start, end):
        source = feature_root / f"month={month:%Y-%m}" / "scores_features.parquet"
        if not source.exists():
            raise FileNotFoundError(source)
        frame = pd.read_parquet(source, columns=[*IDENTITY, "enhanced_base_bps", "base_rank_ts", "base_bps", "efficiency_bps", "timing_bps", "e_minus_t", "e_minus_b0", "t_minus_b0", "base_component_std"])
        leaked = sorted(PROHIBITED.intersection(frame.columns))
        if leaked:
            raise AssertionError(f"{source}: base source is not target-free: {leaked}")
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
        pieces.append(frame.loc[frame.__decision_ts__.ge(start) & frame.__decision_ts__.lt(end)].copy())
    output = pd.concat(pieces, ignore_index=True)
    if output.duplicated(IDENTITY).any():
        raise AssertionError("new-base target-free source has duplicate identities")
    return output


def _read_semantics(root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for month in _months_between(start, end):
        source = root / "parts" / f"month={month:%Y-%m}" / "semantics.parquet"
        if not source.exists():
            raise FileNotFoundError(source)
        data = pd.read_parquet(source)
        data["__decision_ts__"] = pd.to_datetime(data["__decision_ts__"], utc=True, errors="raise")
        parts.append(data.loc[data.__decision_ts__.ge(start) & data.__decision_ts__.lt(end)].copy())
    result = pd.concat(parts, ignore_index=True)
    if result.duplicated(["candidate_id", "__decision_ts__", "side_name"]).any():
        raise AssertionError("semantic source has duplicate identities")
    return result


def _read_policy(path: Path) -> pd.DataFrame:
    output = pd.read_parquet(path, columns=["candidate_id", "policy_path_valid", "policy_net_bps", "policy_label_available_ts"])
    output.policy_path_valid = output.policy_path_valid.fillna(False).astype(bool)
    output.policy_net_bps = pd.to_numeric(output.policy_net_bps, errors="coerce")
    output.policy_label_available_ts = pd.to_datetime(output.policy_label_available_ts, utc=True, errors="coerce")
    if output.candidate_id.duplicated().any():
        raise AssertionError("policy source has duplicate candidate IDs")
    return output


def _read_counterpart(score_root: Path, month: pd.Timestamp, head: str) -> pd.DataFrame:
    source = score_root / "target_free_scores" / head / f"month={month:%Y-%m}.parquet"
    data = pd.read_parquet(source)
    leaked = sorted(PROHIBITED.intersection(data.columns))
    if leaked:
        raise AssertionError(f"{source}: counterpart score leaks outcomes: {leaked}")
    ranks = [name for name in data.columns if name.startswith("head__") and name.endswith("__rank")]
    if len(ranks) != 1:
        raise AssertionError(f"{source}: expected one frozen head rank")
    data = data.loc[:, [*IDENTITY, ranks[0]]].rename(columns={ranks[0]: head})
    data["__decision_ts__"] = pd.to_datetime(data["__decision_ts__"], utc=True, errors="raise")
    return data


def _top30(frame: pd.DataFrame) -> pd.Series:
    work = frame.loc[:, ["candidate_id", "__decision_ts__", "enhanced_base_bps"]].copy()
    work["__row__"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(["__decision_ts__", "enhanced_base_bps", "candidate_id"], ascending=[True, False, True], kind="stable")
    ordinal = work.groupby("__decision_ts__", sort=False).cumcount().to_numpy() + 1
    total = work.groupby("__decision_ts__", sort=False).candidate_id.transform("size").to_numpy(float)
    output = np.zeros(len(frame), dtype=bool)
    output[work.__row__.to_numpy(np.int64)] = ordinal <= np.ceil(total * .30)
    return pd.Series(output, index=frame.index)


def _hash(values: pd.Series, salt: int) -> np.ndarray:
    return pd.util.hash_pandas_object(values.astype(str) + f"|{salt}", index=False).to_numpy(np.uint64)


def _sample_queries(frame: pd.DataFrame, cap: int, *, salt: int) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame.reset_index(drop=True).copy()
    work = frame.copy()
    work["__month__"] = work.__decision_ts__.dt.strftime("%Y-%m")
    queries = work.loc[:, ["__decision_ts__", "__month__"]].drop_duplicates().copy()
    sizes = work.groupby("__decision_ts__", sort=False).size().rename("__rows__")
    queries = queries.merge(sizes, on="__decision_ts__", how="left", validate="one_to_one")
    queries["__hash__"] = _hash(queries["__decision_ts__"].astype(str), salt)
    quota = max(1, cap // max(1, queries.__month__.nunique()))
    keep: list[pd.Timestamp] = []
    for _, group in queries.sort_values(["__month__", "__hash__", "__decision_ts__"], kind="stable").groupby("__month__", sort=False):
        used = 0
        for timestamp, _month, size, _hash_value in group.itertuples(index=False, name=None):
            if used and used + int(size) > quota:
                continue
            keep.append(timestamp)
            used += int(size)
    result = work.loc[work.__decision_ts__.isin(keep)].drop(columns="__month__")
    if result.empty:
        raise AssertionError("query-safe sampling returned no data")
    return result.reset_index(drop=True).copy()


def _prepare_folds(
    *, base_root: Path, raw_root: Path, semantic_root: Path, score_root: Path,
    policy: pd.DataFrame, held_months: Sequence[pd.Timestamp], head: str,
) -> tuple[Fold, ...]:
    earliest = pd.Timestamp("2025-11-01", tz="UTC")
    folds: list[Fold] = []
    for month in held_months:
        reserve = month - pd.Timedelta(days=RESERVE_DAYS)
        start = reserve - pd.DateOffset(months=TRAIN_MONTHS)
        if start < earliest:
            raise AssertionError(f"{month:%Y-%m}: new B/E/T source cannot support six full months from {start:%Y-%m-%d}")
        end = _month_end(month)
        base = _read_base(base_root, start, end)
        semantic = _read_semantics(semantic_root, start, reserve)
        full = base.merge(semantic, on=IDENTITY, how="left", validate="one_to_one")
        full["routed"] = _top30(full).to_numpy(bool)
        # The target contract consumes this canonical route field to construct
        # score geometry.  _read_base intentionally retains only the compact
        # base coordinates, so preserve the freshly recomputed timestamp-local
        # route explicitly rather than relying on a stale source-side flag.
        full["enhanced_base_routed"] = full["routed"].to_numpy(bool)
        full = target_contract._base_geometry(full)
        train = full.loc[full.__decision_ts__.lt(reserve)].copy()
        valid = (
            train.routed
            & train.semantic_path_valid.fillna(False).astype(bool)
            & train.semantic_label_available_ts.lt(reserve)
            & np.isfinite(pd.to_numeric(train.semantic_policy_net_bps, errors="coerce"))
        )
        train = train.loc[valid].copy()
        held = full.loc[full.__decision_ts__.ge(month) & full.routed].copy()
        for counterpart in ("T6_rank_error_ordinal", "T9_exit5_ordinal"):
            held = held.merge(_read_counterpart(score_root, month, counterpart), on=IDENTITY, how="inner", validate="one_to_one")
        held_policy = held.loc[:, ["candidate_id"]].merge(policy, on="candidate_id", how="left", validate="one_to_one")
        if len(held_policy) != len(held):
            raise AssertionError(f"{month:%Y-%m}: policy join changed target-free held identities")
        if len(train) < 20_000 or len(held) < 5_000:
            raise AssertionError(f"{month:%Y-%m}: insufficient support train={len(train)} held={len(held)}")
        folds.append(Fold(month, train, held, held_policy))
    return tuple(folds)


def _raw_matrix(raw_root: Path, frame: pd.DataFrame, fields: Sequence[str]) -> np.ndarray:
    result = np.full((len(frame), len(fields)), np.nan, dtype=np.float32)
    work = frame.reset_index(drop=True)
    for token, positions in work.groupby(work.__decision_ts__.dt.strftime("%Y-%m"), sort=True).groups.items():
        source = raw_root / f"month={token}" / "causal_feature_universe.parquet"
        ids = pd.read_parquet(source, columns=["candidate_id"])
        lookup = pd.Series(np.arange(len(ids), dtype=np.int64), index=ids.candidate_id)
        index = np.asarray(list(positions), dtype=np.int64)
        source_rows = lookup.reindex(work.iloc[index].candidate_id).to_numpy()
        if pd.isna(source_rows).any():
            raise AssertionError(f"{token}: raw feature source misses selected new-base identities")
        for begin in range(0, len(fields), 48):
            end = min(len(fields), begin + 48)
            data = pd.read_parquet(source, columns=list(fields[begin:end])).iloc[source_rows.astype(np.int64)]
            result[np.ix_(index, np.arange(begin, end))] = data.apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    return result


def _core_matrix(frame: pd.DataFrame) -> np.ndarray:
    missing = sorted(set(CORE) - set(frame.columns))
    if missing:
        raise KeyError(f"new-base core geometry missing {missing}")
    return frame.loc[:, list(CORE)].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)


def _impute(train: np.ndarray, held: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    medians = np.nanmedian(train, axis=0).astype(np.float32)
    medians[~np.isfinite(medians)] = 0.0
    output: list[np.ndarray] = []
    for values in (train.copy(), held.copy()):
        missing = ~np.isfinite(values)
        if missing.any():
            values[missing] = np.broadcast_to(medians, values.shape)[missing]
        output.append(values)
    return output[0], output[1]


def _rank(frame: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    work = frame.loc[:, ["candidate_id", "__decision_ts__"]].copy()
    work["value"] = np.asarray(values, dtype=float)
    work["row"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(["__decision_ts__", "value", "candidate_id"], ascending=[True, False, True], kind="stable")
    ordinal = work.groupby("__decision_ts__", sort=False).cumcount().to_numpy(float) + 1.0
    count = work.groupby("__decision_ts__", sort=False).candidate_id.transform("size").to_numpy(float)
    result = np.empty(len(frame), dtype=np.float32)
    result[work.row.to_numpy(np.int64)] = 1.0 - (ordinal - .5) / count
    return result


def _split(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ordered = frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable")
    timestamps = ordered.__decision_ts__.drop_duplicates().to_numpy()
    cut = max(1, int(math.floor(.80 * len(timestamps))))
    fit_ts = set(timestamps[:cut])
    return frame.__decision_ts__.isin(fit_ts).to_numpy(), frame.__decision_ts__.isin(set(timestamps[cut:])).to_numpy()


def _model(params: dict[str, Any], seed: int, n_jobs: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression_l2", n_estimators=int(params.get("n_estimators", 600)),
        learning_rate=float(params.get("learning_rate", .035)), max_depth=int(params.get("max_depth", 5)),
        num_leaves=int(params.get("num_leaves", 31)), min_child_samples=int(params.get("min_child_samples", 300)),
        min_split_gain=float(params.get("min_split_gain", .001)), subsample=float(params.get("subsample", .82)),
        colsample_bytree=float(params.get("feature_fraction", .82)), reg_alpha=float(params.get("lambda_l1", .02)),
        reg_lambda=float(params.get("lambda_l2", 2.0)), random_state=seed, n_jobs=n_jobs, verbosity=-1,
    )


def _fit_predict(
    *, train: pd.DataFrame, held: pd.DataFrame, raw_root: Path, fields: Sequence[str],
    head: str, params: dict[str, Any], seed: int, n_jobs: int,
) -> tuple[np.ndarray, LGBMRegressor, np.ndarray, np.ndarray]:
    target_values, _grade, _objective, _mode = target_contract._anchor_and_targets(train, "T6_rank_error_ordinal" if head == "T6" else "T9_exit5_ordinal")
    train_raw = _raw_matrix(raw_root, train, fields)
    held_raw = _raw_matrix(raw_root, held, fields)
    train_values, held_values = _impute(
        np.hstack([_core_matrix(train), train_raw]), np.hstack([_core_matrix(held), held_raw]),
    )
    fit_mask, valid_mask = _split(train)
    model = _model(params, seed, n_jobs)
    model.fit(
        train_values[fit_mask], np.asarray(target_values, dtype=np.float32)[fit_mask],
        eval_set=[(train_values[valid_mask], np.asarray(target_values, dtype=np.float32)[valid_mask])],
        callbacks=[early_stopping(30, verbose=False)],
    )
    return np.asarray(model.predict(held_values), dtype=np.float32), model, train_values, held_values


def _score_metrics(held: pd.DataFrame, held_policy: pd.DataFrame, raw_score: np.ndarray, head: str) -> tuple[dict[str, float], pd.DataFrame]:
    # This join occurs only after ``raw_score`` has been produced from the
    # target-free held feature frame.  Invalid labels are diagnostic-excluded.
    measured = held.loc[:, ["candidate_id", "__decision_ts__", "base_rank_ts", "T6_rank_error_ordinal", "T9_exit5_ordinal"]].copy()
    measured["candidate_score"] = _rank(held, raw_score)
    if head == "T6":
        measured["combined_score"] = .75 * measured.base_rank_ts + .20 * measured.candidate_score + .05 * measured.T9_exit5_ordinal
    else:
        measured["combined_score"] = .75 * measured.base_rank_ts + .20 * measured.T6_rank_error_ordinal + .05 * measured.candidate_score
    measured = measured.merge(held_policy.loc[:, ["candidate_id", "policy_path_valid", "policy_net_bps"]], on="candidate_id", how="left", validate="one_to_one")
    measured = measured.loc[measured.policy_path_valid & np.isfinite(measured.policy_net_bps)].copy()
    measured = measured.sort_values(["__decision_ts__", "combined_score", "candidate_id"], ascending=[True, False, True], kind="stable")
    measured["rank"] = measured.groupby("__decision_ts__", sort=False).cumcount() + 1
    rows: dict[str, float] = {}
    for k in (1, 2, 3, 5, 10):
        data = measured.loc[measured["rank"].le(k)]
        rows[f"top{k}"] = float(data.groupby("__decision_ts__", sort=False).policy_net_bps.mean().mean())
    top2 = measured.loc[measured["rank"].le(2)].groupby("__decision_ts__", sort=False).policy_net_bps.mean().rename("top2").reset_index()
    top2["week"] = top2.__decision_ts__.dt.to_period("W-SUN").astype(str)
    top2["month"] = top2.__decision_ts__.dt.strftime("%Y-%m")
    week = top2.groupby("week", sort=False).top2.mean()
    month = top2.groupby("month", sort=False).top2.mean()
    for prefix, values in (("week", week), ("month", month)):
        for label, quantile in (("q01", .01), ("q05", .05), ("q25", .25), ("q50", .50)):
            rows[f"{prefix}_{label}_top2"] = float(values.quantile(quantile))
    rows["timestamps"] = float(top2.shape[0])
    rows["trades"] = float(measured.shape[0])
    rows["weighted_top123"] = .70 * rows["top3"] + .20 * rows["top2"] + .10 * rows["top1"]
    rows["selection_objective"] = .80 * rows["weighted_top123"] + .10 * rows["week_q25_top2"] + .10 * rows["month_q25_top2"]
    return rows, measured


def _univariate(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    # Rank correlations use a capped deterministic matrix and are intentionally
    # a rescue term only; they cannot outweigh gain/tail-SHAP/stability evidence.
    y = rankdata(np.asarray(target, dtype=float))
    y = (y - y.mean()) / max(y.std(), 1e-8)
    output = np.zeros(values.shape[1], dtype=np.float32)
    for index in range(values.shape[1]):
        column = values[:, index]
        finite = np.isfinite(column)
        if finite.sum() < 100:
            continue
        x = rankdata(column[finite])
        x = (x - x.mean()) / max(x.std(), 1e-8)
        output[index] = abs(float(np.mean(x * y[finite])))
    return output


def _within_timestamp_permutation(frame: pd.DataFrame, values: np.ndarray, seed: int) -> np.ndarray:
    output = values.copy()
    rng = np.random.default_rng(seed)
    for _, indexes in frame.groupby("__decision_ts__", sort=False).groups.items():
        index = np.asarray(list(indexes), dtype=np.int64)
        if len(index) > 1:
            output[index] = values[index[rng.permutation(len(index))]]
    return output


def _screen(
    *, folds: Sequence[Fold], raw_root: Path, hygiene: pd.DataFrame, head: str,
    n_jobs: int, out: Path,
) -> tuple[list[str], pd.DataFrame]:
    fields = hygiene.loc[hygiene.hygiene_pass, "feature"].astype(str).tolist()
    records: list[dict[str, object]] = []
    params = {"n_estimators": 350, "learning_rate": .045, "max_depth": 4, "num_leaves": 31, "min_child_samples": 350, "feature_fraction": .80, "subsample": .80, "lambda_l2": 2.0}
    for fold_index, fold in enumerate(folds):
        train = _sample_queries(fold.train, MAX_TRAIN_ROWS, salt=SEED + fold_index)
        held = _sample_queries(fold.held, MAX_HELD_ROWS, salt=SEED + 1_000 + fold_index)
        raw_train = _raw_matrix(raw_root, train, fields)
        raw_held = _raw_matrix(raw_root, held, fields)
        target_values, _grade, _objective, _mode = target_contract._anchor_and_targets(train, "T6_rank_error_ordinal" if head == "T6" else "T9_exit5_ordinal")
        x_train, x_held = _impute(np.hstack([_core_matrix(train), raw_train]), np.hstack([_core_matrix(held), raw_held]))
        fit_mask, valid_mask = _split(train)
        model = _model(params, SEED + fold_index, n_jobs)
        model.fit(x_train[fit_mask], target_values[fit_mask], eval_set=[(x_train[valid_mask], target_values[valid_mask])], callbacks=[early_stopping(30, verbose=False)])
        gain = model.feature_importances_[len(CORE):].astype(float)
        raw_score = np.asarray(model.predict(x_held), dtype=float)
        top = _score_metrics(held, fold.held_policy, raw_score, head)[1]
        tail_ids = set(top.loc[top["rank"].le(3), "candidate_id"])
        tail_index = np.flatnonzero(held.candidate_id.isin(tail_ids).to_numpy())
        if len(tail_index):
            contribution = model.predict(x_held[tail_index], pred_contrib=True)
            tail_shap = np.mean(np.abs(contribution[:, len(CORE):-1]), axis=0)
        else:
            tail_shap = np.zeros(len(fields), dtype=float)
        uni = _univariate(raw_train, target_values)
        stability = np.zeros(len(fields), dtype=float)
        sub_count = max(8, int(.65 * len(fields)))
        for replica in range(4):
            rng = np.random.default_rng(SEED + 1000 * fold_index + replica)
            subset = np.sort(rng.choice(len(fields), size=sub_count, replace=False))
            sub = _model(params, SEED + 30_000 + 100 * fold_index + replica, n_jobs)
            sub.fit(x_train[fit_mask][:, np.r_[np.arange(len(CORE)), len(CORE) + subset]], target_values[fit_mask], eval_set=[(x_train[valid_mask][:, np.r_[np.arange(len(CORE)), len(CORE) + subset]], target_values[valid_mask])], callbacks=[early_stopping(30, verbose=False)])
            local = sub.feature_importances_[len(CORE):]
            winner = subset[np.argsort(local)[-min(160, len(subset)):]]
            stability[winner] += 1.0
        for index, field in enumerate(fields):
            records.append({"fold": f"{fold.held_month:%Y-%m}", "feature": field, "family": _family(field), "gain": float(gain[index]), "tail_shap": float(tail_shap[index]), "univariate": float(uni[index]), "stability": float(stability[index] / 4.0)})
        _progress(out, stage="screen_fold_complete", head=head, held_month=f"{fold.held_month:%Y-%m}", fields=len(fields))
        del raw_train, raw_held, x_train, x_held, model
        gc.collect()
    detail = pd.DataFrame(records)
    aggregate = detail.groupby(["feature", "family"], sort=False).agg({"gain": "median", "tail_shap": "median", "univariate": "median", "stability": "mean"}).reset_index()
    for field in ("gain", "tail_shap", "univariate", "stability"):
        aggregate[f"{field}_rank"] = aggregate[field].rank(pct=True, method="average")
    aggregate["screen_score"] = .35 * aggregate.gain_rank + .30 * aggregate.tail_shap_rank + .20 * aggregate.stability_rank + .15 * aggregate.univariate_rank
    aggregate = aggregate.sort_values(["screen_score", "feature"], ascending=[False, True], kind="stable")
    selected: list[str] = aggregate.head(SCREEN_KEEP).feature.astype(str).tolist()
    # Univariate rescue: retain the strongest member of a causal family when a
    # high-gain/SHAP screen otherwise loses that family entirely.
    for family, group in aggregate.groupby("family", sort=True):
        field = str(group.sort_values(["univariate", "feature"], ascending=[False, True], kind="stable").iloc[0].feature)
        if field not in selected:
            selected.append(field)
    selected = selected[:SCREEN_KEEP]
    aggregate["selected_screen120"] = aggregate.feature.isin(selected)
    return selected, aggregate


def _mda(
    *, folds: Sequence[Fold], raw_root: Path, fields: Sequence[str], head: str,
    n_jobs: int, out: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    observations: list[dict[str, object]] = []
    family_observations: list[dict[str, object]] = []
    params = {"n_estimators": 450, "learning_rate": .035, "max_depth": 5, "num_leaves": 31, "min_child_samples": 300, "feature_fraction": .82, "subsample": .82, "lambda_l1": .02, "lambda_l2": 2.0}
    by_family: dict[str, list[int]] = {}
    for index, field in enumerate(fields):
        by_family.setdefault(_family(field), []).append(index)
    for fold_index, fold in enumerate(folds):
        train = _sample_queries(fold.train, MAX_TRAIN_ROWS, salt=SEED + 50_000 + fold_index)
        held = _sample_queries(fold.held, MAX_HELD_ROWS, salt=SEED + 51_000 + fold_index)
        raw_train = _raw_matrix(raw_root, train, fields)
        raw_held = _raw_matrix(raw_root, held, fields)
        target_values, _grade, _objective, _mode = target_contract._anchor_and_targets(train, "T6_rank_error_ordinal" if head == "T6" else "T9_exit5_ordinal")
        x_train, x_held = _impute(np.hstack([_core_matrix(train), raw_train]), np.hstack([_core_matrix(held), raw_held]))
        fit_mask, valid_mask = _split(train)
        for replica in (0, 1):
            model = _model(params, SEED + 60_000 + 100 * fold_index + replica, n_jobs)
            model.fit(x_train[fit_mask], target_values[fit_mask], eval_set=[(x_train[valid_mask], target_values[valid_mask])], callbacks=[early_stopping(30, verbose=False)])
            baseline_raw = np.asarray(model.predict(x_held), dtype=float)
            baseline, _ = _score_metrics(held, fold.held_policy, baseline_raw, head)
            source = _within_timestamp_permutation(held, x_held[:, len(CORE):], SEED + 70_000 + 1000 * fold_index + replica)
            for index, field in enumerate(fields):
                altered = x_held.copy()
                altered[:, len(CORE) + index] = source[:, index]
                metric, _ = _score_metrics(held, fold.held_policy, np.asarray(model.predict(altered), dtype=float), head)
                observations.append({"feature": field, "family": _family(field), "fold": f"{fold.held_month:%Y-%m}", "replica": replica, "mda_objective": baseline["selection_objective"] - metric["selection_objective"], "mda_top3": baseline["top3"] - metric["top3"], "mda_top2": baseline["top2"] - metric["top2"], "mda_top1": baseline["top1"] - metric["top1"], "mda_week_q25": baseline["week_q25_top2"] - metric["week_q25_top2"], "mda_month_q25": baseline["month_q25_top2"] - metric["month_q25_top2"]})
            for family, indexes in by_family.items():
                altered = x_held.copy()
                altered[:, len(CORE) + np.asarray(indexes, dtype=np.int64)] = source[:, indexes]
                metric, _ = _score_metrics(held, fold.held_policy, np.asarray(model.predict(altered), dtype=float), head)
                family_observations.append({"family": family, "fields": len(indexes), "fold": f"{fold.held_month:%Y-%m}", "replica": replica, "mda_objective": baseline["selection_objective"] - metric["selection_objective"], "mda_top3": baseline["top3"] - metric["top3"], "mda_top2": baseline["top2"] - metric["top2"], "mda_top1": baseline["top1"] - metric["top1"], "mda_week_q25": baseline["week_q25_top2"] - metric["week_q25_top2"], "mda_month_q25": baseline["month_q25_top2"] - metric["month_q25_top2"]})
            _progress(out, stage="mda_fold_replica_complete", head=head, held_month=f"{fold.held_month:%Y-%m}", replica=replica)
            del model
        del raw_train, raw_held, x_train, x_held
        gc.collect()
    detail = pd.DataFrame(observations)
    family_detail = pd.DataFrame(family_observations)
    def summary(data: pd.DataFrame, column: str) -> pd.DataFrame:
        group = data.groupby(column, sort=False)
        result = group.agg({"mda_objective": ["median", "min"], "mda_top3": "median", "mda_top2": "median", "mda_top1": "median", "mda_week_q25": "median", "mda_month_q25": "median"})
        result.columns = ["_".join(map(str, name)).strip("_") for name in result.columns]
        return result.reset_index()
    result = summary(detail, "feature").merge(detail.loc[:, ["feature", "family"]].drop_duplicates(), on="feature", how="left")
    result["mda_selection_score"] = (.70 * result.mda_top3_median + .20 * result.mda_top2_median + .10 * result.mda_top1_median + .10 * result.mda_week_q25_median + .10 * result.mda_month_q25_median)
    result = result.sort_values(["mda_selection_score", "feature"], ascending=[False, True], kind="stable")
    return result, summary(family_detail, "family")


def _evaluate_fields(
    *, folds: Sequence[Fold], raw_root: Path, fields: Sequence[str], head: str,
    params: dict[str, Any], n_jobs: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold_index, fold in enumerate(folds):
        raw, _model_value, _train, _held = _fit_predict(train=_sample_queries(fold.train, MAX_TRAIN_ROWS, salt=SEED + 80_000 + fold_index), held=_sample_queries(fold.held, MAX_HELD_ROWS, salt=SEED + 81_000 + fold_index), raw_root=raw_root, fields=fields, head=head, params=params, seed=SEED + 82_000 + fold_index, n_jobs=n_jobs)
        train = _sample_queries(fold.train, MAX_TRAIN_ROWS, salt=SEED + 80_000 + fold_index)
        held = _sample_queries(fold.held, MAX_HELD_ROWS, salt=SEED + 81_000 + fold_index)
        # Recreate deterministic held policy order after the score producer;
        # it remains separate from target-free score generation.
        policy = held.loc[:, ["candidate_id"]].merge(fold.held_policy, on="candidate_id", how="left", validate="one_to_one")
        metric, _ = _score_metrics(held, policy, raw, head)
        rows.append({"held_month": f"{fold.held_month:%Y-%m}", "feature_count": len(fields), **metric})
        del raw, _model_value, _train, _held, train, held
        gc.collect()
    return pd.DataFrame(rows)


def _summarise_evaluation(frame: pd.DataFrame) -> dict[str, float]:
    output: dict[str, float] = {"folds": float(len(frame))}
    for field in ("top1", "top2", "top3", "top5", "top10", "weighted_top123", "selection_objective", "week_q25_top2", "month_q25_top2"):
        values = pd.to_numeric(frame[field], errors="coerce")
        output[f"mean_{field}"] = float(values.mean())
        output[f"q25_{field}"] = float(values.quantile(.25))
        output[f"worst_{field}"] = float(values.min())
    return output


def _select_ladder(summary: pd.DataFrame) -> int:
    best = summary.loc[summary.mean_selection_objective.idxmax()]
    qualify = summary.loc[
        summary.mean_selection_objective.ge(.99 * float(best.mean_selection_objective))
        & summary.q25_week_q25_top2.ge(float(best.q25_week_q25_top2) - 1e-9)
        & summary.q25_month_q25_top2.ge(float(best.q25_month_q25_top2) - 1e-9)
    ].sort_values("feature_count")
    return int(qualify.iloc[0].feature_count) if len(qualify) else int(best.feature_count)


def _hpo(
    *, folds: Sequence[Fold], raw_root: Path, fields: Sequence[str], head: str,
    n_jobs: int, trials: int, out: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    def objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {
            "n_estimators": 1800, "learning_rate": trial.suggest_float("learning_rate", .015, .08, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 6), "num_leaves": trial.suggest_int("num_leaves", 15, 61),
            "min_child_samples": trial.suggest_int("min_child_samples", 180, 900),
            "min_split_gain": trial.suggest_float("min_split_gain", 1e-4, .02, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", .70, .90), "subsample": trial.suggest_float("subsample", .70, .90),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 5.0, log=True), "lambda_l2": trial.suggest_float("lambda_l2", .1, 30.0, log=True),
        }
        rows: list[dict[str, object]] = []
        for index, fold in enumerate(folds):
            train = _sample_queries(fold.train, MAX_TRAIN_ROWS, salt=SEED + 90_000 + index)
            held = _sample_queries(fold.held, MAX_HELD_ROWS, salt=SEED + 91_000 + index)
            raw, _model_value, _train, _held = _fit_predict(train=train, held=held, raw_root=raw_root, fields=fields, head=head, params=params, seed=SEED + 92_000 + 100 * trial.number + index, n_jobs=n_jobs)
            policy = held.loc[:, ["candidate_id"]].merge(fold.held_policy, on="candidate_id", how="left", validate="one_to_one")
            metric, _ = _score_metrics(held, policy, raw, head)
            rows.append(metric)
            partial = _summarise_evaluation(pd.DataFrame(rows))["mean_selection_objective"]
            trial.report(partial, step=index)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return _summarise_evaluation(pd.DataFrame(rows))["mean_selection_objective"]
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=1))
    study.optimize(objective, n_trials=trials, n_jobs=1, show_progress_bar=False)
    rows: list[dict[str, object]] = []
    for trial in study.trials:
        rows.append({"number": trial.number, "state": str(trial.state), "value": trial.value, **trial.params})
    return dict(study.best_trial.params), pd.DataFrame(rows)


def _final_scores(
    *, folds: Sequence[Fold], raw_root: Path, fields: Sequence[str], head: str,
    params: dict[str, Any], n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_parts: list[pd.DataFrame] = []
    metric_parts: list[pd.DataFrame] = []
    for index, fold in enumerate(folds):
        raw, _model_value, _train, _held = _fit_predict(train=fold.train, held=fold.held, raw_root=raw_root, fields=fields, head=head, params=params, seed=SEED + 120_000 + index, n_jobs=n_jobs)
        scores = fold.held.loc[:, list(IDENTITY)].copy()
        scores["head_score"] = raw.astype(np.float32)
        scores["head_rank"] = _rank(fold.held, raw)
        scores["held_month"] = f"{fold.held_month:%Y-%m}"
        policy = fold.held.loc[:, ["candidate_id"]].merge(fold.held_policy, on="candidate_id", how="left", validate="one_to_one")
        metric, ordered = _score_metrics(fold.held, policy, raw, head)
        score_parts.append(scores)
        metric_parts.append(pd.DataFrame([{"held_month": f"{fold.held_month:%Y-%m}", **metric}]))
        del raw, _model_value, _train, _held
    return pd.concat(score_parts, ignore_index=True), pd.concat(metric_parts, ignore_index=True)


def run_head(
    *, head: str, base_root: Path, raw_root: Path, semantic_root: Path, baseline_score_root: Path,
    policy: pd.DataFrame, held_months: Sequence[pd.Timestamp], out: Path, n_jobs: int, hpo_trials: int,
) -> None:
    folds = _prepare_folds(base_root=base_root, raw_root=raw_root, semantic_root=semantic_root, score_root=baseline_score_root, policy=policy, held_months=held_months, head=head)
    months_for_hygiene = tuple(pd.Timestamp(f"{month}-01", tz="UTC") for month in pd.date_range("2025-11-01", "2026-07-01", freq="MS").strftime("%Y-%m"))
    fields = _numeric_fields(raw_root, months_for_hygiene[0])
    hygiene = _hygiene(raw_root, fields, months_for_hygiene)
    hygiene.to_parquet(out / f"{head.lower()}_hygiene.parquet", index=False, compression="zstd")
    screen, screen_metrics = _screen(folds=folds, raw_root=raw_root, hygiene=hygiene, head=head, n_jobs=n_jobs, out=out)
    screen_metrics.to_parquet(out / f"{head.lower()}_screen_metrics.parquet", index=False, compression="zstd")
    _exclusive_json(out / f"{head.lower()}_screen120.json", {"head": head, "process": "gain + tail SHAP + univariate rescue + randomized stability; CMI intentionally omitted", "features": screen, "feature_count": len(screen)})
    mda, family_mda = _mda(folds=folds, raw_root=raw_root, fields=screen, head=head, n_jobs=n_jobs, out=out)
    mda.to_parquet(out / f"{head.lower()}_economic_boundary_mda.parquet", index=False, compression="zstd")
    family_mda.to_parquet(out / f"{head.lower()}_family_mda.parquet", index=False, compression="zstd")
    order = mda.feature.astype(str).tolist()
    fixed = {"n_estimators": 600, "learning_rate": .035, "max_depth": 5, "num_leaves": 31, "min_child_samples": 300, "feature_fraction": .82, "subsample": .82, "lambda_l1": .02, "lambda_l2": 2.0}
    ladder_rows: list[dict[str, object]] = []
    feature_sets: dict[int, list[str]] = {}
    for size in SIZES:
        chosen = order[:size]
        feature_sets[size] = chosen
        metrics = _evaluate_fields(folds=folds, raw_root=raw_root, fields=chosen, head=head, params=fixed, n_jobs=n_jobs)
        summary = _summarise_evaluation(metrics)
        ladder_rows.append({"feature_count": size, **summary})
        metrics.to_parquet(out / f"{head.lower()}_f{size}_fold_metrics.parquet", index=False, compression="zstd")
    ladder = pd.DataFrame(ladder_rows).sort_values("feature_count", ascending=False)
    ladder.to_parquet(out / f"{head.lower()}_subset_ladder.parquet", index=False, compression="zstd")
    selected_size = _select_ladder(ladder)
    selected = list(feature_sets[selected_size])
    # One deterministic semantic-family add-back pass: at most four high-MDA
    # fields from each excluded family.  A family is retained only if it raises
    # the composite objective without weakening either q25 stability measure.
    base_summary = ladder.loc[ladder.feature_count.eq(selected_size)].iloc[0]
    addback_rows: list[dict[str, object]] = []
    for family, group in mda.loc[~mda.feature.isin(selected)].groupby("family", sort=True):
        additions = group.head(4).feature.astype(str).tolist()
        candidate = list(dict.fromkeys([*selected, *additions]))
        metrics = _evaluate_fields(folds=folds, raw_root=raw_root, fields=candidate, head=head, params=fixed, n_jobs=n_jobs)
        summary = _summarise_evaluation(metrics)
        advance = (
            summary["mean_selection_objective"] > float(base_summary.mean_selection_objective)
            and summary["q25_week_q25_top2"] >= float(base_summary.q25_week_q25_top2)
            and summary["q25_month_q25_top2"] >= float(base_summary.q25_month_q25_top2)
        )
        addback_rows.append({"family": family, "additions": additions, "advance": advance, **summary})
        if advance:
            selected = candidate
            base_summary = pd.Series(summary)
    pd.DataFrame(addback_rows).to_parquet(out / f"{head.lower()}_family_addback.parquet", index=False, compression="zstd")
    best_params, trials = _hpo(folds=folds, raw_root=raw_root, fields=selected, head=head, n_jobs=n_jobs, trials=hpo_trials, out=out)
    trials.to_parquet(out / f"{head.lower()}_hpo_trials.parquet", index=False, compression="zstd")
    params = {"n_estimators": 1800, **best_params}
    scores, metrics = _final_scores(folds=folds, raw_root=raw_root, fields=selected, head=head, params=params, n_jobs=n_jobs)
    scores.to_parquet(out / f"{head.lower()}_target_free_oof_scores.parquet", index=False, compression="zstd")
    metrics.to_parquet(out / f"{head.lower()}_final_fold_metrics.parquet", index=False, compression="zstd")
    _exclusive_json(out / f"{head.lower()}_winner.json", {
        "head": head, "target": "rank_error_ordinal" if head == "T6" else "exit5_ordinal",
        "features": selected, "feature_count": len(selected), "fixed_core_features": list(CORE),
        "best_params": params, "hpo_primary_metric": "0.70 timestamp Top3 + 0.20 Top2 + 0.10 Top1; stability is q25 weekly/monthly Top2", "selected_size_before_addback": selected_size,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--raw-feature-root", type=Path, required=True)
    parser.add_argument("--semantic-root", type=Path, required=True)
    parser.add_argument("--baseline-score-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--held-months", default="2026-06,2026-07")
    parser.add_argument("--heads", default="T6,T9")
    parser.add_argument("--hpo-trials", type=int, default=24)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    heads = tuple(token.strip() for token in args.heads.split(",") if token.strip())
    if set(heads) - {"T6", "T9"}:
        raise ValueError("--heads supports only T6,T9")
    held_months = _parse_months(args.held_months)
    if len(held_months) < 2:
        raise ValueError("at least two held months are required for portability metrics")
    args.out.mkdir(parents=True)
    _exclusive_json(args.out / "run_manifest.json", {
        "schema": SCHEMA, "scope": "offline causal T6/T9 selection/HPO only; no live or downstream mutation",
        "base_root": str(args.base_root), "raw_feature_root": str(args.raw_feature_root), "semantic_root": str(args.semantic_root), "baseline_score_root": str(args.baseline_score_root), "policy_path": str(args.policy_path),
        "source_sha256": {"base_root": _sha(args.base_root), "raw_feature_root": _sha(args.raw_feature_root), "semantic_root": _sha(args.semantic_root), "baseline_score_root": _sha(args.baseline_score_root), "policy_path": _sha(args.policy_path)},
        "held_months": [f"{month:%Y-%m}" for month in held_months], "heads": list(heads), "train_contract": {"months": TRAIN_MONTHS, "reserve_days": RESERVE_DAYS, "resolved_only": True, "route": "new-base timestamp-local top30"},
        "selection_process": "hygiene, gain/tail-SHAP, univariate rescue, randomized stability, Screen120, OOF economic/boundary MDA, family MDA, subset ladder, family addback, HPO",
        "primary_hpo_metric": "0.70 top3 + 0.20 top2 + 0.10 top1 timestamp-local net EV, with q25 weekly/monthly top2 stability",
        "cmI": "intentionally omitted; targets are materially distinct and selection is driven by target-specific OOF incremental blend economics",
    })
    policy = _read_policy(args.policy_path)
    for head in heads:
        run_head(head=head, base_root=args.base_root, raw_root=args.raw_feature_root, semantic_root=args.semantic_root, baseline_score_root=args.baseline_score_root, policy=policy, held_months=held_months, out=args.out, n_jobs=args.n_jobs, hpo_trials=args.hpo_trials)
    _exclusive_json(args.out / "correctness_report.json", {
        "schema": SCHEMA, "target_free_held_before_policy_metrics": True, "policy_is_metric_only": True,
        "route": "recomputed deterministic new-base top30", "fit": "six full months before 28-day reserve with resolved semantic labels only", "heads": list(heads),
    })


if __name__ == "__main__":
    main()
