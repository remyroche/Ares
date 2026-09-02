#!/usr/bin/env python3
"""Screen E and T from the frozen-router, full causal feature universe.

This is stages 0--10 of the routed base-layer specification.  It intentionally
does not touch B0 or any live artifact:

    strict-OOF router receipt -> exact timestamp-local top-50% route
      -> direct E/T targets on prior resolved routed rows
      -> full-universe cheap OOF screen
      -> independent E_SCREEN120 / T_SCREEN120 contracts.

The follow-up MDA/subset/HPO producer consumes only these immutable screen
receipts.  All target and policy fields are joined after candidate features;
they are never passed to a model matrix.
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
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMRegressor was fitted with feature names")


ROOT = Path(__file__).resolve().parents[1]
SEED = 1729
IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
ROUTER_FIELD = "router_primary_rank"
HEADS = {
    "B0": {"target": "policy_ordinal_grade", "direction": 1.0, "label_family": "base"},
    "E": {"target": "supportive_path_efficiency_h12", "direction": 1.0, "label_family": "supportive"},
    "T": {"target": "supportive_time_to_meaningful_mfe_h12", "direction": -1.0, "label_family": "supportive"},
}


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _month_end(month: pd.Timestamp) -> pd.Timestamp:
    return month + pd.offsets.MonthBegin(1)


def _month_range(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Timestamp, ...]:
    first = _utc(start).normalize().replace(day=1)
    final = (_utc(end) - pd.Timedelta(nanoseconds=1)).normalize().replace(day=1)
    return tuple(pd.date_range(first, final, freq="MS", tz="UTC"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_dir():
        paths = sorted(path.rglob("*.parquet"))
    else:
        paths = [path]
    for item in paths:
        digest.update(str(item).encode())
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _progress(out: Path, **payload: object) -> None:
    with (out / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _numeric_fields(feature_root: Path, sample_month: str) -> list[str]:
    path = feature_root / f"month={sample_month}" / "causal_feature_universe.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    # Reading a two-row pandas head from a columnar file still materialises all
    # 1,400+ feature columns.  Consult the footer instead: this is equivalent
    # for feature-type hygiene and avoids a multi-gigabyte startup spike.
    schema = pq.ParquetFile(path).schema_arrow
    banned = set(IDENTITY) | {"__ts__", "__symbol__"}
    fields = [
        field.name for field in schema
        if field.name not in banned and pd.api.types.is_numeric_dtype(field.type.to_pandas_dtype())
    ]
    if len(fields) < 700:
        raise AssertionError(f"expected a broad causal universe, found only {len(fields)} numeric fields")
    return fields


def _coverage(feature_root: Path, fields: Sequence[str], months: Sequence[pd.Timestamp]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for month in months:
        path = feature_root / f"month={month:%Y-%m}" / "feature_coverage.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        part = pd.read_parquet(path, columns=["feature", "rows", "finite_rows", "finite_fraction", "n_unique"])
        part = part.loc[part.feature.isin(fields)].copy()
        part["month"] = f"{month:%Y-%m}"
        pieces.append(part)
    work = pd.concat(pieces, ignore_index=True)
    summary = work.groupby("feature", sort=False).agg(
        min_finite_fraction=("finite_fraction", "min"),
        mean_finite_fraction=("finite_fraction", "mean"),
        min_unique=("n_unique", "min"),
        coverage_months=("month", "nunique"),
    ).reset_index()
    summary["coverage_pass"] = (
        summary.min_finite_fraction.ge(.90)
        & summary.min_unique.ge(3)
        & summary.coverage_months.eq(len(months))
    )
    return summary.sort_values(["coverage_pass", "mean_finite_fraction", "feature"], ascending=[False, False, True], kind="stable")


def _feature_family(field: str) -> str:
    name = field.lower()
    if any(token in name for token in ("fund", "carry")):
        return "funding"
    if any(token in name for token in ("oi", "open_interest", "leverage", "liquidat")):
        return "open_interest_flow"
    if any(token in name for token in ("spread", "depth", "amihud", "liquid", "impact", "quote", "ob_")):
        return "liquidity_microstructure"
    if any(token in name for token in ("rv", "vol", "atr", "cvar", "semivariance", "range")):
        return "volatility_tails"
    if any(token in name for token in ("ret", "mom", "accel", "strength", "trend", "ema", "adx", "breakout")):
        return "returns_trend"
    if any(token in name for token in ("corr", "spectral", "eig", "beta", "coher", "pc1")):
        return "correlation_spectral"
    if any(token in name for token in ("breadth", "dispersion", "pct_assets", "xasset", "universe", "mkt_")):
        return "cross_asset_market"
    if any(token in name for token in ("dist", "support", "resistance", "vwap", "donch", "high", "low", "location", "loc_")):
        return "structure_location"
    if any(token in name for token in ("hour", "session", "weekend", "dow", "season")):
        return "calendar_session"
    if any(token in name for token in ("state", "regime", "transition", "entropy", "chop", "hurst")):
        return "regime_transition"
    if any(token in name for token in ("volume", "turnover", "trade_size")):
        return "volume_turnover"
    return "other_structural"


def _read_router(router_root: Path, month: str) -> pd.DataFrame:
    path = router_root / "target_free_scores" / f"month={month}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path, columns=[*IDENTITY, ROUTER_FIELD])
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    if frame.candidate_id.duplicated().any() or not np.isfinite(pd.to_numeric(frame[ROUTER_FIELD], errors="coerce")).all():
        raise AssertionError(f"{path}: invalid router identity or rank")
    return frame


def _route(frame: pd.DataFrame, fraction: float) -> pd.Series:
    work = frame.loc[:, ["candidate_id", "__decision_ts__", ROUTER_FIELD]].copy()
    work["__row__"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(["__decision_ts__", ROUTER_FIELD, "candidate_id"], ascending=[True, False, True], kind="stable")
    work["__ord__"] = work.groupby("__decision_ts__", sort=False).cumcount() + 1
    size = work.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy(float)
    selected = work["__ord__"].to_numpy() <= np.ceil(size * fraction)
    out = pd.Series(False, index=np.arange(len(frame)))
    out.iloc[work["__row__"].to_numpy(np.int64)] = selected
    return out


def _read_features(feature_root: Path, months: Sequence[pd.Timestamp], fields: Sequence[str]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for month in months:
        path = feature_root / f"month={month:%Y-%m}" / "causal_feature_universe.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        part = pd.read_parquet(path, columns=[*IDENTITY, *fields])
        part["__decision_ts__"] = pd.to_datetime(part["__decision_ts__"], utc=True, errors="raise")
        pieces.append(part)
    result = pd.concat(pieces, ignore_index=True)
    if result.candidate_id.duplicated().any():
        raise AssertionError("duplicate feature candidate IDs")
    return result


def _read_labels(labels_root: Path, months: Sequence[pd.Timestamp], base_labels_root: Path | None = None) -> pd.DataFrame:
    columns = [
        "candidate_id", "__decision_ts__", "side_name", "supportive_label_available_ts",
        "supportive_path_valid", "supportive_target_invalid", "supportive_path_efficiency_h12",
        "supportive_time_to_meaningful_mfe_h12",
    ]
    pieces: list[pd.DataFrame] = []
    for month in months:
        path = labels_root / "parts" / f"month={month:%Y-%m}" / "side=long.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        part = pd.read_parquet(path, columns=columns)
        part["__decision_ts__"] = pd.to_datetime(part["__decision_ts__"], utc=True, errors="raise")
        part["supportive_label_available_ts"] = pd.to_datetime(part["supportive_label_available_ts"], utc=True, errors="coerce")
        pieces.append(part)
    result = pd.concat(pieces, ignore_index=True)
    if result.candidate_id.duplicated().any():
        raise AssertionError("duplicate supportive label IDs")
    # B0 keeps its established policy-ordinal target.  Its labels are held in
    # a separately versioned source-repaired ledger, so join them by the exact
    # candidate identity rather than attempting to reconstruct or substitute
    # them from policy outcome values.
    if base_labels_root is not None:
        base_pieces: list[pd.DataFrame] = []
        base_columns = [*IDENTITY, "policy_ordinal_valid", "policy_ordinal_grade", "policy_label_available_ts"]
        for month in months:
            path = base_labels_root / f"month={month:%Y-%m}" / "target_labels.parquet"
            if not path.exists():
                raise FileNotFoundError(path)
            part = pd.read_parquet(path, columns=base_columns)
            part["__decision_ts__"] = pd.to_datetime(part["__decision_ts__"], utc=True, errors="raise")
            part["policy_label_available_ts"] = pd.to_datetime(part["policy_label_available_ts"], utc=True, errors="coerce")
            part = part.rename(columns={"policy_label_available_ts": "base_policy_label_available_ts"})
            base_pieces.append(part)
        base = pd.concat(base_pieces, ignore_index=True)
        if base.candidate_id.duplicated().any():
            raise AssertionError("duplicate base label IDs")
        # The supporting-path ledger carries the same availability name.  B0
        # must use the ordinal ledger's timestamp, which is the target it
        # actually trains on; retain supportive availability under its own name.
        result = result.rename(columns={"policy_label_available_ts": "supportive_policy_label_available_ts"})
        result = result.merge(base, on=list(IDENTITY), how="left", validate="one_to_one")
    else:
        result["policy_ordinal_valid"] = False
        result["policy_ordinal_grade"] = np.nan
        result["base_policy_label_available_ts"] = pd.NaT
    return result


def _read_policy(path: Path, months: Sequence[pd.Timestamp]) -> pd.DataFrame:
    columns = ["candidate_id", "policy_path_valid", "policy_net_bps", "policy_label_available_ts"]
    frame = pd.read_parquet(path, columns=columns)
    frame["policy_label_available_ts"] = pd.to_datetime(frame["policy_label_available_ts"], utc=True, errors="coerce")
    frame["policy_net_bps"] = pd.to_numeric(frame["policy_net_bps"], errors="coerce")
    frame["policy_path_valid"] = frame["policy_path_valid"].fillna(False).astype(bool)
    return frame


def _joined(
    *, feature_root: Path, router_root: Path, labels_root: Path, policy: pd.DataFrame,
    start: pd.Timestamp, end: pd.Timestamp, fields: Sequence[str], route_fraction: float,
    base_labels_root: Path | None = None,
) -> pd.DataFrame:
    months = _month_range(start, end)
    features = _read_features(feature_root, months, fields)
    router = pd.concat([_read_router(router_root, f"{month:%Y-%m}") for month in months], ignore_index=True)
    labels = _read_labels(labels_root, months, base_labels_root)
    # The source-repaired full feature universe is explicitly long-only while
    # the frozen Router receipt may legitimately retain both sides.  The
    # required invariant is therefore *complete feature -> Router coverage*,
    # not an invalid equality against unused short-side router rows.  Keep the
    # surplus count as provenance; never silently accept a missing feature row
    # or a duplicate on either identity key.
    result = features.merge(router, on=list(IDENTITY), how="inner", validate="one_to_one")
    if len(result) != len(features):
        raise AssertionError(
            f"feature/router identity coverage failure: features={len(features)} joined={len(result)}"
        )
    result["router_unused_other_side_rows"] = int(len(router) - len(result))
    result = result.merge(labels, on=list(IDENTITY), how="left", validate="one_to_one")
    result = result.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    result = result.loc[result["__decision_ts__"].ge(start) & result["__decision_ts__"].lt(end)].copy()
    # The target-free universe is authoritative.  A historical outcome join
    # can be absent (for example a later label-substrate repair); retain the
    # candidate and make the missing supervision explicit.  These rows are
    # excluded by _strict_train/_held_eval and are never encoded as failures.
    result["supportive_label_joined"] = result["supportive_path_valid"].notna()
    result["policy_label_joined"] = result["policy_path_valid"].notna()
    result["supportive_path_valid"] = result["supportive_path_valid"].fillna(False).astype(bool)
    result["supportive_target_invalid"] = result["supportive_target_invalid"].fillna(True).astype(bool)
    result["policy_path_valid"] = result["policy_path_valid"].fillna(False).astype(bool)
    result["router_selected"] = _route(result, route_fraction).to_numpy(bool)
    return result


def _selected_feature_matrix(feature_root: Path, selected: pd.DataFrame, fields: Sequence[str]) -> np.ndarray:
    """Materialise selected rows only, reading the wide parquet in small blocks.

    Candidate filtering is entirely metadata/label based and happens before a
    feature value is read.  This avoids materialising four 1,000+ field months
    merely to retain a strict, time-balanced routed subset.
    """
    selected = selected.reset_index(drop=True)
    if selected.candidate_id.duplicated().any():
        raise AssertionError("selected feature materialisation received duplicate IDs")
    result = np.empty((len(selected), len(fields)), dtype=np.float32)
    result.fill(np.nan)
    month_key = selected["__decision_ts__"].dt.strftime("%Y-%m")
    for token, row_ids in month_key.groupby(month_key, sort=True).groups.items():
        path = feature_root / f"month={token}" / "causal_feature_universe.parquet"
        source_ids = pd.read_parquet(path, columns=["candidate_id"])
        source_index = pd.Series(np.arange(len(source_ids), dtype=np.int64), index=source_ids.candidate_id)
        target_rows = np.asarray(list(row_ids), dtype=np.int64)
        source_rows = source_index.reindex(selected.iloc[target_rows].candidate_id).to_numpy()
        if pd.isna(source_rows).any():
            raise AssertionError(f"{token}: selected candidate missing from causal feature universe")
        source_rows = source_rows.astype(np.int64, copy=False)
        for begin in range(0, len(fields), 48):
            end = min(len(fields), begin + 48)
            chunk = list(fields[begin:end])
            values = pd.read_parquet(path, columns=chunk).iloc[source_rows]
            result[np.ix_(target_rows, np.arange(begin, end, dtype=np.int64))] = (
                values.apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
            )
    return result


def _impute_from_train(values: np.ndarray, train_rows: int) -> tuple[np.ndarray, np.ndarray]:
    if train_rows <= 0:
        raise AssertionError("no training rows for feature imputation")
    medians = np.nanmedian(values[:train_rows], axis=0).astype(np.float32)
    medians[~np.isfinite(medians)] = 0.0
    # Fill in blocks so replacing missing values never creates a second
    # full-width copy of the selected feature matrix.
    for begin in range(0, values.shape[1], 48):
        end = min(values.shape[1], begin + 48)
        block = values[:, begin:end]
        bad = ~np.isfinite(block)
        if bad.any():
            block[bad] = np.broadcast_to(medians[begin:end], block.shape)[bad]
    return values, medians


def _time_balanced_sample(frame: pd.DataFrame, cap: int, *, seed: int = SEED) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame.copy()
    work = frame.copy()
    work["__month__"] = work["__decision_ts__"].dt.strftime("%Y-%m")
    work["__hash__"] = pd.util.hash_pandas_object(work["candidate_id"].astype(str) + f"|{seed}", index=False).to_numpy(np.uint64)
    per_month = max(1, int(math.ceil(cap / work["__month__"].nunique())))
    result = (work.sort_values(["__month__", "__hash__", "candidate_id"], kind="stable")
              .groupby("__month__", sort=False, group_keys=False).head(per_month))
    return result.iloc[:cap].drop(columns=["__month__", "__hash__"], errors="ignore").copy()


def _matrix(frame: pd.DataFrame, fields: Sequence[str], medians: pd.Series | None = None) -> tuple[np.ndarray, pd.Series]:
    numeric = frame.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if medians is None:
        medians = numeric.median(axis=0).fillna(0.0)
    return numeric.fillna(medians).fillna(0.0).to_numpy(np.float32), medians


def _params(*, seed: int, n_jobs: int, cheap: bool = False, feature_fraction: float = .80) -> dict[str, object]:
    return {
        "objective": "huber", "alpha": .90, "n_estimators": 110 if cheap else 220,
        "learning_rate": .05 if cheap else .035, "max_depth": 3 if cheap else 4,
        "num_leaves": 15 if cheap else 31, "min_child_samples": 260 if cheap else 180,
        "subsample": .80, "subsample_freq": 1, "colsample_bytree": feature_fraction,
        "reg_lambda": 6.0, "reg_alpha": .05, "random_state": seed, "n_jobs": n_jobs,
        "deterministic": True, "force_col_wise": True, "verbosity": -1,
    }


def _strict_train(frame: pd.DataFrame, reserve_start: pd.Timestamp, target: str, cap: int) -> pd.DataFrame:
    common = (
        frame.router_selected.fillna(False).astype(bool)
        & frame.policy_path_valid.fillna(False).astype(bool)
        & frame.policy_label_available_ts.lt(reserve_start)
        & np.isfinite(pd.to_numeric(frame[target], errors="coerce"))
        & np.isfinite(pd.to_numeric(frame.policy_net_bps, errors="coerce"))
    )
    if target == "policy_ordinal_grade":
        valid = (
            common
            & frame.policy_ordinal_valid.fillna(False).astype(bool)
            & frame.base_policy_label_available_ts.lt(reserve_start)
        )
    else:
        valid = (
            common
            & frame.supportive_path_valid.fillna(False).astype(bool)
            & ~frame.supportive_target_invalid.fillna(True).astype(bool)
            & frame.supportive_label_available_ts.lt(reserve_start)
        )
    return _time_balanced_sample(frame.loc[valid].copy(), cap)


def _held_eval(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    common = (
        frame.router_selected.fillna(False).astype(bool)
        & frame.policy_path_valid.fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(frame.policy_net_bps, errors="coerce"))
    )
    if target == "policy_ordinal_grade":
        valid = common & frame.policy_ordinal_valid.fillna(False).astype(bool)
    else:
        valid = common & frame.supportive_path_valid.fillna(False).astype(bool) & ~frame.supportive_target_invalid.fillna(True).astype(bool)
    return frame.loc[valid].copy()


def _top_metric(frame: pd.DataFrame, score: str, fraction: float) -> dict[str, float]:
    work = frame.loc[:, ["__decision_ts__", "candidate_id", score, "policy_net_bps"]].copy()
    work["__score__"] = pd.to_numeric(work[score], errors="coerce").fillna(-np.inf)
    work = work.sort_values(["__decision_ts__", "__score__", "candidate_id"], ascending=[True, False, True], kind="stable")
    work["__ord__"] = work.groupby("__decision_ts__", sort=False).cumcount() + 1
    size = work.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy(float)
    take = work["__ord__"].to_numpy() <= np.ceil(size * fraction)
    selected = work.loc[take].copy()
    timestamp = selected.groupby("__decision_ts__", sort=False).policy_net_bps.mean()
    positive = selected.assign(__positive__=selected.policy_net_bps.gt(50.0)).groupby("__decision_ts__", sort=False).__positive__.mean()
    week_key = timestamp.index.isocalendar().year.astype(str) + "-" + timestamp.index.isocalendar().week.astype(str)
    month_key = timestamp.index.tz_localize(None).to_period("M")
    return {
        "ev": float(timestamp.mean()), "rows": float(len(selected)), "timestamps": float(len(timestamp)),
        "precision50": float(positive.mean()), "weekly_q10": float(timestamp.groupby(week_key).mean().quantile(.10)),
        "monthly_median": float(timestamp.groupby(month_key).mean().median()),
        "monthly_q25": float(timestamp.groupby(month_key).mean().quantile(.25)),
    }


def _metric_suite(frame: pd.DataFrame, score: str) -> dict[str, float]:
    output: dict[str, float] = {}
    summaries: dict[float, dict[str, float]] = {}
    for fraction, label in ((.01, "01"), (.02, "02"), (.05, "05"), (.10, "10")):
        result = _top_metric(frame, score, fraction)
        summaries[fraction] = result
        output[f"ts_top{label}_ev"] = result["ev"]
        output[f"ts_top{label}_precision50"] = result["precision50"]
        output[f"ts_top{label}_rows"] = result["rows"]
    p10 = summaries[.10]
    output["base_stable_p10"] = float(
        .50 * p10["ev"] + .20 * p10["monthly_median"] + .15 * p10["monthly_q25"] + .15 * p10["weekly_q10"]
    )
    output["q10_week_top10_ev"] = p10["weekly_q10"]
    output["q25_month_top10_ev"] = p10["monthly_q25"]
    return output


def _stratified_index(frame: pd.DataFrame, cap: int, *, rank: pd.Series | None = None, seed: int = SEED) -> np.ndarray:
    if len(frame) <= cap:
        return np.arange(len(frame), dtype=np.int64)
    # All callers index an in-memory held matrix, so return dense local
    # positions rather than the inherited dataframe labels from the monthly
    # source panel.
    work = frame.loc[:, ["candidate_id", "__decision_ts__"]].reset_index(drop=True).copy()
    work["__position__"] = np.arange(len(work), dtype=np.int64)
    work["__week__"] = work["__decision_ts__"].dt.strftime("%Y-%W")
    work["__hash__"] = pd.util.hash_pandas_object(work.candidate_id.astype(str) + f"|{seed}", index=False).to_numpy(np.uint64)
    if rank is not None:
        work["__rank__"] = pd.to_numeric(rank, errors="coerce").fillna(0.0).to_numpy(float)
        # Precision-region selection retains all scarce high-rank rows before
        # balanced sampling from the remaining timestamps.
        high = work.loc[work.__rank__.ge(.70)]
        rest = work.loc[~work.index.isin(high.index)]
        keep_high = high.sort_values(["__week__", "__hash__"], kind="stable").groupby("__week__", group_keys=False).head(max(1, cap // max(1, high.__week__.nunique())))
        remain = max(0, cap - len(keep_high))
        keep_rest = rest.sort_values(["__week__", "__hash__"], kind="stable").groupby("__week__", group_keys=False).head(max(1, remain // max(1, rest.__week__.nunique())))
        return np.sort(np.concatenate([keep_high.__position__.to_numpy(np.int64), keep_rest.__position__.to_numpy(np.int64)])[:cap])
    chosen = work.sort_values(["__week__", "__hash__"], kind="stable").groupby("__week__", group_keys=False).head(max(1, cap // max(1, work.__week__.nunique())))
    return np.sort(chosen.__position__.to_numpy(np.int64)[:cap])


def _univariate(frame: pd.DataFrame, values: np.ndarray, fields: Sequence[str], out: list[dict[str, object]], *, held: str) -> None:
    # Match the local matrix row positions.  Held panels retain source indices
    # across months, while the matrix is dense 0..n-1.
    base = frame.loc[:, ["__decision_ts__", "candidate_id", "policy_net_bps"]].reset_index(drop=True).copy()
    sizes = base.groupby("__decision_ts__", sort=False).candidate_id.transform("size").to_numpy(float)
    for offset in range(0, len(fields), 48):
        chunk = list(fields[offset:offset + 48])
        block = pd.DataFrame(values[:, offset:offset + len(chunk)], columns=chunk)
        # Rank only within a timestamp: this tests candidate discrimination,
        # not the value of a broad market-level level shift.
        ranks = block.groupby(base["__decision_ts__"], sort=False).rank(method="first", ascending=False)
        for field in chunk:
            rank = ranks[field].to_numpy(float)
            take_high = np.isfinite(rank) & (rank <= np.ceil(sizes * .10))
            take_low = np.isfinite(rank) & (rank >= (sizes - np.ceil(sizes * .10) + 1.0))
            candidates: list[tuple[str, np.ndarray]] = [("high", take_high), ("low", take_low)]
            best_direction, best_ev = "high", -np.inf
            for direction, take in candidates:
                work = base.loc[take, ["__decision_ts__", "policy_net_bps"]]
                ev = float(work.groupby("__decision_ts__", sort=False).policy_net_bps.mean().mean()) if len(work) else -np.inf
                if ev > best_ev:
                    best_direction, best_ev = direction, ev
            out.append({"held_month": held, "feature": field, "univariate_direction": best_direction, "univariate_ts_top10_ev": best_ev})


def _correlation_sample(feature_root: Path, fields: Sequence[str], month: pd.Timestamp, cap: int) -> pd.DataFrame:
    path = feature_root / f"month={month:%Y-%m}" / "causal_feature_universe.parquet"
    ids = pd.read_parquet(path, columns=["candidate_id"])
    chosen = (pd.util.hash_pandas_object(ids.candidate_id, index=False).to_numpy(np.uint64).argsort(kind="stable")[:min(cap, len(ids))])
    pieces: list[pd.DataFrame] = []
    for offset in range(0, len(fields), 64):
        chunk = list(fields[offset:offset + 64])
        part = pd.read_parquet(path, columns=chunk).iloc[chosen].reset_index(drop=True)
        pieces.append(part.apply(pd.to_numeric, errors="coerce"))
    return pd.concat(pieces, axis=1)


def _redundancy(fields: Sequence[str], coverage: pd.DataFrame, sample: pd.DataFrame, threshold: float) -> tuple[list[str], pd.DataFrame]:
    # Connected components are used only to remove near duplicates at the
    # requested 0.995 threshold; semantic families remain independent.
    parent = list(range(len(fields)))
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a
    # pandas.DataFrame.corr materialises several full float64 work arrays and
    # exceeds the bounded research worker even for a 4k sample.  Build
    # standardised ranks once, search candidate pairs in small blocks, then
    # verify every candidate with exact pairwise Spearman.  The approximation
    # only reduces the comparison set; it never decides a veto.
    n_rows, n_fields = len(sample), len(fields)
    ranks = np.empty((n_rows, n_fields), dtype=np.float32)
    for index, field in enumerate(fields):
        values = pd.to_numeric(sample[field], errors="coerce").replace([np.inf, -np.inf], np.nan)
        rank = values.rank(method="average", na_option="keep").to_numpy(np.float64)
        fill = float(np.nanmedian(rank)) if np.isfinite(rank).any() else 0.0
        rank = np.nan_to_num(rank, nan=fill)
        scale = float(rank.std())
        ranks[:, index] = ((rank - float(rank.mean())) / (scale if scale > 1e-12 else 1.0)).astype(np.float32)
    candidate_pairs: list[tuple[int, int]] = []
    approximate_floor = max(.985, threshold - .010)
    for begin in range(0, n_fields, 48):
        end = min(n_fields, begin + 48)
        approximate = np.abs((ranks[:, begin:end].T @ ranks) / max(1, n_rows - 1))
        for local, row in enumerate(approximate):
            index = begin + local
            for other in np.flatnonzero(row[index + 1:] >= approximate_floor) + index + 1:
                candidate_pairs.append((index, int(other)))
    for left, right in candidate_pairs:
        x = pd.to_numeric(sample[fields[left]], errors="coerce").to_numpy(float)
        y = pd.to_numeric(sample[fields[right]], errors="coerce").to_numpy(float)
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 32:
            continue
        exact = pd.Series(x[finite]).corr(pd.Series(y[finite]), method="spearman")
        if np.isfinite(exact) and abs(float(exact)) >= threshold:
            union(left, right)
    groups: dict[int, list[str]] = defaultdict(list)
    for index, field in enumerate(fields):
        groups[find(index)].append(field)
    cover = coverage.set_index("feature").mean_finite_fraction.to_dict()
    rows: list[dict[str, object]] = []
    keep: list[str] = []
    for group in groups.values():
        winner = max(group, key=lambda field: (cover.get(field, 0.0), -len(field), field))
        keep.append(winner)
        for field in group:
            rows.append({"feature": field, "correlation_cluster": "|".join(sorted(group)), "cluster_size": len(group), "cluster_representative": winner, "retained_after_redundancy": field == winner})
    return sorted(keep), pd.DataFrame(rows)


def _screen_head(
    *, head: str, folds: Sequence[pd.Timestamp], feature_root: Path, router_root: Path,
    labels_root: Path, policy: pd.DataFrame, fields: Sequence[str], route_fraction: float,
    train_months: int, reserve_days: int, train_cap: int, n_jobs: int, shap_cap: int,
    held_cap: int, random_models_per_fold: int, out: Path, base_labels_root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spec = HEADS[head]
    target, direction = str(spec["target"]), float(spec["direction"])
    gain_rows: list[dict[str, object]] = []
    shap_rows: list[dict[str, object]] = []
    uni_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    for ordinal, held_month in enumerate(folds):
        token = f"{held_month:%Y-%m}"
        reserve = held_month - pd.Timedelta(days=reserve_days)
        start = reserve - pd.DateOffset(months=train_months)
        # First construct strict candidate/label metadata.  Passing no feature
        # fields reads only stable identities; the wide matrix is loaded below
        # for the sampled routed IDs only.
        window = _joined(feature_root=feature_root, router_root=router_root, labels_root=labels_root, policy=policy, start=start, end=_month_end(held_month), fields=(), route_fraction=route_fraction, base_labels_root=base_labels_root)
        train = _strict_train(window.loc[window.__decision_ts__.lt(reserve)].copy(), reserve, target, train_cap)
        held = _time_balanced_sample(_held_eval(window.loc[window.__decision_ts__.ge(held_month)].copy(), target), held_cap, seed=SEED + 500 * ordinal)
        if len(train) < 8_000 or len(held) < 1_000:
            raise AssertionError(f"{head}/{token}: insufficient strict support train={len(train)} held={len(held)}")
        selected = pd.concat([train, held], ignore_index=True)
        selected_values = _selected_feature_matrix(feature_root, selected, fields)
        selected_values, _ = _impute_from_train(selected_values, len(train))
        x_train = selected_values[:len(train)]
        x_held = selected_values[len(train):]
        model = LGBMRegressor(**_params(seed=SEED + 100 * ordinal + (0 if head == "E" else 10_000), n_jobs=n_jobs))
        model.fit(x_train, pd.to_numeric(train[target], errors="coerce").to_numpy(float))
        held["__score__"] = direction * model.predict(x_held)
        metrics = _metric_suite(held, "__score__")
        fold_rows.append({
            "head": head, "held_month": token, "train_rows": len(train), "held_rows": len(held),
            "unlabelled_candidate_rows": int((~window.supportive_label_joined).sum()),
            "policy_unlabelled_candidate_rows": int((~window.policy_label_joined).sum()),
            **metrics,
        })
        gain = model.booster_.feature_importance(importance_type="gain")
        split = model.booster_.feature_importance(importance_type="split")
        total_gain = max(float(gain.sum()), 1e-12)
        for field, raw_gain, raw_split in zip(fields, gain, split, strict=True):
            gain_rows.append({"head": head, "held_month": token, "feature": field, "gain": float(raw_gain), "gain_norm": float(raw_gain / total_gain), "split": int(raw_split), "used": bool(raw_split > 0)})
        global_ix = _stratified_index(held, shap_cap, seed=SEED + ordinal)
        rank = held.groupby("__decision_ts__", sort=False)["__score__"].rank(pct=True, ascending=True, method="first")
        precision_ix = _stratified_index(held, shap_cap, rank=rank, seed=SEED + 1000 + ordinal)
        for sample_name, index in (("general", global_ix), ("precision_region_p70_100", precision_ix)):
            contribution = model.predict(x_held[index], pred_contrib=True)
            values = np.abs(np.asarray(contribution, dtype=np.float64)[:, :-1])
            for field, mean_abs, median_abs in zip(fields, values.mean(axis=0), np.median(values, axis=0), strict=True):
                shap_rows.append({"head": head, "held_month": token, "sample": sample_name, "feature": field, "mean_abs_shap": float(mean_abs), "median_abs_shap": float(median_abs), "rows": int(len(index))})
        _univariate(held, x_held, fields, uni_rows, held=token)
        # Twelve (three folds x four seeds by default) cheap, feature-randomised
        # stability models meet the requested 10--20 model range without making
        # a training row appear in an in-sample held evaluation.
        for repeat in range(random_models_per_fold):
            seed = SEED + 20_000 + 1000 * ordinal + repeat + (0 if head == "E" else 50_000)
            sampled = _time_balanced_sample(train, min(len(train), int(.85 * len(train))), seed=seed)
            # sampled preserves source-row identity but we already hold the
            # fully materialised training matrix in exact row order.
            random_rows = sampled.index.to_numpy(np.int64)
            # _time_balanced_sample retains original indices from train.
            train_positions = pd.Series(np.arange(len(train), dtype=np.int64), index=train.index)
            x_random = x_train[train_positions.reindex(random_rows).to_numpy(np.int64)]
            random_model = LGBMRegressor(**_params(seed=seed, n_jobs=n_jobs, cheap=True, feature_fraction=.5 + .1 * (repeat % 3)))
            random_model.fit(x_random, pd.to_numeric(sampled[target], errors="coerce").to_numpy(float))
            gain_random = random_model.booster_.feature_importance(importance_type="gain")
            split_random = random_model.booster_.feature_importance(importance_type="split")
            total = max(float(gain_random.sum()), 1e-12)
            for field, raw_gain, raw_split in zip(fields, gain_random, split_random, strict=True):
                stability_rows.append({"head": head, "held_month": token, "seed": seed, "feature": field, "gain_norm": float(raw_gain / total), "used": bool(raw_split > 0)})
            del random_model, x_random
        _progress(out, stage="screen_fold_complete", head=head, held_month=token, train_rows=len(train), held_rows=len(held), **metrics)
        # LightGBM owns native buffers outside normal Python reference counting.
        # Dispose them at each temporal fold boundary so full-universe feature
        # screening has a bounded rather than cumulative memory footprint.
        del model, selected_values, x_train, x_held, selected, train, held, window
        del contribution, values
        gc.collect()
    return (pd.DataFrame(gain_rows), pd.DataFrame(shap_rows), pd.DataFrame(uni_rows), pd.DataFrame(stability_rows), pd.DataFrame(fold_rows))


def _rank_metric(frame: pd.DataFrame, column: str, ascending: bool = False) -> pd.Series:
    return frame[column].rank(method="average", ascending=ascending, pct=True).fillna(0.0)


def _shortlist(
    *, head: str, fields: Sequence[str], correlation: pd.DataFrame, gain: pd.DataFrame,
    shap: pd.DataFrame, univariate: pd.DataFrame, stability: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    gain_summary = gain.groupby("feature", sort=False).agg(
        gain_median=("gain_norm", "median"), gain_iqr=("gain_norm", lambda x: x.quantile(.75) - x.quantile(.25)),
        fold_use=("used", "mean"), split_median=("split", "median"),
    )
    gain_summary["stable_gain"] = gain_summary.gain_median * gain_summary.fold_use
    shap_summary = shap.pivot_table(index="feature", columns="sample", values="mean_abs_shap", aggfunc="median").fillna(0.0)
    shap_summary = shap_summary.rename(columns={"general": "global_shap", "precision_region_p70_100": "precision_shap"})
    uni_summary = univariate.groupby("feature", sort=False).univariate_ts_top10_ev.agg(["mean", "min"]).rename(columns={"mean": "univariate_top10_ev", "min": "univariate_worst_fold_ev"})
    stable_summary = stability.groupby("feature", sort=False).agg(random_use=("used", "mean"), random_gain=("gain_norm", "median"))
    result = pd.DataFrame(index=list(fields)).join(gain_summary).join(shap_summary).join(uni_summary).join(stable_summary).fillna(0.0)
    result["family"] = [_feature_family(field) for field in result.index]
    result["stable_gain_rank"] = _rank_metric(result, "stable_gain")
    result["global_shap_rank"] = _rank_metric(result, "global_shap")
    result["precision_shap_rank"] = _rank_metric(result, "precision_shap")
    result["univariate_rank"] = _rank_metric(result, "univariate_top10_ev")
    result["stability_rank"] = _rank_metric(result, "random_use")
    result["screen_score"] = (.30 * result.stable_gain_rank + .20 * result.global_shap_rank + .25 * result.precision_shap_rank + .15 * result.univariate_rank + .10 * result.stability_rank)
    selected: set[str] = set()
    for column, count in (("stable_gain", 80), ("global_shap", 60), ("precision_shap", 60), ("univariate_top10_ev", 40), ("random_use", 60)):
        selected.update(result.nlargest(min(count, len(result)), column).index)
    # Semantic rescue is deliberately modest: it admits the best screened
    # representative of families that otherwise vanish from the union.
    for _, family_frame in result.groupby("family", sort=False):
        selected.add(str(family_frame.nlargest(1, "screen_score").index[0]))
    eligible = result.loc[result.index.isin(selected)].copy()
    eligible["correlation_representative"] = correlation.set_index("feature").reindex(eligible.index).cluster_representative.eq(eligible.index).fillna(True)
    eligible = eligible.loc[eligible.correlation_representative].copy()
    # Keep independent physical-head contracts.  Within the 120 cap, preserve
    # one representative of every observed semantic family then fill by screen.
    representatives = eligible.groupby("family", sort=False).screen_score.idxmax().tolist()
    chosen = list(dict.fromkeys(representatives))
    chosen.extend([field for field in eligible.sort_values("screen_score", ascending=False, kind="stable").index if field not in set(chosen)])
    chosen = chosen[:120]
    result["in_shortlist_union"] = result.index.isin(selected)
    result["selected_screen120"] = result.index.isin(chosen)
    result = result.reset_index(names="feature").sort_values("screen_score", ascending=False, kind="stable")
    return result, chosen


def run(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    folds = tuple(_utc(value) for value in args.held_months)
    if not 1 <= len(folds) <= 3 or tuple(sorted(folds)) != folds:
        raise ValueError("the screen requires one to three chronological held months")
    coverage_months = tuple(sorted({*(folds), *(_utc(folds[0] - pd.DateOffset(months=4) + pd.DateOffset(months=i)) for i in range(4))}))
    all_fields = _numeric_fields(args.feature_root, f"{coverage_months[-1]:%Y-%m}")
    coverage = _coverage(args.feature_root, all_fields, coverage_months)
    eligible = coverage.loc[coverage.coverage_pass, "feature"].tolist()
    if len(eligible) < 700:
        raise AssertionError(f"hygiene left only {len(eligible)} fields; expected roughly 750--900")
    sample = _correlation_sample(args.feature_root, eligible, folds[0], args.correlation_sample_rows)
    survivors, clusters = _redundancy(eligible, coverage, sample, args.redundancy_threshold)
    if len(survivors) < 650:
        raise AssertionError(f"near-duplicate veto was too aggressive: {len(survivors)} survivors")
    policy = _read_policy(args.policy_path, coverage_months)
    coverage.to_parquet(args.out / "feature_hygiene_coverage.parquet", index=False, compression="zstd")
    clusters.to_parquet(args.out / "correlation_clusters.parquet", index=False, compression="zstd")
    _progress(args.out, stage="hygiene_complete", full_numeric_fields=len(all_fields), coverage_eligible=len(eligible), post_redundancy=len(survivors))
    manifest = {
        "schema": "strict_r3_routed_et_fulluniverse_screen_v1",
        "scope": "offline research only; B0, live configuration, exchange and execution artifacts prohibited",
        "router_root": str(args.router_root), "router_sha256": _sha(args.router_root), "router_output": ROUTER_FIELD,
        "route": "exact strict-OOF timestamp-local top 50%", "route_fraction": .50,
        "feature_root": str(args.feature_root), "policy_path": str(args.policy_path), "labels_root": str(args.labels_root),
        "base_labels_root": str(args.base_labels_root) if args.base_labels_root else None,
        "heads": {head: HEADS[head] for head in args.heads}, "held_months": [f"{month:%Y-%m}" for month in folds],
        "strict_train": {"train_months": args.train_months, "reserve_days": args.reserve_days, "train_cap": args.train_cap, "held_cap": args.held_cap, "labels_before_reserve": True},
        "hygiene": {"full_numeric": len(all_fields), "coverage_eligible": len(eligible), "post_near_duplicate": len(survivors), "redundancy_abs_spearman": args.redundancy_threshold},
        "screen": {"folds": len(folds), "seeds": 1, "randomized_models_per_fold": args.randomized_models_per_fold, "shap": "OOF LightGBM pred_contrib / TreeSHAP on stratified samples"},
        "targets_are_inference_features": False,
    }
    _write_json_exclusive(args.out / "run_manifest.json", manifest)
    all_gain: list[pd.DataFrame] = []
    all_shap: list[pd.DataFrame] = []
    all_uni: list[pd.DataFrame] = []
    all_stability: list[pd.DataFrame] = []
    all_folds: list[pd.DataFrame] = []
    for head in args.heads:
        gain, shap, uni, stable, fold = _screen_head(
            head=head, folds=folds, feature_root=args.feature_root, router_root=args.router_root,
            labels_root=args.labels_root, policy=policy, fields=survivors, route_fraction=.50,
            train_months=args.train_months, reserve_days=args.reserve_days, train_cap=args.train_cap,
            n_jobs=args.n_jobs, shap_cap=args.shap_cap, held_cap=args.held_cap,
            random_models_per_fold=args.randomized_models_per_fold, out=args.out, base_labels_root=args.base_labels_root,
        )
        summary, selected = _shortlist(head=head, fields=survivors, correlation=clusters, gain=gain, shap=shap, univariate=uni, stability=stable)
        _write_json_exclusive(args.out / f"{head.lower()}_screen120_contract.json", {
            "head": head, "target": HEADS[head], "feature_contract": selected,
            "feature_contract_sha256": hashlib.sha256("\n".join(selected).encode()).hexdigest(),
            "selection": "full-universe screen union: stable gain, global/precision OOF TreeSHAP, univariate economic rescue, randomized stability, semantic rescue, near-duplicate veto",
        })
        summary.to_parquet(args.out / f"{head.lower()}_screen_feature_summary.parquet", index=False, compression="zstd")
        _progress(args.out, stage="screen120_selected", head=head, features=len(selected), sha256=hashlib.sha256("\n".join(selected).encode()).hexdigest())
        all_gain.append(gain); all_shap.append(shap); all_uni.append(uni); all_stability.append(stable); all_folds.append(fold)
    pd.concat(all_gain, ignore_index=True).to_parquet(args.out / "screen_gain.parquet", index=False, compression="zstd")
    pd.concat(all_shap, ignore_index=True).to_parquet(args.out / "screen_shap.parquet", index=False, compression="zstd")
    pd.concat(all_uni, ignore_index=True).to_parquet(args.out / "screen_univariate.parquet", index=False, compression="zstd")
    pd.concat(all_stability, ignore_index=True).to_parquet(args.out / "screen_randomized_stability.parquet", index=False, compression="zstd")
    pd.concat(all_folds, ignore_index=True).to_parquet(args.out / "screen_fold_metrics.parquet", index=False, compression="zstd")
    _progress(args.out, stage="screen_complete", heads=list(args.heads))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--heads", nargs="+", choices=tuple(HEADS), default=("E", "T"))
    parser.add_argument("--base-labels-root", type=Path, help="required only when screening B0 policy-ordinal target")
    parser.add_argument("--held-months", nargs="+", type=lambda value: _utc(value), default=(_utc("2026-02-01"), _utc("2026-03-01"), _utc("2026-04-01")))
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--train-cap", type=int, default=120_000)
    parser.add_argument("--held-cap", type=int, default=25_000)
    parser.add_argument("--n-jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--shap-cap", type=int, default=8_000)
    parser.add_argument("--randomized-models-per-fold", type=int, default=4)
    # The 0.995 veto detects only practical duplicates.  A deterministic 4k
    # sample is ample for that purpose and keeps the all-pairs Spearman pass
    # safely below the research worker's memory ceiling.
    parser.add_argument("--correlation-sample-rows", type=int, default=4_096)
    parser.add_argument("--redundancy-threshold", type=float, default=.995)
    args = parser.parse_args()
    if args.train_months < 2 or args.reserve_days < 12 or args.train_cap < 20_000 or args.held_cap < 5_000:
        raise ValueError("invalid strict screen temporal contract")
    if "B0" in args.heads and args.base_labels_root is None:
        raise ValueError("--base-labels-root is required when screening B0")
    if not .99 <= args.redundancy_threshold <= .999:
        raise ValueError("full-universe hygiene permits only an essentially-identical correlation veto")
    run(args)


if __name__ == "__main__":
    main()
