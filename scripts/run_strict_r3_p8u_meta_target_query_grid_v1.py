#!/usr/bin/env python3
"""Strict-OOF P8u Meta target × query screen.

The frozen P8u Raw-bps CatBoost Base is the only upstream score owner.  This
program does not inherit the historic E/T model coordinate: it joins the
immutable P8u target-free Base score/rank to its matching causal feature rows,
then trains candidate Meta heads only on labels resolved before each fold.

This is stage 1 of Meta optimisation.  It produces target-free Meta score
receipts and diagnostic metrics; it does not fit MC1, alter admission, modify
the live stack, or perform exchange I/O.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import dataclasses
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMRanker
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mutual_info_score


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "strict_r3_p8u_meta_target_query_grid_v1"
IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
BASE_COLUMNS = (*IDENTITY, "base_score", "base_rank_ts")
POLICY_COLUMNS = (
    "candidate_id", "policy_path_valid", "policy_net_bps", "policy_exit_reason",
    "policy_label_available_ts",
)
PATH_COLUMNS = (
    "candidate_id", "__decision_ts__", "side_name", "supportive_path_valid",
    "supportive_label_available_ts", "path_arch_atr_fraction", "path_arch_peak_mfe_atr",
)
PROHIBITED = frozenset({
    "policy_path_valid", "policy_gross_bps", "policy_net_bps", "policy_exit_bar_15m",
    "policy_entry_price", "policy_exit_price", "policy_exit_reason",
    "policy_label_available_ts", "policy_cost_bps", "policy_outcome_source",
    "label_source_complete_1m_path", "supportive_path_valid",
    "supportive_label_available_ts", "path_arch_peak_mfe_atr", "path_arch_atr_fraction",
})
TRAILING_ACTIVATION_ATR = 0.5
STOP_REASONS = frozenset({"stop_loss", "fast_adverse"})


@dataclasses.dataclass(frozen=True)
class Arm:
    name: str
    family: str
    scale: str
    query: str
    threshold: float | None = None
    edges: tuple[float, ...] = ()
    grade_count: int = 7
    clip: float | None = None


@dataclasses.dataclass(frozen=True)
class Spec:
    raw: Mapping[str, Any]
    config_path: Path

    @property
    def source(self) -> Mapping[str, Any]:
        return self.raw["source"]

    @property
    def folds(self) -> Mapping[str, Any]:
        return self.raw["folds"]

    @property
    def model(self) -> Mapping[str, Any]:
        return self.raw["model"]

    @property
    def blend(self) -> Mapping[str, Any]:
        return self.raw["provisional_meta_blend"]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    targets = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    for target in targets:
        digest.update(str(target).encode())
        with target.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _apply_source_override(raw: Mapping[str, Any], path: Path | None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Bind a source-only receipt without changing target/query semantics."""
    result = deepcopy(dict(raw))
    if path is None:
        return result, None
    payload = json.loads(path.read_text())
    source = payload.get("source", payload) if isinstance(payload, Mapping) else None
    allowed = {"base_target_free_root", "full_feature_roots", "base_f72_contract", "policy_labels", "path_labels"}
    required = {"base_target_free_root", "full_feature_roots", "policy_labels", "path_labels"}
    if not isinstance(source, Mapping) or set(source).difference(allowed) or not required.issubset(source):
        raise AssertionError("source override must declare the complete permitted source mapping")
    if not isinstance(source["full_feature_roots"], list) or not all(isinstance(value, str) for value in source["full_feature_roots"]):
        raise AssertionError("source override full_feature_roots must be a list of strings")
    if any(not isinstance(source[key], str) for key in ("base_target_free_root", "policy_labels", "path_labels")):
        raise AssertionError("source override roots must be strings")
    result["source"] = {**dict(result["source"]), **dict(source)}
    return result, dict(source)


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _progress(root: Path, **payload: object) -> None:
    with (root / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str, sort_keys=True) + "\n")


def _utc_month(value: str) -> pd.Timestamp:
    return pd.Timestamp(f"{value}-01", tz="UTC")


def _month_end(month: pd.Timestamp) -> pd.Timestamp:
    return month + pd.offsets.MonthBegin(1)


def _month_range(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Timestamp, ...]:
    right_edge = end - pd.Timedelta(nanoseconds=1)
    left = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    right = pd.Timestamp(year=right_edge.year, month=right_edge.month, day=1, tz="UTC")
    return tuple(pd.date_range(left, right, freq="MS"))


def _base_path(root: Path, month: pd.Timestamp) -> Path:
    path = root / f"month={month:%Y-%m}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _full_path(roots: Sequence[Path], month: pd.Timestamp) -> Path:
    found = [root / f"month={month:%Y-%m}" / "causal_feature_universe.parquet" for root in roots]
    existing = [path for path in found if path.exists()]
    if len(existing) != 1:
        raise AssertionError(
            f"{month:%Y-%m}: expected exactly one full causal feature owner, got {len(existing)}"
        )
    return existing[0]


def _path_label_paths(root: Path, month: pd.Timestamp) -> tuple[Path, ...]:
    """Return available label partitions that can own ``month`` decisions.

    Historical label layouts differ at month boundaries.  Older sources put
    a small set of next-month decisions in the preceding signal-month part,
    while the current recovered source is partitioned directly by decision
    month.  The current partition is therefore mandatory; a preceding
    partition is an optional additional owner.  Callers still filter by
    decision timestamp and preflight exact candidate identity coverage, so
    omitting an absent preceding partition cannot silently reduce supervision.
    """
    previous = root / f"month={(month - pd.offsets.MonthBegin(1)):%Y-%m}" / "side=long.parquet"
    current = root / f"month={month:%Y-%m}" / "side=long.parquet"
    if not current.exists():
        raise FileNotFoundError(current)
    return tuple(path for path in (previous, current) if path.exists())


def _read_selection(path: Path, *, expected_count: int = 72) -> tuple[str, ...]:
    payload = json.loads(path.read_text())
    # A source feature contract may be a conventional selector receipt
    # (``selected_features``), an older shorthand (``features``), or the
    # append-only F120 + state-stack contract (``all_features``).  The latter
    # is deliberately an additive universe, not a replacement for the parent
    # F120 contract; selection remains a later, fold-local stage.
    fields = payload.get("selected_features", payload.get("all_features", payload.get("features")))
    if not isinstance(fields, list) or len(fields) != expected_count or len(set(fields)) != len(fields):
        raise AssertionError(f"{path}: expected exactly {expected_count} unique selected Meta fields")
    return tuple(str(field) for field in fields)


def _valid_query_contract(query: object) -> bool:
    """Return whether ``query`` is one of the declared causal query forms.

    ``base_band_block<N>`` is intentionally parameterised: the magnitude
    market-state head may need a slower or faster temporal pooling interval,
    while Base-score bands preserve its conditional rather than timestamp-only
    semantics.  The block is a deterministic calendar partition, never a
    look-ahead statistic.
    """
    return str(query) in {"timestamp", "base_band"} or bool(
        re.fullmatch(r"base_band_block(?:7|10|14|21|28|35|42)", str(query))
    )


def _arm_specs(raw: Mapping[str, Any], selected: Sequence[str] | None) -> tuple[Arm, ...]:
    items: list[Arm] = []
    for family, body in raw["target_families"].items():
        for item in body["arms"]:
            query = item.get("query", body.get("query"))
            if not _valid_query_contract(query):
                raise ValueError(f"invalid query {query!r} for {item}")
            scale = str(item["scale"])
            if scale not in {"bps", "atr", "sqrt_atr"}:
                raise ValueError(f"invalid scale {scale!r}")
            threshold = item.get("threshold")
            edges = tuple(float(value) for value in item.get("edges", ()))
            grade_count = int(item.get("grade_count", 7))
            clip = item.get("clip")
            if grade_count < 3 or grade_count > 16:
                raise ValueError(f"{item['name']}: grade_count must be in [3, 16]")
            if clip is not None and float(clip) <= 0.0:
                raise ValueError(f"{item['name']}: clip must be positive when declared")
            if family in {"under", "over"} and threshold is None:
                raise ValueError(f"{item['name']}: missing threshold")
            if family == "state" and len(edges) < 4:
                raise ValueError(f"{item['name']}: missing signed state edges")
            items.append(Arm(
                str(item["name"]), family, scale, str(query), threshold, edges,
                grade_count, float(clip) if clip is not None else None,
            ))
    names = [arm.name for arm in items]
    if len(names) != len(set(names)):
        raise AssertionError("target grid has duplicate arm names")
    if selected is not None:
        wanted = set(selected)
        missing = sorted(wanted.difference(names))
        if missing:
            raise ValueError(f"unknown requested arms: {missing}")
        items = [arm for arm in items if arm.name in wanted]
    return tuple(items)


def _assert_target_free(path: Path) -> None:
    names = set(pq.ParquetFile(path).schema_arrow.names)
    leaked = sorted(PROHIBITED.intersection(names))
    if leaked:
        raise AssertionError(f"{path}: target-free input leaks labels/outcomes {leaked}")


def _rank_desc(frame: pd.DataFrame, column: str) -> np.ndarray:
    work = frame.loc[:, ["candidate_id", "__decision_ts__", column]].copy()
    work["row"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(["__decision_ts__", column, "candidate_id"], ascending=[True, False, True], kind="stable")
    order = work.groupby("__decision_ts__", sort=False).cumcount().to_numpy(float) + 1.0
    size = work.groupby("__decision_ts__", sort=False).candidate_id.transform("size").to_numpy(float)
    result = np.empty(len(work), dtype=np.float32)
    result[work.row.to_numpy(np.int64)] = (1.0 - (order - .5) / size).astype(np.float32)
    return result


def _add_base_geometry(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    persisted = pd.to_numeric(out.base_rank_ts, errors="coerce").to_numpy(float)
    rebuilt = _rank_desc(out, "base_score")
    if not np.allclose(persisted, rebuilt, rtol=0.0, atol=1e-6, equal_nan=False):
        raise AssertionError("P8u Base rank receipt is not equal to its persisted score ordering")
    summary = out.groupby("__decision_ts__", sort=False).base_score.agg(["size", "mean", "std", "min", "max"])
    out["base_query_count"] = out.__decision_ts__.map(summary["size"]).astype(np.float32)
    out["base_query_mean"] = out.__decision_ts__.map(summary["mean"]).astype(np.float32)
    out["base_query_std"] = out.__decision_ts__.map(summary["std"]).fillna(0.0).astype(np.float32)
    out["base_query_range"] = (out.__decision_ts__.map(summary["max"]) - out.__decision_ts__.map(summary["min"])).astype(np.float32)
    out["base_score_z_ts"] = ((out.base_score - out.base_query_mean) / out.base_query_std.replace(0.0, np.nan)).fillna(0.0).astype(np.float32)
    ordered = out.loc[:, ["candidate_id", "__decision_ts__", "base_score"]].sort_values(
        ["__decision_ts__", "base_score", "candidate_id"], ascending=[True, False, True], kind="stable"
    )
    ordered["next"] = ordered.groupby("__decision_ts__", sort=False).base_score.shift(-1)
    ordered["third"] = ordered.groupby("__decision_ts__", sort=False).base_score.shift(-2)
    top = ordered.groupby("__decision_ts__", sort=False).first()
    out["base_top_gap"] = out.__decision_ts__.map(top.base_score - top["next"]).fillna(0.0).astype(np.float32)
    out["base_top2_gap"] = out.__decision_ts__.map(top["next"] - top["third"]).fillna(0.0).astype(np.float32)
    return out


def _read_base_features(
    *, base_root: Path, feature_roots: Sequence[Path], start: pd.Timestamp, end: pd.Timestamp,
    fields: Sequence[str],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for month in _month_range(start, end):
        base_path, full_path = _base_path(base_root, month), _full_path(feature_roots, month)
        _assert_target_free(base_path); _assert_target_free(full_path)
        names = set(pq.ParquetFile(full_path).schema_arrow.names)
        missing = sorted(set(fields).difference(names))
        if missing:
            raise AssertionError(f"{full_path}: missing frozen Meta fields {missing[:8]}")
        base = pd.read_parquet(base_path, columns=list(BASE_COLUMNS))
        full = pd.read_parquet(full_path, columns=[*IDENTITY, *fields])
        for piece in (base, full):
            piece["__decision_ts__"] = pd.to_datetime(piece["__decision_ts__"], utc=True, errors="raise")
            if piece.duplicated(IDENTITY).any():
                raise AssertionError(f"{month:%Y-%m}: duplicate target-free identity")
        merged = base.merge(full, on=list(IDENTITY), how="left", validate="one_to_one")
        if len(merged) != len(base) or merged.loc[:, list(fields)].isna().all(axis=None):
            raise AssertionError(f"{month:%Y-%m}: causal feature identity coverage failure")
        parts.append(merged.loc[merged.__decision_ts__.ge(start) & merged.__decision_ts__.lt(end)].copy())
    out = pd.concat(parts, ignore_index=True)
    if out.duplicated(IDENTITY).any() or not out.side_name.eq("long").all():
        raise AssertionError("invalid P8u target-free population")
    return _add_base_geometry(out.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True))


def _read_policy(path: Path) -> pd.DataFrame:
    policy = pd.read_parquet(path, columns=list(POLICY_COLUMNS))
    if policy.candidate_id.duplicated().any():
        raise AssertionError("policy ledger has duplicate candidate IDs")
    policy["policy_path_valid"] = policy.policy_path_valid.fillna(False).astype(bool)
    policy["policy_net_bps"] = pd.to_numeric(policy.policy_net_bps, errors="coerce")
    policy["policy_label_available_ts"] = pd.to_datetime(policy.policy_label_available_ts, utc=True, errors="coerce")
    return policy


def _read_path(root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for month in _month_range(start, end):
        for path in _path_label_paths(root, month):
            part = pd.read_parquet(path, columns=list(PATH_COLUMNS))
            part["__decision_ts__"] = pd.to_datetime(part["__decision_ts__"], utc=True, errors="raise")
            part["supportive_label_available_ts"] = pd.to_datetime(part["supportive_label_available_ts"], utc=True, errors="coerce")
            parts.append(part.loc[part.__decision_ts__.ge(month) & part.__decision_ts__.lt(_month_end(month))].copy())
    out = pd.concat(parts, ignore_index=True)
    if out.duplicated(IDENTITY).any():
        raise AssertionError("path label source has duplicate candidate IDs")
    out["supportive_path_valid"] = out.supportive_path_valid.fillna(False).astype(bool)
    out["path_arch_atr_fraction"] = pd.to_numeric(out.path_arch_atr_fraction, errors="coerce")
    out["path_arch_peak_mfe_atr"] = pd.to_numeric(out.path_arch_peak_mfe_atr, errors="coerce")
    return out


def _labelled(frame: pd.DataFrame, policy: pd.DataFrame, path_root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    path = _read_path(path_root, start, end)
    out = frame.merge(path, on=list(IDENTITY), how="left", validate="one_to_one")
    out = out.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    if len(out) != len(frame):
        raise AssertionError("label join changed target-free candidate identities")
    out["atr_bps"] = (out.path_arch_atr_fraction * 10_000.0).astype(np.float32)
    return out


def _valid_label(frame: pd.DataFrame, cutoff: pd.Timestamp | None = None) -> np.ndarray:
    valid = (
        frame.policy_path_valid.fillna(False).astype(bool)
        & frame.supportive_path_valid.fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(frame.policy_net_bps, errors="coerce"))
        & np.isfinite(pd.to_numeric(frame.atr_bps, errors="coerce"))
        & pd.to_numeric(frame.atr_bps, errors="coerce").gt(0.0)
    )
    if cutoff is not None:
        valid &= frame.policy_label_available_ts.lt(cutoff) & frame.supportive_label_available_ts.lt(cutoff)
    return valid.to_numpy(bool)


def _sample_queries(frame: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame.reset_index(drop=True).copy()
    work = frame.copy()
    q = work.loc[:, ["__decision_ts__"]].drop_duplicates().copy()
    q["month"] = q.__decision_ts__.dt.strftime("%Y-%m")
    q["hash"] = pd.util.hash_pandas_object(q.__decision_ts__.astype(str) + f"|{seed}", index=False).to_numpy(np.uint64)
    size = work.groupby("__decision_ts__", sort=False).size()
    q["rows"] = q.__decision_ts__.map(size).astype(int)
    quota = max(1, cap // max(1, q.month.nunique()))
    keep: list[pd.Timestamp] = []
    for _, group in q.sort_values(["month", "hash", "__decision_ts__"], kind="stable").groupby("month", sort=False):
        used = 0
        for timestamp, _month, _hash, rows in group.loc[:, ["__decision_ts__", "month", "hash", "rows"]].itertuples(index=False, name=None):
            if used and used + int(rows) > quota:
                continue
            keep.append(timestamp); used += int(rows)
    sampled = work.loc[work.__decision_ts__.isin(keep)].copy()
    if sampled.empty:
        raise AssertionError("query-safe training sample is empty")
    return sampled.reset_index(drop=True)


def _fit_anchor(frame: pd.DataFrame, valid: np.ndarray) -> IsotonicRegression:
    if int(valid.sum()) < 1_000:
        raise AssertionError("insufficient earlier resolved labels for Base-to-EV map")
    return IsotonicRegression(increasing=True, out_of_bounds="clip").fit(
        frame.loc[valid, "base_rank_ts"], frame.loc[valid, "policy_net_bps"],
    )


def _prequential_anchor(frame: pd.DataFrame, *, block_days: int) -> np.ndarray:
    work = frame.copy()
    work["__row__"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    result = np.full(len(work), np.nan, dtype=np.float32)
    start = work.__decision_ts__.min().floor("D")
    work["block"] = ((work.__decision_ts__ - start) / pd.Timedelta(days=block_days)).astype(int)
    for block in sorted(work.block.unique()):
        current = work.block.eq(block)
        block_start = work.loc[current, "__decision_ts__"].min()
        prior = _valid_label(work, block_start) & work.__decision_ts__.lt(block_start).to_numpy(bool)
        if int(prior.sum()) < 1_000:
            continue
        anchor = _fit_anchor(work, prior)
        result[current.to_numpy()] = anchor.predict(work.loc[current, "base_rank_ts"]).astype(np.float32)
    original = np.full(len(frame), np.nan, dtype=np.float32)
    original[work["__row__"].to_numpy(np.int64)] = result
    return original


def _normalise(residual: np.ndarray, atr_bps: np.ndarray, scale: str) -> np.ndarray:
    if scale == "bps":
        return residual.astype(np.float32)
    divisor = np.maximum(np.asarray(atr_bps, dtype=float), 1e-3)
    if scale == "sqrt_atr":
        divisor = np.sqrt(divisor)
    return (np.asarray(residual, dtype=float) / divisor).astype(np.float32)


def _train_target(frame: pd.DataFrame, arm: Arm, *, anchor: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    valid = _valid_label(frame) & np.isfinite(anchor)
    net = pd.to_numeric(frame.policy_net_bps, errors="coerce").to_numpy(float)
    residual = (net - anchor).astype(np.float32)
    scaled = _normalise(residual, frame.atr_bps.to_numpy(float), arm.scale)
    label = np.full(len(frame), -1, dtype=np.int32)
    info: dict[str, Any] = {"family": arm.family, "scale": arm.scale, "valid_rows": int(valid.sum())}
    if arm.family == "magnitude":
        values = scaled[valid]
        if arm.clip is not None:
            values = np.clip(values, -float(arm.clip), float(arm.clip))
        edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, int(arm.grade_count) + 1)[1:-1]))
        label[valid] = np.digitize(values, edges).astype(np.int32)
        info["edges"] = [float(value) for value in edges]
        info["grade_count"] = int(arm.grade_count)
        info["clip"] = float(arm.clip) if arm.clip is not None else None
        info["direction"] = "higher_residual_better"
    elif arm.family == "under":
        reaches = frame.path_arch_peak_mfe_atr.to_numpy(float) >= TRAILING_ACTIVATION_ATR
        label[valid] = (reaches[valid] & (scaled[valid] >= float(arm.threshold))).astype(np.int32)
        info["direction"] = "higher_unexpected_favourable_path_better"
        info["threshold"] = float(arm.threshold)
    elif arm.family == "over":
        stopped = frame.policy_exit_reason.astype(str).isin(STOP_REASONS).to_numpy(bool)
        label[valid] = (stopped[valid] & (scaled[valid] <= -float(arm.threshold))).astype(np.int32)
        info["direction"] = "higher_unexpected_stop_worse; output is inverted"
        info["threshold"] = float(arm.threshold)
    elif arm.family == "state":
        label[valid] = np.digitize(scaled[valid], arm.edges).astype(np.int32)
        info["edges"] = list(arm.edges)
        info["direction"] = "signed_overconfident_to_underconfident_order"
    else:  # pragma: no cover
        raise AssertionError(arm.family)
    return label, residual, info


def _base_band(frame: pd.DataFrame) -> np.ndarray:
    # rank 1.0 is the strongest upstream candidate.  Bands are therefore
    # explicit Base-score strata, rather than global score quantiles.
    return np.minimum(9, np.maximum(0, np.floor((1.0 - frame.base_rank_ts.to_numpy(float)) * 10.0))).astype(np.int16)


def _query_ids(frame: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "timestamp":
        return frame.__decision_ts__.astype(str).to_numpy(object)
    bands = _base_band(frame)
    if mode == "base_band":
        return np.asarray([f"band{value:02d}" for value in bands], dtype=object)
    match = re.fullmatch(r"base_band_block(7|10|14|21|28|35|42)", mode)
    if match:
        days = int(match.group(1))
        epoch = pd.Timestamp("2025-01-01", tz="UTC")
        block = ((frame.__decision_ts__ - epoch) / pd.Timedelta(days=days)).astype(int)
        return np.asarray([f"block{left:03d}|band{right:02d}" for left, right in zip(block, bands)], dtype=object)
    raise ValueError(mode)


def _bounded_queries(frame: pd.DataFrame, query_ids: np.ndarray, cap: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    work = pd.DataFrame({
        "query": query_ids, "timestamp": frame.__decision_ts__.to_numpy(),
        "candidate_id": frame.candidate_id.astype(str).to_numpy(), "row": np.arange(len(frame), dtype=np.int64),
    })
    output = np.empty(len(work), dtype=object)
    for query, group in work.groupby("query", sort=True):
        used, shard = 0, 0
        for _, part in group.sort_values(["timestamp", "candidate_id"], kind="stable").groupby("timestamp", sort=False):
            if used and used + len(part) > cap:
                shard += 1; used = 0
            output[part.row.to_numpy(np.int64)] = f"{query}|s{shard:04d}"
            used += len(part)
    ordered = pd.DataFrame({"query": output, "candidate_id": frame.candidate_id.astype(str), "row": np.arange(len(frame))})
    ordered = ordered.sort_values(["query", "candidate_id"], kind="stable")
    sizes = ordered.groupby("query", sort=False).size()
    keep = set(sizes.index[sizes.ge(2)])
    ordered = ordered.loc[ordered["query"].isin(keep)].copy()
    group_sizes = ordered.groupby("query", sort=False).size().astype(int).tolist()
    if not group_sizes or sum(group_sizes) != len(ordered):
        raise AssertionError("invalid rank query grouping")
    return ordered.row.to_numpy(np.int64), output[ordered.row.to_numpy(np.int64)], group_sizes


def _matrix(frame: pd.DataFrame, fields: Sequence[str]) -> np.ndarray:
    geometry = (
        "base_score", "base_rank_ts", "base_query_count", "base_query_mean", "base_query_std",
        "base_query_range", "base_score_z_ts", "base_top_gap", "base_top2_gap",
    )
    return frame.loc[:, [*geometry, *fields]].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)


def _impute(train: np.ndarray, held: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    medians = np.nanmedian(train, axis=0).astype(np.float32)
    medians[~np.isfinite(medians)] = 0.0
    output: list[np.ndarray] = []
    for values in (train.copy(), held.copy()):
        missing = ~np.isfinite(values)
        if missing.any():
            values[missing] = np.broadcast_to(medians, values.shape)[missing]
        output.append(values.astype(np.float32, copy=False))
    return output[0], output[1]


def _model(spec: Spec, *, label_gain: Sequence[float], seed: int) -> LGBMRanker:
    raw = spec.model
    objective = str(raw.get("objective", "lambdarank"))
    if objective not in {"lambdarank", "rank_xendcg"}:
        raise ValueError(f"unsupported LightGBM ranking objective {objective!r}")
    return LGBMRanker(
        objective=objective, metric="ndcg", label_gain=list(label_gain),
        n_estimators=int(raw["n_estimators"]), learning_rate=float(raw["learning_rate"]),
        max_depth=int(raw["max_depth"]), num_leaves=int(raw["num_leaves"]),
        min_child_samples=int(raw["min_child_samples"]), min_split_gain=float(raw["min_split_gain"]),
        colsample_bytree=float(raw["feature_fraction"]), subsample=float(raw["bagging_fraction"]),
        subsample_freq=1, reg_alpha=float(raw["lambda_l1"]), reg_lambda=float(raw["lambda_l2"]),
        sigmoid=float(raw["sigmoid"]), random_state=seed, n_jobs=1, verbosity=-1,
    )


def _score_fold(
    *, train: pd.DataFrame, held_target_free: pd.DataFrame, arm: Arm, fields: Sequence[str], spec: Spec,
    fold_month: pd.Timestamp, seed: int,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    anchors = _prequential_anchor(train, block_days=int(spec.folds["anchor_block_days"]))
    labels, _residual, target_info = _train_target(train, arm, anchor=anchors)
    valid = labels >= 0
    sampled = _sample_queries(train.loc[valid].copy(), int(spec.folds["max_train_rows"]), seed)
    # Sampling after target construction preserves its strict-prequential Base
    # anchor.  Reindex by the retained original ids rather than recomputing it.
    labels_frame = pd.DataFrame({"candidate_id": train.candidate_id, "label": labels})
    sampled = sampled.merge(labels_frame, on="candidate_id", how="left", validate="one_to_one")
    y = sampled.label.to_numpy(np.int32)
    if len(sampled) < 20_000 or len(np.unique(y)) < 2:
        raise AssertionError(f"{arm.name} {fold_month:%Y-%m}: insufficient target support")
    train_x, held_x = _impute(_matrix(sampled, fields), _matrix(held_target_free, fields))
    order, _queries, groups = _bounded_queries(sampled, _query_ids(sampled, arm.query), int(spec.folds["max_query_rows"]))
    y = y[order]; x = train_x[order]
    gain = (0, 1, 2, 4, 7, 11, 16, 24)[: int(np.nanmax(y)) + 1]
    model = _model(spec, label_gain=gain, seed=seed)
    model.fit(x, y, group=groups)
    raw = np.asarray(model.predict(held_x), dtype=np.float32)
    if arm.family == "over":
        raw *= -1.0
    score = held_target_free.loc[:, list(IDENTITY) + ["base_score", "base_rank_ts"]].copy().reset_index(drop=True)
    score["meta_raw_score"] = raw
    rank_frame = score.loc[:, ["candidate_id", "__decision_ts__"]].copy()
    rank_frame["value"] = raw
    score["meta_rank_ts"] = _rank_desc(rank_frame, "value")
    score["arm"] = arm.name
    score["family"] = arm.family
    score["scale"] = arm.scale
    score["query_contract"] = arm.query
    score["held_month"] = f"{fold_month:%Y-%m}"
    score["target_free"] = True
    audit = {
        "held_month": f"{fold_month:%Y-%m}", "arm": arm.name, "family": arm.family,
        "train_rows_before_sample": int(valid.sum()), "train_rows": int(len(sampled)),
        "train_queries": int(len(groups)), "classes": int(np.nanmax(y) + 1),
        "features": int(len(fields) + 9), **target_info,
    }
    return score, anchors, audit


def _conditional_mi(meta: np.ndarray, base: np.ndarray, outcome: np.ndarray) -> float:
    def bins(values: np.ndarray) -> np.ndarray:
        out = np.full(len(values), -1, dtype=np.int16)
        valid = np.isfinite(values)
        if int(valid.sum()) < 20:
            return out
        rank = pd.Series(values[valid]).rank(method="average", pct=True).to_numpy(float)
        out[valid] = np.minimum(9, np.floor(rank * 10)).astype(np.int16)
        return out
    m, b, y = bins(meta), bins(base), bins(outcome)
    valid = (m >= 0) & (b >= 0) & (y >= 0)
    if int(valid.sum()) < 100:
        return float("nan")
    total, result = float(valid.sum()), 0.0
    for band in np.unique(b[valid]):
        local = valid & (b == band)
        if int(local.sum()) >= 20:
            result += float(local.sum()) / total * mutual_info_score(m[local], y[local])
    return float(result)


def _topk(frame: pd.DataFrame, column: str, k: int) -> pd.DataFrame:
    ordered = frame.sort_values(["__decision_ts__", column, "candidate_id"], ascending=[True, False, True], kind="stable")
    return ordered.groupby("__decision_ts__", sort=False).head(k)


def _top_fraction(frame: pd.DataFrame, column: str, fraction: float) -> pd.DataFrame:
    ordered = frame.sort_values(["__decision_ts__", column, "candidate_id"], ascending=[True, False, True], kind="stable").copy()
    ordered["ordinal"] = ordered.groupby("__decision_ts__", sort=False).cumcount()
    ordered["size"] = ordered.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size")
    return ordered.loc[ordered.ordinal.lt(np.ceil(fraction * ordered["size"]))].copy()


def _timestamp_value(frame: pd.DataFrame, column: str, *, utility: bool) -> pd.Series:
    values = frame["utility_bps"] if utility else frame["policy_net_bps"]
    return pd.DataFrame({"timestamp": frame.__decision_ts__, "value": values}).groupby("timestamp", sort=False).value.mean()


def _band_metrics(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    work = frame.copy()
    # Bands exactly match the requested Base strata.  Higher base rank means a
    # stronger Base prediction; no rows below the Base top-30% enter this
    # conditional diagnostic.
    definitions = (("0_5", .95, 1.001), ("5_10", .90, .95), ("10_20", .80, .90), ("20_30", .70, .80))
    rows: list[dict[str, float | str]] = []
    ic_values: dict[str, float] = {}
    spread_values: dict[str, float] = {}
    recalls: list[tuple[int, int]] = []
    rescues: list[float] = []
    for name, low, high in definitions:
        local = work.loc[work.base_rank_ts.ge(low) & work.base_rank_ts.lt(high)].copy()
        if len(local) < 40:
            continue
        ic = float(spearmanr(local.meta_rank_ts, local.residual_bps).statistic)
        local["q"] = local.groupby("__decision_ts__", sort=False).meta_rank_ts.transform(
            lambda x: pd.qcut(x.rank(method="first"), 4, labels=False, duplicates="drop")
        )
        high_q, low_q = local.loc[local.q.eq(3)], local.loc[local.q.eq(0)]
        spread = float(high_q.utility_bps.mean() - low_q.utility_bps.mean()) if len(high_q) and len(low_q) else float("nan")
        available = int(local.policy_net_bps.gt(100.0).sum())
        captured = int(high_q.policy_net_bps.gt(100.0).sum())
        base_density = float(local.utility_density_bps.mean())
        rescue_density = float(high_q.utility_density_bps.mean()) if len(high_q) else float("nan")
        rows.append({
            "base_band": name, "rows": len(local), "ic": ic, "utility_spread_bps": spread,
            "potential_recall100": captured / available if available else float("nan"),
            "positive_available": available, "positive_captured": captured,
            "net_rescue_separation_bps": rescue_density - base_density,
            "base_utility_bps": float(local.utility_bps.mean()), "meta_high_utility_bps": float(high_q.utility_bps.mean()) if len(high_q) else float("nan"),
            "conversion": float(high_q.utility_bps.mean() / local.utility_bps.mean()) if len(high_q) and abs(float(local.utility_bps.mean())) > 1e-9 else float("nan"),
        })
        ic_values[name] = ic; spread_values[name] = spread
        recalls.append((captured, available)); rescues.append(rescue_density - base_density)
    weights = {"0_5": .10, "5_10": .25, "10_20": .35, "20_30": .25}
    iccond = sum(weights[name] * ic_values.get(name, 0.0) for name in weights)
    spreadcond = sum(weights[name] * spread_values.get(name, 0.0) for name in weights)
    captured, available = sum(left for left, _ in recalls), sum(right for _, right in recalls)
    return pd.DataFrame(rows), {
        "iccond": float(iccond), "utility_spreadcond_bps": float(spreadcond),
        "potential_utility_recall": captured / available if available else float("nan"),
        "net_rescue_separation_bps": float(np.nanmean(rescues)) if rescues else float("nan"),
    }


def _metrics(
    *, score: pd.DataFrame, held_labelled: pd.DataFrame, held_anchor: IsotonicRegression, spec: Spec,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    outcome = held_labelled.loc[:, list(IDENTITY) + ["policy_path_valid", "policy_net_bps"]].copy()
    scored = score.merge(outcome, on=list(IDENTITY), how="left", validate="one_to_one")
    valid = scored.policy_path_valid.fillna(False).astype(bool) & np.isfinite(scored.policy_net_bps)
    work = scored.loc[valid].copy()
    if len(work) < 1_000:
        raise AssertionError("held Meta metrics have inadequate valid policy support")
    work["residual_bps"] = work.policy_net_bps.to_numpy(float) - held_anchor.predict(work.base_rank_ts).astype(float)
    low, high = (float(spec.blend["utility_clip_bps"][0]), float(spec.blend["utility_clip_bps"][1]))
    work["utility_bps"] = work.policy_net_bps.clip(low, high)
    density_low, density_high = (float(spec.blend["utility_density_clip_bps"][0]), float(spec.blend["utility_density_clip_bps"][1]))
    work["utility_density_bps"] = work.policy_net_bps.clip(density_low, density_high)
    work["combined_rank"] = (
        float(spec.blend["base_rank_weight"]) * work.base_rank_ts
        + float(spec.blend["meta_rank_weight"]) * work.meta_rank_ts
    )
    base_top2, combo_top2 = _topk(work, "base_rank_ts", 2), _topk(work, "combined_rank", 2)
    fraction = float(spec.blend["admission_proxy_fraction"])
    base_admit, combo_admit = _top_fraction(work, "base_rank_ts", fraction), _top_fraction(work, "combined_rank", fraction)
    weeks: list[dict[str, Any]] = []
    all_bands: list[pd.DataFrame] = []
    for week, group in work.groupby(work.__decision_ts__.dt.to_period("W-SUN").astype(str), sort=True):
        week_start = group.__decision_ts__.min().normalize()
        base2 = base_top2.loc[base_top2.__decision_ts__.isin(group.__decision_ts__)]
        combo2 = combo_top2.loc[combo_top2.__decision_ts__.isin(group.__decision_ts__)]
        basea = base_admit.loc[base_admit.__decision_ts__.isin(group.__decision_ts__)]
        comboa = combo_admit.loc[combo_admit.__decision_ts__.isin(group.__decision_ts__)]
        base2_utility, combo2_utility = _timestamp_value(base2, "base_rank_ts", utility=True).mean(), _timestamp_value(combo2, "combined_rank", utility=True).mean()
        basea_utility, comboa_utility = _timestamp_value(basea, "base_rank_ts", utility=True).mean(), _timestamp_value(comboa, "combined_rank", utility=True).mean()
        base2_ev, combo2_ev = _timestamp_value(base2, "base_rank_ts", utility=False).mean(), _timestamp_value(combo2, "combined_rank", utility=False).mean()
        bands, conditional = _band_metrics(group)
        if not bands.empty:
            bands["week"] = week; bands["week_start"] = week_start
            all_bands.append(bands)
        dutility2 = float(combo2_utility - base2_utility)
        dutilitya = float(comboa_utility - basea_utility)
        dev2 = float(combo2_ev - base2_ev)
        probe = .50 * dutility2 / 100.0 + .30 * dutilitya / 100.0 + .20 * dev2 / 100.0
        conditional_score = (
            .40 * conditional["iccond"]
            + .40 * conditional["utility_spreadcond_bps"] / 100.0
            + .20 * conditional["potential_utility_recall"]
        )
        smeta = .60 * probe + .25 * conditional_score
        weeks.append({
            "week": week, "week_start": week_start, "rows": len(group),
            "base_top2_utility_bps": base2_utility, "meta_top2_utility_bps": combo2_utility,
            "delta_utility_top2_bps": dutility2, "delta_utility_admission_bps": dutilitya,
            "delta_ev_top2_bps": dev2, "iccond": conditional["iccond"],
            "utility_spreadcond_bps": conditional["utility_spreadcond_bps"],
            "potential_utility_recall": conditional["potential_utility_recall"],
            "net_rescue_separation_bps": conditional["net_rescue_separation_bps"],
            "sprobe": probe, "sconditional": conditional_score, "smeta": smeta,
        })
    weekly = pd.DataFrame(weeks)
    if weekly.empty:
        raise AssertionError("no week-level Meta metrics")
    q20, q80 = weekly.smeta.quantile([.20, .80])
    robust = float(weekly.loc[weekly.smeta.ge(q20) & weekly.smeta.le(q80), "smeta"].mean())
    lower = float(weekly.smeta.quantile(.15) + weekly.smeta.quantile(.10) + weekly.smeta.quantile(.05)) / 3.0
    stable = robust + .50 * lower
    summary = {
        "valid_policy_rows": int(len(work)), "weeks": int(len(weekly)),
        "smeta_week_robust_average": robust, "smeta_week_lower_tail": lower, "sstable_meta": stable,
        "residual_spearman_ic": float(spearmanr(work.meta_rank_ts, work.residual_bps).statistic),
        "conditional_mi_meta_policy_given_base": _conditional_mi(work.meta_rank_ts.to_numpy(float), work.base_rank_ts.to_numpy(float), work.policy_net_bps.to_numpy(float)),
        "mean_top2_substitution_ev_bps": float(weekly.delta_ev_top2_bps.mean()),
        "mean_top2_substitution_utility_bps": float(weekly.delta_utility_top2_bps.mean()),
        "mean_admission_substitution_utility_bps": float(weekly.delta_utility_admission_bps.mean()),
        "worst_week_smeta": float(weekly.smeta.min()),
        "worst_week_delta_ev_top2_bps": float(weekly.delta_ev_top2_bps.min()),
        "mean_iccond": float(weekly.iccond.mean()), "mean_utility_spreadcond_bps": float(weekly.utility_spreadcond_bps.mean()),
        "mean_potential_utility_recall": float(weekly.potential_utility_recall.mean()),
        "mean_net_rescue_separation_bps": float(weekly.net_rescue_separation_bps.mean()),
    }
    return weekly, pd.concat(all_bands, ignore_index=True) if all_bands else pd.DataFrame(), summary


def _preflight(spec: Spec, fields: Sequence[str], months: Sequence[pd.Timestamp]) -> pd.DataFrame:
    base_root = ROOT / str(spec.source["base_target_free_root"])
    feature_roots = tuple(ROOT / str(value) for value in spec.source["full_feature_roots"])
    path_root = ROOT / str(spec.source["path_labels"])
    rows: list[dict[str, Any]] = []
    for month in months:
        base_path, feature_path = _base_path(base_root, month), _full_path(feature_roots, month)
        _assert_target_free(base_path); _assert_target_free(feature_path)
        base = pd.read_parquet(base_path, columns=list(IDENTITY))
        features = pd.read_parquet(feature_path, columns=list(IDENTITY))
        for piece in (base, features):
            piece["__decision_ts__"] = pd.to_datetime(piece["__decision_ts__"], utc=True, errors="raise")
        matched = base.merge(features, on=list(IDENTITY), how="left", indicator=True)["_merge"].eq("both").sum()
        paths = pd.concat([pd.read_parquet(path, columns=list(IDENTITY)) for path in _path_label_paths(path_root, month)], ignore_index=True)
        paths["__decision_ts__"] = pd.to_datetime(paths["__decision_ts__"], utc=True, errors="raise")
        paths = paths.loc[paths.__decision_ts__.ge(month) & paths.__decision_ts__.lt(_month_end(month))].drop_duplicates(list(IDENTITY))
        path_matched = base.merge(paths, on=list(IDENTITY), how="left", indicator=True)["_merge"].eq("both").sum()
        rows.append({"month": f"{month:%Y-%m}", "base_rows": len(base), "feature_matched": int(matched), "path_matched": int(path_matched), "meta_feature_count": len(fields)})
    audit = pd.DataFrame(rows)
    if not (audit.base_rows.eq(audit.feature_matched) & audit.base_rows.eq(audit.path_matched)).all():
        raise AssertionError("P8u Meta preflight identity coverage failure")
    return audit


def run(
    *, config: Path, out: Path, arm_names: Sequence[str] | None = None,
    held_month_values: Sequence[str] | None = None, source_override: Path | None = None,
) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    raw, applied_source_override = _apply_source_override(json.loads(config.read_text()), source_override)
    spec = Spec(raw=raw, config_path=config)
    fields = _read_selection(ROOT / str(spec.source["base_f72_contract"]), expected_count=int(raw.get("meta_feature_count", 72)))
    arms = _arm_specs(raw, arm_names)
    months = tuple(_utc_month(value) for value in (held_month_values or spec.folds["held_months"]))
    if not months or tuple(sorted(months)) != months:
        raise AssertionError("held months must be chronological")
    base_root = ROOT / str(spec.source["base_target_free_root"])
    feature_roots = tuple(ROOT / str(value) for value in spec.source["full_feature_roots"])
    policy_path = ROOT / str(spec.source["policy_labels"])
    path_root = ROOT / str(spec.source["path_labels"])
    policy = _read_policy(policy_path)
    out.mkdir(parents=True)
    preflight_months = tuple(_month_range(months[0] - pd.DateOffset(months=5), _month_end(months[-1])))
    preflight = _preflight(spec, fields, preflight_months)
    preflight.to_parquet(out / "source_coverage_audit.parquet", index=False, compression="zstd")
    # Some compact exact-identity overlays expose the monthly feature panels
    # under ``features/`` while keeping their immutable run manifest at the
    # overlay root.  Resolve that documented layout without weakening the
    # requirement that every feature owner be hash-bound.
    feature_manifests: dict[str, str] = {}
    for root in feature_roots:
        manifest = root / "run_manifest.json"
        if not manifest.exists() and root.name == "features":
            manifest = root.parent / "run_manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(f"missing feature-root manifest for {root}")
        feature_manifests[str(root)] = _sha(manifest)
    _once(out / "run_manifest.json", {
        "schema": SCHEMA, "scope": raw["scope"], "config": str(config),
        "base_contract": raw["base_contract"], "arms": [dataclasses.asdict(arm) for arm in arms],
        # Downstream GateProxy descriptor materialisation must bind the
        # exact frozen feature contract, not merely a derived field hash.
        # This is especially important when multiple target/query screen
        # receipts share the same F72 list but arise from different source
        # bridges.  The path below is immutable configuration provenance;
        # it is not a feature-selection fallback.
        "feature_contract": str(spec.source["base_f72_contract"]),
        "f72_feature_count": len(fields), "f72_feature_sha256": hashlib.sha256("\n".join(fields).encode()).hexdigest(),
        "held_months": [f"{month:%Y-%m}" for month in months],
        "source": raw["source"],
        "source_override": str(source_override) if source_override else None,
        "source_override_sha256": _sha(source_override) if source_override else None,
        "source_override_payload": applied_source_override,
        "source_hashes": {
            "base": _sha(base_root), "policy": _sha(policy_path), "path": _sha(path_root),
            "full_feature_manifests": feature_manifests,
        },
        "causality": raw["causality"],
        "preflight": "all P8u Base IDs must have exactly one causal full-feature owner and one path-label owner; held labels are never source inputs",
    })
    all_summary: list[dict[str, Any]] = []
    all_weekly: list[pd.DataFrame] = []
    all_bands: list[pd.DataFrame] = []
    all_audit: list[dict[str, Any]] = []
    train_months, reserve_days = int(spec.folds["train_months"]), int(spec.folds["resolved_label_reserve_days"])
    for arm_index, arm in enumerate(arms):
        arm_summary: list[dict[str, Any]] = []
        for fold_index, held_month in enumerate(months):
            reserve = held_month - pd.Timedelta(days=reserve_days)
            start, end = reserve - pd.DateOffset(months=train_months), _month_end(held_month)
            base = _read_base_features(base_root=base_root, feature_roots=feature_roots, start=start, end=end, fields=fields)
            train_tf = base.loc[base.__decision_ts__.lt(reserve)].copy()
            held_tf = base.loc[base.__decision_ts__.ge(held_month)].copy()
            # Training labels are materialised only after target-free frame and
            # candidate identities are frozen.  The held source remains
            # target-free through model fit and persisted scoring.
            train = _labelled(train_tf, policy, path_root, start, reserve)
            train = train.loc[_valid_label(train, reserve)].copy()
            if len(train) < 30_000 or len(held_tf) < 10_000:
                raise AssertionError(f"{arm.name} {held_month:%Y-%m}: insufficient support")
            score, _anchors, audit = _score_fold(
                train=train, held_target_free=held_tf, arm=arm, fields=fields, spec=spec,
                fold_month=held_month, seed=int(spec.folds["seed"]) + 1000 * arm_index + fold_index,
            )
            target_free_path = out / "target_free_scores" / arm.name / f"month={held_month:%Y-%m}.parquet"
            target_free_path.parent.mkdir(parents=True, exist_ok=True)
            score.to_parquet(target_free_path, index=False, compression="zstd")
            # Only after the score receipt is immutable do held labels enter
            # metric computation.  They never affect the score receipt.
            held_labelled = _labelled(held_tf, policy, path_root, held_month, end)
            held_valid = _valid_label(train)
            held_anchor = _fit_anchor(train, held_valid)
            weekly, bands, metrics = _metrics(score=score, held_labelled=held_labelled, held_anchor=held_anchor, spec=spec)
            weekly["arm"] = arm.name; weekly["family"] = arm.family; weekly["held_month"] = f"{held_month:%Y-%m}"
            if not bands.empty:
                bands["arm"] = arm.name; bands["family"] = arm.family; bands["held_month"] = f"{held_month:%Y-%m}"
                all_bands.append(bands)
            all_weekly.append(weekly)
            audit.update(metrics); all_audit.append(audit); arm_summary.append(metrics)
            _progress(out, event="arm_fold_complete", arm=arm.name, held_month=f"{held_month:%Y-%m}", target_free_score=str(target_free_path), **metrics)
        agg = pd.DataFrame(arm_summary).mean(numeric_only=True).to_dict()
        all_summary.append({"arm": arm.name, "family": arm.family, "scale": arm.scale, "query": arm.query, **agg})
    summary = pd.DataFrame(all_summary)
    summary = summary.sort_values(["family", "sstable_meta", "arm"], ascending=[True, False, True], kind="stable")
    summary["family_rank"] = summary.groupby("family", sort=False).cumcount() + 1
    summary.to_parquet(out / "target_query_summary.parquet", index=False, compression="zstd")
    pd.DataFrame(all_audit).to_parquet(out / "target_query_fold_metrics.parquet", index=False, compression="zstd")
    pd.concat(all_weekly, ignore_index=True).to_parquet(out / "weekly_sstable_meta.parquet", index=False, compression="zstd")
    (pd.concat(all_bands, ignore_index=True) if all_bands else pd.DataFrame()).to_parquet(out / "base_band_conversion_metrics.parquet", index=False, compression="zstd")
    selected = summary.loc[summary.family_rank.eq(1), ["family", "arm", "sstable_meta", "conditional_mi_meta_policy_given_base", "mean_top2_substitution_ev_bps", "worst_week_smeta"]]
    selected.to_parquet(out / "one_winner_per_family_pre_mc1.parquet", index=False, compression="zstd")
    _once(out / "correctness_report.json", {
        "p8u_base_target_free_score_source": True,
        "frozen_f72_features_merged_by_exact_identity": True,
        "no_policy_or_path_field_in_target_free_inputs": True,
        "all_train_labels_resolved_before_reserve": True,
        "train_residual_anchor_strict_prequential": True,
        "held_scores_persisted_before_held_outcome_metrics": True,
        "base_band_metrics_limited_to_base_top30": True,
        "no_mc1_admission_portfolio_live_or_exchange_mutation": True,
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", default=None, help="comma-separated predeclared arm subset; only for parallel screen shards")
    parser.add_argument("--held-months", default=None, help="comma-separated predeclared held-month subset; pilot only when narrower than config")
    parser.add_argument("--source-override", type=Path, help="immutable source-only binding receipt")
    args = parser.parse_args()
    selected = tuple(item.strip() for item in args.arms.split(",") if item.strip()) if args.arms else None
    held = tuple(item.strip() for item in args.held_months.split(",") if item.strip()) if args.held_months else None
    print(run(config=args.config.resolve(), out=args.out.resolve(), arm_names=selected, held_month_values=held,
              source_override=args.source_override.resolve() if args.source_override else None))


if __name__ == "__main__":
    main()
