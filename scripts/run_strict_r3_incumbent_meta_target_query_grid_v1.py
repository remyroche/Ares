#!/usr/bin/env python3
"""Strict-OOF incumbent E/T meta target × query screen.

The retained base score is fixed:

``incumbent_upstream_bps = 0.50 * efficiency_bps + 0.50 * timing_bps``.

This first meta stage does *not* select features, re-map EV, admit trades, or
change the live stack.  It isolates four requested target families against a
small query grid using the incumbent's existing 120 causal inputs plus
target-free E/T score/rank geometry.  A held score receipt is persisted before
any policy/path outcome is joined for diagnostics.

The later feature-selection stage starts from the separately materialised full
causal universe.  Keeping target/query discovery on the frozen 120 inputs
avoids conflating label semantics with a 1,400-feature search.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMRanker, early_stopping
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mutual_info_score


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_strict_r3_o3v2_target_funnel as target_contract  # noqa: E402


SCHEMA = "strict_r3_incumbent_meta_target_query_grid_v1"
SEED = 1729
IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
TRAIN_MONTHS = 4
RESERVE_DAYS = 28
ANCHOR_BLOCK_DAYS = 14
SCORE_BAND_WIDTH = .05
BASE_BLOCK_EPOCH = pd.Timestamp("2025-01-01", tz="UTC")
ROUTE_FRACTION = .30
MAX_TRAIN_ROWS = 120_000
MAX_LAMBDARANK_QUERY_ROWS = 8_000
DEFAULT_SOURCE_ROOT = ROOT / "data_perp/artifacts/strict_r3_enhanced_base_live_stack_challenger_20260823_v10/target_free_monthly"
DEFAULT_POLICY = ROOT / "data_perp/artifacts/strict_r3_enhanced_base_rich_policy_labels_reconciled_20260823_v1/canonical_reconciled_policy_labels.parquet"
DEFAULT_PATH_ROOT = ROOT / "data_perp/artifacts/strict_r3_long_supportive_path_labels_2024_2026_20260823_v6_observed_entry/parts"

SCORE_COLUMNS = (
    "base_bps", "efficiency_bps", "timing_bps", "enhanced_base_bps", "base_rank_ts",
    "enhanced_base_routed", "e_minus_t", "e_minus_b0", "t_minus_b0", "base_component_std",
)
PATH_COLUMNS = (
    "candidate_id", "__decision_ts__", "side_name", "supportive_path_valid",
    "supportive_label_available_ts", "path_arch_atr_fraction",
)
POLICY_COLUMNS = (
    "candidate_id", "policy_path_valid", "policy_net_bps", "policy_exit_reason",
    "policy_label_available_ts",
)
PROHIBITED = set(target_contract.PROHIBITED_SCORE_COLUMNS)


@dataclasses.dataclass(frozen=True)
class Arm:
    name: str
    family: Literal["magnitude", "under", "over", "state"]
    scale: Literal["bps", "atr", "sqrt_atr"]
    query: Literal["base_band", "timestamp", "base_band_block28"]
    threshold: float | None = None
    classes: int = 7
    state_edges: tuple[float, ...] | None = None
    gain_schedule: Literal["small", "medium", "high"] = "medium"
    truncation_level: int | None = None


@dataclasses.dataclass(frozen=True)
class Fold:
    held_month: pd.Timestamp
    train: pd.DataFrame
    held: pd.DataFrame
    source_features: tuple[str, ...]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    targets = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    for target in targets:
        digest.update(str(target).encode())
        with target.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _exclusive_json(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _progress(root: Path, **payload: object) -> None:
    with (root / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _parse_months(raw: str) -> tuple[pd.Timestamp, ...]:
    months = tuple(
        pd.Timestamp(f"{item.strip()}-01", tz="UTC")
        for item in raw.split(",") if item.strip()
    )
    if not months or len(set(months)) != len(months) or tuple(sorted(months)) != months:
        raise ValueError("--held-months must contain unique chronological YYYY-MM values")
    return months


def _month_end(month: pd.Timestamp) -> pd.Timestamp:
    return month + pd.offsets.MonthBegin(1)


def _months_between(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Timestamp, ...]:
    return tuple(pd.date_range(
        start.to_period("M").to_timestamp().tz_localize("UTC"),
        (end - pd.Timedelta(nanoseconds=1)).to_period("M").to_timestamp().tz_localize("UTC"),
        freq="MS",
    ))


def arms() -> tuple[Arm, ...]:
    result: list[Arm] = []
    for scale in ("bps", "atr", "sqrt_atr"):
        for query in ("base_band", "base_band_block28"):
            result.append(Arm(f"magnitude_{scale}__{query}", "magnitude", scale, query))
    for scale, threshold in (("bps", 50.0), ("atr", 1.0)):
        result.append(Arm(f"under_{scale}{threshold:g}__timestamp", "under", scale, "timestamp", threshold))
        result.append(Arm(f"over_{scale}{threshold:g}__timestamp", "over", scale, "timestamp", threshold))
    for scale in ("bps", "atr", "sqrt_atr"):
        for query in ("base_band", "timestamp", "base_band_block28"):
            result.append(Arm(f"state_{scale}__{query}", "state", scale, query))
    return tuple(result)


def _source_path(root: Path | Sequence[Path], month: pd.Timestamp) -> Path:
    """Resolve one immutable source owner for a calendar month.

    A forward evaluation can append a target-free E/T score month to an
    earlier immutable ledger.  A root sequence avoids copying parquet panels;
    the exact-one-owner check prevents accidental overlap or fallback.
    """
    roots = (root,) if isinstance(root, Path) else tuple(root)
    candidates = [item / f"month={month:%Y-%m}" / "scores_features.parquet" for item in roots]
    existing = [item for item in candidates if item.exists()]
    if len(existing) != 1:
        raise AssertionError(
            f"{month:%Y-%m}: expected exactly one target-free base source across "
            f"{[str(item) for item in roots]}, found {len(existing)}"
        )
    return existing[0]


def _source_feature_columns(root: Path | Sequence[Path], month: pd.Timestamp) -> tuple[str, ...]:
    source = _source_path(root, month)
    names = pq.ParquetFile(source).schema_arrow.names
    leakage = sorted(PROHIBITED.intersection(names))
    if leakage:
        raise AssertionError(f"{source}: target-free source leaks {leakage}")
    fields = tuple(item for item in names if item not in set(IDENTITY) | set(SCORE_COLUMNS) | {"__ts__", "__symbol__"})
    if len(fields) != 120:
        raise AssertionError(f"{source}: expected 120 frozen causal fields, found {len(fields)}")
    return fields


def _load_feature_contract(path: Path) -> tuple[str, ...]:
    """Load an explicit feature-selection receipt, never an implicit column set."""
    payload = json.loads(path.read_text())
    raw = payload.get("features", payload.get("feature_contract")) if isinstance(payload, dict) else payload
    if not isinstance(raw, list) or not raw or not all(isinstance(item, str) and item for item in raw):
        raise ValueError(f"{path}: feature contract must contain a non-empty string list under features")
    fields = tuple(raw)
    if len(fields) != len(set(fields)):
        raise ValueError(f"{path}: feature contract contains duplicate fields")
    return fields


def _full_feature_path(roots: Sequence[Path], month: pd.Timestamp) -> Path:
    matches = [root / f"month={month:%Y-%m}" / "causal_feature_universe.parquet" for root in roots]
    existing = [path for path in matches if path.exists()]
    if len(existing) != 1:
        raise AssertionError(
            f"{month:%Y-%m}: expected exactly one immutable full-feature panel across "
            f"{[str(root) for root in roots]}, found {len(existing)}"
        )
    return existing[0]


def _full_feature_columns(roots: Sequence[Path], month: pd.Timestamp, fields: Sequence[str]) -> tuple[str, ...]:
    source = _full_feature_path(roots, month)
    names = pq.ParquetFile(source).schema_arrow.names
    leakage = sorted(PROHIBITED.intersection(names))
    if leakage:
        raise AssertionError(f"{source}: target-free full feature source leaks {leakage}")
    missing = sorted(set(fields).difference(names))
    if missing:
        raise AssertionError(f"{source}: feature contract missing {missing[:10]}")
    return tuple(fields)


def _rank_desc(frame: pd.DataFrame, score: str) -> np.ndarray:
    work = frame.loc[:, ["candidate_id", "__decision_ts__", score]].copy()
    work["__row__"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(["__decision_ts__", score, "candidate_id"], ascending=[True, False, True], kind="stable")
    ordinal = work.groupby("__decision_ts__", sort=False).cumcount().to_numpy(float) + 1.0
    count = work.groupby("__decision_ts__", sort=False).candidate_id.transform("size").to_numpy(float)
    result = np.empty(len(frame), dtype=np.float32)
    result[work.__row__.to_numpy(np.int64)] = 1.0 - (ordinal - .5) / count
    return result


def _add_base_geometry(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    expected = .5 * pd.to_numeric(out.efficiency_bps, errors="coerce") + .5 * pd.to_numeric(out.timing_bps, errors="coerce")
    if not np.allclose(expected.to_numpy(float), pd.to_numeric(out.enhanced_base_bps, errors="coerce").to_numpy(float), rtol=0.0, atol=1e-6, equal_nan=False):
        raise AssertionError("source upstream is not the retained 50/50 E/T base")
    out["inc_base_rank_ts"] = _rank_desc(out, "enhanced_base_bps")
    # The incumbent source already carries the canonical inference route.  It
    # must be consumed as persisted rather than reconstructed here: exact
    # score ties can otherwise select a different candidate boundary even
    # though the 50/50 E/T score arithmetic is unchanged.
    if out.enhanced_base_routed.isna().any():
        raise AssertionError("incumbent source has missing canonical route state")
    out["inc_routed"] = out.enhanced_base_routed.astype(bool).to_numpy()
    summary = out.groupby("__decision_ts__", sort=False).enhanced_base_bps.agg(["size", "std", "min", "max"])
    out["inc_query_count"] = out.__decision_ts__.map(summary["size"]).astype(np.float32)
    out["inc_query_std"] = out.__decision_ts__.map(summary["std"]).fillna(0.0).astype(np.float32)
    out["inc_query_range"] = (out.__decision_ts__.map(summary["max"]) - out.__decision_ts__.map(summary["min"])).astype(np.float32)
    ordered = out.loc[:, ["candidate_id", "__decision_ts__", "enhanced_base_bps"]].sort_values(
        ["__decision_ts__", "enhanced_base_bps", "candidate_id"], ascending=[True, False, True], kind="stable"
    )
    ordered["__next__"] = ordered.groupby("__decision_ts__", sort=False).enhanced_base_bps.shift(-1)
    ordered["__third__"] = ordered.groupby("__decision_ts__", sort=False).enhanced_base_bps.shift(-2)
    top = ordered.groupby("__decision_ts__", sort=False).first()
    out["inc_top_gap"] = out.__decision_ts__.map(top.enhanced_base_bps - top.__next__).fillna(0.0).astype(np.float32)
    out["inc_top2_gap"] = out.__decision_ts__.map(top.__next__ - top.__third__).fillna(0.0).astype(np.float32)
    out["inc_e_minus_t"] = (pd.to_numeric(out.efficiency_bps, errors="coerce") - pd.to_numeric(out.timing_bps, errors="coerce")).astype(np.float32)
    out["inc_e_t_mean"] = expected.astype(np.float32)
    out["inc_e_t_abs_gap"] = out.inc_e_minus_t.abs().astype(np.float32)
    return out


def _read_base(source_root: Path | Sequence[Path], start: pd.Timestamp, end: pd.Timestamp, fields: Sequence[str]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    columns = [*IDENTITY, *SCORE_COLUMNS, *fields]
    for month in _months_between(start, end):
        source = _source_path(source_root, month)
        if not source.exists():
            raise FileNotFoundError(source)
        part = pd.read_parquet(source, columns=columns)
        part["__decision_ts__"] = pd.to_datetime(part["__decision_ts__"], utc=True, errors="raise")
        pieces.append(part.loc[part.__decision_ts__.ge(start) & part.__decision_ts__.lt(end)].copy())
    out = pd.concat(pieces, ignore_index=True)
    out = out.loc[out.__decision_ts__.ge(start) & out.__decision_ts__.lt(end)].copy()
    if out.duplicated(IDENTITY).any():
        raise AssertionError("target-free base source has duplicate identities")
    if not out.side_name.eq("long").all():
        raise AssertionError("incumbent meta screen is long-only")
    return _add_base_geometry(out)


def _read_full_features(roots: Sequence[Path], start: pd.Timestamp, end: pd.Timestamp, fields: Sequence[str]) -> pd.DataFrame:
    """Read only receipt-selected causal fields and prove one source per month.

    The full universe is intentionally materialised in immutable bounded date
    bundles.  A strict OOF fold may span two such bundles; choosing the one
    panel that owns each calendar month preserves the feature representation
    instead of copying or recomputing it into a new mutable panel.
    """
    pieces: list[pd.DataFrame] = []
    columns = [*IDENTITY, *fields]
    for month in _months_between(start, end):
        _full_feature_columns(roots, month, fields)
        source = _full_feature_path(roots, month)
        part = pd.read_parquet(source, columns=columns)
        part["__decision_ts__"] = pd.to_datetime(part["__decision_ts__"], utc=True, errors="raise")
        pieces.append(part.loc[part.__decision_ts__.ge(start) & part.__decision_ts__.lt(end)].copy())
    out = pd.concat(pieces, ignore_index=True)
    if out.duplicated(IDENTITY).any():
        raise AssertionError("full causal feature source has duplicate identities")
    return out


def _read_policy(path: Path) -> pd.DataFrame:
    data = pd.read_parquet(path, columns=list(POLICY_COLUMNS))
    data["policy_path_valid"] = data.policy_path_valid.fillna(False).astype(bool)
    data["policy_net_bps"] = pd.to_numeric(data.policy_net_bps, errors="coerce")
    data["policy_label_available_ts"] = pd.to_datetime(data.policy_label_available_ts, utc=True, errors="coerce")
    if data.candidate_id.duplicated().any():
        raise AssertionError("canonical policy ledger has duplicate candidate IDs")
    return data


def _read_path(path_root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for month in _months_between(start, end):
        # Path partitions are keyed by signal-close month whereas the target
        # free incumbent ledger is keyed by executable decision month.  The
        # first 00:00 decision can therefore live in the preceding partition.
        # Load both adjacent partitions, then filter by decision time; this is
        # an identity repair only and never selects on a future-path field.
        for token in (month - pd.offsets.MonthBegin(1), month):
            source = path_root / f"month={token:%Y-%m}" / "side=long.parquet"
            if not source.exists():
                raise FileNotFoundError(source)
            part = pd.read_parquet(source, columns=list(PATH_COLUMNS))
            part["__decision_ts__"] = pd.to_datetime(part["__decision_ts__"], utc=True, errors="raise")
            part["supportive_label_available_ts"] = pd.to_datetime(part["supportive_label_available_ts"], utc=True, errors="coerce")
            pieces.append(part.loc[part.__decision_ts__.ge(month) & part.__decision_ts__.lt(_month_end(month))].copy())
    out = pd.concat(pieces, ignore_index=True)
    if out.duplicated(IDENTITY).any():
        raise AssertionError("path label source has duplicate identities")
    out["supportive_path_valid"] = out.supportive_path_valid.fillna(False).astype(bool)
    out["path_arch_atr_fraction"] = pd.to_numeric(out.path_arch_atr_fraction, errors="coerce")
    return out


def _join_training_labels(base: pd.DataFrame, policy: pd.DataFrame, path_root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    path = _read_path(path_root, start, end)
    out = base.merge(path, on=IDENTITY, how="left", validate="one_to_one")
    out = out.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    if len(out) != len(base):
        raise AssertionError("label join changed target-free base identities")
    out["atr_bps"] = (pd.to_numeric(out.path_arch_atr_fraction, errors="coerce") * 10_000.0).astype(np.float32)
    return out


def _prepare_folds(
    *, source_root: Path | Sequence[Path], policy: pd.DataFrame, path_root: Path, held_months: Sequence[pd.Timestamp],
    full_feature_roots: Sequence[Path] = (), full_feature_fields: Sequence[str] = (),
    unlabelled_held_months: Sequence[pd.Timestamp] = (),
) -> tuple[Fold, ...]:
    using_full_features = bool(full_feature_roots or full_feature_fields)
    if using_full_features and (not full_feature_roots or not full_feature_fields):
        raise ValueError("full causal feature mode requires both feature roots and an explicit feature contract")
    fields = tuple(full_feature_fields) if using_full_features else _source_feature_columns(source_root, held_months[0])
    folds: list[Fold] = []
    for month in held_months:
        reserve = month - pd.Timedelta(days=RESERVE_DAYS)
        start = reserve - pd.DateOffset(months=TRAIN_MONTHS)
        end = _month_end(month)
        # The base receipt remains the sole source of E/T and canonical route
        # geometry.  The optional full universe is merged only by immutable
        # point-in-time candidate identity.
        base = _read_base(source_root, start, end, () if using_full_features else fields)
        if using_full_features:
            full = _read_full_features(full_feature_roots, start, end, fields)
            labelled_base = base.merge(full, on=list(IDENTITY), how="left", validate="one_to_one")
            # A point-in-time full-causal panel may deliberately retain the
            # complete universe while the immutable upstream receipt carries
            # only a causal router-selected subset.  The full panel must
            # therefore *cover* every base identity; it must never replace the
            # base population or be filtered by labels/outcomes.  Requiring
            # equal row counts here incorrectly rejects that valid superset
            # representation and encourages needless regenerated panels.
            if len(labelled_base) != len(base) or labelled_base.loc[:, list(fields)].isna().all(axis=None):
                raise AssertionError(f"{month:%Y-%m}: full feature identity merge failed")
            base = labelled_base
        # The current forward month may not yet have a resolved path panel.
        # Train labels are still mandatory and end at the reserve; only its
        # held diagnostics may be deliberately omitted.  This lets an
        # immutable target-free score receipt be produced without opening or
        # requiring unavailable future outcomes.
        unlabelled_held = month in set(unlabelled_held_months)
        labelled_train = _join_training_labels(
            base.loc[base.__decision_ts__.lt(reserve)].copy(), policy, path_root, start, reserve,
        )
        train = labelled_train.loc[
            labelled_train.inc_routed
            & labelled_train.policy_label_available_ts.lt(reserve)
            & labelled_train.supportive_label_available_ts.lt(reserve)
        ].copy()
        if unlabelled_held:
            held = base.loc[base.__decision_ts__.ge(month) & base.inc_routed].copy()
        else:
            labelled = _join_training_labels(base, policy, path_root, start, end)
            held = labelled.loc[labelled.__decision_ts__.ge(month) & labelled.inc_routed].copy()
        # Every target's label availability is rechecked in its own target
        # builder.  This common gate prevents the current path from being
        # treated as observed before it has resolved.
        if len(train) < 30_000 or len(held) < 10_000:
            raise AssertionError(f"{month:%Y-%m}: insufficient routed support train={len(train)} held={len(held)}")
        folds.append(Fold(month, train.reset_index(drop=True), held.reset_index(drop=True), fields))
    return tuple(folds)


def _sample_queries(frame: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame.reset_index(drop=True).copy()
    work = frame.loc[:, [*frame.columns]].copy()
    query = work.loc[:, ["__decision_ts__"]].drop_duplicates()
    query["month"] = query.__decision_ts__.dt.strftime("%Y-%m")
    query["hash"] = pd.util.hash_pandas_object(query.__decision_ts__.astype(str) + f"|{seed}", index=False).to_numpy(np.uint64)
    size = work.groupby("__decision_ts__", sort=False).size()
    query["rows"] = query.__decision_ts__.map(size).astype(int)
    quota = max(1, cap // max(1, query.month.nunique()))
    keep: list[pd.Timestamp] = []
    for _, group in query.sort_values(["month", "hash", "__decision_ts__"], kind="stable").groupby("month", sort=False):
        used = 0
        for ts, _month, _hash, rows in group.loc[:, ["__decision_ts__", "month", "hash", "rows"]].itertuples(index=False, name=None):
            if used and used + int(rows) > quota:
                continue
            keep.append(ts); used += int(rows)
    sampled = work.loc[work.__decision_ts__.isin(keep)].copy()
    if sampled.empty:
        raise AssertionError("query-safe sample is empty")
    return sampled.reset_index(drop=True)


def _fit_anchor(frame: pd.DataFrame) -> IsotonicRegression:
    valid = (
        frame.policy_path_valid.fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(frame.policy_net_bps, errors="coerce"))
        & np.isfinite(pd.to_numeric(frame.inc_base_rank_ts, errors="coerce"))
    )
    if int(valid.sum()) < 1_000:
        raise AssertionError("insufficient rows for base-anchor map")
    return IsotonicRegression(increasing=True, out_of_bounds="clip").fit(
        frame.loc[valid, "inc_base_rank_ts"], frame.loc[valid, "policy_net_bps"],
    )


def _prequential_train_anchor(frame: pd.DataFrame) -> np.ndarray:
    """Create training residual anchors without same-row calibration leakage."""
    work = frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index()
    result = np.full(len(work), np.nan, dtype=np.float32)
    start = work.__decision_ts__.min().floor("D")
    block = ((work.__decision_ts__ - start) / pd.Timedelta(days=ANCHOR_BLOCK_DAYS)).astype(int)
    work["__anchor_block__"] = block
    for value in sorted(work.__anchor_block__.unique()):
        current = work.__anchor_block__.eq(value)
        block_start = work.loc[current, "__decision_ts__"].min()
        prior = (
            work.__decision_ts__.lt(block_start)
            & work.policy_path_valid.fillna(False).astype(bool)
            & work.policy_label_available_ts.lt(block_start)
            & np.isfinite(pd.to_numeric(work.policy_net_bps, errors="coerce"))
            & np.isfinite(pd.to_numeric(work.inc_base_rank_ts, errors="coerce"))
        )
        if int(prior.sum()) < 1_000:
            continue
        mapper = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(
            work.loc[prior, "inc_base_rank_ts"], work.loc[prior, "policy_net_bps"],
        )
        result[current.to_numpy()] = mapper.predict(work.loc[current, "inc_base_rank_ts"]).astype(np.float32)
    original = np.empty(len(frame), dtype=np.float32)
    original[work["index"].to_numpy(np.int64)] = result
    return original


def _normalised_residual(residual_bps: np.ndarray, atr_bps: np.ndarray, scale: str) -> np.ndarray:
    if scale == "bps":
        return residual_bps.astype(np.float32)
    safe = np.maximum(np.asarray(atr_bps, dtype=float), 1e-3)
    divisor = safe if scale == "atr" else np.sqrt(safe)
    return (np.asarray(residual_bps, dtype=float) / divisor).astype(np.float32)


def _quantile_grades(values: np.ndarray, classes: int = 7) -> tuple[np.ndarray, tuple[float, ...]]:
    finite = np.isfinite(values)
    if int(finite.sum()) < 100:
        raise AssertionError("insufficient finite values for ordinal target")
    edges = np.unique(np.quantile(values[finite], np.linspace(0.0, 1.0, classes + 1)[1:-1]))
    grade = np.full(len(values), -1, dtype=np.int32)
    grade[finite] = np.digitize(values[finite], edges).astype(np.int32)
    return grade, tuple(float(value) for value in edges)


def _state_grades(values: np.ndarray, scale: str, edges_override: Sequence[float] | None = None) -> tuple[np.ndarray, tuple[float, ...]]:
    if edges_override is not None:
        edges = tuple(float(value) for value in edges_override)
    elif scale == "bps":
        edges = (-100.0, -25.0, 25.0, 100.0)
    elif scale == "atr":
        edges = (-1.0, -.25, .25, 1.0)
    else:
        # Equivalent bps-scale edges after division by sqrt(ATR bps), around
        # the modal 60--100 bps decision-time ATR regime.
        edges = (-12.0, -3.0, 3.0, 12.0)
    grade = np.full(len(values), -1, dtype=np.int32)
    finite = np.isfinite(values)
    grade[finite] = np.digitize(values[finite], edges).astype(np.int32)
    return grade, tuple(float(value) for value in edges)


def _target(frame: pd.DataFrame, arm: Arm, *, train: bool, held_anchor: IsotonicRegression | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    valid = (
        frame.policy_path_valid.fillna(False).astype(bool)
        & frame.supportive_path_valid.fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(frame.policy_net_bps, errors="coerce"))
        & np.isfinite(pd.to_numeric(frame.atr_bps, errors="coerce"))
        & pd.to_numeric(frame.atr_bps, errors="coerce").gt(0.0)
        & np.isfinite(pd.to_numeric(frame.inc_base_rank_ts, errors="coerce"))
    ).to_numpy(bool)
    if train:
        anchor = _prequential_train_anchor(frame)
        valid &= np.isfinite(anchor)
    else:
        if held_anchor is None:
            raise AssertionError("held target requires train-only anchor")
        anchor = held_anchor.predict(frame.inc_base_rank_ts).astype(np.float32)
    net = pd.to_numeric(frame.policy_net_bps, errors="coerce").to_numpy(float)
    residual_bps = (net - anchor).astype(np.float32)
    normalized = _normalised_residual(residual_bps, pd.to_numeric(frame.atr_bps, errors="coerce").to_numpy(float), arm.scale)
    labels = np.full(len(frame), -1, dtype=np.int32)
    info: dict[str, Any] = {"scale": arm.scale, "family": arm.family, "valid_rows": int(valid.sum())}
    if arm.family == "magnitude":
        masked = normalized.copy()
        masked[~valid] = np.nan
        labels, edges = _quantile_grades(masked, classes=arm.classes)
        info["edges"] = edges
        info["output_direction"] = "higher_residual_is_better"
    elif arm.family == "under":
        event = frame.policy_exit_reason.astype(str).isin(("trailing", "smooth_capital_protect")).to_numpy(bool)
        labels[valid] = (event[valid] & (normalized[valid] >= float(arm.threshold))).astype(np.int32)
        info["threshold"] = float(arm.threshold)
        info["output_direction"] = "higher_unexpected_trailing_is_better"
    elif arm.family == "over":
        bad = frame.policy_exit_reason.astype(str).isin(("stop_loss", "fast_adverse")).to_numpy(bool)
        labels[valid] = (bad[valid] & (normalized[valid] <= -float(arm.threshold))).astype(np.int32)
        info["threshold"] = float(arm.threshold)
        info["output_direction"] = "higher_unexpected_stop_is_worse; score is inverted after prediction"
    elif arm.family == "state":
        masked = normalized.copy()
        masked[~valid] = np.nan
        labels, edges = _state_grades(masked, arm.scale, arm.state_edges)
        info["edges"] = edges
        info["output_direction"] = "overconfident_to_underconfident_signed_order"
    else:  # pragma: no cover - dataclass restricts values
        raise AssertionError(arm.family)
    return labels, residual_bps, info


def _query_ids(frame: pd.DataFrame, query: str) -> np.ndarray:
    if query == "timestamp":
        return frame.__decision_ts__.astype(str).to_numpy(object)
    band = np.minimum(19, np.maximum(0, np.floor(pd.to_numeric(frame.inc_base_rank_ts, errors="coerce") / SCORE_BAND_WIDTH))).astype(int)
    if query == "base_band":
        return np.asarray([f"band{value:02d}" for value in band], dtype=object)
    if query == "base_band_block28":
        block = ((frame.__decision_ts__ - BASE_BLOCK_EPOCH) / pd.Timedelta(days=28)).astype(int)
        return np.asarray([f"block{left:03d}|band{right:02d}" for left, right in zip(block, band)], dtype=object)
    raise ValueError(query)


def _bounded_query_ids(frame: pd.DataFrame, query_ids: np.ndarray) -> np.ndarray:
    """Respect LightGBM's 10k group cap without splitting a timestamp.

    The score-band contract is intentionally cross-timestamp.  A global band
    can nevertheless contain more rows than LightGBM permits in one query.
    We partition only those exceptional groups into consecutive, fixed-size
    causal time shards.  This is a runtime representation constraint, not a
    label-, outcome-, or timestamp-local query change; its exact 8k ceiling is
    recorded in the manifest and every score receipt.
    """
    work = pd.DataFrame({
        "query": query_ids,
        "__decision_ts__": frame.__decision_ts__.to_numpy(),
        "candidate_id": frame.candidate_id.astype(str).to_numpy(),
        "row": np.arange(len(frame), dtype=np.int64),
    })
    output = np.empty(len(work), dtype=object)
    for query, group in work.groupby("query", sort=True):
        if len(group) <= MAX_LAMBDARANK_QUERY_ROWS:
            output[group.row.to_numpy(np.int64)] = str(query)
            continue
        chunk, used = 0, 0
        for _timestamp, part in group.sort_values(["__decision_ts__", "candidate_id"], kind="stable").groupby("__decision_ts__", sort=False):
            if used and used + len(part) > MAX_LAMBDARANK_QUERY_ROWS:
                chunk += 1; used = 0
            output[part.row.to_numpy(np.int64)] = f"{query}|shard{chunk:03d}"
            used += len(part)
    return output


def _ordered_query(frame: pd.DataFrame, query_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    bounded = _bounded_query_ids(frame, query_ids)
    work = pd.DataFrame({"query": bounded, "candidate_id": frame.candidate_id.astype(str), "row": np.arange(len(frame), dtype=np.int64)})
    work = work.sort_values(["query", "candidate_id"], kind="stable")
    sizes = work.groupby("query", sort=False).size()
    keep_queries = set(sizes.index[sizes.ge(2)])
    work = work.loc[work["query"].isin(keep_queries)]
    groups = work.groupby("query", sort=False).size().astype(int).tolist()
    if not groups or sum(groups) != len(work):
        raise AssertionError("invalid LambdaRank query groups")
    order = work.row.to_numpy(np.int64)
    return order, bounded[order], groups


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


def _matrix(frame: pd.DataFrame, source_features: Sequence[str]) -> np.ndarray:
    geometry = (
        "enhanced_base_bps", "efficiency_bps", "timing_bps", "inc_base_rank_ts", "inc_query_count",
        "inc_query_std", "inc_query_range", "inc_top_gap", "inc_top2_gap", "inc_e_minus_t",
        "inc_e_t_mean", "inc_e_t_abs_gap", "base_component_std",
    )
    return frame.loc[:, [*geometry, *source_features]].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)


_MODEL_OVERRIDE_KEYS = {
    "n_estimators", "learning_rate", "max_depth", "num_leaves", "min_child_samples",
    "min_split_gain", "colsample_bytree", "subsample", "subsample_freq", "reg_alpha",
    "reg_lambda", "sigmoid", "lambdarank_truncation_level",
}


def _model(
    *, seed: int, gain: Sequence[float], truncation_level: int | None,
    overrides: Mapping[str, Any] | None = None,
) -> LGBMRanker:
    params: dict[str, Any] = {
        "objective": "lambdarank", "metric": "ndcg", "label_gain": list(gain),
        "n_estimators": 800, "learning_rate": .045, "max_depth": 4, "num_leaves": 15,
        "min_child_samples": 350, "min_split_gain": .001, "colsample_bytree": .80,
        "subsample": .82, "reg_alpha": .02, "reg_lambda": 8.0, "random_state": seed,
        "n_jobs": 2, "verbosity": -1,
    }
    if truncation_level is not None:
        params["lambdarank_truncation_level"] = int(truncation_level)
    if overrides:
        unknown = sorted(set(overrides).difference(_MODEL_OVERRIDE_KEYS))
        if unknown:
            raise ValueError(f"unsupported model override(s): {unknown}")
        params.update({key: value for key, value in overrides.items() if value is not None})
    return LGBMRanker(**params)


GAIN_SCHEDULES: dict[str, tuple[float, ...]] = {
    # Tail convexity is selected using downstream MC1/admission economics,
    # not just in-sample NDCG.
    "small": (0, 1, 2, 3, 5, 7, 10, 14),
    "medium": (0, 1, 2, 4, 7, 11, 16, 24),
    "high": (0, 1, 3, 6, 11, 18, 28, 42),
}


def _gain(labels: np.ndarray, schedule: str) -> list[float]:
    maximum = int(np.nanmax(labels))
    values = GAIN_SCHEDULES.get(schedule)
    if values is None:
        raise ValueError(f"unknown gain schedule {schedule!r}")
    if maximum >= len(values):
        raise AssertionError(f"gain schedule {schedule!r} lacks label {maximum}")
    return [float(value) for value in values[: maximum + 1]]


def _fit_score(
    fold: Fold, arm: Arm, *, seed: int, model_params: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train = _sample_queries(fold.train, MAX_TRAIN_ROWS, seed)
    held = fold.held.copy()
    labels, _residual, info = _target(train, arm, train=True)
    valid = labels >= 0
    train = train.loc[valid].reset_index(drop=True)
    labels = labels[valid]
    if len(train) < 20_000 or len(np.unique(labels)) < 2:
        raise AssertionError(f"{arm.name} {fold.held_month:%Y-%m}: insufficient target support after prequential anchor")
    # Fit the held target anchor solely for later evaluation.  It is never an
    # input to the held score calculation.  A genuinely forward month may be
    # intentionally target-free: score it with the frozen model and defer all
    # diagnostic outcome joins until the labels resolve.
    held_outcome_columns = {
        "policy_path_valid", "policy_net_bps", "policy_exit_reason", "path_arch_atr_fraction",
    }
    held_has_outcomes = held_outcome_columns.issubset(held.columns)
    if held_has_outcomes:
        held_anchor = _fit_anchor(train)
        held_labels, held_residual, _held_info = _target(held, arm, train=False, held_anchor=held_anchor)
    else:
        held_labels = np.full(len(held), -1, dtype=np.int32)
        held_residual = np.full(len(held), np.nan, dtype=np.float32)
    train_x, held_x = _impute(_matrix(train, fold.source_features), _matrix(held, fold.source_features))
    order, ordered_query_ids, groups = _ordered_query(train, _query_ids(train, arm.query))
    x = train_x[order]
    y = labels[order]
    # Last 20% of causal queries forms an early-stop set, disjoint from the
    # fit query groups.  It is not the held month.
    query_ids = ordered_query_ids
    unique = pd.Index(query_ids).drop_duplicates()
    cut = max(1, int(math.floor(.80 * len(unique))))
    fit_queries = set(unique[:cut]); valid_queries = set(unique[cut:])
    fit = np.asarray([value in fit_queries for value in query_ids], dtype=bool)
    tune = np.asarray([value in valid_queries for value in query_ids], dtype=bool)
    if not fit.any() or not tune.any():
        raise AssertionError("insufficient causal query support for early stopping")
    fit_groups = pd.Series(query_ids[fit]).groupby(pd.Series(query_ids[fit]), sort=False).size().astype(int).tolist()
    valid_groups = pd.Series(query_ids[tune]).groupby(pd.Series(query_ids[tune]), sort=False).size().astype(int).tolist()
    overrides = dict(model_params or {})
    # Fractional leaf support is an HPO transport parameter, not a LightGBM
    # option.  Resolve it after the strict causal train/early-stop query split
    # so a finalist preserves its intended support scale in every OOF fold.
    min_fraction = overrides.pop("min_data_fraction", None)
    if min_fraction is not None:
        overrides["min_child_samples"] = max(80, int(round(float(min_fraction) * int(fit.sum()))))
    model = _model(
        seed=seed,
        gain=_gain(labels, arm.gain_schedule),
        truncation_level=arm.truncation_level,
        overrides=overrides,
    )
    model.fit(
        x[fit], y[fit], group=fit_groups,
        eval_set=[(x[tune], y[tune])], eval_group=[valid_groups],
        callbacks=[early_stopping(30, verbose=False)],
    )
    raw = np.asarray(model.predict(held_x), dtype=np.float32)
    if arm.family == "over":
        raw *= -1.0
    score = held.loc[:, list(IDENTITY)].copy()
    score["meta_raw_score"] = raw
    score["meta_rank_ts"] = _rank_desc(pd.concat([held.loc[:, ["candidate_id", "__decision_ts__"]], pd.DataFrame({"score": raw})], axis=1), "score")
    score["base_rank_ts"] = held.inc_base_rank_ts.to_numpy(np.float32)
    score["held_month"] = f"{fold.held_month:%Y-%m}"
    score["arm"] = arm.name
    score["target_family"] = arm.family
    score["query_contract"] = arm.query
    score["gain_schedule"] = arm.gain_schedule
    score["truncation_level"] = arm.truncation_level
    # Target values remain in-memory for metrics only.  They are intentionally
    # excluded from the persisted score receipt.
    metric_cache = {
        "held_labels": held_labels,
        "held_residual_bps": held_residual,
        "held_policy": (
            held.loc[:, ["candidate_id", "policy_path_valid", "policy_net_bps"]].copy()
            if held_has_outcomes else None
        ),
        "held_outcomes_available": bool(held_has_outcomes),
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
        "train_rows": int(len(train)), "train_queries": int(len(groups)),
        "gain_schedule": arm.gain_schedule,
        "truncation_level": arm.truncation_level,
        **info,
    }
    return score, metric_cache


def _bin(values: np.ndarray, bins: int = 10) -> np.ndarray:
    result = np.full(len(values), -1, dtype=np.int16)
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return result
    ranks = pd.Series(values[finite]).rank(method="average", pct=True).to_numpy(float)
    result[finite] = np.minimum(bins - 1, np.floor(ranks * bins)).astype(np.int16)
    return result


def _conditional_mi(meta: np.ndarray, base: np.ndarray, outcome: np.ndarray) -> float:
    m, b, y = _bin(meta), _bin(base), _bin(outcome)
    valid = (m >= 0) & (b >= 0) & (y >= 0)
    if int(valid.sum()) < 100:
        return float("nan")
    total = float(valid.sum()); value = 0.0
    for band in np.unique(b[valid]):
        index = valid & (b == band)
        if int(index.sum()) >= 20:
            value += float(index.sum()) / total * mutual_info_score(m[index], y[index])
    return float(value)


def _substitution(frame: pd.DataFrame, *, score: np.ndarray, policy: np.ndarray, k: int) -> tuple[float, float, float]:
    work = frame.loc[:, ["candidate_id", "__decision_ts__", "inc_base_rank_ts"]].copy()
    work["policy"] = policy; work["meta"] = score
    valid = np.isfinite(work.policy) & np.isfinite(work.meta) & np.isfinite(work.inc_base_rank_ts)
    work = work.loc[valid].copy()
    work["combined"] = .75 * work.inc_base_rank_ts + .25 * work.meta
    def _tail(column: str) -> float:
        ordered = work.sort_values(["__decision_ts__", column, "candidate_id"], ascending=[True, False, True], kind="stable")
        chosen = ordered.groupby("__decision_ts__", sort=False).head(k)
        return float(chosen.groupby("__decision_ts__", sort=False).policy.mean().mean())
    base_value, combined = _tail("inc_base_rank_ts"), _tail("combined")
    return base_value, combined, combined - base_value


def _metrics(fold: Fold, score: pd.DataFrame, cache: dict[str, Any]) -> dict[str, Any]:
    if not bool(cache.get("held_outcomes_available", False)):
        return {
            "arm": str(score.arm.iloc[0]), "family": str(score.target_family.iloc[0]),
            "query": str(score.query_contract.iloc[0]), "held_month": f"{fold.held_month:%Y-%m}",
            "valid_policy_rows": 0, "residual_spearman_ic": float("nan"),
            "conditional_mi_meta_policy_given_base": float("nan"),
            "best_iteration": cache["best_iteration"], "train_rows": cache["train_rows"],
            "train_queries": cache["train_queries"], "held_outcomes_available": False,
        }
    policy = cache["held_policy"].set_index("candidate_id").reindex(score.candidate_id)
    outcomes = pd.to_numeric(policy.policy_net_bps, errors="coerce").to_numpy(float)
    valid = policy.policy_path_valid.fillna(False).to_numpy(bool) & np.isfinite(outcomes)
    meta = score.meta_rank_ts.to_numpy(float); base = score.base_rank_ts.to_numpy(float)
    residual = np.asarray(cache["held_residual_bps"], dtype=float)
    ic = float(spearmanr(meta[valid], residual[valid]).statistic) if int(valid.sum()) >= 20 else float("nan")
    cmi = _conditional_mi(meta[valid], base[valid], outcomes[valid])
    result: dict[str, Any] = {
        "arm": str(score.arm.iloc[0]), "family": str(score.target_family.iloc[0]),
        "query": str(score.query_contract.iloc[0]), "held_month": f"{fold.held_month:%Y-%m}",
        "valid_policy_rows": int(valid.sum()), "residual_spearman_ic": ic,
        "conditional_mi_meta_policy_given_base": cmi, "best_iteration": cache["best_iteration"],
        "train_rows": cache["train_rows"], "train_queries": cache["train_queries"], "held_outcomes_available": True,
    }
    for k in (1, 2):
        base_value, combined, delta = _substitution(fold.held, score=meta, policy=outcomes, k=k)
        result[f"base_top{k}_bps"] = base_value
        result[f"substitution_top{k}_bps"] = combined
        result[f"substitution_delta_top{k}_bps"] = delta
    return result


def _write_scores(out: Path, arm: Arm, fold: Fold, scores: pd.DataFrame) -> Path:
    path = out / "target_free_scores" / arm.name / f"month={fold.held_month:%Y-%m}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(path, index=False, compression="zstd")
    return path


def _custom_arms(path: Path) -> tuple[Arm, ...]:
    """Load a small predeclared target-variant grid without changing code.

    Fine target tuning belongs in an immutable JSON config so every threshold,
    bin count, and signed-state edge is visible in the artifact manifest.
    """
    payload = json.loads(path.read_text())
    raw = payload.get("arms") if isinstance(payload, dict) else payload
    if not isinstance(raw, list) or not raw:
        raise ValueError("--arm-specs-json must contain a non-empty 'arms' list")
    valid_family = {"magnitude", "under", "over", "state"}
    valid_scale = {"bps", "atr", "sqrt_atr"}
    valid_query = {"base_band", "timestamp", "base_band_block28"}
    result: list[Arm] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("arm spec entries must be objects")
        name, family, scale, query = (str(item.get(key, "")) for key in ("name", "family", "scale", "query"))
        if not name or family not in valid_family or scale not in valid_scale or query not in valid_query:
            raise ValueError(f"invalid arm spec: {item}")
        threshold = item.get("threshold")
        if family in {"under", "over"} and threshold is None:
            raise ValueError(f"{name}: binary unexpected-path target requires threshold")
        classes = int(item.get("classes", 7))
        if family == "magnitude" and not 5 <= classes <= 7:
            raise ValueError(f"{name}: magnitude classes must be 5..7")
        raw_edges = item.get("state_edges")
        edges = tuple(float(value) for value in raw_edges) if raw_edges is not None else None
        if family == "state" and edges is not None and not 4 <= len(edges) <= 6:
            raise ValueError(f"{name}: state edges must create 5..7 classes")
        gain_schedule = str(item.get("gain_schedule", "medium"))
        if gain_schedule not in GAIN_SCHEDULES:
            raise ValueError(f"{name}: unsupported gain_schedule {gain_schedule!r}")
        truncation = item.get("truncation_level")
        if truncation is not None and int(truncation) < 2:
            raise ValueError(f"{name}: truncation_level must be >=2 when declared")
        result.append(Arm(
            name, family, scale, query, None if threshold is None else float(threshold),
            classes, edges, gain_schedule, None if truncation is None else int(truncation),
        ))
    names = [arm.name for arm in result]
    if len(set(names)) != len(names):
        raise ValueError("custom arm names must be unique")
    return tuple(result)


def run(
    *,
    source_root: Path,
    policy_path: Path,
    path_root: Path,
    out: Path,
    held_months: Sequence[pd.Timestamp],
    max_arms: int | None = None,
    arm_names: Sequence[str] | None = None,
    arm_specs: Sequence[Arm] | None = None,
    full_feature_roots: Sequence[Path] = (),
    full_feature_contract: Path | None = None,
) -> None:
    if out.exists():
        raise FileExistsError(f"{out}: immutable target/query output already exists")
    if bool(full_feature_roots) != bool(full_feature_contract):
        raise ValueError("full causal feature mode requires both roots and a feature contract")
    full_fields = _load_feature_contract(full_feature_contract) if full_feature_contract else ()
    policy = _read_policy(policy_path)
    folds = _prepare_folds(
        source_root=source_root, policy=policy, path_root=path_root, held_months=held_months,
        full_feature_roots=full_feature_roots, full_feature_fields=full_fields,
    )
    available = {arm.name: arm for arm in arms()}
    if arm_specs is not None:
        all_arms = list(arm_specs)
    elif arm_names:
        unknown = sorted(set(arm_names).difference(available))
        if unknown:
            raise ValueError(f"unknown requested arm(s): {unknown}")
        all_arms = [available[name] for name in arm_names]
    else:
        all_arms = list(arms())[:max_arms] if max_arms else list(arms())
    out.mkdir(parents=True)
    _exclusive_json(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline incumbent meta target/query research; no live, MC1, admission, portfolio, or exchange mutation",
        "incumbent_upstream": "0.50 * efficiency_bps + 0.50 * timing_bps",
        "candidate_source": str(source_root),
        "feature_contract": (
            {"kind": "full_causal_receipt", "fields": list(full_fields),
             "field_sha256": hashlib.sha256("\n".join(full_fields).encode()).hexdigest(),
             "roots": [str(root) for root in full_feature_roots],
             "contract_path": str(full_feature_contract)}
            if full_feature_contract else
            "frozen 120 causal fields plus target-free incumbent E/T score and within-timestamp geometry"
        ),
        "feature_selection_note": (
            "full causal fields are selected only by their explicit immutable receipt"
            if full_feature_contract else
            "the full causal universe is intentionally deferred to the post target/query feature-selection stage"
        ),
        "policy_path": str(policy_path), "path_root": str(path_root),
        "held_months": [f"{month:%Y-%m}" for month in held_months],
        "arms": [dataclasses.asdict(arm) for arm in all_arms],
        "causality": {
            "base": "immutable strict-OOF E/T source; persisted canonical target-free route and score features",
            "labels": "policy/path fields loaded only after target-free source is fixed; valid labels must resolve before the train reserve",
            "train": f"{TRAIN_MONTHS} complete months before a {RESERVE_DAYS}-day reserve",
            "residual_anchor": f"expanding {ANCHOR_BLOCK_DAYS}-day train-only isotonic anchors",
            "lambdarank_query_cap": (
                f"global score-band groups above {MAX_LAMBDARANK_QUERY_ROWS} rows are split into "
                "consecutive time shards without splitting a timestamp"
            ),
            "held": "score is persisted target-free before policy diagnostics are computed",
        },
        "source_hashes": {
            "source": _sha(source_root), "policy": _sha(policy_path), "path": _sha(path_root),
            "full_feature_manifests": {
                str(root): _sha(root / "run_manifest.json") for root in full_feature_roots
            },
        },
    })
    metrics: list[dict[str, Any]] = []
    for arm_index, arm in enumerate(all_arms):
        for fold_index, fold in enumerate(folds):
            scores, cache = _fit_score(fold, arm, seed=SEED + 1000 * arm_index + fold_index)
            receipt = _write_scores(out, arm, fold, scores)
            metrics.append(_metrics(fold, scores, cache))
            _progress(out, event="arm_fold_complete", arm=arm.name, held_month=f"{fold.held_month:%Y-%m}", score_receipt=str(receipt), rows=int(len(scores)))
    report = pd.DataFrame(metrics)
    report.to_parquet(out / "target_query_metrics.parquet", index=False, compression="zstd")
    summary = report.groupby(["arm", "family", "query"], sort=True).agg(
        folds=("held_month", "nunique"),
        residual_ic=("residual_spearman_ic", "mean"),
        cmi=("conditional_mi_meta_policy_given_base", "mean"),
        substitution_delta_top1=("substitution_delta_top1_bps", "mean"),
        substitution_delta_top2=("substitution_delta_top2_bps", "mean"),
        worst_substitution_top2=("substitution_delta_top2_bps", "min"),
    ).reset_index()
    summary["screen_score"] = (
        50.0 * summary.residual_ic.fillna(-1.0)
        + 20.0 * summary.cmi.fillna(-1.0)
        + .20 * summary.substitution_delta_top2.fillna(-1e9)
        + .10 * summary.worst_substitution_top2.fillna(-1e9)
    )
    summary.sort_values(["family", "screen_score", "arm"], ascending=[True, False, True], kind="stable").to_parquet(
        out / "target_query_summary.parquet", index=False, compression="zstd"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--path-root", type=Path, default=DEFAULT_PATH_ROOT)
    parser.add_argument("--held-months", default="2026-01,2026-02,2026-03")
    parser.add_argument("--max-arms", type=int, default=None, help="bounded smoke only; never final selection")
    parser.add_argument(
        "--arms",
        default=None,
        help="comma-separated named arms for an immutable selected-arm history; mutually exclusive with --max-arms",
    )
    parser.add_argument("--arm-specs-json", type=Path, default=None, help="predeclared fine target-variant JSON; mutually exclusive with --arms/--max-arms")
    parser.add_argument(
        "--full-feature-roots", default=None,
        help="comma-separated immutable full causal feature roots; requires --full-feature-contract",
    )
    parser.add_argument(
        "--full-feature-contract", type=Path, default=None,
        help="immutable JSON feature receipt (features or feature_contract list); requires --full-feature-roots",
    )
    args = parser.parse_args()
    selected_mode_count = int(bool(args.arms)) + int(args.max_arms is not None) + int(args.arm_specs_json is not None)
    if selected_mode_count > 1:
        raise ValueError("--arms, --max-arms, and --arm-specs-json are mutually exclusive")
    arm_names = tuple(item.strip() for item in args.arms.split(",") if item.strip()) if args.arms else None
    arm_specs = _custom_arms(args.arm_specs_json) if args.arm_specs_json else None
    full_feature_roots = tuple(
        Path(item.strip()) for item in args.full_feature_roots.split(",") if item.strip()
    ) if args.full_feature_roots else ()
    run(
        source_root=args.source_root, policy_path=args.policy, path_root=args.path_root,
        out=args.out, held_months=_parse_months(args.held_months), max_arms=args.max_arms, arm_names=arm_names, arm_specs=arm_specs,
        full_feature_roots=full_feature_roots, full_feature_contract=args.full_feature_contract,
    )


if __name__ == "__main__":
    main()
