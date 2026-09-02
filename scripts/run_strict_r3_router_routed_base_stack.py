#!/usr/bin/env python3
"""Strict-OOF route-first enhanced-base stack, research only.

This is intentionally *not* the router-only adapter.  Its causal sequence is

    target-free router score on the full timestamp universe
      -> exact timestamp-local router route
      -> refit E/T or B/E/T base coordinates on prior routed, resolved rows
         only
      -> score routed held rows
      -> refit T6/T9 on those routed base outputs
      -> strict-prequential Current/BCF MC1 maps and constrained portfolio.

The router is never retained as a numerical downstream feature.  Conversely,
the inherited enhanced-base coordinates are never reused: every base output
in this producer is fitted from the routed population.  This makes the
experiment distinct from the invalid "router-only but old base" branches.

The exact historical R3 event sidecar was intentionally removed from the
available homogeneous ledger.  The first coordinate is therefore explicitly
named ``r3_proxy_h12_net100``: a three-state H12 TP6/SL4 proxy (clear / weak /
adverse) whose raw score remains ``P(clear) - .5 P(adverse)``.  It is mapped
to canonical rich-policy net bps only from chronological OOF training
predictions.  Efficiency and timing retain their original supportive-path
targets and the same train-only map convention.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.isotonic import IsotonicRegression


ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_strict_r3_enhanced_base_live_stack_challenger as parent  # noqa: E402
import run_strict_r3_router_downstream as downstream  # noqa: E402


SCHEMA = "strict_r3_router_routed_base_stack_v1"
SEED = 1729
# The selected six-month router has target-free scores from July 2025 onward.
# Starting the routed-base layer in October is therefore the earliest point at
# which its two-month supervised fit plus the 28-day reserve can be assembled
# entirely from strict router outputs.  We deliberately do not manufacture
# May--June router ranks from an incomplete feature history merely to gain
# earlier evaluation months.
ROUTED_BASE_MONTHS = tuple(pd.date_range("2025-10-01", "2026-07-01", freq="MS", tz="UTC"))
# Jan 2026 is the first consensus month with three prior routed-base months
# (Oct--Dec).  MC1 then has Jan--Mar consensus scores before its first Apr
# evaluation, so Apr--Jul is the earliest fully supported dual-map block.
CONSENSUS_SCORE_MONTHS = tuple(pd.date_range("2026-01-01", "2026-07-01", freq="MS", tz="UTC"))
EVALUATION_MONTHS = tuple(pd.date_range("2026-04-01", "2026-07-01", freq="MS", tz="UTC"))
BASE_TRAIN_MONTHS = 2
BASE_RESERVE_DAYS = 28
BASE_TRAIN_CAP = 180_000
# A 40% timestamp-local route has 6.6k eligible rows in the brief December
# 2025 warm-up. This is still sufficient for the fixed models once the OOF
# mapper independently clears its 2,000-row support requirement. Never add
# non-routed rows merely to satisfy a round-number warm-up cutoff.
MIN_BASE_TRAIN_ROWS = 6_000
# OOF maps always require 2,000 strictly prior validation predictions. The
# narrow-route warm-up uses a coarser two-fold chronology where the usual
# three-fold geometry could not meet this same floor; see `_oof_bounds`.
MIN_CALIBRATION_ROWS = 2_000
RETAINED_HEADS = ("cap80_ordinary", "cap120_equal_month")
SOURCE_PREFIX = downstream.SOURCE_PREFIX
# The current selected router is primary-only: its historical auxiliary family
# was rejected.  The three persisted wire aliases are therefore byte-identical
# (primary, primary-only, and full-AE).  Use one canonical coordinate so a
# downstream model cannot mistake duplicate copies for independent evidence.
# Route membership remains this same rank's exact timestamp-local cutoff.
ROUTER_OUTPUT_FIELDS = ("router_primary_rank",)


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _month_end(month: pd.Timestamp) -> pd.Timestamp:
    return month + pd.offsets.MonthBegin(1)


def _sha256(paths: Iterable[Path] | Path) -> str:
    items = [paths] if isinstance(paths, Path) else list(paths)
    digest = hashlib.sha256()
    expanded: list[Path] = []
    for item in items:
        expanded.extend(sorted(item.rglob("*.parquet")) if item.is_dir() else [item])
    for item in sorted(expanded):
        digest.update(str(item).encode())
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _progress(out: Path, **payload: object) -> None:
    with (out / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _source_fields(source_root: Path) -> tuple[str, ...]:
    sample = source_root / "target_free_monthly" / "month=2025-08" / "scores_features.parquet"
    if not sample.exists():
        raise FileNotFoundError(sample)
    return downstream._source_base_fields(sample)


def _months_between(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Timestamp, ...]:
    return tuple(pd.date_range(start.normalize().replace(day=1), (end - pd.Timedelta(nanoseconds=1)).normalize().replace(day=1), freq="MS", tz="UTC"))


def _read_router_month(router_root: Path, token: str) -> pd.DataFrame:
    path = router_root / "target_free_scores" / f"month={token}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing strict-OOF router score {path}")
    result = pd.read_parquet(path, columns=["candidate_id", "__decision_ts__", "side_name", *ROUTER_OUTPUT_FIELDS])
    result["__decision_ts__"] = pd.to_datetime(result["__decision_ts__"], utc=True, errors="raise")
    if (result["candidate_id"].duplicated().any()
            or not result["side_name"].astype(str).str.lower().eq("long").all()
            or not np.isfinite(result.loc[:, list(ROUTER_OUTPUT_FIELDS)].apply(pd.to_numeric, errors="coerce").to_numpy(float)).all()):
        raise AssertionError(f"{path}: router identity / side contract failed")
    return result


def _router_route(frame: pd.DataFrame, fraction: float) -> np.ndarray:
    return parent._exact_timestamp_top_fraction(frame, "router_primary_rank", fraction).to_numpy(bool)


def _read_feature_window(
    source_root: Path,
    router_root: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fields: Sequence[str],
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for month in _months_between(start, end):
        token = f"{month:%Y-%m}"
        source_path = source_root / "target_free_monthly" / f"month={token}" / "scores_features.parquet"
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        source = pd.read_parquet(source_path, columns=["candidate_id", "__decision_ts__", "side_name", *fields])
        source["__decision_ts__"] = pd.to_datetime(source["__decision_ts__"], utc=True, errors="raise")
        source = source.loc[source["__decision_ts__"].ge(start) & source["__decision_ts__"].lt(end)].copy()
        router = _read_router_month(router_root, token)
        router = router.loc[router["__decision_ts__"].ge(start) & router["__decision_ts__"].lt(end)].copy()
        merged = source.merge(router, on=["candidate_id", "__decision_ts__", "side_name"], how="inner", validate="one_to_one")
        if len(merged) != len(source) or len(merged) != len(router):
            raise AssertionError(f"{token}: full-universe router/source identity mismatch {len(source)} / {len(router)} / {len(merged)}")
        pieces.append(merged)
    result = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if result.empty or result["candidate_id"].duplicated().any():
        raise AssertionError("empty or duplicate routed-base source window")
    return result


def _read_supportive_window(labels_root: Path, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    columns = [
        "candidate_id", "__decision_ts__", "side_name", "supportive_label_available_ts",
        "supportive_path_valid", "supportive_target_invalid", "supportive_path_efficiency_h12",
        "supportive_time_to_meaningful_mfe_h12", "h12_tp6_sl4_net_bps",
    ]
    pieces: list[pd.DataFrame] = []
    signal_start = start - pd.Timedelta(hours=1)
    for month in _months_between(signal_start, end):
        path = labels_root / "parts" / f"month={month:%Y-%m}" / "side=long.parquet"
        if not path.exists():
            if month < start.normalize().replace(day=1):
                continue
            raise FileNotFoundError(path)
        part = pd.read_parquet(path, columns=columns)
        part["__decision_ts__"] = pd.to_datetime(part["__decision_ts__"], utc=True, errors="raise")
        part["supportive_label_available_ts"] = pd.to_datetime(part["supportive_label_available_ts"], utc=True, errors="coerce")
        pieces.append(part.loc[part["__decision_ts__"].ge(start) & part["__decision_ts__"].lt(end)].copy())
    result = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=columns)
    if result.empty or result["candidate_id"].duplicated().any():
        raise AssertionError("supportive labels missing or duplicate for routed-base window")
    return result


def _load_policy(policy_path: Path) -> pd.DataFrame:
    result = pd.read_parquet(policy_path, columns=list(downstream.POLICY_COLUMNS))
    result["policy_label_available_ts"] = pd.to_datetime(result["policy_label_available_ts"], utc=True, errors="coerce")
    result["policy_path_valid"] = result["policy_path_valid"].fillna(False).astype(bool)
    result["policy_net_bps"] = pd.to_numeric(result["policy_net_bps"], errors="coerce")
    if result["candidate_id"].duplicated().any():
        raise AssertionError("canonical rich policy contains duplicate candidate IDs")
    return result


def _finite_matrix(frame: pd.DataFrame, fields: Sequence[str], medians: pd.Series | None = None) -> tuple[np.ndarray, pd.Series]:
    numeric = frame.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if medians is None:
        medians = numeric.median(axis=0).fillna(0.0)
    return numeric.fillna(medians).fillna(0.0).to_numpy(np.float32), medians


def _sample_time_balanced(frame: pd.DataFrame, cap: int) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame.copy()
    work = frame.copy()
    work["__month__"] = work["__decision_ts__"].dt.strftime("%Y-%m")
    per_month = max(1, int(math.ceil(cap / work["__month__"].nunique())))
    work["__hash__"] = pd.util.hash_pandas_object(work["candidate_id"], index=False).to_numpy(np.uint64)
    result = (work.sort_values(["__month__", "__hash__", "candidate_id"], kind="stable")
              .groupby("__month__", sort=False, group_keys=False).head(per_month))
    return result.iloc[:cap].drop(columns=["__month__", "__hash__"], errors="ignore").copy()


def _b0_classes(h12_net: np.ndarray) -> np.ndarray:
    """Three-state H12 TP6/SL4 clearance proxy; not the removed exact R3 sidecar."""
    value = np.asarray(h12_net, dtype=float)
    # Clear is a material positive H12 outcome, adverse a material loss;
    # mid-range timeouts/marginal outcomes remain weak.
    return np.where(value >= 100.0, 2, np.where(value <= -100.0, 0, 1)).astype(np.int8)


def _oof_bounds(n_rows: int, *, min_fit_rows: int = 4_000) -> tuple[np.ndarray, int]:
    """Return the finest strict chronological OOF geometry with map support.

    Standard history uses four sequential segments / three expanding folds.
    If the valid OOF portion would fall below the fixed calibration floor,
    three segments / two expanding folds are used instead. This is solely a
    warm-up granularity change: every validation prediction is still produced
    by a model trained strictly before it.
    """
    primary = np.linspace(0, n_rows, 5, dtype=int)
    primary_rows = sum(
        int(primary[fold + 2] - primary[fold + 1])
        for fold in range(3)
        if int(primary[fold + 1]) >= min_fit_rows
    )
    if primary_rows >= MIN_CALIBRATION_ROWS:
        return primary, 3
    fallback = np.linspace(0, n_rows, 4, dtype=int)
    fallback_rows = sum(
        int(fallback[fold + 2] - fallback[fold + 1])
        for fold in range(2)
        if int(fallback[fold + 1]) >= min_fit_rows
    )
    if fallback_rows >= MIN_CALIBRATION_ROWS:
        return fallback, 2
    return primary, 3


def _b0_params(n_jobs: int, seed: int) -> dict[str, object]:
    return {
        "objective": "multiclass", "num_class": 3, "n_estimators": 140,
        "learning_rate": .05, "max_depth": 4, "num_leaves": 31,
        "min_child_samples": 350, "subsample": .80, "subsample_freq": 1,
        "colsample_bytree": .80, "reg_lambda": 8.0, "random_state": seed,
        "n_jobs": n_jobs, "deterministic": True, "force_col_wise": True, "verbosity": -1,
    }


def _direct_params(n_jobs: int, seed: int) -> dict[str, object]:
    return {
        "objective": "huber", "alpha": .90, "n_estimators": 220,
        "learning_rate": .035, "max_depth": 4, "num_leaves": 31,
        "min_child_samples": 180, "subsample": .85, "subsample_freq": 1,
        "colsample_bytree": .80, "reg_lambda": 6.0, "reg_alpha": .05,
        "random_state": seed, "n_jobs": n_jobs, "deterministic": True,
        "force_col_wise": True, "verbosity": -1,
    }


def _policy_preservation_utility(values: pd.Series) -> np.ndarray:
    """Bounded learnable utility used only by the opt-in ETP Base head.

    The downstream policy target is still canonical rich-policy net bps.  This
    construction only gives the weak direct head a stable supervised shape:
    it rewards preserving rows above the +50-bps economic hurdle without
    letting a small number of extreme paths dominate its regression loss.
    """
    net = pd.to_numeric(values, errors="coerce").to_numpy(float)
    return np.sqrt(np.minimum(np.maximum(net - 50.0, 0.0), 300.0) / 300.0)


def _chronological_oof_b0(train: pd.DataFrame, fields: Sequence[str], n_jobs: int) -> tuple[np.ndarray, np.ndarray]:
    ordered = train.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    raw = np.full(len(ordered), np.nan, dtype=float)
    policy = ordered["policy_net_bps"].to_numpy(float)
    bounds, n_folds = _oof_bounds(len(ordered))
    for fold in range(n_folds):
        fit_end, valid_end = int(bounds[fold + 1]), int(bounds[fold + 2])
        if fit_end < 4_000 or valid_end <= fit_end:
            continue
        fit, valid = ordered.iloc[:fit_end], ordered.iloc[fit_end:valid_end]
        x_fit, medians = _finite_matrix(fit, fields)
        x_valid, _ = _finite_matrix(valid, fields, medians)
        model = LGBMClassifier(**_b0_params(n_jobs, SEED + 10 + fold))
        model.fit(x_fit, _b0_classes(fit["h12_tp6_sl4_net_bps"].to_numpy(float)))
        probability = model.predict_proba(x_valid)
        aligned = np.zeros((len(valid), 3), dtype=float)
        for column, klass in enumerate(model.classes_.astype(int)):
            if 0 <= klass < 3:
                aligned[:, klass] = probability[:, column]
        raw[fit_end:valid_end] = aligned[:, 2] - .5 * aligned[:, 0]
    return raw, policy


def _fit_b0(train: pd.DataFrame, held: pd.DataFrame, fields: Sequence[str], n_jobs: int) -> tuple[np.ndarray, dict[str, object]]:
    raw_oof, policy = _chronological_oof_b0(train, fields, n_jobs)
    usable = np.isfinite(raw_oof) & np.isfinite(policy)
    if int(usable.sum()) < MIN_CALIBRATION_ROWS or np.unique(raw_oof[usable]).size < 10:
        raise AssertionError("routed B0 proxy has insufficient chronological OOF map support")
    mapper = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(raw_oof[usable], policy[usable])
    x_train, medians = _finite_matrix(train, fields)
    x_held, _ = _finite_matrix(held, fields, medians)
    model = LGBMClassifier(**_b0_params(n_jobs, SEED + 100))
    model.fit(x_train, _b0_classes(train["h12_tp6_sl4_net_bps"].to_numpy(float)))
    probability = model.predict_proba(x_held)
    aligned = np.zeros((len(held), 3), dtype=float)
    for column, klass in enumerate(model.classes_.astype(int)):
        if 0 <= klass < 3:
            aligned[:, klass] = probability[:, column]
    raw = aligned[:, 2] - .5 * aligned[:, 0]
    return mapper.predict(raw).astype(np.float32), {"oof_map_rows": int(usable.sum()), "raw_min": float(np.min(raw)), "raw_max": float(np.max(raw))}


def _chronological_oof_direct(train: pd.DataFrame, fields: Sequence[str], target: str, direction: float, n_jobs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    ordered = train.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    raw = np.full(len(ordered), np.nan, dtype=float)
    policy = ordered["policy_net_bps"].to_numpy(float)
    bounds, n_folds = _oof_bounds(len(ordered))
    for fold in range(n_folds):
        fit_end, valid_end = int(bounds[fold + 1]), int(bounds[fold + 2])
        fit, valid = ordered.iloc[:fit_end], ordered.iloc[fit_end:valid_end]
        usable = np.isfinite(pd.to_numeric(fit[target], errors="coerce").to_numpy(float))
        if int(usable.sum()) < 4_000 or valid_end <= fit_end:
            continue
        x_fit, medians = _finite_matrix(fit.loc[usable], fields)
        x_valid, _ = _finite_matrix(valid, fields, medians)
        model = LGBMRegressor(**_direct_params(n_jobs, seed + fold))
        model.fit(x_fit, pd.to_numeric(fit.loc[usable, target], errors="coerce").to_numpy(float))
        raw[fit_end:valid_end] = direction * model.predict(x_valid)
    return raw, policy


def _fit_direct(train: pd.DataFrame, held: pd.DataFrame, fields: Sequence[str], target: str, direction: float, n_jobs: int, seed: int) -> tuple[np.ndarray, dict[str, object]]:
    raw_oof, policy = _chronological_oof_direct(train, fields, target, direction, n_jobs, seed)
    usable = np.isfinite(raw_oof) & np.isfinite(policy)
    if int(usable.sum()) < MIN_CALIBRATION_ROWS or np.unique(raw_oof[usable]).size < 10:
        raise AssertionError(f"{target}: insufficient routed chronological OOF map support")
    mapper = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(raw_oof[usable], policy[usable])
    valid = np.isfinite(pd.to_numeric(train[target], errors="coerce").to_numpy(float))
    x_train, medians = _finite_matrix(train.loc[valid], fields)
    x_held, _ = _finite_matrix(held, fields, medians)
    model = LGBMRegressor(**_direct_params(n_jobs, seed + 100))
    model.fit(x_train, pd.to_numeric(train.loc[valid, target], errors="coerce").to_numpy(float))
    raw = direction * model.predict(x_held)
    return mapper.predict(raw).astype(np.float32), {"oof_map_rows": int(usable.sum()), "raw_min": float(np.min(raw)), "raw_max": float(np.max(raw))}


def _joined_window(
    source_root: Path, router_root: Path, labels_root: Path, policy: pd.DataFrame,
    *, start: pd.Timestamp, end: pd.Timestamp, fields: Sequence[str], route_fraction: float,
) -> pd.DataFrame:
    features = _read_feature_window(source_root, router_root, start=start, end=end, fields=fields)
    labels = _read_supportive_window(labels_root, start=start, end=end)
    result = features.merge(labels, on=["candidate_id", "__decision_ts__", "side_name"], how="left", validate="one_to_one")
    result = result.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    if result[["supportive_path_valid", "policy_path_valid"]].isna().any().any():
        raise AssertionError("routed-base label identity coverage failed")
    result["router_routed"] = _router_route(result, route_fraction)
    return result


def _strict_train(
    frame: pd.DataFrame,
    reserve_start: pd.Timestamp,
    *,
    include_b0: bool,
) -> pd.DataFrame:
    valid = (
        frame["router_routed"].fillna(False).astype(bool)
        & frame["supportive_path_valid"].fillna(False).astype(bool)
        & ~frame["supportive_target_invalid"].fillna(True).astype(bool)
        & frame["policy_path_valid"].fillna(False).astype(bool)
        & frame["supportive_label_available_ts"].lt(reserve_start)
        & frame["policy_label_available_ts"].lt(reserve_start)
        & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(frame["supportive_path_efficiency_h12"], errors="coerce"))
        & np.isfinite(pd.to_numeric(frame["supportive_time_to_meaningful_mfe_h12"], errors="coerce"))
    )
    if include_b0:
        valid &= np.isfinite(pd.to_numeric(frame["h12_tp6_sl4_net_bps"], errors="coerce"))
    return _sample_time_balanced(frame.loc[valid].copy(), BASE_TRAIN_CAP)


def _score_routed_base(
    *, router_root: Path, source_root: Path, labels_root: Path, policy: pd.DataFrame,
    out: Path, route_fraction: float, n_jobs: int,
    base_router_inputs: bool = False, persist_router_outputs: bool = False,
    base_components: str = "bet",
    preservation_weight: float = 0.25,
    max_new_months: int | None = None,
) -> tuple[Path, tuple[str, ...], pd.DataFrame, bool]:
    if base_components not in {"bet", "et", "etp"}:
        raise ValueError("base_components must be 'bet', 'et', or 'etp'")
    if base_components == "etp" and not 0.0 < float(preservation_weight) < 1.0:
        raise ValueError("etp preservation_weight must lie strictly in (0, 1)")
    include_b0 = base_components == "bet"
    include_preservation = base_components == "etp"
    fields = _source_fields(source_root)
    model_fields = (*fields, *ROUTER_OUTPUT_FIELDS) if base_router_inputs else fields
    target_root = out / "target_free_monthly"
    target_root.mkdir(parents=True, exist_ok=True)
    audits: list[dict[str, object]] = []
    completed_new = 0
    for month in ROUTED_BASE_MONTHS:
        token = f"{month:%Y-%m}"
        destination = target_root / f"month={token}"
        score_path = destination / "scores_features.parquet"
        if destination.exists():
            if not score_path.exists():
                # The original producer creates the month directory just
                # before fitting.  A supervisor interruption can therefore
                # leave an *empty* directory without any score or audit
                # material.  It contains no data to preserve and is safe to
                # remove; any non-empty partial checkpoint remains fatal.
                if any(destination.iterdir()):
                    raise AssertionError(f"{token}: partial routed-base checkpoint; refusing overwrite")
                destination.rmdir()
            else:
                probe = pd.read_parquet(score_path, columns=["candidate_id", "enhanced_base_routed", *fields])
                if probe.empty or not probe["enhanced_base_routed"].fillna(False).astype(bool).all():
                    raise AssertionError(f"{token}: invalid existing routed-base checkpoint")
                audits.append({
                    "month": token, "held_routed_rows": int(len(probe)),
                    "base_feature_complete_fraction": float(probe.loc[:, list(fields)].notna().all(axis=1).mean()),
                    "all_base_train_rows_router_selected": True,
                    "base_components": base_components,
                    "resumed_existing": True,
                })
                continue
        if max_new_months is not None and completed_new >= max_new_months:
            return target_root, fields, pd.DataFrame(audits), False
        destination.mkdir()
        reserve_start = month - pd.Timedelta(days=BASE_RESERVE_DAYS)
        start = reserve_start - pd.DateOffset(months=BASE_TRAIN_MONTHS)
        train_raw = _joined_window(source_root, router_root, labels_root, policy, start=start, end=reserve_start, fields=fields, route_fraction=route_fraction)
        held_raw = _joined_window(source_root, router_root, labels_root, policy, start=month, end=_month_end(month), fields=fields, route_fraction=route_fraction)
        train = _strict_train(train_raw, reserve_start, include_b0=include_b0)
        if include_preservation:
            train["policy_preservation_utility"] = _policy_preservation_utility(train["policy_net_bps"])
        held = held_raw.loc[held_raw["router_routed"].fillna(False).astype(bool)].copy()
        if len(train) < MIN_BASE_TRAIN_ROWS or len(held) < 1_000:
            raise AssertionError(f"{token}: inadequate route-first base support train={len(train)} held={len(held)}")
        efficiency, efficiency_audit = _fit_direct(train, held, model_fields, "supportive_path_efficiency_h12", 1.0, n_jobs, SEED + 1000)
        timing, timing_audit = _fit_direct(train, held, model_fields, "supportive_time_to_meaningful_mfe_h12", -1.0, n_jobs, SEED + 2000)
        if include_b0:
            b0, b0_audit = _fit_b0(train, held, model_fields, n_jobs)
            enhanced = ((b0 + efficiency + timing) / 3.0).astype(np.float32)
            base_bps_semantics = "strict_r3_proxy_common_bps"
        elif include_preservation:
            # This opt-in head is deliberately policy-aligned and is trained
            # only after the U50 route.  `base_bps` carries it through the
            # existing stable Meta contract, while E/T remain explicit
            # orthogonal coordinates.  No R3 output has ranking authority.
            preservation, preservation_audit = _fit_direct(
                train, held, model_fields, "policy_preservation_utility", 1.0, n_jobs, SEED + 3000,
            )
            b0 = preservation
            enhanced = (
                (1.0 - float(preservation_weight)) * ((efficiency + timing) / 2.0)
                + float(preservation_weight) * preservation
            ).astype(np.float32)
            b0_audit = {
                "kind": "policy_net_preservation_direct",
                "target": "bounded_utility(policy_net_bps; +50 floor, 300-bps cap, sqrt)",
                "weight": float(preservation_weight),
                "map": preservation_audit,
                "r3_input": False,
            }
            base_bps_semantics = "policy_net_direct_preservation_common_bps_no_r3"
        else:
            # Parent consensus helpers historically require ``base_bps``.
            # In this E/T-only contract it is an alias of the two-head common
            # bps coordinate, never an R3 model output or ranking component.
            enhanced = ((efficiency + timing) / 2.0).astype(np.float32)
            b0 = enhanced.copy()
            b0_audit = {
                "kind": "derived_et_anchor_alias",
                "formula": "0.50 * efficiency_bps + 0.50 * timing_bps",
                "r3_input": False,
            }
            base_bps_semantics = "et_common_bps_alias_no_r3"
        result_fields = ["candidate_id", "__decision_ts__", "side_name", *fields]
        if persist_router_outputs:
            result_fields.extend(ROUTER_OUTPUT_FIELDS)
        result = held.loc[:, result_fields].copy()
        result.insert(3, "base_bps", b0)
        result.insert(4, "efficiency_bps", efficiency)
        result.insert(5, "timing_bps", timing)
        result.insert(6, "enhanced_base_bps", enhanced)
        result.insert(7, "enhanced_base_routed", True)
        result.insert(8, "base_rank_ts", parent._rank_pct(result, "enhanced_base_bps").to_numpy(np.float32))
        result.insert(9, "e_minus_t", (efficiency - timing).astype(np.float32))
        result.insert(10, "e_minus_b0", (efficiency - b0).astype(np.float32))
        result.insert(11, "t_minus_b0", (timing - b0).astype(np.float32))
        result.insert(12, "base_component_std", np.nanstd(np.column_stack([b0, efficiency, timing]), axis=1).astype(np.float32))
        ordered = [
            "candidate_id", "__decision_ts__", "side_name", "base_bps", "efficiency_bps", "timing_bps",
            "enhanced_base_bps", "base_rank_ts", "enhanced_base_routed", "e_minus_t", "e_minus_b0",
            "t_minus_b0", "base_component_std", *fields,
        ]
        if persist_router_outputs:
            ordered.extend(ROUTER_OUTPUT_FIELDS)
        result = result.loc[:, ordered]
        result.to_parquet(destination / "scores_features.parquet", index=False, compression="zstd")
        audits.append({
            "month": token, "train_start": str(start), "reserve_start": str(reserve_start),
            "train_rows": int(len(train)), "held_full_universe_rows": int(len(held_raw)), "held_routed_rows": int(len(held)),
            "route_fraction": route_fraction, "base_feature_complete_fraction": float(result.loc[:, list(fields)].notna().all(axis=1).mean()),
            "all_base_train_rows_router_selected": True,
            "base_components": base_components,
            "r3_direct_ranking_authority": bool(include_b0),
            "base_bps_semantics": base_bps_semantics,
            "preservation_weight": float(preservation_weight) if include_preservation else None,
            "base_router_inputs": bool(base_router_inputs), "router_outputs_persisted": bool(persist_router_outputs),
            "model_feature_count": int(len(model_fields)),
            "b0_target": "r3_proxy_h12_net100" if include_b0 else None,
            "preservation_target": "bounded_utility(policy_net_bps; +50 floor, 300-bps cap, sqrt)" if include_preservation else None,
            "efficiency_target": "supportive_path_efficiency_h12",
            "timing_target": "supportive_time_to_meaningful_mfe_h12", "b0_map": b0_audit,
            "efficiency_map": efficiency_audit, "timing_map": timing_audit,
        })
        _progress(out, stage="routed_base_month_done", month=token, train_rows=int(len(train)), held_rows=int(len(held)))
        completed_new += 1
        del train_raw, held_raw, train, held, result
        gc.collect()
    audit = pd.DataFrame(audits)
    if audit["base_feature_complete_fraction"].lt(.90).any():
        raise AssertionError("routed-base held coverage below 90%")
    audit.to_parquet(out / "routed_base_fold_audit.parquet", index=False, compression="zstd")
    return target_root, fields, audit, True


def _score_router_folds_resumable(
    *, target_root: Path, policy: pd.DataFrame, fields: tuple[str, ...], out: Path,
    n_jobs: int, max_new_months: int | None,
) -> tuple[pd.DataFrame, bool]:
    """Fit one immutable consensus month at a time and resume safely.

    A target-free current/BCF score pair is an atomic completed fold.  Resume
    never re-fits or rewrites that pair; the compact audit part is persisted
    immediately so an interruption cannot turn completed score panels into
    an ambiguous partial run.
    """
    original_specs = parent._head_specs
    original_fit = parent._fit_heads
    original_train_months = parent.META_TRAIN_MONTHS
    original_reserve_days = parent.RESERVE_DAYS
    audit_root = out / "consensus_audit_parts"
    audit_root.mkdir(exist_ok=True)
    audits: list[dict[str, object]] = []
    completed_new = 0
    try:
        def two_head_specs(core_fields: tuple[str, ...], feature_contract: str = "current") -> tuple[parent.ConsensusHeadSpec, ...]:
            available = original_specs(core_fields, feature_contract)
            selected = tuple(item for item in available if item.name in RETAINED_HEADS)
            if tuple(item.name for item in selected) != RETAINED_HEADS:
                raise AssertionError(f"frozen T6/T9 slots missing: {[item.name for item in available]}")
            return selected

        def parallel_fit(*args: object, **kwargs: object):
            kwargs["n_jobs"] = int(n_jobs)
            return original_fit(*args, **kwargs)

        parent._head_specs = two_head_specs
        parent._fit_heads = parallel_fit
        parent.META_TRAIN_MONTHS = 4
        parent.RESERVE_DAYS = 28
        for month in CONSENSUS_SCORE_MONTHS:
            token = f"{month:%Y-%m}"
            current = out / "target_free_scores" / "current" / f"month={token}.parquet"
            bcf = out / "target_free_scores" / "bcf" / f"month={token}.parquet"
            audit_path = audit_root / f"month={token}.json"
            if current.exists() or bcf.exists():
                if not (current.exists() and bcf.exists()):
                    raise AssertionError(f"{token}: partial consensus checkpoint; refusing overwrite")
                if not audit_path.exists():
                    # A legacy interruption could happen after the atomic
                    # current/BCF pair was written but before this resumable
                    # audit-part protocol existed.  Reconstruct only the
                    # structural audit from those target-free panels; never
                    # re-fit a score bundle or consult a held outcome.
                    probe = pd.read_parquet(current, columns=["candidate_id", "__decision_ts__", "enhanced_base_routed"])
                    if probe.empty or probe["candidate_id"].duplicated().any() or not probe["enhanced_base_routed"].fillna(False).astype(bool).all():
                        raise AssertionError(f"{token}: invalid legacy consensus score checkpoint")
                    audit = {
                        "month": token, "held_rows": int(len(probe)), "held_routed_rows": int(len(probe)),
                        "head_count": len(RETAINED_HEADS), "selected_heads": list(RETAINED_HEADS),
                        "router_meta_inputs": False, "resumed_legacy_structural_audit": True,
                    }
                    _write_json_exclusive(audit_path, audit)
                audits.append(json.loads(audit_path.read_text()))
                continue
            if max_new_months is not None and completed_new >= max_new_months:
                return pd.DataFrame(audits), False
            _progress(out, stage="consensus_fold_start", month=token)
            audit, _, _ = parent._score_fold(
                target_root, policy, fields,
                parent.POLICY_CONVERSION_LABEL_SPECS["direct_policy_economic_200_0_50_150"],
                "base_consensus_correctness", "none",
                parent.BPS_INTEGRATION_SPECS["rank_75_25"], "current",
                month, out, trust_arm="generic_correctness",
            )
            audit["selected_heads"] = list(RETAINED_HEADS)
            audit["router_meta_inputs"] = False
            _write_json_exclusive(audit_path, audit)
            audits.append(audit)
            completed_new += 1
            _progress(out, stage="consensus_fold_done", month=token, held_rows=int(audit["held_rows"]))
        result = pd.DataFrame(audits)
        result.to_parquet(out / "consensus_fold_audit.parquet", index=False, compression="zstd")
        return result, True
    finally:
        parent._head_specs = original_specs
        parent._fit_heads = original_fit
        parent.META_TRAIN_MONTHS = original_train_months
        parent.RESERVE_DAYS = original_reserve_days


def _score_router_folds_with_router_inputs(
    *, target_root: Path, policy: pd.DataFrame, core_fields: tuple[str, ...],
    out: Path, n_jobs: int,
) -> pd.DataFrame:
    """Fit the frozen T6/T9 heads while treating router ranks as causal inputs.

    The core 120-field contract remains immutable.  This local wrapper adds
    only the three strict-OOF router outputs to the *head* matrices and never
    substitutes a router value for an enhanced-base coordinate.
    """
    original_specs = parent._head_specs
    original_fit = parent._fit_heads
    original_train_months = parent.META_TRAIN_MONTHS
    original_reserve_days = parent.RESERVE_DAYS
    model_fields = (*core_fields, *ROUTER_OUTPUT_FIELDS)
    try:
        def selected_specs(_: tuple[str, ...], feature_contract: str = "current") -> tuple[object, ...]:
            available = original_specs(core_fields, feature_contract)
            selected = tuple(spec for spec in available if spec.name in RETAINED_HEADS)
            if tuple(spec.name for spec in selected) != RETAINED_HEADS:
                raise AssertionError("frozen T6/T9 head slots missing")
            return tuple(replace(spec, fields=tuple(dict.fromkeys((*spec.fields, *ROUTER_OUTPUT_FIELDS)))) for spec in selected)

        def parallel_fit(*args: object, **kwargs: object):
            kwargs["n_jobs"] = int(n_jobs)
            return original_fit(*args, **kwargs)

        parent._head_specs = selected_specs
        parent._fit_heads = parallel_fit
        parent.META_TRAIN_MONTHS = 4
        parent.RESERVE_DAYS = 28
        audits: list[dict[str, object]] = []
        for month in CONSENSUS_SCORE_MONTHS:
            _progress(out, stage="consensus_fold_start", month=f"{month:%Y-%m}", router_meta_inputs=True)
            audit, _, _ = parent._score_fold(
                target_root, policy, model_fields,
                parent.POLICY_CONVERSION_LABEL_SPECS["direct_policy_economic_200_0_50_150"],
                "base_consensus_correctness", "none",
                parent.BPS_INTEGRATION_SPECS["rank_75_25"], "current",
                month, out, trust_arm="generic_correctness",
            )
            audit["selected_heads"] = list(RETAINED_HEADS)
            audit["router_meta_inputs"] = True
            audits.append(audit)
            _progress(out, stage="consensus_fold_done", month=f"{month:%Y-%m}", held_rows=int(audit["held_rows"]), router_meta_inputs=True)
        result = pd.DataFrame(audits)
        result.to_parquet(out / "consensus_fold_audit.parquet", index=False, compression="zstd")
        return result
    finally:
        parent._head_specs = original_specs
        parent._fit_heads = original_fit
        parent.META_TRAIN_MONTHS = original_train_months
        parent.RESERVE_DAYS = original_reserve_days


def _router_outputs_for_scores(router_root: Path, frame: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for month in _months_between(frame["__decision_ts__"].min(), frame["__decision_ts__"].max() + pd.Timedelta(hours=1)):
        piece = _read_router_month(router_root, f"{month:%Y-%m}")
        pieces.append(piece)
    router = pd.concat(pieces, ignore_index=True)
    keys = ["candidate_id", "__decision_ts__", "side_name"]
    merged = frame.merge(router, on=keys, how="left", validate="one_to_one")
    if merged.loc[:, list(ROUTER_OUTPUT_FIELDS)].isna().any().any():
        raise AssertionError("strict-OOF router output is missing from an MC1 score row")
    return merged


def _score_mc1_with_router_inputs(
    *, score_root: Path, router_root: Path, policy: pd.DataFrame, out: Path,
    router_mc1_inputs: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    original_months = parent.SCORE_MONTHS
    original_train_months = parent.MC1_TRAIN_MONTHS
    original_features = parent.MC1_FEATURES
    try:
        parent.SCORE_MONTHS = CONSENSUS_SCORE_MONTHS
        parent.MC1_TRAIN_MONTHS = 3
        if router_mc1_inputs:
            parent.MC1_FEATURES = (*original_features, *ROUTER_OUTPUT_FIELDS)

        def read(family: str) -> pd.DataFrame:
            frame = downstream._read_scores_for_mc1(score_root, family, policy)
            return _router_outputs_for_scores(router_root, frame) if router_mc1_inputs else frame

        _progress(out, stage="mc1_current_start", router_mc1_inputs=router_mc1_inputs)
        current_pred, current_audit = parent._mc1_predictions(read("current"), "current", out)
        _progress(out, stage="mc1_current_done", rows=int(len(current_pred)), router_mc1_inputs=router_mc1_inputs)
        _progress(out, stage="mc1_bcf_start", router_mc1_inputs=router_mc1_inputs)
        bcf_pred, bcf_audit = parent._mc1_predictions(read("bcf"), "bcf", out)
        _progress(out, stage="mc1_bcf_done", rows=int(len(bcf_pred)), router_mc1_inputs=router_mc1_inputs)
    finally:
        parent.SCORE_MONTHS = original_months
        parent.MC1_TRAIN_MONTHS = original_train_months
        parent.MC1_FEATURES = original_features
    start, end = min(EVALUATION_MONTHS), _month_end(max(EVALUATION_MONTHS))
    current_pred = current_pred.loc[current_pred["__decision_ts__"].ge(start) & current_pred["__decision_ts__"].lt(end)].copy()
    bcf_pred = bcf_pred.loc[bcf_pred["__decision_ts__"].ge(start) & bcf_pred["__decision_ts__"].lt(end)].copy()
    combined = parent._combined_challenger(current_pred, bcf_pred)
    pd.concat([current_audit, bcf_audit], ignore_index=True).to_parquet(out / "mc1_fit_audit.parquet", index=False, compression="zstd")
    combined.to_parquet(out / "dual_mc1_predictions.parquet", index=False, compression="zstd")
    return combined, current_audit, bcf_audit


def _score_consensus_and_mc1(
    *, target_root: Path, fields: tuple[str, ...], policy: pd.DataFrame,
    router_root: Path, out: Path, thresholds: Sequence[float], n_jobs: int,
    meta_router_inputs: bool = False, mc1_router_inputs: bool = False,
    reuse_score_root: Path | None = None,
    max_new_consensus_months: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path] | None:
    original_all = downstream.ALL_SCORE_MONTHS
    original_eval = downstream.EVALUATION_MONTHS
    try:
        downstream.ALL_SCORE_MONTHS = CONSENSUS_SCORE_MONTHS
        downstream.EVALUATION_MONTHS = EVALUATION_MONTHS
        if reuse_score_root is None:
            # `_score_router_folds` expects the directory containing
            # `month=*` panels, not the outer run directory.  Passing
            # `target_root.parent` silently made every consensus training
            # window empty even though completed route-first base panels
            # existed under `target_free_monthly`.
            if meta_router_inputs:
                folds = _score_router_folds_with_router_inputs(
                    target_root=target_root, policy=policy, core_fields=fields, out=out, n_jobs=n_jobs,
                )
                complete = True
            else:
                folds, complete = _score_router_folds_resumable(
                    target_root=target_root, policy=policy, fields=fields, out=out,
                    n_jobs=n_jobs, max_new_months=max_new_consensus_months,
                )
            if not complete:
                _progress(out, stage="consensus_checkpoint_pause", completed_folds=int(len(folds)))
                return None
            score_root = out
        else:
            score_root = reuse_score_root.resolve()
            if not (score_root / "consensus_fold_audit.parquet").exists():
                raise FileNotFoundError(f"reused consensus source has no fold audit: {score_root}")
            for family in ("current", "bcf"):
                if not list((score_root / "target_free_scores" / family).glob("month=*.parquet")):
                    raise FileNotFoundError(f"reused consensus source has no target-free {family} scores: {score_root}")
            folds = pd.read_parquet(score_root / "consensus_fold_audit.parquet")
            _progress(out, stage="consensus_scores_reused", score_root=str(score_root), folds=int(len(folds)))
        combined, current_audit, bcf_audit = _score_mc1_with_router_inputs(
            score_root=score_root, router_root=router_root, policy=policy, out=out,
            router_mc1_inputs=mc1_router_inputs,
        )
        pd.concat([current_audit, bcf_audit], ignore_index=True).to_parquet(out / "mc1_fit_audit.parquet", index=False, compression="zstd")
        combined.to_parquet(out / "dual_mc1_predictions.parquet", index=False, compression="zstd")
        downstream._per_head_metrics(score_root, out, policy)
        downstream._timestamp_score_metrics(combined, ("current_final_score", "bcf_final_score"), scope="dual").to_parquet(out / "score_timestamp_metrics.parquet", index=False, compression="zstd")
        results: list[dict[str, object]] = []
        original_threshold = parent.MC1_THRESHOLD_BPS
        try:
            for threshold in thresholds:
                parent.MC1_THRESHOLD_BPS = float(threshold)
                metric = parent._portfolio_metrics(combined, f"routed_base_dual_{int(threshold)}", "2026_marjul", out)
                metric["threshold_bps"] = float(threshold)
                results.append(metric)
        finally:
            parent.MC1_THRESHOLD_BPS = original_threshold
        portfolio = pd.DataFrame(results)
        portfolio.to_parquet(out / "portfolio_metrics.parquet", index=False, compression="zstd")
        return folds, combined, portfolio, score_root
    finally:
        downstream.ALL_SCORE_MONTHS = original_all
        downstream.EVALUATION_MONTHS = original_eval


def _reuse_routed_base_source(
    *, target_root: Path, fields: tuple[str, ...], out: Path,
    require_router_outputs: bool,
) -> pd.DataFrame:
    """Validate and reuse immutable routed-base predictions for a matched downstream arm.

    A meta/MC1 comparison must not silently refit the base.  This helper makes
    the shared, target-free base panels an explicit input and checks that every
    required held month has the exact frozen feature contract, router route,
    and (when requested) target-free router coordinates.
    """
    target_root = target_root.resolve()
    if target_root.name != "target_free_monthly":
        raise ValueError("--reuse-target-free must point to the target_free_monthly directory")
    rows: list[dict[str, object]] = []
    for month in ROUTED_BASE_MONTHS:
        token = f"{month:%Y-%m}"
        path = target_root / f"month={token}" / "scores_features.parquet"
        if not path.exists():
            raise FileNotFoundError(f"reused routed-base panel is missing: {path}")
        names = set(pq.ParquetFile(path).schema_arrow.names)
        required = {
            "candidate_id", "__decision_ts__", "side_name", "base_bps", "efficiency_bps",
            "timing_bps", "enhanced_base_bps", "enhanced_base_routed", *fields,
        }
        if require_router_outputs:
            required.update(ROUTER_OUTPUT_FIELDS)
        missing = sorted(required - names)
        if missing:
            raise AssertionError(f"{path}: reused routed-base source lacks {missing[:8]}")
        probe = pd.read_parquet(
            path,
            columns=["candidate_id", "__decision_ts__", "enhanced_base_routed", *fields,
                     *(ROUTER_OUTPUT_FIELDS if require_router_outputs else ())],
        )
        probe["__decision_ts__"] = pd.to_datetime(probe["__decision_ts__"], utc=True, errors="raise")
        if (
            probe.empty
            or probe["candidate_id"].duplicated().any()
            or not probe["enhanced_base_routed"].fillna(False).astype(bool).all()
            or not probe["__decision_ts__"].ge(month).all()
            or not probe["__decision_ts__"].lt(_month_end(month)).all()
        ):
            raise AssertionError(f"{path}: invalid reused routed-base identity or route contract")
        coverage = float(probe.loc[:, list(fields)].notna().all(axis=1).mean())
        if coverage < .90:
            raise AssertionError(f"{path}: reused frozen-base coverage is {coverage:.3f} < .90")
        if require_router_outputs and not np.isfinite(
            probe.loc[:, list(ROUTER_OUTPUT_FIELDS)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        ).all():
            raise AssertionError(f"{path}: missing/non-finite reused router feature")
        rows.append({"month": token, "rows": int(len(probe)), "base_feature_complete_fraction": coverage})
    audit = pd.DataFrame(rows)
    audit.to_parquet(out / "routed_base_reuse_audit.parquet", index=False, compression="zstd")
    _progress(out, stage="routed_base_reused", target_root=str(target_root), rows=int(audit["rows"].sum()))
    return audit


def _reused_base_lineage(target_root: Path) -> dict[str, object]:
    """Recover and validate the immutable routed-base producer contract.

    A downstream-only arm must describe the actual base that generated its
    reused target-free panels.  The prior generic wording incorrectly called
    every reuse a full-trained base, which is materially false for routed-base
    experiments and would invalidate a handover.
    """
    path = target_root.parent / "run_contract.json"
    if not path.exists():
        raise FileNotFoundError(f"reused routed-base source lacks contract: {path}")
    payload = json.loads(path.read_text())
    base = payload.get("base_contract")
    if not isinstance(base, dict):
        raise AssertionError(f"{path}: missing base_contract")
    population = str(base.get("train_population", ""))
    if "router-selected rows only" not in population:
        raise AssertionError(
            f"{path}: reused source is not a routed-population base: {population!r}"
        )
    return {
        "contract_path": str(path),
        "train_population": population,
        "base_components": str(base.get("base_components", "bet")),
        "r3_direct_ranking_authority": bool(base.get("r3_direct_ranking_authority", True)),
        "base_bps_semantics": str(base.get("base_bps_semantics", "strict_r3_proxy_common_bps")),
        "router_outputs_as_base_inputs": bool(base.get("router_outputs_as_base_inputs", False)),
        "router_outputs_persisted_for_downstream_ablation": bool(
            base.get("router_outputs_persisted_for_downstream_ablation", False)
        ),
        "router_output_fields": list(base.get("router_output_fields", [])),
        "minimum_routed_train_rows": base.get("minimum_routed_train_rows"),
    }


def run(
    *, router_root: Path, source_root: Path, labels_root: Path, policy_path: Path,
    out: Path, route_fraction: float, thresholds: Sequence[float], n_jobs: int,
    base_router_inputs: bool = False, persist_router_outputs: bool = False,
    meta_router_inputs: bool = False, mc1_router_inputs: bool = False,
    reuse_target_free: Path | None = None,
    reuse_score_root: Path | None = None,
    base_components: str = "bet",
    preservation_weight: float = 0.25,
    resume: bool = False,
    max_new_base_months: int | None = None,
    max_new_consensus_months: int | None = None,
) -> None:
    if not 0.0 < route_fraction <= 1.0:
        raise ValueError("route fraction must be in (0, 1]")
    if base_components not in {"bet", "et", "etp"}:
        raise ValueError("base_components must be 'bet', 'et', or 'etp'")
    if base_components == "etp" and not 0.0 < float(preservation_weight) < 1.0:
        raise ValueError("etp preservation_weight must lie strictly in (0, 1)")
    if reuse_target_free is not None and (base_router_inputs or persist_router_outputs):
        raise ValueError("reusing routed-base panels cannot also refit or persist base router inputs")
    if reuse_score_root is not None and reuse_target_free is None:
        raise ValueError("reusing consensus scores requires the matching --reuse-target-free source for audit")
    if reuse_score_root is not None and meta_router_inputs:
        raise ValueError("reusing consensus scores cannot also refit meta router inputs")
    if meta_router_inputs and not (persist_router_outputs or reuse_target_free is not None):
        raise ValueError("meta-router-inputs requires persisted router outputs in the target-free routed-base source")
    reused_base = _reused_base_lineage(reuse_target_free.resolve()) if reuse_target_free is not None else None
    components = (
        {
            "base": "r3_proxy_h12_net100: P(clear)-0.5*P(adverse), mapped to rich-policy net through chronological OOF",
            "efficiency": "supportive_path_efficiency_h12, mapped to rich-policy net through chronological OOF",
            "timing": "negative supportive_time_to_meaningful_mfe_h12, mapped to rich-policy net through chronological OOF",
        }
        if base_components == "bet" else
        {
            "efficiency": "supportive_path_efficiency_h12, mapped to rich-policy net through chronological OOF",
            "timing": "negative supportive_time_to_meaningful_mfe_h12, mapped to rich-policy net through chronological OOF",
        }
    )
    if base_components == "etp":
        components["preservation"] = "bounded +50-bps policy utility, mapped back to common policy bps through chronological OOF; no R3 input"
    base_contract = {
        "train_population": (
            "router-selected rows only; labels resolved before same-model 28-day reserve"
            if reused_base is None else reused_base["train_population"]
        ),
        "components": components,
        "blend": (
            "0.50 efficiency + 0.50 timing" if base_components == "et"
            else (f"{1.0 - float(preservation_weight):.2f} * (0.50 efficiency + 0.50 timing) + {float(preservation_weight):.2f} * policy-preservation" if base_components == "etp" else "equal common-bps mean")
        ),
        "base_components": base_components if reused_base is None else reused_base["base_components"],
        "r3_direct_ranking_authority": (
            base_components == "bet" if reused_base is None else reused_base["r3_direct_ranking_authority"]
        ),
        "base_bps_semantics": (
            "et_common_bps_alias_no_r3" if base_components == "et"
            else ("policy_net_direct_preservation_common_bps_no_r3" if base_components == "etp" else "strict_r3_proxy_common_bps")
        ) if reused_base is None else reused_base["base_bps_semantics"],
        "preservation_weight": float(preservation_weight) if base_components == "etp" and reused_base is None else None,
        "train_months_max": BASE_TRAIN_MONTHS,
        "reserve_days": BASE_RESERVE_DAYS,
        "minimum_routed_train_rows": (
            MIN_BASE_TRAIN_ROWS if reused_base is None else reused_base["minimum_routed_train_rows"]
        ),
        "router_outputs_as_base_inputs": (
            bool(base_router_inputs) if reused_base is None else reused_base["router_outputs_as_base_inputs"]
        ),
        "router_outputs_persisted_for_downstream_ablation": (
            bool(persist_router_outputs) if reused_base is None else reused_base["router_outputs_persisted_for_downstream_ablation"]
        ),
        "router_output_fields": (
            list(ROUTER_OUTPUT_FIELDS) if (base_router_inputs or persist_router_outputs)
            else ([] if reused_base is None else reused_base["router_output_fields"])
        ),
        "reused_base_lineage": reused_base,
    }
    contract = {
        "schema": SCHEMA, "scope": "offline research only; no live mutation",
        "router_root": str(router_root), "source_root": str(source_root), "labels_root": str(labels_root),
        "reuse_target_free": str(reuse_target_free) if reuse_target_free else None,
        "reuse_score_root": str(reuse_score_root) if reuse_score_root else None,
        "policy_path": str(policy_path), "route_fraction": route_fraction,
        "router_contract": (
            "strict-OOF full-universe router; router_primary_rank forms exact timestamp-local route membership; "
            "optional router output columns are target-free model inputs only"
        ),
        "base_contract": base_contract,
        "consensus": {"heads": list(RETAINED_HEADS), "train_months": 3, "reserve_days": 28, "router_outputs_as_meta_inputs": bool(meta_router_inputs)},
        "mc1": {"families": ["current", "bcf"], "train_months_max": 3, "thresholds_bps": list(map(float, thresholds)), "router_outputs_as_mc1_inputs": bool(mc1_router_inputs)},
        "base_score_months": [f"{m:%Y-%m}" for m in ROUTED_BASE_MONTHS],
        "consensus_score_months": [f"{m:%Y-%m}" for m in CONSENSUS_SCORE_MONTHS],
        "evaluation_months": [f"{m:%Y-%m}" for m in EVALUATION_MONTHS],
        "source_hashes": {"router": _sha256(router_root), "source": _sha256(source_root), "labels": _sha256(labels_root), "policy": _sha256(policy_path)},
    }
    if out.exists():
        if not resume:
            raise FileExistsError(out)
        existing = out / "run_contract.json"
        if not existing.exists() or json.loads(existing.read_text()) != contract:
            raise AssertionError("refusing resume with a non-identical immutable routed-stack contract")
    else:
        out.mkdir(parents=True)
    if not (out / "run_contract.json").exists():
        _write_json_exclusive(out / "run_contract.json", contract)
    policy = _load_policy(policy_path)
    fields = _source_fields(source_root)
    if reuse_target_free is None:
        _progress(out, stage="routed_base_start")
        target_root, fields, base_audit, base_complete = _score_routed_base(
            router_root=router_root, source_root=source_root, labels_root=labels_root, policy=policy,
            out=out, route_fraction=route_fraction, n_jobs=n_jobs,
            base_router_inputs=base_router_inputs, persist_router_outputs=persist_router_outputs,
            base_components=base_components, max_new_months=max_new_base_months,
            preservation_weight=preservation_weight,
        )
        if not base_complete:
            _progress(out, stage="routed_base_checkpoint_pause", completed_months=int(len(base_audit)))
            return
    else:
        target_root = reuse_target_free.resolve()
        base_audit = _reuse_routed_base_source(
            target_root=target_root, fields=fields, out=out, require_router_outputs=meta_router_inputs,
        )
    _progress(out, stage="consensus_mc1_start")
    scored = _score_consensus_and_mc1(
        target_root=target_root, fields=fields, policy=policy, router_root=router_root,
        out=out, thresholds=thresholds, n_jobs=n_jobs,
        meta_router_inputs=meta_router_inputs, mc1_router_inputs=mc1_router_inputs,
        reuse_score_root=reuse_score_root, max_new_consensus_months=max_new_consensus_months,
    )
    if scored is None:
        return
    folds, combined, portfolio, score_root = scored
    # These checkpoints make the terminal report phase observable and ensure
    # an interrupted report cannot be mistaken for incomplete model scoring.
    _progress(out, stage="score_and_mc1_complete", rows=int(len(combined)))
    audit = downstream._audit(out, target_root, score_root, folds, combined)
    audit.update({
        # `reuse_target_free` means that this *downstream invocation* reuses an
        # already verified routed-base ledger.  It must not be interpreted as
        # saying that the base lineage was full-universe trained: reuse is the
        # normal way to hold the base fixed while ablating MC1 inputs.
        "current_invocation_refit_base": reuse_target_free is None,
        "base_lineage_route_first": bool(
            base_contract.get("train_population") == "router-selected rows only; labels resolved before same-model 28-day reserve"
        ),
        "base_source_reused": str(reuse_target_free) if reuse_target_free else None,
        "router_has_numeric_downstream_authority": bool(base_router_inputs or meta_router_inputs or mc1_router_inputs),
        "router_outputs_as_base_inputs": (
            bool(base_router_inputs) if reused_base is None else bool(reused_base["router_outputs_as_base_inputs"])
        ),
        "router_outputs_as_meta_inputs": bool(meta_router_inputs),
        "router_outputs_as_mc1_inputs": bool(mc1_router_inputs),
        # A re-used target-free base panel has already completed its own
        # row-level audit.  Preserve that fact in this downstream receipt
        # rather than emitting a misleading null merely because this stage
        # did not refit the base models itself.
        "all_base_training_rows_router_selected": (
            bool(base_audit["all_base_train_rows_router_selected"].all())
            if "all_base_train_rows_router_selected" in base_audit
            else bool(
                reused_base is not None
                and reused_base["train_population"]
                == "router-selected rows only; labels resolved before same-model 28-day reserve"
            )
        ),
        "base_score_months": [f"{m:%Y-%m}" for m in ROUTED_BASE_MONTHS],
        "evaluation_months": [f"{m:%Y-%m}" for m in EVALUATION_MONTHS],
    })
    (out / "correctness_report.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    _progress(out, stage="correctness_audit_complete")
    manifest = {
        **contract, "status": "complete", "rows": {
            "routed_base": int(base_audit["held_routed_rows"].sum()) if "held_routed_rows" in base_audit else int(base_audit["rows"].sum()),
            "dual_mc1_evaluation": int(len(combined)),
        },
        "portfolio": portfolio.to_dict(orient="records"), "correctness": audit,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    _progress(out, stage="complete")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--route-fraction", type=float, default=.30)
    parser.add_argument("--thresholds", default="30,50")
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--base-router-inputs", action="store_true", help="feed the three strict-OOF router outputs to the routed base models")
    parser.add_argument("--persist-router-outputs", action="store_true", help="retain strict-OOF router outputs in target-free base panels for downstream ablations")
    parser.add_argument("--meta-router-inputs", action="store_true", help="feed strict-OOF router outputs to routed T6/T9 head matrices")
    parser.add_argument("--mc1-router-inputs", action="store_true", help="feed strict-OOF router outputs to both prequential MC1 maps")
    parser.add_argument(
        "--base-components", choices=("bet", "et", "etp"), default="bet",
        help="base coordinates after router gating; 'et' excludes R3; 'etp' adds a direct policy-preservation coordinate",
    )
    parser.add_argument("--preservation-weight", type=float, default=.25, help="ETP weight on the direct policy-preservation coordinate")
    parser.add_argument("--reuse-target-free", type=Path, default=None, help="immutable target_free_monthly source to reuse for a matched meta/MC1-only arm")
    parser.add_argument("--reuse-score-root", type=Path, default=None, help="completed target-free consensus score root to reuse for a matched MC1-only arm")
    parser.add_argument("--resume", action="store_true", help="resume only from complete immutable base/consensus fold checkpoints")
    parser.add_argument("--max-new-base-months", type=int, default=None, help="checkpoint after this many newly fitted base months")
    parser.add_argument("--max-new-consensus-months", type=int, default=None, help="checkpoint after this many newly fitted consensus months")
    args = parser.parse_args()
    thresholds = tuple(float(item) for item in args.thresholds.split(",") if item)
    if not thresholds or args.n_jobs < 1:
        parser.error("at least one threshold and positive n-jobs are required")
    run(router_root=args.router_root, source_root=args.source_root, labels_root=args.labels_root,
        policy_path=args.policy_path, out=args.out, route_fraction=args.route_fraction,
        thresholds=thresholds, n_jobs=args.n_jobs,
        base_router_inputs=args.base_router_inputs,
        persist_router_outputs=args.persist_router_outputs,
        meta_router_inputs=args.meta_router_inputs,
        mc1_router_inputs=args.mc1_router_inputs,
        reuse_target_free=args.reuse_target_free,
        reuse_score_root=args.reuse_score_root,
        base_components=args.base_components,
        preservation_weight=args.preservation_weight,
        resume=args.resume,
        max_new_base_months=args.max_new_base_months,
        max_new_consensus_months=args.max_new_consensus_months)


if __name__ == "__main__":
    main()
