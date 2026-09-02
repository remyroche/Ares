#!/usr/bin/env python3
"""Frozen-M4 short conversion-regime diagnostic funnel.

This is deliberately a *diagnostic*, not a replacement conversion model.  It
consumes the immutable P0/F90 absolute-conversion artifacts and answers why
the frozen M4 ordinal conversion head is weak in 2024 but strong in 2025–26.

The runner implements the predeclared D0--D10 funnel:

* D0: frozen M4 control;
* D1: matched-history and matched-row-support re-fits of the unchanged M4;
* D2: raw-score versus OOF-isotonic calibration audit;
* D3: gross/net and available exact-policy path/exit decomposition;
* D4--D8: P0 score-domain, market-state, interactions, OOD/support, and
  population-composition diagnostics;
* D9: a research-only hierarchical shrinkage *demotion* map, enabled only
  when predeclared markers have the same-direction relation in all eras; and
* D10: a strict matched replay of frozen M4 p80 plus that demotion gate.

No target, P0 ordering, M4 feature contract, consensus, or policy is changed.
Held outcomes never enter a model fit, score normalisation, marker percentile,
or shrinkage estimate for that held month.  The D9/D10 result is explicitly
non-promoting because marker selection is performed on the diagnostic eras.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left, insort
from collections import deque
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_absolute_module():
    path = ROOT / "scripts/run_strict_r3_short_p0_absolute_conversion_funnel.py"
    spec = importlib.util.spec_from_file_location("short_absolute_conversion_funnel", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen M4 contract from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


ABSOLUTE = _load_absolute_module()
SIDE = "short"
M4_NAME = "M4"
P80_SELECTION = "causal_train_top20"
POLICY_COST_BPS = 100.0
MIN_HISTORY_TRAIN_ROWS = 500
HISTORY_MONTHS_DEFAULT = (3, 6, 9, 12)
# The first three are deliberately retained from the proposal.  An hourly
# top-1 population is usually smaller; ``effective_cap`` makes that fact
# auditable rather than silently turning support matching into a no-op.
ROW_CAPS_DEFAULT: tuple[int | None, ...] = (25_000, 50_000, 100_000, None)
PERCENTILE_WINDOW_DAYS = 90
MIN_DIRECTIONAL_SUPPORT = 50

# These are target-free observables already materialised in the frozen P0
# winner population.  No raw K9 memberships are used because their semantic
# identity is not portable across bundles.
DOMAIN_MARKERS = (
    "prequential_base_rank42",
    "prequential_base_anchor_bps",
    "geom_top1_minus_top2",
    "geom_top1_minus_top4",
    "geom_top1_minus_top8",
    "geom_top1_minus_median",
    "geom_score_std",
    "geom_score_mad",
    "geom_score_iqr",
    "geom_score_p90_p50",
    "geom_score_p99_p90",
    "geom_top_tail_slope",
    "geom_score_entropy",
    "geom_fraction_within_1pct_top",
    "geom_fraction_within_5pct_top",
    "geom_count_rank42_ge_p90",
    "geom_count_rank42_ge_p95",
    "geom_count_rank42_ge_p99",
)
MARKET_MARKERS = (
    "market__mkt_ret_24h",
    "market__mkt_ret_eq_4h",
    "market__pct_assets_price_down_oi_up_1h",
    "market__pct_assets_above_intraday_vwap",
    "market__pct_assets_recovering_from_intraday_low",
    "market__pct_assets_up_24h",
    "market__breadth_dispersion",
    "market__xasset_mkt_ob_stress_z_24h",
    "market__mkt_oi_chg_4h",
    "market__mkt_oi_breadth_rising_24h",
    "market__xs_dispersion__funding_per_hour",
    "market__xs_dispersion__vol_z",
    "market__xs_dispersion__amihud_illiq",
    "market__state_spectral_eig_condition",
    "market__state_spectral_eig_gap_1_2",
)
# A narrow, predeclared list used only if its historical diagnostic relation is
# sign-consistent.  D9 has no authority to create an admission.
SHRINKAGE_CANDIDATES = (
    "prequential_base_rank42",
    "geom_top1_minus_top4",
    "market__pct_assets_price_down_oi_up_1h",
    "market__xasset_mkt_ob_stress_z_24h",
    "market__mkt_ret_eq_4h",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _finite(value: pd.Series) -> pd.Series:
    return pd.to_numeric(value, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _valid_policy(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["policy_path_valid"].fillna(False).astype(bool)
        & _finite(frame["p0_canonical_net_bps"]).notna()
        & frame["policy_label_available_at"].notna()
    )


def _era(timestamp: pd.Series) -> pd.Series:
    year = pd.to_datetime(timestamp, utc=True, errors="raise").dt.year
    return year.astype(str)


def _months_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return (end.year - start.year) * 12 + end.month - start.month


def _read_root(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    population_path = root / "short_p0_top1_hourly_population.parquet"
    prediction_path = root / "short_absolute_conversion_oof_predictions.parquet"
    if not population_path.exists() or not prediction_path.exists():
        raise FileNotFoundError(f"not an immutable M4 absolute-conversion artifact: {root}")
    population = pd.read_parquet(population_path)
    prediction = pd.read_parquet(prediction_path)
    prediction = prediction.loc[prediction["arm"].astype(str).eq(M4_NAME)].copy()
    if prediction.empty:
        raise ValueError(f"{root} has no {M4_NAME} predictions")
    for frame in (population, prediction):
        for field in ("__decision_ts__", "policy_label_available_at"):
            if field in frame:
                frame[field] = pd.to_datetime(frame[field], utc=True, errors="raise")
    if not population["side_name"].astype(str).str.lower().eq(SIDE).all():
        raise ValueError(f"non-short population in {root}")
    return population, prediction, {
        str(population_path): _sha256(population_path),
        str(prediction_path): _sha256(prediction_path),
    }


def _load_inputs(roots: Sequence[Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    population_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    for root in roots:
        population, prediction, local_hashes = _read_root(root)
        population_parts.append(population)
        prediction_parts.append(prediction)
        hashes.update(local_hashes)
    population = pd.concat(population_parts, ignore_index=True)
    # The three source runs share warm-up history.  Identical candidate rows
    # may consequently appear more than once; choose the first immutable copy
    # after checking the identity is genuinely stable on the core fields.
    population = population.sort_values(["candidate_id", "__decision_ts__"], kind="stable")
    compare = ["candidate_id", "__decision_ts__", "__symbol__", "prequential_base_score", "p0_canonical_net_bps"]
    dupes = population.loc[population.duplicated("candidate_id", keep=False), compare]
    if not dupes.empty:
        for _, block in dupes.groupby("candidate_id", sort=False):
            if block["__decision_ts__"].nunique() != 1 or block["__symbol__"].nunique() != 1:
                raise ValueError("overlapping absolute artifacts disagree on P0 candidate identity")
    population = population.drop_duplicates("candidate_id", keep="first").copy()
    predictions = pd.concat(prediction_parts, ignore_index=True)
    if predictions.duplicated("candidate_id").any():
        raise ValueError("M4 OOF source artifacts overlap in scored candidate IDs")
    merged = predictions.merge(
        population,
        on=["candidate_id", "__decision_ts__", "__symbol__", "side_name", "policy_path_valid", "policy_label_available_at", "p0_canonical_net_bps", "prequential_base_score", "prequential_base_rank42", "prequential_base_anchor_bps"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_population"),
    )
    if len(merged) != len(predictions) or merged.isna().all(axis=1).any():
        raise ValueError("could not recover frozen M4 OOF rows from target-free population")
    merged["era"] = _era(merged["__decision_ts__"])
    merged["m4_p80_admitted"] = _finite(merged["expected_net_bps"]).ge(_finite(merged["train_p80_expected_bps"]))
    return population.sort_values("__decision_ts__", kind="stable").reset_index(drop=True), merged.sort_values("__decision_ts__", kind="stable").reset_index(drop=True), hashes


def _whole_day_cap(frame: pd.DataFrame, cap: int | None, *, seed: int) -> tuple[pd.DataFrame, int]:
    """Deterministically cap without selecting individual rows by outcome."""
    if cap is None or len(frame) <= cap:
        return frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable").copy(), len(frame)
    local = frame.copy()
    local["__day__"] = pd.to_datetime(local["__decision_ts__"], utc=True).dt.normalize()
    days = np.asarray(sorted(local["__day__"].unique()))
    # Evenly-spaced day selection preserves the evolving training regime while
    # avoiding target-aware individual-row subsampling.
    target_days = max(1, int(np.floor(cap / max(local.groupby("__day__").size().median(), 1.0))))
    if target_days >= len(days):
        return local.drop(columns="__day__"), len(frame)
    phase = int(seed % max(1, len(days)))
    indices = np.linspace(0, len(days) - 1, target_days, dtype=int)
    indices = np.unique((indices + phase) % len(days))
    chosen_days = set(days[np.sort(indices)])
    chosen = local.loc[local["__day__"].isin(chosen_days)].sort_values(["__decision_ts__", "candidate_id"], kind="stable")
    # Whole-day selection can overshoot by one day.  Retaining the last day is
    # deterministic and is preferable to cutting individual observations.
    while len(chosen) > cap and chosen["__day__"].nunique() > 1:
        latest = chosen["__day__"].max()
        chosen = chosen.loc[chosen["__day__"].ne(latest)]
    return chosen.drop(columns="__day__").copy(), len(frame)


def _fit_m4(train: pd.DataFrame, held: pd.DataFrame, *, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    fields = tuple(ABSOLUTE.BASE_INPUT)
    x_train, medians = ABSOLUTE._matrix(train, fields)
    x_held, _ = ABSOLUTE._matrix(held, fields, medians)
    calibrator, rho, _p70_raw, oof_rows = ABSOLUTE._fit_calibrator(train, fields, "ordinal", seed=seed)
    estimator = ABSOLUTE._model("ordinal", seed=seed + 100)
    estimator.fit(x_train, ABSOLUTE._target(train, "ordinal"))
    raw = ABSOLUTE._raw_prediction(
        estimator, x_held, "ordinal", _finite(held["prequential_base_anchor_bps"]).to_numpy(float)
    )
    _, oof_raw = ABSOLUTE._chronological_oof_raw(train, fields, "ordinal", seed=seed)
    result = held.loc[:, ["candidate_id", "__decision_ts__", "p0_canonical_net_bps", "policy_path_valid", "policy_label_available_at"]].copy()
    result["raw_meta_score"] = raw.astype(np.float32)
    result["expected_net_bps"] = calibrator.predict(raw).astype(np.float32)
    result["train_p80_expected_bps"] = float(calibrator.predict(np.asarray([np.quantile(oof_raw, .80)]))[0])
    return result, {"oof_rows": float(oof_rows), "oof_spearman": float(rho), "features": float(len(fields))}


def _selection_metrics(frame: pd.DataFrame) -> dict[str, float]:
    valid = frame.loc[_valid_policy(frame)].copy()
    if valid.empty:
        return {"trades": 0.0, "net_bps_per_trade": np.nan, "total_net_bps": np.nan, "positive_rate": np.nan}
    net = _finite(valid["p0_canonical_net_bps"])
    return {
        "trades": float(len(valid)),
        "net_bps_per_trade": float(net.mean()),
        "total_net_bps": float(net.sum()),
        "positive_rate": float(net.gt(0.0).mean()),
    }


def _support_matched_refits(population: pd.DataFrame, m4: pd.DataFrame, *, histories: Sequence[int], caps: Sequence[int | None], seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """D1: same frozen M4 contract under history/support controls."""
    predictions: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    months = sorted(pd.to_datetime(m4["__decision_ts__"], utc=True).dt.to_period("M").unique())
    valid_population = population.loc[_valid_policy(population)].copy()
    for month_index, period in enumerate(months):
        month = pd.Timestamp(period.start_time, tz="UTC")
        next_month = month + pd.offsets.MonthBegin(1)
        held_ids = m4.loc[m4["__decision_ts__"].ge(month) & m4["__decision_ts__"].lt(next_month), "candidate_id"]
        held = population.loc[population["candidate_id"].isin(held_ids)].copy()
        if held.empty:
            continue
        for history in histories:
            raw_train = valid_population.loc[
                valid_population["__decision_ts__"].lt(month)
                & valid_population["policy_label_available_at"].lt(month)
                & valid_population["__decision_ts__"].ge(month - pd.DateOffset(months=int(history)))
            ].copy()
            # A top-1 hourly population has fewer than 25k rows in the
            # available eras, so the roadmap's 25k/50k/100k/all controls are
            # often mathematically identical.  Fit each effective sample only
            # once, then retain every requested cap label in the output with an
            # explicit equivalence flag.  This is a compute optimisation, not
            # an omitted ablation.
            effective_groups: dict[int | None, list[int | None]] = {}
            for cap in caps:
                effective = None if cap is None or len(raw_train) <= cap else cap
                effective_groups.setdefault(effective, []).append(cap)
            for cap_index, (effective_cap, requested_caps) in enumerate(effective_groups.items()):
                cap = effective_cap
                train, original_rows = _whole_day_cap(raw_train, cap, seed=seed + 1009 * month_index + 53 * history + cap_index)
                arm = f"D1_h{history}m_cap{'all' if cap is None else cap}"
                if len(train) < MIN_HISTORY_TRAIN_ROWS:
                    for requested in requested_caps:
                        audit_rows.append({"arm": f"D1_h{history}m_cap{'all' if requested is None else requested}", "held_month": str(period), "status": "skipped_insufficient_support", "train_rows_before_cap": original_rows, "train_rows_effective": len(train), "held_rows": len(held), "effective_cap": "all" if cap is None else str(cap), "equivalent_requested_caps": ["all" if item is None else str(item) for item in requested_caps]})
                    continue
                try:
                    predicted, details = _fit_m4(train, held, seed=seed + 10_000 * month_index + 97 * history + cap_index)
                except ValueError as error:
                    for requested in requested_caps:
                        audit_rows.append({"arm": f"D1_h{history}m_cap{'all' if requested is None else requested}", "held_month": str(period), "status": "skipped", "reason": str(error), "train_rows_before_cap": original_rows, "train_rows_effective": len(train), "held_rows": len(held), "effective_cap": "all" if cap is None else str(cap), "equivalent_requested_caps": ["all" if item is None else str(item) for item in requested_caps]})
                    continue
                for requested in requested_caps:
                    local = predicted.copy()
                    requested_arm = f"D1_h{history}m_cap{'all' if requested is None else requested}"
                    local["arm"] = requested_arm
                    local["history_months"] = int(history)
                    # Persist cap labels as strings: Arrow requires one stable
                    # scalar type across an immutable multi-arm artifact.
                    local["row_cap"] = "all" if requested is None else str(int(requested))
                    local["effective_cap"] = "all" if cap is None else str(int(cap))
                    local["held_month"] = str(period)
                    local["era"] = str(month.year)
                    local["m4_p80_admitted"] = _finite(local["expected_net_bps"]).ge(_finite(local["train_p80_expected_bps"]))
                    predictions.append(local)
                    audit_rows.append({"arm": requested_arm, "held_month": str(period), "status": "complete", "train_rows_before_cap": original_rows, "train_rows_effective": len(train), "held_rows": len(held), "effective_cap": "all" if cap is None else str(cap), "equivalent_requested_caps": ["all" if item is None else str(item) for item in requested_caps], **details})
    result = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    return result, pd.DataFrame(audit_rows)


def _aggregate_support_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, block in predictions.groupby(["arm", "history_months", "row_cap", "era"], sort=True):
        selected = block.loc[block["m4_p80_admitted"].fillna(False)].copy()
        rows.append({"selection": P80_SELECTION, "months": int(block["held_month"].nunique()), "eligible_hours": int(len(block)), "arm": keys[0], "history_months": keys[1], "row_cap": keys[2], "era": keys[3], **_selection_metrics(selected)})
    return pd.DataFrame(rows).sort_values(["arm", "era"], kind="stable")


def _causal_percentile(series: pd.Series, timestamps: pd.Series, *, days: int = PERCENTILE_WINDOW_DAYS) -> pd.Series:
    """Strictly-prior rolling percentile for a target-free observed marker."""
    values = _finite(series).to_numpy(float)
    times = pd.to_datetime(timestamps, utc=True, errors="raise")
    result = np.full(len(values), np.nan, dtype=float)
    active: list[float] = []
    queue: deque[tuple[pd.Timestamp, float]] = deque()
    for index, (stamp, value) in enumerate(zip(times, values, strict=True)):
        minimum = stamp - pd.Timedelta(days=days)
        while queue and queue[0][0] < minimum:
            _, old = queue.popleft()
            active.pop(bisect_left(active, old))
        if np.isfinite(value) and active:
            result[index] = bisect_left(active, float(value)) / len(active)
        if np.isfinite(value):
            insort(active, float(value))
            queue.append((stamp, float(value)))
    return pd.Series(result, index=series.index, dtype=float)


def _add_marker_percentiles(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values("__decision_ts__", kind="stable").copy()
    for marker in (*DOMAIN_MARKERS, *MARKET_MARKERS):
        if marker in result:
            result[f"marker_pct90d__{marker}"] = _causal_percentile(result[marker], result["__decision_ts__"])
    return result


def _quantile_bin(values: pd.Series, bins: int) -> pd.Series:
    numeric = _finite(values)
    return pd.cut(numeric, bins=np.linspace(0.0, 1.0, bins + 1), labels=False, include_lowest=True).astype("Float64")


def _stratified_metrics(frame: pd.DataFrame, *, marker: str, family: str, bins: int = 5) -> pd.DataFrame:
    column = f"marker_pct90d__{marker}"
    rows: list[dict[str, object]] = []
    if column not in frame:
        return pd.DataFrame(rows)
    working = frame.loc[_valid_policy(frame) & frame["m4_p80_admitted"].fillna(False)].copy()
    working["bucket"] = _quantile_bin(working[column], bins)
    for keys, block in working.dropna(subset=["bucket"]).groupby(["era", "bucket"], sort=True):
        rows.append({"family": family, "marker": marker, "era": keys[0], "bucket": int(keys[1]), "causal_marker_percentile": True, **_selection_metrics(block)})
    return pd.DataFrame(rows)


def _interaction_metrics(frame: pd.DataFrame, *, left: str, right: str, name: str) -> pd.DataFrame:
    lcol, rcol = f"marker_pct90d__{left}", f"marker_pct90d__{right}"
    if lcol not in frame or rcol not in frame:
        return pd.DataFrame()
    working = frame.loc[_valid_policy(frame) & frame["m4_p80_admitted"].fillna(False)].copy()
    working["left_bucket"] = _quantile_bin(working[lcol], 3)
    working["right_bucket"] = _quantile_bin(working[rcol], 3)
    rows: list[dict[str, object]] = []
    for keys, block in working.dropna(subset=["left_bucket", "right_bucket"]).groupby(["era", "left_bucket", "right_bucket"], sort=True):
        rows.append({"interaction": name, "left_marker": left, "right_marker": right, "era": keys[0], "left_bucket": int(keys[1]), "right_bucket": int(keys[2]), **_selection_metrics(block)})
    return pd.DataFrame(rows)


def _calibration_audit(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = frame.loc[_valid_policy(frame)].copy()
    rows: list[dict[str, object]] = []
    thresholds: list[dict[str, object]] = []
    for era, block in working.groupby("era", sort=True):
        for field, label in (("raw_meta_score", "raw_score"), ("expected_net_bps", "isotonic_expected_bps")):
            values = _finite(block[field])
            try:
                bucket = pd.qcut(values, q=10, labels=False, duplicates="drop")
            except ValueError:
                bucket = pd.Series(np.nan, index=block.index)
            local = block.assign(bucket=bucket)
            for key, part in local.dropna(subset=["bucket"]).groupby("bucket", sort=True):
                rows.append({"era": era, "score_surface": label, "bin_kind": "diagnostic_same_era_decile", "bucket": int(key), "score_mean": float(_finite(part[field]).mean()), **_selection_metrics(part)})
        # These are held-period descriptive thresholds only.  They never feed
        # M4 fitting or admission and are labelled accordingly in artifacts.
        values = _finite(block["raw_meta_score"])
        for percentile in (60, 70, 75, 80, 85, 90, 95):
            cutoff = float(values.quantile(percentile / 100.0))
            selected = block.loc[values.ge(cutoff)]
            thresholds.append({"era": era, "percentile": percentile, "cutoff_raw_score": cutoff, "scope": "diagnostic_same_era_percentile_only", **_selection_metrics(selected)})
    calibration_error = _finite(working["p0_canonical_net_bps"]) - _finite(working["expected_net_bps"])
    error = working.assign(calibration_error_bps=calibration_error).groupby("era", as_index=False).agg(
        rows=("candidate_id", "size"), calibration_error_mean_bps=("calibration_error_bps", "mean"),
        calibration_error_median_bps=("calibration_error_bps", "median"), calibration_error_std_bps=("calibration_error_bps", "std"),
        raw_score_net_spearman=("raw_meta_score", lambda x: float(pd.Series(x).corr(pd.Series(working.loc[x.index, "p0_canonical_net_bps"]), method="spearman"))),
        mapped_ev_net_spearman=("expected_net_bps", lambda x: float(pd.Series(x).corr(pd.Series(working.loc[x.index, "p0_canonical_net_bps"]), method="spearman"))),
    )
    return pd.DataFrame(rows), pd.concat([pd.DataFrame(thresholds), error.assign(percentile=np.nan, cutoff_raw_score=np.nan, scope="calibration_error_summary")], ignore_index=True, sort=False)


def _path_decomposition(frame: pd.DataFrame, policy_label_roots: Sequence[Path]) -> tuple[pd.DataFrame, dict[str, object]]:
    """D3 using all information currently materialised by the canonical policy.

    Exact policy net equals realized gross minus the fixed 100-bps cost, so
    gross economics are available even if the optional per-minute path audit is
    not supplied.  Exit-reason/minute fields are joined when existing policy
    label parts are passed.  MFE/MAE are intentionally reported unavailable
    rather than inferred from a terminal policy outcome.
    """
    exit_parts: list[pd.DataFrame] = []
    for root in policy_label_roots:
        for path in sorted(root.glob("parts/month=*/side=short.parquet")):
            available = set(pd.read_parquet(path, columns=None).columns)
            wanted = ["candidate_id", "p0_canonical_exit_reason", "p0_canonical_exit_minute"]
            if set(wanted).issubset(available):
                exit_parts.append(pd.read_parquet(path, columns=wanted))
    exits = pd.concat(exit_parts, ignore_index=True).drop_duplicates("candidate_id", keep="last") if exit_parts else pd.DataFrame(columns=["candidate_id"])
    working = frame.loc[_valid_policy(frame) & frame["m4_p80_admitted"].fillna(False)].copy()
    working = working.merge(exits, on="candidate_id", how="left", validate="one_to_one")
    working["realized_gross_bps"] = _finite(working["p0_canonical_net_bps"]) + POLICY_COST_BPS
    working["mfe_mae_materialized"] = False
    rows: list[dict[str, object]] = []
    grouped = working.groupby("era", sort=True)
    for era, block in grouped:
        exit_reason = block.get("p0_canonical_exit_reason", pd.Series(index=block.index, dtype=object)).astype("string")
        rows.append({
            "era": era, "trades": int(len(block)),
            "gross_bps_per_trade": float(_finite(block["realized_gross_bps"]).mean()),
            "policy_net_bps_per_trade": float(_finite(block["p0_canonical_net_bps"]).mean()),
            "positive_rate": float(_finite(block["p0_canonical_net_bps"]).gt(0.0).mean()),
            "stop_loss_rate": float(exit_reason.eq("stop_loss").mean()) if exit_reason.notna().any() else np.nan,
            "trailing_exit_rate": float(exit_reason.eq("trailing").mean()) if exit_reason.notna().any() else np.nan,
            "timeout_rate": float(exit_reason.eq("timeout_h12").mean()) if exit_reason.notna().any() else np.nan,
            "mean_exit_minute": float(_finite(block.get("p0_canonical_exit_minute", pd.Series(index=block.index, dtype=float))).replace(-1, np.nan).mean()),
            "mfe_mae_available": False,
        })
    receipt = {
        "gross_identity": "p0_canonical_net_bps + 100 bps fixed policy cost",
        "exit_reason_source": bool(exit_parts),
        "mfe_mae": "not materialised in the immutable policy-label parts; intentionally unavailable rather than inferred",
    }
    return pd.DataFrame(rows), receipt


def _fit_ood_for_month(population: pd.DataFrame, m4_month: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    month = _utc(m4_month["__decision_ts__"].min()).normalize().replace(day=1)
    train = population.loc[_valid_policy(population) & population["__decision_ts__"].lt(month) & population["policy_label_available_at"].lt(month)].copy()
    held_ids = set(m4_month["candidate_id"])
    held = population.loc[population["candidate_id"].isin(held_ids)].copy()
    if len(train) < MIN_HISTORY_TRAIN_ROWS or held.empty:
        return pd.DataFrame()
    fields = tuple(ABSOLUTE.BASE_INPUT)
    x_train, medians = ABSOLUTE._matrix(train, fields)
    x_held, _ = ABSOLUTE._matrix(held, fields, medians)
    estimator = ABSOLUTE._model("ordinal", seed=seed)
    estimator.fit(x_train, ABSOLUTE._target(train, "ordinal"))
    # Robust scale followed by shrinkage covariance gives a stable Mahalanobis
    # diagnostic even where the 41-field training covariance is ill-conditioned.
    median = x_train.median(axis=0)
    mad = (x_train - median).abs().median(axis=0).replace(0.0, np.nan).fillna(1.0) * 1.4826
    z_train = ((x_train - median) / mad).clip(-12.0, 12.0)
    z_held = ((x_held - median) / mad).clip(-12.0, 12.0)
    covariance = LedoitWolf().fit(z_train.to_numpy(float))
    mahal = covariance.mahalanobis(z_held.to_numpy(float))
    low = x_train.quantile(.01)
    high = x_train.quantile(.99)
    outside = ((x_held.lt(low)) | (x_held.gt(high))).mean(axis=1).to_numpy(float)
    # Nearest-neighbour geometry is fitted strictly on the training fold.  A
    # train-derived 90th percentile radius defines local support.
    sample = z_train.to_numpy(float)
    if len(sample) > 8_000:
        sample = sample[np.linspace(0, len(sample) - 1, 8_000, dtype=int)]
    nn = NearestNeighbors(n_neighbors=min(2, len(sample))).fit(sample)
    train_dist = nn.kneighbors(sample, return_distance=True)[0]
    train_radius = float(np.quantile(train_dist[:, -1], .90))
    held_dist = nn.kneighbors(z_held.to_numpy(float), n_neighbors=1, return_distance=True)[0][:, 0]
    leaves_train = estimator.predict(x_train, pred_leaf=True)
    leaves_held = estimator.predict(x_held, pred_leaf=True)
    leaf_support = np.empty(len(held), dtype=float)
    supports: list[dict[int, int]] = []
    for tree in range(leaves_train.shape[1]):
        values, counts = np.unique(leaves_train[:, tree], return_counts=True)
        supports.append(dict(zip(values.tolist(), counts.tolist(), strict=True)))
    for row in range(len(held)):
        leaf_support[row] = min(supports[tree].get(int(leaves_held[row, tree]), 0) for tree in range(leaves_held.shape[1]))
    result = held.loc[:, ["candidate_id", "__decision_ts__"]].copy()
    result["ood_mahalanobis_robust_shrunk"] = mahal
    result["ood_fraction_features_outside_train_p01_p99"] = outside
    result["ood_nearest_train_distance"] = held_dist
    result["ood_train_radius_p90"] = train_radius
    result["ood_within_train_support_radius"] = held_dist <= train_radius
    result["ood_min_leaf_support"] = leaf_support
    result["ood_geometry_k9_aggregate_available"] = False
    return result


def _ood_audit(population: pd.DataFrame, m4: pd.DataFrame, *, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    pieces: list[pd.DataFrame] = []
    for index, (_, block) in enumerate(m4.groupby(m4["__decision_ts__"].dt.to_period("M"), sort=True)):
        local = _fit_ood_for_month(population, block, seed=seed + index)
        if not local.empty:
            pieces.append(local)
    values = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if values.empty:
        return values, pd.DataFrame()
    working = m4.merge(values, on=["candidate_id", "__decision_ts__"], how="left", validate="one_to_one")
    working = working.loc[_valid_policy(working) & working["m4_p80_admitted"].fillna(False)].copy()
    rows: list[dict[str, object]] = []
    for marker in ("ood_mahalanobis_robust_shrunk", "ood_fraction_features_outside_train_p01_p99", "ood_nearest_train_distance", "ood_min_leaf_support"):
        try:
            working["bucket"] = pd.qcut(_finite(working[marker]), q=4, labels=False, duplicates="drop")
        except ValueError:
            continue
        for keys, part in working.dropna(subset=["bucket"]).groupby(["era", "bucket"], sort=True):
            rows.append({"marker": marker, "era": keys[0], "bucket": int(keys[1]), **_selection_metrics(part)})
    return values, pd.DataFrame(rows)


def _population_audit(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = frame.copy().sort_values("__decision_ts__", kind="stable")
    first_seen = working.groupby("__symbol__")["__decision_ts__"].transform("min")
    last_seen = working.groupby("__symbol__")["__decision_ts__"].transform("max")
    working["listing_age_days_causal"] = (working["__decision_ts__"] - first_seen).dt.total_seconds() / 86_400.0
    # This explicitly noncausal stable-core membership is a composition
    # diagnostic, never a feature or deployment rule.
    range_start, range_end = working["__decision_ts__"].min(), working["__decision_ts__"].max()
    stable_symbols = set(working.loc[(first_seen <= range_start + pd.Timedelta(days=31)) & (last_seen >= range_end - pd.Timedelta(days=31)), "__symbol__"])
    working["stable_core_diagnostic_only"] = working["__symbol__"].isin(stable_symbols)
    working["liquidity_depth_proxy"] = -_finite(working["ob_trade_size_to_l1_depth_z_24h"])
    selected = working.loc[_valid_policy(working) & working["m4_p80_admitted"].fillna(False)].copy()
    rows: list[dict[str, object]] = []
    for era, block in selected.groupby("era", sort=True):
        for core, part in block.groupby("stable_core_diagnostic_only", sort=True):
            rows.append({"era": era, "view": "stable_core_diagnostic_only", "bucket": str(bool(core)), **_selection_metrics(part)})
        values = _finite(block["liquidity_depth_proxy"])
        try:
            bucket = pd.qcut(values, q=5, labels=False, duplicates="drop")
        except ValueError:
            bucket = pd.Series(np.nan, index=block.index)
        for key, part in block.assign(bucket=bucket).dropna(subset=["bucket"]).groupby("bucket", sort=True):
            rows.append({"era": era, "view": "depth_proxy_quintile_diagnostic", "bucket": str(int(key)), **_selection_metrics(part)})
    by_symbol = selected.groupby(["era", "__symbol__"], as_index=False).agg(
        M4_admitted_hours=("candidate_id", "size"), net_bps_per_trade=("p0_canonical_net_bps", "mean"), total_net_bps=("p0_canonical_net_bps", "sum"),
        listing_age_days_causal=("listing_age_days_causal", "median"), stable_core_diagnostic_only=("stable_core_diagnostic_only", "first"),
    )
    return pd.DataFrame(rows), by_symbol


def _select_stable_markers(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    admitted = frame.loc[_valid_policy(frame) & frame["m4_p80_admitted"].fillna(False)].copy()
    for marker in SHRINKAGE_CANDIDATES:
        column = f"marker_pct90d__{marker}"
        if column not in admitted:
            continue
        directional: list[float] = []
        support_ok = True
        for era, block in admitted.groupby("era", sort=True):
            values = _finite(block[column])
            low = block.loc[values.le(.20)]
            high = block.loc[values.ge(.80)]
            if len(low) < MIN_DIRECTIONAL_SUPPORT or len(high) < MIN_DIRECTIONAL_SUPPORT:
                support_ok = False
                direction = np.nan
            else:
                direction = float(_finite(high["p0_canonical_net_bps"]).mean() - _finite(low["p0_canonical_net_bps"]).mean())
                directional.append(direction)
            rows.append({"marker": marker, "era": era, "low_rows": len(low), "high_rows": len(high), "high_minus_low_net_bps": direction})
        signs = np.sign(np.asarray(directional, dtype=float))
        stable = support_ok and len(signs) == 3 and np.all(signs == signs[0]) and signs[0] != 0
        for row in rows:
            if row["marker"] == marker:
                row["same_direction_all_eras"] = bool(stable)
    return pd.DataFrame(rows)


def _train_bucket_edges(series: pd.Series, bins: int = 5) -> np.ndarray:
    values = _finite(series).dropna().to_numpy(float)
    if len(values) < bins * 4:
        return np.asarray([-np.inf, np.inf])
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _apply_edges(series: pd.Series, edges: np.ndarray) -> pd.Series:
    if len(edges) <= 2:
        return pd.Series(0, index=series.index, dtype=int)
    return pd.cut(_finite(series), bins=edges, labels=False, include_lowest=True).fillna(-1).astype(int)


def _shrunk_mean(values: pd.Series, parent: float, prior: float) -> float:
    numeric = _finite(values).dropna()
    if numeric.empty:
        return float(parent)
    return float((numeric.sum() + prior * parent) / (len(numeric) + prior))


def _hierarchical_map(train: pd.DataFrame, held: pd.DataFrame, *, strength_marker: str, regime_marker: str) -> pd.DataFrame:
    """Global -> M4 bucket -> strength -> regime map with fixed shrinkage."""
    target = "p0_canonical_net_bps"
    score_edges = _train_bucket_edges(train["expected_net_bps"])
    strength_edges = _train_bucket_edges(train[f"marker_pct90d__{strength_marker}"])
    regime_edges = _train_bucket_edges(train[f"marker_pct90d__{regime_marker}"], bins=3)
    local_train = train.copy()
    local_held = held.copy()
    for local in (local_train, local_held):
        local["m4_bucket"] = _apply_edges(local["expected_net_bps"], score_edges)
        local["strength_bucket"] = _apply_edges(local[f"marker_pct90d__{strength_marker}"], strength_edges)
        local["regime_bucket"] = _apply_edges(local[f"marker_pct90d__{regime_marker}"], regime_edges)
    global_mean = float(_finite(local_train[target]).mean())
    m4_map = {key: _shrunk_mean(part[target], global_mean, 40.0) for key, part in local_train.groupby("m4_bucket")}
    strength_map: dict[tuple[int, int], float] = {}
    for key, part in local_train.groupby(["m4_bucket", "strength_bucket"]):
        strength_map[(int(key[0]), int(key[1]))] = _shrunk_mean(part[target], m4_map.get(int(key[0]), global_mean), 25.0)
    regime_map: dict[tuple[int, int, int], float] = {}
    support_map: dict[tuple[int, int, int], int] = {}
    for key, part in local_train.groupby(["m4_bucket", "strength_bucket", "regime_bucket"]):
        k = (int(key[0]), int(key[1]), int(key[2]))
        regime_map[k] = _shrunk_mean(part[target], strength_map.get(k[:2], global_mean), 15.0)
        support_map[k] = int(len(part))
    keys = list(zip(local_held["m4_bucket"], local_held["strength_bucket"], local_held["regime_bucket"], strict=True))
    local_held["d9_expected_net_bps"] = [regime_map.get((int(a), int(b), int(c)), strength_map.get((int(a), int(b)), m4_map.get(int(a), global_mean))) for a, b, c in keys]
    local_held["d9_effective_support"] = [support_map.get((int(a), int(b), int(c)), 0) for a, b, c in keys]
    local_held["d9_gate"] = _finite(local_held["d9_expected_net_bps"]).ge(0.0)
    return local_held


def _d9_d10(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    marker_audit = _select_stable_markers(frame)
    selected_markers = sorted(marker_audit.loc[marker_audit.get("same_direction_all_eras", False).fillna(False), "marker"].unique().tolist()) if not marker_audit.empty else []
    empty_predictions = pd.DataFrame(columns=[
        "candidate_id", "__decision_ts__", "era", "m4_p80_admitted",
        "d9_expected_net_bps", "d9_effective_support", "d9_gate",
        "d9_strength_marker", "d9_regime_marker",
    ])
    empty_metrics = pd.DataFrame(columns=["era", "arm", "trades", "net_bps_per_trade", "total_net_bps", "positive_rate", "retention_vs_D0"])
    if len(selected_markers) < 2:
        return marker_audit, empty_predictions, empty_metrics, {"status": "no_gate", "reason": "fewer_than_two_predeclared_markers_show_same_direction_with_support_across_all_eras", "selected_markers": selected_markers}
    strength, regime = selected_markers[:2]
    predictions: list[pd.DataFrame] = []
    all_months = sorted(frame["__decision_ts__"].dt.to_period("M").unique())
    for period in all_months:
        month = pd.Timestamp(period.start_time, tz="UTC")
        next_month = month + pd.offsets.MonthBegin(1)
        held = frame.loc[frame["__decision_ts__"].ge(month) & frame["__decision_ts__"].lt(next_month)].copy()
        train = frame.loc[_valid_policy(frame) & frame["__decision_ts__"].lt(month) & frame["policy_label_available_at"].lt(month)].copy()
        if len(train) < MIN_HISTORY_TRAIN_ROWS or held.empty:
            continue
        local = _hierarchical_map(train, held, strength_marker=strength, regime_marker=regime)
        local["d9_strength_marker"] = strength
        local["d9_regime_marker"] = regime
        predictions.append(local)
    d9 = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    if d9.empty:
        return marker_audit, empty_predictions, empty_metrics, {"status": "no_gate", "reason": "insufficient_prequential_m4_oof_support", "selected_markers": selected_markers}
    rows: list[dict[str, object]] = []
    for era, block in d9.groupby("era", sort=True):
        baseline = block.loc[block["m4_p80_admitted"].fillna(False)]
        gated = baseline.loc[baseline["d9_gate"].fillna(False)]
        rows.append({"era": era, "arm": "D0_frozen_M4_p80", **_selection_metrics(baseline)})
        rows.append({"era": era, "arm": "D10_M4_p80_plus_D9_demote_only", **_selection_metrics(gated), "retention_vs_D0": float(len(gated) / len(baseline)) if len(baseline) else np.nan})
    return marker_audit, d9, pd.DataFrame(rows), {"status": "research_only_complete", "selected_markers": selected_markers, "gate": "frozen M4 p80 AND hierarchical posterior expected policy net >= 0; never creates admissions"}


def _write_report(out: Path, *, support: pd.DataFrame, calibration_error: pd.DataFrame, d9_metrics: pd.DataFrame, decision: dict[str, object]) -> None:
    lines = [
        "# Short P0/F90 M4 Domain and Era Diagnostic",
        "",
        "Status: **research-only diagnostic; no short admission promotion**.",
        "",
        "M4 remains frozen. This funnel makes no target, P0 ordering, feature, consensus, policy, or live-inference change.",
        "",
        "## What this artifact answers",
        "",
        "- whether 2024 weakness is explained by history/support maturity;",
        "- whether raw ordinal ordering or the OOF isotonic bps map fails to transport;",
        "- whether P0 score-domain, market-state, support/OOD, or population composition creates a same-direction cross-era condition; and",
        "- whether a conservative, shrinkage-only D9 gate has any research justification.",
        "",
        "## D1 support-matched result table",
        "",
    ]
    if support.empty:
        lines.append("No D1 fit had enough strict-prequential support.")
    else:
        lines.extend(["| Arm | Era | Trades | Net bps/trade | Total net bps |", "|---|---:|---:|---:|---:|"])
        for _, row in support.iterrows():
            lines.append(f"| {row['arm']} | {row['era']} | {int(row['trades'])} | {row['net_bps_per_trade']:.2f} | {row['total_net_bps']:.1f} |")
    lines.extend(["", "## Calibration error", ""])
    if not calibration_error.empty:
        summary = calibration_error.loc[calibration_error["scope"].eq("calibration_error_summary")]
        if not summary.empty:
            lines.extend(["| Era | Mean error bps | Median error bps | Raw-score Spearman | Mapped-EV Spearman |", "|---|---:|---:|---:|---:|"])
            for _, row in summary.iterrows():
                lines.append(f"| {row['era']} | {row['calibration_error_mean_bps']:.2f} | {row['calibration_error_median_bps']:.2f} | {row['raw_score_net_spearman']:.3f} | {row['mapped_ev_net_spearman']:.3f} |")
    lines.extend(["", "## D9/D10 status", "", f"`{json.dumps(decision, sort_keys=True)}`", "", "## Interpretation rules", "", "- Same-era percentile and decile tables are descriptive only; they are never live rules.", "- D9 is allowed only to remove an existing frozen-M4 p80 admission. It cannot rerank P0 or manufacture an admission.", "- Any marker selection used these diagnostic eras, so D9/D10 cannot be promoted without a later frozen test.", "- Raw Geometry/K9 memberships are excluded. The source P0 ledger does not expose a stable bundle-invariant K9 support/OOD aggregate, which is recorded as unavailable rather than imputed.", "- MFE/MAE path metrics require a separate exact-minute path materialisation. Terminal policy net/gross and, when supplied, exit reasons are reported without fabricating those quantities."])
    (out / "SHORT_P0_M4_DOMAIN_REGIME_DIAGNOSTIC_REPORT.md").write_text("\n".join(lines) + "\n")


def run(*, absolute_roots: Sequence[Path], out: Path, histories: Sequence[int], caps: Sequence[int | None], policy_label_roots: Sequence[Path], seed: int) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable diagnostic output exists: {out}")
    population, m4, hashes = _load_inputs(absolute_roots)
    for field in (*DOMAIN_MARKERS, *MARKET_MARKERS):
        if field not in m4:
            raise ValueError(f"frozen M4 population misses required diagnostic marker: {field}")
    m4 = _add_marker_percentiles(m4)
    out.mkdir(parents=True)
    # D0: frozen source table and its causal p80 admission are preserved as-is.
    m4.to_parquet(out / "d0_frozen_m4_oof_population.parquet", index=False, compression="zstd")
    # D1
    d1_predictions, d1_audit = _support_matched_refits(population, m4, histories=histories, caps=caps, seed=seed)
    d1_metrics = _aggregate_support_metrics(d1_predictions)
    d1_predictions.to_parquet(out / "d1_support_matched_predictions.parquet", index=False, compression="zstd")
    d1_audit.to_parquet(out / "d1_support_matched_fold_audit.parquet", index=False, compression="zstd")
    d1_metrics.to_parquet(out / "d1_support_matched_metrics.parquet", index=False, compression="zstd")
    # D2 and D3
    calibration_surface, calibration_error = _calibration_audit(m4)
    calibration_surface.to_parquet(out / "d2_calibration_surfaces.parquet", index=False, compression="zstd")
    calibration_error.to_parquet(out / "d2_calibration_error_by_era.parquet", index=False, compression="zstd")
    path_metrics, path_receipt = _path_decomposition(m4, policy_label_roots)
    path_metrics.to_parquet(out / "d3_policy_path_decomposition.parquet", index=False, compression="zstd")
    # D4--D6
    score_domain = pd.concat([_stratified_metrics(m4, marker=marker, family="D4_score_domain") for marker in DOMAIN_MARKERS], ignore_index=True)
    market_state = pd.concat([_stratified_metrics(m4, marker=marker, family="D5_market_state") for marker in MARKET_MARKERS], ignore_index=True)
    interaction_specs = (
        ("market__pct_assets_price_down_oi_up_1h", "market__xasset_mkt_ob_stress_z_24h", "R1_downside_breadth_x_volatility"),
        ("market__mkt_ret_eq_4h", "market__breadth_dispersion", "R2_direction_x_dispersion"),
        ("market__xs_dispersion__funding_per_hour", "market__mkt_ret_eq_4h", "R3_funding_dispersion_x_direction"),
        ("prequential_base_rank42", "market__pct_assets_price_down_oi_up_1h", "R4_P0_strength_x_downside_breadth"),
        ("geom_top1_minus_top4", "market__mkt_ret_eq_4h", "R5_P0_separation_x_direction"),
    )
    interactions = pd.concat([_interaction_metrics(m4, left=a, right=b, name=name) for a, b, name in interaction_specs], ignore_index=True)
    score_domain.to_parquet(out / "d4_p0_score_domain_stratification.parquet", index=False, compression="zstd")
    market_state.to_parquet(out / "d5_market_state_stratification.parquet", index=False, compression="zstd")
    interactions.to_parquet(out / "d6_domain_market_interactions.parquet", index=False, compression="zstd")
    # D7 and D8
    ood_values, ood_metrics = _ood_audit(population, m4, seed=seed)
    ood_values.to_parquet(out / "d7_causal_ood_support_values.parquet", index=False, compression="zstd")
    ood_metrics.to_parquet(out / "d7_causal_ood_support_stratification.parquet", index=False, compression="zstd")
    population_metrics, by_symbol = _population_audit(m4)
    population_metrics.to_parquet(out / "d8_population_composition_metrics.parquet", index=False, compression="zstd")
    by_symbol.to_parquet(out / "d8_symbol_composition_metrics.parquet", index=False, compression="zstd")
    # D9/D10
    marker_audit, d9_predictions, d10_metrics, d9_decision = _d9_d10(m4)
    marker_audit.to_parquet(out / "d9_marker_direction_audit.parquet", index=False, compression="zstd")
    d9_predictions.to_parquet(out / "d9_prequential_shrinkage_predictions.parquet", index=False, compression="zstd")
    d10_metrics.to_parquet(out / "d10_strict_matched_replay_metrics.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_short_p0_m4_domain_regime_diagnostic_v1",
        "status": "complete_research_only",
        "side": SIDE,
        "control": "frozen short P0/F90 M4 ordinal policy-margin conversion; causal train-p80 admission",
        "contracts": {
            "M4": "unchanged six-class ordinal target and chronological-OOF isotonic policy-net mapping",
            "policy": "canonical short parent policy labels; cost 100 bps exactly once",
            "candidate_order": "frozen target-free P0 rank-1 winner per hour; no reranking",
            "D9_authority": "demotion-only; never creates an admission",
            "raw_k9_memberships": "prohibited",
            "marker_percentiles": "strictly-prior 90-day target-free observed distributions",
        },
        "history_months": list(map(int, histories)),
        "row_caps": ["all" if cap is None else int(cap) for cap in caps],
        "source_hashes": hashes,
        "path_decomposition": path_receipt,
        "d9": d9_decision,
        "promotion": "none; diagnostic-era marker selection requires a later frozen OOS test before any gate can advance",
    }
    (out / "m4_domain_diagnosis.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _write_report(out, support=d1_metrics, calibration_error=calibration_error, d9_metrics=d10_metrics, decision=d9_decision)
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return out


def _parse_cap(value: str) -> int | None:
    return None if value.strip().lower() == "all" else int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--absolute-root", type=Path, action="append", required=True, help="immutable M4 absolute-conversion artifact; repeat for eras")
    parser.add_argument("--policy-label-root", type=Path, action="append", default=[], help="optional exact-policy label roots for exit-reason/minute D3 joins")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--history-months", type=int, nargs="+", default=HISTORY_MONTHS_DEFAULT)
    parser.add_argument("--row-cap", type=_parse_cap, nargs="+", default=list(ROW_CAPS_DEFAULT))
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    if any(value <= 0 for value in args.history_months):
        raise ValueError("history months must be positive")
    print(run(absolute_roots=args.absolute_root, out=args.out, histories=args.history_months, caps=args.row_cap, policy_label_roots=args.policy_label_root, seed=args.seed))


if __name__ == "__main__":
    main()
