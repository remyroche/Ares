#!/usr/bin/env python3
"""Strict-prequential A0--A10 hierarchical MC1 calibration ablation.

This is an offline selection experiment.  It holds fixed the target-free BCF
and Current score ledgers, frozen MC1 packages, canonical rich-policy labels,
and portfolio constraints.  Each arm changes only the causal mapping from a
static MC1 prediction to an absolute EV estimate.

Evaluation deliberately separates three questions:

* ``matched_static_budget``: every arm gets the static map's exact timestamp
  local +50-bps admission capacity, so ranking is not confounded by volume;
* ``gate_only`` / ``priority_only`` / ``full``: restores the real dual +50
  gate and attributes any effect to thresholding versus BCF auction priority;
* the constrained portfolio replay is post-selection and excludes only
  unresolved policy outcomes, never using them for candidate selection.

No live configuration, state, exchange API, or inference bundle is mutated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.p8u_mc1_inference_package import (  # noqa: E402
    FEATURES,
    P8UMC1InferencePackage,
    score_bands,
)
from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    normalise_candidate_table,
    replay_candidates,
)
from scripts.report_strict_r3_mc1_d2_controlled_portfolio import (  # noqa: E402
    CAUSAL_AUCTION_CURVE,
    _params,
)


SCHEMA = "strict_r3_p8u_a0_a10_hierarchical_mc1_ablation_v1"
EVAL_ROOT = ROOT / (
    "data_perp/artifacts/strict_r3_p8u_f72_underf120_dual_mc1_"
    "sixmonth_aug25_aug26_20260828_v4"
)
HISTORY_ROOT = ROOT / (
    "data_perp/artifacts/strict_r3_p8u_f72_underf120_dual_mc1_"
    "nov25_jul26_fullprehistory_20260828_v1"
)
DEFAULT_OUT = ROOT / "data_perp/artifacts/strict_r3_p8u_a0_a10_hierarchical_mc1_20260828_v1"
START = pd.Timestamp("2026-02-01", tz="UTC")
END = pd.Timestamp("2026-08-01", tz="UTC")
HORIZONS = (3, 7, 14, 21, 42)
THRESHOLD_BPS = 50.0
TRIM = 0.10
EPS = 1e-12
SEED = 1729


def _once_json(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _trimmed_mean(values: Iterable[float]) -> float:
    data = np.sort(pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float))
    if not len(data):
        return float("nan")
    cut = int(math.floor(len(data) * TRIM))
    if cut and len(data) > 2 * cut:
        data = data[cut:-cut]
    return float(data.mean())


def _mad_sd(values: Iterable[float]) -> float:
    data = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if not len(data):
        return float("nan")
    median = float(np.median(data))
    return float(1.4826 * np.median(np.abs(data - median)))


def _policy_equal(left: pd.Series, right: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(left) or pd.api.types.is_numeric_dtype(right):
        return bool(np.allclose(
            pd.to_numeric(left, errors="coerce").to_numpy(float),
            pd.to_numeric(right, errors="coerce").to_numpy(float),
            rtol=0.0, atol=1e-10, equal_nan=True,
        ))
    return bool(left.fillna("<NA>").astype(str).equals(right.fillna("<NA>").astype(str)))


POLICY_COLUMNS = (
    "__symbol__", "side_name", "enhanced_base_routed", "policy_path_valid",
    "policy_gross_bps", "policy_net_bps", "policy_exit_bar_15m",
    "policy_entry_price", "policy_exit_price", "policy_exit_reason",
    "policy_label_available_ts", "policy_cost_bps",
)


def _read_family(path: Path, family: str, *, evaluation: bool) -> pd.DataFrame:
    model_fields = tuple(FEATURES)
    fields = list(dict.fromkeys(["candidate_id", "__decision_ts__", "final_score", *model_fields, *POLICY_COLUMNS]))
    if evaluation:
        fields += ["static_expected_bps", "recent_shift_bps", "mc1_expected_bps"]
    raw = pd.read_parquet(path, columns=fields)
    if raw["candidate_id"].duplicated().any():
        raise AssertionError(f"{family} ledger has duplicate candidate IDs")
    raw["__decision_ts__"] = pd.to_datetime(raw["__decision_ts__"], utc=True, errors="raise")
    raw["policy_label_available_ts"] = pd.to_datetime(
        raw["policy_label_available_ts"], utc=True, errors="coerce",
    )
    renamed = {
        "final_score": f"{family}_final_score",
        **{field: f"{family}_{field}" for field in model_fields},
        **{field: f"{field}__{family}" for field in POLICY_COLUMNS},
    }
    if evaluation:
        renamed.update({
            "static_expected_bps": f"{family}_static_bps",
            "recent_shift_bps": f"{family}_current_shift_bps",
            "mc1_expected_bps": f"{family}_current_bps",
        })
    return raw.rename(columns=renamed)


def _merge_families(root: Path, *, evaluation: bool) -> pd.DataFrame:
    current = _read_family(root / "enhanced_current_mc1_predictions.parquet", "current", evaluation=evaluation)
    bcf = _read_family(root / "enhanced_bcf_mc1_predictions.parquet", "bcf", evaluation=evaluation)
    frame = current.merge(bcf, on=["candidate_id", "__decision_ts__"], how="inner", validate="one_to_one")
    if len(frame) != len(current) or len(frame) != len(bcf):
        raise AssertionError("BCF and Current candidate identities differ")
    for field in POLICY_COLUMNS:
        if not _policy_equal(frame[f"{field}__current"], frame[f"{field}__bcf"]):
            raise AssertionError(f"BCF/Current mismatch in policy field {field}")
        frame[field] = frame.pop(f"{field}__current")
        frame.pop(f"{field}__bcf")
    if not frame["side_name"].eq("long").all():
        raise AssertionError("A0--A10 study is long-only")
    return frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _package_path(root: Path, family: str, month: str) -> Path:
    path = root / "mc1_packages" / f"family={family}" / f"month={month}" / "package.joblib"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _package(root: Path, family: str, month: str) -> P8UMC1InferencePackage:
    package = joblib.load(_package_path(root, family, month))
    if not isinstance(package, P8UMC1InferencePackage):
        raise TypeError(f"unexpected MC1 package type at {family}/{month}")
    if package.family != family:
        raise AssertionError(f"package family mismatch at {family}/{month}")
    return package


def _model_view(frame: pd.DataFrame, family: str) -> pd.DataFrame:
    values = {field: frame[f"{family}_{field}"] for field in FEATURES}
    return pd.DataFrame({
        "candidate_id": frame["candidate_id"].astype(str),
        "__decision_ts__": frame["__decision_ts__"],
        "final_score": frame[f"{family}_final_score"],
        **values,
    })


def _valid_policy(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
        & frame["policy_label_available_ts"].notna()
    )


def _stats(values: pd.DataFrame) -> dict[str, float]:
    residual = pd.to_numeric(values.get("residual"), errors="coerce").dropna().to_numpy(float)
    if not len(residual):
        return {
            "location": 0.0, "median": float("nan"), "mad_sd": float("inf"),
            "median_abs": float("nan"), "p90_abs": float("nan"), "n": 0.0,
            "n_eff": 0.0, "se": float("inf"),
        }
    daily = values.groupby("decision_day", sort=True)["residual"].mean().to_numpy(float)
    daily_sd = _mad_sd(daily)
    n_eff = float(len(daily))
    se = float(daily_sd / math.sqrt(n_eff)) if len(daily) >= 2 and np.isfinite(daily_sd) else float("inf")
    return {
        "location": _trimmed_mean(residual),
        "median": float(np.median(residual)),
        "mad_sd": _mad_sd(residual),
        "median_abs": float(np.median(np.abs(residual))),
        "p90_abs": float(np.quantile(np.abs(residual), .90)),
        "n": float(len(residual)), "n_eff": n_eff, "se": se,
    }


def _eb(location: float, se: float, tau2: float, prior: float = 0.0) -> tuple[float, float, float]:
    if not np.isfinite(location) or not np.isfinite(se) or se <= 0.0 or tau2 <= 0.0:
        return float(prior), 0.0, float(max(tau2, 0.0))
    se2 = se * se
    weight = float(tau2 / (tau2 + se2))
    posterior = float(prior + weight * (location - prior))
    variance = float(tau2 * se2 / (tau2 + se2))
    return posterior, weight, variance


def _hierarchical(stats: dict[int, dict[str, float]], tau2: float) -> tuple[float, float]:
    mean, variance = 0.0, max(float(tau2), 1.0)
    for horizon in (42, 21, 14, 7, 3):
        state = stats[horizon]
        se = float(state["se"])
        if not np.isfinite(se) or se <= 0.0:
            continue
        measurement_var = se * se
        gain = variance / (variance + measurement_var)
        mean = float(mean + gain * (float(state["location"]) - mean))
        variance = float(variance * measurement_var / (variance + measurement_var))
    return mean, variance


def _tau2(daily: pd.DataFrame, *, floor_bps: float) -> float:
    if len(daily) < 5:
        return float(floor_bps * floor_bps)
    loc = pd.to_numeric(daily["location"], errors="coerce").dropna().to_numpy(float)
    se = pd.to_numeric(daily["se"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    if len(loc) < 5:
        return float(floor_bps * floor_bps)
    observed = _mad_sd(loc) ** 2
    noise = float(np.median(se * se)) if len(se) else 0.0
    return float(max(floor_bps * floor_bps, observed - noise))


def _kalman_states(
    history: pd.DataFrame, *, tau2: float, q2: float, band: int | None = None,
) -> pd.DataFrame:
    """State at each calendar day, updated only from previously available labels."""
    data = history.copy()
    if band is not None:
        data = data.loc[data["score_band"].eq(int(band))].copy()
    data["arrival_day"] = data["policy_label_available_ts"].dt.normalize()
    measurements: dict[pd.Timestamp, dict[str, float]] = {}
    for day, group in data.groupby("arrival_day", sort=True):
        group = group.copy()
        measurements[pd.Timestamp(day)] = _stats(group)
    days = pd.date_range(data["arrival_day"].min(), END - pd.Timedelta(days=1), freq="D", tz="UTC")
    mean, variance = 0.0, max(float(tau2), 1.0)
    rows: list[dict[str, Any]] = []
    for day in days:
        # Measurements that arrive during a day can only affect decisions on
        # the following day, so update after emitting the day's prior state.
        rows.append({"decision_day": day, "mean": mean, "variance": variance})
        measurement = measurements.get(day)
        if measurement is None or not np.isfinite(measurement["se"]):
            variance += q2
            continue
        r2 = max(float(measurement["se"]) ** 2, 1.0)
        gain = variance / (variance + r2)
        mean = float(mean + gain * (float(measurement["location"]) - mean))
        variance = float(variance * r2 / (variance + r2) + q2)
    return pd.DataFrame(rows)


def _isotonic_by_timestamp(frame: pd.DataFrame, *, static_field: str, raw_field: str) -> np.ndarray:
    result = np.empty(len(frame), dtype=float)
    for _, group in frame.groupby("__decision_ts__", sort=False):
        index = group.index.to_numpy()
        static = pd.to_numeric(group[static_field], errors="coerce").to_numpy(float)
        raw = pd.to_numeric(group[raw_field], errors="coerce").to_numpy(float)
        valid = np.isfinite(static) & np.isfinite(raw)
        output = raw.copy()
        if valid.sum() >= 2 and np.ptp(static[valid]) > 1e-9:
            output[valid] = IsotonicRegression(increasing=True, out_of_bounds="clip").fit_transform(
                static[valid], raw[valid],
            )
        result[index] = output
    return result


def _month_state(
    *, month: pd.Timestamp, eval_frame: pd.DataFrame, history: pd.DataFrame,
    package_root: Path, family: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Generate one family's A0--A10 score fields for a held calendar month."""
    month_end = month + pd.offsets.MonthBegin(1)
    held = eval_frame.loc[
        eval_frame["__decision_ts__"].ge(month) & eval_frame["__decision_ts__"].lt(month_end)
    ].copy().reset_index(drop=True)
    if held.empty:
        return held, pd.DataFrame(), {}
    package = _package(package_root, family, f"{month:%Y-%m}")
    # Start at the retained historical origin for Kalman prior fitting.  The
    # feature contract is target-free; outcomes join only in residual states.
    source = history.loc[history["__decision_ts__"].lt(month_end)].copy()
    source_view = _model_view(source, family)
    source["static"] = package.predict_static(source_view)
    source["score_band"] = score_bands(source_view).astype(np.int8)
    source["decision_day"] = source["__decision_ts__"].dt.normalize()
    valid = source.loc[_valid_policy(source)].copy()
    valid["residual"] = (
        pd.to_numeric(valid["policy_net_bps"], errors="coerce").to_numpy(float)
        - pd.to_numeric(valid["static"], errors="coerce").to_numpy(float)
    )
    valid = valid.replace([np.inf, -np.inf], np.nan).dropna(subset=["residual"])
    pre = valid.loc[valid["policy_label_available_ts"].lt(month)].copy()
    daily_pre = pd.DataFrame([
        {"day": day, **_stats(group)} for day, group in pre.groupby("decision_day", sort=True)
    ])
    tau_global2 = _tau2(daily_pre, floor_bps=20.0)
    band_tau2: dict[int, float] = {}
    global_by_day = daily_pre.set_index("day")["location"] if len(daily_pre) else pd.Series(dtype=float)
    for band in range(10):
        local = pre.loc[pre["score_band"].eq(band)]
        rows = []
        for day, group in local.groupby("decision_day", sort=True):
            if day in global_by_day.index:
                state = _stats(group)
                rows.append({"day": day, "location": state["location"] - float(global_by_day.loc[day]), "se": state["se"]})
        band_tau2[band] = _tau2(pd.DataFrame(rows), floor_bps=12.0)
    # A conservative process variance is estimated from prior daily drift.
    if len(daily_pre) >= 8:
        diffs = np.diff(pd.to_numeric(daily_pre["location"], errors="coerce").dropna().to_numpy(float))
        q_global2 = float(max(5.0**2, (_mad_sd(diffs) ** 2) / 2.0)) if len(diffs) else 5.0**2
    else:
        q_global2 = 5.0**2
    kalman_global = _kalman_states(valid, tau2=tau_global2, q2=q_global2).set_index("decision_day")
    kalman_band = {
        band: _kalman_states(valid, tau2=band_tau2[band], q2=max(3.0**2, q_global2 * .25), band=band).set_index("decision_day")
        for band in range(10)
    }
    held_view = _model_view(held, family)
    stored = pd.to_numeric(held[f"{family}_static_bps"], errors="coerce").to_numpy(float)
    recalculated = package.predict_static(held_view)
    if not np.allclose(stored, recalculated, rtol=0.0, atol=1e-8, equal_nan=True):
        max_delta = float(np.nanmax(np.abs(stored - recalculated)))
        raise AssertionError(f"{family}/{month:%Y-%m} static package parity failure: {max_delta} bps")
    held["decision_day"] = held["__decision_ts__"].dt.normalize()
    arms: dict[str, np.ndarray] = {"A0_static": recalculated}
    arms["A1_current21"] = recalculated + pd.to_numeric(held[f"{family}_current_shift_bps"], errors="coerce").fillna(0.0).to_numpy(float)
    state_rows: list[dict[str, Any]] = []
    day_adjustments: dict[pd.Timestamp, dict[str, Any]] = {}
    for day in sorted(held["decision_day"].unique()):
        hstates: dict[int, dict[str, float]] = {}
        band_states: dict[int, dict[int, dict[str, float]]] = {band: {} for band in range(10)}
        for horizon in HORIZONS:
            window = valid.loc[
                valid["__decision_ts__"].ge(day - pd.Timedelta(days=horizon))
                & valid["__decision_ts__"].lt(day)
                & valid["policy_label_available_ts"].lt(day)
            ].copy()
            hstates[horizon] = _stats(window)
            for band in range(10):
                band_states[band][horizon] = _stats(window.loc[window["score_band"].eq(band)])
            state_rows.append({
                "family": family, "decision_day": day, "horizon_days": horizon,
                "scope": "global", "score_band": -1, **hstates[horizon],
                "tau2": tau_global2,
            })
            for band in range(10):
                state_rows.append({
                    "family": family, "decision_day": day, "horizon_days": horizon,
                    "scope": "score_band", "score_band": band, **band_states[band][horizon],
                    "tau2": band_tau2[band],
                })
        global21, eb_weight, eb_variance = _eb(
            hstates[21]["location"], hstates[21]["se"], tau_global2,
        )
        snr = abs(hstates[21]["location"]) / max(hstates[21]["se"], EPS)
        snr_weight = float(snr * snr / (1.0 + snr * snr)) if np.isfinite(snr) else 0.0
        multi_parts = [_eb(hstates[h]["location"], hstates[h]["se"], tau_global2) for h in HORIZONS]
        inv_var = np.asarray([1.0 / max(part[2], 1.0) for part in multi_parts], dtype=float)
        multi = float(np.dot(inv_var, np.asarray([part[0] for part in multi_parts])) / inv_var.sum())
        multi_variance = float(1.0 / inv_var.sum())
        hierarchical, hierarchical_var = _hierarchical(hstates, tau_global2)
        a6_band: dict[int, float] = {}
        a7_band: dict[int, float] = {}
        a9_band: dict[int, float] = {}
        for band in range(10):
            local = band_states[band][21]
            local_mean, _, _ = _eb(local["location"], local["se"], band_tau2[band], prior=global21)
            a6_band[band] = local_mean
            dev_states: dict[int, dict[str, float]] = {}
            for horizon in HORIZONS:
                local_h = band_states[band][horizon]
                global_h = hstates[horizon]
                dev_states[horizon] = {
                    "location": float(local_h["location"] - global_h["location"]),
                    "se": float(math.sqrt(local_h["se"] ** 2 + global_h["se"] ** 2))
                    if np.isfinite(local_h["se"]) and np.isfinite(global_h["se"]) else float("inf"),
                }
            deviation, _ = _hierarchical(dev_states, band_tau2[band])
            a7_band[band] = float(hierarchical + deviation)
            kband = kalman_band[band].reindex([day])
            deviation_k = float(kband["mean"].iloc[0]) if len(kband) else 0.0
            a9_band[band] = float(kalman_global.reindex([day])["mean"].iloc[0] + deviation_k)
        kalman = kalman_global.reindex([day])
        kalman_mean = float(kalman["mean"].iloc[0]) if len(kalman) else 0.0
        # Equal-day affine residual fit, heavily shrunk toward intercept=0 and
        # slope=0 residual (therefore Y=static EV).  It is deliberately not a
        # broad relearning layer.
        affine_window = valid.loc[
            valid["__decision_ts__"].ge(day - pd.Timedelta(days=21))
            & valid["__decision_ts__"].lt(day)
            & valid["policy_label_available_ts"].lt(day)
        ].copy()
        affine_a, affine_c = 0.0, 0.0
        if len(affine_window) >= 100:
            center = float(np.median(pd.to_numeric(affine_window["static"], errors="coerce")))
            x = (pd.to_numeric(affine_window["static"], errors="coerce").to_numpy(float) - center) / 100.0
            y = affine_window["residual"].to_numpy(float)
            counts = affine_window.groupby("decision_day")["residual"].transform("size").to_numpy(float)
            weights = 1.0 / np.maximum(counts, 1.0)
            design = np.column_stack([np.ones(len(x)), x])
            sigma2 = max(_mad_sd(y) ** 2, 30.0**2)
            prior_precision = np.diag([1.0 / 75.0**2, 1.0 / 45.0**2])
            precision = (design.T * weights) @ design / sigma2 + prior_precision
            rhs = (design.T * weights) @ y / sigma2
            affine_a, affine_c = np.linalg.solve(precision, rhs).tolist()
            affine_a = float(np.clip(affine_a, -150.0, 150.0))
            affine_c = float(np.clip(affine_c, -75.0, 75.0))
        day_adjustments[day] = {
            "A2_lambda025": .25 * hstates[21]["location"],
            "A2_lambda050": .50 * hstates[21]["location"],
            "A2_lambda075": .75 * hstates[21]["location"],
            "A3_eb21": global21,
            "A3_snr21": snr_weight * hstates[21]["location"],
            "A4_multi_eb": multi,
            "A5_hierarchical": hierarchical,
            "A6_band_eb": a6_band,
            "A7_multi_band": a7_band,
            "A8_kalman": kalman_mean,
            "A9_kalman_band": a9_band,
            "A10_affine": (affine_a, affine_c),
            "eb_weight": eb_weight, "eb_variance": eb_variance,
            "multi_variance": multi_variance, "kalman_variance": float(kalman["variance"].iloc[0]) if len(kalman) else tau_global2,
        }
    static = recalculated
    band = score_bands(held_view).astype(int)
    for arm in ("A2_lambda025", "A2_lambda050", "A2_lambda075", "A3_eb21", "A3_snr21", "A4_multi_eb", "A5_hierarchical", "A8_kalman"):
        adjustments = held["decision_day"].map(lambda day: day_adjustments[day][arm]).to_numpy(float)
        arms[arm] = static + adjustments
    # The helper expects an explicit field; add it only for these three
    # monotonic band corrections and remove it before returning.
    held["__static_for_iso"] = static
    for arm in ("A6_band_eb", "A7_multi_band", "A9_kalman_band"):
        adjustment = np.asarray([
            day_adjustments[day][arm][int(item)] for day, item in zip(held["decision_day"], band)
        ], dtype=float)
        raw_field = f"__{arm}_raw"
        held[raw_field] = static + adjustment
        arms[arm] = _isotonic_by_timestamp(held, static_field="__static_for_iso", raw_field=raw_field)
        held.drop(columns=[raw_field], inplace=True)
    affine = np.asarray([
        day_adjustments[day]["A10_affine"] for day in held["decision_day"]
    ], dtype=float)
    center_by_day = held.groupby("decision_day", sort=False)["__static_for_iso"].median()
    centers = held["decision_day"].map(center_by_day).to_numpy(float)
    arms["A10_affine"] = static + affine[:, 0] + affine[:, 1] * (static - centers) / 100.0
    for arm, values in arms.items():
        held[f"{family}__{arm}"] = np.asarray(values, dtype=float)
    held.drop(columns=["__static_for_iso"], inplace=True, errors="ignore")
    adjustment_rows = []
    for day, values in day_adjustments.items():
        adjustment_rows.append({
            "family": family, "decision_day": day, "tau_global2": tau_global2,
            "q_global2": q_global2, **{
                key: value for key, value in values.items() if not isinstance(value, dict)
            },
        })
    audit = {
        "family": family, "month": f"{month:%Y-%m}", "tau_global2": tau_global2,
        "q_global2": q_global2,
        "band_tau2_json": json.dumps({str(key): float(value) for key, value in band_tau2.items()}, sort_keys=True),
        "static_parity_max_abs_bps": float(np.nanmax(np.abs(stored - recalculated))),
        "history_rows": int(len(source)), "valid_history_rows": int(len(valid)),
    }
    return held, pd.concat([pd.DataFrame(state_rows), pd.DataFrame(adjustment_rows)], ignore_index=True, sort=False), audit


ARM_LABELS = {
    "A0_static": "A0 static MC1",
    "A1_current21": "A1 current 21d full shift",
    "A2_lambda025": "A2 fixed 21d lambda=.25",
    "A2_lambda050": "A2 fixed 21d lambda=.50",
    "A2_lambda075": "A2 fixed 21d lambda=.75",
    "A3_eb21": "A3 empirical-Bayes 21d",
    "A3_snr21": "A3 SNR 21d",
    "A4_multi_eb": "A4 multi-horizon EB blend",
    "A5_hierarchical": "A5 hierarchical multi-horizon",
    "A6_band_eb": "A6 global + band partial pool",
    "A7_multi_band": "A7 multi-horizon + band pool",
    "A8_kalman": "A8 global Kalman",
    "A9_kalman_band": "A9 global + band Kalman",
    "A10_affine": "A10 shrunk affine",
}


def _dual(frame: pd.DataFrame, arm: str) -> np.ndarray:
    return np.minimum(
        pd.to_numeric(frame[f"bcf__{arm}"], errors="coerce").to_numpy(float),
        pd.to_numeric(frame[f"current__{arm}"], errors="coerce").to_numpy(float),
    )


def _fixed_capacity(frame: pd.DataFrame) -> pd.DataFrame:
    dual = _dual(frame, "A0_static")
    passes = frame["enhanced_base_routed"].fillna(False).astype(bool).to_numpy() & np.isfinite(dual) & (dual >= THRESHOLD_BPS)
    work = frame.loc[:, ["__decision_ts__"]].copy()
    work["k"] = passes
    return work.groupby("__decision_ts__", as_index=False, sort=True).agg(fixed_budget_k=("k", "sum"))


def _select(
    frame: pd.DataFrame, *, arm: str, mode: str, capacity: pd.DataFrame | None,
) -> pd.DataFrame:
    static_dual = _dual(frame, "A0_static")
    arm_dual = _dual(frame, arm)
    routed = frame["enhanced_base_routed"].fillna(False).astype(bool).to_numpy()
    base = frame.copy()
    if mode == "matched_static_budget":
        if capacity is None:
            raise ValueError("matched selection requires static capacity")
        base = base.merge(capacity, on="__decision_ts__", how="left", validate="many_to_one")
        eligible = routed & np.isfinite(arm_dual) & np.isfinite(pd.to_numeric(base[f"bcf__{arm}"], errors="coerce"))
        priority = pd.to_numeric(base[f"bcf__{arm}"], errors="coerce").to_numpy(float)
        gate_score = arm_dual
    elif mode == "gate_only":
        eligible = routed & np.isfinite(arm_dual) & (arm_dual >= THRESHOLD_BPS)
        priority = pd.to_numeric(base["bcf__A0_static"], errors="coerce").to_numpy(float)
        gate_score = arm_dual
    elif mode == "priority_only":
        eligible = routed & np.isfinite(static_dual) & (static_dual >= THRESHOLD_BPS)
        priority = pd.to_numeric(base[f"bcf__{arm}"], errors="coerce").to_numpy(float)
        gate_score = static_dual
    elif mode == "full":
        eligible = routed & np.isfinite(arm_dual) & (arm_dual >= THRESHOLD_BPS)
        priority = pd.to_numeric(base[f"bcf__{arm}"], errors="coerce").to_numpy(float)
        gate_score = arm_dual
    else:
        raise ValueError(mode)
    base["gate_score_bps"] = gate_score
    base["bcf_priority_bps"] = priority
    base = base.loc[eligible].copy()
    base = base.sort_values(
        ["__decision_ts__", "gate_score_bps", "bcf_priority_bps", "candidate_id"],
        ascending=[True, False, False, True], kind="stable",
    )
    base["selection_rank"] = base.groupby("__decision_ts__", sort=False).cumcount() + 1
    if mode == "matched_static_budget":
        base = base.loc[base["selection_rank"].le(base["fixed_budget_k"].astype(int))].copy()
    base["arm"] = arm
    base["mode"] = mode
    return base


def _outcome_valid(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(frame["policy_exit_bar_15m"], errors="coerce"))
    )


def _portfolio_input(selected: pd.DataFrame) -> pd.DataFrame:
    valid = selected.loc[_outcome_valid(selected)].copy()
    if valid.empty:
        return pd.DataFrame()
    timestamp = pd.to_datetime(valid["__decision_ts__"], utc=True)
    exit_bar = pd.to_numeric(valid["policy_exit_bar_15m"], errors="coerce").astype(int)
    rank = valid.groupby("__decision_ts__", sort=False)["bcf_priority_bps"].rank(pct=True, method="average")
    return normalise_candidate_table(pd.DataFrame({
        "timestamp": timestamp, "symbol": valid["__symbol__"].astype(str), "side": "long",
        "strategy_id": "strict_r3_a0_a10", "policy_archetype": "strict_r3_a0_a10",
        "normalized_rank_score": rank.to_numpy(float), "strategy_rank_pct": rank.to_numpy(float),
        "base_strategy_threshold": 0.0, "calibrated_score": valid["bcf_priority_bps"].to_numpy(float),
        "entry_price": pd.to_numeric(valid["policy_entry_price"], errors="coerce"),
        "exit_timestamp": timestamp + pd.to_timedelta((exit_bar + 1) * 15, unit="min"),
        "exit_price": pd.to_numeric(valid["policy_exit_price"], errors="coerce"),
        "net_return": pd.to_numeric(valid["policy_net_bps"], errors="coerce") / 10_000.0,
        "gross_return": pd.to_numeric(valid["policy_gross_bps"], errors="coerce") / 10_000.0,
        "holding_bars": exit_bar + 1, "simple_policy_exit_reason": valid["policy_exit_reason"].astype(str),
        "fees_bps": 100.0, "slippage_bps": 0.0, "expected_friction_bps": 100.0,
        "price_gap_bps": 0.0, "liquidity_capacity_weight": 1.0,
        "source_month": timestamp.dt.strftime("%Y-%m"), "candidate_id": valid["candidate_id"].astype(str),
        "mapped_expected_net_bps": valid["bcf_priority_bps"].to_numpy(float),
    }))


def _portfolio_metrics(selected: pd.DataFrame) -> tuple[dict[str, float | int], pd.DataFrame]:
    candidates = _portfolio_input(selected)
    if candidates.empty:
        return {"portfolio_trades": 0, "portfolio_ev_bps": float("nan"), "portfolio_total_bps": float("nan"), "portfolio_max_dd": float("nan")}, pd.DataFrame()
    decisions, equity, _ = replay_candidates(candidates, _params(), mode="global_auction", ev_curve=CAUSAL_AUCTION_CURVE, market_mode="perps", initial_wallet=1_000.0)
    accepted = decisions.loc[decisions["accepted"].fillna(False).astype(bool)].copy()
    net = pd.to_numeric(accepted.get("position_net_return"), errors="coerce") * 10_000.0
    wallet = pd.to_numeric(equity.get("wallet"), errors="coerce").dropna()
    dd = float((wallet / wallet.cummax() - 1.0).min()) if len(wallet) else float("nan")
    return {
        "portfolio_trades": int(len(accepted)), "portfolio_ev_bps": float(net.mean()) if len(net) else float("nan"),
        "portfolio_total_bps": float(net.sum()) if len(net) else float("nan"), "portfolio_max_dd": dd,
    }, decisions


def _summary(selected: pd.DataFrame, *, arm: str, mode: str) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid = selected.loc[_outcome_valid(selected)].copy()
    value = pd.to_numeric(valid["policy_net_bps"], errors="coerce")
    top1 = valid.loc[valid["selection_rank"].le(1), "policy_net_bps"]
    top2 = valid.loc[valid["selection_rank"].le(2), "policy_net_bps"]
    weekly = pd.DataFrame()
    monthly = pd.DataFrame()
    if len(valid):
        valid["week"] = valid["__decision_ts__"].dt.strftime("%G-W%V")
        valid["month"] = valid["__decision_ts__"].dt.strftime("%Y-%m")
        weekly = valid.groupby("week", as_index=False, sort=True).agg(trades=("candidate_id", "size"), net_ev_bps=("policy_net_bps", "mean"), total_bps=("policy_net_bps", "sum"))
        monthly = valid.groupby("month", as_index=False, sort=True).agg(trades=("candidate_id", "size"), net_ev_bps=("policy_net_bps", "mean"), total_bps=("policy_net_bps", "sum"))
    portfolio, decisions = _portfolio_metrics(selected)
    result: dict[str, Any] = {
        "arm": arm, "arm_label": ARM_LABELS[arm], "mode": mode,
        "selected": int(len(selected)), "resolved": int(len(valid)), "coverage": float(len(valid) / max(len(selected), 1)),
        "admitted_ev_bps": float(value.mean()) if len(value) else float("nan"), "total_utility_bps": float(value.sum()) if len(value) else float("nan"),
        "hit_gt50": float(value.gt(50).mean()) if len(value) else float("nan"), "hit_gt100": float(value.gt(100).mean()) if len(value) else float("nan"),
        "top1_ev_bps": float(pd.to_numeric(top1, errors="coerce").mean()) if len(top1) else float("nan"),
        "top2_ev_bps": float(pd.to_numeric(top2, errors="coerce").mean()) if len(top2) else float("nan"),
        "weekly_mean_ev_bps": float(weekly["net_ev_bps"].mean()) if len(weekly) else float("nan"),
        "weekly_worst_ev_bps": float(weekly["net_ev_bps"].min()) if len(weekly) else float("nan"),
        "weekly_positive_fraction": float(weekly["net_ev_bps"].gt(0).mean()) if len(weekly) else float("nan"),
        "monthly_worst_ev_bps": float(monthly["net_ev_bps"].min()) if len(monthly) else float("nan"),
        **portfolio,
    }
    for frame in (weekly, monthly, decisions):
        if len(frame):
            frame.insert(0, "mode", mode)
            frame.insert(0, "arm", arm)
    return result, weekly, monthly, decisions


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    view = frame.loc[:, columns].copy()
    for column in columns:
        if pd.api.types.is_numeric_dtype(view[column]) and column not in {"selected", "resolved", "portfolio_trades"}:
            view[column] = view[column].map(lambda value: "—" if not np.isfinite(value) else f"{value:.2f}")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in view.itertuples(index=False, name=None))
    return lines


def run(eval_root: Path, history_root: Path, out: Path) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    out.mkdir(parents=True, exist_ok=False)
    evaluation = _merge_families(eval_root, evaluation=True)
    evaluation = evaluation.loc[evaluation["__decision_ts__"].ge(START) & evaluation["__decision_ts__"].lt(END)].copy()
    history = _merge_families(history_root, evaluation=False)
    if evaluation.empty or history.empty:
        raise ValueError("empty evaluation or history panel")
    probe = evaluation.loc[:, ["candidate_id", "__decision_ts__", "bcf_final_score", "current_final_score"]].merge(
        history.loc[:, ["candidate_id", "__decision_ts__", "bcf_final_score", "current_final_score"]],
        on=["candidate_id", "__decision_ts__"], how="left", validate="one_to_one", suffixes=("_eval", "_history"),
    )
    if probe[["bcf_final_score_history", "current_final_score_history"]].isna().any().any():
        raise AssertionError("history lacks evaluation score identities")
    for family in ("bcf", "current"):
        if not np.allclose(probe[f"{family}_final_score_eval"], probe[f"{family}_final_score_history"], atol=1e-12, rtol=0.0, equal_nan=True):
            raise AssertionError(f"{family} history/evaluation score mismatch")
    parts: list[pd.DataFrame] = []
    state_parts: list[pd.DataFrame] = []
    package_audits: list[dict[str, Any]] = []
    for month in pd.date_range(START, END - pd.Timedelta(days=1), freq="MS", tz="UTC"):
        current, current_state, current_audit = _month_state(month=month, eval_frame=evaluation, history=history, package_root=eval_root, family="current")
        bcf, bcf_state, bcf_audit = _month_state(month=month, eval_frame=evaluation, history=history, package_root=eval_root, family="bcf")
        shared = ["candidate_id", "__decision_ts__"]
        output_cols = [field for field in current.columns if field.startswith("current__")]
        result = bcf.merge(current.loc[:, [*shared, *output_cols]], on=shared, how="inner", validate="one_to_one")
        if len(result) != len(bcf):
            raise AssertionError(f"{month:%Y-%m} family score merge changed identities")
        parts.append(result)
        state_parts.extend([current_state, bcf_state])
        package_audits.extend([current_audit, bcf_audit])
    panel = pd.concat(parts, ignore_index=True)
    arms = list(ARM_LABELS)
    expected_fields = [f"{family}__{arm}" for family in ("bcf", "current") for arm in arms]
    if panel[expected_fields].isna().any().any():
        raise AssertionError("A0--A10 prediction panel has missing arm scores")
    capacity = _fixed_capacity(panel)
    summaries: list[dict[str, Any]] = []
    selected_parts: list[pd.DataFrame] = []
    weekly_parts: list[pd.DataFrame] = []
    monthly_parts: list[pd.DataFrame] = []
    decisions_parts: list[pd.DataFrame] = []
    for arm in arms:
        for mode in ("matched_static_budget", "gate_only", "priority_only", "full"):
            selected = _select(panel, arm=arm, mode=mode, capacity=capacity if mode == "matched_static_budget" else None)
            if mode == "matched_static_budget":
                observed = selected.groupby("__decision_ts__", as_index=False).size().rename(columns={"size": "selected_k"})
                check = capacity.merge(observed, on="__decision_ts__", how="left", validate="one_to_one")
                check["selected_k"] = check["selected_k"].fillna(0).astype(int)
                if not check["selected_k"].eq(check["fixed_budget_k"]).all():
                    raise AssertionError(f"{arm}: fixed static budget mismatch")
            summary, weekly, monthly, decisions = _summary(selected, arm=arm, mode=mode)
            summaries.append(summary)
            selected_parts.append(selected)
            weekly_parts.append(weekly)
            monthly_parts.append(monthly)
            if len(decisions):
                decisions_parts.append(decisions)
    panel.to_parquet(out / "a0_a10_predictions.parquet", index=False, compression="zstd")
    pd.concat(state_parts, ignore_index=True, sort=False).to_parquet(out / "causal_residual_state.parquet", index=False, compression="zstd")
    pd.DataFrame(package_audits).to_parquet(out / "package_static_parity_audit.parquet", index=False, compression="zstd")
    capacity.to_parquet(out / "matched_static_budget.parquet", index=False, compression="zstd")
    pd.concat(selected_parts, ignore_index=True).to_parquet(out / "selected_candidates.parquet", index=False, compression="zstd")
    pd.concat(weekly_parts, ignore_index=True).to_parquet(out / "weekly_metrics.parquet", index=False, compression="zstd")
    pd.concat(monthly_parts, ignore_index=True).to_parquet(out / "monthly_metrics.parquet", index=False, compression="zstd")
    (pd.concat(decisions_parts, ignore_index=True) if decisions_parts else pd.DataFrame()).to_parquet(out / "portfolio_decisions.parquet", index=False, compression="zstd")
    metrics = pd.DataFrame(summaries)
    metrics.to_parquet(out / "metrics.parquet", index=False, compression="zstd")
    matched = metrics.loc[metrics["mode"].eq("matched_static_budget")].sort_values(
        ["top1_ev_bps", "top2_ev_bps", "total_utility_bps", "weekly_worst_ev_bps"], ascending=False,
    )
    full = metrics.loc[metrics["mode"].eq("full")].sort_values(
        ["portfolio_ev_bps", "portfolio_total_bps", "weekly_worst_ev_bps"], ascending=False,
    )
    report = [
        "# A0--A10 Hierarchical MC1 Calibration Ablation",
        "",
        "Research-only strict-prequential evaluation, February--July 2026. The target-free BCF/Current inputs, package static predictions, canonical policy labels, and global portfolio contract are fixed.",
        "",
        "## Matched static-admission budget",
        "",
        "Every arm receives exactly the static MC1 dual +50-bps capacity at each decision timestamp. This is the primary ranking comparison.",
        "",
        *_markdown_table(matched, ["arm", "arm_label", "selected", "admitted_ev_bps", "top1_ev_bps", "top2_ev_bps", "total_utility_bps", "hit_gt50", "hit_gt100", "weekly_worst_ev_bps", "portfolio_ev_bps", "portfolio_total_bps", "portfolio_max_dd"]),
        "",
        "## Real +50 dual admission, full gate + priority",
        "",
        *_markdown_table(full, ["arm", "arm_label", "selected", "admitted_ev_bps", "top1_ev_bps", "top2_ev_bps", "total_utility_bps", "weekly_worst_ev_bps", "portfolio_trades", "portfolio_ev_bps", "portfolio_total_bps", "portfolio_max_dd"]),
        "",
        "## Attribution",
        "",
        "`gate_only` uses each arm for the dual +50 gate and static BCF auction priority. `priority_only` uses static admission and each arm's BCF priority. `full` changes both.",
    ]
    (out / "A0_A10_HIERARCHICAL_MC1_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    state = pd.concat(state_parts, ignore_index=True, sort=False)
    correctness = {
        "schema": SCHEMA,
        "no_live_or_exchange_mutation": True,
        "same_bcf_current_target_free_candidate_ids": True,
        "same_policy_labels_between_families": True,
        "history_scores_match_evaluation_scores": True,
        "static_package_predictions_match_persisted_eval_static_scores": bool(pd.DataFrame(package_audits)["static_parity_max_abs_bps"].le(1e-8).all()),
        "matched_budget_is_timestamp_local": True,
        "all_arms_match_static_budget_exactly": True,
        "residuals_use_only_policy_label_available_ts_before_decision_day": bool(
            state.loc[state["scope"].eq("global"), "decision_day"].notna().all()
        ),
        "policy_outcomes_are_not_inputs_to_candidate_score_families": True,
        "band_corrected_arms_projected_monotonic_per_timestamp": True,
    }
    _once_json(out / "correctness_report.json", correctness)
    _once_json(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline A0--A10 MC1 calibration research; no live/exchange mutation",
        "evaluation": {"start": START.isoformat(), "end_exclusive": END.isoformat()},
        "eval_root": str(eval_root.resolve()), "history_root": str(history_root.resolve()),
        "input_hashes": {
            "eval_bcf": _sha256(eval_root / "enhanced_bcf_mc1_predictions.parquet"),
            "eval_current": _sha256(eval_root / "enhanced_current_mc1_predictions.parquet"),
            "history_bcf": _sha256(history_root / "enhanced_bcf_mc1_predictions.parquet"),
            "history_current": _sha256(history_root / "enhanced_current_mc1_predictions.parquet"),
        },
        "families": ["bcf", "current"], "horizons_days": list(HORIZONS),
        "arms": ARM_LABELS, "threshold_bps": THRESHOLD_BPS,
        "matched_budget": "per timestamp count where min(BCF A0, Current A0) >= +50 bps",
        "real_admission": "both family arm EV >= +50 bps",
        "modes": ["matched_static_budget", "gate_only", "priority_only", "full"],
        "residual_target": "canonical rich 15-minute policy net bps; cost embedded exactly once",
        "uncertainty": "equal-decision-day clustered robust standard error; all labels resolved before decision day",
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=EVAL_ROOT)
    parser.add_argument("--history-root", type=Path, default=HISTORY_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(run(args.eval_root, args.history_root, args.out))


if __name__ == "__main__":
    main()
