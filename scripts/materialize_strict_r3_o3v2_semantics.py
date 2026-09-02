#!/usr/bin/env python3
"""Materialise the strict training-only O3-v2 path/policy semantic sidecar.

The sidecar is deliberately factorised.  It contains *resolved future-path*
descriptors that may be used for labels, sampling and weights while fitting an
O3-v2 correction head.  It must never be joined to a target-free score panel
or an inference feature contract.

The producer is separate from the historical O3 materialiser so its schema is
immutable and so research can compare the two definitions without altering a
live or canonical artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_rich_policy import (  # noqa: E402
    RichPolicyParams,
    _activation_distance,
    _barrier_distances,
    _stop_distance,
    simulate_rich_policy,
)


SCHEMA = "strict_r3_o3v2_semantics_v1"
HORIZON_BARS = 48
BAR_MINUTES = 15
COST_BPS = 100.0
TRAILING_LARGE_GROSS_BPS = 250.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    children: Iterable[Path] = sorted(path.rglob("*.parquet")) if path.is_dir() else (path,)
    for child in children:
        digest.update(str(child).encode())
        with child.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _first_true(mask: np.ndarray) -> np.ndarray:
    present = mask.any(axis=1)
    result = np.zeros(len(mask), dtype=np.int16)
    result[present] = np.argmax(mask[present], axis=1).astype(np.int16) + 1
    return result


def _load_bars(root: Path, symbol: str) -> pd.DataFrame:
    path = root / f"{str(symbol).lower().replace('/', '')}_15m.parquet"
    if not path.exists():
        return pd.DataFrame(columns=("high", "low", "close"))
    bars = pd.read_parquet(path, columns=("high", "low", "close"))
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError(f"15m source index is not datetime: {path}")
    index = pd.DatetimeIndex(bars.index)
    bars.index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    bars = bars.loc[~bars.index.duplicated(keep="last")].sort_index()
    return bars.apply(pd.to_numeric, errors="coerce")


def _windows(bars: pd.DataFrame, decisions: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return exact post-decision completed 15m paths without any filling."""
    index = pd.DatetimeIndex(pd.to_datetime(decisions, utc=True, errors="raise"))
    empty = np.full((len(index), HORIZON_BARS), np.nan, dtype=np.float32)
    if bars.empty:
        return np.zeros(len(index), dtype=bool), empty, empty.copy(), empty.copy()
    start = min(pd.Timestamp(index.min()), pd.Timestamp(bars.index.min())).floor("15min")
    end = max(pd.Timestamp(index.max()) + pd.Timedelta(hours=12), pd.Timestamp(bars.index.max())).ceil("15min")
    grid = pd.date_range(start, end, freq="15min", inclusive="left", tz="UTC")
    values = bars.reindex(grid).loc[:, ["high", "low", "close"]].to_numpy(np.float32)
    offsets = ((index - start) / pd.Timedelta(minutes=BAR_MINUTES)).astype(np.int64)
    high, low, close = empty.copy(), empty.copy(), empty.copy()
    usable = (offsets >= 0) & (offsets + HORIZON_BARS <= len(grid))
    for local in np.flatnonzero(usable):
        part = values[offsets[local] : offsets[local] + HORIZON_BARS]
        high[local], low[local], close[local] = part[:, 0], part[:, 1], part[:, 2]
    complete = (
        np.isfinite(high).all(axis=1)
        & np.isfinite(low).all(axis=1)
        & np.isfinite(close).all(axis=1)
        & (high > 0).all(axis=1)
        & (low > 0).all(axis=1)
        & (close > 0).all(axis=1)
        & (high >= low).all(axis=1)
    )
    return complete, high, low, close


def _fixed_neighbourhood(params: RichPolicyParams) -> tuple[tuple[str, RichPolicyParams], ...]:
    """Small, declared local policy neighbourhood; never selected per row."""
    return (
        ("canonical", params),
        ("sl_tight_90", replace(params, sl_mult=float(params.sl_mult) * 0.90)),
        ("sl_wide_110", replace(params, sl_mult=float(params.sl_mult) * 1.10)),
        ("protect_early", replace(params, protection_activation_atr=max(0.25, float(params.protection_activation_atr) - 0.25))),
        ("protect_late", replace(params, protection_activation_atr=float(params.protection_activation_atr) + 0.25)),
        ("trail_early", replace(params, trailing_activation_mult=max(0.10, float(params.trailing_activation_mult) * 0.90))),
        ("trail_late", replace(params, trailing_activation_mult=float(params.trailing_activation_mult) * 1.10)),
        ("giveback_tight", replace(params, giveback_beta=max(0.05, float(params.giveback_beta) * 0.85))),
        ("giveback_wide", replace(params, giveback_beta=float(params.giveback_beta) * 1.15)),
    )


def _assign_path_metrics(
    output: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    bars_root: Path,
    params: RichPolicyParams,
    median_atr_fraction: float,
) -> None:
    """Calculate path-only values symbol by symbol to bound memory use."""
    required = (
        np.isfinite(pd.to_numeric(frame["entry_price"], errors="coerce").to_numpy(float))
        & (pd.to_numeric(frame["entry_price"], errors="coerce").to_numpy(float) > 0)
        & np.isfinite(pd.to_numeric(frame["path_arch_atr_fraction"], errors="coerce").to_numpy(float))
        & (pd.to_numeric(frame["path_arch_atr_fraction"], errors="coerce").to_numpy(float) > 0)
    )
    for symbol, group in frame.loc[required].groupby("__symbol__", sort=True):
        positions = group.index.to_numpy(np.int64)
        bars = _load_bars(bars_root, str(symbol))
        complete, high, low, close = _windows(bars, group["__decision_ts__"])
        if not complete.any():
            continue
        local = np.flatnonzero(complete)
        target = positions[local]
        entry = pd.to_numeric(group.iloc[local]["entry_price"], errors="coerce").to_numpy(float)
        atr_fraction = pd.to_numeric(group.iloc[local]["path_arch_atr_fraction"], errors="coerce").to_numpy(float)
        atr = entry * atr_fraction
        sl_raw, tp_raw = _barrier_distances(entry, atr, params, median_atr_fraction=median_atr_fraction)
        lower = _stop_distance(sl_raw, entry, params)
        upper = np.maximum(
            _activation_distance(tp_raw, entry, params, bar=0),
            entry * COST_BPS / 10_000.0,
        )
        upper_bar = _first_true(high[local] >= entry[:, None] + upper[:, None])
        lower_bar = _first_true(low[local] <= entry[:, None] - lower[:, None])
        upper_hit, lower_hit = upper_bar > 0, lower_bar > 0
        same = upper_hit & lower_hit & (upper_bar == lower_bar)
        upper_first = upper_hit & (~lower_hit | (upper_bar < lower_bar))
        lower_first = lower_hit & (~upper_hit | (lower_bar < upper_bar))
        event = np.full(len(local), "vertical", dtype=object)
        event[upper_first] = "upper_first"
        event[lower_first] = "lower_first"
        event[same] = "ambiguous"
        fav = np.maximum(high[local] - entry[:, None], 0.0)
        adv = np.maximum(entry[:, None] - low[local], 0.0)
        peak = np.max(fav, axis=1)
        adverse_peak = np.max(adv, axis=1)
        denom = np.maximum(atr, 1e-12)
        peak_atr, adverse_atr = peak / denom, adverse_peak / denom
        below_entry = low[local] < entry[:, None]
        above_half = high[local] >= entry[:, None] + 0.5 * atr[:, None]
        above_one = high[local] >= entry[:, None] + atr[:, None]
        adverse_area = np.mean(adv / denom[:, None], axis=1)
        hourly_peak = np.maximum.accumulate(fav, axis=1)[:, 3::4]
        increments = np.diff(np.column_stack((np.zeros(len(local)), hourly_peak)), axis=1)
        positive_increments = np.maximum(increments, 0.0)
        concentration = np.divide(
            np.max(positive_increments, axis=1),
            np.sum(positive_increments, axis=1),
            out=np.zeros(len(local)), where=np.sum(positive_increments, axis=1) > 1e-12,
        )
        output.loc[target, "semantic_tbm_path_complete"] = True
        output.loc[target, "semantic_tbm_event"] = event
        output.loc[target, "semantic_upper_distance_atr"] = upper / denom
        output.loc[target, "semantic_lower_distance_atr"] = lower / denom
        output.loc[target, "semantic_upper_bar"] = upper_bar
        output.loc[target, "semantic_lower_bar"] = lower_bar
        output.loc[target, "semantic_time_to_event_h"] = np.where(
            upper_first, upper_bar * 0.25, np.where(lower_first, lower_bar * 0.25, np.nan)
        )
        output.loc[target, "semantic_peak_mfe_atr"] = peak_atr
        output.loc[target, "semantic_peak_mae_atr"] = adverse_atr
        output.loc[target, "semantic_adverse_occupancy"] = np.mean(below_entry, axis=1)
        output.loc[target, "semantic_adverse_area_atr"] = adverse_area
        output.loc[target, "semantic_longest_underwater_bars"] = _longest_run(below_entry)
        output.loc[target, "semantic_postop_above_half_fraction"] = np.mean(above_half, axis=1)
        output.loc[target, "semantic_postop_above_one_fraction"] = np.mean(above_one, axis=1)
        output.loc[target, "semantic_impulse_concentration"] = concentration
        output.loc[target, "semantic_favourable_area_atr"] = np.mean(fav / denom[:, None], axis=1)
        output.loc[target, "semantic_policy_neighbourhood_valid"] = True
        neighbourhood = _fixed_neighbourhood(params)
        values = np.empty((len(local), len(neighbourhood)), dtype=np.float32)
        for column, (_name, variant) in enumerate(neighbourhood):
            values[:, column] = simulate_rich_policy(
                entry=entry, atr=atr, highs=high[local], lows=low[local], closes=close[local],
                params=variant, median_atr_fraction=median_atr_fraction, side="long",
            )["net_bps"]
        output.loc[target, "semantic_policy_neighbourhood_mean_bps"] = np.mean(values, axis=1)
        output.loc[target, "semantic_policy_neighbourhood_std_bps"] = np.std(values, axis=1)
        output.loc[target, "semantic_policy_neighbourhood_profitable_fraction"] = np.mean(values > 0.0, axis=1)


def _longest_run(mask: np.ndarray) -> np.ndarray:
    """Return maximum consecutive true observations per row."""
    run = np.zeros(len(mask), dtype=np.int16)
    longest = np.zeros(len(mask), dtype=np.int16)
    for column in range(mask.shape[1]):
        run = np.where(mask[:, column], run + 1, 0)
        longest = np.maximum(longest, run)
    return longest


def _text(series: pd.Series, default: str = "unknown") -> np.ndarray:
    return series.astype("string").fillna(default).astype(str).to_numpy(object)


def _derive_axes(output: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    """Derive fixed A--K semantic axes after all path values are present."""
    out = output.copy()
    valid = out["semantic_path_valid"].to_numpy(bool)
    index = np.flatnonzero(valid)
    if not len(index):
        return out
    event = _text(out.loc[valid, "semantic_tbm_event"], "vertical")
    upper_first, lower_first = event == "upper_first", event == "lower_first"
    peak = pd.to_numeric(out.loc[valid, "semantic_peak_mfe_atr"], errors="coerce").to_numpy(float)
    upper = pd.to_numeric(out.loc[valid, "semantic_upper_distance_atr"], errors="coerce").to_numpy(float)
    pre_mae = pd.to_numeric(source.loc[valid, "path_arch_mae_before_meaningful_mfe_r"], errors="coerce").to_numpy(float)
    atr_frac = pd.to_numeric(source.loc[valid, "path_arch_atr_fraction"], errors="coerce").to_numpy(float)
    pre_mae_atr = np.divide(pre_mae, atr_frac, out=np.full(len(index), np.nan), where=atr_frac > 0)
    retention = pd.to_numeric(source.loc[valid, "path_arch_peak_retention_ratio"], errors="coerce").to_numpy(float)
    final_return = pd.to_numeric(source.loc[valid, "path_arch_final_return_r"], errors="coerce").to_numpy(float)
    policy = pd.to_numeric(out.loc[valid, "semantic_policy_net_bps"], errors="coerce").to_numpy(float)
    gross = pd.to_numeric(out.loc[valid, "semantic_policy_gross_bps"], errors="coerce").to_numpy(float)
    time = pd.to_numeric(source.loc[valid, "path_arch_time_to_first_meaningful_mfe_h"], errors="coerce").to_numpy(float)
    area = pd.to_numeric(out.loc[valid, "semantic_adverse_area_atr"], errors="coerce").to_numpy(float)
    occupancy = pd.to_numeric(out.loc[valid, "semantic_adverse_occupancy"], errors="coerce").to_numpy(float)
    longest = pd.to_numeric(out.loc[valid, "semantic_longest_underwater_bars"], errors="coerce").to_numpy(float)
    concentration = pd.to_numeric(out.loc[valid, "semantic_impulse_concentration"], errors="coerce").to_numpy(float)
    postop_one = pd.to_numeric(out.loc[valid, "semantic_postop_above_one_fraction"], errors="coerce").to_numpy(float)
    robust_mean = pd.to_numeric(out.loc[valid, "semantic_policy_neighbourhood_mean_bps"], errors="coerce").to_numpy(float)
    robust_std = pd.to_numeric(out.loc[valid, "semantic_policy_neighbourhood_std_bps"], errors="coerce").to_numpy(float)
    robust_fraction = pd.to_numeric(out.loc[valid, "semantic_policy_neighbourhood_profitable_fraction"], errors="coerce").to_numpy(float)
    exit_reason = _text(out.loc[valid, "semantic_policy_exit_reason"], "timeout_h12")

    sequence = np.full(len(index), "no_opportunity", dtype=object)
    sequence[lower_first & (peak >= upper)] = "adverse_first_recovery"
    sequence[lower_first & (peak < upper)] = "adverse_first_failure"
    sequence[upper_first & (pre_mae_atr <= 0.5)] = "favourable_first_clean"
    sequence[upper_first & (pre_mae_atr > 0.5)] = "favourable_first_mild_adversity"
    speed = np.full(len(index), "never", dtype=object)
    speed[(peak >= upper) & (time <= 2.0)] = "fast"
    speed[(peak >= upper) & (time > 2.0) & (time <= 6.0)] = "normal"
    speed[(peak >= upper) & (time > 6.0)] = "slow"
    persistence = np.full(len(index), "large_giveback", dtype=object)
    persistence[retention >= 0.75] = "persistent"
    persistence[(retention >= 0.50) & (retention < 0.75)] = "partial_giveback"
    persistence[final_return <= 0.0] = "full_reversal"
    preop = np.full(len(index), "severe", dtype=object)
    preop[(pre_mae_atr <= 0.25) & (area <= 0.10)] = "clean"
    preop[(pre_mae_atr > 0.25) & (pre_mae_atr <= 0.75) & (longest <= 4)] = "brief"
    preop[(pre_mae_atr > 0.75) & (pre_mae_atr <= 1.50) & (longest <= 16)] = "sustained"
    conversion = np.where(
        peak >= upper,
        np.where(policy >= 50.0, "high_opportunity_good_capture", "high_opportunity_poor_capture"),
        np.where(policy >= 50.0, "low_opportunity_good_capture", "low_opportunity_poor_capture"),
    )
    exit4 = np.where(exit_reason == "stop_loss", "stop", np.where(exit_reason == "timeout_h12", "timeout", np.where(np.isin(exit_reason, ("smooth_capital_protect", "capital_protect")), "smooth_protection", "trailing")))
    exit5 = exit4.copy()
    exit5[(exit4 == "trailing") & (gross >= TRAILING_LARGE_GROSS_BPS)] = "large_trailing"
    exit5[(exit4 == "trailing") & (gross < TRAILING_LARGE_GROSS_BPS)] = "regular_trailing"
    margin = peak - pd.to_numeric(out.loc[valid, "semantic_peak_mae_atr"], errors="coerce").to_numpy(float)
    margin_class = np.full(len(index), "fragile", dtype=object)
    margin_class[(margin >= 0.5) & (margin < 1.5)] = "narrow"
    margin_class[(margin >= 1.5) & (margin < 3.0)] = "comfortable"
    margin_class[margin >= 3.0] = "dominant"
    occupancy_class = np.full(len(index), "deep_sustained", dtype=object)
    occupancy_class[(occupancy <= 0.10) & (area <= 0.10)] = "brief"
    occupancy_class[(occupancy > 0.10) & (occupancy <= 0.35) & (area <= 0.30)] = "moderate"
    occupancy_class[(occupancy > 0.35) & (longest <= 20)] = "sustained"
    concentration_class = np.full(len(index), "single_impulse", dtype=object)
    concentration_class[concentration < 0.45] = "distributed"
    concentration_class[(concentration >= 0.45) & (concentration < 0.70)] = "moderate"
    concentration_class[(concentration >= 0.70) & (concentration < 0.90)] = "high"
    persistence_class = np.full(len(index), "transient", dtype=object)
    persistence_class[(postop_one >= 0.10) & (postop_one < 0.30)] = "short_lived"
    persistence_class[(postop_one >= 0.30) & (postop_one < 0.60)] = "moderately_persistent"
    persistence_class[postop_one >= 0.60] = "persistent"
    robust_class = np.full(len(index), "policy_sensitive", dtype=object)
    robust_class[(robust_fraction >= 0.75) & (robust_mean > 0.0) & (robust_std <= 100.0)] = "policy_robust"
    robust_class[(robust_fraction >= 0.50) & (robust_mean > -25.0)] = "mixed_robustness"

    archetype = np.full(len(index), "timeout", dtype=object)
    archetype[(sequence == "favourable_first_clean") & (speed == "fast") & (persistence == "persistent") & (exit4 == "trailing")] = "clean_fast_persistent_trailing"
    archetype[(sequence == "favourable_first_clean") & (speed != "fast")] = "clean_normal"
    archetype[(sequence == "adverse_first_recovery") & (exit4 == "trailing")] = "adverse_first_recovery_trailing"
    archetype[exit4 == "smooth_protection"] = "smooth_protection"
    archetype[(peak >= upper) & np.isin(persistence, ("large_giveback", "full_reversal"))] = "favourable_major_giveback"
    archetype[(peak >= upper) & (policy < 50.0)] = "high_mfe_poor_capture"
    archetype[sequence == "adverse_first_failure"] = "adverse_first_stop"
    archetype[(persistence_class == "transient") & (peak >= upper)] = "transient"
    out.loc[index, "semantic_axis_a_sequence"] = sequence
    out.loc[index, "semantic_axis_b_speed"] = speed
    out.loc[index, "semantic_axis_c_persistence"] = persistence
    out.loc[index, "semantic_axis_d_preop_adversity"] = preop
    out.loc[index, "semantic_axis_e_conversion"] = conversion
    out.loc[index, "semantic_axis_f_exit4"] = exit4
    out.loc[index, "semantic_axis_f_exit5"] = exit5
    out.loc[index, "semantic_axis_g_margin"] = margin_class
    out.loc[index, "semantic_axis_h_adverse_occupancy"] = occupancy_class
    out.loc[index, "semantic_axis_i_impulse_concentration"] = concentration_class
    out.loc[index, "semantic_axis_j_postop_persistence"] = persistence_class
    out.loc[index, "semantic_axis_k_policy_robustness"] = robust_class
    out.loc[index, "semantic_archetype"] = archetype
    return out


def _empty_output(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.loc[:, ["candidate_id", "__decision_ts__", "__symbol__", "side_name"]].copy()
    out["semantic_label_available_ts"] = pd.NaT
    out["semantic_path_valid"] = False
    out["semantic_tbm_path_complete"] = False
    out["semantic_policy_neighbourhood_valid"] = False
    string_fields = (
        "semantic_tbm_event", "semantic_policy_exit_reason", "semantic_axis_a_sequence",
        "semantic_axis_b_speed", "semantic_axis_c_persistence", "semantic_axis_d_preop_adversity",
        "semantic_axis_e_conversion", "semantic_axis_f_exit4", "semantic_axis_f_exit5",
        "semantic_axis_g_margin", "semantic_axis_h_adverse_occupancy",
        "semantic_axis_i_impulse_concentration", "semantic_axis_j_postop_persistence",
        "semantic_axis_k_policy_robustness", "semantic_axis_l_base_error",
        "semantic_axis_m_query_quality", "semantic_archetype",
    )
    for column in string_fields:
        out[column] = pd.Series(pd.NA, index=out.index, dtype="string")
    numeric_fields = (
        "semantic_policy_net_bps", "semantic_policy_gross_bps", "semantic_upper_distance_atr",
        "semantic_lower_distance_atr", "semantic_upper_bar", "semantic_lower_bar",
        "semantic_time_to_event_h", "semantic_peak_mfe_atr", "semantic_peak_mae_atr",
        "semantic_adverse_occupancy", "semantic_adverse_area_atr", "semantic_longest_underwater_bars",
        "semantic_postop_above_half_fraction", "semantic_postop_above_one_fraction",
        "semantic_impulse_concentration", "semantic_favourable_area_atr",
        "semantic_policy_neighbourhood_mean_bps", "semantic_policy_neighbourhood_std_bps",
        "semantic_policy_neighbourhood_profitable_fraction",
    )
    for column in numeric_fields:
        out[column] = np.nan
    return out


def _months(root: Path) -> list[str]:
    return sorted(part.name.split("=", 1)[1] for part in root.glob("month=*") if part.is_dir())


def _write_manifest(out: Path, payload: dict[str, object]) -> None:
    target = out / "run_manifest.json"
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-root", type=Path, required=True)
    parser.add_argument("--canonical-policy-labels", type=Path, required=True)
    parser.add_argument("--policy-json", type=Path, required=True)
    parser.add_argument("--bars-root", type=Path, default=Path("15m_ohlcv_perp"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--months", help="comma-separated YYYY-MM subset")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"immutable output already exists: {args.out}")
    policy_json = json.loads(args.policy_json.read_text())
    params = RichPolicyParams.from_mapping(policy_json["params"])
    median_atr_fraction = float(policy_json.get("median_atr_fraction", policy_json["median_atr_fraction_fitted_on_complete_2024_development"]))
    policy = pd.read_parquet(args.canonical_policy_labels)
    needed = {"candidate_id", "policy_path_valid", "policy_net_bps", "policy_gross_bps", "policy_exit_reason", "policy_label_available_ts"}
    missing = sorted(needed - set(policy.columns))
    if missing:
        raise AssertionError(f"canonical policy labels lack {missing}")
    policy = policy.loc[:, list(needed)].copy()
    if policy["candidate_id"].duplicated().any():
        raise AssertionError("canonical policy labels duplicate candidate IDs")
    tokens = args.months.split(",") if args.months else _months(args.path_root)
    args.out.mkdir(parents=True)
    coverage: list[dict[str, object]] = []
    for token in tokens:
        source = args.path_root / f"month={token}" / "side=long.parquet"
        if not source.exists():
            raise FileNotFoundError(source)
        frame = pd.read_parquet(source).reset_index(drop=True)
        if frame["candidate_id"].duplicated().any():
            raise AssertionError(f"{token}: duplicate path identities")
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
        # Path panels intentionally retain historical convenience outcome
        # fields.  The reconciled canonical policy ledger is the sole source
        # of policy outcomes for this contract, so remove them before joining.
        stale = [
            column for column in (
                "policy_path_valid", "policy_net_bps", "policy_gross_bps",
                "policy_exit_reason", "policy_label_available_ts", "policy_cost_bps",
            ) if column in frame.columns
        ]
        frame = frame.drop(columns=stale)
        frame = frame.merge(policy, on="candidate_id", how="left", validate="one_to_one")
        if frame["policy_path_valid"].isna().any():
            raise AssertionError(f"{token}: canonical policy labels missed candidate identities")
        out = _empty_output(frame)
        out["semantic_policy_net_bps"] = pd.to_numeric(frame["policy_net_bps"], errors="coerce")
        out["semantic_policy_gross_bps"] = pd.to_numeric(frame["policy_gross_bps"], errors="coerce")
        out["semantic_policy_exit_reason"] = frame["policy_exit_reason"].astype("string")
        availability = pd.concat((
            pd.to_datetime(frame["supportive_label_available_ts"], utc=True, errors="coerce"),
            pd.to_datetime(frame["policy_label_available_ts"], utc=True, errors="coerce"),
        ), axis=1).max(axis=1)
        out["semantic_label_available_ts"] = availability
        _assign_path_metrics(out, frame, bars_root=args.bars_root, params=params, median_atr_fraction=median_atr_fraction)
        out["semantic_path_valid"] = (
            frame["supportive_path_valid"].fillna(False).astype(bool)
            & frame["policy_path_valid"].fillna(False).astype(bool)
            & out["semantic_tbm_path_complete"].fillna(False).astype(bool)
            & out["semantic_label_available_ts"].notna()
        )
        out = _derive_axes(out, frame)
        invalid = ~out["semantic_path_valid"].astype(bool)
        semantic_columns = [column for column in out if column.startswith("semantic_") and column not in {"semantic_path_valid", "semantic_label_available_ts", "semantic_tbm_path_complete", "semantic_policy_neighbourhood_valid"}]
        out.loc[invalid, semantic_columns] = np.nan
        target = args.out / "parts" / f"month={token}"
        target.mkdir(parents=True)
        out.to_parquet(target / "semantics.parquet", index=False, compression="zstd")
        valid_fraction = float(out["semantic_path_valid"].mean())
        coverage.append({
            "month": token, "rows": int(len(out)), "valid_rows": int(out["semantic_path_valid"].sum()),
            "valid_fraction": valid_fraction,
            "archetypes": int(out.loc[out["semantic_path_valid"], "semantic_archetype"].nunique()),
        })
        if valid_fraction < 0.90:
            raise AssertionError(f"{token}: semantic coverage {valid_fraction:.3%} < 90%")
        print(json.dumps({"event": "materialized", **coverage[-1]}), flush=True)
    _write_manifest(args.out, {
        "schema": SCHEMA,
        "scope": "resolved future-path support labels only; prohibited from inference and target-free score receipts",
        "path_root": str(args.path_root), "path_root_sha256": _sha256(args.path_root),
        "canonical_policy_labels": str(args.canonical_policy_labels), "canonical_policy_labels_sha256": _sha256(args.canonical_policy_labels),
        "policy_json": str(args.policy_json), "policy_json_sha256": _sha256(args.policy_json),
        "bars_root": str(args.bars_root), "months": tokens, "coverage": coverage,
        "axes": {
            "A": "sequencing", "B": "speed", "C": "persistence", "D": "pre-opportunity adversity",
            "E": "policy conversion", "F": "four/five-way exit mechanism", "G": "fav/adverse margin",
            "H": "adverse occupancy", "I": "impulse concentration", "J": "post-opportunity persistence",
            "K": "frozen local policy-neighbourhood robustness",
            "L": "strict-OOF base error, generated fold-locally by target funnel", "M": "realised query quality, generated fold-locally by target funnel",
        },
        "neighbourhood": [name for name, _ in _fixed_neighbourhood(params)],
        "causality": {
            "labels_available": "semantic_label_available_ts is max(path/policy availability)",
            "invalid": "invalid/incomplete paths remain target-invalid and have no semantic classes",
            "prohibited": "all semantic columns are training-only and must not enter target-free score panels or MC1 inference",
        },
    })


if __name__ == "__main__":
    main()
