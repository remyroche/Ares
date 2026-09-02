#!/usr/bin/env python3
"""Target-free market-state/Kalman representation producer for P8U research.

This materialises *hourly* state rather than copying thousands of candidate
features to every row.  The downstream screen joins selected state columns to
the strict-OOF Base population only after state selection.  All input columns
are contemporaneously generated market fields from the full causal feature
contract; this producer never opens policy, path, outcome, or MC1 data.

The Kalman parameterisation is half-life based.  For a desired steady-state
gain ``a = 1 - exp(-log(2)/half_life_hours)``, the scalar random-walk ratio is
``Q/R = a**2/(1-a)``.  Thus every fast/slow comparison is comparable across
state units and no raw hand-tuned Q/R values leak into the research contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "strict_r3_p8u_market_state_transition_v1"
SEED = 1729
# Frozen, shared half-life pairs requested for the first representation pass.
PAIRS: tuple[tuple[int, int], ...] = ((2, 14), (3, 14), (3, 21), (5, 21), (5, 42), (7, 42))
# All drivers below are timestamp-global upstream market fields.  They are
# repeated on candidate rows because their source calculation happens before
# Router projection; the producer verifies that repetition before using them.
DRIVERS: tuple[tuple[str, str, str], ...] = (
    ("return_iqr", "cs_dispersion_ret_4h", "cross_sectional_return"),
    ("return_tail", "cs_ret_dispersion_4h_pct", "cross_sectional_return"),
    ("breadth", "market_breadth_4h", "breadth"),
    ("breadth_negative", "negative_breadth_pct", "breadth"),
    ("breadth_downside", "downside_breadth_intensity", "breadth"),
    ("volatility_dispersion", "market_dispersion_4h", "volatility"),
    ("volatility_level", "mkt_rv_4h", "volatility"),
    ("volatility_ratio", "mkt_rv_ratio_1h_24h", "volatility"),
    ("volatility_xs", "xs_dispersion__volatility_zscore", "volatility"),
    ("liquidity_depth", "xasset_mkt_depth_to_qv_z", "liquidity"),
    ("execution_spread", "xasset_mkt_spread_bps_z_24h", "liquidity"),
    ("execution_spread_level", "median_spread_bps", "liquidity"),
    ("oi_effective_rank", "xs_cov_effective_rank__xs_open_interest", "leverage_flow"),
    ("oi_eigen_rank", "eig_effective_rank__open_interest", "leverage_flow"),
    ("funding_dispersion", "funding_rate_cross_asset_dispersion", "leverage_flow"),
    ("correlation", "cross_asset_corr_4h", "dependence"),
    ("correlation_break", "correlation_breakdown_dispersion", "dependence"),
    ("portable_effective_rank", "xs_cov_effective_rank__xs_asset_portable_all", "dependence"),
    ("breakout_eigen_rank", "eig_effective_rank__breakout_all", "dependence"),
    ("pc1_share", "market_pc1_variance_share_24h", "spectral"),
    ("spectral_effective_rank", "state_spectral_eig_effective_rank", "spectral"),
    ("spectral_entropy", "state_spectral_eig_entropy", "spectral"),
    ("spectral_lambda1_share", "state_spectral_eig_lambda1_share", "spectral"),
    ("spectral_mahalanobis", "state_spectral_top3_mahalanobis", "spectral"),
    ("btc_decoupling", "btc_decoupling_dispersion", "btc_eth_beta"),
    ("btc_alt_strength", "btc_resilience_alt_weakness", "btc_eth_beta"),
)


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    members = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    for member in members:
        digest.update(str(member).encode())
        with member.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _month(token: str) -> pd.Timestamp:
    return pd.Timestamp(f"{token}-01", tz="UTC")


def _month_range(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Timestamp, ...]:
    result: list[pd.Timestamp] = []
    value = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while value < end:
        result.append(value)
        value += pd.offsets.MonthBegin(1)
    return tuple(result)


def _parse_roots(text: str) -> tuple[Path, ...]:
    values = tuple(ROOT / item.strip() for item in text.split(",") if item.strip())
    if not values or not all(path.exists() for path in values):
        raise FileNotFoundError("one or more raw feature roots do not exist")
    return values


def _raw_path(roots: Iterable[Path], month: pd.Timestamp) -> Path:
    found = [root / f"month={month:%Y-%m}" / "causal_feature_universe.parquet" for root in roots]
    found = [path for path in found if path.exists()]
    if len(found) != 1:
        raise FileNotFoundError(f"expected exactly one full feature panel for {month:%Y-%m}, found={found}")
    return found[0]


def _check_target_free(path: Path) -> None:
    forbidden = {"policy_net_bps", "policy_path_valid", "policy_label_available_ts", "outcome", "label", "mfe", "mae"}
    names = set(pq.ParquetFile(path).schema_arrow.names)
    leaked = sorted(name for name in names if name.lower() in forbidden)
    if leaked:
        raise AssertionError(f"{path}: target/outcome fields are forbidden: {leaked}")


def _shared_market_series(roots: tuple[Path, ...], months: tuple[pd.Timestamp, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read market drivers and verify their value is invariant inside a timestamp.

    The raw full panel is the pre-router feature producer's output.  We never
    calculate a cross-section from the potentially smaller candidate set here:
    an invariant market field is simply deduplicated by timestamp.
    """
    source_fields = [item[1] for item in DRIVERS]
    pieces: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    for month in months:
        path = _raw_path(roots, month)
        _check_target_free(path)
        names = set(pq.ParquetFile(path).schema_arrow.names)
        missing = sorted(set(source_fields).difference(names))
        if missing:
            raise AssertionError(f"{month:%Y-%m}: missing frozen market drivers {missing}")
        part = pd.read_parquet(path, columns=["__decision_ts__", *source_fields])
        part["__decision_ts__"] = pd.to_datetime(part["__decision_ts__"], utc=True, errors="raise")
        if part.empty:
            raise AssertionError(f"{month:%Y-%m}: empty market source")
        for _, source, family in DRIVERS:
            values = pd.to_numeric(part[source], errors="coerce")
            count = part.assign(__value__=values).groupby("__decision_ts__", sort=False).__value__.nunique(dropna=True)
            # A value can be unavailable for an entire timestamp.  Any actual
            # within-time variation means this is not a market-state field.
            if int(count.max()) > 1:
                raise AssertionError(f"{month:%Y-%m}: {source} varies within a timestamp")
            audit.append({
                "month": f"{month:%Y-%m}", "driver": source, "family": family,
                "finite_fraction": float(np.isfinite(values).mean()),
                "timestamps": int(part.__decision_ts__.nunique()),
                "max_within_timestamp_nunique": int(count.max()),
                "source_path": str(path.relative_to(ROOT)),
            })
        # ``first`` skips a partial row but never synthesises a future value.
        grouped = part.groupby("__decision_ts__", sort=True)[source_fields].first().reset_index()
        pieces.append(grouped)
    frame = pd.concat(pieces, ignore_index=True).sort_values("__decision_ts__", kind="stable")
    if frame.__decision_ts__.duplicated().any():
        raise AssertionError("monthly market series overlap")
    frame = frame.set_index("__decision_ts__").asfreq("h")
    for name in source_fields:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame.index.name = "__decision_ts__"
    return frame, pd.DataFrame(audit)


def _gain(half_life_days: int) -> tuple[float, float]:
    half_life_hours = float(half_life_days * 24)
    gain = 1.0 - math.exp(-math.log(2.0) / half_life_hours)
    ratio = gain * gain / max(1e-12, 1.0 - gain)
    return gain, ratio


def _kalman(values: np.ndarray, half_life_days: int) -> dict[str, np.ndarray]:
    """Causal scalar random-walk Kalman filter with adaptive innovation scale."""
    n = len(values)
    _, q = _gain(half_life_days)
    level = np.full(n, np.nan, dtype=np.float32)
    prior = np.full(n, np.nan, dtype=np.float32)
    innovation = np.full(n, np.nan, dtype=np.float32)
    innovation_z = np.full(n, np.nan, dtype=np.float32)
    posterior_variance = np.full(n, np.nan, dtype=np.float32)
    prior_variance = np.full(n, np.nan, dtype=np.float32)
    gain = np.full(n, np.nan, dtype=np.float32)
    level_delta = np.full(n, np.nan, dtype=np.float32)
    # Unit R is intentional: the half-life controls gain.  The adaptive
    # innovation scale supplies dimensionless surprise while keeping levels
    # in their native, interpretable units.
    current = np.nan
    variance = 1.0
    innovation_scale2 = 1.0
    scale_alpha = 1.0 - math.exp(-math.log(2.0) / max(24.0, half_life_days * 12.0))
    previous_level = np.nan
    for i, value in enumerate(values):
        if not np.isfinite(value):
            continue
        if not np.isfinite(current):
            current = float(value)
            variance = 1.0
            previous_level = current
            prior[i] = current
            level[i] = current
            posterior_variance[i] = variance
            prior_variance[i] = variance
            gain[i] = 1.0
            innovation[i] = 0.0
            innovation_z[i] = 0.0
            level_delta[i] = 0.0
            continue
        predicted_variance = variance + q
        predicted = current
        residual = float(value) - predicted
        current_gain = predicted_variance / (predicted_variance + 1.0)
        current = predicted + current_gain * residual
        variance = (1.0 - current_gain) * predicted_variance
        prior[i] = predicted
        level[i] = current
        innovation[i] = residual
        innovation_z[i] = residual / math.sqrt(max(innovation_scale2, 1e-8))
        prior_variance[i] = predicted_variance
        posterior_variance[i] = variance
        gain[i] = current_gain
        level_delta[i] = current - previous_level
        previous_level = current
        innovation_scale2 = (1.0 - scale_alpha) * innovation_scale2 + scale_alpha * residual * residual
    return {
        "kalman_level": level, "kalman_prior": prior, "kalman_innovation": innovation,
        "kalman_innovation_z": innovation_z, "posterior_variance": posterior_variance,
        "prior_variance": prior_variance, "kalman_gain": gain, "level_delta": level_delta,
    }


def _online_mahalanobis(matrix: np.ndarray, half_life_days: int = 21) -> np.ndarray:
    """Prior-only EW covariance surprise with diagonal shrinkage."""
    n, dim = matrix.shape
    result = np.full(n, np.nan, dtype=np.float32)
    mean = np.zeros(dim, dtype=float)
    cov = np.eye(dim, dtype=float)
    alpha = 1.0 - math.exp(-math.log(2.0) / (half_life_days * 24.0))
    count = 0
    for i in range(n):
        row = matrix[i]
        valid = np.isfinite(row)
        if int(valid.sum()) < max(3, dim // 3):
            continue
        x = np.where(valid, row, mean)
        delta = x - mean
        if count >= max(48, dim * 2):
            diagonal = np.diag(np.diag(cov))
            regular = 0.85 * cov + 0.15 * diagonal + np.eye(dim) * 1e-4
            try:
                value = float(delta @ np.linalg.solve(regular, delta))
                result[i] = math.sqrt(max(0.0, value / max(1, int(valid.sum()))))
            except np.linalg.LinAlgError:
                result[i] = np.nan
        mean = (1.0 - alpha) * mean + alpha * x
        cov = (1.0 - alpha) * cov + alpha * np.outer(delta, delta)
        count += 1
    return result


def _transition_no_kalman(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    series = pd.Series(values)
    # Curvature in the 1d/3d and 3d/7d realised movements.  These are direct
    # past-only lags, deliberately distinct from the Kalman transition.
    d1 = series - series.shift(24)
    d3 = series.shift(24) - series.shift(72)
    d7 = series.shift(72) - series.shift(168)
    return (d1 - d3).to_numpy(np.float32), (d3 - d7).to_numpy(np.float32)


def _build_states(series: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Accumulate columns in a dictionary.  Constructing a DataFrame one column
    # at a time creates thousands of fragmented blocks and makes the full
    # 20-month lattice needlessly slow and memory-hungry.
    output: dict[str, np.ndarray] = {}
    dictionary: list[dict[str, object]] = []
    fast_z: dict[tuple[int, int], list[np.ndarray]] = {pair: [] for pair in PAIRS}
    names: list[str] = []
    for semantic, source, family in DRIVERS:
        values = series[source].to_numpy(float)
        d13, d37 = _transition_no_kalman(values)
        for name, value, kind in (
            (f"ms_transition_{semantic}_delta_1d_minus_3d", d13, "direct_transition"),
            (f"ms_transition_{semantic}_delta_3d_minus_7d", d37, "direct_transition"),
        ):
            output[name] = value
            dictionary.append({"feature": name, "semantic_state": semantic, "source_field": source, "family": family, "kind": kind, "pair": None})
        for pair in PAIRS:
            fast, slow = pair
            fast_result = _kalman(values, fast)
            slow_result = _kalman(values, slow)
            prefix_fast = f"ms_kf_{semantic}_fast{fast}d_slow{slow}d"
            prefix_slow = f"ms_kf_{semantic}_slow{slow}d_fast{fast}d"
            for prefix, result, role in ((prefix_fast, fast_result, "fast"), (prefix_slow, slow_result, "slow")):
                for suffix, value in result.items():
                    name = f"{prefix}_{suffix}"
                    output[name] = value
                    dictionary.append({"feature": name, "semantic_state": semantic, "source_field": source, "family": family, "kind": suffix, "pair": f"{fast}d_{slow}d", "role": role})
            for suffix in ("kalman_level", "kalman_innovation_z", "level_delta"):
                name = f"ms_transition_{semantic}_fast{fast}d_slow{slow}d_{suffix}"
                output[name] = (fast_result[suffix] - slow_result[suffix]).astype(np.float32)
                dictionary.append({"feature": name, "semantic_state": semantic, "source_field": source, "family": family, "kind": "kalman_fast_minus_slow", "pair": f"{fast}d_{slow}d"})
            denominator = np.sqrt(np.maximum(np.abs(fast_result["kalman_level"]) + np.abs(slow_result["kalman_level"]), 1e-6))
            name = f"ms_transition_{semantic}_fast{fast}d_slow{slow}d_level_normalized"
            output[name] = ((fast_result["kalman_level"] - slow_result["kalman_level"]) / denominator).astype(np.float32)
            dictionary.append({"feature": name, "semantic_state": semantic, "source_field": source, "family": family, "kind": "kalman_fast_slow_normalized", "pair": f"{fast}d_{slow}d"})
            fast_z[pair].append(fast_result["kalman_innovation_z"])
        names.append(semantic)
    for fast, slow in PAIRS:
        matrix = np.column_stack(fast_z[(fast, slow)]).astype(float)
        finite = np.isfinite(matrix)
        abs_matrix = np.abs(matrix)
        pair_name = f"ms_global_fast{fast}d_slow{slow}d"
        magnitude = np.sqrt(np.nanmean(matrix * matrix, axis=1))
        breadth = np.nanmean(abs_matrix > 1.0, axis=1)
        dispersion = np.nanpercentile(matrix, 75, axis=1) - np.nanpercentile(matrix, 25, axis=1)
        output[f"{pair_name}_innovation_magnitude"] = magnitude.astype(np.float32)
        output[f"{pair_name}_innovation_breadth"] = breadth.astype(np.float32)
        output[f"{pair_name}_innovation_dispersion"] = dispersion.astype(np.float32)
        output[f"{pair_name}_innovation_available_fraction"] = finite.mean(axis=1).astype(np.float32)
        output[f"{pair_name}_innovation_mahalanobis"] = _online_mahalanobis(matrix)
        for suffix in ("innovation_magnitude", "innovation_breadth", "innovation_dispersion", "innovation_available_fraction", "innovation_mahalanobis"):
            dictionary.append({"feature": f"{pair_name}_{suffix}", "semantic_state": "global", "source_field": "|".join(names), "family": "global_innovation", "kind": suffix, "pair": f"{fast}d_{slow}d"})
    frame = pd.DataFrame(output)
    frame.insert(0, "__decision_ts__", series.index.to_numpy())
    return frame.reset_index(drop=True), pd.DataFrame(dictionary)


def _write(root: Path, states: pd.DataFrame, audit: pd.DataFrame, dictionary: pd.DataFrame, *, roots: tuple[Path, ...], months: tuple[pd.Timestamp, ...]) -> None:
    root.mkdir(parents=True, exist_ok=False)
    states.to_parquet(root / "market_state_hourly.parquet", index=False)
    audit.to_parquet(root / "market_state_source_audit.parquet", index=False)
    dictionary.to_parquet(root / "market_state_feature_dictionary.parquet", index=False)
    coverage = pd.DataFrame({
        "feature": [column for column in states.columns if column != "__decision_ts__"],
        "finite_fraction": [float(np.isfinite(pd.to_numeric(states[column], errors="coerce")).mean()) for column in states.columns if column != "__decision_ts__"],
        "n_unique": [int(pd.Series(states[column]).nunique(dropna=True)) for column in states.columns if column != "__decision_ts__"],
    })
    coverage.to_parquet(root / "feature_coverage.parquet", index=False)
    correctness = {
        "schema": SCHEMA,
        "target_free_market_sources": True,
        "market_fields_verified_timestamp_invariant": bool(audit.max_within_timestamp_nunique.le(1).all()),
        "state_is_hourly_and_pre_candidate_join": True,
        "kalman_parameterised_by_half_life_only": True,
        "fast_slow_pairs_predeclared": list(PAIRS),
        "direct_transition_uses_only_past_lags": True,
        "mahalanobis_covariance_is_prior_only": True,
        "no_policy_or_path_source_opened": True,
        "no_live_or_exchange_operation": True,
    }
    _once(root / "correctness_report.json", correctness)
    manifest = {
        "schema": SCHEMA,
        "scope": "offline target-free market-state research only",
        "months": [f"{value:%Y-%m}" for value in months],
        "raw_feature_roots": [str(path.relative_to(ROOT)) for path in roots],
        "raw_feature_root_sha256": {str(path.relative_to(ROOT)): _sha(path / "run_manifest.json") if (path / "run_manifest.json").exists() else None for path in roots},
        "drivers": [{"semantic_state": semantic, "source_field": source, "family": family} for semantic, source, family in DRIVERS],
        "fast_slow_pairs_days": list(PAIRS),
        "state_rows": int(len(states)),
        "state_features": int(len(states.columns) - 1),
        "seed": SEED,
        "correctness": correctness,
    }
    _once(root / "run_manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-feature-roots", required=True, help="comma-separated full causal feature roots")
    parser.add_argument("--start-month", default="2024-12")
    parser.add_argument("--end-month-exclusive", default="2026-08")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    roots = _parse_roots(args.raw_feature_roots)
    months = _month_range(_month(args.start_month), _month(args.end_month_exclusive))
    output = ROOT / args.out
    if output.exists():
        raise FileExistsError(output)
    series, audit = _shared_market_series(roots, months)
    states, dictionary = _build_states(series)
    _write(output, states, audit, dictionary, roots=roots, months=months)
    print(json.dumps({"out": str(output), "rows": len(states), "features": len(states.columns) - 1, "drivers": len(DRIVERS)}, sort_keys=True))


if __name__ == "__main__":
    main()
