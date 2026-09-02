#!/usr/bin/env python3
"""Materialise additional resolved H12 *market-state* labels at timestamps.

This is a label-only producer.  It uses observed 15-minute bars to form
market-wide future outcomes, records a decision+12h availability timestamp,
and never writes a scoring/inference input.  Unlike an asset label table,
one row is emitted per decision timestamp because every target below has
market-state semantics shared by all candidates at that timestamp.

The expensive dependence and structural labels use predeclared, deterministic
proxies (equicorrelation/one-factor moments and a six-variable state vector)
instead of all-pair correlation matrices.  This keeps the materialisation
bounded while preserving the intended economic geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from materialize_strict_r3_o3v2_market_dynamics_labels import H, MIN_ASSETS, PRE, _panel


EPS = 1e-8
BATCH = 256
SCHEMA = "strict_r3_o3v2_market_dynamics_extended_timestamp_labels_v1"
DEFAULT_BASE = ROOT / "data_perp/artifacts/strict_r3_o3v2_market_dynamics_labels_20260825_v2/market_dynamics_labels.parquet"
DEFAULT_BARS = ROOT / "15m_ohlcv_perp"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_exclusive(path: Path, value: object) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)


def _first_hit(condition: np.ndarray) -> np.ndarray:
    """Return first matching 15m step in hours; censor at the H12 horizon."""
    hit = np.asarray(condition, dtype=bool)
    any_hit = hit.any(axis=1)
    return np.where(any_hit, (hit.argmax(axis=1) + 1) / 4.0, float(H / 4))


def _rank_matrix(values: np.ndarray) -> np.ndarray:
    """Stable ordinal ranks for a finite row-major matrix."""
    order = np.argsort(values, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    rows = np.arange(values.shape[0])[:, None]
    ranks[rows, order] = np.arange(values.shape[1], dtype=np.float64)
    return ranks


def _equicorrelation(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return equicorrelation proxy, PC1-share proxy, and mean factor R².

    The proxy uses standardized asset returns and the identity linking the
    variance of their cross-sectional sum to average pairwise correlation. It
    avoids an O(N²) matrix at every timestamp while retaining common-factor
    versus idiosyncratic-state information.
    """
    x = np.where(np.isfinite(returns), returns, np.nan)
    mean = np.nanmean(x, axis=1, keepdims=True)
    sd = np.nanstd(x, axis=1, keepdims=True)
    z = (x - mean) / np.maximum(sd, EPS)
    z = np.where(np.isfinite(z), z, 0.0)
    count = np.isfinite(x).sum(axis=2).astype(float)
    sum_z = z.sum(axis=2)
    numerator = np.sum(sum_z * sum_z - count, axis=1)
    denominator = np.sum(count * np.maximum(count - 1.0, 0.0), axis=1)
    rho = np.clip(numerator / np.maximum(denominator, EPS), -1.0, 1.0)
    n = np.maximum(np.nanmedian(count, axis=1), 2.0)
    pc1 = np.clip((1.0 + (n - 1.0) * rho) / n, 0.0, 1.0)
    mkt = np.nanmedian(x, axis=2, keepdims=True)
    xm = x - mean
    mm = mkt - np.nanmean(mkt, axis=1, keepdims=True)
    covariance = np.nansum(xm * mm, axis=1)
    var_x = np.nansum(xm * xm, axis=1)
    var_m = np.nansum(mm * mm, axis=1)
    r2 = np.nanmean(np.clip(covariance * covariance / np.maximum(var_x * var_m, EPS), 0.0, 1.0), axis=1)
    return rho, pc1, r2


def _state_metrics(pre: np.ndarray, future: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mahalanobis shift, standardized moment break, and first change point."""
    b, _, dimensions = pre.shape
    shift = np.full(b, np.nan, dtype=float)
    distribution = np.full(b, np.nan, dtype=float)
    change_time = np.full(b, float(H / 4), dtype=float)
    for i in range(b):
        p = pre[i]
        f = future[i]
        if not np.isfinite(p).all() or not np.isfinite(f).all():
            continue
        mean = p.mean(axis=0)
        cov = np.cov(p, rowvar=False) + np.eye(dimensions) * 1e-5
        inverse = np.linalg.pinv(cov, hermitian=True)
        delta = f.mean(axis=0) - mean
        shift[i] = float(np.sqrt(max(delta @ inverse @ delta, 0.0)))
        scale = np.std(p, axis=0, ddof=1)
        distribution[i] = float(np.sqrt(np.mean(((f.mean(axis=0) - mean) / np.maximum(scale, EPS)) ** 2)))
        row_delta = f - mean
        distance = np.einsum("ij,jk,ik->i", row_delta, inverse, row_delta)
        change_time[i] = _first_hit((distance > 9.0)[None, :])[0]
    return shift, distribution, change_time


def _build_labels(index: pd.DatetimeIndex, close: np.ndarray, volume: np.ndarray, decisions: pd.DatetimeIndex) -> pd.DataFrame:
    returns = np.full_like(close, np.nan, dtype=np.float32)
    returns[1:] = close[1:] / close[:-1] - 1.0
    market = np.nanmedian(returns, axis=1)
    breadth = np.nanmean(returns > 0, axis=1)
    dispersion = np.nanstd(returns, axis=1)
    dollar = close * volume
    total_volume = np.nansum(dollar, axis=1)
    shares = dollar / np.maximum(total_volume[:, None], EPS)
    hhi = np.nansum(shares * shares, axis=1)
    rv16 = np.sqrt(pd.Series(market).rolling(16, min_periods=16).mean().pow(2).to_numpy())
    # The expression above would compute sqrt(mean)^2.  Use a direct rolling
    # mean of squared returns instead, retaining an explicit causal window.
    rv16 = np.sqrt(pd.Series(market * market).rolling(16, min_periods=16).mean().to_numpy())
    level = np.nancumsum(np.where(np.isfinite(market), market, 0.0))
    anchor = pd.Series(level).rolling(PRE, min_periods=PRE).mean().to_numpy()
    decision_lookup = pd.Series(np.arange(len(index), dtype=int), index=index)
    decision_positions = decision_lookup.reindex(decisions).to_numpy()
    if np.isnan(decision_positions).any():
        raise AssertionError("decision timestamps are not aligned to observed 15m grid")
    positions = decision_positions.astype(int)
    output: dict[str, np.ndarray] = {"__decision_ts__": decisions.to_numpy(), "market_label_valid": np.zeros(len(decisions), dtype=bool)}
    names = (
        "market_anchor_reversion_fraction_12h", "market_time_to_anchor_reentry_12h", "market_reversion_overshoot_12h",
        "market_vol_of_vol_12h", "market_compression_release_ratio_12h", "market_time_to_vol_breakout_12h", "market_low_vol_persistence_12h",
        "market_directional_breadth_12h", "market_breadth_persistence_12h",
        "cross_sectional_tail_spread_12h", "idiosyncratic_variance_share_12h",
        "market_pairwise_correlation_change_12h", "market_pc1_share_change_12h", "market_factor_r2_change_12h",
        "leader_continuation_12h", "cross_sectional_rank_persistence_12h", "topk_leadership_turnover_12h",
        "market_downside_upside_semivol_ratio_12h",
        "market_state_shift_12h", "market_distribution_break_12h", "market_time_to_change_point_12h",
    )
    output.update({name: np.full(len(decisions), np.nan, dtype=np.float32) for name in names})
    offsets_h = np.arange(1, H + 1)
    offsets_pre = np.arange(PRE, 0, -1)
    for start in range(0, len(positions), BATCH):
        stop = min(start + BATCH, len(positions))
        t = positions[start:stop]
        viable = (t >= PRE) & (t + H < len(index))
        if not viable.any():
            continue
        local = np.flatnonzero(viable)
        tv = t[local]
        future = returns[tv[:, None] + offsets_h]
        pre = returns[tv[:, None] - offsets_pre]
        future_market = np.nanmedian(future, axis=2)
        pre_market = np.nanmedian(pre, axis=2)
        valid = (
            np.isfinite(future_market).all(axis=1)
            & np.isfinite(pre_market).all(axis=1)
            & (np.isfinite(future).sum(axis=(1, 2)) >= H * MIN_ASSETS)
            & np.isfinite(anchor[tv])
            & np.isfinite(rv16[tv])
        )
        if not valid.any():
            continue
        valid_local = local[valid]
        q = tv[valid]
        fm = future_market[valid]
        pm = pre_market[valid]
        fut_level = level[q[:, None] + offsets_h]
        # Labels conditional on a stretch are undefined when the initial
        # displacement is essentially zero.  Use a small 5-bps market floor
        # solely to make the H12 normalization numerically well posed; the
        # subsequent stretch gate, not this floor, determines supervision.
        atr = np.maximum(rv16[q] * np.sqrt(16.0), 5e-4)
        d0 = level[q] - anchor[q]
        d_future = level[q + H] - anchor[q]
        sign_d = np.sign(d0)
        stretched = np.abs(d0) >= .25 * atr
        # Mean reversion / stretch resolution.
        reversion = (np.abs(d0) - np.abs(d_future)) / np.maximum(np.abs(d0), EPS)
        reentry = _first_hit(np.abs(fut_level - anchor[q, None]) < atr[:, None])
        overshoot = np.maximum(0.0, -sign_d[:, None] * (fut_level - anchor[q, None])).max(axis=1) / atr
        reversion[~stretched] = np.nan
        reentry[~stretched] = np.nan
        overshoot[~stretched] = np.nan
        output["market_anchor_reversion_fraction_12h"][start + valid_local] = reversion.astype(np.float32)
        output["market_time_to_anchor_reentry_12h"][start + valid_local] = reentry.astype(np.float32)
        output["market_reversion_overshoot_12h"][start + valid_local] = overshoot.astype(np.float32)
        # Volatility regime and compression/release.
        f_rv = rv16[q[:, None] + offsets_h]
        pre_rv = rv16[q[:, None] - np.arange(PRE)]
        high = np.nanquantile(pre_rv, 0.80, axis=1)
        low = np.nanquantile(pre_rv, 0.20, axis=1)
        output["market_vol_of_vol_12h"][start + valid_local] = (np.nanstd(f_rv, axis=1) / np.maximum(np.nanmean(f_rv, axis=1), EPS)).astype(np.float32)
        output["market_compression_release_ratio_12h"][start + valid_local] = np.log(np.maximum(np.nanmax(f_rv, axis=1), EPS) / np.maximum(rv16[q], EPS)).astype(np.float32)
        output["market_time_to_vol_breakout_12h"][start + valid_local] = _first_hit(f_rv > high[:, None]).astype(np.float32)
        output["market_low_vol_persistence_12h"][start + valid_local] = np.mean(f_rv < low[:, None], axis=1).astype(np.float32)
        # Breadth and cross-sectional dispersion.
        asset_future = close[q + H] / close[q] - 1.0
        asset_pre = close[q] / close[q - PRE] - 1.0
        market_sign = np.sign(np.nansum(fm, axis=1))
        output["market_directional_breadth_12h"][start + valid_local] = np.nanmean(np.sign(asset_future) == market_sign[:, None], axis=1).astype(np.float32)
        output["market_breadth_persistence_12h"][start + valid_local] = np.nanmean(np.sign(future[valid]) == np.sign(fm)[:, :, None], axis=(1, 2)).astype(np.float32)
        future_dispersion = np.nanstd(asset_future, axis=1)
        pre_dispersion = np.nanstd(asset_pre, axis=1)
        output["cross_sectional_tail_spread_12h"][start + valid_local] = ((np.nanquantile(asset_future, .90, axis=1) - np.nanquantile(asset_future, .10, axis=1)) / np.maximum(np.nanquantile(asset_pre, .90, axis=1) - np.nanquantile(asset_pre, .10, axis=1), EPS)).astype(np.float32)
        residual = future[valid] - fm[:, :, None]
        output["idiosyncratic_variance_share_12h"][start + valid_local] = (np.nanvar(residual, axis=(1, 2)) / np.maximum(np.nanvar(future[valid], axis=(1, 2)), EPS)).astype(np.float32)
        # Common-factor coupling; bounded, deterministic equicorrelation proxy.
        rho_future, pc1_future, r2_future = _equicorrelation(future[valid])
        rho_pre, pc1_pre, r2_pre = _equicorrelation(pre[valid])
        output["market_pairwise_correlation_change_12h"][start + valid_local] = (rho_future - rho_pre).astype(np.float32)
        output["market_pc1_share_change_12h"][start + valid_local] = (pc1_future - pc1_pre).astype(np.float32)
        output["market_factor_r2_change_12h"][start + valid_local] = (r2_future - r2_pre).astype(np.float32)
        # Leadership / rotation, using a pre-decision top-quintile cohort.
        assets = asset_future.shape[1]
        k = max(1, int(np.ceil(assets * .20)))
        pre_clean = np.where(np.isfinite(asset_pre), asset_pre, -np.inf)
        future_clean = np.where(np.isfinite(asset_future), asset_future, -np.inf)
        leaders_pre = np.argpartition(pre_clean, -k, axis=1)[:, -k:]
        leaders_future = np.argpartition(future_clean, -k, axis=1)[:, -k:]
        row = np.arange(len(q))[:, None]
        leader_mean = np.take_along_axis(asset_future, leaders_pre, axis=1).mean(axis=1)
        total_mean = np.nanmean(asset_future, axis=1)
        nonleader_mean = (assets * total_mean - k * leader_mean) / max(assets - k, 1)
        output["leader_continuation_12h"][start + valid_local] = (leader_mean - nonleader_mean).astype(np.float32)
        rank_pre = _rank_matrix(pre_clean)
        rank_future = _rank_matrix(future_clean)
        rank_pre = (rank_pre - rank_pre.mean(axis=1, keepdims=True)) / np.maximum(rank_pre.std(axis=1, keepdims=True), EPS)
        rank_future = (rank_future - rank_future.mean(axis=1, keepdims=True)) / np.maximum(rank_future.std(axis=1, keepdims=True), EPS)
        output["cross_sectional_rank_persistence_12h"][start + valid_local] = np.mean(rank_pre * rank_future, axis=1).astype(np.float32)
        overlap = np.array([len(np.intersect1d(a, b, assume_unique=False)) for a, b in zip(leaders_pre, leaders_future)], dtype=float)
        output["topk_leadership_turnover_12h"][start + valid_local] = (1.0 - overlap / float(k)).astype(np.float32)
        # Downside asymmetry and structural transition state.
        downside = np.sqrt(np.sum(np.minimum(fm, 0.0) ** 2, axis=1))
        upside = np.sqrt(np.sum(np.maximum(fm, 0.0) ** 2, axis=1))
        output["market_downside_upside_semivol_ratio_12h"][start + valid_local] = np.log(np.maximum(downside, EPS) / np.maximum(upside, EPS)).astype(np.float32)
        # A compact six-variable market-state representation.  Dependence is
        # represented by the deterministic equicorrelation proxy above.
        pre_rho, _, _ = _equicorrelation(pre[valid])
        future_rho, _, _ = _equicorrelation(future[valid])
        state_pre = np.stack((
            pm,
            breadth[q[:, None] - offsets_pre],
            dispersion[q[:, None] - offsets_pre],
            rv16[q[:, None] - offsets_pre],
            np.repeat(pre_rho[:, None], PRE, axis=1),
            np.repeat(np.log(np.maximum(total_volume[q], EPS))[:, None], PRE, axis=1),
        ), axis=2)
        state_future = np.stack((
            fm,
            breadth[q[:, None] + offsets_h],
            dispersion[q[:, None] + offsets_h],
            rv16[q[:, None] + offsets_h],
            np.repeat(future_rho[:, None], H, axis=1),
            np.repeat(np.log(np.maximum(total_volume[q + H], EPS))[:, None], H, axis=1),
        ), axis=2)
        state_shift, distribution, change_time = _state_metrics(state_pre, state_future)
        output["market_state_shift_12h"][start + valid_local] = state_shift.astype(np.float32)
        output["market_distribution_break_12h"][start + valid_local] = distribution.astype(np.float32)
        output["market_time_to_change_point_12h"][start + valid_local] = change_time.astype(np.float32)
        output["market_label_valid"][start + valid_local] = True
    frame = pd.DataFrame(output)
    frame["market_label_available_ts"] = frame["__decision_ts__"] + pd.Timedelta(hours=12)
    return frame


def run(*, base: Path, bars: Path, out: Path) -> Path:
    if out.exists():
        raise FileExistsError(out)
    base_frame = pd.read_parquet(base)
    for column in ("__decision_ts__", "market_label_available_ts"):
        base_frame[column] = pd.to_datetime(base_frame[column], utc=True, errors="raise")
    existing = base_frame.groupby("__decision_ts__", as_index=False, sort=True).first()
    decisions = pd.DatetimeIndex(existing["__decision_ts__"])
    index = pd.date_range(decisions.min().floor("15min") - pd.Timedelta(hours=24), decisions.max().ceil("15min") + pd.Timedelta(hours=12), freq="15min", tz="UTC")
    close, volume = _panel(bars, index)
    extended = _build_labels(index, close, volume, decisions)
    # Preserve all previously materialised labels alongside the new fields so
    # later context screens have one immutable, timestamp-level label source.
    existing = existing.drop(columns=[column for column in ("candidate_id", "side_name", "market_label_valid", "market_label_available_ts") if column in existing])
    result = existing.merge(extended, on="__decision_ts__", how="inner", validate="one_to_one")
    out.mkdir(parents=True, exist_ok=False)
    result.to_parquet(out / "market_dynamics_extended_timestamp_labels.parquet", index=False, compression="zstd")
    _write_exclusive(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "resolved future H12 labels only; explicitly prohibited from candidate scoring and inference",
        "base_labels": str(base.resolve()), "base_labels_sha256": _sha256(base), "bars": str(bars.resolve()),
        "horizon_hours": 12, "pre_window_hours": 24, "min_assets": MIN_ASSETS,
        "rows": int(len(result)), "valid_rows": int(result["market_label_valid"].sum()),
        "labels": sorted(column for column in result if column not in {"__decision_ts__", "market_label_valid", "market_label_available_ts", "candidate_id", "side_name"}),
        "proxies": {
            "dependence": "equicorrelation and PC1-share approximation from standardized asset-return panels",
            "structural": "six-variable state Mahalanobis/moment/change-point proxy",
            "derivatives_positioning": "not emitted here; it requires a separately audited common historical OI/funding source",
        },
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--bars", type=Path, default=DEFAULT_BARS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(run(base=args.base.resolve(), bars=args.bars.resolve(), out=args.out.resolve()))


if __name__ == "__main__":
    main()
