#!/usr/bin/env python3
"""Build causal hourly features and final candidates for the 2022 PI grid.

The input minute data must already have been collected through the immutable
Kraken execution-1m store.  Hourly bars are right-labelled: a feature stamped
at ``t`` uses only minute bars in ``[t-1h, t)``.  Candidate decisions happen
one hour later, so every emitted feature is pre-entry.

This is a separate inverse-perpetual research population.  Returns are
quote-notional price returns, not inverse-collateral ROE, and the artifact must
not be pooled silently with the later USD-linear PF base-top-30 population.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.data_store import (
    PartitionedOHLCVStore,
    read_kraken_execution_1m,
)
from scripts.materialize_historical_backcast_exact1m_label_inputs import (
    wilder_atr_fraction,
)


SCHEMA = "kraken_inverse_market_grid_feature_candidates_v1"
EVIDENCE_SCOPE = "inverse_pi_market_grid_causal_features_research"
POPULATION_LINEAGE = "jan_jul_2022_inverse_pi_market_grid_causal_features_v1"
PRODUCT_LINEAGE = "kraken_inverse_pi_exact_product_binding_v1"
PRODUCTS = {
    "BTC/USD:BTC": "PI_XBTUSD",
    "ETH/USD:ETH": "PI_ETHUSD",
    "LTC/USD:LTC": "PI_LTCUSD",
    "XRP/USD:XRP": "PI_XRPUSD",
    "BCH/USD:BCH": "PI_BCHUSD",
}
POLICY_BY_SIDE = {
    # materialize_historical_backcast_exact1m_label_inputs.py constructs the
    # deployed geometry key as ``side_name + "__" + archetype_policy_key``.
    # This market-grid population has no causal archetype classifier, so bind
    # it explicitly to the frozen side-parent geometries instead of inventing
    # an archetype assignment.
    "long": "parent",
    "short": "parent",
}
SIDE_SIGN = {"long": 1.0, "short": -1.0}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _utc(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError("timestamps must carry an explicit timezone")
    return stamp.tz_convert("UTC")


def _rolling_slope(values: pd.Series, window: int) -> pd.Series:
    """Least-squares slope divided by the contemporaneous level."""

    x = np.arange(window, dtype=np.float64)
    centered = x - x.mean()
    denominator = float(np.square(centered).sum())

    def slope(array: np.ndarray) -> float:
        level = float(array[-1])
        if not np.isfinite(array).all() or not np.isfinite(level) or level <= 0:
            return np.nan
        return float(np.dot(centered, array - array.mean()) / denominator / level)

    return values.rolling(window, min_periods=window).apply(slope, raw=True)


def _path_efficiency(close: pd.Series, window: int) -> pd.Series:
    displacement = close.diff(window).abs()
    travel = close.diff().abs().rolling(window, min_periods=window).sum()
    return displacement.div(travel.replace(0.0, np.nan))


def _hourly_from_exact_minutes(
    minute: pd.DataFrame,
    *,
    required_start: pd.Timestamp,
    required_end: pd.Timestamp,
) -> pd.DataFrame:
    work = minute.loc[
        (minute.index >= required_start - pd.Timedelta(hours=1))
        & (minute.index < required_end)
    ].copy()
    if work.empty:
        raise ValueError("exact minute input is empty")
    work.index = pd.to_datetime(work.index, utc=True, errors="raise").floor("min")
    if work.index.duplicated().any():
        raise ValueError("exact minute input contains duplicate timestamps")
    expected = pd.date_range(
        required_start - pd.Timedelta(hours=1),
        required_end,
        freq="1min",
        inclusive="left",
        tz="UTC",
    )
    if not work.index.equals(expected):
        missing = expected.difference(work.index)
        extra = work.index.difference(expected)
        raise ValueError(
            "hourly feature source is not an uninterrupted exact-minute panel: "
            f"missing={len(missing)} extra={len(extra)}"
        )
    numeric = work.loc[:, ["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if (
        not np.isfinite(numeric.to_numpy(float)).all()
        or (numeric[["open", "high", "low", "close"]] <= 0.0).any().any()
    ):
        raise ValueError("exact minute input has invalid OHLCV")
    grouped = numeric.resample("1h", closed="left", label="right")
    hourly = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        minute_rows=("close", "size"),
    )
    hourly = hourly.loc[
        (hourly.index >= required_start) & (hourly.index <= required_end)
    ]
    if not hourly["minute_rows"].eq(60).all():
        raise ValueError("resampled hourly source does not contain 60 exact minutes")
    return hourly.drop(columns="minute_rows")


def _asset_features(hourly: pd.DataFrame) -> pd.DataFrame:
    close = hourly["close"].astype(float)
    log_return = np.log(close).diff()
    output = hourly.copy()
    for hours in (1, 4, 12, 24, 72, 168):
        output[f"ret_{hours}h"] = close.pct_change(hours, fill_method=None)
    for hours in (6, 24, 72):
        output[f"rv_{hours}h"] = (
            log_return.rolling(hours, min_periods=hours).std(ddof=0)
            * np.sqrt(float(hours))
        )
    output["downside_rv_24h"] = (
        log_return.clip(upper=0.0).rolling(24, min_periods=24).std(ddof=0)
        * np.sqrt(24.0)
    )
    output["atr_fraction_14h"] = wilder_atr_fraction(
        output["high"].to_numpy(float),
        output["low"].to_numpy(float),
        close.to_numpy(float),
        period=14,
    )
    for hours in (12, 24, 72):
        high = output["high"].rolling(hours, min_periods=hours).max()
        low = output["low"].rolling(hours, min_periods=hours).min()
        output[f"range_{hours}h_fraction"] = (high - low).div(close)
        output[f"drawdown_from_{hours}h_high"] = close.div(high) - 1.0
        output[f"recovery_from_{hours}h_low"] = close.div(low) - 1.0
        output[f"trend_slope_{hours}h"] = _rolling_slope(close, hours)
        output[f"path_efficiency_{hours}h"] = _path_efficiency(close, hours)
    volume = output["volume"].astype(float)
    for hours in (24, 72):
        mean = volume.rolling(hours, min_periods=hours).mean()
        std = volume.rolling(hours, min_periods=hours).std(ddof=0)
        output[f"volume_z_{hours}h"] = (volume - mean).div(
            std.replace(0.0, np.nan)
        )
    output["jump_intensity_24h"] = (
        log_return.abs()
        .div(log_return.rolling(72, min_periods=72).std(ddof=0))
        .rolling(24, min_periods=24)
        .max()
    )
    return output


def _market_context(asset: pd.DataFrame) -> pd.DataFrame:
    """Cross-asset and transition-dynamic features at each signal hour."""

    use = asset.reset_index(names="signal_timestamp")
    grouped = use.groupby("signal_timestamp", sort=True, observed=True)
    market = grouped.agg(
        market_median_ret_1h=("ret_1h", "median"),
        market_median_ret_4h=("ret_4h", "median"),
        market_median_ret_24h=("ret_24h", "median"),
        market_dispersion_1h=("ret_1h", "std"),
        market_dispersion_4h=("ret_4h", "std"),
        market_median_rv_24h=("rv_24h", "median"),
        market_median_atr_fraction=("atr_fraction_14h", "median"),
    )
    for horizon in (1, 4, 24):
        breadth = grouped[f"ret_{horizon}h"].apply(
            lambda values: float(pd.to_numeric(values, errors="coerce").gt(0).mean())
        )
        market[f"market_breadth_up_{horizon}h"] = breadth
        market[f"market_negative_breadth_{horizon}h"] = 1.0 - breadth

    returns = use.pivot(
        index="signal_timestamp", columns="symbol", values="ret_1h"
    ).sort_index()
    market["market_average_pair_corr_24h"] = np.nan
    for position, stamp in enumerate(returns.index):
        if position + 1 < 24:
            continue
        correlation = returns.iloc[position - 23 : position + 1].corr()
        values = correlation.to_numpy(float)
        upper = values[np.triu_indices_from(values, k=1)]
        market.at[stamp, "market_average_pair_corr_24h"] = (
            float(np.nanmean(upper)) if np.isfinite(upper).any() else np.nan
        )
    btc = (
        use.loc[use["symbol"].eq("BTC/USD:BTC")]
        .set_index("signal_timestamp")["ret_24h"]
        .reindex(market.index)
    )
    market["btc_minus_alt_median_ret_24h"] = (
        btc - market["market_median_ret_24h"]
    )
    transition_bases = (
        "market_median_rv_24h",
        "market_dispersion_1h",
        "market_negative_breadth_4h",
        "market_average_pair_corr_24h",
        "btc_minus_alt_median_ret_24h",
    )
    for name in transition_bases:
        series = market[name]
        market[f"transition_raw__{name}__delta_1h"] = series.diff(1)
        market[f"transition_raw__{name}__delta_6h"] = series.diff(6)
        market[f"transition_raw__{name}__acceleration_1h"] = series.diff().diff()
        market[f"transition_raw__{name}__cumulative_change_24h"] = series.diff(24)
        rolling = series.rolling(72, min_periods=72)
        market[f"transition_raw__{name}__z_72h"] = (
            (series - rolling.mean()).div(rolling.std(ddof=0).replace(0.0, np.nan))
        )
    return market


def build_feature_candidates(
    minute_by_symbol: dict[str, pd.DataFrame],
    *,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    warmup_days: int,
    cadence_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if set(minute_by_symbol) != set(PRODUCTS):
        raise ValueError("inverse minute sources do not match the frozen product set")
    feature_start = start - pd.Timedelta(days=int(warmup_days))
    hourly_end = end_exclusive + pd.Timedelta(hours=12)
    parts: list[pd.DataFrame] = []
    hourly_parts: list[pd.DataFrame] = []
    for symbol in sorted(PRODUCTS):
        hourly = _hourly_from_exact_minutes(
            minute_by_symbol[symbol],
            required_start=feature_start,
            required_end=hourly_end,
        )
        features = _asset_features(hourly)
        features["symbol"] = symbol
        parts.append(features)
        hourly_parts.append(
            hourly.assign(symbol=symbol).reset_index(names="ts")
        )
    asset = pd.concat(parts).sort_index()
    market = _market_context(asset)
    asset = (
        asset.reset_index(names="signal_timestamp")
        .merge(
            market.reset_index(names="signal_timestamp"),
            on="signal_timestamp",
            how="left",
            validate="many_to_one",
        )
        .set_index("signal_timestamp")
        .sort_index()
    )
    signals = pd.date_range(
        start,
        end_exclusive,
        freq=f"{int(cadence_hours)}h",
        inclusive="left",
        tz="UTC",
    )
    asset = asset.loc[asset.index.isin(signals)].copy()
    expected_asset_rows = len(signals) * len(PRODUCTS)
    if len(asset) != expected_asset_rows:
        raise ValueError(
            f"expected {expected_asset_rows} asset-hours, got {len(asset)}"
        )
    required_finite = [
        "ret_12h",
        "ret_24h",
        "rv_24h",
        "atr_fraction_14h",
        "market_median_ret_24h",
        "market_median_rv_24h",
        "transition_raw__market_median_rv_24h__z_72h",
    ]
    if not np.isfinite(asset[required_finite].to_numpy(float)).all():
        raise ValueError("warmup is insufficient for the declared causal features")

    candidates: list[pd.DataFrame] = []
    for side, sign in SIDE_SIGN.items():
        local = asset.copy()
        local["__ts__"] = local.index
        local["__symbol__"] = local["symbol"].astype(str)
        local["side_name"] = side
        local["base_score"] = (
            sign
            * local["ret_12h"]
            .div(local["atr_fraction_14h"].replace(0.0, np.nan))
            .clip(-12.0, 12.0)
        ).astype(np.float32)
        local["__barrier_pct__"] = (
            1.5 * local["atr_fraction_14h"]
        ).clip(0.005, 0.05).astype(np.float32)
        local["archetype_policy_key"] = POLICY_BY_SIDE[side]
        local["policy_archetype_assignment_source"] = (
            "explicit_deployed_side_parent_inverse_grid"
        )
        local["selected_for_monitor"] = True
        local["evidence_scope"] = EVIDENCE_SCOPE
        local["candidate_population_lineage"] = POPULATION_LINEAGE
        local["source_product_lineage"] = PRODUCT_LINEAGE
        local["bootstrap_barrier_data_acquisition_only"] = False
        local["product_id"] = local["__symbol__"].map(PRODUCTS)
        local["source_product_id"] = local["product_id"]
        local["source_contract_family"] = "PI"
        candidates.append(local.reset_index(drop=True))
    candidate = pd.concat(candidates, ignore_index=True)
    candidate["historical_rank"] = candidate.groupby(
        ["__ts__", "side_name"], observed=True
    )["base_score"].rank(method="average", pct=True)
    candidate["score_meta_base_soft_label"] = np.nan
    candidate = candidate.sort_values(
        ["__ts__", "__symbol__", "side_name"], kind="mergesort"
    ).reset_index(drop=True)
    if candidate.duplicated(["__ts__", "__symbol__", "side_name"]).any():
        raise ValueError("paired market-grid identities are not unique")
    if len(candidate) != 2 * expected_asset_rows:
        raise ValueError("paired market-grid candidate count is incomplete")
    drop = {"open", "high", "low", "close", "volume", "symbol"}
    candidate = candidate.drop(columns=sorted(drop & set(candidate.columns)))
    hourly_panel = pd.concat(hourly_parts, ignore_index=True).sort_values(
        ["ts", "symbol"], kind="mergesort"
    )
    return candidate, hourly_panel


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = _utc(args.start)
    end = _utc(args.end_exclusive)
    if end <= start:
        raise ValueError("--end-exclusive must be later than --start")
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    minute_start = start - pd.Timedelta(days=int(args.warmup_days)) - pd.Timedelta(hours=1)
    minute_end = end + pd.Timedelta(hours=12)
    minute_by_symbol = {
        symbol: read_kraken_execution_1m(
            args.data_root,
            symbol,
            start=minute_start,
            end=minute_end - pd.Timedelta(minutes=1),
        )
        for symbol in PRODUCTS
    }
    candidates, hourly = build_feature_candidates(
        minute_by_symbol,
        start=start,
        end_exclusive=end,
        warmup_days=int(args.warmup_days),
        cadence_hours=int(args.cadence_hours),
    )
    output.mkdir(parents=True, exist_ok=False)
    shard_dir = output / "candidate_shards"
    shard_dir.mkdir()
    records: list[dict[str, Any]] = []
    for month, group in candidates.groupby(
        candidates["__ts__"].dt.strftime("%Y%m"), sort=True
    ):
        path = shard_dir / f"candidates_{month}.parquet"
        group.to_parquet(path, index=False, compression="zstd")
        records.append(
            {
                "month": month,
                "path": str(path.resolve()),
                "rows": int(len(group)),
                "sha256": _sha256(path),
            }
        )
    hourly_path = output / "causal_hourly_ohlcv.parquet"
    hourly.to_parquet(hourly_path, index=False, compression="zstd")
    hourly_store_root = output / "hourly_store"
    hourly_store = PartitionedOHLCVStore(str(hourly_store_root), timeframe="1h")
    for symbol, group in hourly.groupby("symbol", sort=True):
        local = group.drop(columns="symbol").set_index("ts").sort_index()
        hourly_store.save_partitioned(str(symbol), local, defer_compact=True)
    feature_columns = [
        column
        for column in candidates.columns
        if column
        not in {
            "__ts__",
            "__symbol__",
            "side_name",
            "__barrier_pct__",
            "archetype_policy_key",
            "policy_archetype_assignment_source",
            "selected_for_monitor",
            "evidence_scope",
            "candidate_population_lineage",
            "source_product_lineage",
            "bootstrap_barrier_data_acquisition_only",
            "product_id",
            "source_product_id",
            "source_contract_family",
            "base_score",
            "historical_rank",
            "score_meta_base_soft_label",
        }
    ]
    manifest = {
        "schema": SCHEMA,
        "status": "causal_feature_candidates_materialized",
        "evidence_scope": EVIDENCE_SCOPE,
        "candidate_population_lineage": POPULATION_LINEAGE,
        "product_lineage": PRODUCT_LINEAGE,
        "bootstrap_barrier_data_acquisition_only": False,
        "promotion_eligible": False,
        "execution_parity_claim": False,
        "population_comparability": (
            "must remain separate from USD-linear PF and frozen-base populations"
        ),
        "contract_family": "PI_inverse_perpetual",
        "return_unit": "quote_notional_price_return_not_inverse_collateral_roe",
        "products": PRODUCTS,
        "period": {
            "start": start,
            "end_exclusive": end,
            "warmup_days": int(args.warmup_days),
            "cadence_hours": int(args.cadence_hours),
        },
        "timing": {
            "hourly_bar": "[t-1h,t) labelled t",
            "feature_available_at": "signal_timestamp",
            "execution_decision": "signal_timestamp+1h",
            "path": "[decision,decision+12h)",
        },
        "candidate_rows": int(len(candidates)),
        "rows_by_side": {
            str(side): int(rows)
            for side, rows in candidates["side_name"].value_counts().items()
        },
        "signal_hours": int(candidates["__ts__"].nunique()),
        "feature_columns": feature_columns,
        "transition_feature_columns": [
            column for column in feature_columns if column.startswith("transition_raw__")
        ],
        "barrier": "clip(1.5 * causal Wilder ATR14 fraction, 0.5%, 5%)",
        "base_score": "side-signed 12h return / causal ATR14; population includes every row",
        "shards": records,
        "outputs": {
            "causal_hourly_ohlcv": {
                "path": str(hourly_path.resolve()),
                "rows": int(len(hourly)),
                "sha256": _sha256(hourly_path),
            },
            "hourly_store_root": {
                "path": str(hourly_store_root.resolve()),
                "contract": "PartitionedOHLCVStore_1h_right_labelled",
            },
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-root", type=Path, default=Path("data_perp"))
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--start", default="2022-01-01T00:00:00Z")
    result.add_argument("--end-exclusive", default="2022-08-01T00:00:00Z")
    result.add_argument("--warmup-days", type=int, default=30)
    result.add_argument("--cadence-hours", type=int, default=1)
    return result


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2, sort_keys=True, default=str))
