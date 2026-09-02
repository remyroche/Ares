#!/usr/bin/env python3
"""Build the schema-v2 point-in-time signal grid from causal hourly inputs.

The grid is independent of H12 path availability.  A symbol-hour exists when
the signal-hour close and the decision-hour open are both available at the
declared signal+one-hour entry time.  The frozen exact170 spread registry is
the only instrument-level admission input at this stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_canonical_v2 import (  # noqa: E402
    CandidateSpec,
    SCHEMA,
    build_point_in_time_candidates,
)
from scripts.run_tp6_sl4_exact170_canonical_consensus import (  # noqa: E402
    FROZEN_INPUT_BACKFILL_ROOT,
    _make_panel,
    _read_downloaded_15m_decision_open,
)
from extreme_price_movements.strict_r3_shadow_portfolio import (  # noqa: E402
    causal_decision_atr_from_15m,
    causal_flat_fill_omitted_15m,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _signal_hour_spread_panel(
    symbols: list[str],
    signal_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Load the causal signal-hour Kraken bid/ask spread.

    The historical universe registry decides membership only.  Actionability
    uses the official order-book analytics timestamp that precedes the
    signal+one-hour entry decision.  Missing/stale analytics stay missing and
    are rejected by ``build_point_in_time_candidates``.
    """
    values: dict[str, pd.Series] = {}
    for symbol in symbols:
        base = symbol.split("/", 1)[0]
        path = FROZEN_INPUT_BACKFILL_ROOT / f"{base}_USD_USD.parquet"
        if not path.exists():
            continue
        try:
            frame = pd.read_parquet(
                path, columns=["ob_bid_bestPrice", "ob_ask_bestPrice"],
            )
        except Exception:
            continue
        frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
        bid = pd.to_numeric(frame["ob_bid_bestPrice"], errors="coerce")
        ask = pd.to_numeric(frame["ob_ask_bestPrice"], errors="coerce")
        mid = 0.5 * (bid + ask)
        spread = (10_000.0 * (ask - bid) / mid.where(mid > 0.0)).replace(
            [np.inf, -np.inf], np.nan,
        )
        values[symbol] = spread.reindex(signal_index)
    return (
        pd.concat(values, axis=1).reindex(index=signal_index, columns=symbols)
        if values else pd.DataFrame(index=signal_index, columns=symbols, dtype=float)
    )


def _decision_open_panel(
    symbols: list[str],
    signal_index: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return entry prices and non-feature source lineage at decision time."""
    decision_index = signal_index + pd.Timedelta(hours=1)
    values: dict[str, pd.Series] = {}
    sources: dict[str, pd.Series] = {}
    fallback_volumes: dict[str, pd.Series] = {}
    for symbol in symbols:
        lineage = _read_downloaded_15m_decision_open(
            symbol, decision_index, return_lineage=True,
        )
        assert isinstance(lineage, pd.DataFrame)
        values[symbol] = lineage['decision_open'].set_axis(signal_index)
        sources[symbol] = lineage['decision_open_source'].set_axis(signal_index)
        fallback_volumes[symbol] = lineage['decision_open_hourly_volume'].set_axis(signal_index)
    return (
        pd.concat(values, axis=1).reindex(index=signal_index, columns=symbols),
        pd.concat(sources, axis=1).reindex(index=signal_index, columns=symbols),
        pd.concat(fallback_volumes, axis=1).reindex(index=signal_index, columns=symbols),
    )


def _decision_book_panel(
    symbols: list[str],
    signal_index: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return timestamp-exact official bid/ask for hourly-fallback validation.

    The book is only used to reject a stale coarse trade fallback.  It is
    execution lineage rather than a model feature and does not replace the
    frozen entry-price contract with a live quote.
    """
    decision_index = signal_index + pd.Timedelta(hours=1)
    bids: dict[str, pd.Series] = {}
    asks: dict[str, pd.Series] = {}
    for symbol in symbols:
        base = symbol.split('/', 1)[0]
        path = FROZEN_INPUT_BACKFILL_ROOT / f'{base}_USD_USD.parquet'
        if not path.exists():
            continue
        try:
            frame = pd.read_parquet(
                path, columns=['ob_bid_bestPrice', 'ob_ask_bestPrice'],
            )
            frame.index = pd.to_datetime(frame.index, utc=True, errors='raise')
            frame = frame.loc[~frame.index.duplicated(keep='last')].sort_index()
        except Exception:
            continue
        bids[symbol] = pd.to_numeric(
            frame['ob_bid_bestPrice'], errors='coerce',
        ).reindex(decision_index).set_axis(signal_index)
        asks[symbol] = pd.to_numeric(
            frame['ob_ask_bestPrice'], errors='coerce',
        ).reindex(decision_index).set_axis(signal_index)
    return (
        pd.concat(bids, axis=1).reindex(index=signal_index, columns=symbols)
        if bids else pd.DataFrame(index=signal_index, columns=symbols, dtype=float),
        pd.concat(asks, axis=1).reindex(index=signal_index, columns=symbols)
        if asks else pd.DataFrame(index=signal_index, columns=symbols, dtype=float),
    )


def _signal_atr_panel(
    symbols: list[str],
    signal_index: pd.DatetimeIndex,
    *,
    policy_bar_root: Path,
) -> pd.DataFrame:
    """Return the exact policy ATR available at each signal timestamp.

    This is execution readiness, not a model feature.  It deliberately uses
    the same 15-minute bar files and Wilder-14 implementation as the shadow
    portfolio so an admitted row cannot fail only after the auction.
    """
    values: dict[str, pd.Series] = {}
    required_end = signal_index.max() if len(signal_index) else None
    for symbol in symbols:
        stem = symbol.lower().replace("/", "").replace("_", "")
        path = policy_bar_root / f"{stem}_15m.parquet"
        if not path.exists():
            continue
        try:
            bars = pd.read_parquet(path, columns=["open", "high", "low", "close"])
            bars.index = pd.to_datetime(bars.index, utc=True, errors="raise")
            bars = bars.loc[~bars.index.duplicated(keep="last")].sort_index()
            if required_end is not None and not bars.empty and bars.index.min() <= required_end:
                bars = causal_flat_fill_omitted_15m(bars, end=required_end)
            atr = causal_decision_atr_from_15m(bars)
        except Exception:
            continue
        values[symbol] = pd.to_numeric(atr.reindex(signal_index), errors="coerce")
    return (
        pd.concat(values, axis=1).reindex(index=signal_index, columns=symbols)
        if values else pd.DataFrame(index=signal_index, columns=symbols, dtype=float)
    )


def _attach_execution_lineage(
    candidates: pd.DataFrame,
    lineage_values: pd.DataFrame,
) -> pd.DataFrame:
    """Persist exact entry price and ATR without making either a feature.

    The entry price is execution lineage, not model input.  Keeping it beside
    the candidate identity guarantees that the shadow/live portfolio consumes
    the exact value that made ``entry_executable`` true, including the frozen
    official-hourly fallback when an exact 15-minute cell is absent.
    """
    lineage_fields = [
        field for field in lineage_values.columns
        if field not in {'__ts__', '__symbol__'}
    ]
    if set(lineage_fields).intersection(candidates.columns):
        raise ValueError("candidate frame already contains execution lineage")
    lookup = lineage_values[["__ts__", "__symbol__", *lineage_fields]].copy()
    if lookup.duplicated(["__ts__", "__symbol__"]).any():
        raise ValueError("execution-lineage lookup must be unique by timestamp and symbol")
    return candidates.merge(
        lookup,
        on=["__ts__", "__symbol__"],
        how="left",
        validate="many_to_one",
    )


def _label_entry_price_rejections(candidates: pd.DataFrame) -> pd.DataFrame:
    """Make stale coarse-price failures explicit in the rejection audit."""
    result = candidates.copy()
    generic = result['eligibility_reason'].eq('entry_not_executable')
    source = result['decision_open_source'].fillna('unavailable').astype(str)
    result.loc[generic & source.eq('official_hourly_zero_volume'), 'eligibility_reason'] = (
        'entry_hourly_fallback_zero_volume'
    )
    result.loc[generic & source.eq('official_hourly_invalid'), 'eligibility_reason'] = (
        'entry_hourly_fallback_invalid'
    )
    result.loc[
        generic & source.eq('official_hourly_trade')
        & ~result['decision_open_fallback_valid'].fillna(False),
        'eligibility_reason',
    ] = 'entry_hourly_fallback_book_mismatch'
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--universe-csv", type=Path)
    source.add_argument(
        "--universe-manifest", type=Path,
        help="Prior immutable schema-v2 target-free manifest whose source_map keys freeze the universe",
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end-exclusive", required=True)
    parser.add_argument("--sides", default="long")
    parser.add_argument("--spread-limit-bps", type=float, default=100.0)
    parser.add_argument(
        "--policy-bar-root", type=Path, default=ROOT / "15m_ohlcv_perp",
        help="Frozen 15-minute policy bars used only for signal-time ATR readiness.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    if args.universe_manifest is not None:
        prior = json.loads(args.universe_manifest.read_text())
        if prior.get("schema") != f"{SCHEMA}_target_free_hourly_grid":
            raise ValueError("universe manifest is not a schema-v2 target-free grid")
        if float(prior.get("spread_limit_bps", np.nan)) != float(args.spread_limit_bps):
            raise ValueError("universe manifest uses a different spread limit")
        frozen_symbols = list((prior.get("source_map") or {}).keys())
        if not frozen_symbols:
            raise ValueError("universe manifest has no frozen source_map keys")
        # Membership, rather than an invented spread estimate, is the frozen
        # contract.  The boundary value makes every already-admitted member
        # pass the identical <= limit check without misrepresenting a new
        # measured spread.
        universe_table = pd.DataFrame({
            "symbol": frozen_symbols,
            "p90_spread_bps": float(args.spread_limit_bps),
        })
        universe_source = args.universe_manifest
        universe_source_type = "prior_schema_v2_admitted_membership"
    else:
        universe_table = pd.read_csv(args.universe_csv)
        if "p90_spread_bps" not in universe_table and "average_spread_bps" in universe_table:
            universe_table = universe_table.rename(columns={"average_spread_bps": "p90_spread_bps"})
        if "symbol" not in universe_table or "p90_spread_bps" not in universe_table:
            raise ValueError("universe CSV requires symbol and p90_spread_bps")
        universe_table = universe_table.loc[
            pd.to_numeric(universe_table["p90_spread_bps"], errors="coerce")
            .le(float(args.spread_limit_bps))
        ].copy()
        universe_source = args.universe_csv
        universe_source_type = "spread_registry"
    symbols = universe_table["symbol"].dropna().astype(str).drop_duplicates().tolist()
    start = pd.to_datetime(args.start, utc=True)
    end = pd.to_datetime(args.end_exclusive, utc=True)
    if start >= end:
        raise ValueError("start must precede end-exclusive")
    # Features and the signal close use completed hourly bars only. Entry
    # availability is sourced separately from the first 15-minute open at the
    # decision timestamp; it must not wait for that hour to complete.
    panel, source_map = _make_panel(symbols, start, end)
    close = panel["close"].reindex(columns=symbols)
    signal_index = close.index[(close.index >= start) & (close.index < end)]
    close = close.reindex(signal_index)
    decision_open, decision_open_source, decision_open_hourly_volume = _decision_open_panel(
        symbols, signal_index,
    )
    decision_bid, decision_ask = _decision_book_panel(symbols, signal_index)
    policy_bar_root = (
        args.policy_bar_root if args.policy_bar_root.is_absolute()
        else ROOT / args.policy_bar_root
    )
    signal_atr = _signal_atr_panel(
        symbols, signal_index, policy_bar_root=policy_bar_root,
    )
    market = close.stack(dropna=False).rename("signal_close").reset_index()
    market.columns = ["__ts__", "__symbol__", "signal_close"]
    decision_values = decision_open.stack(dropna=False).rename("decision_open").reset_index()
    decision_values.columns = ["__ts__", "__symbol__", "decision_open"]
    market = market.merge(
        decision_values, on=["__ts__", "__symbol__"], how="left", validate="one_to_one"
    )
    source_values = decision_open_source.stack(dropna=False).rename(
        'decision_open_source',
    ).reset_index()
    source_values.columns = ['__ts__', '__symbol__', 'decision_open_source']
    volume_values = decision_open_hourly_volume.stack(dropna=False).rename(
        'decision_open_hourly_volume',
    ).reset_index()
    volume_values.columns = ['__ts__', '__symbol__', 'decision_open_hourly_volume']
    bid_values = decision_bid.stack(dropna=False).rename('decision_book_bid').reset_index()
    bid_values.columns = ['__ts__', '__symbol__', 'decision_book_bid']
    ask_values = decision_ask.stack(dropna=False).rename('decision_book_ask').reset_index()
    ask_values.columns = ['__ts__', '__symbol__', 'decision_book_ask']
    for values in (source_values, volume_values, bid_values, ask_values):
        market = market.merge(
            values, on=['__ts__', '__symbol__'], how='left', validate='one_to_one',
        )
    atr_values = signal_atr.stack(dropna=False).rename("signal_atr").reset_index()
    atr_values.columns = ["__ts__", "__symbol__", "signal_atr"]
    market = market.merge(
        atr_values, on=["__ts__", "__symbol__"], how="left", validate="one_to_one",
    )
    current_spread = _signal_hour_spread_panel(symbols, signal_index)
    spread_values = current_spread.stack(dropna=False).rename("spread_bps").reset_index()
    spread_values.columns = ["__ts__", "__symbol__", "spread_bps"]
    market = market.merge(
        spread_values, on=["__ts__", "__symbol__"], how="left", validate="one_to_one",
    )
    market["__decision_ts__"] = market["__ts__"] + pd.Timedelta(hours=1)
    market["instrument_available"] = np.isfinite(
        pd.to_numeric(market["signal_close"], errors="coerce")
    )
    bid = pd.to_numeric(market['decision_book_bid'], errors='coerce')
    ask = pd.to_numeric(market['decision_book_ask'], errors='coerce')
    decision_open_numeric = pd.to_numeric(market['decision_open'], errors='coerce')
    decision_book_mid = 0.5 * (bid + ask)
    market['decision_open_book_deviation_bps'] = (
        10_000.0 * (decision_open_numeric - decision_book_mid).abs()
        / decision_book_mid.where(decision_book_mid > 0.0)
    ).replace([np.inf, -np.inf], np.nan)
    hourly_fallback = market['decision_open_source'].fillna('').str.startswith(
        'official_hourly_', na=False,
    )
    hourly_fallback_valid = (
        pd.to_numeric(market['decision_open_hourly_volume'], errors='coerce').gt(0.0)
        & bid.gt(0.0)
        & ask.ge(bid)
        & market['decision_open_book_deviation_bps'].le(100.0)
    )
    market['decision_open_fallback_valid'] = (~hourly_fallback) | hourly_fallback_valid
    market["entry_executable"] = (
        np.isfinite(pd.to_numeric(market["decision_open"], errors="coerce"))
        & pd.to_numeric(market["signal_atr"], errors="coerce").gt(0.0)
        & market['decision_open_fallback_valid'].fillna(False)
    )
    sides = tuple(value.strip() for value in args.sides.split(",") if value.strip())
    population, eligible, rejected = build_point_in_time_candidates(
        market,
        universe=symbols,
        feature_fields=(),
        cross_sectional_sources=(),
        spec=CandidateSpec(
            spread_limit_bps=float(args.spread_limit_bps),
            required_feature_fraction=1.0,
            side_names=sides,
        ),
    )
    # ``build_point_in_time_candidates`` deliberately strips undeclared
    # fields from the model-facing contract.  Reattach the entry price only
    # as execution lineage after eligibility has been decided; it is never a
    # frozen feature or scorer input.
    lineage_values = market[[
        '__ts__', '__symbol__', 'decision_open', 'signal_atr',
        'decision_open_source', 'decision_open_hourly_volume',
        'decision_book_bid', 'decision_book_ask',
        'decision_open_book_deviation_bps', 'decision_open_fallback_valid',
    ]].copy()
    population = _attach_execution_lineage(population, lineage_values)
    eligible = _attach_execution_lineage(eligible, lineage_values)
    rejected = _attach_execution_lineage(rejected, lineage_values)
    population = _label_entry_price_rejections(population)
    eligible = _label_entry_price_rejections(eligible)
    rejected = _label_entry_price_rejections(rejected)
    args.out_dir.mkdir(parents=True)
    population.to_parquet(
        args.out_dir / "target_free_candidate_population.parquet",
        index=False,
        compression="zstd",
    )
    eligible.to_parquet(
        args.out_dir / "eligible_candidates.parquet", index=False, compression="zstd"
    )
    rejected.to_parquet(
        args.out_dir / "candidate_rejection_audit.parquet", index=False, compression="zstd"
    )
    summary = population.groupby(
        ["side_name", "eligibility_reason"], as_index=False, dropna=False
    ).agg(rows=("candidate_id", "size"))
    summary.to_parquet(args.out_dir / "candidate_rejection_reason_summary.parquet", index=False)
    manifest = {
        "schema": f"{SCHEMA}_target_free_hourly_grid",
        "start": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "universe_rows": len(symbols),
        "population_rows": len(population),
        "eligible_rows": len(eligible),
        "rejected_rows": len(rejected),
        "entry": "first 15-minute open at signal close + one hour",
        "decision_open_source": (
            "timestamp-exact Kraken trade-candle open; raw exchange 15-minute "
            "cache first, shared HF 15-minute cache second, official Kraken "
            "one-hour open missing-timestamp fallback only with positive trade "
            "volume and a contemporaneous official-book deviation no greater "
            "than 100 bps; no carry"
        ),
        "future_path_columns_consumed": [],
        "spread_limit_bps": float(args.spread_limit_bps),
        "spread_gate": "official_kraken_signal_hour_bid_ask_bps_before_signal_plus_1h_entry",
        "historical_universe_spread_used_for_membership_only": True,
        "universe_sha256": _sha(universe_source),
        "universe_source_type": universe_source_type,
        "source_map": source_map,
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    print(json.dumps({"event": "complete", "output": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
