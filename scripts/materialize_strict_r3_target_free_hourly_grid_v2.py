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
import os
import sys
from concurrent.futures import ThreadPoolExecutor
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


def _parallel_symbol_load(symbols: list[str], loader):
    """Load independent symbol-local source rows without changing ordering.

    The target-free grid previously opened the three independent per-symbol
    Parquet families serially.  These are immutable point-in-time reads, so a
    bounded pool is safe and removes a material live-boundary delay.  ``map``
    preserves the frozen universe order, while every loader keeps the former
    per-symbol fail-closed behaviour.
    """
    # Arrow filtered reads against the mixed-age 15-minute shards are not
    # safely scalable to the full source-refresh fan-out.  Sixteen concurrent
    # readers can leave all workers blocked in parquet metadata/I/O while no
    # candidate row is emitted.  Four retains independent point-in-time reads
    # and frozen ordering but bounds file pressure; an operator may request a
    # smaller pool for diagnosis, never a larger live pool.
    workers = min(
        4,
        max(1, int(os.environ.get("STRICT_R3_GRID_IO_WORKERS", "4"))),
        max(1, len(symbols)),
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(loader, symbols))


def _signal_hour_spread_panel(
    symbols: list[str],
    signal_index: pd.DatetimeIndex,
    *,
    bar_phase_minutes: int = 0,
) -> pd.DataFrame:
    """Load the causal signal-hour Kraken bid/ask spread.

    The historical universe registry decides membership only.  Actionability
    uses the official order-book analytics timestamp that precedes the
    signal+one-hour entry decision.  Missing/stale analytics stay missing and
    are rejected by ``build_point_in_time_candidates``.
    """
    def _load(symbol: str) -> tuple[str, pd.Series | None]:
        base = symbol.split("/", 1)[0]
        path = FROZEN_INPUT_BACKFILL_ROOT / f"{base}_USD_USD.parquet"
        if not path.exists():
            return symbol, None
        try:
            frame = pd.read_parquet(
                path, columns=["ob_bid_bestPrice", "ob_ask_bestPrice"],
            )
        except Exception:
            return symbol, None
        frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
        bid = pd.to_numeric(frame["ob_bid_bestPrice"], errors="coerce")
        ask = pd.to_numeric(frame["ob_ask_bestPrice"], errors="coerce")
        mid = 0.5 * (bid + ask)
        spread = (10_000.0 * (ask - bid) / mid.where(mid > 0.0)).replace(
            [np.inf, -np.inf], np.nan,
        )
        # The frozen historical order-book sidecar is hourly.  At a shifted
        # H1 boundary there is no timestamp-identical historical observation;
        # use only the most recent published book, never a later value.  The
        # phase-zero path remains byte-identical to the existing exact-index
        # contract.
        values = (
            spread.reindex(signal_index)
            if int(bar_phase_minutes) == 0
            else spread.reindex(signal_index, method="ffill")
        )
        return symbol, values

    values = {
        symbol: frame
        for symbol, frame in _parallel_symbol_load(symbols, _load)
        if frame is not None
    }
    return (
        pd.concat(values, axis=1).reindex(index=signal_index, columns=symbols)
        if values else pd.DataFrame(index=signal_index, columns=symbols, dtype=float)
    )


def _decision_open_panel(
    symbols: list[str],
    signal_index: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return entry prices and non-feature source lineage at decision time."""
    decision_index = signal_index + pd.Timedelta(hours=1)
    def _load(symbol: str) -> tuple[str, pd.DataFrame]:
        lineage = _read_downloaded_15m_decision_open(
            symbol, decision_index, return_lineage=True,
        )
        assert isinstance(lineage, pd.DataFrame)
        return symbol, lineage.set_axis(signal_index)

    loaded = dict(_parallel_symbol_load(symbols, _load))
    values = {
        symbol: lineage['decision_open'] for symbol, lineage in loaded.items()
    }
    sources = {
        symbol: lineage['decision_open_source'] for symbol, lineage in loaded.items()
    }
    trade_volumes = {
        symbol: lineage['decision_open_trade_volume'] for symbol, lineage in loaded.items()
    }
    fallback_volumes = {
        symbol: lineage['decision_open_hourly_volume'] for symbol, lineage in loaded.items()
    }
    return (
        pd.concat(values, axis=1).reindex(index=signal_index, columns=symbols),
        pd.concat(sources, axis=1).reindex(index=signal_index, columns=symbols),
        pd.concat(trade_volumes, axis=1).reindex(index=signal_index, columns=symbols),
        pd.concat(fallback_volumes, axis=1).reindex(index=signal_index, columns=symbols),
    )


def _decision_open_source_validity(
    *,
    source: pd.Series,
    trade_volume: pd.Series,
    hourly_volume: pd.Series,
    bid: pd.Series,
    ask: pd.Series,
    book_deviation_bps: pd.Series,
    maximum_deviation_bps: float = 100.0,
) -> tuple[pd.Series, pd.Series]:
    """Return causal candidate-reference validity and the direct-15m audit flag.

    At the exact decision boundary, the just-opened 15-minute candle supplies
    a causal *reference open*, but its eventual traded volume is necessarily
    unknown.  Treating that unknown final volume as evidence of a stale price
    couples candidate availability to later intra-bar information and wrongly
    removes otherwise feature-complete candidates.  A finite direct 15-minute
    open therefore remains eligible only when the contemporaneous bid/ask
    corroborates it within the declared deviation limit.

    ``stale_15m`` is retained as execution lineage.  It also requires this
    point-in-time book corroboration before entering the scored population;
    the later size-aware preflight remains the final authority for an order.
    The official hourly fallback remains stricter: it needs positive trade
    support and a contemporaneous book agreement before it can form a
    candidate reference.
    """
    direct_15m = source.fillna('').isin({'raw_15m', 'shared_15m'})
    # ``gt`` maps both zero and unknown final volume to False.  For a direct
    # 15-minute decision open this is an audit flag, not a candidate veto: the
    # candle is still forming and its future volume cannot be used causally.
    stale_15m = direct_15m & ~trade_volume.gt(0.0).fillna(False)
    hourly_fallback = source.fillna('').str.startswith('official_hourly_', na=False)
    book_aligned = (
        bid.gt(0.0)
        & ask.ge(bid)
        & book_deviation_bps.le(float(maximum_deviation_bps))
    ).fillna(False)
    hourly_valid = hourly_volume.gt(0.0).fillna(False) & book_aligned
    non_direct_valid = ((~hourly_fallback) | hourly_valid)
    # A just-opened direct 15-minute candle has no decision-time final volume:
    # ``trade_volume`` is deliberately unknown so a delayed refresh cannot
    # change candidate eligibility with future intrabar activity.  It is still
    # an executable *price* only when the contemporaneous bid/ask corroborates
    # it.  Otherwise it is a stale/placeholder open and must be excluded before
    # scoring, not left for the post-auction execution preflight to discover.
    direct_valid = direct_15m & (~stale_15m | book_aligned)
    valid = direct_valid | (~direct_15m & non_direct_valid)
    return valid.fillna(False), stale_15m.fillna(False)


def _decision_book_panel(
    symbols: list[str],
    signal_index: pd.DatetimeIndex,
    *,
    bar_phase_minutes: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return timestamp-exact official bid/ask for hourly-fallback validation.

    The book is only used to reject a stale coarse trade fallback.  It is
    execution lineage rather than a model feature and does not replace the
    frozen entry-price contract with a live quote.
    """
    decision_index = signal_index + pd.Timedelta(hours=1)
    def _load(symbol: str) -> tuple[str, pd.Series | None, pd.Series | None]:
        base = symbol.split('/', 1)[0]
        path = FROZEN_INPUT_BACKFILL_ROOT / f'{base}_USD_USD.parquet'
        if not path.exists():
            return symbol, None, None
        try:
            frame = pd.read_parquet(
                path, columns=['ob_bid_bestPrice', 'ob_ask_bestPrice'],
            )
            frame.index = pd.to_datetime(frame.index, utc=True, errors='raise')
            frame = frame.loc[~frame.index.duplicated(keep='last')].sort_index()
        except Exception:
            return symbol, None, None
        bid = pd.to_numeric(frame['ob_bid_bestPrice'], errors='coerce')
        ask = pd.to_numeric(frame['ob_ask_bestPrice'], errors='coerce')
        if int(bar_phase_minutes) != 0:
            # This is used only to corroborate a zero/unknown-volume entry
            # candle.  The sidecar is hourly, so a shifted decision may use
            # the latest known book from at most the preceding hour.  No
            # future quote can qualify an entry.
            bid = bid.reindex(decision_index, method='ffill')
            ask = ask.reindex(decision_index, method='ffill')
        else:
            bid = bid.reindex(decision_index)
            ask = ask.reindex(decision_index)
        return symbol, bid.set_axis(signal_index), ask.set_axis(signal_index)

    loaded = _parallel_symbol_load(symbols, _load)
    bids = {symbol: bid for symbol, bid, ask in loaded if bid is not None}
    asks = {symbol: ask for symbol, bid, ask in loaded if ask is not None}
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
    bar_phase_minutes: int = 0,
) -> pd.DataFrame:
    """Return the exact policy ATR available at each signal timestamp.

    This is execution readiness, not a model feature.  It deliberately uses
    the same 15-minute bar files and Wilder-14 implementation as the shadow
    portfolio so an admitted row cannot fail only after the auction.
    """
    required_end = signal_index.max() if len(signal_index) else None
    def _load(symbol: str) -> tuple[str, pd.Series | None]:
        stem = symbol.lower().replace("/", "").replace("_", "")
        path = policy_bar_root / f"{stem}_15m.parquet"
        if not path.exists():
            return symbol, None
        try:
            bars = pd.read_parquet(path, columns=["open", "high", "low", "close"])
            bars.index = pd.to_datetime(bars.index, utc=True, errors="raise")
            bars = bars.loc[~bars.index.duplicated(keep="last")].sort_index()
            if required_end is not None and not bars.empty and bars.index.min() <= required_end:
                bars = causal_flat_fill_omitted_15m(bars, end=required_end)
            atr = causal_decision_atr_from_15m(
                bars, bar_phase_minutes=int(bar_phase_minutes),
            )
        except Exception:
            return symbol, None
        return symbol, pd.to_numeric(atr.reindex(signal_index), errors="coerce")

    values = {
        symbol: atr
        for symbol, atr in _parallel_symbol_load(symbols, _load)
        if atr is not None
    }
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
    stale_15m = result.get(
        'decision_open_15m_requires_book_validation',
        pd.Series(False, index=result.index),
    ).fillna(False)
    result.loc[
        generic & stale_15m & ~result['decision_open_fallback_valid'].fillna(False),
        'eligibility_reason',
    ] = 'entry_15m_zero_or_unknown_volume_book_mismatch'
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
    parser.add_argument(
        "--bar-phase-minutes", type=int, default=0, choices=(0, 15, 30, 45),
        help=(
            "Completed-H1 boundary for research sampling.  A non-zero phase "
            "uses the four preceding 15-minute bars as one H1 observation; "
            "it does not upsample the frozen hourly model."
        ),
    )
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
    if int(start.minute) != int(args.bar_phase_minutes):
        raise ValueError("start minute must equal --bar-phase-minutes")
    if int(end.minute) != int(args.bar_phase_minutes):
        raise ValueError("end-exclusive minute must equal --bar-phase-minutes")
    # Features and the signal close use completed hourly bars only. Entry
    # availability is sourced separately from the first 15-minute open at the
    # decision timestamp; it must not wait for that hour to complete.
    panel, source_map = _make_panel(
        symbols,
        start,
        end,
        bar_phase_minutes=int(args.bar_phase_minutes),
    )
    close = panel["close"].reindex(columns=symbols)
    signal_index = close.index[(close.index >= start) & (close.index < end)]
    close = close.reindex(signal_index)
    (
        decision_open,
        decision_open_source,
        decision_open_trade_volume,
        decision_open_hourly_volume,
    ) = _decision_open_panel(
        symbols, signal_index,
    )
    decision_bid, decision_ask = _decision_book_panel(
        symbols, signal_index, bar_phase_minutes=int(args.bar_phase_minutes),
    )
    policy_bar_root = (
        args.policy_bar_root if args.policy_bar_root.is_absolute()
        else ROOT / args.policy_bar_root
    )
    signal_atr = _signal_atr_panel(
        symbols,
        signal_index,
        policy_bar_root=policy_bar_root,
        bar_phase_minutes=int(args.bar_phase_minutes),
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
    trade_volume_values = decision_open_trade_volume.stack(dropna=False).rename(
        'decision_open_trade_volume',
    ).reset_index()
    trade_volume_values.columns = ['__ts__', '__symbol__', 'decision_open_trade_volume']
    bid_values = decision_bid.stack(dropna=False).rename('decision_book_bid').reset_index()
    bid_values.columns = ['__ts__', '__symbol__', 'decision_book_bid']
    ask_values = decision_ask.stack(dropna=False).rename('decision_book_ask').reset_index()
    ask_values.columns = ['__ts__', '__symbol__', 'decision_book_ask']
    for values in (source_values, trade_volume_values, volume_values, bid_values, ask_values):
        market = market.merge(
            values, on=['__ts__', '__symbol__'], how='left', validate='one_to_one',
        )
    atr_values = signal_atr.stack(dropna=False).rename("signal_atr").reset_index()
    atr_values.columns = ["__ts__", "__symbol__", "signal_atr"]
    market = market.merge(
        atr_values, on=["__ts__", "__symbol__"], how="left", validate="one_to_one",
    )
    current_spread = _signal_hour_spread_panel(
        symbols, signal_index, bar_phase_minutes=int(args.bar_phase_minutes),
    )
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
    (
        market['decision_open_fallback_valid'],
        market['decision_open_15m_requires_book_validation'],
    ) = _decision_open_source_validity(
        source=market['decision_open_source'],
        trade_volume=pd.to_numeric(market['decision_open_trade_volume'], errors='coerce'),
        hourly_volume=pd.to_numeric(market['decision_open_hourly_volume'], errors='coerce'),
        bid=bid,
        ask=ask,
        book_deviation_bps=market['decision_open_book_deviation_bps'],
    )
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
        'decision_open_source', 'decision_open_trade_volume',
        'decision_open_hourly_volume', 'decision_open_15m_requires_book_validation',
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
        "bar_phase_minutes": int(args.bar_phase_minutes),
        "feature_cadence_contract": (
            "one completed H1 observation per row; phase shifts only the "
            "15-minute-derived H1 boundary, while entry remains the next "
            "timestamp-exact 15-minute open"
        ),
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
