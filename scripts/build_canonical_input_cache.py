#!/usr/bin/env python3
"""Build a compact, source-audited hourly primitive cache for frozen replays.

The cache is deliberately a *primitive* store: it contains only downloaded
trade OHLCV resampled from complete 15-minute bars.  Mark price and open
interest remain in the separate official Kraken backfill cache.  It never
creates synthetic order-book depth/spread inputs.  Every source is read in an
isolated worker with a deadline, so a cloud-backed or corrupt Parquet object
cannot stall a multi-day materialisation.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_15M_ROOT = ROOT / 'data_perp/exchanges/krakenfutures/raw/ohlcv_15m'
PRIMARY_HOURLY_ROOT = ROOT / 'data_perp/ohlcv'
OFFICIAL_BACKFILL_ROOT = ROOT / 'data_perp/exchanges/krakenfutures/frozen_contract_backfill_hourly'
CANONICAL_LABEL_ROOT = ROOT / 'data_perp/artifacts/tp6_sl4_exact170_labels_20260808_v1'
EXPANDED_LABEL_ROOTS = (
    ROOT / 'data_perp/artifacts/stage_i_historical_tp6_sl4_h12_r3_20260803_v1',
    ROOT / 'data_perp/artifacts/stage_i_packb_tp6_sl4_h12_r3_20260803_v1',
)


def _symbols_from_labels(root: Path) -> list[str]:
    values: set[str] = set()
    for path in sorted(root.glob('parts/month=*/side=*.parquet')):
        frame = pd.read_parquet(path, columns=['__symbol__'])
        values.update(frame['__symbol__'].dropna().astype(str))
    return sorted(values)


def _symbols_from_label_roots(roots: tuple[Path, ...]) -> list[str]:
    values: set[str] = set()
    for root in roots:
        values.update(_symbols_from_labels(root))
    return sorted(values)


def _read_requested_symbols(path: Path) -> list[str]:
    """Accept a plain symbol list or a versioned universe registry.

    The latter avoids maintaining a second hand-written file for a forward
    cache refresh.  ``canonical_only`` remains the final authority over the
    actual materialised universe.
    """
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        values = raw.get('expanded_symbols') or raw.get('canonical_symbols') or raw.get('symbols') or []
    else:
        values = raw
    symbols: list[str] = []
    for value in values:
        if isinstance(value, dict):
            value = value.get('perp_symbol') or value.get('symbol')
        if isinstance(value, str) and value:
            symbols.append(value)
    return sorted(set(symbols))


def _raw_path(symbol: str) -> Path:
    name = f"{symbol.split('/')[0]}usd:usd_15m.parquet".lower()
    return RAW_15M_ROOT / name


def _primary_paths(symbol: str) -> list[Path]:
    base = symbol.split('/')[0]
    roots = [
        PRIMARY_HOURLY_ROOT / f'symbol={base}_USDT',
        PRIMARY_HOURLY_ROOT / f'symbol={base}_USDC',
        PRIMARY_HOURLY_ROOT / f'symbol={base}_USD',
        PRIMARY_HOURLY_ROOT / f'symbol={base}_USD:USD',
    ]
    return [path for root in roots if root.exists() for path in sorted(root.glob('year=*/**/*.parquet'))]


def _read_15m(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if not isinstance(frame.index, pd.DatetimeIndex):
        ts_col = next((name for name in ('ts', 'timestamp', 'time') if name in frame), None)
        if ts_col is None:
            raise ValueError('15m source has no timestamp index/column')
        frame[ts_col] = pd.to_datetime(frame[ts_col], utc=True)
        frame = frame.set_index(ts_col)
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame = frame[(frame.index >= start) & (frame.index < end)]
    required = ['open', 'high', 'low', 'close', 'volume']
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f'15m source missing {missing}')
    bars = frame[required].apply(pd.to_numeric, errors='coerce').sort_index()
    # A decision-hour is usable only when all four constituent 15m bars exist.
    hourly = bars.resample('1h', label='left', closed='left').agg(
        open=('open', 'first'), high=('high', 'max'), low=('low', 'min'),
        close=('close', 'last'), volume=('volume', 'sum'), count=('close', 'count'),
        # Explicitly labelled fallback for the frozen
        # ``ob_trade_size_to_l1_depth_z_24h`` input.  Native historical trade
        # counts are unavailable, so this is deliberately the robust median
        # executed *15-minute bar quantity*, not a claim about individual
        # trade size.  It is available at the decision-hour close and is
        # shifted causally by the feature engine before use.
        coarse_trade_size_proxy_15m=('volume', 'median'),
    )
    hourly = hourly.loc[hourly['count'].eq(4), required + ['coarse_trade_size_proxy_15m']]
    return hourly.dropna(subset=['open', 'high', 'low', 'close', 'volume'])


def _read_primary(paths: list[Path], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        x = pd.read_parquet(path)
        if 'ts' not in x:
            continue
        x['ts'] = pd.to_datetime(x['ts'], utc=True)
        x = x[(x['ts'] >= start) & (x['ts'] < end)]
        if len(x):
            frames.append(x.set_index('ts')[['open', 'high', 'low', 'close', 'volume']])
    if not frames:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    return pd.concat(frames).loc[lambda x: ~x.index.duplicated(keep='last')].sort_index()


def _read_official_panel(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Read cached official candles plus source-faithful L2 analytics."""
    path = OFFICIAL_BACKFILL_ROOT / f"{symbol.split('/')[0]}_USD_USD.parquet"
    if not path.exists():
        return pd.DataFrame()
    x = pd.read_parquet(path)
    if not isinstance(x.index, pd.DatetimeIndex):
        if 'ts' not in x:
            return pd.DataFrame()
        x = x.set_index('ts')
    x.index = pd.to_datetime(x.index, utc=True)
    fields = [name for name in x if name in ('open', 'high', 'low', 'close', 'volume') or name.startswith('ob_')]
    return x.loc[(x.index >= start) & (x.index < end), fields].apply(pd.to_numeric, errors='coerce').sort_index()


def _worker(symbol: str, out_dir: str, start: str, end: str) -> dict[str, object]:
    """Subprocess entrypoint: write one atomic compact source cache shard."""
    began = time.monotonic()
    start_ts, end_ts = pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC')
    output_dir = Path(out_dir)
    raw = _raw_path(symbol)
    source = 'none'
    detail = ''
    try:
        if raw.exists():
            frame = _read_15m(raw, start_ts, end_ts)
            source = 'downloaded_15m_ohlcv'
        else:
            frame = _read_primary(_primary_paths(symbol), start_ts, end_ts)
            source = 'primary_hourly_ohlcv' if len(frame) else 'none'
        official = _read_official_panel(symbol, start_ts, end_ts)
        if len(official):
            ohlcv = official.reindex(columns=['open', 'high', 'low', 'close', 'volume'])
            frame = frame.combine_first(ohlcv) if len(frame) else ohlcv
            for field in (name for name in official if name.startswith('ob_')):
                frame[field] = official[field].reindex(frame.index)
            source = f'{source}+official_kraken_trade' if source != 'none' else 'official_kraken_trade'
        if frame.empty:
            raise ValueError('no complete hourly OHLCV')
        frame = frame.copy()
        frame['source_ohlcv'] = source
        frame['coarse_trade_size_proxy_source'] = (
            'downloaded_15m_bar_volume_median_v1'
            if 'coarse_trade_size_proxy_15m' in frame
            else 'unavailable'
        )
        frame.index.name = 'ts'
        target = output_dir / 'hourly' / f"symbol={symbol.replace('/', '_')}" / 'part.parquet'
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = target.with_suffix('.partial.parquet')
        frame.to_parquet(staged, compression='zstd')
        os.replace(staged, target)
        return {
            'symbol': symbol, 'status': 'ok', 'source_ohlcv': source,
            'rows': int(len(frame)), 'start': str(frame.index.min()), 'end': str(frame.index.max()),
            'complete_hour_fraction': float(len(frame) / max(1, int((end_ts - start_ts) / pd.Timedelta(hours=1)))),
            'seconds': round(time.monotonic() - began, 3), 'detail': detail,
        }
    except Exception as exc:
        return {
            'symbol': symbol, 'status': 'error', 'source_ohlcv': source, 'rows': 0,
            'start': None, 'end': None, 'complete_hour_fraction': 0.0,
            'seconds': round(time.monotonic() - began, 3), 'detail': f'{type(exc).__name__}: {exc}',
        }


def _run_isolated(symbol: str, out_dir: Path, start: str, end: str, timeout: int) -> dict[str, object]:
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--symbol', symbol,
               '--out-dir', str(out_dir), '--start', start, '--end', end]
    try:
        completed = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        return json.loads(completed.stdout)
    except subprocess.TimeoutExpired:
        return {'symbol': symbol, 'status': 'timeout', 'source_ohlcv': 'none', 'rows': 0,
                'start': None, 'end': None, 'complete_hour_fraction': 0.0,
                'seconds': float(timeout), 'detail': 'isolated source read exceeded deadline'}
    except Exception as exc:
        return {'symbol': symbol, 'status': 'error', 'source_ohlcv': 'none', 'rows': 0,
                'start': None, 'end': None, 'complete_hour_fraction': 0.0,
                'seconds': 0.0, 'detail': f'worker: {type(exc).__name__}: {exc}'}


def _universe_versions(expanded_symbols: list[str]) -> dict[str, list[str]]:
    canonical = _symbols_from_labels(CANONICAL_LABEL_ROOT)
    labelled_expanded = _symbols_from_label_roots(EXPANDED_LABEL_ROOTS)
    return {'canonical_exact170': canonical, 'expanded': sorted(set(expanded_symbols) | set(labelled_expanded))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--expanded-symbols-json', type=Path)
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--per-symbol-timeout', type=int, default=90)
    ap.add_argument(
        '--canonical-only', action='store_true',
        help=(
            'materialise only the frozen exact-170 universe; retain the full '
            'expanded-universe behaviour by default'
        ),
    )
    ap.add_argument('--registry-only', action='store_true', help='refresh versioned universe manifest without rewriting cache shards')
    ap.add_argument('--worker', action='store_true')
    ap.add_argument('--symbol')
    args = ap.parse_args()
    if args.worker:
        assert args.symbol is not None
        print(json.dumps(_worker(args.symbol, str(args.out_dir), args.start, args.end)))
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.expanded_symbols_json is None:
        # Refresh an existing, versioned cache without maintaining a second
        # hand-written universe file.  The cached registry remains the source
        # of truth and is still reconciled with all label-bearing symbols.
        existing_registry = args.out_dir / 'universe_versions.json'
        if not existing_registry.exists():
            ap.error('--expanded-symbols-json is required when no cached universe registry exists')
        requested_expanded = sorted(set(json.loads(existing_registry.read_text()).get('expanded_symbols', [])))
        if not requested_expanded:
            ap.error('cached universe registry has no expanded_symbols')
    else:
        requested_expanded = _read_requested_symbols(args.expanded_symbols_json)
        if not requested_expanded:
            ap.error('--expanded-symbols-json contains no usable symbols')
    versions = _universe_versions(requested_expanded)
    # The label universe is authoritative for replay coverage.  The supplied
    # download list is retained as provenance, but cannot silently omit a
    # label-bearing symbol (for example ENJ in the previous 242-item list).
    expanded = (
        versions['canonical_exact170']
        if args.canonical_only
        else versions['expanded']
    )
    (args.out_dir / 'universe_versions.json').write_text(json.dumps({
        'schema': 'canonical_universe_registry_v1',
        'canonical_name': 'canonical_exact170', 'canonical_symbols': versions['canonical_exact170'],
        'expanded_name': 'expanded', 'expanded_symbols': versions['expanded'],
        'canonical_count': len(versions['canonical_exact170']), 'expanded_count': len(versions['expanded']),
    }, indent=2))
    if args.registry_only:
        return
    rows: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        pending = [pool.submit(_run_isolated, symbol, args.out_dir, args.start, args.end, args.per_symbol_timeout) for symbol in expanded]
        for number, future in enumerate(concurrent.futures.as_completed(pending), 1):
            row = future.result(); rows.append(row)
            print(json.dumps({'event': 'cache_source_complete', 'completed': number, 'total': len(expanded), 'symbol': row['symbol'], 'status': row['status']}), flush=True)
    audit = pd.DataFrame(rows).sort_values('symbol')
    audit.to_parquet(args.out_dir / 'source_integrity_audit.parquet', index=False)
    summary = {
        'schema': 'canonical_hourly_primitive_cache_v1', 'start': args.start, 'end_exclusive': args.end,
        'expanded_symbols': len(expanded), 'canonical_symbols': len(versions['canonical_exact170']),
        'universe_mode': 'canonical_exact170' if args.canonical_only else 'expanded',
        'ok_symbols': int(audit.status.eq('ok').sum()), 'timeout_symbols': int(audit.status.eq('timeout').sum()),
        'error_symbols': int(audit.status.eq('error').sum()), 'source_policy': 'downloaded 15m OHLCV only; no synthetic order-book primitives',
    }
    (args.out_dir / 'cache_manifest.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
