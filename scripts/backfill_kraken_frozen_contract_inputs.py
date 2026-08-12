#!/usr/bin/env python3
"""Cache official Kraken hourly mark and OI panels for a frozen replay.

The canonical feature contract needs a mark-price dislocation and OI-derived
features.  The local hourly store ends in May 2026, while the approved coarse
15-minute fallback carries trade OHLCV only.  This utility fills that narrow
source gap from Kraken's documented public chart APIs, preserving one source
file per contract symbol and a source/coverage audit.  It deliberately does
not create labels or substitute a trade-price proxy for mark price.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import certifi


API = 'https://futures.kraken.com/api/charts/v1'
BTC_ALIAS = {'BTC': 'XBT'}
TLS_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def _symbols_from_json(path: Path) -> list[str]:
    """Load a raw list or a frozen-universe manifest without widening scope."""
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict) and isinstance(payload.get('source_map'), dict):
        values = list(payload['source_map'])
    elif isinstance(payload, dict) and isinstance(payload.get('symbols'), list):
        values = [
            row.get('perp_symbol') if isinstance(row, dict) else row
            for row in payload['symbols']
        ]
    else:
        raise ValueError('symbols JSON must be a string list or frozen universe manifest')
    symbols = [str(value) for value in values if value]
    if len(symbols) != len(set(symbols)):
        raise ValueError('symbols JSON contains duplicate symbols')
    return sorted(symbols)


def _json(url: str, retries: int = 2) -> dict:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={'Accept': 'application/json', 'User-Agent': 'Ares-frozen-contract-backfill/1'})
            with urllib.request.urlopen(request, timeout=15, context=TLS_CONTEXT) as response:
                return json.loads(response.read().decode('utf-8'))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code in (400, 404):
                break
            time.sleep(min(8.0, 0.75 * (2 ** attempt)))
    raise RuntimeError(f'{type(last).__name__}: {last}')


def _tradeable(symbol: str) -> str:
    base = symbol.split('/')[0]
    return f'PF_{BTC_ALIAS.get(base, base)}USD'


def _ranges(start: int, end: int, chunk_days: int = 45):
    """Chart endpoints cap long ranges; make the API boundary explicit."""
    step = chunk_days * 86400
    cursor = start
    while cursor < end:
        nxt = min(end, cursor + step)
        yield cursor, nxt
        cursor = nxt


def _candles(kind: str, tradeable: str, start: int, end: int) -> pd.DataFrame:
    rows: list[dict] = []
    for left, right in _ranges(start, end):
        query = urllib.parse.urlencode({'from': left, 'to': right})
        try:
            rows.extend(_json(f'{API}/{kind}/{tradeable}/1h?{query}').get('candles', []))
        except RuntimeError:
            # A contract may not have existed during an early chunk.  Keep
            # later chunks available rather than discarding the entire symbol.
            continue
    if not rows:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close'])
    frame = pd.DataFrame(rows)
    frame['ts'] = pd.to_datetime(pd.to_numeric(frame['time'], errors='coerce'), unit='ms', utc=True)
    frame = frame.dropna(subset=['ts']).drop_duplicates('ts', keep='last').set_index('ts').sort_index()
    return frame


def _mark(tradeable: str, start: int, end: int) -> pd.Series:
    frame = _candles('mark', tradeable, start, end)
    if frame.empty:
        return pd.Series(dtype='float32', name='mark_price')
    return pd.to_numeric(frame['close'], errors='coerce').astype('float32').rename('mark_price')


def _trade(tradeable: str, start: int, end: int) -> pd.DataFrame:
    frame = _candles('trade', tradeable, start, end)
    if frame.empty:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    columns = [name for name in ('open', 'high', 'low', 'close', 'volume') if name in frame]
    return frame[columns].apply(pd.to_numeric, errors='coerce').astype('float32')


def _orderbook_analytics(tradeable: str, start: int, end: int) -> pd.DataFrame:
    """Source-faithful historical L2 analytics; do not relabel bands as levels."""
    pieces: list[pd.DataFrame] = []
    fields = ('bestPrice', 'liquidity005', 'liquidity01', 'liquidity025', 'liquidity05', 'liquidity10', 'liquidity100', 'slippage1k', 'slippage10k', 'slippage100k', 'slippage1m')
    for left, right in _ranges(start, end):
        query = urllib.parse.urlencode({'since': left, 'to': right, 'interval': 3600})
        try:
            result = _json(f'{API}/analytics/{tradeable}/orderbook?{query}').get('result', {})
        except RuntimeError:
            continue
        stamps, data = result.get('timestamp', []), result.get('data', {})
        if not stamps or not isinstance(data, dict):
            continue
        ts = pd.to_datetime(pd.Series(stamps, dtype='int64'), unit='s', utc=True)
        frame = pd.DataFrame(index=ts)
        for side in ('bid', 'ask'):
            values = data.get(side, {})
            if not isinstance(values, dict):
                continue
            for field in fields:
                if field in values:
                    name = f'ob_{side}_{field}'
                    frame[name] = np.asarray(pd.to_numeric(values[field], errors='coerce'), dtype='float32')
        if len(frame):
            pieces.append(frame)
    if not pieces:
        return pd.DataFrame()
    out = pd.concat(pieces).loc[lambda x: ~x.index.duplicated(keep='last')].sort_index()
    out['ob_analytics_source'] = 'kraken_futures_orderbook_analytics_1h'
    return out


def _open_interest(tradeable: str, start: int, end: int) -> pd.Series:
    pieces: list[pd.Series] = []
    for left, right in _ranges(start, end):
        query = urllib.parse.urlencode({'since': left, 'to': right, 'interval': 3600})
        try:
            result = _json(f'{API}/analytics/{tradeable}/open-interest?{query}').get('result', {})
        except RuntimeError:
            continue
        stamps, values = result.get('timestamp', []), result.get('data', [])
        if not stamps or not values:
            continue
        close = [row[3] if isinstance(row, list) and len(row) >= 4 else np.nan for row in values]
        ts = pd.to_datetime(pd.Series(stamps, dtype='int64'), unit='s', utc=True)
        pieces.append(pd.Series(np.asarray(pd.to_numeric(close, errors='coerce'), dtype=np.float32), index=ts))
    if not pieces:
        return pd.Series(dtype='float32', name='open_interest')
    return pd.concat(pieces).groupby(level=0).last().astype('float32').rename('open_interest')


def _one(symbol: str, start: int, end: int, out_dir: Path, include_trade: bool = False, include_orderbook_analytics: bool = False, include_base_primitives: bool = True) -> dict[str, object]:
    path = out_dir / f"{symbol.split('/')[0]}_USD_USD.parquet"
    existing = pd.DataFrame()
    if path.exists():
        try:
            existing = pd.read_parquet(path)
            existing.index = pd.to_datetime(existing.index, utc=True)
        except Exception:
            existing = pd.DataFrame()
    tradeable = _tradeable(symbol)
    errors: list[str] = []
    mark = pd.Series(dtype='float32', name='mark_price')
    oi = pd.Series(dtype='float32', name='open_interest')
    if include_base_primitives:
        existing_mark = existing.get('mark_price', pd.Series(dtype='float32'))
        expected_hours = max(1, int((end - start) // 3600))
        existing_mark_hours = int(
            existing_mark.loc[(existing_mark.index >= pd.Timestamp(start, unit='s', tz='UTC')) & (existing_mark.index < pd.Timestamp(end, unit='s', tz='UTC'))]
            .notna().sum()
        ) if isinstance(existing_mark.index, pd.DatetimeIndex) else 0
        if existing_mark_hours < int(0.995 * expected_hours):
            try:
                mark = _mark(tradeable, start, end)
            except Exception as exc:
                errors.append(f'mark:{exc}')
        try:
            oi = _open_interest(tradeable, start, end)
        except Exception as exc:
            errors.append(f'oi:{exc}')
    trade = pd.DataFrame()
    if include_trade:
        try:
            trade = _trade(tradeable, start, end)
        except Exception as exc:
            errors.append(f'trade:{exc}')
    orderbook = pd.DataFrame()
    if include_orderbook_analytics:
        try:
            orderbook = _orderbook_analytics(tradeable, start, end)
        except Exception as exc:
            errors.append(f'orderbook_analytics:{exc}')
    index = existing.index.union(mark.index).union(oi.index).union(trade.index).union(orderbook.index).sort_values()
    panel = existing.reindex(index)
    for series in (mark, oi):
        if series.name not in panel:
            panel[series.name] = np.nan
        panel[series.name] = panel[series.name].combine_first(series.reindex(index))
    for field in trade:
        if field not in panel:
            panel[field] = np.nan
        panel[field] = panel[field].combine_first(trade[field].reindex(index))
    for field in orderbook:
        if field not in panel:
            panel[field] = np.nan
        panel[field] = panel[field].combine_first(orderbook[field].reindex(index))
    panel = panel.sort_index()
    if len(panel):
        panel.to_parquet(path, compression='zstd')
    return {
        'symbol': symbol, 'tradeable': tradeable, 'rows': int(len(panel)),
        'mark_rows': int(panel.get('mark_price', pd.Series(dtype=float)).notna().sum()),
        'oi_rows': int(panel.get('open_interest', pd.Series(dtype=float)).notna().sum()),
        'trade_rows': int(panel.get('close', pd.Series(dtype=float)).notna().sum()),
        'orderbook_analytics_rows': int(panel.get('ob_bid_bestPrice', pd.Series(dtype=float)).notna().sum()),
        'errors': ' | '.join(errors),
    }


def _audit_cached(symbol: str, out_dir: Path) -> dict[str, object]:
    path = out_dir / f"{symbol.split('/')[0]}_USD_USD.parquet"
    try:
        panel = pd.read_parquet(path)
        return {
            'symbol': symbol, 'tradeable': _tradeable(symbol), 'rows': int(len(panel)),
            'mark_rows': int(panel.get('mark_price', pd.Series(dtype=float)).notna().sum()),
            'oi_rows': int(panel.get('open_interest', pd.Series(dtype=float)).notna().sum()),
            'trade_rows': int(panel.get('close', pd.Series(dtype=float)).notna().sum()),
            'orderbook_analytics_rows': int(panel.get('ob_bid_bestPrice', pd.Series(dtype=float)).notna().sum()), 'errors': '',
        }
    except Exception as exc:
        return {'symbol': symbol, 'tradeable': _tradeable(symbol), 'rows': 0, 'mark_rows': 0,
                'oi_rows': 0, 'trade_rows': 0, 'orderbook_analytics_rows': 0, 'errors': f'{type(exc).__name__}: {exc}'}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols-json', type=Path, required=True, help='JSON array of Ares symbols')
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--start', required=True, help='UTC timestamp, inclusive')
    ap.add_argument('--end', required=True, help='UTC timestamp, exclusive')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--include-trade-ohlcv', action='store_true', help='also cache official trade candles for periods missing local OHLCV')
    ap.add_argument('--include-orderbook-analytics', action='store_true', help='cache raw official historical L2 analytics without changing frozen feature semantics')
    ap.add_argument('--orderbook-analytics-only', action='store_true', help='do not refresh existing trade/mark/OI panels while adding order-book analytics')
    ap.add_argument('--skip-existing-since', help='resume aid: skip cache files modified at or after this UTC timestamp')
    ap.add_argument('--audit-only', action='store_true', help='rebuild coverage/manifest from cached panels without API calls')
    ap.add_argument('--extra-symbol', action='append', default=[], help='additional symbol for audit/backfill reconciliation')
    args = ap.parse_args()
    symbols = sorted(set(_symbols_from_json(args.symbols_json)) | set(args.extra_symbol))
    if args.skip_existing_since:
        cutoff = pd.Timestamp(args.skip_existing_since, tz='UTC').timestamp()
        symbols = [symbol for symbol in symbols if not ((args.out_dir / f"{symbol.split('/')[0]}_USD_USD.parquet").exists() and (args.out_dir / f"{symbol.split('/')[0]}_USD_USD.parquet").stat().st_mtime >= cutoff)]
    start = int(pd.Timestamp(args.start, tz='UTC').timestamp())
    end = int(pd.Timestamp(args.end, tz='UTC').timestamp())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    warnings.filterwarnings('ignore', category=FutureWarning, message='The behavior of array concatenation')
    if args.audit_only:
        rows = [_audit_cached(symbol, args.out_dir) for symbol in symbols]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(_one, symbol, start, end, args.out_dir, args.include_trade_ohlcv, args.include_orderbook_analytics, not args.orderbook_analytics_only) for symbol in symbols]
            for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
                row = future.result(); rows.append(row)
                # Persist progress after every symbol: an interrupted public-API
                # run can resume deterministically without losing completed work.
                pd.DataFrame(rows).sort_values('symbol').to_parquet(args.out_dir / 'backfill_progress.parquet', index=False)
                if count % 20 == 0:
                    print(json.dumps({'event': 'backfill_progress', 'completed': count, 'total': len(symbols)}), flush=True)
    audit = pd.DataFrame(rows).sort_values('symbol')
    audit.to_parquet(args.out_dir / 'backfill_coverage.parquet', index=False)
    (args.out_dir / 'backfill_manifest.json').write_text(json.dumps({
        'source': 'Kraken Futures public charts API', 'mark_endpoint': '/mark/:symbol/1h',
        'open_interest_endpoint': '/analytics/:symbol/open-interest?interval=3600',
        'start': args.start, 'end_exclusive': args.end, 'symbols': symbols,
        'rows': int(len(audit)), 'mark_symbols': int(audit.mark_rows.gt(0).sum()),
        'oi_symbols': int(audit.oi_rows.gt(0).sum()), 'trade_symbols': int(audit.trade_rows.gt(0).sum()),
        'orderbook_analytics_symbols': int(audit.orderbook_analytics_rows.gt(0).sum()),
        'trade_endpoint': '/trade/:symbol/1h' if args.include_trade_ohlcv else None,
        'orderbook_analytics_endpoint': '/analytics/:symbol/orderbook?interval=3600' if args.include_orderbook_analytics else None,
    }, indent=2))


if __name__ == '__main__':
    main()
