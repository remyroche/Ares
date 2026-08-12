#!/usr/bin/env python3
"""Canonical simple-policy replay for strict-R3 scores on downloaded 15m bars.

This materialises candidate-local exits first, using
``simple_policy_optimiser.simulate_and_score`` with a deliberately frozen
contract: 3 ATR stop, 0.5 ATR activation, 0.25 ATR fixed trailing gap and a
12-hour timeout.  The optimiser is used only for bar-order and exit mechanics;
the requested 100-bps round-trip cost is deducted exactly once afterwards.
Per-symbol checkpoints make the 2025--2026 source replay restartable.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The declared contract is a fixed 100-bps cost, with no additional proxy
# spread/slippage.  Set before importing the simulator's module constants.
os.environ.setdefault('EPM_SIMPLE_POLICY_STOP_EXIT_BASE_GAP_BPS', '0')
os.environ.setdefault('EPM_SIMPLE_POLICY_STOP_EXIT_MAX_GAP_BPS', '0')
os.environ.setdefault('EPM_SIMPLE_POLICY_SPREAD_MODEL_ENABLED', '0')
from extreme_price_movements.simple_policy_optimiser import simulate_and_score  # noqa: E402


PREDICTIONS = ROOT / 'data_perp/artifacts/strict_r3_full_inference_2025_2026_v2/predictions.parquet'
LABEL_ROOT = ROOT / 'data_perp/artifacts/stage_i_packb_tp6_sl4_h12_r3_20260803_v1'
# The actively refreshed source is the shared HF store used by the downloader.
# Keep the exchange-local cache only as a legacy fallback.
OHLCV_15M = ROOT / '15m_ohlcv_perp'
LEGACY_OHLCV_15M = ROOT / 'data_perp/exchanges/krakenfutures/raw/ohlcv_15m'
EXACT_MINUTE_ROOT = ROOT / 'data_perp/exchanges/krakenfutures/execution_1m/ohlcv'
HORIZON_BARS = 48
COST_BPS = 100.0
OPTIMISED_POLICY = ROOT / 'data_perp/artifacts/strict_r3_simple_policy_optimised_20260809_v1/winner.json'


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--months', default='2025-02:2026-07', help='inclusive YYYY-MM range')
    p.add_argument('--out-dir', type=Path, default=ROOT / 'data_perp/artifacts/strict_r3_simple_policy_15m_2025_2026_20260809_v1')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--policy-json', type=Path, default=OPTIMISED_POLICY)
    return p.parse_args()


def _months(spec: str) -> list[str]:
    start, end = spec.split(':', 1)
    return pd.period_range(start, end, freq='M').astype(str).tolist()


def _symbol_path(symbol: str) -> Path:
    name = f"{str(symbol).lower().replace('/', '')}_15m.parquet"
    preferred = OHLCV_15M / name
    return preferred if preferred.exists() else LEGACY_OHLCV_15M / name


def _load_labels(months: list[str]) -> pd.DataFrame:
    frames = []
    cols = ['candidate_id', '__ts__', '__symbol__', 'label_valid', 'target_invalid', 'atr_1h']
    for month in months:
        path = LABEL_ROOT / 'parts' / f'month={month}' / 'side=long.parquet'
        if path.exists():
            frames.append(pd.read_parquet(path, columns=cols))
    if not frames:
        raise FileNotFoundError('No requested Pack-B long label partitions exist.')
    out = pd.concat(frames, ignore_index=True)
    if out.candidate_id.duplicated().any():
        raise ValueError('Candidate ID is duplicated in the exact label partitions.')
    out['__ts__'] = pd.to_datetime(out['__ts__'], utc=True)
    # Exact label validity is an evaluation property, never an inference or
    # execution-path eligibility predicate.  Missing exact ATR is repaired in
    # `_replay_symbol` from causal completed 15-minute bars.
    return out[['candidate_id', '__ts__', '__symbol__', 'atr_1h']]


def _coarse_causal_atr(ts: np.ndarray, opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> pd.Series:
    """Causal Wilder ATR(14) from four complete 15-minute bars per hour."""
    idx = pd.to_datetime(ts, utc=True)
    raw = pd.DataFrame({'open': opens, 'high': highs, 'low': lows, 'close': closes}, index=idx)
    hourly = raw.resample('1h', label='left', closed='left').agg(open=('open','first'), high=('high','max'), low=('low','min'), close=('close','last'))
    complete = raw.close.resample('1h', label='left', closed='left').count().eq(4)
    hourly.loc[~complete, :] = np.nan
    previous = hourly.close.shift(1)
    tr = pd.concat([hourly.high-hourly.low, (hourly.high-previous).abs(), (hourly.low-previous).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    atr.loc[complete.rolling(14, min_periods=14).sum().ne(14)] = np.nan
    atr.index = atr.index + pd.Timedelta(hours=1)
    return atr


def _load_15m(symbol: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    name = f"{str(symbol).lower().replace('/', '')}_15m.parquet"
    # Both roots are established Kraken coarse sources.  Union them
    # timestamp-wise so a partial refreshed file cannot hide valid history in
    # the legacy cache.  The preferred refreshed source wins overlaps.
    paths = [LEGACY_OHLCV_15M / name, OHLCV_15M / name]
    frames = []
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=['open', 'high', 'low', 'close'])
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError(f'15m OHLC has no datetime index: {path}')
        index = pd.DatetimeIndex(frame.index)
        frame.index = index.tz_localize('UTC') if index.tz is None else index.tz_convert('UTC')
        frames.append(frame)
    if not frames:
        return tuple(np.empty(0, dtype=np.float64) for _ in range(5))  # type: ignore[return-value]
    x = pd.concat(frames).sort_index(kind='stable')
    x = x.loc[~x.index.duplicated(keep='last')]
    index = pd.DatetimeIndex(x.index)
    order = np.argsort(index.asi8)
    ts = index.asi8[order].astype(np.int64, copy=False)
    keep = ~pd.Index(ts).duplicated(keep='last')
    order, ts = order[keep], ts[keep]
    return (
        ts, x.open.to_numpy(np.float64, copy=False)[order], x.high.to_numpy(np.float64, copy=False)[order],
        x.low.to_numpy(np.float64, copy=False)[order], x.close.to_numpy(np.float64, copy=False)[order],
    )


def _arrays_to_bars(
    ts: np.ndarray, opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {'open': opens, 'high': highs, 'low': lows, 'close': closes},
        index=pd.to_datetime(ts, utc=True),
    )


def _bars_to_arrays(
    bars: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bars = bars.sort_index(kind='stable')
    bars = bars.loc[~bars.index.duplicated(keep='last')]
    finite = np.isfinite(bars.loc[:, ['open', 'high', 'low', 'close']].to_numpy(np.float64)).all(axis=1)
    bars = bars.loc[finite]
    return (
        pd.DatetimeIndex(bars.index).asi8.astype(np.int64, copy=False),
        bars.open.to_numpy(np.float64, copy=False),
        bars.high.to_numpy(np.float64, copy=False),
        bars.low.to_numpy(np.float64, copy=False),
        bars.close.to_numpy(np.float64, copy=False),
    )


def _load_exact_minute_as_15m(
    symbol: str, start: pd.Timestamp, end: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load complete 15-minute bars resampled from the canonical exact store."""
    from scripts.materialize_strict_r3_frozen_policy_labels_v2 import _complete_15m_from_minute

    bars = _complete_15m_from_minute(EXACT_MINUTE_ROOT, symbol, start, end)
    if bars.empty:
        return tuple(np.empty(0, dtype=np.float64) for _ in range(5))  # type: ignore[return-value]
    if not np.isfinite(bars.loc[:, ['open', 'high', 'low', 'close']].to_numpy(np.float64)).any():
        return tuple(np.empty(0, dtype=np.float64) for _ in range(5))  # type: ignore[return-value]
    return _bars_to_arrays(bars)


def _paths_for_group(
    group: pd.DataFrame, ts: np.ndarray, opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    decision = pd.to_datetime(group.__ts__, utc=True).astype('int64').to_numpy() + 3_600_000_000_000
    pos = np.searchsorted(ts, decision, side='left')
    valid = pos < len(ts)
    safe = np.minimum(pos, max(len(ts) - 1, 0))
    valid &= ts[safe] == decision
    for bar in range(HORIZON_BARS):
        safe_bar = np.minimum(safe + bar, max(len(ts) - 1, 0))
        valid &= (safe + bar < len(ts)) & (ts[safe_bar] == decision + bar * 900_000_000_000)
    good = np.flatnonzero(valid)
    if not len(good):
        empty = np.empty((0, HORIZON_BARS), dtype=np.float32)
        return valid, empty, empty, empty, empty
    idx = pos[good, None] + np.arange(HORIZON_BARS, dtype=np.int64)[None, :]
    return valid, opens[idx].astype(np.float32), highs[idx].astype(np.float32), lows[idx].astype(np.float32), closes[idx].astype(np.float32)


def _coarse_paths_are_suspicious(
    valid: np.ndarray,
    path_high: np.ndarray,
    path_low: np.ndarray,
    *,
    maximum_flat_fraction: float = 0.20,
) -> bool:
    """Detect timestamp-complete but stale/forward-filled coarse price paths."""

    if not valid.any() or not len(path_high):
        return True
    scale = np.maximum(np.nanmedian(np.abs(path_high), axis=1), 1e-12)
    path_range = np.nanmax(path_high, axis=1) - np.nanmin(path_low, axis=1)
    flat = path_range <= (1e-10 * scale)
    return bool(float(flat.mean()) > maximum_flat_fraction)


def _replay_symbol(group: pd.DataFrame, policy: dict[str, float]) -> pd.DataFrame:
    result = group.copy().reset_index(drop=True)
    result['policy_path_valid'] = False
    result['policy_gross_bps'] = np.nan
    result['policy_net_bps'] = np.nan
    result['policy_exit_bar_15m'] = -1
    result['policy_exit_reason'] = 'invalid_path'
    result['policy_entry_price'] = np.nan
    result['policy_exit_price'] = np.nan
    result['policy_atr_source'] = 'unavailable'
    result['policy_market_data_source'] = 'unavailable'
    result['policy_market_data_quality'] = 'unavailable'
    ts, opens, highs, lows, closes = _load_15m(str(group.__symbol__.iloc[0]))
    # Match the frozen-label materializer: if the coarse file does not cover
    # the entire requested symbol interval, replace it with complete bars
    # resampled from the canonical exact-minute store.  This is an outcome
    # source only; ATR still uses completed bars strictly before the signal.
    coarse_complete = False
    coarse_suspicious = True
    if len(ts):
        coarse_valid, _, coarse_high, coarse_low, _ = _paths_for_group(
            result, ts, opens, highs, lows, closes,
        )
        coarse_complete = bool(coarse_valid.all())
        coarse_suspicious = _coarse_paths_are_suspicious(
            coarse_valid, coarse_high, coarse_low,
        )
    exact_used = False
    # The exact store is cheap enough to consult once per symbol and is the
    # authoritative source on overlaps.  Always doing so also detects a stale
    # interval embedded inside an otherwise moving multi-month coarse series.
    if EXACT_MINUTE_ROOT.exists():
        decision = pd.to_datetime(result.__ts__, utc=True) + pd.Timedelta(hours=1)
        exact_arrays = _load_exact_minute_as_15m(
            str(group.__symbol__.iloc[0]),
            decision.min() - pd.Timedelta(hours=15),
            decision.max() + pd.Timedelta(hours=12),
        )
        if len(exact_arrays[0]):
            coarse = _arrays_to_bars(ts, opens, highs, lows, closes) if len(ts) else pd.DataFrame()
            exact = _arrays_to_bars(*exact_arrays)
            # Exact-minute resampling must replace timestamp-complete but stale
            # coarse bars.  Coarse data fills only intervals absent from exact.
            combined = exact.combine_first(coarse) if len(coarse) else exact
            ts, opens, highs, lows, closes = _bars_to_arrays(combined)
            exact_used = True
    if coarse_suspicious and not exact_used:
        result['policy_market_data_source'] = 'coarse_15m'
        result['policy_market_data_quality'] = 'rejected_stale_without_exact_fallback'
        return result
    if not len(ts):
        return result
    result['policy_market_data_source'] = 'exact_1m_resampled_15m' if exact_used else 'coarse_15m'
    result['policy_market_data_quality'] = 'complete_nonstale'
    fallback_atr = _coarse_causal_atr(ts, opens, highs, lows, closes)
    current_atr = pd.to_numeric(result.atr_1h, errors='coerce')
    fallback = pd.to_datetime(result.__ts__, utc=True).map(fallback_atr)
    use_fallback = ~np.isfinite(current_atr) | current_atr.le(0)
    result['policy_atr'] = current_atr.where(~use_fallback, fallback)
    result.loc[~use_fallback, 'policy_atr_source'] = 'exact_label_atr'
    result.loc[use_fallback & result.policy_atr.gt(0), 'policy_atr_source'] = 'coarse_15m_wilder14'
    valid, f_open, f_high, f_low, f_close = _paths_for_group(result, ts, opens, highs, lows, closes)
    pos = np.flatnonzero(valid)
    if not len(pos):
        return result
    atr = result.iloc[pos].policy_atr.to_numpy(np.float64)
    usable = np.isfinite(atr) & (atr > 0)
    if not usable.all():
        keep = np.flatnonzero(usable)
        pos, f_open, f_high, f_low, f_close, atr = pos[keep], f_open[keep], f_high[keep], f_low[keep], f_close[keep], atr[keep]
        if not len(pos): return result
    entry = f_open[:, 0].astype(np.float64)
    run = pd.DataFrame({
        'timestamp': result.iloc[pos].__ts__.to_numpy(),
        'symbol': result.iloc[pos].__symbol__.astype(str).to_numpy(),
        'side': np.ones(len(pos), dtype=np.float32),
        'rank_pct': np.ones(len(pos), dtype=np.float32),
        'barrier_pct': atr / entry,
        # Explicitly neutralise simulator spread and price-gap assumptions.
        'expected_half_spread_bps': np.zeros(len(pos)), 'exit_quote_half_spread_bps': np.zeros(len(pos)),
        'entry_slippage_proxy_bps': np.zeros(len(pos)), 'market_mode': 'perps',
    })
    sim = simulate_and_score(
        run, f_open, f_high, f_low, f_close,
        cost_pct=0.0, size_power=1.0, replay_timeframe='15m', market_mode='perps',
        sl_mult=float(policy['sl_mult']), sl_abs_cap_pct=0.0, trailing_activation_mult=float(policy['trailing_activation_mult']),
        trailing_activation_cap_pct=0.0, trailing_activation_max_bars=HORIZON_BARS,
        fixed_trailing_gap_mult=float(policy['fixed_trailing_gap_mult']), capital_protect_mfe_mult=0.0,
        adverse_exit_enabled=False, hard_tp_abs_pct=0.0,
        max_concurrent_trades=max(len(run), 1), max_concurrent_per_asset=max(len(run), 1),
        max_new_entries_per_bar=max(len(run), 1),
    )
    if not np.asarray(sim['selected_mask'], dtype=bool).all():
        raise ValueError('Candidate-local exit materialisation unexpectedly applied concurrency.')
    gross = np.asarray(sim['gross_returns'], dtype=np.float64) * 10_000.0
    result.loc[pos, 'policy_path_valid'] = np.isfinite(gross)
    result.loc[pos, 'policy_gross_bps'] = gross
    result.loc[pos, 'policy_net_bps'] = gross - COST_BPS
    result.loc[pos, 'policy_exit_bar_15m'] = np.asarray(sim['exit_bars'], dtype=np.int16)
    result.loc[pos, 'policy_exit_reason'] = np.asarray(sim['exit_reason'], dtype=object)
    result.loc[pos, 'policy_entry_price'] = np.asarray(sim['entry_prices'], dtype=np.float64)
    result.loc[pos, 'policy_exit_price'] = np.asarray(sim['exit_prices'], dtype=np.float64)
    return result


def main() -> None:
    args = _args()
    policy_payload = json.loads(args.policy_json.read_text())
    policy = {k: float(policy_payload['winner'][k]) for k in ('sl_mult','trailing_activation_mult','fixed_trailing_gap_mult')}
    months = _months(args.months)
    if args.out_dir.exists() and not args.resume:
        raise FileExistsError(f'{args.out_dir} already exists; use --resume.')
    args.out_dir.mkdir(parents=True, exist_ok=bool(args.resume))
    pred = pd.read_parquet(PREDICTIONS, columns=['candidate_id', '__ts__', '__symbol__', 'month', 'final_score'])
    pred['__ts__'] = pd.to_datetime(pred['__ts__'], utc=True)
    pred = pred[pred.month.isin(months) & np.isfinite(pred.final_score)].copy()
    x = pred.merge(_load_labels(months), on=['candidate_id', '__ts__', '__symbol__'], how='inner', validate='one_to_one')
    x['__decision_ts__'] = x.__ts__ + pd.Timedelta(hours=1)
    checkpoints = args.out_dir / 'symbol_parts'; checkpoints.mkdir(exist_ok=True)
    symbols = sorted(x.__symbol__.astype(str).unique())
    for number, symbol in enumerate(symbols, 1):
        cp = checkpoints / f'{hashlib.sha256(symbol.encode()).hexdigest()[:20]}.parquet'
        group = x[x.__symbol__.eq(symbol)].copy()
        if cp.exists():
            existing = pd.read_parquet(cp, columns=['candidate_id'])
            if set(existing.candidate_id) != set(group.candidate_id):
                raise ValueError(f'checkpoint identity mismatch: {symbol}')
        else:
            _replay_symbol(group, policy).to_parquet(cp, index=False, compression='zstd')
        if number % 20 == 0 or number == len(symbols):
            gc.collect(); pa.default_memory_pool().release_unused()
            print(json.dumps({'event': 'policy_symbols_complete', 'completed': number, 'total': len(symbols)}), flush=True)
    pieces = [pd.read_parquet(p) for p in sorted(checkpoints.glob('*.parquet'))]
    all_rows = pd.concat(pieces, ignore_index=True)
    if len(all_rows) != len(x) or all_rows.candidate_id.duplicated().any():
        raise ValueError('Policy checkpoint assembly changed candidate identity/cardinality.')
    all_rows.to_parquet(args.out_dir / 'candidate_policy_outcomes.parquet', index=False, compression='zstd')
    coverage = all_rows.groupby('month', as_index=False).agg(rows=('candidate_id', 'size'), valid_policy_rows=('policy_path_valid', 'sum'))
    coverage['policy_path_coverage'] = coverage.valid_policy_rows / coverage.rows
    coverage.to_parquet(args.out_dir / 'policy_coverage.parquet', index=False)
    manifest = {
        'score': str(PREDICTIONS), 'labels': str(LABEL_ROOT), 'ohlcv_15m': str(OHLCV_15M), 'months': months,
        'entry': 'first 15-minute open at signal close + 1h',
        'atr': 'exact label ATR when available; otherwise causal coarse Wilder ATR(14) from four complete prior 15-minute bars per hour',
        'policy_engine': 'extreme_price_movements.simple_policy_optimiser.simulate_and_score',
        'exit': {'optimiser': str(args.policy_json), 'winner': policy, 'timeout': 'H12', 'ordering': 'canonical simulator ordering'},
        'cost': '100 bps round trip, deducted exactly once after gross simulator outcome; no additional spread/slippage proxy',
        'rows': int(len(all_rows)), 'valid_policy_rows': int(all_rows.policy_path_valid.sum()),
    }
    (args.out_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({**manifest, 'output': str(args.out_dir)}))


if __name__ == '__main__':
    main()
