#!/usr/bin/env python3
"""Replay the frozen 15-minute exit policy on strict-R3 2026 scored rows.

Policy: enter at the existing decision-minute open; stop at -3 ATR; arm a
fixed trailing stop only after a *previous* 15-minute bar has reached +0.5
ATR; trail the best prior favourable excursion by 0.25 ATR; timeout at H12.
The ordering deliberately mirrors the canonical simple-policy simulator:
stop first, then trailing based on prior-bar MFE, then update MFE.  A fixed
100-bps round-trip cost is deducted once.  This is a candidate-local replay,
not a portfolio-constrained simulation.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd
import pyarrow as pa
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.materialize_packb_tp6_sl4_h12_labels import _minute_path_pruned
PREDICTIONS = ROOT / 'data_perp/artifacts/strict_r3_exact_h12_2025_2026_v16_approved_15m_proxy/predictions.parquet'
LABEL_ROOT = ROOT / 'data_perp/artifacts/stage_i_packb_tp6_sl4_h12_r3_20260803_v1'
MINUTE_ROOT = ROOT / 'data_perp/exchanges/krakenfutures/execution_1m/ohlcv'
PRIMARY_MONTHS = ('2026-05', '2026-06', '2026-07')
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10)
HORIZON_BARS = 48
COST_BPS = 100.0


@nb.njit(cache=True)
def _replay_long_policy(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, starts: np.ndarray,
    entry: np.ndarray, atr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return gross bps, exit-bar index and reason code (0 stop/1 trail/2 TO)."""
    n = len(starts)
    gross = np.full(n, np.nan, np.float64)
    exit_bar = np.full(n, -1, np.int16)
    reason = np.full(n, -1, np.int8)
    for row in range(n):
        s, e, a = starts[row], entry[row], atr[row]
        if s < 0 or s + HORIZON_BARS > high.shape[0] or not np.isfinite(e) or not np.isfinite(a) or e <= 0. or a <= 0.:
            continue
        max_fav = 0.0
        armed = False
        resolved = False
        for bar in range(HORIZON_BARS):
            pos = s + bar
            hi, lo, cl = high[pos], low[pos], close[pos]
            if not np.isfinite(hi) or not np.isfinite(lo) or not np.isfinite(cl):
                resolved = True
                break
            # 1. Pessimistic stop precedence.
            if lo <= e - 3.0 * a:
                gross[row] = -3.0 * a / e * 10000.0
                exit_bar[row] = bar
                reason[row] = 0
                resolved = True
                break
            # 2. Trailing state uses only the preceding bars' MFE.
            if max_fav > 0.5 * a:
                armed = True
            if armed:
                trail_px = e + max(max_fav - 0.25 * a, 0.0)
                if lo <= trail_px:
                    gross[row] = (trail_px - e) / e * 10000.0
                    exit_bar[row] = bar
                    reason[row] = 1
                    resolved = True
                    break
            # 3. Current high informs the following 15-minute bar only.
            fav = hi - e
            if fav > max_fav:
                max_fav = fav
        if not resolved:
            gross[row] = (close[s + HORIZON_BARS - 1] - e) / e * 10000.0
            exit_bar[row] = HORIZON_BARS - 1
            reason[row] = 2
    return gross, exit_bar, reason


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', type=Path, default=ROOT / 'data_perp/artifacts/strict_r3_15m_frozen_exit_policy_2026_coverage_aware_20260809_v1')
    p.add_argument('--resume', action='store_true', help='reuse identity-checked per-symbol checkpoints')
    return p.parse_args()


def _labels() -> pd.DataFrame:
    columns = ['candidate_id', '__ts__', '__symbol__', 'label_valid', 'target_invalid', 'tp6_sl4_entry_price', 'atr_1h']
    parts = []
    for month in PRIMARY_MONTHS:
        path = LABEL_ROOT / 'parts' / f'month={month}' / 'side=long.parquet'
        parts.append(pd.read_parquet(path, columns=columns))
    x = pd.concat(parts, ignore_index=True)
    x['__ts__'] = pd.to_datetime(x['__ts__'], utc=True)
    x['policy_label_input_valid'] = (
        x.label_valid.astype(bool) & ~x.target_invalid.astype(bool)
        & np.isfinite(pd.to_numeric(x.tp6_sl4_entry_price, errors='coerce'))
        & np.isfinite(pd.to_numeric(x.atr_1h, errors='coerce'))
    )
    return x[['candidate_id', '__ts__', '__symbol__', 'tp6_sl4_entry_price', 'atr_1h', 'policy_label_input_valid']]


def _metrics(x: pd.DataFrame, period: str, level: str) -> list[dict]:
    out: list[dict] = []
    if not len(x):
        return out
    rho = float(spearmanr(x.final_score, x.policy_net_bps).statistic)
    for tail in TAILS:
        n = max(1, int(np.ceil(len(x) * tail)))
        top = x.nlargest(n, 'final_score')
        out.append({
            'period': period, 'level': level, 'tail_fraction': tail,
            'rows_available': int(len(x)), 'trades': n,
            'gross_bps_per_trade': float(top.policy_gross_bps.mean()),
            'net_bps_per_trade': float(top.policy_net_bps.mean()),
            'net_sum_bps': float(top.policy_net_bps.sum()), 'score_net_spearman': rho,
        })
    return out


def main() -> None:
    args = _args()
    if args.out_dir.exists() and not args.resume:
        raise FileExistsError(f'{args.out_dir} exists; choose a new immutable output directory.')
    args.out_dir.mkdir(parents=True, exist_ok=bool(args.resume))
    pred = pd.read_parquet(PREDICTIONS, columns=['candidate_id', '__ts__', '__symbol__', 'month', 'final_score'])
    pred['__ts__'] = pd.to_datetime(pred['__ts__'], utc=True)
    pred = pred[pred.month.isin(PRIMARY_MONTHS) & np.isfinite(pred.final_score)].copy()
    x = pred.merge(_labels(), on=['candidate_id', '__ts__', '__symbol__'], how='inner', validate='one_to_one')
    x = x[x.policy_label_input_valid].copy()
    x['__decision_ts__'] = x.__ts__ + pd.Timedelta(hours=1)
    results: list[pd.DataFrame] = []
    coverage_rows: list[dict] = []
    symbol_count = int(x.__symbol__.nunique())
    checkpoint_root = args.out_dir / 'symbol_parts'
    checkpoint_root.mkdir(exist_ok=True)
    for symbol_index, (symbol, group) in enumerate(x.groupby('__symbol__', sort=True), 1):
        checkpoint = checkpoint_root / f"{hashlib.sha256(str(symbol).encode()).hexdigest()[:20]}.parquet"
        if checkpoint.exists():
            part = pd.read_parquet(checkpoint)
            if set(part.candidate_id) != set(group.candidate_id):
                raise ValueError(f'symbol checkpoint identity mismatch: {symbol}')
            coverage_rows.append({
                'symbol': str(symbol), 'rows': int(len(part)),
                'policy_path_valid_rows': int(part.policy_path_valid.astype(bool).sum()),
                'policy_path_coverage': float(part.policy_path_valid.astype(bool).mean()),
            })
            results.append(part.loc[part.policy_path_valid.astype(bool)].copy())
            continue
        start = group.__decision_ts__.min()
        end = group.__decision_ts__.max() + pd.Timedelta(hours=12)
        minute_symbol = str(symbol).replace('/', '_')
        minute = _minute_path_pruned(MINUTE_ROOT, minute_symbol, start, end)
        bars = minute.resample('15min', label='left', closed='left').agg(
            open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last'), count=('close', 'count')
        )
        bars.loc[bars['count'].ne(15), ['open', 'high', 'low', 'close']] = np.nan
        starts = bars.index.get_indexer(pd.DatetimeIndex(group.__decision_ts__))
        gross, exit_bar, reason = _replay_long_policy(
            bars.high.to_numpy(float), bars.low.to_numpy(float), bars.close.to_numpy(float),
            starts.astype(np.int64), group.tp6_sl4_entry_price.to_numpy(float), group.atr_1h.to_numpy(float),
        )
        part = group.copy()
        part['policy_gross_bps'] = gross
        part['policy_net_bps'] = gross - COST_BPS
        part['policy_exit_bar_15m'] = exit_bar
        part['policy_exit_reason'] = pd.Categorical.from_codes(
            np.where(reason < 0, 3, reason), categories=['sl_3atr', 'trailing_0p25atr', 'timeout_h12', 'invalid_path']
        ).astype(str)
        valid = np.isfinite(gross)
        part['policy_path_valid'] = valid
        coverage_rows.append({
            'symbol': str(symbol), 'rows': int(len(part)), 'policy_path_valid_rows': int(valid.sum()),
            'policy_path_coverage': float(valid.mean()),
        })
        part.to_parquet(checkpoint, index=False, compression='zstd')
        results.append(part.loc[valid])
        # PyArrow can retain parquet scanner buffers across the 226 symbol
        # reads.  Release each bounded source slice rather than letting an
        # otherwise small 15-minute replay grow with the whole universe.
        del minute, bars, part
        if symbol_index % 8 == 0:
            gc.collect()
            pa.default_memory_pool().release_unused()
        if symbol_index % 32 == 0 or symbol_index == symbol_count:
            print(json.dumps({'event': 'symbols_replayed', 'completed': symbol_index, 'total': symbol_count}), flush=True)
    scored = pd.concat(results, ignore_index=True)
    scored['week'] = scored.__ts__.dt.to_period('W-MON').apply(lambda z: str(z.start_time.date()))
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_parquet(args.out_dir / 'policy_path_coverage.parquet', index=False)
    scored.to_parquet(args.out_dir / 'scored_policy_outcomes.parquet', index=False, compression='zstd')
    metric_rows = _metrics(scored, '2026-05_to_2026-07', 'pooled_global_primary_policy_support')
    for month, part in scored.groupby('month', sort=True):
        metric_rows += _metrics(part, str(month), 'monthly_policy_support')
    for week, part in scored.groupby('week', sort=True):
        metric_rows += _metrics(part, str(week), 'weekly_policy_support')
    pd.DataFrame(metric_rows).to_parquet(args.out_dir / 'policy_metrics.parquet', index=False)
    manifest = {
        'score': str(PREDICTIONS), 'label_input': str(LABEL_ROOT), 'primary_months': list(PRIMARY_MONTHS),
        'entry': 'existing next-minute decision entry at signal close + 1h',
        'bar_resolution': '15-minute OHLC from exact minute source; all 15 constituent minute bars required',
        'exit_policy': 'SL 3 ATR; trailing activation 0.5 ATR; fixed giveback 0.25 ATR; H12 timeout; stop precedence; trailing uses preceding-bar MFE',
        'cost': '100 bps round trip deducted once',
        'scope': 'candidate-local outcome replay; no portfolio constraints or position concurrency',
        'input_rows': int(len(x)), 'policy_valid_rows': int(len(scored)),
    }
    (args.out_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({**manifest, 'output': str(args.out_dir)}))


if __name__ == '__main__':
    main()
