#!/usr/bin/env python3
"""Coverage-aware exact-H12 evaluation for the strict-R3 2026 replay.

The early-2026 minute/ATR label substrate is incomplete.  This utility keeps
the score unchanged and reports a primary May--July 2026 pooled ranking, where
exact label coverage is high, separately from a transparent monthly coverage
audit for the whole year.  It never fills invalid paths or treats them as a
negative economic outcome.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / 'data_perp/artifacts/strict_r3_exact_h12_2025_2026_v16_approved_15m_proxy/predictions.parquet'
LABEL_ROOT = ROOT / 'data_perp/artifacts/stage_i_packb_tp6_sl4_h12_r3_20260803_v1'
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10)
PRIMARY_MONTHS = ('2026-05', '2026-06', '2026-07')


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', type=Path, default=ROOT / 'data_perp/artifacts/strict_r3_exact_h12_2026_coverage_aware_metrics_20260809_v1')
    return p.parse_args()


def _metric(x: pd.DataFrame, period: str, level: str) -> list[dict]:
    rows: list[dict] = []
    if not len(x):
        return rows
    rho = float(spearmanr(x.final_score, x.net_bps).statistic)
    for tail in TAILS:
        n = max(1, int(np.ceil(len(x) * tail)))
        top = x.nlargest(n, 'final_score')
        rows.append({
            'period': period, 'level': level, 'tail_fraction': tail,
            'trades': n, 'rows_available': int(len(x)),
            'gross_bps_per_trade': float(top.gross_bps.mean()),
            'net_bps_per_trade': float(top.net_bps.mean()),
            'net_sum_bps': float(top.net_bps.sum()), 'score_net_spearman': rho,
        })
    return rows


def main() -> None:
    args = _args()
    if args.out_dir.exists():
        raise FileExistsError(f'{args.out_dir} exists; select a new immutable path.')
    args.out_dir.mkdir(parents=True)
    pred = pd.read_parquet(PREDICTIONS, columns=['candidate_id', '__ts__', 'month', 'final_score', 'gross_bps', 'net_bps'])
    pred['__ts__'] = pd.to_datetime(pred['__ts__'], utc=True)
    pred = pred[pred.month.astype(str).str.startswith('2026-')].copy()
    pred = pred[np.isfinite(pred.final_score) & np.isfinite(pred.gross_bps) & np.isfinite(pred.net_bps)].copy()
    if pred.candidate_id.duplicated().any():
        raise ValueError('Scored rows must be identity-unique.')

    coverage = pd.read_parquet(LABEL_ROOT / 'coverage.parquet')
    coverage = coverage[(coverage.side == 'long') & coverage.month.astype(str).str.startswith('2026-')].copy()
    coverage['label_valid_fraction'] = coverage.valid_rows / coverage.rows
    scored = pred.groupby('month', as_index=False).agg(scored_rows=('candidate_id', 'size'))
    coverage = coverage.merge(scored, on='month', how='left').fillna({'scored_rows': 0})
    coverage['score_over_label_valid_fraction'] = coverage.scored_rows / coverage.valid_rows.replace(0, np.nan)
    coverage.to_parquet(args.out_dir / 'monthly_label_and_score_coverage.parquet', index=False)

    metrics: list[dict] = []
    primary = pred[pred.month.isin(PRIMARY_MONTHS)].copy()
    metrics.extend(_metric(primary, '2026-05_to_2026-07', 'pooled_global_primary_exact_support'))
    for month, x in pred.groupby('month', sort=True):
        metrics.extend(_metric(x, str(month), 'monthly_exact_support'))
    pred['week'] = pred.__ts__.dt.to_period('W-MON').apply(lambda x: str(x.start_time.date()))
    for week, x in pred[pred.month.isin(PRIMARY_MONTHS)].groupby('week', sort=True):
        metrics.extend(_metric(x, str(week), 'weekly_primary_exact_support'))
    pd.DataFrame(metrics).to_parquet(args.out_dir / 'exact_h12_metrics.parquet', index=False)

    manifest = {
        'purpose': 'coverage-aware exact H12 TP6/SL4 metrics, score unchanged',
        'predictions': str(PREDICTIONS), 'label_coverage': str(LABEL_ROOT / 'coverage.parquet'),
        'exit_contract': 'H12 TP +6 ATR / SL -4 ATR; decision-minute entry; adverse same-minute precedence; gross minus 100 bps once',
        'primary_window': list(PRIMARY_MONTHS),
        'primary_rule': 'pooled global ranking only across scored rows with high exact label availability; Jan-Apr remain reported only as low-coverage diagnostics',
        'primary_rows': int(len(primary)), 'all_2026_scored_rows': int(len(pred)),
    }
    (args.out_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
