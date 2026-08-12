#!/usr/bin/env python3
"""Reconcile stored canonical C0 and reconstructed strict-R3 scores.

The comparison is deliberately restricted to identical 2025 long candidate
IDs.  It evaluates both scores under (a) the stored H12 proxy outcome and
(b) the independently materialised exact H12 TP6/SL4 outcome.  Thus it
separates an outcome-contract change from a score-reconstruction change.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
CODEX_ROOT = Path('/Users/remyroche/Documents/Codex')
STORED = CODEX_ROOT / 'artifacts/canonical_c0_frozen_strict_r3_proxy_20260808_v1/c0_frozen_strict_r3_proxy_scores.parquet'
RECONSTRUCTED = ROOT / 'data_perp/artifacts/strict_r3_exact_h12_2025_2026_v16_approved_15m_proxy/predictions.parquet'
LABEL_ROOTS = (
    ROOT / 'data_perp/artifacts/stage_i_historical_tp6_sl4_h12_r3_20260803_v1',
    ROOT / 'data_perp/artifacts/stage_i_packb_tp6_sl4_h12_r3_20260803_v1',
)
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--out-dir', type=Path,
        default=ROOT / 'data_perp/artifacts/base_plus_consensus25_exact_reconciliation_20260809_v1',
    )
    return parser.parse_args()


def _read_exact_labels() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    wanted = [
        'candidate_id', '__ts__', 'side_name', 'label_valid', 'target_invalid',
        't4_tp6_sl4_gross_bps', 't4_tp6_sl4_net_bps',
    ]
    for root in LABEL_ROOTS:
        for path in sorted((root / 'parts').glob('month=2025-*/side=long.parquet')):
            available = pd.read_parquet(path).columns.tolist()
            cols = [c for c in wanted if c in available]
            part = pd.read_parquet(path, columns=cols)
            if 'side_name' not in part:
                part['side_name'] = 'long'
            part['__ts__'] = pd.to_datetime(part['__ts__'], utc=True)
            part['label_valid'] = part['label_valid'].astype(bool)
            part['target_invalid'] = part['target_invalid'].astype(bool)
            frames.append(part)
    if not frames:
        raise FileNotFoundError('No 2025 long exact H12 label parts found.')
    labels = pd.concat(frames, ignore_index=True)
    # Historical and Pack-B roots can overlap only when a handoff duplicated a
    # partition.  They must agree; taking neither silently avoids mixing label
    # contracts.
    if labels.candidate_id.duplicated().any():
        duplicated = labels[labels.candidate_id.duplicated(False)].sort_values('candidate_id')
        numeric = ['t4_tp6_sl4_gross_bps', 't4_tp6_sl4_net_bps']
        bad = duplicated.groupby('candidate_id')[numeric].nunique(dropna=False).max(axis=1).gt(1)
        if bad.any():
            raise ValueError('Overlapping exact label roots disagree on a candidate outcome.')
        labels = labels.drop_duplicates('candidate_id', keep='first')
    labels['exact_valid'] = (
        labels['label_valid'] & ~labels['target_invalid']
        & np.isfinite(pd.to_numeric(labels['t4_tp6_sl4_net_bps'], errors='coerce'))
        & np.isfinite(pd.to_numeric(labels['t4_tp6_sl4_gross_bps'], errors='coerce'))
    )
    return labels.rename(columns={
        't4_tp6_sl4_gross_bps': 'exact_gross_bps',
        't4_tp6_sl4_net_bps': 'exact_net_bps',
    })


def _tail_rows(x: pd.DataFrame, score: str, outcome: str, period: str) -> list[dict]:
    y = x[np.isfinite(x[score]) & np.isfinite(x[f'{outcome}_net_bps'])].copy()
    rows: list[dict] = []
    if not len(y):
        return rows
    rho = float(spearmanr(y[score], y[f'{outcome}_net_bps']).statistic)
    for tail in TAILS:
        n = max(1, int(np.ceil(len(y) * tail)))
        top = y.nlargest(n, score)
        rows.append({
            'period': period,
            'score': score,
            'outcome': outcome,
            'tail_fraction': tail,
            'rows_available': int(len(y)),
            'trades': n,
            'gross_bps_per_trade': float(top[f'{outcome}_gross_bps'].mean()),
            'net_bps_per_trade': float(top[f'{outcome}_net_bps'].mean()),
            'score_net_spearman': rho,
        })
    return rows


def main() -> None:
    args = _args()
    if args.out_dir.exists():
        raise FileExistsError(f'{args.out_dir} exists; choose a new immutable output directory.')
    args.out_dir.mkdir(parents=True)

    stored = pd.read_parquet(STORED, columns=[
        'candidate_id', '__ts__', 'month', 'tp6_net_bps', 'tp6_gross_bps', 'c0_score',
    ]).rename(columns={
        'month': 'stored_month', 'tp6_net_bps': 'proxy_net_bps',
        'tp6_gross_bps': 'proxy_gross_bps', 'c0_score': 'stored_base_plus_consensus25',
    })
    stored['__ts__'] = pd.to_datetime(stored['__ts__'], utc=True)
    stored = stored[stored['__ts__'].dt.year.eq(2025)].copy()
    if stored.candidate_id.duplicated().any():
        raise ValueError('Stored score has duplicate candidate IDs.')

    rec = pd.read_parquet(RECONSTRUCTED, columns=['candidate_id', '__ts__', 'month', 'final_score'])
    rec['__ts__'] = pd.to_datetime(rec['__ts__'], utc=True)
    rec = rec[rec['__ts__'].dt.year.eq(2025)].rename(columns={
        'month': 'reconstructed_month', 'final_score': 'reconstructed_base_plus_consensus25',
    })
    if rec.candidate_id.duplicated().any():
        raise ValueError('Reconstructed score has duplicate candidate IDs.')

    labels = _read_exact_labels()
    common = stored.merge(rec, on='candidate_id', how='inner', suffixes=('_stored', '_reconstructed'))
    if not common['__ts___stored'].eq(common['__ts___reconstructed']).all():
        raise ValueError('Same candidate ID is assigned a different decision timestamp.')
    common = common.rename(columns={'__ts___stored': '__ts__'}).drop(columns=['__ts___reconstructed'])
    common = common.merge(labels.drop(columns=['__ts__', 'side_name'], errors='ignore'), on='candidate_id', how='left')
    common['month'] = common['__ts__'].dt.to_period('M').astype(str)
    common = common[np.isfinite(common.proxy_net_bps) & common.exact_valid].copy()
    if not len(common):
        raise ValueError('No common rows with both proxy and exact outcomes.')

    score_cols = ['stored_base_plus_consensus25', 'reconstructed_base_plus_consensus25']
    metric_rows: list[dict] = []
    for score in score_cols:
        for outcome in ('proxy', 'exact'):
            metric_rows.extend(_tail_rows(common, score, outcome, '2025_pooled_same_ids'))
            for month, part in common.groupby('month', sort=True):
                metric_rows.extend(_tail_rows(part, score, outcome, month))
    metrics = pd.DataFrame(metric_rows)
    metrics.to_parquet(args.out_dir / 'global_and_monthly_metrics.parquet', index=False)

    agreement_rows = []
    for tail in TAILS:
        n = max(1, int(np.ceil(len(common) * tail)))
        old_ids = set(common.nlargest(n, 'stored_base_plus_consensus25').candidate_id)
        new_ids = set(common.nlargest(n, 'reconstructed_base_plus_consensus25').candidate_id)
        agreement_rows.append({
            'tail_fraction': tail, 'trades': n, 'top_set_jaccard': len(old_ids & new_ids) / len(old_ids | new_ids),
        })
    agreement = pd.DataFrame(agreement_rows)
    agreement['score_spearman'] = spearmanr(
        common['stored_base_plus_consensus25'], common['reconstructed_base_plus_consensus25'],
    ).statistic
    agreement.to_parquet(args.out_dir / 'score_agreement.parquet', index=False)

    delta = common.exact_net_bps - common.proxy_net_bps
    outcome = pd.DataFrame([{
        'common_rows': int(len(common)),
        'first_month': common.month.min(), 'last_month': common.month.max(),
        'proxy_exact_net_mean_difference_bps': float(delta.mean()),
        'proxy_exact_net_median_difference_bps': float(delta.median()),
        'proxy_exact_net_mae_bps': float(delta.abs().mean()),
        'proxy_exact_net_rmse_bps': float(np.sqrt(np.mean(np.square(delta)))),
        'outcome_changed_over_1bp_fraction': float(delta.abs().gt(1.).mean()),
        'outcome_changed_over_100bp_fraction': float(delta.abs().gt(100.).mean()),
    }])
    outcome.to_parquet(args.out_dir / 'outcome_contract_difference.parquet', index=False)
    common.to_parquet(args.out_dir / 'same_id_scores_and_outcomes.parquet', index=False, compression='zstd')

    manifest = {
        'purpose': 'same-ID stored-vs-reconstructed score and proxy-vs-exact H12 reconciliation',
        'stored_score': str(STORED), 'reconstructed_score': str(RECONSTRUCTED),
        'exact_label_roots': [str(p) for p in LABEL_ROOTS],
        'population': '2025 long candidates present in both score artifacts with both valid exact H12 and proxy outcomes',
        'exact_contract': 'H12 TP +6 ATR / SL -4 ATR; decision-minute entry; 100 bps cost once',
        'proxy_contract': 'stored canonical C0 TP6 proxy outcome; unchanged',
        'rows': int(len(common)),
    }
    (args.out_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({**manifest, 'output': str(args.out_dir)}))


if __name__ == '__main__':
    main()
