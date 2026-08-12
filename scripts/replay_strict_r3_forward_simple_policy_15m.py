#!/usr/bin/env python3
"""Materialise policy exits for an unlabeled forward strict-R3 score panel.

The scorer supplies no outcome data.  This utility joins only identity and
decision-time score fields, derives a causal coarse ATR from prior completed
15-minute candles, and applies the frozen SimplePolicyOptimiser exit contract.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.replay_strict_r3_simple_policy_15m import (  # noqa: E402
    COST_BPS,
    OPTIMISED_POLICY,
    _replay_symbol,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--policy-json', type=Path, default=OPTIMISED_POLICY)
    parser.add_argument('--start', default=None, help='optional inclusive decision timestamp')
    parser.add_argument('--end', default=None, help='optional exclusive decision timestamp')
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f'{args.out_dir} exists; immutable forward output required')
    args.out_dir.mkdir(parents=True)
    policy_doc = json.loads(args.policy_json.read_text())
    policy = {key: float(policy_doc['winner'][key]) for key in (
        'sl_mult', 'trailing_activation_mult', 'fixed_trailing_gap_mult'
    )}
    filters = None
    if args.start is not None or args.end is not None:
        filters = []
        if args.start is not None:
            filters.append(('__decision_ts__', '>=', pd.Timestamp(args.start, tz='UTC')))
        if args.end is not None:
            filters.append(('__decision_ts__', '<', pd.Timestamp(args.end, tz='UTC')))
    scores = pd.read_parquet(args.predictions, filters=filters)
    candidates = pd.read_parquet(
        args.candidates,
        columns=['candidate_id', '__ts__', '__decision_ts__', '__symbol__', 'side_name'],
        filters=filters,
    )
    x = candidates.merge(scores, on=['candidate_id', '__ts__', '__decision_ts__', '__symbol__', 'side_name'], how='inner', validate='one_to_one')
    x['__ts__'] = pd.to_datetime(x['__ts__'], utc=True)
    x['__decision_ts__'] = pd.to_datetime(x['__decision_ts__'], utc=True)
    if x.empty or x.candidate_id.duplicated().any() or not x.side_name.astype(str).eq('long').all():
        raise ValueError('Forward prediction identity/side contract failed.')
    x['month'] = x.__ts__.dt.to_period('M').astype(str)
    # Forward labels do not exist at scoring time.  The imported replay helper
    # falls back to decision-time 15m Wilder ATR whenever this is missing.
    x['atr_1h'] = np.nan
    parts = []
    for number, (symbol, group) in enumerate(x.groupby('__symbol__', sort=True), 1):
        parts.append(_replay_symbol(group.copy(), policy))
        if number % 20 == 0 or number == x.__symbol__.nunique():
            print(json.dumps({'event': 'policy_symbols_complete', 'completed': number, 'total': int(x.__symbol__.nunique())}), flush=True)
    out = pd.concat(parts, ignore_index=True).sort_values(['__ts__', '__symbol__'], kind='stable')
    if len(out) != len(x) or out.candidate_id.duplicated().any():
        raise ValueError('Forward policy replay changed candidate identity/cardinality.')
    # Admission/calibration may consume a realised outcome only once the full
    # H12 contract has resolved.  Persist the conservative availability time
    # explicitly so downstream maps cannot infer it from the evaluation row.
    out['policy_label_available_ts'] = out['__decision_ts__'] + pd.Timedelta(hours=12)
    out.to_parquet(args.out_dir / 'candidate_policy_outcomes.parquet', index=False, compression='zstd')
    coverage = out.groupby('month', as_index=False).agg(rows=('candidate_id', 'size'), valid_policy_rows=('policy_path_valid', 'sum'))
    coverage['policy_path_coverage'] = coverage.valid_policy_rows / coverage.rows
    coverage.to_parquet(args.out_dir / 'policy_coverage.parquet', index=False)
    (args.out_dir / 'run_manifest.json').write_text(json.dumps({
        'schema': 'strict_r3_forward_simple_policy_15m_v1',
        'predictions': str(args.predictions), 'candidate_source': str(args.candidates),
        'entry': 'first 15-minute open at signal close +1h',
        'atr': 'causal Wilder ATR(14) from complete prior 15-minute candles',
        'exit': {'winner': policy, 'timeout': 'H12'},
        'cost': f'{COST_BPS:g} bps deducted exactly once after gross simulator outcome',
        'rows': int(len(out)), 'valid_policy_rows': int(out.policy_path_valid.sum()),
        'decision_start': args.start, 'decision_end_exclusive': args.end,
        'market_data_precedence': 'exact-minute resampled 15m replaces coarse overlaps; coarse fills exact gaps',
        'stale_coarse_rule': 'flat-path share >20% is rejected when no exact fallback is available',
    }, indent=2))
    print(json.dumps({'event': 'complete', 'rows': int(len(out)), 'valid_paths': int(out.policy_path_valid.sum())}))


if __name__ == '__main__':
    main()
