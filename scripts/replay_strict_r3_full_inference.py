#!/usr/bin/env python3
"""Score the strict-R3 consensus on all feature-valid candidates.

This is the corrected inference/evaluation boundary for the frozen strict-R3
stack.  Exact-H12 validity is required only for fitting and for reporting
exact-H12 metrics.  It is never a condition for making a prediction.  This is
particularly important for the thin January--April 2026 exact-label support:
those rows are genuine inference candidates, not absent model scores.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_tp6_sl4_exact170_canonical_consensus import (  # noqa: E402
    BASE_PARAMS,
    BASE_TRAIN_CAP,
    CAPS,
    WEIGHT_MODES,
    _fit_ranker,
    _impute,
    _load_contract,
    _pct,
)

FEATURE_PATH = ROOT / (
    'data_perp/artifacts/strict_r3_exact_h12_2025_2026_v16_approved_15m_proxy_features/'
    'canonical120_features.parquet'
)
LABEL_ROOTS = (
    ROOT / 'data_perp/artifacts/stage_i_historical_tp6_sl4_h12_r3_20260803_v1',
    ROOT / 'data_perp/artifacts/stage_i_packb_tp6_sl4_h12_r3_20260803_v1',
)


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--months', default='2025-01:2026-07', help='inclusive YYYY-MM range')
    p.add_argument(
        '--out-dir', type=Path,
        default=ROOT / 'data_perp/artifacts/strict_r3_full_inference_2025_2026_v2',
    )
    p.add_argument('--resume', action='store_true')
    return p.parse_args()


def _months(spec: str) -> list[str]:
    start, end = spec.split(':', 1)
    return pd.period_range(start, end, freq='M').astype(str).tolist()


def _load_labels() -> pd.DataFrame:
    cols = [
        'candidate_id', '__ts__', '__symbol__', 'side_name', '__decision_ts__',
        '__label_available_at__', 'label_valid', 'target_invalid',
        't2_tp6_sl4_event', 'robust_clear_event_b25',
        't4_tp6_sl4_gross_bps', 't4_tp6_sl4_net_bps',
    ]
    frames: list[pd.DataFrame] = []
    for root in LABEL_ROOTS:
        for path in sorted((root / 'parts').glob('month=*/side=long.parquet')):
            frames.append(pd.read_parquet(path, columns=cols))
    if not frames:
        raise FileNotFoundError('No long strict-R3 label partitions found.')
    out = pd.concat(frames, ignore_index=True)
    if out.candidate_id.duplicated().any():
        dup = out.loc[out.candidate_id.duplicated(False)].sort_values('candidate_id')
        check = dup.groupby('candidate_id')[['t4_tp6_sl4_gross_bps', 't4_tp6_sl4_net_bps']].nunique(dropna=False)
        if check.max(axis=1).gt(1).any():
            raise ValueError('Overlapping strict-R3 label roots disagree.')
        out = out.drop_duplicates('candidate_id', keep='first')
    for c in ('__ts__', '__decision_ts__', '__label_available_at__'):
        out[c] = pd.to_datetime(out[c], utc=True)
    out['month'] = out.__ts__.dt.to_period('M').astype(str)
    out['evaluation_exact_label_valid'] = (
        out.label_valid.astype(bool) & ~out.target_invalid.astype(bool)
        & np.isfinite(pd.to_numeric(out.t4_tp6_sl4_gross_bps, errors='coerce'))
        & np.isfinite(pd.to_numeric(out.t4_tp6_sl4_net_bps, errors='coerce'))
    )
    event = pd.to_numeric(out.t2_tp6_sl4_event, errors='coerce')
    robust = pd.to_numeric(out.robust_clear_event_b25, errors='coerce')
    out['r3_class'] = np.nan
    valid = out.evaluation_exact_label_valid
    out.loc[valid & event.eq(1), 'r3_class'] = 0
    out.loc[valid & event.ne(1) & robust.eq(1), 'r3_class'] = 2
    out.loc[valid & event.ne(1) & ~robust.eq(1), 'r3_class'] = 1
    return out


def _load_panel(labels: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    cols = ['__ts__', '__symbol__'] + fields
    feature = pd.read_parquet(FEATURE_PATH, columns=cols)
    feature.__ts__ = pd.to_datetime(feature.__ts__, utc=True)
    panel = labels.merge(feature, on=['__ts__', '__symbol__'], how='left', validate='one_to_one')
    finite = panel[fields].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    panel['feature_contract_complete'] = finite
    return panel


def _score_month(panel: pd.DataFrame, fields: list[str], month: str) -> tuple[pd.DataFrame, dict]:
    cutoff = pd.Timestamp(month + '-01', tz='UTC')
    train = panel.loc[
        panel.__label_available_at__.lt(cutoff)
        & panel.r3_class.notna()
        & panel.feature_contract_complete
    ].copy()
    # Crucially: this population has *no* label-validity predicate.  It is the
    # complete decision-time, frozen-contract inference population.
    test = panel.loc[panel.month.eq(month) & panel.feature_contract_complete].copy()
    if train.empty or test.empty:
        return pd.DataFrame(), {
            'month': month, 'training_rows': int(len(train)), 'feature_valid_inference_rows': int(len(test)),
            'scored_rows': 0, 'reason': 'empty_train_or_test',
        }
    x_train, x_test = _impute(train, test, fields)
    base_fit = train.sort_values('__label_available_at__').tail(BASE_TRAIN_CAP)
    x_base, _ = _impute(base_fit, test, fields)
    base = LGBMClassifier(**BASE_PARAMS)
    base.fit(x_base, base_fit.r3_class.astype(int).to_numpy())
    p_train = base.predict_proba(x_train)
    p_test = base.predict_proba(x_test)
    train['base_score'] = p_train[:, 2] - 0.5 * p_train[:, 0]
    test['base_score'] = p_test[:, 2] - 0.5 * p_test[:, 0]
    iso = IsotonicRegression(increasing=True, out_of_bounds='clip')
    iso.fit(train.base_score.to_numpy(), train.t4_tp6_sl4_net_bps.to_numpy())
    train['base_anchor_bps'] = iso.predict(train.base_score.to_numpy())
    test['base_anchor_bps'] = iso.predict(test.base_score.to_numpy())
    train['residual_bps'] = train.t4_tp6_sl4_net_bps - train.base_anchor_bps
    train['resid_grade'] = np.select(
        [train.residual_bps.le(-150), train.residual_bps.le(-50), train.residual_bps.le(50), train.residual_bps.le(150)],
        [0, 1, 2, 3], default=4,
    ).astype(int)
    head_predictions = []
    for cap in CAPS:
        for mode in WEIGHT_MODES:
            model = _fit_ranker(
                x_train.iloc[:, :cap], train.resid_grade.to_numpy(), train.__ts__, train.month, mode,
            )
            pred = model.predict(x_test.iloc[:, :cap]) if model is not None else np.zeros(len(test), dtype=float)
            head_predictions.append(pred)
    raw = np.column_stack(head_predictions)
    test['consensus_rank'] = np.nanmedian(
        np.column_stack([_pct(raw[:, i]) for i in range(raw.shape[1])]), axis=1,
    )
    test['base_rank'] = _pct(test.base_anchor_bps.to_numpy())
    test['final_score'] = 0.75 * test.base_rank + 0.25 * test.consensus_rank
    keep = [
        'candidate_id', '__ts__', '__decision_ts__', '__symbol__', 'side_name', 'month',
        'evaluation_exact_label_valid', 't4_tp6_sl4_gross_bps', 't4_tp6_sl4_net_bps',
        'base_score', 'base_anchor_bps', 'base_rank', 'consensus_rank', 'final_score',
    ]
    result = test[keep].rename(columns={
        't4_tp6_sl4_gross_bps': 'exact_h12_gross_bps',
        't4_tp6_sl4_net_bps': 'exact_h12_net_bps',
    })
    audit = {
        'month': month, 'training_rows': int(len(train)), 'base_fit_rows': int(len(base_fit)),
        'feature_valid_inference_rows': int(len(test)), 'scored_rows': int(len(result)),
        'exact_h12_evaluation_rows': int(result.evaluation_exact_label_valid.sum()),
        'exact_h12_evaluation_fraction': float(result.evaluation_exact_label_valid.mean()),
        'training_feature_finite_fraction': float(np.isfinite(x_train.to_numpy()).mean()),
        'inference_feature_finite_fraction': float(np.isfinite(x_test.to_numpy()).mean()),
        'inference_excluded_feature_incomplete': int(panel.month.eq(month).sum() - len(test)),
    }
    return result, audit


def main() -> None:
    args = _args()
    months = _months(args.months)
    if args.out_dir.exists() and not args.resume:
        raise FileExistsError(f'{args.out_dir} exists; use --resume to continue.')
    args.out_dir.mkdir(parents=True, exist_ok=bool(args.resume))
    parts = args.out_dir / 'month_parts'
    parts.mkdir(exist_ok=True)
    contract = _load_contract()
    fields = contract['long']
    labels = _load_labels()
    panel = _load_panel(labels, fields)
    audits = []
    for month in months:
        path = parts / f'{month}.parquet'
        audit_path = parts / f'{month}.json'
        if path.exists() and audit_path.exists():
            audits.append(json.loads(audit_path.read_text()))
            continue
        print(json.dumps({'event': 'score_month_start', 'month': month}), flush=True)
        scored, audit = _score_month(panel, fields, month)
        scored.to_parquet(path, index=False, compression='zstd')
        audit_path.write_text(json.dumps(audit, indent=2))
        audits.append(audit)
        print(json.dumps({'event': 'score_month_complete', **audit}), flush=True)
    pred = pd.concat([pd.read_parquet(parts / f'{month}.parquet') for month in months], ignore_index=True)
    pred.to_parquet(args.out_dir / 'predictions.parquet', index=False, compression='zstd')
    coverage = pd.DataFrame(audits).sort_values('month')
    coverage.to_parquet(args.out_dir / 'inference_and_evaluation_coverage.parquet', index=False)
    manifest = {
        'schema': 'strict_r3_full_inference_v1',
        'purpose': 'score feature-valid candidates independently of exact-H12 validity',
        'score_population': 'all long candidates with a complete frozen 120-field contract at decision time',
        'training_population': 'only label-valid, exact-H12-complete rows available before each held-month start',
        'evaluation_population': 'scored rows with evaluation_exact_label_valid=true; invalid labels are excluded, never encoded as failures',
        'feature_path': str(FEATURE_PATH), 'label_roots': [str(x) for x in LABEL_ROOTS],
        'base_params': BASE_PARAMS, 'caps': CAPS, 'weight_modes': WEIGHT_MODES,
        'blend': '0.75 monthly base percentile + 0.25 monthly median ten-head percentile',
        'months': months, 'rows': int(len(pred)),
    }
    (args.out_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps({'event': 'complete', 'rows': int(len(pred)), 'output': str(args.out_dir)}))


if __name__ == '__main__':
    main()
