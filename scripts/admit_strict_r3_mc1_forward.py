#!/usr/bin/env python3
"""Apply frozen Robust-21 control plus MC1_d2 authority to one live snapshot."""

from __future__ import annotations

import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from extreme_price_movements.strict_r3_canonical_v2 import assert_scoring_frame_is_target_free
from extreme_price_movements.strict_r3_mc1_mapper import CONTRACT, MC1D2Bundle
from extreme_price_movements.strict_r3_cell_day_trust import load_cell_day_residual_trust_bundle
from extreme_price_movements.strict_r3_a5_trust import apply_a5_bounded_10pct, load_a5_bundle

def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()

EMPTY_ADMISSION_FLOAT_FIELDS = (
    'a4_effective_support', 'a4_p_ev_positive_raw', 'a4_raw_expected_bps',
    'a4_raw_predictive_sd_bps', 'a5_bounded10_expected_bps',
    'a5_calibrated_expected_bps', 'a5_calibrated_p_positive',
    'auction_rank_adjustment_bps', 'causal_21d_side_expected_net_bps',
    'mc1_d2_expected_net_bps', 'mc1_d2_recent_global_shift_bps',
    'raw_expected_bps', 'robust21_expected_net_bps', 'robust21_support_days',
    'trust_corrected_expected_net_bps', 'trust_effective_support',
    'trust_p_adverse_200bps', 'trust_p_ev_positive',
    'trust_p_map_overestimate_100bps', 'trust_posterior_expected_bps',
    'trust_posterior_predictive_q10_bps', 'trust_residual_q25_bps',
)
EMPTY_ADMISSION_BOOL_FIELDS = (
    'a5_bounded10_admitted', 'a5_bounded10_available', 'a5_timestamp_top15',
    'causal_21d_side_admitted_ge_50bps', 'mc1_d2_admitted_ge_50bps',
    'mc1_d2_available', 'robust21_admitted_ge_50bps',
    'trust_posterior_admitted_ge_50bps', 'trust_posterior_available',
    'trust_risk_corroborated',
)
EMPTY_ADMISSION_TEXT_FIELDS = (
    'admission_rejection_reason', 'ev_mapping_vintage_mode',
    'mc1_d2_bundle_id', 'trust_authority',
)


def write_empty_admission(
    *, current: pd.DataFrame, history: pd.DataFrame, decision: pd.Timestamp,
    out_dir: Path, bundle: MC1D2Bundle, mc1_dir: Path, r5_dir: Path,
    a5_dir: Path,
) -> None:
    """Persist a schema-complete zero-entry decision without scoring models."""
    out = current.copy()
    for field in EMPTY_ADMISSION_FLOAT_FIELDS:
        out[field] = pd.Series(dtype='float64')
    for field in EMPTY_ADMISSION_BOOL_FIELDS:
        out[field] = pd.Series(dtype='bool')
    for field in EMPTY_ADMISSION_TEXT_FIELDS:
        out[field] = pd.Series(dtype='object')
    available = pd.to_datetime(history.policy_label_available_ts, utc=True)
    out_dir.mkdir(parents=True)
    out.to_parquet(
        out_dir/'admitted_predictions.parquet', index=False, compression='zstd',
    )
    pd.DataFrame([{
        'decision_ts': decision, 'rows': 0, 'robust21_admitted': 0,
        'mc1_admitted': 0, 'mc1_available': 0,
        'maximum_history_label_available_ts': available.loc[
            available.le(decision)
        ].max(),
    }]).to_parquet(out_dir/'mc1_admission_audit.parquet', index=False)
    manifest = {
        'schema': 'strict_r3_mc1_forward_admission_v1',
        'ev_mapping_vintage_mode': CONTRACT,
        'current_outcomes_consumed': [],
        'rows': 0, 'eligible_rows': 0, 'mapped_rows': 0,
        'mc1_admitted_rows': 0, 'robust21_control_admitted_rows': 0,
        'mc1_bundle_id': bundle.manifest['bundle_id'],
        'mc1_bundle_manifest_sha256': sha(mc1_dir/'run_manifest.json'),
        'champion_config_sha256': bundle.manifest['champion_config_sha256'],
        'admission_floor_bps': 50.0,
        'auction_order': 'frozen final_score',
        'robust21_role': 'control/fallback telemetry; not blended',
        'adaptive_exit_context': (
            'frozen R5/A5 shadow fields computed from Robust-21; '
            'no admission/ranking authority'
        ),
        'r5_bundle_manifest_sha256': sha(r5_dir/'run_manifest.json'),
        'a5_bundle_manifest_sha256': sha(a5_dir/'run_manifest.json'),
        'causality': (
            'only policy_label_available_ts <= decision enters dynamic '
            'shift/control; current frame target-free'
        ),
        'empty_current_entry_set_failed_closed': True,
    }
    (out_dir/'run_manifest.json').write_text(
        json.dumps(manifest, indent=2, default=str)+'\n'
    )

def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--resolved-score-label-ledger',type=Path,required=True)
    ap.add_argument('--current-predictions',type=Path,required=True)
    ap.add_argument('--mc1-bundle-dir',type=Path,required=True)
    ap.add_argument('--r5-bundle-dir',type=Path,required=True)
    ap.add_argument('--a5-bundle-dir',type=Path,required=True)
    ap.add_argument('--decision-ts',required=True)
    ap.add_argument('--out-dir',type=Path,required=True)
    a=ap.parse_args()
    if a.out_dir.exists(): raise FileExistsError(a.out_dir)
    decision=pd.Timestamp(a.decision_ts)
    decision=decision.tz_localize('UTC') if decision.tzinfo is None else decision.tz_convert('UTC')
    current=pd.read_parquet(a.current_predictions)
    assert_scoring_frame_is_target_free(current)
    current['__decision_ts__']=pd.to_datetime(current.__decision_ts__,utc=True)
    current=current[current.__decision_ts__.eq(decision)].copy()
    history=pd.read_parquet(a.resolved_score_label_ledger)
    bundle=MC1D2Bundle.load(a.mc1_bundle_dir)
    if current.empty:
        write_empty_admission(
            current=current, history=history, decision=decision,
            out_dir=a.out_dir, bundle=bundle, mc1_dir=a.mc1_bundle_dir,
            r5_dir=a.r5_bundle_dir, a5_dir=a.a5_bundle_dir,
        )
        return
    mapped=bundle.score(current,resolved_history=history,decision_ts=decision)
    out=current.merge(mapped,on='candidate_id',how='inner',validate='one_to_one')
    # R5/A5 no longer own admission, but their causal trust-state outputs are
    # part of Adaptive Exit V1's frozen F4 entry-context contract.  Recompute
    # them from the Robust-21 control coordinate so live F4 sees real values,
    # never median-filled placeholders.  These fields cannot affect MC1 or the
    # final-score auction below.
    out['raw_expected_bps']=pd.to_numeric(out.robust21_expected_net_bps,errors='coerce')
    r5=load_cell_day_residual_trust_bundle(a.r5_bundle_dir)
    r5_score=r5.score(out.loc[:,['candidate_id',*r5.fields,'raw_expected_bps']])
    out=out.merge(r5_score,on='candidate_id',how='inner',validate='one_to_one')
    posterior=pd.to_numeric(out.trust_posterior_expected_bps,errors='coerce')
    out['trust_posterior_available']=np.isfinite(posterior)
    out['trust_posterior_admitted_ge_50bps']=out.trust_posterior_available & posterior.ge(50.)
    a4,calibration=load_a5_bundle(a.a5_bundle_dir)
    a4_score=a4.score(out.loc[:,['candidate_id',*a4.fields,'raw_expected_bps']])
    out=out.merge(a4_score,on='candidate_id',how='inner',validate='one_to_one')
    out=out.merge(apply_a5_bounded_10pct(out,calibration=calibration),on='candidate_id',how='inner',validate='one_to_one')
    eligible=(out.frozen_base_contract_complete.fillna(False).astype(bool)
              & out.base_route_timestamp_top20.fillna(False).astype(bool))
    out.loc[~eligible,'mc1_d2_admitted_ge_50bps']=False
    # Compatibility aliases are explicit: MC1, not the historical cell-day
    # map, owns these executable fields under schema-v6.
    out['causal_21d_side_expected_net_bps']=out.mc1_d2_expected_net_bps
    out['causal_21d_side_admitted_ge_50bps']=out.mc1_d2_admitted_ge_50bps
    out['ev_mapping_vintage_mode']=CONTRACT
    out['admission_rejection_reason']=np.where(
        ~out.frozen_base_contract_complete.fillna(False),'frozen_base_contract_incomplete',
        np.where(~out.base_route_timestamp_top20.fillna(False),'stopped_after_base_below_timestamp_top20',
                 np.where(~out.mc1_d2_available,'mc1_inputs_unavailable',
                          np.where(~out.mc1_d2_admitted_ge_50bps,'mc1_expected_net_below_50bps',''))))
    a.out_dir.mkdir(parents=True)
    out.to_parquet(a.out_dir/'admitted_predictions.parquet',index=False,compression='zstd')
    audit=pd.DataFrame([{'decision_ts':decision,'rows':len(out),'robust21_admitted':int(out.robust21_admitted_ge_50bps.sum()),'mc1_admitted':int(out.mc1_d2_admitted_ge_50bps.sum()),'mc1_available':int(out.mc1_d2_available.sum()),'maximum_history_label_available_ts':pd.to_datetime(history.policy_label_available_ts,utc=True).loc[lambda x:x.le(decision)].max()}])
    audit.to_parquet(a.out_dir/'mc1_admission_audit.parquet',index=False)
    manifest={'schema':'strict_r3_mc1_forward_admission_v1','ev_mapping_vintage_mode':CONTRACT,'current_outcomes_consumed':[],'rows':len(out),'eligible_rows':int(eligible.sum()),'mapped_rows':int((out.mc1_d2_expected_net_bps.notna() & eligible).sum()),'mc1_admitted_rows':int(out.mc1_d2_admitted_ge_50bps.sum()),'robust21_control_admitted_rows':int(out.robust21_admitted_ge_50bps.sum()),'mc1_bundle_id':bundle.manifest['bundle_id'],'mc1_bundle_manifest_sha256':sha(a.mc1_bundle_dir/'run_manifest.json'),'champion_config_sha256':bundle.manifest['champion_config_sha256'],'admission_floor_bps':50.0,'auction_order':'frozen final_score','robust21_role':'control/fallback telemetry; not blended','adaptive_exit_context':'frozen R5/A5 shadow fields computed from Robust-21; no admission/ranking authority','r5_bundle_manifest_sha256':sha(a.r5_bundle_dir/'run_manifest.json'),'a5_bundle_manifest_sha256':sha(a.a5_bundle_dir/'run_manifest.json'),'causality':'only policy_label_available_ts <= decision enters dynamic shift/control; current frame target-free'}
    (a.out_dir/'run_manifest.json').write_text(json.dumps(manifest,indent=2,default=str)+'\n')

if __name__=='__main__': main()
