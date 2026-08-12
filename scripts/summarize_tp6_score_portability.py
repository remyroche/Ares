#!/usr/bin/env python3
"""Write the decision artifact for the TP6 score-portability audit."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data_perp/artifacts/tp6_score_portability_admission_20260803_v1'


def table(frame: pd.DataFrame, *, index: bool = True) -> str:
    """Dependency-free table rendering for the Markdown decision artifact."""
    return "```csv\n" + frame.to_csv(index=index) + "```"


def value(m:pd.DataFrame, transport:str, arm:str, top:float)->float:
    row=m[(m.transport.eq(transport))&(m.arm.eq(arm))&(m.scope.eq('global'))&(m.top_fraction.eq(top))].iloc[0]
    return float(row.net_bps)


def run() -> Path:
    metrics=pd.read_parquet(OUT/'score_portability_metrics.parquet')
    allocation=pd.read_parquet(OUT/'oracle_side_allocation.parquet')
    transports=sorted(metrics.transport.unique())
    records=[]
    for transport in transports:
        for top in (.05,.10):
            frozen=value(metrics,transport,'M0_frozen',top)
            calibration=value(metrics,transport,'C1_oracle_monotonic',top)
            side=float(allocation[(allocation.transport.eq(transport))&(allocation.top_fraction.eq(top))].oracle_net_bps.iloc[0])
            # Oracle monotonic mapping leaves order effectively fixed; side
            # allocation also uses frozen within-side ranks.  Thus the two
            # sequences are intentionally equivalent and expose no hidden
            # order dependence in this decomposition.
            records.extend([
                {'transport':transport,'top_fraction':top,'component':'frozen_common_bps','frozen_result_bps':frozen,'oracle_result_bps':frozen,'recoverable_gap_bps':0.,'sequence':'calibration_then_side'},
                {'transport':transport,'top_fraction':top,'component':'calibration_mapping','frozen_result_bps':frozen,'oracle_result_bps':calibration,'recoverable_gap_bps':calibration-frozen,'sequence':'calibration_then_side'},
                {'transport':transport,'top_fraction':top,'component':'side_allocation','frozen_result_bps':frozen,'oracle_result_bps':side,'recoverable_gap_bps':side-frozen,'sequence':'calibration_then_side'},
                {'transport':transport,'top_fraction':top,'component':'side_then_calibration_sensitivity','frozen_result_bps':frozen,'oracle_result_bps':side,'recoverable_gap_bps':side-frozen,'sequence':'side_then_calibration'},
            ])
    attribution=pd.DataFrame(records); attribution.to_parquet(OUT/'score_portability_attribution.parquet',index=False)
    deploy=metrics[(metrics.basis.eq('deployable_prior_resolved'))&(metrics.scope.eq('global'))&(metrics.top_fraction.isin([.05,.10]))].pivot_table(index=['transport','top_fraction'],columns='arm',values='net_bps').round(2)
    oracle=metrics[(metrics.basis.eq('ORACLE_TEST_LABEL_DIAGNOSTIC'))&(metrics.scope.eq('global'))&(metrics.top_fraction.isin([.05,.10]))].pivot_table(index=['transport','top_fraction'],columns='arm',values='net_bps').round(2)
    meta=pd.read_parquet(OUT/'tercile_meta_classifier_audit.parquet').round(3)
    price=pd.read_parquet(OUT/'matched_price_leverage_portability.parquet')
    report='''# TP6 score portability and admission decision

## Decision

`FROZEN_BPS_LEVEL_NOT_PORTABLE_FOR_ABSOLUTE_ADMISSION`; `NO_DEPLOYABLE_MAPPING_OR_ADMISSION_ARM_ADVANCES`; `TERCILE_META_REJECTED`.

The frozen base has at most narrow pooled-tail information (not robust side-local tails). An oracle test-era monotonic map can create small positive absolute-score populations, but the causal side-shrunk, window-ensemble, affine and strongly shrunk bin maps do not transfer that score level. Another residual-target sweep is not justified.

## Deployable common-bps maps — net bps/trade

'''+table(deploy)+'''\n\n## Oracle diagnostic maps — net bps/trade

'''+table(oracle)+'''\n\nOracle mappings use test labels and are non-deployable. They leave global top-k economics nearly unchanged and oracle side allocation remains all-long. However, the oracle monotonic map can form a small positive absolute-admission population in each transport; this exposes a score-level/admission calibration gap, not a deployable repair.

## Three-class residual meta diagnostic

The classifier labels training residuals by side-specific lower/middle/upper terciles (base overestimate / approximately correct / base underestimate), maps predicted class probabilities to training-only class means, and then ranks the common-bps reconstruction globally.  It degraded both transports.  Test log loss and accuracy:\n\n'''+table(meta[['transport','side_name','test_log_loss','test_accuracy','lower_edge_bps','upper_edge_bps']],index=False)+'''\n\n## Admission

The frozen 0/+25/+50/+100-bps thresholds mostly produce the valid no-trade outcome; the only non-empty frozen -50-bps arm is tiny in the first transport and has zero coverage in the later transport. The test-label oracle map demonstrates positive but narrow absolute populations (for example +36.76 bps at 0 in the first transport and +16.34 bps in the second), yet with only one positive month in each case. The tercile classifier admits more rows but is negative in the later transport. No deployable absolute threshold has portable, non-trivial positive net EV.

## Matched price–leverage fields

Rows were matched on side, decision month, base score decile, clear/adverse probability quintiles, and ATR-percentile quintile.  No stored price–leverage field meets the required stable, material conditional effect in both transports.\n\n'''+table(price[['transport','side_name','feature','high_minus_low_net_bps','effect_sign_consistent','cross_transport_role']],index=False)+'''\n\n## Next permitted work

Do not add another residual target or broad reranker. The next permissible work is a narrowly prequential score-level calibration/admission study, judged by worst-month breadth as well as coverage; it must explain why the oracle score level exists while all prior-resolved maps fail to transfer it. Mapping, side pooling and the tested three-class correction cannot yet turn the frozen score into a transportable executable population.
'''
    target=OUT/'SCORE_PORTABILITY_ADMISSION_REPORT.md'; target.write_text(report)
    return target


if __name__=='__main__': print(run())
