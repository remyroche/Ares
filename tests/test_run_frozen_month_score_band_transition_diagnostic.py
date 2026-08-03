import pandas as pd
import hashlib, json
from pathlib import Path
from scripts.run_frozen_month_score_band_transition_diagnostic import assign, thresholds, expected_top_coverage, TOPS, ROOT

def test_frozen_numeric_cutoffs_do_not_re_rank_target_rows():
    source=pd.Series(range(30),dtype=float); cuts=thresholds(source,10)
    target=pd.Series([100.,101.,102.])
    assert assign(target,cuts).tolist()==[9,9,9]
    assert assign(source,cuts).min()==0 and assign(source,cuts).max()==9

def test_required_top_coverage_has_both_schemes_all_months_and_depths():
    keys=expected_top_coverage()
    assert len(keys)==2*3*len(TOPS)
    for pair in ('2025-02->2025-03','2025-03->2025-04'):
        assert any(k[0]==pair and k[2]=='source_frozen' for k in keys)
        assert any(k[0]==pair and k[2]=='target_local' for k in keys)

def test_sealed_v2_has_complete_source_target_top_coverage_and_audits():
    root=ROOT/'data_perp/artifacts/frozen_month_score_band_transition_diagnostic_20260730_v2'
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['runner']['sha256']==hashlib.sha256((ROOT/'scripts/run_frozen_month_score_band_transition_diagnostic.py').read_bytes()).hexdigest()
    top=pd.read_csv(root/'fixed_band_global_top_contribution.csv')
    found=set(tuple(x) for x in top[['pair','evaluation_month','scheme','top_fraction']].drop_duplicates().itertuples(index=False,name=None))
    assert found==expected_top_coverage()
    # Each global top key has contributions from the concurrently emitted
    # decile and ventile decompositions (the number of occupied bands varies
    # by depth); selection remains pooled-global because no scope exists.
    assert (top.groupby(['pair','evaluation_month','scheme','top_fraction']).size()>=2).all()
    bands=pd.read_csv(root/'band_metrics.csv')
    assert set(bands.band_kind)=={'decile','ventile'}
    assert not any(bands.scope.str.startswith('side_selection'))
    audit=pd.read_csv(root/'frozen_cutoffs_tie_audit.csv')
    assert audit.threshold_role.eq('target_application_of_frozen_source_threshold').any()
    assert {'plateau_rows','tie_ambiguous'}.issubset(audit.columns)
    assert {'net_mean_p025_bps','net_mean_p975_bps','net_ic_p025','net_ic_p975'}.issubset(bands.columns)
