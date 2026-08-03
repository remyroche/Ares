from __future__ import annotations
import pandas as pd
import pytest
from pathlib import Path
from scripts.run_final_v2_context_interaction_diagnostics import BASE, RESIDUAL, FEATURES, _strata, assert_schema_source_version, DiagnosticError

def test_interaction_diagnostic_uses_scores_and_continuous_context_not_state_ids() -> None:
    assert BASE in FEATURES and RESIDUAL in FEATURES
    assert all('state_id' not in field and 'gmm' not in field and 'morphology' not in field for field in FEATURES)

def test_combined_strata_preserves_separate_regime_and_transition_layers() -> None:
    d=pd.DataFrame({'regime_change_probability_max':[.1,.2,.8,.9],'transition_lgbm_probability':[.1,.8,.2,.9]})
    assert _strata(d,'regime').nunique()>1
    assert _strata(d,'transition').nunique()>1
    assert _strata(d,'combined').str.contains('|',regex=False).all()

def test_final_v3_source_cannot_use_a_final_v2_schema() -> None:
    with pytest.raises(DiagnosticError, match='requires a final_v3 interaction schema'):
        assert_schema_source_version(Path('data_perp/artifacts/final_identical_row_regime_stack_gam_ablation_20260730_v3'), 'final_v2_context_interaction_diagnostics_v1')
