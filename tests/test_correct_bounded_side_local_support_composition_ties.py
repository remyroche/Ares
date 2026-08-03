from pathlib import Path
import json
import numpy as np
from scripts.correct_bounded_side_local_support_composition_ties import expected_precision

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data_perp/artifacts/bounded_side_local_support_composition_20260730_v2_tie_correction'

def test_expected_precision_uses_tie_positive_rate_not_expected_net_sign():
    # Above book has one winner; the two tied outcomes net to zero but have a
    # 50% positive rate.  Selecting two tied rows must therefore yield 2/3.
    assert np.isclose(expected_precision(np.array([1.0]),np.array([2.0,-2.0]),2),2/3)

def test_correction_binds_v1_and_strict_adverse_proof():
    m=json.loads((OUT/'manifest.json').read_text());p=json.loads((OUT/'adverse_strict_oof_proof.json').read_text())
    assert m['net_parity_assertion'] is True
    assert p['status']=='STRICT_OOF_ADVERSE_SEVERITY_PROVEN'
    assert len(p['rows'])==12
    assert all(x['strict_resolution_assertion'] for x in p['rows'])
