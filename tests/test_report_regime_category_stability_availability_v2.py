from pathlib import Path
def test_availability_report_forbids_gate_and_expost_phase():
 s=Path('scripts/report_regime_category_stability_availability_v2.py').read_text();assert 'SEALED_NO_GATE_INSUFFICIENT_COMPATIBLE_SUPPORT' in s and 'no_ex_post_phase_gate' in s and "'required':3" in s
