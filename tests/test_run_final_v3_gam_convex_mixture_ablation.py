from pathlib import Path
def test_fixed_grid_is_pre2026_and_global_top10_only():
 s=Path('scripts/run_final_v3_gam_convex_mixture_ablation.py').read_text()
 assert "EXPERTS=('gam_regime_only','gam_transition_only','gam_combined')" in s
 assert 'IsotonicRegression(increasing=True' in s and "TOP=.10" in s
 assert "oof_promotion_gate_passed" in s and "no_2026_tuning" in s
 assert "model_sample_cadence':'1h" in s
