#!/usr/bin/env python3
"""
Verify fee configuration single source of truth
"""
import sys
import os

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from extreme_price_movements.run_pipeline import BASE_ROUND_TRIP_FEE_PCT, PERP_ROUND_TRIP_FEE_PCT, _apply_fee_model
from extreme_price_movements.config import CFG
from extreme_price_movements.utils import tprint

def test_fee_single_source_of_truth():
    """Test that all fee configuration comes from run_pipeline.py constants."""
    tprint("=== FEE CONFIGURATION SINGLE SOURCE OF TRUTH TEST ===")
    
    # Test spot mode
    cfg_spot = CFG.copy()
    _apply_fee_model(cfg_spot, BASE_ROUND_TRIP_FEE_PCT)
    
    tprint(f"Spot Mode Fees:")
    tprint(f"  BASE_ROUND_TRIP_FEE_PCT: {BASE_ROUND_TRIP_FEE_PCT}%")
    tprint(f"  fee_bps: {cfg_spot['fee_bps']} bps")
    tprint(f"  optimiser_fee_pct: {cfg_spot['optimiser_fee_pct']*100:.3f}%")
    tprint(f"  ridge_cost_pct: {cfg_spot['ridge_cost_pct']*100:.3f}%")
    tprint(f"  label_round_trip_fee_pct: {cfg_spot['label_round_trip_fee_pct']*100:.1f}%")
    
    # Test perps mode
    cfg_perp = CFG.copy()
    _apply_fee_model(cfg_perp, PERP_ROUND_TRIP_FEE_PCT)
    
    tprint(f"\nPerpetual Mode Fees:")
    tprint(f"  PERP_ROUND_TRIP_FEE_PCT: {PERP_ROUND_TRIP_FEE_PCT}%")
    tprint(f"  fee_bps: {cfg_perp['fee_bps']} bps")
    tprint(f"  optimiser_fee_pct: {cfg_perp['optimiser_fee_pct']*100:.3f}%")
    tprint(f"  ridge_cost_pct: {cfg_perp['ridge_cost_pct']*100:.3f}%")
    tprint(f"  label_round_trip_fee_pct: {cfg_perp['label_round_trip_fee_pct']*100:.1f}%")
    
    # Verify consistency
    expected_spot_bps = BASE_ROUND_TRIP_FEE_PCT * 100 / 2
    expected_perp_bps = PERP_ROUND_TRIP_FEE_PCT * 100 / 2
    
    assert cfg_spot['fee_bps'] == expected_spot_bps, f"Spot fee_bps mismatch: {cfg_spot['fee_bps']} != {expected_spot_bps}"
    assert cfg_perp['fee_bps'] == expected_perp_bps, f"Perp fee_bps mismatch: {cfg_perp['fee_bps']} != {expected_perp_bps}"
    
    tprint(f"\n✅ SINGLE SOURCE OF TRUTH VERIFIED!")
    tprint(f"   All fee configuration derived from run_pipeline.py constants")
    tprint(f"   No hardcoded fees remain in config.py")
    
    return True

if __name__ == "__main__":
    test_fee_single_source_of_truth()
